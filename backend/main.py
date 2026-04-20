"""
GAB Intelligence API — FastAPI backend
Banque Populaire du Maroc
"""
import json
import logging
import os
import sys
import time
import hashlib
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

# ── Local imports ─────────────────────────────────────────────────────────────
import config
from scoring_utils import (
    score_heuristic, level_from_score,
    VILLES_VALID, TYPES_VALID, ENVS_VALID,
)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Structured logging ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("gab_api")

# ── ML inference module ───────────────────────────────────────────────────────
sys.path.insert(0, BASE)
try:
    from scripts.inference import predict_gab_risk as _ml_predict
    import scripts.inference as _inf_module
    _USE_ML_MODEL = True
    logger.info("ML inference module loaded successfully")
except Exception as e:
    logger.warning(f"ML inference module unavailable ({e}) — heuristic fallback active")
    _USE_ML_MODEL = False
    _inf_module = None

# ── Startup data validation ───────────────────────────────────────────────────
def _check_startup_files() -> None:
    """Fail fast with a clear message if critical files are missing."""
    required = [
        Path(BASE) / "data" / "gab_dataset.csv",
        Path(BASE) / "data" / "feature_importance.csv",
        Path(BASE) / "data" / "resultats_modeles.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError(
            f"Fichiers requis manquants : {missing}\n"
            "Vérifiez que le répertoire data/ est complet."
        )

_check_startup_files()

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="GAB Intelligence API", version="1.1.0")

# ── Export router ─────────────────────────────────────────────────────────────
from export import router as export_router
app.include_router(export_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Response cache (TTL dict) ─────────────────────────────────────────────────
_CACHE: dict = {}

def _cache_get(key: str):
    entry = _CACHE.get(key)
    if entry and time.time() - entry[1] < config.CACHE_TTL:
        return entry[0]
    return None

def _cache_set(key: str, value) -> None:
    _CACHE[key] = (value, time.time())

def _cache_clear() -> None:
    _CACHE.clear()

# ── Prediction log ────────────────────────────────────────────────────────────
PREDICTIONS_LOG = Path(BASE) / "data" / "predictions.log"

def _log_prediction(inp: dict, result: dict) -> None:
    entry = {
        "ts":       datetime.utcnow().isoformat() + "Z",
        "ville":    inp.get("ville"),
        "type_gab": inp.get("type_gab"),
        "age":      inp.get("age"),
        "score":    result.get("score"),
        "niveau":   result.get("niveau"),
        "mode":     result.get("mode"),
        "version":  result.get("model_version", "n/a"),
    }
    try:
        with open(PREDICTIONS_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"Could not write prediction log: {e}")

# ── Data loading ──────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(
        os.path.join(BASE, "data", "gab_dataset.csv"), parse_dates=["date"]
    )
    fi = pd.read_csv(os.path.join(BASE, "data", "feature_importance.csv"))
    with open(os.path.join(BASE, "data", "resultats_modeles.json")) as f:
        res = json.load(f)
    logger.info(
        f"Data loaded: {len(df):,} rows, {df['gab_id'].nunique()} GABs, "
        f"{df['date'].min().date()} → {df['date'].max().date()}"
    )
    return df, fi, res

DF, FEAT_IMP, _res_raw = load_data()

RESULTATS         = _res_raw.get("resultats", _res_raw)
MEILLEUR_MODELE   = _res_raw.get("meilleur", "")
COUTS             = _res_raw.get("cout_hypotheses", {"fn": 5000, "tp": 1500, "fp": 500})
MODEL_HASH        = _res_raw.get("model_hash", "n/a")
SELECTION_CRITERE = _res_raw.get("selection_critere", "AUC-PR")

if not MEILLEUR_MODELE:
    non_dummy = {k: v for k, v in RESULTATS.items() if k != "Dummy (Stratified)"}
    MEILLEUR_MODELE = (
        max(non_dummy, key=lambda k: non_dummy[k].get("f1", 0)) if non_dummy else ""
    )

if "importance" in FEAT_IMP.columns and "imp_moy" not in FEAT_IMP.columns:
    FEAT_IMP["imp_moy"] = FEAT_IMP["importance"]
    FEAT_IMP["imp_rf"]  = FEAT_IMP["importance"]
    FEAT_IMP["imp_gb"]  = FEAT_IMP["importance"]
FEAT_IMP = FEAT_IMP.sort_values("imp_moy", ascending=False).reset_index(drop=True)

DATA_LAST_UPDATED = DF["date"].max().isoformat() if not DF.empty else "unknown"


def filter_df(df, villes=None, types=None, annees=None):
    d = df.copy()
    if villes:
        d = d[d["ville"].isin(villes)]
    if types:
        d = d[d["type_gab"].isin(types)]
    if annees:
        d = d[d["date"].dt.year.isin([int(a) for a in annees])]
    return d


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    model_loaded = _USE_ML_MODEL
    files_ok = all(
        (Path(BASE) / "data" / f).exists()
        for f in ["gab_dataset.csv", "feature_importance.csv", "resultats_modeles.json"]
    )
    return {
        "status": "ok" if files_ok else "degraded",
        "mode": "ml" if model_loaded else "heuristic",
        "model_hash": MODEL_HASH,
        "meilleur_modele": MEILLEUR_MODELE,
        "nb_observations": int(len(DF)),
        "nb_gab": int(DF["gab_id"].nunique()),
        "data_last_updated": DATA_LAST_UPDATED,
        "files_ok": files_ok,
    }


# ── Metadata ──────────────────────────────────────────────────────────────────
@app.get("/api/metadata")
def metadata():
    return {
        "villes": sorted(DF["ville"].unique().tolist()),
        "types":  sorted(DF["type_gab"].unique().tolist()),
        "environnements": sorted(DF["environnement"].unique().tolist()),
        "annees": ["2022", "2023"],
        "modeles": [k for k in RESULTATS if k != "Dummy (Stratified)"],
    }


# ── Overview ──────────────────────────────────────────────────────────────────
@app.get("/api/overview")
def overview(
    request: Request,
    villes: List[str] = Query(default=[]),
    types:  List[str] = Query(default=[]),
    annees: List[str] = Query(default=[]),
):
    cache_key = str(request.url)
    if cached := _cache_get(cache_key):
        return cached

    df = filter_df(DF, villes or None, types or None, annees or None)
    meilleur = MEILLEUR_MODELE
    r = RESULTATS[meilleur]

    kpis = {
        "nb_gab":          int(df["gab_id"].nunique()),
        "nb_pannes":       int(df["panne_sous_48h"].sum()),
        "taux_panne":      float(df["panne_sous_48h"].mean() * 100),
        "meilleur_modele": meilleur,
        "f1_best":         float(r["f1"]),
        "recall_best":     float(r["recall"]),
        "nb_obs":          int(len(df)),
        "nb_villes":       int(df["ville"].nunique()),
    }

    monthly = (
        df.assign(mois=df["date"].dt.to_period("M").astype(str))
        .groupby("mois")
        .agg(pannes=("panne_sous_48h", "sum"), obs=("panne_sous_48h", "count"))
        .reset_index()
    )
    monthly["taux"] = (monthly["pannes"] / monthly["obs"] * 100).round(2)

    taux_type = (
        df.groupby("type_gab")["panne_sous_48h"].mean().reset_index()
        .rename(columns={"panne_sous_48h": "taux"})
    )
    taux_type["taux"] = (taux_type["taux"] * 100).round(2)
    taux_type = taux_type.sort_values("taux")

    env_df = (
        df.groupby("environnement")["panne_sous_48h"].mean().reset_index()
        .rename(columns={"panne_sous_48h": "taux"})
    )
    env_df["taux"] = (env_df["taux"] * 100).round(2)

    df_s = df.copy()
    df_s["saison"] = df_s["date"].dt.month.map({
        12: "Hiver", 1: "Hiver", 2: "Hiver",
        3: "Printemps", 4: "Printemps", 5: "Printemps",
        6: "Été", 7: "Été", 8: "Été",
        9: "Automne", 10: "Automne", 11: "Automne",
    })
    sais = df_s.groupby("saison")["panne_sous_48h"].mean() * 100
    sais = sais.reindex(["Printemps", "Été", "Automne", "Hiver"]).fillna(0).round(2)

    age_df = (
        df.groupby("age_annees")["panne_sous_48h"].mean()
        .reset_index().rename(columns={"panne_sous_48h": "taux"})
    )
    age_df["taux"] = (age_df["taux"] * 100).round(2)

    top_city = df.groupby("ville")["panne_sous_48h"].mean().idxmax()

    result = {
        "kpis":      kpis,
        "monthly":   monthly.to_dict(orient="list"),
        "by_type":   taux_type.to_dict(orient="list"),
        "by_env":    env_df.to_dict(orient="list"),
        "by_season": {"saisons": sais.index.tolist(), "taux": sais.values.tolist()},
        "by_age":    age_df.to_dict(orient="list"),
        "top_city":  top_city,
    }
    _cache_set(cache_key, result)
    return result


# ── Geography ─────────────────────────────────────────────────────────────────
@app.get("/api/geography")
def geography(
    request: Request,
    villes: List[str] = Query(default=[]),
    types:  List[str] = Query(default=[]),
    annees: List[str] = Query(default=[]),
):
    cache_key = str(request.url)
    if cached := _cache_get(cache_key):
        return cached

    df = filter_df(DF, villes or None, types or None, annees or None)

    vs = (
        df.groupby("ville")
        .agg(
            nb_pannes=("panne_sous_48h", "sum"),
            nb_obs=("panne_sous_48h", "count"),
            nb_gab=("gab_id", "nunique"),
        )
        .reset_index()
    )
    vs["taux_panne"]     = (vs["nb_pannes"] / vs["nb_obs"] * 100).round(2)
    vs["pannes_par_gab"] = (vs["nb_pannes"] / vs["nb_gab"]).round(2)
    vs = vs.sort_values("taux_panne", ascending=False)

    mois_noms = ["Jan", "Fév", "Mar", "Avr", "Mai", "Jun", "Jul", "Aoû", "Sep", "Oct", "Nov", "Déc"]
    df_h = df.copy()
    df_h["mois_num"] = df_h["date"].dt.month
    hm = df_h.groupby(["ville", "mois_num"])["panne_sous_48h"].mean().reset_index()
    hm_pivot = hm.pivot(index="ville", columns="mois_num", values="panne_sous_48h").fillna(0) * 100

    result = {
        "cities": vs.to_dict(orient="list"),
        "heatmap": {
            "villes": hm_pivot.index.tolist(),
            "mois":   mois_noms[: len(hm_pivot.columns)],
            "z":      hm_pivot.values.round(2).tolist(),
        },
    }
    _cache_set(cache_key, result)
    return result


# ── Models ────────────────────────────────────────────────────────────────────
@app.get("/api/models")
def models(request: Request):
    cache_key = str(request.url)
    if cached := _cache_get(cache_key):
        return cached
    result = {
        "resultats":         RESULTATS,
        "meilleur":          MEILLEUR_MODELE,
        "model_hash":        MODEL_HASH,
        "selection_critere": SELECTION_CRITERE,
        "couts":             COUTS,
    }
    _cache_set(cache_key, result)
    return result


# ── Features ──────────────────────────────────────────────────────────────────
@app.get("/api/features")
def features(request: Request, top: int = Query(default=25, ge=10, le=101)):
    cache_key = str(request.url)
    if cached := _cache_get(cache_key):
        return cached

    def get_family(col: str) -> str:
        if any(f"lag{l}" in col for l in [1, 3, 7]):
            return "Lag"
        if "roll" in col:
            return "Rolling"
        if "tendance" in col or "acceleration" in col:
            return "Tendance"
        if col in [
            "risque_materiel", "stress_thermique", "score_surcharge",
            "score_negligence", "score_connectivite", "ratio_erreurs_tx",
        ]:
            return "Interaction"
        if "type_gab" in col or "environnement" in col:
            return "Catégoriel"
        if col in [
            "mois_sin", "mois_cos", "jour_sin", "jour_cos",
            "est_weekend", "est_ete", "est_fin_mois", "trimestre",
        ]:
            return "Temporel"
        return "Original"

    fam_col = {
        "Original": "#5a8fc4", "Lag": "#a87fb5",
        "Rolling": "#6ba88a",  "Tendance": "#e8a045",
        "Interaction": "#d4645a", "Temporel": "#c9952a",
        "Catégoriel": "#82a04f",
    }
    fam_desc = {
        "Rolling":     "Moyennes glissantes 7–14j",
        "Interaction": "Features métier composites",
        "Original":    "Variables brutes initiales",
        "Lag":         "Valeurs décalées J-1/3/7",
        "Tendance":    "Pente de dégradation",
        "Temporel":    "Cyclicité temporelle",
        "Catégoriel":  "Encodages type/env",
    }

    fi = FEAT_IMP.copy()
    fi["famille"] = fi["feature"].apply(get_family)
    fi["color"]   = fi["famille"].map(fam_col)
    top_df = fi.head(top).sort_values("imp_moy")

    fam_agg = (
        fi.groupby("famille")["imp_moy"].sum().reset_index()
        .sort_values("imp_moy", ascending=False)
    )
    fam_agg["pct"]   = (fam_agg["imp_moy"] / fam_agg["imp_moy"].sum() * 100).round(1)
    fam_agg["color"] = fam_agg["famille"].map(fam_col)
    fam_agg["desc"]  = fam_agg["famille"].map(fam_desc)

    result = {
        "top_features": top_df[["feature", "imp_moy", "imp_rf", "imp_gb", "famille", "color"]].to_dict(orient="list"),
        "families":     fam_agg.to_dict(orient="list"),
    }
    _cache_set(cache_key, result)
    return result


# ── Threshold simulation ──────────────────────────────────────────────────────
@app.get("/api/threshold")
def threshold(
    request: Request,
    model:          str = Query(default=""),
    cout_correctif: int = Query(default=5000),
    cout_preventif: int = Query(default=1500),
    cout_fausse:    int = Query(default=500),
):
    cache_key = str(request.url)
    if cached := _cache_get(cache_key):
        return cached

    nom_modele = model if model in RESULTATS else MEILLEUR_MODELE
    fallback   = RESULTATS.get(nom_modele) or RESULTATS.get("Logistic Regression") or next(iter(RESULTATS.values()))
    r = fallback

    tp_total  = r["tp"] + r["fn"]
    tn_total  = r["tn"] + r["fp"]
    total     = tp_total + tn_total
    base_rate = tp_total / total if total > 0 else 0.1

    seuils = np.linspace(0.01, 0.99, 200)

    def interp3(t, y0, y_mid, y1):
        if t <= 0.5:
            return y0 + (t / 0.5) * (y_mid - y0)
        return y_mid + ((t - 0.5) / 0.5) * (y1 - y_mid)

    rec_at_0   = 1.0
    rec_at_mid = float(r["recall"])
    rec_at_1   = 0.0
    pre_at_0   = float(base_rate)
    pre_at_mid = float(r["precision"])
    pre_at_1   = 1.0

    recalls_s, precs_s, f1s_s, couts_s = [], [], [], []
    for s in seuils:
        rec  = float(np.clip(interp3(float(s), rec_at_0, rec_at_mid, rec_at_1), 0, 1))
        prec = float(np.clip(interp3(float(s), pre_at_0, pre_at_mid, pre_at_1), 0, 1))
        f1   = 2 * prec * rec / (prec + rec + 1e-10)
        tp_s = int(rec * tp_total)
        fp_s = int(tp_s / (prec + 1e-10) - tp_s) if prec > 1e-10 else 0
        fn_s = tp_total - tp_s
        recalls_s.append(round(rec, 4))
        precs_s.append(round(prec, 4))
        f1s_s.append(round(float(f1), 4))
        couts_s.append(int(tp_s * cout_preventif + fp_s * cout_fausse + fn_s * cout_correctif))

    cout_sans  = tp_total * cout_correctif
    seuil_f1   = float(seuils[int(np.argmax(f1s_s))])
    seuil_econ = float(seuils[int(np.argmin(couts_s))])
    economie   = cout_sans - min(couts_s)

    result = {
        "seuils":             seuils.round(3).tolist(),
        "recalls":            recalls_s,
        "precisions":         precs_s,
        "f1s":                f1s_s,
        "couts":              couts_s,
        "cout_sans_modele":   int(cout_sans),
        "seuil_f1_optimal":   round(seuil_f1, 2),
        "seuil_econ_optimal": round(seuil_econ, 2),
        "economie_max":       int(economie),
        "economie_pct":       round(economie / cout_sans * 100, 1) if cout_sans > 0 else 0,
        "tp_total":           int(tp_total),
        "model_metrics":      {k: round(float(v), 4) for k, v in r.items() if k in ["f1", "recall", "precision"]},
        "approximation":      True,
    }
    _cache_set(cache_key, result)
    return result


# ── Live scoring ──────────────────────────────────────────────────────────────
class ScoringInput(BaseModel):
    ville:           str   = "Casablanca"
    type_gab:        str   = "NCR"
    environnement:   str   = "Agence_Interieure"
    age:             int   = 4
    erreurs_lecteur: int   = 2
    erreurs_dist:    int   = 0
    temperature:     float = 35.0
    jours_maint:     int   = 60
    nb_tx:           int   = 120
    taux_erreur_tx:  float = 0.05
    latence_ms:      float = 80.0
    deconnexions:    int   = 0
    erreurs_roll7:   float = 1.5
    temp_roll7:      float = 34.0

    @field_validator("ville")
    @classmethod
    def validate_ville(cls, v: str) -> str:
        if v not in VILLES_VALID:
            raise ValueError(f"Ville inconnue : '{v}'. Valeurs valides : {sorted(VILLES_VALID)}")
        return v

    @field_validator("type_gab")
    @classmethod
    def validate_type_gab(cls, v: str) -> str:
        if v not in TYPES_VALID:
            raise ValueError(f"Type GAB inconnu : '{v}'. Valeurs valides : {sorted(TYPES_VALID)}")
        return v

    @field_validator("environnement")
    @classmethod
    def validate_env(cls, v: str) -> str:
        if v not in ENVS_VALID:
            raise ValueError(f"Environnement inconnu : '{v}'. Valeurs valides : {sorted(ENVS_VALID)}")
        return v

    @field_validator("age")
    @classmethod
    def validate_age(cls, v: int) -> int:
        if not (0 <= v <= 30):
            raise ValueError("L'âge doit être entre 0 et 30 ans")
        return v

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if not (0 <= v <= 80):
            raise ValueError("La température doit être entre 0 et 80 °C")
        return v

    @field_validator("taux_erreur_tx")
    @classmethod
    def validate_taux(cls, v: float) -> float:
        if not (0 <= v <= 1):
            raise ValueError("taux_erreur_tx doit être compris entre 0.0 et 1.0")
        return v

    @field_validator("jours_maint")
    @classmethod
    def validate_jours(cls, v: int) -> int:
        if v < 0:
            raise ValueError("jours_maint ne peut pas être négatif")
        return v


@app.post("/api/scoring")
def scoring(inp: ScoringInput):
    if _USE_ML_MODEL:
        try:
            result = _ml_predict(inp.model_dump())
            response = {
                "score":          result["score"],
                "score_pct":      result["score_pct"],
                "niveau":         result["niveau"],
                "couleur":        result["couleur"],
                "recommandation": result["recommandation"],
                "contributions":  result["contributions"],
                "model_version":  result.get("model_version", "n/a"),
                "mode":           "ml",
            }
            _log_prediction(inp.model_dump(), response)
            return response
        except Exception as e:
            logger.warning(f"ML scoring failed for {inp.ville}/{inp.type_gab}: {e}")

    # ── Heuristic fallback (single source of truth: scoring_utils) ────────────
    d = inp.model_dump()
    score, contributions = score_heuristic(
        erreurs_lecteur=d["erreurs_lecteur"],
        erreurs_dist=d["erreurs_dist"],
        temperature=d["temperature"],
        jours_maint=d["jours_maint"],
        nb_tx=d["nb_tx"],
        taux_erreur_tx=d["taux_erreur_tx"],
        age=d["age"],
        erreurs_roll7=d["erreurs_roll7"],
        temp_roll7=d["temp_roll7"],
        ville=d["ville"],
        environnement=d["environnement"],
        type_gab=d["type_gab"],
    )
    niveau, couleur, recommandation = level_from_score(score)

    response = {
        "score":          round(score, 3),
        "score_pct":      round(score * 100, 1),
        "niveau":         niveau,
        "couleur":        couleur,
        "recommandation": recommandation,
        "contributions":  contributions,
        "mode":           "heuristic",
    }
    _log_prediction(inp.model_dump(), response)
    return response


# ── Batch scoring ─────────────────────────────────────────────────────────────
@app.post("/api/batch-score")
def batch_score(inputs: List[ScoringInput]):
    """Score multiple GABs in a single request (max 50)."""
    if len(inputs) > 50:
        raise HTTPException(status_code=422, detail="Maximum 50 entrées par requête batch")

    results = []
    for inp in inputs:
        d = inp.model_dump()

        if _USE_ML_MODEL:
            try:
                ml = _ml_predict(d)
                results.append({
                    "gab_id":   d.get("gab_id", f"GAB-{len(results)+1}"),
                    "ville":    d["ville"],
                    "type_gab": d["type_gab"],
                    "score":    ml["score"],
                    "score_pct": ml["score_pct"],
                    "niveau":   ml["niveau"],
                    "couleur":  ml["couleur"],
                    "mode":     "ml",
                })
                continue
            except Exception as e:
                logger.warning(f"ML batch scoring failed: {e}")

        score, _ = score_heuristic(
            erreurs_lecteur=d["erreurs_lecteur"],
            erreurs_dist=d["erreurs_dist"],
            temperature=d["temperature"],
            jours_maint=d["jours_maint"],
            nb_tx=d["nb_tx"],
            taux_erreur_tx=d["taux_erreur_tx"],
            age=d["age"],
            erreurs_roll7=d["erreurs_roll7"],
            temp_roll7=d["temp_roll7"],
            ville=d["ville"],
            environnement=d["environnement"],
            type_gab=d["type_gab"],
        )
        niveau, couleur, _ = level_from_score(score)
        results.append({
            "gab_id":    d.get("gab_id", f"GAB-{len(results)+1}"),
            "ville":     d["ville"],
            "type_gab":  d["type_gab"],
            "score":     round(score, 3),
            "score_pct": round(score * 100, 1),
            "niveau":    niveau,
            "couleur":   couleur,
            "mode":      "heuristic",
        })

    return {"count": len(results), "results": results}


# ── Reload model ──────────────────────────────────────────────────────────────
@app.post("/api/reload-model")
def reload_model(x_admin_token: Optional[str] = Header(default=None)):
    if x_admin_token != config.ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Token admin invalide")

    global _USE_ML_MODEL
    _cache_clear()

    if _inf_module is not None:
        try:
            _inf_module._MODEL_ARTIFACT = None
            _inf_module._ENCODERS = None
            _inf_module._lazy_load()
            new_hash = (
                _inf_module._MODEL_ARTIFACT.get("model_hash", "n/a")
                if _inf_module._MODEL_ARTIFACT else "n/a"
            )
            _USE_ML_MODEL = True
            logger.info(f"Model reloaded successfully. Hash: {new_hash}")
            return {"status": "ok", "model_hash": new_hash, "cache": "cleared"}
        except Exception as e:
            logger.error(f"Model reload failed: {e}")
            return {"status": "error", "detail": str(e)}

    return {"status": "skipped", "reason": "inference module not loaded"}


# ── Monitoring simulation ─────────────────────────────────────────────────────
_MONITORING_CACHE: dict = {"data": None, "ts": 0}

# Ring buffer for GAB score history (used by /api/gab/{id}/history)
_MONITORING_HISTORY: deque = deque(maxlen=config.MONITORING_HISTORY_SIZE)


def _get_rng_for_gab(gab_id: str) -> np.random.Generator:
    """Deterministic RNG per GAB ID + current hour → stable scores within polling window."""
    hour_str  = datetime.utcnow().strftime("%Y%m%d%H")
    seed_int  = int(hashlib.md5(f"{gab_id}_{hour_str}".encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed_int)


def _simulate_monitoring():
    """Simulate real-time GAB status with stable, correlated perturbations."""
    now    = datetime.utcnow()
    latest = DF.sort_values("date").groupby("gab_id").last().reset_index()

    gabs, alerts = [], []
    history_snapshot: dict[str, float] = {}

    for _, row in latest.iterrows():
        gab_id     = row["gab_id"]
        age        = float(row["age_annees"])
        rng        = _get_rng_for_gab(gab_id)
        degradation = np.clip(age / 10 + row["jours_depuis_maintenance"] / 365, 0, 1)
        noise_scale = 0.5 + degradation * 1.5

        heat_shock  = rng.uniform(-1, 3) * noise_scale
        err_lect    = max(0, int(row["erreurs_lecteur_carte"] + rng.uniform(-0.5, 2) * noise_scale + 0.3 * max(0, heat_shock)))
        err_dist    = max(0, int(row["erreurs_distributeur"] + rng.uniform(-0.5, 1.5) * noise_scale))
        temp        = round(row["temperature_interne"] + heat_shock, 1)
        jours_m     = max(0, int(row["jours_depuis_maintenance"] + rng.integers(0, 10)))
        nb_tx       = max(1, int(row["nb_transactions"] + rng.integers(-15, 16)))
        taux_err    = round(min(0.3, max(0, row["taux_erreur_tx"] + rng.uniform(-0.01, 0.03) * noise_scale)), 4)
        latence     = round(max(10, row["latence_ms"] + rng.uniform(-5, 20) * noise_scale), 1)
        decos       = max(0, int(row["nb_deconnexions"] + rng.uniform(-0.5, 1.5) * noise_scale))
        err_roll7   = round(max(0, row.get("erreurs_lecteur_carte", 0) * 0.7 + err_lect * 0.3), 2)
        temp_roll7  = round(row["temperature_interne"] * 0.7 + temp * 0.3, 1)

        # Scoring — ML if available, heuristic otherwise
        if _USE_ML_MODEL:
            try:
                ml_result = _ml_predict({
                    "ville": row["ville"], "type_gab": row["type_gab"],
                    "environnement": row["environnement"], "age": int(age),
                    "erreurs_lecteur": err_lect, "erreurs_dist": err_dist,
                    "temperature": temp, "jours_maint": jours_m,
                    "nb_tx": nb_tx, "taux_erreur_tx": taux_err,
                    "latence_ms": latence, "deconnexions": decos,
                    "erreurs_roll7": err_roll7, "temp_roll7": temp_roll7,
                })
                score   = ml_result["score"]
                niveau  = ml_result["niveau"]
                couleur = ml_result["couleur"]
            except Exception:
                score, _ = score_heuristic(
                    err_lect, err_dist, temp, jours_m, nb_tx, taux_err, age,
                    err_roll7, temp_roll7, row["ville"], row["environnement"], row["type_gab"],
                )
                niveau, couleur, _ = level_from_score(score)
        else:
            score, _ = score_heuristic(
                err_lect, err_dist, temp, jours_m, nb_tx, taux_err, age,
                err_roll7, temp_roll7, row["ville"], row["environnement"], row["type_gab"],
            )
            niveau, couleur, _ = level_from_score(score)

        history_snapshot[gab_id] = round(float(score), 3)

        gab_info = {
            "gab_id":          gab_id,
            "ville":           row["ville"],
            "type_gab":        row["type_gab"],
            "environnement":   row["environnement"],
            "age":             int(age),
            "score":           round(float(score), 3),
            "score_pct":       round(float(score * 100), 1),
            "niveau":          niveau,
            "couleur":         couleur,
            "temperature":     temp,
            "erreurs_lecteur": err_lect,
            "jours_maint":     jours_m,
        }
        gabs.append(gab_info)

        if niveau in ("CRITIQUE", "ÉLEVÉ"):
            minutes_ago = int(rng.integers(0, 25))
            alerts.append({
                "gab_id":    gab_id,
                "ville":     row["ville"],
                "type_gab":  row["type_gab"],
                "niveau":    niveau,
                "couleur":   couleur,
                "score_pct": round(float(score * 100), 1),
                "message":   (
                    f"Score {round(score * 100, 1)}% — "
                    f"{'Intervention immédiate requise' if niveau == 'CRITIQUE' else 'Planifier visite sous 48h'}"
                ),
                "timestamp": (now - timedelta(minutes=minutes_ago)).isoformat() + "Z",
            })

    # Store snapshot in history ring-buffer
    _MONITORING_HISTORY.append({"ts": now.isoformat() + "Z", "scores": history_snapshot})

    alerts.sort(key=lambda a: a["timestamp"], reverse=True)
    compteurs: dict[str, int] = {"FAIBLE": 0, "MODÉRÉ": 0, "ÉLEVÉ": 0, "CRITIQUE": 0}
    for g in gabs:
        compteurs[g["niveau"]] = compteurs.get(g["niveau"], 0) + 1

    return {
        "timestamp":     now.isoformat() + "Z",
        "total_gab":     len(gabs),
        "compteurs":     compteurs,
        "top_critiques": sorted(gabs, key=lambda g: g["score"], reverse=True)[:5],
        "all_gabs":      sorted(gabs, key=lambda g: g["score"], reverse=True),
        "alertes":       alerts[:10],
        "nb_alertes":    len(alerts),
        "mode":          "ml" if _USE_ML_MODEL else "heuristic",
    }


@app.get("/api/monitoring")
def monitoring(
    limit:  int = Query(default=config.MONITORING_DEFAULT_LIMIT, ge=1, le=config.MONITORING_MAX_LIMIT),
    offset: int = Query(default=0, ge=0),
):
    now = time.time()
    if _MONITORING_CACHE["data"] and (now - _MONITORING_CACHE["ts"]) < config.MONITORING_TTL:
        full = _MONITORING_CACHE["data"]
    else:
        full = _simulate_monitoring()
        _MONITORING_CACHE["data"] = full
        _MONITORING_CACHE["ts"]   = now

    # Paginate all_gabs without mutating the cached object
    all_gabs      = full["all_gabs"]
    total_gabs    = len(all_gabs)
    paginated_gabs = all_gabs[offset: offset + limit]

    return {
        **{k: v for k, v in full.items() if k != "all_gabs"},
        "all_gabs":   paginated_gabs,
        "total_gabs": total_gabs,
        "limit":      limit,
        "offset":     offset,
    }


# ── GAB history (from monitoring ring-buffer) ─────────────────────────────────
@app.get("/api/gab/{gab_id}/history")
def gab_history(
    gab_id: str,
    points: int = Query(default=24, ge=1, le=48, description="Number of history points to return"),
):
    """Return score history for a specific GAB from the monitoring ring-buffer."""
    history = list(_MONITORING_HISTORY)[-points:]
    series = [
        {"ts": snap["ts"], "score": snap["scores"].get(gab_id)}
        for snap in history
        if gab_id in snap.get("scores", {})
    ]

    # Enrich with current state if available
    current = None
    if _MONITORING_CACHE.get("data"):
        current = next(
            (g for g in _MONITORING_CACHE["data"].get("all_gabs", []) if g["gab_id"] == gab_id),
            None,
        )

    return {
        "gab_id":  gab_id,
        "count":   len(series),
        "history": series,
        "current": current,
    }


# ── Prediction stats ──────────────────────────────────────────────────────────
@app.get("/api/predictions-stats")
def predictions_stats():
    if not PREDICTIONS_LOG.exists():
        return {"total": 0, "par_niveau": {}, "par_mode": {}, "score_moyen": None, "derniere_prediction": None}

    lines = []
    try:
        with open(PREDICTIONS_LOG, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(json.loads(line))
    except Exception as e:
        logger.error(f"Error reading predictions log: {e}")
        return {"error": str(e)}

    if not lines:
        return {"total": 0, "par_niveau": {}, "par_mode": {}, "score_moyen": None, "derniere_prediction": None}

    par_niveau: dict = {}
    par_mode:   dict = {}
    scores = []

    for entry in lines:
        n = entry.get("niveau", "INCONNU")
        m = entry.get("mode", "unknown")
        s = entry.get("score")
        par_niveau[n] = par_niveau.get(n, 0) + 1
        par_mode[m]   = par_mode.get(m, 0) + 1
        if s is not None:
            scores.append(float(s))

    return {
        "total":               len(lines),
        "par_niveau":          par_niveau,
        "par_mode":            par_mode,
        "score_moyen":         round(sum(scores) / len(scores), 3) if scores else None,
        "derniere_prediction": lines[-1].get("ts"),
    }
