"""
GAB Intelligence API — FastAPI backend
Banque Populaire du Maroc
"""
from fastapi import FastAPI, Query, Request, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import json, os, sys, time, random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Import du module d'inférence réel ─────────────────────────
sys.path.insert(0, BASE)
try:
    from scripts.inference import predict_gab_risk as _ml_predict
    import scripts.inference as _inf_module
    _USE_ML_MODEL = True
except Exception:
    _USE_ML_MODEL = False
    _inf_module = None

app = FastAPI(title="GAB Intelligence API")

# ── Export router ─────────────────────────────────────────────
from export import router as export_router
app.include_router(export_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Cache réponses (TTL dict) ──────────────────────────────────
_CACHE: dict = {}
_CACHE_TTL = int(os.getenv("GAB_CACHE_TTL", "300"))   # secondes, défaut 5 min

def _cache_get(key: str):
    entry = _CACHE.get(key)
    if entry and time.time() - entry[1] < _CACHE_TTL:
        return entry[0]
    return None

def _cache_set(key: str, value):
    _CACHE[key] = (value, time.time())

def _cache_clear():
    _CACHE.clear()

# ── Logs de prédiction (JSONL) ────────────────────────────────
PREDICTIONS_LOG = Path(BASE) / "data" / "predictions.log"
_ADMIN_TOKEN = os.getenv("GAB_ADMIN_TOKEN", "dev-token-change-in-prod")

def _log_prediction(inp: dict, result: dict) -> None:
    entry = {
        "ts":      datetime.utcnow().isoformat() + "Z",
        "ville":   inp.get("ville"),
        "type_gab": inp.get("type_gab"),
        "age":     inp.get("age"),
        "score":   result.get("score"),
        "niveau":  result.get("niveau"),
        "mode":    result.get("mode"),
        "version": result.get("model_version", "n/a"),
    }
    try:
        with open(PREDICTIONS_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass

# ── Data loading ──────────────────────────────────────────────
def load_data():
    df = pd.read_csv(os.path.join(BASE, "data", "gab_dataset.csv"), parse_dates=["date"])
    fi = pd.read_csv(os.path.join(BASE, "data", "feature_importance.csv"))
    with open(os.path.join(BASE, "data", "resultats_modeles.json")) as f:
        res = json.load(f)
    return df, fi, res

DF, FEAT_IMP, _res_raw = load_data()

RESULTATS         = _res_raw.get("resultats", _res_raw)
MEILLEUR_MODELE   = _res_raw.get("meilleur", "")
COUTS             = _res_raw.get("cout_hypotheses", {"fn": 5000, "tp": 1500, "fp": 500})
MODEL_HASH        = _res_raw.get("model_hash", "n/a")
SELECTION_CRITERE = _res_raw.get("selection_critere", "AUC-PR")

if not MEILLEUR_MODELE:
    non_dummy = {k: v for k, v in RESULTATS.items() if k != "Dummy (Stratified)"}
    MEILLEUR_MODELE = max(non_dummy, key=lambda k: non_dummy[k].get("f1", 0)) if non_dummy else ""

if "importance" in FEAT_IMP.columns and "imp_moy" not in FEAT_IMP.columns:
    FEAT_IMP["imp_moy"] = FEAT_IMP["importance"]
    FEAT_IMP["imp_rf"]  = FEAT_IMP["importance"]
    FEAT_IMP["imp_gb"]  = FEAT_IMP["importance"]
FEAT_IMP = FEAT_IMP.sort_values("imp_moy", ascending=False).reset_index(drop=True)


def filter_df(df, villes=None, types=None, annees=None):
    d = df.copy()
    if villes:
        d = d[d["ville"].isin(villes)]
    if types:
        d = d[d["type_gab"].isin(types)]
    if annees:
        d = d[d["date"].dt.year.isin([int(a) for a in annees])]
    return d

# ── Health ─────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "mode": "ml" if _USE_ML_MODEL else "heuristic",
        "model_hash": MODEL_HASH,
        "meilleur_modele": MEILLEUR_MODELE,
        "nb_observations": int(len(DF)),
    }

# ── Metadata ──────────────────────────────────────────────────
@app.get("/api/metadata")
def metadata():
    return {
        "villes": sorted(DF["ville"].unique().tolist()),
        "types": sorted(DF["type_gab"].unique().tolist()),
        "environnements": sorted(DF["environnement"].unique().tolist()),
        "annees": ["2022", "2023"],
        "modeles": [k for k in RESULTATS if k != "Dummy (Stratified)"],
    }

# ── Overview ──────────────────────────────────────────────────
@app.get("/api/overview")
def overview(
    request: Request,
    villes: List[str] = Query(default=[]),
    types: List[str] = Query(default=[]),
    annees: List[str] = Query(default=[]),
):
    cache_key = str(request.url)
    if cached := _cache_get(cache_key):
        return cached

    df = filter_df(DF, villes or None, types or None, annees or None)
    meilleur = MEILLEUR_MODELE
    r = RESULTATS[meilleur]

    kpis = {
        "nb_gab":         int(df["gab_id"].nunique()),
        "nb_pannes":      int(df["panne_sous_48h"].sum()),
        "taux_panne":     float(df["panne_sous_48h"].mean() * 100),
        "meilleur_modele": meilleur,
        "f1_best":        float(r["f1"]),
        "recall_best":    float(r["recall"]),
        "nb_obs":         int(len(df)),
        "nb_villes":      int(df["ville"].nunique()),
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

# ── Geography ─────────────────────────────────────────────────
@app.get("/api/geography")
def geography(
    request: Request,
    villes: List[str] = Query(default=[]),
    types: List[str] = Query(default=[]),
    annees: List[str] = Query(default=[]),
):
    cache_key = str(request.url)
    if cached := _cache_get(cache_key):
        return cached

    df = filter_df(DF, villes or None, types or None, annees or None)

    vs = (
        df.groupby("ville")
        .agg(nb_pannes=("panne_sous_48h", "sum"),
             nb_obs=("panne_sous_48h", "count"),
             nb_gab=("gab_id", "nunique"))
        .reset_index()
    )
    vs["taux_panne"]    = (vs["nb_pannes"] / vs["nb_obs"] * 100).round(2)
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
            "mois":   mois_noms[:len(hm_pivot.columns)],
            "z":      hm_pivot.values.round(2).tolist(),
        },
    }
    _cache_set(cache_key, result)
    return result

# ── Models ────────────────────────────────────────────────────
@app.get("/api/models")
def models(request: Request):
    cache_key = str(request.url)
    if cached := _cache_get(cache_key):
        return cached

    result = {
        "resultats":        RESULTATS,
        "meilleur":         MEILLEUR_MODELE,
        "model_hash":       MODEL_HASH,
        "selection_critere": SELECTION_CRITERE,
        "couts":            COUTS,
    }
    _cache_set(cache_key, result)
    return result

# ── Features ──────────────────────────────────────────────────
@app.get("/api/features")
def features(request: Request, top: int = Query(default=25, ge=10, le=101)):
    cache_key = str(request.url)
    if cached := _cache_get(cache_key):
        return cached

    def get_family(col):
        if any(f"lag{l}" in col for l in [1, 3, 7]):
            return "Lag"
        if "roll" in col:
            return "Rolling"
        if "tendance" in col or "acceleration" in col:
            return "Tendance"
        if col in ["risque_materiel", "stress_thermique", "score_surcharge",
                   "score_negligence", "score_connectivite", "ratio_erreurs_tx"]:
            return "Interaction"
        if "type_gab" in col or "environnement" in col:
            return "Catégoriel"
        if col in ["mois_sin", "mois_cos", "jour_sin", "jour_cos",
                   "est_weekend", "est_ete", "est_fin_mois", "trimestre"]:
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

# ── Threshold simulation ──────────────────────────────────────
@app.get("/api/threshold")
def threshold(
    request: Request,
    model: str = Query(default=""),
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

    tp_total = r["tp"] + r["fn"]
    tn_total = r["tn"] + r["fp"]
    total    = tp_total + tn_total
    base_rate = tp_total / total if total > 0 else 0.1

    seuils = np.linspace(0.01, 0.99, 200)

    def interp3(t, y0, y_mid, y1):
        if t <= 0.5:
            alpha = t / 0.5
            return y0 + alpha * (y_mid - y0)
        else:
            alpha = (t - 0.5) / 0.5
            return y_mid + alpha * (y1 - y_mid)

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
        "seuils":            seuils.round(3).tolist(),
        "recalls":           recalls_s,
        "precisions":        precs_s,
        "f1s":               f1s_s,
        "couts":             couts_s,
        "cout_sans_modele":  int(cout_sans),
        "seuil_f1_optimal":  round(seuil_f1, 2),
        "seuil_econ_optimal": round(seuil_econ, 2),
        "economie_max":      int(economie),
        "economie_pct":      round(economie / cout_sans * 100, 1) if cout_sans > 0 else 0,
        "tp_total":          int(tp_total),
        "model_metrics":     {k: round(float(v), 4) for k, v in r.items() if k in ["f1", "recall", "precision"]},
        "approximation":     True,
    }
    _cache_set(cache_key, result)
    return result

# ── Live scoring ──────────────────────────────────────────────
class ScoringInput(BaseModel):
    ville: str = "Casablanca"
    type_gab: str = "NCR"
    environnement: str = "Agence_Interieure"
    age: int = 4
    erreurs_lecteur: int = 2
    erreurs_dist: int = 0
    temperature: float = 35.0
    jours_maint: int = 60
    nb_tx: int = 120
    taux_erreur_tx: float = 0.05
    latence_ms: float = 80.0
    deconnexions: int = 0
    erreurs_roll7: float = 1.5
    temp_roll7: float = 34.0

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
        except Exception:
            pass

    # ── Mode dégradé : formule heuristique (fallback) ─────────
    # Poids recalibrés depuis feature_importance.csv (Logistic Regression)
    # jours_depuis_maintenance: 0.367 | score_negligence: 0.237
    # erreurs_lecteur_carte: 0.225  | temp_tendance_7j: 0.189
    # age_annees: 0.150 | temperature_interne: 0.150 | risque_materiel: 0.122
    # taux_erreur_tx: 0.094 | ville_risk_encoding: 0.038
    taux_ville = {
        "Safi": 0.110, "Agadir": 0.105, "Casablanca": 0.103,
        "El Jadida": 0.099, "Kénitra": 0.099, "Beni Mellal": 0.098,
        "Rabat": 0.096, "Marrakech": 0.096, "Meknès": 0.095,
        "Fès": 0.094, "Oujda": 0.094, "Tanger": 0.093, "Tétouan": 0.092,
    }
    risque_env  = {"Site_Isole": 1.0, "Centre_Commercial": 0.7, "Agence_Facade": 0.6, "Agence_Interieure": 0.5}
    risque_type = {"Wincor": 1.0, "NCR": 0.7, "Hyosung": 0.65, "Diebold": 0.6}

    # Composantes intermédiaires
    ratio_err  = (inp.erreurs_lecteur + inp.erreurs_dist) / (inp.nb_tx + 1)
    sc_negl    = inp.jours_maint * np.log1p(inp.erreurs_roll7)
    rq_mat     = inp.erreurs_lecteur * np.log1p(inp.age)
    st_therm   = inp.temperature * inp.age / 10
    tend_lect  = (inp.erreurs_lecteur - inp.erreurs_roll7) / (inp.erreurs_roll7 + 1e-8)
    tend_temp  = (inp.temperature - inp.temp_roll7) / (inp.temp_roll7 + 1e-8)

    # Poids data-driven (normalisés depuis feature_importance.csv)
    score_raw = (
        (inp.jours_maint / 365) * 0.22          # jours_depuis_maintenance: 0.367
        + (sc_negl / 1000) * 0.14                # score_negligence: 0.237
        + ratio_err * 0.14                       # erreurs_lecteur + ratio: 0.225
        + max(0, tend_temp) * 0.11               # temp_tendance_7j: 0.189
        + (inp.temperature / 65) * 0.09          # temperature_interne: 0.150
        + (inp.age / 10) * 0.09                  # age_annees: 0.150
        + (rq_mat / 30) * 0.07                   # risque_materiel: 0.122
        + inp.taux_erreur_tx * 0.06              # taux_erreur_tx: 0.094
        + max(0, tend_lect) * 0.03               # erreurs_lecteur_tendance: 0.011
        + taux_ville.get(inp.ville, 0.097) * 0.02  # ville_risk_encoding: 0.038
        + risque_env.get(inp.environnement, 0.5) * 0.02  # environnement: ~0.066
        + risque_type.get(inp.type_gab, 0.65) * 0.01     # type_gab: ~0.058
    )

    # Sigmoid normalization (remplace min(1.0, raw * 1.8))
    # Calibré pour : raw≈0.05→0.05, raw≈0.25→0.27, raw≈0.45→0.73
    score = float(1.0 / (1.0 + np.exp(-10.0 * (score_raw - 0.35))))

    # Seuils alignés avec le modèle ML (inference.py)
    if score >= 0.65:
        niveau, couleur, recommandation = "CRITIQUE", "#d4645a", "Intervention immédiate requise"
    elif score >= 0.35:
        niveau, couleur, recommandation = "ÉLEVÉ", "#e8a045", "Planifier une visite dans les 48h"
    elif score >= 0.15:
        niveau, couleur, recommandation = "MODÉRÉ", "#5a8fc4", "Surveiller l'évolution cette semaine"
    else:
        niveau, couleur, recommandation = "FAIBLE", "#6ba88a", "Aucune action immédiate nécessaire"

    response = {
        "score":          round(score, 3),
        "score_pct":      round(score * 100, 1),
        "niveau":         niveau,
        "couleur":        couleur,
        "recommandation": recommandation,
        "contributions": {
            "Ancienneté maint.":    round((inp.jours_maint / 365) * 0.22, 4),
            "Négligence maint.":    round((sc_negl / 1000) * 0.14, 4),
            "Ratio erreurs":        round(ratio_err * 0.14, 4),
            "Tendance température": round(max(0, tend_temp) * 0.11, 4),
            "Température":          round((inp.temperature / 65) * 0.09, 4),
            "Âge GAB":              round((inp.age / 10) * 0.09, 4),
            "Risque matériel":      round((rq_mat / 30) * 0.07, 4),
            "Taux erreur tx":       round(inp.taux_erreur_tx * 0.06, 4),
            "Tendance lecteur":     round(max(0, tend_lect) * 0.03, 4),
        },
        "mode": "heuristic",
    }
    _log_prediction(inp.model_dump(), response)
    return response

# ── Reload modèle à chaud ─────────────────────────────────────
@app.post("/api/reload-model")
def reload_model(x_admin_token: Optional[str] = Header(default=None)):
    if x_admin_token != _ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Token admin invalide")

    global _USE_ML_MODEL
    _cache_clear()

    if _inf_module is not None:
        try:
            # Réinitialise les singletons de scripts/inference.py
            _inf_module._MODEL_ARTIFACT = None
            _inf_module._ENCODERS = None
            _inf_module._lazy_load()
            new_hash = _inf_module._MODEL_ARTIFACT.get("model_hash", "n/a") if _inf_module._MODEL_ARTIFACT else "n/a"
            _USE_ML_MODEL = True
            return {"status": "ok", "model_hash": new_hash, "cache": "cleared"}
        except Exception as e:
            return {"status": "error", "detail": str(e)}

    return {"status": "skipped", "reason": "inference module not loaded"}

# ── Monitoring temps réel (simulation) ────────────────────────
_MONITORING_CACHE = {"data": None, "ts": 0}
_MONITORING_TTL = 25  # secondes — légèrement < poll frontend (30s)

def _level_from_score(score):
    """Seuils alignés ML/heuristique → (niveau, couleur)."""
    if score >= 0.65:
        return "CRITIQUE", "#d4645a"
    elif score >= 0.35:
        return "ÉLEVÉ", "#e8a045"
    elif score >= 0.15:
        return "MODÉRÉ", "#5a8fc4"
    return "FAIBLE", "#6ba88a"

def _score_heuristic_fast(err_lect, err_dist, temp, jours_m, nb_tx,
                          taux_err, age, err_roll7, temp_roll7,
                          ville, env, type_gab):
    """Heuristique rapide alignée avec /api/scoring (poids data-driven + sigmoid)."""
    taux_ville = {
        "Safi": 0.110, "Agadir": 0.105, "Casablanca": 0.103,
        "El Jadida": 0.099, "Kénitra": 0.099, "Beni Mellal": 0.098,
        "Rabat": 0.096, "Marrakech": 0.096, "Meknès": 0.095,
        "Fès": 0.094, "Oujda": 0.094, "Tanger": 0.093, "Tétouan": 0.092,
    }
    risque_env  = {"Site_Isole": 1.0, "Centre_Commercial": 0.7, "Agence_Facade": 0.6, "Agence_Interieure": 0.5}
    risque_type = {"Wincor": 1.0, "NCR": 0.7, "Hyosung": 0.65, "Diebold": 0.6}

    ratio_err  = (err_lect + err_dist) / (nb_tx + 1)
    sc_negl    = jours_m * np.log1p(err_roll7)
    rq_mat     = err_lect * np.log1p(age)
    tend_temp  = max(0, (temp - temp_roll7) / (temp_roll7 + 1e-8))
    tend_lect  = max(0, (err_lect - err_roll7) / (err_roll7 + 1e-8))

    score_raw = (
        (jours_m / 365) * 0.22 + (sc_negl / 1000) * 0.14
        + ratio_err * 0.14 + tend_temp * 0.11
        + (temp / 65) * 0.09 + (age / 10) * 0.09
        + (rq_mat / 30) * 0.07 + taux_err * 0.06
        + tend_lect * 0.03
        + taux_ville.get(ville, 0.097) * 0.02
        + risque_env.get(env, 0.5) * 0.02
        + risque_type.get(type_gab, 0.65) * 0.01
    )
    return float(1.0 / (1.0 + np.exp(-10.0 * (score_raw - 0.35))))


def _simulate_monitoring():
    """Simule le statut temps réel de chaque GAB avec perturbations corrélées."""
    now = datetime.utcnow()
    latest = DF.sort_values("date").groupby("gab_id").last().reset_index()

    gabs = []
    alerts = []
    for _, row in latest.iterrows():
        age = float(row["age_annees"])

        # Facteur de dégradation corrélé : les vieux GAB avec beaucoup
        # de jours sans maintenance se dégradent plus vite
        degradation = np.clip(age / 10 + row["jours_depuis_maintenance"] / 365, 0, 1)
        # Bruit de base : GABs dégradés fluctuent plus
        noise_scale = 0.5 + degradation * 1.5

        # Perturbations corrélées (un GAB qui surchauffe a aussi plus d'erreurs)
        heat_shock  = random.uniform(-1, 3) * noise_scale
        err_lect = max(0, int(row["erreurs_lecteur_carte"] + random.uniform(-0.5, 2) * noise_scale + (0.3 * max(0, heat_shock))))
        err_dist = max(0, int(row["erreurs_distributeur"] + random.uniform(-0.5, 1.5) * noise_scale))
        temp     = round(row["temperature_interne"] + heat_shock, 1)
        jours_m  = max(0, int(row["jours_depuis_maintenance"] + random.randint(0, 10)))
        nb_tx    = max(1, int(row["nb_transactions"] + random.randint(-15, 15)))
        taux_err = round(min(0.3, max(0, row["taux_erreur_tx"] + random.uniform(-0.01, 0.03) * noise_scale)), 4)
        latence  = round(max(10, row["latence_ms"] + random.uniform(-5, 20) * noise_scale), 1)
        decos    = max(0, int(row["nb_deconnexions"] + random.uniform(-0.5, 1.5) * noise_scale))
        err_roll7 = round(max(0, (row.get("erreurs_lecteur_carte", 0) * 0.7 + err_lect * 0.3)), 2)
        temp_roll7 = round(row["temperature_interne"] * 0.7 + temp * 0.3, 1)

        # Scoring — ML si disponible, sinon heuristique
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
                score = _score_heuristic_fast(
                    err_lect, err_dist, temp, jours_m, nb_tx,
                    taux_err, age, err_roll7, temp_roll7,
                    row["ville"], row["environnement"], row["type_gab"])
                niveau, couleur = _level_from_score(score)
        else:
            score = _score_heuristic_fast(
                err_lect, err_dist, temp, jours_m, nb_tx,
                taux_err, age, err_roll7, temp_roll7,
                row["ville"], row["environnement"], row["type_gab"])
            niveau, couleur = _level_from_score(score)

        gab_info = {
            "gab_id":         row["gab_id"],
            "ville":          row["ville"],
            "type_gab":       row["type_gab"],
            "environnement":  row["environnement"],
            "age":            int(age),
            "score":          round(float(score), 3),
            "score_pct":      round(float(score * 100), 1),
            "niveau":         niveau,
            "couleur":        couleur,
            "temperature":    temp,
            "erreurs_lecteur": err_lect,
            "jours_maint":    jours_m,
        }
        gabs.append(gab_info)

        if niveau in ("CRITIQUE", "ÉLEVÉ"):
            minutes_ago = random.randint(0, 25)
            alerts.append({
                "gab_id":    row["gab_id"],
                "ville":     row["ville"],
                "type_gab":  row["type_gab"],
                "niveau":    niveau,
                "couleur":   couleur,
                "score_pct": round(float(score * 100), 1),
                "message":   f"Score {round(score * 100, 1)}% — {'Intervention immédiate requise' if niveau == 'CRITIQUE' else 'Planifier visite sous 48h'}",
                "timestamp": (now - timedelta(minutes=minutes_ago)).isoformat() + "Z",
            })

    alerts.sort(key=lambda a: a["timestamp"], reverse=True)

    compteurs = {"FAIBLE": 0, "MODÉRÉ": 0, "ÉLEVÉ": 0, "CRITIQUE": 0}
    for g in gabs:
        compteurs[g["niveau"]] = compteurs.get(g["niveau"], 0) + 1

    top_critiques = sorted(gabs, key=lambda g: g["score"], reverse=True)[:5]
    all_gabs = sorted(gabs, key=lambda g: g["score"], reverse=True)

    return {
        "timestamp":      now.isoformat() + "Z",
        "total_gab":      len(gabs),
        "compteurs":      compteurs,
        "top_critiques":  top_critiques,
        "all_gabs":       all_gabs,
        "alertes":        alerts[:10],
        "nb_alertes":     len(alerts),
        "mode":           "ml" if _USE_ML_MODEL else "heuristic",
    }

@app.get("/api/monitoring")
def monitoring():
    now = time.time()
    if _MONITORING_CACHE["data"] and (now - _MONITORING_CACHE["ts"]) < _MONITORING_TTL:
        return _MONITORING_CACHE["data"]
    result = _simulate_monitoring()
    _MONITORING_CACHE["data"] = result
    _MONITORING_CACHE["ts"] = now
    return result

# ── Stats des prédictions loggées ─────────────────────────────
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
        return {"error": str(e)}

    if not lines:
        return {"total": 0, "par_niveau": {}, "par_mode": {}, "score_moyen": None, "derniere_prediction": None}

    par_niveau: dict = {}
    par_mode: dict   = {}
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
