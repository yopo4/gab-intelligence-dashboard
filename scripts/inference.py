"""
==============================================================
PROJET : Prédiction des Pannes GAB/ATM - Banque Populaire Maroc
==============================================================
Script 05 — Module d'inférence standalone

Usage depuis FastAPI :
    from scripts.inference import predict_gab_risk

    result = predict_gab_risk({
        "ville": "Casablanca",
        "type_gab": "NCR",
        "environnement": "Agence_Interieure",
        "age": 4,
        "erreurs_lecteur": 2,
        "erreurs_dist": 0,
        "temperature": 35.0,
        "jours_maint": 60,
        "nb_tx": 120,
        "taux_erreur_tx": 0.05,   # 0–1 (pas %)
        "latence_ms": 80.0,
        "deconnexions": 0,
        "erreurs_roll7": 1.5,     # moy. erreurs lecteur / 7j
        "temp_roll7": 34.0,       # moy. température / 7j
    })

Stratégie d'inférence :
  Le frontend envoie ~14 features brutes. Le modèle attend ~100+ features
  (lags, rolling, interactions, encodages...).
  Pour les features non disponibles à J0 isolé → imputation par la médiane
  du train set (stockée dans encoders.pkl).
  Pour les features calculables depuis l'input → calcul à la volée.
  Pour les encodages catégoriels → reconstruction one-hot depuis encoders.pkl.
==============================================================
"""

import hashlib
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# ── Import optionnel SHAP ────────────────────────────────────
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# ══════════════════════════════════════════════════════════
# CHARGEMENT LAZY — une seule fois au premier appel
# ══════════════════════════════════════════════════════════
_MODEL_ARTIFACT  = None
_ENCODERS        = None

def _lazy_load():
    """Charge best_model.pkl et encoders.pkl en mémoire (singleton)."""
    global _MODEL_ARTIFACT, _ENCODERS
    if _MODEL_ARTIFACT is not None:
        return  # déjà chargé

    model_path    = DATA_DIR / "best_model.pkl"
    encoders_path = DATA_DIR / "encoders.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"best_model.pkl introuvable dans {DATA_DIR}. "
            "Exécutez d'abord : python scripts/modeling.py"
        )
    if not encoders_path.exists():
        raise FileNotFoundError(
            f"encoders.pkl introuvable dans {DATA_DIR}. "
            "Exécutez d'abord : python scripts/feature_engineering.py"
        )

    with open(model_path, "rb") as f:
        _MODEL_ARTIFACT = pickle.load(f)
    with open(encoders_path, "rb") as f:
        _ENCODERS = pickle.load(f)


def _get_saison(month: int) -> str:
    """Retourne la saison pour un mois donné (1–12)."""
    if month in [12, 1, 2]: return "Hiver"
    if month in [3, 4, 5]:  return "Printemps"
    if month in [6, 7, 8]:  return "Été"
    return "Automne"


def _build_feature_vector(features: dict) -> np.ndarray:
    """
    Construit le vecteur de features attendu par le modèle à partir
    des features brutes envoyées par le frontend ScoringLive.

    Ordre des priorités :
      1. Features calculables directement depuis l'input → calculées
      2. Features partiellement disponibles → calculées avec ce qu'on a
      3. Features impossibles à reconstituer (lags précis, rolling stds...) →
         imputation par la médiane du train set (encoders["feature_medians"])

    Cette approche est un compromis acceptable pour un outil opérationnel
    en production : la médiane représente le GAB "moyen" du train set.
    Les features directement observées (température, erreurs...) sont
    toujours renseignées correctement.
    """
    enc        = _ENCODERS
    medians    = enc["feature_medians"]
    feat_cols  = enc["feature_cols"]

    # ── Normalisation des noms ──────────────────────────────
    # Mapping frontend → noms internes du modèle
    ville       = features.get("ville", "Casablanca")
    type_gab    = features.get("type_gab", "NCR")
    env         = features.get("environnement", "Agence_Interieure")
    age         = float(features.get("age", 4))
    err_lec     = float(features.get("erreurs_lecteur", 0))
    err_dist    = float(features.get("erreurs_dist", 0))
    temp        = float(features.get("temperature", 35.0))
    jours_maint = float(features.get("jours_maint", 60))
    nb_tx       = float(features.get("nb_tx", 120))
    taux_err_tx = float(features.get("taux_erreur_tx", 0.05))  # 0–1
    latence     = float(features.get("latence_ms", 80.0))
    deconns     = float(features.get("deconnexions", 0))
    err_roll7   = float(features.get("erreurs_roll7", err_lec))   # fallback = J0
    temp_roll7  = float(features.get("temp_roll7", temp))         # fallback = J0

    # ── Date courante → features temporelles ────────────────
    from datetime import date as dt_date
    today = dt_date.today()
    month = today.month
    dow   = today.weekday()  # 0 = lundi
    saison = _get_saison(month)

    temporal = {
        "jour_semaine":  float(dow),
        "mois":          float(month),
        "trimestre":     float((month - 1) // 3 + 1),
        "est_weekend":   float(1 if dow >= 5 else 0),
        "est_ete":       float(1 if month in [6, 7, 8] else 0),
        "est_fin_mois":  float(1 if today.day >= 25 else 0),
        "mois_sin":      float(np.sin(2 * np.pi * month / 12)),
        "mois_cos":      float(np.cos(2 * np.pi * month / 12)),
        "jour_sin":      float(np.sin(2 * np.pi * dow / 7)),
        "jour_cos":      float(np.cos(2 * np.pi * dow / 7)),
    }

    # ── Features directes ────────────────────────────────────
    direct = {
        "age_annees":               age,
        "erreurs_lecteur_carte":    err_lec,
        "erreurs_distributeur":     err_dist,
        "temperature_interne":      temp,
        "jours_depuis_maintenance": jours_maint,
        "nb_transactions":          nb_tx,
        "taux_erreur_tx":           taux_err_tx,
        "latence_ms":               latence,
        "nb_deconnexions":          deconns,
        "niveau_billets_pct":       medians.get("niveau_billets_pct", 50.0),
    }

    # ── Rolling features calculables ─────────────────────────
    rolling_known = {
        "erreurs_lecteur_carte_roll7_mean":  err_roll7,
        "temperature_interne_roll7_mean":    temp_roll7,
    }

    # ── Interactions métier ───────────────────────────────────
    interactions = {
        "risque_materiel":    err_lec * np.log1p(age),
        "stress_thermique":   temp * age / 10,
        "score_surcharge":    nb_tx * taux_err_tx,
        "score_negligence":   jours_maint * np.log1p(err_roll7),
        "score_connectivite": latence * (1 + deconns),
        "ratio_erreurs_tx":   (err_lec + err_dist) / (nb_tx + 1),
    }

    # ── Tendance calculable : J0 vs moy. 7j ──────────────────
    tendency = {
        "erreurs_lecteur_carte_tendance_7j": (err_lec - err_roll7) / (err_roll7 + 1e-8),
        "temperature_interne_tendance_7j":   (temp - temp_roll7) / (temp_roll7 + 1e-8),
    }

    # ── Accélération (approx) : J0 vs J-1 (→ médiane J-1) ───
    err_lag1 = medians.get("erreurs_lecteur_carte_lag1", err_lec)
    acceleration = {
        "erreurs_lecteur_acceleration":
            (err_lec + err_lag1) / 2 - err_roll7,
        "erreurs_lecteur_carte_accel_2nd":
            err_lec - 2 * err_lag1 + medians.get("erreurs_lecteur_carte_lag3", err_lec),
        "erreurs_distributeur_accel_2nd":
            err_dist - 2 * medians.get("erreurs_distributeur_lag1", err_dist)
            + medians.get("erreurs_distributeur_lag3", err_dist),
        "temperature_interne_accel_2nd":
            temp - 2 * medians.get("temperature_interne_lag1", temp)
            + medians.get("temperature_interne_lag3", temp),
    }

    # ── Encodage ville (target encoding) ─────────────────────
    ville_enc = enc["ville_encoding"].get(ville, enc.get("fallback_taux", 0.1))

    # ── ratio_temperature_saison ─────────────────────────────
    temp_ref_key = f"{ville}|{saison}"
    temp_ref = enc["seasonal_temp"].get(temp_ref_key, enc.get("fallback_temp", 35.0))
    ratio_temp_saison = temp / (temp_ref + 1e-8)

    # ── One-hot encoding type_gab & environnement ────────────
    ohe = {}
    for col in enc.get("ohe_type_gab", []):
        # col est de la forme "type_gab_NCR"
        modal = col[len("type_gab_"):]
        ohe[col] = 1.0 if type_gab == modal else 0.0
    for col in enc.get("ohe_environnement", []):
        modal = col[len("environnement_"):]
        ohe[col] = 1.0 if env == modal else 0.0

    # ── Assemblage du vecteur complet ────────────────────────
    # On commence par les médianes pour tous les features,
    # puis on remplace par les valeurs calculées.
    vec = dict(medians)  # copy des médianes (imputation par défaut)
    vec.update(temporal)
    vec.update(direct)
    vec.update(rolling_known)
    vec.update(interactions)
    vec.update(tendency)
    vec.update(acceleration)
    vec["ville_risk_encoding"] = float(ville_enc)
    vec["ratio_temperature_saison"] = float(ratio_temp_saison)
    vec.update(ohe)

    # Construire le tableau dans l'ordre exact attendu par le modèle
    x = np.array([vec.get(col, medians.get(col, 0.0)) for col in feat_cols],
                 dtype=np.float64)
    return x, feat_cols


def _compute_contributions(x: np.ndarray, feat_cols: list) -> dict:
    """
    Calcule la contribution de chaque feature au score de risque.

    Priorité :
      1. SHAP TreeExplainer si shap installé et modèle tree-based
      2. Coefficients × valeur normalisée pour LogisticRegression
      3. feature_importances_ × valeur normalisée pour modèles tree-based
         (approx. non-linéaire mais indicative)

    Ne retourne que les 12 features avec les plus grandes contributions absolues.
    """
    enc      = _ENCODERS
    artifact = _MODEL_ARTIFACT
    pipeline = artifact["pipeline"]

    # Accès au modèle interne (à travers CalibratedClassifierCV → Pipeline)
    def _get_inner_model(fitted_pipeline):
        try:
            if hasattr(fitted_pipeline, "calibrated_classifiers_"):
                inner = fitted_pipeline.calibrated_classifiers_[0].estimator
            else:
                inner = fitted_pipeline
            if hasattr(inner, "named_steps"):
                return inner.named_steps.get("model", None)
            return inner
        except Exception:
            return None

    model = _get_inner_model(pipeline)

    # ── Essai SHAP (tree-based models) ───────────────────────
    if HAS_SHAP and model is not None:
        try:
            explainer = shap.TreeExplainer(model)
            # On a besoin du vecteur déjà transformé par le scaler
            def _get_scaler(fitted_pipeline):
                try:
                    if hasattr(fitted_pipeline, "calibrated_classifiers_"):
                        inner = fitted_pipeline.calibrated_classifiers_[0].estimator
                    else:
                        inner = fitted_pipeline
                    if hasattr(inner, "named_steps"):
                        return inner.named_steps.get("scaler", None)
                except Exception:
                    pass
                return None
            scaler = _get_scaler(pipeline)
            x_scaled = scaler.transform(x.reshape(1, -1))[0] if scaler else x
            shap_vals = explainer.shap_values(x_scaled.reshape(1, -1))
            # Pour les classifieurs binaires : shap_values peut être liste [neg, pos]
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
            contributions = dict(zip(feat_cols, shap_vals[0].tolist()))
        except Exception:
            contributions = None
    else:
        contributions = None

    # ── Fallback : feature_importances_ × valeur normalisée ──
    if contributions is None and model is not None:
        try:
            if hasattr(model, "feature_importances_"):
                fi = model.feature_importances_
            elif hasattr(model, "coef_"):
                fi = np.abs(model.coef_[0])
            else:
                fi = None
            if fi is not None and len(fi) == len(feat_cols):
                # Pondère par la valeur de la feature (normalisée par sa médiane)
                medians = enc["feature_medians"]
                x_norm  = np.array([
                    x[i] / (abs(medians.get(col, 1.0)) + 1e-8)
                    for i, col in enumerate(feat_cols)
                ])
                scores = fi * np.abs(x_norm)
                contributions = dict(zip(feat_cols, scores.tolist()))
        except Exception:
            contributions = None

    if contributions is None:
        return {}

    # Garder les 12 plus importantes (valeur absolue)
    top12 = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)[:12]
    return {k: round(v, 6) for k, v in top12}


def predict_gab_risk(features: dict) -> dict:
    """
    Prédit le risque de panne d'un GAB pour J0.

    Parameters
    ----------
    features : dict
        Features brutes du GAB (voir l'interface dans le docstring du module).

    Returns
    -------
    dict avec les clés :
        probability          : float — probabilité calibrée de panne sous 48h
        risk_level           : str   — FAIBLE | MODÉRÉ | ÉLEVÉ | CRITIQUE
        threshold_economic   : float — seuil économique optimal du modèle déployé
        threshold_f1         : float — seuil F1-optimal du modèle déployé
        decision             : str   — OK | SURVEILLER | INTERVENTION
        feature_contributions: dict  — top 12 contributions au score
        model_name           : str   — nom du modèle champion
        model_version        : str   — hash SHA-256 du best_model.pkl
        score_pct            : float — score en % (0–100) pour l'affichage gauge
        couleur              : str   — hex couleur pour l'affichage
        niveau               : str   — alias risk_level (rétro-compat. ScoringLive)
        recommandation       : str   — texte d'action
        contributions        : dict  — alias feature_contributions (rétro-compat.)
        score                : float — probabilité (alias probability)
    """
    _lazy_load()
    artifact = _MODEL_ARTIFACT

    # ── Construction du vecteur de features ─────────────────
    x, feat_cols = _build_feature_vector(features)

    # ── Prédiction ───────────────────────────────────────────
    pipeline = artifact["pipeline"]
    prob = float(pipeline.predict_proba(x.reshape(1, -1))[0, 1])

    # ── Niveaux de risque ─────────────────────────────────────
    # Les seuils sont basés sur la distribution des probabilités calibrées.
    # Note : on utilise le seuil économique pour la décision opérationnelle.
    threshold_eco = artifact.get("threshold_economic", 0.35)
    threshold_f1  = artifact.get("threshold_f1", 0.40)

    # Niveaux définis par rapport aux quantiles de la distribution
    # (calibrés empiriquement sur le test set 2023)
    if prob < 0.15:
        risk_level    = "FAIBLE"
        couleur       = "#6ba88a"
        decision      = "OK"
        recommandation = "Fonctionnement normal. Aucune action requise."
    elif prob < threshold_eco:
        risk_level    = "MODÉRÉ"
        couleur       = "#5a8fc4"
        decision      = "SURVEILLER"
        recommandation = "Surveiller les métriques les prochains jours."
    elif prob < 0.65:
        risk_level    = "ÉLEVÉ"
        couleur       = "#e8a045"
        decision      = "INTERVENTION"
        recommandation = "Planifier une maintenance préventive sous 48h."
    else:
        risk_level    = "CRITIQUE"
        couleur       = "#d4645a"
        decision      = "INTERVENTION"
        recommandation = "Intervention urgente recommandée — risque de panne imminent."

    # ── Contributions ─────────────────────────────────────────
    contributions = _compute_contributions(x, feat_cols)

    return {
        # Champs principaux
        "probability":           prob,
        "risk_level":            risk_level,
        "threshold_economic":    threshold_eco,
        "threshold_f1":          threshold_f1,
        "decision":              decision,
        "feature_contributions": contributions,
        "model_name":            artifact.get("model_name", "unknown"),
        "model_version":         artifact.get("model_hash", "n/a"),
        # Champs de rétro-compatibilité avec ScoringLive.jsx
        "score":                 prob,
        "score_pct":             round(prob * 100, 2),
        "couleur":               couleur,
        "niveau":                risk_level,
        "recommandation":        recommandation,
        "contributions":         contributions,
    }


# ══════════════════════════════════════════════════════════
# TESTS UNITAIRES MINIMAUX
# Vérifie que le module est opérationnel et que le schéma de
# sortie est correct avant d'intégrer dans le backend FastAPI.
# ══════════════════════════════════════════════════════════
def _run_tests():
    """Tests unitaires basiques sur le module d'inférence."""
    print("=" * 60)
    print("  TESTS UNITAIRES — inference.py")
    print("=" * 60)

    EXPECTED_KEYS = {
        "probability", "risk_level", "threshold_economic", "threshold_f1",
        "decision", "feature_contributions", "model_name", "model_version",
        "score", "score_pct", "couleur", "niveau", "recommandation", "contributions",
    }
    VALID_RISK_LEVELS = {"FAIBLE", "MODÉRÉ", "ÉLEVÉ", "CRITIQUE"}
    VALID_DECISIONS   = {"OK", "SURVEILLER", "INTERVENTION"}

    test_cases = [
        # Cas normal — GAB récent avec peu d'erreurs
        {
            "name": "GAB nominal",
            "input": {
                "ville": "Casablanca", "type_gab": "NCR",
                "environnement": "Agence_Interieure", "age": 2,
                "erreurs_lecteur": 0, "erreurs_dist": 0,
                "temperature": 30.0, "jours_maint": 14, "nb_tx": 150,
                "taux_erreur_tx": 0.01, "latence_ms": 60.0, "deconnexions": 0,
                "erreurs_roll7": 0.2, "temp_roll7": 29.5,
            },
        },
        # Cas à risque élevé — vieux GAB, température haute, beaucoup d'erreurs
        {
            "name": "GAB critique",
            "input": {
                "ville": "Agadir", "type_gab": "Wincor",
                "environnement": "Site_Isole", "age": 9,
                "erreurs_lecteur": 15, "erreurs_dist": 8,
                "temperature": 58.0, "jours_maint": 300, "nb_tx": 40,
                "taux_erreur_tx": 0.25, "latence_ms": 800.0, "deconnexions": 5,
                "erreurs_roll7": 12.0, "temp_roll7": 55.0,
            },
        },
        # Cas avec ville inconnue → fallback encoding
        {
            "name": "Ville inconnue",
            "input": {
                "ville": "VilleInexistante", "type_gab": "Hyosung",
                "environnement": "Centre_Commercial", "age": 5,
                "erreurs_lecteur": 3, "erreurs_dist": 1,
                "temperature": 38.0, "jours_maint": 90, "nb_tx": 100,
                "taux_erreur_tx": 0.05, "latence_ms": 100.0, "deconnexions": 1,
                "erreurs_roll7": 2.5, "temp_roll7": 37.0,
            },
        },
    ]

    all_passed = True
    for tc in test_cases:
        try:
            result = predict_gab_risk(tc["input"])

            # Vérification du schéma
            missing = EXPECTED_KEYS - set(result.keys())
            assert not missing, f"Clés manquantes : {missing}"

            # Vérification des types et valeurs
            assert 0.0 <= result["probability"] <= 1.0, "Probabilité hors [0,1]"
            assert result["risk_level"] in VALID_RISK_LEVELS, \
                f"risk_level invalide : {result['risk_level']}"
            assert result["decision"] in VALID_DECISIONS, \
                f"decision invalide : {result['decision']}"
            assert isinstance(result["feature_contributions"], dict)
            assert isinstance(result["model_version"], str)
            assert abs(result["probability"] - result["score"]) < 1e-9

            print(f"\n  ✅ {tc['name']}")
            print(f"     prob={result['probability']:.3f}  niveau={result['risk_level']}"
                  f"  décision={result['decision']}")
            print(f"     modèle={result['model_name']}  hash={result['model_version']}")
            top3 = list(result["feature_contributions"].items())[:3]
            if top3:
                print(f"     top3 contributions : {top3}")

        except Exception as e:
            print(f"\n  ❌ {tc['name']} ÉCHOUÉ : {e}")
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("  TOUS LES TESTS PASSÉS ✅")
    else:
        print("  CERTAINS TESTS ONT ÉCHOUÉ ❌")
    print("=" * 60)
    return all_passed


# ══════════════════════════════════════════════════════════
# POINT D'ENTRÉE CLI
# python scripts/inference.py              → lance les tests
# python scripts/inference.py --demo      → affiche un résultat complet
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    args = sys.argv[1:]

    if "--demo" in args:
        print("\n[DEMO] Prédiction sur un GAB dégradé...\n")
        result = predict_gab_risk({
            "ville": "Marrakech", "type_gab": "Wincor",
            "environnement": "Site_Isole", "age": 7,
            "erreurs_lecteur": 8, "erreurs_dist": 3,
            "temperature": 52.0, "jours_maint": 180, "nb_tx": 60,
            "taux_erreur_tx": 0.12, "latence_ms": 350.0, "deconnexions": 3,
            "erreurs_roll7": 6.5, "temp_roll7": 50.0,
        })
        import json
        # Exclure les dicts volumineux pour l'affichage
        display = {k: v for k, v in result.items() if k not in ("contributions",)}
        print(json.dumps(display, indent=2, ensure_ascii=False))
        print("\nContributions (top 12) :")
        for feat, val in result["contributions"].items():
            print(f"  {feat:<45} {val:+.5f}")
    else:
        _run_tests()
