"""
Shared heuristic scoring utilities — GAB Intelligence
Used by /api/scoring (fallback), monitoring simulation, and /api/batch-score.
Single source of truth for weights, thresholds, and lookup tables.
"""
import numpy as np

# ── Risk lookup tables ────────────────────────────────────────────────────────
TAUX_VILLE: dict[str, float] = {
    "Safi": 0.110,        "Agadir": 0.105,      "Casablanca": 0.103,
    "El Jadida": 0.099,   "Kénitra": 0.099,      "Beni Mellal": 0.098,
    "Rabat": 0.096,       "Marrakech": 0.096,   "Meknès": 0.095,
    "Fès": 0.094,         "Oujda": 0.094,        "Tanger": 0.093,
    "Tétouan": 0.092,
}
RISQUE_ENV: dict[str, float] = {
    "Site_Isole": 1.0,
    "Centre_Commercial": 0.7,
    "Agence_Facade": 0.6,
    "Agence_Interieure": 0.5,
}
RISQUE_TYPE: dict[str, float] = {
    "Wincor": 1.0,
    "NCR": 0.7,
    "Hyosung": 0.65,
    "Diebold": 0.6,
}

# ── Valid value sets (for input validation) ───────────────────────────────────
VILLES_VALID = set(TAUX_VILLE.keys())
ENVS_VALID   = set(RISQUE_ENV.keys())
TYPES_VALID  = set(RISQUE_TYPE.keys())

# ── Score thresholds ──────────────────────────────────────────────────────────
SEUIL_CRITIQUE = 0.65
SEUIL_ELEVE    = 0.35
SEUIL_MODERE   = 0.15

# ── Level → display config ────────────────────────────────────────────────────
LEVEL_CONFIG: dict[str, dict] = {
    "CRITIQUE": {
        "couleur": "#d4645a",
        "recommandation": "Intervention immédiate requise",
    },
    "ÉLEVÉ": {
        "couleur": "#e8a045",
        "recommandation": "Planifier une visite dans les 48h",
    },
    "MODÉRÉ": {
        "couleur": "#5a8fc4",
        "recommandation": "Surveiller l'évolution cette semaine",
    },
    "FAIBLE": {
        "couleur": "#6ba88a",
        "recommandation": "Aucune action immédiate nécessaire",
    },
}


def level_from_score(score: float) -> tuple[str, str, str]:
    """Return (niveau, couleur, recommandation) for a given probability score."""
    if score >= SEUIL_CRITIQUE:
        n = "CRITIQUE"
    elif score >= SEUIL_ELEVE:
        n = "ÉLEVÉ"
    elif score >= SEUIL_MODERE:
        n = "MODÉRÉ"
    else:
        n = "FAIBLE"
    cfg = LEVEL_CONFIG[n]
    return n, cfg["couleur"], cfg["recommandation"]


def score_heuristic(
    erreurs_lecteur: float,
    erreurs_dist: float,
    temperature: float,
    jours_maint: float,
    nb_tx: float,
    taux_erreur_tx: float,
    age: float,
    erreurs_roll7: float,
    temp_roll7: float,
    ville: str,
    environnement: str,
    type_gab: str,
) -> tuple[float, dict[str, float]]:
    """
    Heuristic scoring aligned with ML feature importance (data-driven weights).
    Returns (score ∈ [0,1], contributions_dict).

    Weights source: feature_importance.csv (Logistic Regression, normalized):
      jours_depuis_maintenance: 0.367 → 0.22
      score_negligence: 0.237          → 0.14
      erreurs_lecteur_carte: 0.225     → 0.14
      temp_tendance_7j: 0.189          → 0.11
      temperature_interne: 0.150       → 0.09
      age_annees: 0.150                → 0.09
      risque_materiel: 0.122           → 0.07
      taux_erreur_tx: 0.094            → 0.06
      erreurs_lecteur_tendance: 0.011  → 0.03
      ville_risk_encoding: 0.038       → 0.02
      environnement: ~0.066            → 0.02
      type_gab: ~0.058                 → 0.01
    """
    ratio_err = (erreurs_lecteur + erreurs_dist) / (nb_tx + 1)
    sc_negl   = jours_maint * np.log1p(erreurs_roll7)
    rq_mat    = erreurs_lecteur * np.log1p(age)
    tend_temp = max(0.0, (temperature - temp_roll7) / (temp_roll7 + 1e-8))
    tend_lect = max(0.0, (erreurs_lecteur - erreurs_roll7) / (erreurs_roll7 + 1e-8))

    contrib = {
        "Ancienneté maint.":    (jours_maint / 365) * 0.22,
        "Négligence maint.":    (sc_negl / 1000) * 0.14,
        "Ratio erreurs":        ratio_err * 0.14,
        "Tendance température": tend_temp * 0.11,
        "Température":          (temperature / 65) * 0.09,
        "Âge GAB":              (age / 10) * 0.09,
        "Risque matériel":      (rq_mat / 30) * 0.07,
        "Taux erreur tx":       taux_erreur_tx * 0.06,
        "Tendance lecteur":     tend_lect * 0.03,
        "Risque ville":         TAUX_VILLE.get(ville, 0.097) * 0.02,
        "Risque env.":          RISQUE_ENV.get(environnement, 0.5) * 0.02,
        "Risque type":          RISQUE_TYPE.get(type_gab, 0.65) * 0.01,
    }

    score_raw = sum(contrib.values())
    # Sigmoid calibré : raw≈0.05→0.05, raw≈0.25→0.27, raw≈0.45→0.73
    score = float(1.0 / (1.0 + np.exp(-10.0 * (score_raw - 0.35))))
    return score, {k: round(v, 4) for k, v in contrib.items()}
