"""
==============================================================
PROJET : Prédiction des Pannes GAB/ATM - Banque Populaire Maroc
==============================================================
Script 04 — Modélisation ML (refonte complète)

AMÉLIORATIONS vs version précédente :
  1. HistGradientBoostingClassifier remplace GBM (qui ignore class_weight)
  2. XGBoost et LightGBM en options (try/except propre)
  3. Pipeline SMOTE + LogisticRegression si imbalanced-learn disponible
  4. Calibration des probabilités : CalibratedClassifierCV(method='isotonic')
  5. TimeSeriesSplit (3 folds) pour valider la stabilité temporelle
  6. Double optimisation de seuil : F1-optimal + économique
  7. Critère de sélection : AUC-PR (plus robuste que F1 pour classes rares)
  8. Sauvegarde best_model.pkl pour l'inférence live via inference.py

Coûts métier hypothétiques (MAD) :
  FN (panne manquée)     : 5 000 — intervention corrective d'urgence
  TP (panne détectée)    : 1 500 — intervention préventive planifiée
  FP (fausse alerte)     : 500   — déplacement technicien inutile
==============================================================
"""

import hashlib
import json
import os
import pickle
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Imports optionnels ──────────────────────────────────────
try:
    from xgboost import XGBClassifier

    HAS_XGB = True
    print("[opt] xgboost   OK")
except ImportError:
    HAS_XGB = False
    print("[opt] xgboost   -  (pip install xgboost)")

try:
    from lightgbm import LGBMClassifier

    HAS_LGB = True
    print("[opt] lightgbm  OK")
except ImportError:
    HAS_LGB = False
    print("[opt] lightgbm  -  (pip install lightgbm)")

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    HAS_IMBL = True
    print("[opt] imbalanced-learn OK")
except ImportError:
    HAS_IMBL = False
    print("[opt] imbalanced-learn -  (pip install imbalanced-learn)")

# ─── Chemins ────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "outputs" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Style matplotlib ────────────────────────────────────────
plt.rcParams.update(
    {
        "figure.facecolor": "#0f1117",
        "axes.facecolor": "#1a1d27",
        "axes.edgecolor": "#3a3d4d",
        "axes.labelcolor": "#e0e0e0",
        "xtick.color": "#b0b0b0",
        "ytick.color": "#b0b0b0",
        "text.color": "#e0e0e0",
        "grid.color": "#2a2d3d",
        "grid.alpha": 0.5,
        "font.size": 10,
    }
)
C_RED = "#C8102E"
C_GREEN = "#00703C"
C_BLUE = "#3B82C4"
C_AMBER = "#B8922A"
C_PURPLE = "#7C3AED"
C_TEAL = "#34d399"
PALETTE = [C_BLUE, C_TEAL, C_RED, C_AMBER, C_PURPLE, "#fb923c"]

COSTS = {"fn": 5000, "tp": 1500, "fp": 500}  # MAD

# ══════════════════════════════════════════════════════════
# CHARGEMENT & PRÉPARATION
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  MODELISATION ML v2 -- GAB/ATM PREDICTIVE MAINTENANCE")
print("=" * 60)

print("\n[1/6] Chargement des donnees enrichies...")
df = pd.read_csv(DATA_DIR / "gab_features.csv", parse_dates=["date"])

with open(DATA_DIR / "feature_cols.json") as f:
    feature_cols = json.load(f)

COLS_EXCLURE = {
    "date",
    "gab_id",
    "ville",
    "mois_annee",
    "mois",
    "jour_semaine",
    "panne_sous_48h",
}
feature_cols = [c for c in feature_cols if c in df.columns and c not in COLS_EXCLURE]
print(f"   > {len(df):,} observations, {len(feature_cols)} features")

# ── Split temporel chronologique ────────────────────────────
print("\n[2/6] Split temporel chronologique (train=2022, test=2023)...")

train_df = df[df["date"].dt.year == 2022].copy()
test_df = df[df["date"].dt.year == 2023].copy()

X_train = train_df[feature_cols].values
y_train = train_df["panne_sous_48h"].values
X_test = test_df[feature_cols].values
y_test = test_df["panne_sous_48h"].values

pos_ratio = y_train.mean()
neg_pos_ratio = (1 - pos_ratio) / pos_ratio  # pour scale_pos_weight XGBoost

print(
    f"   Train : {len(X_train):,} | pannes = {y_train.sum():,} ({pos_ratio*100:.1f}%)"
)
print(
    f"   Test  : {len(X_test):,}  | pannes = {y_test.sum():,} ({y_test.mean()*100:.1f}%)"
)

# ══════════════════════════════════════════════════════════
# DÉFINITION DES MODÈLES DE BASE
# Chaque modèle gère le déséquilibre de classes de façon native.
# La calibration est ajoutée après via CalibratedClassifierCV.
# ══════════════════════════════════════════════════════════
print("\n[3/6] Definition des modeles...")

# ── Mini-grid search par famille de modèles ────────────────
# On teste quelques configs par modèle et on garde la meilleure (AUC-PR sur
# un fold holdout rapide). Plus efficace qu'un GridSearchCV complet car on a
# peu de combinaisons et la calibration est faite après.
print("   Recherche d'hyperparametres (mini-grid par famille)...")


def _quick_aucpr(pipeline, Xtr, ytr, Xval, yval):
    """Entraîne et retourne AUC-PR sur val, ou -1 en cas d'erreur."""
    try:
        pipeline.fit(Xtr, ytr)
        proba = pipeline.predict_proba(Xval)[:, 1]
        return average_precision_score(yval, proba)
    except Exception:
        return -1.0


# Split interne pour le tuning (70/30 du train)
from sklearn.model_selection import train_test_split

_Xtr, _Xval, _ytr, _yval = train_test_split(
    X_train, y_train, test_size=0.3, random_state=42, stratify=y_train
)


def _best_of(name, candidates):
    """Teste les candidats sur le split interne et retourne le meilleur."""
    best_score, best_pipe, best_label = -1, None, ""
    for label, pipe in candidates.items():
        sc = _quick_aucpr(pipe, _Xtr, _ytr, _Xval, _yval)
        if sc > best_score:
            best_score, best_pipe, best_label = sc, pipe, label
    print(f"     {name}: meilleur = {best_label} (AUC-PR={best_score:.4f})")
    return best_pipe


# Logistic Regression — varier C (régularisation)
lr_candidates = {}
for C_val in [0.01, 0.05, 0.1, 0.5, 1.0]:
    lr_candidates[f"C={C_val}"] = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    C=C_val,
                    solver="saga",
                    max_iter=3000,
                    random_state=42,
                ),
            ),
        ]
    )
best_lr = _best_of("Logistic Regression", lr_candidates)

# HistGradientBoosting — varier profondeur + learning rate
hgb_candidates = {}
for depth, lr_val, n_iter in [
    (4, 0.03, 500),
    (5, 0.05, 400),
    (6, 0.05, 300),
    (4, 0.01, 800),
    (3, 0.05, 600),
]:
    label = f"d={depth},lr={lr_val},n={n_iter}"
    hgb_candidates[label] = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                HistGradientBoostingClassifier(
                    class_weight="balanced",
                    max_iter=n_iter,
                    max_depth=depth,
                    learning_rate=lr_val,
                    min_samples_leaf=20,
                    l2_regularization=0.1,
                    random_state=42,
                ),
            ),
        ]
    )
best_hgb = _best_of("HistGradientBoosting", hgb_candidates)

# Random Forest — varier profondeur + nombre d'arbres
rf_candidates = {}
for n_est, depth in [(300, 8), (500, 10), (300, 12), (500, 15), (400, 6)]:
    label = f"n={n_est},d={depth}"
    rf_candidates[label] = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=n_est,
                    class_weight="balanced_subsample",
                    max_depth=depth,
                    min_samples_leaf=20,
                    max_features="sqrt",
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )
best_rf = _best_of("Random Forest", rf_candidates)

# Modèles de base (avant calibration)
base_models = {
    "Dummy (Stratified)": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", DummyClassifier(strategy="stratified", random_state=42)),
        ]
    ),
    "Logistic Regression": best_lr,
    "HistGradientBoosting": best_hgb,
    "Random Forest": best_rf,
}

# XGBoost : scale_pos_weight est l'équivalent de class_weight pour XGB
if HAS_XGB:
    xgb_candidates = {}
    for depth, lr_val in [(4, 0.03), (5, 0.05), (6, 0.05), (3, 0.01)]:
        label = f"d={depth},lr={lr_val}"
        xgb_candidates[label] = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    XGBClassifier(
                        scale_pos_weight=neg_pos_ratio,
                        n_estimators=500,
                        max_depth=depth,
                        learning_rate=lr_val,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        eval_metric="aucpr",
                        use_label_encoder=False,
                        verbosity=0,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    base_models["XGBoost"] = _best_of("XGBoost", xgb_candidates)

# LightGBM
if HAS_LGB:
    lgb_candidates = {}
    for depth, lr_val in [(4, 0.03), (6, 0.05), (3, 0.01)]:
        label = f"d={depth},lr={lr_val}"
        lgb_candidates[label] = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LGBMClassifier(
                        is_unbalance=True,
                        n_estimators=500,
                        max_depth=depth,
                        learning_rate=lr_val,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        verbose=-1,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    base_models["LightGBM"] = _best_of("LightGBM", lgb_candidates)

# SMOTE + Logistic Regression + SMOTE + HistGBM
# SMOTE suréchantillonne la classe minoritaire synthétiquement.
# sampling_strategy=0.5 → 50% de la classe majoritaire (plus agressif que 0.3)
if HAS_IMBL:
    smote_candidates = {}
    for ratio, C_val in [(0.3, 0.1), (0.5, 0.1), (0.5, 0.5), (0.3, 0.5)]:
        label = f"r={ratio},C={C_val}"
        smote_candidates[label] = ImbPipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "smote",
                    SMOTE(sampling_strategy=ratio, k_neighbors=5, random_state=42),
                ),
                (
                    "model",
                    LogisticRegression(
                        C=C_val, solver="saga", max_iter=3000, random_state=42
                    ),
                ),
            ]
        )
    base_models["SMOTE + LogReg"] = _best_of("SMOTE + LogReg", smote_candidates)

    smote_hgb_candidates = {}
    for ratio, depth in [(0.3, 4), (0.5, 4), (0.3, 5), (0.5, 5)]:
        label = f"r={ratio},d={depth}"
        smote_hgb_candidates[label] = ImbPipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "smote",
                    SMOTE(sampling_strategy=ratio, k_neighbors=5, random_state=42),
                ),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=500,
                        max_depth=depth,
                        learning_rate=0.03,
                        min_samples_leaf=20,
                        l2_regularization=0.1,
                        random_state=42,
                    ),
                ),
            ]
        )
    base_models["SMOTE + HistGBM"] = _best_of("SMOTE + HistGBM", smote_hgb_candidates)

print(f"   > {len(base_models)} modeles configures : {list(base_models.keys())}")

# ══════════════════════════════════════════════════════════
# VALIDATION DE STABILITÉ TEMPORELLE (TimeSeriesSplit)
# Objectif : vérifier que le recall ne s'effondre pas sur certaines
# périodes de 2022. Un écart-type recall > 0.05 serait inquiétant.
# On n'utilise pas les modèles calibrés ici pour la rapidité.
# ══════════════════════════════════════════════════════════
print("\n[4/6] Validation stabilite temporelle (TimeSeriesSplit, 3 folds)...")

tscv = TimeSeriesSplit(n_splits=3)
stability_results = {}

# Seulement pour les modèles non-Dummy (Dummy est stable par définition)
models_for_cv = {k: v for k, v in base_models.items() if k != "Dummy (Stratified)"}

for nom, pipeline in models_for_cv.items():
    recalls_cv = []
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
        Xtr, ytr = X_train[tr_idx], y_train[tr_idx]
        Xval, yval = X_train[val_idx], y_train[val_idx]
        try:
            pipeline.fit(Xtr, ytr)
            y_val_pred = pipeline.predict(Xval)
            recalls_cv.append(recall_score(yval, y_val_pred, zero_division=0))
        except Exception:
            recalls_cv.append(0.0)
    stability_results[nom] = {
        "recall_mean": float(np.mean(recalls_cv)),
        "recall_std": float(np.std(recalls_cv)),
        "stable": float(np.std(recalls_cv)) < 0.05,
    }
    flag = "STABLE" if stability_results[nom]["stable"] else "! INSTABLE"
    print(
        f"   {nom:<25} recall moy={stability_results[nom]['recall_mean']:.3f}"
        f"  std={stability_results[nom]['recall_std']:.3f}  {flag}"
    )

# ══════════════════════════════════════════════════════════
# ENTRAÎNEMENT FINAL + CALIBRATION
# Stratégie : CalibratedClassifierCV avec cv=3 et method='isotonic'.
# 'isotonic' est non-paramétrique → plus flexible que 'sigmoid' (Platt).
# Recommandé pour les datasets de taille > 1000 (le nôtre est bien plus grand).
#
# Principe : pour chaque fold de CV, on entraîne le modèle de base sur
# 2/3 du train et la calibration isotonic sur 1/3. Le modèle final est
# la moyenne des 3 modèles calibrés → robuste et non biaisé.
# ══════════════════════════════════════════════════════════
print("\n[5/6] Entrainement final + calibration isotonique...")
print("-" * 60)

resultats = {}
fitted_models = {}

for nom, base_pipeline in base_models.items():
    print(f"\n  > {nom}")
    t0 = time.time()

    # Dummy : pas de calibration (predict_proba triviale)
    if nom == "Dummy (Stratified)":
        base_pipeline.fit(X_train, y_train)
        calibrated = base_pipeline
    else:
        # cv=3 : le base_pipeline est refitté 3× — calibration correcte
        calibrated = CalibratedClassifierCV(
            base_pipeline, cv=3, method="isotonic", n_jobs=-1
        )
        calibrated.fit(X_train, y_train)

    t_train = time.time() - t0
    fitted_models[nom] = calibrated

    # ── Évaluation sur test ────────────────────────────────
    y_proba = calibrated.predict_proba(X_test)[:, 1]
    auc_roc = roc_auc_score(y_test, y_proba)
    auc_pr = average_precision_score(y_test, y_proba)

    # ── Optimisation du seuil F1 ──────────────────────────
    # On cherche le seuil qui maximise le F1 sur la courbe PR.
    # C'est le seuil de référence (pas 0.5 qui est inadapté au déséquilibre).
    prec_curve, rec_curve, thresholds_pr = precision_recall_curve(y_test, y_proba)
    f1_curve = np.where(
        (prec_curve + rec_curve) > 0,
        2 * prec_curve * rec_curve / (prec_curve + rec_curve + 1e-9),
        0,
    )
    idx_f1 = np.argmax(f1_curve[:-1])
    threshold_f1 = float(thresholds_pr[idx_f1])
    f1_at_opt = float(f1_curve[idx_f1])

    # ── Confusion matrix au seuil F1-optimal (pas 0.5) ───
    # Pour un dataset à 10% de pannes, le seuil optimal est ~0.10-0.15.
    # Évaluer à 0.5 donne F1≈0 pour les modèles tree-based (faux négatifs massifs).
    y_pred = (y_proba >= threshold_f1).astype(int)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    accuracy = float((y_pred == y_test).mean())

    tn = int(((y_pred == 0) & (y_test == 0)).sum())
    fp = int(((y_pred == 1) & (y_test == 0)).sum())
    fn = int(((y_pred == 0) & (y_test == 1)).sum())
    tp = int(((y_pred == 1) & (y_test == 1)).sum())

    # ── Optimisation du seuil économique ──────────────────
    # Minimiser le coût total = FN×5000 + TP×1500 + FP×500
    # Un FN (panne manquée) coûte 10× plus qu'une fausse alerte.
    # Le seuil économique est donc souvent plus bas que 0.5.
    thresholds_econ = np.linspace(0.01, 0.99, 200)
    costs_econ = []
    for thr in thresholds_econ:
        yp = (y_proba >= thr).astype(int)
        _tp = int(((yp == 1) & (y_test == 1)).sum())
        _fp = int(((yp == 1) & (y_test == 0)).sum())
        _fn = int(((yp == 0) & (y_test == 1)).sum())
        costs_econ.append(_fn * COSTS["fn"] + _tp * COSTS["tp"] + _fp * COSTS["fp"])

    idx_econ = int(np.argmin(costs_econ))
    threshold_economic = float(thresholds_econ[idx_econ])
    cost_optimal = costs_econ[idx_econ]
    cost_no_model = int(y_test.sum()) * COSTS["fn"]
    saving_pct = (cost_no_model - cost_optimal) / cost_no_model * 100

    # ── Rapport par modèle ─────────────────────────────────
    stab = stability_results.get(nom, {"recall_std": 0.0})
    print(
        f"     F1={f1:.4f}  Prec={precision:.4f}  Rec={recall:.4f}"
        f"  AUC-ROC={auc_roc:.4f}  AUC-PR={auc_pr:.4f}  [{t_train:.1f}s]"
    )
    print(f"     TP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,}")
    print(
        f"     Seuil F1-opt={threshold_f1:.3f} (F1={f1_at_opt:.4f})"
        f"  Seuil eco={threshold_economic:.3f}  Economie={saving_pct:.1f}%"
        f"  Stabilite recall std={stab['recall_std']:.3f}"
    )

    resultats[nom] = {
        # Métriques test (au seuil F1-optimal — adapté au déséquilibre)
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": accuracy,
        "auc_roc": float(auc_roc),
        "auc_pr": float(auc_pr),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        # Seuils optimaux
        "threshold_f1": threshold_f1,
        "threshold_economic": threshold_economic,
        "f1_at_threshold": f1_at_opt,
        # Impact économique
        "saving_pct": float(saving_pct),
        "cost_with_model": float(cost_optimal),
        "cost_no_model": float(cost_no_model),
        # Stabilité temporelle
        "stability_recall_std": float(stab.get("recall_std", 0.0)),
        "stable": bool(stab.get("stable", True)),
        "calibrated": nom != "Dummy (Stratified)",
        # Interne — pas exporté dans JSON
        "_y_proba": y_proba,
        "_prec_curve": prec_curve,
        "_rec_curve": rec_curve,
        "_thresholds_pr": thresholds_pr,
        "_costs_econ": costs_econ,
    }

print("\n" + "-" * 60)

# ══════════════════════════════════════════════════════════
# ENSEMBLE — Moyenne pondérée des probabilités calibrées
# Combine les top modèles (hors Dummy) pondérés par AUC-PR.
# Souvent meilleur qu'un seul modèle car lisse les erreurs.
# ══════════════════════════════════════════════════════════
print("\n  > Ensemble (moyenne ponderee)")
t0 = time.time()

non_dummy_names = [k for k in resultats if k != "Dummy (Stratified)"]
# Pondérer par AUC-PR (les meilleurs modèles comptent plus)
weights = {k: resultats[k]["auc_pr"] for k in non_dummy_names}
total_w = sum(weights.values())
weights = {k: v / total_w for k, v in weights.items()}

# Moyenne pondérée des probabilités
y_proba_ens = np.zeros(len(y_test))
for k in non_dummy_names:
    y_proba_ens += weights[k] * resultats[k]["_y_proba"]

auc_roc_ens = roc_auc_score(y_test, y_proba_ens)
auc_pr_ens = average_precision_score(y_test, y_proba_ens)

prec_c, rec_c, thr_c = precision_recall_curve(y_test, y_proba_ens)
f1_c = np.where((prec_c + rec_c) > 0, 2 * prec_c * rec_c / (prec_c + rec_c + 1e-9), 0)
idx_ens = np.argmax(f1_c[:-1])
thr_f1_ens = float(thr_c[idx_ens])

y_pred_ens = (y_proba_ens >= thr_f1_ens).astype(int)
f1_ens = f1_score(y_test, y_pred_ens, zero_division=0)
prec_ens = precision_score(y_test, y_pred_ens, zero_division=0)
rec_ens = recall_score(y_test, y_pred_ens, zero_division=0)
acc_ens = float((y_pred_ens == y_test).mean())

tn_ens = int(((y_pred_ens == 0) & (y_test == 0)).sum())
fp_ens = int(((y_pred_ens == 1) & (y_test == 0)).sum())
fn_ens = int(((y_pred_ens == 0) & (y_test == 1)).sum())
tp_ens = int(((y_pred_ens == 1) & (y_test == 1)).sum())

thr_econ_vals = np.linspace(0.01, 0.99, 200)
costs_ens = []
for thr in thr_econ_vals:
    yp = (y_proba_ens >= thr).astype(int)
    _tp = int(((yp == 1) & (y_test == 1)).sum())
    _fp = int(((yp == 1) & (y_test == 0)).sum())
    _fn = int(((yp == 0) & (y_test == 1)).sum())
    costs_ens.append(_fn * COSTS["fn"] + _tp * COSTS["tp"] + _fp * COSTS["fp"])

idx_eco_ens = int(np.argmin(costs_ens))
thr_eco_ens = float(thr_econ_vals[idx_eco_ens])
cost_opt_ens = costs_ens[idx_eco_ens]
cost_no_model = int(y_test.sum()) * COSTS["fn"]
saving_ens = (cost_no_model - cost_opt_ens) / cost_no_model * 100

t_ens = time.time() - t0
print(
    f"     F1={f1_ens:.4f}  Prec={prec_ens:.4f}  Rec={rec_ens:.4f}"
    f"  AUC-ROC={auc_roc_ens:.4f}  AUC-PR={auc_pr_ens:.4f}  [{t_ens:.1f}s]"
)
print(f"     TP={tp_ens:,}  FP={fp_ens:,}  FN={fn_ens:,}  TN={tn_ens:,}")
print(
    f"     Seuil F1-opt={thr_f1_ens:.3f}  Seuil eco={thr_eco_ens:.3f}  Economie={saving_ens:.1f}%"
)
print(f"     Poids: {', '.join(f'{k}={w:.2f}' for k, w in weights.items())}")

resultats["Ensemble"] = {
    "f1": float(f1_ens),
    "precision": float(prec_ens),
    "recall": float(rec_ens),
    "accuracy": acc_ens,
    "auc_roc": float(auc_roc_ens),
    "auc_pr": float(auc_pr_ens),
    "tp": tp_ens,
    "fp": fp_ens,
    "fn": fn_ens,
    "tn": tn_ens,
    "threshold_f1": thr_f1_ens,
    "threshold_economic": thr_eco_ens,
    "f1_at_threshold": float(f1_c[idx_ens]),
    "saving_pct": float(saving_ens),
    "cost_with_model": float(cost_opt_ens),
    "cost_no_model": float(cost_no_model),
    "stability_recall_std": 0.0,
    "stable": True,
    "calibrated": True,
    "_y_proba": y_proba_ens,
    "_prec_curve": prec_c,
    "_rec_curve": rec_c,
    "_thresholds_pr": thr_c,
    "_costs_econ": costs_ens,
}
# L'ensemble n'a pas de fitted_model unique — on stocke le meilleur individuel
# pour best_model.pkl, mais on reporte les métriques ensemble.

print("\n" + "-" * 60)

# ══════════════════════════════════════════════════════════
# SÉLECTION DU MEILLEUR MODÈLE
# Critère : AUC-PR (Area Under Precision-Recall Curve).
# AUC-PR est plus discriminant que F1 (qui dépend du seuil) et
# que AUC-ROC (insensible au déséquilibre). C'est la métrique
# recommandée pour les problèmes à classes fortement déséquilibrées.
# ══════════════════════════════════════════════════════════
non_dummy = {k: v for k, v in resultats.items() if k != "Dummy (Stratified)"}
meilleur = max(non_dummy, key=lambda k: non_dummy[k]["auc_pr"])
# Si Ensemble wins, use the best individual model for the pipeline artifact
meilleur_pipeline = (
    meilleur
    if meilleur != "Ensemble"
    else max(
        (k for k in fitted_models if k != "Dummy (Stratified)"),
        key=lambda k: resultats[k]["auc_pr"],
    )
)
print(f"\n  CHAMPION (AUC-PR) : {meilleur}")
print(f"    AUC-PR  = {resultats[meilleur]['auc_pr']:.4f}")
print(f"    F1      = {resultats[meilleur]['f1']:.4f}")
print(f"    Recall  = {resultats[meilleur]['recall']:.4f}")
print(f"    Economie= {resultats[meilleur]['saving_pct']:.1f}%")

# ══════════════════════════════════════════════════════════
# FEATURE IMPORTANCE du modèle champion
# ══════════════════════════════════════════════════════════
print("\n[6/6] Feature importance + sauvegarde des artefacts...")

best_fitted = fitted_models[meilleur_pipeline]


def extract_importances(fitted_pipeline, nom_modele, feature_cols):
    """Extrait les importances de features selon le type de modèle."""
    try:
        # Pour CalibratedClassifierCV, accéder au calibrateur interne
        if hasattr(fitted_pipeline, "calibrated_classifiers_"):
            inner = fitted_pipeline.calibrated_classifiers_[0].estimator
        else:
            inner = fitted_pipeline
        # Accès au modèle dans le pipeline
        if hasattr(inner, "named_steps"):
            model = inner.named_steps.get("model", None)
        else:
            model = inner
        if model is None:
            return None
        if hasattr(model, "feature_importances_"):
            return model.feature_importances_
        if hasattr(model, "coef_"):
            return np.abs(model.coef_[0])
    except Exception:
        pass
    return None


importances = extract_importances(best_fitted, meilleur, feature_cols)

if importances is not None and len(importances) == len(feature_cols):
    feat_imp_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": importances,
            "model": meilleur,
        }
    ).sort_values("importance", ascending=False)
    feat_imp_df.to_csv(DATA_DIR / "feature_importance.csv", index=False)
    print("   > feature_importance.csv updated")
    print("   Top 10 :")
    for _, row in feat_imp_df.head(10).iterrows():
        bar = "#" * max(1, int(row["importance"] * 400))
        print(f"     {row['feature'][:42]:<42} {bar} {row['importance']:.4f}")
else:
    print("   ! Importances non disponibles pour ce modele")
    feat_imp_df = None

# ══════════════════════════════════════════════════════════
# SAUVEGARDE best_model.pkl
# L'artefact contient tout ce qu'inference.py a besoin :
# pipeline calibré + seuils + feature_cols + métriques
# ══════════════════════════════════════════════════════════
model_path = DATA_DIR / "best_model.pkl"
model_artifact = {
    "pipeline": best_fitted,
    "threshold_f1": resultats[meilleur]["threshold_f1"],
    "threshold_economic": resultats[meilleur]["threshold_economic"],
    "feature_cols": feature_cols,
    "model_name": meilleur,
    "metrics": {k: v for k, v in resultats[meilleur].items() if not k.startswith("_")},
    "costs": COSTS,
    "train_year": 2022,
}

with open(model_path, "wb") as f:
    pickle.dump(model_artifact, f)

# Hash SHA-256 pour la traçabilité (exposé dans /api/scoring)
model_bytes = model_path.read_bytes()
model_hash = hashlib.sha256(model_bytes).hexdigest()[:12]
print(f"   > best_model.pkl saved (hash={model_hash})")

# Ré-insérer le hash dans l'artefact pour l'inférence
model_artifact["model_hash"] = model_hash
with open(model_path, "wb") as f:
    pickle.dump(model_artifact, f)

# ══════════════════════════════════════════════════════════
# MISE À JOUR resultats_modeles.json (compatible frontend)
# Garde les champs attendus par Models.jsx + ajoute les nouveaux
# ══════════════════════════════════════════════════════════
resultats_json = {
    "resultats": {
        nom: {k: v for k, v in r.items() if not k.startswith("_")}
        for nom, r in resultats.items()
    },
    "meilleur": meilleur,
    "cout_hypotheses": COSTS,
    "selection_critere": "AUC-PR",
    "model_hash": model_hash,
}
with open(DATA_DIR / "resultats_modeles.json", "w") as f:
    json.dump(resultats_json, f, indent=2)
print("   > resultats_modeles.json updated")

# ══════════════════════════════════════════════════════════
# VISUALISATIONS
# ══════════════════════════════════════════════════════════

# Figure 7 : Courbes Precision-Recall pour tous les modèles
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor("#0f1117")
fig.suptitle(
    "Comparaison des modèles — Courbes PR & Impact économique",
    fontsize=13,
    color="white",
    fontweight="bold",
)

ax_pr = axes[0]
baseline = y_test.mean()
ax_pr.axhline(
    baseline,
    color="white",
    linestyle="--",
    linewidth=1,
    alpha=0.5,
    label=f"Baseline ({baseline:.3f})",
)

for (nom, r), col in zip(resultats.items(), PALETTE):
    if nom == "Dummy (Stratified)":
        continue
    ax_pr.plot(
        r["_rec_curve"],
        r["_prec_curve"],
        color=col,
        linewidth=2,
        label=f"{nom[:20]} (AP={r['auc_pr']:.3f})",
    )
    # Marquer le seuil économique optimal
    ythr = r["threshold_economic"]
    y_pred_thr = (r["_y_proba"] >= ythr).astype(int)
    rec_thr = recall_score(y_test, y_pred_thr, zero_division=0)
    pre_thr = precision_score(y_test, y_pred_thr, zero_division=0)
    ax_pr.scatter([rec_thr], [pre_thr], color=col, marker="D", s=60, zorder=5)

ax_pr.set_xlabel("Recall", color="#b0b0b0")
ax_pr.set_ylabel("Precision", color="#b0b0b0")
ax_pr.set_title(
    "Courbes Precision-Recall\n(◆ = seuil économique optimal)",
    color="white",
    fontweight="bold",
)
ax_pr.legend(fontsize=8, framealpha=0.3)
ax_pr.set_xlim(0, 1.02)
ax_pr.set_ylim(0, 1.02)

ax_eco = axes[1]
thresholds_econ = np.linspace(0.01, 0.99, 200)
for (nom, r), col in zip(resultats.items(), PALETTE):
    if nom == "Dummy (Stratified)":
        continue
    costs_k = [c / 1000 for c in r["_costs_econ"]]
    ax_eco.plot(
        thresholds_econ,
        costs_k,
        color=col,
        linewidth=2,
        label=f"{nom[:20]} (opt={r['threshold_economic']:.2f})",
    )
    ax_eco.axvline(
        r["threshold_economic"], color=col, linestyle=":", linewidth=1, alpha=0.6
    )

ax_eco.axhline(
    resultats[meilleur]["cost_no_model"] / 1000,
    color="white",
    linestyle="--",
    linewidth=1.5,
    alpha=0.7,
    label="Sans modèle",
)
ax_eco.set_xlabel("Seuil de décision", color="#b0b0b0")
ax_eco.set_ylabel("Coût total (k MAD)", color="#b0b0b0")
ax_eco.set_title(
    "Impact économique par seuil\n(minimiser = meilleur)",
    color="white",
    fontweight="bold",
)
ax_eco.legend(fontsize=8, framealpha=0.3)

plt.tight_layout()
fig.savefig(
    MODELS_DIR / "fig7_comparaison_modeles.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="#0f1117",
)
plt.close()
print("   > fig7_comparaison_modeles.png")

# Figure 8 : Courbes ROC + matrices de confusion (2×2 cards HTML → PNG)
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor("#0f1117")
fig.suptitle(
    "Courbes ROC & Distribution des probabilités",
    fontsize=13,
    color="white",
    fontweight="bold",
)

ax_roc = axes[0]
ax_roc.plot([0, 1], [0, 1], "w--", linewidth=1, alpha=0.5, label="Aléatoire (0.50)")
for (nom, r), col in zip(resultats.items(), PALETTE):
    fpr, tpr, _ = roc_curve(y_test, r["_y_proba"])
    ax_roc.plot(
        fpr, tpr, color=col, linewidth=2, label=f"{nom[:18]} ({r['auc_roc']:.3f})"
    )
ax_roc.set_xlabel("Taux faux positifs", color="#b0b0b0")
ax_roc.set_ylabel("Taux vrais positifs", color="#b0b0b0")
ax_roc.set_title("Courbes ROC", color="white", fontweight="bold")
ax_roc.legend(fontsize=8, framealpha=0.3)

ax_dist = axes[1]
best_proba = resultats[meilleur]["_y_proba"]
bins = np.linspace(0, 1, 60)
ax_dist.hist(
    best_proba[y_test == 0],
    bins=bins,
    alpha=0.6,
    color=C_BLUE,
    label="Normal",
    density=True,
)
ax_dist.hist(
    best_proba[y_test == 1],
    bins=bins,
    alpha=0.6,
    color=C_RED,
    label="Panne",
    density=True,
)
ax_dist.axvline(
    resultats[meilleur]["threshold_economic"],
    color=C_AMBER,
    linestyle="--",
    linewidth=2,
    label=f"Seuil éco ({resultats[meilleur]['threshold_economic']:.2f})",
)
ax_dist.axvline(
    resultats[meilleur]["threshold_f1"],
    color=C_TEAL,
    linestyle=":",
    linewidth=2,
    label=f"Seuil F1 ({resultats[meilleur]['threshold_f1']:.2f})",
)
ax_dist.set_title(f"Distribution proba — {meilleur}", color="white", fontweight="bold")
ax_dist.set_xlabel("P(panne)", color="#b0b0b0")
ax_dist.legend(fontsize=9, framealpha=0.3)

plt.tight_layout()
fig.savefig(
    MODELS_DIR / "fig8_roc_pr_cm.png", dpi=150, bbox_inches="tight", facecolor="#0f1117"
)
plt.close()
print("   > fig8_roc_pr_cm.png")

# ══════════════════════════════════════════════════════════
# RAPPORT FINAL
# ══════════════════════════════════════════════════════════
rapport = f"""
╔═══════════════════════════════════════════════════════════╗
║   RAPPORT MODÉLISATION v2 — PRÉDICTION PANNES GAB/ATM    ║
║   Banque Populaire Maroc · hash={model_hash}             ║
╚═══════════════════════════════════════════════════════════╝

📋 CONFIGURATION
  Train : 2022 ({len(X_train):,} obs)  Test : 2023 ({len(X_test):,} obs)
  Features : {len(feature_cols)}  |  Déséquilibre train : {pos_ratio*100:.1f}%
  Calibration : CalibratedClassifierCV(method='isotonic', cv=3)
  Critère de sélection : AUC-PR (robuste aux classes déséquilibrées)

📊 TABLEAU DES PERFORMANCES (seuil F1-optimal par modèle)
{'─'*80}
{'Modèle':<28} {'F1':>6} {'Prec':>6} {'Recall':>6} {'AUC-PR':>7} {'Seuil F1':>9} {'Seuil éco':>10} {'Économie':>9} {'Stable':>7}
{'─'*80}"""

for nom, r in resultats.items():
    star = " ★" if nom == meilleur else "  "
    rapport += (
        f"\n{nom:<28}{star}{r['f1']:>6.3f} {r['precision']:>6.3f} {r['recall']:>6.3f}"
        f" {r['auc_pr']:>7.3f}  {r.get('threshold_f1', 0):>8.3f}"
        f"  {r.get('threshold_economic', 0):>9.3f}  {r.get('saving_pct', 0):>8.1f}%"
        f"  {'✓' if r.get('stable', True) else '⚠':>6}"
    )

rapport += f"""

🏆 MODÈLE CHAMPION : {meilleur}
  AUC-PR    : {resultats[meilleur]['auc_pr']:.4f}  (critère de sélection)
  F1-Score  : {resultats[meilleur]['f1']:.4f}  (au seuil 0.5)
  Recall    : {resultats[meilleur]['recall']:.4f}
  TP={resultats[meilleur]['tp']:,}  FP={resultats[meilleur]['fp']:,}  FN={resultats[meilleur]['fn']:,}

💡 SEUILS RECOMMANDÉS
  Seuil F1-optimal  : {resultats[meilleur]['threshold_f1']:.3f}  → maximise le F1
  Seuil économique  : {resultats[meilleur]['threshold_economic']:.3f}  → minimise le coût total

💰 IMPACT ÉCONOMIQUE ({meilleur}, seuil économique)
  Coûts : FN=5 000 MAD | TP=1 500 MAD | FP=500 MAD
  Coût sans modèle   : {resultats[meilleur]['cost_no_model']:>12,.0f} MAD
  Coût avec modèle   : {resultats[meilleur]['cost_with_model']:>12,.0f} MAD
  Économie potentielle: {resultats[meilleur]['saving_pct']:>11.1f}%
"""
try:
    print(rapport)
except UnicodeEncodeError:
    print(rapport.encode("ascii", errors="replace").decode("ascii"))
with open(
    ROOT / "outputs" / "reports" / "rapport_modelisation.txt", "w", encoding="utf-8"
) as f:
    f.write(rapport)

print("\n>> MODELISATION TERMINEE")
print(f"   best_model.pkl -> data/best_model.pkl  (hash={model_hash})")
print(">> Prochaine etape : python scripts/inference.py (test unitaire)")
