"""
==============================================================
PROJET : Prédiction des Pannes GAB/ATM - Banque Populaire Maroc
==============================================================
Script 06 — Hyperparameter Tuning

Objectif : améliorer le modèle champion issu de modeling.py
en optimisant automatiquement ses hyperparamètres.

Stratégie :
  - Optuna (Bayesian optimization) si disponible → recommandé
  - RandomizedSearchCV (scikit-learn) sinon → fallback
  - Métrique d'optimisation : AUC-PR (cohérent avec modeling.py)
  - Validation : TimeSeriesSplit(3) sur train 2022 → pas de leakage
  - Recalibration après tuning : CalibratedClassifierCV(isotonic)
  - Sauvegarde uniquement si AUC-PR s'améliore (prudence)

Usage :
  python scripts/tuning.py                  # 50 trials, sauvegarde si meilleur
  python scripts/tuning.py --trials 100     # plus de trials
  python scripts/tuning.py --dry-run        # tuning sans écraser best_model.pkl
==============================================================
"""

import argparse
import hashlib
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, recall_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Imports optionnels ──────────────────────────────────────────
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

# ── Chemins ─────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT / "data"
COSTS     = {"fn": 5000, "tp": 1500, "fp": 500}

# ── Arguments CLI ───────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Hyperparameter tuning du modèle champion GAB")
parser.add_argument("--trials",   type=int,  default=50,    help="Nombre de trials Optuna (défaut: 50)")
parser.add_argument("--dry-run",  action="store_true",      help="Ne pas écraser best_model.pkl")
args = parser.parse_args()

# ══════════════════════════════════════════════════════════
# CHARGEMENT
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("  HYPERPARAMETER TUNING — GAB/ATM")
print(f"  {'Optuna (Bayesian)' if HAS_OPTUNA else 'RandomizedSearchCV (fallback)'}")
print(f"  Trials : {args.trials}  |  Dry-run : {args.dry_run}")
print("=" * 60)

model_path = DATA_DIR / "best_model.pkl"
if not model_path.exists():
    print("\n[ERREUR] best_model.pkl introuvable.")
    print("  Exécutez d'abord : python scripts/modeling.py")
    sys.exit(1)

with open(model_path, "rb") as f:
    artifact = pickle.load(f)

model_name       = artifact["model_name"]
feature_cols     = artifact["feature_cols"]
prev_auc_pr      = artifact["metrics"]["auc_pr"]
prev_threshold_eco = artifact.get("threshold_economic", 0.35)

print(f"\n[Champion actuel] {model_name}")
print(f"  AUC-PR     = {prev_auc_pr:.4f}")
print(f"  F1         = {artifact['metrics']['f1']:.4f}")
print(f"  Recall     = {artifact['metrics']['recall']:.4f}")
print(f"  Hash       = {artifact.get('model_hash', 'n/a')}")

# Dataset enrichi
print("\n[Chargement] gab_features.csv...")
df = pd.read_csv(DATA_DIR / "gab_features.csv", parse_dates=["date"])
df = df[[col for col in [*feature_cols, "date", "panne_sous_48h"] if col in df.columns]]

train_df = df[df["date"].dt.year == 2022]
test_df  = df[df["date"].dt.year == 2023]

X_train = train_df[feature_cols].values
y_train = train_df["panne_sous_48h"].values
X_test  = test_df[feature_cols].values
y_test  = test_df["panne_sous_48h"].values

pos_ratio    = y_train.mean()
neg_pos_ratio = (1 - pos_ratio) / pos_ratio

print(f"  Train: {len(X_train):,} | Test: {len(X_test):,} | Déséquilibre: {pos_ratio*100:.1f}%")

# ══════════════════════════════════════════════════════════
# DÉFINITION DES ESPACES DE RECHERCHE
# Un espace par type de modèle, centré autour des valeurs
# manuelles de modeling.py pour guider la recherche initiale.
# ══════════════════════════════════════════════════════════
def get_base_pipeline(trial_or_params, nom):
    """
    Reconstruit un pipeline sklearn non-calibré à partir d'un trial Optuna
    ou d'un dictionnaire de paramètres (RandomizedSearch).
    """
    p = trial_or_params  # raccourci

    def suggest(name, low, high, log=False, choices=None, step=None):
        """Wrapper : Optuna trial ou dict selon le contexte."""
        if isinstance(p, dict):
            return p[name]
        if choices:
            return p.suggest_categorical(name, choices)
        if isinstance(low, float) or isinstance(high, float):
            return p.suggest_float(name, low, high, log=log, step=step)
        return p.suggest_int(name, low, high, step=step or 1)

    if "HistGradientBoosting" in nom:
        model = HistGradientBoostingClassifier(
            class_weight="balanced",
            max_iter       = suggest("max_iter", 100, 600),
            max_depth      = suggest("max_depth", 3, 10),
            learning_rate  = suggest("learning_rate", 0.01, 0.25, log=True),
            min_samples_leaf = suggest("min_samples_leaf", 5, 60),
            l2_regularization = suggest("l2_reg", 1e-4, 2.0, log=True),
            random_state=42,
        )
    elif "Random Forest" in nom:
        model = RandomForestClassifier(
            class_weight="balanced_subsample",
            n_estimators   = suggest("n_estimators", 100, 500, step=50),
            max_depth      = suggest("max_depth", 6, 20),
            min_samples_leaf = suggest("min_samples_leaf", 5, 40),
            max_features   = suggest("max_features", None, None, choices=["sqrt", "log2", 0.3, 0.5]),
            n_jobs=-1, random_state=42,
        )
    elif "Logistic" in nom or "LogReg" in nom:
        model = LogisticRegression(
            class_weight="balanced",
            C        = suggest("C", 1e-3, 10.0, log=True),
            solver   = suggest("solver", None, None, choices=["saga", "lbfgs"]),
            max_iter=2000, random_state=42,
        )
    elif "XGBoost" in nom and HAS_XGB:
        model = XGBClassifier(
            scale_pos_weight = neg_pos_ratio,
            n_estimators     = suggest("n_estimators", 100, 500, step=50),
            max_depth        = suggest("max_depth", 3, 8),
            learning_rate    = suggest("learning_rate", 0.01, 0.25, log=True),
            subsample        = suggest("subsample", 0.5, 1.0),
            colsample_bytree = suggest("colsample_bytree", 0.5, 1.0),
            reg_lambda       = suggest("reg_lambda", 0.1, 10.0, log=True),
            eval_metric="aucpr", use_label_encoder=False, verbosity=0,
            random_state=42, n_jobs=-1,
        )
    elif "LightGBM" in nom and HAS_LGB:
        model = LGBMClassifier(
            is_unbalance=True,
            n_estimators  = suggest("n_estimators", 100, 500, step=50),
            max_depth     = suggest("max_depth", 3, 8),
            learning_rate = suggest("learning_rate", 0.01, 0.25, log=True),
            num_leaves    = suggest("num_leaves", 20, 100),
            subsample     = suggest("subsample", 0.5, 1.0),
            verbose=-1, random_state=42, n_jobs=-1,
        )
    else:
        # Fallback : HistGradientBoosting avec paramètres par défaut
        model = HistGradientBoostingClassifier(
            class_weight="balanced", random_state=42
        )

    return Pipeline([("scaler", StandardScaler()), ("model", model)])


# ══════════════════════════════════════════════════════════
# TUNING — OPTUNA
# Bayesian optimization : chaque trial est guidé par les résultats
# des trials précédents (TPE sampler) → beaucoup plus efficace
# qu'une recherche aléatoire pour le même nombre d'évaluations.
# ══════════════════════════════════════════════════════════
tscv     = TimeSeriesSplit(n_splits=3)
best_params = None

if HAS_OPTUNA:
    print(f"\n[Tuning Optuna] {args.trials} trials × TimeSeriesSplit(3)...")

    def objective(trial):
        pipeline = get_base_pipeline(trial, model_name)
        # AUC-PR en CV — scorer "average_precision" = AUC de la courbe PR
        scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=tscv, scoring="average_precision",
            n_jobs=-1,
        )
        return float(scores.mean())

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))

    # Warm-start : on teste les paramètres actuels en premier trial
    # pour ancrer la recherche autour d'une solution déjà connue.
    try:
        current_params = {}
        inner = artifact["pipeline"]
        if hasattr(inner, "calibrated_classifiers_"):
            inner = inner.calibrated_classifiers_[0].estimator
        if hasattr(inner, "named_steps"):
            m = inner.named_steps.get("model")
            if m:
                for attr in ["max_iter", "max_depth", "learning_rate",
                              "min_samples_leaf", "n_estimators", "C"]:
                    if hasattr(m, attr):
                        current_params[attr] = getattr(m, attr)
                if hasattr(m, "l2_regularization"):
                    current_params["l2_reg"] = m.l2_regularization
        if current_params:
            study.enqueue_trial(current_params)
    except Exception:
        pass

    t0 = time.time()
    with optuna.logging.disable_default_handler():
        for i in range(args.trials):
            study.optimize(objective, n_trials=1, show_progress_bar=False)
            best = study.best_value
            elapsed = time.time() - t0
            print(f"\r  Trial {i+1:3d}/{args.trials}  "
                  f"best AUC-PR={best:.4f}  "
                  f"elapsed={elapsed:.0f}s     ", end="", flush=True)

    print(f"\n\n  Meilleurs paramètres ({args.trials} trials) :")
    best_params = study.best_params
    for k, v in best_params.items():
        print(f"    {k}: {v}")

# ══════════════════════════════════════════════════════════
# TUNING — RANDOMIZEDSEARCHCV (fallback sans Optuna)
# ══════════════════════════════════════════════════════════
else:
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import loguniform, randint, uniform

    print(f"\n[Tuning RandomizedSearchCV] {args.trials} itérations × TimeSeriesSplit(3)...")

    # Espaces de recherche selon le modèle champion
    if "HistGradientBoosting" in model_name:
        param_dist = {
            "model__max_iter":          randint(100, 601),
            "model__max_depth":         randint(3, 11),
            "model__learning_rate":     loguniform(0.01, 0.25),
            "model__min_samples_leaf":  randint(5, 61),
            "model__l2_regularization": loguniform(1e-4, 2.0),
        }
        base = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  HistGradientBoostingClassifier(class_weight="balanced", random_state=42)),
        ])
    elif "Random Forest" in model_name:
        param_dist = {
            "model__n_estimators":     randint(100, 501),
            "model__max_depth":        randint(6, 21),
            "model__min_samples_leaf": randint(5, 41),
            "model__max_features":     ["sqrt", "log2", 0.3, 0.5],
        }
        base = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  RandomForestClassifier(class_weight="balanced_subsample",
                                              n_jobs=-1, random_state=42)),
        ])
    elif "Logistic" in model_name or "LogReg" in model_name:
        param_dist = {
            "model__C":      loguniform(1e-3, 10.0),
            "model__solver": ["saga", "lbfgs"],
        }
        base = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  LogisticRegression(class_weight="balanced",
                                          max_iter=2000, random_state=42)),
        ])
    else:
        param_dist = {
            "model__max_iter":         randint(100, 601),
            "model__learning_rate":    loguniform(0.01, 0.25),
            "model__min_samples_leaf": randint(5, 61),
        }
        base = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  HistGradientBoostingClassifier(class_weight="balanced", random_state=42)),
        ])

    search = RandomizedSearchCV(
        base, param_dist,
        n_iter=args.trials, cv=tscv,
        scoring="average_precision",
        n_jobs=-1, verbose=1, random_state=42,
    )
    search.fit(X_train, y_train)

    best_cv_score = search.best_score_
    best_params   = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    print(f"\n  Meilleur AUC-PR CV : {best_cv_score:.4f}")
    print(f"  Meilleurs paramètres :")
    for k, v in best_params.items():
        print(f"    {k}: {v}")

# ══════════════════════════════════════════════════════════
# RECONSTRUCTION + RECALIBRATION DU MEILLEUR MODÈLE
# ══════════════════════════════════════════════════════════
print("\n[Recalibration] CalibratedClassifierCV(isotonic, cv=3)...")

tuned_base = get_base_pipeline(best_params, model_name)
tuned_calibrated = CalibratedClassifierCV(tuned_base, cv=3, method="isotonic", n_jobs=-1)
tuned_calibrated.fit(X_train, y_train)

# ── Évaluation sur test 2023 ────────────────────────────
y_proba_new = tuned_calibrated.predict_proba(X_test)[:, 1]

new_auc_pr = float(average_precision_score(y_test, y_proba_new))

# Trouver le seuil économique optimal du modèle tunéfié
thresholds = np.linspace(0.01, 0.99, 200)
costs_list = []
for thr in thresholds:
    yp = (y_proba_new >= thr).astype(int)
    tp_ = int(((yp == 1) & (y_test == 1)).sum())
    fp_ = int(((yp == 1) & (y_test == 0)).sum())
    fn_ = int(((yp == 0) & (y_test == 1)).sum())
    costs_list.append(fn_ * COSTS["fn"] + tp_ * COSTS["tp"] + fp_ * COSTS["fp"])
new_threshold_eco = float(thresholds[np.argmin(costs_list)])

y_pred_eco = (y_proba_new >= new_threshold_eco).astype(int)
new_f1     = float(f1_score(y_test, y_pred_eco, zero_division=0))
new_recall = float(recall_score(y_test, y_pred_eco, zero_division=0))

# Seuil F1-optimal
from sklearn.metrics import precision_recall_curve
prec_c, rec_c, thr_c = precision_recall_curve(y_test, y_proba_new)
f1_c = np.where((prec_c + rec_c) > 0,
                2 * prec_c * rec_c / (prec_c + rec_c + 1e-9), 0)
new_threshold_f1 = float(thr_c[np.argmax(f1_c[:-1])]) if len(thr_c) > 0 else 0.4

# ── Tableau avant / après ───────────────────────────────
delta_auc = new_auc_pr - prev_auc_pr
print(f"\n{'='*60}")
print(f"  RÉSULTATS DU TUNING — {model_name}")
print(f"{'='*60}")
print(f"  {'Métrique':<22} {'Avant':>10} {'Après':>10} {'Delta':>10}")
print(f"  {'─'*52}")
print(f"  {'AUC-PR':<22} {prev_auc_pr:>10.4f} {new_auc_pr:>10.4f} {delta_auc:>+10.4f}  ← critère")
print(f"  {'F1 (seuil éco)':<22} {artifact['metrics']['f1']:>10.4f} {new_f1:>10.4f} {new_f1-artifact['metrics']['f1']:>+10.4f}")
print(f"  {'Recall (seuil éco)':<22} {artifact['metrics']['recall']:>10.4f} {new_recall:>10.4f} {new_recall-artifact['metrics']['recall']:>+10.4f}")
print(f"  {'Seuil économique':<22} {prev_threshold_eco:>10.3f} {new_threshold_eco:>10.3f}")
print(f"{'='*60}")

# ══════════════════════════════════════════════════════════
# SAUVEGARDE — uniquement si amélioration AUC-PR
# Évite d'écraser un bon modèle avec un résultat de tuning
# sous-optimal (possible avec peu de trials sur de gros espaces).
# ══════════════════════════════════════════════════════════
improved = new_auc_pr > prev_auc_pr

if args.dry_run:
    print("\n[Dry-run] Aucun fichier écrasé.")
    if improved:
        print(f"  Le tuning aurait AMÉLIORÉ le modèle (+{delta_auc:.4f} AUC-PR).")
    else:
        print(f"  Le tuning n'aurait PAS amélioré le modèle ({delta_auc:.4f} AUC-PR).")
    sys.exit(0)

if not improved:
    print(f"\n[Annulation] Le tuning n'améliore pas AUC-PR ({delta_auc:.4f}).")
    print("  best_model.pkl conservé tel quel.")
    print("  Conseil : augmenter --trials ou vérifier la convergence.")
    sys.exit(0)

# ── Mise à jour de best_model.pkl ────────────────────────
print(f"\n[Sauvegarde] Amélioration AUC-PR +{delta_auc:.4f} → mise à jour...")

new_artifact = {
    **artifact,  # conserve toutes les clés existantes (feature_cols, encoders path, etc.)
    "pipeline":           tuned_calibrated,
    "threshold_f1":       new_threshold_f1,
    "threshold_economic": new_threshold_eco,
    "metrics": {
        **artifact.get("metrics", {}),
        "auc_pr":    new_auc_pr,
        "f1":        new_f1,
        "recall":    new_recall,
        # tp/fp/fn/tn recalculés au seuil économique
        "tp": int(((y_pred_eco == 1) & (y_test == 1)).sum()),
        "fp": int(((y_pred_eco == 1) & (y_test == 0)).sum()),
        "fn": int(((y_pred_eco == 0) & (y_test == 1)).sum()),
        "tn": int(((y_pred_eco == 0) & (y_test == 0)).sum()),
    },
    "tuned":    True,
    "n_trials": args.trials,
    "best_params": best_params,
}

with open(model_path, "wb") as f:
    pickle.dump(new_artifact, f)

new_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()[:12]
new_artifact["model_hash"] = new_hash
with open(model_path, "wb") as f:
    pickle.dump(new_artifact, f)
print(f"  best_model.pkl mis à jour (hash={new_hash})")

# ── Mise à jour de resultats_modeles.json ────────────────
res_path = DATA_DIR / "resultats_modeles.json"
if res_path.exists():
    with open(res_path) as f:
        res_json = json.load(f)
    if "resultats" in res_json and model_name in res_json["resultats"]:
        res_json["resultats"][model_name].update({
            "auc_pr":             new_auc_pr,
            "f1":                 new_f1,
            "recall":             new_recall,
            "threshold_f1":       new_threshold_f1,
            "threshold_economic": new_threshold_eco,
            "tuned":              True,
            "n_trials":           args.trials,
        })
        res_json["model_hash"] = new_hash
    with open(res_path, "w") as f:
        json.dump(res_json, f, indent=2)
    print("  resultats_modeles.json mis à jour")

print(f"\n✅ TUNING TERMINÉ — {model_name} (hash={new_hash})")
print(f"   AUC-PR : {prev_auc_pr:.4f} → {new_auc_pr:.4f}  (+{delta_auc:.4f})")
print(f"🚀 Relancez le backend pour prendre en compte le nouveau modèle.")
