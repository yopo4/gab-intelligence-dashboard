"""
==============================================================
PROJET : Prédiction des Pannes GAB/ATM - Banque Populaire Maroc
==============================================================
Script 03 — Feature Engineering (version corrigée)

CORRECTIFS vs version précédente :
  1. Data leakage corrigé : target encoding fitté sur train uniquement,
     puis appliqué au test via l'encodeur sauvegardé
  2. Encodeurs sérialisés dans encoders.pkl pour l'inférence live
  3. Nouvelles features :
     - Accélération (dérivée seconde) des erreurs sur 7j
     - ratio_temperature_saison : température / moy. saisonnière par ville
==============================================================
"""

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─── Chemins ────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT / "data"

print("=" * 60)
print("  FEATURE ENGINEERING v2 — GAB/ATM (leakage-free)")
print("=" * 60)

# ══════════════════════════════════════════════════════════
# CHARGEMENT
# ══════════════════════════════════════════════════════════
df = pd.read_csv(DATA_DIR / "gab_dataset.csv", parse_dates=["date"])
df = df.sort_values(["gab_id", "date"]).reset_index(drop=True)
print(f"\n[INPUT]  {len(df):,} lignes × {df.shape[1]} colonnes")
print(f"         Dates : {df['date'].min().date()} → {df['date'].max().date()}")

# Features numériques sur lesquelles créer les lags/rolling
FEATURES_LAG = [
    "erreurs_lecteur_carte",
    "erreurs_distributeur",
    "temperature_interne",
    "taux_erreur_tx",
    "latence_ms",
    "nb_deconnexions",
    "nb_transactions",
]

# ══════════════════════════════════════════════════════════
# FAMILLE 1 : Features temporelles (pas de leakage)
# ══════════════════════════════════════════════════════════
print("\n[1/7] Features temporelles...")

df["jour_semaine"] = df["date"].dt.dayofweek
df["mois"]         = df["date"].dt.month
df["trimestre"]    = df["date"].dt.quarter
df["est_weekend"]  = (df["jour_semaine"] >= 5).astype(int)
df["est_ete"]      = df["mois"].isin([6, 7, 8]).astype(int)
df["est_fin_mois"] = (df["date"].dt.day >= 25).astype(int)

# Encodage cyclique : préserve la continuité jan↔déc et lun↔dim
df["mois_sin"]  = np.sin(2 * np.pi * df["mois"] / 12)
df["mois_cos"]  = np.cos(2 * np.pi * df["mois"] / 12)
df["jour_sin"]  = np.sin(2 * np.pi * df["jour_semaine"] / 7)
df["jour_cos"]  = np.cos(2 * np.pi * df["jour_semaine"] / 7)

print("   → 10 features temporelles créées")

# ══════════════════════════════════════════════════════════
# FAMILLE 2 : Lag features (pas de leakage — shift vers l'avant)
# ══════════════════════════════════════════════════════════
print("\n[2/7] Lag features (J-1, J-3, J-7)...")

lag_cols = []
for feat in FEATURES_LAG:
    for lag in [1, 3, 7]:
        col = f"{feat}_lag{lag}"
        df[col] = df.groupby("gab_id")[feat].shift(lag)
        lag_cols.append(col)

print(f"   → {len(lag_cols)} lag features")

# ══════════════════════════════════════════════════════════
# FAMILLE 3 : Rolling statistics (pas de leakage — shift(1) avant rolling)
# ══════════════════════════════════════════════════════════
print("\n[3/7] Rolling statistics (7j, 14j)...")

roll_cols = []
for feat in FEATURES_LAG:
    for window in [7, 14]:
        grp = df.groupby("gab_id")[feat]
        # shift(1) garantit qu'on utilise uniquement des données passées
        base = grp.transform(lambda x: x.shift(1))

        col = f"{feat}_roll{window}_mean"
        df[col] = grp.transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        roll_cols.append(col)

        col = f"{feat}_roll{window}_std"
        df[col] = grp.transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).std().fillna(0)
        )
        roll_cols.append(col)

        col = f"{feat}_roll{window}_max"
        df[col] = grp.transform(lambda x: x.shift(1).rolling(window, min_periods=1).max())
        roll_cols.append(col)

print(f"   → {len(roll_cols)} rolling features")

# ══════════════════════════════════════════════════════════
# FAMILLE 4 : Tendance (dégradation relative sur 7j)
# ══════════════════════════════════════════════════════════
print("\n[4/7] Features de tendance...")

trend_cols = []
for feat in ["erreurs_lecteur_carte", "erreurs_distributeur",
             "temperature_interne", "taux_erreur_tx"]:
    col = f"{feat}_tendance_7j"
    roll_mean = f"{feat}_roll7_mean"
    if roll_mean in df.columns:
        df[col] = (df[feat] - df[roll_mean]) / (df[roll_mean] + 1e-8)
        trend_cols.append(col)

# Accélération (lecture J0 vs J-3) par rapport à la moyenne 7j
df["erreurs_lecteur_acceleration"] = (
    df["erreurs_lecteur_carte_lag1"] + df["erreurs_lecteur_carte_lag3"]
) / 2 - df["erreurs_lecteur_carte_roll7_mean"]
trend_cols.append("erreurs_lecteur_acceleration")

print(f"   → {len(trend_cols)} features de tendance")

# ══════════════════════════════════════════════════════════
# FAMILLE 4b : Accélération — dérivée seconde (NOUVEAU)
# d²f/dt² ≈ f(t) - 2·f(t-1) + f(t-2)
# Capture si la dégradation s'emballe (accélération positive = alerte)
# ══════════════════════════════════════════════════════════
print("\n[4b] Features d'accélération (dérivée seconde)...")

accel_cols = []
for feat in ["erreurs_lecteur_carte", "erreurs_distributeur", "temperature_interne"]:
    col = f"{feat}_accel_2nd"
    lag1 = df.groupby("gab_id")[feat].shift(1)
    lag2 = df.groupby("gab_id")[feat].shift(2)
    df[col] = df[feat] - 2 * lag1 + lag2
    accel_cols.append(col)

print(f"   → {len(accel_cols)} features d'accélération (dérivée seconde)")

# ══════════════════════════════════════════════════════════
# FAMILLE 5 : Interactions métier (pas de leakage)
# ══════════════════════════════════════════════════════════
print("\n[5/7] Features d'interaction métier...")

df["risque_materiel"]    = df["erreurs_lecteur_carte"] * np.log1p(df["age_annees"])
df["stress_thermique"]   = df["temperature_interne"] * df["age_annees"] / 10
df["score_surcharge"]    = df["nb_transactions"] * df["taux_erreur_tx"]
df["score_negligence"]   = (
    df["jours_depuis_maintenance"] * np.log1p(df["erreurs_lecteur_carte_roll7_mean"])
)
df["score_connectivite"] = df["latence_ms"] * (1 + df["nb_deconnexions"])
df["ratio_erreurs_tx"]   = (
    (df["erreurs_lecteur_carte"] + df["erreurs_distributeur"]) / (df["nb_transactions"] + 1)
)

interaction_cols = [
    "risque_materiel", "stress_thermique", "score_surcharge",
    "score_negligence", "score_connectivite", "ratio_erreurs_tx",
]
print(f"   → {len(interaction_cols)} features d'interaction")

# ══════════════════════════════════════════════════════════
# FAMILLE 5b : Historique de pannes (features sur target décalée)
# shift(1) OBLIGATOIRE — utiliser panne_sous_48h[t] directement
# serait du target leakage. shift(1) = on regarde le passé seulement.
# ══════════════════════════════════════════════════════════
print("\n[5b] Historique de pannes (shift obligatoire)...")

# Nombre de pannes dans les 30j / 90j passés
df["nb_pannes_30j"] = df.groupby("gab_id")["panne_sous_48h"].transform(
    lambda x: x.shift(1).rolling(30, min_periods=1).sum()
)
df["nb_pannes_90j"] = df.groupby("gab_id")["panne_sous_48h"].transform(
    lambda x: x.shift(1).rolling(90, min_periods=1).sum()
)

# Jours depuis la dernière panne (sur la série shiftée)
def _days_since_last(x_shifted):
    """Compte le nombre de jours depuis la dernière panne (valeur = 1)."""
    result, count = [], 0
    for v in x_shifted:
        result.append(count)
        count = 0 if (not pd.isna(v) and v == 1) else count + 1
    return result

df["jours_depuis_panne"] = df.groupby("gab_id")["panne_sous_48h"].transform(
    lambda x: pd.Series(_days_since_last(list(x.shift(1))), index=x.index)
)
# Pour les tout premiers enregistrements d'un GAB (aucune panne connue),
# on impute avec le maximum observé — hypothèse conservatrice.
df["jours_depuis_panne"] = df["jours_depuis_panne"].fillna(df["jours_depuis_panne"].max())

hist_cols = ["nb_pannes_30j", "nb_pannes_90j", "jours_depuis_panne"]
print(f"   → {len(hist_cols)} features historiques (nb_pannes_30j, nb_pannes_90j, jours_depuis_panne)")

# ══════════════════════════════════════════════════════════
# SPLIT CHRONOLOGIQUE — AVANT tout encodage target-dépendant
# Train : 2022 | Test : 2023
# RÈGLE : les statistiques target (target encoding) sont calculées
# uniquement sur le train set, puis appliquées au test.
# ══════════════════════════════════════════════════════════
print("\n[6/7] Encodages (leakage-free)...")

mask_train = df["date"].dt.year == 2022
mask_test  = df["date"].dt.year == 2023
train_df   = df[mask_train]
print(f"   Train (2022) : {mask_train.sum():,} obs")
print(f"   Test  (2023) : {mask_test.sum():,} obs")

# ── 6a. Target encoding pour ville — fitté sur train uniquement ──
# Chaque ville reçoit son taux de panne moyen calculé sur 2022.
# Appliquer ce mapping au test évite le leakage : on n'utilise
# JAMAIS les labels de 2023 pour construire une feature de 2023.
taux_panne_ville = train_df.groupby("ville")["panne_sous_48h"].mean()
# Fallback : villes absentes du train → taux global du train
fallback_taux = train_df["panne_sous_48h"].mean()
df["ville_risk_encoding"] = df["ville"].map(taux_panne_ville).fillna(fallback_taux)

# ── 6b. ratio_temperature_saison (NOUVEAU) ────────────────────────
# Normalise la température courante par la moyenne saisonnière
# de la ville (calculée sur le train). Capture si un GAB est
# plus chaud qu'habituellement pour la saison — stress anormal.
def get_saison(month):
    if month in [12, 1, 2]: return "Hiver"
    if month in [3, 4, 5]:  return "Printemps"
    if month in [6, 7, 8]:  return "Été"
    return "Automne"

df["_saison"] = df["date"].dt.month.map(get_saison)

# Moyennes saisonnières par ville — train uniquement
temp_saison_ville = (
    train_df.assign(_saison=train_df["date"].dt.month.map(get_saison))
    .groupby(["ville", "_saison"])["temperature_interne"]
    .mean()
)
fallback_temp = train_df["temperature_interne"].mean()

df["_temp_saison_ref"] = df.apply(
    lambda r: temp_saison_ville.get((r["ville"], r["_saison"]), fallback_temp), axis=1
)
df["ratio_temperature_saison"] = df["temperature_interne"] / (df["_temp_saison_ref"] + 1e-8)
df.drop(columns=["_saison", "_temp_saison_ref"], inplace=True)

# ── 6c. One-hot encoding — catégorielles sans ordre ───────────────
# get_dummies sur le df entier : les colonnes créées sont identiques
# pour train et test (même modalités = cohérence garantie).
df = pd.get_dummies(df, columns=["type_gab", "environnement"],
                    drop_first=False, dtype=int)

print("   → Target encoding ville (train-only) ✓")
print("   → ratio_temperature_saison ✓")
print("   → One-hot encoding type_gab + environnement ✓")

# ══════════════════════════════════════════════════════════
# NETTOYAGE : NaN structurels dus aux lags (premiers jours de chaque GAB)
# ══════════════════════════════════════════════════════════
avant = len(df)
df = df.dropna()
apres = len(df)
print(f"\n[Nettoyage] {avant - apres:,} lignes NaN supprimées ({(avant-apres)/avant*100:.1f}%)")

# ══════════════════════════════════════════════════════════
# LISTE DES FEATURES FINALES
# ══════════════════════════════════════════════════════════
COLS_EXCLURE = {"date", "gab_id", "ville", "mois_annee",
                "mois", "jour_semaine", "panne_sous_48h"}
feature_cols = [c for c in df.columns if c not in COLS_EXCLURE]

print(f"\n{'='*60}")
print(f"  RÉSUMÉ DES FEATURES")
print(f"{'='*60}")
print(f"  Features temporelles  : 10")
print(f"  Lag features          : {len(lag_cols)}")
print(f"  Rolling features      : {len(roll_cols)}")
print(f"  Tendance              : {len(trend_cols)}")
print(f"  Accélération (2nd)    : {len(accel_cols)}")
print(f"  Interactions          : {len(interaction_cols)}")
print(f"  Historique pannes     : {len(hist_cols)}")
print(f"  Encodages             : ville + type_gab + environnement + ratio_temp_saison")
print(f"  ─────────────────────────────────────────────────")
print(f"  TOTAL features modèle : {len(feature_cols)}")
print(f"{'='*60}")

# ══════════════════════════════════════════════════════════
# SAUVEGARDE DES ENCODEURS (nécessaire pour l'inférence live)
# Contient toutes les statistiques calculées sur le train :
#  - ville_encoding : taux de panne par ville
#  - seasonal_temp  : température moyenne (ville × saison) sur train
#  - feature_medians : médianes pour imputation des valeurs manquantes
#  - feature_cols    : liste ordonnée des features attendues par le modèle
# ══════════════════════════════════════════════════════════
print("\n[7/7] Sauvegarde des artefacts...")

train_clean = df[df["date"].dt.year == 2022]

# Médianes d'imputation sur le train (pour l'inférence sur observations isolées)
feature_medians = {col: float(train_clean[col].median()) for col in feature_cols}

# Recalcul des temp saisonnières propres (sur train_clean)
seasonal_temp = {}
for (ville, saison), val in temp_saison_ville.items():
    seasonal_temp[f"{ville}|{saison}"] = float(val)

encoders = {
    "ville_encoding":   {k: float(v) for k, v in taux_panne_ville.items()},
    "fallback_taux":    float(fallback_taux),
    "seasonal_temp":    seasonal_temp,
    "fallback_temp":    float(fallback_temp),
    "feature_medians":  feature_medians,
    "feature_cols":     feature_cols,
    # One-hot columns présents dans le modèle (pour reconstruire à l'inférence)
    "ohe_type_gab":     [c for c in feature_cols if c.startswith("type_gab_")],
    "ohe_environnement":[c for c in feature_cols if c.startswith("environnement_")],
    "train_year":       2022,
}

with open(DATA_DIR / "encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)
print(f"   → encoders.pkl sauvegardé ({len(feature_cols)} features)")

# Dataset enrichi complet
df.to_csv(DATA_DIR / "gab_features.csv", index=False)
print(f"   → gab_features.csv ({len(df):,} lignes)")

# Liste des features pour modeling.py
with open(DATA_DIR / "feature_cols.json", "w") as f:
    json.dump(feature_cols, f, indent=2)
print(f"   → feature_cols.json ({len(feature_cols)} features)")

print(f"\n✅ Feature engineering terminé — artefacts dans data/")
print(f"🚀 Prochaine étape : python scripts/modeling.py")
