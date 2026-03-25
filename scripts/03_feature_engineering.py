"""
==============================================================
PROJET : Prédiction des Pannes GAB/ATM - Banque Populaire Maroc
==============================================================
Script 03 : Feature Engineering

Stratégie :
-----------
Les données brutes ne capturent que l'état instantané d'un GAB.
Or, une panne ne survient pas d'un coup — elle est précédée
d'une DÉGRADATION PROGRESSIVE sur plusieurs jours.

On va donc créer 5 familles de features :

1. Features temporelles       → Saisonnalité, jour semaine, mois
2. Lag features               → État du GAB J-1, J-3, J-7
3. Rolling statistics         → Moyenne / std glissante sur 7j, 14j
4. Features de tendance       → Pente de dégradation sur 7 jours
5. Features d'interaction     → Combinaisons métier pertinentes
6. Encodage des variables cat → One-hot / ordinal

Pourquoi les lag/rolling features sont cruciales en maintenance
prédictive ?
→ Un GAB avec 3 erreurs lecteur AUJOURD'HUI est moins alarmant
  qu'un GAB avec 1 erreur J-7, 2 erreurs J-3, 5 erreurs aujourd'hui.
  La TENDANCE est plus informative que la valeur instantanée.
==============================================================
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

# ─── Chargement ────────────────────────────────────────────
print("=" * 60)
print("  FEATURE ENGINEERING — GAB/ATM")
print("=" * 60)

df = pd.read_csv("./gab_dataset.csv", parse_dates=["date"])
df = df.sort_values(["gab_id", "date"]).reset_index(drop=True)

print(f"\n[INPUT]  {df.shape[0]:,} lignes × {df.shape[1]} colonnes")

# Features numériques sur lesquelles on va créer les lags/rolling
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
# FAMILLE 1 : Features temporelles
# Objectif : capturer les patterns saisonniers et cycliques
# ══════════════════════════════════════════════════════════
print("\n[1/6] Features temporelles...")

df["jour_semaine"]   = df["date"].dt.dayofweek          # 0=lundi, 6=dimanche
df["mois"]           = df["date"].dt.month               # 1-12
df["trimestre"]      = df["date"].dt.quarter             # 1-4
df["est_weekend"]    = (df["jour_semaine"] >= 5).astype(int)
df["est_ete"]        = df["mois"].isin([6, 7, 8]).astype(int)
df["est_fin_mois"]   = (df["date"].dt.day >= 25).astype(int)  # Fin de mois = plus de retraits

# Encodage cyclique du mois et du jour (sin/cos)
# Pourquoi ? Les encodages linéaires cassent la continuité :
# décembre (12) et janvier (1) sont proches mais semblent loin.
# Le sin/cos préserve cette continuité.
df["mois_sin"]       = np.sin(2 * np.pi * df["mois"] / 12)
df["mois_cos"]       = np.cos(2 * np.pi * df["mois"] / 12)
df["jour_sin"]       = np.sin(2 * np.pi * df["jour_semaine"] / 7)
df["jour_cos"]       = np.cos(2 * np.pi * df["jour_semaine"] / 7)

print(f"   → 10 features temporelles créées")


# ══════════════════════════════════════════════════════════
# FAMILLE 2 : Lag features
# Objectif : donner au modèle la mémoire des jours précédents
# IMPORTANT : le groupby par gab_id évite les fuites entre GAB
# ══════════════════════════════════════════════════════════
print("\n[2/6] Lag features (J-1, J-3, J-7)...")

LAGS = [1, 3, 7]
lag_cols_created = []

for feature in FEATURES_LAG:
    for lag in LAGS:
        col_name = f"{feature}_lag{lag}"
        df[col_name] = df.groupby("gab_id")[feature].shift(lag)
        lag_cols_created.append(col_name)

print(f"   → {len(lag_cols_created)} lag features créées ({len(FEATURES_LAG)} features × {len(LAGS)} lags)")


# ══════════════════════════════════════════════════════════
# FAMILLE 3 : Rolling statistics (fenêtres glissantes)
# Objectif : capturer la moyenne et la variabilité récente
# min_periods=1 : évite les NaN en début de série
# ══════════════════════════════════════════════════════════
print("\n[3/6] Rolling statistics (7j, 14j)...")

WINDOWS   = [7, 14]
STATS     = ["mean", "std", "max"]
roll_cols = []

for feature in FEATURES_LAG:
    for window in WINDOWS:
        grp = df.groupby("gab_id")[feature]
        
        # Moyenne glissante : tendance centrale récente
        col = f"{feature}_roll{window}_mean"
        df[col] = grp.transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        roll_cols.append(col)
        
        # Écart-type glissant : volatilité / instabilité récente
        col = f"{feature}_roll{window}_std"
        df[col] = grp.transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).std().fillna(0)
        )
        roll_cols.append(col)
        
        # Maximum glissant : pic récent (1 seul jour peut indiquer une panne)
        col = f"{feature}_roll{window}_max"
        df[col] = grp.transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).max()
        )
        roll_cols.append(col)

print(f"   → {len(roll_cols)} rolling features créées")


# ══════════════════════════════════════════════════════════
# FAMILLE 4 : Features de tendance (pente de dégradation)
# Objectif : détecter si une métrique est EN HAUSSE ou stable
#
# Méthode : différence entre valeur actuelle et moyenne 7j passés
# → Positif = dégradation en cours
# → Négatif = amélioration
# ══════════════════════════════════════════════════════════
print("\n[4/6] Features de tendance (dégradation)...")

trend_cols = []
TREND_FEATURES = ["erreurs_lecteur_carte", "erreurs_distributeur",
                  "temperature_interne", "taux_erreur_tx"]

for feature in TREND_FEATURES:
    col = f"{feature}_tendance_7j"
    roll_mean_col = f"{feature}_roll7_mean"
    
    if roll_mean_col in df.columns:
        # Tendance = écart entre valeur actuelle et moyenne passée
        # Normalisé par la moyenne pour être en pourcentage de variation
        df[col] = (df[feature] - df[roll_mean_col]) / (df[roll_mean_col] + 1e-8)
        trend_cols.append(col)

# Feature spéciale : accélération des erreurs sur 3j vs 7j
# Si la moyenne 3j > moyenne 7j → accélération récente de la dégradation
df["erreurs_lecteur_acceleration"] = (
    df["erreurs_lecteur_carte_lag1"] + df["erreurs_lecteur_carte_lag3"]
) / 2 - df["erreurs_lecteur_carte_roll7_mean"]

trend_cols.append("erreurs_lecteur_acceleration")

print(f"   → {len(trend_cols)} features de tendance créées")


# ══════════════════════════════════════════════════════════
# FAMILLE 5 : Features d'interaction métier
# Objectif : encoder des règles expertes du domaine maintenance
#
# Ces features sont basées sur la connaissance métier :
# "Un vieux GAB avec beaucoup d'erreurs ET pas de maintenance
#  récente = risque très élevé"
# ══════════════════════════════════════════════════════════
print("\n[5/6] Features d'interaction métier...")

# Score de risque matériel : erreurs × âge
# Plus le GAB est vieux ET fait des erreurs, plus c'est grave
df["risque_materiel"] = (
    df["erreurs_lecteur_carte"] * np.log1p(df["age_annees"])
)

# Score de stress thermique : température × âge
# Les composants vieillissants supportent moins bien la chaleur
df["stress_thermique"] = df["temperature_interne"] * df["age_annees"] / 10

# Score de surcharge : transactions × taux d'erreur
# Un GAB très utilisé avec beaucoup d'erreurs est sous pression
df["score_surcharge"] = df["nb_transactions"] * df["taux_erreur_tx"]

# Score de négligence maintenance : jours × erreurs récentes
# Long délai depuis maintenance + erreurs en hausse = danger
df["score_negligence"] = (
    df["jours_depuis_maintenance"] *
    np.log1p(df["erreurs_lecteur_carte_roll7_mean"])
)

# Score de connectivité : latence × déconnexions
df["score_connectivite"] = df["latence_ms"] * (1 + df["nb_deconnexions"])

# Ratio erreurs / transactions : taux d'erreur "matérielle"
# Indépendant du volume = plus robuste
df["ratio_erreurs_tx"] = (
    (df["erreurs_lecteur_carte"] + df["erreurs_distributeur"]) /
    (df["nb_transactions"] + 1)
)

interaction_cols = [
    "risque_materiel", "stress_thermique", "score_surcharge",
    "score_negligence", "score_connectivite", "ratio_erreurs_tx"
]
print(f"   → {len(interaction_cols)} features d'interaction créées")


# ══════════════════════════════════════════════════════════
# FAMILLE 6 : Encodage des variables catégorielles
# Stratégie :
# - One-hot encoding pour type_gab (4 modalités, pas d'ordre)
# - One-hot encoding pour environnement (4 modalités)
# - Target encoding pour ville (13 modalités → évite explosion dims)
# ══════════════════════════════════════════════════════════
print("\n[6/6] Encodage des variables catégorielles...")

# One-hot encoding
df = pd.get_dummies(df, columns=["type_gab", "environnement"],
                    drop_first=False, dtype=int)

# Target encoding pour ville
# Calcul du taux de panne moyen par ville sur l'ensemble
# ATTENTION en production : calculer sur le train set uniquement
taux_panne_ville = df.groupby("ville")["panne_sous_48h"].mean()
df["ville_risk_encoding"] = df["ville"].map(taux_panne_ville)

print(f"   → Variables catégorielles encodées (one-hot + target encoding ville)")


# ══════════════════════════════════════════════════════════
# NETTOYAGE FINAL
# Les lags/rolling créent des NaN sur les premiers jours
# de chaque GAB (pas d'historique disponible).
# On les supprime plutôt que de les imputer (données manquantes
# structurelles, pas aléatoires).
# ══════════════════════════════════════════════════════════
print("\n[Nettoyage] Suppression des NaN structurels...")

avant = len(df)
df = df.dropna()
apres = len(df)
print(f"   → {avant - apres:,} lignes supprimées ({(avant-apres)/avant*100:.1f}%)")
print(f"   → Dataset final : {apres:,} lignes")


# ══════════════════════════════════════════════════════════
# RÉSUMÉ DES FEATURES CRÉÉES
# ══════════════════════════════════════════════════════════
# Colonnes à exclure du modèle (identifiants, dates, target)
COLS_EXCLURE = ["date", "gab_id", "ville", "mois_annee",
                "mois", "jour_semaine", "panne_sous_48h"]

feature_cols = [c for c in df.columns if c not in COLS_EXCLURE]

print(f"\n{'='*60}")
print(f"  RÉSUMÉ DES FEATURES")
print(f"{'='*60}")
print(f"  Features originales    : 10")
print(f"  Features temporelles   : 10")
print(f"  Lag features           : {len(lag_cols_created)}")
print(f"  Rolling features       : {len(roll_cols)}")
print(f"  Features de tendance   : {len(trend_cols)}")
print(f"  Features d'interaction : {len(interaction_cols)}")
print(f"  ─────────────────────────────")
print(f"  TOTAL features modèle  : {len(feature_cols)}")
print(f"{'='*60}")

# Sauvegarde
df.to_csv("./gab_features.csv", index=False)

# Sauvegarde de la liste des features pour réutilisation
import json
with open("./feature_cols.json", "w") as f:
    json.dump(feature_cols, f, indent=2)

print(f"\n✅ Dataset enrichi sauvegardé : ./gab_features.csv")
print(f"✅ Liste features sauvegardée : ./feature_cols.json")
print(f"\n🚀 Prochaine étape : Modélisation (Script 04)")
