"""
==============================================================
PROJET : Prédiction des Pannes GAB/ATM - Banque Populaire Maroc
==============================================================
Script 01 : Génération du dataset synthétique réaliste

Logique de simulation :
- 200 GAB répartis sur différentes villes du Maroc
- Historique de 2 ans (2022-2023)
- Chaque ligne = état journalier d'un GAB
- Features inspirées des vraies métriques opérationnelles ATM
- Taux de panne ~5% (réaliste, classe déséquilibrée)
==============================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# ─── Reproductibilité ──────────────────────────────────────
np.random.seed(42)

# ─── Paramètres de simulation ──────────────────────────────
N_GABS          = 200       # Nombre de GAB dans le réseau
N_DAYS          = 730       # 2 ans d'historique
START_DATE      = datetime(2022, 1, 1)
FAILURE_RATE    = 0.05      # 5% des observations = panne imminente

# ─── Villes du Maroc avec poids (densité réseau BPM) ───────
VILLES = {
    "Casablanca":    0.25,
    "Rabat":         0.12,
    "Marrakech":     0.10,
    "Fès":           0.09,
    "Tanger":        0.08,
    "Agadir":        0.07,
    "Meknès":        0.06,
    "Oujda":         0.05,
    "Kénitra":       0.05,
    "Tétouan":       0.04,
    "Safi":          0.03,
    "El Jadida":     0.03,
    "Beni Mellal":   0.03,
}

TYPES_GAB = ["NCR", "Diebold", "Wincor", "Hyosung"]
ENVIRONNEMENTS = ["Agence_Interieure", "Agence_Facade", "Site_Isole", "Centre_Commercial"]

# ─── Génération du référentiel GAB ─────────────────────────
def generer_referentiel_gab(n_gabs):
    """
    Crée la table de référence des GAB avec leurs caractéristiques fixes.
    L'ancienneté et le type influenceront directement le risque de panne.
    """
    villes_list = list(VILLES.keys())
    poids_list  = list(VILLES.values())

    gabs = pd.DataFrame({
        "gab_id": [f"GAB_{str(i).zfill(4)}" for i in range(1, n_gabs + 1)],

        # Ville assignée selon la distribution réseau réelle
        "ville": np.random.choice(villes_list, size=n_gabs, p=poids_list),

        # Constructeur du GAB
        "type_gab": np.random.choice(TYPES_GAB, size=n_gabs, p=[0.35, 0.30, 0.20, 0.15]),

        # Environnement d'installation (impact sur poussière, chaleur, vandalisme)
        "environnement": np.random.choice(
            ENVIRONNEMENTS, size=n_gabs, p=[0.40, 0.30, 0.15, 0.15]
        ),

        # Ancienneté en années (les vieux GAB tombent plus en panne)
        "age_annees": np.random.choice(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            size=n_gabs,
            p=[0.10, 0.12, 0.13, 0.13, 0.12, 0.11, 0.10, 0.09, 0.06, 0.04]
        ),
    })
    return gabs


# ─── Génération des observations journalières ──────────────
def generer_observations(gabs_df, n_days, start_date):
    """
    Pour chaque GAB et chaque jour, génère les métriques opérationnelles.
    
    Features générées :
    - Métriques transactionnelles (volume, taux d'erreur)
    - Métriques matérielles (température, niveau billets, erreurs lecteur)
    - Métriques réseau (latence, déconnexions)
    - Métriques de maintenance (jours depuis dernière intervention)
    - Target : panne_sous_48h (1 = panne dans les 2 prochains jours)
    """
    records = []
    dates = [start_date + timedelta(days=d) for d in range(n_days)]

    for _, gab in gabs_df.iterrows():
        
        # Facteur de risque de base lié au GAB (fixe pour toute la durée)
        # Plus le GAB est vieux et en site isolé, plus le risque est élevé
        risque_base = (
            (gab["age_annees"] / 10) * 0.4 +
            (1.0 if gab["environnement"] == "Site_Isole" else 0.3) * 0.3 +
            (0.8 if gab["type_gab"] == "Wincor" else 0.3) * 0.3  # Wincor = plus de pannes
        )

        # Jours depuis la dernière maintenance (repart à 0 après intervention)
        jours_depuis_maintenance = np.random.randint(0, 90)

        for date in dates:
            
            # Jour de la semaine (weekends = plus de transactions → plus de stress)
            est_weekend = date.weekday() >= 5
            
            # Saison (été au Maroc = chaleur = stress thermique)
            mois = date.month
            est_ete = mois in [6, 7, 8]

            # ── Métriques transactionnelles ──────────────────────
            # Volume journalier de transactions (Poisson)
            volume_base = 180 if est_weekend else 120
            nb_transactions = max(0, int(np.random.poisson(volume_base)))

            # Taux d'erreur transactions (% transactions échouées)
            taux_erreur_tx = np.clip(
                np.random.beta(1.5, 20) + risque_base * 0.05,
                0, 1
            )

            # ── Métriques matérielles ────────────────────────────
            # Température interne (°C) — augmente en été et avec l'âge
            temp_base = 38 if est_ete else 32
            temperature_interne = np.clip(
                np.random.normal(temp_base + gab["age_annees"] * 0.5, 3),
                25, 70
            )

            # Niveau de billets restants (% de la capacité)
            niveau_billets = np.clip(np.random.beta(5, 2) * 100, 0, 100)

            # Erreurs lecteur de carte (nb/jour)
            erreurs_lecteur = max(0, int(np.random.poisson(
                0.5 + risque_base * 3 + (jours_depuis_maintenance / 90) * 2
            )))

            # Erreurs distributeur de billets
            erreurs_distributeur = max(0, int(np.random.poisson(
                0.3 + risque_base * 2
            )))

            # ── Métriques réseau ─────────────────────────────────
            # Latence réseau en ms
            latence_ms = np.clip(
                np.random.exponential(80) + (50 if gab["environnement"] == "Site_Isole" else 0),
                10, 2000
            )

            # Nombre de déconnexions réseau dans la journée
            nb_deconnexions = max(0, int(np.random.poisson(
                0.2 + (0.8 if gab["environnement"] == "Site_Isole" else 0)
            )))

            # ── Métriques de maintenance ─────────────────────────
            jours_depuis_maintenance += 1  # +1 chaque jour
            
            # Réinitialisation aléatoire (intervention de maintenance)
            if np.random.random() < 0.01:  # ~1% de chance d'intervention/jour
                jours_depuis_maintenance = 0

            # ── Calcul du score de risque réel ───────────────────
            # Ce score non-observé sert à générer le label de panne
            score_risque = (
                risque_base * 0.25 +
                (temperature_interne / 70) * 0.20 +
                (erreurs_lecteur / 10) * 0.20 +
                (jours_depuis_maintenance / 180) * 0.15 +
                taux_erreur_tx * 0.10 +
                (erreurs_distributeur / 5) * 0.10
            )

            # ── Label : panne imminente (sous 48h) ───────────────
            # Bernoulli avec probabilité proportionnelle au score de risque
            prob_panne = np.clip(score_risque * 0.25, 0, 0.40)
            panne_sous_48h = int(np.random.random() < prob_panne)

            records.append({
                # Identifiants
                "date":                     date.strftime("%Y-%m-%d"),
                "gab_id":                   gab["gab_id"],
                "ville":                    gab["ville"],
                "type_gab":                 gab["type_gab"],
                "environnement":            gab["environnement"],
                "age_annees":               gab["age_annees"],
                # Features transactionnelles
                "nb_transactions":          nb_transactions,
                "taux_erreur_tx":           round(taux_erreur_tx, 4),
                # Features matérielles
                "temperature_interne":      round(temperature_interne, 1),
                "niveau_billets_pct":       round(niveau_billets, 1),
                "erreurs_lecteur_carte":    erreurs_lecteur,
                "erreurs_distributeur":     erreurs_distributeur,
                # Features réseau
                "latence_ms":               round(latence_ms, 1),
                "nb_deconnexions":          nb_deconnexions,
                # Features maintenance
                "jours_depuis_maintenance": jours_depuis_maintenance,
                # Target
                "panne_sous_48h":           panne_sous_48h,
            })

    return pd.DataFrame(records)


# ─── Exécution principale ───────────────────────────────────
if __name__ == "__main__":

    print("=" * 60)
    print("  GÉNÉRATION DU DATASET GAB/ATM")
    print("=" * 60)

    print("\n[1/3] Génération du référentiel GAB...")
    gabs_df = generer_referentiel_gab(N_GABS)
    print(f"      → {len(gabs_df)} GAB créés")

    print("\n[2/3] Génération des observations journalières...")
    print(f"      → {N_GABS} GAB × {N_DAYS} jours = {N_GABS * N_DAYS:,} observations")
    df = generer_observations(gabs_df, N_DAYS, START_DATE)
    print(f"      → Dataset généré : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")

    print("\n[3/3] Sauvegarde...")
    df.to_csv("./gab_dataset.csv", index=False)
    gabs_df.to_csv("./gab_referentiel.csv", index=False)
    print("      → ./gab_dataset.csv")
    print("      → ./gab_referentiel.csv")

    # ─── Aperçu statistique rapide ───────────────────────────
    print("\n" + "=" * 60)
    print("  APERÇU DU DATASET")
    print("=" * 60)

    print(f"\nPériode         : {df['date'].min()} → {df['date'].max()}")
    print(f"Nb observations : {len(df):,}")
    print(f"Nb GAB          : {df['gab_id'].nunique()}")
    print(f"Nb villes       : {df['ville'].nunique()}")

    n_pannes = df['panne_sous_48h'].sum()
    taux     = n_pannes / len(df) * 100
    print(f"\nDistribution target :")
    print(f"  → Pas de panne (0) : {len(df) - n_pannes:,} ({100 - taux:.1f}%)")
    print(f"  → Panne imminente  : {n_pannes:,} ({taux:.1f}%)")

    print("\nAperçu des 5 premières lignes :")
    print(df.head().to_string())

    print("\nStatistiques descriptives :")
    print(df.describe().round(2).to_string())

    print("\n✅ Dataset prêt pour l'EDA et la modélisation.")
