"""
==============================================================
PROJET : Prédiction des Pannes GAB/ATM - Banque Populaire Maroc
==============================================================
Script — Drift Monitor (détection de dérive)

Détecte si la distribution des features a changé entre le train
(2022) et les nouvelles données (2023).

Métriques :
  - KS-test (Kolmogorov-Smirnov) : p-value par feature numérique
  - PSI (Population Stability Index) : 10 bins equal-frequency sur train
      PSI < 0.10  → stable
      0.10 ≤ PSI < 0.20 → surveiller
      PSI ≥ 0.20  → drift significatif, réévaluer le modèle

Sorties :
  outputs/reports/rapport_drift.txt
  outputs/figures/fig_drift.png

Exit code 1 si >= 3 features avec PSI > 0.2 (utilisable en CI/CD).

Usage :
  python scripts/drift_monitor.py
  python scripts/drift_monitor.py --ref 2022 --new 2023
==============================================================
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

ROOT       = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT / "data"
REPORT_DIR = ROOT / "outputs" / "reports"
FIG_DIR    = ROOT / "outputs" / "figures"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ─── Seuils PSI ─────────────────────────────────────────────
PSI_STABLE   = 0.10
PSI_MONITOR  = 0.20
N_BINS       = 10
TOP_FEATURES = 20   # pour la figure

# ─── Parsing CLI ────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Drift Monitor — GAB ATM")
parser.add_argument("--ref", type=int, default=2022, help="Année de référence (train)")
parser.add_argument("--new", type=int, default=2023, help="Année à surveiller (test)")
args = parser.parse_args()

print("=" * 60)
print(f"  DRIFT MONITOR  —  {args.ref} (ref) vs {args.new} (new)")
print("=" * 60)

# ══════════════════════════════════════════════════════════
# CHARGEMENT
# ══════════════════════════════════════════════════════════
csv_path = DATA_DIR / "gab_features.csv"
if not csv_path.exists():
    print(f"\n[ERREUR] {csv_path} introuvable.")
    print("  Lancez d'abord : python scripts/feature_engineering.py")
    sys.exit(2)

print(f"\nChargement de {csv_path.name}...")
df = pd.read_csv(csv_path, parse_dates=["date"])
df_ref = df[df["date"].dt.year == args.ref]
df_new = df[df["date"].dt.year == args.new]

if len(df_ref) == 0:
    print(f"[ERREUR] Aucune donnée pour l'année {args.ref}")
    sys.exit(2)
if len(df_new) == 0:
    print(f"[ERREUR] Aucune donnée pour l'année {args.new}")
    sys.exit(2)

print(f"  Ref ({args.ref}) : {len(df_ref):,} observations")
print(f"  New ({args.new}) : {len(df_new):,} observations")

# Colonnes numériques à analyser (on exclut id, cibles, dates)
COLS_EXCLURE = {"date", "gab_id", "ville", "mois_annee", "panne_sous_48h"}
num_cols = [
    c for c in df_ref.columns
    if c not in COLS_EXCLURE and pd.api.types.is_numeric_dtype(df_ref[c])
]
print(f"  Features analysées : {len(num_cols)}")

# ══════════════════════════════════════════════════════════
# CALCUL PSI
# Population Stability Index :
#   PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)
# Les bins sont définis sur la distribution de référence
# (equal-frequency quantiles sur df_ref) pour un test équitable.
# ══════════════════════════════════════════════════════════
def compute_psi(ref: pd.Series, new: pd.Series, n_bins: int = N_BINS) -> float:
    """Retourne le PSI entre ref (train) et new (test)."""
    # Bins définis sur ref (quantiles equal-frequency)
    _, bin_edges = pd.qcut(ref, q=n_bins, duplicates="drop", retbins=True)
    bin_edges[0]  = -np.inf
    bin_edges[-1] =  np.inf

    ref_counts = pd.cut(ref, bins=bin_edges).value_counts(sort=False) + 1e-6
    new_counts = pd.cut(new, bins=bin_edges).value_counts(sort=False) + 1e-6

    ref_pct = ref_counts / ref_counts.sum()
    new_pct = new_counts / new_counts.sum()

    psi = float(((new_pct - ref_pct) * np.log(new_pct / ref_pct)).sum())
    return max(0.0, psi)   # jamais négatif (artefact numérique)

# ══════════════════════════════════════════════════════════
# CALCUL KS-TEST
# ══════════════════════════════════════════════════════════
def compute_ks(ref: pd.Series, new: pd.Series):
    """Retourne (statistic, p_value) du test de Kolmogorov-Smirnov."""
    ks_stat, p_val = stats.ks_2samp(ref.dropna(), new.dropna())
    return float(ks_stat), float(p_val)

# ══════════════════════════════════════════════════════════
# CALCUL SUR TOUTES LES FEATURES
# ══════════════════════════════════════════════════════════
print("\nCalcul des métriques de drift...")
results = []

for col in num_cols:
    psi          = compute_psi(df_ref[col], df_new[col])
    ks_stat, ks_p = compute_ks(df_ref[col], df_new[col])

    if psi >= PSI_MONITOR:
        statut = "DRIFT"
    elif psi >= PSI_STABLE:
        statut = "MONITOR"
    else:
        statut = "STABLE"

    results.append({
        "feature": col,
        "psi":     round(psi, 4),
        "ks_stat": round(ks_stat, 4),
        "ks_pval": round(ks_p, 4),
        "statut":  statut,
    })

res_df = pd.DataFrame(results).sort_values("psi", ascending=False).reset_index(drop=True)

n_drift   = (res_df["statut"] == "DRIFT").sum()
n_monitor = (res_df["statut"] == "MONITOR").sum()
n_stable  = (res_df["statut"] == "STABLE").sum()

# ══════════════════════════════════════════════════════════
# AFFICHAGE CONSOLE
# ══════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  RÉSUMÉ DRIFT  {args.ref} → {args.new}")
print(f"{'='*60}")
print(f"  STABLE  (PSI < {PSI_STABLE})   : {n_stable}")
print(f"  MONITOR (PSI {PSI_STABLE}–{PSI_MONITOR}) : {n_monitor}")
print(f"  DRIFT   (PSI >= {PSI_MONITOR})  : {n_drift}")
print()

if n_drift > 0:
    print("  Top features en DRIFT :")
    top_drift = res_df[res_df["statut"] == "DRIFT"].head(10)
    for _, row in top_drift.iterrows():
        print(f"    {row['feature']:<45} PSI={row['psi']:.3f}  KS_p={row['ks_pval']:.3f}")
    print()

# ══════════════════════════════════════════════════════════
# RAPPORT TEXTE
# ══════════════════════════════════════════════════════════
report_path = REPORT_DIR / "rapport_drift.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(f"DRIFT REPORT — GAB Intelligence\n")
    f.write(f"Généré le : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Référence : {args.ref}  |  Nouvelle période : {args.new}\n")
    f.write(f"{'='*70}\n\n")
    f.write(f"RÉSUMÉ\n")
    f.write(f"  STABLE  (PSI < {PSI_STABLE})   : {n_stable} features\n")
    f.write(f"  MONITOR (PSI {PSI_STABLE}–{PSI_MONITOR}) : {n_monitor} features\n")
    f.write(f"  DRIFT   (PSI >= {PSI_MONITOR})  : {n_drift} features\n\n")
    f.write(f"{'='*70}\n")
    f.write(f"{'Feature':<45} {'PSI':>8} {'KS_stat':>9} {'KS_pval':>9} {'Statut':>9}\n")
    f.write(f"{'-'*70}\n")
    for _, row in res_df.iterrows():
        f.write(
            f"{row['feature']:<45} {row['psi']:>8.4f} {row['ks_stat']:>9.4f}"
            f" {row['ks_pval']:>9.4f} {row['statut']:>9}\n"
        )

print(f"  Rapport texte  → {report_path}")

# ══════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    top_df = res_df.head(TOP_FEATURES).sort_values("psi")

    colors = []
    for statut in top_df["statut"]:
        if statut == "DRIFT":
            colors.append("#d4645a")
        elif statut == "MONITOR":
            colors.append("#e8a045")
        else:
            colors.append("#6ba88a")

    fig, ax = plt.subplots(figsize=(10, max(6, len(top_df) * 0.35)))
    ax.set_facecolor("#0d0f1a")
    fig.patch.set_facecolor("#07080d")

    bars = ax.barh(top_df["feature"], top_df["psi"], color=colors, edgecolor="none", height=0.65)

    # Lignes de seuil
    ax.axvline(PSI_STABLE,  color="#e8a045", linewidth=1.0, linestyle="--", alpha=0.8)
    ax.axvline(PSI_MONITOR, color="#d4645a", linewidth=1.0, linestyle="--", alpha=0.8)

    ax.set_xlabel("PSI (Population Stability Index)", color="#b0b8d0", fontsize=10)
    ax.set_title(
        f"Drift Monitor — {args.ref} (ref) vs {args.new}  |  Top {TOP_FEATURES} features",
        color="#e0e4f0", fontsize=11, pad=12
    )
    ax.tick_params(colors="#b0b8d0", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2d3e")

    legend_patches = [
        mpatches.Patch(color="#6ba88a", label=f"STABLE  (PSI < {PSI_STABLE})"),
        mpatches.Patch(color="#e8a045", label=f"MONITOR ({PSI_STABLE}–{PSI_MONITOR})"),
        mpatches.Patch(color="#d4645a", label=f"DRIFT   (PSI ≥ {PSI_MONITOR})"),
    ]
    ax.legend(handles=legend_patches, loc="lower right",
              facecolor="#1a1d2e", edgecolor="#2a2d3e",
              labelcolor="#b0b8d0", fontsize=8)

    plt.tight_layout()
    fig_path = FIG_DIR / "fig_drift.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Figure          → {fig_path}")

except ImportError:
    print("  [INFO] matplotlib non installé — figure ignorée")

# ══════════════════════════════════════════════════════════
# EXIT CODE CI/CD
# ══════════════════════════════════════════════════════════
SEUIL_ALERTE_CI = 3   # nombre de features DRIFT avant exit 1

print(f"\n{'='*60}")
if n_drift >= SEUIL_ALERTE_CI:
    print(f"  ALERTE : {n_drift} features en DRIFT (>= {SEUIL_ALERTE_CI})")
    print(f"  Recommandation : réévaluer le modèle avec les données {args.new}")
    print(f"{'='*60}")
    sys.exit(1)
else:
    print(f"  OK : {n_drift} feature(s) en DRIFT (seuil = {SEUIL_ALERTE_CI})")
    print(f"  Modèle stable — pas de réévaluation urgente")
    print(f"{'='*60}")
    sys.exit(0)
