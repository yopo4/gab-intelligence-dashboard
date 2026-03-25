"""
==============================================================
PROJET : Prédiction des Pannes GAB/ATM - Banque Populaire Maroc
==============================================================
Script 02 : Exploratory Data Analysis (EDA) Complet

Objectifs :
1. Comprendre la distribution de chaque feature
2. Analyser le déséquilibre de classes
3. Identifier les corrélations avec la target
4. Détecter les outliers
5. Analyser les patterns temporels et géographiques
6. Produire un rapport visuel complet (PDF/PNG)
==============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Backend non-interactif pour génération fichiers
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

# ─── Configuration visuelle ────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0f1117",
    "axes.facecolor":    "#1a1d27",
    "axes.edgecolor":    "#3a3d4d",
    "axes.labelcolor":   "#e0e0e0",
    "xtick.color":       "#b0b0b0",
    "ytick.color":       "#b0b0b0",
    "text.color":        "#e0e0e0",
    "grid.color":        "#2a2d3d",
    "grid.alpha":        0.5,
    "font.family":       "DejaVu Sans",
    "font.size":         10,
})

PALETTE_CLASSES = {0: "#4a9eff", 1: "#ff4a6e"}  # Bleu = ok, Rouge = panne
COLOR_OK    = "#4a9eff"
COLOR_PANNE = "#ff4a6e"
COLOR_ACCENT = "#f0b429"

os.makedirs("/home/claude/outputs/eda", exist_ok=True)

# ─── Chargement des données ────────────────────────────────
print("Chargement des données...")
df = pd.read_csv("/home/claude/data/gab_dataset.csv", parse_dates=["date"])
df_ref = pd.read_csv("/home/claude/data/gab_referentiel.csv")

# Features numériques pour analyses
FEATURES_NUM = [
    "nb_transactions", "taux_erreur_tx", "temperature_interne",
    "niveau_billets_pct", "erreurs_lecteur_carte", "erreurs_distributeur",
    "latence_ms", "nb_deconnexions", "jours_depuis_maintenance", "age_annees"
]

FEATURES_CAT = ["ville", "type_gab", "environnement"]

print(f"Dataset : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
print(f"Période : {df['date'].min().date()} → {df['date'].max().date()}")
print(f"Taux de panne : {df['panne_sous_48h'].mean()*100:.1f}%\n")


# ══════════════════════════════════════════════════════════
# FIGURE 1 : Vue d'ensemble & distribution de la target
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.patch.set_facecolor("#0f1117")
fig.suptitle("EDA — Vue d'ensemble du Dataset GAB/ATM\nBanque Populaire Maroc",
             fontsize=16, fontweight="bold", color="white", y=0.98)

# 1.1 Distribution de la target (donut)
ax = axes[0, 0]
counts = df["panne_sous_48h"].value_counts()
colors = [COLOR_OK, COLOR_PANNE]
wedges, texts, autotexts = ax.pie(
    counts.values, labels=["Pas de panne", "Panne imminente"],
    colors=colors, autopct="%1.1f%%", startangle=90,
    wedgeprops={"width": 0.6, "edgecolor": "#0f1117", "linewidth": 3},
    textprops={"color": "white", "fontsize": 10}
)
for at in autotexts:
    at.set_color("white")
    at.set_fontweight("bold")
ax.set_title("Distribution de la Target\n(Déséquilibre de classes)", 
             color="white", fontweight="bold")

# 1.2 Pannes par ville (barplot horizontal)
ax = axes[0, 1]
pannes_ville = df[df["panne_sous_48h"] == 1].groupby("ville").size().sort_values(ascending=True)
bars = ax.barh(pannes_ville.index, pannes_ville.values, color=COLOR_PANNE, alpha=0.85)
ax.set_title("Nombre de Pannes par Ville", color="white", fontweight="bold")
ax.set_xlabel("Nb pannes", color="#b0b0b0")
for bar, val in zip(bars, pannes_ville.values):
    ax.text(val + 20, bar.get_y() + bar.get_height()/2,
            f"{val:,}", va="center", color="#b0b0b0", fontsize=8)

# 1.3 Pannes par type de GAB
ax = axes[0, 2]
taux_panne_type = df.groupby("type_gab")["panne_sous_48h"].mean() * 100
bars = ax.bar(taux_panne_type.index, taux_panne_type.values,
              color=[COLOR_PANNE if t == "Wincor" else COLOR_OK for t in taux_panne_type.index],
              alpha=0.85, edgecolor="#0f1117", linewidth=1.5)
ax.set_title("Taux de Panne par Type de GAB (%)", color="white", fontweight="bold")
ax.set_ylabel("Taux (%)", color="#b0b0b0")
ax.set_ylim(0, taux_panne_type.max() * 1.3)
for bar, val in zip(bars, taux_panne_type.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f"{val:.1f}%", ha="center", color="white", fontweight="bold", fontsize=9)

# 1.4 Distribution pannes dans le temps (mensuel)
ax = axes[1, 0]
df["mois_annee"] = df["date"].dt.to_period("M")
pannes_mois = df.groupby("mois_annee")["panne_sous_48h"].sum()
x_labels = [str(p) for p in pannes_mois.index]
x_pos = range(len(x_labels))
ax.fill_between(x_pos, pannes_mois.values, alpha=0.4, color=COLOR_PANNE)
ax.plot(x_pos, pannes_mois.values, color=COLOR_PANNE, linewidth=2)
ax.set_title("Évolution Mensuelle des Pannes", color="white", fontweight="bold")
ax.set_ylabel("Nb pannes", color="#b0b0b0")
ax.set_xticks(x_pos[::3])
ax.set_xticklabels(x_labels[::3], rotation=45, ha="right", fontsize=7)

# 1.5 Pannes par environnement
ax = axes[1, 1]
taux_env = df.groupby("environnement")["panne_sous_48h"].mean() * 100
colors_env = [COLOR_PANNE if "Isole" in e else COLOR_OK for e in taux_env.index]
bars = ax.bar(
    [e.replace("_", "\n") for e in taux_env.index],
    taux_env.values, color=colors_env, alpha=0.85,
    edgecolor="#0f1117", linewidth=1.5
)
ax.set_title("Taux de Panne par Environnement (%)", color="white", fontweight="bold")
ax.set_ylabel("Taux (%)", color="#b0b0b0")
for bar, val in zip(bars, taux_env.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f"{val:.1f}%", ha="center", color="white", fontweight="bold", fontsize=9)

# 1.6 Taux de panne par âge du GAB
ax = axes[1, 2]
taux_age = df.groupby("age_annees")["panne_sous_48h"].mean() * 100
ax.plot(taux_age.index, taux_age.values, color=COLOR_ACCENT,
        linewidth=2.5, marker="o", markersize=6, markerfacecolor="white")
ax.fill_between(taux_age.index, taux_age.values, alpha=0.2, color=COLOR_ACCENT)
ax.set_title("Taux de Panne selon l'Âge du GAB", color="white", fontweight="bold")
ax.set_xlabel("Âge (années)", color="#b0b0b0")
ax.set_ylabel("Taux de panne (%)", color="#b0b0b0")

plt.tight_layout()
fig.savefig("./fig1_vue_ensemble.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("✅ Figure 1 : Vue d'ensemble sauvegardée")


# ══════════════════════════════════════════════════════════
# FIGURE 2 : Distributions des features numériques
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 5, figsize=(22, 9))
fig.patch.set_facecolor("#0f1117")
fig.suptitle("Distribution des Features Numériques — Classe 0 vs Classe 1",
             fontsize=14, fontweight="bold", color="white", y=1.01)

for idx, feature in enumerate(FEATURES_NUM):
    ax = axes[idx // 5, idx % 5]
    
    for classe, color, label in [(0, COLOR_OK, "Pas de panne"), (1, COLOR_PANNE, "Panne")]:
        data_c = df[df["panne_sous_48h"] == classe][feature].dropna()
        ax.hist(data_c, bins=40, alpha=0.65, color=color, label=label,
                density=True, edgecolor="none")
    
    ax.set_title(feature.replace("_", "\n"), color="white", fontsize=8, fontweight="bold")
    ax.set_yticks([])
    
    # Ligne médiane pour chaque classe
    for classe, color in [(0, COLOR_OK), (1, COLOR_PANNE)]:
        median = df[df["panne_sous_48h"] == classe][feature].median()
        ax.axvline(median, color=color, linestyle="--", linewidth=1.5, alpha=0.9)

axes[0, 0].legend(fontsize=7, framealpha=0.3)

plt.tight_layout()
fig.savefig("./fig2_distributions.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("✅ Figure 2 : Distributions sauvegardée")


# ══════════════════════════════════════════════════════════
# FIGURE 3 : Matrice de corrélation
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
fig.patch.set_facecolor("#0f1117")
fig.suptitle("Analyse des Corrélations", fontsize=14, fontweight="bold", color="white")

features_corr = FEATURES_NUM + ["panne_sous_48h"]

# 3.1 Matrice de corrélation complète
corr_matrix = df[features_corr].corr()
mask = np.zeros_like(corr_matrix, dtype=bool)
mask[np.triu_indices_from(mask, k=1)] = True

sns.heatmap(
    corr_matrix, ax=axes[0],
    cmap="RdYlGn", center=0, vmin=-1, vmax=1,
    annot=True, fmt=".2f", annot_kws={"size": 7},
    linewidths=0.5, linecolor="#0f1117",
    cbar_kws={"shrink": 0.8}
)
axes[0].set_title("Matrice de Corrélation (Pearson)", color="white", fontweight="bold")
axes[0].tick_params(colors="#b0b0b0", labelsize=7)

# 3.2 Corrélations avec la target (barplot)
corr_target = corr_matrix["panne_sous_48h"].drop("panne_sous_48h").sort_values()
colors_bar = [COLOR_PANNE if v > 0 else COLOR_OK for v in corr_target.values]
axes[1].barh(
    [f.replace("_", " ") for f in corr_target.index],
    corr_target.values, color=colors_bar, alpha=0.85,
    edgecolor="#0f1117", linewidth=1
)
axes[1].axvline(0, color="white", linewidth=0.8, linestyle="-")
axes[1].set_title("Corrélation de chaque Feature\navec la Target (panne_sous_48h)",
                  color="white", fontweight="bold")
axes[1].set_xlabel("Coefficient de corrélation", color="#b0b0b0")
for i, (val, feat) in enumerate(zip(corr_target.values, corr_target.index)):
    axes[1].text(val + (0.003 if val >= 0 else -0.003), i,
                 f"{val:.3f}", va="center",
                 ha="left" if val >= 0 else "right",
                 color="white", fontsize=8)

plt.tight_layout()
fig.savefig("./fig3_correlations.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("✅ Figure 3 : Corrélations sauvegardée")


# ══════════════════════════════════════════════════════════
# FIGURE 4 : Boxplots — Features les plus discriminantes
# ══════════════════════════════════════════════════════════
# Top 6 features les plus corrélées avec la target
corr_target_abs = abs(corr_matrix["panne_sous_48h"].drop("panne_sous_48h")).sort_values(ascending=False)
top6_features = corr_target_abs.head(6).index.tolist()

fig, axes = plt.subplots(2, 3, figsize=(18, 9))
fig.patch.set_facecolor("#0f1117")
fig.suptitle("Boxplots des Features les Plus Discriminantes (Top 6)",
             fontsize=14, fontweight="bold", color="white")

for idx, feature in enumerate(top6_features):
    ax = axes[idx // 3, idx % 3]
    
    data_0 = df[df["panne_sous_48h"] == 0][feature].dropna()
    data_1 = df[df["panne_sous_48h"] == 1][feature].dropna()
    
    bp = ax.boxplot(
        [data_0, data_1],
        patch_artist=True,
        labels=["Pas de panne", "Panne imminente"],
        medianprops={"color": "white", "linewidth": 2},
        flierprops={"marker": "o", "markersize": 2, "alpha": 0.3},
        whiskerprops={"linewidth": 1.5},
        capprops={"linewidth": 1.5}
    )
    
    bp["boxes"][0].set_facecolor(COLOR_OK)
    bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_facecolor(COLOR_PANNE)
    bp["boxes"][1].set_alpha(0.7)
    
    ax.set_title(feature.replace("_", " ").title(), color="white", fontweight="bold")
    ax.tick_params(colors="#b0b0b0", labelsize=8)
    
    # Annotation : différence de médiane
    med0 = data_0.median()
    med1 = data_1.median()
    delta = ((med1 - med0) / med0 * 100) if med0 != 0 else 0
    sign = "+" if delta > 0 else ""
    ax.text(0.98, 0.95, f"Δ médiane: {sign}{delta:.0f}%",
            transform=ax.transAxes, ha="right", va="top",
            color=COLOR_ACCENT, fontsize=8, fontweight="bold",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "#1a1d27", "alpha": 0.8})

plt.tight_layout()
fig.savefig("./fig4_boxplots_top6.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("✅ Figure 4 : Boxplots Top 6 sauvegardée")


# ══════════════════════════════════════════════════════════
# FIGURE 5 : Analyse temporelle des pannes
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(18, 10))
fig.patch.set_facecolor("#0f1117")
fig.suptitle("Analyse Temporelle des Pannes GAB",
             fontsize=14, fontweight="bold", color="white")

# 5.1 Pannes par jour de semaine
ax = axes[0, 0]
jours_noms = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
df["jour_semaine"] = df["date"].dt.dayofweek
taux_jour = df.groupby("jour_semaine")["panne_sous_48h"].mean() * 100
colors_jour = [COLOR_PANNE if j >= 5 else COLOR_OK for j in range(7)]
ax.bar(jours_noms, taux_jour.values, color=colors_jour, alpha=0.85, edgecolor="#0f1117")
ax.set_title("Taux de Panne par Jour de Semaine", color="white", fontweight="bold")
ax.set_ylabel("Taux (%)", color="#b0b0b0")
ax.tick_params(axis="x", rotation=30)

# 5.2 Pannes par mois
ax = axes[0, 1]
mois_noms = ["Jan", "Fév", "Mar", "Avr", "Mai", "Jun",
             "Jul", "Aoû", "Sep", "Oct", "Nov", "Déc"]
df["mois"] = df["date"].dt.month
taux_mois = df.groupby("mois")["panne_sous_48h"].mean() * 100
colors_mois = [COLOR_PANNE if m in [6, 7, 8] else COLOR_OK for m in range(1, 13)]
ax.bar(mois_noms, taux_mois.values, color=colors_mois, alpha=0.85, edgecolor="#0f1117")
ax.set_title("Taux de Panne par Mois", color="white", fontweight="bold")
ax.set_ylabel("Taux (%)", color="#b0b0b0")

# 5.3 Heatmap : Ville × Mois
ax = axes[1, 0]
heatmap_data = df.groupby(["ville", "mois"])["panne_sous_48h"].mean() * 100
heatmap_pivot = heatmap_data.unstack(level="mois")
heatmap_pivot.columns = mois_noms
sns.heatmap(
    heatmap_pivot, ax=ax, cmap="YlOrRd",
    annot=False, linewidths=0.3, linecolor="#0f1117",
    cbar_kws={"label": "Taux panne (%)"}
)
ax.set_title("Taux de Panne : Ville × Mois", color="white", fontweight="bold")
ax.tick_params(colors="#b0b0b0", labelsize=8)

# 5.4 Courbe température vs panne dans le temps
ax = axes[1, 1]
monthly = df.groupby("mois").agg(
    temp_moy=("temperature_interne", "mean"),
    taux_panne=("panne_sous_48h", "mean")
).reset_index()
monthly["taux_panne"] = monthly["taux_panne"] * 100

ax2 = ax.twinx()
line1 = ax.plot(mois_noms, monthly["temp_moy"], color=COLOR_ACCENT,
                linewidth=2.5, marker="s", markersize=5, label="Température interne (°C)")
line2 = ax2.plot(mois_noms, monthly["taux_panne"], color=COLOR_PANNE,
                 linewidth=2.5, marker="o", markersize=5, label="Taux de panne (%)")
ax.set_ylabel("Température interne (°C)", color=COLOR_ACCENT)
ax2.set_ylabel("Taux de panne (%)", color=COLOR_PANNE)
ax.tick_params(axis="x", rotation=30)
ax2.tick_params(colors=COLOR_PANNE)
ax.tick_params(axis="y", colors=COLOR_ACCENT)
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, loc="upper left", fontsize=8, framealpha=0.3)
ax.set_title("Corrélation Température / Taux de Panne", color="white", fontweight="bold")

plt.tight_layout()
fig.savefig("./fig5_analyse_temporelle.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("✅ Figure 5 : Analyse temporelle sauvegardée")


# ══════════════════════════════════════════════════════════
# RAPPORT TEXTE — Insights EDA
# ══════════════════════════════════════════════════════════
rapport = f"""
╔══════════════════════════════════════════════════════════════╗
║     RAPPORT EDA — PRÉDICTION PANNES GAB/ATM                 ║
║     Banque Populaire Maroc                                   ║
╚══════════════════════════════════════════════════════════════╝

📊 STATISTIQUES GÉNÉRALES
─────────────────────────
• Observations totales  : {len(df):,}
• Période               : {df['date'].min().date()} → {df['date'].max().date()}
• Nombre de GAB         : {df['gab_id'].nunique()}
• Villes couvertes      : {df['ville'].nunique()}

🎯 DISTRIBUTION DES CLASSES (TARGET)
──────────────────────────────────────
• Pas de panne (0)     : {(df['panne_sous_48h']==0).sum():,} ({(df['panne_sous_48h']==0).mean()*100:.1f}%)
• Panne imminente (1)  : {(df['panne_sous_48h']==1).sum():,} ({(df['panne_sous_48h']==1).mean()*100:.1f}%)
• Ratio déséquilibre   : 1 : {int((df['panne_sous_48h']==0).sum() / (df['panne_sous_48h']==1).sum())}
→ IMPORTANT : Déséquilibre significatif → nécessite SMOTE ou class_weight

📈 TOP FEATURES CORRÉLÉES AVEC LA TARGET
──────────────────────────────────────────
{corr_target_abs.head(5).to_string()}

🏙️ VILLES À RISQUE ÉLEVÉ
──────────────────────────
{(df.groupby('ville')['panne_sous_48h'].mean()*100).sort_values(ascending=False).head(5).to_string()}

🔧 CONSTRUCTEURS À RISQUE
──────────────────────────
{(df.groupby('type_gab')['panne_sous_48h'].mean()*100).sort_values(ascending=False).to_string()}

🏗️ ENVIRONNEMENTS À RISQUE
───────────────────────────
{(df.groupby('environnement')['panne_sous_48h'].mean()*100).sort_values(ascending=False).to_string()}

⚠️ VALEURS MANQUANTES
──────────────────────
{df[FEATURES_NUM].isnull().sum().to_string()}

💡 INSIGHTS CLÉS POUR LA MODÉLISATION
────────────────────────────────────────
1. Déséquilibre de classes (~90/10) → utiliser SMOTE + F1-score comme métrique
2. Features les plus prédictives : {', '.join(top6_features[:3])}
3. Effet saisonnier : pannes en hausse en été (stress thermique)
4. Âge du GAB : relation quasi-linéaire avec le taux de panne
5. Sites isolés : taux de panne significativement plus élevé
6. Jours depuis maintenance : plus c'est long, plus le risque monte

🚀 PROCHAINE ÉTAPE : Feature Engineering + Modélisation
"""

print(rapport)

with open("./rapport_eda.txt", "w", encoding="utf-8") as f:
    f.write(rapport)

print("\n✅ EDA COMPLET — Tous les fichiers sauvegardés dans ./")
print("   • fig1_vue_ensemble.png")
print("   • fig2_distributions.png")
print("   • fig3_correlations.png")
print("   • fig4_boxplots_top6.png")
print("   • fig5_analyse_temporelle.png")
print("   • rapport_eda.txt")
