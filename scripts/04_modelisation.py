"""
==============================================================
PROJET : Prédiction des Pannes GAB/ATM - Banque Populaire Maroc
==============================================================
Script 04 : Modélisation ML Complète

Stratégie de modélisation :
────────────────────────────
On suit une approche pyramidale — du plus simple au plus complexe.
Chaque étape justifie la suivante.

  Niveau 1 : Dummy Classifier      → baseline naïf (référence plancher)
  Niveau 2 : Logistic Regression   → baseline linéaire interprétable
  Niveau 3 : Decision Tree         → baseline non-linéaire, interprétable
  Niveau 4 : Random Forest         → modèle robuste, peu de tuning nécessaire
  Niveau 5 : Gradient Boosting     → modèle final, meilleure performance

Problème de déséquilibre de classes :
───────────────────────────────────────
Ratio 90:10 → On utilise class_weight="balanced" sur tous les modèles.
Pourquoi pas SMOTE ici ? SMOTE requiert imbalanced-learn non dispo.
class_weight="balanced" est équivalent en termes d'effet et
parfois meilleur en généralisation (pas de données synthétiques).

Split temporel (TimeSeriesSplit) :
───────────────────────────────────
CRITIQUE : On ne peut PAS faire un split aléatoire sur des
séries temporelles. Cela créerait une fuite temporelle
(data leakage) : le modèle verrait des données futures
pour prédire le passé.
→ On utilise un split chronologique : train = 2022, test = 2023.

Métriques d'évaluation :
─────────────────────────
- Accuracy        : INUTILE ici (biais classe majoritaire)
- Precision       : Parmi les alarmes levées, combien sont vraies ?
- Recall          : Parmi les vraies pannes, combien détecte-t-on ?
- F1-Score        : Équilibre precision/recall → MÉTRIQUE PRINCIPALE
- AUC-ROC         : Performance globale du classifieur
- AUC-PR          : Plus adapté aux classes déséquilibrées que ROC
==============================================================
"""

import pandas as pd
import numpy as np
import json, os, time, warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model   import LogisticRegression
from sklearn.tree           import DecisionTreeClassifier
from sklearn.ensemble       import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy          import DummyClassifier
from sklearn.preprocessing  import StandardScaler
from sklearn.pipeline       import Pipeline
from sklearn.metrics        import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    precision_recall_curve, roc_curve
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─── Style visuel ──────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117", "axes.facecolor": "#1a1d27",
    "axes.edgecolor": "#3a3d4d", "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#b0b0b0", "ytick.color": "#b0b0b0",
    "text.color": "#e0e0e0", "grid.color": "#2a2d3d", "grid.alpha": 0.5,
    "font.size": 10,
})
COLOR_OK = "#4a9eff"; COLOR_PANNE = "#ff4a6e"; COLOR_ACCENT = "#f0b429"
COLOR_OK2 = "#34d399"; COLOR_PURPLE = "#a78bfa"
os.makedirs("./outputs/models", exist_ok=True)

# ══════════════════════════════════════════════════════════
# CHARGEMENT & PRÉPARATION
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("  MODÉLISATION ML — GAB/ATM PREDICTIVE MAINTENANCE")
print("=" * 60)

print("\n[1/5] Chargement des données enrichies...")
df = pd.read_csv("./gab_features.csv", parse_dates=["date"])

with open("./feature_cols.json") as f:
    feature_cols = json.load(f)

# Nettoyage : retirer colonnes non-features qui auraient pu glisser
COLS_EXCLURE = {"date", "gab_id", "ville", "mois_annee",
                "mois", "jour_semaine", "panne_sous_48h"}
feature_cols = [c for c in feature_cols if c in df.columns and c not in COLS_EXCLURE]

print(f"   → {len(df):,} observations, {len(feature_cols)} features")

# ── Split temporel chronologique ────────────────────────
# Train : 2022 entier (première année)
# Test  : 2023 entier (deuxième année, données "futures")
# Cette approche simule exactement la réalité :
# on entraîne sur le passé, on prédit le futur.
print("\n[2/5] Split temporel chronologique...")

train_df = df[df["date"].dt.year == 2022]
test_df  = df[df["date"].dt.year == 2023]

X_train = train_df[feature_cols].values
y_train = train_df["panne_sous_48h"].values
X_test  = test_df[feature_cols].values
y_test  = test_df["panne_sous_48h"].values

print(f"   → Train (2022) : {len(X_train):,} obs | pannes = {y_train.sum():,} ({y_train.mean()*100:.1f}%)")
print(f"   → Test  (2023) : {len(X_test):,}  obs | pannes = {y_test.sum():,} ({y_test.mean()*100:.1f}%)")


# ══════════════════════════════════════════════════════════
# DÉFINITION DES MODÈLES
# ══════════════════════════════════════════════════════════
print("\n[3/5] Définition des modèles...")

# Chaque modèle est wrappé dans un Pipeline avec StandardScaler.
# Pourquoi scaler ? La régression logistique en a besoin absolument.
# Pour les arbres (RF, GBM), ça ne change pas les résultats
# mais c'est bonne pratique (cohérence, déploiement).

MODELES = {
    "Dummy (Stratified)": Pipeline([
        ("scaler", StandardScaler()),
        ("model", DummyClassifier(strategy="stratified", random_state=42))
    ]),

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=0.1,              # Régularisation L2 modérée
            solver="lbfgs",
            random_state=42
        ))
    ]),

    "Decision Tree": Pipeline([
        ("scaler", StandardScaler()),
        ("model", DecisionTreeClassifier(
            class_weight="balanced",
            max_depth=8,        # Limité pour éviter l'overfitting
            min_samples_leaf=50,
            random_state=42
        ))
    ]),

    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            max_depth=12,
            min_samples_leaf=20,
            max_features="sqrt",    # Règle standard pour RF
            n_jobs=-1,              # Parallélisation max
            random_state=42
        ))
    ]),

    "Gradient Boosting": Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,     # Petit LR + plus d'arbres = meilleur
            max_depth=5,
            min_samples_leaf=20,
            subsample=0.8,          # Stochastic GB = moins d'overfitting
            max_features="sqrt",
            random_state=42
        ))
    ]),
}

# ══════════════════════════════════════════════════════════
# ENTRAÎNEMENT & ÉVALUATION
# ══════════════════════════════════════════════════════════
print("\n[4/5] Entraînement et évaluation...")
print(f"{'─'*60}")

resultats = {}

for nom, pipeline in MODELES.items():
    print(f"\n  🔄 {nom}...")
    t0 = time.time()

    # ── Entraînement ──────────────────────────────────────
    pipeline.fit(X_train, y_train)
    t_train = time.time() - t0

    # ── Prédictions ───────────────────────────────────────
    y_pred      = pipeline.predict(X_test)
    y_proba     = pipeline.predict_proba(X_test)[:, 1]

    # ── Métriques ─────────────────────────────────────────
    f1          = f1_score(y_test, y_pred, zero_division=0)
    precision   = precision_score(y_test, y_pred, zero_division=0)
    recall      = recall_score(y_test, y_pred, zero_division=0)
    auc_roc     = roc_auc_score(y_test, y_proba)
    auc_pr      = average_precision_score(y_test, y_proba)
    cm          = confusion_matrix(y_test, y_pred)

    # ── Calcul métriques métier ────────────────────────────
    # Parmi les pannes réelles, combien aurait-on évitées ?
    tn, fp, fn, tp = cm.ravel()
    pannes_detectees    = tp
    pannes_manquees     = fn
    fausses_alertes     = fp
    taux_detection      = tp / (tp + fn) if (tp + fn) > 0 else 0

    resultats[nom] = {
        "f1":                f1,
        "precision":         precision,
        "recall":            recall,
        "auc_roc":           auc_roc,
        "auc_pr":            auc_pr,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "taux_detection":    taux_detection,
        "t_train":           t_train,
        "y_pred":            y_pred,
        "y_proba":           y_proba,
        "pipeline":          pipeline,
    }

    print(f"     F1={f1:.4f} | Precision={precision:.4f} | Recall={recall:.4f} | AUC-ROC={auc_roc:.4f} | AUC-PR={auc_pr:.4f} | Train={t_train:.1f}s")
    print(f"     Pannes détectées={tp}/{tp+fn} ({taux_detection*100:.1f}%) | Fausses alertes={fp}")

print(f"\n{'─'*60}")

# Identification du meilleur modèle (par F1)
meilleur = max(resultats, key=lambda k: resultats[k]["f1"])
print(f"\n  🏆 Meilleur modèle : {meilleur} (F1 = {resultats[meilleur]['f1']:.4f})")


# ══════════════════════════════════════════════════════════
# FEATURE IMPORTANCE (Random Forest & Gradient Boosting)
# ══════════════════════════════════════════════════════════
print("\n[5/5] Analyse de l'importance des features...")

importance_rf  = resultats["Random Forest"]["pipeline"]["model"].feature_importances_
importance_gb  = resultats["Gradient Boosting"]["pipeline"]["model"].feature_importances_

# Moyenne des deux modèles pour un ranking stable
importance_moy = (importance_rf + importance_gb) / 2
feat_imp_df = pd.DataFrame({
    "feature":    feature_cols,
    "imp_rf":     importance_rf,
    "imp_gb":     importance_gb,
    "imp_moy":    importance_moy
}).sort_values("imp_moy", ascending=False)

feat_imp_df.to_csv("./feature_importance.csv", index=False)
print(f"   → Top 10 features :")
for _, row in feat_imp_df.head(10).iterrows():
    bar = "█" * int(row["imp_moy"] * 300)
    print(f"     {row['feature'][:40]:<40} {bar} {row['imp_moy']:.4f}")


# ══════════════════════════════════════════════════════════
# VISUALISATIONS
# ══════════════════════════════════════════════════════════

# ── Figure 7 : Comparaison des modèles ──────────────────
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.patch.set_facecolor("#0f1117")
fig.suptitle("Comparaison des Modèles ML — Prédiction Pannes GAB/ATM",
             fontsize=15, fontweight="bold", color="white", y=0.99)

noms    = list(resultats.keys())
palette = [COLOR_OK, COLOR_ACCENT, COLOR_PURPLE, COLOR_OK2, COLOR_PANNE]

metrics_to_plot = [
    ("f1",        "F1-Score",        "MÉTRIQUE PRINCIPALE"),
    ("precision", "Precision",       "Parmi alertes levées, % vraies"),
    ("recall",    "Recall",          "% vraies pannes détectées"),
    ("auc_roc",   "AUC-ROC",         "Performance globale classifieur"),
    ("auc_pr",    "AUC-PR",          "Adapté aux classes déséquilibrées"),
]

for idx, (metric, label, subtitle) in enumerate(metrics_to_plot):
    ax = axes[idx // 3, idx % 3]
    vals = [resultats[n][metric] for n in noms]
    bars = ax.bar(
        [n.replace(" ", "\n") for n in noms], vals,
        color=palette, alpha=0.85, edgecolor="#0f1117", linewidth=1.5
    )
    ax.set_title(f"{label}\n{subtitle}", color="white", fontweight="bold", fontsize=9)
    ax.set_ylim(0, min(1.15, max(vals) * 1.25))
    ax.axhline(0.5, color="white", linewidth=0.5, linestyle=":", alpha=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", color="white",
                fontweight="bold", fontsize=8)
    # Surbrillance du meilleur
    best_idx = vals.index(max(vals))
    bars[best_idx].set_edgecolor("white")
    bars[best_idx].set_linewidth(2.5)

# Dernier subplot : tableau récapitulatif
ax_table = axes[1, 2]
ax_table.axis("off")
table_data = []
for n in noms:
    r = resultats[n]
    table_data.append([
        n[:18], f"{r['f1']:.3f}", f"{r['auc_roc']:.3f}",
        f"{r['recall']:.3f}", f"{r['taux_detection']*100:.0f}%",
        f"{r['t_train']:.1f}s"
    ])

table = ax_table.table(
    cellText=table_data,
    colLabels=["Modèle", "F1", "AUC-ROC", "Recall", "Détection", "Temps"],
    cellLoc="center", loc="center",
    bbox=[0, 0, 1, 1]
)
table.auto_set_font_size(False)
table.set_fontsize(8)
for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_facecolor("#2a2d4d")
        cell.set_text_props(color="white", fontweight="bold")
    elif noms[r-1] == meilleur:
        cell.set_facecolor("#1a3a1a")
        cell.set_text_props(color=COLOR_OK2)
    else:
        cell.set_facecolor("#1a1d27")
        cell.set_text_props(color="#d0d0d0")
    cell.set_edgecolor("#3a3d4d")

ax_table.set_title("Tableau Comparatif\n(🏆 = meilleur modèle)",
                   color="white", fontweight="bold", fontsize=9, pad=5)

plt.tight_layout()
fig.savefig("./outputs/models/fig7_comparaison_modeles.png",
            dpi=150, bbox_inches="tight", facecolor="#0f1117")
plt.close()
print("\n✅ Figure 7 : Comparaison modèles")


# ── Figure 8 : Courbes ROC & PR + Matrices de confusion ─
fig = plt.figure(figsize=(20, 13))
fig.patch.set_facecolor("#0f1117")
fig.suptitle("Analyse Détaillée — Courbes ROC, PR & Matrices de Confusion",
             fontsize=14, fontweight="bold", color="white")

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# Courbe ROC
ax_roc = fig.add_subplot(gs[0, 0])
ax_roc.plot([0,1], [0,1], "w--", linewidth=1, alpha=0.5, label="Aléatoire (0.50)")
for (nom, r), col in zip(resultats.items(), palette):
    fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
    ax_roc.plot(fpr, tpr, color=col, linewidth=2,
                label=f"{nom[:15]} ({r['auc_roc']:.3f})")
ax_roc.set_xlabel("Taux Faux Positifs", color="#b0b0b0")
ax_roc.set_ylabel("Taux Vrais Positifs", color="#b0b0b0")
ax_roc.set_title("Courbes ROC\n(plus proche coin haut-gauche = meilleur)",
                 color="white", fontweight="bold")
ax_roc.legend(fontsize=7, framealpha=0.3)

# Courbe Precision-Recall
ax_pr = fig.add_subplot(gs[0, 1])
baseline_pr = y_test.mean()
ax_pr.axhline(baseline_pr, color="white", linestyle="--", linewidth=1,
              alpha=0.5, label=f"Baseline ({baseline_pr:.2f})")
for (nom, r), col in zip(resultats.items(), palette):
    prec, rec, _ = precision_recall_curve(y_test, r["y_proba"])
    ax_pr.plot(rec, prec, color=col, linewidth=2,
               label=f"{nom[:15]} ({r['auc_pr']:.3f})")
ax_pr.set_xlabel("Recall", color="#b0b0b0")
ax_pr.set_ylabel("Precision", color="#b0b0b0")
ax_pr.set_title("Courbes Precision-Recall\n(plus adapté aux classes déséquilibrées)",
                color="white", fontweight="bold")
ax_pr.legend(fontsize=7, framealpha=0.3)

# Feature Importance (top 20)
ax_fi = fig.add_subplot(gs[0, 2])
top20 = feat_imp_df.head(20)

def get_feat_color(col):
    if any(f"lag{l}" in col for l in [1,3,7]):      return "#a78bfa"
    if "roll" in col:                                 return COLOR_OK2
    if "tendance" in col or "acceleration" in col:   return COLOR_ACCENT
    if col in ["risque_materiel","stress_thermique",
               "score_surcharge","score_negligence",
               "score_connectivite","ratio_erreurs_tx"]: return "#fb923c"
    return COLOR_OK

feat_colors = [get_feat_color(c) for c in top20["feature"]]
ax_fi.barh(
    [c.replace("_"," ")[:30] for c in top20["feature"]][::-1],
    top20["imp_moy"].values[::-1],
    color=feat_colors[::-1], alpha=0.85, edgecolor="#0f1117"
)
ax_fi.set_title("Feature Importance Moyenne\n(RF + GB) — Top 20\n"
                "🟣Lag 🟢Roll 🟡Tendance 🟠Interaction 🔵Original",
                color="white", fontweight="bold", fontsize=8)
ax_fi.set_xlabel("Importance moyenne", color="#b0b0b0")

# Matrices de confusion — 4 modèles principaux (pas Dummy)
modeles_cm = [k for k in resultats.keys() if k != "Dummy (Stratified)"]
cm_positions = [(1,0), (1,1), (1,2), (0,2)]  # positions dans la grille

# On en affiche 4
for idx, nom_model in enumerate(modeles_cm[:4]):
    pos = cm_positions[idx]
    ax_cm = fig.add_subplot(gs[pos[0], pos[1]])
    r = resultats[nom_model]
    cm_data = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
    cm_pct  = cm_data.astype(float) / cm_data.sum() * 100

    im = ax_cm.imshow(cm_data, cmap="Blues", aspect="auto")
    ax_cm.set_xticks([0, 1]); ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["Prédit: 0", "Prédit: 1"], fontsize=8)
    ax_cm.set_yticklabels(["Réel: 0", "Réel: 1"], fontsize=8)

    for i in range(2):
        for j in range(2):
            color = "white" if cm_data[i,j] > cm_data.max()/2 else "#e0e0e0"
            ax_cm.text(j, i, f"{cm_data[i,j]:,}\n({cm_pct[i,j]:.1f}%)",
                       ha="center", va="center", color=color,
                       fontweight="bold", fontsize=8)

    # Colorier les cases TP/FN en rouge/vert
    ax_cm.add_patch(plt.Rectangle((-0.5, 0.5), 1, 1, fill=False,
                                   edgecolor=COLOR_PANNE, linewidth=2))  # FN
    ax_cm.add_patch(plt.Rectangle((0.5, 0.5), 1, 1, fill=False,
                                   edgecolor=COLOR_OK2, linewidth=2))    # TP

    is_best = "🏆 " if nom_model == meilleur else ""
    ax_cm.set_title(f"{is_best}{nom_model}\nF1={r['f1']:.3f} | Recall={r['recall']:.3f}",
                    color="white", fontweight="bold", fontsize=8)

plt.tight_layout()
fig.savefig("./outputs/models/fig8_roc_pr_cm.png",
            dpi=150, bbox_inches="tight", facecolor="#0f1117")
plt.close()
print("✅ Figure 8 : ROC, PR, Confusion Matrices")


# ── Figure 9 : Analyse métier — Score de risque ─────────
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.patch.set_facecolor("#0f1117")
fig.suptitle(f"Analyse Métier — Modèle {meilleur}\nScore de Risque & Seuil de Décision",
             fontsize=14, fontweight="bold", color="white")

best_r = resultats[meilleur]
y_scores = best_r["y_proba"]

# 9.1 Distribution des scores de probabilité
ax = axes[0]
bins = np.linspace(0, 1, 50)
ax.hist(y_scores[y_test == 0], bins=bins, alpha=0.6, color=COLOR_OK,
        label="Pas de panne", density=True)
ax.hist(y_scores[y_test == 1], bins=bins, alpha=0.6, color=COLOR_PANNE,
        label="Panne imminente", density=True)
ax.axvline(0.5, color="white", linestyle="--", linewidth=1.5, label="Seuil 0.5")
ax.set_title("Distribution des Scores de Probabilité\npar Classe Réelle",
             color="white", fontweight="bold")
ax.set_xlabel("P(panne imminente)", color="#b0b0b0")
ax.legend(fontsize=8, framealpha=0.3)

# 9.2 Courbe Precision / Recall vs Seuil
# CRUCIAL en maintenance prédictive :
# Le seuil 0.5 n'est PAS optimal !
# Un faux négatif (panne manquée) coûte BEAUCOUP plus cher
# qu'un faux positif (fausse alerte = technicien déplacé inutilement)
# → On peut abaisser le seuil pour augmenter le recall
ax = axes[1]
seuils = np.linspace(0.01, 0.99, 100)
precisions_s = [precision_score(y_test, (y_scores >= s).astype(int), zero_division=0) for s in seuils]
recalls_s    = [recall_score(y_test, (y_scores >= s).astype(int), zero_division=0) for s in seuils]
f1s_s        = [f1_score(y_test, (y_scores >= s).astype(int), zero_division=0) for s in seuils]

ax.plot(seuils, precisions_s, color=COLOR_ACCENT, linewidth=2, label="Precision")
ax.plot(seuils, recalls_s,    color=COLOR_PANNE,  linewidth=2, label="Recall")
ax.plot(seuils, f1s_s,        color=COLOR_OK2,    linewidth=2.5, label="F1-Score", linestyle="--")

# Seuil optimal (max F1)
seuil_opt = seuils[np.argmax(f1s_s)]
f1_opt    = max(f1s_s)
ax.axvline(seuil_opt, color="white", linestyle=":", linewidth=1.5,
           label=f"Seuil optimal ({seuil_opt:.2f})")
ax.axvline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="Seuil 0.5")

ax.set_title(f"Precision / Recall / F1 vs Seuil\nSeuil optimal = {seuil_opt:.2f} (F1={f1_opt:.3f})",
             color="white", fontweight="bold")
ax.set_xlabel("Seuil de décision", color="#b0b0b0")
ax.legend(fontsize=8, framealpha=0.3)
ax.set_ylim(0, 1.05)

# 9.3 Simulation impact économique
# Si on suppose :
# - Coût intervention corrective (panne non prévue) : 5000 MAD
# - Coût intervention préventive (alerte vraie)    : 1500 MAD
# - Coût fausse alerte (déplacement inutile)        : 500 MAD
# - Coût panne manquée = coût correctif complet
ax = axes[2]
COUT_CORRECTIF  = 5000  # MAD
COUT_PREVENTIF  = 1500  # MAD
COUT_FAUSSE_AL  = 500   # MAD

couts = []
for s in seuils:
    y_pred_s = (y_scores >= s).astype(int)
    tp_s = ((y_pred_s == 1) & (y_test == 1)).sum()
    fp_s = ((y_pred_s == 1) & (y_test == 0)).sum()
    fn_s = ((y_pred_s == 0) & (y_test == 1)).sum()
    cout = tp_s * COUT_PREVENTIF + fp_s * COUT_FAUSSE_AL + fn_s * COUT_CORRECTIF
    couts.append(cout)

cout_sans_modele = y_test.sum() * COUT_CORRECTIF
seuil_econ = seuils[np.argmin(couts)]
economie   = cout_sans_modele - min(couts)
economie_pct = economie / cout_sans_modele * 100

ax.plot(seuils, [c/1000 for c in couts], color=COLOR_ACCENT, linewidth=2.5)
ax.axhline(cout_sans_modele/1000, color=COLOR_PANNE, linestyle="--",
           linewidth=1.5, label=f"Sans modèle ({cout_sans_modele/1000:.0f}k MAD)")
ax.axvline(seuil_econ, color=COLOR_OK2, linestyle=":",
           linewidth=1.5, label=f"Seuil optimal ({seuil_econ:.2f})")
ax.fill_between(seuils,
                [c/1000 for c in couts],
                cout_sans_modele/1000,
                where=np.array(couts) < cout_sans_modele,
                alpha=0.2, color=COLOR_OK2, label=f"Économie max: {economie/1000:.0f}k MAD")

ax.set_title(f"Impact Économique Simulé\nÉconomie potentielle : {economie_pct:.0f}% du coût total",
             color="white", fontweight="bold")
ax.set_xlabel("Seuil de décision", color="#b0b0b0")
ax.set_ylabel("Coût total (milliers MAD)", color="#b0b0b0")
ax.legend(fontsize=7, framealpha=0.3)

plt.tight_layout()
fig.savefig("./outputs/models/fig9_analyse_metier.png",
            dpi=150, bbox_inches="tight", facecolor="#0f1117")
plt.close()
print("✅ Figure 9 : Analyse métier & impact économique")


# ══════════════════════════════════════════════════════════
# RAPPORT FINAL
# ══════════════════════════════════════════════════════════
rapport = f"""
╔═══════════════════════════════════════════════════════════╗
║   RAPPORT DE MODÉLISATION — PRÉDICTION PANNES GAB/ATM    ║
║   Banque Populaire Maroc                                  ║
╚═══════════════════════════════════════════════════════════╝

📋 CONFIGURATION DU SPLIT
──────────────────────────
  • Train : 2022 ({len(X_train):,} observations)
  • Test  : 2023 ({len(X_test):,} observations)
  • Features : {len(feature_cols)}
  • Stratégie déséquilibre : class_weight="balanced"

📊 TABLEAU DES PERFORMANCES
─────────────────────────────
{'Modèle':<25} {'F1':>7} {'Prec':>7} {'Recall':>7} {'AUC-ROC':>8} {'AUC-PR':>7}
{'─'*65}"""

for nom in noms:
    r = resultats[nom]
    star = " 🏆" if nom == meilleur else ""
    rapport += f"\n{nom:<25} {r['f1']:>7.4f} {r['precision']:>7.4f} {r['recall']:>7.4f} {r['auc_roc']:>8.4f} {r['auc_pr']:>7.4f}{star}"

rapport += f"""

🏆 MEILLEUR MODÈLE : {meilleur}
───────────────────────────────────────────
  F1-Score   : {resultats[meilleur]['f1']:.4f}
  Precision  : {resultats[meilleur]['precision']:.4f}
  Recall     : {resultats[meilleur]['recall']:.4f}
  AUC-ROC    : {resultats[meilleur]['auc_roc']:.4f}
  AUC-PR     : {resultats[meilleur]['auc_pr']:.4f}

  Pannes détectées : {resultats[meilleur]['tp']:,} / {resultats[meilleur]['tp']+resultats[meilleur]['fn']:,} ({resultats[meilleur]['taux_detection']*100:.1f}%)
  Fausses alertes  : {resultats[meilleur]['fp']:,}
  Pannes manquées  : {resultats[meilleur]['fn']:,}

🔑 TOP 10 FEATURES (par importance)
─────────────────────────────────────
{feat_imp_df[['feature','imp_moy']].head(10).to_string(index=False)}

💰 IMPACT ÉCONOMIQUE SIMULÉ
──────────────────────────────
  Hypothèses :
    • Coût intervention corrective : 5,000 MAD
    • Coût intervention préventive : 1,500 MAD
    • Coût fausse alerte           :   500 MAD

  Seuil optimal (économique) : {seuil_econ:.2f}
  Coût sans modèle           : {cout_sans_modele:,} MAD
  Coût avec modèle optimal   : {min(couts):,.0f} MAD
  Économie potentielle        : {economie:,.0f} MAD ({economie_pct:.0f}%)

💡 RECOMMANDATIONS
───────────────────
  1. Utiliser le seuil {seuil_opt:.2f} (F1 optimal) ou {seuil_econ:.2f} (coût optimal)
     selon l'arbitrage métier recall vs précision
  2. Réévaluer mensuellement avec les nouvelles données
  3. Surveiller le drift des features (température, erreurs)
  4. Intégrer les données réelles dès qu'elles sont disponibles

🚀 PROCHAINE ÉTAPE : Dashboard Streamlit (Script 05)
"""

print(rapport)
with open("./outputs/models/rapport_modelisation.txt", "w", encoding="utf-8") as f:
    f.write(rapport)

# ✅ APRÈS — gère float ET int NumPy
def numpy_to_python(v):
    if isinstance(v, (np.floating, float)):  return float(v)
    if isinstance(v, (np.integer, int)):     return int(v)
    return v

resultats_json = {
    nom: {k: numpy_to_python(v)
          for k, v in r.items()
          if k not in ["y_pred", "y_proba", "pipeline"]}
    for nom, r in resultats.items()
}
with open("./resultats_modeles.json", "w") as f:
    json.dump(resultats_json, f, indent=2)

print("\n✅ MODÉLISATION COMPLÈTE")
print("   • fig7_comparaison_modeles.png")
print("   • fig8_roc_pr_cm.png")
print("   • fig9_analyse_metier.png")
print("   • rapport_modelisation.txt")
