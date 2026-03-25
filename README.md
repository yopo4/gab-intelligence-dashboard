# 🏧 GAB Intelligence Dashboard
## Prédiction des Pannes ATM — Banque Populaire Maroc

---

### Structure du projet

```
dashboard/
├── app.py                  ← Application Streamlit principale
├── requirements.txt        ← Dépendances Python
├── README.md               ← Ce fichier
└── data/
    ├── gab_dataset.csv         ← Dataset brut (146k obs)
    ├── gab_features.csv        ← Dataset enrichi (101 features)
    ├── feature_cols.json       ← Liste des features du modèle
    ├── feature_importance.csv  ← Importance RF + GB
    └── resultats_modeles.json  ← Métriques des 5 modèles
```

---

### Installation & lancement

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Lancer le dashboard
python -m streamlit run app.py

# 3. Ouvrir dans le navigateur
# → http://localhost:8501
```

---

### Pages du dashboard

| Page | Description |
|------|-------------|
| 🏠 Vue d'ensemble | KPIs globaux, tendances, répartition par type et environnement |
| 🗺️ Analyse Géographique | Taux de panne par ville, heatmap ville × mois |
| 📊 Performance Modèles | Comparaison F1/AUC, matrices de confusion, courbe ROC |
| 🔬 Feature Importance | Top features, répartition par famille |
| ⚙️ Simulateur de Seuil | Impact du seuil sur précision/rappel/coût |
| 🏧 Scoring GAB | Calcul du score de risque d'un GAB en temps réel |

---

### Données

- **Période** : Janvier 2022 — Décembre 2023
- **GAB** : 200 unités simulées, 13 villes du Maroc
- **Features** : 101 (originales + lag + rolling + interaction)
- **Target** : `panne_sous_48h` (panne imminente dans les 48h)
- **Taux de panne** : ~9.8% (classe déséquilibrée)

---

### Stack technique

- **Dashboard** : Streamlit 1.32+
- **Visualisation** : Plotly
- **ML** : scikit-learn (LogisticRegression, RandomForest, GradientBoosting)
- **Data** : pandas, numpy

---

*Projet de stage — Data Science — Banque Populaire du Maroc*
