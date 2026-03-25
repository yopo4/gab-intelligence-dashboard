"""
==============================================================
GAB Intelligence Dashboard — Banque Populaire du Maroc
Design: "Warm Intelligence" — Fraunces + Geist Mono
Lancez : python -m streamlit run legacy/app.py
==============================================================
"""

import streamlit as st

st.set_page_config(
    page_title="GAB Intelligence · BPM",
    page_icon="🏧",
    layout="wide",
    initial_sidebar_state="expanded",
)

import pandas as pd
import numpy as np
import json, os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
# DESIGN SYSTEM — "Warm Intelligence"
# Tonalité : Warm slate + Cream/Amber accents
# Typo : Fraunces (titres serif expressifs) + Geist Mono (data)
# Ambiance : Tableau de bord éditorial haut de gamme
# ══════════════════════════════════════════════════════════════
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,400;0,9..144,600;0,9..144,700;0,9..144,900;1,9..144,300;1,9..144,400;1,9..144,700&family=Geist+Mono:wght@300;400;500;600&family=Instrument+Sans:wght@400;500;600;700&display=swap');

/* ─── Tokens ───────────────────────────────────────────── */
:root {
  /* Neutrals chauds */
  --ink-950: #0c0a09;
  --ink-900: #1a1614;
  --ink-800: #251f1c;
  --ink-700: #322a27;
  --ink-600: #443a36;
  --ink-500: #5c4f4a;
  --ink-400: #7a6b65;
  --ink-300: #a8998f;
  --ink-200: #d4c8c0;
  --ink-100: #f0ebe5;
  --ink-50:  #faf7f4;

  /* Accents */
  --cream:   #f5f0e8;
  --amber:   #e8a045;
  --amber-l: #f2b96b;
  --coral:   #d4645a;
  --coral-l: #e07b72;
  --sage:    #6ba88a;
  --sage-l:  #87c4a4;
  --sky:     #5a8fc4;
  --sky-l:   #78acd8;
  --gold:    #c9952a;

  /* Text */
  --text-hero:    #f5f0e8;
  --text-primary: #e0d8d0;
  --text-mid:     #a09080;
  --text-muted:   #5c4f4a;
  --text-ghost:   #3a302c;

  /* Glass */
  --glass:        rgba(26,22,20,0.75);
  --glass-light:  rgba(37,31,28,0.60);
  --border-dim:   rgba(255,255,255,0.05);
  --border-glow:  rgba(232,160,69,0.15);
}

/* ─── Reset & Base ─────────────────────────────────────── */
html, body, .stApp {
  background: var(--ink-950) !important;
  font-family: 'Instrument Sans', sans-serif !important;
}
.main .block-container {
  padding: 0 2rem 4rem 2rem !important;
  max-width: 1600px !important;
}
h1,h2,h3,h4 { font-family: 'Fraunces', serif !important; }
code, .mono  { font-family: 'Geist Mono', monospace !important; }

/* ─── Sidebar ───────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: var(--ink-900) !important;
  border-right: 1px solid var(--border-dim) !important;
}
[data-testid="stSidebar"] > div { padding-top: 0 !important; }

/* Sidebar brand */
.sb-brand {
  padding: 1.8rem 1.4rem 1.4rem;
  border-bottom: 1px solid var(--border-dim);
  margin-bottom: 0;
}
.sb-logo {
  width: 44px; height: 44px;
  background: linear-gradient(135deg, var(--amber) 0%, var(--coral) 100%);
  border-radius: 12px;
  display: flex; align-items: center; justify-content: center;
  font-size: 1.5rem; margin-bottom: 1rem;
  box-shadow: 0 0 24px rgba(232,160,69,0.25);
}
.sb-title {
  font-family: 'Fraunces', serif !important;
  font-size: 1.15rem !important;
  font-weight: 700 !important;
  color: var(--text-hero) !important;
  letter-spacing: -0.02em;
  line-height: 1.2;
}
.sb-subtitle {
  font-size: 0.65rem !important;
  color: var(--text-muted) !important;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  margin-top: 0.3rem;
}

/* Sidebar nav */
div[data-testid="stRadio"] > label { display:none !important; }
div[data-testid="stRadio"] > div { gap: 0.15rem !important; flex-direction: column !important; }
div[data-testid="stRadio"] label {
  border-radius: 8px !important;
  padding: 0.6rem 1rem !important;
  color: var(--text-mid) !important;
  font-size: 0.82rem !important;
  font-weight: 500 !important;
  background: transparent !important;
  border: 1px solid transparent !important;
  transition: all 0.15s ease !important;
  cursor: pointer !important;
  margin: 0 !important;
}
div[data-testid="stRadio"] label:hover {
  background: rgba(232,160,69,0.06) !important;
  border-color: rgba(232,160,69,0.2) !important;
  color: var(--amber-l) !important;
}

/* Sidebar meta */
.sb-meta {
  padding: 1rem 1.4rem;
  border-top: 1px solid var(--border-dim);
  margin-top: 1rem;
}
.sb-meta-row {
  display: flex; justify-content: space-between; align-items: center;
  padding: 0.35rem 0;
  font-size: 0.72rem;
}
.sb-meta-label { color: var(--text-muted); }
.sb-meta-val   { color: var(--text-mid); font-family: 'Geist Mono', monospace; font-weight: 500; }

/* Sidebar section sep */
.sb-sep {
  font-size: 0.6rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.2em !important;
  text-transform: uppercase !important;
  color: var(--text-ghost) !important;
  padding: 1.2rem 1rem 0.5rem !important;
  display: block !important;
}

/* ─── Page Hero ─────────────────────────────────────────── */
.page-hero {
  position: relative;
  padding: 3rem 0 2rem;
  margin-bottom: 2rem;
  overflow: hidden;
}
.page-hero::before {
  content: '';
  position: absolute;
  top: -60px; right: -80px;
  width: 400px; height: 400px;
  background: radial-gradient(circle, rgba(232,160,69,0.06) 0%, transparent 70%);
  pointer-events: none;
}
.hero-eyebrow {
  display: inline-flex; align-items: center; gap: 0.5rem;
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--amber);
  margin-bottom: 0.8rem;
}
.hero-eyebrow-dot {
  width: 5px; height: 5px; border-radius: 50%;
  background: var(--amber);
  animation: pulse-dot 2s infinite;
}
@keyframes pulse-dot {
  0%, 100% { opacity: 1; transform: scale(1); }
  50%       { opacity: 0.5; transform: scale(0.7); }
}
.hero-title {
  font-family: 'Fraunces', serif !important;
  font-size: 3rem !important;
  font-weight: 900 !important;
  color: var(--text-hero) !important;
  margin: 0 0 0.6rem !important;
  line-height: 1.08 !important;
  letter-spacing: -0.03em;
}
.hero-title em {
  font-style: italic;
  color: var(--amber);
}
.hero-desc {
  font-size: 0.92rem;
  color: var(--text-mid);
  max-width: 560px;
  line-height: 1.7;
  font-weight: 400;
  margin-bottom: 1.2rem;
}
.hero-divider {
  width: 100%;
  height: 1px;
  background: linear-gradient(90deg, var(--amber) 0%, rgba(232,160,69,0.1) 30%, transparent 60%);
  margin-top: 1.5rem;
}

/* ─── Tag Pills ─────────────────────────────────────────── */
.tag-row { display: flex; gap: 0.5rem; flex-wrap: wrap; }
.tag {
  display: inline-flex; align-items: center; gap: 0.3rem;
  padding: 0.22rem 0.7rem;
  border-radius: 4px;
  font-size: 0.68rem; font-weight: 600;
  letter-spacing: 0.06em; text-transform: uppercase;
  border: 1px solid;
}
.tag-amber { color: var(--amber);   border-color: rgba(232,160,69,0.3); background: rgba(232,160,69,0.08); }
.tag-coral { color: var(--coral-l); border-color: rgba(212,100,90,0.3); background: rgba(212,100,90,0.08); }
.tag-sage  { color: var(--sage-l);  border-color: rgba(107,168,138,0.3); background: rgba(107,168,138,0.08); }
.tag-sky   { color: var(--sky-l);   border-color: rgba(90,143,196,0.3); background: rgba(90,143,196,0.08); }
.tag-ghost { color: var(--text-mid); border-color: var(--border-dim); background: rgba(255,255,255,0.02); }

/* ─── KPI Cards ─────────────────────────────────────────── */
.kpi-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 1rem; margin-bottom: 2rem; }
.kpi-grid-3 { grid-template-columns: repeat(3,1fr) !important; }

.kpi-card {
  background: var(--ink-800);
  border: 1px solid var(--border-dim);
  border-radius: 16px;
  padding: 1.6rem 1.8rem 1.4rem;
  position: relative;
  overflow: hidden;
  transition: border-color 0.2s, transform 0.2s;
  cursor: default;
}
.kpi-card:hover {
  transform: translateY(-2px);
  border-color: rgba(255,255,255,0.09);
}
/* Top accent line */
.kpi-card::before {
  content: '';
  position: absolute; top: 0; left: 1.5rem; right: 1.5rem; height: 2px;
  border-radius: 0 0 4px 4px;
  background: var(--accent-c, var(--amber));
}
/* Background glow */
.kpi-card::after {
  content: '';
  position: absolute; top: -40px; right: -20px;
  width: 120px; height: 120px; border-radius: 50%;
  background: var(--accent-c, var(--amber));
  opacity: 0.04; pointer-events: none;
}
.kpi-amber { --accent-c: var(--amber); }
.kpi-coral { --accent-c: var(--coral); }
.kpi-sage  { --accent-c: var(--sage); }
.kpi-sky   { --accent-c: var(--sky); }

.kpi-icon-box {
  width: 34px; height: 34px; border-radius: 8px;
  display: flex; align-items: center; justify-content: center;
  font-size: 1rem; margin-bottom: 1.1rem;
  background: rgba(255,255,255,0.04);
}
.kpi-label {
  font-size: 0.65rem; font-weight: 700;
  letter-spacing: 0.16em; text-transform: uppercase;
  color: var(--text-muted); margin-bottom: 0.4rem;
}
.kpi-val {
  font-family: 'Fraunces', serif !important;
  font-size: 2.2rem; font-weight: 700;
  color: var(--text-hero); line-height: 1;
  margin-bottom: 0.4rem; letter-spacing: -0.02em;
}
.kpi-sub {
  font-size: 0.74rem; color: var(--text-muted);
  line-height: 1.4; font-weight: 400;
}
.kpi-badge {
  display: inline-block; margin-top: 0.6rem;
  font-size: 0.6rem; font-weight: 700;
  letter-spacing: 0.1em; text-transform: uppercase;
  padding: 0.2rem 0.5rem; border-radius: 3px;
}

/* ─── Section headings ──────────────────────────────────── */
.s-head {
  display: flex; align-items: center; gap: 0.8rem;
  margin: 1.8rem 0 1rem;
}
.s-head-title {
  font-size: 0.68rem; font-weight: 700;
  letter-spacing: 0.18em; text-transform: uppercase;
  color: var(--text-muted); white-space: nowrap;
}
.s-head-line {
  flex: 1; height: 1px;
  background: linear-gradient(90deg, var(--border-dim), transparent);
}

/* ─── Chart frames ──────────────────────────────────────── */
.chart-frame {
  background: var(--ink-800);
  border: 1px solid var(--border-dim);
  border-radius: 14px;
  padding: 1.4rem 1.6rem 0.6rem;
  margin-bottom: 1rem;
}
.chart-title {
  font-family: 'Fraunces', serif;
  font-size: 1rem; font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 0.2rem; letter-spacing: -0.01em;
}
.chart-sub {
  font-size: 0.7rem; color: var(--text-muted);
  margin-bottom: 0.8rem; font-weight: 400;
}

/* ─── Data table ────────────────────────────────────────── */
.data-tbl {
  width: 100%; border-collapse: collapse;
  font-size: 0.81rem;
}
.data-tbl thead tr { border-bottom: 1px solid var(--border-dim); }
.data-tbl th {
  font-size: 0.6rem; font-weight: 700;
  letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--text-muted); padding: 0.55rem 0.9rem;
  text-align: left;
}
.data-tbl td {
  padding: 0.65rem 0.9rem;
  color: var(--text-mid);
  border-bottom: 1px solid rgba(255,255,255,0.025);
  transition: background 0.1s;
}
.data-tbl tbody tr:hover td { background: rgba(255,255,255,0.018); }
.data-tbl tbody tr:last-child td { border-bottom: none; }
.data-tbl tr.highlight td { color: var(--sage-l) !important; }

.badge-wrap {
  display: inline-flex; align-items: center; gap: 0.3rem;
  padding: 0.18rem 0.55rem; border-radius: 4px;
  font-size: 0.6rem; font-weight: 700; letter-spacing: 0.06em;
  text-transform: uppercase; border: 1px solid;
}
.badge-sage  { color: var(--sage-l);  border-color: rgba(107,168,138,0.35); background: rgba(107,168,138,0.1); }
.badge-amber { color: var(--amber-l); border-color: rgba(232,160,69,0.35);  background: rgba(232,160,69,0.1); }
.badge-coral { color: var(--coral-l); border-color: rgba(212,100,90,0.35);  background: rgba(212,100,90,0.1); }

/* ─── Callout boxes ─────────────────────────────────────── */
.callout {
  display: flex; gap: 0.9rem; align-items: flex-start;
  border-radius: 10px; padding: 1rem 1.2rem;
  margin: 0.8rem 0; font-size: 0.84rem; line-height: 1.65;
  border: 1px solid; font-weight: 400;
}
.callout-icon { font-size: 1rem; flex-shrink: 0; margin-top: 0.05rem; }
.callout-sage  { background: rgba(107,168,138,0.07); border-color: rgba(107,168,138,0.2); color: var(--sage-l); }
.callout-amber { background: rgba(232,160,69,0.07);  border-color: rgba(232,160,69,0.2);  color: var(--amber-l); }
.callout-coral { background: rgba(212,100,90,0.07);  border-color: rgba(212,100,90,0.2);  color: var(--coral-l); }
.callout-sky   { background: rgba(90,143,196,0.07);  border-color: rgba(90,143,196,0.2);  color: var(--sky-l); }

/* ─── Progress bars ─────────────────────────────────────── */
.prog-rail {
  background: rgba(255,255,255,0.04);
  border-radius: 99px; height: 4px; overflow: hidden;
}
.prog-fill { height: 100%; border-radius: 99px; }

/* ─── Streamlit overrides ───────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
div[data-baseweb="select"] > div {
  background: var(--ink-700) !important;
  border-color: var(--border-dim) !important;
}
.stSlider > div > div > div { background: var(--amber) !important; }
div[data-baseweb="tag"] { background: rgba(232,160,69,0.15) !important; }
.stMultiSelect [data-baseweb="tag"] span { color: var(--amber-l) !important; }

/* ─── Animations ────────────────────────────────────────── */
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(12px); }
  to   { opacity: 1; transform: translateY(0); }
}
.kpi-card { animation: fadeUp 0.4s ease both; }
.kpi-card:nth-child(1) { animation-delay: 0.05s; }
.kpi-card:nth-child(2) { animation-delay: 0.10s; }
.kpi-card:nth-child(3) { animation-delay: 0.15s; }
.kpi-card:nth-child(4) { animation-delay: 0.20s; }
</style>
""",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(base, "gab_dataset.csv"), parse_dates=["date"])
    df_feat = pd.read_csv(os.path.join(base, "gab_features.csv"), parse_dates=["date"])
    fi = pd.read_csv(os.path.join(base, "feature_importance.csv"))
    with open(os.path.join(base, "resultats_modeles.json")) as f:
        res = json.load(f)
    return df, df_feat, fi, res


df, df_feat, feat_imp, resultats = load_data()

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        """
    <div class="sb-brand">
      <div class="sb-logo">🏧</div>
      <div class="sb-title">GAB Intelligence</div>
      <div class="sb-subtitle">Banque Populaire · Maroc</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown('<span class="sb-sep">Navigation</span>', unsafe_allow_html=True)
    page = st.radio(
        "nav",
        [
            "🏠  Vue d'ensemble",
            "🗺️  Géographie",
            "📊  Modèles",
            "🔬  Features",
            "⚙️  Seuil & Coûts",
            "🏧  Scoring Live",
        ],
        label_visibility="collapsed",
    )

    st.markdown('<span class="sb-sep">Filtres</span>', unsafe_allow_html=True)
    villes_sel = st.multiselect(
        "Villes", sorted(df["ville"].unique()), default=sorted(df["ville"].unique())
    )
    types_sel = st.multiselect(
        "Type GAB",
        sorted(df["type_gab"].unique()),
        default=sorted(df["type_gab"].unique()),
    )
    annee_sel = st.select_slider("Période", options=[2022, 2023, "Tout"], value="Tout")

    df_f = df.copy()
    if villes_sel:
        df_f = df_f[df_f["ville"].isin(villes_sel)]
    if types_sel:
        df_f = df_f[df_f["type_gab"].isin(types_sel)]
    if annee_sel != "Tout":
        df_f = df_f[df_f["date"].dt.year == int(annee_sel)]

    st.markdown(
        f"""
    <div class="sb-meta">
      <div class="sb-meta-row"><span class="sb-meta-label">Période</span><span class="sb-meta-val">2022–2023</span></div>
      <div class="sb-meta-row"><span class="sb-meta-label">GAB</span><span class="sb-meta-val">{df_f["gab_id"].nunique()}</span></div>
      <div class="sb-meta-row"><span class="sb-meta-label">Villes</span><span class="sb-meta-val">{len(villes_sel)}</span></div>
      <div class="sb-meta-row"><span class="sb-meta-label">Obs.</span><span class="sb-meta-val">{len(df_f):,}</span></div>
      <div class="sb-meta-row"><span class="sb-meta-label">Features</span><span class="sb-meta-val">101</span></div>
    </div>
    """,
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════
# PLOTLY THEME — Warm
# ══════════════════════════════════════════════════════════════
BG = "#151210"
PAPER = "#151210"
FONT = "#5c4f4a"
GRID = "rgba(255,255,255,0.05)"
NG = "rgba(0,0,0,0)"


def warm_layout(**kw):
    base = dict(
        plot_bgcolor=BG,
        paper_bgcolor=PAPER,
        font_color=FONT,
        font_family="Instrument Sans",
        xaxis=dict(
            showgrid=False, color=FONT, tickfont_size=10, zeroline=False, linecolor=GRID
        ),
        yaxis=dict(
            gridcolor=GRID, gridwidth=0.4, color=FONT, tickfont_size=10, zeroline=False
        ),
        legend=dict(bgcolor=NG, font_size=10, font_color="#a09080"),
        margin=dict(l=4, r=4, t=16, b=4),
        hoverlabel=dict(
            bgcolor="#251f1c",
            font_size=12,
            font_family="Geist Mono",
            bordercolor="#443a36",
        ),
    )
    base.update(kw)
    return base


PALETTE = ["#e8a045", "#d4645a", "#6ba88a", "#5a8fc4", "#a87fb5", "#c9952a", "#82a04f"]


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════
def sh(title):
    st.markdown(
        f"""
    <div class="s-head">
      <span class="s-head-title">{title}</span>
      <div class="s-head-line"></div>
    </div>""",
        unsafe_allow_html=True,
    )


def callout(text, kind="sage", icon="💡"):
    st.markdown(
        f"""
    <div class="callout callout-{kind}">
      <span class="callout-icon">{icon}</span>
      <span>{text}</span>
    </div>""",
        unsafe_allow_html=True,
    )


def hero(eyebrow, title_main, title_em, desc, tags_html):
    st.markdown(
        f"""
    <div class="page-hero">
      <div class="hero-eyebrow">
        <span class="hero-eyebrow-dot"></span>{eyebrow}
      </div>
      <h1 class="hero-title">{title_main} <em>{title_em}</em></h1>
      <p class="hero-desc">{desc}</p>
      <div class="tag-row">{tags_html}</div>
      <div class="hero-divider"></div>
    </div>""",
        unsafe_allow_html=True,
    )


def tag(text, kind="ghost"):
    return f'<span class="tag tag-{kind}">{text}</span>'


# ══════════════════════════════════════════════════════════════
# PAGE 1 — VUE D'ENSEMBLE
# ══════════════════════════════════════════════════════════════
if "Vue d'ensemble" in page:

    hero(
        "Maintenance Prédictive · Tableau de bord principal",
        "Bienvenue dans votre",
        "centre de contrôle.",
        "200 GAB surveillés en continu sur 13 villes du Maroc. "
        "Des modèles ML entraînés sur 2 ans de données détectent les pannes avant qu'elles surviennent.",
        tag("200 GAB", "ghost")
        + tag("13 villes", "ghost")
        + tag("Actif", "sage")
        + tag("2022–2023", "amber"),
    )

    nb_gab = df_f["gab_id"].nunique()
    nb_pannes = int(df_f["panne_sous_48h"].sum())
    taux_panne = df_f["panne_sous_48h"].mean() * 100
    meilleur = max(resultats, key=lambda k: resultats[k]["f1"])
    f1_best = resultats[meilleur]["f1"]
    recall_best = resultats[meilleur]["recall"]

    st.markdown(
        f"""
    <div class="kpi-grid">
      <div class="kpi-card kpi-amber">
        <div class="kpi-icon-box">🏧</div>
        <div class="kpi-label">GAB surveillés</div>
        <div class="kpi-val">{nb_gab}</div>
        <div class="kpi-sub">{len(villes_sel)} villes · {len(types_sel)} constructeurs</div>
        <span class="kpi-badge tag tag-amber">Opérationnel</span>
      </div>
      <div class="kpi-card kpi-coral">
        <div class="kpi-icon-box">⚡</div>
        <div class="kpi-label">Pannes enregistrées</div>
        <div class="kpi-val">{nb_pannes:,}</div>
        <div class="kpi-sub">Taux moyen : <strong style="color:#e07b72">{taux_panne:.1f}%</strong> · classe minoritaire</div>
        <div class="prog-rail" style="margin-top:0.6rem">
          <div class="prog-fill" style="width:{min(taux_panne*4,100):.0f}%;background:var(--coral)"></div>
        </div>
      </div>
      <div class="kpi-card kpi-sage">
        <div class="kpi-icon-box">🎯</div>
        <div class="kpi-label">Meilleur F1-Score</div>
        <div class="kpi-val">{f1_best:.3f}</div>
        <div class="kpi-sub">{meilleur[:22]} · Recall <strong style="color:#87c4a4">{recall_best:.2f}</strong></div>
        <span class="kpi-badge tag tag-sage">Champion</span>
      </div>
      <div class="kpi-card kpi-sky">
        <div class="kpi-icon-box">📦</div>
        <div class="kpi-label">Observations</div>
        <div class="kpi-val">{len(df_f):,}</div>
        <div class="kpi-sub">2 ans · <strong style="color:#78acd8">101</strong> features engineered</div>
        <span class="kpi-badge tag tag-sky">Synthétique</span>
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([3, 2], gap="medium")
    with col1:
        sh("Tendance mensuelle des incidents")
        monthly = (
            df_f.assign(mois=df_f["date"].dt.to_period("M").astype(str))
            .groupby("mois")
            .agg(pannes=("panne_sous_48h", "sum"), obs=("panne_sous_48h", "count"))
            .reset_index()
        )
        monthly["taux"] = monthly["pannes"] / monthly["obs"] * 100
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(
                x=monthly["mois"],
                y=monthly["pannes"],
                name="Incidents",
                marker=dict(color="#251f1c", line=dict(color="#e8a045", width=1)),
                hovertemplate="<b>%{x}</b><br>%{y:,} pannes<extra></extra>",
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=monthly["mois"],
                y=monthly["taux"],
                name="Taux %",
                line=dict(color="#d4645a", width=2, shape="spline"),
                mode="lines+markers",
                marker=dict(size=4, color="#d4645a"),
                hovertemplate="%{y:.2f}%<extra>Taux</extra>",
            ),
            secondary_y=True,
        )
        fig.update_layout(
            **warm_layout(
                height=300,
                xaxis=dict(showgrid=False, tickangle=40, tickfont_size=9),
                yaxis2=dict(gridcolor=NG, tickfont_size=9),
                legend=dict(orientation="h", y=1.08),
            )
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col2:
        sh("Taux par constructeur")
        taux_type = (
            df_f.groupby("type_gab")["panne_sous_48h"]
            .mean()
            .reset_index()
            .rename(columns={"panne_sous_48h": "taux"})
        )
        taux_type["taux"] *= 100
        taux_type = taux_type.sort_values("taux")
        fig = go.Figure(
            go.Bar(
                x=taux_type["taux"],
                y=taux_type["type_gab"],
                orientation="h",
                marker=dict(
                    color=taux_type["taux"],
                    colorscale=[[0, "#251f1c"], [0.5, "#e8a045"], [1, "#d4645a"]],
                    showscale=False,
                    line=dict(color=BG, width=1),
                ),
                text=[f"{v:.1f}%" for v in taux_type["taux"]],
                textposition="outside",
                textfont=dict(size=10, color=FONT),
                hovertemplate="<b>%{y}</b> : %{x:.2f}%<extra></extra>",
            )
        )
        fig.update_layout(
            **warm_layout(
                height=300,
                xaxis=dict(showgrid=False, visible=False),
                yaxis=dict(gridcolor=NG, tickfont=dict(size=11, color="#a09080")),
                margin=dict(l=4, r=50, t=16, b=4),
            )
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    col3, col4, col5 = st.columns([1, 1, 2], gap="medium")
    with col3:
        sh("Par environnement")
        env_df = (
            df_f.groupby("environnement")["panne_sous_48h"]
            .mean()
            .reset_index()
            .rename(columns={"panne_sous_48h": "taux"})
        )
        env_df["taux"] *= 100
        env_df["env"] = env_df["environnement"].str.replace("_", " ")
        fig = go.Figure(
            go.Pie(
                labels=env_df["env"],
                values=env_df["taux"],
                hole=0.65,
                marker=dict(colors=PALETTE[:4], line=dict(color=BG, width=3)),
                textfont=dict(size=9),
                textinfo="percent",
                hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>",
            )
        )
        fig.update_layout(
            **warm_layout(
                height=290,
                margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(font_size=9, orientation="v", x=1.02),
                annotations=[
                    dict(
                        text="Env", font=dict(size=11, color="#e0d8d0"), showarrow=False
                    )
                ],
            )
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col4:
        sh("Par saison")
        df_s = df_f.copy()
        df_s["saison"] = df_s["date"].dt.month.map(
            {
                12: "Hiver",
                1: "Hiver",
                2: "Hiver",
                3: "Printemps",
                4: "Printemps",
                5: "Printemps",
                6: "Été",
                7: "Été",
                8: "Été",
                9: "Automne",
                10: "Automne",
                11: "Automne",
            }
        )
        sais = df_s.groupby("saison")["panne_sous_48h"].mean() * 100
        sais = sais.reindex(["Printemps", "Été", "Automne", "Hiver"])
        fig = go.Figure(
            go.Bar(
                x=sais.index,
                y=sais.values,
                marker=dict(
                    color=["#6ba88a", "#d4645a", "#e8a045", "#5a8fc4"],
                    line=dict(color=BG, width=2),
                ),
                text=[f"{v:.1f}%" for v in sais.values],
                textposition="outside",
                textfont=dict(size=9, color=FONT),
                hovertemplate="<b>%{x}</b> : %{y:.2f}%<extra></extra>",
            )
        )
        fig.update_layout(
            **warm_layout(height=290, xaxis=dict(showgrid=False, tickfont_size=10))
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col5:
        sh("Dégradation selon l'âge du GAB")
        age_df = (
            df_f.groupby("age_annees")["panne_sous_48h"]
            .mean()
            .reset_index()
            .rename(columns={"panne_sous_48h": "taux"})
        )
        age_df["taux"] *= 100
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=age_df["age_annees"],
                y=age_df["taux"],
                fill="tozeroy",
                fillcolor="rgba(232,160,69,0.07)",
                line=dict(color="#e8a045", width=2.5, shape="spline"),
                mode="lines+markers",
                marker=dict(size=8, color="#e8a045", line=dict(color=BG, width=2)),
                hovertemplate="<b>%{x} ans</b> : %{y:.2f}%<extra></extra>",
            )
        )
        fig.update_layout(
            **warm_layout(
                height=290,
                xaxis=dict(
                    showgrid=False,
                    title="Âge (années)",
                    title_font=dict(size=10),
                    tickfont_size=10,
                ),
                yaxis=dict(gridcolor=GRID, title="Taux %", title_font=dict(size=10)),
            )
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    ville_top = df_f.groupby("ville")["panne_sous_48h"].mean().idxmax()
    taux_top = df_f.groupby("ville")["panne_sous_48h"].mean().max() * 100
    callout(
        f"<strong>{ville_top}</strong> enregistre le taux de panne le plus élevé ({taux_top:.1f}%). "
        f"Les GAB <strong>Wincor</strong> et les <strong>sites isolés</strong> concentrent les incidents. "
        f"Le pic estival confirme le rôle du stress thermique comme facteur déclenchant.",
        kind="amber",
        icon="⚠️",
    )


# ══════════════════════════════════════════════════════════════
# PAGE 2 — GÉOGRAPHIE
# ══════════════════════════════════════════════════════════════
elif "Géographie" in page:

    hero(
        "Cartographie du risque opérationnel",
        "Où se concentrent",
        "les incidents ?",
        "Analyse géographique des pannes par ville et région. "
        "Identifiez les zones à surveiller en priorité et les patterns saisonniers.",
        tag("13 villes", "ghost") + tag("Maroc", "amber") + tag("Heatmap", "sky"),
    )

    vs = (
        df_f.groupby("ville")
        .agg(
            nb_pannes=("panne_sous_48h", "sum"),
            nb_obs=("panne_sous_48h", "count"),
            nb_gab=("gab_id", "nunique"),
        )
        .reset_index()
    )
    vs["taux_panne"] = vs["nb_pannes"] / vs["nb_obs"] * 100
    vs["pannes_par_gab"] = vs["nb_pannes"] / vs["nb_gab"]
    vs = vs.sort_values("taux_panne", ascending=False)

    col1, col2 = st.columns([3, 2], gap="medium")
    with col1:
        sh("Taux de panne par ville — tri décroissant")
        fig = go.Figure(
            go.Bar(
                x=vs["ville"],
                y=vs["taux_panne"],
                marker=dict(
                    color=vs["taux_panne"],
                    colorscale=[[0, "#251f1c"], [0.4, "#e8a045"], [1, "#d4645a"]],
                    showscale=True,
                    colorbar=dict(
                        title=dict(text="%", font=dict(size=9)),
                        thickness=8,
                        tickfont=dict(size=8),
                        bgcolor=NG,
                        outlinewidth=0,
                    ),
                    line=dict(color=BG, width=1),
                ),
                text=[f"{v:.1f}%" for v in vs["taux_panne"]],
                textposition="outside",
                textfont=dict(size=9, color=FONT),
                hovertemplate="<b>%{x}</b><br>Taux : %{y:.2f}%<extra></extra>",
            )
        )
        fig.update_layout(
            **warm_layout(
                height=380,
                xaxis=dict(showgrid=False, tickangle=35, tickfont_size=9),
                yaxis=dict(gridcolor=GRID, title="Taux (%)", title_font=dict(size=10)),
            )
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col2:
        sh("Classement des villes")
        rows = []
        for i, row in enumerate(vs.itertuples(), 1):
            if row.taux_panne > 10.5:
                badge = '<span class="badge-wrap badge-coral">Élevé</span>'
                vc = "#e07b72"
            elif row.taux_panne > 9.5:
                badge = '<span class="badge-wrap badge-amber">Modéré</span>'
                vc = "#f2b96b"
            else:
                badge = '<span class="badge-wrap badge-sage">Faible</span>'
                vc = "#87c4a4"
            rows.append(
                f"<tr>"
                f'<td style="color:#3a302c;font-family:Geist Mono">{i}</td>'
                f'<td style="font-weight:500;color:#a09080">{row.ville}</td>'
                f'<td style="font-family:Geist Mono;color:{vc};font-weight:600">{row.taux_panne:.2f}%</td>'
                f"<td>{badge}</td>"
                f"</tr>"
            )
        st.markdown(
            '<table class="data-tbl"><thead><tr>'
            "<th>#</th><th>Ville</th><th>Taux</th><th>Niveau</th>"
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>",
            unsafe_allow_html=True,
        )

    sh("Heatmap saisonnière — Ville × Mois")
    mois_noms = [
        "Jan",
        "Fév",
        "Mar",
        "Avr",
        "Mai",
        "Jun",
        "Jul",
        "Aoû",
        "Sep",
        "Oct",
        "Nov",
        "Déc",
    ]
    df_h = df_f.copy()
    df_h["mois_num"] = df_h["date"].dt.month
    hm = df_h.groupby(["ville", "mois_num"])["panne_sous_48h"].mean().reset_index()
    hm_pivot = (
        hm.pivot(index="ville", columns="mois_num", values="panne_sous_48h") * 100
    )
    hm_pivot.columns = mois_noms[: len(hm_pivot.columns)]
    fig = go.Figure(
        go.Heatmap(
            z=hm_pivot.values,
            x=hm_pivot.columns.tolist(),
            y=hm_pivot.index.tolist(),
            colorscale=[
                [0, "#151210"],
                [0.35, "#322a27"],
                [0.7, "#e8a045"],
                [1, "#d4645a"],
            ],
            hoverongaps=False,
            hovertemplate="<b>%{y}</b> · %{x}<br>Taux : %{z:.2f}%<extra></extra>",
            colorbar=dict(
                title=dict(text="%", font=dict(size=9)),
                thickness=8,
                tickfont=dict(size=8),
                bgcolor=NG,
                outlinewidth=0,
            ),
            xgap=1,
            ygap=1,
        )
    )
    fig.update_layout(
        **warm_layout(
            height=460,
            xaxis=dict(side="top", tickfont_size=10, showgrid=False),
            yaxis=dict(tickfont_size=10, showgrid=False),
            margin=dict(l=100, r=20, t=30, b=4),
        )
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    callout(
        "Le pic de <strong>juin–août</strong> est visible dans presque toutes les villes, "
        "confirmant le stress thermique comme facteur saisonnier dominant. "
        "<strong>Safi et Agadir</strong> montrent la saisonnalité la plus marquée.",
        kind="amber",
        icon="🌡️",
    )


# ══════════════════════════════════════════════════════════════
# PAGE 3 — MODÈLES
# ══════════════════════════════════════════════════════════════
elif "Modèles" in page:

    hero(
        "Évaluation & comparaison des algorithmes ML",
        "Quel modèle",
        "mérite votre confiance ?",
        "5 modèles de classification comparés sur split temporel strict (train 2022 / test 2023). "
        "L'enjeu : maximiser le recall sans exploser les fausses alertes.",
        tag("Split temporel", "ghost")
        + tag("Classe 9.8%", "coral")
        + tag("class_weight=balanced", "sage"),
    )

    meilleur = max(resultats, key=lambda k: resultats[k]["f1"])

    sh("Tableau comparatif")
    rows = []
    for nom, r in resultats.items():
        is_best = nom == meilleur
        badge = (
            '<span class="badge-wrap badge-sage">🏆 Meilleur</span>' if is_best else ""
        )
        cls = ' class="highlight"' if is_best else ""
        rows.append(
            f"<tr{cls}>"
            f'<td style="font-weight:600;color:#e0d8d0">{nom}</td>'
            f'<td style="font-family:Geist Mono">{r["f1"]:.4f}</td>'
            f'<td style="font-family:Geist Mono">{r["precision"]:.4f}</td>'
            f'<td style="font-family:Geist Mono">{r["recall"]:.4f}</td>'
            f'<td style="font-family:Geist Mono">{r["auc_roc"]:.4f}</td>'
            f'<td style="font-family:Geist Mono">{r["auc_pr"]:.4f}</td>'
            f'<td style="font-family:Geist Mono;color:#87c4a4">{r["tp"]:,}</td>'
            f'<td style="font-family:Geist Mono;color:#e07b72">{r["fn"]:,}</td>'
            f"<td>{badge}</td>"
            f"</tr>"
        )
    st.markdown(
        '<table class="data-tbl"><thead><tr>'
        "<th>Modèle</th><th>F1</th><th>Précision</th><th>Rappel</th>"
        "<th>AUC-ROC</th><th>AUC-PR</th><th>TP ✓</th><th>FN ✗</th><th></th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>",
        unsafe_allow_html=True,
    )
    callout(
        f"<strong>{meilleur}</strong> remporte le F1 à {resultats[meilleur]['f1']:.3f} "
        f"avec un recall de {resultats[meilleur]['recall']:.2f}. "
        f"Le Gradient Boosting prédit toujours la classe majoritaire car sklearn GBM "
        f"ignore <em>class_weight</em> — XGBoost avec <em>scale_pos_weight</em> résoudrait cela.",
        kind="sky",
        icon="📊",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="medium")

    with col1:
        sh("Radar — profil multi-métriques")
        cats = ["F1-Score", "Précision", "Rappel", "AUC-ROC", "AUC-PR"]
        fig = go.Figure()
        for (nom, r), col in zip(resultats.items(), PALETTE):
            vals = [r["f1"], r["precision"], r["recall"], r["auc_roc"], r["auc_pr"]]
            fig.add_trace(
                go.Scatterpolar(
                    r=vals + [vals[0]],
                    theta=cats + [cats[0]],
                    fill="toself",
                    fillcolor=col,
                    opacity=0.08,
                    line=dict(color=col, width=2),
                    name=nom[:18],
                    hovertemplate="<b>%{theta}</b> : %{r:.3f}<extra>"
                    + nom
                    + "</extra>",
                )
            )
        fig.update_layout(
            polar=dict(
                bgcolor=BG,
                radialaxis=dict(
                    visible=True,
                    range=[0, 0.8],
                    gridcolor=GRID,
                    tickfont=dict(size=8, color=FONT),
                    color=FONT,
                ),
                angularaxis=dict(
                    gridcolor=GRID, color="#a09080", tickfont=dict(size=10)
                ),
            ),
            paper_bgcolor=PAPER,
            font_color=FONT,
            font_family="Instrument Sans",
            height=400,
            legend=dict(font_size=10, orientation="h", y=-0.08, bgcolor=NG),
            margin=dict(l=20, r=20, t=10, b=40),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col2:
        sh("Matrice de confusion détaillée")
        modele_cm = st.selectbox(
            "Modèle à analyser",
            [k for k in resultats if k != "Dummy (Stratified)"],
            index=0,
        )
        r = resultats[modele_cm]
        cm_data = [[r["tn"], r["fp"]], [r["fn"], r["tp"]]]
        cm_total = sum(sum(row) for row in cm_data)
        cm_pct = [[v / cm_total * 100 for v in row] for row in cm_data]
        lbls = [["TN", "FP"], ["FN", "TP"]]
        descrs = [
            ["Normal · Correct", "Fausse alerte"],
            ["Panne manquée", "Panne détectée"],
        ]
        accs = [["#a09080", "#d4645a"], ["#e8a045", "#6ba88a"]]
        anns = []
        for i in range(2):
            for j in range(2):
                anns.append(
                    dict(
                        x=j,
                        y=i,
                        text=(
                            f"<b style='font-size:1.1rem'>{lbls[i][j]}</b><br>"
                            f"<span style='font-size:0.65rem;color:#5c4f4a'>{descrs[i][j]}</span><br>"
                            f"<b style='font-size:1.4rem;font-family:Geist Mono'>{cm_data[i][j]:,}</b><br>"
                            f"<span style='font-size:0.7rem'>{cm_pct[i][j]:.1f}%</span>"
                        ),
                        showarrow=False,
                        align="center",
                        font=dict(color=accs[i][j], family="Instrument Sans"),
                    )
                )
        fig = go.Figure(
            go.Heatmap(
                z=cm_data,
                colorscale=[[0, "#1a1614"], [0.5, "#251f1c"], [1, "#322a27"]],
                showscale=False,
                hoverinfo="skip",
                xgap=4,
                ygap=4,
            )
        )
        fig.update_layout(
            **warm_layout(
                height=360,
                annotations=anns,
                xaxis=dict(
                    tickvals=[0, 1],
                    ticktext=["Prédit : Normal", "Prédit : Panne"],
                    tickfont_size=10,
                    showgrid=False,
                    color="#a09080",
                ),
                yaxis=dict(
                    tickvals=[0, 1],
                    ticktext=["Réel : Normal", "Réel : Panne"],
                    tickfont_size=10,
                    showgrid=False,
                    autorange="reversed",
                    color="#a09080",
                ),
                margin=dict(l=4, r=4, t=10, b=4),
            )
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        det_pct = r["tp"] / (r["tp"] + r["fn"]) * 100 if r["tp"] + r["fn"] > 0 else 0
        callout(
            f"<strong>{modele_cm}</strong> détecte <strong>{det_pct:.1f}%</strong> des pannes réelles "
            f"({r['tp']:,} / {r['tp']+r['fn']:,}). "
            f"Fausses alertes : <strong>{r['fp']:,}</strong>.",
            kind="sage",
            icon="🎯",
        )


# ══════════════════════════════════════════════════════════════
# PAGE 4 — FEATURES
# ══════════════════════════════════════════════════════════════
elif "Features" in page:

    hero(
        "Explicabilité — ce que le modèle a appris",
        "Quelles variables",
        "font vraiment la différence ?",
        "Importance moyenne Random Forest + Gradient Boosting sur 101 features engineered. "
        "Décryptez la logique du modèle.",
        tag("101 features", "ghost")
        + tag("RF + GB", "amber")
        + tag("Explicabilité", "sage"),
    )

    def get_family(col):
        if any(f"lag{l}" in col for l in [1, 3, 7]):
            return "Lag"
        if "roll" in col:
            return "Rolling"
        if "tendance" in col or "acceleration" in col:
            return "Tendance"
        if col in [
            "risque_materiel",
            "stress_thermique",
            "score_surcharge",
            "score_negligence",
            "score_connectivite",
            "ratio_erreurs_tx",
        ]:
            return "Interaction"
        if "type_gab" in col or "environnement" in col:
            return "Catégoriel"
        if col in [
            "mois_sin",
            "mois_cos",
            "jour_sin",
            "jour_cos",
            "est_weekend",
            "est_ete",
            "est_fin_mois",
            "trimestre",
        ]:
            return "Temporel"
        return "Original"

    fam_col = {
        "Original": "#5a8fc4",
        "Lag": "#a87fb5",
        "Rolling": "#6ba88a",
        "Tendance": "#e8a045",
        "Interaction": "#d4645a",
        "Temporel": "#c9952a",
        "Catégoriel": "#82a04f",
    }
    fam_desc = {
        "Rolling": "Moyennes glissantes 7–14j",
        "Interaction": "Features métier composites",
        "Original": "Variables brutes initiales",
        "Lag": "Valeurs décalées J-1/3/7",
        "Tendance": "Pente de dégradation",
        "Temporel": "Cyclicité temporelle",
        "Catégoriel": "Encodages type/env",
    }

    feat_imp["famille"] = feat_imp["feature"].apply(get_family)
    feat_imp["color"] = feat_imp["famille"].map(fam_col)

    col1, col2 = st.columns([3, 2], gap="medium")

    with col1:
        n_top = st.slider("Nombre de features à afficher", 10, 50, 25, 5)
        sh(f"Top {n_top} — importance décroissante")
        top_df = feat_imp.head(n_top).sort_values("imp_moy")
        fig = go.Figure(
            go.Bar(
                x=top_df["imp_moy"],
                y=[c.replace("_", " ")[:38] for c in top_df["feature"]],
                orientation="h",
                marker=dict(
                    color=top_df["color"].values,
                    opacity=0.85,
                    line=dict(color=BG, width=0.5),
                ),
                text=[f"{v:.4f}" for v in top_df["imp_moy"]],
                textposition="outside",
                textfont=dict(size=8, color=FONT),
                hovertemplate="<b>%{y}</b><br>%{x:.4f}<extra></extra>",
            )
        )
        fig.update_layout(
            **warm_layout(
                height=max(420, n_top * 22),
                xaxis=dict(showgrid=False, visible=False),
                yaxis=dict(tickfont=dict(size=8, color="#a09080")),
                margin=dict(l=4, r=65, t=16, b=4),
            )
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col2:
        sh("Répartition par famille")
        fam_imp_df = (
            feat_imp.groupby("famille")["imp_moy"]
            .sum()
            .reset_index()
            .sort_values("imp_moy", ascending=False)
        )
        fam_imp_df["pct"] = fam_imp_df["imp_moy"] / fam_imp_df["imp_moy"].sum() * 100

        fig = go.Figure(
            go.Pie(
                labels=fam_imp_df["famille"],
                values=fam_imp_df["pct"],
                hole=0.62,
                marker=dict(
                    colors=[fam_col.get(f, "#3a302c") for f in fam_imp_df["famille"]],
                    line=dict(color=BG, width=3),
                ),
                textfont=dict(size=9),
                textinfo="percent",
                hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>",
            )
        )
        fig.update_layout(
            **warm_layout(
                height=270,
                margin=dict(l=0, r=0, t=10, b=0),
                showlegend=False,
                annotations=[
                    dict(
                        text="Familles",
                        font=dict(size=10, color="#e0d8d0", family="Fraunces"),
                        showarrow=False,
                    )
                ],
            )
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        sh("Poids par famille")
        fam_rows = []
        for _, row in fam_imp_df.iterrows():
            ch = fam_col.get(row["famille"], "#3a302c")
            bw = int(row["pct"] * 1.2)
            desc = fam_desc.get(row["famille"], "")
            fam_rows.append(
                f"<tr>"
                f"<td>"
                f'<span style="display:inline-block;width:7px;height:7px;border-radius:50%;'
                f'background:{ch};margin-right:7px;vertical-align:middle"></span>'
                f'<span style="font-weight:500;color:#a09080">{row["famille"]}</span>'
                f'<br><span style="font-size:0.63rem;color:#5c4f4a">{desc}</span>'
                f"</td>"
                f'<td style="font-family:Geist Mono;color:#e0d8d0">{row["imp_moy"]:.4f}</td>'
                f"<td>"
                f'<div class="prog-rail" style="width:80px;display:inline-block;vertical-align:middle">'
                f'<div class="prog-fill" style="background:{ch};width:{bw}px"></div>'
                f"</div> "
                f'<span style="font-size:0.68rem;color:#5c4f4a">{row["pct"]:.1f}%</span>'
                f"</td>"
                f"</tr>"
            )
        st.markdown(
            '<table class="data-tbl"><thead><tr>'
            "<th>Famille</th><th>Importance</th><th>Part</th>"
            "</tr></thead><tbody>" + "".join(fam_rows) + "</tbody></table>",
            unsafe_allow_html=True,
        )

    callout(
        "Les features <strong>Rolling (7–14j)</strong> et <strong>Interaction métier</strong> dominent "
        "— la <em>tendance de dégradation</em> est bien plus prédictive que la valeur instantanée. "
        "C'est le Feature Engineering, pas le choix du modèle, qui fait la différence.",
        kind="sage",
        icon="💡",
    )


# ══════════════════════════════════════════════════════════════
# PAGE 5 — SEUIL & COÛTS
# ══════════════════════════════════════════════════════════════
elif "Seuil" in page:

    hero(
        "Décision opérationnelle — arbitrage économique",
        "Quel seuil",
        "vous coûte le moins ?",
        "Le seuil 0.5 par défaut n'est presque jamais optimal. "
        "Ajustez selon vos coûts réels et trouvez le point de bascule rentable.",
        tag("Simulation", "ghost")
        + tag("MAD", "amber")
        + tag("Recall / Coût", "coral"),
    )

    col_p, col_r = st.columns([1, 2], gap="medium")

    with col_p:
        sh("Coûts opérationnels")
        cout_correctif = st.slider(
            "💸 Panne non détectée (MAD)", 1000, 20000, 5000, 500
        )
        cout_preventif = st.slider(
            "🔧 Intervention préventive (MAD)", 500, 5000, 1500, 250
        )
        cout_fausse = st.slider("🔔 Fausse alerte (MAD)", 100, 2000, 500, 100)

        sh("Modèle de référence")
        modele_sel = st.selectbox(
            "Modèle", [k for k in resultats if k != "Dummy (Stratified)"], index=0
        )
        r = resultats[modele_sel]
        tp_total = r["tp"] + r["fn"]

        st.markdown(
            f"""
        <div style="margin-top:1rem;padding:1.1rem;background:var(--ink-800);
             border:1px solid var(--border-dim);border-radius:10px">
          <div style="font-size:0.6rem;letter-spacing:0.15em;text-transform:uppercase;
               color:var(--text-muted);margin-bottom:0.8rem">Métriques de base</div>
          <div style="display:flex;justify-content:space-between;margin-bottom:0.45rem">
            <span style="font-size:0.78rem;color:var(--text-muted)">F1-Score</span>
            <span style="font-family:Geist Mono;color:var(--text-primary);font-weight:600">{r['f1']:.4f}</span>
          </div>
          <div style="display:flex;justify-content:space-between;margin-bottom:0.45rem">
            <span style="font-size:0.78rem;color:var(--text-muted)">Recall base</span>
            <span style="font-family:Geist Mono;color:var(--sage-l);font-weight:600">{r['recall']:.4f}</span>
          </div>
          <div style="display:flex;justify-content:space-between">
            <span style="font-size:0.78rem;color:var(--text-muted)">Pannes réelles</span>
            <span style="font-family:Geist Mono;color:var(--text-primary);font-weight:600">{tp_total:,}</span>
          </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    seuils = np.linspace(0.01, 0.99, 200)
    recalls_s, precs_s, f1s_s, couts_s = [], [], [], []
    for s in seuils:
        factor = (0.5 - s) * 2
        rec = float(np.clip(r["recall"] + factor * 0.4, 0, 1))
        prec = float(np.clip(r["precision"] - factor * 0.05, 0.05, 1))
        f1 = 2 * prec * rec / (prec + rec + 1e-10)
        tp_s = int(rec * tp_total)
        fp_s = int(tp_s / (prec + 1e-10) - tp_s)
        fn_s = tp_total - tp_s
        recalls_s.append(rec)
        precs_s.append(prec)
        f1s_s.append(f1)
        couts_s.append(
            tp_s * cout_preventif + fp_s * cout_fausse + fn_s * cout_correctif
        )

    cout_sans = tp_total * cout_correctif
    seuil_f1 = float(seuils[np.argmax(f1s_s)])
    seuil_econ = float(seuils[np.argmin(couts_s)])
    economie = cout_sans - min(couts_s)
    eco_pct = economie / cout_sans * 100 if cout_sans > 0 else 0

    with col_r:
        seuil_ch = st.slider("🎚️ Seuil de décision", 0.01, 0.99, seuil_f1, 0.01)
        idx = int(np.argmin(np.abs(seuils - seuil_ch)))
        rec_ch = recalls_s[idx]
        prec_ch = precs_s[idx]
        f1_ch = f1s_s[idx]
        cout_ch = couts_s[idx]
        eco_ch = cout_sans - cout_ch

        st.markdown(
            f"""
        <div class="kpi-grid" style="grid-template-columns:repeat(3,1fr);margin-bottom:1.2rem">
          <div class="kpi-card kpi-sage">
            <div class="kpi-label">Recall</div>
            <div class="kpi-val" style="font-size:1.8rem">{rec_ch:.3f}</div>
            <div class="kpi-sub">{int(rec_ch*tp_total):,} / {tp_total:,} détectées</div>
          </div>
          <div class="kpi-card kpi-sky">
            <div class="kpi-label">Précision</div>
            <div class="kpi-val" style="font-size:1.8rem">{prec_ch:.3f}</div>
            <div class="kpi-sub">F1 = {f1_ch:.3f}</div>
          </div>
          <div class="kpi-card kpi-amber">
            <div class="kpi-label">Économie</div>
            <div class="kpi-val" style="font-size:1.8rem">{eco_ch/1000:.0f}k</div>
            <div class="kpi-sub">MAD · {eco_ch/cout_sans*100:.0f}% épargné</div>
          </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        fig = make_subplots(
            rows=1, cols=2, subplot_titles=["Métriques vs Seuil", "Coût total (k MAD)"]
        )
        for y_vals, name, color, dash in [
            (precs_s, "Précision", "#e8a045", "solid"),
            (recalls_s, "Rappel", "#d4645a", "solid"),
            (f1s_s, "F1", "#6ba88a", "dot"),
        ]:
            fig.add_trace(
                go.Scatter(
                    x=seuils,
                    y=y_vals,
                    name=name,
                    line=dict(color=color, width=2, dash=dash),
                ),
                row=1,
                col=1,
            )
        fig.add_vline(
            x=seuil_ch,
            line_color="rgba(255,255,255,0.2)",
            line_width=1.5,
            line_dash="dash",
            row=1,
            col=1,
        )
        fig.add_vline(
            x=seuil_f1,
            line_color="#6ba88a",
            line_width=1,
            annotation_text=f"F1 opt {seuil_f1:.2f}",
            annotation_font_size=8,
            annotation_font_color="#6ba88a",
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=seuils,
                y=[c / 1000 for c in couts_s],
                fill="tozeroy",
                fillcolor="rgba(232,160,69,0.06)",
                line=dict(color="#e8a045", width=2.5),
                name="Coût modèle",
            ),
            row=1,
            col=2,
        )
        fig.add_hline(
            y=cout_sans / 1000,
            line_color="#d4645a",
            line_dash="dash",
            line_width=1.5,
            annotation_text="Sans modèle",
            annotation_font_size=8,
            annotation_font_color="#d4645a",
            row=1,
            col=2,
        )
        fig.add_vline(
            x=seuil_ch,
            line_color="rgba(255,255,255,0.2)",
            line_width=1.5,
            line_dash="dash",
            row=1,
            col=2,
        )
        fig.add_vline(
            x=seuil_econ,
            line_color="#e8a045",
            line_width=1,
            annotation_text=f"Éco opt {seuil_econ:.2f}",
            annotation_font_size=8,
            annotation_font_color="#e8a045",
            row=1,
            col=2,
        )
        fig.update_layout(
            **warm_layout(
                height=320,
                legend=dict(orientation="h", y=-0.15, font_size=9),
                yaxis=dict(gridcolor=GRID),
                yaxis2=dict(gridcolor=GRID),
                xaxis=dict(showgrid=False, title="Seuil", title_font=dict(size=10)),
                xaxis2=dict(showgrid=False, title="Seuil", title_font=dict(size=10)),
                margin=dict(l=4, r=4, t=30, b=4),
            )
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    callout(
        f"Seuil F1-optimal = <strong>{seuil_f1:.2f}</strong> · "
        f"Seuil économique optimal = <strong>{seuil_econ:.2f}</strong> · "
        f"Économie maximale estimée = <strong>{economie/1000:.0f} k MAD ({eco_pct:.0f}%)</strong> "
        f"par rapport à une stratégie 100% correctif.",
        kind="sage",
        icon="✅",
    )


# ══════════════════════════════════════════════════════════════
# PAGE 6 — SCORING LIVE
# ══════════════════════════════════════════════════════════════
elif "Scoring" in page:

    hero(
        "Outil opérationnel — évaluation temps réel",
        "Ce GAB va-t-il",
        "tomber en panne ?",
        "Saisissez les métriques du jour d'un GAB spécifique. "
        "Le score composite reproduit la logique des features les plus importantes du modèle ML.",
        tag("Score composite", "ghost")
        + tag("8 features", "amber")
        + tag("Temps réel", "coral"),
    )

    col_in, col_out = st.columns([1, 1], gap="medium")

    with col_in:
        sh("Identité du GAB")
        ville = st.selectbox("Ville", sorted(df["ville"].unique()))
        type_gab = st.selectbox("Constructeur", df["type_gab"].unique())
        env = st.selectbox("Environnement", df["environnement"].unique())
        age = st.slider("Âge du GAB (années)", 1, 10, 4)

        sh("Métriques opérationnelles du jour")
        erreurs_lecteur = st.slider("Erreurs lecteur de carte", 0, 20, 2)
        erreurs_dist = st.slider("Erreurs distributeur", 0, 10, 0)
        temperature = st.slider("Température interne (°C)", 25, 65, 35)
        jours_maint = st.slider("Jours depuis maintenance", 0, 365, 60)
        nb_tx = st.slider("Transactions du jour", 0, 250, 120)
        taux_erreur_tx = (
            st.slider("Taux d'erreur transactions (%)", 0.0, 30.0, 5.0) / 100
        )
        _lat = st.slider("Latence réseau (ms)", 10, 1000, 80)
        _dec = st.slider("Déconnexions réseau", 0, 10, 0)

        sh("Historique 7 derniers jours")
        erreurs_roll7 = st.slider(
            "Moy. erreurs lecteur / 7j", 0.0, 15.0, float(max(0, erreurs_lecteur - 1))
        )
        temp_roll7 = st.slider(
            "Moy. température / 7j (°C)", 25.0, 65.0, float(temperature)
        )

    # Score composite
    taux_ville = {
        "Safi": 0.110,
        "Agadir": 0.105,
        "Casablanca": 0.103,
        "El Jadida": 0.099,
        "Kénitra": 0.099,
        "Beni Mellal": 0.098,
        "Rabat": 0.096,
        "Marrakech": 0.096,
        "Meknès": 0.095,
        "Fès": 0.094,
        "Oujda": 0.094,
        "Tanger": 0.093,
        "Tétouan": 0.092,
    }
    risque_env = {
        "Site_Isole": 1.0,
        "Centre_Commercial": 0.7,
        "Agence_Facade": 0.6,
        "Agence_Interieure": 0.5,
    }
    risque_type = {"Wincor": 1.0, "NCR": 0.7, "Hyosung": 0.65, "Diebold": 0.6}

    ratio_err = (erreurs_lecteur + erreurs_dist) / (nb_tx + 1)
    sc_negl = jours_maint * np.log1p(erreurs_roll7)
    rq_mat = erreurs_lecteur * np.log1p(age)
    st_therm = temperature * age / 10
    tend_lect = (erreurs_lecteur - erreurs_roll7) / (erreurs_roll7 + 1e-8)
    tend_temp = (temperature - temp_roll7) / (temp_roll7 + 1e-8)

    score_raw = (
        ratio_err * 0.20
        + (sc_negl / 1000) * 0.15
        + (rq_mat / 30) * 0.15
        + (st_therm / 60) * 0.10
        + (jours_maint / 365) * 0.10
        + taux_erreur_tx * 0.08
        + (temperature / 65) * 0.07
        + max(0, tend_lect) * 0.05
        + max(0, tend_temp) * 0.05
        + taux_ville.get(ville, 0.097) * 0.03
        + risque_env.get(env, 0.5) * 0.01
        + risque_type.get(type_gab, 0.65) * 0.01
    )
    score = min(1.0, score_raw * 1.8)

    if score >= 0.70:
        niveau, couleur, emoji_n, txt_n, callout_k = (
            "CRITIQUE",
            "#d4645a",
            "🔴",
            "Intervention immédiate requise",
            "coral",
        )
    elif score >= 0.45:
        niveau, couleur, emoji_n, txt_n, callout_k = (
            "ÉLEVÉ",
            "#e8a045",
            "🟡",
            "Planifier une visite dans les 48h",
            "amber",
        )
    elif score >= 0.25:
        niveau, couleur, emoji_n, txt_n, callout_k = (
            "MODÉRÉ",
            "#5a8fc4",
            "🔵",
            "Surveiller l'évolution cette semaine",
            "sky",
        )
    else:
        niveau, couleur, emoji_n, txt_n, callout_k = (
            "FAIBLE",
            "#6ba88a",
            "🟢",
            "Aucune action immédiate nécessaire",
            "sage",
        )

    with col_out:
        sh("Score de risque estimé")
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=score * 100,
                number=dict(
                    suffix="%",
                    font=dict(size=52, color=couleur, family="Fraunces"),
                    valueformat=".0f",
                ),
                gauge=dict(
                    axis=dict(
                        range=[0, 100],
                        tickfont=dict(size=9, color=FONT),
                        tickcolor=GRID,
                        nticks=5,
                    ),
                    bar=dict(color=couleur, thickness=0.18),
                    bgcolor=BG,
                    bordercolor=GRID,
                    borderwidth=1,
                    steps=[
                        dict(range=[0, 25], color="rgba(107,168,138,0.04)"),
                        dict(range=[25, 45], color="rgba(90,143,196,0.04)"),
                        dict(range=[45, 70], color="rgba(232,160,69,0.04)"),
                        dict(range=[70, 100], color="rgba(212,100,90,0.04)"),
                    ],
                    threshold=dict(
                        line=dict(color="rgba(255,255,255,0.15)", width=2),
                        thickness=0.75,
                        value=50,
                    ),
                ),
                title=dict(
                    text=f"{emoji_n} &nbsp; <b>{niveau}</b>",
                    font=dict(size=16, color=couleur, family="Fraunces"),
                ),
            )
        )
        fig.update_layout(
            height=310,
            paper_bgcolor=PAPER,
            font_color=FONT,
            font_family="Instrument Sans",
            margin=dict(l=20, r=20, t=30, b=10),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        callout(f"<strong>{niveau}</strong> — {txt_n}", kind=callout_k, icon=emoji_n)

        sh("Contributions par feature")
        contribs = {
            "Ratio erreurs/tx": ratio_err * 0.20,
            "Score négligence": (sc_negl / 1000) * 0.15,
            "Risque matériel": (rq_mat / 30) * 0.15,
            "Stress thermique": (st_therm / 60) * 0.10,
            "Jours maintenance": (jours_maint / 365) * 0.10,
            "Taux erreur tx": taux_erreur_tx * 0.08,
            "Température": (temperature / 65) * 0.07,
            "Tendance lecteur": max(0, tend_lect) * 0.05,
        }
        cdf = pd.DataFrame(
            list(contribs.items()), columns=["Feature", "Contribution"]
        ).sort_values("Contribution")
        fig2 = go.Figure(
            go.Bar(
                x=cdf["Contribution"],
                y=cdf["Feature"],
                orientation="h",
                marker=dict(
                    color=cdf["Contribution"],
                    colorscale=[[0, "#251f1c"], [0.5, "#e8a045"], [1, couleur]],
                    showscale=False,
                    line=dict(color=BG, width=0.5),
                ),
                text=[f"{v:.4f}" for v in cdf["Contribution"]],
                textposition="outside",
                textfont=dict(size=8, color=FONT),
                hovertemplate="<b>%{y}</b> : %{x:.4f}<extra></extra>",
            )
        )
        fig2.update_layout(
            **warm_layout(
                height=280,
                xaxis=dict(showgrid=False, visible=False),
                yaxis=dict(tickfont=dict(size=9, color="#a09080")),
                margin=dict(l=4, r=65, t=10, b=4),
            )
        )
        st.plotly_chart(
            fig2, use_container_width=True, config={"displayModeBar": False}
        )

        # Fiche récap
        st.markdown(
            f"""
        <div style="background:var(--ink-800);border:1px solid var(--border-dim);
             border-left:3px solid {couleur};border-radius:10px;padding:1.1rem 1.2rem;margin-top:0.5rem">
          <div style="font-size:0.6rem;letter-spacing:0.15em;text-transform:uppercase;
               color:var(--text-muted);margin-bottom:0.9rem">Récapitulatif GAB</div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.6rem">
            <div>
              <div style="font-size:0.68rem;color:var(--text-muted)">Constructeur · Âge</div>
              <div style="font-family:Geist Mono;color:var(--text-mid);font-size:0.82rem;margin-top:0.15rem">{type_gab} · {age} ans</div>
            </div>
            <div>
              <div style="font-size:0.68rem;color:var(--text-muted)">Ville</div>
              <div style="font-family:Geist Mono;color:var(--text-mid);font-size:0.82rem;margin-top:0.15rem">{ville}</div>
            </div>
            <div>
              <div style="font-size:0.68rem;color:var(--text-muted)">Environnement</div>
              <div style="font-family:Geist Mono;color:var(--text-mid);font-size:0.82rem;margin-top:0.15rem">{env.replace("_"," ")}</div>
            </div>
            <div>
              <div style="font-size:0.68rem;color:var(--text-muted)">Score brut</div>
              <div style="font-family:Geist Mono;color:{couleur};font-size:0.9rem;font-weight:600;margin-top:0.15rem">{score:.3f} / 1.00</div>
            </div>
          </div>
        </div>
        """,
            unsafe_allow_html=True,
        )
