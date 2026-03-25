import { useState, useContext, useMemo } from "react";
import { FilterContext, ThemeContext, MonitoringContext } from "../App";
import { useApi, buildFilterQuery } from "../hooks/useApi";
import Chart, { chartColors } from "../components/Chart";
import Hero from "../components/Hero";
import SectionHead from "../components/SectionHead";
import Callout from "../components/Callout";
import KpiCard from "../components/KpiCard";
import { PageSkeleton, ErrorState } from "../components/Skeleton";
import MonitoringModal, {
  GabDetailModal,
} from "../components/MonitoringModal";
import {
  IconAtm,
  IconZap,
  IconTarget,
  IconDatabase,
  IconAlert,
  IconShield,
  StatusDot,
} from "../components/Icons";
import ExportMenu, { ExportReportButton } from "../components/ExportMenu";

export default function Overview() {
  const { filters } = useContext(FilterContext);
  const { monitoring, refreshMonitoring } = useContext(MonitoringContext) || {};
  const [showAllGab, setShowAllGab] = useState(false);
  const [selectedGab, setSelectedGab] = useState(null);
  const theme = useContext(ThemeContext);
  const cc = chartColors(theme === "dark");
  const query = useMemo(() => buildFilterQuery(filters), [filters]);
  const { data, loading, error } = useApi(`/api/overview?${query}`, {
    enabled: filters.villes.length > 0,
  });

  if (loading) return <PageSkeleton />;
  if (error)
    return (
      <ErrorState
        message={`Erreur : ${error}`}
        onRetry={() => window.location.reload()}
      />
    );
  if (!data) return null;

  const { kpis, monthly, by_type, by_env, by_season, by_age, top_city } = data;
  const GRID = cc.grid;

  const peakMonthIdx =
    monthly.pannes.length > 0
      ? monthly.pannes.indexOf(Math.max(...monthly.pannes))
      : -1;
  const avgByType =
    by_type.taux.length > 0
      ? by_type.taux.reduce((a, b) => a + b, 0) / by_type.taux.length
      : null;
  const avgBySeason =
    by_season.taux.length > 0
      ? by_season.taux.reduce((a, b) => a + b, 0) / by_season.taux.length
      : null;
  const peakAgeIdx =
    by_age.taux.length > 0 ? by_age.taux.indexOf(Math.max(...by_age.taux)) : -1;

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
        <Hero
          eyebrow="Maintenance Prédictive · Tableau de bord principal"
          titleMain="Bienvenue dans votre"
          titleEm="centre de contrôle."
          desc="200 GAB surveillés en continu sur 13 villes du Maroc. Des modèles ML entraînés sur 2 ans de données détectent les pannes avant qu'elles surviennent."
          tags={[
            { label: "200 GAB", kind: "ghost" },
            { label: "13 villes", kind: "ghost" },
            { label: "Actif", kind: "sage" },
            { label: "2022–2023", kind: "amber" },
          ]}
        />
        <div style={{ display: "flex", gap: "0.5rem", paddingTop: "2rem", flexShrink: 0 }}>
          <ExportMenu section="overview" />
          <ExportReportButton />
        </div>
      </div>

      <div
        className="kpi-grid"
        style={{ gridTemplateColumns: "repeat(5,1fr)" }}
      >
        <KpiCard
          icon={<IconAtm width={17} height={17} />}
          label="GAB surveillés"
          value={kpis.nb_gab}
          sub={`${kpis.nb_villes} villes · filtres actifs`}
          color="amber"
        />
        <KpiCard
          icon={<IconZap width={16} height={16} />}
          label="Pannes enregistrées"
          value={kpis.nb_pannes.toLocaleString()}
          sub={`Taux moyen : <strong style="color:#E8374B">${kpis.taux_panne.toFixed(1)}%</strong> · classe minoritaire`}
          color="coral"
        />
        <KpiCard
          icon={<IconTarget width={17} height={17} />}
          label="Meilleur F1-Score"
          value={kpis.f1_best.toFixed(3)}
          sub={`${kpis.meilleur_modele.slice(0, 22)} · Recall <strong style="color:#33C47A">${kpis.recall_best.toFixed(2)}</strong><br/><span style="color:var(--text-muted);font-size:0.65rem">Cohérent avec 9.8% classe minoritaire · split temporel strict. F1 > 0.4 indiquerait du data leakage.</span>`}
          color="sage"
        />
        <KpiCard
          icon={<IconDatabase width={17} height={17} />}
          label="Observations"
          value={kpis.nb_obs.toLocaleString()}
          sub={`2 ans · <strong style="color:#60A5E8">101</strong> features engineered`}
          color="sky"
        />
        <KpiCard
          icon={<IconShield width={17} height={17} />}
          label="Disponibilité"
          value={
            kpis.taux_panne != null
              ? `${(100 - kpis.taux_panne).toFixed(1)}%`
              : "—"
          }
          sub={`Taux de panne : <strong style="color:#E8374B">${kpis.taux_panne?.toFixed(1) ?? "—"}%</strong> · base de calcul`}
          color="sage"
        />
      </div>

      <div className="col-grid col-6-4">
        <div>
          <SectionHead title="Tendance mensuelle des incidents" />
          <Chart
            data={[
              {
                type: "bar",
                x: monthly.mois,
                y: monthly.pannes,
                name: "Incidents",
                yaxis: "y1",
                marker: {
                  color: cc.barFill,
                  line: { color: "#C8102E", width: 1 },
                },
                hovertemplate: "<b>%{x}</b><br>%{y:,} pannes<extra></extra>",
              },
              {
                type: "scatter",
                x: monthly.mois,
                y: monthly.taux,
                name: "Taux %",
                yaxis: "y2",
                line: { color: "#C8102E", width: 2, shape: "spline" },
                mode: "lines+markers",
                marker: { size: 4, color: "#C8102E" },
                hovertemplate: "%{y:.2f}%<extra>Taux</extra>",
              },
            ]}
            layout={{
              height: 300,
              xaxis: { showgrid: false, tickangle: 40, tickfont: { size: 9 } },
              yaxis: { gridcolor: GRID, zeroline: false },
              yaxis2: {
                overlaying: "y",
                side: "right",
                tickfont: { size: 9 },
                zeroline: false,
                showgrid: false,
              },
              legend: {
                orientation: "h",
                y: 1.08,
                font: { size: 10, color: cc.fontMuted },
                bgcolor: "rgba(0,0,0,0)",
              },
            }}
          />
        </div>
        <div>
          <SectionHead title="Taux de panne par constructeur" />
          <Chart
            data={[
              {
                type: "bar",
                orientation: "h",
                x: by_type.taux,
                y: by_type.type_gab,
                marker: {
                  color: by_type.taux,
                  colorscale: [
                    [0, cc.barFill],
                    [0.5, "#9E0C24"],
                    [1, "#C8102E"],
                  ],
                  showscale: false,
                  line: { color: "#0a0a0c", width: 1 },
                },
                text: by_type.taux.map((v) => `${v.toFixed(1)}%`),
                textposition: "outside",
                textfont: { size: 10, color: cc.fontMuted },
                hovertemplate: "<b>%{y}</b> : %{x:.2f}%<extra></extra>",
              },
            ]}
            layout={{
              height: 300,
              bargap: 0.5,
              xaxis: { showgrid: false, visible: false },
              yaxis: {
                showgrid: false,
                tickfont: { size: 11, color: cc.fontMuted },
              },
              margin: { l: 70, r: 50, t: 16, b: 4 },
              shapes:
                avgByType != null
                  ? [
                      {
                        type: "line",
                        x0: avgByType,
                        x1: avgByType,
                        y0: -0.5,
                        y1: by_type.type_gab.length - 0.5,
                        xref: "x",
                        yref: "y",
                        line: { color: cc.vLine, width: 1.5, dash: "dot" },
                      },
                    ]
                  : [],
              annotations:
                avgByType != null
                  ? [
                      {
                        x: avgByType,
                        y: by_type.type_gab.length - 0.5,
                        xref: "x",
                        yref: "y",
                        text: `Moy. ${avgByType.toFixed(1)}%`,
                        showarrow: false,
                        xanchor: "left",
                        yanchor: "bottom",
                        font: { size: 9, color: cc.fontMuted },
                      },
                    ]
                  : [],
            }}
          />
        </div>
      </div>

      <div className="col-grid col-3" style={{ marginTop: "1.5rem" }}>
        <div>
          <SectionHead title="Par environnement" />
          <Chart
            data={[
              {
                type: "pie",
                hole: 0.65,
                labels: by_env.environnement.map((e) => e.replace(/_/g, " ")),
                values: by_env.taux,
                marker: {
                  colors: ["#C8102E", "#00703C", "#3B82C4", "#B8922A"],
                  line: { color: cc.outline, width: 2 },
                },
                textfont: { size: 9 },
                textinfo: "percent",
                hovertemplate: "<b>%{label}</b><br>%{percent}<extra></extra>",
              },
            ]}
            layout={{
              height: 280,
              margin: { l: 0, r: 0, t: 10, b: 0 },
              legend: { font: { size: 9 }, orientation: "v", x: 1.02 },
              annotations: [
                {
                  text: "Env",
                  font: { size: 11, color: cc.annotText },
                  showarrow: false,
                },
              ],
            }}
          />
        </div>
        <div>
          <SectionHead title="Par saison" />
          <Chart
            data={[
              {
                type: "bar",
                x: by_season.saisons,
                y: by_season.taux,
                marker: {
                  color: by_season.saisons.map(
                    (s) =>
                      ({
                        Printemps: "#00A65A",
                        Été: "#C8102E",
                        Automne: "#B8922A",
                        Hiver: "#3B82C4",
                      })[s] || "#C8102E",
                  ),
                  line: { color: cc.outline, width: 1 },
                },
                text: by_season.taux.map((v) => `${v.toFixed(1)}%`),
                textposition: "outside",
                textfont: { size: 9, color: cc.fontMuted },
                hovertemplate: "<b>%{x}</b> : %{y:.2f}%<extra></extra>",
              },
            ]}
            layout={{
              height: 280,
              bargap: 0.45,
              xaxis: { showgrid: false, tickfont: { size: 10 } },
              yaxis: { gridcolor: GRID, zeroline: false },
              shapes:
                avgBySeason != null
                  ? [
                      {
                        type: "line",
                        x0: -0.5,
                        x1: by_season.saisons.length - 0.5,
                        y0: avgBySeason,
                        y1: avgBySeason,
                        xref: "x",
                        yref: "y",
                        line: { color: cc.vLine, width: 1.5, dash: "dot" },
                      },
                    ]
                  : [],
              annotations:
                avgBySeason != null
                  ? [
                      {
                        x: by_season.saisons.length - 0.5,
                        y: avgBySeason,
                        xref: "x",
                        yref: "y",
                        text: `Moy. ${avgBySeason.toFixed(1)}%`,
                        showarrow: false,
                        xanchor: "right",
                        yanchor: "bottom",
                        font: { size: 9, color: cc.fontMuted },
                      },
                    ]
                  : [],
            }}
          />
        </div>
        <div>
          <SectionHead title="Dégradation selon l'âge du GAB" />
          <Chart
            data={[
              {
                type: "scatter",
                x: by_age.age_annees,
                y: by_age.taux,
                fill: "tozeroy",
                fillcolor: "rgba(200,16,46,0.07)",
                line: { color: "#C8102E", width: 2.5, shape: "spline" },
                mode: "lines+markers",
                marker: {
                  size: 8,
                  color: "#C8102E",
                  line: { color: cc.outline, width: 2 },
                },
                hovertemplate: "<b>%{x} ans</b> : %{y:.2f}%<extra></extra>",
              },
            ]}
            layout={{
              height: 280,
              xaxis: {
                showgrid: false,
                title: { text: "Âge (années)", font: { size: 10 } },
                tickfont: { size: 10 },
              },
              yaxis: {
                gridcolor: GRID,
                title: { text: "Taux %", font: { size: 10 } },
              },
              annotations:
                peakAgeIdx >= 0
                  ? [
                      {
                        x: by_age.age_annees[peakAgeIdx],
                        y: by_age.taux[peakAgeIdx],
                        xref: "x",
                        yref: "y",
                        text: `Max : ${by_age.taux[peakAgeIdx].toFixed(1)}%`,
                        showarrow: true,
                        arrowhead: 2,
                        arrowwidth: 1.5,
                        arrowcolor: "#C8102E",
                        ax: 30,
                        ay: -30,
                        font: { size: 10, color: cc.annotText },
                        bgcolor: "rgba(200,16,46,0.10)",
                        bordercolor: "#C8102E",
                        borderwidth: 1,
                        borderpad: 4,
                      },
                    ]
                  : [],
            }}
          />
        </div>
      </div>

      <Callout
        icon={<IconAlert width={16} height={16} />}
        text={`<strong>${top_city}</strong> enregistre le taux de panne le plus élevé. Les GAB <strong>Wincor</strong> et les <strong>sites isolés</strong> concentrent les incidents. Le pic estival confirme le rôle du stress thermique comme facteur déclenchant.`}
        kind="amber"
      />

      {/* ── Monitoring temps réel ──────────────────────────── */}
      {monitoring && (
        <div className="col-grid col-5-5" style={{ marginTop: "1.5rem" }}>
          <div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <SectionHead title="État des GAB" />
              <button
                onClick={() => setShowAllGab(true)}
                style={{
                  background: "none",
                  border: "1px solid var(--border-dim)",
                  borderRadius: 6,
                  color: "var(--text-muted)",
                  fontSize: "0.7rem",
                  padding: "0.25rem 0.6rem",
                  cursor: "pointer",
                }}
              >
                Voir tout
              </button>
            </div>
            <div
              style={{
                display: "flex",
                gap: "0.6rem",
                marginBottom: "1rem",
                flexWrap: "wrap",
              }}
            >
              {[
                { key: "FAIBLE", label: "Normal", color: "#6ba88a" },
                { key: "MODÉRÉ", label: "Modéré", color: "#5a8fc4" },
                { key: "ÉLEVÉ", label: "Élevé", color: "#e8a045" },
                { key: "CRITIQUE", label: "Critique", color: "#d4645a" },
              ].map(({ key, label, color }) => (
                <div
                  key={key}
                  style={{
                    flex: 1,
                    minWidth: 80,
                    background: `${color}11`,
                    border: `1px solid ${color}33`,
                    borderRadius: 8,
                    padding: "0.6rem 0.8rem",
                    textAlign: "center",
                  }}
                >
                  <div
                    style={{
                      fontFamily: "Geist Mono, monospace",
                      fontSize: "1.4rem",
                      fontWeight: 700,
                      color,
                      lineHeight: 1,
                    }}
                  >
                    {monitoring.compteurs[key] ?? 0}
                  </div>
                  <div
                    style={{
                      fontSize: "0.65rem",
                      color: "var(--text-muted)",
                      marginTop: 4,
                    }}
                  >
                    {label}
                  </div>
                </div>
              ))}
            </div>

            <div
              style={{
                fontSize: "0.62rem",
                letterSpacing: "0.12em",
                textTransform: "uppercase",
                color: "var(--text-ghost)",
                marginBottom: "0.5rem",
              }}
            >
              Top 5 — risque le plus élevé
            </div>
            {monitoring.top_critiques.map((g) => (
              <div
                key={g.gab_id}
                onClick={() => setSelectedGab(g)}
                style={{
                  display: "flex",
                  alignItems: "center",
                  cursor: "pointer",
                  gap: "0.5rem",
                  padding: "0.45rem 0.6rem",
                  marginBottom: 4,
                  background: `${g.couleur}0a`,
                  border: `1px solid ${g.couleur}22`,
                  borderRadius: 8,
                }}
              >
                <StatusDot color={g.couleur} size={8} />
                <span
                  style={{
                    fontWeight: 600,
                    fontSize: "0.78rem",
                    color: "var(--text-primary)",
                    flex: 1,
                  }}
                >
                  {g.gab_id}
                </span>
                <span
                  style={{
                    fontSize: "0.68rem",
                    color: "var(--text-muted)",
                  }}
                >
                  {g.ville}
                </span>
                <span
                  style={{
                    fontFamily: "Geist Mono, monospace",
                    fontSize: "0.75rem",
                    fontWeight: 700,
                    color: g.couleur,
                  }}
                >
                  {g.score_pct}%
                </span>
              </div>
            ))}
            <div
              style={{
                marginTop: "0.5rem",
                fontSize: "0.62rem",
                color: "var(--text-ghost)",
                textAlign: "right",
                display: "flex",
                justifyContent: "flex-end",
                alignItems: "center",
                gap: "0.4rem",
              }}
            >
              Rafraîchi à{" "}
              {new Date(monitoring.timestamp).toLocaleTimeString("fr-FR", {
                hour: "2-digit",
                minute: "2-digit",
                second: "2-digit",
              })}
              {refreshMonitoring && (
                <button
                  onClick={refreshMonitoring}
                  title="Rafraîchir maintenant"
                  style={{
                    background: "none",
                    border: "1px solid var(--border-dim)",
                    borderRadius: 4,
                    color: "var(--text-ghost)",
                    cursor: "pointer",
                    padding: "2px 5px",
                    fontSize: "0.6rem",
                    lineHeight: 1,
                    transition: "color 0.15s, border-color 0.15s",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.color = "var(--text-muted)";
                    e.currentTarget.style.borderColor = "var(--text-muted)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.color = "var(--text-ghost)";
                    e.currentTarget.style.borderColor = "var(--border-dim)";
                  }}
                >
                  ↻
                </button>
              )}
            </div>
          </div>

          <div>
            <SectionHead title="Alertes récentes" />
            {monitoring.alertes.length === 0 ? (
              <div
                style={{
                  padding: "2rem",
                  textAlign: "center",
                  color: "var(--text-ghost)",
                  fontSize: "0.82rem",
                }}
              >
                Aucune alerte active
              </div>
            ) : (
              monitoring.alertes.slice(0, 6).map((a, i) => (
                <div
                  key={i}
                  onClick={() => {
                    const g = monitoring.all_gabs?.find((x) => x.gab_id === a.gab_id);
                    if (g) setSelectedGab(g);
                  }}
                  style={{
                    display: "flex",
                    gap: "0.5rem",
                    alignItems: "flex-start",
                    padding: "0.55rem 0.6rem",
                    marginBottom: 4,
                    cursor: "pointer",
                    background: `${a.couleur}0a`,
                    border: `1px solid ${a.couleur}22`,
                    borderRadius: 8,
                  }}
                >
                  <StatusDot color={a.couleur} size={8} />
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                      <span
                        style={{
                          fontWeight: 600,
                          fontSize: "0.78rem",
                          color: "var(--text-primary)",
                        }}
                      >
                        {a.gab_id}
                        <span
                          style={{
                            fontWeight: 400,
                            color: "var(--text-muted)",
                            marginLeft: "0.3rem",
                          }}
                        >
                          · {a.ville} · {a.type_gab}
                        </span>
                      </span>
                      <span
                        style={{
                          fontSize: "0.6rem",
                          color: "var(--text-ghost)",
                          whiteSpace: "nowrap",
                        }}
                      >
                        {new Date(a.timestamp).toLocaleTimeString("fr-FR", {
                          hour: "2-digit",
                          minute: "2-digit",
                        })}
                      </span>
                    </div>
                    <div
                      style={{
                        fontSize: "0.7rem",
                        color: a.couleur,
                        marginTop: 2,
                      }}
                    >
                      {a.message}
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      )}

      {showAllGab && monitoring?.all_gabs && (
        <MonitoringModal
          gabs={monitoring.all_gabs}
          onClose={() => setShowAllGab(false)}
          onSelect={(g) => {
            setShowAllGab(false);
            setSelectedGab(g);
          }}
        />
      )}

      {selectedGab && (
        <GabDetailModal
          gab={selectedGab}
          onClose={() => setSelectedGab(null)}
        />
      )}
    </div>
  );
}
