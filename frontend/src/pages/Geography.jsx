import { useContext, useMemo } from "react";
import { FilterContext, ThemeContext } from "../App";
import { useApi, buildFilterQuery } from "../hooks/useApi";
import Chart, { chartColors } from "../components/Chart";
import Hero from "../components/Hero";
import SectionHead from "../components/SectionHead";
import Callout from "../components/Callout";
import { PageSkeleton, ErrorState } from "../components/Skeleton";
import { IconThermo } from "../components/Icons";

function levelBadge(taux) {
  if (taux > 10.5)
    return { label: "Élevé", cls: "badge-coral", color: "#F07070" };
  if (taux > 9.5)
    return { label: "Modéré", cls: "badge-amber", color: "#E8374B" };
  return { label: "Faible", cls: "badge-sage", color: "#33C47A" };
}

export default function Geography() {
  const { filters } = useContext(FilterContext);
  const theme = useContext(ThemeContext);
  const cc = chartColors(theme === "dark");
  const query = useMemo(() => buildFilterQuery(filters), [filters]);
  const { data, loading, error } = useApi(`/api/geography?${query}`, {
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

  const { cities, heatmap } = data;
  const avgCity =
    cities.taux_panne.length > 0
      ? cities.taux_panne.reduce((a, b) => a + b, 0) / cities.taux_panne.length
      : null;

  return (
    <div>
      <Hero
        eyebrow="Cartographie du risque opérationnel"
        titleMain="Où se concentrent"
        titleEm="les incidents ?"
        desc="Analyse géographique des pannes par ville et région. Identifiez les zones à surveiller en priorité et les patterns saisonniers."
        tags={[
          { label: "13 villes", kind: "ghost" },
          { label: "Maroc", kind: "amber" },
          { label: "Heatmap", kind: "sky" },
        ]}
      />

      <div className="col-grid col-6-4">
        <div>
          <SectionHead title="Taux de panne par ville — tri décroissant" />
          <Chart
            data={[
              {
                type: "bar",
                x: cities.ville,
                y: cities.taux_panne,
                marker: {
                  color: cities.taux_panne,
                  colorscale: [
                    [0, cc.barFill],
                    [0.4, "#9E0C24"],
                    [1, "#C8102E"],
                  ],
                  showscale: true,
                  colorbar: {
                    title: { text: "%", font: { size: 9 } },
                    thickness: 8,
                    tickfont: { size: 8 },
                    bgcolor: "rgba(0,0,0,0)",
                    outlinewidth: 0,
                  },
                  line: { color: cc.outline, width: 1 },
                },
                text: cities.taux_panne.map((v) => `${v.toFixed(1)}%`),
                textposition: "outside",
                textfont: { size: 9, color: cc.fontMuted },
                hovertemplate: "<b>%{x}</b><br>Taux : %{y:.2f}%<extra></extra>",
              },
            ]}
            layout={{
              height: 380,
              xaxis: { showgrid: false, tickangle: 35, tickfont: { size: 9 } },
              yaxis: {
                gridcolor: cc.grid,
                title: { text: "Taux (%)", font: { size: 10 } },
              },
              shapes:
                avgCity != null
                  ? [
                      {
                        type: "line",
                        x0: -0.5,
                        x1: cities.ville.length - 0.5,
                        y0: avgCity,
                        y1: avgCity,
                        xref: "x",
                        yref: "y",
                        line: { color: cc.vLine, width: 1.5, dash: "dot" },
                      },
                    ]
                  : [],
              annotations:
                avgCity != null
                  ? [
                      {
                        x: cities.ville.length - 0.5,
                        y: avgCity,
                        xref: "x",
                        yref: "y",
                        text: `Moy. ${avgCity.toFixed(1)}%`,
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
          <SectionHead title="Classement des villes" />
          <table className="data-tbl">
            <thead>
              <tr>
                <th>#</th>
                <th>Ville</th>
                <th>Taux</th>
                <th>Niveau</th>
              </tr>
            </thead>
            <tbody>
              {cities.ville.map((v, i) => {
                const taux = cities.taux_panne[i];
                const { label, cls, color } = levelBadge(taux);
                return (
                  <tr key={v}>
                    <td
                      style={{
                        color: "var(--text-ghost)",
                        fontFamily: "Geist Mono",
                      }}
                    >
                      {i + 1}
                    </td>
                    <td style={{ fontWeight: 500, color: "var(--text-mid)" }}>{v}</td>
                    <td
                      style={{
                        fontFamily: "Geist Mono",
                        color,
                        fontWeight: 600,
                      }}
                    >
                      {taux.toFixed(2)}%
                    </td>
                    <td>
                      <span className={`badge ${cls}`}>{label}</span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      <div style={{ marginTop: "1.5rem" }}>
        <SectionHead title="Heatmap saisonnière — Ville × Mois" />
      </div>
      <Chart
        data={[
          {
            type: "heatmap",
            z: heatmap.z,
            x: heatmap.mois,
            y: heatmap.villes,
            colorscale: [
              [0, cc.scaleMin],
              [0.35, cc.scaleMid],
              [0.7, "#9E0C24"],
              [1, "#C8102E"],
            ],
            hoverongaps: false,
            hovertemplate:
              "<b>%{y}</b> · %{x}<br>Taux : %{z:.2f}%<extra></extra>",
            colorbar: {
              title: { text: "%", font: { size: 9 } },
              thickness: 8,
              tickfont: { size: 8 },
              bgcolor: "rgba(0,0,0,0)",
              outlinewidth: 0,
            },
            xgap: 1,
            ygap: 1,
          },
        ]}
        layout={{
          height: 460,
          xaxis: { side: "top", tickfont: { size: 10 }, showgrid: false },
          yaxis: { tickfont: { size: 10 }, showgrid: false },
          margin: { l: 100, r: 20, t: 30, b: 16 },
        }}
      />

      <Callout
        icon={<IconThermo width={16} height={16} />}
        text="Le pic de <strong>juin–août</strong> est visible dans presque toutes les villes, confirmant le stress thermique comme facteur saisonnier dominant. <strong>Safi et Agadir</strong> montrent la saisonnalité la plus marquée."
        kind="amber"
      />
    </div>
  );
}
