import { useState, useContext } from "react";
import { useApi } from "../hooks/useApi";
import { ThemeContext } from "../App";
import Chart, { chartColors } from "../components/Chart";
import Hero from "../components/Hero";
import SectionHead from "../components/SectionHead";
import Callout from "../components/Callout";
import { PageSkeleton, ErrorState } from "../components/Skeleton";
import { IconBulb } from "../components/Icons";
import ExportMenu from "../components/ExportMenu";

export default function Features() {
  const [top, setTop] = useState(25);
  const theme = useContext(ThemeContext);
  const cc = chartColors(theme === "dark");
  const { data, loading, error } = useApi(`/api/features?top=${top}`);

  if (loading) return <PageSkeleton />;
  if (error)
    return (
      <ErrorState
        message={`Erreur : ${error}`}
        onRetry={() => window.location.reload()}
      />
    );
  if (!data) return null;

  const { top_features, families } = data;
  const avgImp =
    top_features.imp_moy.length > 0
      ? top_features.imp_moy.reduce((a, b) => a + b, 0) /
        top_features.imp_moy.length
      : null;
  const maxFamPct = Math.max(...families.pct);

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
        <Hero
          eyebrow="Explicabilité — ce que le modèle a appris"
          titleMain="Quelles variables"
          titleEm="font vraiment la différence ?"
          desc="Importance moyenne Random Forest + Gradient Boosting sur 101 features engineered. Décryptez la logique du modèle."
          tags={[
            { label: "101 features", kind: "ghost" },
            { label: "RF + GB", kind: "amber" },
            { label: "Explicabilité", kind: "sage" },
          ]}
        />
        <div style={{ paddingTop: "2rem", flexShrink: 0 }}>
          <ExportMenu section="features" formats={["csv", "json", "excel"]} />
        </div>
      </div>

      <div className="col-grid col-6-4">
        <div>
          <div className="slider-wrap">
            <div className="slider-label">
              <span>Nombre de features à afficher</span>
              <span className="slider-val">{top}</span>
            </div>
            <input
              type="range"
              min={10}
              max={101}
              step={5}
              value={top}
              onChange={(e) => setTop(Number(e.target.value))}
            />
          </div>
          <SectionHead title={`Top ${top} — importance décroissante`} />
          <Chart
            data={[
              {
                type: "bar",
                orientation: "h",
                x: top_features.imp_moy,
                y: top_features.feature.map((f) =>
                  f.replace(/_/g, " ").slice(0, 38),
                ),
                marker: {
                  color: top_features.color,
                  opacity: 0.85,
                  line: { color: cc.outline, width: 0.5 },
                },
                text: top_features.imp_moy.map((v) => v.toFixed(4)),
                textposition: "outside",
                textfont: { size: 8, color: cc.fontMuted },
                hovertemplate: "<b>%{y}</b><br>%{x:.4f}<extra></extra>",
              },
            ]}
            layout={{
              height: Math.max(420, top * 22),
              xaxis: { showgrid: false, visible: false },
              yaxis: { tickfont: { size: 8, color: cc.fontMuted } },
              margin: { l: 200, r: 65, t: 16, b: 4 },
              shapes:
                avgImp != null
                  ? [
                      {
                        type: "line",
                        x0: avgImp,
                        x1: avgImp,
                        y0: -0.5,
                        y1: top_features.feature.length - 0.5,
                        xref: "x",
                        yref: "y",
                        line: { color: cc.vLine, width: 1.5, dash: "dot" },
                      },
                    ]
                  : [],
              annotations:
                avgImp != null
                  ? [
                      {
                        x: avgImp,
                        y: top_features.feature.length - 0.5,
                        xref: "x",
                        yref: "y",
                        text: `Moy. ${avgImp.toFixed(4)}`,
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

        <div>
          <SectionHead title="Répartition par famille" />
          <Chart
            data={[
              {
                type: "pie",
                hole: 0.62,
                labels: families.famille,
                values: families.pct,
                marker: {
                  colors: families.color,
                  line: { color: cc.outline, width: 2 },
                },
                textfont: { size: 9 },
                textinfo: "percent",
                hovertemplate: "<b>%{label}</b><br>%{percent}<extra></extra>",
              },
            ]}
            layout={{
              height: 260,
              margin: { l: 0, r: 0, t: 10, b: 0 },
              showlegend: false,
              annotations: [
                {
                  text: "Familles",
                  font: { size: 10, color: cc.annotText, family: "Fraunces" },
                  showarrow: false,
                },
              ],
            }}
          />

          <SectionHead title="Poids par famille" />
          <table className="data-tbl">
            <thead>
              <tr>
                <th>Famille</th>
                <th>Importance</th>
                <th>Part</th>
              </tr>
            </thead>
            <tbody>
              {families.famille.map((fam, i) => (
                <tr key={fam}>
                  <td>
                    <span
                      style={{
                        display: "inline-block",
                        width: 7,
                        height: 7,
                        borderRadius: "50%",
                        background: families.color[i],
                        marginRight: 7,
                        verticalAlign: "middle",
                      }}
                    />
                    <span style={{ fontWeight: 500, color: "var(--text-mid)" }}>
                      {fam}
                    </span>
                    <br />
                    <span style={{ fontSize: "0.63rem", color: "var(--text-muted)" }}>
                      {families.desc[i]}
                    </span>
                  </td>
                  <td className="mono" style={{ color: "var(--text-primary)" }}>
                    {families.imp_moy[i].toFixed(4)}
                  </td>
                  <td>
                    <div
                      className="prog-rail"
                      style={{
                        width: 80,
                        display: "inline-block",
                        verticalAlign: "middle",
                      }}
                    >
                      <div
                        className="prog-fill"
                        style={{
                          background: families.color[i],
                          width: `${(families.pct[i] / maxFamPct) * 80}px`,
                        }}
                      />
                    </div>{" "}
                    <span style={{ fontSize: "0.68rem", color: "var(--text-muted)" }}>
                      {families.pct[i].toFixed(1)}%
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <Callout
        icon={<IconBulb width={16} height={16} />}
        text="Les features <strong>Rolling (7–14j)</strong> et <strong>Interaction métier</strong> dominent — la <em>tendance de dégradation</em> est bien plus prédictive que la valeur instantanée. C'est le Feature Engineering, pas le choix du modèle, qui fait la différence."
        kind="sage"
      />
    </div>
  );
}
