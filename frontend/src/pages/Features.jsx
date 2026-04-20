import { useState, useContext, useMemo } from "react";
import { useApi } from "../hooks/useApi";
import { ThemeContext } from "../App";
import Chart, { chartColors } from "../components/Chart";
import Hero from "../components/Hero";
import SectionHead from "../components/SectionHead";
import Callout from "../components/Callout";
import Slider from "../components/Slider";
import { PageSkeleton, ErrorState } from "../components/Skeleton";
import { IconBulb } from "../components/Icons";
import ExportMenu from "../components/ExportMenu";

export default function Features() {
  const [top, setTop] = useState(25);
  const theme = useContext(ThemeContext);
  const cc = chartColors(theme === "dark");
  const { data, loading, error } = useApi(`/api/features?top=${top}`);

  // Sort families by pct descending — must be before early returns (hooks rule)
  const famSorted = useMemo(() => {
    if (!data) return null;
    const { families } = data;
    const idx = families.pct
      .map((v, i) => ({ v, i }))
      .sort((a, b) => b.v - a.v)
      .map((x) => x.i);
    return {
      famille: idx.map((i) => families.famille[i]),
      imp_moy: idx.map((i) => families.imp_moy[i]),
      pct: idx.map((i) => families.pct[i]),
      color: idx.map((i) => families.color[i]),
      desc: idx.map((i) => families.desc[i]),
    };
  }, [data]);

  if (loading) return <PageSkeleton />;
  if (error)
    return (
      <ErrorState
        message={`Erreur : ${error}`}
        onRetry={() => window.location.reload()}
      />
    );
  if (!data || !famSorted) return null;

  const { top_features } = data;

  const avgImp =
    top_features.imp_moy.length > 0
      ? top_features.imp_moy.reduce((a, b) => a + b, 0) /
        top_features.imp_moy.length
      : null;

  const maxFamPct = famSorted.pct[0]; // already sorted desc

  return (
    <div>
      {/* ── Header ── */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-start",
        }}
      >
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

      {/* ── Main grid ── */}
      <div className="col-grid col-6-4">
        {/* Left — bar chart */}
        <div>
          <Slider
            label="Nombre de features à afficher"
            value={top}
            min={10}
            max={101}
            step={5}
            onChange={setTop}
          />
          <SectionHead title={`Top ${top} — importance décroissante`} />
          <Chart
            data={[
              {
                type: "bar",
                orientation: "h",
                x: top_features.imp_moy,
                y: top_features.feature.map((f) =>
                  f.replace(/_/g, " ").slice(0, 38)
                ),
                marker: {
                  color: top_features.color,
                  opacity: 0.88,
                  line: { color: cc.outline, width: 0.5 },
                },
                text: top_features.imp_moy.map((v) => v.toFixed(4)),
                textposition: "outside",
                textfont: { size: 8, color: cc.fontMuted },
                hovertemplate: "<b>%{y}</b><br>Importance : %{x:.4f}<extra></extra>",
              },
            ]}
            layout={{
              height: Math.max(440, top * 22),
              xaxis: { showgrid: false, visible: false },
              yaxis: { tickfont: { size: 8, color: cc.fontMuted } },
              margin: { l: 200, r: 70, t: 16, b: 4 },
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

        {/* Right — donut + table */}
        <div>
          <SectionHead title={`Répartition par famille (${famSorted.famille.length})`} />
          <Chart
            data={[
              {
                type: "pie",
                hole: 0.62,
                labels: famSorted.famille,
                values: famSorted.pct,
                marker: {
                  colors: famSorted.color,
                  line: { color: cc.outline, width: 2 },
                },
                textfont: { size: 9 },
                textinfo: "percent",
                hovertemplate:
                  "<b>%{label}</b><br>%{percent} (%{value:.2f})<extra></extra>",
                sort: false,
              },
            ]}
            layout={{
              height: 270,
              margin: { l: 0, r: 0, t: 10, b: 0 },
              showlegend: false,
              annotations: [
                {
                  text: `<b>${famSorted.famille.length}</b><br><span style="font-size:9px">familles</span>`,
                  font: { size: 13, color: cc.annotText, family: "Instrument Sans" },
                  showarrow: false,
                },
              ],
            }}
          />

          <SectionHead title="Poids par famille" />
          <table className="data-tbl">
            <thead>
              <tr>
                <th>#</th>
                <th>Famille</th>
                <th>Importance</th>
                <th>Part</th>
              </tr>
            </thead>
            <tbody>
              {famSorted.famille.map((fam, i) => (
                <tr key={fam} className={i === 0 ? "highlight" : ""}>
                  <td
                    className="mono"
                    style={{
                      color: "var(--text-ghost)",
                      fontSize: "0.65rem",
                      width: "1.5rem",
                    }}
                  >
                    {i + 1}
                  </td>
                  <td>
                    <span
                      style={{
                        display: "inline-block",
                        width: 7,
                        height: 7,
                        borderRadius: "50%",
                        background: famSorted.color[i],
                        marginRight: 7,
                        verticalAlign: "middle",
                        flexShrink: 0,
                      }}
                    />
                    <span style={{ fontWeight: 500, color: "var(--text-mid)" }}>
                      {fam}
                    </span>
                    <br />
                    <span
                      style={{
                        fontSize: "0.62rem",
                        color: "var(--text-muted)",
                      }}
                    >
                      {famSorted.desc[i]}
                    </span>
                  </td>
                  <td className="mono" style={{ color: "var(--text-primary)" }}>
                    {famSorted.imp_moy[i].toFixed(4)}
                  </td>
                  <td style={{ whiteSpace: "nowrap" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                      <div className="prog-rail" style={{ width: 64, flexShrink: 0 }}>
                        <div
                          className="prog-fill"
                          style={{
                            background: famSorted.color[i],
                            width: `${(famSorted.pct[i] / maxFamPct) * 100}%`,
                          }}
                        />
                      </div>
                      <span
                        className="mono"
                        style={{ fontSize: "0.68rem", color: "var(--text-muted)" }}
                      >
                        {famSorted.pct[i].toFixed(1)}%
                      </span>
                    </div>
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
