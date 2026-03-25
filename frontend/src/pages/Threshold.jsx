import { useState, useMemo, useContext } from "react";
import { useApi } from "../hooks/useApi";
import { ThemeContext } from "../App";
import Chart, { chartColors } from "../components/Chart";
import Hero from "../components/Hero";
import SectionHead from "../components/SectionHead";
import Callout from "../components/Callout";
import KpiCard from "../components/KpiCard";
import { PageSkeleton, ErrorState } from "../components/Skeleton";
import { IconCheck, IconTarget, IconSliders } from "../components/Icons";

export default function Threshold() {
  const { data: meta } = useApi("/api/metadata");
  const [model, setModel] = useState("");
  const [coutCorrectif, setCoutCorrectif] = useState(5000);
  const [coutPreventif, setCoutPreventif] = useState(1500);
  const [coutFausse, setCoutFausse] = useState(500);
  const [seuil, setSeuil] = useState(null);

  const url = useMemo(
    () =>
      `/api/threshold?model=${encodeURIComponent(model)}&cout_correctif=${coutCorrectif}&cout_preventif=${coutPreventif}&cout_fausse=${coutFausse}`,
    [model, coutCorrectif, coutPreventif, coutFausse],
  );

  const { data, loading, error } = useApi(url);

  if (loading) return <PageSkeleton />;
  if (error)
    return (
      <ErrorState
        message={`Erreur : ${error}`}
        onRetry={() => window.location.reload()}
      />
    );
  if (!data) return null;

  const activeSeuil = seuil ?? data.seuil_f1_optimal;
  const idx = data.seuils.reduce(
    (bi, s, i) =>
      Math.abs(s - activeSeuil) < Math.abs(data.seuils[bi] - activeSeuil)
        ? i
        : bi,
    0,
  );

  const recCh = data.recalls[idx];
  const precCh = data.precisions[idx];
  const f1Ch = data.f1s[idx];
  const coutCh = data.couts[idx];
  const ecoCh = data.cout_sans_modele - coutCh;
  const theme = useContext(ThemeContext);
  const cc = chartColors(theme === "dark");
  const GRID = cc.grid;

  return (
    <div>
      <Hero
        eyebrow="Décision opérationnelle — arbitrage économique"
        titleMain="Quel seuil"
        titleEm="vous coûte le moins ?"
        desc="Le seuil 0.5 par défaut n'est presque jamais optimal. Ajustez selon vos coûts réels et trouvez le point de bascule rentable."
        tags={[
          { label: "Simulation", kind: "ghost" },
          { label: "MAD", kind: "amber" },
          { label: "Recall / Coût", kind: "coral" },
        ]}
      />

      <div className="col-grid col-1-2">
        <div>
          <SectionHead title="Coûts opérationnels" />

          {[
            [
              "Panne non détectée (MAD)",
              coutCorrectif,
              1000,
              20000,
              500,
              setCoutCorrectif,
            ],
            [
              "Intervention préventive (MAD)",
              coutPreventif,
              500,
              5000,
              250,
              setCoutPreventif,
            ],
            ["Fausse alerte (MAD)", coutFausse, 100, 2000, 100, setCoutFausse],
          ].map(([label, val, min, max, step, setter]) => (
            <div key={label} className="slider-wrap">
              <div className="slider-label">
                <span>{label}</span>
                <span className="slider-val">{val.toLocaleString()} MAD</span>
              </div>
              <input
                type="range"
                min={min}
                max={max}
                step={step}
                value={val}
                onChange={(e) => {
                  setter(Number(e.target.value));
                  setSeuil(null);
                }}
              />
            </div>
          ))}

          <SectionHead title="Modèle de référence" />
          <select
            className="sb-select"
            value={model}
            onChange={(e) => {
              setModel(e.target.value);
              setSeuil(null);
            }}
            style={{ marginBottom: "1rem" }}
          >
            <option value="">Champion (auto)</option>
            {(meta?.modeles || []).map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>

          <div
            style={{
              padding: "1.1rem",
              background: "var(--ink-800)",
              border: "1px solid var(--border-dim)",
              borderRadius: 10,
            }}
          >
            <div
              style={{
                fontSize: "0.6rem",
                letterSpacing: "0.15em",
                textTransform: "uppercase",
                color: "var(--text-muted)",
                marginBottom: "0.8rem",
              }}
            >
              Métriques de base
            </div>
            {[
              ["F1-Score", data.model_metrics.f1?.toFixed(4)],
              ["Recall", data.model_metrics.recall?.toFixed(4)],
              ["Pannes réelles", data.tp_total.toLocaleString()],
            ].map(([k, v]) => (
              <div
                key={k}
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  marginBottom: "0.45rem",
                }}
              >
                <span
                  style={{ fontSize: "0.78rem", color: "var(--text-muted)" }}
                >
                  {k}
                </span>
                <span
                  className="mono"
                  style={{ color: "var(--text-primary)", fontWeight: 600 }}
                >
                  {v}
                </span>
              </div>
            ))}
          </div>
        </div>

        <div>
          <div className="slider-wrap">
            <div className="slider-label">
              <span>Seuil de décision</span>
              <span className="slider-val">{activeSeuil.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min={0.01}
              max={0.99}
              step={0.01}
              value={activeSeuil}
              onChange={(e) => setSeuil(Number(e.target.value))}
            />
          </div>

          <div
            className="kpi-grid kpi-grid-3"
            style={{ marginBottom: "1.2rem" }}
          >
            <KpiCard
              icon={<IconTarget width={16} height={16} />}
              label="Recall"
              value={recCh.toFixed(3)}
              sub={`${Math.round(recCh * data.tp_total).toLocaleString()} / ${data.tp_total.toLocaleString()}`}
              color="sage"
            />
            <KpiCard
              icon={<IconSliders width={16} height={16} />}
              label="Précision"
              value={precCh.toFixed(3)}
              sub={`F1 = ${f1Ch.toFixed(3)}`}
              color="sky"
            />
            <KpiCard
              icon={<IconCheck width={16} height={16} />}
              label="Économie"
              value={`${Math.round(ecoCh / 1000)}k`}
              sub={`MAD · ${Math.round((ecoCh / data.cout_sans_modele) * 100)}% épargné`}
              color="amber"
            />
          </div>

          <Chart
            data={[
              {
                type: "scatter",
                x: data.seuils,
                y: data.precisions,
                name: "Précision",
                line: { color: "#C8102E", width: 2 },
              },
              {
                type: "scatter",
                x: data.seuils,
                y: data.recalls,
                name: "Rappel",
                line: { color: "#E8374B", width: 2 },
              },
              {
                type: "scatter",
                x: data.seuils,
                y: data.f1s,
                name: "F1",
                line: { color: "#00A65A", width: 2, dash: "dot" },
              },
            ]}
            layout={{
              height: 280,
              shapes: [
                {
                  type: "line",
                  x0: activeSeuil,
                  x1: activeSeuil,
                  y0: 0,
                  y1: 1,
                  yref: "paper",
                  line: { color: cc.vLine, width: 1.5, dash: "dash" },
                },
                {
                  type: "line",
                  x0: data.seuil_f1_optimal,
                  x1: data.seuil_f1_optimal,
                  y0: 0,
                  y1: 1,
                  yref: "paper",
                  line: { color: "#00A65A", width: 1 },
                },
              ],
              annotations: [
                {
                  x: data.seuil_f1_optimal,
                  y: 1,
                  yref: "paper",
                  text: `F1 opt ${data.seuil_f1_optimal}`,
                  font: { size: 8, color: "#00A65A" },
                  showarrow: false,
                  xanchor: "left",
                },
              ],
              xaxis: {
                showgrid: false,
                title: { text: "Seuil", font: { size: 10 } },
              },
              yaxis: { gridcolor: GRID, zeroline: false },
              legend: { orientation: "h", y: -0.2, font: { size: 9 } },
              title: {
                text: "Métriques vs Seuil",
                font: { size: 11, color: cc.fontMuted },
              },
            }}
          />

          <Chart
            data={[
              {
                type: "scatter",
                x: data.seuils,
                y: data.couts.map((c) => c / 1000),
                fill: "tozeroy",
                fillcolor: "rgba(200,16,46,0.07)",
                line: { color: "#C8102E", width: 2.5 },
                name: "Coût modèle",
              },
            ]}
            layout={{
              height: 280,
              shapes: [
                {
                  type: "line",
                  x0: 0,
                  x1: 1,
                  y0: data.cout_sans_modele / 1000,
                  y1: data.cout_sans_modele / 1000,
                  line: { color: "#E05252", width: 1.5, dash: "dash" },
                },
                {
                  type: "line",
                  x0: activeSeuil,
                  x1: activeSeuil,
                  y0: 0,
                  y1: 1,
                  yref: "paper",
                  line: { color: cc.vLine, width: 1.5, dash: "dash" },
                },
                {
                  type: "line",
                  x0: data.seuil_econ_optimal,
                  x1: data.seuil_econ_optimal,
                  y0: 0,
                  y1: 1,
                  yref: "paper",
                  line: { color: "#C8102E", width: 1 },
                },
              ],
              annotations: [
                {
                  x: data.seuil_econ_optimal,
                  y: 1,
                  yref: "paper",
                  text: `Éco opt ${data.seuil_econ_optimal}`,
                  font: { size: 8, color: "#C8102E" },
                  showarrow: false,
                  xanchor: "left",
                },
                {
                  x: 0.02,
                  y: data.cout_sans_modele / 1000,
                  text: "Sans modèle",
                  font: { size: 8, color: "#E05252" },
                  showarrow: false,
                  xanchor: "left",
                  yanchor: "bottom",
                },
              ],
              xaxis: {
                showgrid: false,
                title: { text: "Seuil", font: { size: 10 } },
              },
              yaxis: {
                gridcolor: GRID,
                zeroline: false,
                title: { text: "k MAD", font: { size: 10 } },
              },
              title: {
                text: "Coût total (k MAD)",
                font: { size: 11, color: cc.fontMuted },
              },
            }}
          />
        </div>
      </div>

      <Callout
        icon={<IconCheck width={16} height={16} />}
        text={`Seuil F1-optimal = <strong>${data.seuil_f1_optimal}</strong> · Seuil économique optimal = <strong>${data.seuil_econ_optimal}</strong> · Économie maximale estimée = <strong>${Math.round(data.economie_max / 1000)} k MAD (${data.economie_pct}%)</strong> par rapport à une stratégie 100% correctif.`}
        kind="sage"
      />
    </div>
  );
}
