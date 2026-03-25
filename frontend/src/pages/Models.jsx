import { useState, useContext } from "react";
import { useApi } from "../hooks/useApi";
import { ThemeContext } from "../App";
import Chart, { chartColors } from "../components/Chart";
import Hero from "../components/Hero";
import SectionHead from "../components/SectionHead";
import Callout from "../components/Callout";
import { PageSkeleton, ErrorState } from "../components/Skeleton";
import { IconTarget, IconBarChart } from "../components/Icons";

const PALETTE = [
  "#C8102E", "#00703C", "#3B82C4", "#B8922A", "#7C3AED",
  "#E05252", "#00A65A", "#5a8fc4", "#e8a045",
];

export default function Models() {
  const { data, loading, error } = useApi("/api/models", {
    transform: (d) => ({
      ...d,
      _selectedModel: Object.keys(d.resultats).find(
        (k) => k !== "Dummy (Stratified)",
      ),
    }),
  });
  const [selectedModel, setSelectedModel] = useState(null);
  const theme = useContext(ThemeContext);
  const cc = chartColors(theme === "dark");

  if (loading) return <PageSkeleton />;
  if (error)
    return (
      <ErrorState
        message={`Erreur : ${error}`}
        onRetry={() => window.location.reload()}
      />
    );
  if (!data) return null;

  const { resultats, meilleur } = data;
  const modeles = Object.keys(resultats).filter(
    (k) => k !== "Dummy (Stratified)",
  );
  const active = selectedModel || data._selectedModel;

  const cats = ["F1-Score", "Précision", "Rappel", "AUC-ROC", "AUC-PR"];
  const radarData = modeles.map((nom, i) => {
    const r = resultats[nom];
    const vals = [r.f1, r.precision, r.recall, r.auc_roc, r.auc_pr];
    return {
      type: "scatterpolar",
      r: [...vals, vals[0]],
      theta: [...cats, cats[0]],
      fill: "toself",
      fillcolor: PALETTE[i] + "33",
      line: { color: PALETTE[i], width: 2.5 },
      name: nom.slice(0, 18),
      hovertemplate: "<b>%{theta}</b> : %{r:.3f}<extra>" + nom + "</extra>",
    };
  });

  const cmData = (() => {
    const r = resultats[active];
    if (!r) return null;
    const total = r.tn + r.fp + r.fn + r.tp;
    return { r, total };
  })();

  return (
    <div>
      <Hero
        eyebrow="Évaluation & comparaison des algorithmes ML"
        titleMain="Quel modèle"
        titleEm="mérite votre confiance ?"
        desc="Modèles comparés sur split temporel strict (train 2022 / test 2023) avec calibration isotonique. L'enjeu : maximiser le recall sans exploser les fausses alertes."
        tags={[
          { label: "Split temporel", kind: "ghost" },
          { label: "Classe 9.8%", kind: "coral" },
          { label: "class_weight=balanced", kind: "sage" },
        ]}
      />

      <SectionHead title="Tableau comparatif" />
      <table className="data-tbl" style={{ marginBottom: "1.5rem" }}>
        <thead>
          <tr>
            <th>Modèle</th>
            <th>F1</th>
            <th>Précision</th>
            <th>Rappel</th>
            <th>AUC-ROC</th>
            <th>AUC-PR</th>
            <th>TP</th>
            <th>FN</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(resultats).map(([nom, r]) => (
            <tr key={nom} className={nom === meilleur ? "highlight" : ""} style={nom === "Dummy (Stratified)" ? {opacity:0.55} : {}}>
              <td style={{ fontWeight: 600, color: "var(--text-primary)" }}>
                {nom}
              </td>
              <td className="mono">{r.f1.toFixed(4)}</td>
              <td className="mono">{r.precision.toFixed(4)}</td>
              <td className="mono">{r.recall.toFixed(4)}</td>
              <td className="mono">{r.auc_roc.toFixed(4)}</td>
              <td className="mono">{r.auc_pr.toFixed(4)}</td>
              <td className="mono" style={{ color: "#33C47A" }}>
                {r.tp.toLocaleString()}
              </td>
              <td className="mono" style={{ color: "#F07070" }}>
                {r.fn.toLocaleString()}
              </td>
              <td>
                {nom === meilleur && (
                  <span className="badge badge-sage">Champion</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <Callout
        icon={<IconBarChart width={16} height={16} />}
        text={`<strong>${meilleur}</strong> sélectionné par AUC-PR (${resultats[meilleur].auc_pr.toFixed(3)}) — Recall <strong>${(resultats[meilleur].recall * 100).toFixed(1)}%</strong>, F1 <strong>${resultats[meilleur].f1.toFixed(3)}</strong> au seuil optimal (${resultats[meilleur].threshold_f1?.toFixed(3) || "0.5"}).`}
        kind="sky"
      />

      <div className="col-grid col-5-5" style={{ marginTop: "1.5rem" }}>
        <div>
          <SectionHead title="Radar — profil multi-métriques" />
          <Chart
            data={radarData}
            layout={{
              polar: {
                bgcolor: cc.bg,
                radialaxis: {
                  visible: true,
                  range: [0, 0.7],
                  gridcolor: cc.radarGrid,
                  tickfont: { size: 8, color: cc.fontMuted },
                  color: cc.fontMuted,
                },
                angularaxis: {
                  gridcolor: cc.radarGrid,
                  color: cc.fontMid,
                  tickfont: { size: 10 },
                },
              },
              paper_bgcolor: cc.bg,
              font: { color: cc.fontMuted, family: "Instrument Sans" },
              height: 400,
              legend: {
                font: { size: 10 },
                orientation: "h",
                y: -0.08,
                bgcolor: "rgba(0,0,0,0)",
              },
              margin: { l: 20, r: 20, t: 10, b: 40 },
            }}
          />
        </div>

        <div>
          <SectionHead title="Matrice de confusion détaillée" />
          <select
            className="sb-select"
            style={{ marginBottom: "1rem" }}
            value={active || ""}
            onChange={(e) => setSelectedModel(e.target.value)}
          >
            {modeles.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>

          {cmData && (
            <>
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "1fr 1fr",
                  gap: 6,
                  marginBottom: "1rem",
                }}
              >
                {[
                  {
                    lbl: "TN",
                    desc: "Normal · Correct",
                    val: cmData.r.tn,
                    pct: ((cmData.r.tn / cmData.total) * 100).toFixed(1),
                    accent: "#8891a8",
                    bg: "rgba(136,145,168,0.07)",
                  },
                  {
                    lbl: "FP",
                    desc: "Fausse alerte",
                    val: cmData.r.fp,
                    pct: ((cmData.r.fp / cmData.total) * 100).toFixed(1),
                    accent: "#C8102E",
                    bg: "rgba(200,16,46,0.07)",
                  },
                  {
                    lbl: "FN",
                    desc: "Panne manquée",
                    val: cmData.r.fn,
                    pct: ((cmData.r.fn / cmData.total) * 100).toFixed(1),
                    accent: "#B8922A",
                    bg: "rgba(184,146,42,0.08)",
                  },
                  {
                    lbl: "TP",
                    desc: "Panne détectée",
                    val: cmData.r.tp,
                    pct: ((cmData.r.tp / cmData.total) * 100).toFixed(1),
                    accent: "#00A65A",
                    bg: "rgba(0,166,90,0.08)",
                  },
                ].map(({ lbl, desc, val, pct, accent, bg }) => (
                  <div
                    key={lbl}
                    style={{
                      background: bg,
                      border: `1px solid ${accent}33`,
                      borderRadius: 10,
                      padding: "1rem",
                      display: "flex",
                      flexDirection: "column",
                      gap: "0.3rem",
                    }}
                  >
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                      }}
                    >
                      <span
                        style={{
                          fontSize: "0.62rem",
                          fontWeight: 700,
                          letterSpacing: "0.12em",
                          textTransform: "uppercase",
                          color: accent,
                        }}
                      >
                        {lbl}
                      </span>
                      <span
                        style={{
                          fontSize: "0.62rem",
                          color: "var(--text-ghost)",
                        }}
                      >
                        {pct}%
                      </span>
                    </div>
                    <div
                      style={{
                        fontFamily: "Geist Mono, monospace",
                        fontSize: "1.6rem",
                        fontWeight: 700,
                        color: accent,
                        letterSpacing: "-0.02em",
                        lineHeight: 1,
                      }}
                    >
                      {val.toLocaleString()}
                    </div>
                    <div
                      style={{
                        fontSize: "0.68rem",
                        color: "var(--text-muted)",
                      }}
                    >
                      {desc}
                    </div>
                    <div
                      style={{
                        marginTop: "0.3rem",
                        height: 3,
                        borderRadius: 2,
                        background: `${accent}22`,
                      }}
                    >
                      <div
                        style={{
                          height: "100%",
                          borderRadius: 2,
                          background: accent,
                          width: `${Math.min(pct * 2, 100)}%`,
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
              <Callout
                icon={<IconTarget width={16} height={16} />}
                text={`<strong>${active}</strong> détecte <strong>${((cmData.r.tp / (cmData.r.tp + cmData.r.fn)) * 100).toFixed(1)}%</strong> des pannes réelles (${cmData.r.tp.toLocaleString()} / ${(cmData.r.tp + cmData.r.fn).toLocaleString()}). Fausses alertes : <strong>${cmData.r.fp.toLocaleString()}</strong>.`}
                kind="sage"
              />
            </>
          )}
        </div>
      </div>
    </div>
  );
}
