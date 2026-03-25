import { useState, useContext } from "react";
import { usePost } from "../hooks/useApi";
import { ThemeContext } from "../App";
import Chart, { chartColors } from "../components/Chart";
import Hero from "../components/Hero";
import SectionHead from "../components/SectionHead";
import Callout from "../components/Callout";
import Slider from "../components/Slider";
import { ErrorState } from "../components/Skeleton";
import {
  IconShield,
  IconAlert,
  IconCheck,
  IconInfo,
} from "../components/Icons";

const VILLES = [
  "Agadir",
  "Beni Mellal",
  "Casablanca",
  "El Jadida",
  "Fès",
  "Kénitra",
  "Marrakech",
  "Meknès",
  "Oujda",
  "Rabat",
  "Safi",
  "Tanger",
  "Tétouan",
];
const TYPES = ["Diebold", "Hyosung", "NCR", "Wincor"];
const ENVS = [
  "Agence_Facade",
  "Agence_Interieure",
  "Centre_Commercial",
  "Site_Isole",
];

function niveauCallout(n) {
  if (n === "CRITIQUE") return "coral";
  if (n === "ÉLEVÉ") return "amber";
  if (n === "MODÉRÉ") return "sky";
  return "sage";
}
function niveauIcon(n) {
  if (n === "CRITIQUE") return <IconAlert width={16} height={16} />;
  if (n === "ÉLEVÉ") return <IconAlert width={16} height={16} />;
  if (n === "MODÉRÉ") return <IconInfo width={16} height={16} />;
  return <IconCheck width={16} height={16} />;
}

const DEFAULTS = {
  ville: "Casablanca",
  type_gab: "NCR",
  environnement: "Agence_Interieure",
  age: 4,
  erreurs_lecteur: 2,
  erreurs_dist: 0,
  temperature: 35.0,
  jours_maint: 60,
  nb_tx: 120,
  taux_erreur_tx: 5.0,
  latence_ms: 80.0,
  deconnexions: 0,
  erreurs_roll7: 1.5,
  temp_roll7: 34.0,
};

export default function ScoringLive() {
  const [inp, setInp] = useState(DEFAULTS);
  const [infoOpen, setInfoOpen] = useState(false);
  const set = (k) => (v) => setInp((p) => ({ ...p, [k]: v }));
  const theme = useContext(ThemeContext);
  const cc = chartColors(theme === "dark");

  const postBody = { ...inp, taux_erreur_tx: inp.taux_erreur_tx / 100 };
  const { data: result, error } = usePost("/api/scoring", postBody);

  const gaugeData = result
    ? [
        {
          type: "indicator",
          mode: "gauge+number",
          value: result.score_pct,
          number: {
            suffix: "%",
            font: { size: 36, family: "Fraunces", color: result.couleur },
          },
          gauge: {
            axis: {
              range: [0, 100],
              tickfont: { size: 8, color: cc.fontMuted },
              tickcolor: cc.scaleMid,
            },
            bar: { color: result.couleur, thickness: 0.2 },
            bgcolor: cc.gaugeBg,
            borderwidth: 0,
            steps: [
              { range: [0, 15], color: "rgba(107,168,138,0.10)" },
              { range: [15, 35], color: "rgba(90,143,196,0.10)" },
              { range: [35, 65], color: "rgba(232,160,69,0.10)" },
              { range: [65, 100], color: "rgba(212,100,90,0.14)" },
            ],
            threshold: {
              line: { color: result.couleur, width: 3 },
              thickness: 0.8,
              value: result.score_pct,
            },
          },
        },
      ]
    : [];

  const contribData = result
    ? [
        {
          type: "bar",
          orientation: "h",
          x: Object.values(result.contributions),
          y: Object.keys(result.contributions),
          marker: {
            color: "#C8102E",
            opacity: 0.8,
            line: { color: cc.outline, width: 0.5 },
          },
          hovertemplate: "<b>%{y}</b><br>%{x:.4f}<extra></extra>",
        },
      ]
    : [];

  return (
    <div>
      <Hero
        eyebrow="Outil opérationnel — évaluation temps réel"
        titleMain="Ce GAB va-t-il"
        titleEm="tomber en panne ?"
        desc="Saisissez les métriques du jour d'un GAB spécifique. Le modèle ML calibré (105 features) prédit la probabilité de panne sous 48h."
        tags={[
          { label: "ML calibré", kind: "ghost" },
          { label: "105 features", kind: "amber" },
          { label: "Temps réel", kind: "coral" },
        ]}
      />

      {error && <ErrorState message={`Erreur de scoring : ${error}`} />}

      <div className="col-grid col-5-5">
        {/* Inputs */}
        <div>
          <SectionHead title="Identité du GAB" />
          {[
            ["Ville", "ville", VILLES],
            ["Constructeur", "type_gab", TYPES],
            ["Environnement", "environnement", ENVS],
          ].map(([label, key, opts]) => (
            <div key={key} style={{ marginBottom: "0.8rem" }}>
              <label className="sb-filter-label">{label}</label>
              <select
                className="sb-select"
                value={inp[key]}
                onChange={(e) =>
                  setInp((p) => ({ ...p, [key]: e.target.value }))
                }
              >
                {opts.map((o) => (
                  <option key={o} value={o}>
                    {o.replace(/_/g, " ")}
                  </option>
                ))}
              </select>
            </div>
          ))}
          <Slider
            label="Âge du GAB (années)"
            value={inp.age}
            min={1}
            max={10}
            onChange={set("age")}
          />

          <SectionHead title="Métriques du jour" />
          <Slider
            label="Erreurs lecteur de carte"
            value={inp.erreurs_lecteur}
            min={0}
            max={20}
            onChange={set("erreurs_lecteur")}
          />
          <Slider
            label="Erreurs distributeur"
            value={inp.erreurs_dist}
            min={0}
            max={10}
            onChange={set("erreurs_dist")}
          />
          <Slider
            label="Température interne (°C)"
            value={inp.temperature}
            min={25}
            max={65}
            step={0.5}
            onChange={set("temperature")}
          />
          <Slider
            label="Jours depuis maintenance"
            value={inp.jours_maint}
            min={0}
            max={365}
            onChange={set("jours_maint")}
          />
          <Slider
            label="Transactions du jour"
            value={inp.nb_tx}
            min={0}
            max={250}
            onChange={set("nb_tx")}
          />
          <Slider
            label="Taux d'erreur transactions (%)"
            value={inp.taux_erreur_tx}
            min={0}
            max={30}
            step={0.5}
            onChange={set("taux_erreur_tx")}
            format={(v) => `${v.toFixed(1)}%`}
          />
          <Slider
            label="Latence réseau (ms)"
            value={inp.latence_ms}
            min={10}
            max={1000}
            step={5}
            onChange={set("latence_ms")}
            format={(v) => `${v} ms`}
          />
          <Slider
            label="Déconnexions réseau"
            value={inp.deconnexions}
            min={0}
            max={10}
            onChange={set("deconnexions")}
          />

          <SectionHead title="Historique 7 jours" />
          <Slider
            label="Moy. erreurs lecteur / 7j"
            value={inp.erreurs_roll7}
            min={0}
            max={15}
            step={0.5}
            onChange={set("erreurs_roll7")}
            format={(v) => v.toFixed(1)}
          />
          <Slider
            label="Moy. température / 7j (°C)"
            value={inp.temp_roll7}
            min={25}
            max={65}
            step={0.5}
            onChange={set("temp_roll7")}
            format={(v) => `${v.toFixed(1)} °C`}
          />
        </div>

        {/* Result */}
        <div>
          <SectionHead title="Score de risque estimé" />
          {result ? (
            <>
              <Chart
                data={gaugeData}
                layout={{
                  height: 260,
                  margin: { l: 20, r: 20, t: 20, b: 10 },
                  paper_bgcolor: cc.bg,
                  font: { color: cc.fontMuted, family: "Instrument Sans" },
                }}
              />

              <div
                style={{
                  textAlign: "center",
                  padding: "1.1rem",
                  background: "var(--ink-800)",
                  border: `1px solid ${result.couleur}44`,
                  borderRadius: 12,
                  marginBottom: "1.5rem",
                }}
              >
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    gap: "0.6rem",
                    marginBottom: "0.4rem",
                  }}
                >
                  <IconShield
                    width={20}
                    height={20}
                    style={{ color: result.couleur }}
                  />
                  <span
                    style={{
                      fontFamily: "Fraunces",
                      fontSize: "1.8rem",
                      fontWeight: 900,
                      color: result.couleur,
                      letterSpacing: "-0.02em",
                    }}
                  >
                    {result.niveau}
                  </span>
                </div>
                <div
                  style={{ fontSize: "0.82rem", color: "var(--text-muted)" }}
                >
                  {result.recommandation}
                </div>
                {result.mode && (
                  <div
                    style={{
                      marginTop: "0.5rem",
                      fontSize: "0.62rem",
                      fontWeight: 600,
                      letterSpacing: "0.1em",
                      textTransform: "uppercase",
                      color: "var(--text-ghost)",
                    }}
                  >
                    {result.mode === "ml" ? "Modèle ML calibré" : "Mode heuristique"}
                    {result.model_version && ` · ${result.model_version}`}
                  </div>
                )}
              </div>

              <SectionHead title="Contributions au score" />
              <Chart
                data={contribData}
                layout={{
                  height: 280,
                  xaxis: { showgrid: false, visible: false },
                  yaxis: {
                    tickfont: { size: 9, color: cc.fontMuted },
                    showgrid: false,
                  },
                  margin: { l: 160, r: 40, t: 10, b: 4 },
                }}
              />

              <Callout
                icon={niveauIcon(result.niveau)}
                text={`P(panne) : <strong>${(result.score * 100).toFixed(1)}%</strong> · Niveau : <strong style="color:${result.couleur}">${result.niveau}</strong> · ${result.recommandation}`}
                kind={niveauCallout(result.niveau)}
              />
            </>
          ) : (
            <div className="loading" style={{ minHeight: 200 }}>
              Calcul en cours…
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
