import { useContext } from "react";
import Plot from "react-plotly.js";
import { ThemeContext } from "../App";

const NG = "rgba(0,0,0,0)";

export const PALETTE = [
  "#C8102E",
  "#00703C",
  "#3B82C4",
  "#B8922A",
  "#7C3AED",
  "#E05252",
  "#00A65A",
];

export function chartColors(isDark) {
  return {
    bg: isDark ? "#0a0a0c" : "#ffffff",
    barFill: isDark ? "#1a1d24" : "#c8cdd8",
    outline: isDark ? "#0a0a0c" : "#ffffff",
    grid: isDark ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.07)",
    radarGrid: isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.08)",
    scaleMin: isDark ? "#0d1017" : "#f0f1f4",
    scaleMid: isDark ? "#232730" : "#d0d5e0",
    annotText: isDark ? "#f0f2f8" : "#1e2230",
    gaugeBg: isDark ? "#1a1d24" : "#f4f5f8",
    vLine: isDark ? "rgba(255,255,255,0.2)" : "rgba(0,0,0,0.2)",
    fontMid: isDark ? "#8891a8" : "#5c6480",
    fontMuted: isDark ? "#5c6480" : "#8891a8",
  };
}

export function baseLayout(overrides = {}) {
  const isDark = document.documentElement.dataset.theme === "dark";
  const bg   = isDark ? "#0a0a0c" : "#ffffff";
  const font = isDark ? "#404759" : "#5c6480";
  const grid = isDark ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.07)";
  const hbg  = isDark ? "#252830" : "#f4f5f8";
  const hbdr = isDark ? "#444b5e" : "#d0d5e0";
  const hfnt = isDark ? "#e8ecf5" : "#1e2230";
  return {
    plot_bgcolor: bg,
    paper_bgcolor: bg,
    font: { color: font, family: "Instrument Sans" },
    xaxis: {
      showgrid: false,
      color: font,
      tickfont: { size: 10 },
      zeroline: false,
      showline: true,
      linecolor: font,
      linewidth: 1,
    },
    yaxis: {
      gridcolor: grid,
      gridwidth: 0.4,
      color: font,
      tickfont: { size: 10 },
      zeroline: false,
      showline: true,
      linecolor: font,
      linewidth: 1,
    },
    legend: { bgcolor: NG, font: { size: 10, color: font } },
    margin: { l: 50, r: 16, t: 24, b: 44 },
    hoverlabel: {
      bgcolor: hbg,
      bordercolor: hbdr,
      font: { size: 12, family: "Geist Mono", color: hfnt },
    },
    ...overrides,
  };
}

export default function Chart({ data, layout, style }) {
  const theme = useContext(ThemeContext); // subscribe → re-render on theme change
  return (
    <Plot
      data={data}
      layout={{ ...baseLayout(), ...layout }}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: "100%", ...style }}
      useResizeHandler
      key={theme} // force full remount on theme change
    />
  );
}
