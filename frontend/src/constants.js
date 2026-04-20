/**
 * Shared constants — single source of truth for the frontend.
 * Import from here instead of defining inline in pages/components.
 */

// ── GAB metadata ──────────────────────────────────────────────────────────────
export const VILLES = [
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

export const TYPES = ["Diebold", "Hyosung", "NCR", "Wincor"];

export const ENVS = [
  "Agence_Facade",
  "Agence_Interieure",
  "Centre_Commercial",
  "Site_Isole",
];

// ── Risk levels ────────────────────────────────────────────────────────────────
export const NIVEAUX_CONFIG = [
  { key: "FAIBLE",   label: "Normal",   color: "#6ba88a" },
  { key: "MODÉRÉ",   label: "Modéré",   color: "#5a8fc4" },
  { key: "ÉLEVÉ",    label: "Élevé",    color: "#e8a045" },
  { key: "CRITIQUE", label: "Critique", color: "#d4645a" },
];

export const NIVEAU_COLORS = {
  FAIBLE:   "#6ba88a",
  "MODÉRÉ": "#5a8fc4",
  "ÉLEVÉ":  "#e8a045",
  CRITIQUE: "#d4645a",
};

// ── Score thresholds (must match backend scoring_utils.py) ────────────────────
export const SCORE_THRESHOLDS = {
  CRITIQUE: 65,
  ELEVE:    35,
  MODERE:   15,
};

// ── Polling & UI ──────────────────────────────────────────────────────────────
export const POLL_INTERVAL_MS  = 30_000;   // monitoring poll interval
export const NOTIF_MAX_DISPLAY = 8;        // max alerts shown in notification bell
