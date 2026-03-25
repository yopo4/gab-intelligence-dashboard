import { useState, useRef, useEffect, useContext, useMemo } from "react";
import { FilterContext } from "../App";
import { buildFilterQuery } from "../hooks/useApi";
import {
  IconDownload,
  IconFileText,
  IconFileSpreadsheet,
  IconChevronDown,
} from "./Icons";

/* ── Download helper ─────────────────────────────────────────── */

function downloadFromApi(url, fallbackName) {
  const a = document.createElement("a");
  a.href = url;
  a.download = fallbackName;
  document.body.appendChild(a);
  a.click();
  a.remove();
}

/* ── Format definitions ─────────────────────────────────────── */

const FORMATS = {
  csv: {
    label: "CSV",
    desc: "Données brutes",
    icon: (p) => <IconFileText {...p} />,
    ext: "csv",
  },
  excel: {
    label: "Excel",
    desc: "Rapport multi-feuilles",
    icon: (p) => <IconFileSpreadsheet {...p} />,
    ext: "xlsx",
  },
  json: {
    label: "JSON",
    desc: "Format structuré",
    icon: (p) => (
      <svg viewBox="0 0 24 24" width={p.width || 15} height={p.height || 15}
        fill="none" stroke="currentColor" strokeWidth={1.75}
        strokeLinecap="round" strokeLinejoin="round">
        <path d="M8 3H7a2 2 0 0 0-2 2v5a2 2 0 0 1-2 2 2 2 0 0 1 2 2v5a2 2 0 0 0 2 2h1" />
        <path d="M16 3h1a2 2 0 0 1 2 2v5a2 2 0 0 0 2 2 2 2 0 0 0-2 2v5a2 2 0 0 1-2 2h-1" />
      </svg>
    ),
    ext: "json",
  },
  pdf: {
    label: "PDF",
    desc: "Rapport complet",
    icon: (p) => (
      <svg viewBox="0 0 24 24" width={p.width || 15} height={p.height || 15}
        fill="none" stroke="currentColor" strokeWidth={1.75}
        strokeLinecap="round" strokeLinejoin="round">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
        <polyline points="14 2 14 8 20 8" />
        <path d="M9 15v-2h1.5a1.5 1.5 0 0 1 0 3H9" />
      </svg>
    ),
    ext: "pdf",
  },
};

/* ── ExportMenu Component ──────────────────────────────────── */

export default function ExportMenu({
  section = "overview",
  formats = ["csv", "excel", "json", "pdf"],
  compact = false,
}) {
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(null);
  const ref = useRef(null);
  const { filters } = useContext(FilterContext);
  const query = useMemo(() => buildFilterQuery(filters), [filters]);

  useEffect(() => {
    const handler = (e) => {
      if (ref.current && !ref.current.contains(e.target)) setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const handleExport = async (format) => {
    setLoading(format);
    try {
      const base = `/api/export/${format}`;
      const sep = base.includes("?") ? "&" : "?";
      const sectionParam = format !== "excel" && format !== "pdf"
        ? `section=${section}&`
        : "";
      const url = `${base}${sep}${sectionParam}${query}`;
      downloadFromApi(url, `gab_${section}.${FORMATS[format].ext}`);
    } finally {
      setTimeout(() => setLoading(null), 1500);
    }
  };

  const visibleFormats = formats.filter((f) => FORMATS[f]);

  if (compact) {
    return (
      <div ref={ref} style={{ position: "relative", display: "inline-flex" }}>
        <button
          className="export-btn export-btn-compact"
          onClick={() => setOpen((o) => !o)}
          title="Exporter"
        >
          <IconDownload width={14} height={14} />
          <IconChevronDown
            width={12}
            height={12}
            style={{
              transition: "transform 0.15s",
              transform: open ? "rotate(180deg)" : "rotate(0)",
            }}
          />
        </button>
        {open && (
          <div className="export-dropdown">
            {visibleFormats.map((f) => {
              const fmt = FORMATS[f];
              const Icon = fmt.icon;
              return (
                <button
                  key={f}
                  className="export-dropdown-item"
                  onClick={() => {
                    handleExport(f);
                    setOpen(false);
                  }}
                  disabled={loading === f}
                >
                  <Icon width={15} height={15} />
                  <span className="export-dropdown-label">{fmt.label}</span>
                  <span className="export-dropdown-desc">{fmt.desc}</span>
                  {loading === f && <span className="export-spinner" />}
                </button>
              );
            })}
          </div>
        )}
      </div>
    );
  }

  return (
    <div ref={ref} style={{ position: "relative", display: "inline-flex" }}>
      <button
        className="export-btn"
        onClick={() => setOpen((o) => !o)}
      >
        <IconDownload width={14} height={14} />
        <span>Exporter</span>
        <IconChevronDown
          width={12}
          height={12}
          style={{
            transition: "transform 0.15s",
            transform: open ? "rotate(180deg)" : "rotate(0)",
          }}
        />
      </button>
      {open && (
        <div className="export-dropdown">
          {visibleFormats.map((f) => {
            const fmt = FORMATS[f];
            const Icon = fmt.icon;
            return (
              <button
                key={f}
                className="export-dropdown-item"
                onClick={() => {
                  handleExport(f);
                  setOpen(false);
                }}
                disabled={loading === f}
              >
                <Icon width={15} height={15} />
                <span className="export-dropdown-label">{fmt.label}</span>
                <span className="export-dropdown-desc">{fmt.desc}</span>
                {loading === f && <span className="export-spinner" />}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

/* ── Standalone full report button ─────────────────────────── */

export function ExportReportButton() {
  const [loading, setLoading] = useState(false);
  const { filters } = useContext(FilterContext);
  const query = useMemo(() => buildFilterQuery(filters), [filters]);

  const handleClick = () => {
    setLoading(true);
    downloadFromApi(`/api/export/pdf?${query}`, "gab_rapport.pdf");
    setTimeout(() => setLoading(false), 2000);
  };

  return (
    <button
      className="export-btn export-btn-report"
      onClick={handleClick}
      disabled={loading}
    >
      <IconDownload width={14} height={14} />
      <span>{loading ? "Génération…" : "Rapport PDF"}</span>
    </button>
  );
}
