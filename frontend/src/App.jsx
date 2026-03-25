import { useState, useEffect, useRef, createContext } from "react";
import { createPortal } from "react-dom";
import {
  BrowserRouter,
  Routes,
  Route,
  useNavigate,
  useLocation,
} from "react-router-dom";
import MultiSelect from "./components/MultiSelect";
import Overview from "./pages/Overview";
import Geography from "./pages/Geography";
import Models from "./pages/Models";
import Features from "./pages/Features";
import Threshold from "./pages/Threshold";
import ScoringLive from "./pages/ScoringLive";
import {
  BpmLogo,
  IconHome,
  IconMap,
  IconBarChart,
  IconCpu,
  IconSettings,
  IconTerminal,
  IconSun,
  IconMoon,
  IconBell,
  StatusDot,
} from "./components/Icons";
import { useApiPoll } from "./hooks/useApi";
import { GabDetailModal } from "./components/MonitoringModal";

export const FilterContext = createContext(null);
export const ThemeContext = createContext("light");
export const MonitoringContext = createContext(null);

const NAV = [
  { path: "/", Icon: IconHome, label: "Vue d'ensemble" },
  { path: "/geo", Icon: IconMap, label: "Géographie" },
  { path: "/modeles", Icon: IconBarChart, label: "Modèles ML" },
  { path: "/features", Icon: IconCpu, label: "Features" },
  { path: "/seuil", Icon: IconSettings, label: "Seuil & Coûts" },
  { path: "/scoring", Icon: IconTerminal, label: "Scoring Live" },
];

function NotifBell({ monitoring, onSelectGab }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);
  const btnRef = useRef(null);
  const [pos, setPos] = useState({ top: 0, left: 0 });

  useEffect(() => {
    const handler = (e) => {
      if (ref.current && !ref.current.contains(e.target) && !btnRef.current?.contains(e.target))
        setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const handleToggle = () => {
    if (!open && btnRef.current) {
      const rect = btnRef.current.getBoundingClientRect();
      setPos({ top: rect.bottom + 8, left: rect.right });
    }
    setOpen((o) => !o);
  };

  const handleClick = (a) => {
    setOpen(false);
    const g = monitoring?.all_gabs?.find((x) => x.gab_id === a.gab_id);
    if (g) onSelectGab(g);
  };

  const nb = monitoring?.nb_alertes ?? 0;
  const alertes = monitoring?.alertes ?? [];

  return (
    <>
      <button
        ref={btnRef}
        className="theme-toggle"
        onClick={handleToggle}
        title="Alertes temps réel"
        style={{ position: "relative" }}
      >
        <IconBell width={14} height={14} />
        {nb > 0 && (
          <span
            style={{
              position: "absolute",
              top: -4,
              right: -4,
              background: "#C8102E",
              color: "#fff",
              fontSize: "0.55rem",
              fontWeight: 700,
              borderRadius: "50%",
              width: 15,
              height: 15,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              lineHeight: 1,
            }}
          >
            {nb > 9 ? "9+" : nb}
          </span>
        )}
      </button>
      {open && createPortal(
        <div
          ref={ref}
          style={{
            position: "fixed",
            top: pos.top,
            left: pos.left,
            zIndex: 99999,
            width: 320,
            maxHeight: 400,
            overflowY: "auto",
            background: "var(--surface-2)",
            border: "1px solid var(--border-dim)",
            borderRadius: 10,
            boxShadow: "var(--shadow-lg)",
            padding: "0.5rem 0",
          }}
        >
          <div
            style={{
              padding: "0.5rem 0.75rem",
              fontSize: "0.68rem",
              fontWeight: 700,
              letterSpacing: "0.1em",
              textTransform: "uppercase",
              color: "var(--text-muted)",
              borderBottom: "1px solid var(--border-dim)",
            }}
          >
            Alertes récentes ({nb})
          </div>
          {alertes.length === 0 ? (
            <div
              style={{
                padding: "1.5rem 0.75rem",
                textAlign: "center",
                fontSize: "0.78rem",
                color: "var(--text-ghost)",
              }}
            >
              Aucune alerte active
            </div>
          ) : (
            alertes.slice(0, 8).map((a, i) => (
              <div
                key={i}
                onClick={() => handleClick(a)}
                style={{
                  padding: "0.55rem 0.75rem",
                  borderBottom:
                    i < alertes.length - 1
                      ? "1px solid var(--border-dim)"
                      : "none",
                  display: "flex",
                  gap: "0.5rem",
                  alignItems: "flex-start",
                  cursor: "pointer",
                  transition: "background 0.15s",
                }}
                onMouseEnter={(e) =>
                  (e.currentTarget.style.background = `${a.couleur}0d`)
                }
                onMouseLeave={(e) =>
                  (e.currentTarget.style.background = "transparent")
                }
              >
                <StatusDot color={a.couleur} size={8} />
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div
                    style={{
                      fontSize: "0.75rem",
                      fontWeight: 600,
                      color: "var(--text-primary)",
                    }}
                  >
                    {a.gab_id}{" "}
                    <span
                      style={{
                        fontWeight: 400,
                        color: "var(--text-muted)",
                      }}
                    >
                      · {a.ville}
                    </span>
                  </div>
                  <div
                    style={{
                      fontSize: "0.68rem",
                      color: a.couleur,
                      marginTop: 2,
                    }}
                  >
                    {a.message}
                  </div>
                  <div
                    style={{
                      fontSize: "0.6rem",
                      color: "var(--text-ghost)",
                      marginTop: 2,
                    }}
                  >
                    {new Date(a.timestamp).toLocaleTimeString("fr-FR", {
                      hour: "2-digit",
                      minute: "2-digit",
                    })}
                  </div>
                </div>
              </div>
            ))
          )}
        </div>,
        document.body,
      )}
    </>
  );
}

function Sidebar({ filters, setFilters, meta, theme, toggleTheme, monitoring, onSelectGab }) {
  const location = useLocation();
  const navigate = useNavigate();

  return (
    <aside className="sidebar">
      <div className="sb-brand">
        <BpmLogo size={38} />
        <div style={{ flex: 1 }}>
          <div className="sb-title">GAB Intelligence</div>
          <div className="sb-subtitle">Banque Populaire · Maroc</div>
        </div>
        <NotifBell monitoring={monitoring} onSelectGab={onSelectGab} />
        <button
          className="theme-toggle"
          onClick={toggleTheme}
          title={theme === "dark" ? "Mode clair" : "Mode sombre"}
        >
          {theme === "dark" ? (
            <IconSun width={14} height={14} />
          ) : (
            <IconMoon width={14} height={14} />
          )}
        </button>
      </div>

      <nav className="sb-nav">
        {NAV.map(({ path, Icon, label }) => (
          <button
            key={path}
            className={`sb-nav-item${location.pathname === path ? " active" : ""}`}
            onClick={() => navigate(path)}
          >
            <Icon width={16} height={16} />
            <span>{label}</span>
          </button>
        ))}
      </nav>

      {meta && (
        <div className="sb-filters">
          <label className="sb-filter-label">Villes</label>
          <MultiSelect
            options={meta.villes}
            value={filters.villes}
            onChange={(sel) =>
              setFilters((f) => ({
                ...f,
                villes: sel.length ? sel : [...meta.villes],
              }))
            }
            allLabel="Toutes les villes"
          />

          <label className="sb-filter-label">Type GAB</label>
          <MultiSelect
            options={meta.types}
            value={filters.types}
            onChange={(sel) =>
              setFilters((f) => ({
                ...f,
                types: sel.length ? sel : [...meta.types],
              }))
            }
            allLabel="Tous les types"
          />

          <label className="sb-filter-label">Période</label>
          <MultiSelect
            options={meta.annees.filter((a) => a !== "Tout")}
            value={filters.annees.filter((a) => a !== "Tout")}
            onChange={(sel) =>
              setFilters((f) => ({
                ...f,
                annees: sel.length ? sel : [...meta.annees],
              }))
            }
            allLabel="Toutes les années"
          />
        </div>
      )}

      <div className="sb-meta">
        <div className="sb-meta-row">
          <span className="sb-meta-label">Période</span>
          <span className="sb-meta-val">2022–2023</span>
        </div>
        <div className="sb-meta-row">
          <span className="sb-meta-label">Villes</span>
          <span className="sb-meta-val">{filters.villes.length}</span>
        </div>
        <div className="sb-meta-row">
          <span className="sb-meta-label">Features</span>
          <span className="sb-meta-val">101</span>
        </div>
      </div>
    </aside>
  );
}

function Shell() {
  const [meta, setMeta] = useState(null);
  const [filters, setFilters] = useState({ villes: [], types: [], annees: [] });
  const [theme, setTheme] = useState(
    () => localStorage.getItem("theme") || "light",
  );
  const { data: monitoring, refresh: refreshMonitoring } = useApiPoll("/api/monitoring", 30000);
  const [bellGab, setBellGab] = useState(null);

  // Synchroniser le data-theme au montage et à chaque changement
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  const toggleTheme = () => {
    const next = theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next; // synchrone avant le re-render
    localStorage.setItem("theme", next);
    setTheme(next);
  };

  useEffect(() => {
    fetch("/api/metadata")
      .then((r) => r.json())
      .then((d) => {
        setMeta(d);
        setFilters({ villes: d.villes, types: d.types, annees: [...d.annees] });
      });
  }, []);

  return (
    <ThemeContext.Provider value={theme}>
      <FilterContext.Provider value={{ filters, meta }}>
        <MonitoringContext.Provider value={{ monitoring, refreshMonitoring }}>
          <div className="app-shell">
            <Sidebar
              filters={filters}
              setFilters={setFilters}
              meta={meta}
              theme={theme}
              toggleTheme={toggleTheme}
              monitoring={monitoring}
              onSelectGab={setBellGab}
            />
            <main className="main-content">
              <Routes>
                <Route path="/" element={<Overview />} />
                <Route path="/geo" element={<Geography />} />
                <Route path="/modeles" element={<Models />} />
                <Route path="/features" element={<Features />} />
                <Route path="/seuil" element={<Threshold />} />
                <Route path="/scoring" element={<ScoringLive />} />
              </Routes>
            </main>
          </div>
          {bellGab && (
            <GabDetailModal
              gab={bellGab}
              onClose={() => setBellGab(null)}
            />
          )}
        </MonitoringContext.Provider>
      </FilterContext.Provider>
    </ThemeContext.Provider>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <Shell />
    </BrowserRouter>
  );
}
