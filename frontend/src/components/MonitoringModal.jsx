import { useState, useRef, useEffect, useMemo } from "react";
import { StatusDot } from "./Icons";

const NIVEAUX = ["Tous", "CRITIQUE", "ÉLEVÉ", "MODÉRÉ", "FAIBLE"];
const NIVEAU_COLORS = {
  CRITIQUE: "#d4645a",
  "ÉLEVÉ": "#e8a045",
  "MODÉRÉ": "#5a8fc4",
  FAIBLE: "#6ba88a",
};

export default function MonitoringModal({ gabs, onClose, onSelect }) {
  const [search, setSearch] = useState("");
  const [niveau, setNiveau] = useState("Tous");
  const overlayRef = useRef(null);
  // Snapshot à l'ouverture — les données ne bougent plus pendant la recherche
  const snapshot = useMemo(() => [...(gabs || [])], []);

  useEffect(() => {
    const handler = (e) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [onClose]);

  const filtered = snapshot.filter((g) => {
    if (niveau !== "Tous" && g.niveau !== niveau) return false;
    if (search) {
      const q = search.toLowerCase();
      return (
        g.gab_id.toLowerCase().includes(q) ||
        g.ville.toLowerCase().includes(q) ||
        g.type_gab.toLowerCase().includes(q)
      );
    }
    return true;
  });

  return (
    <div
      ref={overlayRef}
      onClick={(e) => {
        if (e.target === overlayRef.current) onClose();
      }}
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 10000,
        background: "rgba(0,0,0,0.55)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: "2rem",
      }}
    >
      <div
        style={{
          background: "var(--surface-1)",
          border: "1px solid var(--border-dim)",
          borderRadius: 14,
          boxShadow: "var(--shadow-lg)",
          width: "100%",
          maxWidth: 720,
          maxHeight: "80vh",
          display: "flex",
          flexDirection: "column",
          overflow: "hidden",
        }}
      >
        {/* Header */}
        <div
          style={{
            padding: "1rem 1.2rem",
            borderBottom: "1px solid var(--border-dim)",
            display: "flex",
            alignItems: "center",
            gap: "0.75rem",
          }}
        >
          <div style={{ flex: 1 }}>
            <div
              style={{
                fontSize: "1rem",
                fontWeight: 700,
                fontFamily: "Fraunces",
                color: "var(--text-primary)",
              }}
            >
              État des GAB
              <span
                style={{
                  fontWeight: 400,
                  fontSize: "0.78rem",
                  color: "var(--text-muted)",
                  marginLeft: "0.5rem",
                }}
              >
                {filtered.length} / {snapshot.length}
              </span>
            </div>
          </div>
          <button
            onClick={onClose}
            style={{
              background: "none",
              border: "none",
              color: "var(--text-muted)",
              fontSize: "1.2rem",
              cursor: "pointer",
              padding: "0.2rem 0.4rem",
              lineHeight: 1,
            }}
          >
            ✕
          </button>
        </div>

        {/* Filters */}
        <div
          style={{
            padding: "0.75rem 1.2rem",
            display: "flex",
            gap: "0.6rem",
            alignItems: "center",
            flexWrap: "wrap",
            borderBottom: "1px solid var(--border-dim)",
          }}
        >
          <input
            type="text"
            placeholder="Rechercher GAB, ville, constructeur…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            style={{
              flex: 1,
              minWidth: 180,
              padding: "0.4rem 0.7rem",
              fontSize: "0.78rem",
              background: "var(--surface-3)",
              border: "1px solid var(--border-dim)",
              borderRadius: 6,
              color: "var(--text-primary)",
              outline: "none",
            }}
          />
          <div style={{ display: "flex", gap: 4 }}>
            {NIVEAUX.map((n) => (
              <button
                key={n}
                onClick={() => setNiveau((prev) => (prev === n ? "Tous" : n))}
                style={{
                  padding: "0.3rem 0.6rem",
                  fontSize: "0.68rem",
                  fontWeight: niveau === n ? 700 : 500,
                  borderRadius: 5,
                  border: `1px solid ${
                    niveau === n
                      ? NIVEAU_COLORS[n] || "var(--text-muted)"
                      : "var(--border-dim)"
                  }`,
                  background:
                    niveau === n
                      ? `${NIVEAU_COLORS[n] || "var(--text-muted)"}22`
                      : "transparent",
                  color:
                    niveau === n
                      ? NIVEAU_COLORS[n] || "var(--text-primary)"
                      : "var(--text-muted)",
                  cursor: "pointer",
                }}
              >
                {n === "Tous" ? "Tous" : n}
              </button>
            ))}
          </div>
        </div>

        {/* List */}
        <div style={{ flex: 1, overflowY: "auto", padding: "0.5rem 0" }}>
          {filtered.length === 0 ? (
            <div
              style={{
                padding: "2rem",
                textAlign: "center",
                color: "var(--text-ghost)",
                fontSize: "0.82rem",
              }}
            >
              Aucun GAB trouvé
            </div>
          ) : (
            filtered.map((g) => (
              <div
                key={g.gab_id}
                onClick={() => onSelect(g)}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "0.6rem",
                  padding: "0.55rem 1.2rem",
                  cursor: "pointer",
                  transition: "background 0.15s",
                }}
                onMouseEnter={(e) =>
                  (e.currentTarget.style.background = `${g.couleur}0d`)
                }
                onMouseLeave={(e) =>
                  (e.currentTarget.style.background = "transparent")
                }
              >
                <StatusDot color={g.couleur} size={8} />
                <span
                  style={{
                    fontWeight: 600,
                    fontSize: "0.8rem",
                    color: "var(--text-primary)",
                    fontFamily: "Geist Mono, monospace",
                    width: 90,
                  }}
                >
                  {g.gab_id}
                </span>
                <span
                  style={{
                    fontSize: "0.75rem",
                    color: "var(--text-mid)",
                    flex: 1,
                  }}
                >
                  {g.ville}
                </span>
                <span
                  style={{
                    fontSize: "0.7rem",
                    color: "var(--text-muted)",
                    width: 70,
                  }}
                >
                  {g.type_gab}
                </span>
                <span
                  style={{
                    fontSize: "0.68rem",
                    color: "var(--text-muted)",
                    width: 75,
                  }}
                >
                  {g.environnement.replace(/_/g, " ")}
                </span>
                <span
                  style={{
                    fontFamily: "Geist Mono, monospace",
                    fontSize: "0.78rem",
                    fontWeight: 700,
                    color: g.couleur,
                    width: 50,
                    textAlign: "right",
                  }}
                >
                  {g.score_pct}%
                </span>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}

export function GabDetailModal({ gab, onClose }) {
  if (!gab) return null;

  const metrics = [
    { label: "Score de risque", value: `${gab.score_pct}%`, color: gab.couleur, big: true },
    { label: "Niveau", value: gab.niveau, color: gab.couleur },
    { label: "Température", value: `${gab.temperature}°C` },
    { label: "Erreurs lecteur", value: gab.erreurs_lecteur },
    { label: "Jours depuis maintenance", value: gab.jours_maint },
    { label: "Âge", value: `${gab.age} ans` },
  ];

  return (
    <div
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 10001,
        background: "rgba(0,0,0,0.55)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: "2rem",
      }}
    >
      <div
        style={{
          background: "var(--surface-1)",
          border: `1px solid ${gab.couleur}44`,
          borderRadius: 14,
          boxShadow: "var(--shadow-lg)",
          width: "100%",
          maxWidth: 400,
          padding: "1.5rem",
        }}
      >
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: "1.2rem",
          }}
        >
          <div>
            <div
              style={{
                fontFamily: "Geist Mono, monospace",
                fontSize: "1.1rem",
                fontWeight: 700,
                color: "var(--text-primary)",
              }}
            >
              {gab.gab_id}
            </div>
            <div
              style={{
                fontSize: "0.78rem",
                color: "var(--text-muted)",
                marginTop: 2,
              }}
            >
              {gab.ville} · {gab.type_gab} · {gab.environnement.replace(/_/g, " ")}
            </div>
          </div>
          <button
            onClick={onClose}
            style={{
              background: "none",
              border: "none",
              color: "var(--text-muted)",
              fontSize: "1.2rem",
              cursor: "pointer",
            }}
          >
            ✕
          </button>
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: 8,
          }}
        >
          {metrics.map(({ label, value, color, big }) => (
            <div
              key={label}
              style={{
                background: color ? `${color}11` : "var(--surface-3)",
                border: `1px solid ${color ? `${color}33` : "var(--border-dim)"}`,
                borderRadius: 8,
                padding: "0.6rem 0.8rem",
                gridColumn: big ? "1 / -1" : undefined,
                textAlign: big ? "center" : undefined,
              }}
            >
              <div
                style={{
                  fontSize: "0.6rem",
                  letterSpacing: "0.1em",
                  textTransform: "uppercase",
                  color: "var(--text-muted)",
                  marginBottom: 4,
                }}
              >
                {label}
              </div>
              <div
                style={{
                  fontFamily: "Geist Mono, monospace",
                  fontSize: big ? "1.8rem" : "1.1rem",
                  fontWeight: 700,
                  color: color || "var(--text-primary)",
                  lineHeight: 1,
                }}
              >
                {value}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
