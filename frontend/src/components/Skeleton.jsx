/** Shimmer skeleton — drop-in replacement while data loads */
export function Skeleton({ width = "100%", height = 16, radius = 6, style }) {
  return (
    <div
      className="skeleton"
      style={{ width, height, borderRadius: radius, flexShrink: 0, ...style }}
    />
  );
}

/** Full KPI card skeleton */
export function KpiSkeleton() {
  return (
    <div className="kpi-card kpi-amber" style={{ pointerEvents: "none" }}>
      <Skeleton
        width={32}
        height={32}
        radius={8}
        style={{ marginBottom: "1rem" }}
      />
      <Skeleton width="55%" height={10} style={{ marginBottom: "0.55rem" }} />
      <Skeleton width="70%" height={28} style={{ marginBottom: "0.55rem" }} />
      <Skeleton width="85%" height={10} />
    </div>
  );
}

/** Generic page loading — 4 KPI + chart area */
export function PageSkeleton() {
  return (
    <div>
      {/* hero */}
      <div style={{ padding: "2.8rem 0 1.8rem" }}>
        <Skeleton width={180} height={10} style={{ marginBottom: "0.9rem" }} />
        <Skeleton width="50%" height={36} style={{ marginBottom: "0.6rem" }} />
        <Skeleton width="40%" height={36} style={{ marginBottom: "0.9rem" }} />
        <Skeleton width={380} height={13} style={{ marginBottom: "0.4rem" }} />
        <Skeleton width={300} height={13} style={{ marginBottom: "1.1rem" }} />
        <div style={{ display: "flex", gap: "0.5rem" }}>
          {[60, 70, 90, 80].map((w, i) => (
            <Skeleton key={i} width={w} height={20} radius={4} />
          ))}
        </div>
        <Skeleton width="100%" height={1} style={{ marginTop: "1.4rem" }} />
      </div>

      {/* KPI row */}
      <div className="kpi-grid" style={{ marginBottom: "2rem" }}>
        {[...Array(4)].map((_, i) => (
          <KpiSkeleton key={i} />
        ))}
      </div>

      {/* Chart areas */}
      <div className="col-grid col-6-4">
        <Skeleton width="100%" height={320} radius={12} />
        <Skeleton width="100%" height={320} radius={12} />
      </div>
    </div>
  );
}

/** Error state */
export function ErrorState({ message = "Erreur de chargement", onRetry }) {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        minHeight: 260,
        gap: "1rem",
      }}
    >
      <svg
        width={40}
        height={40}
        viewBox="0 0 24 24"
        fill="none"
        stroke="#C8102E"
        strokeWidth={1.5}
        strokeLinecap="round"
      >
        <circle cx="12" cy="12" r="10" />
        <line x1="12" y1="8" x2="12" y2="12" />
        <line x1="12" y1="16" x2="12.01" y2="16" strokeWidth={2.5} />
      </svg>
      <p style={{ color: "var(--text-muted)", fontSize: "0.88rem" }}>
        {message}
      </p>
      {onRetry && (
        <button
          onClick={onRetry}
          style={{
            padding: "0.45rem 1rem",
            borderRadius: 6,
            border: "1px solid var(--border-dim)",
            background: "var(--surface-2)",
            color: "var(--text-mid)",
            cursor: "pointer",
            fontSize: "0.8rem",
          }}
        >
          Réessayer
        </button>
      )}
    </div>
  );
}
