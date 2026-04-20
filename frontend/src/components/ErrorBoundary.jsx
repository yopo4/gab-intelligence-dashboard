import { Component } from "react";

/**
 * Global error boundary — catches render errors anywhere in the tree
 * and shows a recovery UI instead of a blank crash screen.
 */
export default class ErrorBoundary extends Component {
  state = { error: null, info: null };

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    console.error("[ErrorBoundary]", error, info.componentStack);
    this.setState({ info });
  }

  handleReset = () => {
    this.setState({ error: null, info: null });
  };

  render() {
    if (!this.state.error) return this.props.children;

    const msg = this.state.error?.message || "Erreur inattendue";

    return (
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          minHeight: "100vh",
          gap: "1.2rem",
          padding: "2rem",
          background: "var(--surface-0, #07080d)",
          color: "var(--text-primary, #e8ecf5)",
          fontFamily: "Instrument Sans, sans-serif",
        }}
      >
        <svg
          width={48}
          height={48}
          viewBox="0 0 24 24"
          fill="none"
          stroke="#C8102E"
          strokeWidth={1.5}
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <circle cx="12" cy="12" r="10" />
          <line x1="12" y1="8" x2="12" y2="12" />
          <line x1="12" y1="16" x2="12.01" y2="16" strokeWidth={2.5} />
        </svg>

        <div style={{ textAlign: "center" }}>
          <div
            style={{
              fontFamily: "Fraunces, serif",
              fontSize: "1.4rem",
              fontWeight: 700,
              marginBottom: "0.4rem",
            }}
          >
            Une erreur est survenue
          </div>
          <div
            style={{
              fontSize: "0.82rem",
              color: "var(--text-muted, #8891a8)",
              maxWidth: 400,
            }}
          >
            {msg}
          </div>
        </div>

        <button
          onClick={this.handleReset}
          style={{
            padding: "0.5rem 1.2rem",
            borderRadius: 7,
            border: "1px solid var(--border-dim, #2a2d38)",
            background: "var(--surface-2, #1a1d24)",
            color: "var(--text-mid, #c4c9d8)",
            cursor: "pointer",
            fontSize: "0.82rem",
          }}
        >
          Réessayer
        </button>

        {import.meta.env.DEV && this.state.info && (
          <details
            style={{
              maxWidth: 640,
              width: "100%",
              fontSize: "0.7rem",
              color: "var(--text-ghost, #404759)",
              background: "var(--surface-1, #10121a)",
              border: "1px solid var(--border-dim, #2a2d38)",
              borderRadius: 8,
              padding: "0.75rem 1rem",
            }}
          >
            <summary style={{ cursor: "pointer", marginBottom: "0.5rem" }}>
              Détail technique
            </summary>
            <pre style={{ overflowX: "auto", margin: 0, whiteSpace: "pre-wrap" }}>
              {this.state.info.componentStack}
            </pre>
          </details>
        )}
      </div>
    );
  }
}
