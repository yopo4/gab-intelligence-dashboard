import { useState, useEffect, useRef, useCallback } from "react";

/**
 * Generic data-fetching hook with:
 * - Automatic abort on dep change (no stale data)
 * - Distinct loading / error states
 * - Optional transform function
 */
export function useApi(url, { transform, enabled = true } = {}) {
  const [state, setState] = useState({
    data: null,
    loading: true,
    error: null,
  });
  const abortRef = useRef(null);

  useEffect(() => {
    if (!enabled || !url) {
      setState((s) => ({ ...s, loading: false }));
      return;
    }

    abortRef.current?.abort();
    abortRef.current = new AbortController();

    setState({ data: null, loading: true, error: null });

    fetch(url, { signal: abortRef.current.signal })
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((raw) => {
        const data = transform ? transform(raw) : raw;
        setState({ data, loading: false, error: null });
      })
      .catch((err) => {
        if (err.name === "AbortError") return;
        setState({ data: null, loading: false, error: err.message });
      });

    return () => abortRef.current?.abort();
  }, [url, enabled]);

  return state;
}

/**
 * POST variant — re-fires when `body` reference changes
 */
export function usePost(url, body) {
  const [state, setState] = useState({
    data: null,
    loading: true,
    error: null,
  });
  const abortRef = useRef(null);
  const bodyStr = JSON.stringify(body);

  useEffect(() => {
    abortRef.current?.abort();
    abortRef.current = new AbortController();
    setState({ data: null, loading: true, error: null });

    fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: bodyStr,
      signal: abortRef.current.signal,
    })
      .then((r) => r.json())
      .then((data) => setState({ data, loading: false, error: null }))
      .catch((err) => {
        if (err.name === "AbortError") return;
        setState({ data: null, loading: false, error: err.message });
      });

    return () => abortRef.current?.abort();
  }, [url, bodyStr]);

  return state;
}

/**
 * Polling variant — refetches every `interval` ms
 */
export function useApiPoll(url, interval = 30000) {
  const [state, setState] = useState({
    data: null,
    loading: true,
    error: null,
  });

  const fetchData = useCallback(() => {
    fetch(url)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((data) => setState({ data, loading: false, error: null }))
      .catch((err) =>
        setState((s) => ({ ...s, loading: false, error: err.message })),
      );
  }, [url]);

  useEffect(() => {
    fetchData();
    const id = setInterval(fetchData, interval);
    return () => clearInterval(id);
  }, [fetchData, interval]);

  return { ...state, refresh: fetchData };
}

/** Build ?villes=…&types=…&annees=… query string from filter object */
export function buildFilterQuery(filters) {
  const p = new URLSearchParams();
  (filters.villes || []).forEach((v) => p.append("villes", v));
  (filters.types || []).forEach((t) => p.append("types", t));
  (filters.annees || []).forEach((a) => p.append("annees", a));
  return p.toString();
}
