import { useState, useEffect, useRef, useCallback } from "react";

/**
 * Generic data-fetching hook with:
 * - Automatic abort on dep change (no stale data)
 * - Distinct loading / error states
 * - Optional transform function
 * - refetch() to manually re-trigger without page reload
 */
export function useApi(url, { transform, enabled = true } = {}) {
  const [state, setState] = useState({
    data: null,
    loading: true,
    error: null,
  });
  const abortRef  = useRef(null);
  const counterRef = useRef(0); // bumped by refetch() to re-run the effect

  const [tick, setTick] = useState(0);
  const refetch = useCallback(() => setTick((t) => t + 1), []);

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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url, enabled, tick]);

  return { ...state, refetch };
}

/**
 * POST variant — re-fires when `body` content changes (stable JSON comparison)
 */
export function usePost(url, body) {
  const [state, setState] = useState({
    data: null,
    loading: true,
    error: null,
  });
  const abortRef = useRef(null);
  // Stable string comparison avoids re-firing on same-content object references
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
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
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
 * Polling variant — refetches every `interval` ms with AbortController cleanup
 */
export function useApiPoll(url, interval = 30000) {
  const [state, setState] = useState({
    data: null,
    loading: true,
    error: null,
  });
  const abortRef = useRef(null);

  const fetchData = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = new AbortController();

    fetch(url, { signal: abortRef.current.signal })
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((data) => setState({ data, loading: false, error: null }))
      .catch((err) => {
        if (err.name === "AbortError") return;
        setState((s) => ({ ...s, loading: false, error: err.message }));
      });
  }, [url]);

  useEffect(() => {
    fetchData();
    const id = setInterval(fetchData, interval);
    return () => {
      clearInterval(id);
      abortRef.current?.abort();
    };
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
