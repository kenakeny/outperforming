import { useCallback, useEffect, useRef, useState } from "react";

/** Value that trails `value` by `delay` ms of quiet.
 *  Search and the screener both refetch on it, so typing costs one request at
 *  the end of a word rather than one per character. */
export function useDebounced<T>(value: T, delay = 220): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const id = setTimeout(() => setDebounced(value), delay);
    return () => clearTimeout(id);
  }, [value, delay]);
  return debounced;
}

export interface AsyncState<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
}

/** Run an aborting fetch whenever `deps` change.
 *
 *  The abort matters for correctness, not just politeness: without it a slow
 *  response for "gol" can land after the fast one for "gold" and repaint the
 *  table with results for a query the user has already moved past.
 *
 *  `data` is deliberately kept during a refetch so the table doesn't blank out
 *  between keystrokes -- `loading` is what dims it.
 */
export function useAsync<T>(
  fn: (signal: AbortSignal) => Promise<T>,
  deps: unknown[],
  enabled = true,
): AsyncState<T> & { reload: () => void } {
  const [state, setState] = useState<AsyncState<T>>({ data: null, error: null, loading: enabled });
  const [nonce, setNonce] = useState(0);
  const fnRef = useRef(fn);
  fnRef.current = fn;

  useEffect(() => {
    if (!enabled) {
      setState((s) => ({ ...s, loading: false }));
      return;
    }
    const controller = new AbortController();
    setState((s) => ({ ...s, loading: true, error: null }));
    fnRef.current(controller.signal)
      .then((data) => {
        if (!controller.signal.aborted) setState({ data, error: null, loading: false });
      })
      .catch((err: unknown) => {
        if (controller.signal.aborted || (err as Error)?.name === "AbortError") return;
        setState({ data: null, error: (err as Error).message ?? "request failed", loading: false });
      });
    return () => controller.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, enabled, nonce]);

  return { ...state, reload: useCallback(() => setNonce((n) => n + 1), []) };
}

export type Theme = "light" | "dark" | "system";

/** Theme stamped on <html> so CSS can resolve it, persisted across reloads.
 *  "system" removes the stamp entirely and lets the media query decide. */
export function useTheme(): [Theme, (t: Theme) => void] {
  const [theme, setTheme] = useState<Theme>(
    () => (localStorage.getItem("theme") as Theme) || "system",
  );
  useEffect(() => {
    const root = document.documentElement;
    if (theme === "system") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", theme);
    localStorage.setItem("theme", theme);
  }, [theme]);
  return [theme, setTheme];
}

/** Call `handler` on Escape. Used by the drawer and the search dropdown. */
export function useEscape(handler: () => void, active = true) {
  useEffect(() => {
    if (!active) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && handler();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [handler, active]);
}
