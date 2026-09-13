/** Thin typed client over serve/api.py.
 *
 *  Two things live here that don't belong in components: query-string building
 *  (the backend takes repeated params for multi-value facets, which
 *  URLSearchParams does but object spreads don't), and request cancellation --
 *  the search box and the screener both fire on every keystroke, and a slow
 *  response arriving after a fast one would otherwise overwrite it.
 */
import type {
  CategoriesResponse, CsvResponse, DatesResponse, FacetField, FacetsResponse,
  HealthResponse, Query, SaudiArm, SaudiBenchmarkResponse, SaudiMeta,
  SaudiSignalsResponse, SaudiTickerResponse, SearchHit, SignalsResponse,
  TickerResponse,
} from "./types";

/** Empty in dev (Vite proxies) and in production (same origin as the API). */
const BASE = "";

export class ApiError extends Error {
  constructor(readonly status: number, message: string) {
    super(message);
  }
}

type Param = string | number | boolean | null | undefined;

function qs(params: Record<string, Param | Param[]>): string {
  const out = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === null || value === undefined || value === "") continue;
    // A repeated key is how FastAPI receives list[str] -- ?category=A&category=B.
    if (Array.isArray(value)) value.forEach((v) => v != null && out.append(key, String(v)));
    else out.append(key, String(value));
  }
  const s = out.toString();
  return s ? `?${s}` : "";
}

async function get<T>(path: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { signal });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* a non-JSON error body is still worth surfacing as the status text */
    }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

export const getHealth = (s?: AbortSignal) => get<HealthResponse>("/health", s);
export const getFacets = (s?: AbortSignal) => get<FacetsResponse>("/facets", s);
export const getDates = (s?: AbortSignal) => get<DatesResponse>("/dates?limit=1200", s);

export const getSearch = (q: string, model: string, date: string | null, s?: AbortSignal) =>
  get<{ results: SearchHit[] }>(`/search${qs({ q, model, date, limit: 12 })}`, s);

export function getSignals(query: Query, s?: AbortSignal) {
  const path = `/signals${qs({
    model: query.model,
    date: query.date,
    q: query.q || null,
    category: query.filters.category,
    category_group: query.filters.category_group,
    family: query.filters.family,
    exchange: query.filters.exchange,
    prediction: query.prediction,
    exclude_leveraged: query.excludeLeveraged || null,
    min_confidence: query.minConfidence || null,
    sort: query.sort,
    order: query.order,
    top_n: query.pageSize,
    offset: query.page * query.pageSize,
  })}`;
  return get<SignalsResponse>(path, s);
}

export const getTicker = (ticker: string, model: string, days = 180, s?: AbortSignal) =>
  get<TickerResponse>(`/tickers/${encodeURIComponent(ticker)}${qs({ model, days })}`, s);

export const getCategories = (
  field: FacetField, model: string, date: string | null, s?: AbortSignal,
) => get<CategoriesResponse>(`/categories${qs({ field, model, date, min_funds: 3 })}`, s);

// --------------------------------------------------------------------- //
//  Saudi (Tadawul) transfer-learning record                              //
// --------------------------------------------------------------------- //

export const getSaudiMeta = (s?: AbortSignal) => get<SaudiMeta>("/saudi/meta", s);

export const getSaudiSignals = (arm: SaudiArm, date: string | null, s?: AbortSignal) =>
  get<SaudiSignalsResponse>(`/saudi/signals${qs({ arm, date })}`, s);

export const getSaudiTicker = (ticker: string, arm: SaudiArm, days = 180, s?: AbortSignal) =>
  get<SaudiTickerResponse>(`/saudi/tickers/${encodeURIComponent(ticker)}${qs({ arm, days })}`, s);

export const getSaudiBenchmark = (arm: SaudiArm, s?: AbortSignal) =>
  get<SaudiBenchmarkResponse>(`/saudi/benchmark${qs({ arm })}`, s);

export async function postCsv(
  file: File, model: string, latestOnly: boolean, s?: AbortSignal,
): Promise<CsvResponse> {
  const body = new FormData();
  body.append("file", file);
  const res = await fetch(
    `${BASE}/predict/csv${qs({ model, latest_only: latestOnly, top_n: 200 })}`,
    { method: "POST", body, signal: s },
  );
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch { /* keep the status text */ }
    throw new ApiError(res.status, detail);
  }
  return res.json();
}
