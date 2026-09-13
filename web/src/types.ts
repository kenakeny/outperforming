/** Wire types for serve/api.py. Kept hand-written and narrow: only the fields
 *  the UI actually reads, so a backend change that matters shows up as a type
 *  error rather than as a blank cell. */

export type PredictionClass = "outperform" | "neutral" | "underperform";

/** One fund's identity. Every lookup surface returns at least this. */
export interface FundMeta {
  ticker: string;
  name: string | null;
  category_group: string | null;
  category: string | null;
  family: string | null;
  exchange: string | null;
  is_leveraged: boolean | null;
}

/** A scored row in the signal table. */
export interface Signal extends FundMeta {
  score: number;
  confidence: number;
  prediction: PredictionClass;
  outperform: number;
  neutral: number;
  underperform: number;
}

/** A search hit: identity, plus the score when the fund is in the scored panel. */
export interface SearchHit extends FundMeta {
  scored?: boolean;
  score?: number;
  confidence?: number;
  prediction?: PredictionClass;
}

export interface Summary {
  n: number;
  mean_score: number | null;
  mean_confidence: number | null;
  outperform: number;
  neutral: number;
  underperform: number;
}

export interface SignalsResponse {
  date: string;
  model: string;
  n_scored: number;
  universe_scored: number;
  offset: number;
  limit: number;
  sort: string;
  order: "asc" | "desc";
  summary: Summary;
  ignored_filters: string[];
  signals: Signal[];
}

export interface FacetValue {
  value: string;
  count: number;
}

export interface FacetsResponse {
  fields: FacetField[];
  facets: Record<FacetField, FacetValue[]>;
  predictions: PredictionClass[];
}

export type FacetField = "category_group" | "category" | "family" | "exchange";

export interface DatesResponse {
  latest: string;
  earliest: string;
  n_total: number;
  dates: string[];
}

export interface HealthResponse {
  status: "ok" | "degraded";
  models_available: string[];
  panel_rows: number;
  latest_date: string | null;
}

export interface HistoryPoint {
  date: string;
  score: number;
  outperform: number;
  neutral: number;
  underperform: number;
  confidence: number;
  prediction: PredictionClass;
  realized_target?: number | null;
}

export interface PricePoint {
  date: string;
  close: number;
}

export interface TickerResponse {
  ticker: string;
  model: string;
  meta: FundMeta | null;
  latest: HistoryPoint | null;
  n_points: number;
  history: HistoryPoint[];
  prices: PricePoint[];
}

export interface CategoryGroupRow {
  [field: string]: string | number;
  n_funds: number;
  mean_score: number;
  mean_confidence: number;
  n_outperform: number;
  n_underperform: number;
  top_ticker: string;
}

export interface CategoriesResponse {
  date: string;
  model: string;
  field: FacetField;
  n_groups: number;
  groups: CategoryGroupRow[];
}

/** A scored row from an uploaded CSV. Same shape as a Signal minus the
 *  metadata join, plus the date it was scored on. */
export interface CsvPrediction {
  date: string;
  ticker: string;
  score: number;
  confidence: number;
  prediction: PredictionClass;
  outperform: number;
  neutral: number;
  underperform: number;
}

export interface CsvResponse {
  model: string;
  n_input_rows: number;
  n_scored: number;
  n_skipped_insufficient_history: number;
  predictions: CsvPrediction[];
}

// --------------------------------------------------------------------- //
//  Saudi (Tadawul) transfer-learning record -- see serve/saudi.py.       //
//  Every response here describes a walk-forward EVALUATION, not a live   //
//  model: `is_realized_history` is always true, and each row's outcome   //
//  (`fwd_ret` / `realized_class`) already happened. Kept as distinct      //
//  types rather than reusing Signal/SignalsResponse so that distinction  //
//  can't quietly get erased by a shared shape.                           //
// --------------------------------------------------------------------- //

export type SaudiArm = "saudi_only" | "zeroshot" | "zeroshot_pure" | "finetune";

export interface SaudiFund {
  ticker: string;
  name: string;
}

export interface SaudiMeta {
  arms: SaudiArm[];
  horizon_days: number;
  funds: SaudiFund[];
  earliest: string;
  latest: string;
  dates: string[];
  is_realized_history: true;
}

export interface SaudiSignal {
  ticker: string;
  name: string;
  score: number;
  prediction: PredictionClass;
  realized_class: PredictionClass;
  fwd_ret: number;
}

export interface SaudiSignalsResponse {
  arm: SaudiArm;
  date: string;
  n: number;
  is_realized_history: true;
  signals: SaudiSignal[];
}

export interface SaudiHistoryPoint {
  date: string;
  score: number;
  prediction: PredictionClass;
  realized_class: PredictionClass;
  fwd_ret: number;
}

export interface SaudiTickerResponse {
  ticker: string;
  name: string;
  arm: SaudiArm;
  is_realized_history: true;
  history: SaudiHistoryPoint[];
}

export interface SaudiSeriesStats {
  ann_return: number;
  ir: number | null;
  hit_rate: number | null;
  n: number;
}

export interface SaudiBenchmarkResponse {
  arm: SaudiArm;
  horizon_days: number;
  is_realized_history: true;
  series: Record<string, { date: string; value: number }[]>;
  stats: Record<string, SaudiSeriesStats>;
}

/** Everything the screener can ask the backend for, in one object. */
export interface Query {
  model: string;
  date: string | null;
  q: string;
  filters: Record<FacetField, string[]>;
  prediction: PredictionClass[];
  excludeLeveraged: boolean;
  minConfidence: number;
  sort: string;
  order: "asc" | "desc";
  page: number;
  pageSize: number;
}
