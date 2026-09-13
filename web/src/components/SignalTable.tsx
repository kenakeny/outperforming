import { PREDICTION_LABEL, fmtInt, fmtPct, fmtScore, toneOf } from "../format";
import type { Query, Signal, SignalsResponse } from "../types";

interface Props {
  data: SignalsResponse | null;
  loading: boolean;
  query: Query;
  onChange: (patch: Partial<Query>) => void;
  onPick: (ticker: string) => void;
}

const COLUMNS: { key: string; label: string; sortable: boolean; right?: boolean }[] = [
  { key: "ticker", label: "Ticker", sortable: true },
  { key: "name", label: "Fund", sortable: true },
  { key: "category", label: "Category", sortable: true },
  { key: "prediction", label: "Signal", sortable: false },
  { key: "confidence", label: "Confidence", sortable: true, right: true },
  { key: "score", label: "Score", sortable: true, right: true },
];

export default function SignalTable({ data, loading, query, onChange, onPick }: Props) {
  const rows = data?.signals ?? [];
  const total = data?.n_scored ?? 0;
  const from = query.page * query.pageSize;

  const sortBy = (key: string) => {
    if (!COLUMNS.find((c) => c.key === key)?.sortable) return;
    // Re-clicking the active column flips direction; a new column starts at the
    // direction that puts the interesting end first (descending for numbers,
    // ascending for text).
    const isActive = query.sort === key;
    const order = isActive
      ? query.order === "desc" ? "asc" : "desc"
      : key === "ticker" || key === "name" || key === "category" ? "asc" : "desc";
    onChange({ sort: key, order, page: 0 });
  };

  return (
    <div className="panel" style={{ opacity: loading && rows.length > 0 ? 0.55 : 1, transition: "opacity 0.12s" }}>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              {COLUMNS.map((col) => (
                <th
                  key={col.key}
                  className={`${col.sortable ? "sortable" : ""} ${col.right ? "right" : ""}`}
                  onClick={() => sortBy(col.key)}
                  aria-sort={
                    query.sort === col.key
                      ? query.order === "asc" ? "ascending" : "descending"
                      : undefined
                  }
                >
                  {col.label}
                  {query.sort === col.key && (
                    <span className="arrow" aria-hidden="true">{query.order === "asc" ? "▲" : "▼"}</span>
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => <Row key={row.ticker} row={row} onPick={onPick} />)}
          </tbody>
        </table>

        {rows.length === 0 && !loading && (
          <div className="empty">
            <strong>Nothing matches these filters</strong>
            Widen the categories, drop the confidence floor, or clear the search.
          </div>
        )}
        {rows.length === 0 && loading && (
          <div style={{ padding: 12 }}>
            {Array.from({ length: 8 }, (_, i) => (
              <div key={i} className="skeleton" style={{ height: 30, marginBottom: 5 }} />
            ))}
          </div>
        )}
      </div>

      <div className="pager">
        <span style={{ color: "var(--ink-muted)", fontSize: 12 }}>
          {total > 0
            ? `${fmtInt(from + 1)}–${fmtInt(Math.min(from + query.pageSize, total))} of ${fmtInt(total)}`
            : "No results"}
          {data && data.n_scored !== data.universe_scored && (
            <> · filtered from {fmtInt(data.universe_scored)} scored</>
          )}
        </span>

        <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, color: "var(--ink-muted)" }}>
          Rows
          <select
            value={query.pageSize}
            onChange={(e) => onChange({ pageSize: Number(e.target.value), page: 0 })}
            style={{ background: "var(--surface-sunken)", border: "1px solid var(--border)", borderRadius: 5, padding: "3px 5px" }}
          >
            {[25, 50, 100, 250].map((n) => <option key={n} value={n}>{n}</option>)}
          </select>
        </label>

        <span className="spacer" />
        <button className="btn" disabled={query.page === 0} onClick={() => onChange({ page: query.page - 1 })}>
          ← Prev
        </button>
        <button
          className="btn"
          disabled={from + query.pageSize >= total}
          onClick={() => onChange({ page: query.page + 1 })}
        >
          Next →
        </button>
      </div>
    </div>
  );
}

function Row({ row, onPick }: { row: Signal; onPick: (t: string) => void }) {
  return (
    <tr onClick={() => onPick(row.ticker)} tabIndex={0}
        onKeyDown={(e) => e.key === "Enter" && onPick(row.ticker)}>
      <td className="ticker">
        {row.ticker}
        {row.is_leveraged && <span className="lev" title="Leveraged or inverse product">LEV</span>}
      </td>
      <td className="name" title={row.name ?? ""}>{row.name ?? "—"}</td>
      <td className="meta" title={row.category_group ?? ""}>{row.category ?? "—"}</td>
      <td>
        <span className={`pred ${toneOf(row.prediction)}`}>{PREDICTION_LABEL[row.prediction]}</span>
      </td>
      <td className="right num">
        <span className="conf-bar" aria-hidden="true">
          <i style={{ width: `${Math.round(row.confidence * 100)}%` }} />
        </span>
        {fmtPct(row.confidence)}
      </td>
      <td className="right">
        <ScoreBar value={row.score} />
      </td>
    </tr>
  );
}

/** Diverging bar around a zero centreline. The numeric score sits beside it in
 *  every row, so the colour is a scanning aid and never the only encoding.
 *
 *  `barScale` stretches the *bar only*, for contexts whose values occupy a much
 *  narrower range than a single fund's score (group means cluster near zero and
 *  would otherwise render as invisible slivers). The printed number is always
 *  the true value -- scaling what the reader can measure against the axis is
 *  fine, scaling the number they read off it is a lie.
 */
export function ScoreBar({ value, barScale = 1 }: { value: number; barScale?: number }) {
  const magnitude = Math.min(Math.abs(value * barScale), 1) * 50;
  return (
    <span className="scorebar">
      <span className="track">
        <span className="zero" />
        <span
          className={`fill ${value >= 0 ? "up" : "down"}`}
          style={{ width: `${magnitude}%` }}
        />
      </span>
      <span className="n">{fmtScore(value)}</span>
    </span>
  );
}
