import { useState } from "react";

import { PREDICTION_LABEL, fmtPct, toneOf } from "../format";
import type { SaudiSignal } from "../types";
import { ScoreBar } from "./SignalTable";

interface Props {
  rows: SaudiSignal[];
  loading: boolean;
  onPick: (ticker: string) => void;
}

type SortKey = "ticker" | "name" | "score" | "fwd_ret";

/** The 12-fund Saudi table. No pagination, no facets -- there's no peer-group
 *  metadata for Tadawul funds the way there is for the US universe, and 12
 *  rows don't need a page size control. What earns its own column here is
 *  `realized_class`: unlike the US table, every row already knows its
 *  outcome, so "did the model call it right" is a real, displayable fact.
 */
export default function SaudiSignalTable({ rows, loading, onPick }: Props) {
  const [sort, setSort] = useState<SortKey>("score");
  const [order, setOrder] = useState<"asc" | "desc">("desc");

  const sortBy = (key: SortKey) => {
    const isActive = sort === key;
    setOrder(isActive ? (order === "desc" ? "asc" : "desc") : key === "ticker" || key === "name" ? "asc" : "desc");
    setSort(key);
  };

  const sorted = [...rows].sort((a, b) => {
    const dir = order === "asc" ? 1 : -1;
    if (sort === "ticker" || sort === "name") return a[sort].localeCompare(b[sort]) * dir;
    return (a[sort] - b[sort]) * dir;
  });

  const arrow = (key: SortKey) =>
    sort === key ? <span className="arrow" aria-hidden="true">{order === "asc" ? "▲" : "▼"}</span> : null;
  const ariaSort = (key: SortKey): "ascending" | "descending" | undefined =>
    sort === key ? (order === "asc" ? "ascending" : "descending") : undefined;

  return (
    <div className="panel" style={{ opacity: loading && rows.length > 0 ? 0.55 : 1, transition: "opacity 0.12s" }}>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th className="sortable" onClick={() => sortBy("ticker")} aria-sort={ariaSort("ticker")}>Ticker{arrow("ticker")}</th>
              <th className="sortable" onClick={() => sortBy("name")} aria-sort={ariaSort("name")}>Fund{arrow("name")}</th>
              <th>Predicted</th>
              <th>Realized outcome</th>
              <th className="right sortable" onClick={() => sortBy("fwd_ret")} aria-sort={ariaSort("fwd_ret")}>
                Fwd return{arrow("fwd_ret")}
              </th>
              <th className="right sortable" onClick={() => sortBy("score")} aria-sort={ariaSort("score")}>
                Score{arrow("score")}
              </th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((row) => {
              const hit = row.prediction === row.realized_class;
              return (
                <tr key={row.ticker} onClick={() => onPick(row.ticker)} tabIndex={0}
                    onKeyDown={(e) => e.key === "Enter" && onPick(row.ticker)}>
                  <td className="ticker">{row.ticker}</td>
                  <td className="name" title={row.name}>{row.name}</td>
                  <td><span className={`pred ${toneOf(row.prediction)}`}>{PREDICTION_LABEL[row.prediction]}</span></td>
                  <td>
                    <span className={`pred ${toneOf(row.realized_class)}`}>{PREDICTION_LABEL[row.realized_class]}</span>
                    <span
                      style={{ marginLeft: 6, fontSize: 11, color: hit ? "var(--up)" : "var(--ink-muted)" }}
                      title={hit ? "Predicted class matched the realized outcome" : "Predicted class missed the realized outcome"}
                    >
                      {hit ? "✓ hit" : "miss"}
                    </span>
                  </td>
                  <td className={`right num ${row.fwd_ret >= 0 ? "up" : "down"}`}>{fmtPct(row.fwd_ret, 2)}</td>
                  <td className="right"><ScoreBar value={row.score} /></td>
                </tr>
              );
            })}
          </tbody>
        </table>

        {rows.length === 0 && !loading && (
          <div className="empty"><strong>No predictions for this date.</strong></div>
        )}
        {rows.length === 0 && loading && (
          <div style={{ padding: 12 }}>
            {Array.from({ length: 6 }, (_, i) => (
              <div key={i} className="skeleton" style={{ height: 30, marginBottom: 5 }} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
