import { useState } from "react";

import { getSaudiTicker } from "../api";
import { PREDICTION_LABEL, fmtDate, fmtDateShort, fmtPct, fmtScore, toneOf } from "../format";
import { useAsync, useEscape } from "../hooks";
import type { SaudiArm } from "../types";
import LineChart from "./LineChart";

interface Props {
  ticker: string;
  arm: SaudiArm;
  onClose: () => void;
}

/** One Tadawul fund's walk-forward record. Deliberately thinner than the US
 *  TickerDrawer: no category/family tags (Tadawul funds aren't in
 *  financedatabase), no price series (not served for this market), and every
 *  number here is historical -- the banner says so, because a drawer that
 *  looks like the US one otherwise invites reading it as a live call.
 */
export default function SaudiTickerDrawer({ ticker, arm, onClose }: Props) {
  const [showData, setShowData] = useState(false);
  const { data, error, loading } = useAsync(
    (signal) => getSaudiTicker(ticker, arm, 180, signal),
    [ticker, arm],
  );

  useEscape(onClose);

  const history = data?.history ?? [];
  const latest = history[history.length - 1] ?? null;
  const scorePoints = history.map((h) => ({ date: h.date, value: h.score }));
  const hit = latest ? latest.prediction === latest.realized_class : null;

  return (
    <>
      <div className="scrim" onClick={onClose} />
      <div className="drawer" role="dialog" aria-modal="true" aria-label={`${ticker} detail`}>
        <div className="drawer-head">
          <div style={{ flex: 1, minWidth: 0 }}>
            <h2>{ticker}</h2>
            <div className="sub">{data?.name ?? (loading ? "Loading…" : "—")}</div>
            <div className="tags">
              <span className="tag-pill">Tadawul</span>
              <span className="tag-pill">{arm.replace(/_/g, " ")}</span>
            </div>
          </div>
          <button className="icon-btn" onClick={onClose} aria-label="Close">✕</button>
        </div>

        <div className="drawer-body">
          <div className="notice" style={{ marginBottom: 16 }}>
            Walk-forward evaluation record — every row&apos;s outcome already happened when this
            was written. Not a live signal.
          </div>

          {error && <div className="notice error">{error}</div>}
          {loading && !data && <div className="skeleton" style={{ height: 148 }} />}

          {latest && (
            <>
              <div className="tiles" style={{ marginBottom: 4 }}>
                <div className="tile">
                  <div className="k">Score</div>
                  <div className={`v num ${toneOf(latest.prediction)}`}>{fmtScore(latest.score)}</div>
                  <div className="sub">{fmtDate(latest.date)}</div>
                </div>
                <div className="tile">
                  <div className="k">Predicted</div>
                  <div className="v" style={{ fontSize: 15 }}>
                    <span className={`pred ${toneOf(latest.prediction)}`}>{PREDICTION_LABEL[latest.prediction]}</span>
                  </div>
                </div>
                <div className="tile">
                  <div className="k">Realized</div>
                  <div className="v" style={{ fontSize: 15 }}>
                    <span className={`pred ${toneOf(latest.realized_class)}`}>{PREDICTION_LABEL[latest.realized_class]}</span>
                  </div>
                  <div className="sub">{hit ? "✓ prediction matched" : "prediction missed"}</div>
                </div>
                <div className="tile">
                  <div className="k">Fwd return</div>
                  <div className={`v num ${latest.fwd_ret >= 0 ? "up" : "down"}`}>{fmtPct(latest.fwd_ret, 2)}</div>
                  <div className="sub">20-day, already realized</div>
                </div>
              </div>
            </>
          )}

          {scorePoints.length > 0 && (
            <>
              <div className="section-title">Score history · last {scorePoints.length} evaluated days</div>
              <LineChart
                points={scorePoints}
                label="Score history"
                zeroBaseline
                format={(v) => v.toFixed(2)}
                formatDate={fmtDateShort}
              />
            </>
          )}

          {history.length > 0 && (
            <>
              <button className="link-btn" style={{ marginTop: 16, paddingLeft: 0 }} onClick={() => setShowData((s) => !s)}>
                {showData ? "Hide" : "Show"} the numbers
              </button>
              {showData && (
                <div className="table-wrap" style={{ maxHeight: 280, overflowY: "auto", marginTop: 8 }}>
                  <table>
                    <thead>
                      <tr>
                        <th>Date</th><th className="right">Score</th>
                        <th>Predicted</th><th>Realized</th><th className="right">Fwd return</th>
                      </tr>
                    </thead>
                    <tbody>
                      {[...history].reverse().map((h) => (
                        <tr key={h.date} style={{ cursor: "default" }}>
                          <td className="num">{h.date}</td>
                          <td className="right num">{fmtScore(h.score)}</td>
                          <td><span className={`pred ${toneOf(h.prediction)}`}>{PREDICTION_LABEL[h.prediction]}</span></td>
                          <td><span className={`pred ${toneOf(h.realized_class)}`}>{PREDICTION_LABEL[h.realized_class]}</span></td>
                          <td className={`right num ${h.fwd_ret >= 0 ? "up" : "down"}`}>{fmtPct(h.fwd_ret, 2)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </>
  );
}
