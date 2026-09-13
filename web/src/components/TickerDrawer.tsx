import { useState } from "react";

import { getTicker } from "../api";
import { PREDICTION_LABEL, fmtDate, fmtDateShort, fmtPct, fmtScore, toneOf } from "../format";
import { useAsync, useEscape } from "../hooks";
import type { PredictionClass } from "../types";
import LineChart from "./LineChart";

interface Props {
  ticker: string;
  model: string;
  onClose: () => void;
}

const CLASSES: PredictionClass[] = ["outperform", "neutral", "underperform"];
const BAR_COLOR: Record<PredictionClass, string> = {
  outperform: "var(--up)",
  neutral: "var(--flat)",
  underperform: "var(--down)",
};

/** Everything about one fund: what it is, what the model says today, and how
 *  that has moved. Score and price are two charts stacked on a shared x-range
 *  rather than one chart with two y-axes -- they have nothing in common but
 *  their dates, and a second axis would invent a relationship between them. */
export default function TickerDrawer({ ticker, model, onClose }: Props) {
  const [showData, setShowData] = useState(false);
  const { data, error, loading } = useAsync(
    (signal) => getTicker(ticker, model, 180, signal),
    [ticker, model],
  );

  useEscape(onClose);

  const latest = data?.latest ?? null;
  const scorePoints = (data?.history ?? []).map((h) => ({ date: h.date, value: h.score }));
  const pricePoints = (data?.prices ?? []).map((p) => ({ date: p.date, value: p.close }));

  return (
    <>
      <div className="scrim" onClick={onClose} />
      <div className="drawer" role="dialog" aria-modal="true" aria-label={`${ticker} detail`}>
        <div className="drawer-head">
          <div style={{ flex: 1, minWidth: 0 }}>
            <h2>{ticker}</h2>
            <div className="sub">{data?.meta?.name ?? (loading ? "Loading…" : "—")}</div>
            {data?.meta && (
              <div className="tags">
                {[data.meta.category_group, data.meta.category, data.meta.family, data.meta.exchange]
                  .filter(Boolean)
                  .map((t) => <span className="tag-pill" key={t as string}>{t}</span>)}
                {data.meta.is_leveraged && <span className="tag-pill">Leveraged / inverse</span>}
              </div>
            )}
          </div>
          <button className="icon-btn" onClick={onClose} aria-label="Close">✕</button>
        </div>

        <div className="drawer-body">
          {error && <div className="notice error">{error}</div>}

          {loading && !data && (
            <>
              <div className="skeleton" style={{ height: 84, marginBottom: 12 }} />
              <div className="skeleton" style={{ height: 148 }} />
            </>
          )}

          {latest && (
            <>
              <div className="tiles" style={{ marginBottom: 4 }}>
                <div className="tile">
                  <div className="k">Score</div>
                  <div className={`v num ${toneOf(latest.prediction)}`}>{fmtScore(latest.score)}</div>
                  <div className="sub">{fmtDate(latest.date)}</div>
                </div>
                <div className="tile">
                  <div className="k">Signal</div>
                  <div className="v" style={{ fontSize: 17 }}>
                    <span className={`pred ${toneOf(latest.prediction)}`}>
                      {PREDICTION_LABEL[latest.prediction]}
                    </span>
                  </div>
                  <div className="sub">{fmtPct(latest.confidence, 1)} confidence</div>
                </div>
              </div>

              <div className="section-title">Class probabilities</div>
              <div className="proba">
                {CLASSES.map((c) => (
                  <div className="proba-row" key={c}>
                    <span>{PREDICTION_LABEL[c]}</span>
                    <span className="bar">
                      <i style={{ width: `${latest[c] * 100}%`, background: BAR_COLOR[c] }} />
                    </span>
                    <span className="pct">{fmtPct(latest[c], 1)}</span>
                  </div>
                ))}
              </div>
            </>
          )}

          {scorePoints.length > 0 && (
            <>
              <div className="section-title">
                Score history · last {scorePoints.length} sessions
              </div>
              <LineChart
                points={scorePoints}
                label="Score history"
                zeroBaseline
                format={(v) => v.toFixed(2)}
                formatDate={fmtDateShort}
              />
              <p className="footnote">
                Score is P(outperform) − P(underperform) against the fund&apos;s own category
                peers over the next 5 trading days. It ranks funds against each other; it is
                not a return forecast.
              </p>
            </>
          )}

          {pricePoints.length > 0 && (
            <>
              <div className="section-title">Close price · same window</div>
              <LineChart
                points={pricePoints}
                label="Close price"
                format={(v) => `$${v.toFixed(0)}`}
                formatDate={fmtDateShort}
              />
            </>
          )}

          {scorePoints.length > 0 && (
            <>
              <button
                className="link-btn"
                style={{ marginTop: 16, paddingLeft: 0 }}
                onClick={() => setShowData((s) => !s)}
              >
                {showData ? "Hide" : "Show"} the numbers
              </button>
              {showData && (
                <div className="table-wrap" style={{ maxHeight: 280, overflowY: "auto", marginTop: 8 }}>
                  <table>
                    <thead>
                      <tr>
                        <th>Date</th><th className="right">Score</th>
                        <th>Signal</th><th className="right">Confidence</th>
                      </tr>
                    </thead>
                    <tbody>
                      {[...data!.history].reverse().map((h) => (
                        <tr key={h.date} style={{ cursor: "default" }}>
                          <td className="num">{h.date}</td>
                          <td className="right num">{fmtScore(h.score)}</td>
                          <td><span className={`pred ${toneOf(h.prediction)}`}>{PREDICTION_LABEL[h.prediction]}</span></td>
                          <td className="right num">{fmtPct(h.confidence, 1)}</td>
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
