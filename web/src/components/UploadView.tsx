import { useRef, useState } from "react";

import { postCsv } from "../api";
import { PREDICTION_LABEL, fmtInt, fmtPct, toneOf } from "../format";
import type { CsvResponse } from "../types";
import { ScoreBar } from "./SignalTable";

/** Score an OHLCV file the model has never seen.
 *
 *  Features are rebuilt by the same etl.py code the training panel used, so the
 *  numbers here are directly comparable to the screener's -- with one caveat
 *  worth stating on screen: the cross-sectional features are computed over the
 *  tickers *in the upload*, so a one-fund file has no real peer group.
 */
export default function UploadView({ model }: { model: string }) {
  const [file, setFile] = useState<File | null>(null);
  const [latestOnly, setLatestOnly] = useState(true);
  const [result, setResult] = useState<CsvResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [over, setOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const submit = async (f: File) => {
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      setResult(await postCsv(f, model, latestOnly));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const take = (f: File | null | undefined) => {
    if (!f) return;
    setFile(f);
    void submit(f);
  };

  return (
    <div className="panel" style={{ padding: 16 }}>
      <div
        className="dropzone"
        data-over={over}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setOver(true); }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => { e.preventDefault(); setOver(false); take(e.dataTransfer.files?.[0]); }}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => e.key === "Enter" && inputRef.current?.click()}
      >
        <strong>{file ? file.name : "Drop an OHLCV CSV, or click to choose"}</strong>
        <div className="hint">
          Columns: <code>date, ticker, high, low, close, volume</code> — optional{" "}
          <code>category</code>. <code>symbol</code> and <code>adj close</code> also work.
        </div>
        <input
          ref={inputRef}
          type="file"
          accept=".csv,text/csv"
          hidden
          onChange={(e) => take(e.target.files?.[0])}
        />
      </div>

      <label className="check" style={{ marginTop: 12, maxWidth: 320 }}>
        <input
          type="checkbox"
          checked={latestOnly}
          onChange={(e) => {
            setLatestOnly(e.target.checked);
            if (file) void submit(file);
          }}
        />
        <span className="lbl">Score only the latest date per ticker</span>
      </label>

      {busy && <div className="notice" style={{ marginTop: 12 }}>Building features and scoring…</div>}
      {error && <div className="notice error" style={{ marginTop: 12 }}>{error}</div>}

      {result && (
        <>
          <div className="tiles" style={{ marginTop: 16 }}>
            <div className="tile"><div className="k">Rows in</div><div className="v num">{fmtInt(result.n_input_rows)}</div></div>
            <div className="tile"><div className="k">Scored</div><div className="v num">{fmtInt(result.n_scored)}</div></div>
            <div className="tile">
              <div className="k">Skipped</div>
              <div className="v num">{fmtInt(result.n_skipped_insufficient_history)}</div>
              <div className="sub">too little history</div>
            </div>
          </div>

          <div className="table-wrap" style={{ marginTop: 4 }}>
            <table>
              <thead>
                <tr>
                  <th>Date</th><th>Ticker</th><th>Signal</th>
                  <th className="right">Confidence</th><th className="right">Score</th>
                </tr>
              </thead>
              <tbody>
                {result.predictions.map((p) => (
                  <tr key={`${p.ticker}-${p.date}`} style={{ cursor: "default" }}>
                    <td className="num">{p.date}</td>
                    <td className="ticker">{p.ticker}</td>
                    <td><span className={`pred ${toneOf(p.prediction)}`}>{PREDICTION_LABEL[p.prediction]}</span></td>
                    <td className="right num">{fmtPct(p.confidence, 1)}</td>
                    <td className="right"><ScoreBar value={p.score} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <p className="footnote">
            Peer features (category rank, breadth, category returns) are computed across the
            tickers in <em>this file</em>. Tickers the universe recognises use their real
            category as the peer group; the rest are pooled into one catch-all group. A
            single-ticker upload therefore has no meaningful peer group, and its
            cross-sectional features should be read with that in mind.
          </p>
        </>
      )}
    </div>
  );
}
