import { useState } from "react";

import { getSaudiBenchmark, getSaudiMeta, getSaudiSignals } from "./api";
import DatePicker from "./components/DatePicker";
import MultiSeriesChart from "./components/MultiSeriesChart";
import SaudiSignalTable from "./components/SaudiSignalTable";
import SaudiTickerDrawer from "./components/SaudiTickerDrawer";
import { fmtDate, fmtInt, fmtPct } from "./format";
import { useAsync } from "./hooks";
import type { SaudiArm } from "./types";

const ARM_LABEL: Record<SaudiArm, string> = {
  saudi_only: "Saudi-only",
  zeroshot: "Zero-shot (US net)",
  zeroshot_pure: "Zero-shot, pure",
  finetune: "Fine-tuned",
};

const ARM_HINT: Record<SaudiArm, string> = {
  saudi_only: "an LSTM trained from scratch on the 12 Tadawul funds alone",
  zeroshot: "the US-trained net, unmodified, scored with Saudi-fold feature scaling",
  zeroshot_pure: "the US-trained net, unmodified, with no Saudi data touching it at all — not even feature centering",
  finetune: "the US-trained net with its head retrained on Saudi data",
};

const SERIES_COLOR: Record<string, string> = {
  model: "var(--series-1)",
  TASI: "var(--series-2)",
  "FALCOM Saudi Equity": "var(--series-3)",
};
const SERIES_LABEL: Record<string, string> = { model: "Model portfolio" };

export default function SaudiMarketView() {
  const [arm, setArm] = useState<SaudiArm>("saudi_only");
  const [date, setDate] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);

  const meta = useAsync((s) => getSaudiMeta(s), []);
  const signals = useAsync((s) => getSaudiSignals(arm, date, s), [arm, date]);
  const bench = useAsync((s) => getSaudiBenchmark(arm, s), [arm]);

  const resolvedDate = signals.data?.date ?? meta.data?.latest ?? null;
  const datesForPicker = meta.data
    ? { latest: meta.data.latest, earliest: meta.data.earliest, n_total: meta.data.dates.length, dates: meta.data.dates }
    : null;

  const series = bench.data
    ? Object.entries(bench.data.series).map(([key, points]) => ({
        key, points,
        label: SERIES_LABEL[key] ?? key,
        color: SERIES_COLOR[key] ?? "var(--series-1)",
      }))
    : [];

  return (
    <>
      <div className="notice" style={{ marginBottom: 14 }}>
        <strong style={{ display: "block", marginBottom: 2 }}>Walk-forward evaluation record</strong>
        {meta.data
          ? `${fmtInt(meta.data.funds.length)} Tadawul-listed funds, ${fmtDate(meta.data.earliest)} – ${fmtDate(meta.data.latest)}. `
          : ""}
        No live model runs here (see the README) — this is the tested transfer-learning
        record from <code>saudi.py</code>, so every row&apos;s outcome already happened.
      </div>

      <div className="tabs" role="tablist" style={{ borderBottom: "none", marginBottom: 10, flexWrap: "wrap", gap: 6 }}>
        {meta.data?.arms.map((a) => (
          <button
            key={a}
            className="chip"
            aria-pressed={arm === a}
            title={ARM_HINT[a]}
            onClick={() => setArm(a)}
          >
            {ARM_LABEL[a]}
          </button>
        ))}
        <span style={{ marginLeft: "auto" }}>
          <DatePicker value={date} resolved={resolvedDate} dates={datesForPicker} onChange={setDate} />
        </span>
      </div>

      {bench.data && (
        <div className="tiles">
          {Object.entries(bench.data.stats).map(([name, st]) => (
            <div className="tile" key={name}>
              <div className="k">{SERIES_LABEL[name] ?? name}</div>
              <div className={`v num ${st.ann_return >= 0 ? "up" : "down"}`}>{fmtPct(st.ann_return)}</div>
              <div className="sub">
                ann. return · IR {st.ir != null ? st.ir.toFixed(2) : "—"} ·{" "}
                {st.hit_rate != null ? fmtPct(st.hit_rate, 0) : "—"} hit rate
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="panel" style={{ padding: 14, marginBottom: 16 }}>
        <div className="section-title" style={{ marginTop: 0 }}>
          Model vs. TASI &amp; FALCOM · growth of $1
        </div>
        {bench.error && <div className="notice error">{bench.error}</div>}
        {bench.loading && !bench.data && (
          <div className="notice">Fetching TASI / FALCOM benchmark data…</div>
        )}
        {bench.data && (
          <MultiSeriesChart
            series={series}
            format={(v) => `$${v.toFixed(2)}`}
            formatDate={fmtDate}
            height={220}
          />
        )}
      </div>

      {signals.error && <div className="notice error" style={{ marginBottom: 12 }}>{signals.error}</div>}

      <SaudiSignalTable
        rows={signals.data?.signals ?? []}
        loading={signals.loading}
        onPick={setSelected}
      />

      <p className="footnote">
        Score is a transfer-learning arm&apos;s P(outperform) − P(underperform) against the
        Tadawul universe, 20-trading-day horizon. Only 12 funds are covered — Tadawul funds
        aren&apos;t in the financedatabase universe the US side uses, so this list is fixed
        rather than searchable. &ldquo;Realized outcome&rdquo; and &ldquo;Fwd return&rdquo; are
        what already happened by the time this record was written; they are not a forecast.
      </p>

      {selected && (
        <SaudiTickerDrawer ticker={selected} arm={arm} onClose={() => setSelected(null)} />
      )}
    </>
  );
}
