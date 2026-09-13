import { useMemo } from "react";

import { fmtDate } from "../format";
import type { DatesResponse } from "../types";

interface Props {
  value: string | null;          // null = "latest"
  resolved: string | null;       // the date the backend actually scored
  dates: DatesResponse | null;
  onChange: (date: string | null) => void;
}

/** Trading-date picker.
 *
 *  A native date input rather than a <select>: the panel holds ~2,500 dates and
 *  rendering them all as options is unusable with a mouse and worse with a
 *  screen reader. The catch is that a calendar happily offers weekends and
 *  holidays, which the panel has no rows for -- so any pick is snapped back to
 *  the most recent trading day at or before it, and the arrows step by trading
 *  day rather than by calendar day.
 */
export default function DatePicker({ value, resolved, dates, onChange }: Props) {
  const list = dates?.dates ?? [];          // newest first, from the API
  const active = value ?? resolved ?? "";

  const index = useMemo(() => new Map(list.map((d, i) => [d, i])), [list]);

  const snap = (picked: string) => {
    if (!picked) return onChange(null);
    if (index.has(picked)) return onChange(picked);
    // `list` is newest-first, so the first entry <= the pick is the one just
    // before a weekend or holiday.
    const previous = list.find((d) => d <= picked);
    onChange(previous ?? list[list.length - 1] ?? null);
  };

  const step = (delta: number) => {
    const at = index.get(active);
    if (at == null) return;
    const next = list[at + delta];
    if (next) onChange(next);
  };

  const at = index.get(active);
  const isLatest = value === null || active === dates?.latest;

  return (
    <div className="control datepicker">
      <label htmlFor="date">Date</label>
      <button
        className="step"
        onClick={() => step(1)}
        disabled={at == null || at >= list.length - 1}
        aria-label="Previous trading day"
        title="Previous trading day"
      >
        ‹
      </button>
      <input
        id="date"
        type="date"
        value={active}
        min={dates?.earliest}
        max={dates?.latest}
        aria-label="Scoring date"
        onChange={(e) => snap(e.target.value)}
      />
      <button
        className="step"
        onClick={() => step(-1)}
        disabled={at == null || at <= 0}
        aria-label="Next trading day"
        title="Next trading day"
      >
        ›
      </button>
      <button
        className="link-btn"
        onClick={() => onChange(null)}
        disabled={isLatest}
        title={dates ? `Jump to ${fmtDate(dates.latest)}` : undefined}
      >
        Latest
      </button>
    </div>
  );
}
