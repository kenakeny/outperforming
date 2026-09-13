import { useEffect, useMemo, useRef, useState } from "react";

export interface Series {
  key: string;
  label: string;
  color: string;
  points: { date: string; value: number }[];
}

interface Props {
  series: Series[];
  height?: number;
  format: (v: number) => string;
  formatDate: (iso: string) => string;
}

const PAD = { top: 8, right: 10, bottom: 20, left: 46 };

/** Several series on one shared axis, indexed to a common base (equity growth
 *  curves) rather than plotted on two y-scales -- the dataviz rule this exists
 *  to honour is "one axis": model vs benchmark returns are the same unit, so
 *  they share a scale instead of inventing a second one.
 *
 *  A legend is mandatory here (>=2 series): the aqua series fails the light-
 *  surface 3:1 contrast floor in isolation, so the legend's text label is the
 *  relief the palette check requires, not decoration.
 */
export default function MultiSeriesChart({ series, height = 220, format, formatDate }: Props) {
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const [hover, setHover] = useState<number | null>(null);
  const [width, setWidth] = useState(640);
  const [hidden, setHidden] = useState<Set<string>>(new Set());

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const observer = new ResizeObserver(([entry]) => {
      const next = entry.contentRect.width;
      if (next > 0) setWidth((w) => (Math.abs(next - w) > 1 ? next : w));
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, [series.length > 0]);

  const visible = series.filter((s) => !hidden.has(s.key) && s.points.length > 0);
  const dates = visible[0]?.points.map((p) => p.date) ?? [];

  const geom = useMemo(() => {
    if (visible.length === 0 || dates.length === 0) return null;
    const values = visible.flatMap((s) => s.points.map((p) => p.value));
    let min = Math.min(...values);
    let max = Math.max(...values);
    if (min === max) { min -= 1; max += 1; }
    const pad = (max - min) * 0.08;
    min -= pad;
    max += pad;

    const innerW = Math.max(width - PAD.left - PAD.right, 10);
    const innerH = height - PAD.top - PAD.bottom;
    const n = dates.length;
    const x = (i: number) => PAD.left + (n === 1 ? innerW / 2 : (i / (n - 1)) * innerW);
    const y = (v: number) => PAD.top + innerH - ((v - min) / (max - min)) * innerH;

    const paths = visible.map((s) => ({
      key: s.key, color: s.color,
      d: s.points.map((p, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(2)},${y(p.value).toFixed(2)}`).join(" "),
    }));
    const ticks = [max - pad, (min + max) / 2, min + pad];
    return { x, y, paths, ticks, innerW };
  }, [visible, width, height, dates.length]);

  const toggle = (key: string) => setHidden((h) => {
    const next = new Set(h);
    next.has(key) ? next.delete(key) : next.add(key);
    return next;
  });

  return (
    <div>
      <div className="legend-row">
        {series.map((s) => (
          <button
            key={s.key}
            className="legend-item"
            aria-pressed={!hidden.has(s.key)}
            onClick={() => toggle(s.key)}
            title={hidden.has(s.key) ? `Show ${s.label}` : `Hide ${s.label}`}
          >
            <span className="swatch" style={{ background: s.color, opacity: hidden.has(s.key) ? 0.25 : 1 }} />
            <span style={{ opacity: hidden.has(s.key) ? 0.45 : 1 }}>{s.label}</span>
          </button>
        ))}
      </div>

      <div className="chart" ref={wrapRef}>
        {!geom && <div className="notice">No data for these series.</div>}
        {geom && (
          <>
            <svg
              viewBox={`0 0 ${width} ${height}`}
              role="img"
              aria-label={`${visible.map((s) => s.label).join(" vs ")}: ${dates.length} points from ${formatDate(dates[0])} to ${formatDate(dates[dates.length - 1])}`}
              onMouseMove={(e) => {
                const rect = e.currentTarget.getBoundingClientRect();
                const rel = ((e.clientX - rect.left) / rect.width) * width - PAD.left;
                const i = Math.round((rel / geom.innerW) * (dates.length - 1));
                setHover(Math.min(dates.length - 1, Math.max(0, i)));
              }}
              onMouseLeave={() => setHover(null)}
            >
              {geom.ticks.map((t, i) => (
                <g key={i}>
                  <line className="gridline" x1={PAD.left} x2={width - PAD.right} y1={geom.y(t)} y2={geom.y(t)} />
                  <text className="axis-label" x={PAD.left - 6} y={geom.y(t) + 3} textAnchor="end">{format(t)}</text>
                </g>
              ))}

              {geom.paths.map((p) => <path key={p.key} className="series" d={p.d} stroke={p.color} />)}

              <text className="axis-label" x={PAD.left} y={height - 5}>{formatDate(dates[0])}</text>
              <text className="axis-label" x={width - PAD.right} y={height - 5} textAnchor="end">
                {formatDate(dates[dates.length - 1])}
              </text>

              {hover != null && (
                <line className="crosshair" x1={geom.x(hover)} x2={geom.x(hover)} y1={PAD.top} y2={height - PAD.bottom} />
              )}
              {hover != null && visible.map((s) => (
                <circle key={s.key} className="marker" cx={geom.x(hover)} cy={geom.y(s.points[hover].value)} r={4} fill={s.color} />
              ))}
            </svg>

            {hover != null && (
              <div
                className="tooltip multi"
                style={{ left: `${(geom.x(hover) / width) * 100}%`, top: `${PAD.top}px` }}
              >
                <div className="t-date">{formatDate(dates[hover])}</div>
                {visible.map((s) => (
                  <div key={s.key} className="t-row">
                    <span className="swatch" style={{ background: s.color }} />
                    {s.label}: <span className="t-val">{format(s.points[hover].value)}</span>
                  </div>
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
