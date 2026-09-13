import { useEffect, useMemo, useRef, useState } from "react";

export interface Point {
  date: string;
  value: number;
}

interface Props {
  points: Point[];
  /** Names the single series -- there is no legend box because there is only
   *  one line, so the title is what identifies it. */
  label: string;
  height?: number;
  /** Draw a zero baseline and let the domain include 0. For the score series,
   *  where zero is "no view", the baseline is the most important reference on
   *  the chart; for a price series it is meaningless and would flatten the line. */
  zeroBaseline?: boolean;
  format: (v: number) => string;
  formatDate: (iso: string) => string;
}

const PAD = { top: 8, right: 10, bottom: 20, left: 46 };

/** Single-series line chart with a crosshair + tooltip hover layer.
 *
 *  Hand-rolled SVG rather than a charting library: two small charts is not
 *  worth a dependency, and this way the marks follow the same tokens as the
 *  rest of the page (2px stroke, recessive grid, surface-ringed marker).
 */
export default function LineChart({
  points, label, height = 148, zeroBaseline = false, format, formatDate,
}: Props) {
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const [hover, setHover] = useState<number | null>(null);
  const [width, setWidth] = useState(560);

  // Track the rendered width so the x-scale and the tick spacing match what's
  // actually on screen. The SVG is width:100%; only the viewBox depends on this.
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const observer = new ResizeObserver(([entry]) => {
      const next = entry.contentRect.width;
      if (next > 0) setWidth((w) => (Math.abs(next - w) > 1 ? next : w));
    });
    observer.observe(el);
    return () => observer.disconnect();
    // Re-attach when the chart goes from its empty state to having data: the
    // wrapper element doesn't exist to observe until there's something to plot.
  }, [points.length > 0]);

  const geom = useMemo(() => {
    if (points.length === 0) return null;
    const values = points.map((p) => p.value);
    let min = Math.min(...values);
    let max = Math.max(...values);
    if (zeroBaseline) {
      const reach = Math.max(Math.abs(min), Math.abs(max), 1e-6);
      min = -reach;
      max = reach;
    }
    if (min === max) { min -= 1; max += 1; }
    const pad = (max - min) * 0.08;
    min -= pad;
    max += pad;

    const innerW = Math.max(width - PAD.left - PAD.right, 10);
    const innerH = height - PAD.top - PAD.bottom;
    const x = (i: number) =>
      PAD.left + (points.length === 1 ? innerW / 2 : (i / (points.length - 1)) * innerW);
    const y = (v: number) => PAD.top + innerH - ((v - min) / (max - min)) * innerH;

    const path = points.map((p, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(2)},${y(p.value).toFixed(2)}`).join(" ");
    const ticks = [max - pad, (min + max) / 2, min + pad];
    return { x, y, path, min, max, ticks, innerW, innerH };
  }, [points, width, height, zeroBaseline]);

  if (!geom || points.length === 0) {
    return <div className="notice">No {label.toLowerCase()} to plot.</div>;
  }

  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const rel = ((e.clientX - rect.left) / rect.width) * width - PAD.left;
    const i = Math.round((rel / geom.innerW) * (points.length - 1));
    setHover(Math.min(points.length - 1, Math.max(0, i)));
  };

  const active = hover != null ? points[hover] : null;
  const tone = (v: number) => (zeroBaseline && v < 0 ? "var(--down)" : "var(--up)");

  return (
    <div className="chart" ref={wrapRef}>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`${label}: ${points.length} points from ${formatDate(points[0].date)} to ${formatDate(points[points.length - 1].date)}, latest ${format(points[points.length - 1].value)}`}
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
      >
        {geom.ticks.map((t, i) => (
          <g key={i}>
            <line className="gridline" x1={PAD.left} x2={width - PAD.right} y1={geom.y(t)} y2={geom.y(t)} />
            <text className="axis-label" x={PAD.left - 6} y={geom.y(t) + 3} textAnchor="end">{format(t)}</text>
          </g>
        ))}

        {zeroBaseline && (
          <line className="baseline" x1={PAD.left} x2={width - PAD.right} y1={geom.y(0)} y2={geom.y(0)} />
        )}

        <path className="series" d={geom.path} stroke={tone(points[points.length - 1].value)} />

        <text className="axis-label" x={PAD.left} y={height - 5}>{formatDate(points[0].date)}</text>
        <text className="axis-label" x={width - PAD.right} y={height - 5} textAnchor="end">
          {formatDate(points[points.length - 1].date)}
        </text>

        {active && hover != null && (
          <g>
            <line className="crosshair" x1={geom.x(hover)} x2={geom.x(hover)} y1={PAD.top} y2={height - PAD.bottom} />
            <circle
              className="marker"
              cx={geom.x(hover)}
              cy={geom.y(active.value)}
              r={4.5}
              fill={tone(active.value)}
            />
          </g>
        )}
      </svg>

      {active && hover != null && (
        <div
          className="tooltip"
          style={{
            left: `${(geom.x(hover) / width) * 100}%`,
            top: `${geom.y(active.value) - 10}px`,
          }}
        >
          <div className="t-date">{formatDate(active.date)}</div>
          <div className="t-val">{format(active.value)}</div>
        </div>
      )}
    </div>
  );
}
