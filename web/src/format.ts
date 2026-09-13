import type { PredictionClass } from "./types";

/** Signed, fixed-width score. Always shows the sign: the whole point of the
 *  score is its direction, and "+0.04" vs "0.04" is the difference between
 *  reading a rank and reading a magnitude. */
export const fmtScore = (v: number | null | undefined) =>
  v == null || Number.isNaN(v) ? "—" : `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(3)}`;

export const fmtPct = (v: number | null | undefined, digits = 0) =>
  v == null || Number.isNaN(v) ? "—" : `${(v * 100).toFixed(digits)}%`;

export const fmtInt = (v: number | null | undefined) =>
  v == null ? "—" : v.toLocaleString("en-US");

/** CSS modifier for the diverging encoding. Kept in one place so the table,
 *  the tiles and the drawer can never disagree about which way is up. */
export const toneOf = (p: PredictionClass | null | undefined) =>
  p === "outperform" ? "up" : p === "underperform" ? "down" : "flat";

export const scoreTone = (v: number) => (v > 0.005 ? "up" : v < -0.005 ? "down" : "flat");

export const PREDICTION_LABEL: Record<PredictionClass, string> = {
  outperform: "Outperform",
  neutral: "Neutral",
  underperform: "Underperform",
};

/** "2026-07-02" -> "Jul 2, 2026", without dragging in a date library or
 *  tripping over the local-timezone shift `new Date("2026-07-02")` causes. */
export function fmtDate(iso: string): string {
  const [y, m, d] = iso.split("-").map(Number);
  if (!y || !m || !d) return iso;
  return new Date(y, m - 1, d).toLocaleDateString("en-US", {
    month: "short", day: "numeric", year: "numeric",
  });
}

export const fmtDateShort = (iso: string): string => {
  const [y, m, d] = iso.split("-").map(Number);
  if (!y || !m || !d) return iso;
  return new Date(y, m - 1, d).toLocaleDateString("en-US", { month: "short", day: "numeric" });
};
