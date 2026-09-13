import { getCategories } from "../api";
import { fmtInt, fmtScore, scoreTone } from "../format";
import { useAsync } from "../hooks";
import type { FacetField, Query } from "../types";
import { ScoreBar } from "./SignalTable";

interface Props {
  query: Query;
  field: FacetField;
  onFieldChange: (f: FacetField) => void;
  /** Clicking a group jumps back to the screener with that filter applied --
   *  browsing and screening are the same data, so the views have to connect. */
  onDrill: (field: FacetField, value: string) => void;
}

const FIELD_LABEL: Record<FacetField, string> = {
  category_group: "Asset group",
  category: "Category",
  family: "Issuer",
  exchange: "Exchange",
};

export default function CategoryView({ query, field, onFieldChange, onDrill }: Props) {
  const { data, error, loading } = useAsync(
    (signal) => getCategories(field, query.model, query.date, signal),
    [field, query.model, query.date],
  );

  return (
    <div className="panel">
      <div className="sidebar-head">
        <h2>Mean score by {FIELD_LABEL[field].toLowerCase()}</h2>
        <select
          value={field}
          onChange={(e) => onFieldChange(e.target.value as FacetField)}
          aria-label="Group by"
          style={{ background: "var(--surface-sunken)", border: "1px solid var(--border)", borderRadius: 5, padding: "3px 6px", fontSize: 12 }}
        >
          {(Object.keys(FIELD_LABEL) as FacetField[]).map((f) => (
            <option key={f} value={f}>{FIELD_LABEL[f]}</option>
          ))}
        </select>
      </div>

      {error && <div className="notice error" style={{ margin: 13 }}>{error}</div>}

      {loading && !data && (
        <div style={{ padding: 12 }}>
          {Array.from({ length: 10 }, (_, i) => (
            <div key={i} className="skeleton" style={{ height: 28, marginBottom: 5 }} />
          ))}
        </div>
      )}

      {data && (
        <>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>{FIELD_LABEL[field]}</th>
                  <th className="right">Funds</th>
                  <th className="right">Outperform</th>
                  <th className="right">Underperform</th>
                  <th>Top pick</th>
                  <th className="right">Mean score</th>
                </tr>
              </thead>
              <tbody>
                {data.groups.map((g) => {
                  const name = String(g[field]);
                  return (
                    <tr key={name} onClick={() => onDrill(field, name)}>
                      <td className="ticker" style={{ fontWeight: 550 }}>{name}</td>
                      <td className="right num">{fmtInt(g.n_funds)}</td>
                      <td className="right num" style={{ color: g.n_outperform ? "var(--up)" : undefined }}>
                        {fmtInt(g.n_outperform)}
                      </td>
                      <td className="right num" style={{ color: g.n_underperform ? "var(--down)" : undefined }}>
                        {fmtInt(g.n_underperform)}
                      </td>
                      <td className="ticker">{g.top_ticker}</td>
                      <td className="right">
                        {/* Group means are an order of magnitude smaller than
                            single-fund scores, so the bar is stretched to stay
                            readable. The number beside it is the true mean. */}
                        <ScoreBar value={g.mean_score} barScale={6} />
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <p className="footnote" style={{ padding: "0 13px 14px" }}>
            The label ranks each fund <em>within</em> its own category, so a group mean near zero
            is the expected result — the interesting reading is the spread between groups, and
            which groups the model splits most confidently. Groups with fewer than 3 scored funds
            are omitted. Bars are scaled to group means, not to the full score range.
            Strongest: <strong className={scoreTone(data.groups[0]?.mean_score ?? 0)}>
              {String(data.groups[0]?.[field] ?? "—")}
            </strong> at {fmtScore(data.groups[0]?.mean_score)}.
          </p>
        </>
      )}
    </div>
  );
}
