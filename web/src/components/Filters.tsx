import { useState } from "react";

import { PREDICTION_LABEL, fmtInt, toneOf } from "../format";
import type { FacetField, FacetValue, FacetsResponse, PredictionClass, Query } from "../types";

interface Props {
  facets: FacetsResponse | null;
  query: Query;
  onChange: (patch: Partial<Query>) => void;
}

const FIELD_LABEL: Record<FacetField, string> = {
  category_group: "Asset group",
  category: "Category",
  family: "Issuer",
  exchange: "Exchange",
};

/** How many values a facet shows before it needs its own filter box. Issuer has
 *  ~150 values, exchange has 4; a search input on the latter is just clutter. */
const SEARCHABLE_AT = 12;

export default function Filters({ facets, query, onChange }: Props) {
  const activeCount =
    Object.values(query.filters).reduce((n, v) => n + v.length, 0) +
    query.prediction.length +
    (query.excludeLeveraged ? 1 : 0) +
    (query.minConfidence > 0 ? 1 : 0);

  const toggle = (field: FacetField, value: string) => {
    const current = query.filters[field];
    const next = current.includes(value)
      ? current.filter((v) => v !== value)
      : [...current, value];
    onChange({ filters: { ...query.filters, [field]: next }, page: 0 });
  };

  const togglePrediction = (p: PredictionClass) => {
    const next = query.prediction.includes(p)
      ? query.prediction.filter((v) => v !== p)
      : [...query.prediction, p];
    onChange({ prediction: next, page: 0 });
  };

  const clearAll = () =>
    onChange({
      filters: { category_group: [], category: [], family: [], exchange: [] },
      prediction: [],
      excludeLeveraged: false,
      minConfidence: 0,
      page: 0,
    });

  return (
    <aside className="panel sidebar">
      <div className="sidebar-head">
        <h2>Filters</h2>
        <button className="link-btn" onClick={clearAll} disabled={activeCount === 0}>
          Clear{activeCount > 0 ? ` (${activeCount})` : ""}
        </button>
      </div>

      <details className="facet" open>
        <summary>
          Signal
          {query.prediction.length > 0 && <span className="count-pill">{query.prediction.length}</span>}
        </summary>
        <div className="facet-body">
          <div className="chips">
            {(["outperform", "neutral", "underperform"] as PredictionClass[]).map((p) => (
              <button
                key={p}
                className={`chip ${toneOf(p)}`}
                aria-pressed={query.prediction.includes(p)}
                onClick={() => togglePrediction(p)}
              >
                {PREDICTION_LABEL[p]}
              </button>
            ))}
          </div>

          <div className="section-title" style={{ marginTop: 14 }}>Min confidence</div>
          <div className="slider-row">
            <input
              type="range" min={0} max={0.9} step={0.05}
              value={query.minConfidence}
              aria-label="Minimum confidence"
              onChange={(e) => onChange({ minConfidence: Number(e.target.value), page: 0 })}
            />
            <span className="val">{query.minConfidence === 0 ? "any" : query.minConfidence.toFixed(2)}</span>
          </div>

          <label className="check" style={{ marginTop: 8 }}>
            <input
              type="checkbox"
              aria-label="Exclude leveraged and inverse products"
              checked={query.excludeLeveraged}
              onChange={(e) => onChange({ excludeLeveraged: e.target.checked, page: 0 })}
            />
            <span className="lbl">Exclude leveraged &amp; inverse</span>
          </label>
        </div>
      </details>

      {facets?.fields.map((field) => (
        <FacetGroup
          key={field}
          field={field}
          values={facets.facets[field] ?? []}
          selected={query.filters[field]}
          onToggle={(v) => toggle(field, v)}
        />
      ))}

      {!facets && <div className="facet-body" style={{ padding: 13 }}>
        {[0, 1, 2].map((i) => <div key={i} className="skeleton" style={{ height: 28, marginBottom: 6 }} />)}
      </div>}
    </aside>
  );
}

function FacetGroup({
  field, values, selected, onToggle,
}: {
  field: FacetField;
  values: FacetValue[];
  selected: string[];
  onToggle: (value: string) => void;
}) {
  const [needle, setNeedle] = useState("");
  const searchable = values.length >= SEARCHABLE_AT;

  const visible = needle
    ? values.filter((v) => v.value.toLowerCase().includes(needle.toLowerCase()))
    : values;

  // Selected values are pinned to the top: once a list is scrolled or filtered,
  // a checked box you can no longer see is how people lose track of what they
  // have applied.
  const ordered = [
    ...visible.filter((v) => selected.includes(v.value)),
    ...visible.filter((v) => !selected.includes(v.value)),
  ];

  return (
    <details className="facet" open={selected.length > 0}>
      <summary>
        {FIELD_LABEL[field]}
        {selected.length > 0 && <span className="count-pill">{selected.length}</span>}
      </summary>
      <div className="facet-body">
        {searchable && (
          <input
            className="facet-filter"
            placeholder={`Filter ${FIELD_LABEL[field].toLowerCase()}…`}
            value={needle}
            aria-label={`Filter the ${FIELD_LABEL[field].toLowerCase()} list`}
            onChange={(e) => setNeedle(e.target.value)}
          />
        )}
        <div className="facet-list">
          {ordered.map((v) => (
            <label className="check" key={v.value}>
              {/* Explicit name: the wrapped text would otherwise be announced as
                  "Fixed Income386", the label and the count running together. */}
              <input
                type="checkbox"
                aria-label={`${v.value} (${fmtInt(v.count)} funds)`}
                checked={selected.includes(v.value)}
                onChange={() => onToggle(v.value)}
              />
              <span className="lbl" title={v.value}>{v.value}</span>
              <span className="n">{fmtInt(v.count)}</span>
            </label>
          ))}
          {ordered.length === 0 && (
            <div style={{ padding: "8px 6px", fontSize: 12, color: "var(--ink-muted)" }}>
              No match.
            </div>
          )}
        </div>
      </div>
    </details>
  );
}
