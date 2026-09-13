import { useCallback, useEffect, useState } from "react";

import { getDates, getFacets, getHealth, getSignals } from "./api";
import CategoryView from "./components/CategoryView";
import DatePicker from "./components/DatePicker";
import Filters from "./components/Filters";
import SearchBox from "./components/SearchBox";
import SignalTable from "./components/SignalTable";
import TickerDrawer from "./components/TickerDrawer";
import UploadView from "./components/UploadView";
import { fmtInt, fmtPct, fmtScore, scoreTone } from "./format";
import { useAsync, useDebounced } from "./hooks";
import type { FacetField, Query } from "./types";

type Tab = "screener" | "categories" | "upload";

const EMPTY_FILTERS: Record<FacetField, string[]> = {
  category_group: [], category: [], family: [], exchange: [],
};

const INITIAL: Query = {
  model: "catboost_sr",
  date: null,           // null = "latest", resolved by the backend
  q: "",
  filters: EMPTY_FILTERS,
  prediction: [],
  excludeLeveraged: false,
  minConfidence: 0,
  sort: "score",
  order: "desc",
  page: 0,
  pageSize: 50,
};

/** The original screener: US ETFs, live models, search + facet filters. Its
 *  own control row (search, model, date) lives here rather than in the app
 *  shell -- the Saudi market has a different control set (arm, date, no
 *  search), so the two views don't share a header beyond brand/theme. */
export default function UsMarketView() {
  const [query, setQuery] = useState<Query>(INITIAL);
  const [tab, setTab] = useState<Tab>("screener");
  const [groupBy, setGroupBy] = useState<FacetField>("category");
  // The open fund lives in the URL hash so a detail view can be linked and
  // survives a reload -- cheaper than pulling in a router for one route.
  const [selected, setSelected] = useState<string | null>(
    () => decodeURIComponent(location.hash.replace(/^#\/?/, "")) || null,
  );

  useEffect(() => {
    const target = selected ? `#/${selected}` : "";
    if (location.hash !== target) history.replaceState(null, "", target || location.pathname);
  }, [selected]);

  useEffect(() => {
    const onHash = () => setSelected(decodeURIComponent(location.hash.replace(/^#\/?/, "")) || null);
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const health = useAsync((s) => getHealth(s), []);
  const facets = useAsync((s) => getFacets(s), []);
  const dates = useAsync((s) => getDates(s), []);

  // Free text hits the same endpoint as the filters, so it must be debounced
  // the same way the search box is -- otherwise every keystroke re-scores a page.
  const debouncedQ = useDebounced(query.q, 260);
  const effective = { ...query, q: debouncedQ };
  const signals = useAsync(
    (s) => getSignals(effective, s),
    [
      effective.model, effective.date, effective.q,
      JSON.stringify(effective.filters), JSON.stringify(effective.prediction),
      effective.excludeLeveraged, effective.minConfidence,
      effective.sort, effective.order, effective.page, effective.pageSize,
    ],
    tab === "screener",
  );

  const patch = useCallback((p: Partial<Query>) => setQuery((q) => ({ ...q, ...p })), []);

  const drill = (field: FacetField, value: string) => {
    setQuery((q) => ({ ...q, filters: { ...EMPTY_FILTERS, [field]: [value] }, page: 0 }));
    setTab("screener");
  };

  const models = health.data?.models_available ?? [];
  const summary = signals.data?.summary;
  const activeDate = signals.data?.date ?? health.data?.latest_date ?? null;

  // The backend reports any filter it couldn't honour rather than silently
  // dropping it; if that ever happens the user needs to see it.
  const ignored = signals.data?.ignored_filters ?? [];

  return (
    <>
      <div className="us-controls">
        <SearchBox model={query.model} date={query.date} onPick={setSelected} />

        <div className="control">
          <label htmlFor="model">Model</label>
          <select
            id="model"
            value={query.model}
            onChange={(e) => patch({ model: e.target.value, page: 0 })}
          >
            {models.length === 0 && <option value={query.model}>{query.model}</option>}
            {models.map((m) => <option key={m} value={m}>{m}</option>)}
          </select>
        </div>

        <DatePicker
          value={query.date}
          resolved={activeDate}
          dates={dates.data}
          onChange={(date) => patch({ date, page: 0 })}
        />
      </div>

      {/* The filter rail only drives the screener. Leaving it on screen for the
          other tabs would imply it filters them too -- the category roll-up is
          deliberately a whole-universe view, and the CSV tab scores a file. */}
      <div className={`shell${tab === "screener" ? "" : " shell-wide"}`}>
        {tab === "screener" && (
          <Filters facets={facets.data} query={query} onChange={patch} />
        )}

        <div>
          <div className="tabs" role="tablist">
            {([["screener", "Screener"], ["categories", "Browse categories"], ["upload", "Score a CSV"]] as const)
              .map(([key, label]) => (
                <button
                  key={key}
                  className="tab"
                  role="tab"
                  aria-selected={tab === key}
                  onClick={() => setTab(key)}
                >
                  {label}
                </button>
              ))}
          </div>

          {health.error && (
            <div className="notice error" style={{ marginBottom: 12 }}>
              Can’t reach the API: {health.error}. Start it with{" "}
              <code>uvicorn serve.api:app --reload</code>.
            </div>
          )}

          {tab === "screener" && (
            <>
              {summary && (
                <div className="tiles">
                  <div className="tile">
                    <div className="k">Matching</div>
                    <div className="v num">{fmtInt(summary.n)}</div>
                    <div className="sub">of {fmtInt(signals.data?.universe_scored ?? 0)} scored</div>
                  </div>
                  <div className="tile">
                    <div className="k">Outperform</div>
                    <div className="v num up">{fmtInt(summary.outperform)}</div>
                    <div className="sub">{fmtPct(summary.n ? summary.outperform / summary.n : 0)} of matches</div>
                  </div>
                  <div className="tile">
                    <div className="k">Underperform</div>
                    <div className="v num down">{fmtInt(summary.underperform)}</div>
                    <div className="sub">{fmtPct(summary.n ? summary.underperform / summary.n : 0)} of matches</div>
                  </div>
                  <div className="tile">
                    <div className="k">Mean score</div>
                    <div className={`v num ${scoreTone(summary.mean_score ?? 0)}`}>
                      {fmtScore(summary.mean_score)}
                    </div>
                    <div className="sub">{fmtPct(summary.mean_confidence, 1)} mean confidence</div>
                  </div>
                </div>
              )}

              <div style={{ marginBottom: 12 }}>
                <input
                  className="facet-filter"
                  style={{ maxWidth: 340, marginBottom: 0, padding: "7px 10px", fontSize: 13 }}
                  placeholder="Filter these results by ticker or name…"
                  value={query.q}
                  aria-label="Filter results by ticker or name"
                  onChange={(e) => patch({ q: e.target.value, page: 0 })}
                />
              </div>

              {ignored.length > 0 && (
                <div className="notice" style={{ marginBottom: 12 }}>
                  Ignored {ignored.join(", ")} — this deployment’s metadata has no such column.
                </div>
              )}
              {signals.error && (
                <div className="notice error" style={{ marginBottom: 12 }}>{signals.error}</div>
              )}

              <SignalTable
                data={signals.data}
                loading={signals.loading}
                query={query}
                onChange={patch}
                onPick={setSelected}
              />

              <p className="footnote">
                Score = P(outperform) − P(underperform) over the next 5 trading days, ranked
                against the fund’s own category peers. Categories come from a periodically
                refreshed metadata snapshot and are not point-in-time, so peer groups —
                and therefore scores — can shift between refreshes. Research output, not
                investment advice.
              </p>
            </>
          )}

          {tab === "categories" && (
            <CategoryView
              query={query}
              field={groupBy}
              onFieldChange={setGroupBy}
              onDrill={drill}
            />
          )}

          {tab === "upload" && <UploadView model={query.model} />}
        </div>
      </div>

      {selected && (
        <TickerDrawer ticker={selected} model={query.model} onClose={() => setSelected(null)} />
      )}
    </>
  );
}
