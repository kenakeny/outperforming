import { useEffect, useRef, useState } from "react";

import { getSearch } from "../api";
import { fmtScore, toneOf } from "../format";
import { useAsync, useDebounced, useEscape } from "../hooks";
import type { SearchHit } from "../types";

interface Props {
  model: string;
  date: string | null;
  onPick: (ticker: string) => void;
}

/** The "look anything up" box: ticker, fund name, issuer or category.
 *
 *  Ranking lives on the server (serve/universe.py) so the API and this box
 *  can't disagree about what the best match is. This component owns the
 *  interaction: debounce, keyboard nav, and dismissal.
 */
export default function SearchBox({ model, date, onPick }: Props) {
  const [text, setText] = useState("");
  const [open, setOpen] = useState(false);
  const [cursor, setCursor] = useState(0);
  const boxRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const query = useDebounced(text.trim(), 180);
  const { data, loading } = useAsync(
    (signal) => getSearch(query, model, date, signal),
    [query, model, date],
    query.length > 0,
  );

  const hits: SearchHit[] = query ? data?.results ?? [] : [];

  useEffect(() => setCursor(0), [query]);

  // "/" focuses the box from anywhere, the way every other screener works --
  // but not while the user is already typing into some other field.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null;
      const typing = el && (el.tagName === "INPUT" || el.tagName === "SELECT" || el.isContentEditable);
      if (e.key === "/" && !typing) {
        e.preventDefault();
        inputRef.current?.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  useEscape(() => setOpen(false), open);

  useEffect(() => {
    const onClick = (e: MouseEvent) => {
      if (!boxRef.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onClick);
    return () => document.removeEventListener("mousedown", onClick);
  }, []);

  const pick = (hit: SearchHit) => {
    onPick(hit.ticker);
    setOpen(false);
    setText("");
    inputRef.current?.blur();
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (!open || hits.length === 0) return;
    if (e.key === "ArrowDown") { e.preventDefault(); setCursor((c) => (c + 1) % hits.length); }
    else if (e.key === "ArrowUp") { e.preventDefault(); setCursor((c) => (c - 1 + hits.length) % hits.length); }
    else if (e.key === "Enter") { e.preventDefault(); pick(hits[cursor]); }
  };

  return (
    <div className="search" ref={boxRef}>
      <div className="search-input-wrap">
        <svg width="15" height="15" viewBox="0 0 16 16" fill="none" aria-hidden="true">
          <circle cx="7" cy="7" r="4.6" stroke="currentColor" strokeWidth="1.6" />
          <path d="M10.4 10.4L14 14" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
        </svg>
        <input
          ref={inputRef}
          value={text}
          placeholder="Search ticker, fund name, issuer or category…"
          aria-label="Search funds by ticker, name, issuer or category"
          role="combobox"
          aria-expanded={open && hits.length > 0}
          aria-controls="search-results"
          autoComplete="off"
          spellCheck={false}
          onChange={(e) => { setText(e.target.value); setOpen(true); }}
          onFocus={() => setOpen(true)}
          onKeyDown={onKeyDown}
        />
        {!text && <kbd>/</kbd>}
      </div>

      {open && query.length > 0 && (
        <div className="search-results" id="search-results" role="listbox">
          {hits.map((hit, i) => (
            <button
              key={hit.ticker}
              className="search-hit"
              role="option"
              aria-selected={i === cursor}
              data-active={i === cursor}
              onMouseEnter={() => setCursor(i)}
              onClick={() => pick(hit)}
            >
              <span className="tk">{hit.ticker}</span>
              <span className="nm">{hit.name ?? "—"}</span>
              <span className="ct">{hit.category ?? hit.category_group ?? ""}</span>
              {hit.scored && hit.score != null ? (
                <span className={`pred ${toneOf(hit.prediction)} num`}>{fmtScore(hit.score)}</span>
              ) : (
                /* Findable but outside the scored panel -- saying so beats an
                   empty cell the user has to guess the meaning of. */
                <span className="ct" title="In the fund universe but not in the scored panel">
                  not scored
                </span>
              )}
            </button>
          ))}
          {hits.length === 0 && (
            <div className="search-empty">
              {loading ? "Searching…" : `Nothing matches “${query}”.`}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
