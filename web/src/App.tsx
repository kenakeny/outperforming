import { useState } from "react";

import SaudiMarketView from "./SaudiMarketView";
import UsMarketView from "./UsMarketView";
import { getHealth } from "./api";
import { useAsync, useTheme } from "./hooks";

type Market = "us" | "saudi";

/** App shell: brand, the US/Saudi market switch, theme toggle, and a shared
 *  health indicator (one FastAPI process serves both markets). Everything
 *  market-specific -- search, model/arm pickers, date control, tabs -- lives
 *  inside the two market views, because the two markets don't share a control
 *  set beyond "pick a date". */
export default function App() {
  const [market, setMarket] = useState<Market>("us");
  const [theme, setTheme] = useTheme();
  const health = useAsync((s) => getHealth(s), []);

  return (
    <>
      <header className="header">
        <div className="brand">
          <h1>Outperformance</h1>
          <span className="tag">
            {market === "us" ? "US ETFs · 5-day peer-relative signal" : "Tadawul · transfer-learning record"}
          </span>
        </div>

        <div className="market-switch" role="tablist" aria-label="Market">
          <button role="tab" aria-pressed={market === "us"} aria-selected={market === "us"} onClick={() => setMarket("us")}>
            US Markets
          </button>
          <button role="tab" aria-pressed={market === "saudi"} aria-selected={market === "saudi"} onClick={() => setMarket("saudi")}>
            Saudi Market
          </button>
        </div>

        <div className="header-controls">
          <button
            className="icon-btn"
            title={`Theme: ${theme}`}
            aria-label={`Switch theme (currently ${theme})`}
            onClick={() => setTheme(theme === "light" ? "dark" : theme === "dark" ? "system" : "light")}
          >
            {theme === "light" ? "☀" : theme === "dark" ? "☾" : "◐"}
          </button>

          <span
            className={`status-dot ${health.data?.status ?? ""}`}
            title={
              health.data
                ? `${health.data.status} · ${health.data.panel_rows.toLocaleString()} panel rows · latest ${health.data.latest_date}`
                : "checking…"
            }
          />
        </div>
      </header>

      <main>
        {market === "us" ? <UsMarketView /> : (
          <div className="shell shell-wide">
            <div>
              <SaudiMarketView />
            </div>
          </div>
        )}
      </main>
    </>
  );
}
