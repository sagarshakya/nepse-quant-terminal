// Tab 1: Market — Gainers, Losers, Volume Leaders, 52-Week Extremes, Live Quotes

import { useState } from "react";
import { useTerminalDimensions, useKeyboard } from "@opentui/react";
import { useMarketOverview } from "../data/hooks";
import { MarketPanel } from "../components/ui/market-panel";
import { BG_BASE, FG_DIM } from "../theme/colors";

export function MarketTab() {
  const { data, loading, error } = useMarketOverview();
  const { width, height } = useTerminalDimensions();
  const [focusedPanel, setFocusedPanel] = useState(0);

  // Calculate panel height (split available height between rows)
  const availableHeight = height - 4; // minus header, tab-bar, status-bar, padding
  const topRowHeight = Math.floor(availableHeight * 0.5);
  const bottomRowHeight = availableHeight - topRowHeight;

  if (!data) {
    return (
      <box flexGrow={1} justifyContent="center" alignItems="center">
        <text fg={FG_DIM}>{loading ? "Loading market data..." : `Error: ${error}`}</text>
      </box>
    );
  }

  return (
    <box flexDirection="column" flexGrow={1} backgroundColor={BG_BASE}>
      {/* Top row: Gainers | Losers | Volume Leaders */}
      <box flexDirection="row" height={topRowHeight}>
        <MarketPanel
          title="GAINERS"
          data={data.gainers}
          focused={focusedPanel === 0}
          height={topRowHeight}
          flexGrow={1}
        />
        <MarketPanel
          title="LOSERS"
          data={data.losers}
          focused={focusedPanel === 1}
          height={topRowHeight}
          flexGrow={1}
        />
        <MarketPanel
          title="VOLUME LEADERS"
          data={data.volume_leaders}
          focused={focusedPanel === 2}
          height={topRowHeight}
          flexGrow={1}
        />
      </box>

      {/* Bottom row: 52-Week Extremes | Live Quotes */}
      <box flexDirection="row" height={bottomRowHeight}>
        <MarketPanel
          title="52-WEEK EXTREMES"
          data={[...data.near_52w_high, ...data.near_52w_low]}
          focused={focusedPanel === 3}
          height={bottomRowHeight}
          flexGrow={1}
        />
        <MarketPanel
          title="LIVE QUOTES"
          data={data.live_quotes}
          focused={focusedPanel === 4}
          height={bottomRowHeight}
          flexGrow={1}
        />
      </box>
    </box>
  );
}
