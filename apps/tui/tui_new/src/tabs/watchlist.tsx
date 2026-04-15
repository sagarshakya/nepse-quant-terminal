// Tab: Watchlist — Tracked stocks, Forex rates, Commodities

import { useState } from "react";
import { useTerminalDimensions } from "@opentui/react";
import { TextAttributes } from "@opentui/core";
import { useWatchlist, useRates } from "../data/hooks";
import { BloombergPanel } from "../components/ui/panel";
import { DataTable, type Column, type CellStyle } from "../components/ui/data-table";
import * as colors from "../theme/colors";
import { fmtPrice, fmtChg, fmtVol, chgColor, truncate } from "../components/ui/helpers";

// ── Column definitions ──

const watchlistCols: Column[] = [
  { id: "symbol", label: "SYMBOL", width: 10, align: "left" },
  { id: "ltp", label: "LTP", width: 10, align: "right" },
  { id: "change_pct", label: "CHG%", width: 8, align: "right" },
  { id: "volume", label: "VOLUME", width: 10, align: "right" },
];

const forexCols: Column[] = [
  { id: "currency", label: "CURRENCY", width: 10, align: "left" },
  { id: "buy", label: "BUY", width: 10, align: "right" },
  { id: "sell", label: "SELL", width: 10, align: "right" },
  { id: "unit", label: "UNIT", width: 6, align: "right" },
];

const commodityCols: Column[] = [
  { id: "name", label: "COMMODITY", width: 14, align: "left" },
  { id: "price", label: "PRICE", width: 10, align: "right" },
  { id: "change", label: "CHG", width: 10, align: "right" },
];

export function WatchlistTab() {
  const { data: watchlist, loading: wLoading, error: wError } = useWatchlist();
  const { data: rates, loading: rLoading, error: rError } = useRates();
  const { width, height } = useTerminalDimensions();
  const [focusedPanel, setFocusedPanel] = useState(0);

  const availableHeight = height - 4;
  const rightTopHeight = Math.floor(availableHeight * 0.5);
  const rightBottomHeight = availableHeight - rightTopHeight;

  const loading = wLoading || rLoading;
  const error = wError || rError;

  if (loading && !watchlist && !rates) {
    return (
      <box flexGrow={1} justifyContent="center" alignItems="center">
        <text fg={colors.FG_DIM}>Loading watchlist data...</text>
      </box>
    );
  }

  if (error && !watchlist && !rates) {
    return (
      <box flexGrow={1} justifyContent="center" alignItems="center">
        <text fg={colors.LOSS}>Error: {error}</text>
      </box>
    );
  }

  const watchlistData = watchlist ?? [];
  const forexData = rates?.forex ?? [];
  // Combine metals + energy into a single commodities list
  const commodityData = [
    ...(rates?.metals ?? []).map((m) => ({
      name: m.name,
      price: m.price,
      change: m.change,
    })),
    ...(rates?.energy ?? []).map((e) => ({
      name: e.name,
      price: e.price,
      change: 0,
    })),
  ];

  return (
    <box flexDirection="row" flexGrow={1} backgroundColor={colors.BG_BASE}>
      {/* Left: Stock Watchlist */}
      <BloombergPanel
        title="WATCHLIST"
        subtitle={`${watchlistData.length} symbols`}
        focused={focusedPanel === 0}
        flexGrow={1}
      >
        <DataTable
          columns={watchlistCols}
          data={watchlistData}
          height={availableHeight - 2}
          focused={focusedPanel === 0}
          emptyText="No symbols tracked"
          renderCell={(item, colId) => {
            switch (colId) {
              case "symbol":
                return { text: truncate(item.symbol, 10), style: { fg: colors.FG_AMBER, bold: true } };
              case "ltp":
                return { text: fmtPrice(item.ltp), style: { fg: colors.FG_PRIMARY } };
              case "change_pct":
                return { text: fmtChg(item.change_pct), style: { fg: chgColor(item.change_pct) } };
              case "volume":
                return { text: fmtVol(item.volume), style: { fg: colors.FG_SECONDARY } };
              default:
                return { text: "" };
            }
          }}
        />
      </BloombergPanel>

      {/* Right column: Forex + Commodities */}
      <box flexDirection="column" flexGrow={1}>
        {/* Right-top: Forex Rates */}
        <BloombergPanel
          title="FOREX RATES"
          subtitle="NRB"
          focused={focusedPanel === 1}
          height={rightTopHeight}
        >
          <DataTable
            columns={forexCols}
            data={forexData}
            height={rightTopHeight - 3}
            focused={focusedPanel === 1}
            emptyText="No forex data"
            renderCell={(item, colId) => {
              switch (colId) {
                case "currency":
                  return { text: truncate(item.currency, 10), style: { fg: colors.CYAN, bold: true } };
                case "buy":
                  return { text: fmtPrice(item.buy), style: { fg: colors.GAIN } };
                case "sell":
                  return { text: fmtPrice(item.sell), style: { fg: colors.LOSS } };
                case "unit":
                  return { text: String(item.unit), style: { fg: colors.FG_DIM } };
                default:
                  return { text: "" };
              }
            }}
          />
        </BloombergPanel>

        {/* Right-bottom: Commodities (Metals + Energy) */}
        <BloombergPanel
          title="COMMODITIES"
          subtitle="Metals & Energy"
          focused={focusedPanel === 2}
          height={rightBottomHeight}
        >
          <DataTable
            columns={commodityCols}
            data={commodityData}
            height={rightBottomHeight - 3}
            focused={focusedPanel === 2}
            emptyText="No commodity data"
            renderCell={(item, colId) => {
              switch (colId) {
                case "name":
                  return { text: truncate(item.name, 14), style: { fg: colors.FG_AMBER, bold: true } };
                case "price":
                  return { text: fmtPrice(item.price), style: { fg: colors.FG_PRIMARY } };
                case "change":
                  return { text: item.change !== 0 ? fmtChg(item.change) : "--", style: { fg: chgColor(item.change) } };
                default:
                  return { text: "" };
              }
            }}
          />
        </BloombergPanel>
      </box>
    </box>
  );
}
