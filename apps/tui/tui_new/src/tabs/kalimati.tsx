// Tab: Kalimati — Vegetable/commodity prices, Metals/Energy, Forex

import { useState } from "react";
import { useTerminalDimensions } from "@opentui/react";
import { TextAttributes } from "@opentui/core";
import { useRates } from "../data/hooks";
import { BloombergPanel } from "../components/ui/panel";
import { DataTable, type Column, type CellStyle } from "../components/ui/data-table";
import * as colors from "../theme/colors";
import { fmtPrice, chgColor, truncate } from "../components/ui/helpers";

// ── Column definitions ──

const kalimatiCols: Column[] = [
  { id: "name", label: "COMMODITY", width: 18, align: "left" },
  { id: "unit", label: "UNIT", width: 6, align: "left" },
  { id: "min_price", label: "MIN", width: 8, align: "right" },
  { id: "max_price", label: "MAX", width: 8, align: "right" },
  { id: "avg_price", label: "AVG", width: 8, align: "right" },
];

const metalsCols: Column[] = [
  { id: "name", label: "METAL/ENERGY", width: 14, align: "left" },
  { id: "price", label: "PRICE", width: 12, align: "right" },
  { id: "change", label: "CHG", width: 10, align: "right" },
];

const forexCols: Column[] = [
  { id: "currency", label: "CURRENCY", width: 10, align: "left" },
  { id: "buy", label: "BUY", width: 10, align: "right" },
  { id: "sell", label: "SELL", width: 10, align: "right" },
];

export function KalimatiTab() {
  const { data: rates, loading, error } = useRates();
  const { width, height } = useTerminalDimensions();
  const [focusedPanel, setFocusedPanel] = useState(0);
  const [search, setSearch] = useState("");
  const [searchFocused, setSearchFocused] = useState(false);

  const availableHeight = height - 4;
  const rightTopHeight = Math.floor(availableHeight * 0.5);
  const rightBottomHeight = availableHeight - rightTopHeight;

  if (loading && !rates) {
    return (
      <box flexGrow={1} justifyContent="center" alignItems="center">
        <text fg={colors.FG_DIM}>Loading Kalimati data...</text>
      </box>
    );
  }

  if (error && !rates) {
    return (
      <box flexGrow={1} justifyContent="center" alignItems="center">
        <text fg={colors.LOSS}>Error: {error}</text>
      </box>
    );
  }

  // Filter kalimati data by search term
  const allKalimati = rates?.kalimati ?? [];
  const filteredKalimati = search.length > 0
    ? allKalimati.filter((item) =>
        item.name.toLowerCase().includes(search.toLowerCase())
      )
    : allKalimati;

  // Metals + Energy combined for right-top panel
  const metalsEnergy = [
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

  // Key forex currencies for right-bottom
  const forexData = rates?.forex ?? [];

  return (
    <box flexDirection="row" flexGrow={1} backgroundColor={colors.BG_BASE}>
      {/* Left: Kalimati Market Prices */}
      <BloombergPanel
        title="KALIMATI MARKET"
        subtitle={`${filteredKalimati.length}/${allKalimati.length} items`}
        focused={focusedPanel === 0}
        flexGrow={2}
      >
        <box flexDirection="column" flexGrow={1}>
          {/* Search bar */}
          <box height={1} paddingLeft={1} paddingRight={1} backgroundColor={colors.BG_SURFACE}>
            <text fg={colors.FG_DIM}>Search: </text>
            <input
              placeholder="Filter commodities..."
              onInput={(val: string) => setSearch(val)}
              focused={searchFocused}
            />
          </box>

          <DataTable
            columns={kalimatiCols}
            data={filteredKalimati}
            height={availableHeight - 4}
            focused={focusedPanel === 0}
            emptyText={search ? "No matches found" : "No Kalimati data"}
            renderCell={(item, colId) => {
              switch (colId) {
                case "name":
                  return { text: truncate(item.name, 18), style: { fg: colors.FG_AMBER, bold: true } };
                case "unit":
                  return { text: truncate(item.unit, 6), style: { fg: colors.FG_DIM } };
                case "min_price":
                  return { text: fmtPrice(item.min_price), style: { fg: colors.GAIN } };
                case "max_price":
                  return { text: fmtPrice(item.max_price), style: { fg: colors.LOSS } };
                case "avg_price":
                  return { text: fmtPrice(item.avg_price), style: { fg: colors.CYAN } };
                default:
                  return { text: "" };
              }
            }}
          />
        </box>
      </BloombergPanel>

      {/* Right column */}
      <box flexDirection="column" flexGrow={1}>
        {/* Right-top: Metals & Energy */}
        <BloombergPanel
          title="METALS & ENERGY"
          focused={focusedPanel === 1}
          height={rightTopHeight}
        >
          <DataTable
            columns={metalsCols}
            data={metalsEnergy}
            height={rightTopHeight - 3}
            focused={focusedPanel === 1}
            emptyText="No metals data"
            renderCell={(item, colId) => {
              switch (colId) {
                case "name":
                  return { text: truncate(item.name, 14), style: { fg: colors.YELLOW, bold: true } };
                case "price":
                  return { text: fmtPrice(item.price), style: { fg: colors.FG_PRIMARY } };
                case "change":
                  return { text: item.change !== 0 ? `${item.change > 0 ? "+" : ""}${item.change.toFixed(2)}%` : "--", style: { fg: chgColor(item.change) } };
                default:
                  return { text: "" };
              }
            }}
          />
        </BloombergPanel>

        {/* Right-bottom: Forex */}
        <BloombergPanel
          title="FOREX vs NPR"
          subtitle="NRB Rates"
          focused={focusedPanel === 2}
          height={rightBottomHeight}
        >
          <DataTable
            columns={forexCols}
            data={forexData}
            height={rightBottomHeight - 3}
            focused={focusedPanel === 2}
            emptyText="No forex data"
            renderCell={(item, colId) => {
              switch (colId) {
                case "currency":
                  return { text: truncate(item.currency, 10), style: { fg: colors.CYAN, bold: true } };
                case "buy":
                  return { text: fmtPrice(item.buy), style: { fg: colors.GAIN } };
                case "sell":
                  return { text: fmtPrice(item.sell), style: { fg: colors.LOSS } };
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
