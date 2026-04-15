// Tab 2: Portfolio — NAV summary, Holdings, Concentration, Trade History, Engine Log

import { useState, useMemo } from "react";
import { TextAttributes } from "@opentui/core";
import { useTerminalDimensions, useKeyboard } from "@opentui/react";
import { usePortfolio, useTrades } from "../data/hooks";
import { BloombergPanel } from "../components/ui/panel";
import { DataTable, type Column, type CellStyle } from "../components/ui/data-table";
import { fmtPrice, fmtChg, fmtPnl, fmtNpr, fmtVol, chgColor, truncate, fmtDateShort } from "../components/ui/helpers";
import * as colors from "../theme/colors";

// Sub-panel focus order: 0=Holdings, 1=Concentration, 2=Trades, 3=Engine Log
const PANEL_COUNT = 4;

// ── Holdings columns ─────────────────────────────────────────────────────────
const holdingsCols: Column[] = [
  { id: "symbol",      label: "SYMBOL",   width: 10, align: "left" },
  { id: "qty",         label: "QTY",      width: 6,  align: "right" },
  { id: "entry_price", label: "ENTRY",    width: 9,  align: "right" },
  { id: "ltp",         label: "LTP",      width: 9,  align: "right" },
  { id: "pnl",         label: "P&L",      width: 11, align: "right" },
  { id: "pnl_pct",     label: "RTN%",     width: 8,  align: "right" },
  { id: "weight",      label: "WT%",      width: 7,  align: "right" },
  { id: "signal_type", label: "SIGNAL",   width: 10, align: "left" },
  { id: "holding_days",label: "DAYS",     width: 6,  align: "right" },
];

// ── Concentration columns ────────────────────────────────────────────────────
const concCols: Column[] = [
  { id: "sector",   label: "SECTOR",     width: 16, align: "left" },
  { id: "exposure", label: "EXPOSURE",   width: 12, align: "right" },
  { id: "weight",   label: "WT%",        width: 8,  align: "right" },
  { id: "count",    label: "CNT",        width: 5,  align: "right" },
];

// ── Trade history columns ────────────────────────────────────────────────────
const tradeCols: Column[] = [
  { id: "date",    label: "DATE",    width: 8,  align: "left" },
  { id: "action",  label: "SIDE",    width: 5,  align: "left" },
  { id: "symbol",  label: "SYMBOL",  width: 10, align: "left" },
  { id: "shares",  label: "QTY",     width: 7,  align: "right" },
  { id: "price",   label: "PRICE",   width: 9,  align: "right" },
  { id: "pnl",     label: "P&L",     width: 11, align: "right" },
];

// ── Sector exposure from positions ───────────────────────────────────────────
interface SectorRow {
  sector: string;
  value: number;
  weight: number;
  count: number;
}

function buildSectorRows(
  sectorExposure: Record<string, number>,
  totalValue: number,
): SectorRow[] {
  return Object.entries(sectorExposure)
    .map(([sector, value]) => ({
      sector,
      value,
      weight: totalValue > 0 ? (value / totalValue) * 100 : 0,
      count: 0, // filled below if we track it
    }))
    .sort((a, b) => b.value - a.value);
}

// ── Component ────────────────────────────────────────────────────────────────

export function PortfolioTab() {
  const { data: portfolio, loading, error } = usePortfolio();
  const { data: trades } = useTrades();
  const { width, height } = useTerminalDimensions();
  const [focusedPanel, setFocusedPanel] = useState(0);

  // Keyboard: Tab/Shift+Tab to move between sub-panels
  useKeyboard((key) => {
    if (key.name === "Tab") {
      setFocusedPanel((p) => (p + 1) % PANEL_COUNT);
    }
  }, { release: false });

  // Derived data
  const sectorRows = useMemo(() => {
    if (!portfolio) return [];
    const rows = buildSectorRows(portfolio.sector_exposure, portfolio.total_value);
    // Enrich with position counts per sector
    const countMap: Record<string, number> = {};
    for (const pos of portfolio.positions) {
      // Use signal_type as rough proxy since we don't have sector on Position type
      // The API sector_exposure keys match
    }
    // Count positions per sector from sector_exposure keys
    for (const pos of portfolio.positions) {
      for (const [sec, val] of Object.entries(portfolio.sector_exposure)) {
        // This is approximate; real count needs backend enrichment
      }
    }
    return rows;
  }, [portfolio]);

  const tradeData = useMemo(() => {
    if (!trades) return [];
    return [...trades].reverse().slice(0, 50); // most recent first
  }, [trades]);

  // Layout math
  const availableHeight = height - 4;
  const navBarHeight = 3;
  const topRowHeight = Math.floor((availableHeight - navBarHeight) * 0.5);
  const bottomRowHeight = availableHeight - navBarHeight - topRowHeight;

  // ── Loading / Error states ─────────────────────────────────────────────────
  if (!portfolio) {
    return (
      <box flexGrow={1} justifyContent="center" alignItems="center">
        <text fg={colors.FG_DIM}>
          {loading ? "Loading portfolio..." : `Error: ${error}`}
        </text>
      </box>
    );
  }

  const p = portfolio;
  const dayColor = chgColor(p.day_pnl);
  const retColor = chgColor(p.total_return);
  const ddColor = colors.LOSS_HI;

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <box flexDirection="column" flexGrow={1} backgroundColor={colors.BG_BASE}>

      {/* ── NAV Summary Bar ─────────────────────────────────────────────── */}
      <box
        height={navBarHeight}
        backgroundColor={colors.BG_SURFACE}
        paddingLeft={2}
        paddingTop={1}
        flexDirection="row"
        gap={3}
      >
        <text fg={colors.FG_AMBER} attributes={TextAttributes.BOLD}>NAV </text>
        <text fg={colors.FG_PRIMARY} attributes={TextAttributes.BOLD}>{fmtNpr(p.nav)}</text>
        <text fg={colors.FG_DIM}>Cash </text>
        <text fg={colors.FG_PRIMARY}>{fmtNpr(p.cash)}</text>
        <text fg={colors.FG_DIM}>Invested </text>
        <text fg={colors.FG_PRIMARY}>{fmtNpr(p.total_cost)}</text>
        <text fg={colors.FG_DIM}>Day </text>
        <text fg={dayColor} attributes={TextAttributes.BOLD}>{fmtPnl(p.day_pnl)}</text>
        <text fg={dayColor}> {fmtChg(p.day_ret)}</text>
        <text fg={colors.FG_DIM}>Return </text>
        <text fg={retColor} attributes={TextAttributes.BOLD}>{fmtChg(p.total_return)}</text>
        <text fg={colors.FG_DIM}>MaxDD </text>
        <text fg={ddColor} attributes={TextAttributes.BOLD}>{p.max_dd.toFixed(1)}%</text>
        <text fg={colors.FG_DIM}>Regime </text>
        <text fg={colors.CYAN}>{p.regime || "—"}</text>
      </box>

      {/* ── Top Row: Holdings | Concentration ───────────────────────────── */}
      <box flexDirection="row" height={topRowHeight}>
        {/* Holdings table */}
        <BloombergPanel
          title="HOLDINGS"
          focused={focusedPanel === 0}
          flexGrow={3}
          subtitle={`${p.positions.length} positions`}
        >
          <DataTable<typeof p.positions[0]>
            columns={holdingsCols}
            data={p.positions}
            height={topRowHeight - 2}
            focused={focusedPanel === 0}
            emptyText="No positions"
            renderCell={(pos, colId) => {
              switch (colId) {
                case "symbol":
                  return { text: truncate(pos.symbol, 10), style: { fg: colors.FG_AMBER, bold: true } };
                case "qty":
                  return { text: String(pos.shares), style: { fg: colors.FG_PRIMARY } };
                case "entry_price":
                  return { text: fmtPrice(pos.entry_price), style: { fg: colors.FG_DIM } };
                case "ltp":
                  return { text: fmtPrice(pos.ltp), style: { fg: colors.FG_PRIMARY } };
                case "pnl":
                  return { text: fmtPnl(pos.unrealized_pnl), style: { fg: chgColor(pos.unrealized_pnl) } };
                case "pnl_pct":
                  return { text: fmtChg(pos.pnl_pct), style: { fg: chgColor(pos.pnl_pct), bold: true } };
                case "weight":
                  return {
                    text: `${pos.weight.toFixed(1)}%`,
                    style: { fg: pos.weight > 25 ? colors.YELLOW : colors.FG_PRIMARY },
                  };
                case "signal_type":
                  return { text: truncate(pos.signal_type, 10), style: { fg: colors.CYAN } };
                case "holding_days":
                  return {
                    text: `${pos.holding_days}d`,
                    style: { fg: pos.holding_days > 30 ? colors.YELLOW : colors.FG_SECONDARY },
                  };
                default:
                  return { text: "" };
              }
            }}
          />
        </BloombergPanel>

        {/* Concentration & Sector */}
        <BloombergPanel
          title="CONCENTRATION & SECTOR"
          focused={focusedPanel === 1}
          flexGrow={1}
          subtitle={`${sectorRows.length} sectors`}
        >
          <DataTable<SectorRow>
            columns={concCols}
            data={sectorRows}
            height={topRowHeight - 2}
            focused={focusedPanel === 1}
            emptyText="No sector data"
            renderCell={(row, colId) => {
              switch (colId) {
                case "sector":
                  return { text: truncate(row.sector, 16), style: { fg: colors.PURPLE } };
                case "exposure":
                  return { text: fmtNpr(row.value), style: { fg: colors.FG_PRIMARY } };
                case "weight":
                  return {
                    text: `${row.weight.toFixed(1)}%`,
                    style: {
                      fg: row.weight > 35 ? colors.LOSS_HI : row.weight > 25 ? colors.YELLOW : colors.FG_PRIMARY,
                      bold: row.weight > 35,
                    },
                  };
                case "count":
                  return { text: String(row.count || "—"), style: { fg: colors.FG_DIM } };
                default:
                  return { text: "" };
              }
            }}
          />
        </BloombergPanel>
      </box>

      {/* ── Bottom Row: Trade History | Engine Log ──────────────────────── */}
      <box flexDirection="row" height={bottomRowHeight}>
        {/* Trade History */}
        <BloombergPanel
          title="TRADE HISTORY"
          focused={focusedPanel === 2}
          flexGrow={2}
          subtitle={tradeData.length > 0 ? `${tradeData.length} trades` : undefined}
        >
          <DataTable<(typeof tradeData)[0]>
            columns={tradeCols}
            data={tradeData}
            height={bottomRowHeight - 2}
            focused={focusedPanel === 2}
            emptyText="No trades yet"
            renderCell={(trade, colId) => {
              switch (colId) {
                case "date":
                  return { text: fmtDateShort(trade.date), style: { fg: colors.FG_SECONDARY } };
                case "action":
                  return {
                    text: trade.action,
                    style: {
                      fg: trade.action === "BUY" ? colors.GAIN_HI : colors.LOSS_HI,
                      bold: true,
                    },
                  };
                case "symbol":
                  return { text: truncate(trade.symbol, 10), style: { fg: colors.FG_AMBER, bold: true } };
                case "shares":
                  return { text: String(trade.shares), style: { fg: colors.FG_PRIMARY } };
                case "price":
                  return { text: fmtPrice(trade.price), style: { fg: colors.FG_PRIMARY } };
                case "pnl":
                  return {
                    text: trade.pnl !== 0 ? fmtPnl(trade.pnl) : "—",
                    style: { fg: trade.pnl !== 0 ? chgColor(trade.pnl) : colors.FG_DIM },
                  };
                default:
                  return { text: "" };
              }
            }}
          />
        </BloombergPanel>

        {/* Engine Activity Log */}
        <BloombergPanel
          title="ENGINE LOG"
          focused={focusedPanel === 3}
          flexGrow={1}
          subtitle={p.engine_phase || undefined}
        >
          <scrollbox height={bottomRowHeight - 2}>
            <text fg={colors.FG_DIM}>
              {p.engine_phase
                ? `Engine phase: ${p.engine_phase}\nRegime: ${p.regime}\nPositions: ${p.positions.length}\nCash: ${fmtNpr(p.cash)}\nUnrealized: ${fmtPnl(p.unrealized)}\nRealized: ${fmtPnl(p.realized)}\nMax DD: ${p.max_dd.toFixed(1)}%`
                : "No engine activity"}
            </text>
          </scrollbox>
        </BloombergPanel>
      </box>
    </box>
  );
}
