// Tab 3: Signals — Signal table, Corporate Actions Calendar, Sector Heatmap

import { useState, useMemo } from "react";
import { TextAttributes } from "@opentui/core";
import { useTerminalDimensions, useKeyboard } from "@opentui/react";
import { useSignals, useCalendar } from "../data/hooks";
import { BloombergPanel } from "../components/ui/panel";
import { DataTable, type Column, type CellStyle } from "../components/ui/data-table";
import * as colors from "../theme/colors";
import { fmtPrice, fmtChg, truncate, fmtDateShort } from "../components/ui/helpers";

// Sub-panel focus: 0=Signals, 1=Calendar, 2=Heatmap
const PANEL_COUNT = 3;

// ── Signals table columns ────────────────────────────────────────────────────
const signalCols: Column[] = [
  { id: "rank",        label: "#",       width: 4,  align: "right" },
  { id: "symbol",      label: "SYMBOL",  width: 10, align: "left" },
  { id: "score",       label: "SCORE",   width: 8,  align: "right" },
  { id: "signal_type", label: "TYPE",    width: 14, align: "left" },
  { id: "strength",    label: "STR",     width: 7,  align: "right" },
  { id: "confidence",  label: "CONF",    width: 7,  align: "right" },
  { id: "direction",   label: "DIR",     width: 8,  align: "left" },
];

// ── Corporate actions calendar columns ───────────────────────────────────────
const calendarCols: Column[] = [
  { id: "symbol",         label: "SYMBOL",     width: 10, align: "left" },
  { id: "bookclose_date", label: "BOOK CLOSE", width: 12, align: "left" },
  { id: "days_to",        label: "DAYS",       width: 6,  align: "right" },
  { id: "cash_dividend",  label: "CASH%",      width: 8,  align: "right" },
  { id: "bonus_share",    label: "BONUS%",     width: 8,  align: "right" },
  { id: "right_share",    label: "RIGHT",      width: 8,  align: "left" },
  { id: "buy_by",         label: "BUY BY",     width: 12, align: "left" },
];

// ── Heatmap block colors by avg change ───────────────────────────────────────
function heatColor(avgChg: number): string {
  if (avgChg > 2)  return colors.GAIN_HI;   // #00ff7f
  if (avgChg > 1)  return "#00cc60";
  if (avgChg > 0)  return "#66cc66";
  if (avgChg > -1) return "#cc9933";
  if (avgChg > -3) return "#cc4444";
  return colors.LOSS_HI;                     // #ff4545
}

function heatBlocks(avgChg: number, maxAbs: number): string {
  const filled = Math.max(1, Math.round((Math.abs(avgChg) / maxAbs) * 8));
  return "\u2588".repeat(filled) + "\u2591".repeat(8 - filled);
}

// ── Enriched calendar row ────────────────────────────────────────────────────
interface CalendarRow {
  symbol: string;
  bookclose_date: string;
  days_to: number;
  cash_dividend_pct: number;
  bonus_share_pct: number;
  right_share_ratio: string;
  buy_by: string;
}

function enrichCalendar(actions: ReturnType<typeof useCalendar>["data"]): CalendarRow[] {
  if (!actions || actions.length === 0) return [];

  const now = Date.now();
  return actions
    .map((ca) => {
      const bcDate = new Date(ca.bookclose_date);
      const daysTo = Math.ceil((bcDate.getTime() - now) / 86_400_000);
      // Buy by = 5 trading days before book close (approx calendar days)
      const buyBy = new Date(bcDate.getTime() - 5 * 86_400_000);
      return {
        symbol: ca.symbol,
        bookclose_date: ca.bookclose_date,
        days_to: daysTo,
        cash_dividend_pct: ca.cash_dividend_pct,
        bonus_share_pct: ca.bonus_share_pct,
        right_share_ratio: ca.right_share_ratio,
        buy_by: buyBy.toISOString().slice(0, 10),
      };
    })
    .filter((r) => r.days_to >= 0)
    .sort((a, b) => a.days_to - b.days_to);
}

// ── Sector heatmap from signals (group by signal_type as proxy) ──────────────
interface SectorHeat {
  name: string;
  avgChg: number;
  count: number;
}

function buildSectorHeat(signals: ReturnType<typeof useSignals>["data"]): SectorHeat[] {
  if (!signals || signals.length === 0) return [];

  // Group signals by signal_type and compute average score
  const groups: Record<string, { totalScore: number; count: number }> = {};
  for (const sig of signals) {
    const key = sig.signal_type || "Other";
    if (!groups[key]) groups[key] = { totalScore: 0, count: 0 };
    groups[key].totalScore += sig.score;
    groups[key].count += 1;
  }

  return Object.entries(groups)
    .map(([name, g]) => ({
      name,
      avgChg: g.totalScore / g.count,
      count: g.count,
    }))
    .sort((a, b) => b.avgChg - a.avgChg);
}

// ── Component ────────────────────────────────────────────────────────────────

export function SignalsTab() {
  const { data: signals, loading: sigLoading, error: sigError } = useSignals();
  const { data: calendar, loading: calLoading } = useCalendar();
  const { width, height } = useTerminalDimensions();
  const [focusedPanel, setFocusedPanel] = useState(0);

  useKeyboard((key) => {
    if (key.name === "Tab") {
      setFocusedPanel((p) => (p + 1) % PANEL_COUNT);
    }
  }, { release: false });

  // Derived data
  const signalData = useMemo(() => {
    if (!signals) return [];
    return [...signals].sort((a, b) => b.score - a.score).slice(0, 40);
  }, [signals]);

  const calendarRows = useMemo(() => enrichCalendar(calendar), [calendar]);
  const sectorHeat = useMemo(() => buildSectorHeat(signals), [signals]);

  // Layout
  const availableHeight = height - 4;
  const topRowHeight = Math.floor(availableHeight * 0.55);
  const bottomRowHeight = availableHeight - topRowHeight;

  // ── Loading state ──────────────────────────────────────────────────────────
  if (!signals && sigLoading) {
    return (
      <box flexGrow={1} justifyContent="center" alignItems="center">
        <text fg={colors.FG_DIM}>Loading signals...</text>
      </box>
    );
  }

  if (!signals && sigError) {
    return (
      <box flexGrow={1} justifyContent="center" alignItems="center">
        <text fg={colors.LOSS}>Error: {sigError}</text>
      </box>
    );
  }

  const maxAbsChg = sectorHeat.length > 0
    ? Math.max(...sectorHeat.map((s) => Math.abs(s.avgChg)), 0.01)
    : 1;

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <box flexDirection="column" flexGrow={1} backgroundColor={colors.BG_BASE}>

      {/* ── Status bar ────────────────────────────────────────────────────── */}
      <box height={1} paddingLeft={2} backgroundColor={colors.BG_SURFACE}>
        <text fg={colors.FG_AMBER} attributes={TextAttributes.BOLD}>SIGNALS WORKSPACE</text>
        <text fg={colors.FG_DIM}>
          {"   "}
          {signalData.length} signals
          {"   "}
          {calendarRows.length} upcoming events
        </text>
      </box>

      {/* ── Top: Signals table ────────────────────────────────────────────── */}
      <box flexDirection="row" height={topRowHeight}>
        <BloombergPanel
          title="SIGNALS"
          focused={focusedPanel === 0}
          flexGrow={1}
          subtitle={`Top ${signalData.length}`}
        >
          <DataTable<(typeof signalData)[0] & { _rank: number }>
            columns={signalCols}
            data={signalData.map((s, i) => ({ ...s, _rank: i + 1 }))}
            height={topRowHeight - 2}
            focused={focusedPanel === 0}
            emptyText="No signals — run screener"
            renderCell={(sig, colId) => {
              switch (colId) {
                case "rank":
                  return { text: String(sig._rank), style: { fg: colors.FG_DIM } };
                case "symbol":
                  return { text: truncate(sig.symbol, 10), style: { fg: colors.FG_AMBER, bold: true } };
                case "score":
                  return {
                    text: sig.score.toFixed(3),
                    style: {
                      fg: sig.score > 0 ? colors.GAIN_HI : sig.score < 0 ? colors.LOSS : colors.FG_SECONDARY,
                    },
                  };
                case "signal_type":
                  return { text: truncate(sig.signal_type, 14), style: { fg: colors.CYAN } };
                case "strength":
                  return { text: sig.strength.toFixed(2), style: { fg: colors.FG_PRIMARY } };
                case "confidence":
                  return { text: sig.confidence.toFixed(2), style: { fg: colors.FG_PRIMARY } };
                case "direction":
                  return sig.score > 0
                    ? { text: "\u25b2 LONG", style: { fg: colors.GAIN_HI, bold: true } }
                    : { text: "\u2014", style: { fg: colors.FG_DIM } };
                default:
                  return { text: "" };
              }
            }}
          />
        </BloombergPanel>
      </box>

      {/* ── Bottom: Calendar | Sector Heatmap ────────────────────────────── */}
      <box flexDirection="row" height={bottomRowHeight}>

        {/* Corporate Actions Calendar */}
        <BloombergPanel
          title="CORPORATE ACTIONS \u2014 Upcoming"
          focused={focusedPanel === 1}
          flexGrow={2}
          subtitle={calLoading ? "Loading..." : undefined}
        >
          <DataTable<CalendarRow>
            columns={calendarCols}
            data={calendarRows}
            height={bottomRowHeight - 2}
            focused={focusedPanel === 1}
            emptyText="No upcoming corporate actions"
            renderCell={(row, colId) => {
              switch (colId) {
                case "symbol":
                  return { text: truncate(row.symbol, 10), style: { fg: colors.FG_AMBER, bold: true } };
                case "bookclose_date":
                  return { text: row.bookclose_date.slice(0, 10), style: { fg: colors.FG_PRIMARY } };
                case "days_to": {
                  const urgent = row.days_to <= 7;
                  const soon = row.days_to <= 14;
                  return {
                    text: `${row.days_to}d`,
                    style: {
                      fg: urgent ? colors.YELLOW : soon ? colors.YELLOW : colors.FG_PRIMARY,
                      bold: urgent,
                    },
                  };
                }
                case "cash_dividend":
                  return row.cash_dividend_pct >= 5
                    ? { text: `${row.cash_dividend_pct.toFixed(1)}%`, style: { fg: colors.GAIN_HI, bold: true } }
                    : { text: row.cash_dividend_pct > 0 ? `${row.cash_dividend_pct.toFixed(1)}%` : "\u2014", style: { fg: colors.FG_DIM } };
                case "bonus_share":
                  return row.bonus_share_pct >= 10
                    ? { text: `${row.bonus_share_pct.toFixed(1)}%`, style: { fg: colors.GAIN_HI, bold: true } }
                    : { text: row.bonus_share_pct > 0 ? `${row.bonus_share_pct.toFixed(1)}%` : "\u2014", style: { fg: colors.FG_DIM } };
                case "right_share":
                  return {
                    text: row.right_share_ratio || "\u2014",
                    style: { fg: row.right_share_ratio ? colors.FG_PRIMARY : colors.FG_DIM },
                  };
                case "buy_by":
                  return { text: row.buy_by, style: { fg: colors.CYAN } };
                default:
                  return { text: "" };
              }
            }}
          />
        </BloombergPanel>

        {/* Sector Performance Heatmap */}
        <BloombergPanel
          title="SECTOR PERFORMANCE"
          focused={focusedPanel === 2}
          flexGrow={1}
        >
          <scrollbox height={bottomRowHeight - 2}>
            {sectorHeat.length === 0 ? (
              <text fg={colors.FG_DIM}>  No sector data available</text>
            ) : (
              <box flexDirection="column" paddingLeft={1} paddingTop={1}>
                {sectorHeat.map((sector, i) => {
                  const fg = heatColor(sector.avgChg);
                  const blocks = heatBlocks(sector.avgChg, maxAbsChg);
                  return (
                    <box key={i} height={1} flexDirection="row">
                      <text fg={fg}>{blocks} </text>
                      <text fg={fg} attributes={TextAttributes.BOLD}>
                        {sector.avgChg >= 0 ? "+" : ""}
                        {sector.avgChg.toFixed(1)}%
                      </text>
                      <text fg={colors.FG_PRIMARY}>{"  "}{truncate(sector.name, 16)}</text>
                      <text fg={colors.FG_DIM}>{"  "}{sector.count}</text>
                    </box>
                  );
                })}
                {/* Summary line */}
                <box height={1} paddingTop={1}>
                  <text fg={colors.FG_DIM}>
                    {"  "}
                    {sectorHeat.reduce((s, r) => s + r.count, 0)} signals across{" "}
                    {sectorHeat.length} types
                  </text>
                </box>
              </box>
            )}
          </scrollbox>
        </BloombergPanel>
      </box>
    </box>
  );
}
