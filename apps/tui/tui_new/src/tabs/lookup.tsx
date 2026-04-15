// Tab 4: Lookup — Deep-dive on a single symbol with chart, OHLCV, stats, financials

import { useState } from "react";
import { useTerminalDimensions, useKeyboard } from "@opentui/react";
import { TextAttributes } from "@opentui/core";
import { useLookup } from "../data/hooks";
import { useAppState, useDispatch } from "../state/app-context";
import { BloombergPanel } from "../components/ui/panel";
import { DataTable, type Column, type CellStyle } from "../components/ui/data-table";
import { CandlestickChart } from "../components/chart/candlestick";
import * as colors from "../theme/colors";
import {
  fmtPrice,
  fmtChg,
  fmtVol,
  fmtNpr,
  chgColor,
  truncate,
  fmtDateShort,
} from "../components/ui/helpers";

// ── OHLCV table columns ────────────────────────────────────────────

const ohlcvColumns: Column[] = [
  { id: "date", label: "Date", width: 8 },
  { id: "open", label: "Open", width: 9, align: "right" },
  { id: "high", label: "High", width: 9, align: "right" },
  { id: "low", label: "Low", width: 9, align: "right" },
  { id: "close", label: "Close", width: 9, align: "right" },
  { id: "volume", label: "Volume", width: 8, align: "right" },
];

// ── Financial metrics table columns ────────────────────────────────

const financialColumns: Column[] = [
  { id: "metric", label: "Metric", width: 20 },
  { id: "value", label: "Value", width: 14, align: "right" },
];

// ── Corporate actions table columns ────────────────────────────────

const corpActionColumns: Column[] = [
  { id: "fiscal_year", label: "FY", width: 8 },
  { id: "bookclose", label: "Book Close", width: 10 },
  { id: "cash", label: "Cash %", width: 8, align: "right" },
  { id: "bonus", label: "Bonus %", width: 8, align: "right" },
  { id: "right", label: "Right", width: 8 },
];

// ── Focus zones for keyboard navigation ────────────────────────────

type FocusZone = "chart" | "ohlcv" | "stats" | "financials" | "corpActions" | "report";
const ZONES: FocusZone[] = ["chart", "ohlcv", "stats", "financials", "corpActions", "report"];

export function LookupTab() {
  const { width: termW, height: termH } = useTerminalDimensions();
  const appState = useAppState();
  const dispatch = useDispatch();
  const symbol = appState.lookupSymbol;
  const timeframe = appState.chartTimeframe;

  const { data, loading, error } = useLookup(symbol);
  const [focusIdx, setFocusIdx] = useState(0);
  const [ohlcvSelected, setOhlcvSelected] = useState(0);
  const [finSelected, setFinSelected] = useState(0);
  const [corpSelected, setCorpSelected] = useState(0);
  const [reportScroll, setReportScroll] = useState(0);

  const focusZone = ZONES[focusIdx];

  // Keyboard: Tab to cycle focus, / to enter symbol
  useKeyboard(
    (key) => {
      if (key.name === "Tab") {
        setFocusIdx((prev) => (prev + 1) % ZONES.length);
      } else if (key.name === "Tab" && key.shift) {
        setFocusIdx((prev) => (prev - 1 + ZONES.length) % ZONES.length);
      }
      // Timeframe switching
      if (focusZone === "chart") {
        if (key.name === "d") dispatch({ type: "SET_CHART_TIMEFRAME", tf: "D" });
        if (key.name === "w") dispatch({ type: "SET_CHART_TIMEFRAME", tf: "W" });
        if (key.name === "m") dispatch({ type: "SET_CHART_TIMEFRAME", tf: "M" });
        if (key.name === "y") dispatch({ type: "SET_CHART_TIMEFRAME", tf: "Y" });
      }
      // Report scrolling
      if (focusZone === "report") {
        if (key.name === "ArrowDown" || key.name === "j") {
          setReportScroll((s) => s + 1);
        } else if (key.name === "ArrowUp" || key.name === "k") {
          setReportScroll((s) => Math.max(0, s - 1));
        }
      }
    },
    { release: false },
  );

  // ── Loading / Error ────────────────────────────────────────────

  if (!data) {
    return (
      <box flexGrow={1} justifyContent="center" alignItems="center">
        <text fg={colors.FG_DIM}>
          {loading ? `Loading ${symbol}...` : `Error: ${error}`}
        </text>
      </box>
    );
  }

  const { detail, ohlcv, financials, corporate_actions, report } = data;
  const availH = termH - 4;
  const leftW = Math.floor(termW * 0.55);
  const rightW = termW - leftW;
  const chartH = Math.min(Math.floor(availH * 0.45), 16);
  const ohlcvH = availH - chartH - 3; // header row

  // Stats entries
  const statEntries: { label: string; value: string }[] = [
    { label: "P/E Ratio", value: detail.pe_ratio != null ? detail.pe_ratio.toFixed(2) : "N/A" },
    { label: "Market Cap", value: detail.market_cap != null ? fmtNpr(detail.market_cap) : "N/A" },
    { label: "52W High", value: detail.high_52w != null ? fmtPrice(detail.high_52w) : "N/A" },
    { label: "52W Low", value: detail.low_52w != null ? fmtPrice(detail.low_52w) : "N/A" },
    { label: "EPS", value: detail.eps != null ? detail.eps.toFixed(2) : "N/A" },
    { label: "Book Value", value: detail.book_value != null ? fmtPrice(detail.book_value) : "N/A" },
  ];

  // Financials as rows
  const finRows = financials.map((row) => {
    const entries = Object.entries(row);
    return { metric: String(entries[0]?.[0] ?? ""), value: String(entries[0]?.[1] ?? "") };
  });

  // Report lines
  const reportLines = report ? report.split("\n") : ["No report available"];
  const visibleReportLines = reportLines.slice(reportScroll, reportScroll + 10);

  return (
    <box flexDirection="column" flexGrow={1} backgroundColor={colors.BG_BASE}>
      {/* ── Symbol Header ────────────────────────────────────── */}
      <box
        height={2}
        backgroundColor={colors.BG_HEADER}
        paddingLeft={2}
        paddingRight={2}
        flexDirection="row"
        alignItems="center"
      >
        <text fg={colors.FG_AMBER} attributes={TextAttributes.BOLD}>
          {detail.symbol}
        </text>
        <text fg={colors.FG_SECONDARY}>{"  "}{detail.name}</text>
        <text fg={colors.FG_DIM}>{"  |  "}{detail.sector}</text>
        <text fg={colors.FG_PRIMARY}>{"    "}LTP: {fmtPrice(detail.ltp)}</text>
        <text fg={chgColor(detail.change)}>
          {"  "}{fmtChg(detail.change_pct)} ({detail.change >= 0 ? "+" : ""}{fmtPrice(detail.change)})
        </text>
      </box>

      {/* ── Main Content: Left + Right ───────────────────────── */}
      <box flexDirection="row" flexGrow={1}>
        {/* ── Left Column ──────────────────────────────────── */}
        <box flexDirection="column" width={leftW}>
          {/* Candlestick Chart */}
          <BloombergPanel
            title="CHART"
            subtitle={`${timeframe} | d/w/m/y to switch`}
            focused={focusZone === "chart"}
            height={chartH}
          >
            <CandlestickChart
              data={ohlcv}
              width={leftW - 4}
              height={chartH - 3}
              timeframe={timeframe}
            />
          </BloombergPanel>

          {/* OHLCV Data Table */}
          <BloombergPanel
            title="OHLCV"
            focused={focusZone === "ohlcv"}
            flexGrow={1}
          >
            <DataTable
              columns={ohlcvColumns}
              data={[...ohlcv].reverse()}
              selectedIndex={ohlcvSelected}
              onSelect={setOhlcvSelected}
              height={ohlcvH}
              focused={focusZone === "ohlcv"}
              emptyText="No price data"
              renderCell={(bar, colId) => {
                switch (colId) {
                  case "date":
                    return { text: fmtDateShort(bar.date), style: { fg: colors.FG_SECONDARY } };
                  case "open":
                    return { text: fmtPrice(bar.open) };
                  case "high":
                    return { text: fmtPrice(bar.high), style: { fg: colors.GAIN } };
                  case "low":
                    return { text: fmtPrice(bar.low), style: { fg: colors.LOSS } };
                  case "close": {
                    const chg = bar.close - bar.open;
                    return { text: fmtPrice(bar.close), style: { fg: chgColor(chg), bold: true } };
                  }
                  case "volume":
                    return { text: fmtVol(bar.volume), style: { fg: colors.CYAN } };
                  default:
                    return { text: "" };
                }
              }}
            />
          </BloombergPanel>
        </box>

        {/* ── Right Column ─────────────────────────────────── */}
        <box flexDirection="column" width={rightW}>
          {/* Stats Panel */}
          <BloombergPanel
            title="STATS"
            focused={focusZone === "stats"}
            height={statEntries.length + 3}
          >
            <box flexDirection="column" paddingLeft={1}>
              {statEntries.map((stat, i) => (
                <box key={i} height={1} flexDirection="row">
                  <text fg={colors.FG_SECONDARY}>
                    {stat.label.padEnd(14)}
                  </text>
                  <text fg={colors.FG_PRIMARY}>{stat.value}</text>
                </box>
              ))}
            </box>
          </BloombergPanel>

          {/* Financial Metrics */}
          <BloombergPanel
            title="FINANCIALS"
            focused={focusZone === "financials"}
            height={Math.min(finRows.length + 3, 10)}
          >
            <DataTable
              columns={financialColumns}
              data={finRows}
              selectedIndex={finSelected}
              onSelect={setFinSelected}
              height={Math.min(finRows.length + 1, 8)}
              focused={focusZone === "financials"}
              emptyText="No financial data"
              renderCell={(row, colId) => {
                if (colId === "metric") {
                  return { text: truncate(row.metric, 20), style: { fg: colors.FG_SECONDARY } };
                }
                return { text: truncate(row.value, 14) };
              }}
            />
          </BloombergPanel>

          {/* Corporate Actions */}
          <BloombergPanel
            title="CORPORATE ACTIONS"
            focused={focusZone === "corpActions"}
            height={Math.min(corporate_actions.length + 3, 8)}
          >
            <DataTable
              columns={corpActionColumns}
              data={corporate_actions}
              selectedIndex={corpSelected}
              onSelect={setCorpSelected}
              height={Math.min(corporate_actions.length + 1, 6)}
              focused={focusZone === "corpActions"}
              emptyText="No corporate actions"
              renderCell={(ca, colId) => {
                switch (colId) {
                  case "fiscal_year":
                    return { text: ca.fiscal_year, style: { fg: colors.FG_AMBER } };
                  case "bookclose":
                    return { text: fmtDateShort(ca.bookclose_date), style: { fg: colors.FG_SECONDARY } };
                  case "cash":
                    return {
                      text: ca.cash_dividend_pct > 0 ? `${ca.cash_dividend_pct}%` : "-",
                      style: { fg: ca.cash_dividend_pct > 0 ? colors.GAIN : colors.FG_DIM },
                    };
                  case "bonus":
                    return {
                      text: ca.bonus_share_pct > 0 ? `${ca.bonus_share_pct}%` : "-",
                      style: { fg: ca.bonus_share_pct > 0 ? colors.CYAN : colors.FG_DIM },
                    };
                  case "right":
                    return {
                      text: ca.right_share_ratio || "-",
                      style: { fg: colors.FG_SECONDARY },
                    };
                  default:
                    return { text: "" };
                }
              }}
            />
          </BloombergPanel>

          {/* Report */}
          <BloombergPanel
            title="REPORT"
            subtitle="j/k to scroll"
            focused={focusZone === "report"}
            flexGrow={1}
          >
            <box flexDirection="column" paddingLeft={1} flexGrow={1}>
              {visibleReportLines.map((line, i) => (
                <box key={i} height={1}>
                  <text fg={colors.FG_PRIMARY}>{truncate(line, rightW - 4)}</text>
                </box>
              ))}
              {reportLines.length > 10 && (
                <box height={1}>
                  <text fg={colors.FG_DIM}>
                    [{reportScroll + 1}-{Math.min(reportScroll + 10, reportLines.length)}/{reportLines.length}]
                  </text>
                </box>
              )}
            </box>
          </BloombergPanel>
        </box>
      </box>
    </box>
  );
}
