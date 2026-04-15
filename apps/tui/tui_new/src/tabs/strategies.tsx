// Tab 0: Strategies — Strategy configuration and backtesting

import { useState } from "react";
import { TextAttributes } from "@opentui/core";
import { api } from "../data/api-client";
import { BloombergPanel } from "../components/ui/panel";
import { DataTable, type Column, type CellStyle } from "../components/ui/data-table";
import * as colors from "../theme/colors";
import { fmtPrice, fmtChg } from "../components/ui/helpers";
import type { StrategyConfig, BacktestResult } from "../data/types";

// ── Column definitions ──

const strategyCols: Column[] = [
  { id: "name", label: "STRATEGY", width: 16, align: "left" },
  { id: "description", label: "DESCRIPTION", width: 22, align: "left" },
  { id: "signals", label: "SIGNALS", width: 10, align: "right" },
  { id: "holding", label: "HOLD", width: 6, align: "right" },
  { id: "max_pos", label: "MAX", width: 5, align: "right" },
];

// Default strategies when API unavailable
const DEFAULT_STRATEGIES: StrategyConfig[] = [
  {
    name: "C31 Regime",
    description: "Regime-adaptive sector limits",
    signal_types: ["quality", "volume", "low_vol", "mean_rev", "qf", "xsec_mom"],
    holding_days: 40,
    max_positions: 5,
    stop_loss_pct: 15,
    trailing_stop_pct: 10,
    sector_limit: 0.5,
  },
  {
    name: "C5 Baseline",
    description: "QF + cross-sectional momentum",
    signal_types: ["quality", "volume", "low_vol", "mean_rev", "qf", "xsec_mom"],
    holding_days: 40,
    max_positions: 5,
    stop_loss_pct: 15,
    trailing_stop_pct: 10,
    sector_limit: 0.4,
  },
];

export function StrategiesTab() {
  const [strategies, setStrategies] = useState<StrategyConfig[]>(DEFAULT_STRATEGIES);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [focusedPanel, setFocusedPanel] = useState(0);

  // Backtest state
  const [btStartDate, setBtStartDate] = useState("2024-01-01");
  const [btEndDate, setBtEndDate] = useState("2025-12-31");
  const [btCapital, setBtCapital] = useState("1000000");
  const [btRunning, setBtRunning] = useState(false);
  const [btResult, setBtResult] = useState<BacktestResult | null>(null);
  const [btError, setBtError] = useState("");

  const selected = strategies[selectedIndex] ?? null;

  // Load strategies from API
  useState(() => {
    api.getStrategies().then(setStrategies).catch(() => {});
  });

  async function runBacktest() {
    if (!selected || btRunning) return;
    setBtRunning(true);
    setBtResult(null);
    setBtError("");
    try {
      const result = await api.runBacktest(selected);
      setBtResult(result);
    } catch (err: any) {
      setBtError(err.message || "Backtest failed");
    } finally {
      setBtRunning(false);
    }
  }

  return (
    <box flexDirection="column" flexGrow={1} backgroundColor={colors.BG_BASE}>
      {/* Top: Strategy list | Strategy detail */}
      <box flexDirection="row" flexGrow={1}>
        {/* Left: Strategy list */}
        <BloombergPanel
          title="STRATEGIES"
          subtitle={`${strategies.length} configs`}
          focused={focusedPanel === 0}
          flexGrow={1}
        >
          <DataTable
            columns={strategyCols}
            data={strategies}
            height={16}
            focused={focusedPanel === 0}
            selectedIndex={selectedIndex}
            onSelect={setSelectedIndex}
            emptyText="No strategies loaded"
            renderCell={(item: StrategyConfig, colId) => {
              switch (colId) {
                case "name":
                  return {
                    text: item.name.slice(0, 16),
                    style: { fg: colors.FG_AMBER, bold: true },
                  };
                case "description":
                  return {
                    text: item.description.slice(0, 22),
                    style: { fg: colors.FG_SECONDARY },
                  };
                case "signals":
                  return {
                    text: `${item.signal_types.length} sigs`,
                    style: { fg: colors.CYAN },
                  };
                case "holding":
                  return {
                    text: `${item.holding_days}d`,
                    style: { fg: colors.FG_PRIMARY },
                  };
                case "max_pos":
                  return {
                    text: String(item.max_positions),
                    style: { fg: colors.FG_PRIMARY },
                  };
                default:
                  return { text: "" };
              }
            }}
          />
        </BloombergPanel>

        {/* Right: Strategy detail */}
        <BloombergPanel
          title="PARAMETERS"
          subtitle={selected?.name ?? ""}
          focused={focusedPanel === 1}
          flexGrow={1}
        >
          {selected ? (
            <box flexDirection="column" paddingLeft={1} paddingTop={1} gap={0}>
              <box flexDirection="row" height={1}>
                <text fg={colors.FG_SECONDARY}>{"Name:            ".slice(0, 18)}</text>
                <text fg={colors.FG_AMBER} attributes={TextAttributes.BOLD}>
                  {selected.name}
                </text>
              </box>
              <box flexDirection="row" height={1}>
                <text fg={colors.FG_SECONDARY}>{"Description:     ".slice(0, 18)}</text>
                <text fg={colors.FG_PRIMARY}>{selected.description}</text>
              </box>
              <box flexDirection="row" height={1}>
                <text fg={colors.FG_SECONDARY}>{"Signals:         ".slice(0, 18)}</text>
                <text fg={colors.CYAN}>{selected.signal_types.join(", ")}</text>
              </box>
              <box flexDirection="row" height={1}>
                <text fg={colors.FG_SECONDARY}>{"Holding Days:    ".slice(0, 18)}</text>
                <text fg={colors.FG_PRIMARY}>{selected.holding_days}</text>
              </box>
              <box flexDirection="row" height={1}>
                <text fg={colors.FG_SECONDARY}>{"Max Positions:   ".slice(0, 18)}</text>
                <text fg={colors.FG_PRIMARY}>{selected.max_positions}</text>
              </box>
              <box flexDirection="row" height={1}>
                <text fg={colors.FG_SECONDARY}>{"Stop Loss:       ".slice(0, 18)}</text>
                <text fg={colors.LOSS}>{selected.stop_loss_pct}%</text>
              </box>
              <box flexDirection="row" height={1}>
                <text fg={colors.FG_SECONDARY}>{"Trailing Stop:   ".slice(0, 18)}</text>
                <text fg={colors.ORANGE}>{selected.trailing_stop_pct}%</text>
              </box>
              <box flexDirection="row" height={1}>
                <text fg={colors.FG_SECONDARY}>{"Sector Limit:    ".slice(0, 18)}</text>
                <text fg={colors.PURPLE}>{(selected.sector_limit * 100).toFixed(0)}%</text>
              </box>
            </box>
          ) : (
            <box flexGrow={1} justifyContent="center" alignItems="center">
              <text fg={colors.FG_DIM}>Select a strategy to view parameters</text>
            </box>
          )}
        </BloombergPanel>
      </box>

      {/* Bottom: Backtest runner */}
      <BloombergPanel title="BACKTEST" focused={focusedPanel === 2}>
        <box flexDirection="row" paddingLeft={1} paddingTop={1} gap={3} height={5}>
          {/* Input fields */}
          <box flexDirection="column" gap={0}>
            <box flexDirection="row" height={1} gap={1}>
              <text fg={colors.FG_SECONDARY}>Start:</text>
              <input
                placeholder="2024-01-01"
                onInput={(v: string) => setBtStartDate(v)}
                focused={focusedPanel === 2}
              />
            </box>
            <box flexDirection="row" height={1} gap={1}>
              <text fg={colors.FG_SECONDARY}>End:  </text>
              <input
                placeholder="2025-12-31"
                onInput={(v: string) => setBtEndDate(v)}
                focused={false}
              />
            </box>
            <box flexDirection="row" height={1} gap={1}>
              <text fg={colors.FG_SECONDARY}>Capital:</text>
              <input
                placeholder="1000000"
                onInput={(v: string) => setBtCapital(v)}
                focused={false}
              />
            </box>
          </box>

          {/* Run button */}
          <box flexDirection="column" justifyContent="center">
            <box
              backgroundColor={btRunning ? colors.FG_DIM : colors.BLUE}
              paddingLeft={2}
              paddingRight={2}
            >
              <text fg={colors.BG_BASE} attributes={TextAttributes.BOLD}>
                {btRunning ? "[ RUNNING... ]" : "[ RUN BACKTEST ]"}
              </text>
            </box>
          </box>

          {/* Results */}
          <box flexDirection="column" gap={0} flexGrow={1}>
            {btResult ? (
              <>
                <box flexDirection="row" height={1} gap={2}>
                  <text fg={colors.FG_SECONDARY}>Return:</text>
                  <text fg={colors.priceColor(btResult.total_return)}>
                    {fmtChg(btResult.total_return)}
                  </text>
                  <text fg={colors.FG_SECONDARY}>Sharpe:</text>
                  <text fg={colors.FG_PRIMARY}>{btResult.sharpe_ratio.toFixed(3)}</text>
                </box>
                <box flexDirection="row" height={1} gap={2}>
                  <text fg={colors.FG_SECONDARY}>MaxDD:</text>
                  <text fg={colors.LOSS}>{btResult.max_drawdown.toFixed(2)}%</text>
                  <text fg={colors.FG_SECONDARY}>WinRate:</text>
                  <text fg={colors.GAIN}>{btResult.win_rate.toFixed(1)}%</text>
                </box>
                <box flexDirection="row" height={1} gap={2}>
                  <text fg={colors.FG_SECONDARY}>Trades:</text>
                  <text fg={colors.FG_PRIMARY}>{btResult.total_trades}</text>
                  <text fg={colors.FG_SECONDARY}>Period:</text>
                  <text fg={colors.FG_DIM}>
                    {btResult.start_date} to {btResult.end_date}
                  </text>
                </box>
              </>
            ) : btError ? (
              <text fg={colors.LOSS}>{btError}</text>
            ) : (
              <text fg={colors.FG_DIM}>Configure and run a backtest</text>
            )}
          </box>
        </box>
      </BloombergPanel>
    </box>
  );
}
