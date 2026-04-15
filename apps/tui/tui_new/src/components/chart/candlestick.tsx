// CandlestickChart — braille OHLCV chart with price axis, date axis, and volume bars

import { useMemo } from "react";
import { TextAttributes } from "@opentui/core";
import { renderCandlestickBraille } from "./braille";
import * as colors from "../../theme/colors";
import type { OHLCVBar } from "../../data/types";
import { fmtPrice, fmtDateShort } from "../ui/helpers";

interface CandlestickChartProps {
  data: OHLCVBar[];
  width: number;
  height: number; // terminal rows total (price + volume + axes)
  timeframe: string;
}

export function CandlestickChart({
  data,
  width,
  height,
  timeframe,
}: CandlestickChartProps) {
  const axisWidth = 10; // right-side price axis
  const chartWidth = Math.max(4, width - axisWidth - 2);
  const dateAxisRows = 1;
  const volumeRows = 2;
  const priceRows = Math.max(2, height - volumeRows - dateAxisRows - 1);

  const rendered = useMemo(() => {
    if (data.length === 0) return null;

    const bars = data.map((d) => ({
      open: d.open,
      high: d.high,
      low: d.low,
      close: d.close,
      volume: d.volume,
    }));

    return renderCandlestickBraille(bars, chartWidth, priceRows);
  }, [data, chartWidth, priceRows]);

  if (!rendered || data.length === 0) {
    return (
      <box
        flexDirection="column"
        height={height}
        justifyContent="center"
        alignItems="center"
      >
        <text fg={colors.FG_DIM}>No chart data</text>
      </box>
    );
  }

  // Build price axis labels — spread evenly across price rows
  const labelCount = Math.min(priceRows, 5);
  const priceLabels: { row: number; label: string }[] = [];
  for (let i = 0; i < labelCount; i++) {
    const row = Math.round((i / (labelCount - 1)) * (priceRows - 1));
    const price =
      rendered.priceMax -
      (row / (priceRows - 1)) * (rendered.priceMax - rendered.priceMin);
    priceLabels.push({ row, label: fmtPrice(price) });
  }

  // Build date labels along bottom
  const dateLabels: string[] = [];
  const labelSpacing = Math.max(1, Math.floor(data.length / 5));
  for (let i = 0; i < data.length; i += labelSpacing) {
    dateLabels.push(fmtDateShort(data[i].date));
  }
  const dateRow = dateLabels
    .map((d) => d.padEnd(Math.floor(chartWidth / dateLabels.length)))
    .join("")
    .slice(0, chartWidth);

  // Determine candle colors per bar for the last bar (used for overall trend display)
  const lastBar = data[data.length - 1];
  const bullish = lastBar.close >= lastBar.open;
  const candleColor = bullish ? colors.GAIN : colors.LOSS;
  const volColor = colors.FG_SECONDARY;

  return (
    <box flexDirection="column" height={height}>
      {/* Timeframe label */}
      <box height={1} paddingLeft={1} flexDirection="row">
        <text fg={colors.FG_AMBER} attributes={TextAttributes.BOLD}>
          {timeframe}
        </text>
        <text fg={colors.FG_DIM}>
          {"  "}
          {data.length} bars
        </text>
      </box>

      {/* Price chart + axis */}
      {rendered.lines.map((line, rowIdx) => {
        const priceLabel = priceLabels.find((p) => p.row === rowIdx);
        const axisText = priceLabel
          ? priceLabel.label.padStart(axisWidth)
          : " ".repeat(axisWidth);

        return (
          <box key={`p-${rowIdx}`} height={1} flexDirection="row">
            <text fg={candleColor}>{line}</text>
            <text fg={colors.FG_DIM}> {"\u2502"}</text>
            <text fg={colors.FG_SECONDARY}>{axisText}</text>
          </box>
        );
      })}

      {/* Separator between price and volume */}
      <box height={1} flexDirection="row">
        <text fg={colors.FG_DIM}>
          {"\u2500".repeat(chartWidth)} {"\u253c"}
          {"\u2500".repeat(axisWidth)}
        </text>
      </box>

      {/* Volume bars */}
      {rendered.volumeLines.map((line, rowIdx) => (
        <box key={`v-${rowIdx}`} height={1} flexDirection="row">
          <text fg={volColor}>{line}</text>
          <text fg={colors.FG_DIM}> {"\u2502"}</text>
          <text fg={colors.FG_DIM}>{"".padStart(axisWidth)}</text>
        </box>
      ))}

      {/* Date axis */}
      <box height={1} paddingLeft={1}>
        <text fg={colors.FG_DIM}>{dateRow}</text>
      </box>
    </box>
  );
}
