// Formatting helpers for market data display

import { GAIN, GAIN_HI, LOSS, LOSS_HI, UNCHANGED, FG_AMBER, FG_DIM, FG_SECONDARY } from "../../theme/colors";

// Format price change with color
export function chgColor(value: number): string {
  if (value > 0) return GAIN;
  if (value < 0) return LOSS;
  return UNCHANGED;
}

// Format change text: "+5.23%" or "-2.10%"
export function fmtChg(value: number): string {
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(2)}%`;
}

// Format price: "1,234.56"
export function fmtPrice(value: number): string {
  return value.toLocaleString("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

// Format volume: "1.2M" or "456K" or "1,234"
export function fmtVol(value: number): string {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}K`;
  return value.toLocaleString("en-US");
}

// Format NPR: "Rs 1.2M" or "Rs 456K"
export function fmtNpr(value: number): string {
  if (value >= 10_000_000) return `Rs ${(value / 10_000_000).toFixed(2)}Cr`;
  if (value >= 100_000) return `Rs ${(value / 100_000).toFixed(2)}L`;
  if (value >= 1_000) return `Rs ${(value / 1_000).toFixed(1)}K`;
  return `Rs ${value.toFixed(0)}`;
}

// Format P&L with sign
export function fmtPnl(value: number): string {
  const sign = value > 0 ? "+" : "";
  return `${sign}${fmtPrice(value)}`;
}

// Pad string to width (right-aligned for numbers)
export function padRight(s: string, width: number): string {
  return s.padEnd(width);
}

export function padLeft(s: string, width: number): string {
  return s.padStart(width);
}

// Truncate string with ellipsis
export function truncate(s: string, maxLen: number): string {
  if (s.length <= maxLen) return s;
  return s.slice(0, maxLen - 1) + "…";
}

// Format date: "2024-01-15" → "Jan 15"
export function fmtDateShort(dateStr: string): string {
  const d = new Date(dateStr);
  const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  return `${months[d.getMonth()]} ${d.getDate()}`;
}

// Format timestamp: "15:23:45"
export function fmtTime(date: Date = new Date()): string {
  return date.toLocaleTimeString("en-US", { hour12: false });
}

// Verdict color
export function verdictColor(verdict: string): string {
  switch (verdict.toUpperCase()) {
    case "BUY":
    case "APPROVE":
      return GAIN_HI;
    case "SELL":
    case "REJECT":
      return LOSS_HI;
    case "HOLD":
      return FG_AMBER;
    case "REVIEW":
      return "#bb88ff";
    default:
      return FG_SECONDARY;
  }
}
