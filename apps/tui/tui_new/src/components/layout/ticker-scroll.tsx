// Scrolling news ticker — single line at bottom of header area

import { useState, useEffect } from "react";
import * as colors from "../../theme/colors";

const DEFAULT_ITEMS = [
  "NEPSE TUI v1.0 -- Paper Trading System",
  "Sun-Thu trading hours: 11:00-15:00 NPT",
  "Use Ctrl+P for command palette",
  "Press / for quick symbol lookup",
];

interface TickerScrollProps {
  items?: string[];
  intervalMs?: number;
}

export function TickerScroll({
  items = DEFAULT_ITEMS,
  intervalMs = 4000,
}: TickerScrollProps) {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [offset, setOffset] = useState(0);

  const displayItems = items.length > 0 ? items : DEFAULT_ITEMS;
  const currentText = displayItems[currentIndex % displayItems.length] ?? "";

  // Cycle through items
  useEffect(() => {
    if (displayItems.length <= 1) return;
    const timer = setInterval(() => {
      setCurrentIndex((i) => (i + 1) % displayItems.length);
    }, intervalMs);
    return () => clearInterval(timer);
  }, [displayItems.length, intervalMs]);

  // Scroll text within the line
  useEffect(() => {
    setOffset(0);
    const timer = setInterval(() => {
      setOffset((o) => o + 1);
    }, 200);
    return () => clearInterval(timer);
  }, [currentIndex]);

  // Build scrolling display
  const padded = `     ${currentText}     `;
  const scrollPos = offset % padded.length;
  const display = (padded.slice(scrollPos) + padded).slice(0, 80);

  return (
    <box height={1} paddingLeft={1} paddingRight={1} flexDirection="row" gap={1}>
      <text fg={colors.FG_AMBER}>
        {">>>"}
      </text>
      <text fg={colors.FG_SECONDARY}>{display}</text>
    </box>
  );
}
