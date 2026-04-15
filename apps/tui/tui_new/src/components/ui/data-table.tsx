// DataTable — Gloomberb-style table with keyboard nav and row highlighting

import { useState, useMemo } from "react";
import { useKeyboard } from "@opentui/react";
import { TextAttributes } from "@opentui/core";
import * as colors from "../../theme/colors";

export interface Column {
  id: string;
  label: string;
  width: number;
  align?: "left" | "right" | "center";
}

export interface CellStyle {
  fg?: string;
  bold?: boolean;
  dim?: boolean;
}

export interface DataTableProps<T> {
  columns: Column[];
  data: T[];
  renderCell: (item: T, columnId: string, rowIndex: number) => { text: string; style?: CellStyle };
  selectedIndex?: number;
  onSelect?: (index: number) => void;
  onActivate?: (index: number) => void;
  height?: number;
  focused?: boolean;
  emptyText?: string;
}

function padTo(str: string, width: number, align?: string): string {
  if (str.length >= width) return str.slice(0, width);
  if (align === "right") return str.padStart(width);
  return str.padEnd(width);
}

export function DataTable<T>({
  columns,
  data,
  renderCell,
  selectedIndex: controlledIndex,
  onSelect,
  onActivate,
  height = 20,
  focused = false,
  emptyText = "No data",
}: DataTableProps<T>) {
  const [internalIndex, setInternalIndex] = useState(0);
  const selectedIndex = controlledIndex ?? internalIndex;
  const [scrollOffset, setScrollOffset] = useState(0);

  const visibleRows = height - 1; // minus header row
  const maxScroll = Math.max(0, data.length - visibleRows);

  const adjustedOffset = useMemo(() => {
    let offset = scrollOffset;
    if (selectedIndex < offset) offset = selectedIndex;
    if (selectedIndex >= offset + visibleRows) offset = selectedIndex - visibleRows + 1;
    return Math.min(Math.max(0, offset), maxScroll);
  }, [selectedIndex, scrollOffset, visibleRows, maxScroll]);

  useKeyboard(
    (key) => {
      if (!focused) return;

      let newIndex = selectedIndex;

      if (key.name === "ArrowUp" || key.name === "k") {
        newIndex = Math.max(0, selectedIndex - 1);
      } else if (key.name === "ArrowDown" || key.name === "j") {
        newIndex = Math.min(data.length - 1, selectedIndex + 1);
      } else if (key.name === "Home") {
        newIndex = 0;
      } else if (key.name === "End") {
        newIndex = data.length - 1;
      } else if (key.name === "PageUp") {
        newIndex = Math.max(0, selectedIndex - visibleRows);
      } else if (key.name === "PageDown") {
        newIndex = Math.min(data.length - 1, selectedIndex + visibleRows);
      } else if (key.name === "Return" && onActivate) {
        onActivate(selectedIndex);
        return;
      } else {
        return;
      }

      if (newIndex !== selectedIndex) {
        if (onSelect) onSelect(newIndex);
        else setInternalIndex(newIndex);

        let newOffset = adjustedOffset;
        if (newIndex < newOffset) newOffset = newIndex;
        if (newIndex >= newOffset + visibleRows) newOffset = newIndex - visibleRows + 1;
        setScrollOffset(newOffset);
      }
    },
    { release: false }
  );

  const visibleData = data.slice(adjustedOffset, adjustedOffset + visibleRows);

  // Header text
  const headerText = columns
    .map((col) => padTo(col.label, col.width, col.align))
    .join(" ");

  return (
    <box flexDirection="column" height={height}>
      {/* Column header */}
      <box height={1} paddingLeft={1} backgroundColor={colors.panel}>
        {columns.map((col, ci) => (
          <box key={col.id} width={col.width + 1}>
            <text
              fg={colors.text}
              attributes={TextAttributes.BOLD}
            >
              {padTo(col.label, col.width, col.align)}
              {ci < columns.length - 1 ? " " : ""}
            </text>
          </box>
        ))}
      </box>

      {/* Rows */}
      {data.length === 0 ? (
        <box justifyContent="center" alignItems="center" flexGrow={1}>
          <text fg={colors.textMuted}>{emptyText}</text>
        </box>
      ) : (
        visibleData.map((item, vi) => {
          const actualIndex = adjustedOffset + vi;
          const isSelected = actualIndex === selectedIndex && focused;
          const rowBg = isSelected ? colors.selected : colors.bg;

          const cells = columns.map((col) => {
            const { text: cellText, style } = renderCell(item, col.id, actualIndex);
            return { text: padTo(cellText, col.width, col.align), style };
          });

          return (
            <box
              key={actualIndex}
              backgroundColor={rowBg}
              height={1}
              paddingLeft={1}
              flexDirection="row"
            >
              {cells.map((cell, ci) => {
                let attrs = 0;
                if (cell.style?.bold) attrs |= TextAttributes.BOLD;
                if (cell.style?.dim) attrs |= TextAttributes.DIM;

                // Selected row overrides text color
                const fg = isSelected
                  ? colors.selectedText
                  : (cell.style?.fg || colors.text);

                return (
                  <box key={ci} width={columns[ci].width + 1}>
                    <text fg={fg} attributes={attrs}>
                      {cell.text}
                      {ci < cells.length - 1 ? " " : ""}
                    </text>
                  </box>
                );
              })}
            </box>
          );
        })
      )}
    </box>
  );
}
