// MarketPanel — Pane + DataTable for market data display

import { BloombergPanel } from "./panel";
import { DataTable, type Column, type CellStyle } from "./data-table";
import { fmtPrice, fmtChg, fmtVol, chgColor, truncate } from "./helpers";
import * as colors from "../../theme/colors";
import type { MarketQuote } from "../../data/types";

interface MarketPanelProps {
  title: string;
  data: MarketQuote[];
  focused?: boolean;
  height?: number;
  flexGrow?: number;
}

const COLUMNS: Column[] = [
  { id: "symbol", label: "SYMBOL", width: 9 },
  { id: "ltp", label: "LTP", width: 10, align: "right" },
  { id: "change_pct", label: "CHG%", width: 8, align: "right" },
  { id: "volume", label: "VOL", width: 9, align: "right" },
];

function renderCell(
  item: MarketQuote,
  columnId: string,
): { text: string; style?: CellStyle } {
  switch (columnId) {
    case "symbol":
      return { text: truncate(item.symbol, 9), style: { fg: colors.textBright, bold: true } };
    case "ltp":
      return { text: fmtPrice(item.ltp), style: { fg: colors.text } };
    case "change_pct": {
      const pct = item.change_pct;
      const fg = pct > 0 ? colors.positive : pct < 0 ? colors.negative : colors.neutral;
      return { text: fmtChg(pct), style: { fg, bold: Math.abs(pct) > 2 } };
    }
    case "volume":
      return { text: fmtVol(item.volume), style: { fg: colors.textDim } };
    default:
      return { text: "" };
  }
}

export function MarketPanel({ title, data, focused, height, flexGrow }: MarketPanelProps) {
  return (
    <BloombergPanel
      title={title}
      focused={focused}
      flexGrow={flexGrow}
      subtitle={`${data.length}`}
    >
      <DataTable
        columns={COLUMNS}
        data={data}
        renderCell={(item, colId) => renderCell(item, colId)}
        height={height ? height - 2 : 15}
        focused={focused}
        emptyText="No data"
      />
    </BloombergPanel>
  );
}
