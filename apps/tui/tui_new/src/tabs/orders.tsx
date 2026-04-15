// Tab: Orders — Order entry form, Daily orders, Order history

import { useState } from "react";
import { useTerminalDimensions } from "@opentui/react";
import { TextAttributes } from "@opentui/core";
import { useDailyOrders, useOrderHistory } from "../data/hooks";
import { api } from "../data/api-client";
import { BloombergPanel } from "../components/ui/panel";
import { DataTable, type Column, type CellStyle } from "../components/ui/data-table";
import * as colors from "../theme/colors";
import { fmtPrice, truncate } from "../components/ui/helpers";

// ── Column definitions ──

const dailyOrderCols: Column[] = [
  { id: "symbol", label: "SYMBOL", width: 10, align: "left" },
  { id: "side", label: "SIDE", width: 5, align: "left" },
  { id: "qty", label: "QTY", width: 8, align: "right" },
  { id: "price", label: "PRICE", width: 10, align: "right" },
  { id: "status", label: "STATUS", width: 10, align: "left" },
];

const historyCols: Column[] = [
  { id: "timestamp", label: "DATE", width: 12, align: "left" },
  { id: "symbol", label: "SYMBOL", width: 10, align: "left" },
  { id: "side", label: "SIDE", width: 5, align: "left" },
  { id: "qty", label: "QTY", width: 8, align: "right" },
  { id: "price", label: "PRICE", width: 10, align: "right" },
  { id: "status", label: "STATUS", width: 10, align: "left" },
];

function statusColor(status: string): string {
  switch (status.toUpperCase()) {
    case "FILLED":
    case "COMPLETE":
      return colors.GAIN;
    case "PENDING":
    case "OPEN":
      return colors.YELLOW;
    case "CANCELLED":
    case "REJECTED":
    case "FAILED":
      return colors.LOSS;
    case "PARTIAL":
      return colors.ORANGE;
    default:
      return colors.FG_SECONDARY;
  }
}

function sideColor(side: string): string {
  return side === "BUY" ? colors.GAIN_HI : colors.LOSS_HI;
}

export function OrdersTab() {
  const { data: dailyOrders, loading: dLoading, refresh: refreshDaily } = useDailyOrders();
  const { data: orderHistory, loading: hLoading, refresh: refreshHistory } = useOrderHistory();
  const { width, height } = useTerminalDimensions();

  // Form state
  const [side, setSide] = useState<"BUY" | "SELL">("BUY");
  const [symbol, setSymbol] = useState("");
  const [qty, setQty] = useState("");
  const [price, setPrice] = useState("");
  const [slippage, setSlippage] = useState("0.5");
  const [submitting, setSubmitting] = useState(false);
  const [statusMsg, setStatusMsg] = useState("");

  // Focus tracking
  const [focusedField, setFocusedField] = useState(0); // 0=side, 1=symbol, 2=qty, 3=price, 4=slippage, 5=submit
  const [focusedPanel, setFocusedPanel] = useState(0); // 0=form, 1=daily, 2=history

  const formHeight = 7;
  const availableHeight = height - 4;
  const tableHeight = availableHeight - formHeight;
  const bottomHalf = Math.max(8, tableHeight);

  const handleSubmit = async () => {
    if (!symbol || !qty || !price) {
      setStatusMsg("Fill all fields");
      return;
    }

    const qtyNum = parseInt(qty, 10);
    const priceNum = parseFloat(price);
    const slipNum = parseFloat(slippage) || 0;

    if (isNaN(qtyNum) || isNaN(priceNum) || qtyNum <= 0 || priceNum <= 0) {
      setStatusMsg("Invalid qty/price");
      return;
    }

    setSubmitting(true);
    setStatusMsg("Submitting...");

    try {
      const result = await api.submitOrder({
        side,
        symbol: symbol.toUpperCase(),
        qty: qtyNum,
        price: priceNum,
        slippage: slipNum,
      });
      setStatusMsg(`Order placed: ${result.order_id}`);
      setSymbol("");
      setQty("");
      setPrice("");
      refreshDaily();
      refreshHistory();
    } catch (err: any) {
      setStatusMsg(`Error: ${err.message}`);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <box flexDirection="column" flexGrow={1} backgroundColor={colors.BG_BASE}>
      {/* Top: Order Entry Form */}
      <BloombergPanel
        title="ORDER ENTRY"
        subtitle={statusMsg || undefined}
        focused={focusedPanel === 0}
        height={formHeight}
      >
        <box flexDirection="column" paddingLeft={1} paddingRight={1}>
          {/* Row 1: Side toggle + Symbol */}
          <box flexDirection="row" height={1}>
            <text fg={colors.FG_SECONDARY}>Side: </text>
            <box
              backgroundColor={side === "BUY" ? colors.GAIN : colors.BG_SURFACE}
              paddingLeft={1}
              paddingRight={1}
              height={1}
            >
              <text
                fg={side === "BUY" ? colors.BG_BASE : colors.FG_DIM}
                attributes={side === "BUY" ? TextAttributes.BOLD : 0}
              >
                BUY
              </text>
            </box>
            <text fg={colors.FG_DIM}>/</text>
            <box
              backgroundColor={side === "SELL" ? colors.LOSS : colors.BG_SURFACE}
              paddingLeft={1}
              paddingRight={1}
              height={1}
            >
              <text
                fg={side === "SELL" ? colors.BG_BASE : colors.FG_DIM}
                attributes={side === "SELL" ? TextAttributes.BOLD : 0}
              >
                SELL
              </text>
            </box>
            <text fg={colors.FG_DIM}>  </text>
            <text fg={colors.FG_SECONDARY}>Symbol: </text>
            <input
              placeholder="e.g. NABIL"
              onInput={(val: string) => setSymbol(val)}
              focused={focusedField === 1 && focusedPanel === 0}
            />
          </box>

          {/* Row 2: Qty, Price, Slippage, Submit */}
          <box flexDirection="row" height={1}>
            <text fg={colors.FG_SECONDARY}>Qty: </text>
            <input
              placeholder="100"
              onInput={(val: string) => setQty(val)}
              focused={focusedField === 2 && focusedPanel === 0}
            />
            <text fg={colors.FG_DIM}>  </text>
            <text fg={colors.FG_SECONDARY}>Price: </text>
            <input
              placeholder="1000.00"
              onInput={(val: string) => setPrice(val)}
              focused={focusedField === 3 && focusedPanel === 0}
            />
            <text fg={colors.FG_DIM}>  </text>
            <text fg={colors.FG_SECONDARY}>Slip%: </text>
            <input
              placeholder="0.5"
              onInput={(val: string) => setSlippage(val)}
              focused={focusedField === 4 && focusedPanel === 0}
            />
            <text fg={colors.FG_DIM}>  </text>
            <box
              backgroundColor={submitting ? colors.BG_SURFACE : colors.BLUE}
              paddingLeft={2}
              paddingRight={2}
              height={1}
            >
              <text
                fg={colors.BG_BASE}
                attributes={TextAttributes.BOLD}
              >
                {submitting ? "..." : "SUBMIT"}
              </text>
            </box>
          </box>

          {/* Row 3: Status message */}
          {statusMsg ? (
            <box height={1}>
              <text fg={statusMsg.startsWith("Error") ? colors.LOSS : colors.GAIN}>
                {statusMsg}
              </text>
            </box>
          ) : (
            <box height={1}>
              <text fg={colors.FG_DIM}>
                Tab to switch fields | Enter to submit
              </text>
            </box>
          )}
        </box>
      </BloombergPanel>

      {/* Bottom row: Daily Orders | Order History */}
      <box flexDirection="row" height={bottomHalf}>
        {/* Bottom-left: Today's Orders */}
        <BloombergPanel
          title="DAILY ORDERS"
          subtitle={`${(dailyOrders ?? []).length} orders`}
          focused={focusedPanel === 1}
          flexGrow={1}
        >
          <DataTable
            columns={dailyOrderCols}
            data={dailyOrders ?? []}
            height={bottomHalf - 3}
            focused={focusedPanel === 1}
            emptyText="No orders today"
            renderCell={(item, colId) => {
              switch (colId) {
                case "symbol":
                  return { text: truncate(item.symbol, 10), style: { fg: colors.FG_AMBER, bold: true } };
                case "side":
                  return { text: item.side, style: { fg: sideColor(item.side), bold: true } };
                case "qty":
                  return { text: String(item.qty), style: { fg: colors.FG_PRIMARY } };
                case "price":
                  return { text: fmtPrice(item.price), style: { fg: colors.FG_PRIMARY } };
                case "status":
                  return { text: truncate(item.status, 10), style: { fg: statusColor(item.status) } };
                default:
                  return { text: "" };
              }
            }}
          />
        </BloombergPanel>

        {/* Bottom-right: Order History */}
        <BloombergPanel
          title="ORDER HISTORY"
          subtitle={`${(orderHistory ?? []).length} total`}
          focused={focusedPanel === 2}
          flexGrow={1}
        >
          <DataTable
            columns={historyCols}
            data={orderHistory ?? []}
            height={bottomHalf - 3}
            focused={focusedPanel === 2}
            emptyText="No order history"
            renderCell={(item, colId) => {
              switch (colId) {
                case "timestamp": {
                  const dateStr = item.timestamp.slice(0, 10);
                  return { text: dateStr, style: { fg: colors.FG_DIM } };
                }
                case "symbol":
                  return { text: truncate(item.symbol, 10), style: { fg: colors.FG_AMBER, bold: true } };
                case "side":
                  return { text: item.side, style: { fg: sideColor(item.side), bold: true } };
                case "qty":
                  return { text: String(item.qty), style: { fg: colors.FG_PRIMARY } };
                case "price":
                  return { text: fmtPrice(item.filled_price ?? item.price), style: { fg: colors.FG_PRIMARY } };
                case "status":
                  return { text: truncate(item.status, 10), style: { fg: statusColor(item.status) } };
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
