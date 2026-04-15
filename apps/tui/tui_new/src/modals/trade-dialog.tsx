// Modal: Trade dialog — Buy/Sell order entry

import { useState } from "react";
import { useKeyboard, useTerminalDimensions } from "@opentui/react";
import { TextAttributes } from "@opentui/core";
import { api } from "../data/api-client";
import * as colors from "../theme/colors";

type TradeField = "side" | "symbol" | "qty" | "price" | "slippage";

const FIELD_ORDER: TradeField[] = ["side", "symbol", "qty", "price", "slippage"];

export function TradeDialog({
  side: initialSide,
  onClose,
}: {
  side: "BUY" | "SELL";
  onClose: () => void;
}) {
  const [side, setSide] = useState<"BUY" | "SELL">(initialSide);
  const [symbol, setSymbol] = useState("");
  const [qty, setQty] = useState("");
  const [price, setPrice] = useState("");
  const [slippage, setSlippage] = useState("0.5");
  const [focusField, setFocusField] = useState<TradeField>("symbol");
  const [status, setStatus] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const { width, height } = useTerminalDimensions();

  const sideColor = side === "BUY" ? colors.GAIN_HI : colors.LOSS_HI;
  const sideBg = side === "BUY" ? "#0a2a0a" : "#2a0a0a";

  useKeyboard(
    (key) => {
      if (key.name === "Escape") {
        onClose();
        return;
      }
      if (key.name === "Tab") {
        const idx = FIELD_ORDER.indexOf(focusField);
        const next = FIELD_ORDER[(idx + 1) % FIELD_ORDER.length];
        setFocusField(next);
        return;
      }
      // Toggle side with space when on side field
      if (focusField === "side" && key.name === "Space") {
        setSide((s) => (s === "BUY" ? "SELL" : "BUY"));
        return;
      }
      if (key.name === "Return" && !submitting) {
        handleSubmit();
      }
    },
    { release: false }
  );

  async function handleSubmit() {
    const qtyNum = parseInt(qty, 10);
    const priceNum = parseFloat(price);
    const slippageNum = parseFloat(slippage);

    if (!symbol.trim()) {
      setStatus("Symbol is required");
      return;
    }
    if (isNaN(qtyNum) || qtyNum <= 0) {
      setStatus("Invalid quantity");
      return;
    }
    if (isNaN(priceNum) || priceNum <= 0) {
      setStatus("Invalid price");
      return;
    }

    setSubmitting(true);
    setStatus("Submitting...");
    try {
      const result = await api.submitOrder({
        side,
        symbol: symbol.trim().toUpperCase(),
        qty: qtyNum,
        price: priceNum,
        slippage: isNaN(slippageNum) ? undefined : slippageNum,
      });
      setStatus(`Order placed: ${result.order_id}`);
      setTimeout(onClose, 1500);
    } catch (err: any) {
      setStatus(`Error: ${err.message}`);
      setSubmitting(false);
    }
  }

  const boxWidth = 44;
  const boxHeight = 18;
  const left = Math.floor((width - boxWidth) / 2);
  const top = Math.floor((height - boxHeight) / 2);

  return (
    <box
      width={width}
      height={height}
      backgroundColor="#00000088"
      position="absolute"
      top={0}
      left={0}
    >
      <box
        position="absolute"
        left={left}
        top={top}
        width={boxWidth}
        height={boxHeight}
        backgroundColor={colors.BG_PANEL}
        borderStyle="single"
        borderColor={sideColor}
        flexDirection="column"
      >
        {/* Title */}
        <box backgroundColor={sideBg} height={1} paddingLeft={1} paddingRight={1}>
          <text fg={sideColor} attributes={TextAttributes.BOLD}>
            :: {side} ORDER
          </text>
        </box>

        <box flexDirection="column" paddingLeft={2} paddingTop={1} gap={0}>
          {/* Side toggle */}
          <box flexDirection="row" height={1} gap={1}>
            <text fg={colors.FG_SECONDARY}>{"Side:     "}</text>
            <box
              backgroundColor={focusField === "side" ? sideBg : colors.BG_SURFACE}
              paddingLeft={1}
              paddingRight={1}
            >
              <text fg={sideColor} attributes={TextAttributes.BOLD}>
                {side}
              </text>
            </box>
            {focusField === "side" && (
              <text fg={colors.FG_DIM}>(Space to toggle)</text>
            )}
          </box>

          {/* Symbol */}
          <box flexDirection="row" height={1} gap={1}>
            <text fg={colors.FG_SECONDARY}>{"Symbol:   "}</text>
            <input
              placeholder="NABIL"
              onInput={(v: string) => setSymbol(v)}
              focused={focusField === "symbol"}
            />
          </box>

          {/* Quantity */}
          <box flexDirection="row" height={1} gap={1}>
            <text fg={colors.FG_SECONDARY}>{"Quantity: "}</text>
            <input
              placeholder="100"
              onInput={(v: string) => setQty(v)}
              focused={focusField === "qty"}
            />
          </box>

          {/* Price */}
          <box flexDirection="row" height={1} gap={1}>
            <text fg={colors.FG_SECONDARY}>{"Price:    "}</text>
            <input
              placeholder="1250.00"
              onInput={(v: string) => setPrice(v)}
              focused={focusField === "price"}
            />
          </box>

          {/* Slippage */}
          <box flexDirection="row" height={1} gap={1}>
            <text fg={colors.FG_SECONDARY}>{"Slippage: "}</text>
            <input
              placeholder="0.5"
              onInput={(v: string) => setSlippage(v)}
              focused={focusField === "slippage"}
            />
          </box>
        </box>

        {/* Buttons */}
        <box flexDirection="row" paddingLeft={2} paddingTop={1} gap={2}>
          <box
            backgroundColor={submitting ? colors.FG_DIM : sideColor}
            paddingLeft={2}
            paddingRight={2}
          >
            <text fg={colors.BG_BASE} attributes={TextAttributes.BOLD}>
              [ SUBMIT ]
            </text>
          </box>
          <box
            backgroundColor={colors.BG_SURFACE}
            paddingLeft={2}
            paddingRight={2}
          >
            <text fg={colors.FG_SECONDARY}>[ CANCEL ]</text>
          </box>
        </box>

        {/* Status */}
        {status && (
          <box paddingLeft={2} paddingTop={1}>
            <text
              fg={
                status.startsWith("Error")
                  ? colors.LOSS
                  : status.startsWith("Order")
                  ? colors.GAIN
                  : colors.FG_SECONDARY
              }
            >
              {status}
            </text>
          </box>
        )}

        {/* Footer help */}
        <box paddingLeft={2} paddingTop={1}>
          <text fg={colors.FG_DIM}>Tab: Next field  Enter: Submit  Esc: Close</text>
        </box>
      </box>
    </box>
  );
}
