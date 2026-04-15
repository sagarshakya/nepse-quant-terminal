// Modal: Quick symbol lookup

import { useState } from "react";
import { useKeyboard, useTerminalDimensions } from "@opentui/react";
import { TextAttributes } from "@opentui/core";
import { useDispatch } from "../state/app-context";
import * as colors from "../theme/colors";

export function LookupModal({ onClose }: { onClose: () => void }) {
  const [symbol, setSymbol] = useState("");
  const dispatch = useDispatch();
  const { width, height } = useTerminalDimensions();

  useKeyboard(
    (key) => {
      if (key.name === "Escape") {
        onClose();
        return;
      }
      if (key.name === "Return") {
        const trimmed = symbol.trim().toUpperCase();
        if (trimmed) {
          dispatch({ type: "SET_LOOKUP_SYMBOL", symbol: trimmed });
          dispatch({ type: "SET_TAB", tab: "lookup" });
          onClose();
        }
        return;
      }
    },
    { release: false }
  );

  const boxWidth = 38;
  const boxHeight = 6;
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
        borderColor={colors.CYAN}
        flexDirection="column"
      >
        {/* Title */}
        <box
          backgroundColor={colors.BG_HEADER}
          height={1}
          paddingLeft={1}
        >
          <text fg={colors.CYAN} attributes={TextAttributes.BOLD}>
            :: SYMBOL LOOKUP
          </text>
        </box>

        {/* Input */}
        <box paddingLeft={1} paddingRight={1} height={1} flexDirection="row" gap={1}>
          <text fg={colors.FG_AMBER} attributes={TextAttributes.BOLD}>
            {">"}
          </text>
          <input
            placeholder="Enter symbol (e.g. NABIL)"
            onInput={(v: string) => setSymbol(v)}
            focused={true}
          />
        </box>

        {/* Footer */}
        <box paddingLeft={1} height={1}>
          <text fg={colors.FG_DIM}>Enter: Search  Esc: Cancel</text>
        </box>
      </box>
    </box>
  );
}
