// Modal: Mode selection — Paper vs Live trading

import { useState } from "react";
import { useKeyboard, useTerminalDimensions } from "@opentui/react";
import { TextAttributes } from "@opentui/core";
import * as colors from "../theme/colors";

export function ModeSelectModal({
  onSelect,
}: {
  onSelect: (mode: "paper" | "live") => void;
}) {
  const [selected, setSelected] = useState<0 | 1>(0);
  const { width, height } = useTerminalDimensions();

  useKeyboard(
    (key) => {
      if (key.name === "ArrowUp" || key.name === "k") {
        setSelected(0);
      } else if (key.name === "ArrowDown" || key.name === "j") {
        setSelected(1);
      } else if (key.name === "Return") {
        onSelect(selected === 0 ? "paper" : "live");
      } else if (key.name === "Escape") {
        // Close without changing — caller can handle
      }
    },
    { release: false }
  );

  const boxWidth = 36;
  const boxHeight = 10;
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
        borderColor={colors.BORDER_FOCUS}
        flexDirection="column"
      >
        {/* Title */}
        <box
          backgroundColor={colors.BG_HEADER}
          height={1}
          paddingLeft={1}
          paddingRight={1}
        >
          <text fg={colors.FG_AMBER} attributes={TextAttributes.BOLD}>
            :: SELECT TRADING MODE
          </text>
        </box>

        {/* Spacer */}
        <box height={1} />

        {/* Paper option */}
        <box
          height={2}
          paddingLeft={2}
          backgroundColor={selected === 0 ? colors.BG_FOCUS : colors.BG_PANEL}
          flexDirection="column"
        >
          <text
            fg={selected === 0 ? colors.GAIN_HI : colors.FG_PRIMARY}
            attributes={selected === 0 ? TextAttributes.BOLD : 0}
          >
            {selected === 0 ? "> " : "  "}PAPER MODE
          </text>
          <text fg={colors.FG_DIM}>
            {"    Simulated trading with virtual capital"}
          </text>
        </box>

        {/* Live option */}
        <box
          height={2}
          paddingLeft={2}
          backgroundColor={selected === 1 ? colors.BG_FOCUS : colors.BG_PANEL}
          flexDirection="column"
        >
          <text
            fg={selected === 1 ? colors.LOSS_HI : colors.FG_PRIMARY}
            attributes={selected === 1 ? TextAttributes.BOLD : 0}
          >
            {selected === 1 ? "> " : "  "}LIVE MODE
          </text>
          <text fg={colors.FG_DIM}>
            {"    Real orders via broker connection"}
          </text>
        </box>

        {/* Footer */}
        <box height={1} paddingLeft={1}>
          <text fg={colors.FG_DIM}>Arrow keys to select, Enter to confirm</text>
        </box>
      </box>
    </box>
  );
}
