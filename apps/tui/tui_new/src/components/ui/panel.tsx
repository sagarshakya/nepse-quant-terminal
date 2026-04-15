// Pane — Gloomberb-style panel with :: TITLE ─── header pattern
// Focused panes get ┌─ :: TITLE ───────── ─┐ border header
// Unfocused panes get plain :: TITLE ─────── header

import { useState } from "react";
import type { ReactNode } from "react";
import { TextAttributes } from "@opentui/core";
import { useTerminalDimensions } from "@opentui/react";
import * as colors from "../../theme/colors";

interface PanelProps {
  title: string;
  children: ReactNode;
  focused?: boolean;
  width?: number | string;
  height?: number | string;
  flexGrow?: number;
  subtitle?: string;
}

export function BloombergPanel({
  title,
  children,
  focused = false,
  width,
  height,
  flexGrow,
  subtitle,
}: PanelProps) {
  const titleBg = colors.paneTitleBg(focused);
  const paneBg = colors.paneBg(focused);
  const bc = focused ? colors.borderFocused : colors.border;
  const titleColor = focused ? colors.textBright : colors.text;

  // Build the header text: :: TITLE (subtitle)
  const headerLabel = subtitle ? `:: ${title} (${subtitle})` : `:: ${title}`;

  return (
    <box
      flexDirection="column"
      backgroundColor={paneBg}
      width={width as any}
      height={height as any}
      flexGrow={flexGrow}
    >
      {/* Pane header — Gloomberb pattern */}
      <box
        backgroundColor={titleBg}
        height={1}
        flexDirection="row"
      >
        {focused ? (
          <>
            <text fg={bc}>{"┌─"}</text>
            <text fg={titleColor} attributes={TextAttributes.BOLD}>
              {headerLabel}
            </text>
            <text fg={bc} attributes={TextAttributes.DIM}>
              {" "}
            </text>
          </>
        ) : (
          <text fg={titleColor} attributes={TextAttributes.BOLD}>
            {"  "}{headerLabel}
          </text>
        )}
      </box>

      {/* Panel content */}
      <box flexDirection="column" flexGrow={1}>
        {children}
      </box>
    </box>
  );
}
