// Tab bar — Gloomberb-style with ▔ underline on active tab

import { TextAttributes } from "@opentui/core";
import { useAppState } from "../../state/app-context";
import { TAB_LIST } from "../../state/types";
import * as colors from "../../theme/colors";

export function TabBar() {
  const { activeTab } = useAppState();

  return (
    <box
      backgroundColor={colors.bg}
      height={1}
      flexDirection="row"
      paddingLeft={1}
      gap={1}
    >
      {TAB_LIST.map((tab) => {
        const isActive = tab.id === activeTab;
        return (
          <box key={tab.id} flexDirection="row">
            <text
              fg={isActive ? colors.textBright : colors.textDim}
              attributes={isActive ? TextAttributes.BOLD : 0}
            >
              {tab.key}:{tab.label}
            </text>
          </box>
        );
      })}
    </box>
  );
}
