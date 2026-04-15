// Status bar — Gloomberb-style bottom bar with keybindings and status

import { TextAttributes } from "@opentui/core";
import { useAppState } from "../../state/app-context";
import * as colors from "../../theme/colors";

export function StatusBar() {
  const { connected } = useAppState();

  const connColor = connected ? colors.positive : colors.negative;
  const connText = connected ? "ONLINE" : "OFFLINE";

  return (
    <box
      backgroundColor={colors.panel}
      height={1}
      flexDirection="row"
      paddingLeft={1}
      paddingRight={1}
      justifyContent="space-between"
    >
      {/* Key hints */}
      <text fg={colors.textDim}>
        [1-0] Tab  [b] Buy  [s] Sell  [l] Lookup  [/] Cmd  [q] Quit
      </text>

      {/* Connection status */}
      <text fg={connColor} attributes={TextAttributes.BOLD}>
        ● {connText}
      </text>
    </box>
  );
}
