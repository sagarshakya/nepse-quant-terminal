import * as defaultColors from "./colors";

export interface Theme {
  name: string;
  bg: string;
  panel: string;
  surface: string;
  header: string;
  hover: string;
  focus: string;
  text: string;
  textDim: string;
  accent: string;
  border: string;
  borderFocus: string;
  gain: string;
  loss: string;
}

export const bloombergDark: Theme = {
  name: "Bloomberg Dark",
  bg: defaultColors.BG_BASE,
  panel: defaultColors.BG_PANEL,
  surface: defaultColors.BG_SURFACE,
  header: defaultColors.BG_HEADER,
  hover: defaultColors.BG_HOVER,
  focus: defaultColors.BG_FOCUS,
  text: defaultColors.FG_PRIMARY,
  textDim: defaultColors.FG_SECONDARY,
  accent: defaultColors.FG_AMBER,
  border: defaultColors.BORDER,
  borderFocus: defaultColors.BORDER_FOCUS,
  gain: defaultColors.GAIN,
  loss: defaultColors.LOSS,
};

// Active theme (mutable for future theme switching)
export let activeTheme: Theme = bloombergDark;

export function setTheme(theme: Theme) {
  activeTheme = theme;
}
