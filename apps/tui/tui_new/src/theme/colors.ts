// Theme system — matches Gloomberb's amber Bloomberg terminal aesthetic
// Dynamic color blending for derived colors

// ── Hex blend utility ────────────────────────────────────────────────────────
function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace("#", "");
  return [
    parseInt(h.slice(0, 2), 16),
    parseInt(h.slice(2, 4), 16),
    parseInt(h.slice(4, 6), 16),
  ];
}

function rgbToHex(r: number, g: number, b: number): string {
  return (
    "#" +
    [r, g, b]
      .map((v) => Math.round(Math.max(0, Math.min(255, v))).toString(16).padStart(2, "0"))
      .join("")
  );
}

export function blendHex(a: string, b: string, ratio: number): string {
  const [r1, g1, b1] = hexToRgb(a);
  const [r2, g2, b2] = hexToRgb(b);
  return rgbToHex(
    r1 + (r2 - r1) * ratio,
    g1 + (g2 - g1) * ratio,
    b1 + (b2 - b1) * ratio,
  );
}

// ── Core palette (Gloomberb amber theme) ─────────────────────────────────────
export const bg = "#000000"; // pure black background
export const panel = "#0a0a14"; // very dark blue panel
export const border = "#1a3a5c"; // medium blue border
export const borderFocused = "#ff8800"; // orange/amber focus

export const text = "#ff8800"; // amber text (primary)
export const textDim = "#886622"; // dim amber
export const textBright = "#ffaa00"; // bright amber
export const textMuted = "#555555"; // very dim gray

export const positive = "#00cc66"; // green gains
export const negative = "#ff3333"; // red losses
export const neutral = "#888888"; // unchanged
export const warning = "#ffaa00"; // warning amber

export const header = "#0044aa"; // deep blue header bar
export const headerText = "#ffffff"; // white header text

export const selected = "#1a3a5c"; // selected row background
export const selectedText = "#ffaa00"; // selected row text

// ── Derived colors (dynamic blending like Gloomberb) ─────────────────────────
export function paneBg(focused: boolean): string {
  if (focused) return blendHex(bg, borderFocused, 0.06);
  return blendHex(panel, border, 0.08);
}

export function paneTitleBg(focused: boolean): string {
  if (focused) return blendHex(bg, borderFocused, 0.22);
  return blendHex(panel, border, 0.15);
}

export function hoverBg(): string {
  return blendHex(bg, selected, 0.5);
}

// ── Backward-compat aliases (used by existing tabs) ──────────────────────────
export const BG_BASE = bg;
export const BG_PANEL = panel;
export const BG_ROW_ALT = blendHex(bg, panel, 0.5);
export const BG_SURFACE = blendHex(panel, border, 0.15);
export const BG_HEADER = header;
export const BG_HOVER = hoverBg();
export const BG_FOCUS = selected;
export const BG_INPUT = blendHex(bg, panel, 0.6);
export const BG_INPUT_FOCUS = blendHex(bg, border, 0.3);

export const FG_PRIMARY = text;
export const FG_SECONDARY = textDim;
export const FG_DIM = textMuted;
export const FG_AMBER = textBright;
export const FG_BRIGHT = "#ffffff";

export const GAIN = positive;
export const GAIN_HI = "#00ff7f";
export const LOSS = negative;
export const LOSS_HI = "#ff4545";
export const UNCHANGED = neutral;

export const CYAN = "#00cfff";
export const YELLOW = warning;
export const PURPLE = "#bb88ff";
export const BLUE = "#5599ff";
export const ORANGE = "#ff9944";

export const BORDER = border;
export const BORDER_FOCUS = borderFocused;
export const BORDER_DIM = blendHex(bg, border, 0.4);

export const SCROLLBAR_BG = blendHex(bg, border, 0.2);
export const SCROLLBAR_FG = blendHex(bg, border, 0.5);
export const SCROLLBAR_ACTIVE = borderFocused;

export const VERDICT_BUY = "#00ff7f";
export const VERDICT_SELL = "#ff4545";
export const VERDICT_HOLD = warning;
export const VERDICT_REVIEW = "#bb88ff";

export function priceColor(value: number): string {
  if (value > 0) return positive;
  if (value < 0) return negative;
  return neutral;
}

export function priceColorBright(value: number): string {
  if (value > 0) return GAIN_HI;
  if (value < 0) return LOSS_HI;
  return neutral;
}
