// Braille chart rendering utilities
//
// Each braille character (U+2800 base) represents a 2x4 dot grid.
// Dot bit positions within a cell:
//   col 0  col 1
//   0x01   0x08   row 0
//   0x02   0x10   row 1
//   0x04   0x20   row 2
//   0x40   0x80   row 3

const BRAILLE_BASE = 0x2800;

const DOT_MAP: number[][] = [
  [0x01, 0x08], // row 0
  [0x02, 0x10], // row 1
  [0x04, 0x20], // row 2
  [0x40, 0x80], // row 3
];

/** Create a zeroed pixel buffer: pixelWidth x pixelHeight booleans packed into
 *  a braille-cell grid of (cellCols x cellRows). */
function createBuffer(cellCols: number, cellRows: number): number[][] {
  // buffer[row][col] holds the braille offset bits for that cell
  const buf: number[][] = [];
  for (let r = 0; r < cellRows; r++) {
    buf.push(new Array(cellCols).fill(0));
  }
  return buf;
}

/** Set a single pixel in the braille buffer.
 *  px/py are in pixel coordinates (0-based, top-left origin). */
function setPixel(
  buf: number[][],
  cellCols: number,
  cellRows: number,
  px: number,
  py: number,
): void {
  if (px < 0 || py < 0) return;
  const cellCol = Math.floor(px / 2);
  const cellRow = Math.floor(py / 4);
  if (cellCol >= cellCols || cellRow >= cellRows) return;
  const dotCol = px % 2;
  const dotRow = py % 4;
  buf[cellRow][cellCol] |= DOT_MAP[dotRow][dotCol];
}

/** Convert a braille buffer into an array of strings (one per row). */
function bufferToStrings(buf: number[][]): string[] {
  return buf.map((row) =>
    row.map((bits) => String.fromCharCode(BRAILLE_BASE + bits)).join(""),
  );
}

// ── Line Chart ──────────────────────────────────────────────────────

export function renderBrailleLine(
  values: number[],
  width: number,
  height: number, // terminal rows (each row = 4 braille dots)
  color: string,
): { lines: string[]; min: number; max: number } {
  if (values.length === 0) {
    return { lines: Array(height).fill(" ".repeat(width)), min: 0, max: 0 };
  }

  let min = Infinity;
  let max = -Infinity;
  for (const v of values) {
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (min === max) {
    max = min + 1; // avoid division by zero
  }

  const pixelW = width * 2;
  const pixelH = height * 4;
  const buf = createBuffer(width, height);

  // Map each value to a pixel column, interpolating across pixelW
  for (let px = 0; px < pixelW; px++) {
    // Which value index corresponds to this pixel column?
    const dataIdx = (px / (pixelW - 1)) * (values.length - 1);
    const lo = Math.floor(dataIdx);
    const hi = Math.min(lo + 1, values.length - 1);
    const t = dataIdx - lo;
    const v = values[lo] * (1 - t) + values[hi] * t;

    // Map value to pixel row (inverted: 0=top)
    const py = Math.round((1 - (v - min) / (max - min)) * (pixelH - 1));
    setPixel(buf, width, height, px, py);

    // Draw vertical line between consecutive points for continuity
    if (px > 0) {
      const prevIdx = ((px - 1) / (pixelW - 1)) * (values.length - 1);
      const plo = Math.floor(prevIdx);
      const phi = Math.min(plo + 1, values.length - 1);
      const pt = prevIdx - plo;
      const pv = values[plo] * (1 - pt) + values[phi] * pt;
      const prevPy = Math.round((1 - (pv - min) / (max - min)) * (pixelH - 1));

      const step = prevPy < py ? 1 : -1;
      for (let y = prevPy; y !== py; y += step) {
        setPixel(buf, width, height, px, y);
      }
    }
  }

  return { lines: bufferToStrings(buf), min, max };
}

// ── Candlestick Chart ───────────────────────────────────────────────

export interface CandleBar {
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export function renderCandlestickBraille(
  bars: CandleBar[],
  width: number,
  height: number, // terminal rows for price area
): {
  lines: string[];
  priceMin: number;
  priceMax: number;
  volumeLines: string[];
} {
  const volHeight = 2; // terminal rows for volume
  const emptyResult = {
    lines: Array(height).fill(" ".repeat(width)),
    priceMin: 0,
    priceMax: 0,
    volumeLines: Array(volHeight).fill(" ".repeat(width)),
  };

  if (bars.length === 0) return emptyResult;

  // Find price range
  let priceMin = Infinity;
  let priceMax = -Infinity;
  let volMax = 0;
  for (const b of bars) {
    if (b.low < priceMin) priceMin = b.low;
    if (b.high > priceMax) priceMax = b.high;
    if (b.volume > volMax) volMax = b.volume;
  }
  if (priceMin === priceMax) priceMax = priceMin + 1;
  if (volMax === 0) volMax = 1;

  const pixelH = height * 4;
  const pixelW = width * 2;
  const priceBuf = createBuffer(width, height);

  // Each bar gets some pixel columns
  const barPixelWidth = Math.max(1, Math.floor(pixelW / bars.length));
  const gap = barPixelWidth >= 3 ? 1 : 0;
  const bodyWidth = Math.max(1, barPixelWidth - gap);

  function priceToY(price: number): number {
    return Math.round((1 - (price - priceMin) / (priceMax - priceMin)) * (pixelH - 1));
  }

  for (let i = 0; i < bars.length; i++) {
    const bar = bars[i];
    const startPx = Math.floor((i / bars.length) * pixelW);
    const midPx = startPx + Math.floor(bodyWidth / 2);

    const highY = priceToY(bar.high);
    const lowY = priceToY(bar.low);
    const openY = priceToY(bar.open);
    const closeY = priceToY(bar.close);

    // Draw wick (single pixel column at center)
    const wickTop = Math.min(highY, lowY);
    const wickBot = Math.max(highY, lowY);
    for (let y = wickTop; y <= wickBot; y++) {
      setPixel(priceBuf, width, height, midPx, y);
    }

    // Draw body (filled rectangle between open and close)
    const bodyTop = Math.min(openY, closeY);
    const bodyBot = Math.max(openY, closeY);
    for (let y = bodyTop; y <= bodyBot; y++) {
      for (let x = startPx; x < startPx + bodyWidth && x < pixelW; x++) {
        setPixel(priceBuf, width, height, x, y);
      }
    }
    // If open === close, at least draw a line
    if (bodyTop === bodyBot) {
      for (let x = startPx; x < startPx + bodyWidth && x < pixelW; x++) {
        setPixel(priceBuf, width, height, x, bodyTop);
      }
    }
  }

  // Volume bars
  const volPixelH = volHeight * 4;
  const volBuf = createBuffer(width, volHeight);

  for (let i = 0; i < bars.length; i++) {
    const bar = bars[i];
    const startPx = Math.floor((i / bars.length) * pixelW);
    const volH = Math.round((bar.volume / volMax) * (volPixelH - 1));

    for (let y = volPixelH - 1; y >= volPixelH - volH; y--) {
      for (let x = startPx; x < startPx + bodyWidth && x < pixelW; x++) {
        setPixel(volBuf, width, volHeight, x, y);
      }
    }
  }

  return {
    lines: bufferToStrings(priceBuf),
    priceMin,
    priceMax,
    volumeLines: bufferToStrings(volBuf),
  };
}
