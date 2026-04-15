"""Diagnose how the TMS captcha <img> is served.

Prints src, naturalWidth/Height, the tag HTML, and attempts three fetches:
  1. JS fetch(src, {credentials:'include'})
  2. Playwright page.request.get(src)
  3. element.screenshot()

Writes each to artifacts/captcha_debug/diag_*.png for comparison.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from playwright.sync_api import sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROFILE_DIR = PROJECT_ROOT / ".tms_chrome_profile"
OUT_DIR = PROJECT_ROOT / "artifacts" / "captcha_debug"
LOGIN_URL = "https://tms19.nepsetms.com.np/login"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%H%M%S")

    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR),
            channel="chrome",
            headless=False,
            viewport={"width": 1280, "height": 800},
        )
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto(LOGIN_URL, wait_until="domcontentloaded")
        page.wait_for_timeout(3000)

        # Dump a screenshot of the full page for context
        page.screenshot(path=str(OUT_DIR / f"diag_{stamp}_page.png"))

        # Find all img tags, their src and natural dimensions
        imgs = page.evaluate(
            """
            () => Array.from(document.querySelectorAll('img')).map(img => ({
              className: img.className,
              id: img.id,
              src: (img.src || '').slice(0, 200),
              naturalWidth: img.naturalWidth,
              naturalHeight: img.naturalHeight,
              renderedWidth: img.width,
              renderedHeight: img.height,
            }))
            """
        )
        print("=== all <img> on login page ===")
        for i, info in enumerate(imgs):
            print(f"  [{i}] {info}")

        # Locate the captcha
        loc = page.locator("img.captcha-image-dimension").first
        try:
            loc.wait_for(timeout=5000)
        except Exception as exc:
            print(f"captcha locator not found: {exc}")

        src = loc.get_attribute("src") or ""
        print(f"\ncaptcha src (first 200): {src[:200]!r}")
        nat = loc.evaluate("img => ({nw: img.naturalWidth, nh: img.naturalHeight, w: img.width, h: img.height})")
        print(f"natural: {nat}")

        # Method 1: JS fetch
        try:
            if src.startswith("data:"):
                import base64
                _, _, b64 = src.partition(",")
                b = base64.b64decode(b64)
            else:
                arr = page.evaluate(
                    "async (u) => { const r = await fetch(u, {credentials:'include'}); const b = await r.arrayBuffer(); return Array.from(new Uint8Array(b)); }",
                    src,
                )
                b = bytes(arr) if arr else b""
            out = OUT_DIR / f"diag_{stamp}_jsfetch.png"
            out.write_bytes(b)
            print(f"js fetch: {len(b)} bytes -> {out.name}")
        except Exception as exc:
            print(f"js fetch failed: {exc}")

        # Method 2: page.request.get
        try:
            resp = page.request.get(src) if src.startswith("http") else None
            if resp is not None:
                b = resp.body()
                out = OUT_DIR / f"diag_{stamp}_pwreq.png"
                out.write_bytes(b)
                print(f"pw request: {len(b)} bytes -> {out.name}")
        except Exception as exc:
            print(f"pw request failed: {exc}")

        # Method 3: element screenshot (for comparison)
        try:
            out = OUT_DIR / f"diag_{stamp}_screenshot.png"
            loc.screenshot(path=str(out))
            print(f"screenshot -> {out.name}")
        except Exception as exc:
            print(f"screenshot failed: {exc}")

        page.wait_for_timeout(1000)
        ctx.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
