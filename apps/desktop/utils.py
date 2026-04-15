"""Small formatting helpers."""
from __future__ import annotations


def fmt_number(v: float, decimals: int = 2) -> str:
    return f"{v:,.{decimals}f}"


def fmt_signed_pct(v: float, decimals: int = 2) -> str:
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.{decimals}f}%"


def fmt_signed(v: float, decimals: int = 2) -> str:
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:,.{decimals}f}"


def fmt_volume(v: float) -> str:
    if v >= 1e9:
        return f"{v/1e9:.2f}B"
    if v >= 1e6:
        return f"{v/1e6:.2f}M"
    if v >= 1e3:
        return f"{v/1e3:.1f}K"
    return f"{v:,.0f}"
