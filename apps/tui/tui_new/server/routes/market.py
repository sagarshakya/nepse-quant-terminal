"""Market data endpoints — wraps MD class."""
from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter()


def _quote_to_dict(row) -> dict:
    return {
        "symbol": str(row.get("symbol", "")),
        "ltp": float(row.get("ltp", row.get("close", 0))),
        "change": float(row.get("chg", 0)),
        "change_pct": float(row.get("pc", row.get("chg_pct", 0))),
        "volume": int(row.get("vol", row.get("volume", 0))),
        "prev_close": float(row.get("prev_close", 0)),
        "high": float(row.get("high", 0)),
        "low": float(row.get("low", 0)),
        "open": float(row.get("open", 0)),
    }


def _df_to_quotes(df) -> list[dict]:
    if df is None or df.empty:
        return []
    return [_quote_to_dict(row) for _, row in df.head(25).iterrows()]


@router.get("/overview")
async def market_overview(request: Request):
    md = request.app.state.md
    try:
        md.refresh()
    except Exception:
        pass

    return {
        "gainers": _df_to_quotes(md.gainers),
        "losers": _df_to_quotes(md.losers),
        "volume_leaders": _df_to_quotes(md.vol_top),
        "near_52w_high": _df_to_quotes(md.near_hi),
        "near_52w_low": _df_to_quotes(md.near_lo),
        "live_quotes": _df_to_quotes(md.quotes) if hasattr(md, 'quotes') and not md.quotes.empty else [],
        "timestamp": md.ts.isoformat() if hasattr(md, 'ts') else "",
    }


@router.get("/indices")
async def market_indices(request: Request):
    md = request.app.state.md
    try:
        md.refresh()
    except Exception:
        pass

    nepse_val = 0.0
    nepse_chg = 0.0
    if hasattr(md, 'nepse') and not md.nepse.empty:
        try:
            nepse_val = float(md.nepse.iloc[-1].get("close", 0))
            if len(md.nepse) >= 2:
                prev = float(md.nepse.iloc[-2].get("close", nepse_val))
                nepse_chg = nepse_val - prev
        except Exception:
            pass

    nepse_chg_pct = (nepse_chg / (nepse_val - nepse_chg) * 100) if nepse_val != nepse_chg and nepse_chg != 0 else 0

    total_vol = 0
    total_turn = 0
    if hasattr(md, 'df') and not md.df.empty:
        total_vol = int(md.df["vol"].sum()) if "vol" in md.df.columns else 0
        total_turn = int(md.df.get("turnover", md.df.get("vol", 0)).sum()) if "turnover" in md.df.columns else total_vol

    return {
        "nepse_index": nepse_val,
        "nepse_change": nepse_chg,
        "nepse_change_pct": round(nepse_chg_pct, 2),
        "market_cap": 0,  # TODO: compute from df
        "total_volume": total_vol,
        "total_turnover": total_turn,
        "advances": md.adv,
        "declines": md.dec,
        "unchanged": md.unch,
    }
