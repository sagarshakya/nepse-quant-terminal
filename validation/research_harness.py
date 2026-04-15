"""Strict autoresearch harness with actual-NEPSE benchmarking."""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from backend.backtesting.simple_backtest import (
    BacktestResult,
    compute_market_regime,
    load_all_prices,
    run_backtest,
)
from backend.quant_pro.database import get_db_path

NEPSE_TRADING_DAYS = 240
DEFAULT_WARMUP_START = "2023-01-01"
DEFAULT_OOS_START = "2024-01-01"
DEFAULT_OOS_END = "2025-12-31"
DEFAULT_ADJACENT_SLICES: Sequence[Tuple[str, str]] = (
    ("2023-01-01", "2024-12-31"),
    ("2024-01-01", "2025-12-31"),
    ("2025-01-01", "2025-12-31"),
)

VALIDATED_CORE_CONFIG: Dict[str, Any] = {
    "holding_days": 40,
    "max_positions": 5,
    "signal_types": ["volume", "quality", "low_vol", "mean_reversion"],
    "rebalance_frequency": 5,
    "stop_loss_pct": 0.08,
    "trailing_stop_pct": 0.10,
    "use_regime_filter": True,
    "sector_limit": 0.35,
    "regime_max_positions": {"bull": 5, "neutral": 4, "bear": 2},
    "bear_threshold": -0.05,
    "initial_capital": 1_000_000,
    "profit_target_pct": None,
    "event_exit_mode": False,
}


@dataclass(frozen=True)
class ResearchPaths:
    """Filesystem locations for research artifacts."""

    root: Path
    artifact_dir: Path
    ledger_path: Path


def get_research_paths(
    root: str | Path = "reports/autoresearch",
) -> ResearchPaths:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    artifact_dir = root_path / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = root_path / "experiment_ledger.jsonl"
    return ResearchPaths(root=root_path, artifact_dir=artifact_dir, ledger_path=ledger_path)


def load_actual_nepse_series(
    conn: sqlite3.Connection,
    start: str,
    end: str,
) -> pd.Series:
    """Load normalized actual NEPSE index levels from the database."""
    q = """
        SELECT date, close
        FROM stock_prices
        WHERE symbol = 'NEPSE' AND date BETWEEN ? AND ?
        ORDER BY date
    """
    df = pd.read_sql_query(q, conn, params=[start, end])
    if df.empty:
        return pd.Series(dtype=float)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    series = pd.Series(df["close"].astype(float).values, index=df["date"])
    series = series / float(series.iloc[0])
    series.iloc[0] = 1.0
    return series


def _nav_frame(result: BacktestResult) -> pd.DataFrame:
    nav_df = pd.DataFrame(result.daily_nav, columns=["date", "nav"])
    if nav_df.empty:
        return nav_df
    nav_df["date"] = pd.to_datetime(nav_df["date"])
    return nav_df.sort_values("date").set_index("date")


def _slice_nav(nav_df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    if nav_df.empty:
        return nav_df.copy()
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    return nav_df.loc[(nav_df.index >= start_ts) & (nav_df.index <= end_ts)].copy()


def _window_total_return(nav_df: pd.DataFrame) -> float:
    if nav_df.empty or len(nav_df) < 2:
        return 0.0
    return float(nav_df["nav"].iloc[-1] / nav_df["nav"].iloc[0] - 1.0)


def _window_sharpe(nav_df: pd.DataFrame) -> float:
    if nav_df.empty or len(nav_df) < 30:
        return 0.0
    rets = nav_df["nav"].pct_change().dropna()
    if len(rets) < 30:
        return 0.0
    std = float(rets.std(ddof=1))
    if std == 0:
        return 0.0
    return float(rets.mean() / std * math.sqrt(NEPSE_TRADING_DAYS))


def _window_max_drawdown(nav_df: pd.DataFrame) -> float:
    if nav_df.empty or len(nav_df) < 2:
        return 0.0
    running_max = nav_df["nav"].cummax()
    drawdown = nav_df["nav"] / running_max - 1.0
    return float(drawdown.min())


def _window_years(nav_df: pd.DataFrame) -> float:
    if nav_df.empty or len(nav_df) < 2:
        return 0.0
    days = (nav_df.index[-1] - nav_df.index[0]).days
    if days <= 0:
        return 0.0
    return days / 365.0


def _trade_in_window(trade, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    entry = pd.Timestamp(trade.entry_date)
    exit_date = pd.Timestamp(trade.exit_date) if trade.exit_date is not None else None
    if start <= entry <= end:
        return True
    if exit_date is not None and start <= exit_date <= end:
        return True
    return False


def _window_trade_set(result: BacktestResult, start: str, end: str) -> List:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    return [trade for trade in result.completed_trades if _trade_in_window(trade, start_ts, end_ts)]


def _turnover_ratio(trades: Sequence, nav_df: pd.DataFrame) -> float:
    if not trades or nav_df.empty:
        return 0.0
    years = _window_years(nav_df)
    if years <= 0:
        return 0.0
    total_turnover = 0.0
    for trade in trades:
        total_turnover += float(trade.position_value or 0.0)
        if trade.exit_price is not None:
            total_turnover += float(trade.exit_price * trade.shares)
    avg_nav = float(nav_df["nav"].mean())
    if avg_nav <= 0:
        return 0.0
    return float(total_turnover / avg_nav / years)


def _win_rate_pct(trades: Sequence) -> float:
    if not trades:
        return 0.0
    wins = sum(1 for trade in trades if (trade.net_pnl or 0.0) > 0.0)
    return 100.0 * wins / len(trades)


def _trade_count(trades: Sequence) -> int:
    return len(trades)


def _concentration_metrics(trades: Sequence) -> Dict[str, float]:
    if not trades:
        return {
            "top_name_pnl_share": 0.0,
            "top3_name_pnl_share": 0.0,
            "pnl_name_hhi": 0.0,
            "concentration_penalty": 0.0,
        }
    pnl_by_symbol: Dict[str, float] = {}
    for trade in trades:
        pnl_by_symbol[trade.symbol] = pnl_by_symbol.get(trade.symbol, 0.0) + float(trade.net_pnl or 0.0)
    abs_pnl = pd.Series({k: abs(v) for k, v in pnl_by_symbol.items()}, dtype=float)
    total_abs = float(abs_pnl.sum())
    if total_abs <= 0:
        return {
            "top_name_pnl_share": 0.0,
            "top3_name_pnl_share": 0.0,
            "pnl_name_hhi": 0.0,
            "concentration_penalty": 0.0,
        }
    shares = abs_pnl / total_abs
    top_name = float(shares.max())
    top3 = float(shares.nlargest(min(3, len(shares))).sum())
    hhi = float((shares**2).sum())
    penalty = max(0.0, top_name - 0.35) * 12.0 + max(0.0, top3 - 0.70) * 8.0 + max(0.0, hhi - 0.20) * 20.0
    return {
        "top_name_pnl_share": top_name,
        "top3_name_pnl_share": top3,
        "pnl_name_hhi": hhi,
        "concentration_penalty": penalty,
    }


def _instability_metrics(nav_df: pd.DataFrame) -> Dict[str, float]:
    if nav_df.empty or len(nav_df) < 10:
        return {
            "top5_day_pnl_share": 0.0,
            "monthly_hit_rate_pct": 0.0,
            "instability_penalty": 0.0,
        }
    daily_pnl = nav_df["nav"].diff().dropna()
    total_abs = float(daily_pnl.abs().sum())
    if total_abs > 0:
        top5_share = float(daily_pnl.abs().nlargest(min(5, len(daily_pnl))).sum() / total_abs)
    else:
        top5_share = 0.0
    monthly = nav_df["nav"].resample("ME").last().pct_change().dropna()
    monthly_hit_rate = float((monthly > 0).mean() * 100.0) if len(monthly) else 0.0
    penalty = max(0.0, top5_share - 0.35) * 10.0
    return {
        "top5_day_pnl_share": top5_share,
        "monthly_hit_rate_pct": monthly_hit_rate,
        "instability_penalty": penalty,
    }


def _regime_breakdown(
    prices_df: pd.DataFrame,
    nav_df: pd.DataFrame,
    start: str,
    end: str,
) -> List[Dict[str, Any]]:
    if nav_df.empty:
        return []
    rows: List[Dict[str, Any]] = []
    nav_with_ret = nav_df.copy()
    nav_with_ret["ret"] = nav_with_ret["nav"].pct_change()
    nav_with_ret = nav_with_ret.dropna()
    for regime in ("bull", "neutral", "bear"):
        regime_dates = []
        for day in nav_with_ret.index:
            current_regime = compute_market_regime(prices_df, pd.Timestamp(day))
            if current_regime == regime:
                regime_dates.append(day)
        if not regime_dates:
            continue
        regime_rets = nav_with_ret.loc[regime_dates, "ret"]
        total_return = float((1.0 + regime_rets).prod() - 1.0)
        rows.append(
            {
                "regime": regime,
                "days": int(len(regime_rets)),
                "mean_daily_return_pct": float(regime_rets.mean() * 100.0),
                "total_return_pct": float(total_return * 100.0),
                "sharpe": float(
                    regime_rets.mean() / regime_rets.std(ddof=1) * math.sqrt(NEPSE_TRADING_DAYS)
                ) if len(regime_rets) >= 10 and float(regime_rets.std(ddof=1)) > 0 else 0.0,
            }
        )
    return rows


def _average_gross_exposure(
    trades: Sequence,
    prices_df: pd.DataFrame,
    nav_df: pd.DataFrame,
) -> Dict[str, float]:
    if not trades or nav_df.empty:
        return {"avg_exposure": 0.0, "max_exposure": 0.0}
    date_index = pd.Index(nav_df.index.unique())
    symbols = sorted({trade.symbol for trade in trades})
    window_prices = prices_df[
        (prices_df["symbol"].isin(symbols))
        & (prices_df["date"] >= date_index.min())
        & (prices_df["date"] <= date_index.max())
    ][["symbol", "date", "close"]].copy()
    if window_prices.empty:
        return {"avg_exposure": 0.0, "max_exposure": 0.0}
    window_prices["date"] = pd.to_datetime(window_prices["date"])
    close_pivot = window_prices.pivot(index="date", columns="symbol", values="close").sort_index()
    exposure = pd.Series(0.0, index=date_index)
    for trade in trades:
        entry = pd.Timestamp(trade.entry_date)
        exit_date = pd.Timestamp(trade.exit_date) if trade.exit_date is not None else date_index.max() + pd.Timedelta(days=1)
        active_dates = date_index[(date_index >= entry) & (date_index < exit_date)]
        if len(active_dates) == 0:
            continue
        prices = close_pivot.get(trade.symbol)
        if prices is None:
            prices = pd.Series(float(trade.entry_price), index=active_dates)
        else:
            prices = prices.reindex(active_dates).ffill().fillna(float(trade.entry_price))
        exposure.loc[active_dates] += prices.astype(float) * float(trade.shares)
    nav_series = nav_df.reindex(date_index)["nav"].astype(float)
    gross_exposure = exposure / nav_series.replace(0.0, np.nan)
    gross_exposure = gross_exposure.replace([np.inf, -np.inf], np.nan).dropna()
    if gross_exposure.empty:
        return {"avg_exposure": 0.0, "max_exposure": 0.0}
    return {
        "avg_exposure": float(gross_exposure.mean()),
        "max_exposure": float(gross_exposure.max()),
    }


def _evaluate_window(
    result: BacktestResult,
    prices_df: pd.DataFrame,
    benchmark: pd.Series,
    start: str,
    end: str,
) -> Dict[str, Any]:
    nav_df = _slice_nav(_nav_frame(result), start, end)
    if nav_df.empty:
        return {
            "window_start": start,
            "window_end": end,
            "strategy_return_pct": 0.0,
            "nepse_return_pct": 0.0,
            "relative_return_vs_nepse": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "win_rate_pct": 0.0,
            "trade_count": 0,
            "annualized_turnover_ratio": 0.0,
            "avg_exposure": 0.0,
            "max_exposure": 0.0,
            "top_name_pnl_share": 0.0,
            "top3_name_pnl_share": 0.0,
            "pnl_name_hhi": 0.0,
            "top5_day_pnl_share": 0.0,
            "monthly_hit_rate_pct": 0.0,
            "instability_penalty": 0.0,
            "concentration_penalty": 0.0,
        }

    strategy_return_pct = _window_total_return(nav_df) * 100.0
    sharpe_ratio = _window_sharpe(nav_df)
    max_drawdown_pct = abs(_window_max_drawdown(nav_df) * 100.0)
    trades = _window_trade_set(result, start, end)
    turnover_ratio = _turnover_ratio(trades, nav_df)
    exposure = _average_gross_exposure(trades, prices_df, nav_df)
    concentration = _concentration_metrics(trades)
    instability = _instability_metrics(nav_df)

    bench_slice = benchmark.loc[(benchmark.index >= pd.Timestamp(start)) & (benchmark.index <= pd.Timestamp(end))]
    nepse_return_pct = float((bench_slice.iloc[-1] / bench_slice.iloc[0] - 1.0) * 100.0) if len(bench_slice) >= 2 else 0.0
    relative_return = strategy_return_pct - nepse_return_pct

    return {
        "window_start": start,
        "window_end": end,
        "strategy_return_pct": float(strategy_return_pct),
        "nepse_return_pct": float(nepse_return_pct),
        "relative_return_vs_nepse": float(relative_return),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown_pct": float(max_drawdown_pct),
        "win_rate_pct": float(_win_rate_pct(trades)),
        "trade_count": int(_trade_count(trades)),
        "annualized_turnover_ratio": float(turnover_ratio),
        "avg_exposure": float(exposure["avg_exposure"]),
        "max_exposure": float(exposure["max_exposure"]),
        **concentration,
        **instability,
    }


def _fragility_penalty(slice_rows: Sequence[Dict[str, Any]]) -> Tuple[float, Dict[str, float]]:
    if not slice_rows:
        return 0.0, {"negative_slice_ratio": 0.0, "relative_return_std": 0.0}
    rel = np.array([float(row["relative_return_vs_nepse"]) for row in slice_rows], dtype=float)
    neg_ratio = float(np.mean(rel <= 0.0))
    rel_std = float(np.std(rel, ddof=0)) if len(rel) > 1 else 0.0
    penalty = neg_ratio * 8.0 + max(0.0, rel_std - 15.0) * 0.20
    return penalty, {"negative_slice_ratio": neg_ratio, "relative_return_std": rel_std}


def composite_score(window_metrics: Dict[str, Any], fragility_penalty: float) -> float:
    return float(
        1.50 * float(window_metrics["relative_return_vs_nepse"])
        + 1.50 * float(window_metrics["sharpe_ratio"])
        - 0.75 * abs(float(window_metrics["max_drawdown_pct"]))
        - 0.20 * float(window_metrics["annualized_turnover_ratio"])
        + 0.05 * float(window_metrics["win_rate_pct"])
        - float(window_metrics["instability_penalty"])
        - float(window_metrics["concentration_penalty"])
        - float(fragility_penalty)
    )


def breakthrough_status(window_metrics: Dict[str, Any]) -> Dict[str, bool]:
    strat = float(window_metrics["strategy_return_pct"])
    bench = float(window_metrics["nepse_return_pct"])
    return {
        "return_2x_nepse": bench > 0.0 and strat >= 2.0 * bench,
        "return_5x_nepse": bench > 0.0 and strat >= 5.0 * bench,
        "positive_sharpe": float(window_metrics["sharpe_ratio"]) > 0.0,
        "controlled_drawdown": float(window_metrics["max_drawdown_pct"]) <= 35.0,
        "realistic_turnover": float(window_metrics["annualized_turnover_ratio"]) <= 8.0,
    }


def run_research_evaluation(
    *,
    name: str,
    config: Dict[str, Any],
    hypothesis: str,
    rationale: str,
    changed_files: Optional[Iterable[str]] = None,
    commands: Optional[Iterable[str]] = None,
    warmup_start: str = DEFAULT_WARMUP_START,
    oos_start: str = DEFAULT_OOS_START,
    oos_end: str = DEFAULT_OOS_END,
    adjacent_slices: Sequence[Tuple[str, str]] = DEFAULT_ADJACENT_SLICES,
    artifact_root: str | Path = "reports/autoresearch",
) -> Dict[str, Any]:
    """Run a single warm-up-aware research evaluation and persist its artifact."""
    paths = get_research_paths(artifact_root)
    conn = sqlite3.connect(str(get_db_path()))
    prices_df = load_all_prices(conn)
    benchmark = load_actual_nepse_series(conn, warmup_start, oos_end)
    conn.close()

    result = run_backtest(start_date=warmup_start, end_date=oos_end, **config)
    primary = _evaluate_window(result, prices_df, benchmark, oos_start, oos_end)
    slice_rows = [
        _evaluate_window(result, prices_df, benchmark, slice_start, slice_end)
        for slice_start, slice_end in adjacent_slices
    ]
    fragility_penalty, fragility_meta = _fragility_penalty(slice_rows)
    score = composite_score(primary, fragility_penalty)
    breakthrough = breakthrough_status(primary)
    regime_rows = _regime_breakdown(prices_df, _slice_nav(_nav_frame(result), oos_start, oos_end), oos_start, oos_end)

    artifact = {
        "name": name,
        "hypothesis": hypothesis,
        "rationale": rationale,
        "config": config,
        "commands": list(commands or []),
        "changed_files": list(changed_files or []),
        "warmup_start": warmup_start,
        "oos_start": oos_start,
        "oos_end": oos_end,
        "primary_window": primary,
        "slice_stability": slice_rows,
        "regime_breakdown": regime_rows,
        "fragility_penalty": float(fragility_penalty),
        "fragility_meta": fragility_meta,
        "score": float(score),
        "breakthrough": breakthrough,
    }

    artifact_path = paths.artifact_dir / f"{name}.json"
    artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    with paths.ledger_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"name": name, "artifact_path": str(artifact_path), "score": score}) + "\n")
    artifact["artifact_path"] = str(artifact_path)
    artifact["ledger_path"] = str(paths.ledger_path)
    return artifact


def compare_artifacts(candidate: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, float]:
    cand = candidate["primary_window"]
    base = baseline["primary_window"]
    return {
        "score_delta": float(candidate["score"] - baseline["score"]),
        "strategy_return_pct_delta": float(cand["strategy_return_pct"] - base["strategy_return_pct"]),
        "relative_return_vs_nepse_delta": float(cand["relative_return_vs_nepse"] - base["relative_return_vs_nepse"]),
        "sharpe_ratio_delta": float(cand["sharpe_ratio"] - base["sharpe_ratio"]),
        "max_drawdown_pct_delta": float(cand["max_drawdown_pct"] - base["max_drawdown_pct"]),
        "win_rate_pct_delta": float(cand["win_rate_pct"] - base["win_rate_pct"]),
        "annualized_turnover_ratio_delta": float(cand["annualized_turnover_ratio"] - base["annualized_turnover_ratio"]),
        "avg_exposure_delta": float(cand["avg_exposure"] - base["avg_exposure"]),
    }
