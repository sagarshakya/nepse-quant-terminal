from __future__ import annotations

import copy
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from configs.long_term import LONG_TERM_CONFIG
from backend.quant_pro.paths import ensure_dir, get_project_root

PROJECT_ROOT = get_project_root(__file__)
STRATEGY_REGISTRY_DIR = ensure_dir(PROJECT_ROOT / "data" / "strategy_registry")
BUILTIN_STRATEGY_DIR = ensure_dir(STRATEGY_REGISTRY_DIR / "builtin")
CUSTOM_STRATEGY_DIR = ensure_dir(STRATEGY_REGISTRY_DIR / "custom")
BACKTEST_RESULTS_DIR = ensure_dir(STRATEGY_REGISTRY_DIR / "backtests")
COMPARISON_LATEST_JSON = BACKTEST_RESULTS_DIR / "registry_strategies_vs_nepse_latest.json"
COMPARISON_LATEST_CSV = BACKTEST_RESULTS_DIR / "registry_strategies_vs_nepse_latest.csv"
COMPARISON_LATEST_PNG = BACKTEST_RESULTS_DIR / "registry_strategies_vs_nepse_latest.png"


def _timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _json_write(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _baseline_c31_config() -> Dict[str, Any]:
    config = copy.deepcopy(LONG_TERM_CONFIG)
    config.setdefault("use_trailing_stop", True)
    config.setdefault("profit_target_pct", None)
    config.setdefault("event_exit_mode", False)
    return config


def _temp_forward_winner_config() -> Dict[str, Any]:
    return {
        "holding_days": 45,
        "max_positions": 5,
        "signal_types": ["quality", "quarterly_fundamental", "xsec_momentum"],
        "rebalance_frequency": 5,
        "stop_loss_pct": 0.06,
        "trailing_stop_pct": 0.15,
        "use_regime_filter": True,
        "sector_limit": 0.35,
        "regime_max_positions": {"bull": 5, "neutral": 4, "bear": 2},
        "bear_threshold": -0.05,
        "initial_capital": float(LONG_TERM_CONFIG.get("initial_capital") or 1_000_000.0),
        "regime_sector_limits": {"bull": 0.5, "neutral": 0.35, "bear": 0.25},
        "profit_target_pct": None,
        "event_exit_mode": False,
        "use_trailing_stop": True,
    }


def _builtin_payloads() -> List[Dict[str, Any]]:
    return []


def ensure_builtin_strategies() -> None:
    for payload in _builtin_payloads():
        path = BUILTIN_STRATEGY_DIR / f"{payload['id']}.json"
        _json_write(path, payload)


def _load_strategy_file(path: Path) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    payload.setdefault("id", path.stem)
    payload.setdefault("name", payload["id"])
    payload.setdefault("description", "")
    payload.setdefault("source", "custom" if path.parent == CUSTOM_STRATEGY_DIR else "builtin")
    payload.setdefault("editable", payload.get("source") != "builtin")
    payload.setdefault("runner_mode", "temp_patched")
    payload.setdefault("execution_mode", "paper_runtime")
    payload.setdefault("config", {})
    payload.setdefault("ranking_overlay", {"mode": "baseline"})
    payload["_path"] = str(path)
    return payload


def list_strategies() -> List[Dict[str, Any]]:
    ensure_builtin_strategies()
    rows: List[Dict[str, Any]] = []
    for base in (BUILTIN_STRATEGY_DIR, CUSTOM_STRATEGY_DIR):
        for path in sorted(base.glob("*.json")):
            payload = _load_strategy_file(path)
            if payload:
                rows.append(payload)
    rows.sort(key=lambda item: (0 if str(item.get("source")) == "builtin" else 1, str(item.get("name") or item.get("id") or "").lower()))
    return rows


def load_strategy(strategy_id: str) -> Optional[Dict[str, Any]]:
    sid = str(strategy_id or "").strip()
    if not sid:
        return None
    ensure_builtin_strategies()
    for base in (CUSTOM_STRATEGY_DIR, BUILTIN_STRATEGY_DIR):
        path = base / f"{sid}.json"
        if path.exists():
            return _load_strategy_file(path)
    return None


def strategy_name(strategy_id: str) -> str:
    payload = load_strategy(strategy_id)
    if payload:
        return str(payload.get("name") or strategy_id)
    return str(strategy_id or "").strip()


def default_strategy_for_account(account_id: str) -> str:
    """Return the default strategy ID for an account. All accounts start with the C5 baseline."""
    return "default_c5"


def ensure_account_strategy_ids(accounts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ensure_builtin_strategies()
    updated: List[Dict[str, Any]] = []
    for account in list(accounts or []):
        row = dict(account or {})
        if not str(row.get("strategy_id") or "").strip():
            row["strategy_id"] = default_strategy_for_account(str(row.get("id") or ""))
        updated.append(row)
    return updated


def _slugify(token: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", str(token or "").strip().lower()).strip("_")
    return cleaned or "strategy"


def save_custom_strategy(payload: Dict[str, Any], *, strategy_id: Optional[str] = None, overwrite: bool = False) -> Dict[str, Any]:
    ensure_builtin_strategies()
    now = _timestamp()
    sid = _slugify(strategy_id or payload.get("id") or payload.get("name") or now)
    path = CUSTOM_STRATEGY_DIR / f"{sid}.json"
    existing = _load_strategy_file(path) if path.exists() else None
    if path.exists() and not overwrite:
        suffix = 2
        while (CUSTOM_STRATEGY_DIR / f"{sid}_{suffix}.json").exists():
            suffix += 1
        sid = f"{sid}_{suffix}"
        path = CUSTOM_STRATEGY_DIR / f"{sid}.json"
        existing = None
    record = {
        "id": sid,
        "name": str(payload.get("name") or sid),
        "description": str(payload.get("description") or "").strip(),
        "source": "custom",
        "editable": True,
        "runner_mode": str(payload.get("runner_mode") or "temp_patched"),
        "execution_mode": str(payload.get("execution_mode") or "paper_runtime"),
        "config": copy.deepcopy(payload.get("config") or {}),
        "ranking_overlay": copy.deepcopy(payload.get("ranking_overlay") or {"mode": "baseline"}),
        "created_at": str((existing or {}).get("created_at") or now),
        "updated_at": now,
        "notes": copy.deepcopy(payload.get("notes") or {}),
    }
    _json_write(path, record)
    return load_strategy(sid) or record


def run_strategy_backtest(strategy: Dict[str, Any], *, start_date: str, end_date: str, capital: float) -> Dict[str, Any]:
    import run_live_trader_temp_forward_experiments as temp_runner
    from backend.backtesting.simple_backtest import run_backtest

    temp_runner._build_fast_signal_patches()
    temp_runner._patch_ranker()
    temp_runner.RANKING_OVERLAY = dict(strategy.get("ranking_overlay") or {"mode": "baseline"})

    config = copy.deepcopy(strategy.get("config") or {})
    config["initial_capital"] = float(capital)
    result = run_backtest(start_date=start_date, end_date=end_date, **config)
    summary = temp_runner._summarize_result(
        str(strategy.get("id") or "strategy"),
        str(strategy.get("description") or strategy.get("name") or "Strategy backtest"),
        result,
    )
    nepse = temp_runner._nepse_return(start_date, end_date)
    summary["vs_nepse_pct_points"] = round(float(summary["total_return_pct"]) - float(nepse.get("return_pct") or 0.0), 4)

    payload = {
        "strategy": {
            "id": str(strategy.get("id") or ""),
            "name": str(strategy.get("name") or ""),
            "runner_mode": str(strategy.get("runner_mode") or "temp_patched"),
            "execution_mode": str(strategy.get("execution_mode") or "paper_runtime"),
            "config": copy.deepcopy(strategy.get("config") or {}),
            "ranking_overlay": copy.deepcopy(strategy.get("ranking_overlay") or {"mode": "baseline"}),
        },
        "window": {
            "start": str(start_date),
            "end": str(end_date),
            "capital": float(capital),
        },
        "nepse": nepse,
        "summary": summary,
        "generated_at": _timestamp(),
    }
    _json_write(BACKTEST_RESULTS_DIR / f"{strategy['id']}_latest.json", payload)
    return payload


def load_strategy_comparison_snapshot() -> Optional[Dict[str, Any]]:
    if not COMPARISON_LATEST_JSON.exists():
        return None
    try:
        payload = json.loads(COMPARISON_LATEST_JSON.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def comparison_metrics_for_strategy(strategy_id: str) -> Optional[Dict[str, Any]]:
    snapshot = load_strategy_comparison_snapshot()
    if not snapshot:
        return None
    target = str(strategy_id or "").strip()
    for row in list(snapshot.get("strategies") or []):
        if str(row.get("id") or "") == target:
            metrics = dict(row.get("summary") or {})
            metrics["window"] = dict(snapshot.get("window") or {})
            metrics["nepse"] = dict(snapshot.get("nepse") or {})
            return metrics
    return None
