#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.trading import strategy_registry


ACCOUNTS_REGISTRY = PROJECT_ROOT / "data" / "runtime" / "accounts" / "registry.json"
ACCOUNTS_DIR = PROJECT_ROOT / "data" / "runtime" / "accounts"


def _load_accounts() -> list[dict[str, Any]]:
    if not ACCOUNTS_REGISTRY.exists():
        return []
    try:
        payload = json.loads(ACCOUNTS_REGISTRY.read_text(encoding="utf-8"))
    except Exception:
        return []
    return list(payload.get("accounts") or [])


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _launch_account(account: dict[str, Any]) -> dict[str, Any]:
    account_id = str(account.get("id") or "").strip()
    strategy_id = str(account.get("strategy_id") or strategy_registry.default_strategy_for_account(account_id)).strip()
    strategy = strategy_registry.load_strategy(strategy_id)
    if strategy is None:
        raise RuntimeError(f"{account_id}: unknown strategy {strategy_id}")

    account_dir = ACCOUNTS_DIR / account_id
    portfolio_path = account_dir / "paper_portfolio.csv"
    pid_path = account_dir / "live_trader.pid"
    log_path = account_dir / "live_trader.log"

    if pid_path.exists():
        try:
            pid = int(pid_path.read_text(encoding="utf-8").strip())
        except Exception:
            pid = 0
        if pid > 0 and _pid_is_alive(pid):
            return {
                "account_id": account_id,
                "strategy_id": strategy_id,
                "status": "already_running",
                "pid": pid,
                "log": str(log_path),
            }

    capital = float((strategy.get("config") or {}).get("initial_capital") or 1_000_000.0)
    cmd = [
        sys.executable,
        "-m",
        "backend.trading.live_trader",
        "--mode",
        "paper",
        "--continuous",
        "--headless",
        "--no-telegram",
        "--strategy-id",
        strategy_id,
        "--capital",
        str(capital),
        "--paper-portfolio",
        str(portfolio_path),
    ]

    account_dir.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as handle:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pid_path.write_text(f"{proc.pid}\n", encoding="utf-8")
    return {
        "account_id": account_id,
        "strategy_id": strategy_id,
        "status": "started",
        "pid": proc.pid,
        "log": str(log_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch one live_trader paper process per saved account strategy.")
    parser.add_argument(
        "--accounts",
        default="account_1,account_2,account_3",
        help="Comma-separated account ids to start",
    )
    args = parser.parse_args()

    targets = {token.strip() for token in str(args.accounts).split(",") if token.strip()}
    accounts = [row for row in _load_accounts() if str(row.get("id") or "") in targets]
    if not accounts:
        raise SystemExit("No matching accounts found")

    results = [_launch_account(account) for account in accounts]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
