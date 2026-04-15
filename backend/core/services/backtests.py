"""BacktestService — read-only listing of existing backtest artifacts.

Phase 0 scope: enumerate + load JSON artifacts that already exist on disk
(autoresearch / reports / research). Running new backtests is intentionally
deferred — the GUI MVP only needs to *browse* and *compare* saved runs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

from backend.core.types import BacktestSummary


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


class BacktestService:
    def __init__(self, roots: Optional[Iterable[Path]] = None):
        root = _project_root()
        self._roots: list[Path] = list(roots) if roots else [
            root / "research",
            root / "reports",
        ]

    # ---------- listing ---------------------------------------------------
    def list(self) -> list[BacktestSummary]:
        summaries: list[BacktestSummary] = []
        for r in self._roots:
            if not r.exists():
                continue
            for p in r.rglob("artifact_*.json"):
                s = self._load_summary(p)
                if s is not None:
                    summaries.append(s)
            for p in r.rglob("best_config_v2.json"):
                s = self._load_summary(p)
                if s is not None:
                    summaries.append(s)
        # stable ordering: highest total_return first, tie-broken by artifact_id
        summaries.sort(key=lambda s: (-(s.total_return or 0.0), s.artifact_id))
        return summaries

    def load(self, artifact_id: str) -> Optional[dict]:
        for r in self._roots:
            for p in r.rglob(f"*{artifact_id}*.json"):
                try:
                    return json.loads(p.read_text())
                except Exception:
                    continue
        return None

    # ---------- helpers ---------------------------------------------------
    def _load_summary(self, path: Path) -> Optional[BacktestSummary]:
        try:
            blob = json.loads(path.read_text())
        except Exception:
            return None

        def _get(keys: list[str], default=0.0) -> float:
            for k in keys:
                if k in blob:
                    try:
                        return float(blob[k])
                    except (TypeError, ValueError):
                        pass
                if isinstance(blob.get("metrics"), dict) and k in blob["metrics"]:
                    try:
                        return float(blob["metrics"][k])
                    except (TypeError, ValueError):
                        pass
            return float(default)

        name = (
            blob.get("name")
            or blob.get("run_id")
            or blob.get("config_name")
            or path.stem
        )
        return BacktestSummary(
            artifact_id=path.stem,
            name=str(name),
            total_return=_get(["total_return", "return", "pct_return"], 0.0),
            sharpe=_get(["sharpe", "sharpe_ratio"], 0.0),
            max_dd=_get(["max_dd", "max_drawdown", "mdd"], 0.0),
            n_trades=int(_get(["n_trades", "num_trades", "trades"], 0)),
            path=str(path),
        )
