#!/usr/bin/env python3
"""CLI for the strict autoresearch harness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from validation.research_harness import (
    VALIDATED_CORE_CONFIG,
    compare_artifacts,
    run_research_evaluation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run warm-up-aware NEPSE research evaluation.")
    parser.add_argument("--name", required=True, help="Artifact name, e.g. baseline_core")
    parser.add_argument("--signals", default=",".join(VALIDATED_CORE_CONFIG["signal_types"]))
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--rationale", required=True)
    parser.add_argument("--changed-files", default="")
    parser.add_argument("--commands", default="")
    parser.add_argument("--baseline-artifact", default="")
    parser.add_argument("--warmup-start", default="2023-01-01")
    parser.add_argument("--oos-start", default="2024-01-01")
    parser.add_argument("--oos-end", default="2025-12-31")
    args = parser.parse_args()

    config = dict(VALIDATED_CORE_CONFIG)
    config["signal_types"] = [item.strip() for item in args.signals.split(",") if item.strip()]

    artifact = run_research_evaluation(
        name=args.name,
        config=config,
        hypothesis=args.hypothesis,
        rationale=args.rationale,
        changed_files=[item.strip() for item in args.changed_files.split(",") if item.strip()],
        commands=[item.strip() for item in args.commands.split("||") if item.strip()],
        warmup_start=args.warmup_start,
        oos_start=args.oos_start,
        oos_end=args.oos_end,
    )

    output = {"artifact": artifact}
    if args.baseline_artifact:
        with open(args.baseline_artifact, "r", encoding="utf-8") as fh:
            baseline = json.load(fh)
        output["delta_vs_baseline"] = compare_artifacts(artifact, baseline)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
