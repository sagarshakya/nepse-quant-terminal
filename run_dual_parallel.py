#!/usr/bin/env python3
"""
Parallel launcher for dual-window research experiments.

Launches up to N_PARALLEL experiments simultaneously, each in a separate process.
Results are collected when all processes finish.

Usage:
    python3 run_dual_parallel.py [--batch 1|2|all] [--n-parallel 4]
"""
import sys
import json
import subprocess
import argparse
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
ARTIFACT_DIR = PROJECT_ROOT / "reports/autoresearch/artifacts"
LOG_DIR = PROJECT_ROOT / "reports/autoresearch/parallel_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# All experiment short names from run_dual_window_research.py
# C53 = C31 baseline, C54 = temp_forward_winner — highest priority
BATCH_1_PRIORITY = [
    "C54_temp_winner",    # The key pending measurement
    "C55_hold35",         # C31 + shorter hold
    "C56_hold30",         # C31 + even shorter hold
    "C57_neutral5_bear3", # More neutral exposure
    "C58_stop12",         # Wider stop
]

BATCH_2 = [
    "C59_no_trailing",
    "C60_bull6",
    "C61_sector65",
    "C62_bear_thresh07",
    "C63_4signal",
]

BATCH_3 = [
    "C64_stop10_trail12",
    "C65_2signal",
    "C66_rebalance3",
    "C67_hold45",
    "C68_wider_neutral",
]

ALL_EXPERIMENTS = BATCH_1_PRIORITY + BATCH_2 + BATCH_3


def run_experiment(short_name: str) -> subprocess.Popen:
    """Launch a single experiment as a background subprocess."""
    log_path = LOG_DIR / f"{short_name}.log"
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "run_dual_window_research.py"),
        "--only", short_name,
    ]
    with open(log_path, "w") as f:
        proc = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
        )
    return proc, log_path


def wait_for_batch(procs_logs: list, poll_interval: int = 30):
    """Wait for a batch of processes to complete, showing periodic status."""
    start = time.time()
    while True:
        still_running = [(p, l, n) for p, l, n in procs_logs if p.poll() is None]
        done = [(p, l, n) for p, l, n in procs_logs if p.poll() is not None]

        elapsed = int(time.time() - start)
        print(f"\r[{elapsed}s] Running: {len(still_running)} | Done: {len(done)}", end="", flush=True)

        # Print progress for any that just finished
        for p, log_path, name in done:
            artifact_path = ARTIFACT_DIR / f"{name}.json"
            if artifact_path.exists():
                try:
                    d = json.loads(artifact_path.read_text())
                    fw = d.get("forward_window", {})
                    hw = d.get("historic_window", {})
                    bt = "BREAKTHROUGH!" if d.get("breakthrough") else ""
                    print(f"\n  DONE {name}: Fwd {fw.get('strategy_return_pct',0):+.2f}% ({fw.get('nepse_multiple',0):.2f}x) | His {hw.get('strategy_return_pct',0):+.2f}% ({hw.get('nepse_multiple',0):.2f}x) {bt}")
                    # Remove from still-running check on next iteration
                except Exception:
                    pass

        if not still_running:
            break

        time.sleep(poll_interval)

    print(f"\nBatch complete in {int(time.time() - start)}s")


def collect_results(experiment_names: list) -> list:
    """Read all artifacts and print summary."""
    results = []
    missing = []
    for name in experiment_names:
        artifact_path = ARTIFACT_DIR / f"{name}.json"
        if artifact_path.exists():
            try:
                d = json.loads(artifact_path.read_text())
                results.append(d)
            except Exception as e:
                print(f"  Error reading {artifact_path}: {e}")
        else:
            missing.append(name)

    if missing:
        print(f"\nMissing artifacts: {missing}")

    # Sort by forward multiple * historic multiple
    results.sort(key=lambda r: (
        r.get("forward_window", {}).get("nepse_multiple", 0) *
        r.get("historic_window", {}).get("nepse_multiple", 0)
    ), reverse=True)

    print(f"\n{'='*80}")
    print("DUAL-WINDOW RESEARCH SUMMARY")
    print(f"Breakthrough: Fwd ≥1.5x AND Historic ≥2.73x NEPSE")
    print(f"C31 baseline: Fwd=0.17x, Historic=2.73x")
    print(f"{'='*80}")
    print(f"{'Name':<45} {'Fwd%':>7} {'FwdX':>6} {'His%':>7} {'HisX':>6} {'Status':>14}")
    print(f"{'-'*45} {'-'*7} {'-'*6} {'-'*7} {'-'*6} {'-'*14}")

    for r in results:
        fw = r.get("forward_window", {})
        hw = r.get("historic_window", {})
        bt = "BREAKTHROUGH" if r.get("breakthrough") else ""
        fw_x = fw.get("nepse_multiple", 0)
        hw_x = hw.get("nepse_multiple", 0)
        fw_p = fw.get("strategy_return_pct", 0)
        hw_p = hw.get("strategy_return_pct", 0)
        fwd_ok = ">" if fw_x >= 1.5 else " "
        his_ok = ">" if hw_x >= 2.73 else " "
        print(f"{r['name']:<45} {fw_p:>+6.2f}% {fwd_ok}{fw_x:>4.2f}x {hw_p:>+6.2f}% {his_ok}{hw_x:>4.2f}x  {bt}")

    breakthroughs = [r for r in results if r.get("breakthrough")]
    print(f"\nBreakthroughs: {len(breakthroughs)} / {len(results)}")

    if breakthroughs:
        print("\n*** BREAKTHROUGH CANDIDATES ***")
        for r in breakthroughs:
            fw = r["forward_window"]
            hw = r["historic_window"]
            cfg = r.get("config", {})
            print(f"\n  {r['name']}")
            print(f"    Forward:  {fw['strategy_return_pct']:+.2f}% vs NEPSE {fw['nepse_return_pct']:+.2f}% = {fw['nepse_multiple']:.2f}x | Sharpe {fw['sharpe_ratio']:.3f} | Trades {fw['trade_count']}")
            print(f"    Historic: {hw['strategy_return_pct']:+.2f}% vs NEPSE {hw['nepse_return_pct']:+.2f}% = {hw['nepse_multiple']:.2f}x | Sharpe {hw['sharpe_ratio']:.3f} | Trades {hw['trade_count']}")
            print(f"    Signals: {cfg.get('signal_types')}")
            print(f"    Hold: {cfg.get('holding_days')}d | Stop: {cfg.get('stop_loss_pct',0)*100:.0f}% | Trailing: {cfg.get('trailing_stop_pct',0)*100:.0f}%")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", default="1", choices=["1", "2", "3", "all"], help="Which batch to run")
    parser.add_argument("--n-parallel", type=int, default=5, help="Max parallel experiments")
    parser.add_argument("--collect-only", action="store_true", help="Just print summary of existing artifacts")
    args = parser.parse_args()

    if args.collect_only:
        collect_results(ALL_EXPERIMENTS)
        return

    if args.batch == "1":
        experiments = BATCH_1_PRIORITY
    elif args.batch == "2":
        experiments = BATCH_2
    elif args.batch == "3":
        experiments = BATCH_3
    else:
        experiments = ALL_EXPERIMENTS

    print(f"Launching {len(experiments)} experiments (max {args.n_parallel} parallel)")
    print(f"Experiments: {experiments}")
    print(f"Logs: {LOG_DIR}")
    print()

    # Launch in chunks of n_parallel
    for chunk_start in range(0, len(experiments), args.n_parallel):
        chunk = experiments[chunk_start:chunk_start + args.n_parallel]
        print(f"\n--- Launching chunk: {chunk} ---")

        procs_logs = []
        for name in chunk:
            # Skip if artifact already exists
            artifact_path = ARTIFACT_DIR / f"{name}.json"
            if artifact_path.exists():
                print(f"  [skip] {name} — artifact already exists")
                continue
            proc, log_path = run_experiment(name)
            procs_logs.append((proc, log_path, name))
            print(f"  [launched] {name} (PID {proc.pid}, log: {log_path.name})")

        if procs_logs:
            wait_for_batch(procs_logs)

    # Final summary
    print("\n\nFINAL SUMMARY:")
    collect_results(experiments)


if __name__ == "__main__":
    main()
