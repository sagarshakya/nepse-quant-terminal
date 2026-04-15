#!/usr/bin/env python3
"""Re-run C31 baseline with non-tradable symbols (NEPSE, SECTOR::*) filtered out."""
import sys
sys.path.insert(0, '/Users/samriddhagc/Desktop/Projects/Nepse_quant_clean')

from validation.research_harness import run_research_evaluation

C31_CONFIG = dict(
    holding_days=40,
    max_positions=5,
    signal_types=["volume","quality","low_vol","mean_reversion","quarterly_fundamental","xsec_momentum"],
    rebalance_frequency=5,
    stop_loss_pct=0.08,
    trailing_stop_pct=0.10,
    use_regime_filter=True,
    sector_limit=0.35,
    regime_max_positions={"bull":5,"neutral":4,"bear":2},
    bear_threshold=-0.05,
    initial_capital=1_000_000,
    regime_sector_limits={"bull":0.50,"neutral":0.35,"bear":0.25},
    use_trailing_stop=True,
)

print("Running C31 clean baseline (NEPSE + SECTOR:: filtered)...")
artifact = run_research_evaluation(
    name="C31_clean_no_indices",
    config=C31_CONFIG,
    hypothesis="C31 re-run after filtering NEPSE and SECTOR:: from tradable universe",
    rationale="Verify C31 score is unchanged after removing 10 non-tradable symbols.",
    warmup_start="2023-01-01",
    oos_start="2024-01-01",
    oos_end="2025-12-31",
    adjacent_slices=[
        ("2023-01-01", "2024-12-31"),
        ("2024-01-01", "2025-12-31"),
        ("2025-01-01", "2025-12-31"),
    ],
)

pw = artifact["primary_window"]
slices = artifact.get("slice_stability", [])
s2025 = next((s for s in slices if str(s.get("window_start","")).startswith("2025")), {})

print(f"\n=== C31 CLEAN (filtered) vs original ===")
print(f"  Return:  {pw['strategy_return_pct']:.2f}%  (original: 93.48%)")
print(f"  NEPSE:   {pw['nepse_return_pct']:.2f}%  (original: 27.84%)")
print(f"  Sharpe:  {pw['sharpe_ratio']:.3f}  (original: 2.279)")
print(f"  MaxDD:   {pw['max_drawdown_pct']:.2f}%  (original: 12.42%)")
print(f"  Trades:  {pw['trade_count']}  (original: 63)")
print(f"  Score:   {artifact['score']:.2f}  (original: 93.42)")
print(f"  2025:    ret={s2025.get('strategy_return_pct',0):.2f}% | trades={s2025.get('trade_count',0)} | exp={s2025.get('avg_exposure',0):.3f}")
print(f"\nSaved to: {artifact['artifact_path']}")
