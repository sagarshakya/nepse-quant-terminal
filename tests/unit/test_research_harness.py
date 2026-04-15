"""Unit tests for the strict autoresearch harness."""

from __future__ import annotations

from validation.research_harness import breakthrough_status, compare_artifacts, composite_score


def test_composite_score_penalizes_drawdown_and_turnover():
    metrics = {
        "relative_return_vs_nepse": 35.0,
        "sharpe_ratio": 1.8,
        "max_drawdown_pct": 18.0,
        "annualized_turnover_ratio": 3.5,
        "win_rate_pct": 58.0,
        "instability_penalty": 1.0,
        "concentration_penalty": 0.5,
    }
    score = composite_score(metrics, fragility_penalty=2.0)
    assert round(score, 2) == 40.40


def test_breakthrough_status_requires_positive_benchmark_and_2x_return():
    status = breakthrough_status(
        {
            "strategy_return_pct": 80.0,
            "nepse_return_pct": 35.0,
            "sharpe_ratio": 0.9,
            "max_drawdown_pct": 20.0,
            "annualized_turnover_ratio": 4.0,
        }
    )
    assert status["return_2x_nepse"] is True
    assert status["return_5x_nepse"] is False
    assert status["positive_sharpe"] is True


def test_compare_artifacts_reports_primary_deltas():
    baseline = {
        "score": 10.0,
        "primary_window": {
            "strategy_return_pct": 25.0,
            "relative_return_vs_nepse": 5.0,
            "sharpe_ratio": 0.8,
            "max_drawdown_pct": 22.0,
            "win_rate_pct": 51.0,
            "annualized_turnover_ratio": 3.0,
            "avg_exposure": 0.72,
        },
    }
    candidate = {
        "score": 14.0,
        "primary_window": {
            "strategy_return_pct": 35.0,
            "relative_return_vs_nepse": 12.0,
            "sharpe_ratio": 1.1,
            "max_drawdown_pct": 19.0,
            "win_rate_pct": 55.0,
            "annualized_turnover_ratio": 2.4,
            "avg_exposure": 0.68,
        },
    }
    delta = compare_artifacts(candidate, baseline)
    assert delta == {
        "score_delta": 4.0,
        "strategy_return_pct_delta": 10.0,
        "relative_return_vs_nepse_delta": 7.0,
        "sharpe_ratio_delta": 0.30000000000000004,
        "max_drawdown_pct_delta": -3.0,
        "win_rate_pct_delta": 4.0,
        "annualized_turnover_ratio_delta": -0.6000000000000001,
        "avg_exposure_delta": -0.039999999999999925,
    }
