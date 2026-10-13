"""Tests for src/evaluation/metrics.py and src/evaluation/report.py.

No real model/env is needed here -- everything works off hand-built
portfolio_values / return series, per the task spec. Run with:

    d:\\LUAN_VAN\\LV\\hifa\\.venv\\Scripts\\python.exe -m pytest tests/test_metrics.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import compute_all_metrics
from src.evaluation.report import (
    build_comparison_table,
    export_report,
    paired_significance_test,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _steady_growth_values(n_days: int = 252, base_return: float = 0.001, wobble: float = 0.0003):
    """Steadily increasing portfolio values: every daily return is positive
    (base_return +/- wobble, always > 0), so there should be no drawdown at
    all and a well-defined (finite, positive) Sharpe ratio.
    """
    returns = [base_return + wobble * math.sin(i) for i in range(n_days)]
    assert all(r > 0 for r in returns), "test fixture must have strictly positive daily returns"
    values = [1_000_000.0]
    for r in returns:
        values.append(values[-1] * (1 + r))
    return values


def _hand_rolled_max_drawdown(values: list[float]) -> float:
    """Standard peak-to-trough max drawdown, as a negative fraction (matches
    quantstats' sign convention), computed independently of quantstats.
    """
    arr = np.asarray(values, dtype=float)
    running_max = np.maximum.accumulate(arr)
    drawdowns = (arr - running_max) / running_max
    return float(drawdowns.min())


def _crash_scenario_values():
    """20 days of +1%/day growth, 5 days of -8%/day crash, 20 days of
    +1%/day partial recovery -- a single clear peak-to-trough drawdown.
    """
    values = [1_000_000.0]
    for _ in range(20):
        values.append(values[-1] * 1.01)
    for _ in range(5):
        values.append(values[-1] * 0.92)
    for _ in range(20):
        values.append(values[-1] * 1.01)
    return values


# --------------------------------------------------------------------------- #
# compute_all_metrics
# --------------------------------------------------------------------------- #


class TestComputeAllMetrics:
    def test_steady_growth_has_positive_sharpe_and_near_zero_drawdown(self):
        values = _steady_growth_values()
        metrics = compute_all_metrics(values)

        assert metrics["sharpe"] > 0
        assert math.isfinite(metrics["sharpe"])
        assert metrics["max_drawdown"] == pytest.approx(0.0, abs=1e-9)
        assert metrics["cumulative_return"] > 0

    def test_sharp_drawdown_matches_hand_computed_value(self):
        values = _crash_scenario_values()
        metrics = compute_all_metrics(values)

        expected_dd = _hand_rolled_max_drawdown(values)
        assert metrics["max_drawdown"] == pytest.approx(expected_dd, abs=1e-6)
        # sanity: this scenario really is a loss of roughly 30-35% peak-to-trough
        assert -0.5 < metrics["max_drawdown"] < -0.1

    def test_empty_portfolio_values_does_not_crash(self):
        metrics = compute_all_metrics([])

        assert set(metrics.keys()) == {
            "cumulative_return",
            "annualized_return",
            "sharpe",
            "sortino",
            "max_drawdown",
            "cvar_95",
        }
        for value in metrics.values():
            assert isinstance(value, float)
            assert math.isnan(value)

    def test_single_value_portfolio_does_not_crash(self):
        metrics = compute_all_metrics([1_000_000.0])

        assert metrics["cumulative_return"] == 0.0
        assert metrics["max_drawdown"] == 0.0
        # not enough data for a return-based ratio -- must be NaN, not a crash
        assert math.isnan(metrics["sharpe"])
        assert math.isnan(metrics["sortino"])
        assert math.isnan(metrics["annualized_return"])
        assert math.isnan(metrics["cvar_95"])

    def test_none_and_empty_benchmark_lengths_are_handled(self):
        values = _steady_growth_values(n_days=30)

        metrics_no_bench = compute_all_metrics(values, benchmark_values=None)
        assert "excess_return" not in metrics_no_bench
        assert "information_ratio" not in metrics_no_bench

        metrics_short_bench = compute_all_metrics(values, benchmark_values=[1_000_000.0])
        assert math.isnan(metrics_short_bench["excess_return"])
        assert math.isnan(metrics_short_bench["information_ratio"])

    def test_benchmark_adds_excess_return_and_information_ratio(self):
        port = _steady_growth_values(n_days=60, base_return=0.0015, wobble=0.0003)
        bench = _steady_growth_values(n_days=60, base_return=0.0008, wobble=0.0002)

        metrics = compute_all_metrics(port, benchmark_values=bench)

        expected_excess = (port[-1] / port[0] - 1) - (bench[-1] / bench[0] - 1)
        assert metrics["excess_return"] == pytest.approx(expected_excess, rel=1e-9)
        # portfolio outgrows benchmark every day here -> positive excess return
        # and a well-defined, positive information ratio.
        assert metrics["excess_return"] > 0
        assert metrics["information_ratio"] > 0
        assert math.isfinite(metrics["information_ratio"])

    def test_all_zero_returns_does_not_raise(self):
        # constant portfolio value -> zero variance in returns, a classic
        # quantstats division-by-zero trap. Must degrade to NaN, not crash.
        values = [1_000_000.0] * 50
        metrics = compute_all_metrics(values)
        assert metrics["cumulative_return"] == 0.0
        assert metrics["max_drawdown"] == 0.0
        # sharpe is 0/0 for a constant series -- must not raise, NaN is acceptable
        assert isinstance(metrics["sharpe"], float)


# --------------------------------------------------------------------------- #
# build_comparison_table
# --------------------------------------------------------------------------- #


class TestBuildComparisonTable:
    def _fake_results(self):
        return {
            "strategy_a": [
                {"sharpe": 1.0, "max_drawdown": -0.10},
                {"sharpe": 2.0, "max_drawdown": -0.20},
                {"sharpe": 3.0, "max_drawdown": -0.30},
            ],
            "strategy_b": [
                {"sharpe": 0.5, "max_drawdown": -0.05},
                {"sharpe": 1.5, "max_drawdown": -0.15},
                {"sharpe": 2.5, "max_drawdown": -0.25},
            ],
        }

    def test_mean_and_std_match_hand_computation(self):
        results = self._fake_results()
        table = build_comparison_table(results)

        assert set(table.index) == {"strategy_a", "strategy_b"}
        assert {"sharpe_mean", "sharpe_std", "max_drawdown_mean", "max_drawdown_std"}.issubset(
            table.columns
        )

        for strategy, seed_dicts in results.items():
            sharpe_values = np.array([d["sharpe"] for d in seed_dicts])
            dd_values = np.array([d["max_drawdown"] for d in seed_dicts])

            assert table.loc[strategy, "sharpe_mean"] == pytest.approx(sharpe_values.mean())
            assert table.loc[strategy, "sharpe_std"] == pytest.approx(
                sharpe_values.std(ddof=1)
            )
            assert table.loc[strategy, "max_drawdown_mean"] == pytest.approx(dd_values.mean())
            assert table.loc[strategy, "max_drawdown_std"] == pytest.approx(
                dd_values.std(ddof=1)
            )
            assert table.loc[strategy, "n_seeds"] == len(seed_dicts)

    def test_empty_results_returns_empty_dataframe(self):
        table = build_comparison_table({})
        assert isinstance(table, pd.DataFrame)
        assert table.empty

    def test_strategy_with_no_seeds_is_all_nan(self):
        results = self._fake_results()
        results["strategy_c_no_seeds"] = []
        table = build_comparison_table(results)

        assert table.loc["strategy_c_no_seeds", "n_seeds"] == 0
        assert math.isnan(table.loc["strategy_c_no_seeds", "sharpe_mean"])


# --------------------------------------------------------------------------- #
# paired_significance_test
# --------------------------------------------------------------------------- #


class TestPairedSignificanceTest:
    def test_identical_series_gives_p_value_near_one(self):
        returns = [0.01, -0.02, 0.03, -0.01, 0.02, 0.015, -0.005, 0.008] * 5
        result = paired_significance_test(returns, list(returns))

        assert result["p_value"] == pytest.approx(1.0, abs=1e-6)
        assert result["t_stat"] == pytest.approx(0.0, abs=1e-6)

    def test_clearly_different_series_gives_small_p_value(self):
        rng = np.random.default_rng(42)
        returns_a = list(rng.normal(0.01, 0.001, 200))
        returns_b = list(rng.normal(-0.01, 0.001, 200))

        result = paired_significance_test(returns_a, returns_b)

        assert result["p_value"] < 0.01
        assert math.isfinite(result["t_stat"])

    def test_mismatched_lengths_raise_value_error(self):
        with pytest.raises(ValueError):
            paired_significance_test([0.01, 0.02, 0.03], [0.01, 0.02])

    def test_too_short_returns_nan_instead_of_crashing(self):
        result = paired_significance_test([0.01], [0.02])
        assert math.isnan(result["t_stat"])
        assert math.isnan(result["p_value"])


# --------------------------------------------------------------------------- #
# export_report
# --------------------------------------------------------------------------- #


class TestExportReport:
    def test_creates_csv_and_markdown_in_given_dir(self, tmp_path):
        results = {
            "strategy_a": [{"sharpe": 1.0, "max_drawdown": -0.1}],
            "strategy_b": [{"sharpe": 2.0, "max_drawdown": -0.2}],
        }
        table = build_comparison_table(results)
        significance = {
            "strategy_a vs strategy_b": {"t_stat": 1.2345, "p_value": 0.6789},
        }

        output_dir = tmp_path / "reports_phase_a"
        paths = export_report(table, significance, str(output_dir), phase="a")

        csv_path = output_dir / "comparison_table.csv"
        md_path = output_dir / "comparison_table.md"

        assert csv_path.exists()
        assert md_path.exists()
        assert paths["csv_path"] == str(csv_path)
        assert paths["md_path"] == str(md_path)

        # CSV round-trips the numeric data written.
        csv_df = pd.read_csv(csv_path, index_col=0)
        assert csv_df.loc["strategy_a", "sharpe_mean"] == pytest.approx(1.0)
        assert csv_df.loc["strategy_b", "sharpe_mean"] == pytest.approx(2.0)

        md_text = md_path.read_text(encoding="utf-8")
        assert "strategy_a" in md_text
        assert "strategy_b" in md_text
        assert "1.2345" in md_text
        assert "0.6789" in md_text

    def test_does_not_touch_real_reports_directory(self, tmp_path, monkeypatch):
        # Regression guard: export_report must only ever write under the
        # explicit output_dir it is given, never a hardcoded reports/ path.
        monkeypatch.chdir(tmp_path)
        table = build_comparison_table(
            {"strategy_a": [{"sharpe": 1.0}]}
        )
        export_report(table, {}, str(tmp_path / "out"), phase="b")

        assert not (tmp_path / "reports").exists()
        assert (tmp_path / "out" / "comparison_table.csv").exists()
