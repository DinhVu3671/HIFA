"""Comparison-table reporting and significance testing for HIFA backtests.

Turns a collection of per-strategy, per-seed metric dicts (as produced by
:func:`src.evaluation.backtest.run_backtest_multi_seed`) into the mean±std
comparison table used in the thesis, plus paired significance tests between
strategies and CSV/Markdown export.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

logger = logging.getLogger(__name__)


def build_comparison_table(results: dict[str, list[dict]]) -> pd.DataFrame:
    """Build a mean±std comparison table across seeds, one row per strategy.

    Args:
        results: ``{strategy_name: [metrics_dict_seed_0, metrics_dict_seed_1, ...]}``,
            where each metrics dict is typically the ``"metrics"`` output of
            :func:`compute_all_metrics` / ``run_backtest``.

    Returns:
        DataFrame indexed by strategy name. For every metric key encountered
        (union across all strategies/seeds) there are two columns,
        ``"<metric>_mean"`` and ``"<metric>_std"`` (sample std, ddof=1; 0.0
        when only a single seed is available), plus an ``"n_seeds"`` column.
        Numeric mean/std columns (rather than a single formatted "x ± y"
        string) are used so the table stays directly testable/plottable;
        :func:`export_report` formats a human-readable "mean ± std" view for
        the Markdown export.
    """
    if not results:
        return pd.DataFrame()

    # Union of metric names, preserving first-seen order for readability.
    metric_names: list[str] = []
    for seed_list in results.values():
        for metrics_dict in seed_list:
            for key in metrics_dict.keys():
                if key not in metric_names:
                    metric_names.append(key)

    rows: dict[str, dict] = {}
    for strategy, seed_list in results.items():
        row: dict[str, float] = {}
        if not seed_list:
            logger.warning(
                "build_comparison_table: strategy '%s' has no seed results; "
                "filling row with NaN.",
                strategy,
            )
            for metric in metric_names:
                row[f"{metric}_mean"] = float("nan")
                row[f"{metric}_std"] = float("nan")
            row["n_seeds"] = 0
            rows[strategy] = row
            continue

        seeds_df = pd.DataFrame(seed_list)
        for metric in metric_names:
            if metric in seeds_df.columns:
                values = seeds_df[metric].astype(float)
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            else:
                row[f"{metric}_mean"] = float("nan")
                row[f"{metric}_std"] = float("nan")
        row["n_seeds"] = len(seed_list)
        rows[strategy] = row

    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "strategy"
    return table


def paired_significance_test(returns_a: list[float], returns_b: list[float]) -> dict:
    """Paired t-test (``scipy.stats.ttest_rel``) between two daily return series.

    Args:
        returns_a: Daily returns of strategy A.
        returns_b: Daily returns of strategy B, must be the same length as
            ``returns_a`` (paired/matched by day).

    Returns:
        ``{"t_stat": float, "p_value": float}``.

    Raises:
        ValueError: If ``returns_a`` and ``returns_b`` have different lengths.
    """
    a = np.asarray(returns_a, dtype=float)
    b = np.asarray(returns_b, dtype=float)

    if a.shape[0] != b.shape[0]:
        raise ValueError(
            "paired_significance_test requires returns_a and returns_b to have the "
            f"same length (a paired/matched-by-day t-test needs 1:1 pairs), got "
            f"len(returns_a)={a.shape[0]} vs len(returns_b)={b.shape[0]}."
        )

    if a.shape[0] < 2:
        logger.warning(
            "paired_significance_test: need at least 2 paired observations, got %d; "
            "returning NaN.",
            a.shape[0],
        )
        return {"t_stat": float("nan"), "p_value": float("nan")}

    diff = a - b
    # scipy.stats.ttest_rel divides by the (zero) std of the differences when
    # the two series are exactly identical, which yields NaN/NaN rather than
    # the intuitive "no significant difference" result. We special-case a
    # zero-variance diff (including float noise) to t=0, p=1.
    if np.allclose(diff, 0.0) or np.std(diff, ddof=1) == 0:
        return {"t_stat": 0.0, "p_value": 1.0}

    result = ttest_rel(a, b)
    return {"t_stat": float(result.statistic), "p_value": float(result.pvalue)}


def export_report(
    comparison_df: pd.DataFrame,
    significance_results: dict,
    output_dir: str,
    phase: str = "a",
) -> dict:
    """Write the comparison table (and significance results) to CSV + Markdown.

    Args:
        comparison_df: Output of :func:`build_comparison_table`.
        significance_results: ``{"StrategyA vs StrategyB": {"t_stat": ..., "p_value": ...}, ...}``.
            May be an empty dict.
        output_dir: Directory to write into (created if missing). Callers
            pass e.g. ``reports/phase_a`` for real runs, or a pytest
            ``tmp_path`` for tests.
        phase: Phase label used in the Markdown report title ("a" or "b").

    Returns:
        ``{"csv_path": str, "md_path": str}``.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "comparison_table.csv"
    md_path = out_dir / "comparison_table.md"

    comparison_df.to_csv(csv_path)

    lines = [f"# HIFA - Bang so sanh ket qua (Giai doan {phase.upper()})", ""]
    if comparison_df.empty:
        lines.append("_(khong co du lieu de so sanh)_")
    else:
        lines.append(_format_mean_std_markdown(comparison_df))
    lines.append("")

    if significance_results:
        lines.append("## Kiem dinh y nghia thong ke (paired t-test tren return hang ngay)")
        lines.append("")
        lines.append("| So sanh | t-statistic | p-value |")
        lines.append("|---|---|---|")
        for name, res in significance_results.items():
            t_stat = res.get("t_stat", float("nan"))
            p_value = res.get("p_value", float("nan"))
            lines.append(f"| {name} | {t_stat:.4f} | {p_value:.4f} |")
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")

    return {"csv_path": str(csv_path), "md_path": str(md_path)}


def _format_mean_std_markdown(comparison_df: pd.DataFrame) -> str:
    """Collapse the "<metric>_mean" / "<metric>_std" column pairs produced by
    :func:`build_comparison_table` into a single human-readable "mean ± std"
    Markdown table (falls back to a plain .to_markdown() dump for any
    columns that don't follow that convention, e.g. "n_seeds").
    """
    mean_cols = [c for c in comparison_df.columns if c.endswith("_mean")]
    metric_names = [c[: -len("_mean")] for c in mean_cols]

    display = pd.DataFrame(index=comparison_df.index)
    for metric in metric_names:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        if std_col in comparison_df.columns:
            display[metric] = [
                f"{m:.4f} ± {s:.4f}" if pd.notna(m) else "NaN"
                for m, s in zip(comparison_df[mean_col], comparison_df[std_col])
            ]
        else:
            display[metric] = comparison_df[mean_col]

    leftover_cols = [
        c for c in comparison_df.columns if not c.endswith("_mean") and not c.endswith("_std")
    ]
    for col in leftover_cols:
        display[col] = comparison_df[col]

    return display.to_markdown()


# --------------------------------------------------------------------------- #
# Best-effort CLI
# --------------------------------------------------------------------------- #


def _load_strategy_results(results_dir: Path) -> dict[str, list[dict]]:
    """Best-effort load of per-strategy seed-metrics JSON files.

    Convention: each ``<results_dir>/<strategy_name>.json`` contains a JSON
    list of metrics dicts (one per seed). This directory/format is produced
    by whatever script orchestrates ``run_backtest_multi_seed`` across
    strategies -- it is not yet wired up automatically since the
    training/baseline modules are still under development, so this loader
    just does its best and reports clearly if nothing is found.
    """
    results: dict[str, list[dict]] = {}
    if not results_dir.exists():
        return results

    for json_path in sorted(glob.glob(str(results_dir / "*.json"))):
        strategy_name = Path(json_path).stem
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                print(
                    f"[WARN] Expected a list of per-seed metrics dicts in {json_path}, "
                    f"got {type(data).__name__}. Skipping."
                )
                continue
            results[strategy_name] = data
        except Exception as exc:
            print(f"[WARN] Failed to load {json_path} ({type(exc).__name__}: {exc}). Skipping.")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and export the HIFA strategy comparison report."
    )
    parser.add_argument("--phase", type=str, choices=["a", "b"], default="a", help="Phase A or B.")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory with <strategy>.json files (list of per-seed metrics dicts). "
        "Defaults to reports/phase_<phase>/raw.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to write comparison_table.csv/.md. Defaults to reports/phase_<phase>.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir or f"reports/phase_{args.phase}/raw")
    output_dir = args.output_dir or f"reports/phase_{args.phase}"

    strategy_results = _load_strategy_results(results_dir)

    if not strategy_results:
        print(
            f"[WARN] No per-strategy result JSON files found in {results_dir}. "
            "Run the backtests first (see src.evaluation.backtest) and save each "
            "strategy's list of per-seed metrics dicts as "
            f"{results_dir}/<strategy_name>.json. Nothing to export; exiting."
        )
        return

    comparison_df = build_comparison_table(strategy_results)
    export_paths = export_report(comparison_df, {}, output_dir, phase=args.phase)
    print(f"[INFO] Wrote comparison table to: {export_paths['csv_path']} and {export_paths['md_path']}")


if __name__ == "__main__":
    main()
