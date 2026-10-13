"""Portfolio evaluation metrics for HIFA.

Wraps ``quantstats`` to compute a standard set of backtest metrics from a
portfolio value history (and, optionally, a benchmark value history), with
defensive error handling so that a single failing metric never crashes the
whole evaluation pipeline.

Real API quirks discovered while implementing this module (see also the
final report handed back to the task owner):

- ``quantstats.stats.max_drawdown`` (and ``to_drawdown_series``) internally do
  ``prices.index[0] - time_delta`` where ``time_delta`` is a
  ``pandas.Timedelta``. This crashes with a plain integer ``RangeIndex``
  (``TypeError: unsupported operand type(s) for -: 'int' and 'Timedelta'``).
  A ``DatetimeIndex`` is therefore required — we build one internally so
  callers can keep passing plain ``list[float]``.
- ``quantstats.stats.cvar`` / ``var`` take a ``confidence`` kwarg for the
  confidence level (default already ``0.95``); ``sigma`` is a *volatility
  multiplier*, not a confidence level. The example in the implementation
  plan calls ``qs.stats.cvar(returns, sigma=0.95)`` which does **not**
  compute a 95% CVaR — it scales the VaR threshold by 0.95. We use
  ``confidence=0.95`` instead to get the metric the spec actually wants.
- ``quantstats.stats.information_ratio(returns, benchmark)`` exists but
  returns a *non-annualized* daily ratio (no ``sqrt(252)`` factor), unlike
  ``sharpe``/``sortino`` which annualize by default. To keep all ratio
  metrics on a comparable (annualized) scale we compute the information
  ratio manually per the fallback formula given in the spec
  (``mean(excess_returns) / std(excess_returns) * sqrt(252)``).
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
import quantstats as qs

logger = logging.getLogger(__name__)

_TRADING_DAYS_PER_YEAR = 252

# Metrics always present in the output dict (portfolio-only).
_BASE_METRIC_KEYS = (
    "cumulative_return",
    "annualized_return",
    "sharpe",
    "sortino",
    "max_drawdown",
    "cvar_95",
)

# Extra metrics added when a benchmark is supplied.
_BENCHMARK_METRIC_KEYS = ("excess_return", "information_ratio")


def _values_to_returns(values: np.ndarray) -> np.ndarray:
    """Simple daily returns from a price/portfolio-value series."""
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = np.diff(values) / values[:-1]
    return returns


def _to_series(values: np.ndarray, name: str = "value") -> pd.Series:
    """Wrap a numeric array in a pandas Series with a DatetimeIndex.

    A DatetimeIndex is required for some quantstats functions (notably
    ``max_drawdown``) which do date arithmetic internally; a plain
    ``RangeIndex`` raises a TypeError there. The actual dates are
    arbitrary (metrics below are computed on trading-day counts, not
    calendar spans), so a synthetic daily range is sufficient.
    """
    index = pd.date_range("2000-01-01", periods=len(values), freq="D")
    return pd.Series(np.asarray(values, dtype=float), index=index, name=name)


def _safe_metric(name: str, fn):
    """Run ``fn`` and return its float value, or NaN (with a logged warning)
    if it raises. Individual metric failures must never abort the whole
    ``compute_all_metrics`` call.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            value = fn()
        if value is None:
            return float("nan")
        return float(value)
    except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
        logger.warning("compute_all_metrics: failed to compute '%s': %s", name, exc)
        return float("nan")


def _empty_metrics(n: int, with_benchmark: bool) -> dict:
    """Reasonable NaN/0 defaults for portfolio_values shorter than 2 points."""
    metrics = {
        "cumulative_return": 0.0 if n == 1 else float("nan"),
        "annualized_return": float("nan"),
        "sharpe": float("nan"),
        "sortino": float("nan"),
        "max_drawdown": 0.0 if n >= 1 else float("nan"),
        "cvar_95": float("nan"),
    }
    if with_benchmark:
        metrics["excess_return"] = float("nan")
        metrics["information_ratio"] = float("nan")
    return metrics


def compute_all_metrics(
    portfolio_values: list[float],
    benchmark_values: list[float] | None = None,
    turnover_history: list[float] | None = None,
) -> dict:
    """Compute standard backtest metrics from a portfolio value history.

    Args:
        portfolio_values: Portfolio value (e.g. NAV) at each timestep,
            in chronological order. Needs at least 2 points to produce
            meaningful returns-based metrics.
        benchmark_values: Optional benchmark value history, same
            convention as ``portfolio_values``. When provided (and long
            enough), ``excess_return`` and ``information_ratio`` are added.
        turnover_history: Optional per-step turnover series (``info["turnover"]``
            from the env). When provided, ``avg_turnover`` is added (De-Cuong §9).

    Returns:
        dict with keys ``cumulative_return``, ``annualized_return``,
        ``sharpe``, ``sortino``, ``max_drawdown``, ``cvar_95``, and
        (when a usable benchmark is given) ``excess_return`` and
        ``information_ratio``. Any metric that cannot be computed is set
        to ``float("nan")`` instead of raising.
    """
    portfolio_values = list(portfolio_values) if portfolio_values is not None else []
    n = len(portfolio_values)
    with_benchmark = benchmark_values is not None

    if n < 2:
        logger.warning(
            "compute_all_metrics: portfolio_values has length %d (<2); "
            "returning NaN/0 placeholder metrics instead of crashing.",
            n,
        )
        return _empty_metrics(n, with_benchmark)

    values = np.asarray(portfolio_values, dtype=float)
    returns = _values_to_returns(values)
    returns_series = _to_series(returns, name="returns")
    prices_series = _to_series(values, name="prices")

    metrics: dict = {}
    metrics["cumulative_return"] = _safe_metric(
        "cumulative_return", lambda: values[-1] / values[0] - 1.0
    )
    metrics["annualized_return"] = _safe_metric(
        "annualized_return", lambda: qs.stats.cagr(returns_series)
    )
    metrics["sharpe"] = _safe_metric("sharpe", lambda: qs.stats.sharpe(returns_series))
    metrics["sortino"] = _safe_metric("sortino", lambda: qs.stats.sortino(returns_series))
    # NOTE: pass *prices*, not returns, to max_drawdown -- see module docstring.
    metrics["max_drawdown"] = _safe_metric(
        "max_drawdown", lambda: qs.stats.max_drawdown(prices_series)
    )
    # NOTE: confidence=0.95 (not sigma=0.95) is the correct kwarg for a 95% CVaR.
    metrics["cvar_95"] = _safe_metric(
        "cvar_95", lambda: qs.stats.cvar(returns_series, confidence=0.95)
    )

    if with_benchmark:
        metrics.update(_compute_benchmark_metrics(values, returns, benchmark_values))

    if turnover_history is not None:
        metrics["avg_turnover"] = average_turnover(turnover_history)

    return metrics


def _compute_benchmark_metrics(
    values: np.ndarray, returns: np.ndarray, benchmark_values: list[float]
) -> dict:
    bench_values = np.asarray(
        list(benchmark_values) if benchmark_values is not None else [], dtype=float
    )
    nb = len(bench_values)

    if nb < 2:
        logger.warning(
            "compute_all_metrics: benchmark_values has length %d (<2); "
            "excess_return/information_ratio set to NaN.",
            nb,
        )
        return {"excess_return": float("nan"), "information_ratio": float("nan")}

    result = {}

    result["excess_return"] = _safe_metric(
        "excess_return",
        lambda: (values[-1] / values[0] - 1.0) - (bench_values[-1] / bench_values[0] - 1.0),
    )

    bench_returns = _values_to_returns(bench_values)
    m = min(len(returns), len(bench_returns))
    if m != len(returns) or m != len(bench_returns):
        logger.warning(
            "compute_all_metrics: portfolio (%d) and benchmark (%d) return "
            "series lengths differ; truncating both to %d points to align "
            "for information_ratio.",
            len(returns),
            len(bench_returns),
            m,
        )

    def _information_ratio():
        if m < 2:
            return float("nan")
        excess = returns[:m] - bench_returns[:m]
        std = excess.std(ddof=1)
        if std == 0 or not np.isfinite(std):
            return float("nan")
        return excess.mean() / std * np.sqrt(_TRADING_DAYS_PER_YEAR)

    # qs.stats.information_ratio exists but returns a *non-annualized* ratio
    # (see module docstring) -- we compute the annualized version manually
    # so it is on the same scale as sharpe/sortino.
    result["information_ratio"] = _safe_metric("information_ratio", _information_ratio)

    return result


# ---------------------------------------------------------------------------
# Chi phí thực thi & độ ổn định (De-Cuong §6.4.4, §9)
# ---------------------------------------------------------------------------
def average_turnover(turnover_history: list[float]) -> float:
    """Turnover trung bình mỗi bước (De-Cuong §9 — chi phí thực thi).

    ``turnover_history`` là chuỗi turnover từng ngày do env trả về trong
    ``info["turnover"]`` (= Σ|w_t − w_{t-1}|). Trả về trung bình cộng; chuỗi
    rỗng -> NaN.
    """
    arr = np.asarray(list(turnover_history) if turnover_history is not None else [], dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))


def is_policy_collapsed(
    weights_history: "list | np.ndarray",
    concentration_threshold: float = 0.99,
    frac_days_threshold: float = 0.9,
) -> bool:
    """Phát hiện chính sách "sập" (De-Cuong §6.4.4 — độ ổn định huấn luyện).

    Chính sách bị coi là sập nếu trong phần lớn số ngày, danh mục dồn gần như
    toàn bộ vào MỘT ô duy nhất (1 mã hoặc toàn cash) — tức max tỷ trọng mỗi
    ngày >= ``concentration_threshold`` ở ít nhất ``frac_days_threshold`` số
    ngày.

    Args:
        weights_history: mảng (n_days, n_assets+1) tỷ trọng theo ngày (gồm cash
            ở cột cuối), mỗi hàng tổng ~1.
        concentration_threshold: ngưỡng coi là "dồn hết" (mặc định 0.99).
        frac_days_threshold: tỷ lệ ngày vượt ngưỡng để kết luận sập (mặc định 0.9).

    Returns:
        True nếu chính sách sập, False nếu không. Mảng rỗng -> False.
    """
    w = np.asarray(weights_history, dtype=float)
    if w.ndim != 2 or w.shape[0] == 0:
        return False
    max_per_day = w.max(axis=1)
    frac_concentrated = float((max_per_day >= concentration_threshold).mean())
    return frac_concentrated >= frac_days_threshold


def policy_collapse_rate(collapsed_flags: list[bool]) -> float:
    """Tỷ lệ seed bị sập chính sách (De-Cuong §6.4.4).

    ``collapsed_flags`` là list bool, mỗi phần tử = kết quả ``is_policy_collapsed``
    cho 1 seed. Trả về % seed sập (0.0–1.0). List rỗng -> NaN.
    """
    flags = list(collapsed_flags) if collapsed_flags is not None else []
    if not flags:
        return float("nan")
    return float(np.mean([1.0 if f else 0.0 for f in flags]))
