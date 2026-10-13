"""Baseline B3 — Mean-Variance Optimization (Markowitz), walk-forward.

`mean_variance_weights` dùng đúng công thức mẫu trong đặc tả kỹ thuật (mục
3.5): tối đa hoá utility `w @ mu - risk_aversion * 0.5 * w @ cov @ w` với
ràng buộc `sum(w) == 1`, `0 <= w_i <= 1` (không cho bán khống), giải bằng
`scipy.optimize.minimize`.

⚠️ Lưu ý quan trọng (đã được cảnh báo trong đặc tả): `scipy.optimize.minimize`
có thể KHÔNG hội tụ (hoặc trả kết quả suy biến) nếu covariance matrix gần
singular — trường hợp rất dễ gặp trong thực tế khi 2 mã có chuỗi return gần
như giống hệt nhau (ví dụ 2 cổ phiếu cùng ngành di chuyển gần như đồng pha).
Để tránh lỗi này, ta thêm regularization nhỏ vào covariance trước khi đưa
vào optimizer: `cov + 1e-6 * I`. Việc này làm cov luôn positive-definite
(thêm 1e-6 vào từng eigenvalue), giúp bài toán quadratic optimization luôn
hội tụ ổn định mà gần như không ảnh hưởng tới kết quả khi cov đã well-conditioned.

`run_b3_mean_variance` chạy walk-forward trên toàn bộ `state_df`: cứ mỗi
`rebalance_freq` ngày, tính lại weights từ `lookback` ngày return gần nhất
(dùng `mean_variance_weights`), rồi áp dụng weights đó (mua/bán để đạt đúng
tỷ trọng mới) cho đến lần rebalance tiếp theo.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def mean_variance_weights(returns_df: pd.DataFrame, risk_aversion: float = 1.0) -> np.ndarray:
    """Tính trọng số danh mục Markowitz (mean-variance) từ chuỗi return.

    Args:
        returns_df: DataFrame return theo ngày, columns = các mã (n_assets cột).
        risk_aversion: hệ số ngại rủi ro (risk aversion), càng lớn thì
            optimizer càng ưu tiên giảm phương sai hơn là tối đa hoá return kỳ vọng.

    Returns:
        np.ndarray độ dài n_assets, tổng xấp xỉ 1, mỗi phần tử trong [0, 1].
    """
    mu = returns_df.mean().values
    cov = returns_df.cov().values
    n = len(mu)

    # Regularization nhỏ để tránh lỗi không hội tụ khi covariance matrix
    # gần singular (ví dụ 2 mã có return gần như giống hệt nhau).
    cov = cov + 1e-6 * np.eye(n)

    def neg_utility(w):
        return -(w @ mu - risk_aversion * 0.5 * w @ cov @ w)

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(0, 1)] * n
    result = minimize(neg_utility, x0=np.ones(n) / n, bounds=bounds, constraints=constraints)

    weights = result.x
    # An toàn số học: kẹp về [0, 1] và chuẩn hoá lại tổng = 1 (SLSQP có thể
    # trả về nghiệm lệch tổng một lượng rất nhỏ do dung sai hội tụ).
    weights = np.clip(weights, 0.0, 1.0)
    total = weights.sum()
    if total > 0:
        weights = weights / total
    else:
        weights = np.ones(n) / n
    return weights


def _extract_close_prices(state_df: pd.DataFrame, tickers: Sequence[str]) -> pd.DataFrame:
    close = state_df.xs("close", axis=1, level=1)
    close = close.reindex(columns=list(tickers))
    return close


def _compute_basic_metrics(portfolio_value_history: np.ndarray) -> dict:
    pv = np.asarray(portfolio_value_history, dtype=float)
    if len(pv) < 2 or pv[0] <= 0:
        return {
            "cumulative_return": 0.0,
            "annualized_return": 0.0,
            "volatility_annualized": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
        }

    daily_returns = np.diff(pv) / pv[:-1]
    n_days = len(daily_returns)
    trading_days_per_year = 252

    cumulative_return = float(pv[-1] / pv[0] - 1)
    if n_days > 0 and pv[-1] > 0:
        annualized_return = float((pv[-1] / pv[0]) ** (trading_days_per_year / n_days) - 1)
    else:
        annualized_return = 0.0

    std = daily_returns.std(ddof=1) if n_days > 1 else 0.0
    volatility_annualized = float(std * np.sqrt(trading_days_per_year))
    mean = daily_returns.mean() if n_days > 0 else 0.0
    sharpe = float((mean / std) * np.sqrt(trading_days_per_year)) if std > 1e-12 else 0.0

    running_max = np.maximum.accumulate(pv)
    drawdown = (pv - running_max) / running_max
    max_drawdown = float(drawdown.min())

    return {
        "cumulative_return": cumulative_return,
        "annualized_return": annualized_return,
        "volatility_annualized": volatility_annualized,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }


def run_b3_mean_variance(
    state_df: pd.DataFrame,
    tickers: Sequence[str],
    lookback: int = 60,
    rebalance_freq: int = 20,
    initial_cash: float = 1_000_000_000,
    risk_aversion: float = 1.0,
) -> dict:
    """Chạy baseline B3 (Mean-Variance, walk-forward rebalance).

    Cứ mỗi `rebalance_freq` ngày giao dịch (và chỉ khi đã có đủ `lookback`
    ngày return lịch sử để ước lượng mu/cov), tính lại weights bằng
    `mean_variance_weights` trên cửa sổ `lookback` ngày return gần nhất, rồi
    tái phân bổ toàn bộ danh mục theo weights mới. Giữa 2 lần rebalance,
    weights được giữ cố định về SỐ CỔ PHIẾU (giống buy & hold), nên tỷ trọng
    thực tế (`weights_history`) vẫn trôi nhẹ theo giá cho tới lần rebalance kế tiếp.

    Trước khi có đủ `lookback` ngày dữ liệu, danh mục dùng equal-weight tạm thời.

    Args:
        state_df: DataFrame trạng thái (index=date, columns=MultiIndex(ticker, feature)).
        tickers: danh sách mã trong rổ.
        lookback: số ngày return lịch sử dùng để ước lượng mu/cov mỗi lần rebalance.
        rebalance_freq: số ngày giữa 2 lần rebalance liên tiếp.
        initial_cash: vốn ban đầu (VND).
        risk_aversion: hệ số ngại rủi ro truyền vào `mean_variance_weights`.

    Returns:
        dict gồm portfolio_value_history, weights_history, dates, metrics,
        rebalance_dates (danh sách ngày đã rebalance thật sự).
    """
    close = _extract_close_prices(state_df, tickers)
    returns = close.pct_change().fillna(0.0)
    n_days = len(close)
    n_assets = close.shape[1]
    prices = close.to_numpy()

    weights_history = np.zeros((n_days, n_assets))
    portfolio_value_history = np.zeros(n_days)
    rebalance_dates = []

    current_weights = np.full(n_assets, 1.0 / n_assets)
    shares = (initial_cash * current_weights) / prices[0]
    portfolio_value_history[0] = float(np.sum(shares * prices[0]))
    weights_history[0] = current_weights

    for t in range(1, n_days):
        if t % rebalance_freq == 0 and t >= lookback:
            window_returns = returns.iloc[t - lookback : t]
            current_weights = mean_variance_weights(window_returns, risk_aversion=risk_aversion)
            port_value_before_rebalance = float(np.sum(shares * prices[t]))
            shares = (port_value_before_rebalance * current_weights) / prices[t]
            rebalance_dates.append(close.index[t])

        portfolio_value_history[t] = float(np.sum(shares * prices[t]))
        # Tỷ trọng thực tế trôi theo giá giữa 2 lần rebalance.
        weights_history[t] = (shares * prices[t]) / portfolio_value_history[t]

    metrics = _compute_basic_metrics(portfolio_value_history)

    return {
        "portfolio_value_history": portfolio_value_history,
        "weights_history": weights_history,
        "dates": close.index,
        "rebalance_dates": rebalance_dates,
        "metrics": metrics,
    }
