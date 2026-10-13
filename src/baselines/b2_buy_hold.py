"""Baseline B2 — Buy & Hold equal-weight (không rebalance).

Chiến lược: tại ngày đầu tiên của `state_df`, mua đều (equal-weight) toàn bộ
rổ mã Giai đoạn A/B (`tickers`) bằng `initial_cash`, sau đó GIỮ NGUYÊN số cổ
phiếu đã mua cho đến hết chuỗi dữ liệu — không bán, không mua thêm, không
rebalance. Tỷ trọng (weights) theo đó sẽ TRÔI DẦN theo thời gian do giá các
mã biến động khác nhau — đây là điểm khác biệt cốt lõi so với B1 (proxy
index dùng equal-weight rebalance hàng ngày).

B2 dùng để trả lời câu hỏi: "chỉ cần nắm giữ thụ động rổ mã này có tốt
tương đương với chiến lược chủ động (B3, B4, Đề xuất) hay không?".
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def _extract_close_prices(state_df: pd.DataFrame, tickers: Sequence[str]) -> pd.DataFrame:
    """Lấy DataFrame giá đóng cửa (index=date, columns=tickers) từ state_df.

    Dùng level vị trí (level=1) cho feature "close" trong MultiIndex
    (ticker, feature) để không phụ thuộc việc các level có được đặt tên hay
    không.
    """
    close = state_df.xs("close", axis=1, level=1)
    close = close.reindex(columns=list(tickers))
    return close


def _compute_basic_metrics(portfolio_value_history: np.ndarray) -> dict:
    """Metric cơ bản, tự chứa (không phụ thuộc quantstats/src.evaluation)."""
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


def run_b2_buy_hold(
    state_df: pd.DataFrame,
    tickers: Sequence[str],
    initial_cash: float = 1_000_000_000,
) -> dict:
    """Chạy baseline B2 (Buy & Hold equal-weight, không rebalance).

    Args:
        state_df: DataFrame trạng thái (index=date, columns=MultiIndex(ticker, feature)).
        tickers: danh sách mã trong rổ Giai đoạn A/B.
        initial_cash: vốn ban đầu (VND), chia đều cho từng mã ở ngày đầu tiên.

    Returns:
        dict gồm:
            - portfolio_value_history: np.ndarray độ dài = số ngày trong state_df
            - weights_history: np.ndarray (n_days, n_assets), mỗi hàng tổng = 1,
              không âm, TRÔI THEO GIÁ (không rebalance)
            - shares: dict {ticker: số cổ phiếu nắm giữ cố định}
            - dates: index ngày tương ứng
            - metrics: dict các metric cơ bản
    """
    close = _extract_close_prices(state_df, tickers)
    n_assets = close.shape[1]

    initial_prices = close.iloc[0]
    cash_per_asset = initial_cash / n_assets
    shares = cash_per_asset / initial_prices  # pd.Series indexed by ticker, mua 1 lần duy nhất

    holdings_value = close.mul(shares, axis=1)  # (n_days, n_assets) giá trị nắm giữ theo thời gian
    portfolio_value_history = holdings_value.sum(axis=1).to_numpy()
    weights_history = holdings_value.div(holdings_value.sum(axis=1), axis=0).to_numpy()

    metrics = _compute_basic_metrics(portfolio_value_history)

    return {
        "portfolio_value_history": portfolio_value_history,
        "weights_history": weights_history,
        "shares": shares.to_dict(),
        "dates": close.index,
        "metrics": metrics,
    }
