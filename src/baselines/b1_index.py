"""Baseline B1 — VN30-Index (buy & hold benchmark).

Baseline B1 mô phỏng chiến lược "mua và giữ" chỉ số VN30 — đây là benchmark
thụ động chuẩn mà bất kỳ chiến lược chủ động nào (kể cả Đề xuất HIFA)
cũng phải đánh bại thì mới có ý nghĩa.

QUAN TRỌNG — về nguồn dữ liệu chỉ số:
    `state_df` (do `src/data/build_dataset.py::build_state_dataframe` tạo ra)
    chỉ chứa dữ liệu OHLCV + chỉ báo kỹ thuật + sentiment cho **từng mã cổ
    phiếu** trong rổ Giai đoạn A/B (`tickers`), KHÔNG chứa giá đóng cửa của
    chỉ số VN30 thật.

    Vì vậy, mặc định (khi không truyền `index_close`), hàm `run_b1_index`
    dùng một **proxy equal-weight**: mỗi ngày lấy trung bình cộng (equal
    weight, ngầm định rebalance mỗi ngày) return của toàn bộ mã có trong
    `tickers`, coi đó là return xấp xỉ của "chỉ số". Đây CHỈ LÀ PROXY vì:
      - VN30 thật là index vốn hoá gia quyền (market-cap weighted), không
        phải equal-weight.
      - VN30 gồm 30 mã, còn `tickers` ở Giai đoạn A chỉ có 5 mã (Giai đoạn B
        có 10 mã) — là tập con nhỏ của VN30.

    Nếu có dữ liệu chỉ số VN30 thật (ví dụ tải qua `vnstock`), hãy truyền
    vào tham số `index_close` (một `pd.Series` giá đóng cửa VN30 theo ngày,
    cùng lưới ngày với `state_df.index`) — khi đó hàm sẽ mô phỏng buy-and-hold
    THẬT trên chỉ số thay vì dùng proxy equal-weight.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def _extract_close_prices(state_df: pd.DataFrame, tickers: Sequence[str]) -> pd.DataFrame:
    """Lấy DataFrame giá đóng cửa (index=date, columns=tickers) từ state_df.

    state_df có columns=MultiIndex(ticker, feature) — level 0 là ticker,
    level 1 là feature (close, rsi_14, macd, macd_signal, volatility_20d,
    sentiment). Dùng vị trí level (level=1) thay vì tên level để không phụ
    thuộc việc MultiIndex có được đặt tên ("ticker", "feature") hay không.
    """
    close = state_df.xs("close", axis=1, level=1)
    close = close.reindex(columns=list(tickers))
    return close


def _compute_basic_metrics(portfolio_value_history: np.ndarray) -> dict:
    """Tính nhanh một số metric cơ bản từ chuỗi giá trị danh mục.

    Đây là bản rút gọn (không phụ thuộc `quantstats`/`src.evaluation`) để
    các module baseline có thể chạy độc lập. Metric đầy đủ hơn (Sortino,
    CVaR...) sẽ do `src/evaluation/metrics.py::compute_all_metrics` đảm nhận
    khi ghép pipeline đánh giá đầy đủ.
    """
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


def run_b1_index(
    state_df: pd.DataFrame,
    tickers: Sequence[str],
    initial_cash: float = 1_000_000_000,
    index_close: Optional[pd.Series] = None,
) -> dict:
    """Chạy baseline B1 (VN30-Index buy & hold).

    Args:
        state_df: DataFrame trạng thái (index=date, columns=MultiIndex(ticker, feature)).
        tickers: danh sách mã trong rổ (dùng cho proxy equal-weight nếu
            không có `index_close`).
        initial_cash: vốn ban đầu (VND).
        index_close: (tuỳ chọn) chuỗi giá đóng cửa VN30 thật, cùng lưới ngày
            với `state_df.index`. Nếu truyền vào, dùng buy-and-hold thật
            trên chỉ số thay vì proxy equal-weight.

    Returns:
        dict gồm:
            - portfolio_value_history: np.ndarray độ dài = số ngày trong state_df
            - weights_history: np.ndarray (n_days, n_assets), mỗi hàng tổng = 1, không âm
            - dates: index ngày tương ứng
            - source: "vn30_index_real" hoặc "equal_weight_proxy"
            - metrics: dict các metric cơ bản
    """
    dates = state_df.index
    n_days = len(dates)

    if index_close is not None:
        index_close = pd.Series(index_close).reindex(dates)
        if index_close.isna().any():
            raise ValueError(
                "index_close phải có giá trị đầy đủ cho toàn bộ index ngày của state_df"
            )
        shares = initial_cash / float(index_close.iloc[0])
        portfolio_value_history = (shares * index_close).to_numpy()
        weights_history = np.ones((n_days, 1))
        source = "vn30_index_real"
    else:
        close = _extract_close_prices(state_df, tickers)
        n_assets = close.shape[1]
        daily_returns = close.pct_change().fillna(0.0)
        # Proxy "index": trung bình cộng return mỗi ngày (equal-weight,
        # ngầm định rebalance hàng ngày) trên toàn bộ tickers.
        equal_weight_daily_return = daily_returns.mean(axis=1).to_numpy()
        portfolio_value_history = initial_cash * np.cumprod(1.0 + equal_weight_daily_return)
        weights_history = np.tile(np.full(n_assets, 1.0 / n_assets), (n_days, 1))
        source = "equal_weight_proxy"

    metrics = _compute_basic_metrics(portfolio_value_history)

    return {
        "portfolio_value_history": portfolio_value_history,
        "weights_history": weights_history,
        "dates": dates,
        "source": source,
        "metrics": metrics,
    }
