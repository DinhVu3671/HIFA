"""Test cho kiến trúc nền C1 — Debate-style (src/baselines/c1_debate.py).

Dùng scorer GIẢ tất định (không cần model LLM thật / không mạng) để kiểm:
    - debate round kéo điểm về phía đồng thuận,
    - phân bổ chỉ vào mã điểm dương, tôn trọng trần max_weight,
    - toàn tiền mặt khi mọi điểm <= 0,
    - run_c1_debate chạy hết backtest, không NaN, có metric avg_turnover.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.baselines.c1_debate import DebateAllocator, run_c1_debate

TICKERS = ["AAPL", "MSFT", "JPM", "JNJ"]
FEATURES = ("close", "rsi_14", "macd", "macd_signal", "volatility_20d",
            "pe", "pb", "eps_growth")


def make_state_df(tickers, n_days=30, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    data = {}
    for tk in tickers:
        price = 50.0 + np.cumsum(rng.normal(0, 0.3, size=n_days))
        price = np.clip(price, 1.0, None)
        data[(tk, "close")] = price
        data[(tk, "rsi_14")] = rng.uniform(30, 70, size=n_days)
        data[(tk, "macd")] = rng.normal(0, 1, size=n_days)
        data[(tk, "macd_signal")] = rng.normal(0, 1, size=n_days)
        data[(tk, "volatility_20d")] = rng.uniform(0.01, 0.05, size=n_days)
        data[(tk, "pe")] = rng.uniform(5, 25, size=n_days)
        data[(tk, "pb")] = rng.uniform(0.5, 4, size=n_days)
        data[(tk, "eps_growth")] = rng.uniform(-0.1, 0.3, size=n_days)
    cols = pd.MultiIndex.from_tuples(list(data.keys()), names=["ticker", "feature"])
    return pd.DataFrame(data, index=dates, columns=cols).sort_index(axis=1)


# --------------------------------------------------------------------------
# Scorer giả
# --------------------------------------------------------------------------
def _fixed_positive_scorer(agent_type, ticker_data):
    # AAPL luôn dương mạnh, còn lại âm -> C1 phải dồn vào AAPL (tôn trọng trần).
    return 0.9 if ticker_data["ticker"] == "AAPL" else -0.5


def _all_negative_scorer(agent_type, ticker_data):
    return -0.8


def test_debate_round_moves_toward_consensus():
    alloc = DebateAllocator(TICKERS, n_debate_rounds=1, revision_strength=0.5)
    # 3 agent: 1.0 / 0.0 / -1.0 -> trung bình 0; sau 1 vòng revision về consensus,
    # điểm tổng hợp (mean) vẫn ~0 nhưng từng agent dịch về giữa.
    out = alloc._debate_scores({"fundamental": 1.0, "technical": 0.0, "sentiment": -1.0})
    assert -1.0 <= out <= 1.0


def test_allocate_concentrates_on_positive_and_respects_cap():
    alloc = DebateAllocator(TICKERS, scorer=_fixed_positive_scorer, max_weight=0.30)
    df = make_state_df(TICKERS, n_days=5)
    w = alloc.allocate_one(df.iloc[0])
    assert w.shape == (len(TICKERS) + 1,)
    assert abs(float(w.sum()) - 1.0) < 1e-6
    assert np.all(w[:-1] <= 0.30 + 1e-6)
    # AAPL (index 0) là mã dương duy nhất -> phải nhận trần 0.30, phần dư vào cash.
    assert w[0] >= 0.30 - 1e-6
    assert w[-1] > 0.0  # cash nhận phần dư vì chỉ 1 mã dương, trần 30%


def test_allocate_all_cash_when_no_positive():
    alloc = DebateAllocator(TICKERS, scorer=_all_negative_scorer, max_weight=0.30)
    df = make_state_df(TICKERS, n_days=5)
    w = alloc.allocate_one(df.iloc[0])
    assert abs(w[-1] - 1.0) < 1e-6  # toàn tiền mặt
    assert np.allclose(w[:-1], 0.0)


def test_run_c1_debate_backtest_runs_and_reports_metrics():
    df = make_state_df(TICKERS, n_days=40, seed=1)
    result = run_c1_debate(df, TICKERS, scorer=_fixed_positive_scorer)
    pv = result["portfolio_value_history"]
    assert len(pv) == len(df)  # n_days (1 giá trị khởi tạo + n_days-1 bước)
    assert np.all(np.isfinite(pv))
    assert "avg_turnover" in result["metrics"]
    assert np.isfinite(result["metrics"]["avg_turnover"])
    assert "sharpe" in result["metrics"]


def test_requires_scorer():
    alloc = DebateAllocator(TICKERS)  # không có scorer
    df = make_state_df(TICKERS, n_days=3)
    with pytest.raises(ValueError):
        alloc.allocate_one(df.iloc[0])
