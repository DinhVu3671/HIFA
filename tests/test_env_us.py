"""Test cho các tính năng thị trường MỸ (De-Cuong.md):
    - `apply_position_cap`  : trần 30%/mã + phân bổ lại phần dư.
    - `apply_action_mask(price_limit=None)` : không khoá mã (Mỹ không có biên độ).
    - `PortfolioEnvUS`      : default đúng đề cương (price_limit=None, cap 0.30,
      cash 1tr USD), chạy hết episode, không mã nào vượt trần.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.envs.action_masking import apply_action_mask, apply_position_cap
from src.envs.portfolio_env import PortfolioEnvUS

# Tái dùng helper tạo state_df giả lập của test_env.py.
from tests.test_env import make_state_df

TICKERS = ["AAPL", "MSFT", "JPM", "JNJ", "XOM", "PG", "KO", "NVDA"]


# --------------------------------------------------------------------------
# apply_position_cap
# --------------------------------------------------------------------------
def test_position_cap_clips_and_renormalizes():
    w = np.array([0.9, 0.05, 0.03, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02])
    capped = apply_position_cap(w, max_weight=0.30)
    assert np.all(capped[:-1] <= 0.30 + 1e-6)
    assert abs(float(capped.sum()) - 1.0) < 1e-6


def test_position_cap_no_change_when_under_cap():
    w = np.array([0.2, 0.2, 0.2, 0.1, 0.1, 0.0, 0.0, 0.0, 0.2])
    capped = apply_position_cap(w, max_weight=0.30)
    assert np.allclose(capped, w / w.sum(), atol=1e-6)


def test_position_cap_excess_flows_to_cash_when_all_assets_capped():
    # 2 mã, trần 0.3 -> tối đa 0.6 vào cổ phiếu, phần còn lại buộc vào cash.
    w = np.array([0.8, 0.8, 0.0])  # 2 assets + cash
    capped = apply_position_cap(w, max_weight=0.30)
    assert capped[0] <= 0.30 + 1e-6
    assert capped[1] <= 0.30 + 1e-6
    assert capped[-1] >= 0.40 - 1e-6  # cash nhận phần dư
    assert abs(float(capped.sum()) - 1.0) < 1e-6


# --------------------------------------------------------------------------
# apply_action_mask với price_limit=None (thị trường Mỹ)
# --------------------------------------------------------------------------
def test_action_mask_none_price_limit_locks_nothing():
    df = make_state_df(TICKERS, n_days=10, seed=1)
    action = np.full(len(TICKERS) + 1, 1.0 / (len(TICKERS) + 1))
    # Dù t>0, price_limit=None => không mã nào bị khoá; chỉ normalize.
    out = apply_action_mask(
        action=action,
        state_df=df,
        t=5,
        tickers=TICKERS,
        price_limit=None,
        current_weights=None,
    )
    assert abs(float(out.sum()) - 1.0) < 1e-6
    # Tất cả bằng nhau (không mã nào bị zero do mask).
    assert np.allclose(out, out[0], atol=1e-6)


# --------------------------------------------------------------------------
# PortfolioEnvUS
# --------------------------------------------------------------------------
def test_us_env_defaults():
    df = make_state_df(TICKERS, n_days=30, seed=2)
    env = PortfolioEnvUS(df, TICKERS)
    assert env.price_limit is None
    assert abs(env.max_weight - 0.30) < 1e-9
    assert abs(env.initial_cash - 1_000_000.0) < 1e-6
    assert abs(env.transaction_cost - 0.0005) < 1e-9


def test_us_env_respects_cap_over_episode():
    df = make_state_df(TICKERS, n_days=60, seed=3)
    env = PortfolioEnvUS(df, TICKERS)
    rng = np.random.default_rng(0)
    obs, _ = env.reset()
    max_seen = 0.0
    done = False
    while not done:
        a = rng.random(len(TICKERS) + 1)
        obs, r, term, trunc, info = env.step(a)
        assert np.isfinite(r)
        assert np.isfinite(info["portfolio_value"])
        max_seen = max(max_seen, float(env.weights[:-1].max()))
        done = term or trunc
    assert max_seen <= 0.30 + 1e-6
