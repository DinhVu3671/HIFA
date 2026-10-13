"""Test cho `src/envs/portfolio_env.py` và `src/envs/action_masking.py`.

Checklist bắt buộc theo đặc tả mục 3.3:
    - action luôn được normalize về tổng = 1 sau mask.
    - action-masking thực sự chặn đúng mã chạm biên độ +-7% (dữ liệu giả lập
      biết trước).
    - reward không NaN/Inf ở episode đầu tiên (window rolling chưa đủ dữ liệu).
    - action = [1, 0, ..., 0] (dồn hết vào 1 mã) -> turnover/cost tính đúng
      theo đúng công thức trong đặc tả.
    - random policy chạy hết 1 episode không crash, không NaN trong
      `portfolio_value_history`.

Dữ liệu dùng trong test là DataFrame giả lập (synthetic) với MultiIndex columns
(ticker, feature), đúng format mà `PortfolioEnvPhaseA` kỳ vọng — không phụ
thuộc vào `src/data/*` thật (đang được code song song bởi phần khác).
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd
import pytest

from src.envs.action_masking import apply_action_mask
from src.envs.portfolio_env import PortfolioEnvPhaseA, PortfolioEnvPhaseB

FEATURES = ("close", "rsi_14", "macd", "macd_signal", "volatility_20d")


def make_state_df(
    tickers: Sequence[str],
    n_days: int = 40,
    seed: int = 0,
    start_price: float = 50_000.0,
    daily_vol: float = 0.01,
) -> pd.DataFrame:
    """Tạo DataFrame giả lập MultiIndex (ticker, feature) cho test.

    Giá đi theo random-walk nhân với daily return nhỏ (~N(0, daily_vol)), các
    feature indicator khác (rsi_14, macd, macd_signal, volatility_20d) được
    sinh ngẫu nhiên nhưng hợp lý về khoảng giá trị — env không kiểm tra tính
    đúng đắn của các indicator này, chỉ cần đúng shape/kiểu dữ liệu.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days)

    columns = pd.MultiIndex.from_product([tickers, FEATURES], names=["ticker", "feature"])
    data = pd.DataFrame(index=dates, columns=columns, dtype=float)

    for tk in tickers:
        daily_returns = rng.normal(loc=0.0003, scale=daily_vol, size=n_days)
        # đảm bảo không có ngày nào tình cờ chạm +-7% (test riêng sẽ tự chèn ngày đó)
        daily_returns = np.clip(daily_returns, -0.05, 0.05)
        prices = start_price * np.cumprod(1 + daily_returns)
        data[(tk, "close")] = prices
        data[(tk, "rsi_14")] = rng.uniform(20, 80, size=n_days)
        data[(tk, "macd")] = rng.normal(0, 1, size=n_days)
        data[(tk, "macd_signal")] = rng.normal(0, 1, size=n_days)
        data[(tk, "volatility_20d")] = rng.uniform(0.01, 0.05, size=n_days)

    return data.sort_index(axis=1)


def make_env(tickers: Sequence[str] = ("AAA", "BBB", "CCC"), n_days: int = 40, seed: int = 0) -> PortfolioEnvPhaseA:
    state_df = make_state_df(tickers, n_days=n_days, seed=seed)
    return PortfolioEnvPhaseA(state_df=state_df, tickers=list(tickers), initial_cash=1_000_000_000)


# ----------------------------------------------------------------------
# 1. Action luôn normalize về tổng = 1 sau mask
# ----------------------------------------------------------------------
class TestActionNormalization:
    def test_random_unnormalized_action_sums_to_one(self):
        tickers = ["AAA", "BBB", "CCC"]
        state_df = make_state_df(tickers, n_days=10, seed=1)
        rng = np.random.default_rng(42)

        for t in range(1, 9):
            raw_action = rng.uniform(0, 5, size=len(tickers) + 1)  # chưa normalize, scale bất kỳ
            masked = apply_action_mask(raw_action, state_df, t, tickers)
            assert np.isfinite(masked).all()
            assert masked.sum() == pytest.approx(1.0, abs=1e-5)
            assert (masked >= -1e-8).all()

    def test_env_step_action_sums_to_one(self):
        env = make_env()
        env.reset()
        rng = np.random.default_rng(7)
        for _ in range(5):
            raw_action = rng.uniform(0, 3, size=env.n_assets + 1)
            obs, reward, terminated, truncated, info = env.step(raw_action)
            assert env.weights.sum() == pytest.approx(1.0, abs=1e-5)
            if terminated:
                break


# ----------------------------------------------------------------------
# 2. Action-masking thực sự chặn mã chạm biên độ +-7%
# ----------------------------------------------------------------------
class TestActionMasking:
    def _make_df_with_limit_hit(self, hit_ticker_idx: int = 0, direction: float = 1.0):
        """Trả về (state_df, tickers, t) sao cho tại ngày `t`, ticker ở
        `hit_ticker_idx` tăng/giảm đúng +8% (>= 7% biên độ) so với ngày t-1,
        các mã khác chỉ đổi nhẹ (~1%), không chạm biên."""
        tickers = ["AAA", "BBB", "CCC"]
        n_days = 20
        state_df = make_state_df(tickers, n_days=n_days, seed=3, daily_vol=0.005)

        t = 10
        hit_ticker = tickers[hit_ticker_idx]
        prev_close = float(state_df.iloc[t - 1][(hit_ticker, "close")])
        new_close = prev_close * (1 + direction * 0.08)  # 8% > 7% biên độ
        state_df.iloc[t, state_df.columns.get_loc((hit_ticker, "close"))] = new_close
        return state_df, tickers, t

    @pytest.mark.parametrize("direction", [1.0, -1.0])
    def test_mask_blocks_ticker_at_price_limit_zero_fallback(self, direction):
        state_df, tickers, t = self._make_df_with_limit_hit(hit_ticker_idx=0, direction=direction)

        # Đề xuất dồn gần hết vào mã bị khoá (index 0) -> phải bị chặn.
        action = np.array([0.9, 0.05, 0.05, 0.0])
        masked = apply_action_mask(action, state_df, t, tickers)  # current_weights=None -> fallback zero

        assert masked.sum() == pytest.approx(1.0, abs=1e-5)
        assert masked[0] == pytest.approx(0.0, abs=1e-6), "mã chạm biên phải bị đặt trọng số đề xuất về 0"
        # phần còn lại phân bổ theo đúng tỷ lệ 0.05:0.05:0.0 giữa BBB, CCC, cash
        assert masked[1] == pytest.approx(0.5, abs=1e-5)
        assert masked[2] == pytest.approx(0.5, abs=1e-5)

    @pytest.mark.parametrize("direction", [1.0, -1.0])
    def test_mask_preserves_current_weight_when_provided(self, direction):
        state_df, tickers, t = self._make_df_with_limit_hit(hit_ticker_idx=1, direction=direction)

        current_weights = np.array([0.2, 0.3, 0.1, 0.4])  # BBB (index 1) đang giữ 0.3
        action = np.array([0.1, 0.8, 0.1, 0.0])  # policy muốn dồn mạnh vào BBB -> phải bị chặn

        masked = apply_action_mask(action, state_df, t, tickers, current_weights=current_weights)

        assert masked.sum() == pytest.approx(1.0, abs=1e-5)
        # BBB bị khoá -> giữ nguyên đúng tỷ trọng hiện tại (0.3), KHÔNG được là 0.8
        assert masked[1] == pytest.approx(0.3, abs=1e-5)
        assert masked[1] != pytest.approx(0.8, abs=1e-2)

    def test_no_mask_when_within_price_limit(self):
        tickers = ["AAA", "BBB", "CCC"]
        state_df = make_state_df(tickers, n_days=20, seed=5, daily_vol=0.005)
        t = 10
        action = np.array([0.3, 0.3, 0.3, 0.1])
        masked = apply_action_mask(action, state_df, t, tickers)
        # không mã nào chạm biên -> chỉ normalize, tỷ lệ tương đối giữ nguyên
        expected = action / action.sum()
        np.testing.assert_allclose(masked, expected, atol=1e-5)

    def test_t_zero_never_masks(self):
        """t=0 không có ngày trước để so sánh -> không mã nào bị khoá."""
        tickers = ["AAA", "BBB"]
        state_df = make_state_df(tickers, n_days=10, seed=9)
        action = np.array([0.5, 0.5, 0.0])
        masked = apply_action_mask(action, state_df, 0, tickers)
        np.testing.assert_allclose(masked, action, atol=1e-5)

    def test_invalid_action_shape_raises(self):
        tickers = ["AAA", "BBB"]
        state_df = make_state_df(tickers, n_days=5, seed=1)
        with pytest.raises(ValueError):
            apply_action_mask(np.array([1.0, 0.0]), state_df, 1, tickers)  # thiếu 1 phần tử


# ----------------------------------------------------------------------
# 3. Reward không NaN/Inf ở episode đầu tiên
# ----------------------------------------------------------------------
class TestRewardStability:
    def test_reward_finite_at_first_steps(self):
        env = make_env(n_days=40)
        env.reset()
        rng = np.random.default_rng(11)
        for step_idx in range(5):
            action = rng.uniform(0, 1, size=env.n_assets + 1)
            obs, reward, terminated, truncated, info = env.step(action)
            assert np.isfinite(reward), f"reward không hữu hạn ở step {step_idx}: {reward}"
            assert not np.isnan(reward)
            if terminated:
                break

    def test_compute_reward_directly_with_short_history(self):
        env = make_env(n_days=40)
        env.reset()
        # portfolio_value_history chỉ có 1 giá trị (episode chưa chạy step nào)
        assert len(env.portfolio_value_history) == 1
        reward = env._compute_reward(net_return=0.01, turnover=0.5)
        assert np.isfinite(reward)

        # thêm 1 giá trị nữa -> len == 2 -> vẫn phải sharpe_local = 0 (không NaN)
        env.portfolio_value_history.append(env.initial_cash * 1.01)
        reward2 = env._compute_reward(net_return=0.01, turnover=0.5)
        assert np.isfinite(reward2)


# ----------------------------------------------------------------------
# 4. action = [1, 0, ..., 0] -> turnover/cost đúng công thức
# ----------------------------------------------------------------------
class TestTurnoverAndCost:
    def test_all_in_one_asset_turnover_and_cost(self):
        tickers = ["AAA", "BBB", "CCC"]
        state_df = make_state_df(tickers, n_days=20, seed=21, daily_vol=0.005)
        env = PortfolioEnvPhaseA(
            state_df=state_df, tickers=tickers, initial_cash=1_000_000_000, transaction_cost=0.002
        )
        env.reset()

        # weights ban đầu = [0, 0, 0, 1] (100% cash)
        old_weights = env.weights.copy()
        assert old_weights.tolist() == [0.0, 0.0, 0.0, 1.0]

        action = np.array([1.0, 0.0, 0.0, 0.0])  # dồn hết vào AAA
        row = state_df.iloc[env.t]
        obs, reward, terminated, truncated, info = env.step(action)

        # Công thức thủ công theo đặc tả:
        # turnover = sum(|action_masked - old_weights|)
        expected_turnover = np.abs(np.array([1.0, 0.0, 0.0, 0.0]) - old_weights).sum()
        assert expected_turnover == pytest.approx(2.0)  # |1-0|+|0-0|+|0-0|+|0-1| = 2
        assert info["turnover"] == pytest.approx(expected_turnover, abs=1e-5)

        expected_cost = expected_turnover * env.transaction_cost
        assert info["cost"] == pytest.approx(expected_cost, abs=1e-6)

        next_row = state_df.iloc[env.t]  # env.t đã += 1 bên trong step()
        aaa_return = (float(next_row[("AAA", "close")]) - float(row[("AAA", "close")])) / float(
            row[("AAA", "close")]
        )
        expected_gross_return = 1.0 * aaa_return  # 100% AAA, các mã khác weight=0
        expected_net_return = expected_gross_return - expected_cost

        assert info["net_return"] == pytest.approx(expected_net_return, abs=1e-6)

        expected_new_value = 1_000_000_000 * (1 + expected_net_return)
        assert info["portfolio_value"] == pytest.approx(expected_new_value, rel=1e-6)
        assert env.portfolio_value_history[-1] == pytest.approx(expected_new_value, rel=1e-6)


# ----------------------------------------------------------------------
# 5. Random policy chạy hết 1 episode không crash, không NaN
# ----------------------------------------------------------------------
class TestFullEpisodeRandomPolicy:
    def test_random_policy_completes_episode_no_nan(self):
        tickers = ["AAA", "BBB", "CCC"]
        env = make_env(tickers=tickers, n_days=60, seed=99)
        obs, info = env.reset()
        assert obs.shape == env.observation_space.shape
        assert np.isfinite(obs).all()

        rng = np.random.default_rng(123)
        n_steps = 0
        terminated = False
        truncated = False
        while not (terminated or truncated):
            raw_action = rng.random(env.n_assets + 1)
            raw_action = raw_action / raw_action.sum()  # normalize trước khi gửi vào env
            obs, reward, terminated, truncated, info = env.step(raw_action)
            n_steps += 1

            assert np.isfinite(reward)
            assert not np.isnan(info["portfolio_value"])
            assert np.isfinite(info["portfolio_value"])
            assert n_steps <= len(env.state_df)  # phòng vòng lặp vô hạn nếu logic done sai

        assert n_steps == len(env.state_df) - 1
        assert len(env.portfolio_value_history) == n_steps + 1
        assert not np.any(np.isnan(env.portfolio_value_history))
        assert np.all(np.isfinite(env.portfolio_value_history))
        # sau khi terminated, obs cuối cùng phải là vector zero đúng shape (spec)
        assert obs.shape == env.observation_space.shape
        np.testing.assert_allclose(obs, np.zeros_like(obs))


# ----------------------------------------------------------------------
# Extra: PortfolioEnvPhaseB tái sử dụng nguyên vẹn logic Phase A
# ----------------------------------------------------------------------
class TestPhaseBAlias:
    def test_phase_b_env_behaves_like_phase_a(self):
        tickers = ["AAA", "BBB", "CCC", "DDD", "EEE"]
        state_df = make_state_df(tickers, n_days=30, seed=55)
        env = PortfolioEnvPhaseB(state_df=state_df, tickers=tickers)
        obs, info = env.reset()
        assert isinstance(env, PortfolioEnvPhaseA)
        assert env.action_space.shape == (len(tickers) + 1,)

        action = np.ones(len(tickers) + 1) / (len(tickers) + 1)
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.isfinite(reward)
        assert not terminated
