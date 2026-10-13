"""Test cho các baseline B1-B4 (src/baselines/).

Dùng dữ liệu GIẢ LẬP tự tạo trong file này (không phụ thuộc
`src/data/build_dataset.py` thật) để test độc lập được ngay cả khi pipeline
dữ liệu thật chưa sẵn sàng / đang được chỉnh sửa song song. Dữ liệu giả lập
tuân đúng format do `build_state_dataframe` tạo ra: index=date,
columns=MultiIndex(ticker, feature) với feature gồm
["close", "rsi_14", "macd", "macd_signal", "volatility_20d", "sentiment"].

Test B4 (PPO) cần `src.envs.portfolio_env.PortfolioEnvPhaseA` — module này
được một phần khác của dự án viết song song. Nếu chưa import được, test B4
sẽ tự SKIP (không FAIL) kèm lý do rõ ràng, thay vì làm cả suite đỏ.
"""
from __future__ import annotations

import os

import pytest

pd = pytest.importorskip("pandas")
import numpy as np

from src.baselines.b1_index import run_b1_index
from src.baselines.b2_buy_hold import run_b2_buy_hold
from src.baselines.b3_mean_variance import mean_variance_weights, run_b3_mean_variance


FEATURES = ["close", "rsi_14", "macd", "macd_signal", "volatility_20d", "sentiment"]


def _make_fake_state_df(n_days=80, tickers=("AAA", "BBB", "CCC"), seed=42):
    """Tạo state_df giả lập đúng format MultiIndex(ticker, feature)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    columns = pd.MultiIndex.from_product([tickers, FEATURES], names=["ticker", "feature"])

    data = {}
    for ticker in tickers:
        base_price = rng.uniform(20, 100)
        daily_ret = rng.normal(0.0005, 0.015, size=n_days)
        close = base_price * np.cumprod(1 + daily_ret)
        data[(ticker, "close")] = close
        data[(ticker, "rsi_14")] = rng.uniform(30, 70, size=n_days)
        data[(ticker, "macd")] = rng.normal(0, 1, size=n_days)
        data[(ticker, "macd_signal")] = rng.normal(0, 1, size=n_days)
        data[(ticker, "volatility_20d")] = rng.uniform(0.1, 0.4, size=n_days)
        data[(ticker, "sentiment")] = rng.uniform(-1, 1, size=n_days)

    state_df = pd.DataFrame(data, index=dates)
    state_df.columns = columns
    return state_df, list(tickers)


def _assert_valid_weights(weights_history: np.ndarray, n_days: int, n_assets: int):
    w = np.asarray(weights_history, dtype=float)
    assert w.shape == (n_days, n_assets)
    assert not np.isnan(w).any(), "weights_history chứa NaN"
    assert (w >= -1e-9).all(), "weights_history có giá trị âm"
    row_sums = w.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6), f"weights mỗi ngày phải tổng = 1, got {row_sums}"


def _assert_valid_portfolio_value(pv_history: np.ndarray, n_days: int, initial_cash: float):
    pv = np.asarray(pv_history, dtype=float)
    assert len(pv) == n_days
    assert not np.isnan(pv).any(), "portfolio_value_history chứa NaN"
    assert pv[0] == pytest.approx(initial_cash, rel=1e-6)


class TestB1Index:
    def test_proxy_equal_weight_valid(self):
        n_days = 80
        state_df, tickers = _make_fake_state_df(n_days=n_days)
        initial_cash = 1_000_000_000
        result = run_b1_index(state_df, tickers, initial_cash=initial_cash)

        assert result["source"] == "equal_weight_proxy"
        _assert_valid_portfolio_value(result["portfolio_value_history"], n_days, initial_cash)
        _assert_valid_weights(result["weights_history"], n_days, len(tickers))
        assert "metrics" in result
        assert not any(np.isnan(v) for v in result["metrics"].values())

    def test_real_index_close_path(self):
        n_days = 40
        state_df, tickers = _make_fake_state_df(n_days=n_days)
        initial_cash = 1_000_000_000
        rng = np.random.default_rng(0)
        fake_vn30 = pd.Series(
            1000 * np.cumprod(1 + rng.normal(0.0003, 0.01, size=n_days)),
            index=state_df.index,
        )
        result = run_b1_index(state_df, tickers, initial_cash=initial_cash, index_close=fake_vn30)

        assert result["source"] == "vn30_index_real"
        _assert_valid_portfolio_value(result["portfolio_value_history"], n_days, initial_cash)
        # Buy & hold 1 "tài sản" duy nhất (chỉ số) -> weights luôn = 1.
        w = np.asarray(result["weights_history"])
        assert np.allclose(w, 1.0)


class TestB2BuyHold:
    def test_equal_weight_buy_hold_valid(self):
        n_days = 80
        state_df, tickers = _make_fake_state_df(n_days=n_days)
        initial_cash = 1_000_000_000
        result = run_b2_buy_hold(state_df, tickers, initial_cash=initial_cash)

        _assert_valid_portfolio_value(result["portfolio_value_history"], n_days, initial_cash)
        _assert_valid_weights(result["weights_history"], n_days, len(tickers))

        w = np.asarray(result["weights_history"])
        # Ngày đầu tiên phải là equal-weight (vừa mua đều xong).
        assert np.allclose(w[0], np.full(len(tickers), 1.0 / len(tickers)), atol=1e-6)
        # Không rebalance -> tỷ trọng phải TRÔI theo giá theo thời gian
        # (khác B1, vốn rebalance ngầm định mỗi ngày).
        assert not np.allclose(w[0], w[-1], atol=1e-3)

        assert "shares" in result
        assert set(result["shares"].keys()) == set(tickers)


class TestB3MeanVariance:
    def test_mean_variance_weights_basic(self):
        rng = np.random.default_rng(1)
        returns_df = pd.DataFrame(
            rng.normal(0.001, 0.01, size=(60, 4)), columns=["A", "B", "C", "D"]
        )
        w = mean_variance_weights(returns_df)

        assert w.shape == (4,)
        assert not np.isnan(w).any()
        assert (w >= -1e-8).all()
        assert (w <= 1 + 1e-8).all()
        assert w.sum() == pytest.approx(1.0, abs=1e-4)

    def test_mean_variance_weights_near_singular_covariance(self):
        """2 mã có return gần như giống hệt nhau -> cov gần singular.

        Đây chính là trường hợp đặc tả cảnh báo có thể làm scipy.optimize
        không hội tụ nếu không regularize covariance. Test xác nhận
        `mean_variance_weights` (đã có `cov + 1e-6*I`) vẫn trả về nghiệm hợp
        lệ, không NaN, không lỗi.
        """
        rng = np.random.default_rng(2)
        n = 60
        base = rng.normal(0.001, 0.01, size=n)
        col_a = base
        col_b = base + rng.normal(0, 1e-9, size=n)  # nhiễu cực nhỏ -> gần như trùng col_a
        col_c = rng.normal(0.0005, 0.008, size=n)
        returns_df = pd.DataFrame({"A": col_a, "B": col_b, "C": col_c})

        # Sanity check: covariance thực sự gần singular (eigenvalue nhỏ nhất ~ 0).
        cov = returns_df.cov().values
        eigvals = np.linalg.eigvalsh(cov)
        assert eigvals.min() < 1e-6

        w = mean_variance_weights(returns_df)

        assert w.shape == (3,)
        assert not np.isnan(w).any(), "mean_variance_weights trả NaN khi cov gần singular"
        assert (w >= -1e-8).all()
        assert (w <= 1 + 1e-8).all()
        assert w.sum() == pytest.approx(1.0, abs=1e-4)

    def test_run_b3_walk_forward_valid(self):
        n_days = 90
        state_df, tickers = _make_fake_state_df(n_days=n_days)
        initial_cash = 1_000_000_000
        result = run_b3_mean_variance(
            state_df,
            tickers,
            lookback=20,
            rebalance_freq=10,
            initial_cash=initial_cash,
        )

        _assert_valid_portfolio_value(result["portfolio_value_history"], n_days, initial_cash)
        _assert_valid_weights(result["weights_history"], n_days, len(tickers))
        assert len(result["rebalance_dates"]) >= 1
        assert "metrics" in result


class TestB4PPONoLLM:
    def test_train_b4_smoke(self, tmp_path):
        """Smoke test: train_b4 chạy được với total_timesteps rất nhỏ, không crash.

        Mục tiêu: xác nhận không có lỗi shape/API khi ghép PortfolioEnvPhaseA
        với SB3 PPO — KHÔNG kiểm tra hội tụ/hiệu năng.

        Nếu `src.envs.portfolio_env.PortfolioEnvPhaseA` hoặc `stable_baselines3`
        chưa sẵn sàng (đang được viết song song / chưa cài xong), test SKIP
        với lý do rõ ràng thay vì FAIL cả suite.
        """
        pytest.importorskip("stable_baselines3")
        pytest.importorskip("gymnasium")
        try:
            from src.envs.portfolio_env import PortfolioEnvPhaseA  # noqa: F401
        except Exception as exc:  # module có thể chưa tồn tại / đang được ghi đè
            pytest.skip(f"src.envs.portfolio_env chưa sẵn sàng: {exc!r}")

        from src.baselines.b4_ppo_no_llm import train_b4

        n_days = 60
        state_df, tickers = _make_fake_state_df(n_days=n_days)
        save_path = str(tmp_path / "b4_smoke_test")

        model = train_b4(
            state_df,
            tickers,
            total_timesteps=128,
            save_path=save_path,
            seed=0,
            tensorboard_log=None,  # tránh phụ thuộc gói tensorboard khi smoke-test
            verbose=0,
            n_steps=32,
            batch_size=16,
        )

        assert model is not None
        assert os.path.exists(save_path + ".zip"), "train_b4 phải lưu checkpoint .zip"
