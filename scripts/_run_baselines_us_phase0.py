"""Driver — chạy 4 baseline tài chính (B1 Index / B2 Buy&Hold / B3 Mean-Variance
/ B4 PPO-no-LLM) trên THỊ TRƯỜNG MỸ, Giai đoạn 0 (De-Cuong §6.4, §8).

Đặc điểm theo đề cương (khác bản VN):
  - Dữ liệu: 8 mã Mỹ (phase_0.parquet), môi trường `PortfolioEnvUS`
    (price_limit=None, trần 30%/mã, cash 1tr USD, phí 0.05%).
  - B1 (Index) dùng CHỈ SỐ THẬT ^DJI (benchmark_dow.parquet) qua `index_close`,
    không dùng proxy equal-weight.
  - Walk-forward split 1 cửa sổ: train 2019-2022 / val 2023H1 / test 2023H2.
    B1/B2/B3 chạy trực tiếp trên test (out-of-sample). B4 train lại từ đầu chỉ
    trên train_df rồi backtest trên test (tránh look-ahead — bài học TODO cũ).

Kết quả xuất ra reports/phase_0/.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.baselines.b1_index import run_b1_index  # noqa: E402
from src.baselines.b2_buy_hold import run_b2_buy_hold  # noqa: E402
from src.baselines.b3_mean_variance import run_b3_mean_variance  # noqa: E402
from src.baselines.b4_ppo_no_llm import train_b4  # noqa: E402
from src.data.build_dataset import load_state_dataframe  # noqa: E402
from src.data.fetch_ohlcv_us import TICKERS_PHASE_0  # noqa: E402
from src.data.walk_forward_split import single_window_phase_a  # noqa: E402
from src.envs.portfolio_env import PortfolioEnvUS  # noqa: E402
from src.evaluation.backtest import run_backtest  # noqa: E402
from src.evaluation.metrics import compute_all_metrics  # noqa: E402
from src.evaluation.report import build_comparison_table, export_report  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SEED = 0
INITIAL_CASH = 1_000_000.0  # USD
TRANSACTION_COST = 0.0005
MAX_WEIGHT = 0.30


def _slice(state_df, start, end):
    return state_df.loc[(state_df.index >= start) & (state_df.index <= end)]


def _load_benchmark_close(dates) -> pd.Series:
    """Đọc giá đóng cửa ^DJI, reindex về đúng lưới ngày của state_df (ffill)."""
    path = REPO_ROOT / "data" / "raw" / "ohlcv" / "benchmark_dow.parquet"
    bench = pd.read_parquet(path)
    bench["date"] = pd.to_datetime(bench["date"])
    s = bench.set_index("date")["close"].sort_index()
    # Chỉ số và cổ phiếu Mỹ cùng lịch giao dịch -> reindex khít; ffill phòng lệch nhẹ.
    return s.reindex(dates).ffill().bfill()


def main() -> None:
    state_df = load_state_dataframe(REPO_ROOT / "data" / "processed" / "phase_0.parquet")
    print(f"State df (US): {state_df.shape}, range {state_df.index.min().date()} -> {state_df.index.max().date()}")

    (train_range, val_range, test_range) = single_window_phase_a(dates=state_df.index)
    train_df = _slice(state_df, *train_range)
    val_df = _slice(state_df, *val_range)
    test_df = _slice(state_df, *test_range)

    for name, (rng, df) in {
        "train": (train_range, train_df),
        "val": (val_range, val_df),
        "test": (test_range, test_df),
    }.items():
        print(
            f"  {name}: {rng[0].date()} -> {rng[1].date()}, "
            f"thuc te {len(df)} phien "
            f"({df.index.min().date() if len(df) else None} -> {df.index.max().date() if len(df) else None})"
        )
        if len(df) == 0:
            raise RuntimeError(f"Split '{name}' rong.")

    results = {}

    print("\n=== B1 (Dow Jones Index THAT ^DJI, buy&hold) tren TEST ===")
    bench_close = _load_benchmark_close(test_df.index)
    b1 = run_b1_index(test_df, TICKERS_PHASE_0, initial_cash=INITIAL_CASH, index_close=bench_close)
    m1 = compute_all_metrics(list(b1["portfolio_value_history"]))
    results["B1_DowJonesIndex"] = [m1]
    print("B1 source:", b1["source"], {k: round(v, 4) for k, v in m1.items()})

    print("\n=== B2 (Buy & Hold 1/N) tren TEST ===")
    b2 = run_b2_buy_hold(test_df, TICKERS_PHASE_0, initial_cash=INITIAL_CASH)
    m2 = compute_all_metrics(list(b2["portfolio_value_history"]))
    results["B2_BuyHold"] = [m2]
    print("B2 done:", {k: round(v, 4) for k, v in m2.items()})

    print("\n=== B3 (Mean-Variance) tren TEST ===")
    b3 = run_b3_mean_variance(test_df, TICKERS_PHASE_0, initial_cash=INITIAL_CASH)
    m3 = compute_all_metrics(list(b3["portfolio_value_history"]))
    results["B3_MeanVariance"] = [m3]
    print("B3 done:", {k: round(v, 4) for k, v in m3.items()})

    print(
        f"\n=== Train B4 (PPO MlpPolicy) tu dau tren TRAIN ({len(train_df)} phien, "
        f"seed={SEED}, 200000 timesteps, US env) ==="
    )
    b4_ckpt = REPO_ROOT / "checkpoints" / "b4_ppo_no_llm_phase_0"
    model = train_b4(
        train_df,
        TICKERS_PHASE_0,
        total_timesteps=200_000,
        save_path=str(b4_ckpt),
        seed=SEED,
        tensorboard_log=None,
        verbose=0,
        env_cls=PortfolioEnvUS,
        env_kwargs={
            "initial_cash": INITIAL_CASH,
            "transaction_cost": TRANSACTION_COST,
            "max_weight": MAX_WEIGHT,
        },
    )
    print(f"B4 train xong: {b4_ckpt}.zip")

    print("\n=== Backtest B4 tren TEST (out-of-sample) ===")
    test_env = PortfolioEnvUS(
        test_df,
        TICKERS_PHASE_0,
        initial_cash=INITIAL_CASH,
        transaction_cost=TRANSACTION_COST,
        max_weight=MAX_WEIGHT,
    )
    b4_result = run_backtest(model, test_env, seed=SEED, deterministic=True)
    results["B4_PPO_NoLLM"] = [b4_result["metrics"]]
    print("B4 done:", {k: round(v, 4) for k, v in b4_result["metrics"].items()})

    table = build_comparison_table(results)
    print("\n=== Bang so sanh (US, TEST 2023H2, 1 seed) ===")
    print(table.to_string())

    out_dir = REPO_ROOT / "reports" / "phase_0"
    export_report(table, {}, out_dir, phase="0")
    print(f"\nDa xuat bao cao vao {out_dir}")


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    main()
