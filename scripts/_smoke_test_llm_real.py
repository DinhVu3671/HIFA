"""scripts/_smoke_test_llm_real.py

Script TAM THOI (khong phai production) de verify PortfolioPolicyLLM +
SimplePPOTrainer voi model THAT Qwen/Qwen2.5-1.5B-Instruct tren CPU (khong
GPU). KHONG co gang train hoi tu - chi verify (a) model tai duoc, (b) forward
pass dung shape/khong NaN, (c) 1 vong lap PPO cuc nho (n_steps nho, 1 update,
n_epochs=1) chay het khong loi runtime, do thoi gian that.

Chay: .venv\\Scripts\\python.exe scripts\\_smoke_test_llm_real.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.agents.super_agent_head import PortfolioPolicyLLM  # noqa: E402
from src.data.build_dataset import load_state_dataframe  # noqa: E402
from src.data.fetch_ohlcv_us import TICKERS_PHASE_0  # noqa: E402
from src.envs.portfolio_env import PortfolioEnvUS  # noqa: E402
from src.training.ppo_trainer import SimplePPOTrainer  # noqa: E402

TICKERS = TICKERS_PHASE_0  # 8 mã Mỹ Giai đoạn 0
N_STEPS = 4
N_SLICE_DAYS = 40  # cat nho de env chay nhanh, chi can > n_steps + 1


def log(msg: str) -> None:
    print(f"[smoke-llm-real] {msg}", flush=True)


def main() -> None:
    total_t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # 1) Load policy voi model THAT
    # ------------------------------------------------------------------
    log("Dang khoi tao PortfolioPolicyLLM voi Qwen/Qwen2.5-1.5B-Instruct ...")
    t0 = time.perf_counter()
    policy = PortfolioPolicyLLM(
        base_model_name="Qwen/Qwen2.5-1.5B-Instruct",
        n_assets=len(TICKERS),
        torch_dtype=torch.float32,  # xem ghi chu trong bao cao: bf16 matmul rat cham tren CPU
    )
    t_init = time.perf_counter() - t0
    log(f"Khoi tao policy xong trong {t_init:.1f}s")

    n_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in policy.parameters())
    log(
        f"Tham so trainable: {n_trainable:,} / {n_total:,} tong "
        f"({100.0 * n_trainable / max(n_total, 1):.4f}%)"
    )

    # ------------------------------------------------------------------
    # 2) Forward pass don le - kiem tra shape / NaN
    # ------------------------------------------------------------------
    log("Kiem tra forward() don le...")
    t0 = time.perf_counter()
    dummy_prompt = policy.state_to_prompt(
        {
            tk: {
                "close": 50.0 + i,
                "rsi_14": 45.0,
                "macd": 0.1,
                "macd_signal": 0.05,
                "volatility_20d": 0.02,
            }
            for i, tk in enumerate(TICKERS)
        }
    )
    with torch.no_grad():
        weights, value = policy([dummy_prompt])
    t_fwd = time.perf_counter() - t0
    log(f"forward() 1 prompt: {t_fwd:.2f}s | weights.shape={tuple(weights.shape)} value.shape={tuple(value.shape)}")
    log(f"weights={weights.detach().cpu().numpy().tolist()} sum={float(weights.sum()):.6f}")
    log(f"value={value.detach().cpu().numpy().tolist()}")
    assert weights.shape == (1, len(TICKERS) + 1), f"weights shape sai: {weights.shape}"
    assert value.shape == (1, 1), f"value shape sai: {value.shape}"
    assert torch.isfinite(weights).all(), "weights co NaN/Inf!"
    assert torch.isfinite(value).all(), "value co NaN/Inf!"
    assert abs(float(weights.sum()) - 1.0) < 1e-4, "weights khong sum ve 1!"
    log("Forward pass OK: dung shape, khong NaN/Inf, weights sum=1.")

    # ------------------------------------------------------------------
    # 3) Env voi du lieu that (cat nho)
    # ------------------------------------------------------------------
    log("Load state_df that tu data/processed/phase_0.parquet...")
    state_df = load_state_dataframe(REPO_ROOT / "data" / "processed" / "phase_0.parquet")
    state_slice = state_df.iloc[:N_SLICE_DAYS]
    log(f"state_slice shape={state_slice.shape} ({state_slice.index[0].date()} -> {state_slice.index[-1].date()})")

    env = PortfolioEnvUS(state_df=state_slice, tickers=TICKERS)

    # ------------------------------------------------------------------
    # 4) SimplePPOTrainer: 1 rollout n_steps nho + 1 update n_epochs=1
    # ------------------------------------------------------------------
    trainer = SimplePPOTrainer(
        policy=policy,
        env=env,
        n_steps=N_STEPS,
        gradient_checkpointing=True,
        rollout_warn_seconds=1e9,  # tat canh bao trong script nay, ta tu log thoi gian
    )

    log(f"Bat dau collect_rollout(n_steps={N_STEPS})...")
    t0 = time.perf_counter()
    buffer, last_value = trainer.collect_rollout(n_steps=N_STEPS)
    t_rollout = time.perf_counter() - t0
    log(f"collect_rollout xong trong {t_rollout:.2f}s ({t_rollout / N_STEPS:.2f}s/step) | last_value={last_value:.6f}")

    data = buffer.get()
    rewards = data["rewards"]
    log(f"rewards={rewards.tolist()} mean={float(rewards.mean()):.6f}")
    assert torch.isfinite(rewards).all(), "rewards co NaN/Inf!"
    assert torch.isfinite(data["values"]).all(), "values co NaN/Inf!"
    assert torch.isfinite(data["log_probs"]).all(), "log_probs co NaN/Inf!"

    advantages, returns = trainer.compute_gae(data["rewards"], data["values"], data["dones"], last_value)
    data["advantages"] = advantages
    data["returns"] = returns
    assert torch.isfinite(advantages).all(), "advantages co NaN/Inf!"
    assert torch.isfinite(returns).all(), "returns co NaN/Inf!"
    log(f"advantages={advantages.tolist()}")
    log(f"returns={returns.tolist()}")

    log("Bat dau update(n_epochs=1)...")
    t0 = time.perf_counter()
    stats = trainer.update(data, n_epochs=1, batch_size=N_STEPS)
    t_update = time.perf_counter() - t0
    log(f"update() xong trong {t_update:.2f}s")
    log(f"stats={stats}")
    for k, v in stats.items():
        assert v == v, f"stats['{k}'] la NaN!"  # NaN != NaN

    total_elapsed = time.perf_counter() - total_t0
    log("=" * 70)
    log("TOM TAT THOI GIAN THAT (CPU 12-core, khong GPU):")
    log(f"  - Khoi tao policy (tokenizer+model+LoRA, KHONG tinh download lan dau): {t_init:.1f}s")
    log(f"  - forward() 1 prompt don le: {t_fwd:.2f}s")
    log(f"  - collect_rollout(n_steps={N_STEPS}): {t_rollout:.2f}s => {t_rollout / N_STEPS:.2f}s/step")
    log(f"  - update(n_epochs=1, batch_size={N_STEPS}): {t_update:.2f}s")
    log(f"  - TONG script: {total_elapsed:.1f}s")
    per_step = t_rollout / N_STEPS
    for n in (256, 512):
        est_min = per_step * n / 60.0
        log(f"  => Uoc luong collect_rollout(n_steps={n}) ~ {est_min:.1f} phut (ngoai suy tuyen tinh tu step/giay do duoc)")
    log("KET LUAN: pipeline chay het KHONG loi runtime voi model that.")


if __name__ == "__main__":
    main()
