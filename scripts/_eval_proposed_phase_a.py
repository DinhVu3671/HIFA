"""scripts/_eval_proposed_phase_a.py

Backtest ĐỀ XUẤT (PortfolioPolicyLLM đã train, Giai đoạn A) trên TEST split
(walk-forward: test_range trong config) cho từng seed, tính metrics qua
src.evaluation.metrics.compute_all_metrics, và lưu kết quả per-seed thành
reports/phase_a/raw/Proposed_LLM_PPO.json (đúng format mà
src.evaluation.report._load_strategy_results kỳ vọng: list các dict metrics).

Chạy trên CPU được (inference thuần, torch.no_grad) — chậm vì mỗi bước forward
qua Qwen2.5-1.5B, nhưng test split chỉ ~120-130 ngày nên khả thi.

Dùng:
    PYTHONPATH=. python scripts/_eval_proposed_phase_a.py \
        --config configs/phase_a.yaml \
        --checkpoints-dir checkpoints_from_gpu/phase_a \
        --seeds 0 1 2

Đây là DRIVER TẠM (giống _run_baselines_*.py), không phải file production.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agents.super_agent_head import PortfolioPolicyLLM  # noqa: E402
from src.envs.portfolio_env import PortfolioEnvPhaseA  # noqa: E402
from src.evaluation.metrics import compute_all_metrics  # noqa: E402
from src.training.ppo_trainer import obs_array_to_dict  # noqa: E402


def load_state_df(processed_path: Path) -> pd.DataFrame:
    """Đọc parquet + khôi phục MultiIndex columns (ticker, feature) — giống
    train_phase_a.load_state_df."""
    df = pd.read_parquet(processed_path)
    df.columns = pd.MultiIndex.from_tuples(
        [tuple(c.split("__", 1)) for c in df.columns], names=["ticker", "feature"]
    )
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def slice_test_range(state_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    start_str, end_str = cfg["data"]["walk_forward"]["test_range"]
    start, end = pd.Timestamp(start_str), pd.Timestamp(end_str)
    test_df = state_df.loc[(state_df.index >= start) & (state_df.index <= end)]
    if len(test_df) < 2:
        raise ValueError(
            f"Test split [{start_str}, {end_str}] chỉ có {len(test_df)} dòng (<2)."
        )
    return test_df


def build_policy_once(cfg: dict) -> PortfolioPolicyLLM:
    """Dựng PortfolioPolicyLLM MỘT LẦN (tải Qwen 1 lần). Các seed sau chỉ nạp
    lại trainable_state_dict (17MB) vào cùng instance — tránh reload 3GB mỗi seed."""
    model_cfg = cfg["model"]
    lora_cfg = model_cfg.get("lora", {})
    tickers = list(cfg["data"]["tickers"])
    policy = PortfolioPolicyLLM(
        base_model_name=model_cfg["base_model_name"],
        n_assets=len(tickers),
        lora_r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        lora_target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]),
        max_prompt_length=model_cfg.get("max_prompt_length", 512),
        torch_dtype=torch.float32,   # CPU -> float32 (bf16 trên CPU chậm hơn)
        device="cpu",
    )
    policy.eval()
    return policy


def load_seed_weights(policy: PortfolioPolicyLLM, ckpt_path: Path) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("trainable_state_dict", ckpt)
    missing, unexpected = policy.load_state_dict(state, strict=False)
    if unexpected:
        print(f"  [warn] {len(unexpected)} key thừa khi load (bỏ qua): {list(unexpected)[:3]}", flush=True)
    policy.eval()


def run_one_seed_backtest(cfg: dict, test_df: pd.DataFrame, policy: PortfolioPolicyLLM) -> dict:
    tickers = list(cfg["data"]["tickers"])
    env_cfg = cfg["env"]
    env = PortfolioEnvPhaseA(
        state_df=test_df,
        tickers=tickers,
        initial_cash=env_cfg["initial_cash"],
        transaction_cost=env_cfg["transaction_cost"],
        price_limit=env_cfg["price_limit"],
        max_weight=env_cfg.get("max_weight"),
    )
    base_close = getattr(env, "_base_close", None)

    obs, _ = env.reset()
    pv_hist = [float(v) for v in getattr(env, "portfolio_value_history", [])]
    done = False
    steps = 0
    with torch.no_grad():
        while not done:
            obs_dict = obs_array_to_dict(obs, tickers, base_close=base_close)
            prompt = policy.state_to_prompt(obs_dict)
            weights, _value = policy([prompt])
            action = weights[0].detach().cpu().numpy().astype(np.float32)
            obs, _reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            steps += 1
            if isinstance(info, dict) and "portfolio_value" in info:
                pv_hist.append(float(info["portfolio_value"]))

    env_hist = getattr(env, "portfolio_value_history", None)
    if isinstance(env_hist, list) and len(env_hist) > len(pv_hist):
        pv_hist = [float(v) for v in env_hist]

    metrics = compute_all_metrics(pv_hist)
    print(f"  steps={steps}, final_value={pv_hist[-1]:,.0f}, metrics={ {k: round(v,4) for k,v in metrics.items()} }", flush=True)
    return metrics


def main() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase_a.yaml")
    ap.add_argument("--checkpoints-dir", default="checkpoints_from_gpu/phase_a")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--output", default="reports/phase_a/raw/Proposed_LLM_PPO.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(REPO_ROOT / args.config, encoding="utf-8"))
    processed_path = REPO_ROOT / cfg["data"]["processed_path"]
    state_df = load_state_df(processed_path)
    test_df = slice_test_range(state_df, cfg)
    print(f"Test split: {test_df.index[0].date()} -> {test_df.index[-1].date()} ({len(test_df)} phiên)")

    print("Đang tải Qwen2.5-1.5B (một lần duy nhất) ...", flush=True)
    policy = build_policy_once(cfg)

    out_path = REPO_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: List[Dict] = []
    for seed in args.seeds:
        ckpt_path = REPO_ROOT / args.checkpoints_dir / f"seed_{seed}" / "policy_trainable.pt"
        if not ckpt_path.exists():
            print(f"[seed {seed}] KHÔNG tìm thấy checkpoint {ckpt_path} — bỏ qua.", flush=True)
            continue
        print(f"[seed {seed}] nạp weights + backtest ...", flush=True)
        load_seed_weights(policy, ckpt_path)
        metrics = run_one_seed_backtest(cfg, test_df, policy)
        results.append(metrics)
        # Ghi JSON sau MỖI seed (tránh mất kết quả nếu treo giữa chừng)
        json.dump(results, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"  -> đã ghi {len(results)} seed vào {out_path}", flush=True)

    print(f"\nHOÀN TẤT: {len(results)} seed metrics -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
