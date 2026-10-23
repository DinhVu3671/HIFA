"""src/training/train_phase_a.py

Script CLI chính — train `PortfolioPolicyLLM` bằng `SimplePPOTrainer` cho
Giai đoạn A (mục 3.6 đặc tả `HIFA_Implementation_Plan_GiaiDoan_A_B.md`).

Chạy:
    python -m src.training.train_phase_a --config configs/phase_a.yaml
    python -m src.training.train_phase_a --config configs/phase_a.yaml --seed 0
    python -m src.training.train_phase_a --config configs/phase_a.yaml --smoke-test

`--smoke-test`: chạy CỰC NHANH (1 seed, n_steps=8, n_epochs=1, total_updates=1)
chỉ để kiểm tra pipeline không có lỗi runtime (shape mismatch, NaN loss...) —
đúng theo checklist mục 3.6 của đặc tả ("Chạy thử pipeline với 1 epoch,
n_steps nhỏ chỉ để kiểm tra không có lỗi runtime"). KHÔNG dùng kết quả từ
smoke-test cho luận văn.

wandb là OPTIONAL: import + init đều bọc try/except rộng — nếu chưa cài
`wandb`, chưa có `WANDB_API_KEY`, hoặc `wandb.init()` lỗi vì bất kỳ lý do gì,
script CHỈ log ra console/tensorboard, KHÔNG crash.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Console mặc định trên Windows (cp1252) không encode được tiếng Việt có dấu
# trong log/--help — ép UTF-8 để script không crash vì UnicodeEncodeError khi
# in log/argparse help. Bọc try/except vì stdout/stderr có thể đã bị thay thế
# (vd. chạy dưới pytest capsys) và không hỗ trợ reconfigure().
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.agents.super_agent_head import PortfolioPolicyLLM  # noqa: E402
from src.training.ppo_trainer import SimplePPOTrainer  # noqa: E402

# Import mềm: `src/envs/portfolio_env.py` được 1 subagent khác viết song song
# (có thể chưa tồn tại tại thời điểm module này được import lần đầu, vd. khi
# chạy `tests/test_ppo_trainer.py` không cần env thật). Lỗi thật sự chỉ nổ ra
# khi `run_one_seed()` thực sự cần dùng PortfolioEnvPhaseA.
try:
    from src.envs.portfolio_env import PortfolioEnvPhaseA  # noqa: E402
except ImportError as _env_import_exc:  # pragma: no cover
    PortfolioEnvPhaseA = None  # type: ignore[assignment]
    _ENV_IMPORT_ERROR: Optional[ImportError] = _env_import_exc
else:
    _ENV_IMPORT_ERROR = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tiện ích
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - không có GPU trong sandbox
        torch.cuda.manual_seed_all(seed)


def resolve_path(path_str: str) -> Path:
    """Resolve đường dẫn tương đối theo thứ tự ưu tiên: absolute -> cwd ->
    REPO_ROOT — để script chạy đúng dù được gọi từ đâu."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    cwd_candidate = Path.cwd() / p
    if cwd_candidate.exists():
        return cwd_candidate
    return REPO_ROOT / p


def load_config(config_arg: str) -> Dict[str, Any]:
    path = resolve_path(config_arg)
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file config tại '{config_arg}' (đã thử: {path})."
        )
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_state_df(cfg: Dict[str, Any]) -> pd.DataFrame:
    """Đọc parquet state dataframe đã xử lý bởi
    `src/data/build_dataset.py::build_state_dataframe` và khôi phục
    MultiIndex columns (ticker, feature) — bị "flatten" thành chuỗi
    'TICKER__feature' lúc lưu (parquet không hỗ trợ MultiIndex columns)."""
    processed_path = resolve_path(cfg["data"]["processed_path"])
    if not processed_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy dataset đã xử lý tại {processed_path}. Chạy trước:\n"
            "  python -m src.data.fetch_ohlcv_us --phase 0\n"
            "  python -m src.data.build_dataset --phase 0"
        )
    df = pd.read_parquet(processed_path)
    df.columns = pd.MultiIndex.from_tuples(
        [tuple(c.split("__", 1)) for c in df.columns], names=["ticker", "feature"]
    )
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def slice_train_range(state_df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Cắt `state_df` theo `data.walk_forward.train_range` trong config —
    Giai đoạn A chỉ train trên đoạn 2019-2022 (mục 3.2/3.3 đặc tả), val/test
    dành cho `src/evaluation` (không thuộc phạm vi file này)."""
    start_str, end_str = cfg["data"]["walk_forward"]["train_range"]
    start, end = pd.Timestamp(start_str), pd.Timestamp(end_str)
    train_df = state_df.loc[(state_df.index >= start) & (state_df.index <= end)]
    if len(train_df) < 2:
        raise ValueError(
            f"Không đủ dữ liệu train trong khoảng [{start_str}, {end_str}] "
            f"(chỉ có {len(train_df)} dòng, cần >= 2)."
        )
    return train_df


def trainable_state_dict(policy: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Chỉ lưu phần state_dict TRAINABLE (LoRA + policy_head + value_head) —
    checkpoint nhẹ hơn nhiều so với lưu cả backbone gốc (đã frozen, có thể
    tải lại từ `base_model_name` bất cứ lúc nào)."""
    trainable_names = {name for name, p in policy.named_parameters() if p.requires_grad}
    full_state = policy.state_dict()
    return {k: v for k, v in full_state.items() if k in trainable_names}


def save_checkpoint(
    ckpt_path: Path,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    seed: int,
    next_update: int,
    total_updates: int,
    n_trainable: int,
    history: List[Dict[str, float]],
    cfg: Dict[str, Any],
) -> None:
    """Lưu checkpoint ĐẦY ĐỦ để resume: trainable weights + optimizer state +
    vị trí update đang dở (`next_update`) + lịch sử train. Ghi ra file tạm rồi
    `os.replace` (atomic) để KHÔNG bao giờ để lại checkpoint hỏng nếu Colab bị
    ngắt ngay giữa lúc đang ghi."""
    tmp_path = ckpt_path.with_suffix(ckpt_path.suffix + ".tmp")
    torch.save(
        {
            "trainable_state_dict": trainable_state_dict(policy),
            "optimizer_state_dict": optimizer.state_dict(),
            "seed": seed,
            "next_update": next_update,
            "total_updates": total_updates,
            "n_trainable_params": n_trainable,
            "history": history,
            "config": cfg,
        },
        tmp_path,
    )
    os.replace(tmp_path, ckpt_path)


def load_checkpoint_if_any(
    ckpt_path: Path,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> Optional[Dict[str, Any]]:
    """Nạp lại checkpoint dở (nếu có) vào `policy` + `optimizer`. Trả về dict
    metadata (`next_update`, `history`, ...) hoặc None nếu chưa có checkpoint.

    Dùng `strict=False` vì checkpoint CHỈ chứa param trainable (LoRA + 2 head),
    backbone frozen đã được nạp lại từ `base_model_name` lúc khởi tạo policy."""
    if not ckpt_path.exists():
        return None
    device = next((p.device for p in policy.parameters()), torch.device("cpu"))
    ckpt = torch.load(ckpt_path, map_location=device)
    missing, unexpected = policy.load_state_dict(ckpt["trainable_state_dict"], strict=False)
    if unexpected:
        logger.warning("Checkpoint có %d key không khớp policy (bỏ qua): %s", len(unexpected), unexpected[:5])
    try:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    except Exception as exc:  # pragma: no cover
        logger.warning("Không nạp được optimizer state (%s) — optimizer khởi tạo lại từ đầu.", exc)
    return ckpt


def try_init_wandb(cfg: Dict[str, Any], run_name: str):
    """wandb OPTIONAL (xem docstring module). Trả về module `wandb` đã init
    thành công, hoặc None nếu không dùng được vì BẤT KỲ lý do gì."""
    try:
        import wandb
    except ImportError:
        logger.warning("Không tìm thấy package `wandb` — chỉ log ra console/tensorboard.")
        return None

    logging_cfg = cfg.get("logging", {})
    has_api_key = bool(os.environ.get("WANDB_API_KEY"))
    mode = logging_cfg.get("wandb_mode")
    if mode is None:
        # Không có API key trong môi trường -> KHÔNG thử "online" (tránh
        # wandb chờ nhập API key / lỗi login làm gián đoạn script không tương tác).
        mode = "online" if has_api_key else "disabled"
    if mode == "disabled":
        logger.info(
            "wandb ở chế độ 'disabled' (không tìm thấy WANDB_API_KEY) — "
            "chỉ log ra console/tensorboard."
        )
        return None

    try:
        wandb.init(
            project=logging_cfg.get("wandb_project", "hifa-phase-a"),
            name=run_name,
            config=cfg,
            reinit=True,
            mode=mode,
        )
        return wandb
    except Exception as exc:  # bắt MỌI lỗi (thiếu API key, không mạng, quota, ...)
        logger.warning("wandb.init() thất bại (%s) — tiếp tục chỉ log console/tensorboard.", exc)
        try:
            wandb.finish()
        except Exception:  # pragma: no cover
            pass
        return None


def try_init_tensorboard(cfg: Dict[str, Any], run_name: str):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        logger.warning("Không có torch.utils.tensorboard (thiếu `tensorboard`?) — bỏ qua.")
        return None
    log_dir = resolve_path(cfg.get("logging", {}).get("tensorboard_dir", "./logs")) / run_name
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        return SummaryWriter(log_dir=str(log_dir))
    except Exception as exc:  # pragma: no cover
        logger.warning("Không thể khởi tạo tensorboard SummaryWriter (%s) — bỏ qua.", exc)
        return None


# ---------------------------------------------------------------------------
# Vòng lặp train 1 seed
# ---------------------------------------------------------------------------
def run_one_seed(cfg: Dict[str, Any], seed: int, smoke_test: bool, resume: bool = True) -> Path:
    if PortfolioEnvPhaseA is None:
        raise ImportError(
            "Không import được `src.envs.portfolio_env.PortfolioEnvPhaseA` "
            f"({_ENV_IMPORT_ERROR!r}). Module này do 1 phần khác của dự án viết — "
            "nếu vừa mới tạo, thử chạy lại; nếu vẫn lỗi, kiểm tra "
            "src/envs/portfolio_env.py có tồn tại và import được không."
        ) from _ENV_IMPORT_ERROR

    set_seed(seed)
    run_name = f"phase_a_seed_{seed}" + ("_smoke" if smoke_test else "")
    logger.info("=== Seed %d (%s) ===", seed, "SMOKE-TEST" if smoke_test else "full run")

    state_df = load_state_df(cfg)
    train_df = slice_train_range(state_df, cfg)
    tickers: List[str] = list(cfg["data"]["tickers"])
    logger.info("Dữ liệu train: %d dòng, %d mã (%s).", len(train_df), len(tickers), tickers)

    env_cfg = cfg["env"]
    env = PortfolioEnvPhaseA(
        state_df=train_df,
        tickers=tickers,
        initial_cash=env_cfg["initial_cash"],
        transaction_cost=env_cfg["transaction_cost"],
        price_limit=env_cfg["price_limit"],
        # Trần tỷ trọng mỗi mã (De-Cuong §6.2) — PHẢI truyền để train áp đúng
        # ràng buộc 30%/mã; nếu bỏ qua, agent có thể all-in 1 mã, lệch đề cương.
        max_weight=env_cfg.get("max_weight"),
    )

    model_cfg = cfg["model"]
    lora_cfg = model_cfg.get("lora", {})
    policy = PortfolioPolicyLLM(
        base_model_name=model_cfg["base_model_name"],
        n_assets=len(tickers),
        lora_r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        lora_target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]),
        max_prompt_length=model_cfg.get("max_prompt_length", 512),
    )

    # Yêu cầu bắt buộc mục 3.6: confirm số tham số trainable rất nhỏ (chỉ
    # LoRA + 2 head) ngay khi khởi tạo.
    n_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in policy.parameters())
    print(
        f"[phase_a][seed={seed}] Tham số trainable: {n_trainable:,} / {n_total:,} tổng "
        f"({100.0 * n_trainable / max(n_total, 1):.4f}%) — chỉ LoRA + policy_head + value_head."
    )
    logger.info("Trainable params = %d, total params = %d", n_trainable, n_total)

    train_cfg = cfg["training"]
    if smoke_test:
        n_steps = min(8, train_cfg.get("n_steps", 512))
        n_epochs = 1
        batch_size = min(4, n_steps)
        total_updates = 1
    else:
        n_steps = train_cfg.get("n_steps", 512)
        n_epochs = train_cfg.get("n_epochs", 4)
        batch_size = train_cfg.get("batch_size", 64)
        total_updates = train_cfg.get("total_updates", 200)

    trainer = SimplePPOTrainer(
        policy=policy,
        env=env,
        lr=train_cfg.get("lr", 1e-5),
        clip_eps=train_cfg.get("clip_eps", 0.2),
        gamma=train_cfg.get("gamma", 0.99),
        gae_lambda=train_cfg.get("gae_lambda", 0.95),
        n_steps=n_steps,
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        dirichlet_concentration_scale=train_cfg.get("dirichlet_concentration_scale", 10.0),
        entropy_coef=train_cfg.get("entropy_coef", 0.01),
        target_kl=train_cfg.get("target_kl", 0.05),
    )

    wandb_mod = None if smoke_test else try_init_wandb(cfg, run_name)
    tb_writer = try_init_tensorboard(cfg, run_name)

    ckpt_dir = resolve_path("checkpoints/phase_a") / f"seed_{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint "đang chạy" (resume được) khác với checkpoint "hoàn tất" cuối.
    resume_ckpt_path = ckpt_dir / "resume.pt"
    final_ckpt_path = ckpt_dir / "policy_trainable.pt"

    # Lưu định kỳ mỗi `save_every` update để không mất quá nhiều tiến độ nếu
    # Colab ngắt phiên. Smoke-test không cần (chỉ 1 update).
    save_every = 1 if smoke_test else int(train_cfg.get("save_every", 10))

    history: List[Dict[str, float]] = []
    start_update = 0

    if resume and not smoke_test:
        meta = load_checkpoint_if_any(resume_ckpt_path, policy, trainer.optimizer)
        if meta is not None:
            start_update = int(meta.get("next_update", 0))
            history = list(meta.get("history", []))
            if start_update >= total_updates:
                logger.info(
                    "Seed %d đã train đủ %d update (resume.pt). Bỏ qua, chỉ ghi checkpoint cuối.",
                    seed,
                    total_updates,
                )
            else:
                logger.info(
                    "Seed %d: RESUME từ update %d/%d (nạp lại trainable weights + optimizer).",
                    seed,
                    start_update,
                    total_updates,
                )

    try:
        for update_idx in range(start_update, total_updates):
            t0 = time.perf_counter()
            stats = trainer.train_step(n_steps=n_steps, n_epochs=n_epochs, batch_size=batch_size)
            stats["update"] = float(update_idx)
            stats["wall_seconds"] = time.perf_counter() - t0
            history.append(stats)

            logger.info(
                "[seed=%d update=%d/%d] reward=%.5f policy_loss=%.5f value_loss=%.5f "
                "entropy=%.5f approx_kl=%.5f clip_frac=%.3f kl_stop_ep=%.0f rollout=%.1fs total=%.1fs",
                seed,
                update_idx + 1,
                total_updates,
                stats["mean_reward"],
                stats["policy_loss"],
                stats["value_loss"],
                stats["entropy"],
                stats["approx_kl"],
                stats["clip_frac"],
                stats.get("kl_early_stop_epoch", 0.0),
                stats["rollout_seconds"],
                stats["wall_seconds"],
            )

            if tb_writer is not None:
                for k, v in stats.items():
                    if k != "update":
                        tb_writer.add_scalar(f"phase_a/{k}", v, update_idx)
            if wandb_mod is not None:
                wandb_mod.log(stats, step=update_idx)

            # Lưu checkpoint resume định kỳ (và luôn ở update cuối). `next_update`
            # = update_idx + 1 vì update này đã xong -> lần resume sẽ bắt đầu từ đó.
            is_last = update_idx == total_updates - 1
            if (update_idx + 1) % save_every == 0 or is_last:
                save_checkpoint(
                    resume_ckpt_path,
                    policy,
                    trainer.optimizer,
                    seed=seed,
                    next_update=update_idx + 1,
                    total_updates=total_updates,
                    n_trainable=n_trainable,
                    history=history,
                    cfg=cfg,
                )
                logger.info(
                    "  ↳ Đã lưu checkpoint resume tại update %d/%d: %s",
                    update_idx + 1,
                    total_updates,
                    resume_ckpt_path,
                )
    finally:
        if tb_writer is not None:
            tb_writer.close()
        if wandb_mod is not None:
            wandb_mod.finish()

    # Checkpoint "hoàn tất" — nhẹ, chỉ chứa trainable weights (dùng cho evaluation).
    torch.save(
        {
            "trainable_state_dict": trainable_state_dict(policy),
            "seed": seed,
            "n_trainable_params": n_trainable,
            "config": cfg,
        },
        final_ckpt_path,
    )
    history_path = ckpt_dir / "train_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    logger.info(
        "Seed %d xong. Checkpoint cuối: %s | Resume: %s | Lịch sử: %s",
        seed,
        final_ckpt_path,
        resume_ckpt_path,
        history_path,
    )
    return final_ckpt_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train PortfolioPolicyLLM (PPO custom, mục 3.6) — Giai đoạn A HIFA."
    )
    parser.add_argument(
        "--config", type=str, default="configs/phase_a.yaml", help="Đường dẫn file config YAML."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Chỉ chạy đúng 1 seed này (override list `training.seeds` trong config).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Chạy CỰC NHANH (1 seed, n_steps nhỏ, 1 epoch, 1 update) chỉ để kiểm tra "
            "pipeline không lỗi runtime — KHÔNG dùng kết quả cho luận văn."
        ),
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help=(
            "Tắt tự động resume: train lại từ đầu dù có checkpoint dở "
            "(mặc định LUÔN resume nếu tìm thấy checkpoints/phase_a/seed_N/resume.pt)."
        ),
    )
    parser.set_defaults(resume=True)
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_arg_parser().parse_args(argv)

    cfg = load_config(args.config)

    if args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = list(cfg.get("training", {}).get("seeds", [0]))
    if args.smoke_test:
        seeds = seeds[:1]

    logger.info(
        "Sẽ train %d seed(s): %s%s (resume=%s)",
        len(seeds),
        seeds,
        " [SMOKE-TEST]" if args.smoke_test else "",
        args.resume,
    )

    for seed in seeds:
        run_one_seed(cfg, seed, smoke_test=args.smoke_test, resume=args.resume)


if __name__ == "__main__":
    main()
