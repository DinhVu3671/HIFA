"""Baseline B4 — PPO (Stable-Baselines3) không có LLM.

Đây là baseline QUAN TRỌNG NHẤT để so sánh với mô hình Đề xuất (HIFA):
cùng thuật toán PPO, cùng môi trường `PortfolioEnvPhaseA`, chỉ khác là B4
dùng policy MLP thuần (SB3 "MlpPolicy") học trực tiếp trên state vector số
học — KHÔNG qua LLM/prompt. Nếu Đề xuất không vượt trội B4 một cách có ý
nghĩa thống kê, phần kiến trúc LLM chưa chứng minh được giá trị gia tăng.

`train_b4(...)` giữ đúng lời gọi API mẫu trong đặc tả kỹ thuật
(`PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/b4")`,
`model.learn(total_timesteps=200_000)`, `model.save(...)`), có bổ sung thêm
vài tham số tuỳ chọn (seed, **ppo_kwargs) để tiện test/smoke-test và tái lập
kết quả (reproducibility qua nhiều seed — Definition of Done yêu cầu tối
thiểu 5 seed) mà không phá vỡ chữ ký gọi hàm gốc.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Sequence

from stable_baselines3 import PPO

from src.envs.portfolio_env import PortfolioEnvPhaseA

# Console mặc định trên Windows (cp1252/cp437) không encode được tiếng Việt
# có dấu trong help text/log của CLI này -> ép UTF-8 để tránh UnicodeEncodeError
# khi chạy trực tiếp trên cmd.exe/PowerShell không bật UTF-8.
if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def train_b4(
    state_df,
    tickers: Sequence[str],
    total_timesteps: int = 200_000,
    save_path: str = "checkpoints/b4_ppo_no_llm",
    seed: Optional[int] = None,
    tensorboard_log: Optional[str] = "./logs/b4",
    verbose: int = 1,
    policy: str = "MlpPolicy",
    env_cls=None,
    env_kwargs: Optional[dict] = None,
    **ppo_kwargs,
):
    """Huấn luyện baseline B4 (PPO, không LLM) trên `PortfolioEnvPhaseA`.

    Args:
        state_df: DataFrame trạng thái (index=date, columns=MultiIndex(ticker, feature)).
        tickers: danh sách mã trong rổ Giai đoạn A/B.
        total_timesteps: tổng số bước huấn luyện PPO (mặc định 200_000 theo đặc tả).
        save_path: đường dẫn lưu checkpoint (không cần đuôi .zip, SB3 tự thêm).
        seed: seed để tái lập kết quả (nên chạy tối thiểu 5 seed khác nhau
            theo Definition of Done Giai đoạn A).
        tensorboard_log: thư mục log tensorboard, truyền None để tắt (hữu ích
            khi chạy smoke-test/CI mà chưa cài `tensorboard`).
        verbose: mức verbose của SB3.
        policy: loại policy SB3 (mặc định "MlpPolicy" theo đặc tả).
        env_cls: lớp môi trường dùng để train. Mặc định `None` =
            `PortfolioEnvPhaseA` (giữ hành vi cũ). Truyền `PortfolioEnvUS` cho
            thí nghiệm thị trường Mỹ (De-Cuong).
        env_kwargs: dict tham số khởi tạo env (vd. initial_cash, max_weight,
            transaction_cost cho PortfolioEnvUS).
        **ppo_kwargs: tham số bổ sung truyền thẳng vào `PPO(...)` (ví dụ
            `n_steps`, `batch_size`, `learning_rate`...) — dùng để override
            hyperparameter khi smoke-test (n_steps nhỏ) mà không đổi API chính.

    Returns:
        model: đối tượng `stable_baselines3.PPO` đã huấn luyện.
    """
    cls = env_cls if env_cls is not None else PortfolioEnvPhaseA
    env = cls(state_df, tickers, **(env_kwargs or {}))
    model = PPO(
        policy,
        env,
        verbose=verbose,
        tensorboard_log=tensorboard_log,
        seed=seed,
        **ppo_kwargs,
    )
    model.learn(total_timesteps=total_timesteps)

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    model.save(save_path)
    return model


def _load_phase_data(phase: str):
    """Load dữ liệu thật cho Giai đoạn A hoặc B.

    Import trễ (lazy import) bên trong hàm này để module `b4_ppo_no_llm`
    (và `train_b4`) vẫn import/dùng được ngay cả khi
    `src/data/build_dataset.py` chưa tồn tại hoặc đang được chỉnh sửa song
    song bởi phần khác của dự án — chỉ khi thực sự gọi CLI (`main()`) mới
    cần module dữ liệu sẵn sàng.

    Ưu tiên đọc lại state dataframe đã xử lý sẵn ở `data/processed/phase_{phase}.parquet`
    (output của `python -m src.data.build_dataset --phase {phase}`) qua
    `load_state_dataframe()`; nếu chưa có, tự build từ OHLCV thô ở
    `data/raw/ohlcv/phase_{phase}.parquet` (output của `fetch_ohlcv.py`).
    """
    from pathlib import Path

    from src.data.build_dataset import build_state_dataframe, load_state_dataframe

    repo_root = Path(__file__).resolve().parents[2]
    tickers_phase_a = ["FPT", "VPB", "HPG", "VNM", "VCB"]
    tickers_phase_b = tickers_phase_a + ["MWG", "MSN", "GAS", "VIC", "TCB"]
    tickers = tickers_phase_a if phase == "a" else tickers_phase_b

    processed_path = repo_root / "data" / "processed" / f"phase_{phase}.parquet"
    if processed_path.exists():
        state_df = load_state_dataframe(processed_path)
    else:
        ohlcv_path = repo_root / "data" / "raw" / "ohlcv" / f"phase_{phase}.parquet"
        state_df = build_state_dataframe(tickers=tickers, ohlcv_path=ohlcv_path)
    return state_df, tickers


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Huấn luyện baseline B4 (PPO, không LLM) trên PortfolioEnvPhaseA."
    )
    parser.add_argument("--phase", choices=["a", "b"], default="a", help="Giai đoạn A (5 mã) hoặc B (10 mã).")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Tổng số bước PPO. Mặc định 200000 (hoặc 2000 nếu dùng --smoke-test và không truyền cờ này).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Chạy nhanh để kiểm tra không lỗi runtime (mặc định total_timesteps=2000 nếu không override).",
    )
    parser.add_argument("--save-path", default=None, help="Đường dẫn lưu checkpoint.")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    if args.total_timesteps is not None:
        total_timesteps = args.total_timesteps
    elif args.smoke_test:
        total_timesteps = 2_000
    else:
        total_timesteps = 200_000

    save_path = args.save_path or f"checkpoints/b4_ppo_no_llm_phase_{args.phase}"

    state_df, tickers = _load_phase_data(args.phase)
    train_b4(
        state_df,
        tickers,
        total_timesteps=total_timesteps,
        save_path=save_path,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
