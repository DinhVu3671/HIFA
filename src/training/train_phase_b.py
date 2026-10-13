"""src/training/train_phase_b.py

Staged training — huấn luyện tuần tự theo tầng (Giai đoạn B, mục 4.5 đặc tả
`HIFA_Implementation_Plan_GiaiDoan_A_B.md`).

3 bước tuần tự, ĐÚNG theo mẫu đặc tả `train_staged()`:
    Bước 1 (`train_base_agent`): train từng base agent (fundamental/
        technical/sentiment) ĐỘC LẬP trên "reward cục bộ" (tự định nghĩa —
        xem docstring hàm để biết công thức và lý do đơn giản hoá MVP).
    Bước 2 (`train_meta_aggregator`): ĐÓNG BĂNG Tầng 1 (3 base agent),
        train `MetaAggregator`. ĐIỂM DỄ LỖI NHẤT theo cảnh báo mục 4.5 —
        quên freeze khiến gradient chảy ngược vào Tầng 1, phá vỡ giả định
        huấn luyện tuần tự. Hàm ASSERT tường minh trước khi train.
    Bước 3 (`train_super_agent`): ĐÓNG BĂNG Tầng 1+2, train Super-Agent
        bằng PPO THẬT (tái sử dụng `SimplePPOTrainer` của Giai đoạn A,
        src/training/ppo_trainer.py) trên `MetaScoreObsWrapper` bọc quanh
        `PortfolioEnvPhaseB`.

`train_staged()` orchestrate cả 3 bước, LƯU CHECKPOINT SAU MỖI BƯỚC riêng
biệt (`checkpoints/phase_b/{base_agents,meta,super_agent}/`) — nếu Bước 3
lỗi, KHÔNG cần train lại Bước 1-2 (yêu cầu tường minh của đặc tả mục 4.5).

CLI: `python -m src.training.train_phase_b --config configs/phase_b.yaml
[--smoke-test]` — `--smoke-test` dùng dữ liệu giả lập rất nhỏ (vài chục
ngày, vài mã) + vài bước huấn luyện, CHỈ để verify pipeline chạy hết không
lỗi runtime, KHÔNG dùng để train thật (train thật cần Qwen2.5-1.5B-Instruct
tải qua mạng + GPU/dữ liệu 10 mã thật — ngoài phạm vi kiểm chứng ở đây).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.agents.base_agent import AGENT_TYPES, BaseAgent, build_prompt_for_agent
from src.agents.meta_agent import MetaAggregator
from src.agents.super_agent_head import PortfolioPolicyLLM
from src.data.build_dataset import load_state_dataframe

# Placeholder tin tức khi chưa có pipeline tin tức thật cho thị trường Mỹ.
# Giai đoạn 0/A không dùng tin tức; Giai đoạn B/C sẽ thay bằng nguồn thật
# (Yahoo Finance RSS / NewsAPI) qua một module fetch_news_us riêng.
_NO_NEWS_TEXT = "No recent news."


def _attach_placeholder_news(state_df, tickers, text: str = _NO_NEWS_TEXT):
    """Gắn cột (ticker, 'recent_news') = placeholder cho mọi ngày.

    Tạm thời cho Giai đoạn 0/A (thị trường Mỹ chưa có pipeline tin tức thật).
    Giữ đúng layout mà prompt của Sentiment Agent kỳ vọng.
    """
    df = state_df.copy()
    for tk in tickers:
        df[(tk, "recent_news")] = text
    return df.sort_index(axis=1)

DEFAULT_CHECKPOINT_ROOT = "checkpoints/phase_b"


# ---------------------------------------------------------------------------
# Helper dùng chung: trích 1 dict ticker_data từ 1 hàng của state_df
# ---------------------------------------------------------------------------
def _extract_ticker_data(row: "pd.Series", ticker: str, agent_type: str) -> Dict[str, Any]:
    """Trích dict `ticker_data` từ 1 hàng của `state_df` (MultiIndex columns
    `(ticker, feature)`) theo đúng field mà `build_xxx_prompt()`
    (src/agents/base_agent.py, mục 4.3) cần cho `agent_type`.

    Dùng fallback an toàn (`try/except KeyError`) thay vì để crash nếu cột
    thiếu — vì schema đầy đủ của `state_df` (đặc biệt P/E, P/B, EPS growth
    cho fundamental, vốn KHÔNG có trong dataset numeric tối thiểu của Giai
    đoạn A, xem `build_dataset.py`) do pipeline dữ liệu Giai đoạn B cung cấp
    và không thuộc phạm vi các file được giao ở đây; fallback giúp
    `train_phase_b.py` không phụ thuộc cứng vào 1 schema chưa chốt, chỉ mất
    độ chính xác của prompt khi thiếu field.
    """

    def g(feat: str, default: float = 0.0) -> Any:
        try:
            return row[(ticker, feat)]
        except KeyError:
            return default

    data: Dict[str, Any] = {"ticker": ticker}
    if agent_type == "fundamental":
        data["pe"] = g("pe", 0.0)
        data["pb"] = g("pb", 0.0)
        data["eps_growth"] = g("eps_growth", 0.0)
    elif agent_type == "technical":
        data["rsi_14"] = g("rsi_14", 50.0)
        data["macd"] = g("macd", 0.0)
        data["macd_signal"] = g("macd_signal", 0.0)
        data["volatility_20d"] = g("volatility_20d", 0.0)
    elif agent_type == "sentiment":
        data["recent_news"] = g("recent_news", "Không có tin tức mới.")
    else:  # pragma: no cover - đã được validate ở tầng gọi
        raise ValueError(f"agent_type không hợp lệ: {agent_type!r}")
    return data


def _next_day_returns(row: "pd.Series", next_row: "pd.Series", tickers: List[str]) -> List[float]:
    return [
        (float(next_row[(tk, "close")]) - float(row[(tk, "close")])) / (float(row[(tk, "close")]) + 1e-8)
        for tk in tickers
    ]


# ---------------------------------------------------------------------------
# BƯỚC 1 — train từng base agent độc lập
# ---------------------------------------------------------------------------
def train_base_agent(
    agent_type: str, state_df: "pd.DataFrame", tickers: List[str], config: Dict[str, Any]
) -> BaseAgent:
    """BƯỚC 1 của staged training (mục 4.5) — train 1 `BaseAgent` chuyên
    biệt ĐỘC LẬP trên "reward cục bộ".

    ĐỊNH NGHĨA REWARD CỤC BỘ (TỰ CHỌN — đặc tả mục 4.5 KHÔNG cho công thức cụ
    thể, chỉ nói "lợi nhuận nếu hành động theo khuyến nghị riêng của agent
    đó"):
        reward_local(t, ticker) = score_agent(t, ticker) * next_day_return(t, ticker)
    Diễn giải: nếu agent khuyến nghị mua (score > 0) và giá THẬT SỰ tăng ngày
    hôm sau (next_day_return > 0) thì reward dương ("agent đúng"); nếu score
    và next_day_return trái dấu thì reward âm ("agent sai"). |score| lớn
    (mức tự tin cao) khuếch đại cả thưởng lẫn phạt — đây là dạng "directional
    trading reward" chuẩn cho tín hiệu long/short, không cần chuẩn hoá
    phương sai vì chỉ dùng để attribution cục bộ cho 1 agent (không phải
    reward cấp danh mục).

    ĐƠN GIẢN HOÁ CHO MVP GIAI ĐOẠN B (ghi rõ theo yêu cầu — ĐÂY KHÔNG PHẢI
    PPO ĐẦY ĐỦ): `reward_local` ở trên là hàm TUYẾN TÍNH theo `score` (đạo
    hàm riêng theo score = next_day_return, hằng số với batch cố định) — tối
    đa hoá trực tiếp kỳ vọng reward_local bằng gradient ascent sẽ đẩy score
    về +-1 với biên độ LỚN NHẤT có thể cho MỌI mẫu (miễn đúng dấu), làm mất
    ý nghĩa "mức độ tự tin định lượng" và không cần cơ chế PPO (không có
    action rời rạc/liên tục cần lấy mẫu qua env, không có concept "episode").
    Vì BaseAgent chỉ output 1 SCORE liên tục (không phải hành động phân bổ
    danh mục qua `PortfolioEnvPhaseB`), ta dùng 1 VÒNG LẶP SUPERVISED-STYLE
    TỐI GIẢN thay cho PPO đầy đủ: train agent bằng
        MSE(score, tanh(next_day_return / return_scale))
    (regression về hướng + biên độ chuẩn hoá của return thực tế ngày kế
    tiếp). `reward_local` vẫn được TÍNH và LOG lại mỗi epoch (KHÔNG dùng để
    lan truyền ngược) để theo dõi "agent có học được tín hiệu có lợi nhuận
    hay không", đúng checklist mục 4.5 ("theo dõi reward cục bộ ... trước
    khi ghép Bước 2"). PPO đầy đủ (`SimplePPOTrainer` qua env thật) CHỈ bắt
    buộc cho Super-Agent ở Bước 3 (`train_super_agent`) — không bắt buộc cho
    base agent theo đặc tả.

    Args:
        agent_type: "fundamental" | "technical" | "sentiment".
        state_df: DataFrame giá/indicator, index=ngày tăng dần, columns =
            MultiIndex `(ticker, feature)` (xem `_extract_ticker_data`).
        tickers: danh sách mã dùng để train (thứ tự không quan trọng ở bước
            này vì mỗi mã được xử lý độc lập trong cùng 1 batch/ngày).
        config: dict cấu hình (xem `configs/phase_b.yaml`), đọc từ
            `config["agents"]["base_model_name"]` và
            `config["training"]["stage1_base_agents"]`.

    Returns:
        `BaseAgent` đã train, có thêm attribute `_train_history` (dict với
        "loss" và "reward_local_mean" theo từng epoch) để `train_staged()`
        log/kiểm tra "agent có học được gì không" trước khi sang Bước 2.
    """
    if agent_type not in AGENT_TYPES:
        raise ValueError(f"agent_type phải thuộc {AGENT_TYPES}, nhận được: {agent_type!r}")

    agent_cfg = config.get("agents", {})
    base_model_name = agent_cfg.get("base_model_name", "Qwen/Qwen2.5-1.5B-Instruct")

    stage_cfg = config.get("training", {}).get("stage1_base_agents", {})
    n_epochs = int(stage_cfg.get("n_epochs", 4))
    lr = float(stage_cfg.get("lr", 1e-5))
    # return_scale: chuẩn hoá next_day_return trước khi đưa qua tanh() làm
    # target hồi quy — return hàng ngày cổ phiếu VN30 thường trong khoảng
    # vài %, dùng 0.05 (5%) làm mốc "biến động điển hình 1 ngày" mặc định.
    return_scale = float(stage_cfg.get("return_scale", 0.05))

    n_dates = len(state_df)
    if n_dates < 2:
        raise ValueError("state_df cần tối thiểu 2 ngày để tính next-day-return.")

    agent = BaseAgent(base_model_name, agent_type)
    trainable_params = [p for p in agent.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    history: Dict[str, List[float]] = {"loss": [], "reward_local_mean": []}

    for _epoch in range(n_epochs):
        epoch_losses: List[float] = []
        epoch_rewards: List[float] = []

        for t in range(n_dates - 1):
            row = state_df.iloc[t]
            next_row = state_df.iloc[t + 1]

            prompts = [
                build_prompt_for_agent(agent_type, _extract_ticker_data(row, tk, agent_type))
                for tk in tickers
            ]
            next_returns = torch.tensor(
                _next_day_returns(row, next_row, tickers), dtype=torch.float32
            ).unsqueeze(-1)  # (n_tickers, 1)
            target = torch.tanh(next_returns / return_scale)

            score, _value = agent(prompts)  # (n_tickers, 1)

            # reward cục bộ CHỈ để log/theo dõi — KHÔNG backward (xem docstring).
            with torch.no_grad():
                reward_local = float((score * next_returns).mean().item())
            epoch_rewards.append(reward_local)

            loss = nn.functional.mse_loss(score, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(float(loss.item()))

        history["loss"].append(sum(epoch_losses) / max(len(epoch_losses), 1))
        history["reward_local_mean"].append(sum(epoch_rewards) / max(len(epoch_rewards), 1))

    agent._train_history = history  # gắn lại để train_staged() log/kiểm tra
    return agent


# ---------------------------------------------------------------------------
# BƯỚC 2 — freeze Tầng 1, train MetaAggregator
# ---------------------------------------------------------------------------
def train_meta_aggregator(
    fundamental_agent: BaseAgent,
    technical_agent: BaseAgent,
    sentiment_agent: BaseAgent,
    state_df: "pd.DataFrame",
    tickers: List[str],
    config: Dict[str, Any],
) -> MetaAggregator:
    """BƯỚC 2 của staged training (mục 4.5): đóng băng cả 3 base agent, chỉ
    train `MetaAggregator` (supervised đơn giản: MSE giữa `meta_score` và
    return thực tế ngày kế tiếp đã chuẩn hoá).

    ĐIỂM DỄ LỖI NHẤT theo cảnh báo tường minh của đặc tả mục 4.5: "quên
    freeze khiến gradient vẫn chảy ngược vào tầng trước, làm sai giả định
    huấn luyện tuần tự". Hàm này (a) TỰ set `requires_grad=False` cho cả 3
    agent (không tin tưởng lời gọi bên ngoài đã làm đúng), (b) ASSERT tường
    minh ngay sau đó để chặn sớm nếu freeze không thành công, THAY VÌ chỉ
    phát hiện muộn qua debug loss không hội tụ.
    """
    agents: Dict[str, BaseAgent] = {
        "fundamental": fundamental_agent,
        "technical": technical_agent,
        "sentiment": sentiment_agent,
    }
    for name, agent in agents.items():
        for p in agent.parameters():
            p.requires_grad = False
        assert not any(p.requires_grad for p in agent.parameters()), (
            f"Base agent '{name}' chưa được freeze đúng cách (còn tham số "
            "requires_grad=True) — KHÔNG được train MetaAggregator khi Tầng 1 "
            "chưa đóng băng hoàn toàn (cảnh báo tường minh mục 4.5)."
        )
        agent.eval()

    meta_cfg = config.get("meta_agent", {})
    n_sources = int(meta_cfg.get("n_sources", 3))
    hidden = int(meta_cfg.get("hidden", 32))

    stage_cfg = config.get("training", {}).get("stage2_meta", {})
    lr = float(stage_cfg.get("lr", 1e-3))
    epochs = int(stage_cfg.get("epochs", 50))
    return_scale = float(stage_cfg.get("return_scale", 0.05))
    vol_window = int(stage_cfg.get("confidence_vol_window", 5))

    n_dates = len(state_df)
    if n_dates < vol_window + 2:
        raise ValueError(
            f"state_df có {n_dates} ngày, quá ngắn so với confidence_vol_window="
            f"{vol_window} (+2 cho ít nhất 1 cặp train + rolling window)."
        )

    order = list(agents.keys())  # thứ tự nguồn CỐ ĐỊNH: fundamental, technical, sentiment

    # Precompute score của 3 base agent cho MỌI (ngày, mã) — 1 LẦN, vì cả 3
    # đã freeze (không cần gradient) và forward LLM là phần chậm nhất; tránh
    # forward lặp lại mỗi epoch (epochs có thể tới 50, xem config mẫu).
    with torch.no_grad():
        scores_per_type: Dict[str, torch.Tensor] = {name: [] for name in order}
        for t in range(n_dates):
            row = state_df.iloc[t]
            for name, agent in agents.items():
                prompts = [
                    build_prompt_for_agent(name, _extract_ticker_data(row, tk, name))
                    for tk in tickers
                ]
                score, _ = agent(prompts)  # (n_tickers, 1)
                scores_per_type[name].append(score.squeeze(-1))
        for name in order:
            scores_per_type[name] = torch.stack(scores_per_type[name], dim=0)  # (n_dates, n_tickers)

    # confidence động = compute_confidence(độ lệch chuẩn rolling của chính
    # score nguồn đó qua `vol_window` ngày gần nhất) — "nguồn càng ổn định
    # càng tin cậy", đúng docstring MetaAggregator.compute_confidence().
    meta = MetaAggregator(n_sources=n_sources, hidden=hidden)
    confidences_per_type: Dict[str, torch.Tensor] = {}
    for name in order:
        scores = scores_per_type[name]
        conf = torch.zeros_like(scores)
        for t in range(n_dates):
            lo = max(0, t - vol_window + 1)
            window = scores[lo : t + 1]
            vol = float(window.std(unbiased=False).item()) if window.shape[0] > 1 else 0.0
            conf[t] = meta.compute_confidence(vol)
        confidences_per_type[name] = conf

    optimizer = torch.optim.AdamW(meta.parameters(), lr=lr)
    history: Dict[str, List[float]] = {"loss": []}

    for _epoch in range(epochs):
        epoch_losses: List[float] = []
        for t in range(n_dates - 1):
            row = state_df.iloc[t]
            next_row = state_df.iloc[t + 1]

            scores_t = torch.stack([scores_per_type[name][t] for name in order], dim=-1)  # (n_tickers, n_sources)
            conf_t = torch.stack([confidences_per_type[name][t] for name in order], dim=-1)

            next_returns = torch.tensor(
                _next_day_returns(row, next_row, tickers), dtype=torch.float32
            ).unsqueeze(-1)
            target = torch.tanh(next_returns / return_scale)

            meta_score, _value = meta(scores_t, conf_t)
            loss = nn.functional.mse_loss(meta_score, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        history["loss"].append(sum(epoch_losses) / max(len(epoch_losses), 1))

    meta._train_history = history
    return meta


# ---------------------------------------------------------------------------
# Adapter Bước 3 — obs = meta score (không phải state numeric gốc)
# ---------------------------------------------------------------------------
class MetaScoreObsWrapper(gym.Wrapper):
    """Adapter cho Super-Agent (Bước 3, mục 4.5): thay vì Super-Agent nhận
    state numeric gốc (giá + indicator) như `PortfolioEnvPhaseB` cung cấp
    mặc định, wrapper này khiến obs trả về là (meta score + 3 base score +
    độ tin cậy trung bình) TÍNH TỪ Tầng 1+2 đã freeze — đúng tinh thần kiến
    trúc 3 tầng: Super-Agent CHỈ nhìn thấy đầu ra đã tổng hợp, không nhìn
    thấy dữ liệu thô nữa.

    DECISION QUAN TRỌNG (ĐƠN GIẢN HOÁ CHO MVP, ghi rõ vì ảnh hưởng tới cách
    đọc code): `SimplePPOTrainer` (src/training/ppo_trainer.py, viết song
    song bởi 1 subagent khác, KHÔNG thuộc phạm vi sửa ở đây) gọi
    `obs_array_to_dict(obs, tickers, ...)` rồi `policy.state_to_prompt(obs_dict)`
    để biến numeric obs thành prompt — hàm `obs_array_to_dict` CỐ ĐỊNH kỳ
    vọng obs có ÍT NHẤT `n_tickers * 5` phần tử, đọc theo 5-slot/mã với tên
    khoá CỐ ĐỊNH `("close", "rsi_14", "macd", "macd_signal", "volatility_20d")`.
    Để KHÔNG phải sửa `ppo_trainer.py`/`super_agent_head.py` (ngoài phạm vi
    5 file được giao) mà vẫn tái sử dụng nguyên vẹn `SimplePPOTrainer`, obs
    ở đây được ĐÓNG GÓI lại đúng layout 5-slot/mã đó, nhưng với Ý NGHĨA khác:

        slot "close"          -> meta_score    (Tầng 2, tổng hợp)
        slot "rsi_14"         -> fundamental_score (Tầng 1)
        slot "macd"           -> technical_score   (Tầng 1)
        slot "macd_signal"    -> sentiment_score   (Tầng 1)
        slot "volatility_20d" -> độ tin cậy trung bình (3 nguồn)
        + nối thêm current_weights (n_tickers+1) ở cuối (giữ đúng tinh thần
          "10 meta score + current weights" của đặc tả — `obs_array_to_dict`
          bỏ qua phần này, chỉ đọc `n_tickers*5` phần tử đầu).

    Vì các khoá dict bị TÁI SỬ DỤNG sai ngữ nghĩa gốc (vd "close" không còn
    là giá), `policy.state_to_prompt` MẶC ĐỊNH (vốn in ra "giá=...,
    RSI=...") sẽ tạo prompt GÂY HIỂU NHẦM cho LLM nếu dùng nguyên bản — nên
    `train_super_agent()` GHI ĐÈ `state_to_prompt` ở CẤP INSTANCE (không sửa
    file `super_agent_head.py`) bằng `build_super_agent_state_to_prompt()`
    bên dưới, diễn giải lại đúng ý nghĩa thật của từng khoá.

    Một hệ quả kỹ thuật khác cần xử lý: `SimplePPOTrainer._obs_to_prompt()`
    còn gọi `obs_array_to_dict(..., base_close=env._base_close)` để "khôi
    phục giá thật" bằng cách NHÂN slot "close" với giá đóng cửa gốc — nếu để
    mặc định, meta_score (dải [-1,1]) sẽ bị nhân với giá cổ phiếu (vd
    87000), phá hỏng hoàn toàn ý nghĩa số liệu TRƯỚC KHI tới
    `state_to_prompt`. Do đó `__init__` bên dưới GHI ĐÈ `self._base_close =
    None` (thuộc tính instance, không phải qua `__getattr__` delegation của
    `gym.Wrapper`) để tắt hẳn cơ chế "khôi phục giá" này cho wrapper.

    Đây là simplification hợp lý cho MVP (tính meta score ONLINE theo từng
    bước, không cache/precompute như `train_meta_aggregator`, vì Super-Agent
    tương tác tuần tự qua episode) — bản đầy đủ/production nên precompute
    toàn bộ meta score theo ngày 1 lần để tránh forward LLM lặp lại.
    """

    def __init__(
        self,
        env: Any,
        fundamental_agent: BaseAgent,
        technical_agent: BaseAgent,
        sentiment_agent: BaseAgent,
        meta_aggregator: MetaAggregator,
        tickers: List[str],
        extract_ticker_data_fn: Optional[Callable[["pd.Series", str, str], Dict[str, Any]]] = None,
    ) -> None:
        super().__init__(env)
        self.fundamental_agent = fundamental_agent
        self.technical_agent = technical_agent
        self.sentiment_agent = sentiment_agent
        self.meta_aggregator = meta_aggregator
        self.tickers = list(tickers)
        self.n_tickers = len(self.tickers)
        self._extract_ticker_data_fn = extract_ticker_data_fn or _extract_ticker_data

        obs_dim = self.n_tickers * 5 + (self.n_tickers + 1)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        # Tắt cơ chế "khôi phục giá thật" của SimplePPOTrainer._obs_to_prompt()
        # cho wrapper này — xem giải thích ở docstring class.
        self._base_close = None

    # ------------------------------------------------------------------
    def _agent_scores(self, agent: BaseAgent, agent_type: str, row: "pd.Series") -> torch.Tensor:
        prompts = [
            build_prompt_for_agent(agent_type, self._extract_ticker_data_fn(row, tk, agent_type))
            for tk in self.tickers
        ]
        with torch.no_grad():
            score, _ = agent(prompts)  # (n_tickers, 1)
        return score.squeeze(-1)

    def _build_obs(self) -> np.ndarray:
        row = self.env.state_df.iloc[self.env.t]

        with torch.no_grad():
            fund_scores = self._agent_scores(self.fundamental_agent, "fundamental", row)
            tech_scores = self._agent_scores(self.technical_agent, "technical", row)
            sent_scores = self._agent_scores(self.sentiment_agent, "sentiment", row)
            base_scores = torch.stack([fund_scores, tech_scores, sent_scores], dim=-1)  # (n_tickers, 3)

            # Proxy "biến động gần đây của nguồn" theo phương ONLINE (đơn
            # giản hoá MVP): độ lệch chuẩn của mỗi nguồn GIỮA CÁC MÃ trong
            # cùng 1 ngày — bản offline đầy đủ (train_meta_aggregator) dùng
            # rolling theo thời gian, ở đây không khả thi vì Super-Agent
            # tương tác tuần tự, không có sẵn lịch sử score đã precompute.
            source_vol = base_scores.std(dim=0, unbiased=False)  # (3,)
            conf_values = [self.meta_aggregator.compute_confidence(float(v.item())) for v in source_vol]
            confidences = torch.tensor(conf_values, dtype=torch.float32).unsqueeze(0).expand(
                self.n_tickers, -1
            )
            avg_confidence = confidences.mean(dim=-1)  # (n_tickers,)

            meta_score, _ = self.meta_aggregator(base_scores, confidences)  # (n_tickers, 1)

        per_ticker = np.stack(
            [
                meta_score.squeeze(-1).numpy(),
                fund_scores.numpy(),
                tech_scores.numpy(),
                sent_scores.numpy(),
                avg_confidence.numpy(),
            ],
            axis=-1,
        )  # (n_tickers, 5) -- layout tương thích obs_array_to_dict(), xem docstring class
        weights = np.asarray(self.env.weights, dtype=np.float32)
        obs = np.concatenate([per_ticker.reshape(-1).astype(np.float32), weights])
        return obs.astype(np.float32)

    # ------------------------------------------------------------------
    def reset(self, **kwargs: Any):
        _obs, info = self.env.reset(**kwargs)
        return self._build_obs(), info

    def step(self, action: np.ndarray):
        _obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            return np.zeros(self.observation_space.shape, dtype=np.float32), reward, terminated, truncated, info
        return self._build_obs(), reward, terminated, truncated, info


def build_super_agent_state_to_prompt(obs_dict: Dict[str, Dict[str, float]]) -> str:
    """Thay thế cho `PortfolioPolicyLLM.state_to_prompt()` mặc định, DÙNG
    RIÊNG cho Super-Agent ở Bước 3 (gán qua `super_agent.state_to_prompt =
    build_super_agent_state_to_prompt` trong `train_super_agent()`).

    `obs_dict` ở đây được `SimplePPOTrainer._obs_to_prompt()` dựng từ obs
    numeric của `MetaScoreObsWrapper` qua `obs_array_to_dict()` — các khoá
    "close"/"rsi_14"/"macd"/"macd_signal"/"volatility_20d" bị TÁI SỬ DỤNG để
    mang (meta_score, fundamental_score, technical_score, sentiment_score,
    độ tin cậy trung bình) thay vì giá/chỉ báo kỹ thuật thật (xem docstring
    `MetaScoreObsWrapper`). Hàm này diễn giải lại ĐÚNG ý nghĩa thật khi in
    ra prompt, tránh đưa cho LLM 1 mô tả sai lệch kiểu "giá=..., RSI=...".
    """
    lines = ["Điểm tổng hợp từ Tầng 1 (base agent) + Tầng 2 (meta aggregator):"]
    for tk, feat in obs_dict.items():
        lines.append(
            f"- {tk}: meta_score={feat['close']:.2f}, "
            f"fundamental={feat['rsi_14']:.2f}, technical={feat['macd']:.2f}, "
            f"sentiment={feat['macd_signal']:.2f}, do_tin_cay_tb={feat['volatility_20d']:.2f}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# BƯỚC 3 — freeze Tầng 1+2, train Super-Agent bằng PPO thật
# ---------------------------------------------------------------------------
def train_super_agent(meta_aggregator: MetaAggregator, env: Any, config: Dict[str, Any]) -> nn.Module:
    """BƯỚC 3 của staged training (mục 4.5): đóng băng `MetaAggregator`
    (kiểm tra tường minh bằng assert, không tin tưởng lời gọi bên ngoài),
    rồi train 1 Super-Agent (`PortfolioPolicyLLM` mới, output phân bổ danh
    mục đầy đủ) bằng PPO THẬT — tái sử dụng nguyên vẹn `SimplePPOTrainer`
    của Giai đoạn A (src/training/ppo_trainer.py) trên `env` (BẮT BUỘC đã
    được bọc bởi `MetaScoreObsWrapper`, xem class đó để biết chi tiết).
    """
    assert not any(p.requires_grad for p in meta_aggregator.parameters()), (
        "MetaAggregator (Tầng 2) chưa được freeze — PHẢI freeze trước khi train "
        "Super-Agent (Bước 3, cảnh báo tường minh mục 4.5). Lưu ý: ở thiết kế "
        "hiện tại, MetaScoreObsWrapper luôn gọi meta_aggregator trong "
        "torch.no_grad() nên gradient PPO thực tế không thể lan vào Tầng 2 dù "
        "quên freeze — nhưng assert này VẪN bắt buộc để chặn sớm nếu code sau "
        "này thay đổi và vô tình để lộ gradient path."
    )
    meta_aggregator.eval()

    # Import trễ: module do 1 subagent khác viết song song với file này —
    # không import ở đầu file để train_phase_b.py vẫn load được (vd. Bước 1,
    # Bước 2, test) ngay cả khi ppo_trainer.py tạm thời chưa sẵn sàng/đang
    # được chỉnh sửa đồng thời.
    from src.training.ppo_trainer import SimplePPOTrainer

    agent_cfg = config.get("agents", {})
    base_model_name = agent_cfg.get("base_model_name", "Qwen/Qwen2.5-1.5B-Instruct")

    tickers: List[str] = list(getattr(env, "tickers"))
    n_assets = len(tickers)

    super_agent = PortfolioPolicyLLM(base_model_name, n_assets=n_assets)
    # Override state_to_prompt CHỈ cho instance này (không sửa
    # super_agent_head.py) — xem MetaScoreObsWrapper +
    # build_super_agent_state_to_prompt() để biết lý do.
    super_agent.state_to_prompt = build_super_agent_state_to_prompt  # type: ignore[assignment]

    stage_cfg = config.get("training", {}).get("stage3_super_agent", {})
    n_steps = int(stage_cfg.get("n_steps", 512))
    n_epochs = int(stage_cfg.get("n_epochs", 4))
    batch_size = int(stage_cfg.get("batch_size", 64))
    lr = float(stage_cfg.get("lr", 1e-5))
    # n_updates: số vòng lặp PPO (collect_rollout -> update) — KHÔNG có
    # trong khung config mẫu (configs/phase_b.yaml chỉ có n_steps/n_epochs/
    # batch_size/lr) nên mặc định 1 (đủ để verify pipeline chạy hết ở
    # smoke-test); train thật nên tăng qua
    # config["training"]["stage3_super_agent"]["n_updates"].
    n_updates = int(stage_cfg.get("n_updates", 1))

    trainer = SimplePPOTrainer(super_agent, env, lr=lr, n_steps=n_steps, tickers=tickers)

    history: List[Dict[str, float]] = []
    for update_i in range(n_updates):
        stats = trainer.train_step(n_steps=n_steps, n_epochs=n_epochs, batch_size=batch_size)
        history.append(stats)
        print(f"[train_super_agent] update {update_i + 1}/{n_updates}: {stats}")

    super_agent._train_history = history  # type: ignore[attr-defined]
    return super_agent


# ---------------------------------------------------------------------------
# Orchestration — 3 bước tuần tự + checkpoint riêng biệt sau mỗi bước
# ---------------------------------------------------------------------------
def train_staged(state_df: "pd.DataFrame", tickers: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
    """Orchestration ĐÚNG 3 bước tuần tự theo mẫu đặc tả mục 4.5.

    LƯU CHECKPOINT SAU MỖI BƯỚC riêng biệt (`checkpoints/phase_b/
    {base_agents,meta,super_agent}/`) — nếu Bước 3 lỗi, KHÔNG cần train lại
    Bước 1-2 (yêu cầu tường minh của đặc tả mục 4.5).
    """
    ckpt_root = Path(config.get("checkpoints_dir", DEFAULT_CHECKPOINT_ROOT))
    base_ckpt_dir = ckpt_root / "base_agents"
    meta_ckpt_dir = ckpt_root / "meta"
    super_ckpt_dir = ckpt_root / "super_agent"
    for d in (base_ckpt_dir, meta_ckpt_dir, super_ckpt_dir):
        d.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {}

    # ---------- BƯỚC 1 ----------
    base_agents: Dict[str, BaseAgent] = {}
    for agent_type in AGENT_TYPES:
        print(f"[train_staged] Bước 1: training base agent '{agent_type}' ...")
        agent = train_base_agent(agent_type, state_df, tickers, config)
        ckpt_path = base_ckpt_dir / f"{agent_type}.pt"
        torch.save(agent.state_dict(), ckpt_path)
        base_agents[agent_type] = agent
        results[f"base_agent_{agent_type}_ckpt"] = str(ckpt_path)

        local_rewards = getattr(agent, "_train_history", {}).get("reward_local_mean", [])
        if local_rewards:
            print(f"[train_staged]   reward cục bộ trung bình theo epoch ({agent_type}): {local_rewards}")

    # ---------- BƯỚC 2 ----------
    print("[train_staged] Bước 2: freeze Tầng 1, training MetaAggregator ...")
    meta = train_meta_aggregator(
        base_agents["fundamental"], base_agents["technical"], base_agents["sentiment"],
        state_df, tickers, config,
    )
    meta_ckpt_path = meta_ckpt_dir / "meta_aggregator.pt"
    torch.save(meta.state_dict(), meta_ckpt_path)
    results["meta_ckpt"] = str(meta_ckpt_path)

    # ---------- BƯỚC 3 ----------
    print("[train_staged] Bước 3: freeze Tầng 1+2, training Super-Agent (PPO) ...")
    for p in meta.parameters():
        p.requires_grad = False
    assert not any(p.requires_grad for p in meta.parameters()), (
        "MetaAggregator chưa được freeze đúng trước Bước 3 — kiểm tra lại logic freeze."
    )

    # Import trễ: PortfolioEnvPhaseB do 1 subagent khác viết song song.
    from src.envs.portfolio_env import PortfolioEnvPhaseB

    env_cfg = config.get("env", {})
    base_env = PortfolioEnvPhaseB(
        state_df,
        tickers,
        initial_cash=env_cfg.get("initial_cash", 1_000_000_000),
        transaction_cost=env_cfg.get("transaction_cost", 0.002),
        price_limit=env_cfg.get("price_limit", 0.07),
    )
    wrapped_env = MetaScoreObsWrapper(
        base_env,
        base_agents["fundamental"], base_agents["technical"], base_agents["sentiment"],
        meta, tickers,
    )

    try:
        super_agent = train_super_agent(meta, wrapped_env, config)
        super_ckpt_path = super_ckpt_dir / "super_agent.pt"
        torch.save(super_agent.state_dict(), super_ckpt_path)
        results["super_agent_ckpt"] = str(super_ckpt_path)
        results["super_agent"] = super_agent
    except Exception as e:
        print(
            f"[train_staged] Bước 3 THẤT BẠI ({e}) — checkpoint Bước 1-2 vẫn còn tại "
            f"{base_ckpt_dir} và {meta_ckpt_dir}, KHÔNG cần train lại khi retry Bước 3."
        )
        raise

    results["base_agents"] = base_agents
    results["meta_aggregator"] = meta
    return results


# ---------------------------------------------------------------------------
# CLI — smoke test với dữ liệu giả lập
# ---------------------------------------------------------------------------
def _make_smoke_test_state_df(tickers: List[str], n_days: int = 30, seed: int = 0) -> "pd.DataFrame":
    """Sinh `state_df` giả lập RẤT NHỎ cho `--smoke-test` — KHÔNG dùng dữ
    liệu thật. Columns: MultiIndex `(ticker, feature)` gồm đủ field mà cả 3
    prompt builder (mục 4.3: fundamental/technical/sentiment) cần, cộng với
    "close" để tính next-day-return.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    feature_names = [
        "close", "rsi_14", "macd", "macd_signal", "volatility_20d",
        "pe", "pb", "eps_growth", "recent_news",
    ]
    columns = pd.MultiIndex.from_product([tickers, feature_names])

    data: Dict[Any, Any] = {}
    for tk in tickers:
        price = 20.0 + np.cumsum(rng.normal(0, 0.2, size=n_days))
        price = np.clip(price, 1.0, None)
        data[(tk, "close")] = price
        data[(tk, "rsi_14")] = rng.uniform(20, 80, size=n_days)
        data[(tk, "macd")] = rng.normal(0, 1, size=n_days)
        data[(tk, "macd_signal")] = rng.normal(0, 1, size=n_days)
        data[(tk, "volatility_20d")] = rng.uniform(0.01, 0.05, size=n_days)
        data[(tk, "pe")] = rng.uniform(5, 25, size=n_days)
        data[(tk, "pb")] = rng.uniform(0.5, 4, size=n_days)
        data[(tk, "eps_growth")] = rng.uniform(-0.1, 0.3, size=n_days)
        data[(tk, "recent_news")] = ["Không có tin tức mới."] * n_days

    return pd.DataFrame(data, index=dates, columns=columns)


def _load_config(path: str) -> Dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Staged training cho kiến trúc 3 tầng Giai đoạn B (HIFA)."
    )
    parser.add_argument("--config", type=str, default="configs/phase_b.yaml")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Chạy pipeline với dữ liệu giả lập rất nhỏ (vài chục ngày, vài mã) "
            "+ vài bước huấn luyện để verify không lỗi runtime. KHÔNG dùng để "
            "train thật (train thật cần Qwen2.5-1.5B-Instruct qua mạng + "
            "GPU/dữ liệu 10 mã thật)."
        ),
    )
    args = parser.parse_args()

    config = _load_config(args.config)

    if args.smoke_test:
        all_tickers = config.get("data", {}).get("tickers", ["FPT", "VPB", "HPG"])
        tickers = list(all_tickers[:3])  # 3 mã đủ để verify pipeline, không cần cả 10
        state_df = _make_smoke_test_state_df(tickers, n_days=30)

        # Ghi đè hyperparameter cho nhanh (vài chục bước) — KHÔNG phải config
        # dùng để train thật.
        config = dict(config)
        config["training"] = dict(config.get("training", {}))
        config["training"]["stage1_base_agents"] = {"n_epochs": 1, "lr": 1e-4, "return_scale": 0.05}
        config["training"]["stage2_meta"] = {
            "lr": 1e-3, "epochs": 2, "return_scale": 0.05, "confidence_vol_window": 3,
        }
        config["training"]["stage3_super_agent"] = {
            "n_steps": 16, "n_epochs": 1, "batch_size": 4, "lr": 1e-4, "n_updates": 1,
        }
        config["checkpoints_dir"] = "checkpoints/phase_b_smoke_test"
        print(f"[smoke-test] tickers={tickers}, n_days={len(state_df)}")
    else:
        tickers = list(config["data"]["tickers"])
        processed_path = config["data"]["processed_path"]
        # Dùng load_state_dataframe() (KHÔNG pd.read_parquet trực tiếp) —
        # file processed lưu cột phẳng "TICKER__feature" (parquet không hỗ
        # trợ MultiIndex columns); load_state_dataframe khôi phục lại đúng
        # MultiIndex (ticker, feature) mà toàn bộ code bên dưới kỳ vọng
        # (xem TODO.md mục 1.1 — lỗi tích hợp tương tự đã sửa ở b4_ppo_no_llm.py).
        state_df = load_state_dataframe(processed_path)
        # Thị trường Mỹ: pipeline tin tức thật (Yahoo RSS/NewsAPI) sẽ bổ sung
        # ở Giai đoạn B/C. Hiện dùng placeholder để prompt sentiment agent có
        # đúng layout cột (ticker, 'recent_news').
        state_df = _attach_placeholder_news(state_df, tickers)

    results = train_staged(state_df, tickers, config)
    ckpt_root = config.get("checkpoints_dir", DEFAULT_CHECKPOINT_ROOT)
    print(f"[train_phase_b] Hoàn tất staged training. Checkpoint tại {ckpt_root}/.")
    return results


if __name__ == "__main__":
    main()
