"""src/agents/base_agent.py

Tầng 1 — Base Agent chuyên biệt (Giai đoạn B, mục 4.3 đặc tả
`HIFA_Implementation_Plan_GiaiDoan_A_B.md`).

Ở Giai đoạn A, `PortfolioPolicyLLM` (src/agents/super_agent_head.py) gộp cả
Tầng 1 + Tầng 3: 1 LLM duy nhất nhận toàn bộ state và xuất thẳng vector phân
bổ danh mục. Giai đoạn B tách Tầng 1 ra thành 3 `BaseAgent` CHUYÊN BIỆT
("fundamental", "technical", "sentiment"), mỗi agent chỉ đọc 1 loại thông tin
và xuất ra 1 SCORE liên tục trong [-1, 1] (điểm khuyến nghị mua/bán cho MỘT
mã) — không phải vector phân bổ. 3 score này sau đó được `MetaAggregator`
(Tầng 2, src/agents/meta_agent.py) tổng hợp lại thành 1 tín hiệu duy nhất cho
Super-Agent (Tầng 3) ra quyết định phân bổ cuối cùng.

`BaseAgent` kế thừa `PortfolioPolicyLLM` để tái sử dụng backbone Qwen +
LoRA + tokenizer đã cấu hình sẵn, chỉ thay `policy_head` (đầu ra 1 chiều
thay vì n_assets+1 chiều) và `forward()` (dùng `tanh` thay vì `softmax`, vì
đây là 1 score trong [-1, 1], không phải phân bổ có tổng = 1).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from src.agents.super_agent_head import PortfolioPolicyLLM

# 3 loại base agent chuyên biệt theo đúng mục 4.3 của đặc tả.
AGENT_TYPES: Tuple[str, ...] = ("fundamental", "technical", "sentiment")


class BaseAgent(PortfolioPolicyLLM):
    """Base Agent chuyên biệt cho Tầng 1 của kiến trúc 3 tầng (Giai đoạn B).

    So với `PortfolioPolicyLLM` gốc (Giai đoạn A):
        - `n_assets=1` khi gọi `super().__init__()` — không dùng để biểu diễn
          "1 tài sản" theo nghĩa danh mục, mà chỉ là cách tái sử dụng logic
          khởi tạo backbone/tokenizer/value_head của lớp cha; `policy_head`
          sinh ra bởi `super().__init__()` (shape `(hidden, n_assets+1)`) bị
          GHI ĐÈ ngay sau đó thành `nn.Linear(hidden_size, 1)` theo đúng yêu
          cầu "output là 1 điểm số, không phải vector phân bổ" của đặc tả.
        - `forward()` dùng `torch.tanh` (không phải `softmax`) để đảm bảo
          score nằm trong [-1, 1] — không cần ràng buộc tổng = 1 vì đây
          không phải phân bổ danh mục.
    """

    def __init__(self, base_model_name: str, agent_type: str, **kwargs) -> None:
        """
        Args:
            base_model_name: tên model HuggingFace dùng làm backbone (vd.
                "Qwen/Qwen2.5-1.5B-Instruct"), truyền thẳng cho lớp cha.
            agent_type: 1 trong 3 giá trị `AGENT_TYPES`
                ("fundamental" | "technical" | "sentiment").
            **kwargs: các tham số khác của `PortfolioPolicyLLM.__init__`
                (vd. lora_r, use_lora, `_test_backbone_override` cho test,
                ...) được truyền xuyên qua nguyên vẹn.
        """
        if agent_type not in AGENT_TYPES:
            raise ValueError(
                f"agent_type phải thuộc {AGENT_TYPES}, nhận được: {agent_type!r}"
            )

        # n_assets=1: output là 1 score, không phải vector phân bổ (xem
        # docstring class). policy_head tạm thời do lớp cha khởi tạo sẽ bị
        # ghi đè ngay bên dưới.
        super().__init__(base_model_name, n_assets=1, **kwargs)
        self.agent_type = agent_type

        hidden_size = self.backbone.config.hidden_size
        # Ghi đè policy_head: 1 điểm số duy nhất trong [-1, 1] (đúng mẫu 4.3),
        # thay vì Linear(hidden, n_assets+1) do PortfolioPolicyLLM tạo mặc định.
        self.policy_head = nn.Linear(hidden_size, 1)

    def forward(self, prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize `prompts` -> backbone (LoRA) -> hidden token cuối ->
        policy_head (tanh, [-1, 1]) + value_head.

        Trả về (score, value):
            score: FloatTensor (batch, 1), trong đoạn [-1, 1].
            value: FloatTensor (batch, 1).
        """
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_prompt_length,
        )
        device = self._get_device()
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self.backbone(**inputs)
        last_hidden = outputs.last_hidden_state[:, -1, :]  # hidden state token cuối cùng
        score = torch.tanh(self.policy_head(last_hidden))  # đảm bảo output trong [-1, 1]
        value = self.value_head(last_hidden)
        return score, value


# ---------------------------------------------------------------------------
# Prompt builders — ĐÚNG theo mẫu đặc tả mục 4.3.
# ---------------------------------------------------------------------------
def build_fundamental_prompt(ticker_data: Dict) -> str:
    """Fundamental-analysis prompt: P/E, P/B, YoY EPS growth (English, US market)."""
    return (
        f"Fundamental analysis for {ticker_data['ticker']}: P/E={ticker_data['pe']:.1f}, "
        f"P/B={ticker_data['pb']:.1f}, YoY earnings growth="
        f"{ticker_data['eps_growth'] * 100:.1f}%. "
        f"Provide a buy/sell recommendation score."
    )


def build_technical_prompt(ticker_data: Dict) -> str:
    """Technical-analysis prompt: RSI, MACD (bullish/bearish), 20-day volatility."""
    return (
        f"Technical analysis for {ticker_data['ticker']}: RSI={ticker_data['rsi_14']:.1f}, "
        f"MACD={'bullish' if ticker_data['macd'] > ticker_data['macd_signal'] else 'bearish'}, "
        f"20-day volatility={ticker_data['volatility_20d'] * 100:.1f}%. "
        f"Provide a buy/sell recommendation score."
    )


def build_sentiment_prompt(ticker_data: Dict) -> str:
    """Market-sentiment prompt based on recent news (English, US market)."""
    news = ticker_data.get("recent_news", "No recent news.")
    return (
        f"Recent news about {ticker_data['ticker']}: {news}\n"
        f"Assess the market sentiment (positive/negative)."
    )


_PROMPT_BUILDERS = {
    "fundamental": build_fundamental_prompt,
    "technical": build_technical_prompt,
    "sentiment": build_sentiment_prompt,
}


def build_prompt_for_agent(agent_type: str, ticker_data: Dict) -> str:
    """Dispatch tới đúng hàm `build_xxx_prompt()` theo `agent_type`.

    Raises:
        ValueError: nếu `agent_type` không thuộc `AGENT_TYPES`.
    """
    if agent_type not in _PROMPT_BUILDERS:
        raise ValueError(
            f"agent_type phải thuộc {tuple(_PROMPT_BUILDERS)}, nhận được: {agent_type!r}"
        )
    return _PROMPT_BUILDERS[agent_type](ticker_data)
