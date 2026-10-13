"""src/agents/meta_agent.py

Tầng 2 — Meta Aggregation (Giai đoạn B, mục 4.4 đặc tả
`HIFA_Implementation_Plan_GiaiDoan_A_B.md`).

`MetaAggregator` là 1 MẠNG NHỎ THÔNG THƯỜNG (`nn.Sequential` gồm vài lớp
`Linear`+`ReLU`) — KHÔNG phải LLM. Nó nhận (score, confidence) từ 3 base
agent chuyên biệt (fundamental/technical/sentiment, xem
`src/agents/base_agent.py`) và tổng hợp thành 1 `meta_score` duy nhất trong
[-1, 1] + 1 `value` (dùng làm baseline cho PPO ở Tầng 3, Super-Agent).

`generate_explanation()` là 1 nhánh PHỤ, tách biệt khỏi luồng huấn luyện: chỉ
dùng model sentiment (đã fine-tune LoRA ở Tầng 1) ở chế độ suy luận thuần
(`eval()` + `torch.no_grad()`) để sinh 1 đoạn văn giải thích quyết định đầu
tư — không tham gia backprop.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class MetaAggregator(nn.Module):
    """Mạng nhỏ (KHÔNG phải LLM) — nhận 3 score từ base agent + độ tin cậy
    động, xuất 1 score tổng hợp (`meta_score`) trong [-1, 1] + 1 `value`.

    Kiến trúc: `Linear(n_sources*2, hidden) -> ReLU -> Linear(hidden, hidden)
    -> ReLU -> Linear(hidden, 1)`. `value_head` là 1 `Linear(hidden, 1)`
    RIÊNG, lấy input từ hidden layer TRƯỚC layer cuối của `self.net` (tức
    `self.net[:-1](x)`) — đúng theo mẫu đặc tả mục 4.4, để `value` chia sẻ
    biểu diễn ẩn với `meta_score` nhưng có đầu ra độc lập (cần cho PPO baseline
    ở Tầng 3 sau này).
    """

    def __init__(self, n_sources: int = 3, hidden: int = 32) -> None:
        super().__init__()
        self.n_sources = n_sources
        self.net = nn.Sequential(
            nn.Linear(n_sources * 2, hidden),  # x2 vì mỗi source có (score, confidence)
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.value_head = nn.Linear(hidden, 1)  # value head riêng cho PPO ở tầng này

    def compute_confidence(self, recent_volatility_of_source: float) -> float:
        """Trọng số tin cậy động theo độ biến động gần đây của từng nguồn tín
        hiệu — càng ổn định (biến động thấp) càng tin cậy.

        confidence = 1 / (1 + volatility). volatility=0 -> confidence=1.0
        (tối đa); volatility càng lớn -> confidence càng tiến về 0 (không bao
        giờ âm/vượt 1, vì volatility >= 0).
        """
        return 1.0 / (1.0 + recent_volatility_of_source)

    def forward(
        self, scores: torch.Tensor, confidences: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            scores: FloatTensor (batch, n_sources) — score của từng base
                agent (fundamental/technical/sentiment), mỗi giá trị trong
                [-1, 1].
            confidences: FloatTensor (batch, n_sources) — độ tin cậy động
                tương ứng (xem `compute_confidence`), cùng shape `scores`.

        Returns:
            (meta_score, value): meta_score shape (batch, 1) trong [-1, 1];
            value shape (batch, 1).
        """
        x = torch.cat([scores, confidences], dim=-1)
        h = self.net[:-1](x)  # lấy hidden trước layer cuối để dùng cho value head
        meta_score = torch.tanh(self.net[-1](h))
        value = self.value_head(h)
        return meta_score, value


def generate_explanation(sentiment_llm, base_scores: Dict[str, float], ticker: str) -> str:
    """Nhánh giải thích — CHỈ suy luận (inference), KHÔNG cập nhật gradient.

    Dùng `sentiment_llm` (kỳ vọng là 1 `BaseAgent`/`PortfolioPolicyLLM` đã
    fine-tune, có `.tokenizer` + `.backbone` kiểu HuggingFace với method
    `.generate()`) ở chế độ `eval()` + `torch.no_grad()` để sinh 1 đoạn giải
    thích ngắn gọn cho quyết định đầu tư, tổng hợp từ 3 score base agent.

    LƯU Ý: hàm này CẦN model thật (có `.generate()` khả dụng) để chạy — với
    backbone giả dùng trong unit test (không hỗ trợ sinh văn bản tự do), hàm
    sẽ raise `RuntimeError` với thông báo rõ ràng thay vì lỗi mơ hồ kiểu
    `AttributeError: 'X' object has no attribute 'generate'`.

    Raises:
        ValueError: nếu `base_scores` thiếu 1 trong 3 khóa bắt buộc.
        RuntimeError: nếu `sentiment_llm` không hỗ trợ sinh văn bản
            (không có `.backbone.generate()` khả dụng) hoặc quá trình sinh
            văn bản thất bại vì bất kỳ lý do gì khác.
    """
    required_keys = ("fundamental", "technical", "sentiment")
    missing = [k for k in required_keys if k not in base_scores]
    if missing:
        raise ValueError(f"base_scores thiếu khóa bắt buộc: {missing}")

    prompt = (
        f"Three analysts evaluated {ticker} as follows: "
        f"Fundamental analysis: {base_scores['fundamental']:.2f}. "
        f"Technical analysis: {base_scores['technical']:.2f}. "
        f"Sentiment analysis: {base_scores['sentiment']:.2f}. "
        f"Synthesize these into a brief explanation for the investment decision."
    )

    try:
        backbone = sentiment_llm.backbone
        tokenizer = sentiment_llm.tokenizer
        if not hasattr(backbone, "generate") or not callable(getattr(backbone, "generate")):
            raise AttributeError(
                "`sentiment_llm.backbone` không có method .generate() khả dụng"
            )

        backbone.eval()  # chuyển sang chế độ eval TRƯỚC khi generate — không train ở đây
        with torch.no_grad():
            # dùng generate() thông thường của HuggingFace, không qua policy_head
            if hasattr(sentiment_llm, "_get_device"):
                device = sentiment_llm._get_device()
            else:
                device = getattr(backbone, "device", "cpu")
            encoded = tokenizer(prompt, return_tensors="pt")
            encoded = {k: v.to(device) for k, v in encoded.items()}
            output_ids = backbone.generate(**encoded, max_new_tokens=100)
            explanation = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return explanation
    except AttributeError as e:
        raise RuntimeError(
            "generate_explanation() yêu cầu `sentiment_llm.backbone` là 1 model "
            "HuggingFace THẬT có method .generate() (vd backbone Qwen đã nạp qua "
            "PortfolioPolicyLLM/BaseAgent với model thật, KHÔNG dùng "
            "`_test_backbone_override`/dummy backbone dùng cho unit test — dummy "
            "backbone không hỗ trợ sinh văn bản tự do). "
            f"Chi tiết lỗi gốc: {e}"
        ) from e
    except Exception as e:  # noqa: BLE001 - bọc mọi lỗi runtime khác (device/model) thành thông báo rõ ràng
        raise RuntimeError(
            f"generate_explanation() thất bại khi sinh giải thích cho mã {ticker}: {e}"
        ) from e
