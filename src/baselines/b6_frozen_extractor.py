"""src/baselines/b6_frozen_extractor.py

Baseline B6 — biến thể kiểu HARLF (Giai đoạn B, mục 4.6 đặc tả
`HIFA_Implementation_Plan_GiaiDoan_A_B.md`).

SO SÁNH CỐT LÕI với `BaseAgent` (src/agents/base_agent.py, mục 4.3) — đây
chính là biến thiên dùng cho Ablation A:
    - `BaseAgent`: backbone LLM được chèn LoRA (`peft`) và fine-tune tiếp
      (supervised ở Bước 1, rồi gián tiếp qua staged training) — trọng số
      LoRA + `policy_head` + `value_head` CÓ cập nhật gradient.
    - `FrozenSentimentExtractor` (B6): backbone LLM pretrained ĐÓNG BĂNG
      HOÀN TOÀN — KHÔNG LoRA, KHÔNG fine-tune bất kỳ phần nào
      (`requires_grad=False` cho MỌI tham số của backbone), chỉ dùng để
      trích xuất hidden state TĨNH. Hidden state được map sang 1 score qua
      `self.fixed_projection` — 1 `nn.Linear` khởi tạo NGẪU NHIÊN nhưng CỐ
      ĐỊNH (seed cụ thể, KHÔNG BAO GIỜ được train) — mô phỏng đúng tinh thần
      "feature extractor tĩnh" kiểu FinBERT/HARLF (trích xuất nông, không
      fine-tune sâu).

Ablation A (mục 4.6, việc cần làm) = so sánh B6 (đóng băng hoàn toàn) vs
`BaseAgent` (fine-tune LoRA+PPO), giữ mọi thứ khác (Meta Aggregator,
Super-Agent) giống hệt nhau.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

try:  # pragma: no cover - phụ thuộc môi trường (nặng, cần mạng lần đầu)
    from transformers import AutoModel, AutoTokenizer
except ImportError:  # pragma: no cover
    AutoModel = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]


class FrozenSentimentExtractor:
    """Không train — chỉ dùng LLM pretrained để sinh sentiment/feature score
    TĨNH, tương đương FinBERT-style / HARLF-style feature extraction.

    KHÔNG kế thừa `nn.Module` (khác với `BaseAgent`) vì đây không phải 1
    thành phần có tham số huấn luyện được trong pipeline — toàn bộ state bên
    trong (`model`, `fixed_projection`) đều bị đóng băng ngay sau khi khởi
    tạo, dùng thuần cho suy luận (`get_score`).
    """

    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        projection_seed: int = 42,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
    ) -> None:
        """
        Args:
            base_model_name: tên model HuggingFace dùng làm feature
                extractor (mặc định giống backbone của `BaseAgent`, để so
                sánh công bằng ở Ablation A — cùng kiến trúc, chỉ khác cách
                huấn luyện).
            projection_seed: seed cố định cho `fixed_projection` — BẮT BUỘC
                cụ thể (không phải `None`/ngẫu nhiên mỗi lần) để tái lập được
                (reproducible) giữa các lần chạy baseline.
            model, tokenizer: cho phép TIÊM (dependency injection) 1
                backbone/tokenizer đã có sẵn — dùng cho unit test không cần
                mạng (xem `tests/test_phase_b_agents.py`) hoặc để tái sử
                dụng 1 model đã nạp sẵn ở nơi khác. Nếu bỏ trống (mặc định),
                sẽ tải `base_model_name` thật qua HuggingFace (cần mạng).
        """
        if model is not None:
            self.model = model
        else:
            if AutoModel is None:
                raise ImportError(
                    "Thư viện `transformers` chưa được cài trong môi trường này — "
                    "cần thiết để tải backbone thật cho FrozenSentimentExtractor. "
                    "Với unit test không cần mạng, truyền `model=<dummy backbone>` "
                    "qua tham số `model`."
                )
            self.model = AutoModel.from_pretrained(base_model_name)

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False  # đóng băng HOÀN TOÀN — đây là điểm khác biệt với BaseAgent ở 4.3

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            if AutoTokenizer is None:
                raise ImportError(
                    "Thư viện `transformers` chưa được cài trong môi trường này — "
                    "cần thiết để tải tokenizer thật cho FrozenSentimentExtractor. "
                    "Với unit test không cần mạng, truyền `tokenizer=<dummy tokenizer>` "
                    "qua tham số `tokenizer`."
                )
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        config = getattr(self.model, "config", None)
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError(
                "Không xác định được `hidden_size` từ `model.config.hidden_size` — "
                "nếu inject `model` giả cho test, hãy đảm bảo nó có `.config.hidden_size`."
            )

        # 1 linear layer NGẪU NHIÊN CỐ ĐỊNH (không train bao giờ) để map
        # hidden -> score, dùng generator riêng (seed cụ thể) để không ảnh
        # hưởng/lệ thuộc vào trạng thái RNG toàn cục và tái lập được.
        generator = torch.Generator().manual_seed(projection_seed)
        self.fixed_projection = nn.Linear(hidden_size, 1)
        with torch.no_grad():
            self.fixed_projection.weight.copy_(
                torch.randn(self.fixed_projection.weight.shape, generator=generator)
            )
            self.fixed_projection.bias.copy_(
                torch.randn(self.fixed_projection.bias.shape, generator=generator)
            )
        for p in self.fixed_projection.parameters():
            p.requires_grad = False  # cố định vĩnh viễn, không bao giờ train

    def get_score(self, prompt: str) -> float:
        """Sinh 1 score tĩnh trong [-1, 1] từ `prompt`, không update gradient.

        Pipeline: tokenize -> backbone ĐÓNG BĂNG (lấy hidden state token
        cuối) -> `fixed_projection` NGẪU NHIÊN CỐ ĐỊNH -> `tanh`.
        """
        with torch.no_grad():
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model(**inputs)
            hidden = outputs.last_hidden_state[:, -1, :]
            score = torch.tanh(self.fixed_projection(hidden))
            return float(score.item())
