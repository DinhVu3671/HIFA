"""src/agents/super_agent_head.py

Kiến trúc LLM-policy cho Giai đoạn A (mục 3.4 đặc tả
`HIFA_Implementation_Plan_GiaiDoan_A_B.md`) — `PortfolioPolicyLLM`.

QUYẾT ĐỊNH THIẾT KẾ CỐT LÕI (đã chốt theo khuyến nghị mục 3.4 của đặc tả):
state được biểu diễn dưới dạng TEXT PROMPT tiếng Việt (LLM đọc mô tả bằng
ngôn ngữ tự nhiên qua `state_to_prompt()`), KHÔNG dùng phương án "embedding
số học nối trực tiếp" (feed thẳng vector số vào 1 lớp projection trước khi
vào transformer, bỏ qua tokenize). Lý do (trích đặc tả 3.4): đây chính là
điểm khác biệt cốt lõi cần chứng minh ở RQ3 (đóng góp của "language
understanding" so với 1 MLP/embedding số học thuần) — nếu dùng embedding số
học ngay từ đầu, Giai đoạn A sẽ không tách bạch được đóng góp của phần ngôn
ngữ so với baseline B4 (PPO thuần, không LLM). Cái giá phải trả là tốc độ
(phải tokenize text mỗi step) và độ chính xác biểu diễn số qua text — chấp
nhận được cho MVP Giai đoạn A theo đúng đánh giá của đặc tả.

KIỂM THỬ MÀ KHÔNG CẦN TẢI QWEN2.5-1.5B THẬT:
`Qwen/Qwen2.5-1.5B-Instruct` nặng (~3GB weights bf16) và cần mạng để tải lần
đầu — không khả thi để unit-test tự động trong CI/sandbox. Class được thiết
kế để `base_model_name` có thể thay bằng một model nhỏ công khai trên HF Hub
(vd. "hf-internal-testing/tiny-random-gpt2", vài MB) nếu môi trường CÓ mạng.
Nếu môi trường KHÔNG có mạng (trường hợp của sandbox hiện tại), dùng tham số
nội bộ `_test_backbone_override` (+ `_test_tokenizer_override` tuỳ chọn) để
tiêm vào 1 backbone giả `_DummyBackboneForTest` (torch.nn thuần, không phụ
thuộc `transformers`/mạng) — cho phép test toàn bộ luồng dữ liệu qua
policy_head/value_head. Hai tham số này CHỈ dùng cho test, KHÔNG dùng trong
production (xem `tests/test_ppo_trainer.py`).
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

# transformers/peft là dependency nặng và cần mạng để tải weights thật lần
# đầu — import mềm (try/except) để module này (và mọi thứ import nó, vd.
# tests) vẫn load được trong môi trường chưa cài đủ 2 thư viện này, miễn là
# chỉ dùng path test (`_test_backbone_override`).
try:  # pragma: no cover - phụ thuộc môi trường
    from transformers import AutoModel, AutoTokenizer
except ImportError:  # pragma: no cover
    AutoModel = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]

try:  # pragma: no cover - phụ thuộc môi trường
    from peft import LoraConfig, get_peft_model
except ImportError:  # pragma: no cover
    LoraConfig = None  # type: ignore[assignment]
    get_peft_model = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Backbone/tokenizer giả lập — CHỈ dùng cho unit test (mục "Kiểm thử" ở trên).
# ---------------------------------------------------------------------------
class _DummyBackboneForTest(nn.Module):
    """Backbone transformer giả lập, torch.nn thuần, KHÔNG cần `transformers`
    hay mạng — dùng để unit-test kiến trúc `PortfolioPolicyLLM` (luồng
    tokenize -> backbone -> policy_head/value_head) mà không cần tải
    Qwen2.5-1.5B thật.

    Có 1 khối self-attention đơn giản với các submodule được ĐẶT TÊN TRÙNG
    với `target_modules` dùng trong LoRA production (`q_proj`, `k_proj`,
    `v_proj`, `o_proj`) — nhờ vậy `peft.get_peft_model` có thể gắn LoRA
    adapter thật lên backbone giả này (không cần mạng, `peft` chỉ thao tác
    trên các `nn.Module` cục bộ), cho phép test "chỉ LoRA + head có
    requires_grad=True" bằng cơ chế thật thay vì giả lập.

    KHÔNG dùng trong production — chỉ được tiêm vào qua
    `PortfolioPolicyLLM(_test_backbone_override=...)`.
    """

    class _Config:
        def __init__(self, hidden_size: int, vocab_size: int) -> None:
            self.hidden_size = hidden_size
            self.vocab_size = vocab_size
            self.model_type = "dummy-for-test"

    class _Output:
        def __init__(self, last_hidden_state: torch.Tensor) -> None:
            self.last_hidden_state = last_hidden_state

    def __init__(self, vocab_size: int = 1000, hidden_size: int = 32) -> None:
        super().__init__()
        self.config = self._Config(hidden_size=hidden_size, vocab_size=vocab_size)
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        # Đặt tên trùng target_modules LoRA production để peft có thể gắn adapter.
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> "_DummyBackboneForTest._Output":
        x = self.embed_tokens(input_ids)
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        scale = x.shape[-1] ** 0.5
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / scale
        if attention_mask is not None:
            # padding mask: chặn attend vào vị trí pad (giống hành vi backbone thật)
            pad_mask = (1.0 - attention_mask[:, None, :].to(attn_scores.dtype)) * -1e9
            attn_scores = attn_scores + pad_mask
        attn = torch.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn, v)
        hidden = torch.tanh(self.o_proj(out))
        return self._Output(last_hidden_state=hidden)

    def gradient_checkpointing_enable(self, **kwargs: Any) -> None:
        """No-op — chỉ để tương thích interface với backbone HF thật, cho
        phép `SimplePPOTrainer` gọi `hasattr(backbone, "gradient_checkpointing_enable")`
        mà không phân biệt dummy hay thật."""
        return None


class _DummyTokenizerForTest:
    """Tokenizer giả lập (băm từ theo whitespace -> id) — KHÔNG cần tải
    vocab/config từ HF Hub. Chỉ dùng cùng `_DummyBackboneForTest` cho test.
    """

    def __init__(self, vocab_size: int = 1000, pad_token_id: int = 0) -> None:
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id

    def __call__(
        self,
        texts: Any,
        return_tensors: str = "pt",
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 512,
    ) -> Dict[str, torch.Tensor]:
        if isinstance(texts, str):
            texts = [texts]
        token_lists: List[List[int]] = []
        for text in texts:
            tokens = text.split() or ["<empty>"]
            ids = [(hash(tok) % (self.vocab_size - 1)) + 1 for tok in tokens]
            if truncation:
                ids = ids[:max_length]
            token_lists.append(ids)
        max_len = max((len(ids) for ids in token_lists), default=1)
        if not padding:
            max_len = max_len  # vẫn cần độ dài đồng nhất để stack thành tensor
        input_ids: List[List[int]] = []
        attention_mask: List[List[int]] = []
        for ids in token_lists:
            pad_len = max_len - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Kiến trúc chính
# ---------------------------------------------------------------------------
class PortfolioPolicyLLM(nn.Module):
    """LLM policy (gộp Tầng 1 + Tầng 3) cho Giai đoạn A — theo ĐÚNG code mẫu
    mục 3.4 của đặc tả, có value_head bắt buộc cho PPO.

    Backbone (Qwen2.5-1.5B-Instruct hoặc model nhỏ tương đương) được đóng
    băng hoàn toàn; chỉ adapter LoRA (r=16, alpha=32, dropout=0.05, gắn vào
    q_proj/k_proj/v_proj/o_proj) + `policy_head` + `value_head` là trainable
    — số lượng tham số train được rất nhỏ so với backbone (xem
    `train_phase_a.py`, in ra `sum(p.numel() for p in model.parameters() if
    p.requires_grad)` để confirm).
    """

    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        n_assets: int = 5,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Sequence[str] = ("q_proj", "v_proj", "k_proj", "o_proj"),
        max_prompt_length: int = 512,
        use_lora: bool = True,
        torch_dtype: "torch.dtype | None" = torch.bfloat16,
        device: "str | torch.device | None" = None,
        # --- CHỈ DÙNG CHO UNIT TEST (xem docstring module) — KHÔNG dùng ở production ---
        _test_backbone_override: Optional[nn.Module] = None,
        _test_tokenizer_override: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.n_assets = n_assets
        self.max_prompt_length = max_prompt_length
        self.base_model_name = base_model_name
        self._is_test_backbone = _test_backbone_override is not None
        # device: None -> tự dò ("cuda" nếu có, else "cpu"). Trên Colab GPU,
        # đặt None là đủ để model tự lên GPU. Truyền "cpu" để ép CPU (dev/test).
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if _test_backbone_override is not None:
            # --- Path unit-test: không chạm mạng, không tải pretrained weights ---
            backbone: nn.Module = _test_backbone_override
            vocab_size = getattr(backbone.config, "vocab_size", 1000)
            self.tokenizer = (
                _test_tokenizer_override
                if _test_tokenizer_override is not None
                else _DummyTokenizerForTest(vocab_size=vocab_size)
            )
        else:
            # --- Path production: Qwen2.5-1.5B-Instruct (hoặc model tiny công khai
            # khác truyền qua base_model_name, vd. "hf-internal-testing/tiny-random-gpt2"
            # nếu môi trường có mạng nhưng muốn test nhẹ) ---
            if AutoTokenizer is None or AutoModel is None:
                raise ImportError(
                    "Thư viện `transformers` chưa được cài trong môi trường này. "
                    "Cần `pip install transformers` để dùng PortfolioPolicyLLM ở "
                    "production (base_model_name thật). Với unit test không cần "
                    "mạng, dùng tham số `_test_backbone_override` "
                    "(xem `_DummyBackboneForTest`)."
                )
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            if getattr(self.tokenizer, "pad_token", None) is None:
                # Nhiều causal-LM tokenizer (kể cả Qwen2.5) không có pad_token mặc
                # định — cần gán để `padding=True` trong forward() không lỗi.
                self.tokenizer.pad_token = self.tokenizer.eos_token
            backbone = AutoModel.from_pretrained(base_model_name, torch_dtype=torch_dtype)
            # Đưa backbone lên GPU TRƯỚC khi wrap LoRA — an toàn hơn gọi
            # `self.to(device)` trên PeftModel (một số phiên bản peft/torch xử
            # lý `.to()` trên toàn PeftModel không nhất quán). Không nuốt lỗi:
            # nếu CUDA báo available mà đây lỗi, ta muốn biết ngay chứ KHÔNG
            # âm thầm rớt về CPU (chạy chậm gấp ~10 lần suốt nhiều giờ).
            if not self._is_test_backbone and self.device.type == "cuda":
                backbone = backbone.to(self.device)

        if use_lora:
            if LoraConfig is None or get_peft_model is None:
                raise ImportError(
                    "Thư viện `peft` chưa được cài trong môi trường này — cần "
                    "thiết để gắn LoRA adapter lên backbone (production hoặc "
                    "test với `_DummyBackboneForTest`). Với test không cần LoRA, "
                    "truyền `use_lora=False`."
                )
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=list(lora_target_modules),
                # FEATURE_EXTRACTION vì ta dùng backbone như 1 bộ trích xuất
                # hidden state (không dùng LM head sinh text) — đúng theo mẫu đặc tả 3.4.
                task_type="FEATURE_EXTRACTION",
            )
            backbone = get_peft_model(backbone, lora_config)

        self.backbone = backbone

        hidden_size = self.backbone.config.hidden_size
        self.policy_head = nn.Linear(hidden_size, n_assets + 1)  # +1 cho cash
        # BẮT BUỘC có value_head cho PPO — bản thiết kế trước thiếu baseline này.
        self.value_head = nn.Linear(hidden_size, 1)

        # Chuyển toàn bộ (backbone đã lên GPU ở trên + 2 head float32) lên
        # device đã chọn. Backbone giả (unit test) giữ nguyên trên CPU.
        #
        # KHÔNG nuốt lỗi device nữa: bản cũ bọc try/except rồi âm thầm rớt về
        # CPU — hệ quả là model chạy CPU chậm gấp ~10 lần suốt nhiều giờ mà
        # người dùng không hề biết (warning bị chôn trong log). Nếu CUDA báo
        # available mà không đưa model lên GPU được, ta RAISE để dừng ngay.
        if not self._is_test_backbone and self.device.type == "cuda":
            # Backbone đã được .to('cuda') TRƯỚC khi wrap LoRA (ở trên); LoRA
            # adapter được peft tạo trên cùng device với backbone. Chỉ còn 2
            # head float32 cần đưa lên GPU riêng — tránh gọi .to() trên toàn
            # PeftModel wrapper (nguồn gây lỗi device trong bản cũ).
            self.policy_head = self.policy_head.to(self.device)
            self.value_head = self.value_head.to(self.device)
            # Phòng hờ: ép mọi param còn sót trên CPU (vd. LoRA) lên GPU.
            self.to(self.device)
            # Xác minh THẬT params đã nằm trên GPU (không tin suông is_available).
            actual = next(self.parameters()).device
            if actual.type != "cuda":
                raise RuntimeError(
                    f"torch.cuda.is_available()=True nhưng model vẫn ở {actual} "
                    "sau khi .to('cuda'). Training trên CPU sẽ chậm gấp ~10 lần — "
                    "dừng để bạn kiểm tra runtime GPU của Colab (Runtime → Change "
                    "runtime type → GPU) thay vì âm thầm chạy CPU."
                )
            print(f"[PortfolioPolicyLLM] Model đã lên GPU: {torch.cuda.get_device_name(0)} ({actual}).")
        elif not self._is_test_backbone:
            print(f"[PortfolioPolicyLLM] Chạy trên CPU (device={self.device}) — "
                  "CẢNH BÁO: training LLM trên CPU rất chậm.")

    # ------------------------------------------------------------------
    def _get_device(self) -> torch.device:
        try:
            return next(self.backbone.parameters()).device
        except StopIteration:  # pragma: no cover - backbone không có param
            return torch.device("cpu")

    def gradient_checkpointing_enable(self, **kwargs: Any) -> None:
        """Passthrough tiện dụng — `SimplePPOTrainer` cũng tự kiểm tra
        `hasattr(policy.backbone, "gradient_checkpointing_enable")` trực tiếp,
        method này chỉ để gọi qua `policy.gradient_checkpointing_enable()`."""
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable(**kwargs)
        else:  # pragma: no cover
            warnings.warn(
                "Backbone hiện tại không hỗ trợ gradient_checkpointing_enable() — bỏ qua."
            )

    # ------------------------------------------------------------------
    def state_to_prompt(self, obs_dict: Dict[str, Dict[str, float]]) -> str:
        """Chuyển state numeric (+ tin tức, nếu có ở Giai đoạn B) thành 1
        prompt text TIẾNG ANH cho LLM (thị trường Mỹ, De-Cuong §6.1).

        `obs_dict`: {ticker: {"close":.., "rsi_14":.., "macd":.., "macd_signal":..,
        "volatility_20d":..}}. Dùng cùng `obs_array_to_dict()` trong
        `src/training/ppo_trainer.py` để chuyển ngược numeric obs array (từ
        `PortfolioEnvPhaseA`) sang dict này trước khi gọi hàm.

        LƯU Ý: khi obs_dict được dựng từ `PortfolioEnvPhaseA._get_obs()` (qua
        `obs_array_to_dict`), trường "close" mặc định là giá ĐÃ CHUẨN HÓA
        (close / close ngày đầu episode) theo quyết định thiết kế của env
        (xem `src/envs/portfolio_env.py`), không phải giá USD thô — trừ khi
        gọi `obs_array_to_dict(..., base_close=env._base_close)` để khôi phục
        giá thật. Hàm này KHÔNG tự phân biệt 2 trường hợp, chỉ hiển thị đúng
        giá trị số nhận được.
        """
        lines = ["Current market data:"]
        for tk, feat in obs_dict.items():
            lines.append(
                f"- {tk}: price={feat['close']:.2f}, RSI={feat['rsi_14']:.1f}, "
                f"MACD={'bullish' if feat['macd'] > feat['macd_signal'] else 'bearish'}, "
                f"20-day volatility={feat['volatility_20d'] * 100:.1f}%"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def forward(self, prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize `prompts` -> forward qua backbone (LoRA) -> lấy hidden
        state token cuối -> policy_head (softmax, tổng=1) + value_head.

        Trả về (weights, value):
            weights: FloatTensor (batch, n_assets+1), mỗi hàng sum=1.
            value:   FloatTensor (batch, 1).
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
        # BUG FIX (PPO KL nổ với batch > 1): KHÔNG dùng [:, -1, :] — với
        # padding=True và Qwen pad BÊN PHẢI, vị trí -1 của các prompt ngắn hơn
        # prompt dài nhất trong batch là TOKEN PAD (hidden vô nghĩa). Vì lượng
        # pad phụ thuộc thành phần batch, cùng 1 prompt ở 2 batch khác nhau cho
        # ra hidden khác nhau -> weights khác -> log_prob khác -> ratio != 1 ->
        # approx_kl ~2.5, clip_frac ~0.83 ngay update đầu dù policy chưa học.
        # Cách đúng: lấy hidden state của TOKEN THẬT CUỐI CÙNG mỗi prompt qua
        # attention_mask (bất biến với padding/thành phần batch).
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden)
        attn_mask = inputs.get("attention_mask")
        if attn_mask is not None:
            # index token thật cuối = số token thật - 1 (mask: 1=thật, 0=pad)
            last_idx = attn_mask.sum(dim=1) - 1  # (batch,)
            last_idx = last_idx.clamp_min(0).long()
            batch_idx = torch.arange(hidden_states.shape[0], device=hidden_states.device)
            last_hidden = hidden_states[batch_idx, last_idx, :]
        else:
            # Không có mask (vd. backbone test) -> fallback token cuối tensor.
            last_hidden = hidden_states[:, -1, :]
        # policy_head/value_head là nn.Linear float32 (mặc định), trong khi
        # backbone có thể chạy ở bfloat16 (torch_dtype) — ép về float32 trước
        # khi vào head để tránh lệch dtype khi nhân ma trận (heads rất nhỏ,
        # không đáng kể về bộ nhớ/tốc độ).
        last_hidden = last_hidden.float()
        logits = self.policy_head(last_hidden)
        weights = torch.softmax(logits, dim=-1)  # đảm bảo sum = 1
        value = self.value_head(last_hidden)
        return weights, value
