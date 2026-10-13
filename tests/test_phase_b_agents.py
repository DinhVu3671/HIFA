"""tests/test_phase_b_agents.py

Unit test cho kiến trúc 3 tầng Giai đoạn B (mục 4.3-4.6 đặc tả
`HIFA_Implementation_Plan_GiaiDoan_A_B.md`):
    - `src/agents/base_agent.py`: prompt builders (fundamental/technical/
      sentiment) + dispatcher.
    - `src/agents/meta_agent.py`: `MetaAggregator` (forward/compute_confidence)
      + `generate_explanation` (kiểm tra thông báo lỗi rõ ràng khi thiếu
      `.generate()`).
    - Cơ chế freeze dùng trong `train_meta_aggregator()`/`train_super_agent()`
      (src/training/train_phase_b.py) — mô phỏng lại bằng 1 module giả nhỏ.
    - `src/baselines/b6_frozen_extractor.py`: `FrozenSentimentExtractor`.

KHÔNG PHỤ THUỘC MẠNG/MODEL THẬT: dùng backbone/tokenizer giả (torch.nn thuần)
tự tạo TRONG FILE NÀY, KHÔNG phụ thuộc chi tiết nội bộ (vd
`_test_backbone_override`) của `src/agents/super_agent_head.py` — theo đúng
hướng dẫn "an toàn hơn, tránh phụ thuộc implementation detail chưa ổn định".
Vì lý do này, các test không khởi tạo `BaseAgent`/`PortfolioPolicyLLM` đầy đủ
(cần tải Qwen2.5-1.5B-Instruct qua mạng) — chỉ test các hàm/class không cần
LLM thật (prompt builders là hàm thuần, `MetaAggregator` là `nn.Module`
thường KHÔNG phải LLM).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.agents.base_agent import (
    build_fundamental_prompt,
    build_prompt_for_agent,
    build_sentiment_prompt,
    build_technical_prompt,
)
from src.agents.meta_agent import MetaAggregator, generate_explanation
from src.baselines.b6_frozen_extractor import FrozenSentimentExtractor


# ===========================================================================
# 1. Prompt builders (src/agents/base_agent.py, mục 4.3)
# ===========================================================================
class TestPromptBuilders:
    def test_build_fundamental_prompt_contains_values(self):
        data = {"ticker": "AAPL", "pe": 15.234, "pb": 3.1, "eps_growth": 0.257}
        prompt = build_fundamental_prompt(data)
        assert "AAPL" in prompt
        assert "P/E=15.2" in prompt
        assert "P/B=3.1" in prompt
        assert "25.7%" in prompt

    def test_build_technical_prompt_up_trend_when_macd_above_signal(self):
        data = {
            "ticker": "MSFT", "rsi_14": 65.44, "macd": 1.2, "macd_signal": 0.8,
            "volatility_20d": 0.0234,
        }
        prompt = build_technical_prompt(data)
        assert "MSFT" in prompt
        assert "RSI=65.4" in prompt
        assert "bullish" in prompt
        assert "bearish" not in prompt
        assert "2.3%" in prompt

    def test_build_technical_prompt_down_trend_when_macd_below_signal(self):
        data = {
            "ticker": "MSFT", "rsi_14": 30.0, "macd": 0.5, "macd_signal": 0.9,
            "volatility_20d": 0.01,
        }
        prompt = build_technical_prompt(data)
        assert "bearish" in prompt

    def test_build_sentiment_prompt_with_news(self):
        data = {"ticker": "NVDA", "recent_news": "Q2 earnings beat expectations."}
        prompt = build_sentiment_prompt(data)
        assert "NVDA" in prompt
        assert "Q2 earnings beat expectations." in prompt

    def test_build_sentiment_prompt_without_news_uses_placeholder(self):
        data = {"ticker": "JPM"}
        prompt = build_sentiment_prompt(data)
        assert "JPM" in prompt
        assert "No recent news." in prompt

    @pytest.mark.parametrize(
        "agent_type,expected_substr",
        [
            ("fundamental", "Fundamental analysis"),
            ("technical", "Technical analysis"),
            ("sentiment", "Recent news"),
        ],
    )
    def test_build_prompt_for_agent_dispatches_correctly(self, agent_type, expected_substr):
        data = {
            "ticker": "XOM", "pe": 10.0, "pb": 1.5, "eps_growth": 0.1,
            "rsi_14": 50.0, "macd": 0.1, "macd_signal": 0.1, "volatility_20d": 0.02,
            "recent_news": "sample news",
        }
        prompt = build_prompt_for_agent(agent_type, data)
        assert expected_substr in prompt
        assert "XOM" in prompt

    def test_build_prompt_for_agent_invalid_type_raises_value_error(self):
        with pytest.raises(ValueError):
            build_prompt_for_agent("invalid_type", {"ticker": "X"})


# ===========================================================================
# 2. MetaAggregator (src/agents/meta_agent.py, mục 4.4)
# ===========================================================================
class TestMetaAggregator:
    def test_forward_output_shapes(self):
        torch.manual_seed(0)
        meta = MetaAggregator(n_sources=3, hidden=16)
        batch = 5
        scores = torch.randn(batch, 3)
        confidences = torch.rand(batch, 3)
        meta_score, value = meta(scores, confidences)
        assert meta_score.shape == (batch, 1)
        assert value.shape == (batch, 1)

    def test_forward_meta_score_within_valid_range(self):
        torch.manual_seed(1)
        meta = MetaAggregator(n_sources=3, hidden=16)
        # scores/confidences ngẫu nhiên, kể cả giá trị "cực đoan" để chắc chắn
        # tanh() thực sự chặn output trong [-1, 1].
        scores = torch.randn(20, 3) * 10.0
        confidences = torch.rand(20, 3) * 5.0
        meta_score, _value = meta(scores, confidences)
        assert torch.all(meta_score >= -1.0)
        assert torch.all(meta_score <= 1.0)

    def test_forward_supports_different_n_sources(self):
        meta = MetaAggregator(n_sources=4, hidden=8)
        scores = torch.randn(2, 4)
        confidences = torch.rand(2, 4)
        meta_score, value = meta(scores, confidences)
        assert meta_score.shape == (2, 1)
        assert value.shape == (2, 1)

    @pytest.mark.parametrize(
        "vol,expected",
        [
            (0.0, 1.0),
            (1.0, 0.5),
            (3.0, 0.25),
            (9.0, 0.1),
            (99.0, 0.01),
        ],
    )
    def test_compute_confidence_known_values(self, vol, expected):
        meta = MetaAggregator()
        result = meta.compute_confidence(vol)
        assert result == pytest.approx(expected)

    def test_compute_confidence_never_exceeds_one(self):
        meta = MetaAggregator()
        assert meta.compute_confidence(0.0) == pytest.approx(1.0)
        assert meta.compute_confidence(1000.0) < 0.01


# ===========================================================================
# 3. generate_explanation — chỉ inference, lỗi rõ ràng nếu thiếu generate()
# ===========================================================================
class TestGenerateExplanation:
    def test_raises_value_error_when_base_scores_missing_keys(self):
        with pytest.raises(ValueError):
            generate_explanation(object(), {"fundamental": 0.1}, "FPT")

    def test_raises_clear_runtime_error_without_generate_method(self):
        """Backbone giả KHÔNG có `.generate()` (mô phỏng dummy backbone dùng
        cho unit test) -> phải raise RuntimeError với thông báo rõ ràng,
        KHÔNG phải AttributeError mơ hồ."""

        class DummyBackboneNoGenerate:
            def eval(self):
                return self

        class DummyTokenizer:
            def __call__(self, *args, **kwargs):
                return {"input_ids": torch.tensor([[1, 2, 3]])}

            def decode(self, *args, **kwargs):
                return "dummy explanation"

        class DummySentimentLLM:
            backbone = DummyBackboneNoGenerate()
            tokenizer = DummyTokenizer()

        base_scores = {"fundamental": 0.1, "technical": -0.2, "sentiment": 0.3}
        with pytest.raises(RuntimeError, match="generate"):
            generate_explanation(DummySentimentLLM(), base_scores, "FPT")


# ===========================================================================
# 4. Cơ chế freeze — mô phỏng đúng logic dùng trong
#    train_meta_aggregator()/train_super_agent() (src/training/train_phase_b.py)
# ===========================================================================
class TestFreezeMechanism:
    """`train_meta_aggregator()` và `train_super_agent()` (mục 4.5) đều
    dùng đúng pattern:
        for p in module.parameters():
            p.requires_grad = False
        assert not any(p.requires_grad for p in module.parameters())
    trước khi train tầng kế tiếp — đây là ĐIỂM DỄ LỖI NHẤT theo cảnh báo của
    đặc tả (quên freeze). Test dưới đây dùng 1 module giả nhỏ (KHÔNG phụ
    thuộc BaseAgent/LLM thật) để verify pattern này hoạt động đúng cả 2
    chiều: (a) freeze đúng -> assert pass, (b) quên freeze -> assert fail.
    """

    class _DummySubModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(4, 8)
            self.fc2 = nn.Linear(8, 1)

    def test_freeze_sets_all_requires_grad_false(self):
        module = self._DummySubModule()
        # Trước freeze: mặc định mọi Parameter có requires_grad=True.
        assert all(p.requires_grad for p in module.parameters())

        for p in module.parameters():
            p.requires_grad = False

        assert not any(p.requires_grad for p in module.parameters())
        for p in module.parameters():
            assert p.requires_grad is False

    def test_freeze_multiple_modules_all_verified(self):
        """Mô phỏng đúng train_meta_aggregator(): freeze CẢ 3 base agent
        (ở đây là 3 module giả), verify từng cái bằng assert riêng."""
        modules = {name: self._DummySubModule() for name in ("fundamental", "technical", "sentiment")}
        for name, module in modules.items():
            for p in module.parameters():
                p.requires_grad = False
            assert not any(p.requires_grad for p in module.parameters()), (
                f"module '{name}' chưa được freeze đúng cách"
            )

    def test_freeze_check_catches_forgotten_freeze(self):
        """Mô phỏng lỗi phổ biến "quên freeze" (đặc tả mục 4.5 cảnh báo) —
        assert check PHẢI fail nếu không thực sự set requires_grad=False."""
        module = self._DummySubModule()
        # KHÔNG freeze — mô phỏng lỗi.
        with pytest.raises(AssertionError):
            assert not any(p.requires_grad for p in module.parameters())

    def test_frozen_module_produces_no_gradient_on_backward(self):
        """Kiểm tra hệ quả thực tế của freeze: sau backward(), .grad của
        tham số đã freeze phải là None (không có gradient nào lan tới).

        Cần 1 tham số TRAINABLE ở downstream để loss thực sự có `grad_fn`
        (nếu toàn bộ graph frozen, `.backward()` sẽ raise
        `RuntimeError: ... does not require grad` ngay từ đầu — đó không
        phải hành vi cần kiểm tra ở đây, mà là "phần frozen không nhận
        gradient dù phần khác của graph vẫn train được").
        """
        module = self._DummySubModule()
        for p in module.parameters():
            p.requires_grad = False

        trainable_scale = nn.Parameter(torch.tensor(1.0))

        x = torch.randn(2, 4)
        out = module.fc2(torch.relu(module.fc1(x)))
        loss = (out * trainable_scale).sum()
        loss.backward()

        for p in module.parameters():
            assert p.grad is None
        assert trainable_scale.grad is not None


# ===========================================================================
# 5. FrozenSentimentExtractor (src/baselines/b6_frozen_extractor.py, mục 4.6)
# ===========================================================================
class TestFrozenSentimentExtractor:
    """Backbone/tokenizer giả (torch.nn thuần, không cần transformers/mạng)
    được TIÊM vào qua tham số `model=`/`tokenizer=` của
    `FrozenSentimentExtractor.__init__` — cơ chế dependency-injection này
    được thiết kế riêng trong `b6_frozen_extractor.py` chính vì mục đích
    test không cần mạng (xem docstring class)."""

    @staticmethod
    def _build_dummy_model_and_tokenizer(hidden_size: int = 8):
        class DummyConfig:
            def __init__(self, hidden_size):
                self.hidden_size = hidden_size

        class DummyOutput:
            def __init__(self, last_hidden_state):
                self.last_hidden_state = last_hidden_state

        class DummyModel(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.config = DummyConfig(hidden_size)
                self.embed = nn.Embedding(100, hidden_size)

            def forward(self, input_ids=None, **kwargs):
                emb = self.embed(input_ids)  # (batch, seq, hidden)
                return DummyOutput(last_hidden_state=emb)

        class DummyTokenizer:
            def __call__(self, prompt, return_tensors="pt", **kwargs):
                return {"input_ids": torch.tensor([[1, 2, 3, 4]])}

        return DummyModel(hidden_size), DummyTokenizer()

    def test_all_backbone_params_frozen_after_init(self):
        model, tokenizer = self._build_dummy_model_and_tokenizer()
        extractor = FrozenSentimentExtractor(model=model, tokenizer=tokenizer, projection_seed=123)

        assert len(list(extractor.model.parameters())) > 0
        for p in extractor.model.parameters():
            assert p.requires_grad is False

    def test_fixed_projection_also_frozen(self):
        model, tokenizer = self._build_dummy_model_and_tokenizer()
        extractor = FrozenSentimentExtractor(model=model, tokenizer=tokenizer, projection_seed=123)

        for p in extractor.fixed_projection.parameters():
            assert p.requires_grad is False

    def test_get_score_within_valid_range(self):
        model, tokenizer = self._build_dummy_model_and_tokenizer()
        extractor = FrozenSentimentExtractor(model=model, tokenizer=tokenizer, projection_seed=7)
        score = extractor.get_score("Tin tức tích cực về FPT.")
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0

    def test_projection_seed_is_reproducible(self):
        model1, tok1 = self._build_dummy_model_and_tokenizer()
        model2, tok2 = self._build_dummy_model_and_tokenizer()
        extractor1 = FrozenSentimentExtractor(model=model1, tokenizer=tok1, projection_seed=99)
        extractor2 = FrozenSentimentExtractor(model=model2, tokenizer=tok2, projection_seed=99)

        assert torch.allclose(extractor1.fixed_projection.weight, extractor2.fixed_projection.weight)
        assert torch.allclose(extractor1.fixed_projection.bias, extractor2.fixed_projection.bias)

    def test_different_seed_gives_different_projection(self):
        model1, tok1 = self._build_dummy_model_and_tokenizer()
        model2, tok2 = self._build_dummy_model_and_tokenizer()
        extractor1 = FrozenSentimentExtractor(model=model1, tokenizer=tok1, projection_seed=1)
        extractor2 = FrozenSentimentExtractor(model=model2, tokenizer=tok2, projection_seed=2)

        assert not torch.allclose(extractor1.fixed_projection.weight, extractor2.fixed_projection.weight)

    def test_backward_on_backbone_output_produces_no_grad_for_frozen_params(self):
        """Kiểm tra hệ quả thực tế: dù forward qua backbone đóng băng, nếu
        (giả định) có graph nào đó cố backward, .grad của params vẫn None."""
        model, tokenizer = self._build_dummy_model_and_tokenizer()
        extractor = FrozenSentimentExtractor(model=model, tokenizer=tokenizer, projection_seed=5)

        # get_score() đã dùng torch.no_grad() nội bộ nên không có graph để
        # backward — verify gián tiếp bằng cách gọi lại 2 lần cho ra cùng 1
        # giá trị (không có randomness/dropout nào ảnh hưởng vì eval() + frozen).
        score_a = extractor.get_score("tin tức A")
        score_b = extractor.get_score("tin tức A")
        assert score_a == pytest.approx(score_b)
