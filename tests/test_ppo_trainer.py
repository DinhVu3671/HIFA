"""tests/test_ppo_trainer.py

Unit test cho kiến trúc LLM-policy (`PortfolioPolicyLLM`) và vòng lặp PPO
custom (`SimplePPOTrainer`, `RolloutBuffer`) — mục 3.4/3.6 đặc tả
`HIFA_Implementation_Plan_GiaiDoan_A_B.md`.

THIẾT KẾ TEST: KHÔNG tải Qwen2.5-1.5B thật (nặng ~3GB, cần mạng) — mọi test
liên quan `PortfolioPolicyLLM` dùng `_DummyBackboneForTest` (torch.nn thuần,
đặt tên submodule q_proj/k_proj/v_proj/o_proj trùng target_modules LoRA
production, xem `src/agents/super_agent_head.py`) tiêm qua tham số nội bộ
`_test_backbone_override`. Test `compute_gae`/`RolloutBuffer` không phụ
thuộc `transformers`/`peft`/mạng ở bất kỳ mức nào.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.agents.super_agent_head import PortfolioPolicyLLM, _DummyBackboneForTest
from src.training.ppo_trainer import SimplePPOTrainer, obs_array_to_dict
from src.training.rollout_buffer import RolloutBuffer


# ---------------------------------------------------------------------------
# Stub tối giản để khởi tạo SimplePPOTrainer mà không cần policy/env thật —
# dùng cho các test chỉ cần compute_gae() (không đụng tới forward()).
# ---------------------------------------------------------------------------
class _StubPolicy(nn.Module):
    def __init__(self, n_assets: int = 1) -> None:
        super().__init__()
        self.n_assets = n_assets
        self.dummy = nn.Linear(2, 2)  # có >=1 param requires_grad=True cho optimizer


def _make_trainer(gamma: float, gae_lambda: float, n_assets: int = 1) -> SimplePPOTrainer:
    policy = _StubPolicy(n_assets=n_assets)
    env = SimpleNamespace(tickers=[f"T{i}" for i in range(n_assets)])
    return SimplePPOTrainer(
        policy=policy,
        env=env,
        gamma=gamma,
        gae_lambda=gae_lambda,
        gradient_checkpointing=False,
    )


# ---------------------------------------------------------------------------
# 1. compute_gae — verify công thức GAE bằng tay (KHÔNG phải REINFORCE thuần)
# ---------------------------------------------------------------------------
def test_compute_gae_matches_hand_computed_values():
    """gamma=1, lambda=1 (không discount) + done=True ở step cuối cùng =>
    GAE suy biến về công thức Monte-Carlo return trừ baseline:
        A_t = sum_{k=t}^{T-1} r_k - V(s_t)
    Tính tay: rewards=[1,2,3], values=[0.5,0.5,0.5], done=[F,F,T]
        A_0 = (1+2+3) - 0.5 = 5.5   ; return_0 = 5.5+0.5 = 6.0
        A_1 = (2+3)   - 0.5 = 4.5   ; return_1 = 4.5+0.5 = 5.0
        A_2 = (3)     - 0.5 = 2.5   ; return_2 = 2.5+0.5 = 3.0
    """
    trainer = _make_trainer(gamma=1.0, gae_lambda=1.0)
    rewards = torch.tensor([1.0, 2.0, 3.0])
    values = torch.tensor([0.5, 0.5, 0.5])
    dones = torch.tensor([0.0, 0.0, 1.0])

    advantages, returns = trainer.compute_gae(rewards, values, dones, last_value=0.0)

    expected_adv = torch.tensor([5.5, 4.5, 2.5])
    expected_ret = torch.tensor([6.0, 5.0, 3.0])
    assert torch.allclose(advantages, expected_adv, atol=1e-6)
    assert torch.allclose(returns, expected_ret, atol=1e-6)


def test_compute_gae_single_step_with_bootstrap():
    """T=1, done=False -> phải bootstrap bằng last_value (khác REINFORCE
    thuần vốn không bootstrap gì cả):
        delta_0 = r_0 + gamma*last_value*(1-done_0) - v_0
        A_0 = delta_0 (không có đệ quy vì T=1)
    """
    trainer = _make_trainer(gamma=0.9, gae_lambda=0.95)
    rewards = torch.tensor([2.0])
    values = torch.tensor([1.0])
    dones = torch.tensor([0.0])
    last_value = 4.0

    advantages, returns = trainer.compute_gae(rewards, values, dones, last_value=last_value)

    expected_delta = 2.0 + 0.9 * 4.0 * 1.0 - 1.0  # = 4.6
    assert advantages[0].item() == pytest.approx(expected_delta, abs=1e-6)
    assert returns[0].item() == pytest.approx(expected_delta + 1.0, abs=1e-6)


def test_compute_gae_done_zeroes_bootstrap():
    """done=True ở step cuối phải chặn hoàn toàn bootstrap (next_non_terminal=0),
    bất kể last_value là bao nhiêu — khác biệt quan trọng so với việc quên
    nhân (1-done) (lỗi REINFORCE/TD thường gặp)."""
    trainer = _make_trainer(gamma=0.99, gae_lambda=0.95)
    rewards = torch.tensor([5.0])
    values = torch.tensor([2.0])
    dones = torch.tensor([1.0])

    advantages, _ = trainer.compute_gae(rewards, values, dones, last_value=1000.0)
    # last_value=1000 KHÔNG được lọt vào vì done=True -> delta = 5 - 2 = 3
    assert advantages[0].item() == pytest.approx(3.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 2. RolloutBuffer — add/get/reset, shape & số lượng đúng
# ---------------------------------------------------------------------------
def test_rollout_buffer_add_and_get_shapes():
    buf = RolloutBuffer()
    n_assets_plus_cash = 4
    n_steps = 5
    for i in range(n_steps):
        buf.add(
            prompt=f"prompt số {i}",
            action=torch.tensor([0.1, 0.2, 0.3, 0.4]),
            reward=float(i) * 0.1,
            log_prob=-0.5 - i,
            value=1.0 + i,
            done=(i == n_steps - 1),
        )

    assert len(buf) == n_steps

    data = buf.get()
    assert len(data["prompts"]) == n_steps
    assert data["actions"].shape == (n_steps, n_assets_plus_cash)
    assert data["rewards"].shape == (n_steps,)
    assert data["log_probs"].shape == (n_steps,)
    assert data["values"].shape == (n_steps,)
    assert data["dones"].shape == (n_steps,)

    assert torch.allclose(data["rewards"], torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4]), atol=1e-6)
    assert data["dones"][-1].item() == 1.0
    assert data["dones"][0].item() == 0.0


def test_rollout_buffer_reset_clears_and_get_raises_when_empty():
    buf = RolloutBuffer()
    buf.add(prompt="p", action=torch.tensor([1.0]), reward=1.0, log_prob=0.0, value=0.0, done=True)
    assert len(buf) == 1
    buf.reset()
    assert len(buf) == 0
    with pytest.raises(ValueError):
        buf.get()


# ---------------------------------------------------------------------------
# 3. obs_array_to_dict — chuyển ngược numeric obs -> dict cho state_to_prompt
# ---------------------------------------------------------------------------
def test_obs_array_to_dict_basic_layout():
    tickers = ["FPT", "VPB"]
    # 2 tickers x 5 feature (close, rsi_14, macd, macd_signal, volatility_20d) + 3 weights (cash)
    features = [1.05, 55.0, 0.30, 0.20, 0.15, 0.98, 40.0, -0.10, -0.05, 0.22]
    weights = [0.4, 0.4, 0.2]
    obs = np.array(features + weights, dtype=np.float32)

    d = obs_array_to_dict(obs, tickers)

    assert set(d.keys()) == {"FPT", "VPB"}
    assert d["FPT"]["close"] == pytest.approx(1.05, abs=1e-4)
    assert d["FPT"]["rsi_14"] == pytest.approx(55.0, abs=1e-3)
    assert d["VPB"]["macd_signal"] == pytest.approx(-0.05, abs=1e-4)
    assert d["VPB"]["volatility_20d"] == pytest.approx(0.22, abs=1e-4)


def test_obs_array_to_dict_restores_real_price_with_base_close():
    tickers = ["FPT"]
    obs = np.array([1.05, 55.0, 0.3, 0.2, 0.15, 1.0], dtype=np.float32)
    d = obs_array_to_dict(obs, tickers, base_close={"FPT": 90_000.0})
    assert d["FPT"]["close"] == pytest.approx(1.05 * 90_000.0, rel=1e-4)


def test_obs_array_to_dict_raises_on_too_short_array():
    with pytest.raises(ValueError):
        obs_array_to_dict(np.array([1.0, 2.0]), ["FPT", "VPB"])


# ---------------------------------------------------------------------------
# 4. PortfolioPolicyLLM.forward với _DummyBackboneForTest — không cần mạng
# ---------------------------------------------------------------------------
def test_policy_forward_dummy_backbone_weights_sum_to_one_and_value_shape():
    n_assets = 3
    backbone = _DummyBackboneForTest(vocab_size=200, hidden_size=16)
    policy = PortfolioPolicyLLM(
        n_assets=n_assets,
        use_lora=False,  # test kiến trúc thuần, không cần peft ở đây
        max_prompt_length=32,
        _test_backbone_override=backbone,
    )

    prompts = [
        "Dữ liệu thị trường hiện tại:\n- FPT: giá=120000, RSI=65.0, MACD=tăng, biến động 20 ngày=1.8%",
        "Dữ liệu thị trường hiện tại:\n- VPB: giá=18000, RSI=30.0, MACD=giảm, biến động 20 ngày=2.5%",
    ]
    weights, value = policy(prompts)

    assert weights.shape == (len(prompts), n_assets + 1)
    assert value.shape == (len(prompts), 1)

    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones(len(prompts)), atol=1e-5)
    assert torch.all(weights >= 0.0)
    assert torch.isfinite(value).all()


def test_state_to_prompt_produces_english_text_with_expected_fields():
    backbone = _DummyBackboneForTest(vocab_size=50, hidden_size=8)
    policy = PortfolioPolicyLLM(
        n_assets=2, use_lora=False, _test_backbone_override=backbone
    )
    obs_dict = {
        "AAPL": {"close": 180.0, "rsi_14": 65.0, "macd": 0.5, "macd_signal": 0.2, "volatility_20d": 0.018},
        "MSFT": {"close": 320.0, "rsi_14": 30.0, "macd": -0.1, "macd_signal": 0.0, "volatility_20d": 0.025},
    }
    prompt = policy.state_to_prompt(obs_dict)
    assert "AAPL" in prompt and "MSFT" in prompt
    assert "RSI=65.0" in prompt
    assert "MACD=bullish" in prompt  # AAPL: macd(0.5) > macd_signal(0.2)
    assert "MACD=bearish" in prompt  # MSFT: macd(-0.1) < macd_signal(0.0)
    assert "20-day volatility=1.8%" in prompt


# ---------------------------------------------------------------------------
# 5. requires_grad: chỉ LoRA + head trainable, backbone gốc frozen.
#
# GIỚI HẠN CỦA TEST NÀY: `_DummyBackboneForTest` không phải model HF thật —
# nó chỉ có 4 submodule tuyến tính (q/k/v/o_proj) được đặt tên khớp
# target_modules để `peft.get_peft_model` có thể gắn adapter LoRA THẬT lên
# (không cần mạng, peft chỉ thao tác cục bộ trên nn.Module truyền vào). Vì
# vậy assertion "mọi param tên chứa 'backbone' mà KHÔNG chứa 'lora' phải
# requires_grad=False" ở đây kiểm tra ĐÚNG cơ chế `get_peft_model` thật (cùng
# cơ chế áp dụng cho Qwen2.5-1.5B ở production), chỉ khác input là backbone
# nhỏ/giả — nếu peft đổi hành vi đóng băng tham số ở phiên bản khác, test này
# vẫn phát hiện được. Test KHÔNG chứng minh gì thêm về kiến trúc/attention
# thật của Qwen2.5 (bản thân backbone dummy không giống Qwen).
# ---------------------------------------------------------------------------
def test_lora_wrapping_freezes_backbone_except_adapters():
    pytest.importorskip("peft")
    backbone = _DummyBackboneForTest(vocab_size=200, hidden_size=16)
    try:
        policy = PortfolioPolicyLLM(
            n_assets=2,
            use_lora=True,
            lora_r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            _test_backbone_override=backbone,
        )
    except Exception as exc:  # pragma: no cover - phụ thuộc phiên bản peft cài đặt
        pytest.skip(
            "peft không gắn được LoRA lên _DummyBackboneForTest trong môi trường "
            f"này (không ảnh hưởng production path với Qwen2.5 thật): {exc}"
        )

    found_lora_param = False
    for name, p in policy.named_parameters():
        if "backbone" not in name:
            continue
        if "lora" in name.lower():
            found_lora_param = True
            assert p.requires_grad, f"Tham số LoRA '{name}' phải requires_grad=True"
        else:
            assert not p.requires_grad, f"Tham số backbone gốc '{name}' phải bị đóng băng"

    assert found_lora_param, (
        "Không tìm thấy tham số LoRA nào trong backbone — target_modules có thể "
        "không khớp tên submodule."
    )

    for name, p in policy.named_parameters():
        if "policy_head" in name or "value_head" in name:
            assert p.requires_grad, f"'{name}' (policy_head/value_head) phải luôn trainable"

    n_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in policy.parameters())
    assert 0 < n_trainable < n_total, "Số tham số trainable phải nhỏ hơn hẳn tổng số tham số"


# ---------------------------------------------------------------------------
# 6. (Bổ sung) Smoke test toàn luồng SimplePPOTrainer với FakeEnv + dummy
# backbone — không cần policy/env thật, chỉ để xác nhận collect_rollout ->
# compute_gae -> update chạy hết không lỗi runtime (đúng checklist mục 3.6:
# "Chạy thử pipeline với 1 epoch, n_steps nhỏ chỉ để kiểm tra không có lỗi
# runtime"), và loss hữu hạn (không NaN/Inf).
# ---------------------------------------------------------------------------
class _FakeEnv:
    """Env tối giản, tuân thủ API Gymnasium (`reset`/`step`) và có
    `tickers`/`n_assets` giống `PortfolioEnvPhaseA`, nhưng KHÔNG phụ thuộc
    `src.envs.portfolio_env` (module do 1 phần khác của dự án viết song
    song) — obs sinh ngẫu nhiên, đúng layout `n_assets*5 + (n_assets+1)`."""

    def __init__(self, n_assets: int = 2, episode_len: int = 4, seed: int = 0) -> None:
        self.n_assets = n_assets
        self.tickers = [f"T{i}" for i in range(n_assets)]
        self.episode_len = episode_len
        self._rng = np.random.default_rng(seed)
        self._t = 0

    def _obs(self) -> np.ndarray:
        feat = self._rng.normal(loc=1.0, scale=0.1, size=self.n_assets * 5).astype(np.float32)
        weights = np.full(self.n_assets + 1, 1.0 / (self.n_assets + 1), dtype=np.float32)
        return np.concatenate([feat, weights])

    def reset(self, seed=None, options=None):
        self._t = 0
        return self._obs(), {}

    def step(self, action):
        self._t += 1
        terminated = self._t >= self.episode_len
        reward = float(self._rng.normal())
        return self._obs(), reward, terminated, False, {}


def test_ppo_trainer_end_to_end_smoke_no_runtime_errors():
    n_assets = 2
    backbone = _DummyBackboneForTest(vocab_size=100, hidden_size=8)
    policy = PortfolioPolicyLLM(
        n_assets=n_assets, use_lora=False, max_prompt_length=32, _test_backbone_override=backbone
    )
    env = _FakeEnv(n_assets=n_assets, episode_len=4, seed=1)

    trainer = SimplePPOTrainer(
        policy=policy,
        env=env,
        lr=1e-3,
        n_steps=6,
        gradient_checkpointing=False,
    )

    stats = trainer.train_step(n_steps=6, n_epochs=1, batch_size=3)

    for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "mean_reward"):
        assert key in stats
        assert np.isfinite(stats[key]), f"stats['{key}'] không hữu hạn: {stats[key]}"
