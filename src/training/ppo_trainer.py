"""src/training/ppo_trainer.py

`SimplePPOTrainer` — PPO custom cho LLM policy (mục 3.6 đặc tả
`HIFA_Implementation_Plan_GiaiDoan_A_B.md`), hoàn thiện đầy đủ khung mà
đặc tả để TODO: `collect_rollout`, `compute_gae` (GAE thật, KHÔNG REINFORCE
thuần — lỗi đã bị chỉ ra ở bản thiết kế trước), `update` (PPO clipped
objective + value loss + entropy bonus).

⚠️ CẢNH BÁO HIỆU NĂNG (trích đặc tả 3.6): vì policy là LLM, mỗi forward pass
chậm hơn MLP hàng trăm lần. `n_steps` mặc định ở đây là 512 (KHÔNG phải 2048
như PPO chuẩn) theo đúng khuyến nghị. `collect_rollout()` luôn chạy trong
`torch.no_grad()` — chỉ cần gradient lúc `update()`.

QUYẾT ĐỊNH THIẾT KẾ BỔ SUNG (không có trong đặc tả, cần thiết để hoàn thiện
TODO một cách nhất quán toán học):
`PortfolioPolicyLLM.forward()` trả về `weights` là kết quả CHÍNH XÁC của
softmax(logits) — tức MỘT vector xác định (deterministic) cho mỗi prompt,
không có phân phối xác suất tường minh để lấy mẫu (sample) hành động và tính
log-prob (bắt buộc cho PPO ratio = exp(new_log_prob - old_log_prob)). Ở đây,
`weights` được diễn giải như VECTOR KỲ VỌNG (mean) của một phân phối
Dirichlet: nếu concentration alpha_i = weights_i * kappa (kappa =
`dirichlet_concentration_scale`, hằng số dương), thì
E[Dirichlet(alpha)] = alpha / sum(alpha) = weights (vì softmax đã chuẩn hoá
sum=1) — action lấy mẫu từ Dirichlet này luôn nằm trên simplex hợp lệ (mọi
thành phần >=0, tổng=1, đúng ràng buộc action space của
`PortfolioEnvPhaseA`), có log_prob và entropy dạng đóng (closed-form), sẵn
sàng cho PPO. `kappa` càng lớn thì phân phối càng "chụm" quanh weights (ít
exploration hơn); càng nhỏ thì exploration nhiều hơn. Đây là lựa chọn tối
giản, không làm thay đổi API của `PortfolioPolicyLLM.forward()` (vẫn đúng
`(weights, value)` như code mẫu 3.4).
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.agents.super_agent_head import PortfolioPolicyLLM
from src.training.rollout_buffer import RolloutBuffer

# Layout numeric obs của PortfolioEnvPhaseA._get_obs() (mục 3.3 đặc tả +
# src/envs/portfolio_env.py): 5 feature/mã theo đúng thứ tự sau, lặp lại cho
# từng ticker, rồi nối thêm current weights (n_assets+1).
_OBS_FEATURE_NAMES: Tuple[str, ...] = ("close", "rsi_14", "macd", "macd_signal", "volatility_20d")


def obs_array_to_dict(
    obs_array: "np.ndarray | Sequence[float]",
    tickers: Sequence[str],
    base_close: Optional[Dict[str, float]] = None,
) -> Dict[str, Dict[str, float]]:
    """Chuyển ngược numeric obs array (từ `PortfolioEnvPhaseA._get_obs()`,
    numeric fallback dùng cho B4/debug) thành dict theo layout dùng bởi
    `PortfolioPolicyLLM.state_to_prompt()`.

    Layout obs (mục 3.3 đặc tả, xem `src/envs/portfolio_env.py::_get_obs`):
        obs = [close_1, rsi14_1, macd_1, macd_signal_1, vol20d_1,
               close_2, rsi14_2, ..., ] (n_assets bộ 5 feature)
              + list(current_weights)  (n_assets+1, KHÔNG dùng bởi hàm này)

    Args:
        obs_array: mảng numeric trả về bởi `env.reset()`/`env.step()`.
        tickers: danh sách mã theo ĐÚNG thứ tự dùng để build obs (env.tickers).
        base_close: tuỳ chọn — {ticker: giá đóng cửa ngày đầu episode}. Nếu
            truyền vào (vd. `env._base_close`), trường "close" trả về sẽ
            được NHÂN NGƯỢC lại (vì `PortfolioEnvPhaseA` lưu giá đã CHUẨN
            HÓA = close/base_close, xem docstring `portfolio_env.py`) để
            khôi phục giá VND thật cho prompt dễ đọc hơn ("giá=87000" thay
            vì "giá=1" khi giá gần bằng mốc chuẩn hoá). Nếu không truyền,
            "close" giữ nguyên giá trị chuẩn hoá gốc.

    Returns:
        {ticker: {"close":.., "rsi_14":.., "macd":.., "macd_signal":..,
        "volatility_20d":..}}
    """
    tickers = list(tickers)
    n_assets = len(tickers)
    n_feat = len(_OBS_FEATURE_NAMES)
    arr = np.asarray(obs_array, dtype=np.float64).reshape(-1)

    expected_min_len = n_assets * n_feat
    if arr.shape[0] < expected_min_len:
        raise ValueError(
            f"obs_array có {arr.shape[0]} phần tử, cần tối thiểu {expected_min_len} "
            f"({n_assets} tickers x {n_feat} feature/mã) theo layout _get_obs() (mục 3.3)."
        )

    obs_dict: Dict[str, Dict[str, float]] = {}
    for i, tk in enumerate(tickers):
        chunk = arr[i * n_feat : (i + 1) * n_feat]
        feat = dict(zip(_OBS_FEATURE_NAMES, chunk.tolist()))
        if base_close is not None and tk in base_close:
            feat["close"] = feat["close"] * float(base_close[tk])
        obs_dict[tk] = feat
    return obs_dict


def _weights_to_dirichlet(
    weights: torch.Tensor, concentration_scale: float, floor: float = 0.02
) -> torch.distributions.Dirichlet:
    """Xây phân phối Dirichlet có mean ≈ `weights` (xem giải thích ở docstring
    module).

    FIX approx_kl cao (quan sát thực nghiệm): softmax ở policy_head có thể tạo
    ra vài thành phần weights CỰC NHỎ (~0.0003). Khi đó concentration của mã đó
    = weights_i * scale ≈ 0, phân phối Dirichlet trở nên cực kỳ không ổn định
    (mật độ dồn vào biên simplex, entropy nổ về -∞, log_prob biến động dữ dội
    giữa 2 lần forward gần như giống nhau) -> ratio=exp(new-old) lệch lớn ->
    approx_kl ~5, clip_frac ~0.94 dù policy chưa học gì nhiều.

    Thay vì cộng eps tuyệt đối (quá nhỏ, không cứu được), ta TRỘN weights với
    phân phối ĐỀU (uniform smoothing trên simplex): mỗi thành phần được kéo về
    tối thiểu `floor` phần của khối lượng đều. Nhờ vậy mọi concentration_i >=
    floor/n * scale > 0 đủ lớn, Dirichlet ổn định, mean vẫn ≈ weights (chỉ
    smoothing nhẹ). Đây là kỹ thuật chuẩn (giống label smoothing) và giữ
    nguyên API `(weights, value)` của policy.
    """
    w = weights.clamp_min(0.0)
    w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    n = w.shape[-1]
    # Trộn: (1 - floor) * w + floor * uniform. floor=0.02 -> mỗi thành phần có
    # tối thiểu ~0.02/n khối lượng, đủ để concentration không gần 0.
    w_smoothed = (1.0 - floor) * w + floor / n
    concentration = w_smoothed * concentration_scale
    return torch.distributions.Dirichlet(concentration)


def _set_dropout_eval(module: nn.Module) -> None:
    """Đặt MỌI lớp Dropout (gồm cả LoRA dropout) về chế độ eval, GIỮ NGUYÊN
    các lớp khác ở chế độ train.

    LÝ DO (fix lỗi approx_kl/clip_frac cao): PPO yêu cầu log_prob của "policy
    cũ" (tính lúc collect_rollout dưới policy.eval(), dropout TẮT) và log_prob
    "policy mới" (tính lúc update dưới policy.train(), dropout BẬT) phải nhất
    quán về mặt stochasticity — nếu không, ratio = exp(new - old) sẽ lệch chỉ
    vì NHIỄU DROPOUT khác nhau giữa 2 lần forward, dù policy chưa học gì. Điều
    này đẩy approx_kl lên hàng chục và clip_frac ~1.0 ngay từ update đầu.

    Ta vẫn cần policy.train() để LoRA adapter + 2 head nhận gradient và cập
    nhật; chỉ cần vô hiệu hoá tính ngẫu nhiên của Dropout. Gradient checkpoint
    / BatchNorm (không dùng ở đây) không bị ảnh hưởng."""
    for m in module.modules():
        if isinstance(m, nn.Dropout):
            m.eval()


class SimplePPOTrainer:
    """PPO custom tối giản cho `PortfolioPolicyLLM`, theo khung mục 3.6 đặc tả."""

    def __init__(
        self,
        policy: PortfolioPolicyLLM,
        env: Any,
        lr: float = 1e-5,
        clip_eps: float = 0.2,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        n_steps: int = 512,  # khuyến nghị 256-512 (mục 3.6), KHÔNG dùng 2048 mặc định PPO chuẩn
        tickers: Optional[Sequence[str]] = None,
        dirichlet_concentration_scale: float = 10.0,
        gradient_checkpointing: bool = True,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = 0.05,
        rollout_warn_seconds: float = 1800.0,  # 30 phút (mục 3.6)
    ) -> None:
        self.policy = policy
        self.env = env
        # optimizer CHỈ trên params có requires_grad=True (LoRA + 2 head) —
        # backbone gốc frozen bởi get_peft_model(). Đúng khuyến nghị đặc tả.
        trainable_params = [p for p in policy.parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            raise ValueError(
                "Không có tham số nào requires_grad=True trong policy — kiểm tra "
                "lại LoRA/policy_head/value_head đã được khởi tạo đúng chưa."
            )
        self.optimizer = torch.optim.AdamW(trainable_params, lr=lr)

        self.clip_eps = clip_eps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.n_steps = n_steps
        self.dirichlet_concentration_scale = dirichlet_concentration_scale
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        # target_kl: nếu approx_kl trung bình của 1 epoch vượt ngưỡng này ->
        # DỪNG SỚM các epoch còn lại (early stopping theo KL, thực hành chuẩn
        # PPO) để policy không trôi quá xa "policy cũ". None = tắt early stop.
        self.target_kl = target_kl
        self.rollout_warn_seconds = rollout_warn_seconds

        resolved_tickers = tickers if tickers is not None else getattr(env, "tickers", None)
        if resolved_tickers is None:
            raise ValueError(
                "Không xác định được `tickers` — truyền qua tham số `tickers=` hoặc "
                "dùng env có attribute `env.tickers` (vd. PortfolioEnvPhaseA)."
            )
        self.tickers: List[str] = list(resolved_tickers)
        if len(self.tickers) != policy.n_assets:
            raise ValueError(
                f"Số ticker ({len(self.tickers)}) khác n_assets của policy "
                f"({policy.n_assets}) — action_space sẽ không khớp."
            )

        self._last_rollout_seconds: Optional[float] = None

        # Bật gradient checkpointing để tiết kiệm VRAM (mục 3.6) — chỉ áp
        # dụng nếu backbone hỗ trợ (HF PreTrainedModel/PeftModel hoặc
        # `_DummyBackboneForTest` có no-op tương thích).
        if gradient_checkpointing:
            backbone = getattr(policy, "backbone", None)
            if backbone is not None and hasattr(backbone, "gradient_checkpointing_enable"):
                try:
                    backbone.gradient_checkpointing_enable()
                except Exception as exc:  # pragma: no cover - phụ thuộc backbone thật
                    warnings.warn(f"Không thể bật gradient checkpointing: {exc}")

    # ------------------------------------------------------------------
    def _base_close(self) -> Optional[Dict[str, float]]:
        """Lấy `env._base_close` nếu env cung cấp (PortfolioEnvPhaseA) để
        `obs_array_to_dict` khôi phục giá VND thật trong prompt. Trả về None
        nếu env không có attribute này (obs "close" sẽ giữ nguyên giá trị thô)."""
        return getattr(self.env, "_base_close", None)

    def _obs_to_prompt(self, obs: "np.ndarray") -> str:
        obs_dict = obs_array_to_dict(obs, self.tickers, base_close=self._base_close())
        return self.policy.state_to_prompt(obs_dict)

    # ------------------------------------------------------------------
    def collect_rollout(self, n_steps: Optional[int] = None) -> Tuple[RolloutBuffer, float]:
        """Thu thập (state/prompt, action, reward, log_prob, value, done) qua
        `n_steps` bước tương tác với `self.env`.

        Dùng `torch.no_grad()` xuyên suốt (chỉ cần gradient lúc `update()`,
        theo đúng cảnh báo hiệu năng mục 3.6) và `policy.eval()` (tắt LoRA
        dropout khi thu thập, chỉ bật lại lúc update để không làm nhiễu ước
        lượng log_prob/value dùng làm "policy cũ" cố định).

        Mỗi step: `policy.state_to_prompt()` rồi `forward()` — batch=1
        (không thể batch nhiều timestep cùng lúc vì mỗi action phụ thuộc kết
        quả step trước của env, môi trường tuần tự).

        Returns:
            (buffer, last_value): `buffer` là `RolloutBuffer` đã đầy;
            `last_value` là V(s_T) dùng bootstrap cho GAE — 0.0 nếu step
            cuối cùng của rollout kết thúc episode (done=True), ngược lại là
            value_head ước lượng tại obs kế tiếp (chưa dùng để hành động).
        """
        n_steps = n_steps or self.n_steps
        buffer = RolloutBuffer()

        self.policy.eval()
        start_time = time.perf_counter()

        obs, _ = self.env.reset()
        last_done = True
        with torch.no_grad():
            for _ in range(n_steps):
                prompt = self._obs_to_prompt(obs)
                weights, value = self.policy([prompt])
                weights = weights[0]
                value_scalar = float(value[0].item())

                dist = _weights_to_dirichlet(weights, self.dirichlet_concentration_scale)
                action = dist.sample()
                action = action / action.sum().clamp_min(1e-8)  # an toàn số học
                log_prob = float(dist.log_prob(action).item())

                action_np = action.detach().cpu().numpy().astype(np.float32)
                next_obs, reward, terminated, truncated, _info = self.env.step(action_np)
                done = bool(terminated or truncated)

                buffer.add(
                    prompt=prompt,
                    action=action,
                    reward=reward,
                    log_prob=log_prob,
                    value=value_scalar,
                    done=done,
                )

                last_done = done
                if done:
                    obs, _ = self.env.reset()
                else:
                    obs = next_obs

            # Bootstrap V(s_T) cho GAE (obs hiện tại CHƯA được dùng để hành động).
            if last_done:
                last_value = 0.0
            else:
                prompt = self._obs_to_prompt(obs)
                _, value = self.policy([prompt])
                last_value = float(value[0].item())

        elapsed = time.perf_counter() - start_time
        self._last_rollout_seconds = elapsed
        if elapsed > self.rollout_warn_seconds:
            warnings.warn(
                f"collect_rollout() mất {elapsed / 60:.1f} phút cho n_steps={n_steps} "
                f"(> {self.rollout_warn_seconds / 60:.0f} phút) — theo khuyến nghị mục 3.6, "
                "cần giảm thêm n_steps hoặc chuyển sang model nhỏ hơn (0.5B) để debug "
                "pipeline trước khi scale lên 1.5B."
            )

        self.policy.train()
        return buffer, last_value

    # ------------------------------------------------------------------
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        last_value: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generalized Advantage Estimation (Schulman et al., 2016) — BẮT
        BUỘC để giảm phương sai, KHÔNG dùng công thức REINFORCE thuần
        (return Monte Carlo trừ baseline không bootstrap) như bản thiết kế
        đầu tiên đã bị chỉ ra lỗi.

        delta_t   = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
        A_t (GAE) = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
        return_t  = A_t + V(s_t)   # target cho value loss

        Args:
            rewards, values, dones: FloatTensor shape (T,) (từ `RolloutBuffer.get()`).
            last_value: V(s_T) bootstrap cho step cuối rollout (xem `collect_rollout`).

        Returns:
            (advantages, returns): FloatTensor shape (T,) mỗi cái.
        """
        T = rewards.shape[0]
        advantages = torch.zeros(T, dtype=torch.float32)
        gae = 0.0
        next_value = float(last_value)

        for t in reversed(range(T)):
            next_non_terminal = 1.0 - float(dones[t].item())
            delta = float(rewards[t].item()) + self.gamma * next_value * next_non_terminal - float(
                values[t].item()
            )
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
            next_value = float(values[t].item())

        returns = advantages + values
        return advantages, returns

    # ------------------------------------------------------------------
    def update(
        self,
        rollout_data: Dict[str, Any],
        n_epochs: int = 4,
        batch_size: int = 64,
    ) -> Dict[str, float]:
        """PPO clipped objective + value loss (MSE) + entropy bonus.

        `rollout_data` PHẢI chứa (xem `train_step()` để build đúng dict này
        từ `RolloutBuffer.get()` + `compute_gae()`):
            prompts, actions, log_probs (log_prob CŨ), values (value CŨ,
            không dùng trực tiếp trong loss nhưng giữ lại để log/debug),
            advantages, returns.

        Lặp `n_epochs` lần qua toàn bộ rollout, mỗi lần chia minibatch
        `batch_size` (shuffle lại mỗi epoch — thực hành chuẩn PPO).
        """
        prompts: List[str] = rollout_data["prompts"]
        old_actions: torch.Tensor = rollout_data["actions"]
        old_log_probs: torch.Tensor = rollout_data["log_probs"]
        advantages: torch.Tensor = rollout_data["advantages"]
        returns: torch.Tensor = rollout_data["returns"]

        T = old_actions.shape[0]
        if T == 0:
            raise ValueError("rollout_data rỗng — không có gì để update().")

        # Chuẩn hoá advantage (giảm phương sai gradient) — thực hành chuẩn PPO,
        # KHÔNG làm thay đổi công thức GAE đã tính ở compute_gae().
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        stats: Dict[str, List[float]] = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
            "clip_frac": [],
            "kl_early_stop_epoch": [],
        }

        # policy.train() để LoRA + 2 head nhận gradient, NHƯNG tắt Dropout để
        # log_prob mới nhất quán với log_prob cũ.
        self.policy.train()
        _set_dropout_eval(self.policy)
        trainable_params = [p for p in self.policy.parameters() if p.requires_grad]

        # FIX approx_kl cao do BF16: log_prob "cũ" lưu từ collect_rollout được
        # tính dưới policy.eval() + đường bf16 KHÁC với forward trong update()
        # (train mode, có gradient). Với bf16, cùng input nhưng 2 đường tính cho
        # weights lệch nhẹ -> Dirichlet.log_prob (nhạy) lệch vài đơn vị ->
        # ratio=exp(new-old) sai -> approx_kl ~5 dù policy CHƯA học gì. (Kiểm
        # chứng: với backbone float32 tất định, KL ở forward đầu = 0.000000.)
        #
        # Khắc phục chuẩn: TÍNH LẠI old_log_prob bằng CHÍNH forward đầu tiên của
        # update() (no_grad), rồi dùng nó làm mốc "policy cũ". Nhờ vậy ratio ở
        # minibatch đầu epoch đầu = 1 chính xác (KL=0), loại bỏ nhiễu bf16 giữa
        # 2 pha. Đây là biến thể hợp lệ của PPO (old_logprob = policy tại đầu
        # update, trước mọi optimizer.step()). KHÔNG dùng old_log_probs lưu từ
        # rollout nữa — nó chỉ còn để log/debug.
        recomputed_old_lp: Dict[int, torch.Tensor] = {}
        with torch.no_grad():
            for start in range(0, T, batch_size):
                idx_r = list(range(start, min(start + batch_size, T)))
                w_r, _ = self.policy([prompts[i] for i in idx_r])
                dist_r = _weights_to_dirichlet(w_r, self.dirichlet_concentration_scale)
                act_r = old_actions[start : start + batch_size].to(w_r.device)
                lp_r = dist_r.log_prob(act_r).detach()
                for j, i in enumerate(idx_r):
                    recomputed_old_lp[i] = lp_r[j]
        old_log_probs = torch.stack([recomputed_old_lp[i] for i in range(T)]).to(advantages.device)

        for _epoch in range(n_epochs):
            epoch_kls: List[float] = []
            perm = torch.randperm(T)
            for start in range(0, T, batch_size):
                idx = perm[start : start + batch_size]
                batch_prompts = [prompts[i] for i in idx.tolist()]
                batch_actions = old_actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_advantages = advantages[idx]
                batch_returns = returns[idx]

                # Gradient BẬT ở đây (khác hẳn collect_rollout dùng no_grad).
                weights, values_pred = self.policy(batch_prompts)
                dist = _weights_to_dirichlet(weights, self.dirichlet_concentration_scale)

                # Rollout buffer giữ tensor trên CPU, còn policy chạy trên GPU khi
                # có CUDA -> phải đưa các batch tensor về đúng device của output
                # policy trước khi dùng, tránh "tensors on different devices".
                policy_device = weights.device
                batch_actions = batch_actions.to(policy_device)
                batch_old_log_probs = batch_old_log_probs.to(policy_device)
                batch_advantages = batch_advantages.to(policy_device)
                batch_returns = batch_returns.to(policy_device)

                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(values_pred.squeeze(-1), batch_returns)

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm is not None and self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = float((batch_old_log_probs - new_log_probs).mean().item())
                    clip_frac = float(((ratio - 1.0).abs() > self.clip_eps).float().mean().item())

                stats["policy_loss"].append(float(policy_loss.item()))
                stats["value_loss"].append(float(value_loss.item()))
                stats["entropy"].append(float(entropy.item()))
                stats["approx_kl"].append(approx_kl)
                stats["clip_frac"].append(clip_frac)
                epoch_kls.append(approx_kl)

            # Early stopping theo KL: nếu KL trung bình của epoch này vượt
            # target_kl -> dừng, không chạy các epoch còn lại (policy đã trôi
            # đủ xa "policy cũ", update thêm chỉ gây bất ổn). approx_kl ở đây
            # tính theo (old - new).mean() nên dùng abs() cho an toàn.
            if self.target_kl is not None and len(epoch_kls) > 0:
                if abs(float(np.mean(epoch_kls))) > self.target_kl:
                    stats["kl_early_stop_epoch"].append(float(_epoch + 1))
                    break

        result = {k: float(np.mean(v)) for k, v in stats.items() if len(v) > 0}
        # Đánh dấu không early-stop (chạy đủ n_epochs) bằng 0 để log luôn có khoá này.
        result.setdefault("kl_early_stop_epoch", 0.0)
        return result

    # ------------------------------------------------------------------
    def train_step(
        self,
        n_steps: Optional[int] = None,
        n_epochs: int = 4,
        batch_size: int = 64,
    ) -> Dict[str, float]:
        """Tiện ích: 1 update PPO đầy đủ = collect_rollout -> compute_gae ->
        update. `train_phase_a.py` gọi hàm này lặp lại `total_updates` lần."""
        buffer, last_value = self.collect_rollout(n_steps)
        data = buffer.get()
        advantages, returns = self.compute_gae(data["rewards"], data["values"], data["dones"], last_value)
        data["advantages"] = advantages
        data["returns"] = returns

        stats = self.update(data, n_epochs=n_epochs, batch_size=batch_size)
        stats["mean_reward"] = float(data["rewards"].mean().item())
        stats["mean_value"] = float(data["values"].mean().item())
        stats["n_steps"] = float(len(buffer))
        stats["rollout_seconds"] = float(self._last_rollout_seconds or 0.0)
        return stats
