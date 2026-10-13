"""src/training/rollout_buffer.py

`RolloutBuffer` — lưu trữ transitions (state/prompt, action, reward,
log_prob, value, done) thu thập được trong 1 rollout PPO (mục 3.6 đặc tả
`HIFA_Implementation_Plan_GiaiDoan_A_B.md`).

Vì policy là LLM dùng state dạng TEXT PROMPT (quyết định thiết kế mục 3.4),
buffer lưu prompt dạng chuỗi (không phải obs numeric) — cần prompt gốc để
`SimplePPOTrainer.update()` forward lại qua policy (có gradient) khi tính
log_prob mới cho PPO ratio, KHÔNG lưu lại numeric obs vì policy không dùng nó
trực tiếp.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch


class RolloutBuffer:
    """Buffer đơn giản, dùng list Python nội bộ rồi stack thành tensor khi
    `get()` được gọi (rollout của Giai đoạn A nhỏ — n_steps 256-512 — nên
    không cần tối ưu bằng preallocated numpy array như SB3)."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Xoá sạch buffer — gọi sau mỗi lần `update()` dùng xong rollout
        hiện tại (PPO on-policy: dữ liệu cũ không được tái sử dụng)."""
        self.prompts: List[str] = []
        self.actions: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []

    def add(
        self,
        prompt: str,
        action: torch.Tensor,
        reward: float,
        log_prob: float,
        value: float,
        done: bool,
    ) -> None:
        """Thêm 1 transition vào buffer.

        Args:
            prompt: state đã chuyển thành text (output của `state_to_prompt`).
            action: tensor shape (n_assets+1,) — action đã lấy mẫu từ policy
                (đã normalize tổng=1).
            reward: reward nhận được sau khi env.step(action).
            log_prob: log-probability của `action` dưới policy TẠI THỜI ĐIỂM
                thu thập (policy cũ — dùng làm mẫu số cố định cho PPO ratio).
            value: V(s) ước lượng bởi value_head tại state này (baseline cho GAE).
            done: True nếu step này kết thúc episode (terminated hoặc truncated).
        """
        # .detach() để chắc chắn không giữ graph tính toán của lần forward
        # dùng để collect (đã ở trong torch.no_grad() nhưng an toàn thêm 1 lớp).
        self.prompts.append(prompt)
        self.actions.append(action.detach().float().cpu())
        self.rewards.append(float(reward))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.dones.append(bool(done))

    def __len__(self) -> int:
        return len(self.rewards)

    def get(self) -> Dict[str, Any]:
        """Trả về toàn bộ buffer, sẵn sàng cho `compute_gae()`/`update()`.

        Returns:
            dict với các khoá:
                prompts: list[str], độ dài N (KHÔNG stack thành tensor — là text)
                actions: FloatTensor (N, n_assets+1)
                rewards: FloatTensor (N,)
                log_probs: FloatTensor (N,)
                values: FloatTensor (N,)
                dones: FloatTensor (N,) — 0.0/1.0
        """
        if len(self) == 0:
            raise ValueError("RolloutBuffer rỗng — gọi add() ít nhất 1 lần trước khi get().")
        return {
            "prompts": list(self.prompts),
            "actions": torch.stack(self.actions, dim=0),
            "rewards": torch.tensor(self.rewards, dtype=torch.float32),
            "log_probs": torch.tensor(self.log_probs, dtype=torch.float32),
            "values": torch.tensor(self.values, dtype=torch.float32),
            "dones": torch.tensor(self.dones, dtype=torch.float32),
        }
