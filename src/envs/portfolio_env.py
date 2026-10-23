"""Môi trường Gymnasium cho bài toán tối ưu danh mục VN30 (HIFA).

Cài đặt theo đặc tả mục 3.3 (`HIFA_Implementation_Plan_GiaiDoan_A_B.md`), đã
hoàn thiện/sửa các điểm đặc tả để ngỏ hoặc còn lỗi:
    - Logic action-masking được tách sang `src/envs/action_masking.py`
      (`apply_action_mask`) thay vì định nghĩa lại trong class, đồng thời SỬA
      lỗi hành vi: bản nháp trong đặc tả có chú thích "giữ nguyên trọng số cũ
      cho phần bị khoá" nhưng code lại đặt về 0 — ở đây mã bị khoá thực sự giữ
      nguyên tỷ trọng hiện tại (`self.weights`) vì về bản chất không thể khớp
      lệnh khi cổ phiếu chạm trần/sàn.
    - Giá đưa vào observation được CHUẨN HOÁ (chia cho giá đóng cửa ngày đầu
      episode) thay vì dùng giá đóng cửa thô — bản nháp trong đặc tả gọi biến là
      "giá chuẩn hoá" trong comment nhưng lại nhét giá thô vào obs; các mã VN30
      có biên độ giá chênh lệch rất lớn (vài nghìn tới hơn trăm nghìn VND) nên
      nếu không chuẩn hoá, obs sẽ mất cân bằng scale giữa các mã.
    - `reset`/`step` tuân theo API Gymnasium mới (5 giá trị trả về từ
      `step`: obs, reward, terminated, truncated, info).

State DataFrame kỳ vọng (định dạng do `src/data/build_dataset.py::
build_state_dataframe` tạo ra — KHÔNG import trực tiếp từ `src.data` ở đây để
tránh phụ thuộc chéo lúc các phần khác của dự án đang được code song song, chỉ
giả định đúng cấu trúc sau):
    - index: ngày giao dịch, tăng dần theo thời gian, đã align chung giữa các
      mã (không có ngày lệch nhau giữa các mã).
    - columns: `pandas.MultiIndex` 2 cấp `(ticker, feature)` với tối thiểu 5
      feature cho mỗi ticker: "close", "rsi_14", "macd", "macd_signal",
      "volatility_20d".
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd

from src.envs.action_masking import apply_action_mask, apply_position_cap

# Feature numeric tối thiểu / mã, theo đúng thứ tự nhét vào observation.
_FEATURES: Tuple[str, ...] = ("close", "rsi_14", "macd", "macd_signal", "volatility_20d")


class PortfolioEnvPhaseA(gym.Env):
    """Môi trường RL tối ưu danh mục cho Giai đoạn A (numeric fallback).

    Action: vector phân bổ `(n_assets + 1,)` — `n_assets` mã theo `tickers` +
    1 phần tử cuối là tiền mặt (cash). Không bắt buộc đã chuẩn hoá sẵn;
    `step()` sẽ clip/normalize (qua `apply_action_mask`) để tổng luôn = 1.

    Observation: vector numeric `(n_assets * 5 + n_assets + 1,)` gồm:
        - Với mỗi mã (theo thứ tự `tickers`): giá đóng cửa đã chuẩn hoá (so
          với giá đóng cửa ngày đầu episode), rsi_14, macd, macd_signal,
          volatility_20d (5 giá trị/mã).
        - Tỷ trọng danh mục hiện tại `(n_assets + 1,)` (bao gồm cash).

    Đây là numeric fallback dùng cho baseline B4 (PPO SB3 thuần) và debug —
    policy LLM chính (mục 3.4) dùng state dạng text prompt riêng, không dùng
    trực tiếp observation này.

    LƯU Ý: `reset()`/`step()` theo API Gymnasium mới — `step()` trả về 5 giá
    trị `(obs, reward, terminated, truncated, info)`.
    """

    metadata: Dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        state_df: pd.DataFrame,
        tickers: Sequence[str],
        initial_cash: float = 1_000_000_000,
        transaction_cost: float = 0.002,
        price_limit: Optional[float] = 0.07,
        max_weight: Optional[float] = None,
    ) -> None:
        """
        Args:
            state_df: DataFrame giá/indicator theo format mô tả ở docstring
                module (MultiIndex columns (ticker, feature)).
            tickers: Danh sách mã (thứ tự cố định, dùng xuyên suốt episode).
            initial_cash: Vốn ban đầu (VND cho VN30, USD cho Dow30/Nasdaq-100 —
                giá trị tuyệt đối không ảnh hưởng %-return/Sharpe).
            max_weight: Trần tỷ trọng tối đa mỗi mã (vd. 0.30 = 30% theo
                De-Cuong §6.2). ``None`` (mặc định) = không áp trần, giữ hành
                vi cũ. Áp dụng sau action-masking, trước khi tính turnover.
            price_limit: Biên độ dao động giá tối đa 1 phiên. ``None`` = thị
                trường không có biên độ (Mỹ). Mặc định 0.07 (HOSE) để tương
                thích env VN cũ.
            transaction_cost: Phí giao dịch tính trên turnover (vd. 0.002 =
                0.2%/lượt xoay vòng danh mục).
            price_limit: Biên độ dao động giá tối đa 1 phiên (mặc định 0.07 =
                +-7% theo HOSE), dùng cho action-masking.
        """
        super().__init__()

        if len(state_df) < 2:
            raise ValueError(
                "state_df cần ít nhất 2 hàng (1 để quan sát, 1 để tính return "
                f"bước kế tiếp), nhận được {len(state_df)} hàng."
            )

        self.state_df = state_df
        self.tickers: List[str] = list(tickers)
        self.n_assets = len(self.tickers)
        self.initial_cash = float(initial_cash)
        self.transaction_cost = float(transaction_cost)
        self.price_limit = None if price_limit is None else float(price_limit)
        self.max_weight = None if max_weight is None else float(max_weight)

        # Action: vector phân bổ (n_assets + 1 cho cash), tổng = 1 sau chuẩn hoá.
        self.action_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self.n_assets + 1,), dtype=np.float32
        )

        # Observation: giá chuẩn hoá + indicator + tỷ trọng hiện tại — numeric
        # fallback cho B4/debug (xem docstring class).
        obs_dim = self.n_assets * len(_FEATURES) + (self.n_assets + 1)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Giá đóng cửa ngày đầu tiên của từng mã — dùng làm mốc chuẩn hoá giá
        # trong observation (đặt ở __init__ vì cố định trong suốt episode/dataset,
        # tính lại ở reset() để rõ ràng về mặt ngữ nghĩa "đầu episode").
        self._base_close: Dict[str, float] = {}

        self.t: int = 0
        self.cash: float = self.initial_cash
        self.weights: np.ndarray = np.zeros(self.n_assets + 1, dtype=np.float64)
        self.portfolio_value_history: List[float] = [self.initial_cash]
        # Drawdown của bước TRƯỚC — dùng để tính *differential* drawdown
        # (phần drawdown tăng thêm ở bước này) trong reward, thay vì drawdown
        # tuyệt đối (luôn >= 0 nên luôn trừ, biến reward thành "thuế cố định").
        self._prev_drawdown: float = 0.0

        self.reset()

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.t = 0
        self.cash = self.initial_cash
        # Bắt đầu 100% tiền mặt: n_assets phần tử 0 + 1 (cash).
        self.weights = np.array([0.0] * self.n_assets + [1.0], dtype=np.float64)
        self.portfolio_value_history = [self.initial_cash]
        self._prev_drawdown = 0.0

        self._base_close = {
            tk: float(self.state_df.iloc[0][(tk, "close")]) for tk in self.tickers
        }

        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        row = self.state_df.iloc[self.t]
        features: List[float] = []
        for tk in self.tickers:
            base = self._base_close.get(tk, 1.0) or 1.0
            norm_close = float(row[(tk, "close")]) / base
            features += [
                norm_close,
                float(row[(tk, "rsi_14")]),
                float(row[(tk, "macd")]),
                float(row[(tk, "macd_signal")]),
                float(row[(tk, "volatility_20d")]),
            ]
        obs = np.array(features + list(self.weights), dtype=np.float32)
        return obs

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        row = self.state_df.iloc[self.t]

        # Áp action-masking (chặn mã đang chạm biên độ +-price_limit) — logic
        # đầy đủ (clip, normalize, khoá mã trần/sàn) nằm trong action_masking.py.
        action = apply_action_mask(
            action=np.asarray(action, dtype=np.float64),
            state_df=self.state_df,
            t=self.t,
            tickers=self.tickers,
            price_limit=self.price_limit,
            current_weights=self.weights,
        )

        # Áp trần tỷ trọng mỗi mã (De-Cuong §6.2) nếu được cấu hình.
        if self.max_weight is not None:
            action = apply_position_cap(action, max_weight=self.max_weight)

        turnover = float(np.abs(action - self.weights).sum())
        old_value = self.portfolio_value_history[-1]

        # Return danh mục dựa trên giá đóng cửa ngày kế tiếp so với ngày hiện tại.
        self.t += 1
        terminated = self.t >= len(self.state_df) - 1
        next_row = self.state_df.iloc[self.t]

        asset_returns = np.array(
            [
                (float(next_row[(tk, "close")]) - float(row[(tk, "close")]))
                / float(row[(tk, "close")])
                for tk in self.tickers
            ]
            + [0.0],  # cash return = 0
            dtype=np.float64,
        )

        gross_return = float(np.dot(action, asset_returns))
        cost = turnover * self.transaction_cost
        net_return = gross_return - cost
        new_value = old_value * (1.0 + net_return)

        self.portfolio_value_history.append(new_value)
        self.weights = action

        reward = self._compute_reward(net_return, turnover)

        obs = (
            self._get_obs()
            if not terminated
            else np.zeros(self.observation_space.shape, dtype=np.float32)
        )
        info: Dict[str, Any] = {
            "portfolio_value": new_value,
            "turnover": turnover,
            "net_return": net_return,
            "gross_return": gross_return,
            "cost": cost,
        }
        truncated = False
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def _compute_reward(
        self,
        net_return: float,
        turnover: float,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.2,
        window: int = 20,
    ) -> float:
        """Reward = alpha*(net_return*100) + beta*sharpe_local
        - gamma*(delta_drawdown*100) - 0.001*turnover.

        THAY ĐỔI SCALE (so với bản nháp ban đầu) — lý do & cơ sở để ghi vào
        phần phương pháp luận văn:

        1. `net_return * 100` (quy ra PHẦN TRĂM): daily return cổ phiếu ~±1%
           (tức ±0.01) nên ở dạng thô, số hạng lợi nhuận chỉ ~±0.005 — quá
           nhỏ, bị các số hạng khác nhấn chìm, PPO gần như không nhận được
           gradient "danh mục nào sinh lời hơn". Nhân 100 đưa net_return về
           thang %-đơn-vị, tín hiệu học rõ hơn ~100 lần.

        2. Bỏ hệ số `*0.01` ở sharpe: bản nháp nhân sharpe với 0.01 khiến số
           hạng này ~±0.0001 (gần như tắt), mâu thuẫn với ý nghĩa trọng số
           `beta=0.3`. Sharpe cục bộ thường nằm trong ~[-3, 3] nên `beta*sharpe`
           đóng góp ~[-0.9, 0.9] — cùng thang với số hạng net_return, đúng vai
           trò "trọng số" như tên gọi.

        3. DIFFERENTIAL drawdown thay cho drawdown tuyệt đối: drawdown tuyệt
           đối `(peak - current)/peak` LUÔN >= 0, nên `- gamma*drawdown` là một
           "thuế cố định" luôn âm, kéo reward xuống ~-0.002..-0.005 BẤT KỂ agent
           làm gì (đây là lý do reward bị "dính" quanh -0.003 và không cải thiện
           trong log train). Thay bằng `delta_drawdown = drawdown_t -
           drawdown_{t-1}`: chỉ phạt phần drawdown TĂNG THÊM ở bước này; khi
           danh mục hồi phục (drawdown giảm) thì delta âm -> `-gamma*delta` DƯƠNG
           (thưởng). Đây là tín hiệu "reward shaping" trung lập theo thời gian,
           không bias về âm.

        LƯU Ý: alpha/beta/gamma vẫn là placeholder — PHẢI grid-search lại trên
        tập validation theo đề cương (6.5), không dùng cố định cho kết quả cuối.

        Ổn định số học ở episode đầu (history < 3 giá trị): `sharpe_local=0.0`.
        """
        values = np.array(self.portfolio_value_history[-window:], dtype=np.float64)

        if len(values) > 2:
            rets = np.diff(values) / values[:-1]
            sharpe_local = float(rets.mean() / (rets.std() + 1e-8) * np.sqrt(252))
        else:
            sharpe_local = 0.0

        drawdown = 0.0
        if len(values) > 1:
            peak = values.max()
            drawdown = float((peak - values[-1]) / (peak + 1e-8))

        # Differential drawdown: phần drawdown tăng thêm so với bước trước.
        delta_drawdown = drawdown - self._prev_drawdown
        self._prev_drawdown = drawdown

        reward = (
            alpha * (net_return * 100.0)
            + beta * sharpe_local
            - gamma * (delta_drawdown * 100.0)
            - 0.001 * turnover
        )
        return float(reward)


class PortfolioEnvPhaseB(PortfolioEnvPhaseA):
    """Môi trường RL cho Giai đoạn B — tái sử dụng nguyên vẹn logic Giai đoạn A.

    Giai đoạn B chỉ mở rộng rổ tài sản (vd. 10 mã VN30 thay vì 5 mã của Giai
    đoạn A qua tham số `tickers` dài hơn khi khởi tạo) — action-masking, cách
    tính turnover/transaction cost, và reward giữ nguyên như
    `PortfolioEnvPhaseA`, không có logic mới. Class này tồn tại như một tên
    riêng để `src/training/train_phase_b.py` import và để checkpoint/log tách
    bạch rõ ràng giữa 2 giai đoạn.
    """

    pass


class PortfolioEnvUS(PortfolioEnvPhaseA):
    """Môi trường RL cho **thị trường Mỹ** (Dow30/Nasdaq-100) theo đề cương
    `De-Cuong.md` (§6.2).

    Khác biệt duy nhất so với `PortfolioEnvPhaseA` là bộ default đúng với ràng
    buộc thị trường Mỹ:
        - ``price_limit=None``  : Mỹ KHÔNG có biên độ giá hằng ngày.
        - ``max_weight=0.30``   : trần 30%/mã (clip + renormalize sau softmax).
        - ``initial_cash``      : 1.000.000 USD (giá trị tuyệt đối không ảnh
                                  hưởng %-return/Sharpe, chỉ cần cố định).
        - ``transaction_cost``  : 0.0005–0.001 (0.05–0.1%/lệnh, biểu phí phổ
                                  biến broker Mỹ) — mặc định 0.0005, có thể
                                  override.

    Toàn bộ logic reward/turnover/observation tái sử dụng nguyên vẹn từ lớp cha
    (thị-trường-độc-lập). Dùng lớp này cho MỌI thí nghiệm A1/B1/C1 + HIFA theo
    đề cương để đảm bảo cùng một môi trường mô phỏng (De-Cuong §6.2).
    """

    def __init__(
        self,
        state_df: pd.DataFrame,
        tickers: Sequence[str],
        initial_cash: float = 1_000_000.0,
        transaction_cost: float = 0.0005,
        max_weight: Optional[float] = 0.30,
    ) -> None:
        super().__init__(
            state_df=state_df,
            tickers=tickers,
            initial_cash=initial_cash,
            transaction_cost=transaction_cost,
            price_limit=None,  # Mỹ không có biên độ giá
            max_weight=max_weight,
        )
