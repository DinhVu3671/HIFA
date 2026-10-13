"""Action masking cho PortfolioEnvPhaseA / PortfolioEnvPhaseB.

Theo quy định biên độ dao động giá của HOSE (mặc định +-7%), khi một cổ phiếu
chạm giá trần hoặc giá sàn trong phiên, sổ lệnh phía đối ứng thường cạn thanh
khoản — nhà đầu tư trên thực tế không thể chủ động mua thêm (khi trần) hay bán
ra (khi sàn) ở mức giá đó. Module này tách riêng logic "action masking" khỏi
`PortfolioEnvPhaseA` (đặc tả mục 3.3) để có thể unit-test độc lập và tái sử dụng
cho Giai đoạn B.

Format `state_df` kỳ vọng (giống output của `src/data/build_dataset.py::
build_state_dataframe`, KHÔNG import trực tiếp từ đó để tránh phụ thuộc chéo):
    - index: ngày giao dịch, tăng dần theo thời gian, không có ngày trùng/khuyết
      giữa các mã (đã align).
    - columns: `pandas.MultiIndex` 2 cấp `(ticker, feature)`, với tối thiểu
      feature "close" cho mỗi ticker trong danh sách `tickers`.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def apply_action_mask(
    action: np.ndarray,
    state_df: pd.DataFrame,
    t: int,
    tickers: Sequence[str],
    price_limit: Optional[float] = 0.07,
    current_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Chặn mua/bán các mã đang chạm biên độ giá +-``price_limit`` tại ngày ``t``.

    THỊ TRƯỜNG MỸ: Dow30/Nasdaq-100 KHÔNG có cơ chế biên độ giá hằng ngày như
    HOSE. Truyền ``price_limit=None`` để VÔ HIỆU HOÀN TOÀN logic masking theo
    biên độ — khi đó hàm chỉ clip action về [0, +inf) và chuẩn hoá tổng=1 (có
    xét ``current_weights`` nếu được cung cấp), không khoá mã nào. Đây là chế độ
    dùng cho toàn bộ thí nghiệm theo đề cương `De-Cuong.md` (thị trường Mỹ).

    So sánh giá đóng cửa ngày ``t`` với ngày ``t-1`` cho từng mã trong ``tickers``;
    nếu |% thay đổi| >= ``price_limit`` (trừ sai số nhỏ 1e-4 để tránh lỗi số thực
    dấu phẩy động khi giá đúng bằng biên trần/sàn), mã đó bị khoá — phần hành động
    (action) muốn thay đổi tỷ trọng mã đó trong bước hiện tại bị vô hiệu.

    Cổ phần cuối cùng của vector action luôn được hiểu là tiền mặt (cash) và
    KHÔNG BAO GIỜ bị khoá (không có "biên độ giá" cho cash).

    Args:
        action: Vector phân bổ do policy đề xuất, shape ``(n_assets + 1,)`` —
            ``n_assets`` phần tử đầu theo đúng thứ tự ``tickers``, phần tử cuối
            là cash. Không bắt buộc đã chuẩn hoá sẵn (tổng có thể khác 1, có thể
            âm) — hàm tự clip về [0, +inf) rồi chuẩn hoá lại.
        state_df: DataFrame giá/indicator, xem mô tả format ở docstring module.
        t: Vị trí hàng (0-based, theo ``iloc``) trong ``state_df`` ứng với ngày
            ra quyết định hiện tại. Biên độ được tính so với hàng ``t - 1``.
            Nếu ``t == 0`` (không có phiên trước để so sánh) thì không mã nào bị
            coi là chạm biên (toàn bộ mask = 1).
        tickers: Danh sách mã theo đúng thứ tự các phần tử đầu của ``action``.
        price_limit: Biên độ dao động giá tối đa 1 phiên (mặc định 0.07 = +-7%
            theo HOSE).
        current_weights: Tỷ trọng danh mục hiện tại (trước khi áp dụng action ở
            bước này), shape ``(n_assets + 1,)``, nếu có.
            - Nếu được cung cấp (khuyến nghị — đúng bản chất vật lý của việc
              "không khớp được lệnh"): mã bị khoá GIỮ NGUYÊN tỷ trọng hiện tại;
              phần ngân sách còn lại (``1 - tổng tỷ trọng các mã bị khoá``)
              được phân bổ cho các mã không bị khoá theo đúng tỷ lệ mà action đề
              xuất giữa chúng.
            - Nếu ``None`` (mặc định, để tương thích chữ ký đơn giản và cho phép
              gọi hàm mà không cần biết trọng số hiện tại): dùng cách đơn giản
              hoá — đặt phần đề xuất cho mã bị khoá về 0 rồi chuẩn hoá lại toàn
              bộ vector.

    Returns:
        np.ndarray, dtype float32, shape ``(n_assets + 1,)``, tổng xấp xỉ 1
        (trong sai số ~1e-6), đã áp mask.

    Raises:
        ValueError: nếu shape của ``action`` (hoặc ``current_weights``) không
            khớp ``len(tickers) + 1``.
    """
    action = np.asarray(action, dtype=np.float64)
    n_assets = len(tickers)
    expected_shape = n_assets + 1
    if action.shape[0] != expected_shape:
        raise ValueError(
            f"action phải có shape ({expected_shape},) (n_assets={n_assets} mã "
            f"+ 1 cash), nhận được shape {action.shape}"
        )

    # Phòng vệ: action có thể chưa được clip/normalize (logits thô, v.v.)
    action = np.clip(action, 0.0, None)

    mask = np.ones(expected_shape, dtype=np.float64)
    # price_limit=None -> thị trường không có biên độ giá (Mỹ): bỏ qua masking,
    # mọi mã luôn giao dịch được.
    if price_limit is not None and t > 0:
        prev_row = state_df.iloc[t - 1]
        curr_row = state_df.iloc[t]
        for i, tk in enumerate(tickers):
            prev_close = float(prev_row[(tk, "close")])
            curr_close = float(curr_row[(tk, "close")])
            if prev_close == 0.0:
                continue  # tránh chia 0 với dữ liệu hỏng; coi như không mask
            change = (curr_close - prev_close) / prev_close
            if abs(change) >= price_limit - 1e-4:
                mask[i] = 0.0
    # mask[-1] (cash) luôn giữ = 1

    if current_weights is None:
        return _mask_zero_and_renormalize(action, mask)
    return _mask_preserve_locked_weights(action, mask, current_weights)


def _mask_zero_and_renormalize(action: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Đặt phần bị khoá về 0 rồi chuẩn hoá lại toàn vector (tổng = 1)."""
    masked_action = action * mask
    total = masked_action.sum()
    if total <= 1e-8:
        # Toàn bộ ngân sách bị khoá (trường hợp hiếm, vd. mọi mã đều trần/sàn
        # và action không đặt gì vào cash) -> fallback an toàn: giữ 100% cash.
        result = np.zeros_like(action)
        result[-1] = 1.0
        return result.astype(np.float32)
    return (masked_action / total).astype(np.float32)


def _mask_preserve_locked_weights(
    action: np.ndarray, mask: np.ndarray, current_weights: np.ndarray
) -> np.ndarray:
    """Giữ nguyên tỷ trọng hiện tại cho phần bị khoá, phân bổ lại phần còn lại."""
    current_weights = np.asarray(current_weights, dtype=np.float64)
    if current_weights.shape != action.shape:
        raise ValueError(
            "current_weights phải cùng shape với action: "
            f"{current_weights.shape} != {action.shape}"
        )

    locked = mask == 0.0
    free = ~locked

    result = np.zeros_like(action)
    result[locked] = current_weights[locked]
    remaining_budget = max(1.0 - result[locked].sum(), 0.0)

    free_action_sum = action[free].sum()
    if free_action_sum <= 1e-8:
        # Policy không đề xuất gì cho phần "tự do" -> giữ nguyên tỷ trọng hiện
        # tại của phần đó thay vì đặt về 0 (renormalize ở bước cuối xử lý phần
        # còn lại của tổng).
        result[free] = current_weights[free]
    else:
        result[free] = action[free] / free_action_sum * remaining_budget

    total = result.sum()
    if total <= 1e-8:
        result = np.zeros_like(action)
        result[-1] = 1.0
        return result.astype(np.float32)
    return (result / total).astype(np.float32)


def apply_position_cap(weights: np.ndarray, max_weight: float = 0.30) -> np.ndarray:
    """Áp trần tỷ trọng tối đa mỗi mã (De-Cuong §6.2: trần 30%/mã).

    Sau softmax ở Tầng 3, nếu một mã vượt ``max_weight`` thì clip về
    ``max_weight`` rồi phân bổ lại phần dư (``excess``) cho các mã CHƯA chạm
    trần theo đúng tỷ lệ tương đối giữa chúng. Lặp lại cho đến khi không còn mã
    nào vượt trần (hoặc không thể phân bổ thêm). Phần tử cuối (cash) KHÔNG bị
    áp trần — tiền mặt được phép giữ bao nhiêu tuỳ ý.

    Args:
        weights: vector tỷ trọng ``(n_assets + 1,)``, phần tử cuối là cash,
            đã chuẩn hoá tổng ~1 (nếu chưa, hàm tự chuẩn hoá trước).
        max_weight: trần tỷ trọng mỗi mã (mặc định 0.30 = 30%).

    Returns:
        np.ndarray float32, tổng ~1, mọi mã (trừ cash) <= ``max_weight``.

    Ghi chú: nếu ``n_assets * max_weight < 1`` thì về mặt toán học không thể dồn
    hết 100% vào các mã (phần dư bắt buộc chảy vào cash) — đây là hành vi đúng,
    không phải lỗi.
    """
    w = np.asarray(weights, dtype=np.float64).copy()
    if w.shape[0] < 2:
        return w.astype(np.float32)

    total = w.sum()
    if total > 1e-12:
        w = w / total

    n_assets = w.shape[0] - 1  # phần tử cuối là cash
    asset_idx = np.arange(n_assets)

    # Lặp phân bổ lại phần vượt trần. Số vòng tối đa = n_assets (hội tụ chắc chắn).
    for _ in range(n_assets + 1):
        over = asset_idx[w[asset_idx] > max_weight + 1e-9]
        if over.size == 0:
            break
        excess = float((w[over] - max_weight).sum())
        w[over] = max_weight
        # Các mã chưa chạm trần (còn "chỗ" để nhận thêm).
        under = asset_idx[w[asset_idx] < max_weight - 1e-9]
        if under.size == 0 or excess <= 1e-12:
            # Không còn mã nào nhận thêm được -> phần dư chảy vào cash.
            w[-1] += excess
            break
        under_sum = float(w[under].sum())
        if under_sum <= 1e-12:
            # Các mã under đều ~0 -> chia đều phần dư cho chúng.
            w[under] += excess / under.size
        else:
            w[under] += excess * (w[under] / under_sum)

    # Chuẩn hoá lại lần cuối cho chắc (sai số số thực).
    s = w.sum()
    if s > 1e-12:
        w = w / s
    return w.astype(np.float32)
