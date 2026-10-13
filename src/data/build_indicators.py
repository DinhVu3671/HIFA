"""
src/data/build_indicators.py
------------------------------
Bước 2 (mục 3.2 đặc tả HIFA) — tính chỉ báo kỹ thuật (RSI14, MACD,
Bollinger Bands, volatility 20 ngày) bằng `pandas_ta`.

⚠️ CẢNH BÁO QUAN TRỌNG (đã nêu rõ trong đặc tả): các chỉ báo dùng rolling
window (RSI, MACD, Bollinger Bands, volatility) chỉ nhìn về quá khứ nên bản
thân không gây look-ahead bias — NHƯNG nếu tính trên một DataFrame đã trộn
lẫn nhiều mã cổ phiếu (nhiều ticker) mà không tách riêng trước, rolling
window sẽ "lẫn" dữ liệu giữa các mã tại điểm nối giao giữa 2 mã liên tiếp
(vd: rolling(20) tại ngày đầu tiên của mã B sẽ dùng luôn 19 ngày cuối của mã
A). Đây là lỗi nghiêm trọng cần tránh tuyệt đối.

=> `add_indicators()` PHẢI luôn được gọi trên DataFrame chỉ chứa DUY NHẤT một
   mã cổ phiếu. Dùng `add_indicators_multi()` để áp dụng an toàn cho nhiều mã
   (tính riêng từng mã rồi mới gộp lại).
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compatibility shim: các bản pandas_ta cũ (<=0.3.14b, chưa có release mới hỗ
# trợ numpy>=2.0) import `numpy.NaN` (viết hoa) — alias này đã bị xoá khỏi
# numpy>=2.0 (chỉ còn `numpy.nan`). Patch lại TRƯỚC khi import pandas_ta để
# tránh AttributeError khi import.
# ---------------------------------------------------------------------------
if not hasattr(np, "NaN"):
    np.NaN = np.nan  # type: ignore[attr-defined]

import pandas_ta as ta  # noqa: E402  (phải import sau shim ở trên)

EXPECTED_INDICATOR_COLUMNS: List[str] = [
    "rsi_14",
    "macd",
    "macd_signal",
    "bb_upper",
    "bb_lower",
    "volatility_20d",
]


def _first_col_startswith(frame: pd.DataFrame, prefix: str) -> str:
    """Tìm tên cột đầu tiên bắt đầu bằng `prefix`.

    pandas_ta đặt tên cột output kèm theo tham số (vd "BBU_20_2.0"), và cách
    format số thập phân có thể khác nhau giữa các phiên bản pandas_ta
    (vd "2.0" vs "2"). Dò theo prefix để không phụ thuộc cứng vào phiên bản.
    """
    candidates = [c for c in frame.columns if c.startswith(prefix)]
    if not candidates:
        raise KeyError(f"Không tìm thấy cột nào bắt đầu bằng '{prefix}' trong {list(frame.columns)}")
    return candidates[0]


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Tính chỉ báo kỹ thuật cho DataFrame giá của MỘT mã cổ phiếu duy nhất.

    Args:
        df: DataFrame có tối thiểu các cột "date", "close" (và tuỳ chọn
            "ticker" — nếu có, phải chỉ chứa đúng 1 giá trị ticker duy nhất).

    Returns:
        DataFrame đã thêm các cột: rsi_14, macd, macd_signal, bb_upper,
        bb_lower, volatility_20d; đã dropna() các dòng đầu chuỗi bị thiếu do
        rolling window (warm-up period).

    Raises:
        ValueError: nếu phát hiện df chứa nhiều hơn 1 ticker (dấu hiệu gọi
            nhầm hàm này trên dữ liệu đã trộn nhiều mã).
    """
    if df.empty:
        logger.warning("add_indicators() nhận DataFrame rỗng, trả về nguyên trạng.")
        return df.copy()

    if "ticker" in df.columns and df["ticker"].nunique() > 1:
        raise ValueError(
            "add_indicators() chỉ được gọi trên dữ liệu của MỘT mã cổ phiếu duy nhất, "
            f"nhưng nhận được {df['ticker'].nunique()} mã: {sorted(df['ticker'].unique())}. "
            "Dùng add_indicators_multi() cho dữ liệu nhiều mã."
        )
    if "date" not in df.columns or "close" not in df.columns:
        raise ValueError("df cần có tối thiểu 2 cột 'date' và 'close'.")

    df = df.sort_values("date").reset_index(drop=True).copy()

    df["rsi_14"] = ta.rsi(df["close"], length=14)

    macd = ta.macd(df["close"])
    macd_col = _first_col_startswith(macd, "MACD_")
    macds_col = _first_col_startswith(macd, "MACDs_")
    df["macd"] = macd[macd_col]
    df["macd_signal"] = macd[macds_col]

    bb = ta.bbands(df["close"], length=20)
    bbu_col = _first_col_startswith(bb, "BBU_")
    bbl_col = _first_col_startswith(bb, "BBL_")
    df["bb_upper"] = bb[bbu_col]
    df["bb_lower"] = bb[bbl_col]

    df["volatility_20d"] = df["close"].pct_change().rolling(20).std()

    df = df.dropna().reset_index(drop=True)  # loại bỏ NaN đầu chuỗi do rolling window
    return df


def add_indicators_multi(df: pd.DataFrame, ticker_col: str = "ticker") -> pd.DataFrame:
    """Áp dụng `add_indicators` RIÊNG cho từng mã rồi gộp kết quả lại.

    Đây là cách AN TOÀN duy nhất để tính chỉ báo trên dữ liệu nhiều mã — mỗi
    nhóm (group) theo `ticker_col` được xử lý độc lập, tránh rolling window
    lẫn dữ liệu giữa các mã kề nhau.

    Args:
        df: DataFrame chứa nhiều mã, có cột `ticker_col`, "date", "close".
        ticker_col: tên cột định danh mã cổ phiếu.

    Returns:
        DataFrame gộp kết quả add_indicators() của từng mã, sắp xếp theo
        [ticker_col, "date"].
    """
    if ticker_col not in df.columns:
        raise ValueError(f"df không có cột '{ticker_col}'.")

    processed_frames = []
    # Loop tường minh qua từng mã (thay vì groupby().apply()) để tránh các vấn
    # đề tương thích/deprecation của pandas quanh việc group_keys / cột nhóm
    # có được giữ lại trong groupby.apply() hay không giữa các phiên bản.
    for ticker_value, group in df.groupby(ticker_col, sort=False):
        single = group.copy()
        processed = add_indicators(single)
        processed_frames.append(processed)

    if not processed_frames:
        return df.iloc[0:0].copy()

    out = pd.concat(processed_frames, ignore_index=True)
    out = out.sort_values([ticker_col, "date"]).reset_index(drop=True)
    return out
