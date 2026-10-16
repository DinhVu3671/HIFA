"""
src/data/build_indicators.py
------------------------------
Bước 2 (mục 3.2 đặc tả HIFA) — tính chỉ báo kỹ thuật (RSI14, MACD,
Bollinger Bands, volatility 20 ngày) bằng PANDAS THUẦN (không phụ thuộc
`pandas_ta`).

Vì sao bỏ `pandas_ta`?
    `pandas_ta==0.3.14b0` (bản mới nhất trên PyPI) không build được với
    `pip>=24.1` (lỗi `setup.py egg_info`) và chưa tương thích numpy>=2.0 /
    pandas 3.x. Trên Colab (pip/pandas luôn mới) việc ghim ngược phiên bản rất
    chậm và dễ gãy. 3 chỉ báo dự án dùng (RSI, MACD, Bollinger) đều tính được
    bằng pandas trong vài dòng, nên ta tự implement để chạy trên MỌI phiên bản
    pandas/numpy mới nhất mà KHÔNG đổi tên cột output hay ý nghĩa chỉ báo.

Công thức khớp với mặc định pandas_ta để số liệu nhất quán với kết quả đã
chạy trước đó:
    - RSI(14): Wilder's smoothing — EMA với alpha = 1/length (adjust=False).
    - MACD(12, 26, 9): EMA(12) - EMA(26); signal = EMA(9) của MACD; EMA
      dùng adjust=False (giống pandas_ta).
    - Bollinger(length=20, std=2.0): SMA(20) ± 2 * rolling std(ddof=0)
      (pandas_ta dùng độ lệch chuẩn population, ddof=0).
    - volatility_20d: rolling std(20) của pct_change() (giữ nguyên như cũ).

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

EXPECTED_INDICATOR_COLUMNS: List[str] = [
    "rsi_14",
    "macd",
    "macd_signal",
    "bb_upper",
    "bb_lower",
    "volatility_20d",
]


# ---------------------------------------------------------------------------
# Các chỉ báo thuần pandas (thay cho pandas_ta)
# ---------------------------------------------------------------------------
def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """RSI theo Wilder (khớp mặc định pandas_ta.rsi).

    Wilder's smoothing tương đương EMA với alpha = 1/length, adjust=False.
    `min_periods=length` để các dòng warm-up đầu chuỗi là NaN (sau đó bị
    dropna() ở add_indicators) — giữ đúng hành vi của pandas_ta.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # avg_loss == 0 -> rs = inf -> rsi = 100 (toàn bộ kỳ tăng giá). Dùng 100.0.
    rsi = rsi.where(avg_loss != 0.0, 100.0)
    return rsi


def _macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """MACD line + signal line (khớp mặc định pandas_ta.macd, adjust=False)."""
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": macd_line, "macd_signal": signal_line})


def _bbands(
    close: pd.Series, length: int = 20, n_std: float = 2.0
) -> pd.DataFrame:
    """Bollinger Bands upper/lower (khớp mặc định pandas_ta.bbands).

    pandas_ta dùng độ lệch chuẩn population (ddof=0).
    """
    mid = close.rolling(window=length, min_periods=length).mean()
    std = close.rolling(window=length, min_periods=length).std(ddof=0)
    upper = mid + n_std * std
    lower = mid - n_std * std
    return pd.DataFrame({"bb_upper": upper, "bb_lower": lower})


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

    df["rsi_14"] = _rsi(df["close"], length=14)

    macd = _macd(df["close"], fast=12, slow=26, signal=9)
    df["macd"] = macd["macd"]
    df["macd_signal"] = macd["macd_signal"]

    bb = _bbands(df["close"], length=20, n_std=2.0)
    df["bb_upper"] = bb["bb_upper"]
    df["bb_lower"] = bb["bb_lower"]

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
