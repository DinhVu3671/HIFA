"""
src/data/build_dataset.py
----------------------------
Bước 4 (mục 3.2 đặc tả HIFA) — ghép OHLCV + chỉ báo kỹ thuật + sentiment
placeholder thành 1 dataset "state" chuẩn cho môi trường RL.

Output: 1 DataFrame với index = date, columns = MultiIndex (ticker, feature).
Feature tối thiểu Giai đoạn A: close, rsi_14, macd, macd_signal,
volatility_20d, sentiment (placeholder = 0.0 ở Giai đoạn A).

Việc align dữ liệu giữa nhiều mã dùng nguyên tắc INNER JOIN theo ngày giao
dịch chung — KHÔNG giả định index liên tục (lịch nghỉ lễ VN khác Mỹ/các thị
trường khác, và mỗi mã có thể có warm-up period dropna() khác nhau tuỳ ngày
niêm yết dữ liệu bắt đầu).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Union

import pandas as pd

from .build_indicators import add_indicators_multi

logger = logging.getLogger(__name__)


def get_sentiment_placeholder(ticker: str, date) -> float:
    """Sentiment placeholder = 0.0 (neutral) cho Giai đoạn 0/A.

    Giai đoạn 0/A chưa dùng tín hiệu sentiment số học — cột `sentiment` trong
    state_df để = 0 cho mọi mã/ngày. Sentiment thật (từ tin tức Yahoo Finance
    RSS / NewsAPI) chỉ đưa vào từ Giai đoạn B qua prompt của Sentiment Agent,
    không ghi đè cột số học này (De-Cuong §6.1, Plan §0.1.1).
    """
    return 0.0

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_DIR = REPO_ROOT / "data" / "processed"

# Feature tối thiểu Giai đoạn A theo mục 3.2 bước 4 của đặc tả.
FEATURE_COLUMNS: List[str] = [
    "close",
    "rsi_14",
    "macd",
    "macd_signal",
    "volatility_20d",
    "sentiment",
]


def build_state_dataframe(
    tickers: Iterable[str],
    ohlcv_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Đọc parquet OHLCV thô, tính chỉ báo, và pivot thành state dataframe.

    Args:
        tickers: danh sách mã cần đưa vào dataset (thứ tự này được giữ lại
            trong MultiIndex columns của kết quả).
        ohlcv_path: đường dẫn tới file parquet OHLCV thô (cột tối thiểu:
            date, open, high, low, close, volume, ticker) — thường là output
            của `fetch_all()` trong fetch_ohlcv.py. Chỉ báo kỹ thuật sẽ được
            TÍNH RIÊNG cho từng mã ngay trong hàm này (qua add_indicators_multi)
            để tránh lỗi rolling window lẫn dữ liệu giữa các mã.
        output_path: nếu truyền vào, lưu kết quả ra parquet tại đường dẫn này
            (cột MultiIndex sẽ được "flatten" thành chuỗi "TICKER__feature"
            vì định dạng parquet không hỗ trợ MultiIndex columns).

    Returns:
        DataFrame index = date (DatetimeIndex, tăng dần), columns =
        MultiIndex (ticker, feature), chỉ giữ lại các ngày giao dịch chung
        (inner join) của TẤT CẢ các mã trong `tickers`.
    """
    tickers = list(tickers)
    if not tickers:
        raise ValueError("tickers không được rỗng.")

    ohlcv_path = Path(ohlcv_path)
    if not ohlcv_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file OHLCV: {ohlcv_path}")

    raw = pd.read_parquet(ohlcv_path)
    if "date" not in raw.columns or "ticker" not in raw.columns:
        raise ValueError("File OHLCV cần có tối thiểu 2 cột 'date' và 'ticker'.")

    raw = raw.copy()
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw[raw["ticker"].isin(tickers)]

    present = set(raw["ticker"].unique())
    missing = [t for t in tickers if t not in present]
    if missing:
        logger.warning(
            "Thiếu dữ liệu OHLCV cho các mã sau (sẽ khiến toàn bộ dataset "
            "rỗng sau inner-join nếu không được loại bỏ khỏi `tickers`): %s",
            missing,
        )

    # Tính chỉ báo RIÊNG cho từng mã (an toàn, không lẫn dữ liệu) rồi mới gộp.
    indicators_df = add_indicators_multi(raw, ticker_col="ticker")

    # Giai đoạn 0/A: sentiment = placeholder 0.0 cho mọi mã/ngày.
    indicators_df["sentiment"] = [
        get_sentiment_placeholder(t, d)
        for t, d in zip(indicators_df["ticker"], indicators_df["date"])
    ]

    missing_features = [c for c in FEATURE_COLUMNS if c not in indicators_df.columns]
    if missing_features:
        raise ValueError(f"Thiếu cột feature bắt buộc sau khi tính indicator: {missing_features}")

    # pivot_table: mặc định pandas sẽ tạo MultiIndex columns dạng (feature, ticker)
    # khi values là danh sách nhiều cột — ta sẽ swaplevel về (ticker, feature)
    # ngay sau đó theo đúng yêu cầu của đặc tả.
    wide = indicators_df.pivot_table(index="date", columns="ticker", values=FEATURE_COLUMNS)
    wide = wide.swaplevel(axis=1).sort_index(axis=1)
    wide.columns = wide.columns.set_names(["ticker", "feature"])

    # Giữ đúng thứ tự tickers như tham số đầu vào (nếu ticker nào không có dữ
    # liệu, cột của nó sẽ toàn NaN và bị loại ở bước dropna inner-join dưới đây).
    desired_columns = pd.MultiIndex.from_product([tickers, FEATURE_COLUMNS], names=["ticker", "feature"])
    wide = wide.reindex(columns=desired_columns)

    # INNER JOIN theo ngày giao dịch chung: loại các ngày mà BẤT KỲ mã nào
    # thiếu dữ liệu (không giả định index liên tục — lịch nghỉ lễ khác nhau,
    # ngày niêm yết dữ liệu khác nhau, warm-up period của indicator khác nhau).
    n_before = len(wide)
    wide = wide.dropna(how="any")
    n_after = len(wide)
    if n_after < n_before:
        logger.info(
            "Inner-join theo ngày giao dịch chung: giữ lại %d/%d ngày.",
            n_after, n_before,
        )
    if n_after == 0:
        raise ValueError(
            "Không tìm thấy ngày giao dịch chung nào giữa các mã "
            f"{tickers} — kiểm tra lại dữ liệu đầu vào (thiếu mã, lệch khoảng thời gian...)."
        )

    wide = wide.sort_index()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        flat = wide.copy()
        # parquet không hỗ trợ MultiIndex columns trực tiếp -> flatten tên cột.
        flat.columns = [f"{tk}__{feat}" for tk, feat in flat.columns]
        flat.to_parquet(output_path)
        logger.info("Đã lưu state dataframe (%d dòng) vào %s", len(flat), output_path)

    return wide


def load_state_dataframe(path: Union[str, Path]) -> pd.DataFrame:
    """Đọc lại state dataframe đã lưu bởi `build_state_dataframe(..., output_path=...)`.

    Khôi phục MultiIndex(ticker, feature) từ tên cột phẳng "TICKER__feature"
    (parquet không hỗ trợ MultiIndex columns trực tiếp, xem docstring
    `build_state_dataframe`). Đây là điểm vào chuẩn để mọi module khác
    (env, baselines, training, evaluation) load state dataframe từ đĩa.
    """
    path = Path(path)
    flat = pd.read_parquet(path)
    columns = [tuple(c.split("__", 1)) for c in flat.columns]
    flat.columns = pd.MultiIndex.from_tuples(columns, names=["ticker", "feature"])
    return flat.sort_index(axis=1)


def _build_arg_parser() -> argparse.ArgumentParser:
    # Luu y: giu help/description text thuan ASCII de tranh UnicodeEncodeError
    # khi in --help tren console Windows dung codepage khong phai UTF-8 (vd
    # cp1252 qua Git Bash); giai thich chi tiet (tieng Viet co dau) o docstring.
    parser = argparse.ArgumentParser(
        description="Merge US OHLCV + indicators + sentiment placeholder into a state dataframe."
    )
    parser.add_argument(
        "--phase", choices=["0", "a", "b"], default="0",
        help="Phase 0/A (8 US tickers) or B (16 US tickers).",
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Rổ mã thị trường Mỹ (Dow30/Nasdaq-100) — xem src/data/fetch_ohlcv_us.py.
    from .fetch_ohlcv_us import TICKERS_PHASE_0, TICKERS_PHASE_B

    args = _build_arg_parser().parse_args()
    phase_tickers = TICKERS_PHASE_0 if args.phase in ("0", "a") else TICKERS_PHASE_B

    raw_name = "phase_0.parquet" if args.phase in ("0", "a") else "phase_b.parquet"
    out_name = f"phase_{args.phase}.parquet"
    in_path = REPO_ROOT / "data" / "raw" / "ohlcv" / raw_name
    out_path = DEFAULT_PROCESSED_DIR / out_name

    build_state_dataframe(phase_tickers, in_path, out_path)
