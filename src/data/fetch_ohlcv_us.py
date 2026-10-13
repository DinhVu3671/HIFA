"""
src/data/fetch_ohlcv_us.py
---------------------------
Thu thập dữ liệu giá OHLCV cho rổ cổ phiếu **thị trường Mỹ** (Dow Jones 30 /
Nasdaq-100) qua thư viện `yfinance`, theo đúng đề cương `De-Cuong.md` (mục 6.1).

Đây là phiên bản thay thế `fetch_ohlcv.py` (vốn dùng `vnstock`/VN30). Module
giữ NGUYÊN schema đầu ra `date, open, high, low, close, volume, ticker` để toàn
bộ pipeline downstream (`build_indicators.py`, `build_dataset.py`,
`portfolio_env.py`, baselines, training) tái sử dụng được không cần sửa.

QUAN TRỌNG: module KHÔNG gọi network khi chỉ import. Mọi hành vi mạng chỉ xảy
ra khi gọi `fetch_ohlcv()` / `fetch_all()` hoặc chạy CLI
(`python -m src.data.fetch_ohlcv_us --phase 0`).

Tham chiếu dữ liệu (De-Cuong §6.1, §6.4.3; Implementation Plan §0.1):
    - Giai đoạn 0 & A (8 mã, dùng chung, không đổi giữa 2 giai đoạn):
        AAPL, MSFT, JPM, JNJ, XOM, PG, KO, NVDA
    - Giai đoạn B (16 mã = 8 mã cũ + 8 mã mới):
        + GOOGL, AMZN, V, HD, DIS, CSCO, MRK, CVX
    - Benchmark chỉ số thật: ^DJI (Dow Jones), ^NDX (Nasdaq-100).
    - Khoảng fetch: 2015-01-01 → 2025-12-31 (dư để indicator warm-up và để
      Giai đoạn B có nhiều cửa sổ walk-forward).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Danh mục mã cổ phiếu theo từng giai đoạn (De-Cuong §6.4.3 / Plan §0.1.2)
# ---------------------------------------------------------------------------
# Giai đoạn 0 và A dùng chung 8 mã, đa dạng ngành (công nghệ, tài chính, y tế,
# năng lượng, tiêu dùng) trong Dow30/Nasdaq-100.
TICKERS_PHASE_0: List[str] = ["AAPL", "MSFT", "JPM", "JNJ", "XOM", "PG", "KO", "NVDA"]
TICKERS_PHASE_A: List[str] = TICKERS_PHASE_0  # alias: Giai đoạn A dùng y hệt bộ 8 mã

# Giai đoạn B mở rộng lên 16 mã: giữ nguyên 8 mã cũ + thêm 8 mã mới.
TICKERS_PHASE_B: List[str] = TICKERS_PHASE_0 + [
    "GOOGL",
    "AMZN",
    "V",
    "HD",
    "DIS",
    "CSCO",
    "MRK",
    "CVX",
]

# Chỉ số tham chiếu (benchmark) — dùng cho baseline B1 (Index) thay vì proxy
# equal-weight. yfinance nhận ký hiệu index với tiền tố '^'.
BENCHMARK_TICKERS = {"dow": "^DJI", "nasdaq100": "^NDX"}

DEFAULT_START = "2015-01-01"
DEFAULT_END = "2025-12-31"

# Thư mục gốc của repo (src/data/fetch_ohlcv_us.py -> parents[2] = repo root)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OHLCV_DIR = REPO_ROOT / "data" / "raw" / "ohlcv"

_REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume", "ticker"]


def _flatten_yf_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Chuẩn hoá output của yfinance về schema phẳng chuẩn của dự án.

    `yfinance.download` có thể trả MultiIndex columns (khi tải nhiều mã) hoặc
    cột đơn (1 mã). Hàm này xử lý cả hai, rồi đổi tên về
    date/open/high/low/close/volume + thêm cột ticker.

    Ưu tiên dùng giá đã điều chỉnh cổ tức/chia tách: khi gọi với
    `auto_adjust=True` (mặc định mới của yfinance), cột 'Close' CHÍNH LÀ giá đã
    điều chỉnh — phù hợp cho backtest (tránh nhảy giá giả do chia tách).
    """
    out = df.copy()

    # Nếu là MultiIndex columns (level 0 = field, level 1 = ticker), rút gọn
    # về 1 mã bằng cách lấy cross-section theo ticker.
    if isinstance(out.columns, pd.MultiIndex):
        # yfinance có thể đặt ticker ở level 0 hoặc level 1 tuỳ phiên bản.
        if ticker in out.columns.get_level_values(-1):
            out = out.xs(ticker, axis=1, level=-1)
        elif ticker in out.columns.get_level_values(0):
            out = out.xs(ticker, axis=1, level=0)
        else:
            # Chỉ còn 1 mã -> gộp bỏ level thừa.
            out.columns = out.columns.get_level_values(0)

    out = out.reset_index()  # đưa index ngày (Date) thành cột

    rename_map = {
        "Date": "date",
        "Datetime": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    out = out.rename(columns={c: rename_map.get(c, str(c).lower()) for c in out.columns})

    out["ticker"] = ticker
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    out = out.reset_index(drop=True)

    missing = [c for c in _REQUIRED_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(
            f"Dữ liệu yfinance cho {ticker} thiếu cột {missing}; "
            f"cột nhận được: {list(out.columns)}"
        )
    return out[_REQUIRED_COLUMNS]


def fetch_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Lấy OHLCV hằng ngày cho MỘT mã (hoặc index) qua yfinance.

    Args:
        ticker: mã cổ phiếu, vd "AAPL", hoặc index "^DJI".
        start, end: "YYYY-MM-DD".

    Returns:
        DataFrame cột: date, open, high, low, close, volume, ticker.
        Giá là giá đã auto-adjust (chia tách/cổ tức) — phù hợp backtest.
    """
    # Lazy-import: tránh khởi tạo nặng khi module chỉ được import để lấy hằng số.
    import yfinance as yf

    logger.info("Fetching (yfinance) OHLCV cho %s từ %s đến %s", ticker, start, end)
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,  # Close = giá điều chỉnh; tránh nhảy giá giả do split
        progress=False,
        threads=False,
    )

    if df is None or len(df) == 0:
        logger.warning("Không có dữ liệu yfinance cho %s (%s -> %s)", ticker, start, end)
        return pd.DataFrame(columns=_REQUIRED_COLUMNS)

    return _flatten_yf_columns(df, ticker)


def fetch_all(
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    tickers: Optional[Iterable[str]] = None,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch OHLCV cho nhiều mã, gộp và lưu parquet.

    Args:
        start, end: khoảng thời gian.
        tickers: danh sách mã; mặc định TICKERS_PHASE_0 (8 mã Giai đoạn 0/A).
        output_path: parquet đầu ra; mặc định data/raw/ohlcv/phase_0.parquet.

    Returns:
        DataFrame gộp toàn bộ mã.
    """
    if tickers is None:
        tickers = TICKERS_PHASE_0
    tickers = list(tickers)

    frames = []
    failed = []
    for t in tickers:
        try:
            frame = fetch_ohlcv(t, start, end)
            if len(frame) == 0:
                failed.append(t)
            else:
                frames.append(frame)
        except Exception:  # noqa: BLE001 - tiếp tục các mã còn lại nếu 1 mã lỗi
            logger.exception("Lỗi khi fetch %s, bỏ qua mã này", t)
            failed.append(t)

    if failed:
        logger.warning("Các mã fetch thất bại/rỗng: %s", failed)

    if not frames:
        raise RuntimeError("Không fetch được dữ liệu cho bất kỳ mã nào.")

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)

    counts = out.groupby("ticker")["date"].count()
    logger.info("Số phiên giao dịch theo từng mã:\n%s", counts.to_string())

    if output_path is None:
        output_path = DEFAULT_OHLCV_DIR / "phase_0.parquet"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path)
    logger.info("Đã lưu %d dòng vào %s", len(out), output_path)

    return out


def fetch_benchmark(
    which: str = "dow",
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch chỉ số tham chiếu thật (^DJI hoặc ^NDX) cho baseline B1.

    Args:
        which: "dow" (^DJI) hoặc "nasdaq100" (^NDX).
        start, end: khoảng thời gian.
        output_path: parquet đầu ra; mặc định data/raw/ohlcv/benchmark_{which}.parquet.

    Returns:
        DataFrame OHLCV của chỉ số (cột ticker = ký hiệu index).
    """
    if which not in BENCHMARK_TICKERS:
        raise ValueError(f"'which' phải thuộc {list(BENCHMARK_TICKERS)}, nhận: {which}")
    symbol = BENCHMARK_TICKERS[which]
    df = fetch_ohlcv(symbol, start, end)

    if output_path is None:
        output_path = DEFAULT_OHLCV_DIR / f"benchmark_{which}.parquet"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path)
    logger.info("Đã lưu benchmark %s (%d dòng) vào %s", symbol, len(df), output_path)
    return df


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch US-market OHLCV (Dow30/Nasdaq-100) for HIFA via yfinance."
    )
    parser.add_argument(
        "--phase",
        choices=["0", "a", "b"],
        default="0",
        help="Phase 0/A (8 tickers) or B (16 tickers).",
    )
    parser.add_argument("--start", default=DEFAULT_START, help="Start date YYYY-MM-DD.")
    parser.add_argument("--end", default=DEFAULT_END, help="End date YYYY-MM-DD.")
    parser.add_argument(
        "--benchmark",
        choices=["dow", "nasdaq100", "both", "none"],
        default="both",
        help="Also fetch benchmark index (^DJI / ^NDX).",
    )
    return parser


if __name__ == "__main__":
    import sys

    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args = _build_arg_parser().parse_args()
    if args.phase in ("0", "a"):
        phase_tickers = TICKERS_PHASE_0
        out_name = "phase_0.parquet" if args.phase == "0" else "phase_a.parquet"
    else:
        phase_tickers = TICKERS_PHASE_B
        out_name = "phase_b.parquet"
    out_path = DEFAULT_OHLCV_DIR / out_name

    fetch_all(start=args.start, end=args.end, tickers=phase_tickers, output_path=out_path)

    if args.benchmark in ("dow", "both"):
        fetch_benchmark("dow", start=args.start, end=args.end)
    if args.benchmark in ("nasdaq100", "both"):
        fetch_benchmark("nasdaq100", start=args.start, end=args.end)
