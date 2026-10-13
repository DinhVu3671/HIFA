"""
src/data/walk_forward_split.py
---------------------------------
Bước 5 (mục 3.2 đặc tả HIFA) — walk-forward split.

- `make_walk_forward_windows`: tạo NHIỀU cửa sổ (train, val, test) trượt dần
  theo thời gian — dùng cho Giai đoạn B.
- `single_window_phase_a`: trả về ĐÚNG 1 cửa sổ cố định (train 2019-2022,
  val 2023 H1, test 2023 H2) — dùng cho Giai đoạn A để tiết kiệm thời gian
  (đặc tả: "Giai đoạn A chỉ cần 1 cửa sổ duy nhất").
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Tuple

import pandas as pd
from pandas.tseries.offsets import DateOffset

logger = logging.getLogger(__name__)

DateRange = Tuple[pd.Timestamp, pd.Timestamp]
Window = Tuple[DateRange, DateRange, DateRange]

# Cửa sổ cố định của Giai đoạn A, theo đúng mục 3.2 bước 5 của đặc tả.
PHASE_A_TRAIN_RANGE: DateRange = (pd.Timestamp("2019-01-01"), pd.Timestamp("2022-12-31"))
PHASE_A_VAL_RANGE: DateRange = (pd.Timestamp("2023-01-01"), pd.Timestamp("2023-06-30"))
PHASE_A_TEST_RANGE: DateRange = (pd.Timestamp("2023-07-01"), pd.Timestamp("2023-12-31"))


def _to_sorted_unique_dates(dates: Iterable) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(pd.Index(dates).unique())).sort_values()
    return idx


def _has_data_in_range(idx: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    if len(idx) == 0:
        return False
    return bool(((idx >= start) & (idx <= end)).any())


def make_walk_forward_windows(
    dates: Iterable,
    train_years: int = 3,
    val_months: int = 6,
    test_months: int = 6,
    step_months: int = 6,
) -> List[Window]:
    """Tạo danh sách các cửa sổ (train_range, val_range, test_range) trượt dần.

    Mỗi range là tuple (start_date, end_date) dạng pd.Timestamp, LIÊN TIẾP và
    KHÔNG CHỒNG LẤN trong cùng 1 cửa sổ: train_range kết thúc ngay trước khi
    val_range bắt đầu, val_range kết thúc ngay trước khi test_range bắt đầu.
    Cửa sổ kế tiếp có train_start trượt tới thêm `step_months` tháng so với
    cửa sổ trước — dùng cho walk-forward validation ở Giai đoạn B.

    Args:
        dates: iterable các ngày giao dịch thực tế (dùng để xác định biên dữ
            liệu đầu/cuối và loại bỏ các cửa sổ không có dữ liệu thực).
        train_years: độ dài đoạn train tính theo năm.
        val_months: độ dài đoạn validation tính theo tháng.
        test_months: độ dài đoạn test tính theo tháng.
        step_months: bước trượt giữa 2 cửa sổ liên tiếp, tính theo tháng.

    Returns:
        List[Window] — danh sách rỗng nếu không đủ dữ liệu cho dù chỉ 1 cửa sổ.
    """
    idx = _to_sorted_unique_dates(dates)
    if len(idx) == 0:
        return []

    data_start, data_end = idx[0], idx[-1]

    windows: List[Window] = []
    train_start = data_start

    while True:
        train_end = train_start + DateOffset(years=train_years) - pd.Timedelta(days=1)
        val_start = train_end + pd.Timedelta(days=1)
        val_end = val_start + DateOffset(months=val_months) - pd.Timedelta(days=1)
        test_start = val_end + pd.Timedelta(days=1)
        test_end = test_start + DateOffset(months=test_months) - pd.Timedelta(days=1)

        if test_end > data_end:
            break

        if (
            _has_data_in_range(idx, train_start, train_end)
            and _has_data_in_range(idx, val_start, val_end)
            and _has_data_in_range(idx, test_start, test_end)
        ):
            windows.append(
                ((train_start, train_end), (val_start, val_end), (test_start, test_end))
            )
        else:
            logger.debug(
                "Bỏ qua cửa sổ train_start=%s do thiếu dữ liệu thực tế trong 1 trong 3 đoạn.",
                train_start,
            )

        train_start = train_start + DateOffset(months=step_months)

    return windows


def single_window_phase_a(dates: Optional[Iterable] = None) -> Window:
    """Trả về ĐÚNG 1 cửa sổ cố định cho Giai đoạn A (không trượt cửa sổ).

    Theo mục 3.2 bước 5 của đặc tả: Giai đoạn A chỉ cần 1 cửa sổ duy nhất để
    tiết kiệm thời gian — walk-forward đầy đủ nhiều cửa sổ (make_walk_forward_windows)
    để dành cho Giai đoạn B.

    Args:
        dates: (tuỳ chọn) iterable các ngày giao dịch thực tế, chỉ dùng để
            cảnh báo (log warning) nếu 1 trong 3 đoạn cố định không có dữ
            liệu thực tế nào — không làm thay đổi giá trị trả về.

    Returns:
        Window = (train_range, val_range, test_range) với:
            train_range = (2019-01-01, 2022-12-31)
            val_range   = (2023-01-01, 2023-06-30)
            test_range  = (2023-07-01, 2023-12-31)
    """
    if dates is not None:
        idx = _to_sorted_unique_dates(dates)
        for name, (start, end) in (
            ("train", PHASE_A_TRAIN_RANGE),
            ("val", PHASE_A_VAL_RANGE),
            ("test", PHASE_A_TEST_RANGE),
        ):
            if not _has_data_in_range(idx, start, end):
                logger.warning(
                    "Không có ngày giao dịch nào trong đoạn %s (%s -> %s) của cửa sổ Giai đoạn A cố định.",
                    name, start.date(), end.date(),
                )

    return (PHASE_A_TRAIN_RANGE, PHASE_A_VAL_RANGE, PHASE_A_TEST_RANGE)
