"""
tests/test_data_pipeline.py
------------------------------
Test cho pipeline dữ liệu (mục 3.2 đặc tả HIFA), dùng TOÀN BỘ dữ liệu giả
lập (synthetic) — KHÔNG cần mạng / vnstock — để kiểm tra:

1. add_indicators(): không NaN sau xử lý, đủ cột kỳ vọng.
2. Không look-ahead bias: feature tại ngày t chỉ phụ thuộc dữ liệu <= t.
3. add_indicators_multi() tính RIÊNG từng ticker (không lẫn dữ liệu giữa các
   mã) — đây là test quan trọng nhất theo cảnh báo của đặc tả.
4. build_state_dataframe(): align đúng theo ngày giao dịch chung (inner join)
   giữa các mã có khoảng ngày lệch nhau.
5. make_walk_forward_windows(): thứ tự train < val < test, không chồng lấn,
   cửa sổ trượt đúng hướng.
6. single_window_phase_a(): đúng 3 khoảng cố định theo đặc tả.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.build_indicators import (
    EXPECTED_INDICATOR_COLUMNS,
    add_indicators,
    add_indicators_multi,
)
from src.data.build_dataset import (
    FEATURE_COLUMNS,
    build_state_dataframe,
    get_sentiment_placeholder,
)
from src.data.walk_forward_split import (
    make_walk_forward_windows,
    single_window_phase_a,
)


# ---------------------------------------------------------------------------
# Helpers dữ liệu giả lập
# ---------------------------------------------------------------------------
def make_synthetic_ohlcv(
    ticker: str,
    start: str = "2020-01-01",
    periods: int = 150,
    seed: int = 0,
    base_price: float = 100.0,
) -> pd.DataFrame:
    """Tạo DataFrame OHLCV giả lập cho MỘT mã, dùng random-walk giá đóng cửa."""
    dates = pd.bdate_range(start=start, periods=periods)
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=0.0003, scale=0.015, size=periods)
    close = base_price * np.cumprod(1 + returns)
    open_ = close * (1 + rng.normal(0, 0.002, size=periods))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, size=periods)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, size=periods)))
    volume = rng.integers(100_000, 1_000_000, size=periods)

    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "ticker": ticker,
        }
    )


MIN_WARMUP_ROWS = 40  # đủ lớn hơn warm-up của MACD(12,26,9)/BBands(20)/RSI(14)/vol(20)


# ---------------------------------------------------------------------------
# 1. add_indicators: không NaN, đủ cột kỳ vọng
# ---------------------------------------------------------------------------
class TestAddIndicators:
    def test_no_nan_after_processing(self):
        df = make_synthetic_ohlcv("AAA", periods=150, seed=1)
        result = add_indicators(df)

        assert len(result) > 0
        assert not result.isna().any().any(), "Không được có NaN sau add_indicators()"

    def test_has_expected_columns(self):
        df = make_synthetic_ohlcv("AAA", periods=150, seed=1)
        result = add_indicators(df)

        for col in EXPECTED_INDICATOR_COLUMNS:
            assert col in result.columns, f"Thiếu cột chỉ báo '{col}'"
        # cột gốc vẫn còn
        for col in ["date", "close", "ticker"]:
            assert col in result.columns

    def test_drops_warmup_rows_only(self):
        df = make_synthetic_ohlcv("AAA", periods=150, seed=1)
        result = add_indicators(df)

        # dropna() chỉ loại bỏ các dòng đầu chuỗi (warm-up), không loại giữa chuỗi
        assert len(result) < len(df)
        assert len(result) > len(df) - MIN_WARMUP_ROWS * 2  # sanity: không bị mất quá nhiều dòng
        # các ngày còn lại phải là 1 dải liên tục ở "đuôi" của df gốc
        expected_tail_dates = df["date"].iloc[-len(result):].reset_index(drop=True)
        pd.testing.assert_series_equal(
            result["date"].reset_index(drop=True), expected_tail_dates, check_names=False
        )

    def test_raises_on_multi_ticker_input(self):
        df_a = make_synthetic_ohlcv("AAA", periods=60, seed=1)
        df_b = make_synthetic_ohlcv("BBB", periods=60, seed=2)
        combined = pd.concat([df_a, df_b], ignore_index=True)

        with pytest.raises(ValueError):
            add_indicators(combined)


# ---------------------------------------------------------------------------
# 2. Không look-ahead bias: feature tại ngày t chỉ phụ thuộc dữ liệu <= t
# ---------------------------------------------------------------------------
class TestNoLookaheadBias:
    def test_future_price_changes_do_not_affect_past_features(self):
        df = make_synthetic_ohlcv("AAA", periods=120, seed=7)

        cutoff_idx = 90  # đủ xa warm-up (~35 dòng) để có feature hợp lệ tại đây
        cutoff_date = df["date"].iloc[cutoff_idx]

        result_original = add_indicators(df)
        row_original = result_original.loc[result_original["date"] == cutoff_date]
        assert len(row_original) == 1
        row_original = row_original.iloc[0]

        # Thay đổi MẠNH toàn bộ giá SAU cutoff_date (tương lai so với t=cutoff)
        df_modified = df.copy()
        future_mask = df_modified["date"] > cutoff_date
        df_modified.loc[future_mask, "close"] = (
            df_modified.loc[future_mask, "close"] * 5.0 + 10_000
        )
        df_modified.loc[future_mask, "open"] = df_modified.loc[future_mask, "close"]
        df_modified.loc[future_mask, "high"] = df_modified.loc[future_mask, "close"] * 1.02
        df_modified.loc[future_mask, "low"] = df_modified.loc[future_mask, "close"] * 0.98

        result_modified = add_indicators(df_modified)
        row_modified = result_modified.loc[result_modified["date"] == cutoff_date]
        assert len(row_modified) == 1
        row_modified = row_modified.iloc[0]

        for col in EXPECTED_INDICATOR_COLUMNS + ["close"]:
            assert np.isclose(row_original[col], row_modified[col], rtol=1e-9, atol=1e-9), (
                f"LEAK phát hiện: cột '{col}' tại ngày cutoff thay đổi khi dữ liệu "
                f"TƯƠNG LAI (sau cutoff) bị thay đổi -> vi phạm no-lookahead."
            )

    def test_past_price_changes_do_affect_features_at_t_sanity_check(self):
        """Sanity check ngược lại: thay đổi dữ liệu TRƯỚC/tại cutoff PHẢI làm
        (ít nhất một) feature tại cutoff thay đổi — nếu không, test look-ahead
        ở trên có thể đang pass "giả" (vd do bug khiến add_indicators không
        tính gì cả).

        Lưu ý: dùng nhiễu NGẪU NHIÊN (không phải affine transform đơn giản
        close' = a*close+b) để nhiễu loạn quá khứ, vì RSI về mặt toán học BẤT
        BIẾN với phép affine dương (chỉ phụ thuộc tỉ lệ gain/loss, không phụ
        thuộc thang đo giá) — affine transform sẽ khiến test này false-negative
        một cách "đúng về toán" chứ không phải lỗi của add_indicators().
        """
        df = make_synthetic_ohlcv("AAA", periods=120, seed=7)
        cutoff_idx = 90
        cutoff_date = df["date"].iloc[cutoff_idx]

        result_original = add_indicators(df)
        row_original = result_original.loc[result_original["date"] == cutoff_date].iloc[0]

        df_modified = df.copy()
        past_mask = df_modified["date"] <= cutoff_date
        rng = np.random.default_rng(999)
        noise = rng.normal(loc=1.0, scale=0.08, size=int(past_mask.sum()))
        df_modified.loc[past_mask, "close"] = df_modified.loc[past_mask, "close"].to_numpy() * noise

        result_modified = add_indicators(df_modified)
        row_modified = result_modified.loc[result_modified["date"] == cutoff_date].iloc[0]

        changed_columns = [
            col
            for col in EXPECTED_INDICATOR_COLUMNS
            if not np.isclose(row_original[col], row_modified[col], rtol=1e-6, atol=1e-9)
        ]
        assert len(changed_columns) > 0, (
            "Thay đổi dữ liệu QUÁ KHỨ (nhiễu ngẫu nhiên) phải làm ít nhất 1 "
            "feature tại t thay đổi — nếu không thì bản thân add_indicators() "
            "có vấn đề (không thực sự tính rolling)."
        )


# ---------------------------------------------------------------------------
# 3. add_indicators_multi(): tính RIÊNG từng ticker, không lẫn dữ liệu
#    (test quan trọng nhất — phát hiện lỗi rolling-window mixing giữa các mã)
# ---------------------------------------------------------------------------
class TestAddIndicatorsMulti:
    def test_matches_individual_per_ticker_computation(self):
        df_a = make_synthetic_ohlcv("AAA", periods=150, seed=11, base_price=100.0)
        df_b = make_synthetic_ohlcv("BBB", periods=150, seed=222, base_price=800.0)

        combined = pd.concat([df_a, df_b], ignore_index=True)

        result_multi = add_indicators_multi(combined, ticker_col="ticker")
        result_multi = result_multi.sort_values(["ticker", "date"]).reset_index(drop=True)

        result_individual = pd.concat(
            [add_indicators(df_a), add_indicators(df_b)], ignore_index=True
        )
        result_individual = result_individual.sort_values(["ticker", "date"]).reset_index(drop=True)

        pd.testing.assert_frame_equal(result_multi, result_individual, check_dtype=False)

    def test_row_count_equals_sum_of_individual_dropna(self):
        df_a = make_synthetic_ohlcv("AAA", periods=100, seed=3)
        df_b = make_synthetic_ohlcv("BBB", periods=80, seed=4)
        combined = pd.concat([df_a, df_b], ignore_index=True)

        result_multi = add_indicators_multi(combined, ticker_col="ticker")

        expected_len = len(add_indicators(df_a)) + len(add_indicators(df_b))
        assert len(result_multi) == expected_len

    def test_detects_cross_ticker_leakage_if_naively_mixed(self):
        """Test đối chứng: chứng minh rằng NẾU tính indicator trên dữ liệu đã
        trộn (sort theo date, bỏ qua ticker) như lỗi đặc tả cảnh báo, kết quả
        SẼ khác với add_indicators_multi() đúng cách — nghĩa là test ở trên
        thực sự có khả năng phát hiện lỗi này chứ không phải test rỗng."""
        df_a = make_synthetic_ohlcv("AAA", periods=60, seed=11, base_price=100.0)
        df_b = make_synthetic_ohlcv("BBB", periods=60, seed=222, base_price=800.0)
        # cùng khoảng ngày -> khi sort theo date (bỏ qua ticker) các dòng của
        # 2 mã sẽ xen kẽ nhau, mô phỏng đúng lỗi "trộn dữ liệu" mà đặc tả cảnh báo.
        combined = pd.concat([df_a, df_b], ignore_index=True)

        correct = add_indicators_multi(combined, ticker_col="ticker")
        correct = correct.sort_values(["ticker", "date"]).reset_index(drop=True)

        # Mô phỏng implementation SAI: sort toàn bộ theo date rồi tính rolling
        # thẳng trên "close" như thể chỉ có 1 chuỗi duy nhất.
        naive_mixed = combined.sort_values("date").reset_index(drop=True).copy()
        naive_mixed["rsi_14_naive"] = _naive_rsi(naive_mixed["close"], length=14)

        # Với dữ liệu 2 mã base_price rất khác nhau (100 vs 800) xen kẽ theo
        # ngày, RSI "naive" (tính lẫn) phải khác RSI đúng (tính riêng từng mã)
        # tại phần lớn các điểm sau warm-up -> chứng minh test ở trên có sức mạnh
        # phát hiện lỗi.
        merged = naive_mixed.merge(
            correct[["ticker", "date", "rsi_14"]], on=["ticker", "date"], how="inner"
        )
        assert len(merged) > 0
        n_different = (~np.isclose(merged["rsi_14_naive"], merged["rsi_14"], equal_nan=False)).sum()
        assert n_different > 0, (
            "Kỳ vọng RSI tính lẫn dữ liệu (naive, sai) phải khác RSI tính đúng "
            "từng mã riêng (add_indicators_multi) — nếu bằng nhau thì test không "
            "có khả năng phát hiện lỗi mixing."
        )


def _naive_rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """RSI thủ công (Wilder) chỉ dùng để mô phỏng implementation SAI (tính
    trên chuỗi đã trộn nhiều mã) trong test đối chứng ở trên — KHÔNG dùng
    trong pipeline thật."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ---------------------------------------------------------------------------
# 4. build_state_dataframe(): align đúng theo inner-join ngày giao dịch chung
# ---------------------------------------------------------------------------
class TestBuildStateDataframe:
    def test_inner_join_alignment_with_offset_date_ranges(self, tmp_path):
        # Ticker A: 2020-01-01 -> ~2020-07 (150 business days)
        df_a = make_synthetic_ohlcv("AAA", start="2020-01-01", periods=150, seed=21)
        # Ticker B: bắt đầu trễ hơn 30 ngày và kết thúc trễ hơn -> lệch khoảng ngày
        df_b = make_synthetic_ohlcv("BBB", start="2020-02-15", periods=150, seed=22)

        raw = pd.concat([df_a, df_b], ignore_index=True)
        ohlcv_path = tmp_path / "raw_ohlcv.parquet"
        raw.to_parquet(ohlcv_path)

        wide = build_state_dataframe(["AAA", "BBB"], ohlcv_path=ohlcv_path)

        # Tính "common dates" kỳ vọng một cách độc lập, không qua build_state_dataframe
        indicators_a = add_indicators(df_a)
        indicators_b = add_indicators(df_b)
        expected_common_dates = sorted(
            set(indicators_a["date"]).intersection(set(indicators_b["date"]))
        )

        assert len(wide) == len(expected_common_dates)
        assert list(wide.index) == list(pd.DatetimeIndex(expected_common_dates))
        assert not wide.isna().any().any()

        assert isinstance(wide.columns, pd.MultiIndex)
        assert set(wide.columns.get_level_values("ticker")) == {"AAA", "BBB"}
        for feat in FEATURE_COLUMNS:
            assert feat in wide.columns.get_level_values("feature")

    def test_sentiment_placeholder_is_zero(self, tmp_path):
        df_a = make_synthetic_ohlcv("AAA", periods=80, seed=1)
        df_b = make_synthetic_ohlcv("BBB", periods=80, seed=2)
        raw = pd.concat([df_a, df_b], ignore_index=True)
        ohlcv_path = tmp_path / "raw_ohlcv.parquet"
        raw.to_parquet(ohlcv_path)

        wide = build_state_dataframe(["AAA", "BBB"], ohlcv_path=ohlcv_path)

        sentiment_cols = wide.loc[:, (slice(None), "sentiment")]
        assert (sentiment_cols == 0.0).all().all()

    def test_saves_to_output_path(self, tmp_path):
        df_a = make_synthetic_ohlcv("AAA", periods=80, seed=1)
        df_b = make_synthetic_ohlcv("BBB", periods=80, seed=2)
        raw = pd.concat([df_a, df_b], ignore_index=True)
        ohlcv_path = tmp_path / "raw_ohlcv.parquet"
        raw.to_parquet(ohlcv_path)

        out_path = tmp_path / "processed" / "state.parquet"
        wide = build_state_dataframe(["AAA", "BBB"], ohlcv_path=ohlcv_path, output_path=out_path)

        assert out_path.exists()
        reloaded = pd.read_parquet(out_path)
        assert len(reloaded) == len(wide)

    def test_raises_for_no_common_dates(self, tmp_path):
        # Hai mã có khoảng ngày HOÀN TOÀN không giao nhau -> inner join rỗng
        df_a = make_synthetic_ohlcv("AAA", start="2019-01-01", periods=60, seed=1)
        df_b = make_synthetic_ohlcv("BBB", start="2023-01-01", periods=60, seed=2)
        raw = pd.concat([df_a, df_b], ignore_index=True)
        ohlcv_path = tmp_path / "raw_ohlcv.parquet"
        raw.to_parquet(ohlcv_path)

        with pytest.raises(ValueError):
            build_state_dataframe(["AAA", "BBB"], ohlcv_path=ohlcv_path)


# ---------------------------------------------------------------------------
# 5. make_walk_forward_windows(): thứ tự train < val < test, trượt đúng hướng
# ---------------------------------------------------------------------------
class TestMakeWalkForwardWindows:
    def test_windows_ordering_and_no_overlap(self):
        dates = pd.bdate_range("2015-01-01", "2024-12-31")
        windows = make_walk_forward_windows(
            dates, train_years=3, val_months=6, test_months=6, step_months=6
        )

        assert len(windows) >= 2, "Kỳ vọng nhiều hơn 1 cửa sổ với 10 năm dữ liệu"

        for train_range, val_range, test_range in windows:
            train_start, train_end = train_range
            val_start, val_end = val_range
            test_start, test_end = test_range

            assert train_start <= train_end
            assert val_start <= val_end
            assert test_start <= test_end

            # train luôn kết thúc TRƯỚC khi val bắt đầu
            assert train_end < val_start
            # val luôn kết thúc TRƯỚC khi test bắt đầu
            assert val_end < test_start

    def test_windows_slide_forward_monotonically(self):
        dates = pd.bdate_range("2015-01-01", "2024-12-31")
        windows = make_walk_forward_windows(
            dates, train_years=3, val_months=6, test_months=6, step_months=6
        )

        train_starts = [w[0][0] for w in windows]
        assert train_starts == sorted(train_starts)
        assert len(set(train_starts)) == len(train_starts)  # không trùng lặp
        for i in range(len(train_starts) - 1):
            assert train_starts[i] < train_starts[i + 1]

    def test_all_windows_within_data_bounds(self):
        dates = pd.bdate_range("2015-01-01", "2024-12-31")
        windows = make_walk_forward_windows(
            dates, train_years=3, val_months=6, test_months=6, step_months=6
        )
        data_start, data_end = dates.min(), dates.max()
        for train_range, _val_range, test_range in windows:
            assert train_range[0] >= data_start
            assert test_range[1] <= data_end

    def test_empty_when_insufficient_data(self):
        # Chỉ có 6 tháng dữ liệu -> không đủ cho train_years=3
        dates = pd.bdate_range("2023-01-01", "2023-06-30")
        windows = make_walk_forward_windows(
            dates, train_years=3, val_months=6, test_months=6, step_months=6
        )
        assert windows == []


# ---------------------------------------------------------------------------
# 6. single_window_phase_a(): đúng 1 cửa sổ cố định theo đặc tả
# ---------------------------------------------------------------------------
class TestSingleWindowPhaseA:
    def test_fixed_ranges_match_spec(self):
        train_range, val_range, test_range = single_window_phase_a()

        assert train_range == (pd.Timestamp("2019-01-01"), pd.Timestamp("2022-12-31"))
        assert val_range == (pd.Timestamp("2023-01-01"), pd.Timestamp("2023-06-30"))
        assert test_range == (pd.Timestamp("2023-07-01"), pd.Timestamp("2023-12-31"))

    def test_ordering(self):
        train_range, val_range, test_range = single_window_phase_a()
        assert train_range[1] < val_range[0]
        assert val_range[1] < test_range[0]

    def test_accepts_optional_dates_arg_without_changing_output(self):
        dates = pd.bdate_range("2019-01-01", "2023-12-31")
        window_with_dates = single_window_phase_a(dates)
        window_without_dates = single_window_phase_a()
        assert window_with_dates == window_without_dates


# ---------------------------------------------------------------------------
# Bonus: sanity check cho fetch_news placeholder và danh sách ticker
# ---------------------------------------------------------------------------
def test_get_sentiment_placeholder_returns_neutral_zero():
    value = get_sentiment_placeholder("FPT", "2023-01-01")
    assert value == 0.0
    assert isinstance(value, float)


def test_ticker_lists_phase_0_and_b():
    from src.data.fetch_ohlcv_us import (
        TICKERS_PHASE_0,
        TICKERS_PHASE_A,
        TICKERS_PHASE_B,
    )

    assert TICKERS_PHASE_0 == ["AAPL", "MSFT", "JPM", "JNJ", "XOM", "PG", "KO", "NVDA"]
    assert TICKERS_PHASE_A == TICKERS_PHASE_0  # Giai đoạn A dùng chung bộ 8 mã
    assert len(TICKERS_PHASE_B) == 16
    assert set(TICKERS_PHASE_0).issubset(set(TICKERS_PHASE_B))
