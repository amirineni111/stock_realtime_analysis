from datetime import datetime, timezone

import pandas as pd

from stocks.yf_client import _bar_end_utc, _frame_to_bars, _slice_ticker

NY = "America/New_York"


def _intraday_frame(periods: int = 5) -> pd.DataFrame:
    idx = pd.date_range("2026-07-08 09:30", periods=periods, freq="5min", tz=NY)
    base = 100.0
    return pd.DataFrame({
        "Open": [base + i for i in range(periods)],
        "High": [base + i + 0.5 for i in range(periods)],
        "Low": [base + i - 0.5 for i in range(periods)],
        "Close": [base + i + 0.2 for i in range(periods)],
        "Volume": [1000 + i for i in range(periods)],
    }, index=idx)


def test_forming_bar_dropped_during_market():
    df = _intraday_frame(5)  # last bar starts 09:50 ET, completes 09:55 ET = 13:55 UTC
    now = datetime(2026, 7, 8, 13, 52, tzinfo=timezone.utc)  # 09:52 ET → last bar forming
    bars = _frame_to_bars(df, "AAPL", "5m", now)
    assert len(bars) == 4
    assert bars[-1].timestamp == "2026-07-08T13:45:00+00:00"  # 09:45 ET bar


def test_completed_bars_all_kept():
    df = _intraday_frame(5)
    now = datetime(2026, 7, 8, 14, 30, tzinfo=timezone.utc)
    bars = _frame_to_bars(df, "AAPL", "5m", now)
    assert len(bars) == 5
    # Uniform UTC-ISO format → lexicographic order is chronological
    timestamps = [b.timestamp for b in bars]
    assert timestamps == sorted(timestamps)
    assert all(ts.endswith("+00:00") for ts in timestamps)


def test_nan_rows_dropped():
    df = _intraday_frame(5)
    df.iloc[2, df.columns.get_loc("Close")] = float("nan")
    now = datetime(2026, 7, 8, 14, 30, tzinfo=timezone.utc)
    bars = _frame_to_bars(df, "AAPL", "5m", now)
    assert len(bars) == 4


def test_keep_forming_bar_for_quotes():
    df = _intraday_frame(5)
    now = datetime(2026, 7, 8, 13, 52, tzinfo=timezone.utc)
    bars = _frame_to_bars(df, "AAPL", "5m", now, drop_forming=False)
    assert len(bars) == 5


def test_daily_bar_completes_at_close():
    idx = pd.date_range("2026-07-06", periods=3, freq="D")  # tz-naive daily index
    df = pd.DataFrame({
        "Open": [10.0, 11.0, 12.0],
        "High": [10.5, 11.5, 12.5],
        "Low": [9.5, 10.5, 11.5],
        "Close": [10.2, 11.2, 12.2],
        "Volume": [100, 200, 300],
    }, index=idx)
    # 2026-07-08 12:00 ET (16:00 UTC): today's daily bar still forming → dropped
    mid_session = datetime(2026, 7, 8, 16, 0, tzinfo=timezone.utc)
    assert len(_frame_to_bars(df, "AAPL", "1d", mid_session)) == 2
    # 2026-07-08 17:00 ET (21:00 UTC): after the close → kept
    after_close = datetime(2026, 7, 8, 21, 0, tzinfo=timezone.utc)
    assert len(_frame_to_bars(df, "AAPL", "1d", after_close)) == 3


def test_bar_end_utc_daily():
    ts = pd.Timestamp("2026-07-08")  # tz-naive
    end = _bar_end_utc(ts, "1d")
    assert end == datetime(2026, 7, 8, 20, 0, tzinfo=timezone.utc)  # 16:00 EDT


def test_slice_ticker_multiindex():
    df = _intraday_frame(3)
    multi = pd.concat({"AAPL": df, "MSFT": df * 2}, axis=1)
    sub = _slice_ticker(multi, "AAPL")
    assert sub is not None
    assert list(sub.columns) == list(df.columns)
    assert _slice_ticker(multi, "NVDA") is None


def test_slice_ticker_single_level():
    df = _intraday_frame(3)
    sub = _slice_ticker(df, "AAPL")
    assert sub is df


def test_empty_frame():
    assert _frame_to_bars(None, "AAPL", "5m", datetime.now(timezone.utc)) == []
    assert _frame_to_bars(pd.DataFrame(), "AAPL", "5m", datetime.now(timezone.utc)) == []
