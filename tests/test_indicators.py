from datetime import datetime, timezone

from stocks.indicators import (
    calculate_adx,
    calculate_rsi,
    compute_all,
    range_high_low,
    window_high_low,
)


def _bar(ts: str, high: float, low: float, close: float) -> dict:
    return {"timestamp": ts, "open": close, "high": high, "low": low, "close": close, "volume": 100}


def test_rsi_all_gains_is_100():
    closes = [float(i) for i in range(1, 20)]
    assert calculate_rsi(closes) == 100.0


def test_rsi_insufficient_data():
    assert calculate_rsi([1.0, 2.0, 3.0]) is None


def test_adx_insufficient_data():
    assert calculate_adx([1.0] * 10, [0.5] * 10, [0.7] * 10) is None


def test_range_high_low_filters_by_start():
    bars = [
        _bar("2026-07-08T13:00:00+00:00", 10.0, 9.0, 9.5),
        _bar("2026-07-08T13:30:00+00:00", 12.0, 8.0, 11.0),
        _bar("2026-07-08T14:00:00+00:00", 11.0, 10.0, 10.5),
    ]
    start = datetime(2026, 7, 8, 13, 30, tzinfo=timezone.utc)
    hi, lo = range_high_low(bars, start)
    assert hi == 12.0
    assert lo == 8.0


def test_range_high_low_none_start():
    assert range_high_low([_bar("2026-07-08T13:00:00+00:00", 1, 0, 0.5)], None) == (None, None)


def test_window_high_low_half_open_interval():
    bars = [
        _bar("2026-07-08T13:30:00+00:00", 10.0, 9.0, 9.5),
        _bar("2026-07-08T13:55:00+00:00", 12.0, 8.0, 11.0),
        _bar("2026-07-08T14:00:00+00:00", 15.0, 14.0, 14.5),  # excluded (end bound)
    ]
    start = datetime(2026, 7, 8, 13, 30, tzinfo=timezone.utc)
    end = datetime(2026, 7, 8, 14, 0, tzinfo=timezone.utc)
    hi, lo = window_high_low(bars, start, end)
    assert hi == 12.0
    assert lo == 8.0


def test_compute_all_keys():
    bars = [
        _bar(f"2026-07-08T{13 + i // 12:02d}:{(i % 12) * 5:02d}:00+00:00", 100 + i * 0.1, 99 + i * 0.1, 99.5 + i * 0.1)
        for i in range(60)
    ]
    result = compute_all(bars)
    for key in ("close", "rsi14", "ema9", "ema20", "macd", "atr14", "adx14", "bb_upper"):
        assert key in result
    assert result["close"] == bars[-1]["close"]
