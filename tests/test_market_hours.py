from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from stocks.market_hours import (
    current_market_phase,
    is_regular_market_hours,
    market_open_today_utc,
    minutes_since_open,
    minutes_to_close,
    opening_range_end_utc,
    phase_badge_color,
)

ET = ZoneInfo("America/New_York")


def test_regular_hours_weekday():
    now = datetime(2026, 7, 8, 10, 0, tzinfo=ET)  # Wednesday
    assert is_regular_market_hours(now)
    assert current_market_phase(now) == "REGULAR"


def test_pre_market():
    now = datetime(2026, 7, 8, 8, 0, tzinfo=ET)
    assert not is_regular_market_hours(now)
    assert current_market_phase(now) == "PRE_MARKET"


def test_after_hours():
    now = datetime(2026, 7, 8, 17, 30, tzinfo=ET)
    assert current_market_phase(now) == "AFTER_HOURS"


def test_overnight_closed():
    now = datetime(2026, 7, 8, 22, 0, tzinfo=ET)
    assert current_market_phase(now) == "CLOSED"


def test_weekend_closed():
    now = datetime(2026, 7, 11, 11, 0, tzinfo=ET)  # Saturday
    assert current_market_phase(now) == "CLOSED"
    assert market_open_today_utc(now) is None
    assert minutes_since_open(now) is None
    assert minutes_to_close(now) is None


def test_winter_regular_hours():
    # EST (no DST): 10:00 ET on a Wednesday in January
    now = datetime(2026, 1, 14, 10, 0, tzinfo=ET)
    assert current_market_phase(now) == "REGULAR"
    open_utc = market_open_today_utc(now)
    assert open_utc == datetime(2026, 1, 14, 14, 30, tzinfo=timezone.utc)  # 09:30 EST = 14:30 UTC


def test_open_time_and_ranges():
    now = datetime(2026, 7, 8, 10, 15, tzinfo=ET)  # EDT: 09:30 = 13:30 UTC
    open_utc = market_open_today_utc(now)
    assert open_utc == datetime(2026, 7, 8, 13, 30, tzinfo=timezone.utc)
    assert opening_range_end_utc(now) == datetime(2026, 7, 8, 14, 0, tzinfo=timezone.utc)
    assert minutes_since_open(now) == 45.0
    assert minutes_to_close(now) == 345.0


def test_minutes_since_open_before_open():
    now = datetime(2026, 7, 8, 9, 0, tzinfo=ET)
    assert minutes_since_open(now) is None


def test_badges():
    assert phase_badge_color("REGULAR") == "🟢"
    assert phase_badge_color("CLOSED") == "⚫"
    assert phase_badge_color("???") == "⚫"
