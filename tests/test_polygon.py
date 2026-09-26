"""Polygon client (pagination, throttling, retries), bar conversion, and the
resumable per-ticker cache — all against a fake HTTP opener."""
from __future__ import annotations

import io
import json
from datetime import date, datetime, timedelta, timezone
from urllib.error import HTTPError

import pytest

from stocks import history
from stocks.market_hours import US_EASTERN
from stocks.polygon_client import (
    PolygonClient, PolygonError, regular_5m_bars, resample_daily, resample_hourly,
)


class FakeClock:
    def __init__(self):
        self.t = 0.0
        self.slept = []

    def clock(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


class FakeOpener:
    """Serves queued responses; each is a dict (200 JSON) or an int (HTTP error)."""
    def __init__(self, responses):
        self.responses = list(responses)
        self.urls = []

    def __call__(self, url, timeout=None):
        self.urls.append(url)
        r = self.responses.pop(0)
        if isinstance(r, int):
            raise HTTPError(url, r, "err", {}, io.BytesIO(b'{"status":"ERROR"}'))
        return io.BytesIO(json.dumps(r).encode())


def _client(responses, cpm=5.0):
    fc, op = FakeClock(), FakeOpener(responses)
    return PolygonClient("KEY", cpm, opener=op, sleep=fc.sleep, clock=fc.clock), op, fc


def _ms(local: datetime) -> int:
    return int(local.replace(tzinfo=US_EASTERN).timestamp() * 1000)


def _row(local: datetime, c=100.0):
    return {"t": _ms(local), "o": c, "h": c + 1, "l": c - 1, "c": c, "v": 10.4}


def test_pagination_follows_next_url_with_the_key():
    c, op, _ = _client([
        {"results": [{"t": 1}], "next_url": "https://api.polygon.io/v2/aggs/cursor1"},
        {"results": [{"t": 2}]},
    ])
    rows = c.aggregates("AAPL", 5, "minute", date(2026, 1, 2), date(2026, 3, 31))
    assert [r["t"] for r in rows] == [1, 2]
    assert "/range/5/minute/2026-01-02/2026-03-31" in op.urls[0]
    assert op.urls[1] == "https://api.polygon.io/v2/aggs/cursor1?apiKey=KEY"


def test_calls_are_spaced_to_the_plan_rate():
    c, _, fc = _client([{"results": []}] * 3, cpm=5.0)
    for _ in range(3):
        c.aggregates("AAPL", 5, "minute", date(2026, 1, 2), date(2026, 1, 3))
    assert fc.slept == [12.0, 12.0]


def test_rate_limit_backs_off_and_retries():
    c, _, fc = _client([429, {"results": [{"t": 7}]}])
    assert c.aggregates("AAPL", 5, "minute", date(2026, 1, 2), date(2026, 1, 3)) == [{"t": 7}]
    assert 60.0 in fc.slept


def test_plan_refusal_is_reported_not_retried():
    c, op, _ = _client([403])
    with pytest.raises(PolygonError, match="HTTP 403"):
        c.aggregates("AAPL", 5, "minute", date(2023, 1, 2), date(2023, 1, 3))
    assert len(op.urls) == 1


def test_missing_key_fails_fast():
    with pytest.raises(PolygonError, match="POLYGON_API_KEY"):
        PolygonClient("")


def test_conversion_keeps_regular_completed_bars_in_yahoo_format():
    rows = [
        _row(datetime(2026, 9, 24, 9, 25)),          # pre-market
        _row(datetime(2026, 9, 24, 9, 30)),
        _row(datetime(2026, 9, 24, 15, 55)),
        _row(datetime(2026, 9, 24, 16, 0)),          # after hours
    ]
    now = datetime(2026, 9, 24, 15, 58, tzinfo=US_EASTERN)   # 15:55 bar still forming
    bars = regular_5m_bars(rows, "AAPL", now=now)
    assert [b["timestamp"] for b in bars] == ["2026-09-24T13:30:00+00:00"]
    assert bars[0]["volume"] == 10 and bars[0]["timeframe"] == "5m"


def test_hourly_and_daily_resampling_match_yahoo_alignment():
    rows = [_row(datetime(2026, 9, 24, 9, 30) + timedelta(minutes=5 * k), 100 + k) for k in range(78)]
    bars = regular_5m_bars(rows, "AAPL", now=datetime(2026, 9, 25, tzinfo=timezone.utc))
    hourly = resample_hourly(bars)
    starts = [datetime.fromisoformat(h["timestamp"]).astimezone(US_EASTERN).strftime("%H:%M") for h in hourly]
    assert starts == ["09:30", "10:30", "11:30", "12:30", "13:30", "14:30", "15:30"]
    assert hourly[0]["open"] == 100 and hourly[0]["close"] == 111 and hourly[0]["high"] == 112
    assert hourly[-1]["volume"] == 6 * 10                     # 15:30–16:00 is half an hour
    (daily,) = resample_daily(bars)
    assert daily["timestamp"] == "2026-09-24T04:00:00+00:00"  # ET midnight, as yfinance
    assert daily["open"] == 100 and daily["close"] == 177 and daily["volume"] == 780


def test_cache_fetches_only_the_missing_tail_and_cache_only_never_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "POLYGON_CACHE_DIR", tmp_path)
    day1 = [_row(datetime(2026, 9, 21, 10, 0))]
    day2 = [_row(datetime(2026, 9, 21, 10, 0)), _row(datetime(2026, 9, 22, 10, 0))]
    c, op, _ = _client([{"results": day1}, {"results": day2}], cpm=0)

    first = history.polygon_5m(c, "AAPL", date(2026, 9, 21), date(2026, 9, 21))
    assert len(first) == 1
    second = history.polygon_5m(c, "AAPL", date(2026, 9, 21), date(2026, 9, 22))
    assert len(second) == 2
    assert "/2026-09-21/2026-09-22" in op.urls[1]            # tail only, from the last cached day

    offline = history.polygon_5m(None, "AAPL", date(2026, 9, 21), date(2026, 9, 30), cache_only=True)
    assert offline == second and len(op.urls) == 2
    assert history.polygon_5m(None, "MSFT", date(2026, 9, 21), date(2026, 9, 30), cache_only=True) == []
