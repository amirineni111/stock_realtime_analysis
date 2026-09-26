"""
Polygon.io (Massive) aggregates client for backtest history.

Used for research only: the free Basic plan serves two years of minute
aggregates but no same-day data (bars appear after the close), so live scanning
stays on yfinance. See README "Data sources".

Things this client has to get right:

- **Rate limit.** Basic allows 5 calls/minute. Calls are spaced evenly rather than
  fired in bursts, and a 429 backs off and retries instead of failing a
  multi-hour download.
- **Pagination.** Polygon's ``limit`` counts *base* 1-minute bars, so one 5-minute
  request returns only ~3 months (12k bars). ``next_url`` is followed until done.
- **Plan boundary.** A request reaching past the plan's history window fails
  whole (403 NOT_AUTHORIZED), so the caller must stay inside it.
- **Bar conventions.** Timestamps are emitted exactly as ``yf_client`` emits them
  (bar start, UTC ISO-8601), and hourly/daily bars are *derived* from regular-
  session 5m bars with Yahoo's 09:30 alignment. The replay then sees the same
  shapes of data the live scanner sees, instead of Polygon's clock-hour and
  extended-hours bars.
"""
from __future__ import annotations

import json
import time
import urllib.request
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional
from urllib.error import HTTPError, URLError

from .market_hours import US_EASTERN

DEFAULT_BASE_URL = "https://api.polygon.io"


class PolygonError(RuntimeError):
    """A request Polygon refused or that kept failing after retries."""


class PolygonClient:
    def __init__(
        self,
        api_key: str,
        calls_per_minute: float = 5.0,
        base_url: str = DEFAULT_BASE_URL,
        opener: Callable = urllib.request.urlopen,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        max_retries: int = 4,
    ) -> None:
        if not api_key:
            raise PolygonError("POLYGON_API_KEY is not set (see .env.example)")
        self._key = api_key
        self._base = base_url.rstrip("/")
        self._interval = 60.0 / calls_per_minute if calls_per_minute > 0 else 0.0
        self._open, self._sleep, self._clock = opener, sleep, clock
        self._max_retries = max_retries
        self._last_call: Optional[float] = None
        self.calls = 0

    def _throttle(self) -> None:
        if self._interval and self._last_call is not None:
            wait = self._last_call + self._interval - self._clock()
            if wait > 0:
                self._sleep(wait)
        self._last_call = self._clock()

    def _get(self, url: str) -> dict:
        sep = "&" if "?" in url else "?"
        full = f"{url}{sep}apiKey={self._key}"
        for attempt in range(self._max_retries + 1):
            self._throttle()
            self.calls += 1
            try:
                with self._open(full, timeout=30) as resp:
                    return json.loads(resp.read())
            except HTTPError as exc:
                body = exc.read()[:300].decode("utf-8", "replace")
                if exc.code == 429 and attempt < self._max_retries:
                    self._sleep(60.0)            # minute window exhausted: wait it out
                    continue
                if exc.code >= 500 and attempt < self._max_retries:
                    self._sleep(5.0 * (attempt + 1))
                    continue
                raise PolygonError(f"HTTP {exc.code}: {body}") from None
            except (URLError, OSError) as exc:
                if attempt < self._max_retries:
                    self._sleep(5.0 * (attempt + 1))
                    continue
                raise PolygonError(f"network error: {exc}") from None
        raise PolygonError("retries exhausted")

    def aggregates(
        self, ticker: str, multiplier: int, timespan: str, start: date, end: date,
    ) -> List[dict]:
        """Raw aggregate rows for [start, end], following pagination."""
        url = (
            f"{self._base}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/"
            f"{start.isoformat()}/{end.isoformat()}?adjusted=true&sort=asc&limit=50000"
        )
        rows: List[dict] = []
        while url:
            page = self._get(url)
            rows.extend(page.get("results") or [])
            url = page.get("next_url")
        return rows


# ── Conversions to the repo's bar-dict shape ────────────────────────────────

def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def _is_regular_start(ms: int) -> bool:
    local = datetime.fromtimestamp(ms / 1000, tz=timezone.utc).astimezone(US_EASTERN)
    minutes = local.hour * 60 + local.minute
    return local.weekday() < 5 and 9 * 60 + 30 <= minutes < 16 * 60


def regular_5m_bars(rows: List[dict], ticker: str, now: Optional[datetime] = None) -> List[dict]:
    """Regular-session 5m bars as bar dicts, dropping any still-forming bar."""
    now = now or datetime.now(timezone.utc)
    now_ms = now.timestamp() * 1000
    out = []
    for r in rows:
        t = int(r["t"])
        if not _is_regular_start(t) or t + 5 * 60_000 > now_ms:
            continue
        out.append({
            "ticker": ticker, "timeframe": "5m", "timestamp": _iso(t),
            "open": float(r["o"]), "high": float(r["h"]), "low": float(r["l"]),
            "close": float(r["c"]), "volume": int(round(r.get("v") or 0)),
        })
    out.sort(key=lambda b: b["timestamp"])
    return out


def _bucket(bars: List[dict], key_fn, timeframe: str) -> List[dict]:
    groups: Dict[str, List[dict]] = {}
    order: List[str] = []
    for b in bars:
        k = key_fn(b)
        if k not in groups:
            groups[k] = []
            order.append(k)
        groups[k].append(b)
    out = []
    for k in order:
        g = groups[k]
        out.append({
            "ticker": g[0]["ticker"], "timeframe": timeframe, "timestamp": k,
            "open": g[0]["open"], "high": max(b["high"] for b in g),
            "low": min(b["low"] for b in g), "close": g[-1]["close"],
            "volume": sum(b["volume"] for b in g),
        })
    return out


def resample_hourly(bars_5m: List[dict]) -> List[dict]:
    """Hourly bars starting 09:30, 10:30, … 15:30 ET (the last is 30 minutes),
    matching Yahoo's regular-session hourly alignment."""
    def key(b):
        local = datetime.fromisoformat(b["timestamp"]).astimezone(US_EASTERN)
        minutes = local.hour * 60 + local.minute - (9 * 60 + 30)
        start = local.replace(hour=9, minute=30, second=0, microsecond=0) + timedelta(hours=minutes // 60)
        return start.astimezone(timezone.utc).isoformat()
    return _bucket(bars_5m, key, "1h")


def resample_daily(bars_5m: List[dict]) -> List[dict]:
    """Daily bars stamped at ET midnight of the session date, as yfinance's are."""
    def key(b):
        local = datetime.fromisoformat(b["timestamp"]).astimezone(US_EASTERN)
        midnight = local.replace(hour=0, minute=0, second=0, microsecond=0)
        return midnight.astimezone(timezone.utc).isoformat()
    return _bucket(bars_5m, key, "1d")
