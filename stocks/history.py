"""
Historical bar downloads for the backtesters, cached per calendar day.

Yahoo serves at most 60 days of 5m history, which bounds every intraday replay.
The cache lives under data/ (gitignored) and is keyed by date + ticker set, so
re-running a backtest the same day costs no network calls.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from .tickers import parse_watchlist

PREFS_PATH = Path("data/app_preferences.json")
CACHE_DIR = Path("data/backtest_cache")
DEFAULT_PERIODS: Mapping[str, str] = {"5m": "60d", "1h": "6mo", "1d": "2y"}


def dashboard_watchlist() -> List[str]:
    """The watchlist saved by the dashboard, or [] when there is none."""
    try:
        prefs = json.loads(PREFS_PATH.read_text(encoding="utf-8"))
        return parse_watchlist(prefs.get("watchlist_raw", ""))
    except (OSError, ValueError):
        return []


def fetch_history(
    tickers: Sequence[str],
    periods: Mapping[str, str] = DEFAULT_PERIODS,
    refresh: bool = False,
) -> Dict[str, Dict[str, List[dict]]]:
    """{interval: {ticker: [bar dicts]}} with forming bars dropped, cached per day."""
    import yfinance as yf
    from .yf_client import _frame_to_bars, _slice_ticker

    tickers = sorted(set(tickers))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # hashlib, not hash(): str hashes are salted per process, so hash() never hits.
    spec = ",".join(tickers) + "|" + ",".join(f"{k}={v}" for k, v in sorted(periods.items()))
    digest = hashlib.sha1(spec.encode()).hexdigest()[:10]
    path = CACHE_DIR / f"{date.today().isoformat()}_{digest}.pkl"
    if path.exists() and not refresh:
        return pickle.loads(path.read_bytes())

    now = datetime.now(timezone.utc)
    data: Dict[str, Dict[str, List[dict]]] = {}
    for interval, period in periods.items():
        print(f"  downloading {interval} ({period}) for {len(tickers)} tickers...", flush=True)
        df = yf.download(
            tickers, interval=interval, period=period, group_by="ticker",
            auto_adjust=False, progress=False, threads=True,
        )
        data[interval] = {
            t: [b.model_dump() for b in _frame_to_bars(_slice_ticker(df, t), t, interval, now)]
            for t in tickers
        }
    path.write_bytes(pickle.dumps(data))
    return data


# ── Polygon (Massive) history ────────────────────────────────────────────────

POLYGON_CACHE_DIR = Path("data/polygon_cache/5m")
# Stay a few days inside the plan's rolling window: a request that reaches past
# it is refused whole, not truncated.
_PLAN_EDGE_DAYS = 3
# Calendar days at the start of a Polygon window used only to warm up indicators:
# daily trend needs ~26 sessions and 20-day dollar volume needs 20.
WARMUP_DAYS = 60


def _polygon_cache_path(ticker: str) -> Path:
    return POLYGON_CACHE_DIR / f"{ticker}.pkl"


def _merge(a: List[dict], b: List[dict]) -> List[dict]:
    by_ts = {x["timestamp"]: x for x in a}
    by_ts.update({x["timestamp"]: x for x in b})
    return [by_ts[k] for k in sorted(by_ts)]


def polygon_5m(
    client, ticker: str, start: date, end: date, refresh: bool = False, cache_only: bool = False,
) -> List[dict]:
    """
    Regular-session 5m bars for [start, end], cached per ticker. Only the missing
    head/tail is fetched, and the file is written as soon as the ticker is done,
    so an interrupted multi-hour download resumes where it stopped.

    ``cache_only`` never touches the network: whatever is cached is returned as
    is ([] if nothing is). Backtests use it so they cannot compete with a running
    download for the plan's per-minute call budget.
    """
    from .polygon_client import regular_5m_bars

    POLYGON_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _polygon_cache_path(ticker)
    cached = None if refresh or not path.exists() else pickle.loads(path.read_bytes())
    if cache_only:
        bars = cached["bars"] if cached else []
        return [b for b in bars if b["timestamp"][:10] >= start.isoformat()]
    if cached is None:
        bars = regular_5m_bars(client.aggregates(ticker, 5, "minute", start, end), ticker)
        cached = {"start": start, "end": end, "bars": bars}
    else:
        bars = cached["bars"]
        if cached["start"] > start + timedelta(days=5):
            head = client.aggregates(ticker, 5, "minute", start, cached["start"])
            bars = _merge(regular_5m_bars(head, ticker), bars)
            cached["start"] = start
        if cached["end"] < end:
            # Re-fetch the last cached day too: it may have been partial.
            tail = client.aggregates(ticker, 5, "minute", cached["end"], end)
            bars = _merge(bars, regular_5m_bars(tail, ticker))
            cached["end"] = end
        cached["bars"] = bars
    path.write_bytes(pickle.dumps(cached))
    return [b for b in cached["bars"] if b["timestamp"][:10] >= start.isoformat()]


def polygon_history(
    tickers: Sequence[str],
    years: float = 2.0,
    refresh: bool = False,
    client=None,
    log=print,
    cache_only: bool = False,
) -> Dict[str, Dict[str, List[dict]]]:
    """
    Same shape as ``fetch_history``: {"5m"|"1h"|"1d": {ticker: [bar dicts]}}. The
    hourly and daily frames are derived from the 5m bars (see polygon_client).
    Tickers Polygon cannot serve are logged and left empty.
    """
    from .config import get_settings
    from .polygon_client import PolygonClient, PolygonError, resample_daily, resample_hourly

    if client is None and not cache_only:
        settings = get_settings()
        client = PolygonClient(settings.polygon_api_key, settings.polygon_calls_per_minute)
    end = date.today()
    start = end - timedelta(days=int(365 * years) - _PLAN_EDGE_DAYS)
    data: Dict[str, Dict[str, List[dict]]] = {"5m": {}, "1h": {}, "1d": {}}
    for i, t in enumerate(tickers, 1):
        try:
            bars = polygon_5m(client, t, start, end, refresh, cache_only)
        except PolygonError as exc:
            log(f"  [{i}/{len(tickers)}] {t}: {exc}")
            bars = []
        else:
            calls = f" (API calls so far: {client.calls})" if client is not None else ""
            log(f"  [{i}/{len(tickers)}] {t}: {len(bars)} bars{calls}")
        data["5m"][t] = bars
        data["1h"][t] = resample_hourly(bars)
        data["1d"][t] = resample_daily(bars)
    return data
