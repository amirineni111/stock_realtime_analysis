from __future__ import annotations
import logging
import threading
import time as _time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence

import pandas as pd
import yfinance as yf
from yfinance import shared as yf_shared
from zoneinfo import ZoneInfo

from .models import StockBar, StockQuote

US_EASTERN = ZoneInfo("America/New_York")

# One batched download per timeframe per scan, regardless of watchlist size.
INTERVAL_PERIOD = {
    "1m": "1d",    # Live Quotes page
    "5m": "5d",    # primary scoring timeframe
    "1h": "1mo",   # MTF hourly + S/R
    "1d": "6mo",   # MTF daily + S/R + prev close + avg $ volume
}

_INTERVAL_DELTA = {
    "1m": timedelta(minutes=1),
    "5m": timedelta(minutes=5),
    "1h": timedelta(hours=1),
}

# Higher timeframes change slowly — cache them so steady-state auto-refresh costs
# a single 5m request per scan instead of three.
_CACHE_TTL_SECONDS = {"1h": 600.0, "1d": 1800.0}

# yf.download keeps per-call results in module-level globals (yfinance.shared._DFS),
# reset at the start of every call. Concurrent calls from different YFClient
# instances (scanner, indices, paper, quotes page) clobber each other and tickers
# fail with "'NoneType' object is not subscriptable" — so serialize them process-wide.
_DOWNLOAD_LOCK = threading.Lock()
# Per-request timeout (seconds). yfinance's default of 10s is tight when a batch
# fires ~20 requests at once.
_DOWNLOAD_TIMEOUT = 20


class DataFetchError(RuntimeError):
    """Raised when yfinance cannot return data (network/rate-limit). The scanner
    logs it once per scan; no retry loops inside a Streamlit rerun."""


def _bar_end_utc(ts: pd.Timestamp, interval: str) -> datetime:
    """UTC instant at which a bar starting at ``ts`` is complete. Daily bars
    complete at 16:00 ET on their own date."""
    if getattr(ts, "tzinfo", None) is None:
        # Daily frames can come back tz-naive (dates); intraday frames are tz-aware.
        ts = ts.tz_localize(US_EASTERN)
    if interval == "1d":
        local = ts.astimezone(US_EASTERN)
        end_local = local.replace(hour=16, minute=0, second=0, microsecond=0)
        return end_local.astimezone(timezone.utc)
    return (ts + _INTERVAL_DELTA[interval]).astimezone(timezone.utc)


def _frame_to_bars(
    df: Optional[pd.DataFrame],
    ticker: str,
    interval: str,
    now_utc: datetime,
    drop_forming: bool = True,
) -> List[StockBar]:
    """Convert one ticker's OHLCV frame into StockBars with uniform UTC-ISO
    timestamps. Drops NaN rows (halts/partial data) and — unless told otherwise —
    the still-forming last bar, so signals don't repaint between scans."""
    bars: List[StockBar] = []
    if df is None or df.empty:
        return bars
    for ts, row in df.iterrows():
        close = row.get("Close")
        if close is None or pd.isna(close):
            continue
        if pd.isna(row.get("Open")) or pd.isna(row.get("High")) or pd.isna(row.get("Low")):
            continue
        if drop_forming and _bar_end_utc(ts, interval) > now_utc:
            continue
        ts_aware = ts if getattr(ts, "tzinfo", None) is not None else ts.tz_localize(US_EASTERN)
        volume = row.get("Volume")
        bars.append(StockBar(
            ticker=ticker,
            timeframe=interval,
            timestamp=ts_aware.astimezone(timezone.utc).isoformat(),
            open=float(row["Open"]),
            high=float(row["High"]),
            low=float(row["Low"]),
            close=float(close),
            volume=int(volume) if volume is not None and not pd.isna(volume) else 0,
        ))
    bars.sort(key=lambda b: b.timestamp)
    return bars


def _slice_ticker(df: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
    """Pull one ticker's sub-frame out of a batched download. yfinance drops the
    ticker column level when only one symbol comes back, so handle both shapes."""
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        top = df.columns.get_level_values(0)
        if ticker in set(top):
            return df[ticker]
        return None
    return df


def _merge_retry(
    df: pd.DataFrame, df_retry: Optional[pd.DataFrame], retry: List[str]
) -> pd.DataFrame:
    """Replace the failed tickers' (empty) columns in a batched frame with the
    retried data. Leaves ``df`` as-is when the retry produced nothing."""
    if df_retry is None or df_retry.empty:
        return df
    if not isinstance(df_retry.columns, pd.MultiIndex):
        df_retry = pd.concat({retry[0]: df_retry}, axis=1)
    if not isinstance(df.columns, pd.MultiIndex):
        return df_retry if df.empty else df
    got = set(df_retry.columns.get_level_values(0))
    keep = df.loc[:, ~df.columns.get_level_values(0).isin(got)]
    return pd.concat([keep, df_retry], axis=1).sort_index()


class YFClient:
    def __init__(self) -> None:
        self._cache: dict = {}  # interval -> (monotonic_ts, tickers_set, {ticker: [StockBar]})
        self._lock = threading.Lock()

    @staticmethod
    def _download_locked(tickers: List[str], interval: str, quiet: bool) -> tuple:
        """One yf.download under the process-wide lock. Returns (frame, failed
        tickers). ``quiet`` mutes yfinance's own "N Failed downloads" log line."""
        yf_logger = logging.getLogger("yfinance")
        with _DOWNLOAD_LOCK:
            prev_level = yf_logger.level
            if quiet:
                yf_logger.setLevel(logging.CRITICAL)
            try:
                df = yf.download(
                    tickers=tickers,
                    interval=interval,
                    period=INTERVAL_PERIOD[interval],
                    group_by="ticker",
                    auto_adjust=False,
                    prepost=False,
                    progress=False,
                    threads=True,
                    timeout=_DOWNLOAD_TIMEOUT,
                )
                failed = set(getattr(yf_shared, "_ERRORS", {}) or {})
            finally:
                yf_logger.setLevel(prev_level)
        return df, failed

    def _download(self, tickers: Sequence[str], interval: str) -> pd.DataFrame:
        tickers = list(tickers)
        try:
            df, failed = self._download_locked(tickers, interval, quiet=True)
        except Exception as exc:
            raise DataFetchError(f"yfinance download failed ({interval}): {exc}") from exc
        if df is None:
            raise DataFetchError(f"yfinance returned no data ({interval})")

        # A transient Yahoo timeout/reset surfaces per ticker as
        # TypeError("'NoneType' object is not subscriptable") (yfinance swallows the
        # network error, then indexes the None response). Retry those once.
        retry = [t for t in tickers if t.upper() in failed]
        if retry:
            try:
                df_retry, _ = self._download_locked(retry, interval, quiet=False)
            except Exception:
                df_retry = None
            df = _merge_retry(df, df_retry, retry)
        return df

    def get_bars(self, tickers: Sequence[str], interval: str) -> Dict[str, List[StockBar]]:
        """Completed bars per ticker for one timeframe, from a single batched request.
        A ticker with no data maps to an empty list — callers degrade gracefully."""
        tickers = list(dict.fromkeys(t.upper() for t in tickers))
        ttl = _CACHE_TTL_SECONDS.get(interval)
        if ttl:
            with self._lock:
                cached = self._cache.get(interval)
            if cached and _time.monotonic() - cached[0] < ttl and set(tickers) <= cached[1]:
                return {t: cached[2].get(t, []) for t in tickers}

        df = self._download(tickers, interval)
        now_utc = datetime.now(timezone.utc)
        result: Dict[str, List[StockBar]] = {}
        for t in tickers:
            sub = _slice_ticker(df, t)
            result[t] = _frame_to_bars(sub, t, interval, now_utc)

        if ttl:
            with self._lock:
                self._cache[interval] = (_time.monotonic(), set(tickers), result)
        return result

    def get_quotes(self, tickers: Sequence[str]) -> List[StockQuote]:
        """Latest price per ticker from a batched 1m download (keeping the forming
        bar — freshest price wins for display), with prev close from daily bars."""
        tickers = list(dict.fromkeys(t.upper() for t in tickers))
        daily = self.get_bars(tickers, "1d")
        try:
            df_1m = self._download(tickers, "1m")
        except DataFetchError:
            df_1m = None

        now_utc = datetime.now(timezone.utc)
        quotes: List[StockQuote] = []
        for t in tickers:
            sub = _slice_ticker(df_1m, t) if df_1m is not None else None
            minute_bars = _frame_to_bars(sub, t, "1m", now_utc, drop_forming=False)
            d_bars = daily.get(t) or []

            last: Optional[float] = None
            as_of = ""
            volume = 0
            if minute_bars:
                last = minute_bars[-1].close
                as_of = minute_bars[-1].timestamp
                last_date = minute_bars[-1].timestamp[:10]
                volume = sum(b.volume for b in minute_bars if b.timestamp[:10] == last_date)
            else:
                last, as_of = self._fast_info_last(t)
            if last is None:
                continue

            prev_close = self._prev_close(d_bars, as_of)
            change_pct = (
                round((last - prev_close) / prev_close * 100, 3)
                if prev_close
                else None
            )
            quotes.append(StockQuote(
                ticker=t,
                last=round(last, 4),
                prev_close=prev_close,
                change_pct=change_pct,
                volume=volume,
                as_of=as_of or now_utc.isoformat(),
            ))
        return quotes

    @staticmethod
    def _prev_close(d_bars: List[StockBar], as_of: str) -> Optional[float]:
        """Previous session's close: skip the daily bar of the same session the
        quote came from, so intraday change is measured against yesterday."""
        if not d_bars:
            return None
        last_daily = d_bars[-1]
        if as_of and last_daily.timestamp[:10] == as_of[:10]:
            return d_bars[-2].close if len(d_bars) >= 2 else None
        return last_daily.close

    @staticmethod
    def _fast_info_last(ticker: str):
        try:
            info = yf.Ticker(ticker).fast_info
            last = info.get("last_price") if hasattr(info, "get") else getattr(info, "last_price", None)
            if last:
                return float(last), datetime.now(timezone.utc).isoformat()
        except Exception:
            pass
        return None, ""
