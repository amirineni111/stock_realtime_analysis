from __future__ import annotations
from typing import Dict, List, Optional, Tuple

from .config import AppSettings
from .models import StockBar, StockQuote, StockSnapshot, ScanRequest, ScanSummary
from .yf_client import YFClient, DataFetchError
from .storage import Storage
from .tickers import BENCHMARK
from .indicators import (
    compute_all,
    compute_trend_direction,
    detect_sr_levels,
    range_high_low,
    window_high_low,
)
from .signals import score_ticker, MIN_TARGET_PCT
from .market_hours import (
    current_market_phase,
    market_open_today_utc,
    opening_range_end_utc,
    minutes_since_open,
    minutes_to_close,
)
from .relative_strength import calculate_rs, rs_assessment, rs_bonus

# Reused across scans within one Streamlit session so the 1h/1d TTL caches survive.
_shared_client = YFClient()


def _prev_session_close(d1_dicts: List[dict], as_of: str) -> Optional[float]:
    """Previous session's daily close relative to ``as_of`` (skips the daily bar of
    the same session so intraday change is measured against yesterday)."""
    if not d1_dicts:
        return None
    if as_of and d1_dicts[-1].get("timestamp", "")[:10] == as_of[:10]:
        return d1_dicts[-2]["close"] if len(d1_dicts) >= 2 else None
    return d1_dicts[-1]["close"]


def _avg_dollar_volume(d1_dicts: List[dict], days: int = 20) -> Optional[float]:
    recent = d1_dicts[-days:]
    if not recent:
        return None
    values = [b["close"] * b["volume"] for b in recent]
    return round(sum(values) / len(values), 0)


def run_scan(
    settings: AppSettings,
    storage: Storage,
    request: ScanRequest,
) -> ScanSummary:
    client = _shared_client
    summary = ScanSummary()
    scan_id = storage.start_scan()

    tickers = list(dict.fromkeys(request.tickers))
    fetch_list = tickers + ([BENCHMARK] if BENCHMARK not in tickers else [])

    # Three batched downloads per scan (1h/1d are TTL-cached inside the client).
    def _fetch(interval: str) -> Dict[str, List[StockBar]]:
        try:
            return client.get_bars(fetch_list, interval)
        except DataFetchError as exc:
            storage.log_ticker(scan_id, "ALL", None, str(exc))
            return {}

    bars_5m = _fetch("5m")
    bars_1h = _fetch("1h")
    bars_1d = _fetch("1d")

    phase = current_market_phase()
    open_utc = market_open_today_utc()
    or_end_utc = opening_range_end_utc()
    mins_open = minutes_since_open()
    mins_close = minutes_to_close()

    # SPY day change anchors the relative-strength post-pass.
    spy_change_pct: Optional[float] = None
    spy_5m = bars_5m.get(BENCHMARK) or []
    spy_1d = [b.model_dump() for b in (bars_1d.get(BENCHMARK) or [])]
    if spy_5m and spy_1d:
        spy_last = spy_5m[-1].close
        spy_prev = _prev_session_close(spy_1d, spy_5m[-1].timestamp)
        if spy_prev:
            spy_change_pct = round((spy_last - spy_prev) / spy_prev * 100, 3)

    snapshots: List[StockSnapshot] = []
    quotes: List[StockQuote] = []
    bars_by_ticker: Dict[str, List[dict]] = {}

    for ticker in tickers:
        try:
            m5_bars = bars_5m.get(ticker) or []
            if not m5_bars:
                summary.errors += 1
                storage.log_ticker(scan_id, ticker, None, "No candle data")
                continue

            bar_dicts = [b.model_dump() for b in m5_bars]
            h1_dicts = [b.model_dump() for b in (bars_1h.get(ticker) or [])]
            d1_dicts = [b.model_dump() for b in (bars_1d.get(ticker) or [])]

            indicators = compute_all(bar_dicts)
            as_of = bar_dicts[-1]["timestamp"]
            last = indicators.get("close")

            # Day range since the 09:30 open, EXCLUDING the current bar so its own
            # extreme can actually be "broken". Falls back to a rolling 4h window
            # (48 5m bars) outside market hours or when no session bars exist yet.
            prior_dicts = bar_dicts[:-1] if len(bar_dicts) > 1 else bar_dicts
            day_high, day_low = range_high_low(prior_dicts, open_utc)
            if day_high is None or day_low is None:
                recent = prior_dicts[-48:] if len(prior_dicts) >= 48 else prior_dicts
                day_high = max(b["high"] for b in recent)
                day_low = min(b["low"] for b in recent)
            or_high, or_low = window_high_low(bar_dicts, open_utc, or_end_utc)
            indicators["day_high"] = day_high
            indicators["day_low"] = day_low
            indicators["or_high"] = or_high
            indicators["or_low"] = or_low

            hourly_direction = compute_trend_direction(h1_dicts) if h1_dicts else None
            daily_direction = compute_trend_direction(d1_dicts) if d1_dicts else None

            # Replace the noisy 5-min "day change" with change vs the previous
            # session's daily close — this also drives relative strength vs SPY.
            prev_close = _prev_session_close(d1_dicts, as_of)
            if prev_close and last:
                indicators["day_change_pct"] = round((last - prev_close) / prev_close * 100, 3)

            avg_dollar_volume = _avg_dollar_volume(d1_dicts)

            # S/R levels from daily (longer-term structure) + hourly (shorter-term)
            sr_levels: List[dict] = []
            if d1_dicts:
                sr_levels += detect_sr_levels(d1_dicts, lookback=50)
            if h1_dicts:
                sr_levels += detect_sr_levels(h1_dicts, lookback=30)
            sr_levels.sort(key=lambda x: x["strength"], reverse=True)

            scoring = score_ticker(
                ticker=ticker,
                last=last,
                avg_dollar_volume=avg_dollar_volume,
                indicators=indicators,
                phase=phase,
                minutes_since_open=mins_open,
                minutes_to_close=mins_close,
                min_avg_dollar_volume=request.min_avg_dollar_volume,
                hourly_direction=hourly_direction,
                daily_direction=daily_direction,
                sr_levels=sr_levels,
            )

            snapshot = StockSnapshot(
                ticker=ticker,
                last=last,
                prev_close=prev_close,
                avg_dollar_volume=avg_dollar_volume,
                open=indicators.get("open"),
                high=indicators.get("high"),
                low=indicators.get("low"),
                close=indicators.get("close"),
                day_change_pct=indicators.get("day_change_pct"),
                rsi14=indicators.get("rsi14"),
                ema9=indicators.get("ema9"),
                ema20=indicators.get("ema20"),
                ema50=indicators.get("ema50"),
                macd=indicators.get("macd"),
                macd_signal=indicators.get("macd_signal"),
                macd_histogram=indicators.get("macd_histogram"),
                atr14=indicators.get("atr14"),
                adx14=indicators.get("adx14"),
                bb_upper=indicators.get("bb_upper"),
                bb_middle=indicators.get("bb_middle"),
                bb_lower=indicators.get("bb_lower"),
                bb_width_pct=indicators.get("bb_width_pct"),
                market_phase=scoring.get("market_phase"),
                day_high=day_high,
                day_low=day_low,
                or_high=or_high,
                or_low=or_low,
                momentum_score=scoring.get("momentum_score", 0.0),
                reversion_score=scoring.get("reversion_score", 0.0),
                breakout_score=scoring.get("breakout_score", 0.0),
                regime=scoring.get("regime"),
                total_score=scoring.get("total_score", 0.0),
                trade_signal=scoring.get("trade_signal", "AVOID"),
                signal_reason=scoring.get("signal_reason", ""),
                risk_notes=scoring.get("risk_notes", ""),
                as_of=as_of,
                suggested_entry=scoring.get("suggested_entry"),
                suggested_stop=scoring.get("suggested_stop"),
                suggested_target=scoring.get("suggested_target"),
                stop_dollars=scoring.get("stop_dollars"),
                target_dollars=scoring.get("target_dollars"),
                stop_pct=scoring.get("stop_pct"),
                rr_ratio=scoring.get("rr_ratio"),
                hourly_direction=hourly_direction,
                daily_direction=daily_direction,
                mtf_score=scoring.get("mtf_score", 0.0),
                mtf_confluence=scoring.get("mtf_confluence"),
                nearest_support=scoring.get("nearest_support"),
                nearest_resistance=scoring.get("nearest_resistance"),
                sr_score=scoring.get("sr_score", 0.0),
                at_key_level=scoring.get("at_key_level", False),
                sr_levels_json=scoring.get("sr_levels_json"),
            )
            snapshots.append(snapshot)
            bars_by_ticker[ticker] = bar_dicts
            summary.tickers_scanned += 1
            storage.log_ticker(scan_id, ticker, snapshot.trade_signal, None)
            if snapshot.trade_signal not in ("AVOID", "WATCH_ONLY"):
                summary.signals_found += 1

            # Quote derived from the same bars — no extra API call during a scan.
            if last is not None:
                last_date = as_of[:10]
                day_volume = sum(b["volume"] for b in bar_dicts if b["timestamp"][:10] == last_date)
                quotes.append(StockQuote(
                    ticker=ticker,
                    last=last,
                    prev_close=prev_close,
                    change_pct=indicators.get("day_change_pct"),
                    volume=day_volume,
                    as_of=as_of,
                ))
        except Exception as exc:
            summary.errors += 1
            storage.log_ticker(scan_id, ticker, None, str(exc))

    # Forward-evaluate previously-tracked signals against this scan's fresh bars,
    # then arm any new actionable signals.
    _ACTIONABLE = ("STRONG_BUY", "BUY_CANDIDATE", "STRONG_SHORT", "SHORT_CANDIDATE")
    for s in snapshots:
        ticker_bars = bars_by_ticker.get(s.ticker) or []
        if ticker_bars:
            try:
                storage.evaluate_tracked_signals(s.ticker, ticker_bars)
            except Exception as exc:
                storage.log_ticker(scan_id, s.ticker, None, f"Tracking eval failed: {exc}")
        # Thin-edge gate: a target under MIN_TARGET_PCT of entry is untradeable
        # noise after commissions/slippage — tracking it pollutes win-rate stats.
        thin_edge = (
            s.suggested_entry is not None
            and s.target_dollars is not None
            and s.suggested_entry > 0
            and (s.target_dollars / s.suggested_entry * 100) < MIN_TARGET_PCT
        )
        if (
            s.trade_signal in _ACTIONABLE
            and s.suggested_stop is not None
            and s.suggested_target is not None
            and ticker_bars
            and not thin_edge
        ):
            direction = -1 if "SHORT" in s.trade_signal else 1
            try:
                storage.record_tracked_signal(
                    ticker=s.ticker,
                    signal=s.trade_signal,
                    direction=direction,
                    entry=s.suggested_entry,
                    stop=s.suggested_stop,
                    target=s.suggested_target,
                    stop_dollars=s.stop_dollars or 0.0,
                    target_dollars=s.target_dollars or 0.0,
                    atr14=s.atr14 or 0.0,
                    entry_ts=ticker_bars[-1]["timestamp"],
                )
            except Exception as exc:
                storage.log_ticker(scan_id, s.ticker, None, f"Tracking record failed: {exc}")

    # Post-scan: relative strength vs SPY adjusts total_score (mirrors the forex
    # currency-strength bonus).
    for s in snapshots:
        s.spy_change_pct = spy_change_pct
        s.rs_vs_spy = calculate_rs(s.day_change_pct, spy_change_pct)
        s.rs_assessment = rs_assessment(s.rs_vs_spy)
        s.total_score = round(s.total_score + rs_bonus(s.rs_assessment, s.trade_signal), 1)

    # Sort by score descending (after RS adjustment)
    snapshots.sort(key=lambda s: s.total_score, reverse=True)
    storage.save_snapshots(scan_id, snapshots)

    if quotes:
        storage.save_quotes(quotes)

    storage.finish_scan(scan_id, summary)
    return summary
