from __future__ import annotations
import json
from typing import List, Optional, Tuple


def _macd_magnitude_pts(macd_histogram: float, atr14: Optional[float], macd: Optional[float]) -> float:
    """
    Convert MACD-histogram size into 0-15 pts. Normalizes against ATR (both in price
    units, so the ratio is unitless): a histogram ~0.3×ATR is treated as full strength.
    Falls back to the legacy hist/macd ratio when ATR is unavailable.
    """
    if atr14 and atr14 > 0:
        return min(15.0, 50.0 * abs(macd_histogram) / atr14)
    return min(15.0, 15 * abs(macd_histogram) / max(abs(macd or 0.0), 0.00001))


def _momentum(
    ema9: Optional[float],
    ema20: Optional[float],
    macd_histogram: Optional[float],
    macd: Optional[float],
    rsi14: Optional[float],
    atr14: Optional[float] = None,
) -> Tuple[float, str, str]:
    """Score momentum signal (max 40 pts). Returns (score, direction, reason)."""
    score = 0.0
    reasons = []
    direction = "NEUTRAL"

    # EMA alignment (max 15 pts)
    if ema9 is not None and ema20 is not None:
        if ema9 > ema20:
            score += 15
            direction = "LONG"
            reasons.append("EMA9>EMA20 bullish")
        elif ema9 < ema20:
            score += 15
            direction = "SHORT"
            reasons.append("EMA9<EMA20 bearish")

    # MACD histogram direction + ATR-normalized strength (max 15 pts)
    if macd_histogram is not None and macd is not None:
        if macd_histogram > 0 and macd > 0:
            pts = _macd_magnitude_pts(macd_histogram, atr14, macd)
            score += pts
            reasons.append(f"MACD bullish (+{pts:.0f}pts)")
        elif macd_histogram < 0 and macd < 0:
            pts = _macd_magnitude_pts(macd_histogram, atr14, macd)
            score += pts
            reasons.append(f"MACD bearish (+{pts:.0f}pts)")
        elif macd_histogram > 0:  # histogram positive but MACD crossing zero
            score += 8
            reasons.append("MACD histogram positive crossing")
        elif macd_histogram < 0:
            score += 8
            reasons.append("MACD histogram negative crossing")

    # RSI 40-60 trending confirmation (max 10 pts)
    if rsi14 is not None:
        if direction == "LONG" and 40 <= rsi14 <= 65:
            score += 10
            reasons.append(f"RSI {rsi14:.1f} momentum zone")
        elif direction == "SHORT" and 35 <= rsi14 <= 60:
            score += 10
            reasons.append(f"RSI {rsi14:.1f} momentum zone")

    signal = "LONG_MOMENTUM" if direction == "LONG" else (
        "SHORT_MOMENTUM" if direction == "SHORT" else "NEUTRAL_MOMENTUM"
    )
    return round(score, 1), signal, "; ".join(reasons)


def _mean_reversion(
    rsi14: Optional[float],
    close: Optional[float],
    bb_upper: Optional[float],
    bb_lower: Optional[float],
    bb_middle: Optional[float],
    day_high: Optional[float],
    day_low: Optional[float],
) -> Tuple[float, str, str]:
    """Score mean reversion signal (max 40 pts). Returns (score, direction, reason)."""
    score = 0.0
    reasons = []
    direction = "NEUTRAL"

    # RSI extreme (max 20 pts)
    if rsi14 is not None:
        if rsi14 < 30:
            pts = 20 * (30 - rsi14) / 30
            score += min(20, pts)
            direction = "LONG"
            reasons.append(f"RSI oversold {rsi14:.1f}")
        elif rsi14 > 70:
            pts = 20 * (rsi14 - 70) / 30
            score += min(20, pts)
            direction = "SHORT"
            reasons.append(f"RSI overbought {rsi14:.1f}")

    # Bollinger Band proximity (max 10 pts)
    if close is not None and bb_upper is not None and bb_lower is not None and bb_middle is not None:
        band_width = bb_upper - bb_lower
        if band_width > 0:
            dist_lower = (close - bb_lower) / band_width
            dist_upper = (bb_upper - close) / band_width
            if dist_lower <= 0.15:
                score += 10
                direction = "LONG"
                reasons.append("Price at lower Bollinger Band")
            elif dist_upper <= 0.15:
                score += 10
                direction = "SHORT"
                reasons.append("Price at upper Bollinger Band")
            elif dist_lower <= 0.30:
                score += 5
                direction = direction if direction != "NEUTRAL" else "LONG"
                reasons.append("Price near lower Bollinger Band")
            elif dist_upper <= 0.30:
                score += 5
                direction = direction if direction != "NEUTRAL" else "SHORT"
                reasons.append("Price near upper Bollinger Band")

    # Day range position (max 10 pts)
    if (
        close is not None
        and day_high is not None
        and day_low is not None
        and day_high > day_low
    ):
        day_range = day_high - day_low
        pos = (close - day_low) / day_range
        if pos <= 0.30:
            score += 10
            direction = "LONG" if direction == "NEUTRAL" else direction
            reasons.append(f"Bottom 30% of day range ({pos*100:.0f}%)")
        elif pos >= 0.70:
            score += 10
            direction = "SHORT" if direction == "NEUTRAL" else direction
            reasons.append(f"Top 30% of day range ({pos*100:.0f}%)")

    signal = "LONG_REVERSION" if direction == "LONG" else (
        "SHORT_REVERSION" if direction == "SHORT" else "NEUTRAL_REVERSION"
    )
    return round(score, 1), signal, "; ".join(reasons)


def _day_breakout(
    close: Optional[float],
    day_high: Optional[float],
    day_low: Optional[float],
    or_high: Optional[float],
    or_low: Optional[float],
    atr14: Optional[float],
    phase: Optional[str],
    minutes_since_open: Optional[float],
    minutes_to_close: Optional[float],
) -> Tuple[float, str, str]:
    """
    Score day/opening-range breakout (max 20 pts). Replaces the forex session breakout:
    the liquidity-window bonus rewards the high-volume first/last hour of the regular
    session, and breakout points measure how far the close pushed beyond the prior day
    extreme or the 09:30-10:00 opening range, in ATR units.
    """
    score = 0.0
    reasons = []
    direction = "NEUTRAL"

    # Liquidity-window bonus (max 10 pts)
    if phase == "REGULAR":
        near_open = minutes_since_open is not None and minutes_since_open <= 60
        near_close = minutes_to_close is not None and minutes_to_close <= 60
        if near_open or near_close:
            score += 10
            reasons.append("Open/close high-volume window")
        else:
            score += 5
            reasons.append("Regular session active")

    # Breakout points (max 10 pts): strongest of day-extreme or opening-range break.
    # day_high/day_low exclude the current bar so the extreme can actually be broken.
    if close is not None and atr14 is not None and atr14 > 0:
        long_pts = 0.0
        short_pts = 0.0
        long_reason = ""
        short_reason = ""
        if day_high is not None and close > day_high:
            dist = (close - day_high) / atr14
            long_pts = min(10, 10 * dist)
            long_reason = f"Breaking day high ({dist:.1f}×ATR)"
        if or_high is not None and close > or_high:
            dist = (close - or_high) / atr14
            pts = min(10, 10 * dist)
            if pts > long_pts:
                long_pts = pts
                long_reason = f"Breaking opening range high ({dist:.1f}×ATR)"
        if day_low is not None and close < day_low:
            dist = (day_low - close) / atr14
            short_pts = min(10, 10 * dist)
            short_reason = f"Breaking day low ({dist:.1f}×ATR)"
        if or_low is not None and close < or_low:
            dist = (or_low - close) / atr14
            pts = min(10, 10 * dist)
            if pts > short_pts:
                short_pts = pts
                short_reason = f"Breaking opening range low ({dist:.1f}×ATR)"

        if long_pts > short_pts and long_pts > 0:
            score += long_pts
            direction = "LONG"
            reasons.append(long_reason)
        elif short_pts > 0:
            score += short_pts
            direction = "SHORT"
            reasons.append(short_reason)

    signal = "LONG_BREAKOUT" if direction == "LONG" else (
        "SHORT_BREAKOUT" if direction == "SHORT" else "NEUTRAL_BREAKOUT"
    )
    return round(score, 1), signal, "; ".join(reasons)


def _mtf_confluence(
    m5_dir: Optional[str],
    hourly_dir: Optional[str],
    daily_dir: Optional[str],
) -> Tuple[float, str]:
    """
    Score multi-timeframe alignment.
    Full (all 3 agree): +30 pts, "FULL"
    Two of three agree: +15 pts, "PARTIAL"
    Conflict/missing: 0 pts, "NONE"
    """
    dirs = [d for d in (m5_dir, hourly_dir, daily_dir) if d and d != "NEUTRAL"]
    if not dirs:
        return 0.0, "NONE"
    long_count = dirs.count("LONG")
    short_count = dirs.count("SHORT")
    total = len(dirs)
    if long_count == total or short_count == total:
        return 30.0, "FULL"
    elif long_count >= 2 or short_count >= 2:
        return 15.0, "PARTIAL"
    return 0.0, "NONE"


def _sr_proximity(
    close: Optional[float],
    atr14: Optional[float],
    sr_levels: list,
    dominant_direction: str,
) -> Tuple[float, str, bool, Optional[float], Optional[float]]:
    """
    Score proximity to key S/R levels.
    Returns (score, reason, at_key_level, nearest_support, nearest_resistance).
    """
    if not sr_levels or not close or not atr14 or atr14 == 0:
        return 0.0, "", False, None, None

    supports = [lv["price"] for lv in sr_levels if lv["type"] == "S" and lv["price"] <= close]
    resistances = [lv["price"] for lv in sr_levels if lv["type"] == "R" and lv["price"] >= close]

    nearest_support = max(supports) if supports else None
    nearest_resistance = min(resistances) if resistances else None

    score = 0.0
    reasons: List[str] = []
    at_key_level = False

    if nearest_support is not None:
        dist = (close - nearest_support) / atr14
        if dist <= 0.3:
            score += 25
            at_key_level = True
            reasons.append(f"AT support {nearest_support:.2f}")
        elif dist <= 1.0 and dominant_direction == "LONG":
            score += 15
            reasons.append(f"Near support {nearest_support:.2f}")

    if nearest_resistance is not None:
        dist = (nearest_resistance - close) / atr14
        if dist <= 0.3:
            score += 25
            at_key_level = True
            reasons.append(f"AT resistance {nearest_resistance:.2f}")
        elif dist <= 1.0 and dominant_direction == "SHORT":
            score += 15
            reasons.append(f"Near resistance {nearest_resistance:.2f}")

    return round(min(score, 25), 1), "; ".join(reasons), at_key_level, nearest_support, nearest_resistance


# Regime thresholds on ADX: above TREND → momentum playbook, below RANGE → reversion.
_ADX_TREND = 25.0
_ADX_RANGE = 18.0

# ATR multiples for suggested stop/target. Reward:risk stays fixed — the target is
# derived from the final stop distance, so widening the stop widens the target too.
_STOP_ATR_MULT = 1.5
_RR = 1.5
# Noise floor for the stop: 1×ATR on 5m bars can be a few cents, inside ordinary
# bid/ask noise. Never risk less than 0.30% of the entry price.
_MIN_STOP_PCT = 0.003

# Forward-testing a target under this % of entry is untradeable noise after
# commissions/slippage (replaces the forex 3×spread thin-edge gate).
MIN_TARGET_PCT = 0.15


def _regime_weights(adx14: Optional[float]) -> Tuple[float, float, str]:
    """
    Decide how much to trust momentum vs mean-reversion given trend strength.
    Returns (momentum_weight, reversion_weight, regime_label). Blends linearly
    between the range/trend thresholds to avoid hard flip-flopping.
    """
    if adx14 is None:
        return 1.0, 1.0, "UNKNOWN"
    if adx14 >= _ADX_TREND:
        return 1.0, 0.0, "TREND"
    if adx14 <= _ADX_RANGE:
        return 0.0, 1.0, "RANGE"
    t = (adx14 - _ADX_RANGE) / (_ADX_TREND - _ADX_RANGE)
    return round(t, 3), round(1 - t, 3), "MIXED"


def _trade_levels(
    direction: str,
    entry: Optional[float],
    atr14: Optional[float],
) -> dict:
    """ATR-based stop/target/RR for an actionable direction. Empty dict if not computable."""
    if direction not in ("LONG", "SHORT") or not entry or not atr14 or atr14 <= 0:
        return {}
    stop_dist = max(_STOP_ATR_MULT * atr14, _MIN_STOP_PCT * entry)
    tgt_dist = _RR * stop_dist
    if direction == "LONG":
        stop = entry - stop_dist
        target = entry + tgt_dist
    else:
        stop = entry + stop_dist
        target = entry - tgt_dist
    return {
        "suggested_entry": round(entry, 2),
        "suggested_stop": round(stop, 2),
        "suggested_target": round(target, 2),
        "stop_dollars": round(stop_dist, 2),
        "target_dollars": round(tgt_dist, 2),
        "stop_pct": round(stop_dist / entry * 100, 2),
        "target_pct": round(tgt_dist / entry * 100, 2),
        "rr_ratio": round(_RR, 2),
    }


def score_ticker(
    ticker: str,
    last: Optional[float],
    avg_dollar_volume: Optional[float],
    indicators: dict,
    phase: Optional[str] = None,
    minutes_since_open: Optional[float] = None,
    minutes_to_close: Optional[float] = None,
    min_avg_dollar_volume: float = 0.0,
    hourly_direction: Optional[str] = None,
    daily_direction: Optional[str] = None,
    sr_levels: Optional[list] = None,
) -> dict:
    """
    Compute all signal scores and produce final trade_signal.
    Returns a dict merging into StockSnapshot.
    """
    close = indicators.get("close")
    rsi14 = indicators.get("rsi14")
    ema9 = indicators.get("ema9")
    ema20 = indicators.get("ema20")
    macd = indicators.get("macd")
    macd_hist = indicators.get("macd_histogram")
    atr14 = indicators.get("atr14")
    adx14 = indicators.get("adx14")
    bb_upper = indicators.get("bb_upper")
    bb_lower = indicators.get("bb_lower")
    bb_middle = indicators.get("bb_middle")
    day_high = indicators.get("day_high")
    day_low = indicators.get("day_low")
    or_high = indicators.get("or_high")
    or_low = indicators.get("or_low")

    # Liquidity gate (replaces the forex spread gate). 0 threshold disables it.
    risk_notes = []
    thin_liquidity = False
    illiquid = False
    if min_avg_dollar_volume > 0 and avg_dollar_volume is not None:
        if avg_dollar_volume < min_avg_dollar_volume * 0.1:
            illiquid = True
            risk_notes.append(
                f"Illiquid: avg ${avg_dollar_volume:,.0f}/day (min ${min_avg_dollar_volume:,.0f})"
            )
        elif avg_dollar_volume < min_avg_dollar_volume:
            thin_liquidity = True
            risk_notes.append(
                f"Thin liquidity: avg ${avg_dollar_volume:,.0f}/day (min ${min_avg_dollar_volume:,.0f})"
            )

    mom_raw, mom_signal, mom_reason = _momentum(ema9, ema20, macd_hist, macd, rsi14, atr14)
    rev_raw, rev_signal, rev_reason = _mean_reversion(
        rsi14, close, bb_upper, bb_lower, bb_middle, day_high, day_low
    )
    brk_score, brk_signal, brk_reason = _day_breakout(
        close, day_high, day_low, or_high, or_low, atr14,
        phase, minutes_since_open, minutes_to_close,
    )

    # Regime gate: in trends trust momentum, in ranges trust reversion. These are
    # opposite playbooks — weighting (instead of summing both) stops them cancelling.
    w_mom, w_rev, regime = _regime_weights(adx14)
    mom_score = round(mom_raw * w_mom, 1)
    rev_score = round(rev_raw * w_rev, 1)

    # Weighted dominant direction — a suppressed playbook gets no vote.
    long_w = sum(sc for sig, sc in (
        (mom_signal, mom_score), (rev_signal, rev_score), (brk_signal, brk_score)
    ) if "LONG" in sig)
    short_w = sum(sc for sig, sc in (
        (mom_signal, mom_score), (rev_signal, rev_score), (brk_signal, brk_score)
    ) if "SHORT" in sig)
    dominant = "LONG" if long_w > short_w else (
        "SHORT" if short_w > long_w else "NEUTRAL"
    )

    # MTF confluence bonus (0-30 pts)
    mtf_bonus, mtf_confluence = _mtf_confluence(dominant, hourly_direction, daily_direction)

    # S/R proximity bonus (0-25 pts)
    sr_bonus, sr_reason, at_key_level, nearest_support, nearest_resistance = _sr_proximity(
        close, atr14, sr_levels or [], dominant
    )

    total = mom_score + rev_score + brk_score + mtf_bonus + sr_bonus

    # Penalize thin liquidity
    if thin_liquidity:
        total = max(0, total - 20)

    # Hourly alignment gate: forex forward-testing showed candidates firing against
    # the next timeframe up were the biggest loss bucket. An actionable signal must
    # not fight the hourly trend.
    hourly_opposes = (
        dominant in ("LONG", "SHORT")
        and hourly_direction in ("LONG", "SHORT")
        and hourly_direction != dominant
    )

    if illiquid:
        trade_signal = "AVOID"
        reason = f"Too illiquid (avg ${(avg_dollar_volume or 0):,.0f}/day)"
    elif hourly_opposes and total >= 45:
        trade_signal = "WATCH_ONLY"
        reason = f"{dominant} setup ({total:.0f}pts) but hourly trend is {hourly_direction} — countertrend"
    elif total >= 70 and dominant == "LONG" and mtf_bonus >= 15:
        trade_signal = "STRONG_BUY"
        reason = f"Strong long setup ({total:.0f}pts, MTF:{mtf_confluence})"
    elif total >= 70 and dominant == "SHORT" and mtf_bonus >= 15:
        trade_signal = "STRONG_SHORT"
        reason = f"Strong short setup ({total:.0f}pts, MTF:{mtf_confluence})"
    elif total >= 45 and dominant == "LONG":
        trade_signal = "BUY_CANDIDATE"
        reason = f"Long candidate ({total:.0f}pts)"
    elif total >= 45 and dominant == "SHORT":
        trade_signal = "SHORT_CANDIDATE"
        reason = f"Short candidate ({total:.0f}pts)"
    elif total >= 25:
        trade_signal = "WATCH_ONLY"
        reason = f"Mixed signals ({total:.0f}pts)"
    else:
        trade_signal = "AVOID"
        reason = f"No clear setup ({total:.0f}pts)"

    # ATR-based stop/target/RR for actionable directions. Entry is the last completed
    # 5m close (yfinance has no bid/ask).
    entry = last or close
    levels = _trade_levels(dominant, entry, atr14) if trade_signal not in (
        "AVOID", "WATCH_ONLY"
    ) else {}
    if levels and levels.get("target_pct", 100.0) < MIN_TARGET_PCT:
        risk_notes.append(
            f"Target {levels['target_pct']:.2f}% of price — thin edge"
        )

    signal_parts = []
    if mom_reason:
        signal_parts.append(f"Momentum: {mom_reason}")
    if rev_reason:
        signal_parts.append(f"Reversion: {rev_reason}")
    if brk_reason:
        signal_parts.append(f"Breakout: {brk_reason}")
    if mtf_confluence != "NONE":
        signal_parts.append(f"MTF: {mtf_confluence} ({hourly_direction or '?'}/{daily_direction or '?'})")
    if sr_reason:
        signal_parts.append(f"S/R: {sr_reason}")

    return {
        "momentum_score": mom_score,
        "reversion_score": rev_score,
        "breakout_score": brk_score,
        "adx14": adx14,
        "regime": regime,
        "suggested_entry": levels.get("suggested_entry"),
        "suggested_stop": levels.get("suggested_stop"),
        "suggested_target": levels.get("suggested_target"),
        "stop_dollars": levels.get("stop_dollars"),
        "target_dollars": levels.get("target_dollars"),
        "stop_pct": levels.get("stop_pct"),
        "rr_ratio": levels.get("rr_ratio"),
        "mtf_score": mtf_bonus,
        "mtf_confluence": mtf_confluence,
        "sr_score": sr_bonus,
        "at_key_level": at_key_level,
        "nearest_support": nearest_support,
        "nearest_resistance": nearest_resistance,
        "sr_levels_json": json.dumps(sr_levels[:5]) if sr_levels else None,
        "total_score": round(total, 1),
        "trade_signal": trade_signal,
        "signal_reason": reason,
        "risk_notes": "; ".join(risk_notes + signal_parts),
        "market_phase": phase,
    }
