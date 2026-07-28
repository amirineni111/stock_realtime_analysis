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

    # Liquidity-window bonus (max 10 pts). Forward-testing showed the first hour
    # was the worst entry window (27% win rate, −0.33R/trade): the old +10
    # near-open bonus pushed marginal setups over the actionable threshold right
    # into opening chop. Only the closing hour keeps the full bonus now.
    if phase == "REGULAR":
        near_close = minutes_to_close is not None and minutes_to_close <= 60
        if near_close:
            score += 10
            reasons.append("Closing high-volume window")
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
    dominant: Optional[str],
    hourly_dir: Optional[str],
    daily_dir: Optional[str],
) -> Tuple[float, str]:
    """
    Score higher-timeframe confirmation of the 5m ``dominant`` direction.

    The 5m direction is the thing being confirmed — it is NOT a vote. It used to be
    passed in as one of three votes, so a setup whose hourly and daily reads were
    both NEUTRAL scored "FULL" (+30 pts) purely on its own say-so. Those 30 free
    points are most of what let a mediocre setup clear the 70-point STRONG
    threshold, which is why the STRONG tier never outperformed. (The identical bug
    was measured in the forex sibling: 103 of 143 FULL rows had a NEUTRAL/absent
    higher timeframe, and 20 had *both* neutral.)

    FULL now requires both the hourly and the daily trend present and agreeing.
    """
    if dominant not in ("LONG", "SHORT"):
        return 0.0, "NONE"

    votes = [d for d in (hourly_dir, daily_dir) if d in ("LONG", "SHORT")]
    if not votes:
        return 0.0, "UNCONFIRMED"

    agree = sum(1 for v in votes if v == dominant)
    oppose = len(votes) - agree
    if oppose:
        return 0.0, "OPPOSED" if agree == 0 else "CONFLICT"
    return (30.0, "FULL") if len(votes) == 2 else (15.0, "PARTIAL")


def _sr_proximity(
    close: Optional[float],
    atr14: Optional[float],
    sr_levels: list,
    dominant_direction: str,
) -> Tuple[float, str, bool, bool, Optional[float], Optional[float]]:
    """
    Score structure *relative to the trade direction*.

    A level only helps when it sits **behind** the trade (support beneath a long,
    resistance above a short) — that is where the stop shelters. A level directly
    **ahead** is a wall: it caps the move before the target can be reached.

    The previous version was direction-blind — both the `dist <= 0.3` branches
    awarded +25 and set at_key_level regardless of which way the trade pointed, so
    a long pinned under resistance scored identically to a long bouncing off
    support. Combined with the MTF bug this manufactured the STRONG tier.

    Returns (score, reason, at_key_level, blocked_ahead, nearest_support, nearest_resistance).
    ``score`` may be negative when structure opposes the trade.
    """
    if not sr_levels or not close or not atr14 or atr14 <= 0:
        return 0.0, "", False, False, None, None

    supports = [lv["price"] for lv in sr_levels if lv["type"] == "S" and lv["price"] <= close]
    resistances = [lv["price"] for lv in sr_levels if lv["type"] == "R" and lv["price"] >= close]

    nearest_support = max(supports) if supports else None
    nearest_resistance = min(resistances) if resistances else None

    if dominant_direction not in ("LONG", "SHORT"):
        return 0.0, "", False, False, nearest_support, nearest_resistance

    if dominant_direction == "LONG":
        behind, ahead = nearest_support, nearest_resistance
        behind_label, ahead_label = "support", "resistance"
    else:
        behind, ahead = nearest_resistance, nearest_support
        behind_label, ahead_label = "resistance", "support"

    score = 0.0
    reasons: List[str] = []
    at_key_level = False
    blocked_ahead = False

    if behind is not None:
        dist = abs(close - behind) / atr14
        if dist <= 0.3:
            score += 25
            at_key_level = True
            reasons.append(f"AT {behind_label} {behind:.2f} (entry at structure)")
        elif dist <= 1.0:
            score += 15
            reasons.append(f"Near {behind_label} {behind:.2f}")

    if ahead is not None:
        dist = abs(ahead - close) / atr14
        # Target sits ~3.75×ATR out (1.5×RR on a 2.5×ATR stop), so a level inside
        # 1.5×ATR means the trade is very unlikely to reach target unimpeded.
        if dist <= 1.5:
            score -= 25
            blocked_ahead = True
            reasons.append(f"BLOCKED by {ahead_label} {ahead:.2f} ({dist:.1f}×ATR ahead)")
        elif dist <= 2.5:
            score -= 10
            reasons.append(f"{ahead_label.capitalize()} {ahead:.2f} close ahead ({dist:.1f}×ATR)")

    return (
        round(max(-25.0, min(score, 25.0)), 1),
        "; ".join(reasons),
        at_key_level,
        blocked_ahead,
        nearest_support,
        nearest_resistance,
    )


# Regime thresholds on ADX: above TREND → momentum playbook, below RANGE → reversion.
_ADX_TREND = 25.0
_ADX_RANGE = 18.0

# ATR multiples for suggested stop/target. Reward:risk stays fixed — the target is
# derived from the final stop distance, so widening the stop widens the target too.
# Forward-testing at 1.5×ATR / 0.30% floor put the median stop at 0.52% of price:
# 54% of trades stopped out (median 48 min), i.e. the stop sat inside ordinary
# 5m-bar noise. Widened to keep the stop outside one bar's wiggle.
_STOP_ATR_MULT = 2.5
_RR = 1.5
# Noise floor for the stop: never risk less than 0.50% of the entry price.
_MIN_STOP_PCT = 0.005

# Over-extension gate: STRONG signals fire when momentum + breakout + MTF all
# align — i.e. late in a move. Forward-tested STRONG_BUYs stopped out in a median
# of 18 minutes (buying the local extreme). Beyond this many ATRs from EMA20 the
# signal downgrades to WATCH_ONLY until price pulls back.
_MAX_EXTENSION_ATR = 2.0

# Forward-testing a target under this % of entry is untradeable noise after
# commissions/slippage (replaces the forex 3×spread thin-edge gate).
MIN_TARGET_PCT = 0.15

# ── Transaction cost ────────────────────────────────────────────────────────
# yfinance returns last-trade prices with no bid/ask, so unlike the forex sibling
# the round-trip cost cannot be observed — it has to be estimated. Liquidity is by
# far the strongest determinant of effective spread, so cost is tiered on 20-day
# average dollar volume. Figures are round-trip basis points of price and are
# deliberately conservative: understating cost is what makes a backtest lie.
_COST_TIERS_BPS = (
    (50_000_000.0, 3.0),    # mega/large cap: ~1c on a $100 name, both sides
    (10_000_000.0, 8.0),
    (2_000_000.0, 20.0),
)
_COST_BPS_THIN = 50.0       # below $2M/day the spread alone eats a third of a 1.5R target

# Hard veto: above this fraction of risk, cost dominates any plausible edge in the
# score. At a 0.6% stop this vetoes anything costing more than ~9bps round trip,
# i.e. roughly the sub-$10M-ADV tier.
_MAX_COST_RATIO = 0.15

# How far above cost-adjusted breakeven a modelled probability must sit before the
# trade is worth taking. Trading at exactly breakeven just donates the spread to
# the market maker while adding variance, so demand a real cushion.
_PROB_MARGIN = 0.04


def estimate_cost_pct(avg_dollar_volume: Optional[float]) -> float:
    """
    Estimated round-trip transaction cost as a percentage of price.

    Returns the most pessimistic tier when liquidity is unknown — an unmeasured
    cost is not a zero cost, and treating it as one is how a screener talks itself
    into illiquid names.
    """
    if avg_dollar_volume is None or avg_dollar_volume <= 0:
        return _COST_BPS_THIN / 100.0
    for floor, bps in _COST_TIERS_BPS:
        if avg_dollar_volume >= floor:
            return bps / 100.0
    return _COST_BPS_THIN / 100.0


def breakeven_win_rate(rr: float = _RR, cost_ratio: float = 0.0) -> float:
    """
    Win rate at which a bracket exactly breaks even, including round-trip cost.

    expectancy = p·(rr − c) − (1 − p)·(1 + c) = 0  ⇒  p = (1 + c) / (1 + rr)

    with everything expressed in units of the stop distance. At rr=1.5 and zero
    cost this is exactly 0.400 — which is why an edgeless system lands on ~40%,
    and why a measured 40% win rate is indistinguishable from random entry.
    """
    return (1.0 + cost_ratio) / (1.0 + rr)


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
    model_prob: Optional[float] = None,
    strength_bonus: float = 0.0,
) -> dict:
    """
    Compute all signal scores and produce final trade_signal.
    Returns a dict merging into StockSnapshot.

    ``model_prob`` is the trained model's P(target before stop) for this setup, when
    one is available. It acts as a veto, never as a promoter: the rule-based score
    still has to propose the setup, and the model decides whether the measured odds
    justify paying the cost of the round trip.

    ``strength_bonus`` is the relative-strength-vs-SPY adjustment. It is passed in
    rather than applied afterwards so that the score driving the decision, the score
    shown on the dashboard, and the score the model trains on are the same number.
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

    # MTF confluence bonus (0-30 pts) — requires real higher-timeframe confirmation
    mtf_bonus, mtf_confluence = _mtf_confluence(dominant, hourly_direction, daily_direction)

    # S/R structure, direction-aware (-25 to +25 pts)
    (
        sr_bonus, sr_reason, at_key_level, blocked_ahead,
        nearest_support, nearest_resistance,
    ) = _sr_proximity(close, atr14, sr_levels or [], dominant)

    total = mom_score + rev_score + brk_score + mtf_bonus + sr_bonus + strength_bonus

    # Penalize thin liquidity
    if thin_liquidity:
        total = max(0, total - 20)

    # Cost ratio: estimated round-trip cost as a fraction of the risk being taken.
    # Computed from the stop we would actually use, so it reflects the real drag on
    # expectancy rather than an abstract bps number.
    entry_px = last or close
    provisional = _trade_levels(dominant, entry_px, atr14)
    prov_stop_pct = provisional.get("stop_pct") or 0.0
    cost_pct = estimate_cost_pct(avg_dollar_volume)
    cost_ratio = round(cost_pct / prov_stop_pct, 4) if prov_stop_pct > 0 else None
    cost_veto = cost_ratio is not None and cost_ratio > _MAX_COST_RATIO
    if cost_veto:
        risk_notes.append(
            f"Cost {cost_ratio:.0%} of risk (est. {cost_pct:.2f}% round trip vs "
            f"{prov_stop_pct:.2f}% stop) — above {_MAX_COST_RATIO:.0%} limit"
        )

    ahead_block_label = "resistance" if dominant == "LONG" else "support"
    if blocked_ahead:
        risk_notes.append(f"Nearby {ahead_block_label} blocks the path to target")

    # Hourly alignment gate: forward-testing showed candidates firing against
    # the next timeframe up were the biggest loss bucket. An actionable signal must
    # not fight the hourly trend.
    hourly_opposes = (
        dominant in ("LONG", "SHORT")
        and hourly_direction in ("LONG", "SHORT")
        and hourly_direction != dominant
    )

    # Probability gate. When a trained model is available, an actionable signal must
    # clear the cost-adjusted breakeven win rate by a margin — this is the whole
    # point of the learning loop: the score proposes, the measured model disposes.
    be_p = breakeven_win_rate(_RR, cost_ratio or 0.0)
    required_p = round(be_p + _PROB_MARGIN, 4)
    prob_veto = model_prob is not None and model_prob < required_p

    if illiquid:
        trade_signal = "AVOID"
        reason = f"Too illiquid (avg ${(avg_dollar_volume or 0):,.0f}/day)"
    elif cost_veto:
        trade_signal = "AVOID"
        reason = f"Est. cost {cost_ratio:.0%} of risk exceeds {_MAX_COST_RATIO:.0%} limit"
    elif hourly_opposes and total >= 45:
        trade_signal = "WATCH_ONLY"
        reason = f"{dominant} setup ({total:.0f}pts) but hourly trend is {hourly_direction} — countertrend"
    elif blocked_ahead and total >= 45:
        trade_signal = "WATCH_ONLY"
        reason = f"{dominant} setup ({total:.0f}pts) but {ahead_block_label} blocks the target"
    elif prob_veto and total >= 45:
        trade_signal = "WATCH_ONLY"
        reason = (
            f"{dominant} setup ({total:.0f}pts) but model P(win)={model_prob:.0%} "
            f"< {required_p:.0%} required"
        )
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

    # Over-extension gate (see _MAX_EXTENSION_ATR): don't chase a STRONG signal
    # that is already stretched far from its EMA20.
    extension_atr: Optional[float] = None
    if close is not None and ema20 is not None and atr14:
        extension_atr = round((close - ema20) / atr14, 2)
    if trade_signal in ("STRONG_BUY", "STRONG_SHORT") and extension_atr is not None:
        if trade_signal == "STRONG_BUY" and extension_atr > _MAX_EXTENSION_ATR:
            trade_signal = "WATCH_ONLY"
            reason = f"Extended {extension_atr:.1f}×ATR above EMA20 — wait for pullback"
        elif trade_signal == "STRONG_SHORT" and extension_atr < -_MAX_EXTENSION_ATR:
            trade_signal = "WATCH_ONLY"
            reason = f"Extended {abs(extension_atr):.1f}×ATR below EMA20 — wait for pullback"

    if model_prob is not None and trade_signal not in ("AVOID", "WATCH_ONLY"):
        reason += f" — P(win) {model_prob:.0%} vs {required_p:.0%} needed"

    # ATR-based stop/target/RR for actionable directions. Entry is the last completed
    # 5m close (yfinance has no bid/ask).
    levels = provisional if trade_signal not in ("AVOID", "WATCH_ONLY") else {}
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
        "blocked_ahead": blocked_ahead,
        "nearest_support": nearest_support,
        "nearest_resistance": nearest_resistance,
        "sr_levels_json": json.dumps(sr_levels[:5]) if sr_levels else None,
        "extension_atr": extension_atr,
        "cost_pct": round(cost_pct, 4),
        "cost_ratio": cost_ratio,
        "model_prob": model_prob,
        "required_prob": required_p,
        "dominant": dominant,
        # Levels the trade *would* use, exposed even when the signal is not actionable
        # so feature extraction always sees a real stop size rather than a zero.
        "prov_stop_pct": provisional.get("stop_pct"),
        "prov_target_pct": provisional.get("target_pct"),
        "total_score": round(total, 1),
        "trade_signal": trade_signal,
        "signal_reason": reason,
        "risk_notes": "; ".join(risk_notes + signal_parts),
        "market_phase": phase,
    }
