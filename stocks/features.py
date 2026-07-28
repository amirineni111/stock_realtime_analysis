"""
Canonical feature extraction for the stock direction model.

This module is the single source of truth for what the model sees. Both the live
scanner (serving) and the trainer (from stored ``features_json``) go through
``build_features``, which is what keeps train/serve skew from creeping in.

Design note — everything here is **direction-relative**. A feature is expressed
from the point of view of the trade being taken, so the model learns "does this
kind of setup work" rather than having to learn long and short as separate
regimes. ``direction`` itself is kept as a feature so a genuine long/short
asymmetry can still be represented — and for equities that asymmetry is real:
index drift is upward, so shorts start from a worse base rate than longs.

What differs from the forex feature set
---------------------------------------
- No spread. yfinance quotes are last-trade prices, so transaction cost is
  *estimated* from liquidity (see ``signals.estimate_cost_pct``) rather than
  observed. ``cost_ratio`` still enters the model, because a 25bps round trip
  against a 0.5% stop is half the edge regardless of how it was measured.
- ``session_progress`` replaces the forex hour-of-day sin/cos. The equity day has
  hard 09:30/16:00 boundaries and a well-known volatility smile; a cyclical
  24-hour encoding would spend most of its range on hours that never trade.
- ``rel_volume`` and ``log_dollar_volume`` have no forex analogue and are among
  the few genuinely stock-specific predictors: a breakout on 3× normal volume is
  a different event from the same price move on 0.4× volume.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

from .timeutil import session_progress

# Ordered feature contract. Changing this list invalidates stored models, which
# is why FEATURE_VERSION travels with every persisted model and every logged row.
FEATURE_NAMES: Sequence[str] = (
    "direction",
    "total_score",
    "momentum_score",
    "reversion_score",
    "breakout_score",
    "mtf_score",
    "sr_score",
    "adx14",
    "rsi_dir",
    "atr_pct",
    "bb_width_pct",
    "stop_pct",
    "cost_ratio",
    "ema_gap_atr",
    "ema50_gap_atr",
    "macd_hist_atr",
    "extension_atr",
    "dist_to_target_level_atr",
    "dist_to_protective_level_atr",
    "hourly_agrees",
    "daily_agrees",
    "rs_dir",
    "day_change_dir",
    "range_pos_dir",
    "rel_volume",
    "log_dollar_volume",
    "session_progress",
    "first_hour",
    "last_hour",
    "phase_regular",
)

FEATURE_VERSION = 1

# Cap for ATR-normalised ratios. Without this a near-zero ATR turns one bar into
# an enormous outlier that dominates a linear model's fit.
_CLIP = 10.0


def _f(value: Optional[float], default: float = 0.0) -> float:
    """Coerce to float, mapping None/NaN/inf to a neutral default."""
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def _clip(value: float, limit: float = _CLIP) -> float:
    return max(-limit, min(limit, value))


def _agreement(htf_direction: Optional[str], direction: int) -> float:
    """+1 when the higher timeframe backs the trade, -1 when it fights it, 0 if flat."""
    if htf_direction not in ("LONG", "SHORT"):
        return 0.0
    htf = 1 if htf_direction == "LONG" else -1
    return 1.0 if htf == direction else -1.0


def build_features(snap: dict, direction: int) -> dict:
    """
    Build the model feature dict for taking ``direction`` (+1 long / -1 short) on
    the setup described by ``snap`` (a StockSnapshot dump merged with the scoring
    result, or a stock_snapshots row).

    Every value is finite; missing inputs collapse to a neutral default so a
    partially populated snapshot still scores rather than raising.
    """
    d = 1.0 if direction >= 0 else -1.0

    close = _f(snap.get("close"))
    atr = _f(snap.get("atr14"))
    # Guard every ATR-normalised ratio behind a positive-ATR check.
    atr_ok = atr > 0

    stop_pct = _f(snap.get("stop_pct"))
    cost_pct = _f(snap.get("cost_pct"))
    cost_ratio = (cost_pct / stop_pct) if stop_pct > 0 else 0.0

    rsi = snap.get("rsi14")
    # Orient RSI to the trade: high means "RSI supports this direction".
    rsi_dir = (_f(rsi, 50.0) if d > 0 else 100.0 - _f(rsi, 50.0)) if rsi is not None else 50.0

    ema9 = _f(snap.get("ema9"))
    ema20 = _f(snap.get("ema20"))
    ema50 = _f(snap.get("ema50"))
    ema_gap_atr = _clip((ema9 - ema20) / atr * d) if atr_ok else 0.0
    ema50_gap_atr = _clip((ema20 - ema50) / atr * d) if (atr_ok and ema50) else 0.0
    macd_hist_atr = _clip(_f(snap.get("macd_histogram")) / atr * d) if atr_ok else 0.0

    # How stretched price is from its own mean, oriented to the trade. Positive =
    # already extended in our favour (chasing); negative = entering on a pullback.
    # This is the quantity the STRONG-signal over-extension gate thresholds on, so
    # the model gets to learn the cutoff instead of inheriting a hand-picked one.
    extension_atr = _clip((close - ema20) / atr * d) if (atr_ok and ema20) else 0.0

    # Structure ahead of / behind the trade. For a long the "target level" is the
    # resistance overhead (how much runway before it stalls) and the "protective
    # level" is the support beneath. Mirrored for a short.
    support = snap.get("nearest_support")
    resistance = snap.get("nearest_resistance")
    ahead = resistance if d > 0 else support
    behind = support if d > 0 else resistance

    if atr_ok and ahead is not None and close:
        dist_target = _clip(abs(_f(ahead) - close) / atr)
    else:
        dist_target = 0.0
    if atr_ok and behind is not None and close:
        dist_protective = _clip(abs(close - _f(behind)) / atr)
    else:
        dist_protective = 0.0

    # Relative strength vs SPY, oriented: positive means the name is moving with
    # the trade relative to the market.
    rs = snap.get("rs_vs_spy")
    rs_dir = _clip(_f(rs) * d, 10.0) if rs is not None else 0.0

    # Where price sits in the day's range, oriented to the trade.
    # +1 = extended in our favour (breakout entry), -1 = against us (reversion entry).
    hi = _f(snap.get("day_high"))
    lo = _f(snap.get("day_low"))
    if hi > lo and close:
        pos = (close - lo) / (hi - lo)
        range_pos_dir = _clip((pos - 0.5) * 2.0 * d, 2.0)
    else:
        range_pos_dir = 0.0

    # Liquidity on a log scale: the difference between a $2M/day and a $20M/day name
    # matters, the difference between $2B and $20B barely does.
    adv = _f(snap.get("avg_dollar_volume"))
    log_dollar_volume = math.log10(adv) if adv > 0 else 0.0

    progress = session_progress(snap.get("as_of"))
    if progress is None:
        progress = 0.5
        first_hour = last_hour = 0.0
    else:
        # Forward-testing showed the opening hour was the worst entry window by a
        # wide margin, and the closing hour the best. Flagged explicitly so the
        # model can price them without having to bend a linear session-progress term.
        first_hour = 1.0 if progress <= (60.0 / 390.0) else 0.0
        last_hour = 1.0 if progress >= (330.0 / 390.0) else 0.0

    phase = snap.get("market_phase") or ""

    return {
        "direction": d,
        "total_score": _f(snap.get("total_score")),
        "momentum_score": _f(snap.get("momentum_score")),
        "reversion_score": _f(snap.get("reversion_score")),
        "breakout_score": _f(snap.get("breakout_score")),
        "mtf_score": _f(snap.get("mtf_score")),
        "sr_score": _f(snap.get("sr_score")),
        "adx14": _f(snap.get("adx14"), 20.0),
        "rsi_dir": rsi_dir,
        # Scale-free volatility. Raw ATR is not comparable across tickers (a $900
        # stock's ATR is ~100× a $9 stock's in price units), which would otherwise
        # let the model use ATR as a ticker identifier.
        "atr_pct": _clip(atr / close * 100.0, 20.0) if close else 0.0,
        "bb_width_pct": _clip(_f(snap.get("bb_width_pct")), 50.0),
        "stop_pct": _clip(stop_pct, 20.0),
        "cost_ratio": _clip(cost_ratio, 2.0),
        "ema_gap_atr": ema_gap_atr,
        "ema50_gap_atr": ema50_gap_atr,
        "macd_hist_atr": macd_hist_atr,
        "extension_atr": extension_atr,
        "dist_to_target_level_atr": dist_target,
        "dist_to_protective_level_atr": dist_protective,
        "hourly_agrees": _agreement(snap.get("hourly_direction"), int(d)),
        "daily_agrees": _agreement(snap.get("daily_direction"), int(d)),
        "rs_dir": rs_dir,
        "day_change_dir": _clip(_f(snap.get("day_change_pct")) * d, 20.0),
        "range_pos_dir": range_pos_dir,
        "rel_volume": _clip(_f(snap.get("rel_volume"), 1.0), 10.0),
        "log_dollar_volume": log_dollar_volume,
        "session_progress": progress,
        "first_hour": first_hour,
        "last_hour": last_hour,
        "phase_regular": 1.0 if phase == "REGULAR" else 0.0,
    }


def to_vector(features: dict) -> list:
    """Flatten a feature dict into FEATURE_NAMES order."""
    return [_f(features.get(name)) for name in FEATURE_NAMES]
