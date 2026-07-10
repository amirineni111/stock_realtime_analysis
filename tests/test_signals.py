from stocks.signals import (
    _day_breakout,
    _mtf_confluence,
    _regime_weights,
    _trade_levels,
    score_ticker,
)
from stocks.relative_strength import calculate_rs, rs_assessment, rs_bonus


# ── _trade_levels ─────────────────────────────────────────────────────────────

def test_trade_levels_long_atr_stop():
    levels = _trade_levels("LONG", 100.0, 0.8)
    assert levels["suggested_entry"] == 100.0
    assert levels["suggested_stop"] == 98.0      # 2.5 × 0.8 = 2.0 > 0.50% floor
    assert levels["suggested_target"] == 103.0   # 1.5R
    assert levels["stop_dollars"] == 2.0
    assert levels["target_dollars"] == 3.0
    assert levels["stop_pct"] == 2.0
    assert levels["rr_ratio"] == 1.5


def test_trade_levels_min_stop_pct_floor():
    levels = _trade_levels("LONG", 100.0, 0.1)
    assert levels["stop_dollars"] == 0.5         # floor: 0.50% of 100 beats 2.5×0.1
    assert levels["suggested_stop"] == 99.5


def test_trade_levels_short():
    levels = _trade_levels("SHORT", 50.0, 1.0)
    assert levels["suggested_stop"] == 52.5      # 2.5 × 1.0
    assert levels["suggested_target"] == 46.25   # 1.5R below entry


def test_trade_levels_neutral_empty():
    assert _trade_levels("NEUTRAL", 100.0, 0.8) == {}
    assert _trade_levels("LONG", None, 0.8) == {}
    assert _trade_levels("LONG", 100.0, 0.0) == {}


# ── _day_breakout ─────────────────────────────────────────────────────────────

def test_breakout_day_high_near_open_no_bonus():
    score, signal, reason = _day_breakout(
        close=101.0, day_high=100.0, day_low=95.0, or_high=None, or_low=None,
        atr14=0.5, phase="REGULAR", minutes_since_open=30, minutes_to_close=330,
    )
    # First hour gets only the base session 5 pts (open bonus removed after
    # forward-testing) + capped 10 breakout ((101-100)/0.5 = 2×ATR)
    assert score == 15.0
    assert signal == "LONG_BREAKOUT"
    assert "day high" in reason.lower()


def test_breakout_day_high_near_close_keeps_bonus():
    score, signal, reason = _day_breakout(
        close=101.0, day_high=100.0, day_low=95.0, or_high=None, or_low=None,
        atr14=0.5, phase="REGULAR", minutes_since_open=330, minutes_to_close=30,
    )
    # 10 closing-window bonus + capped 10 breakout
    assert score == 20.0
    assert signal == "LONG_BREAKOUT"
    assert "closing" in reason.lower()


def test_breakout_opening_range_short():
    score, signal, reason = _day_breakout(
        close=94.8, day_high=100.0, day_low=94.0, or_high=96.0, or_low=95.0,
        atr14=0.5, phase="REGULAR", minutes_since_open=120, minutes_to_close=180,
    )
    # mid-session bonus 5 + OR-low break (95−94.8)/0.5 = 0.4×ATR → 4 pts
    assert score == 9.0
    assert signal == "SHORT_BREAKOUT"
    assert "opening range low" in reason.lower()


def test_breakout_closed_phase_no_bonus():
    score, signal, _ = _day_breakout(
        close=99.0, day_high=100.0, day_low=95.0, or_high=None, or_low=None,
        atr14=0.5, phase="CLOSED", minutes_since_open=None, minutes_to_close=None,
    )
    assert score == 0.0
    assert signal == "NEUTRAL_BREAKOUT"


# ── regime / MTF ──────────────────────────────────────────────────────────────

def test_regime_weights():
    assert _regime_weights(30.0) == (1.0, 0.0, "TREND")
    assert _regime_weights(10.0) == (0.0, 1.0, "RANGE")
    w_mom, w_rev, label = _regime_weights(21.5)
    assert label == "MIXED"
    assert w_mom == 0.5 and w_rev == 0.5
    assert _regime_weights(None) == (1.0, 1.0, "UNKNOWN")


def test_mtf_confluence():
    assert _mtf_confluence("LONG", "LONG", "LONG") == (30.0, "FULL")
    assert _mtf_confluence("LONG", "LONG", "SHORT") == (15.0, "PARTIAL")
    assert _mtf_confluence("LONG", "SHORT", None) == (0.0, "NONE")
    assert _mtf_confluence(None, None, None) == (0.0, "NONE")


# ── relative strength ─────────────────────────────────────────────────────────

def test_calculate_rs():
    assert calculate_rs(2.5, 1.0) == 1.5
    assert calculate_rs(None, 1.0) is None
    assert calculate_rs(2.5, None) is None


def test_rs_assessment_thresholds():
    assert rs_assessment(1.5) == "OUTPERFORMING"
    assert rs_assessment(-1.5) == "UNDERPERFORMING"
    assert rs_assessment(0.5) == "IN_LINE"
    assert rs_assessment(None) is None


def test_rs_bonus_matrix():
    assert rs_bonus("OUTPERFORMING", "STRONG_BUY") == 10.0
    assert rs_bonus("OUTPERFORMING", "SHORT_CANDIDATE") == -5.0
    assert rs_bonus("UNDERPERFORMING", "STRONG_SHORT") == 10.0
    assert rs_bonus("UNDERPERFORMING", "BUY_CANDIDATE") == -5.0
    assert rs_bonus("IN_LINE", "STRONG_BUY") == 0.0
    assert rs_bonus("OUTPERFORMING", "WATCH_ONLY") == 0.0
    assert rs_bonus(None, "STRONG_BUY") == 0.0


# ── over-extension gate ───────────────────────────────────────────────────────

def _strong_long_kwargs(close: float, ema20: float, day_high: float) -> dict:
    """A setup that scores STRONG_BUY: trending regime, aligned momentum,
    day-high breakout in the closing window, full MTF confluence."""
    return dict(
        ticker="TEST",
        last=close,
        avg_dollar_volume=None,
        indicators={
            "close": close, "rsi14": 50.0, "ema9": ema20 + 1.0, "ema20": ema20,
            "macd": 1.0, "macd_histogram": 1.0, "atr14": 2.0, "adx14": 30.0,
            "day_high": day_high, "day_low": day_high - 5.0,
        },
        phase="REGULAR",
        minutes_since_open=330,
        minutes_to_close=30,
        hourly_direction="LONG",
        daily_direction="LONG",
    )


def test_strong_buy_within_extension_limit_keeps_signal():
    # close 1.5×ATR above EMA20 — inside the 2×ATR limit
    result = score_ticker(**_strong_long_kwargs(close=103.0, ema20=100.0, day_high=102.0))
    assert result["trade_signal"] == "STRONG_BUY"
    assert result["suggested_stop"] == 98.0     # 2.5 × ATR(2.0) below entry


def test_strong_buy_overextended_downgrades_to_watch_only():
    # close 2.5×ATR above EMA20 — chasing; downgrade and suppress trade levels
    result = score_ticker(**_strong_long_kwargs(close=105.0, ema20=100.0, day_high=104.0))
    assert result["trade_signal"] == "WATCH_ONLY"
    assert "Extended" in result["signal_reason"]
    assert result["suggested_entry"] is None


def test_strong_short_overextended_downgrades_to_watch_only():
    result = score_ticker(
        ticker="TEST",
        last=95.0,
        avg_dollar_volume=None,
        indicators={
            "close": 95.0, "rsi14": 50.0, "ema9": 99.0, "ema20": 100.0,
            "macd": -1.0, "macd_histogram": -1.0, "atr14": 2.0, "adx14": 30.0,
            "day_high": 101.0, "day_low": 96.0,
        },
        phase="REGULAR",
        minutes_since_open=330,
        minutes_to_close=30,
        hourly_direction="SHORT",
        daily_direction="SHORT",
    )
    assert result["trade_signal"] == "WATCH_ONLY"
    assert "Extended" in result["signal_reason"]
