from stocks.signals import (
    _day_breakout,
    _mtf_confluence,
    _regime_weights,
    trade_levels,
    score_ticker,
)
from stocks.relative_strength import calculate_rs, rs_assessment, rs_bonus


# ── trade_levels ─────────────────────────────────────────────────────────────

def test_trade_levels_long_atr_stop():
    levels = trade_levels("LONG", 100.0, 0.8)
    assert levels["suggested_entry"] == 100.0
    assert levels["suggested_stop"] == 98.0      # 2.5 × 0.8 = 2.0 > 0.50% floor
    assert levels["suggested_target"] == 103.0   # 1.5R
    assert levels["stop_dollars"] == 2.0
    assert levels["target_dollars"] == 3.0
    assert levels["stop_pct"] == 2.0
    assert levels["rr_ratio"] == 1.5


def test_trade_levels_min_stop_pct_floor():
    levels = trade_levels("LONG", 100.0, 0.1)
    assert levels["stop_dollars"] == 0.5         # floor: 0.50% of 100 beats 2.5×0.1
    assert levels["suggested_stop"] == 99.5


def test_trade_levels_short():
    levels = trade_levels("SHORT", 50.0, 1.0)
    assert levels["suggested_stop"] == 52.5      # 2.5 × 1.0
    assert levels["suggested_target"] == 46.25   # 1.5R below entry


def test_trade_levels_neutral_empty():
    assert trade_levels("NEUTRAL", 100.0, 0.8) == {}
    assert trade_levels("LONG", None, 0.8) == {}
    assert trade_levels("LONG", 100.0, 0.0) == {}


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
    # The 5m direction is what gets confirmed, not a third vote.
    assert _mtf_confluence("LONG", "LONG", "LONG") == (30.0, "FULL")
    assert _mtf_confluence("LONG", "LONG", None) == (15.0, "PARTIAL")
    assert _mtf_confluence("LONG", "LONG", "SHORT") == (0.0, "CONFLICT")
    assert _mtf_confluence("LONG", "SHORT", "SHORT") == (0.0, "OPPOSED")
    assert _mtf_confluence(None, None, None) == (0.0, "NONE")


def test_mtf_confluence_requires_real_higher_timeframe_confirmation():
    """
    The regression this fix exists for: a 5m read with NOTHING confirming it used
    to score FULL (+30 pts), which is most of what pushed mediocre setups over the
    70-point STRONG threshold.
    """
    assert _mtf_confluence("LONG", None, None) == (0.0, "UNCONFIRMED")
    assert _mtf_confluence("SHORT", "NEUTRAL", "NEUTRAL") == (0.0, "UNCONFIRMED")


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
    result = score_ticker(**_strong_long_kwargs(close=103.0, ema20=100.0, day_high=102.0),
                          max_entry_range_pos=None)
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


# ── Pullback-entry gate ──────────────────────────────────────────────────────

def test_entry_range_pos_is_oriented_to_the_trade():
    from stocks.signals import entry_range_pos
    assert entry_range_pos(100.0, 110.0, 100.0, "LONG") == -1.0   # long at the low
    assert entry_range_pos(110.0, 110.0, 100.0, "LONG") == 1.0    # long at the high
    assert entry_range_pos(110.0, 110.0, 100.0, "SHORT") == -1.0  # short at the high
    assert entry_range_pos(105.0, 105.0, 105.0, "LONG") is None   # no range yet
    assert entry_range_pos(105.0, 110.0, 100.0, "NEUTRAL") is None


def test_pullback_gate_blocks_a_long_bought_at_the_day_high():
    # Same STRONG_BUY setup that passes the extension test, closing at the high.
    result = score_ticker(**_strong_long_kwargs(close=103.0, ema20=100.0, day_high=102.0))
    assert result["trade_signal"] == "WATCH_ONLY"
    assert "pullback" in result["signal_reason"]
    assert result["suggested_entry"] is None
    assert result["entry_range_pos"] > 0


def test_pullback_gate_keeps_a_long_bought_in_the_bottom_quarter():
    kwargs = _strong_long_kwargs(close=103.0, ema20=100.0, day_high=112.0)
    kwargs["indicators"]["day_low"] = 102.0          # close sits 10% up the range
    result = score_ticker(**kwargs)
    assert result["trade_signal"] == "STRONG_BUY"
    assert result["entry_range_pos"] == -0.8


def test_pullback_gate_fails_closed_without_a_day_range():
    kwargs = _strong_long_kwargs(close=103.0, ema20=100.0, day_high=102.0)
    kwargs["indicators"]["day_low"] = kwargs["indicators"]["day_high"]
    result = score_ticker(**kwargs)
    assert result["trade_signal"] == "WATCH_ONLY"
    assert "unknown" in result["signal_reason"]


# ── Optional market-regime gate ─────────────────────────────────────────────

def test_market_trend_compares_last_close_with_its_50_day_average():
    from stocks.signals import market_trend
    assert market_trend([100.0] * 49 + [101.0]) == "UP"
    assert market_trend([100.0] * 49 + [99.0]) == "DOWN"
    assert market_trend([100.0] * 49) is None            # not enough history


def test_regime_gate_is_off_unless_enabled():
    kwargs = _strong_long_kwargs(close=103.0, ema20=100.0, day_high=112.0)
    kwargs["indicators"]["day_low"] = 102.0
    assert score_ticker(**kwargs, market_trend="DOWN")["trade_signal"] == "STRONG_BUY"
    gated = score_ticker(**kwargs, market_trend="DOWN", stand_aside_in_downtrend=True)
    assert gated["trade_signal"] == "WATCH_ONLY"
    assert "50-day" in gated["signal_reason"] and gated["suggested_entry"] is None
    up = score_ticker(**kwargs, market_trend="UP", stand_aside_in_downtrend=True)
    assert up["trade_signal"] == "STRONG_BUY" and up["market_trend"] == "UP"
