from stocks.signals import (
    _day_breakout,
    _mtf_confluence,
    _regime_weights,
    _trade_levels,
)
from stocks.relative_strength import calculate_rs, rs_assessment, rs_bonus


# ── _trade_levels ─────────────────────────────────────────────────────────────

def test_trade_levels_long_atr_stop():
    levels = _trade_levels("LONG", 100.0, 0.8)
    assert levels["suggested_entry"] == 100.0
    assert levels["suggested_stop"] == 98.8      # 1.5 × 0.8 = 1.2 > 0.30% floor
    assert levels["suggested_target"] == 101.8   # 1.5R
    assert levels["stop_dollars"] == 1.2
    assert levels["target_dollars"] == 1.8
    assert levels["stop_pct"] == 1.2
    assert levels["rr_ratio"] == 1.5


def test_trade_levels_min_stop_pct_floor():
    levels = _trade_levels("LONG", 100.0, 0.1)
    assert levels["stop_dollars"] == 0.3         # floor: 0.30% of 100 beats 1.5×0.1
    assert levels["suggested_stop"] == 99.7


def test_trade_levels_short():
    levels = _trade_levels("SHORT", 50.0, 1.0)
    assert levels["suggested_stop"] == 51.5
    assert levels["suggested_target"] == 47.75


def test_trade_levels_neutral_empty():
    assert _trade_levels("NEUTRAL", 100.0, 0.8) == {}
    assert _trade_levels("LONG", None, 0.8) == {}
    assert _trade_levels("LONG", 100.0, 0.0) == {}


# ── _day_breakout ─────────────────────────────────────────────────────────────

def test_breakout_day_high_in_open_window():
    score, signal, reason = _day_breakout(
        close=101.0, day_high=100.0, day_low=95.0, or_high=None, or_low=None,
        atr14=0.5, phase="REGULAR", minutes_since_open=30, minutes_to_close=330,
    )
    # 10 window bonus + capped 10 breakout ((101-100)/0.5 = 2×ATR)
    assert score == 20.0
    assert signal == "LONG_BREAKOUT"
    assert "day high" in reason.lower()


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
