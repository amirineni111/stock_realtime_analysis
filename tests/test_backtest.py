"""The replay's trade resolution and arming rules, on hand-built bars."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from stocks.backtest import Candidate, resolve, simulate, summarize

T0 = datetime(2026, 9, 15, 15, 0, tzinfo=timezone.utc)   # 11:00 ET


def _bar(i: int, high: float, low: float, close: float) -> dict:
    ts = (T0 + timedelta(minutes=5 * i)).isoformat()
    return {"timestamp": ts, "open": close, "high": high, "low": low, "close": close, "volume": 1}


def _cand(direction=1, entry_dt=T0, ticker="AAPL", mins=90.0, cost_pct=0.0) -> Candidate:
    return Candidate(
        ticker=ticker, signal="BUY_CANDIDATE" if direction == 1 else "SHORT_CANDIDATE",
        direction=direction, entry_ts=entry_dt.isoformat(), entry_dt=entry_dt,
        entry=100.0, stop_dist=1.0, cost_pct=cost_pct, total_score=50.0,
        extension_atr=0.0, range_pos=-1.0, minutes_since_open=mins, features=None,
    )


def test_long_hits_target_net_of_cost():
    c = _cand(cost_pct=0.1)                       # 0.1% of $100 = $0.10 = 0.1R
    resolve(c, [_bar(1, 100.5, 99.5, 100.2), _bar(2, 101.6, 100.1, 101.5)], rrs=(1.5,))
    _, r, reason = c.outcomes[1.5]
    assert reason == "TARGET"
    assert abs(r - 1.4) < 1e-9


def test_stop_is_checked_before_target_within_a_bar():
    c = _cand()
    resolve(c, [_bar(1, 102.0, 98.5, 100.0)], rrs=(1.5,))
    assert c.outcomes[1.5][1:] == (-1.0, "STOP")


def test_short_mirrors_long():
    c = _cand(direction=-1)
    resolve(c, [_bar(1, 100.2, 98.4, 98.5)], rrs=(1.5,))
    assert c.outcomes[1.5][1:] == (1.5, "TARGET")


def test_unresolved_trade_exits_at_the_session_close():
    c = _cand()
    resolve(c, [_bar(1, 100.4, 99.8, 100.1), _bar(2, 100.6, 100.0, 100.5)], rrs=(1.0, 2.0))
    assert c.outcomes[1.0][1:] == (0.5, "CLOSE")
    assert c.outcomes[2.0][1:] == (0.5, "CLOSE")


def test_simulate_applies_first_hour_open_trade_and_cooldown_rules():
    early = _cand(mins=30.0)                                  # opening chop
    a = _cand(entry_dt=T0)
    overlapping = _cand(entry_dt=T0 + timedelta(minutes=10))  # a still open
    other_side = _cand(direction=-1, entry_dt=T0 + timedelta(minutes=10))
    after_cooldown = _cand(entry_dt=T0 + timedelta(minutes=60))
    for c in (early, a, overlapping, other_side, after_cooldown):
        c.outcomes[1.5] = (c.entry_dt + timedelta(minutes=20), 1.0, "TARGET")
    taken = simulate([after_cooldown, overlapping, early, other_side, a], lambda c: True)
    assert taken == [a, other_side, after_cooldown]


def test_summarize_reports_expectancy_not_just_win_rate():
    cs = [_cand(), _cand(), _cand()]
    for c, r in zip(cs, (1.5, -1.0, -1.0)):
        c.outcomes[1.5] = (c.entry_dt, r, "")
    s = summarize(cs)
    assert s["n"] == 3
    assert abs(s["win_rate"] - 1 / 3) < 1e-9
    assert abs(s["avg_r"] - (-0.5 / 3)) < 1e-9
    assert abs(s["breakeven"] - 0.4) < 1e-9


def test_regime_gate_drops_trades_taken_below_the_50_day_average():
    from stocks.backtest import pullback_gate, regime_gate
    keep = regime_gate(pullback_gate())
    up, down, unknown = _cand(), _cand(), _cand()
    up.market_trend, down.market_trend = "UP", "DOWN"
    assert keep(up) and not keep(down) and keep(unknown)
