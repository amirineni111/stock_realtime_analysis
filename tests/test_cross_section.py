"""Ranking backtest mechanics on synthetic panels: session handling, no look-ahead,
cost timing, and that a planted reversal is actually found."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from stocks.cross_section import (
    TO_CLOSE, Schedule, bar_in_session, build_panels, cost_panel, default_signals,
    forward_return, long_short, reversal, return_vol, summarize,
)
from stocks.market_hours import US_EASTERN

DAYS = ["2026-09-14", "2026-09-15", "2026-09-16", "2026-09-17", "2026-09-18",
        "2026-09-21", "2026-09-22", "2026-09-23"]


def _bars(closes_by_day, extra_premarket=False):
    """{day: [78 closes]} -> bar dicts for one ticker (UTC ISO like the client)."""
    rows = []
    for day, closes in closes_by_day.items():
        open_local = datetime.fromisoformat(f"{day}T09:30:00").replace(tzinfo=US_EASTERN)
        if extra_premarket:
            ts = (open_local - timedelta(minutes=30)).astimezone(timezone.utc).isoformat()
            rows.append({"timestamp": ts, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1})
        prev = closes[0]
        for k, c in enumerate(closes):
            ts = (open_local + timedelta(minutes=5 * k)).astimezone(timezone.utc).isoformat()
            rows.append({"timestamp": ts, "open": prev, "high": max(prev, c) + 0.01,
                         "low": min(prev, c) - 0.01, "close": c, "volume": 1000})
            prev = c
    return rows


def _panels(n_tickers=30, seed=0, reverting=True):
    """Random walks where (optionally) each 30m move partly reverses over the next 30m."""
    rng = np.random.default_rng(seed)
    bars = {}
    for i in range(n_tickers):
        by_day = {}
        for day in DAYS:
            px, closes, shocks = 100.0, [], []
            for k in range(78):
                shock = rng.normal(0, 0.001)
                if reverting and k >= 6:
                    shock -= 0.1 * sum(shocks[-6:])     # pay back some of the last 30m
                shocks.append(shock)
                px *= 1 + shock
                closes.append(px)
            by_day[day] = closes
        bars[f"T{i:02d}"] = _bars(by_day)
    return build_panels(bars), bars


def _flat_cost(close, value=0.0):
    return pd.DataFrame(value, index=close.index, columns=close.columns)


def test_build_panels_keeps_regular_session_only():
    panels = build_panels({"A": _bars({DAYS[0]: [100.0] * 78}, extra_premarket=True)})
    idx = panels["close"].index
    assert len(idx) == 78
    assert idx[0].hour == 9 and idx[0].minute == 30
    assert bar_in_session(panels["close"]).iloc[-1] == 77


def test_forward_return_never_crosses_the_session():
    panels, _ = _panels(n_tickers=2)
    fwd = forward_return(panels["close"], 6)
    k = bar_in_session(panels["close"])
    assert fwd[k >= 72].isna().all().all()     # last 6 bars of each day have no 30m future
    assert fwd[k < 72].notna().all().all()


def test_signals_do_not_look_ahead():
    panels, bars = _panels(n_tickers=5)
    before = default_signals(panels)
    # Rewrite every bar after a cut-off: signals up to the cut-off must not change.
    cut = panels["close"].index[300]
    changed = {f: p.copy() for f, p in panels.items()}
    for f in ("open", "high", "low", "close"):
        changed[f].loc[changed[f].index > cut] *= 1.5
    after = default_signals(changed)
    for name in before:
        a = before[name].loc[:cut]
        b = after[name].loc[:cut]
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-12)


def test_planted_reversal_is_found_and_absent_when_not_planted():
    panels, _ = _panels(reverting=True, seed=1)
    close = panels["close"]
    sig = reversal(close, 6, return_vol(close, lookback=156))
    book = long_short(sig, forward_return(close, 6), _flat_cost(close), Schedule(horizon=6))
    s = summarize(book)
    assert s["ic"] > 0.05 and s["gross_bps"] > 0

    panels, _ = _panels(reverting=False, seed=1)
    close = panels["close"]
    sig = reversal(close, 6, return_vol(close, lookback=156))
    s0 = summarize(long_short(sig, forward_return(close, 6), _flat_cost(close), Schedule(horizon=6)))
    assert abs(s0["ic"]) < s["ic"]


def test_costs_come_off_both_legs():
    panels, _ = _panels(n_tickers=30, seed=2)
    close = panels["close"]
    sig = reversal(close, 6, return_vol(close, lookback=156))
    fwd = forward_return(close, 6)
    free = long_short(sig, fwd, _flat_cost(close, 0.0), Schedule(horizon=6))
    paid = long_short(sig, fwd, _flat_cost(close, 0.0003), Schedule(horizon=6))
    assert np.allclose(free["gross"] - paid["net"], 0.0006)
    s = summarize(free)
    assert abs(s["breakeven_cost_bps"] - s["gross_bps"] / 2) < 1e-9


def test_rebalances_do_not_overlap_by_default():
    panels, _ = _panels(n_tickers=25, seed=3)
    close = panels["close"]
    sig = reversal(close, 6, return_vol(close, lookback=156))
    book = long_short(sig, forward_return(close, 12), _flat_cost(close), Schedule(horizon=12))
    gaps = book.groupby(book["ts"].dt.date)["ts"].diff().dropna()
    assert (gaps == pd.Timedelta(minutes=60)).all()
    daily = long_short(sig, forward_return(close, 0), _flat_cost(close), TO_CLOSE)
    assert daily["ts"].dt.date.is_unique
    assert (daily["ts"].dt.strftime("%H:%M") == "10:55").all()   # bar 17 starts 10:55, closes 11:00


def test_cost_panel_uses_only_prior_sessions():
    idx = pd.DatetimeIndex([
        datetime(2026, 9, 22, 10, 0, tzinfo=US_EASTERN), datetime(2026, 9, 23, 10, 0, tzinfo=US_EASTERN),
    ])
    daily = {"A": [
        {"timestamp": datetime(2026, 9, d, 4, 0, tzinfo=timezone.utc).isoformat(),
         "close": 100.0, "volume": vol}
        # Thin for 20 days, then one huge day on 9/22 itself.
        for d, vol in [(x, 1_000.0) for x in range(1, 22)] + [(22, 10_000_000.0)]
    ]}
    cost = cost_panel(daily, idx, ["A"])
    assert cost.iloc[0, 0] == 0.005        # 9/22 priced from thin prior days (50 bps tier)
    assert cost.iloc[1, 0] < 0.005         # 9/23 sees 9/22's volume
