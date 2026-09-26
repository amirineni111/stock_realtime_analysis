"""
Historical replay of the live scanner.

Why this exists: the forward-test loop needs weeks of live scans to say anything
about a rule change, and by the time it answers the market regime has moved. The
replay walks every completed 5m bar of the last ~60 sessions through
``scanner.analyze_ticker`` — the exact function the live scan calls — and resolves
each armed trade against the bars that followed. A rule can be judged in minutes.

No look-ahead, by construction:

- the 5m window ends at the bar that just completed (entry = its close);
- an hourly bar is only visible once its hour has ended, a daily bar only after
  16:00 ET on its own date — the same ``drop_forming`` rule the live client uses;
- resolution only ever looks at bars strictly after the entry bar.

Rule variants are applied *after* the expensive indicator pass. Gates only ever
demote an actionable signal to WATCH_ONLY, so one ungated pass produces the
candidate stream and each variant filters it — then re-runs the arming rules
(dedupe/cooldown), because which trades are open depends on which were taken.

Known differences from the live forward-test (see README):

- trades are flat at the session close (an intraday alert is an intraday trade),
  whereas the live tracker holds up to 8 wall-clock hours, often overnight;
- no trained model is applied — this measures the rules alone;
- a stop gapped through is filled at the stop, as live.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .market_hours import US_EASTERN
from .scanner import OPEN_CHOP_MINUTES, analyze_ticker, spy_day_change
from .signals import _MAX_ENTRY_RANGE_POS, MIN_TARGET_PCT, breakeven_win_rate, market_trend
from .timeutil import parse_ts

ACTIONABLE = ("STRONG_BUY", "BUY_CANDIDATE", "STRONG_SHORT", "SHORT_CANDIDATE")

# Mirrors the live fetch windows in yf_client.INTERVAL_PERIOD so indicators are
# computed over the same amount of history as they are live.
_M5_SESSIONS = 5          # "5d" of 5m bars
_H1_BARS = 154            # "1mo" of 1h bars ≈ 22 sessions × 7
_D1_BARS = 126            # "6mo" of daily bars

REARM_COOLDOWN_MINUTES = 45   # storage.Storage.REARM_COOLDOWN_MINUTES
TARGET_RRS = (1.0, 1.5, 2.0)


@dataclass
class Candidate:
    """An actionable signal as the live scanner would have shown it."""
    ticker: str
    signal: str
    direction: int
    entry_ts: str
    entry_dt: datetime
    entry: float
    stop_dist: float
    cost_pct: float
    total_score: float
    extension_atr: Optional[float]
    range_pos: Optional[float]
    minutes_since_open: float
    features: Optional[dict]
    market_trend: Optional[str] = None      # SPY vs its 50-day average at entry
    # rr -> (exit_dt, net_R, exit_reason); filled by resolve()
    outcomes: Dict[float, Tuple[datetime, float, str]] = field(default_factory=dict)

    @property
    def stretch_atr(self) -> Optional[float]:
        """Extension from EMA20 measured *in the trade's direction* (+ = chasing)."""
        if self.extension_atr is None:
            return None
        return self.extension_atr * self.direction


def _eastern_date(dt: datetime):
    return dt.astimezone(US_EASTERN).date()


def _d1_end(ts: str) -> datetime:
    local = parse_ts(ts).astimezone(US_EASTERN)
    return local.replace(hour=16, minute=0, second=0, microsecond=0).astimezone(timezone.utc)


def _is_regular_bar(dt: datetime) -> bool:
    local = dt.astimezone(US_EASTERN)
    minutes = local.hour * 60 + local.minute
    return local.weekday() < 5 and 9 * 60 + 30 <= minutes < 16 * 60


def resolve(cand: Candidate, forward: Sequence[dict], rrs: Iterable[float] = TARGET_RRS) -> None:
    """
    Bracket-resolve ``cand`` against ``forward`` bars (strictly after entry, same
    session) for each target multiple in ``rrs``. Stop is checked before target
    within a bar, as in ``Storage.evaluate_tracked_signals``. Unresolved trades
    exit at the last forward bar's close (the session close). R is net of the
    estimated round-trip cost.
    """
    cost = cand.cost_pct / 100.0 * cand.entry
    d = cand.direction
    stop = cand.entry - d * cand.stop_dist
    for rr in rrs:
        target = cand.entry + d * rr * cand.stop_dist
        exit_px, reason, exit_dt = None, "CLOSE", cand.entry_dt
        for b in forward:
            hi, lo = b["high"], b["low"]
            if (d == 1 and lo <= stop) or (d == -1 and hi >= stop):
                exit_px, reason = stop, "STOP"
            elif (d == 1 and hi >= target) or (d == -1 and lo <= target):
                exit_px, reason = target, "TARGET"
            if exit_px is not None:
                exit_dt = parse_ts(b["timestamp"]) + timedelta(minutes=5)
                break
        if exit_px is None:
            if forward:
                exit_px = forward[-1]["close"]
                exit_dt = parse_ts(forward[-1]["timestamp"]) + timedelta(minutes=5)
            else:
                exit_px = cand.entry
        net = (exit_px - cand.entry) * d - cost
        cand.outcomes[rr] = (exit_dt, net / cand.stop_dist, reason)


def generate_candidates(
    ticker: str,
    m5: List[dict],
    h1: List[dict],
    d1: List[dict],
    spy_m5: List[dict],
    spy_d1: List[dict],
    min_avg_dollar_volume: float,
    start: Optional[datetime] = None,
    rrs: Iterable[float] = TARGET_RRS,
) -> List[Candidate]:
    """
    Replay one ticker bar by bar and return every actionable signal, resolved.

    The rule set replayed is the *ungated* one (``max_entry_range_pos=None``)
    so variants can be applied afterwards; see ``simulate``.
    """
    m5 = [b for b in m5 if _is_regular_bar(parse_ts(b["timestamp"]))]
    if not m5:
        return []
    m5_dt = [parse_ts(b["timestamp"]) for b in m5]
    m5_date = [_eastern_date(dt) for dt in m5_dt]
    h1_end = [parse_ts(b["timestamp"]) + timedelta(hours=1) for b in h1]
    d1_end = [_d1_end(b["timestamp"]) for b in d1]
    spy_ts = {b["timestamp"]: i for i, b in enumerate(spy_m5)}
    spy_d1_end = [_d1_end(b["timestamp"]) for b in spy_d1]

    # First index of each session, for the 5-session window and the forward slice.
    session_start: Dict[object, int] = {}
    session_end: Dict[object, int] = {}
    for i, day in enumerate(m5_date):
        session_start.setdefault(day, i)
        session_end[day] = i
    days = list(session_start)
    day_pos = {d: i for i, d in enumerate(days)}

    out: List[Candidate] = []
    for i, dt in enumerate(m5_dt):
        if start is not None and dt < start:
            continue
        day = m5_date[i]
        day_idx = day_pos[day]
        if day_idx < 1:
            continue  # need a previous session for prev-close / warm indicators
        now = dt + timedelta(minutes=5)
        window_start = session_start[days[max(0, day_idx - (_M5_SESSIONS - 1))]]
        bars = m5[window_start:i + 1]
        if len(bars) < 60:
            continue

        h1_vis = h1[max(0, bisect.bisect_right(h1_end, now) - _H1_BARS):bisect.bisect_right(h1_end, now)]
        d1_hi = bisect.bisect_right(d1_end, now)
        d1_vis = d1[max(0, d1_hi - _D1_BARS):d1_hi]

        spy_change = None
        spy_d1_vis = spy_d1[:bisect.bisect_right(spy_d1_end, now)]
        j = spy_ts.get(m5[i]["timestamp"])
        if j is not None:
            spy_change = spy_day_change(spy_m5[:j + 1], spy_d1_vis)
        trend = market_trend([b["close"] for b in spy_d1_vis])

        result = analyze_ticker(
            ticker, bars, h1_vis, d1_vis, spy_change, min_avg_dollar_volume,
            now=now, signal_params={"max_entry_range_pos": None, "market_trend": trend},
        )
        sc = result["scoring"]
        if sc.get("trade_signal") not in ACTIONABLE or not sc.get("stop_dollars"):
            continue
        entry = sc["suggested_entry"]
        if not entry or (sc["target_dollars"] / entry * 100) < MIN_TARGET_PCT:
            continue
        forward = m5[i + 1:session_end[day] + 1]
        if not forward:
            continue  # signal on the closing bar: nothing left to trade today
        mins_open = (now - now.astimezone(US_EASTERN).replace(
            hour=9, minute=30, second=0, microsecond=0)).total_seconds() / 60
        cand = Candidate(
            ticker=ticker,
            signal=sc["trade_signal"],
            direction=-1 if "SHORT" in sc["trade_signal"] else 1,
            entry_ts=m5[i]["timestamp"],
            entry_dt=now,
            entry=entry,
            stop_dist=sc["stop_dollars"],
            cost_pct=sc.get("cost_pct") or 0.0,
            total_score=sc.get("total_score") or 0.0,
            extension_atr=sc.get("extension_atr"),
            range_pos=sc.get("entry_range_pos"),
            minutes_since_open=mins_open,
            features=result["features"],
            market_trend=trend,
        )
        resolve(cand, forward, rrs)
        out.append(cand)
    return out


Filter = Callable[[Candidate], bool]


def regime_gate(keep: Filter) -> Filter:
    """``keep`` plus the optional market-regime gate (no trades while SPY < 50d avg)."""
    return lambda c: c.market_trend != "DOWN" and keep(c)


def pullback_gate(limit: Optional[float] = _MAX_ENTRY_RANGE_POS) -> Filter:
    """The live pullback-entry gate as a candidate filter (same rule as score_ticker)."""
    if limit is None:
        return lambda c: True
    return lambda c: c.range_pos is not None and c.range_pos <= limit


def simulate(
    candidates: Sequence[Candidate],
    keep: Filter,
    rr: float = 1.5,
    skip_opening_minutes: float = OPEN_CHOP_MINUTES,
) -> List[Candidate]:
    """
    Apply a rule variant (``keep``) and the live arming rules to a candidate
    stream: no first-hour entries, at most one open trade per ticker+direction,
    and a cooldown after each arm. Returns the trades that would have been taken,
    in time order.
    """
    taken: List[Candidate] = []
    open_until: Dict[Tuple[str, int], datetime] = {}
    last_armed: Dict[Tuple[str, int], datetime] = {}
    for c in sorted(candidates, key=lambda c: c.entry_dt):
        if c.minutes_since_open < skip_opening_minutes or not keep(c):
            continue
        key = (c.ticker, c.direction)
        if key in open_until and c.entry_dt < open_until[key]:
            continue
        if key in last_armed and c.entry_dt < last_armed[key] + timedelta(minutes=REARM_COOLDOWN_MINUTES):
            continue
        last_armed[key] = c.entry_dt
        open_until[key] = c.outcomes[rr][0]
        taken.append(c)
    return taken


def summarize(trades: Sequence[Candidate], rr: float = 1.5) -> dict:
    """Win rate, expectancy and a rough per-day consistency read for ``trades``."""
    rs = [t.outcomes[rr][1] for t in trades]
    n = len(rs)
    if not n:
        return {"n": 0}
    wins = sum(1 for r in rs if r > 0)
    by_day: Dict[object, float] = {}
    for t, r in zip(trades, rs):
        day = _eastern_date(t.entry_dt)
        by_day[day] = by_day.get(day, 0.0) + r
    days_up = sum(1 for v in by_day.values() if v > 0)
    mean = sum(rs) / n
    var = sum((r - mean) ** 2 for r in rs) / max(1, n - 1)
    return {
        "n": n,
        "win_rate": wins / n,
        "avg_r": mean,
        "total_r": sum(rs),
        # Standard error of the mean R — trades on the same day are correlated, so
        # treat this as a lower bound on the real uncertainty.
        "se_r": (var / n) ** 0.5,
        "days": len(by_day),
        "pct_days_up": days_up / len(by_day),
        "breakeven": breakeven_win_rate(rr),
    }
