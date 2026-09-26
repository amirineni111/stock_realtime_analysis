"""
Cross-sectional (ranking) research: long the best-ranked stocks, short the worst.

This is the structural difference between the per-ticker scanner and how
systematic funds trade. The scanner asks "will AAPL go up?", which is mostly a
question about the market, because intraday most large caps move with it. A
ranking asks "will the top of today's list beat the bottom?". Being long and short
in equal dollars cancels the market move, and what remains is the stock-specific
part a signal can actually predict. Ranking a hundred names at once also gives
many small, partly independent bets, instead of a few correlated ones.

Mechanics (all vectorised over a time × ticker panel of regular-session 5m bars):

- a *signal* is a panel where higher = more attractive to own, computed from
  data up to and including each bar's close, and never from anything later;
- at each rebalance bar the universe is ranked; the top ``frac`` is bought and the
  bottom ``frac`` sold short, equal-weighted, entered at that bar's close;
- the book is held ``horizon`` bars (or to the session close) and never overnight;
- each leg pays its names' estimated round-trip cost (``signals.estimate_cost_pct``
  tiers, from trailing 20-day dollar volume known *before* the day started).

Metrics reported per rebalance: the long-minus-short return (gross and net), and
the information coefficient (IC), the rank correlation between the signal and the
next-horizon return. IC is the number quant desks track. An IC of 0.02–0.05 that
holds up is a good signal; "accuracy" in the 60–70% sense does not exist at this
horizon.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from .market_hours import US_EASTERN
from .signals import estimate_cost_pct

BARS_PER_SESSION = 78
_FIELDS = ("open", "high", "low", "close", "volume")


# ── Panels ───────────────────────────────────────────────────────────────────

def build_panels(bars: Mapping[str, List[dict]]) -> Dict[str, pd.DataFrame]:
    """
    {field: DataFrame(index = bar start in US/Eastern, columns = tickers)} over
    regular-session bars only (09:30–15:55 starts). Tickers with no bars are dropped.
    """
    series: Dict[str, Dict[str, pd.Series]] = {f: {} for f in _FIELDS}
    for ticker, rows in bars.items():
        if not rows:
            continue
        idx = pd.to_datetime([b["timestamp"] for b in rows], utc=True).tz_convert(US_EASTERN)
        for f in _FIELDS:
            series[f][ticker] = pd.Series([b[f] for b in rows], index=idx, dtype=float)
    panels = {f: pd.DataFrame(cols).sort_index() for f, cols in series.items()}
    idx = panels["close"].index
    minutes = idx.hour * 60 + idx.minute
    regular = (idx.dayofweek < 5) & (minutes >= 9 * 60 + 30) & (minutes < 16 * 60)
    return {f: p.loc[regular] for f, p in panels.items()}


def _session(panel: pd.DataFrame) -> np.ndarray:
    return np.asarray(panel.index.date)


def _within_session_shift(panel: pd.DataFrame, n: int) -> pd.DataFrame:
    """Shift by ``n`` bars without crossing a session boundary (NaN instead)."""
    return panel.groupby(_session(panel)).shift(n)


def bar_in_session(panel: pd.DataFrame) -> pd.Series:
    """0 for the 09:30 bar, 1 for 09:35, … (the bar closes at 09:35 + 5·k)."""
    return pd.Series(panel.groupby(_session(panel)).cumcount().values, index=panel.index)


def return_vol(close: pd.DataFrame, lookback: int = 5 * BARS_PER_SESSION) -> pd.DataFrame:
    """Trailing std of within-session 5m returns: the per-name unit for scaling moves."""
    r1 = close / _within_session_shift(close, 1) - 1.0
    return r1.rolling(lookback, min_periods=lookback // 2).std()


# ── Signals (higher = more attractive long) ──────────────────────────────────

def reversal(close: pd.DataFrame, n: int, vol: pd.DataFrame) -> pd.DataFrame:
    """Minus the last ``n`` bars' return, in units of the name's own volatility."""
    move = close / _within_session_shift(close, n) - 1.0
    return -move / (vol * math.sqrt(n))


def day_reversal(panels: Mapping[str, pd.DataFrame], vol: pd.DataFrame) -> pd.DataFrame:
    """Minus the return since today's open, vol-scaled by the bars elapsed."""
    close = panels["close"]
    day_open = panels["open"].groupby(_session(close)).transform("first")
    k = bar_in_session(close).to_numpy()[:, None] + 1
    return -(close / day_open - 1.0) / (vol * np.sqrt(k))


def range_pullback(panels: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Minus where the close sits in today's range so far (+0.5 at the low, −0.5 at
    the high). The cross-sectional version of the scanner's pullback-entry gate.
    """
    close = panels["close"]
    sess = _session(close)
    hi = panels["high"].groupby(sess).cummax()
    lo = panels["low"].groupby(sess).cummin()
    width = (hi - lo).where(hi > lo)
    return -((close - lo) / width - 0.5)


def default_signals(panels: Mapping[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """The published short-horizon families this repo tests. A negative IC means
    the *opposite* (momentum) is what works for that family."""
    vol = return_vol(panels["close"])
    return {
        "reversal_30m": reversal(panels["close"], 6, vol),
        "reversal_60m": reversal(panels["close"], 12, vol),
        "reversal_day": day_reversal(panels, vol),
        "range_pullback": range_pullback(panels),
    }


# ── Forward returns and costs ────────────────────────────────────────────────

def forward_return(close: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Return from this bar's close to ``horizon`` bars later, same session only.
    ``horizon=0`` means to the session's last close."""
    if horizon == 0:
        last = close.groupby(_session(close)).transform("last")
        return last / close - 1.0
    return _within_session_shift(close, -horizon) / close - 1.0


def cost_panel(daily: Mapping[str, List[dict]], index: pd.DatetimeIndex, columns) -> pd.DataFrame:
    """
    Round-trip cost (as a fraction of price) per name per bar, from the 20-day
    average dollar volume of the sessions *before* each bar's date: no same-day
    volume leaks into the cost estimate.
    """
    dates = sorted(set(index.date))
    out = pd.DataFrame(index=dates, columns=list(columns), dtype=float)
    for t in columns:
        rows = daily.get(t) or []
        if not rows:
            continue
        d = pd.Series(
            [b["close"] * b["volume"] for b in rows],
            index=pd.to_datetime([b["timestamp"] for b in rows], utc=True).tz_convert(US_EASTERN).date,
        )
        d = d[~d.index.duplicated(keep="last")].sort_index()
        adv = d.rolling(20, min_periods=5).mean()
        # Last value from a session strictly before each date: known at the open.
        pos = np.searchsorted(np.array(adv.index, dtype=object), np.array(dates, dtype=object), side="left") - 1
        for day, j in zip(dates, pos):
            value = adv.iloc[j] if j >= 0 else None
            out.at[day, t] = estimate_cost_pct(None if value is None or pd.isna(value) else value) / 100.0
    return out.reindex(index.date).set_axis(index)


# ── Long/short simulation ────────────────────────────────────────────────────

@dataclass(frozen=True)
class Schedule:
    horizon: int              # bars held; 0 = to the session close
    first_bar: int = 5        # 09:55 bar → entry at its 10:00 close (skips the open auction chop)
    step: Optional[int] = None  # bars between rebalances; defaults to horizon (no overlap)

    def label(self) -> str:
        if self.horizon == 0:
            return f"{_bar_close_label(self.first_bar)} -> close"
        return f"{self.horizon * 5}m"


# The hold-to-close book enters once a day. 11:00 ET (bar 17 closes then) leaves
# every signal a full hour of same-session lookback to rank on.
TO_CLOSE = Schedule(horizon=0, first_bar=17)


def _bar_close_label(k: int) -> str:
    minutes = 9 * 60 + 35 + 5 * k
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def long_short(
    signal: pd.DataFrame,
    fwd: pd.DataFrame,
    cost: pd.DataFrame,
    schedule: Schedule,
    frac: float = 0.2,
    min_names: int = 20,
) -> pd.DataFrame:
    """
    One row per rebalance: gross and net long-minus-short return, IC, and legs.
    Returns are per unit of capital on each side (a $1 long / $1 short book).
    """
    k = bar_in_session(signal)
    if schedule.horizon == 0:
        rebalance = k == schedule.first_bar            # one book per day
    else:
        step = schedule.step or schedule.horizon
        rebalance = (k >= schedule.first_bar) & ((k - schedule.first_bar) % step == 0)
    rows = []
    for ts in signal.index[rebalance.to_numpy()]:
        s, f, c = signal.loc[ts], fwd.loc[ts], cost.loc[ts]
        ok = s.notna() & f.notna() & c.notna() & np.isfinite(s)
        if ok.sum() < min_names:
            continue
        s, f, c = s[ok], f[ok], c[ok]
        n_leg = max(1, int(len(s) * frac))
        order = s.sort_values(kind="mergesort")
        short, long_ = order.index[:n_leg], order.index[-n_leg:]
        gross = f[long_].mean() - f[short].mean()
        rows.append({
            "ts": ts,
            "gross": gross,
            "net": gross - (c[long_].mean() + c[short].mean()),
            "ic": s.rank().corr(f.rank()),
            "names": int(ok.sum()),
            "long": ",".join(long_),
            "short": ",".join(short),
        })
    return pd.DataFrame(rows)


def summarize(book: pd.DataFrame) -> dict:
    """Headline stats for a long_short() result. t-stats assume independent
    rebalances, which non-overlapping schedules roughly give."""
    if book.empty:
        return {"n": 0}
    n = len(book)
    daily = book.groupby(book["ts"].dt.date)["net"].sum()
    ic_sd = book["ic"].std(ddof=1) if n > 1 else float("nan")
    net_sd = book["net"].std(ddof=1) if n > 1 else float("nan")
    third = np.array_split(np.arange(n), 3)
    return {
        "n": n,
        "days": len(daily),
        "ic": book["ic"].mean(),
        "ic_t": book["ic"].mean() / ic_sd * math.sqrt(n) if ic_sd else float("nan"),
        "gross_bps": book["gross"].mean() * 1e4,
        "net_bps": book["net"].mean() * 1e4,
        "net_t": book["net"].mean() / net_sd * math.sqrt(n) if net_sd else float("nan"),
        # Round-trip cost per name (bps) at which net would be zero: each rebalance
        # pays one round trip on the long leg and one on the short leg.
        "breakeven_cost_bps": book["gross"].mean() * 1e4 / 2,
        "hit": (book["net"] > 0).mean(),
        "day_hit": (daily > 0).mean(),
        "sharpe": daily.mean() / daily.std(ddof=1) * math.sqrt(252) if len(daily) > 1 and daily.std(ddof=1) else float("nan"),
        "thirds_bps": [book["net"].iloc[ix].mean() * 1e4 if len(ix) else float("nan") for ix in third],
    }

