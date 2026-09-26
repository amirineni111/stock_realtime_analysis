"""
Replay the live scanner over the last ~60 sessions and compare rule variants.

Always run from the project root:

    python scripts/backtest.py                      # watchlist from the dashboard prefs
    python scripts/backtest.py --tickers AAPL NVDA  # explicit tickers
    python scripts/backtest.py --refresh            # re-download instead of using today's cache
    python scripts/backtest.py --csv trades.csv     # dump the chosen variant's trades

Yahoo serves at most 60 days of 5m history, which is what bounds the window.

How to read the output: win rate on its own is not the goal. A 1.5R bracket
breaks even at a 40% win rate and a 1.0R bracket at 50%, so moving the target
trades win rate for payoff without changing the edge. Judge a variant on avg R
(expectancy per trade, net of estimated cost) and on whether it holds in BOTH
halves of the window — a filter that only works in one half is noise.
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stocks.backtest import (  # noqa: E402
    TARGET_RRS, generate_candidates, pullback_gate, regime_gate, simulate, summarize,
)
from stocks.market_hours import US_EASTERN  # noqa: E402
from stocks.history import (  # noqa: E402
    WARMUP_DAYS, dashboard_watchlist, fetch_history, polygon_history,
)
from stocks.tickers import BENCHMARK  # noqa: E402

# Name -> keep(candidate). "baseline" is the rule set before the pullback-entry
# gate; "live" is what the scanner runs now. The others are comparisons: the
# gate at other thresholds, and an EMA20-extension cap (the first idea tried —
# it trims losses but never turned positive).
VARIANTS = {
    "baseline (no gate)": pullback_gate(None),
    "live: range pos <=-0.5": pullback_gate(),
    "range pos <=0": pullback_gate(0.0),
    "range pos <=-0.75": pullback_gate(-0.75),
    "EMA20 stretch <=1 ATR": lambda c: c.stretch_atr is None or c.stretch_atr <= 1.0,
    # Optional STOCKS_MARKET_REGIME_GATE: no trades while SPY is below its 50d average.
    "baseline + SPY>50d": regime_gate(pullback_gate(None)),
    "live + SPY>50d": regime_gate(pullback_gate()),
}


def _replay_one(args):
    ticker, bars, min_adv, start = args
    return generate_candidates(
        ticker, bars["5m"], bars["1h"], bars["1d"], bars["spy_5m"], bars["spy_1d"],
        min_adv, start=start,
    )


def _slice(data: dict, ticker: str) -> dict:
    """Only what one worker needs — pickling the whole universe to every process
    is slow at two years of history."""
    return {
        "5m": data["5m"].get(ticker) or [],
        "1h": data["1h"].get(ticker) or [],
        "1d": data["1d"].get(ticker) or [],
        "spy_5m": data["5m"].get(BENCHMARK) or [],
        "spy_1d": data["1d"].get(BENCHMARK) or [],
    }


def _row(name: str, s: dict) -> str:
    if not s.get("n"):
        return f"  {name:22s}  n=   0"
    return (
        f"  {name:22s}  n={s['n']:4d}  win={s['win_rate']:5.1%} (BE {s['breakeven']:.0%})  "
        f"avgR={s['avg_r']:+.3f} +/-{s['se_r']:.3f}  totR={s['total_r']:+7.1f}  "
        f"days up={s['pct_days_up']:4.0%}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay the scanner over recent history")
    ap.add_argument("--tickers", nargs="*", help="defaults to the dashboard watchlist")
    ap.add_argument("--min-dollar-volume-m", type=float, default=5.0,
                    help="liquidity gate in $M/day, as in the sidebar (default 5)")
    ap.add_argument("--refresh", action="store_true", help="ignore today's download cache")
    ap.add_argument("--source", choices=("yahoo", "polygon"), default="yahoo",
                    help="yahoo = last 60 days; polygon = --years of cached Polygon history "
                         "(filled by scripts/download_history.py; never calls the API)")
    ap.add_argument("--years", type=float, default=2.0, help="history depth for --source polygon")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--csv", help="write the trades of --variant at 1.5R to this file")
    ap.add_argument("--variant", default="live: range pos <=-0.5", choices=list(VARIANTS))
    args = ap.parse_args()

    tickers = [t.upper() for t in (args.tickers or dashboard_watchlist())]
    if not tickers:
        print("No tickers: pass --tickers or set a watchlist in the dashboard.")
        return 1
    fetch_list = sorted(set(tickers) | {BENCHMARK})

    print(f"Fetching history for {len(tickers)} tickers (+{BENCHMARK})")
    start = None
    if args.source == "polygon":
        data = polygon_history(fetch_list, years=args.years, log=lambda m: None, cache_only=True)
        missing = [t for t in fetch_list if not data["5m"].get(t)]
        if missing:
            print(f"  not in the Polygon cache yet (run scripts/download_history.py): {', '.join(missing)}")
        tickers = [t for t in tickers if data["5m"].get(t)]
        # Hourly/daily bars are derived from the same 5m history, so the first weeks
        # have too little of them for trend and S/R: treat those as warm-up.
        first = min((b[0]["timestamp"] for b in data["5m"].values() if b), default=None)
        if first:
            start = datetime.fromisoformat(first) + timedelta(days=WARMUP_DAYS)
    else:
        data = fetch_history(fetch_list, refresh=args.refresh)

    print(f"Replaying with {args.workers} workers (a few minutes for ~35 tickers)...", flush=True)
    min_adv = args.min_dollar_volume_m * 1_000_000
    candidates = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for t, cands in zip(tickers, pool.map(_replay_one, [(t, _slice(data, t), min_adv, start) for t in tickers])):
            candidates.extend(cands)
    if not candidates:
        print("No actionable signals in the window.")
        return 0

    first = min(c.entry_dt for c in candidates)
    last = max(c.entry_dt for c in candidates)
    mid = first + (last - first) / 2
    print(f"\n{len(candidates)} raw actionable signals, {first:%Y-%m-%d} to {last:%Y-%m-%d}"
          f"  (halves split at {mid:%Y-%m-%d})")

    for rr in TARGET_RRS:
        print(f"\n=== Target {rr:.1f}R (stop unchanged; flat at session close) ===")
        for name, keep in VARIANTS.items():
            trades = simulate(candidates, keep, rr=rr)
            print(_row(name, summarize(trades, rr)))
            h1 = [t for t in trades if t.entry_dt < mid]
            h2 = [t for t in trades if t.entry_dt >= mid]
            s1, s2 = summarize(h1, rr), summarize(h2, rr)
            if s1.get("n") and s2.get("n"):
                print(f"  {'':22s}    1st half avgR={s1['avg_r']:+.3f} (n={s1['n']})"
                      f"   2nd half avgR={s2['avg_r']:+.3f} (n={s2['n']})")

    chosen = simulate(candidates, VARIANTS[args.variant], rr=1.5)
    print(f"\n=== '{args.variant}' at 1.5R, by signal / entry hour (ET) ===")
    groups: dict = {}
    for t in chosen:
        groups.setdefault(("signal", t.signal), []).append(t)
        groups.setdefault(("hour", f"{t.entry_dt.astimezone(US_EASTERN):%H}:00"), []).append(t)
    for key in sorted(groups):
        print(_row(f"{key[0]} {key[1]}", summarize(groups[key], 1.5)))

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["ticker", "signal", "direction", "entry_ts", "entry", "stop_dist",
                        "stretch_atr", "total_score"] + [f"R_{rr}" for rr in TARGET_RRS])
            for t in chosen:
                w.writerow([t.ticker, t.signal, t.direction, t.entry_ts, t.entry, t.stop_dist,
                            t.stretch_atr, t.total_score] + [round(t.outcomes[rr][1], 3) for rr in TARGET_RRS])
        print(f"\nWrote {len(chosen)} trades to {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
