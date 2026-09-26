"""
Cross-sectional backtest: rank the universe, go long the top and short the bottom.

Always run from the project root:

    python scripts/rank_backtest.py                       # NASDAQ-100, all signals x horizons
    python scripts/rank_backtest.py --universe watchlist  # the dashboard watchlist instead
    python scripts/rank_backtest.py --frac 0.1            # deciles instead of quintiles
    python scripts/rank_backtest.py --csv book.csv --signal range_pullback --horizon 12

How to read it (see stocks/cross_section.py for the mechanics):

- IC: rank correlation of signal vs. next-horizon return, averaged over rebalances.
  |IC| of 0.02-0.05 with |t| > 2 is a usable signal. Negative IC = the opposite
  direction (momentum instead of reversal) is what worked.
- net bps: average long-minus-short return per rebalance after estimated costs, in
  basis points of one side's capital. This is the number that pays.
- BEcost: round-trip cost per name (bps) at which net would be zero. Compare it
  with the estimated cost printed at the top: that gap is the whole question.
- hit / day hit: share of rebalances / days that made money. This is the honest
  version of "accuracy" for a portfolio.
- thirds: net bps in each third of the window. Keep only what holds in all three.

Every signal x horizon combination printed is a separate test. With ~16 tests one
will look good by luck, so require t > 2 and consistency across thirds before
believing any of them.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stocks.cross_section import (  # noqa: E402
    TO_CLOSE, Schedule, build_panels, cost_panel, default_signals, forward_return, long_short, summarize,
)
from stocks.history import dashboard_watchlist, fetch_history, polygon_history  # noqa: E402
from stocks.universe import NASDAQ_100  # noqa: E402

HORIZONS = (6, 12, 24, 0)   # 30m, 1h, 2h (in 5m bars), and 11:00 -> close


def _row(name: str, s: dict) -> str:
    if not s.get("n"):
        return f"  {name:28s} n=0"
    thirds = " ".join(f"{v:+6.1f}" for v in s["thirds_bps"])
    return (
        f"  {name:28s} n={s['n']:4d}  IC={s['ic']:+.3f} (t={s['ic_t']:+5.1f})  "
        f"gross={s['gross_bps']:+6.1f}  net={s['net_bps']:+6.1f}bps (t={s['net_t']:+5.1f})  "
        f"BEcost={s['breakeven_cost_bps']:+5.1f}  "
        f"hit={s['hit']:4.0%} dayhit={s['day_hit']:4.0%}  Sharpe={s['sharpe']:+5.1f}  "
        f"thirds [{thirds}]"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Long/short ranking backtest")
    ap.add_argument("--universe", choices=("nasdaq100", "watchlist"), default="nasdaq100")
    ap.add_argument("--tickers", nargs="*", help="explicit universe (overrides --universe)")
    ap.add_argument("--frac", type=float, default=0.2, help="fraction of names per leg (default 0.2)")
    ap.add_argument("--refresh", action="store_true", help="ignore today's download cache")
    ap.add_argument("--source", choices=("yahoo", "polygon"), default="yahoo",
                    help="yahoo = last 60 days; polygon = --years of cached Polygon history "
                         "(filled by scripts/download_history.py; never calls the API)")
    ap.add_argument("--years", type=float, default=2.0, help="history depth for --source polygon")
    ap.add_argument("--csv", help="write one signal/horizon's rebalances to this file")
    ap.add_argument("--signal", default="range_pullback")
    ap.add_argument("--horizon", type=int, default=12, help="bars for --csv (0 = to close)")
    args = ap.parse_args()

    if args.tickers:
        universe = [t.upper() for t in args.tickers]
    elif args.universe == "watchlist":
        universe = dashboard_watchlist()
    else:
        universe = list(NASDAQ_100)
    if len(universe) < 10:
        print("Need at least 10 tickers for a meaningful ranking.")
        return 1

    if args.source == "polygon":
        print(f"Loading {args.years:g}y of cached Polygon 5m bars for {len(universe)} tickers")
        data = polygon_history(universe, years=args.years, log=lambda m: None, cache_only=True)
    else:
        print(f"Fetching 60d of 5m bars for {len(universe)} tickers")
        data = fetch_history(universe, periods={"5m": "60d", "1d": "6mo"}, refresh=args.refresh)
    data["5m"] = {t: b for t, b in data["5m"].items() if b}
    panels = build_panels(data["5m"])
    close = panels["close"]
    missing = sorted(set(universe) - set(close.columns))
    print(f"  {close.shape[1]} tickers with data, {len(set(close.index.date))} sessions "
          f"({close.index[0]:%Y-%m-%d} to {close.index[-1]:%Y-%m-%d})"
          + (f"; no data for {', '.join(missing)}" if missing else ""))
    cost = cost_panel(data["1d"], close.index, close.columns)
    print(f"  median est. round-trip cost {cost.stack().median() * 1e4:.1f} bps per name, "
          f"so a long/short rebalance must earn >{cost.stack().median() * 2e4:.0f} bps gross")

    signals = default_signals(panels)
    fwd = {h: forward_return(close, h) for h in HORIZONS}

    print(f"\nLong top {args.frac:.0%} / short bottom {args.frac:.0%}, equal weight, "
          f"intraday entries from 10:00 ET, flat by the close, net of est. cost")
    for h in HORIZONS:
        sched = Schedule(horizon=h) if h else TO_CLOSE
        print(f"\n=== Hold {sched.label()} ===")
        for name, sig in signals.items():
            print(_row(name, summarize(long_short(sig, fwd[h], cost, sched, frac=args.frac))))

    window = f"{len(set(close.index.date))} sessions"
    print(f"\nCaveats: current-constituent universe (survivorship bias); {window} of history; "
          "costs are estimated, not observed; ~16 tests were run above.")

    if args.csv:
        if args.signal not in signals or args.horizon not in fwd:
            print(f"--signal must be one of {sorted(signals)}; --horizon one of {HORIZONS}")
            return 1
        book = long_short(signals[args.signal], fwd[args.horizon], cost,
                          Schedule(horizon=args.horizon) if args.horizon else TO_CLOSE,
                          frac=args.frac)
        book.to_csv(args.csv, index=False)
        print(f"Wrote {len(book)} rebalances to {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
