"""
Download (or top up) Polygon/Massive 5m history into data/polygon_cache/.

Always run from the project root:

    python scripts/download_history.py                 # watchlist + SPY, then NASDAQ-100
    python scripts/download_history.py --only-watchlist
    python scripts/download_history.py --years 1

On the free Basic plan (5 calls/minute) two years is ~9 calls per ticker, so the
first full run takes about 3.5 hours for ~115 tickers; the watchlist comes first
(~1 hour) so scripts/backtest.py --source polygon is usable early. Safe to stop
and re-run: finished tickers are cached, and later runs only fetch new days.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stocks.history import dashboard_watchlist, polygon_history  # noqa: E402
from stocks.tickers import BENCHMARK  # noqa: E402
from stocks.universe import NASDAQ_100  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Download Polygon 5m history into the cache")
    ap.add_argument("--years", type=float, default=2.0, help="history depth (Basic plan max: 2)")
    ap.add_argument("--only-watchlist", action="store_true")
    ap.add_argument("--refresh", action="store_true", help="re-download instead of topping up")
    args = ap.parse_args()

    first = [BENCHMARK] + [t for t in dashboard_watchlist() if t != BENCHMARK]
    rest = [] if args.only_watchlist else [t for t in NASDAQ_100 if t not in first]
    tickers = first + rest
    print(f"Downloading {args.years:g}y of 5m bars for {len(tickers)} tickers "
          f"(watchlist + SPY first: {len(first)})", flush=True)
    t0 = time.time()
    data = polygon_history(tickers, years=args.years, refresh=args.refresh,
                           log=lambda m: print(m, flush=True))
    empty = [t for t, bars in data["5m"].items() if not bars]
    print(f"Done in {(time.time() - t0) / 60:.0f} min."
          + (f" No data for: {', '.join(empty)}" if empty else ""), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
