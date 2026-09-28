"""
Headless real-time scanner: scan on every 5m bar close and push new signals.

The dashboard only scans while a browser tab is open and its auto-refresh timer
fires, and that timer is not aligned to bar closes — a signal can sit unseen for
most of a refresh interval. This runner wakes a fixed lag after each 5-minute
boundary (the moment a new completed bar exists), runs the same ``run_scan``, and
pushes whatever it newly armed. Alerts arrive ~1–2 minutes after the bar closes,
which is the floor Yahoo's own publishing delay allows.

Always run from the project root:

    python scripts/run_alerts.py                 # watchlist + liquidity from dashboard prefs
    python scripts/run_alerts.py --once          # one scan now, then exit (a smoke test)
    python scripts/run_alerts.py --test-push     # send a test alert to the webhook and exit
    python scripts/run_alerts.py --sample-alert  # send a sample signal in the real alert format

Safe to run alongside the dashboard: both write to the same database, and the
arming dedupe means a setup is only ever armed — and alerted — once.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stocks.alerts import channel_name, console_line, deliver, notify, send  # noqa: E402
from stocks.config import get_settings  # noqa: E402
from stocks.market_hours import US_EASTERN, current_market_phase  # noqa: E402
from stocks.models import ArmedSignal, ScanRequest  # noqa: E402
from stocks.scanner import run_scan  # noqa: E402
from stocks.storage import Storage  # noqa: E402
from stocks.tickers import parse_watchlist  # noqa: E402

PREFS_PATH = Path("data/app_preferences.json")
BAR_MINUTES = 5


def _prefs() -> dict:
    try:
        return json.loads(PREFS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def next_wake(now: datetime, lag_seconds: float) -> datetime:
    """The next 5-minute boundary plus ``lag_seconds`` strictly after ``now``."""
    base = now.replace(second=0, microsecond=0)
    base -= timedelta(minutes=base.minute % BAR_MINUTES)
    wake = base + timedelta(seconds=lag_seconds)
    while wake <= now:
        wake += timedelta(minutes=BAR_MINUTES)
    return wake


def _scan_once(settings, storage, request, url: str, quiet: bool) -> None:
    summary = run_scan(settings, storage, request)
    stamp = datetime.now(US_EASTERN).strftime("%H:%M:%S")
    if not quiet or summary.armed:
        print(f"[{stamp}] scanned {summary.tickers_scanned}, "
              f"{summary.signals_found} actionable, {len(summary.armed)} new, "
              f"{summary.errors} errors", flush=True)
    for sig in summary.armed:
        print("  " + console_line(sig), flush=True)
    for err in deliver(summary.armed, url, storage, "runner"):
        print(f"  push failed - {err}", flush=True)


def main() -> int:
    prefs = _prefs()
    ap = argparse.ArgumentParser(description="Scan on each 5m bar close and push new signals")
    ap.add_argument("--tickers", nargs="*", help="defaults to the dashboard watchlist")
    ap.add_argument("--min-dollar-volume-m", type=float,
                    default=float(prefs.get("min_dollar_volume_m", 5.0)))
    ap.add_argument("--lag", type=float, default=75.0,
                    help="seconds after each bar boundary to scan (Yahoo publishes 5m bars "
                         "with ~1 min delay; default 75)")
    ap.add_argument("--offhours", action="store_true", help="also scan outside 09:30-16:00 ET")
    ap.add_argument("--once", action="store_true", help="run one scan now and exit")
    ap.add_argument("--test-push", action="store_true", help="send a test alert and exit")
    ap.add_argument("--sample-alert", action="store_true",
                    help="send a made-up signal through the real alert path (format check) and exit")
    ap.add_argument("--quiet", action="store_true", help="only print scans that armed something")
    args = ap.parse_args()

    settings = get_settings()
    # .env wins; otherwise use the URL entered in the dashboard sidebar.
    url = settings.alert_webhook_url or (prefs.get("alert_webhook") or "").strip()
    if args.sample_alert:
        if not url:
            print("STOCKS_ALERT_WEBHOOK_URL is not set (see .env.example).")
            return 1
        sample = ArmedSignal(
            tracking_id=0, ticker="SAMPLE", signal="STRONG_BUY", entry=100.00, stop=98.50,
            target=102.25, rr_ratio=1.5, total_score=72.0,
            reason="SAMPLE ALERT - not a real signal. Real alerts look exactly like this.",
            as_of=datetime.now(timezone.utc).isoformat(),
        )
        errors = notify([sample], url)
        print("sent" if not errors else f"failed: {errors}")
        return 0 if not errors else 1

    if args.test_push:
        if not url:
            print("STOCKS_ALERT_WEBHOOK_URL is not set (see .env.example).")
            return 1
        err = send(url, "Stock scanner test alert",
                   "If you can read this, push alerts are working.")
        print("sent" if err is None else f"failed: {err}")
        return 0 if err is None else 1

    tickers = [t.upper() for t in (args.tickers or parse_watchlist(prefs.get("watchlist_raw", "")))]
    if not tickers:
        print("No tickers: pass --tickers or set a watchlist in the dashboard.")
        return 1

    storage = Storage(settings.db_path)
    request = ScanRequest(tickers=tickers, min_avg_dollar_volume=args.min_dollar_volume_m * 1_000_000)
    push = f"push -> {channel_name(url)}" if url else "console only (no STOCKS_ALERT_WEBHOOK_URL)"
    print(f"Watching {len(tickers)} tickers; {push}. Ctrl+C to stop.", flush=True)

    if args.once:
        _scan_once(settings, storage, request, url, quiet=False)
        return 0

    try:
        while True:
            now = datetime.now(timezone.utc)
            wake = next_wake(now, args.lag)
            time.sleep(max(0.0, (wake - now).total_seconds()))
            if not args.offhours and current_market_phase() != "REGULAR":
                continue
            try:
                _scan_once(settings, storage, request, url, args.quiet)
            except Exception as exc:  # keep the loop alive through network blips
                print(f"[{datetime.now(US_EASTERN):%H:%M:%S}] scan failed: {exc}", flush=True)
    except KeyboardInterrupt:
        print("stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
