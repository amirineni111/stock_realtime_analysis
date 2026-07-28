"""
Timestamp parsing and intraday clock helpers shared by the storage and feature layers.

The yfinance client emits uniform UTC ISO-8601 strings, but SQLite writes bare
``CURRENT_TIMESTAMP`` values ("2026-07-28 13:45:02") with no zone. Both land in the
same columns, so one parser has to handle both — a silent None here is expensive:
in the forex sibling it is what made trade durations unmeasurable for months.

Intraday time-of-day matters far more for stocks than for forex: the 09:30 open and
the 16:00 close are hard boundaries with completely different volatility regimes, so
the model gets *session progress* rather than a wall-clock hour.
"""
from __future__ import annotations

from datetime import datetime, time, timezone
from typing import Optional
from zoneinfo import ZoneInfo

US_EASTERN = ZoneInfo("America/New_York")

REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)
# 09:30 → 16:00 ET
SESSION_MINUTES = 390.0


def parse_ts(value: Optional[str]) -> Optional[datetime]:
    """
    Parse an ISO-8601 timestamp or a SQLite ``CURRENT_TIMESTAMP`` string into a
    timezone-aware UTC datetime. Returns None when the value cannot be parsed.

    Naive inputs are assumed UTC, which is correct for both sources here. Fractional
    seconds longer than 6 digits (which ``fromisoformat`` rejects) are truncated
    rather than causing a parse failure.
    """
    if not value:
        return None
    text = str(value).strip()
    try:
        if "." in text:
            head, frac = text.split(".", 1)
            # Leading digits are the fraction; whatever follows is the zone suffix.
            i = 0
            while i < len(frac) and frac[i].isdigit():
                i += 1
            digits = frac[:i][:6].ljust(6, "0")
            rest = frac[i:]
            tz = "+00:00" if rest in ("", "Z", "z") else rest
            text = f"{head}.{digits}{tz}"
        elif text.endswith(("Z", "z")):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
    except (ValueError, TypeError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def minutes_since_open_at(value: Optional[str]) -> Optional[float]:
    """
    Minutes from that day's 09:30 ET open to ``value``. Negative before the open
    (pre-market), above 390 after the close. None if the timestamp is unparseable.

    Keyed to the timestamp's own date, not to "now" — a feature has to describe the
    bar it was built from, or every backfill and retrain silently shifts.
    """
    dt = parse_ts(value)
    if dt is None:
        return None
    local = dt.astimezone(US_EASTERN)
    open_local = local.replace(hour=9, minute=30, second=0, microsecond=0)
    return (local - open_local).total_seconds() / 60.0


def session_progress(value: Optional[str]) -> Optional[float]:
    """
    Position within the regular session as 0.0 (open) → 1.0 (close), clipped outside.

    Clipping rather than extrapolating keeps pre-market and after-hours bars at the
    endpoints instead of handing the model large out-of-range values it has almost
    no examples of.
    """
    mins = minutes_since_open_at(value)
    if mins is None:
        return None
    return max(0.0, min(1.0, mins / SESSION_MINUTES))


def is_regular_at(value: Optional[str]) -> Optional[bool]:
    """True when the timestamp falls inside 09:30–16:00 ET on a weekday."""
    dt = parse_ts(value)
    if dt is None:
        return None
    local = dt.astimezone(US_EASTERN)
    if local.weekday() >= 5:
        return False
    return REGULAR_OPEN <= local.time() <= REGULAR_CLOSE
