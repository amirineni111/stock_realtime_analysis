"""
Push alerts for newly armed signals.

An alert fires exactly when the scanner *arms* a signal for tracking — that is the
moment it becomes a trade the system stands behind, and the storage layer's
dedupe/cooldown (one open signal per ticker+direction, 45-minute re-arm cooldown)
already guarantees each setup alerts once rather than on every scan it persists.

Delivery is a single webhook URL from ``STOCKS_ALERT_WEBHOOK_URL``, shaped to what
the receiving service expects:

- ntfy (``ntfy.sh`` or a self-hosted server with ``/`` topic path): plain-text body
  with Title/Priority/Tags headers → free phone push with no account;
- Discord / Slack incoming webhooks: their JSON message shapes;
- anything else: a generic JSON POST with the signal fields.

Standard library only, a short timeout, and failures are returned rather than
raised — a flaky push endpoint must never break a scan.
"""
from __future__ import annotations

import json
import urllib.request
from datetime import datetime
from typing import List, Optional, Sequence, Tuple
from urllib.error import URLError
from urllib.parse import urlparse

from .market_hours import US_EASTERN
from .models import ArmedSignal
from .timeutil import parse_ts

_TIMEOUT_SECONDS = 5.0
# A burst beyond this is folded into one summary message rather than spamming.
_MAX_MESSAGES_PER_SCAN = 8

_ARROW = {"STRONG_BUY": "LONG", "BUY_CANDIDATE": "LONG",
          "STRONG_SHORT": "SHORT", "SHORT_CANDIDATE": "SHORT"}


def _px(value: Optional[float]) -> str:
    return "?" if value is None else f"{value:,.2f}"


def format_alert(sig: ArmedSignal) -> Tuple[str, str]:
    """(title, body). The title is ASCII so it survives an HTTP header (ntfy)."""
    side = _ARROW.get(sig.signal, sig.signal)
    title = f"{sig.ticker} {side} @ {_px(sig.entry)} ({sig.signal})"
    bar = parse_ts(sig.as_of)
    bar_txt = f" | bar {bar.astimezone(US_EASTERN):%H:%M} ET" if bar else ""
    rr = f" ({sig.rr_ratio:.1f}R)" if sig.rr_ratio else ""
    body = (
        f"Stop {_px(sig.stop)} | Target {_px(sig.target)}{rr} | score {sig.total_score:.0f}"
        f"{bar_txt}\n{sig.reason}"
    )
    return title, body


def _request(url: str, title: str, body: str, sig: Optional[ArmedSignal]) -> urllib.request.Request:
    host = urlparse(url).netloc.lower()
    if "discord.com" in host or "discordapp.com" in host:
        payload = json.dumps({"content": f"**{title}**\n{body}"}).encode()
        return urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    if "hooks.slack.com" in host:
        payload = json.dumps({"text": f"*{title}*\n{body}"}).encode()
        return urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    if "ntfy" in host:
        strong = sig is not None and sig.signal.startswith("STRONG")
        long_side = sig is not None and _ARROW.get(sig.signal) == "LONG"
        headers = {
            "Title": title.encode("ascii", "replace").decode(),
            "Priority": "high" if strong else "default",
        }
        if sig is not None:   # a plain test message gets no direction icon
            headers["Tags"] = "chart_with_upwards_trend" if long_side else "chart_with_downwards_trend"
        return urllib.request.Request(url, data=body.encode("utf-8"), headers=headers)
    payload = json.dumps({
        "title": title, "body": body,
        "signal": sig.model_dump() if sig is not None else None,
    }).encode()
    return urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})


def send(url: str, title: str, body: str, sig: Optional[ArmedSignal] = None) -> Optional[str]:
    """POST one alert. Returns None on success, else a short error string."""
    try:
        with urllib.request.urlopen(_request(url, title, body, sig), timeout=_TIMEOUT_SECONDS) as resp:
            if resp.status >= 300:
                return f"HTTP {resp.status}"
    except (URLError, OSError, ValueError) as exc:
        return str(getattr(exc, "reason", exc))
    return None


def notify(armed: Sequence[ArmedSignal], url: str) -> List[str]:
    """
    Push every newly armed signal to ``url``. Returns the list of delivery errors
    (empty = all delivered, or nothing to send / no URL configured).
    """
    if not url or not armed:
        return []
    errors: List[str] = []
    head, rest = list(armed[:_MAX_MESSAGES_PER_SCAN]), list(armed[_MAX_MESSAGES_PER_SCAN:])
    for sig in head:
        title, body = format_alert(sig)
        err = send(url, title, body, sig)
        if err:
            errors.append(f"{sig.ticker}: {err}")
    if rest:
        title = f"+{len(rest)} more signals"
        body = ", ".join(f"{s.ticker} {_ARROW.get(s.signal, s.signal)}" for s in rest)
        err = send(url, title, body)
        if err:
            errors.append(f"summary: {err}")
    return errors


def console_line(sig: ArmedSignal, now: Optional[datetime] = None) -> str:
    title, body = format_alert(sig)
    stamp = (now or datetime.now(US_EASTERN)).astimezone(US_EASTERN).strftime("%H:%M:%S")
    return f"[{stamp}] {title} | {body.splitlines()[0]}"
