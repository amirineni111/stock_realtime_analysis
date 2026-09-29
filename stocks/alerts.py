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


def channel_name(url: str) -> Optional[str]:
    """Which service ``url`` points at, for the alert log. None when no URL is set."""
    if not url:
        return None
    host = urlparse(url).netloc.lower()
    for key, name in (("ntfy", "ntfy"), ("discord", "discord"), ("hooks.slack.com", "slack")):
        if key in host:
            return name
    return "webhook"


def push_each(armed: Sequence[ArmedSignal], url: str) -> List[Tuple[ArmedSignal, Optional[str]]]:
    """
    Push every newly armed signal to ``url``; returns (signal, error-or-None) per
    signal. Signals past the per-scan cap share the summary message's outcome.
    """
    if not url or not armed:
        return []
    results: List[Tuple[ArmedSignal, Optional[str]]] = []
    head, rest = list(armed[:_MAX_MESSAGES_PER_SCAN]), list(armed[_MAX_MESSAGES_PER_SCAN:])
    for sig in head:
        title, body = format_alert(sig)
        results.append((sig, send(url, title, body, sig)))
    if rest:
        title = f"+{len(rest)} more signals"
        body = ", ".join(f"{s.ticker} {_ARROW.get(s.signal, s.signal)}" for s in rest)
        err = send(url, title, body)
        results.extend((s, err) for s in rest)
    return results


def notify(armed: Sequence[ArmedSignal], url: str) -> List[str]:
    """
    Push every newly armed signal to ``url``. Returns the list of delivery errors
    (empty = all delivered, or nothing to send / no URL configured).
    """
    return [f"{sig.ticker}: {err}" for sig, err in push_each(armed, url) if err]


def _missed_signal(row: dict) -> ArmedSignal:
    """An armed-but-never-alerted tracking row, rebuilt as an alert."""
    armed_at = parse_ts(row.get("created_at"))
    late = f" | late alert, armed {armed_at.astimezone(US_EASTERN):%H:%M} ET" if armed_at else " | late alert"
    stop_d, target_d = row.get("stop_dollars"), row.get("target_dollars")
    score = row.get("total_score") or 0.0
    return ArmedSignal(
        tracking_id=row["id"],
        ticker=row["ticker"],
        signal=row["signal"],
        entry=row.get("entry_price"),
        stop=row.get("stop_price"),
        target=row.get("target_price"),
        rr_ratio=(target_d / stop_d) if stop_d and target_d else None,
        total_score=score,
        reason=f"{row['signal']} ({score:.0f}pts){late}",
        as_of=row.get("entry_ts") or "",
    )


def deliver(armed: Sequence[ArmedSignal], url: str, storage, source: str) -> List[str]:
    """
    Push ``armed`` (when ``url`` is set) and log every alert with its outcome to the
    ``stock_alerts`` table the dashboard's Alerts tab reads. Returns delivery errors.

    Also sweeps up recently armed signals that never got an alert (a scan cut off
    between arming and delivery). Each alert is claimed in the log before it is
    pushed, so two processes sweeping at once push it only once. Logging failures
    are swallowed — like a push failure, they must not break a scan.
    """
    pending = list(armed)
    try:
        seen = {s.tracking_id for s in pending}
        pending += [_missed_signal(r) for r in storage.load_unalerted_signals()
                    if r["id"] not in seen]
    except Exception:
        pass
    if not pending:
        return []
    channel = channel_name(url)
    claimed = []
    for sig in pending:
        try:
            if not storage.claim_alert(sig, source, channel):
                continue   # already alerted by this or another process
        except Exception:
            pass           # can't log it — still better to push than to drop it
        claimed.append(sig)
    outcomes = dict((sig.tracking_id, err) for sig, err in push_each(claimed, url))
    if channel is not None:
        for sig in claimed:
            try:
                storage.set_alert_delivery(sig.tracking_id, outcomes.get(sig.tracking_id))
            except Exception:
                pass
    return [f"{sig.ticker}: {outcomes[sig.tracking_id]}"
            for sig in claimed if outcomes.get(sig.tracking_id)]


def console_line(sig: ArmedSignal, now: Optional[datetime] = None) -> str:
    title, body = format_alert(sig)
    stamp = (now or datetime.now(US_EASTERN)).astimezone(US_EASTERN).strftime("%H:%M:%S")
    return f"[{stamp}] {title} | {body.splitlines()[0]}"
