"""Alert formatting, per-service request shapes, and the bar-close wake schedule."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from stocks.alerts import _request, format_alert, notify, send
from stocks.models import ArmedSignal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from run_alerts import next_wake  # noqa: E402

SIG = ArmedSignal(
    tracking_id=1, ticker="NVDA", signal="STRONG_BUY", entry=180.25, stop=178.0,
    target=183.62, rr_ratio=1.5, total_score=74.0, reason="Strong long setup",
    as_of="2026-09-24T18:35:00+00:00",
)


def test_format_alert_carries_the_trade_levels():
    title, body = format_alert(SIG)
    assert title == "NVDA LONG @ 180.25 (STRONG_BUY)"
    assert "Stop 178.00" in body and "Target 183.62 (1.5R)" in body
    assert "bar 14:35 ET" in body
    assert title.isascii()


def test_ntfy_gets_plain_text_with_headers():
    req = _request("https://ntfy.sh/my-topic", *format_alert(SIG), SIG)
    assert req.get_header("Title") == "NVDA LONG @ 180.25 (STRONG_BUY)"
    assert req.get_header("Priority") == "high"
    assert req.data.decode().startswith("Stop 178.00")


def test_discord_and_slack_get_their_json_shapes():
    d = _request("https://discord.com/api/webhooks/1/x", "T", "B", SIG)
    s = _request("https://hooks.slack.com/services/x", "T", "B", SIG)
    assert json.loads(d.data) == {"content": "**T**\nB"}
    assert json.loads(s.data) == {"text": "*T*\nB"}


def test_generic_url_gets_the_signal_fields():
    req = _request("https://example.com/hook", "T", "B", SIG)
    payload = json.loads(req.data)
    assert payload["signal"]["ticker"] == "NVDA"


def test_notify_is_a_no_op_without_a_url():
    assert notify([SIG], "") == []


def test_unreachable_endpoint_reports_instead_of_raising():
    assert send("http://127.0.0.1:9/nothing-listens-here", "T", "B") is not None


def test_next_wake_lands_just_after_the_next_bar_close():
    now = datetime(2026, 9, 24, 18, 37, 0, tzinfo=timezone.utc)
    assert next_wake(now, 75) == datetime(2026, 9, 24, 18, 41, 15, tzinfo=timezone.utc)
    # Inside the lag window of the bar that just closed: wake for that bar.
    early = datetime(2026, 9, 24, 18, 35, 30, tzinfo=timezone.utc)
    assert next_wake(early, 75) == datetime(2026, 9, 24, 18, 36, 15, tzinfo=timezone.utc)
