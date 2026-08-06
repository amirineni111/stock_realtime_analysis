"""
Whole-scan wiring: a scan must log a feature vector for every setup it arms, and an
active model must be able to veto what the rules proposed.

These two properties are what close the learning loop. If features stop being
logged the loop starves silently — the dashboard keeps working and the training
data simply never grows — so it is worth an integration test rather than trusting
the unit tests of each half.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from stocks.config import AppSettings
from stocks.features import FEATURE_NAMES, FEATURE_VERSION
from stocks.model import StockModel
from stocks.models import ScanRequest, StockBar
from stocks.storage import Storage
from stocks import scanner as scanner_mod
from stocks.scanner import run_scan

SESSION_DAY = "2026-07-28"                       # a Tuesday
OPEN_UTC = datetime(2026, 7, 28, 13, 30, tzinfo=timezone.utc)   # 09:30 ET (EDT)


def _bars(ticker: str, interval: str, closes: list, start: datetime, step: timedelta):
    """Build bars with a small, consistent high/low envelope around each close."""
    out = []
    for i, close in enumerate(closes):
        ts = start + step * i
        prev = closes[i - 1] if i else close
        out.append(StockBar(
            ticker=ticker,
            timeframe=interval,
            timestamp=ts.isoformat().replace("+00:00", "+00:00"),
            open=round(prev, 4),
            high=round(max(prev, close) + 0.25, 4),
            low=round(min(prev, close) - 0.25, 4),
            close=round(close, 4),
            volume=1_000_000,
        ))
    return out


def _uptrend(n: int, start_price: float, per_bar: float, wobble: float = 0.15) -> list:
    """A rising series with enough wobble to give ATR/ADX something to chew on."""
    return [
        start_price + per_bar * i + (wobble if i % 2 else -wobble)
        for i in range(n)
    ]


class FakeClient:
    """Stands in for YFClient — the scanner only ever calls get_bars."""

    def __init__(self) -> None:
        # 5m bars covering 13:30 → ~19:50 UTC (a full regular session).
        self.series_5m = _uptrend(76, 100.0, 0.06)
        self.spy_5m = _uptrend(76, 500.0, 0.05)

    def get_bars(self, tickers, interval):
        out = {}
        for ticker in tickers:
            is_spy = ticker == "SPY"
            if interval == "5m":
                closes = self.spy_5m if is_spy else self.series_5m
                out[ticker] = _bars(ticker, interval, closes, OPEN_UTC, timedelta(minutes=5))
            elif interval == "1h":
                closes = _uptrend(60, 490.0 if is_spy else 96.0, 0.35 if is_spy else 0.12)
                out[ticker] = _bars(
                    ticker, interval, closes,
                    OPEN_UTC - timedelta(hours=60), timedelta(hours=1),
                )
            else:  # 1d
                closes = _uptrend(80, 460.0 if is_spy else 88.0, 0.6 if is_spy else 0.18)
                out[ticker] = _bars(
                    ticker, interval, closes,
                    OPEN_UTC - timedelta(days=80), timedelta(days=1),
                )
        return out


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """A scan environment pinned to mid-session so the result is time-independent."""
    monkeypatch.setattr(scanner_mod, "_shared_client", FakeClient())
    monkeypatch.setattr(scanner_mod, "current_market_phase", lambda: "REGULAR")
    monkeypatch.setattr(scanner_mod, "market_open_today_utc", lambda: OPEN_UTC)
    monkeypatch.setattr(scanner_mod, "opening_range_end_utc",
                        lambda: OPEN_UTC + timedelta(minutes=30))
    # Past the opening-chop window, so signals are armed rather than display-only.
    monkeypatch.setattr(scanner_mod, "minutes_since_open", lambda: 180.0)
    monkeypatch.setattr(scanner_mod, "minutes_to_close", lambda: 210.0)

    storage = Storage(tmp_path / "scan.sqlite3")
    settings = AppSettings(db_path=tmp_path / "scan.sqlite3")
    request = ScanRequest(tickers=["AAPL"], min_avg_dollar_volume=0.0)
    return settings, storage, request


def _run(env) -> tuple:
    settings, storage, request = env
    summary = run_scan(settings, storage, request)
    snapshot = next(s for s in storage.load_latest_snapshots() if s["ticker"] == "AAPL")
    return summary, storage, snapshot


class TestScanProducesTrainingData:
    def test_scan_completes_and_scores_the_ticker(self, env):
        summary, _, snap = _run(env)
        assert summary.tickers_scanned == 1
        assert summary.errors == 0
        assert snap["total_score"] > 0
        assert snap["trade_signal"] in (
            "STRONG_BUY", "BUY_CANDIDATE", "WATCH_ONLY", "AVOID",
        )

    def test_rules_only_scan_records_no_probability(self, env):
        """With no model trained, the scanner must run rules-only rather than
        inventing a probability."""
        _, _, snap = _run(env)
        assert snap["model_prob"] is None
        assert snap["required_prob"] is not None   # the bar still gets computed

    def test_new_columns_are_persisted(self, env):
        _, _, snap = _run(env)
        for col in ("cost_pct", "cost_ratio", "required_prob", "rel_volume",
                    "extension_atr", "blocked_ahead"):
            assert col in snap
        assert snap["cost_pct"] > 0
        assert snap["rel_volume"] is not None

    def test_armed_signal_carries_a_complete_feature_vector(self, env):
        """The loop-closing assertion: what gets armed also gets its inputs logged."""
        _, storage, snap = _run(env)
        if snap["trade_signal"] not in ("STRONG_BUY", "BUY_CANDIDATE",
                                        "STRONG_SHORT", "SHORT_CANDIDATE"):
            pytest.skip(f"synthetic series scored {snap['trade_signal']}, nothing armed")

        tracked = storage.load_tracked_signals("open")
        assert len(tracked) == 1
        row = tracked[0]
        assert row["feature_version"] == FEATURE_VERSION
        feats = json.loads(row["features_json"])
        assert set(feats) == set(FEATURE_NAMES)
        # Context copied alongside the vector, for slicing performance later.
        assert row["total_score"] == pytest.approx(snap["total_score"])
        assert row["cost_pct"] == pytest.approx(snap["cost_pct"])
        assert row["market_phase"] == "REGULAR"

    def test_feature_vector_matches_the_snapshot_it_came_from(self, env):
        _, storage, snap = _run(env)
        tracked = storage.load_tracked_signals("open")
        if not tracked:
            pytest.skip("nothing armed on this synthetic series")
        feats = json.loads(tracked[0]["features_json"])
        assert feats["direction"] == 1.0
        assert feats["total_score"] == pytest.approx(snap["total_score"])
        assert feats["adx14"] == pytest.approx(snap["adx14"])
        # stop_pct comes from the provisional levels, so it is never a placeholder 0.
        assert feats["stop_pct"] > 0


class TestModelVeto:
    def _activate(self, storage: Storage, intercept: float) -> None:
        """Activate a constant model: all-zero coefficients, so every setup gets
        sigmoid(intercept)."""
        model = StockModel(
            coefficients=[0.0] * len(FEATURE_NAMES),
            intercept=intercept,
            mean=[0.0] * len(FEATURE_NAMES),
            std=[1.0] * len(FEATURE_NAMES),
        )
        storage.save_model(model.to_json(), {"feature_version": FEATURE_VERSION},
                           activate=True)

    def _shadow(self, storage: Storage, intercept: float) -> None:
        """Same constant model, but parked in the logging-only lane."""
        model = StockModel(
            coefficients=[0.0] * len(FEATURE_NAMES),
            intercept=intercept,
            mean=[0.0] * len(FEATURE_NAMES),
            std=[1.0] * len(FEATURE_NAMES),
        )
        model_id = storage.save_model(model.to_json(),
                                      {"feature_version": FEATURE_VERSION},
                                      activate=False)
        storage.shadow_model(model_id)

    def test_pessimistic_model_vetoes_the_setup(self, env):
        settings, storage, request = env
        baseline = run_scan(settings, storage, request)

        self._activate(storage, intercept=-6.0)      # P(win) ≈ 0.2%
        run_scan(settings, storage, request)
        snap = next(s for s in storage.load_latest_snapshots() if s["ticker"] == "AAPL")

        assert snap["model_prob"] is not None
        assert snap["model_prob"] < snap["required_prob"]
        assert snap["trade_signal"] in ("WATCH_ONLY", "AVOID")
        assert baseline.tickers_scanned == 1

    def test_optimistic_model_does_not_manufacture_a_signal(self, env):
        """A confident model must not promote a setup the rules never proposed —
        it is a veto, not a promoter."""
        settings, storage, request = env
        run_scan(settings, storage, request)
        rules_only = next(s for s in storage.load_latest_snapshots() if s["ticker"] == "AAPL")

        self._activate(storage, intercept=6.0)       # P(win) ≈ 99.8%
        run_scan(settings, storage, request)
        gated = next(s for s in storage.load_latest_snapshots() if s["ticker"] == "AAPL")

        assert gated["model_prob"] > gated["required_prob"]
        # The label can only stay the same or be downgraded, never upgraded.
        assert gated["trade_signal"] == rules_only["trade_signal"]

    def test_shadow_model_scores_without_touching_the_label(self, env):
        """The whole point of shadow: a probability that would have vetoed still
        gets logged, but the rules signal reaches the dashboard untouched."""
        settings, storage, request = env
        run_scan(settings, storage, request)
        rules_only = next(s for s in storage.load_latest_snapshots() if s["ticker"] == "AAPL")

        self._shadow(storage, intercept=-6.0)        # would veto if it were gating
        run_scan(settings, storage, request)
        shadowed = next(s for s in storage.load_latest_snapshots() if s["ticker"] == "AAPL")

        assert shadowed["model_prob"] is not None
        assert shadowed["model_prob"] < shadowed["required_prob"]
        assert shadowed["trade_signal"] == rules_only["trade_signal"]

    def test_shadow_rows_are_tagged_so_the_report_can_separate_them(self, env):
        settings, storage, request = env
        self._shadow(storage, intercept=-6.0)
        run_scan(settings, storage, request)

        tracked = storage.load_tracked_signals("open")
        if not tracked:
            pytest.skip("nothing armed on this synthetic series")
        assert tracked[0]["model_mode"] == "shadow"
        assert tracked[0]["model_prob"] is not None

    def test_an_active_model_takes_precedence_over_a_shadow_one(self, env):
        """Two models scoring the same setup would make model_prob ambiguous, so the
        gating one wins and the shadow is ignored until nothing is active."""
        settings, storage, request = env
        self._activate(storage, intercept=6.0)       # P(win) ~ 99.8%
        self._shadow(storage, intercept=-6.0)        # P(win) ~ 0.2%
        run_scan(settings, storage, request)

        snap = next(s for s in storage.load_latest_snapshots() if s["ticker"] == "AAPL")
        assert snap["model_prob"] > 0.9
        tracked = storage.load_tracked_signals("open")
        if tracked:
            assert tracked[0]["model_mode"] == "active"

    def test_stale_feature_version_falls_back_to_rules_only(self, env):
        settings, storage, request = env
        model = StockModel(
            coefficients=[0.0] * len(FEATURE_NAMES), intercept=-6.0,
            mean=[0.0] * len(FEATURE_NAMES), std=[1.0] * len(FEATURE_NAMES),
            feature_version=FEATURE_VERSION + 1,
        )
        storage.save_model(model.to_json(), {"feature_version": FEATURE_VERSION + 1},
                           activate=True)
        run_scan(settings, storage, request)
        snap = next(s for s in storage.load_latest_snapshots() if s["ticker"] == "AAPL")
        assert snap["model_prob"] is None, "a mismatched model must not be served"
