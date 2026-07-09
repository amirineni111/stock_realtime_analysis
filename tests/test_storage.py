from pathlib import Path

import pytest

from stocks.storage import Storage


@pytest.fixture()
def storage(tmp_path: Path) -> Storage:
    return Storage(tmp_path / "test.sqlite3")


def _arm(storage: Storage, ticker: str, direction: int = 1) -> None:
    # entry 100, stop 98.8, target 101.8 for longs (mirrored for shorts)
    if direction == 1:
        stop, target = 98.8, 101.8
    else:
        stop, target = 101.2, 98.2
    storage.record_tracked_signal(
        ticker=ticker, signal="BUY_CANDIDATE" if direction == 1 else "SHORT_CANDIDATE",
        direction=direction, entry=100.0, stop=stop, target=target,
        stop_dollars=1.2, target_dollars=1.8, atr14=0.8,
        entry_ts="2026-07-08T14:00:00+00:00",
    )


def _bar(ts: str, high: float, low: float, close: float) -> dict:
    return {"timestamp": ts, "open": close, "high": high, "low": low, "close": close, "volume": 100}


def test_target_touch_wins(storage: Storage):
    _arm(storage, "WINR")
    bars = [_bar("2026-07-08T14:05:00+00:00", 102.0, 99.5, 101.5)]
    assert storage.evaluate_tracked_signals("WINR", bars) == 1
    outcomes = storage.load_trade_outcomes()
    assert len(outcomes) == 1
    o = outcomes[0]
    assert o["outcome"] == "WIN"
    assert o["exit_price"] == 101.8
    assert o["exit_dollars"] == pytest.approx(1.8)
    assert o["r_multiple"] == pytest.approx(1.5)
    assert storage.load_tracked_signals("open") == []


def test_stop_touch_loses(storage: Storage):
    _arm(storage, "LOSR")
    bars = [_bar("2026-07-08T14:05:00+00:00", 101.0, 98.5, 99.0)]
    assert storage.evaluate_tracked_signals("LOSR", bars) == 1
    o = storage.load_trade_outcomes()[0]
    assert o["outcome"] == "LOSS"
    assert o["exit_price"] == 98.8
    assert o["r_multiple"] == pytest.approx(-1.0)


def test_both_touched_is_conservative_loss(storage: Storage):
    _arm(storage, "BOTH")
    bars = [_bar("2026-07-08T14:05:00+00:00", 102.5, 98.5, 100.0)]
    storage.evaluate_tracked_signals("BOTH", bars)
    assert storage.load_trade_outcomes()[0]["outcome"] == "LOSS"


def test_short_direction(storage: Storage):
    _arm(storage, "SHRT", direction=-1)
    bars = [_bar("2026-07-08T14:05:00+00:00", 100.5, 98.0, 98.5)]  # target 98.2 not hit... low 98.0 <= 98.2 → WIN
    storage.evaluate_tracked_signals("SHRT", bars)
    o = storage.load_trade_outcomes()[0]
    assert o["outcome"] == "WIN"
    assert o["exit_price"] == 98.2
    assert o["exit_dollars"] == pytest.approx(1.8)


def test_no_touch_stays_open(storage: Storage):
    _arm(storage, "OPEN")
    bars = [_bar("2026-07-08T14:05:00+00:00", 100.5, 99.5, 100.1)]
    assert storage.evaluate_tracked_signals("OPEN", bars) == 0
    assert len(storage.load_tracked_signals("open")) == 1


def test_timeout_closes_at_last_close(storage: Storage):
    _arm(storage, "TOUT")
    bars = [_bar("2026-07-08T14:05:00+00:00", 100.5, 99.5, 100.4)]
    # max_hold_hours=0 → already aged out; closes at last close 100.4 → WIN (pnl > 0)
    assert storage.evaluate_tracked_signals("TOUT", bars, max_hold_hours=0.0) == 1
    o = storage.load_trade_outcomes()[0]
    assert o["outcome"] == "WIN"
    assert o["exit_price"] == 100.4


def test_rearm_cooldown_skips_duplicate(storage: Storage):
    _arm(storage, "DUPE")
    _arm(storage, "DUPE")
    assert len(storage.load_tracked_signals("open")) == 1


def test_forward_bars_only(storage: Storage):
    _arm(storage, "PAST")
    # Bar BEFORE entry_ts touches the target — must be ignored
    bars = [_bar("2026-07-08T13:55:00+00:00", 102.0, 99.0, 101.9)]
    assert storage.evaluate_tracked_signals("PAST", bars) == 0


def test_watchlist_outcome_and_performance(storage: Storage):
    storage.add_watchlist("AAPL", "BUY_CANDIDATE", 100.0, 101.8, 98.8, 1.2, 1.8, "test")
    row = storage.load_watchlist("watching")[0]
    storage.close_watchlist_with_outcome(row["id"], 101.0)
    outcomes = storage.load_trade_outcomes()
    assert outcomes[0]["outcome"] == "WIN"
    assert outcomes[0]["exit_dollars"] == pytest.approx(1.0)
    assert outcomes[0]["exit_pct"] == pytest.approx(1.0)
    perf = storage.load_performance_by_dimension("overall")
    assert perf and perf[0]["trades"] == 1 and perf[0]["wins"] == 1


def test_scan_lifecycle(storage: Storage):
    from stocks.models import ScanSummary
    scan_id = storage.start_scan()
    storage.log_ticker(scan_id, "AAPL", "WATCH_ONLY", None)
    storage.finish_scan(scan_id, ScanSummary(tickers_scanned=1, errors=0, signals_found=0))
    run = storage.load_latest_scan_run()
    assert run["tickers_scanned"] == 1
    assert run["finished_at"] is not None
