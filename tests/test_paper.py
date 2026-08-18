from pathlib import Path

import pytest

from stocks.paper import (
    pnl_pct,
    position_r_multiple,
    realized_pnl,
    scale_in,
    unrealized_pnl,
)
from stocks.storage import Storage


@pytest.fixture()
def storage(tmp_path: Path) -> Storage:
    return Storage(tmp_path / "test.sqlite3")


# ── Money math ───────────────────────────────────────────────────────────────

def test_scale_in_weighted_average():
    qty, avg = scale_in(100, 241.30, 50, 244.00)
    assert qty == 150
    assert avg == pytest.approx(242.20)


def test_scale_in_twice_stays_weighted():
    qty, avg = scale_in(100, 100.0, 100, 110.0)
    assert (qty, avg) == (200, pytest.approx(105.0))
    # A third lot must average against the running 200 shares, not the last fill.
    qty, avg = scale_in(qty, avg, 200, 115.0)
    assert qty == 400
    assert avg == pytest.approx(110.0)


def test_realized_pnl_long():
    assert realized_pnl(1, 100.0, 105.0, 10) == pytest.approx(50.0)
    assert realized_pnl(1, 100.0, 95.0, 10) == pytest.approx(-50.0)


def test_realized_pnl_short_inverts():
    """A short profits when price falls — the direction sign is the whole trick."""
    assert realized_pnl(-1, 100.0, 95.0, 10) == pytest.approx(50.0)
    assert realized_pnl(-1, 100.0, 105.0, 10) == pytest.approx(-50.0)


def test_unrealized_matches_realized_at_same_price():
    assert unrealized_pnl(1, 100.0, 103.0, 25) == realized_pnl(1, 100.0, 103.0, 25)


def test_pnl_pct_both_directions():
    assert pnl_pct(1, 100.0, 102.0) == pytest.approx(2.0)
    assert pnl_pct(-1, 100.0, 98.0) == pytest.approx(2.0)
    assert pnl_pct(1, 0.0, 100.0) is None


def test_position_r_multiple():
    # Long 100 with a stop at 98 risks 2/share; +2 at 102 is exactly 1R.
    assert position_r_multiple(1, 100.0, 102.0, 98.0) == pytest.approx(1.0)
    assert position_r_multiple(1, 100.0, 98.0, 98.0) == pytest.approx(-1.0)
    # Short 100 stopped at 102 risks 2/share; a drop to 98 is 1R.
    assert position_r_multiple(-1, 100.0, 98.0, 102.0) == pytest.approx(1.0)


def test_position_r_multiple_none_without_usable_stop():
    assert position_r_multiple(1, 100.0, 102.0, None) is None
    assert position_r_multiple(1, 100.0, 102.0, 0) is None
    # Stop at the entry leaves zero risk — nothing to divide by.
    assert position_r_multiple(1, 100.0, 102.0, 100.0) is None
    # Stop on the wrong side of the entry is not a stop.
    assert position_r_multiple(1, 100.0, 102.0, 105.0) is None


# ── Position lifecycle ───────────────────────────────────────────────────────

def test_open_position_records_fill(storage: Storage):
    pid = storage.open_paper_position("AAPL", 1, 100, 241.30, 237.10, 247.60, "STRONG_BUY")
    p = storage.load_paper_position(pid)
    assert (p["ticker"], p["qty"], p["status"]) == ("AAPL", 100, "open")
    assert p["avg_entry"] == pytest.approx(241.30)

    fills = storage.load_paper_fills(pid)
    assert len(fills) == 1
    assert (fills[0]["kind"], fills[0]["side"], fills[0]["qty"]) == ("OPEN", "BUY", 100)


def test_short_open_records_sell_side(storage: Storage):
    pid = storage.open_paper_position("TSLA", -1, 10, 400.0)
    assert storage.load_paper_fills(pid)[0]["side"] == "SELL"


def test_add_reaverages_entry(storage: Storage):
    pid = storage.open_paper_position("AAPL", 1, 100, 241.30)
    storage.add_to_paper_position(pid, 50, 244.00)
    p = storage.load_paper_position(pid)
    assert p["qty"] == 150
    assert p["avg_entry"] == pytest.approx(242.20)
    assert [f["kind"] for f in storage.load_paper_fills(pid)] == ["ADD", "OPEN"]


def test_partial_close_keeps_position_open(storage: Storage):
    pid = storage.open_paper_position("AAPL", 1, 150, 242.20)
    pnl = storage.close_paper_position(pid, 50, 246.00)
    assert pnl == pytest.approx(190.0)

    p = storage.load_paper_position(pid)
    assert p["qty"] == 100
    assert p["status"] == "open"
    assert p["closed_at"] is None
    assert p["realized_pnl"] == pytest.approx(190.0)


def test_final_close_flips_status_and_accumulates(storage: Storage):
    pid = storage.open_paper_position("AAPL", 1, 150, 242.20)
    storage.close_paper_position(pid, 50, 246.00)
    storage.close_paper_position(pid, 100, 250.00)

    p = storage.load_paper_position(pid)
    assert p["qty"] == 0
    assert p["status"] == "closed"
    assert p["closed_at"] is not None
    assert p["realized_pnl"] == pytest.approx(970.0)  # 190 + 780


def test_close_caps_at_open_quantity(storage: Storage):
    """Over-closing must not manufacture shares or negative size."""
    pid = storage.open_paper_position("AAPL", 1, 100, 100.0)
    pnl = storage.close_paper_position(pid, 500, 101.0)
    assert pnl == pytest.approx(100.0)  # only the 100 held
    p = storage.load_paper_position(pid)
    assert p["qty"] == 0 and p["status"] == "closed"


def test_close_on_closed_position_is_noop(storage: Storage):
    pid = storage.open_paper_position("AAPL", 1, 100, 100.0)
    storage.close_paper_position(pid, 100, 101.0)
    assert storage.close_paper_position(pid, 10, 105.0) is None
    assert storage.add_to_paper_position(pid, 10, 105.0) is None


def test_short_round_trip_profits_when_price_falls(storage: Storage):
    pid = storage.open_paper_position("TSLA", -1, 10, 400.0, 410.0, 385.0)
    pnl = storage.close_paper_position(pid, 10, 390.0)
    assert pnl == pytest.approx(100.0)
    fills = storage.load_paper_fills(pid)
    # Covering a short is a buy.
    assert fills[0]["kind"] == "CLOSE" and fills[0]["side"] == "BUY"


def test_open_position_lookup_by_ticker(storage: Storage):
    assert storage.load_open_paper_position("AAPL") is None
    pid = storage.open_paper_position("AAPL", 1, 100, 100.0)
    assert storage.load_open_paper_position("aapl")["id"] == pid
    storage.close_paper_position(pid, 100, 101.0)
    assert storage.load_open_paper_position("AAPL") is None


# ── Account ──────────────────────────────────────────────────────────────────

def test_account_created_on_demand_and_tracks_realized(storage: Storage):
    account = storage.paper_account()
    assert account["starting_equity"] == pytest.approx(100000.0)
    assert account["equity"] == pytest.approx(100000.0)

    pid = storage.open_paper_position("AAPL", 1, 100, 100.0)
    storage.close_paper_position(pid, 100, 105.0)

    account = storage.paper_account()
    assert account["realized_pnl"] == pytest.approx(500.0)
    assert account["equity"] == pytest.approx(100500.0)


def test_reset_clears_positions_and_fills(storage: Storage):
    pid = storage.open_paper_position("AAPL", 1, 100, 100.0)
    storage.close_paper_position(pid, 100, 105.0)

    storage.reset_paper_account(50000.0)
    assert storage.load_paper_positions("all") == []
    assert storage.load_paper_fills() == []
    account = storage.paper_account()
    assert account["starting_equity"] == pytest.approx(50000.0)
    assert account["equity"] == pytest.approx(50000.0)
    assert account["reset_at"] is not None


# ── Isolation from the model's forward-test record ───────────────────────────

def test_paper_trades_never_touch_model_stats(storage: Storage):
    """The Performance tab counts every stock_trade_outcomes row as a model
    prediction, so paper trading must stay entirely out of that table."""
    before_outcomes = storage.load_trade_outcomes()
    before_training = storage.load_training_rows()

    pid = storage.open_paper_position("AAPL", 1, 100, 241.30, 237.10, 247.60, "STRONG_BUY")
    storage.add_to_paper_position(pid, 50, 244.00)
    storage.close_paper_position(pid, 75, 246.00)
    storage.close_paper_position(pid, 75, 250.00)
    sid = storage.open_paper_position("TSLA", -1, 10, 400.0)
    storage.close_paper_position(sid, 10, 390.0)

    assert storage.load_trade_outcomes() == before_outcomes
    assert storage.load_training_rows() == before_training
    assert storage.load_paper_positions("closed")  # the trades really did happen
