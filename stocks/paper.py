from __future__ import annotations
from typing import List, Optional, Sequence, Tuple

from .models import StockQuote
from .yf_client import YFClient

# Paper-trading money math. Kept free of DB and Streamlit imports so the
# arithmetic is unit-testable on its own.
#
# `direction` is +1 (long) or -1 (short) throughout — the same convention as
# stock_signal_tracking — so one formula covers both sides.

LONG = 1
SHORT = -1


def scale_in(
    qty: int, avg_entry: float, add_qty: int, add_price: float
) -> Tuple[int, float]:
    """Share count and weighted average cost after adding to a position."""
    new_qty = qty + add_qty
    if new_qty <= 0:
        return 0, round(avg_entry, 4)
    new_avg = (avg_entry * qty + add_price * add_qty) / new_qty
    return new_qty, round(new_avg, 4)


def realized_pnl(direction: int, avg_entry: float, exit_price: float, qty: int) -> float:
    """Dollar P&L from closing ``qty`` shares. The direction sign handles shorts:
    a short profits when the exit is below the entry."""
    return round((exit_price - avg_entry) * direction * qty, 2)


def unrealized_pnl(direction: int, avg_entry: float, last: float, qty: int) -> float:
    """Open P&L at the current price — same math as a close that hasn't happened."""
    return realized_pnl(direction, avg_entry, last, qty)


def pnl_pct(direction: int, avg_entry: float, price: float) -> Optional[float]:
    """P&L as a percent of entry, per share. None when entry is unusable."""
    if not avg_entry:
        return None
    return round((price - avg_entry) * direction / avg_entry * 100, 3)


def position_r_multiple(
    direction: int, avg_entry: float, price: float, stop_price: Optional[float]
) -> Optional[float]:
    """P&L in units of initial risk. None when there is no stop, or the stop sits
    at the entry — both leave risk-per-share at zero with nothing to divide by."""
    if not stop_price or not avg_entry:
        return None
    risk_per_share = (avg_entry - stop_price) * direction
    if risk_per_share <= 0:
        return None
    return round((price - avg_entry) * direction / risk_per_share, 2)


def position_cost(qty: int, price: float) -> float:
    """Notional dollars committed."""
    return round(qty * price, 2)


# ── Live quotes for open positions ───────────────────────────────────────────

# Dedicated client: the scanner's shared client caches 1d bars keyed by the
# watchlist set, and asking it for other tickers would evict that cache.
_quote_client = YFClient()


def fetch_position_quotes(tickers: Sequence[str]) -> List[StockQuote]:
    """Latest price per held ticker, one batched request. Empty in, empty out."""
    if not tickers:
        return []
    return _quote_client.get_quotes(list(tickers))
