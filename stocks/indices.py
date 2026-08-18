from __future__ import annotations
from typing import List, Tuple

from .models import StockQuote
from .yf_client import YFClient

# (yahoo_symbol, display_label) for the page-header strip. These bypass
# parse_watchlist() on purpose — its ticker regex rejects the leading '^'.
INDEX_SYMBOLS: List[Tuple[str, str]] = [
    ("^IXIC", "NASDAQ"),
    ("^DJI", "DOW"),
    ("^GSPC", "S&P 500"),
]

# Dedicated client: the scanner's shared client caches 1d bars keyed by the
# watchlist set, and asking it for index symbols would evict that cache.
_index_client = YFClient()


def fetch_index_quotes() -> List[StockQuote]:
    """Latest level + prev close for the header indices, one batched request.
    Points up/down are derived by the caller as ``last - prev_close``."""
    return _index_client.get_quotes([sym for sym, _ in INDEX_SYMBOLS])
