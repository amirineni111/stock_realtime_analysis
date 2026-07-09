from __future__ import annotations
import re
from typing import List

# Benchmark used for relative-strength comparison; always fetched alongside the watchlist.
BENCHMARK = "SPY"

# Yahoo-style tickers: letters, digits, dots (BRK.B) and dashes (BRK-B).
_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")


def parse_watchlist(raw: str) -> List[str]:
    """Parse a free-form watchlist string (newlines/commas/spaces) into a
    deduplicated, upper-cased list of valid tickers, preserving input order."""
    if not raw:
        return []
    tokens = re.split(r"[\s,;]+", raw.upper())
    seen = set()
    result: List[str] = []
    for token in tokens:
        token = token.strip()
        if not token or token in seen or not _TICKER_RE.match(token):
            continue
        seen.add(token)
        result.append(token)
    return result
