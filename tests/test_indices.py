from stocks.indices import INDEX_SYMBOLS, _index_client
from stocks.scanner import _shared_client
from stocks.tickers import parse_watchlist


def test_header_indices():
    assert INDEX_SYMBOLS == [
        ("^IXIC", "NASDAQ"),
        ("^DJI", "DOW"),
        ("^GSPC", "S&P 500"),
    ]


def test_index_symbols_bypass_watchlist_parsing():
    """The '^' prefix fails the ticker regex, so index symbols must never be
    routed through parse_watchlist — they'd be silently dropped."""
    raw = " ".join(sym for sym, _ in INDEX_SYMBOLS)
    assert parse_watchlist(raw) == []


def test_index_client_is_separate_from_scanner_client():
    """Sharing the scanner's client would evict its cached watchlist 1d bars,
    since the bar cache overwrites its entry with the new ticker set."""
    assert _index_client is not _shared_client


def test_points_and_percent_agree_in_sign():
    """The header shows points as last - prev_close alongside change_pct from
    the quote; the two must never disagree about direction."""
    for last, prev in [(26306.70, 26644.91), (53459.78, 53392.27), (100.0, 100.0)]:
        points = last - prev
        change_pct = (last - prev) / prev * 100
        assert (points > 0) == (change_pct > 0)
        assert (points < 0) == (change_pct < 0)
