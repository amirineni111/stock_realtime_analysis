from stocks.tickers import parse_watchlist, BENCHMARK


def test_parse_mixed_separators():
    raw = " aapl, msft\nnvda;tsla  amd "
    assert parse_watchlist(raw) == ["AAPL", "MSFT", "NVDA", "TSLA", "AMD"]


def test_dedupe_preserves_order():
    assert parse_watchlist("MSFT AAPL msft") == ["MSFT", "AAPL"]


def test_invalid_tokens_dropped():
    assert parse_watchlist("AAPL 123 BAD$TICKER BRK.B BRK-B") == ["AAPL", "BRK.B", "BRK-B"]


def test_empty():
    assert parse_watchlist("") == []
    assert parse_watchlist("   \n , ") == []


def test_benchmark_is_spy():
    assert BENCHMARK == "SPY"
