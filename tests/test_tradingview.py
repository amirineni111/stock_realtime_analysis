import re

from stocks.tradingview import DISPLAY_REGEX, tradingview_symbol, tradingview_url


def test_plain_ticker():
    assert tradingview_url("aapl") == "https://www.tradingview.com/chart/?symbol=AAPL#AAPL"


def test_share_class_uses_dot():
    assert tradingview_symbol("BRK-B") == "BRK.B"


def test_yahoo_index_maps_to_tradingview_name():
    assert tradingview_url("^GSPC") == "https://www.tradingview.com/chart/?symbol=SP%3ASPX#^GSPC"


def test_display_regex_shows_the_original_ticker():
    for t in ("NVDA", "BRK-B", "^IXIC"):
        assert re.search(DISPLAY_REGEX, tradingview_url(t)).group(1) == t
