"""TradingView chart links for ticker cells in the dashboard tables."""
from __future__ import annotations

from urllib.parse import quote

_CHART_URL = "https://www.tradingview.com/chart/?symbol="

# Yahoo index symbols -> TradingView's names for the same index.
_INDEX_MAP = {
    "^GSPC": "SP:SPX",
    "^DJI": "DJ:DJI",
    "^IXIC": "NASDAQ:IXIC",
    "^NDX": "NASDAQ:NDX",
    "^RUT": "TVC:RUT",
    "^VIX": "TVC:VIX",
}

# Pulls the display ticker back out of the URL for st.column_config.LinkColumn,
# so the cell still reads "AAPL" rather than the whole link.
DISPLAY_REGEX = r"#(.+)$"


def tradingview_symbol(ticker: str) -> str:
    """Yahoo ticker -> TradingView symbol. Share classes use a dot there
    (BRK-B -> BRK.B); without an exchange prefix TradingView picks the primary
    listing."""
    t = str(ticker).strip().upper()
    return _INDEX_MAP.get(t, t.replace("-", "."))


def tradingview_url(ticker: str) -> str:
    """Chart URL for ``ticker``. The original ticker rides in the fragment (which
    TradingView ignores) so the table can display it via DISPLAY_REGEX."""
    t = str(ticker).strip().upper()
    return f"{_CHART_URL}{quote(tradingview_symbol(t), safe='')}#{t}"
