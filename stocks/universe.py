"""
Stock universes for cross-sectional (ranking) research.

NASDAQ_100 is a hand-maintained approximation of the index's recent members, not
a point-in-time constituent history. Two consequences, both stated in every
report that uses it:

- survivorship bias: names that fell out of the index (usually after falling)
  are missing, which flatters anything that buys losers;
- drift: membership changes every December and on ad-hoc replacements. Tickers
  Yahoo no longer serves simply drop out of the download.

GOOG is omitted deliberately: it is the same company as GOOGL, and a long/short
ranker would happily "discover" the share-class spread.
"""
from __future__ import annotations

NASDAQ_100 = (
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "COST", "NFLX",
    "AMD", "PEP", "ADBE", "CSCO", "TMUS", "INTC", "QCOM", "TXN", "AMGN", "INTU",
    "CMCSA", "HON", "AMAT", "ISRG", "BKNG", "VRTX", "ADP", "SBUX", "GILD", "MDLZ",
    "ADI", "REGN", "LRCX", "PANW", "MU", "KLAC", "SNPS", "CDNS", "MELI", "PYPL",
    "CSX", "ASML", "MAR", "ORLY", "CTAS", "ABNB", "MNST", "CRWD", "FTNT", "NXPI",
    "WDAY", "PCAR", "CHTR", "ROP", "ADSK", "MRVL", "CPRT", "AEP", "PAYX", "KDP",
    "ROST", "ODFL", "DXCM", "FAST", "EA", "KHC", "IDXX", "CTSH", "VRSK", "EXC",
    "XEL", "BKR", "GEHC", "CCEP", "LULU", "TTWO", "CSGP", "ON", "ZS", "DDOG",
    "TEAM", "BIIB", "CDW", "WBD", "GFS", "ARM", "PLTR", "APP", "AXON", "MSTR",
    "TRI", "LIN", "PDD", "AZN", "SHOP", "DASH", "FANG", "MCHP", "CEG",
)
