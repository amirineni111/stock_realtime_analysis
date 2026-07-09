# Stock Real-Time Screening Dashboard

A near-real-time US stock screening dashboard built with Streamlit and free Yahoo Finance data (yfinance — no API key required). Add tickers to a watchlist, and the app polls candles on a configurable interval, scores each ticker with a multi-factor engine, and shows BUY/SELL signals with suggested entry/stop/target levels. Every actionable signal is automatically forward-tested against its stop/target, so the Performance tab reports real win rates and expectancy over time.

## Features

- **Manual watchlist** — type tickers in the sidebar (persisted between sessions)
- **Signals**: 🟢 STRONG_BUY · 🔵 BUY_CANDIDATE · 🟠 SHORT_CANDIDATE · 🔴 STRONG_SHORT · 🟡 WATCH_ONLY · ⚫ AVOID
- **Scoring engine** (per ticker, on completed 5-minute bars):
  - Momentum (EMA9/20, MACD, RSI) and mean reversion (RSI extremes, Bollinger Bands, day-range position), blended by an ADX regime gate (trend vs range)
  - Opening-range / day high-low breakout with a first/last-hour liquidity bonus
  - Multi-timeframe confluence across 5m / hourly / daily trends
  - Support/resistance proximity from daily + hourly pivot levels
  - Relative strength vs SPY (outperformers get a long bonus, underperformers a short bonus)
  - Average-dollar-volume liquidity gate
- **Trade levels**: ATR-based stop (1.5×ATR, floor 0.30% of price) and 1.5R target, in dollars per share
- **Self-calibrating performance tracking**: each scan checks open tracked signals against fresh bars; stop/target touches record WIN/LOSS outcomes feeding win-rate, expectancy, and equity-curve stats
- **Market-hours aware**: phase badge (pre-market / regular / after-hours / closed); auto-refresh pauses off-hours unless overridden; manual scans always work against the last session's bars
- **Efficient fetching**: 3 batched yfinance requests per scan regardless of watchlist size, with TTL caches on hourly/daily frames; the still-forming candle is dropped so signals never repaint

## Quick start

```
python -m venv venv
venv\Scripts\pip install -r requirements.txt
venv\Scripts\streamlit run app.py --server.port 8502
```

Or on Windows just run `start_stock_dashboard.bat` and open http://localhost:8502.

## Configuration

No API keys needed. Optional overrides go in `.env` (see `.env.example`):

- `STOCKS_DB_PATH` — SQLite location. Defaults to `%LOCALAPPDATA%\StocksRealtimeScreening\stocks_screening.sqlite3`, deliberately outside synced folders (OneDrive + SQLite WAL files cause lock/sync conflicts).

## Project layout

```
app.py                    Streamlit UI (Scanner + Live Quotes pages)
stocks/
  yf_client.py            Batched yfinance data client (forming-candle drop, TTL caches)
  scanner.py              Scan orchestration: fetch → indicators → score → persist → forward-test
  signals.py              Scoring engine and trade-level computation
  indicators.py           RSI/EMA/MACD/ATR/ADX/Bollinger/S&R (pure-Python OHLC math)
  relative_strength.py    Relative strength vs SPY
  market_hours.py         US equity market phases (America/New_York)
  storage.py              SQLite persistence + forward-test WIN/LOSS resolution
  models.py               Pydantic models
  config.py, tickers.py, refresh.py
tests/                    pytest suite (52 tests, all offline)
```

## Tests

```
venv\Scripts\python -m pytest tests/ -q
```

## Known limitations (v1)

- No market-holiday calendar: on a holiday the phase may read REGULAR while data simply stays stale (the "last bar N min ago" caption reveals it)
- yfinance has no bid/ask, so entries use the last completed 5m close; Yahoo intraday data can lag ~1–2 minutes
- Keep watchlists under ~50 tickers to stay friendly with Yahoo rate limits

> Not financial advice. Signals are heuristics for screening, not trade recommendations.
