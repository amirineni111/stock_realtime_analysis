# Stock Real-Time Screening Dashboard

A near-real-time US stock screening dashboard built with Streamlit and free Yahoo Finance data (yfinance — no API key required). Add tickers to a watchlist, and the app polls candles on a configurable interval, scores each ticker with a multi-factor engine, and shows BUY/SELL signals with suggested entry/stop/target levels. Every actionable signal is automatically forward-tested against its stop/target, so the Performance tab reports real win rates and expectancy over time.

## Features

- **Manual watchlist** — type tickers in the sidebar (persisted between sessions)
- **Signals**: 🟢 STRONG_BUY · 🔵 BUY_CANDIDATE · 🟠 SHORT_CANDIDATE · 🔴 STRONG_SHORT · 🟡 WATCH_ONLY · ⚫ AVOID
- **Scoring engine** (per ticker, on completed 5-minute bars):
  - Momentum (EMA9/20, MACD, RSI) and mean reversion (RSI extremes, Bollinger Bands, day-range position), blended by an ADX regime gate (trend vs range)
  - Opening-range / day high-low breakout with a last-hour liquidity bonus
  - Multi-timeframe confluence: the 5m read has to be *confirmed* by the hourly and daily trend (both present and agreeing for full credit — an unconfirmed 5m read scores nothing)
  - Direction-aware support/resistance from daily + hourly pivots: a level behind the trade shelters the stop (+25), one blocking the path to target is a penalty (−25) and downgrades the signal
  - Relative strength vs SPY, folded into the score itself so the displayed, decided, and trained score are one number
  - Average-dollar-volume liquidity gate, plus a transaction-cost veto
- **Trade levels**: ATR-based stop (2.5×ATR, floor 0.50% of price) and 1.5R target, in dollars per share
- **Cost-aware accounting**: round-trip cost is estimated from liquidity and charged against every trade, so reported R is net rather than flattering. A setup whose estimated cost exceeds 15% of the risk is vetoed outright
- **Trained direction model** (see below): a calibrated P(target before stop) that vetoes setups whose measured odds don't clear cost-adjusted breakeven
- **Self-calibrating performance tracking**: each scan checks open tracked signals against fresh bars; stop/target touches record WIN/LOSS outcomes feeding win-rate, expectancy, and equity-curve stats
- **Calibration-driven guards** (from forward-test results): signals in the first hour after the open display but are not forward-tested as trades, and a STRONG signal stretched more than 2×ATR from its EMA20 downgrades to WATCH_ONLY instead of chasing
- **Interactive results grid**: click a row to highlight it end-to-end and see its entry/stop/target/R:R at a glance; arrange/hide columns in any order you like (saved between sessions, "Reset to default" one click away)
- **Market-hours aware**: phase badge (pre-market / regular / after-hours / closed); auto-refresh pauses off-hours unless overridden; manual scans always work against the last session's bars
- **Efficient fetching**: 3 batched yfinance requests per scan regardless of watchlist size, with TTL caches on hourly/daily frames; the still-forming candle is dropped so signals never repaint

## Quick start

```
python -m venv venv
venv\Scripts\pip install -r requirements.txt
venv\Scripts\streamlit run app.py --server.port 8502
```

Or on Windows just run `start_stock_dashboard.bat` and open http://localhost:8502.

## The learning loop

The rules score **proposes** a setup; a trained model **disposes**. The two never
swap roles: the model can only veto, never promote something the rules rejected.

1. **Log.** Every actionable signal is armed for forward-testing *with its feature
   vector attached* (`stock_signal_tracking.features_json`). Without this the
   outcome rows are unlearnable — the snapshot table is pruned to the last 20
   scans, so the inputs are gone by the time the trade resolves.
2. **Resolve.** Later scans check open signals against fresh bars. A stop or target
   touch (stop checked first — conservative) records a WIN/LOSS **net of estimated
   cost**, linked back to the tracking row that produced it.
3. **Train.** Once ~120 resolved trades have accumulated, the Model tab (or
   `scripts/train_model.py`) walk-forward evaluates an L2 logistic regression over
   the logged features and reports out-of-sample AUC, Brier, calibration, and
   expectancy at each candidate threshold.
4. **Gate.** A candidate is saved **inactive**. It only passes if it beats chance
   out of sample *and* turns a profit at its decision threshold — promoting is a
   separate, deliberate click, and any earlier model can be rolled back.
5. **Serve.** The active model scores each setup live. Below the cost-adjusted
   breakeven win rate plus a margin, an otherwise-actionable signal is downgraded
   to WATCH_ONLY with the reason shown.

```
python scripts/train_model.py              # evaluate + save a candidate, do not activate
python scripts/train_model.py --activate   # promote it, if it clears the gate
python scripts/backfill_links.py --apply   # one-off: link pre-existing outcome rows
```

Editing `FEATURE_NAMES` means bumping `FEATURE_VERSION` in `stocks/features.py`:
the scanner refuses to serve a model built on a different feature contract, so it
fails closed to rules-only rather than serving misaligned probabilities.

Why logistic regression and not gradient boosting: the effective sample is far
smaller than the row count, because equities move together intraday — simultaneous
longs across AAPL/MSFT/NVDA are largely one beta bet. A linear model in probability
space is readable and close to calibrated, which the decision rule requires. See
the module docstring in `stocks/model.py` for the full reasoning.

## Configuration

No API keys needed. Optional overrides go in `.env` (see `.env.example`):

- `STOCKS_DB_PATH` — SQLite location. Defaults to `%LOCALAPPDATA%\StocksRealtimeScreening\stocks_screening.sqlite3`, deliberately outside synced folders (OneDrive + SQLite WAL files cause lock/sync conflicts).

## Project layout

```
app.py                    Streamlit UI (Scanner + Live Quotes pages; Model tab)
stocks/
  yf_client.py            Batched yfinance data client (forming-candle drop, TTL caches)
  scanner.py              Scan orchestration: fetch → indicators → score → model → persist → forward-test
  signals.py              Scoring engine, cost model, trade-level computation
  indicators.py           RSI/EMA/MACD/ATR/ADX/Bollinger/S&R/relative volume (pure-Python OHLC math)
  features.py             Canonical feature contract — the one place serving and training agree
  model.py                Logistic model, walk-forward evaluation, calibration metrics
  training.py             Retrain pipeline + quality gate, shared by the CLI and the Model tab
  relative_strength.py    Relative strength vs SPY
  market_hours.py         US equity market phases (America/New_York)
  timeutil.py             Timestamp parsing + intraday session clock
  storage.py              SQLite persistence, forward-test resolution, model store
  models.py               Pydantic models
  config.py, tickers.py, refresh.py
scripts/
  train_model.py          Retrain from the CLI, with the same gate as the dashboard
  backfill_links.py       One-off link of legacy outcome rows to their tracking rows
tests/                    pytest suite (140 tests, all offline)
```

## Tests

```
venv\Scripts\python -m pytest tests/ -q
```

## Known limitations (v1)

- No market-holiday calendar: on a holiday the phase may read REGULAR while data simply stays stale (the "last bar N min ago" caption reveals it)
- yfinance has no bid/ask, so entries use the last completed 5m close and transaction cost is **estimated** from liquidity tiers rather than observed; Yahoo intraday data can lag ~1–2 minutes
- Forward-tested fills are optimistic in one direction: a bar that trades through the target is credited at the target, with no slippage beyond the modelled round-trip cost
- Trades armed before feature logging existed are not trainable — their inputs are unrecoverable, and they are excluded rather than imputed (see `scripts/backfill_links.py`)
- Keep watchlists under ~50 tickers to stay friendly with Yahoo rate limits

> Not financial advice. Signals are heuristics for screening, not trade recommendations.
