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
- **Pullback-entry gate**: an actionable signal is only taken from the *adverse* side of the day's range — a long in the bottom quarter, a short in the top quarter. Otherwise it shows as WATCH_ONLY ("wait for a pullback…"). See [Why the pullback gate](#why-the-pullback-gate)
- **Real-time alerts**: each newly armed signal pops a dashboard toast and, if configured, a phone/desktop push (ntfy, Discord, Slack or any webhook). `scripts/run_alerts.py` scans headlessly on every 5m bar close, so alerts don't depend on a browser tab being open
- **Historical replay backtester**: `scripts/backtest.py` runs the last ~60 sessions through the exact live scoring code and compares rule variants in a couple of minutes, instead of waiting weeks for forward-test results
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

## Real-time alerts

1. Pick a push target and put it in `.env` (see `.env.example`). The simplest is
   [ntfy](https://ntfy.sh): install the phone app, subscribe to a long, unguessable
   topic name, and set `STOCKS_ALERT_WEBHOOK_URL=https://ntfy.sh/<that-topic>`.
   Discord and Slack incoming-webhook URLs also work.
2. Check delivery: `python scripts/run_alerts.py --test-push`
3. During market hours run `python scripts/run_alerts.py`. It wakes 75 s after
   every 5-minute boundary (Yahoo publishes bars ~1 min late), scans the dashboard
   watchlist, and pushes each newly armed signal: ticker, side, entry, stop,
   target and reason.

An alert fires when a signal is **armed**, i.e. when it passes every gate
(including the pullback gate and the first-hour guard). Arming is de-duplicated
per ticker+direction with a 45-minute cooldown, so each setup alerts once. The
dashboard and the runner can run side by side; they share the database and its
dedupe.

## Data sources

| Use | Source | Why |
|---|---|---|
| Live scanning and alerts | yfinance | Free, ~1–2 min behind. Polygon's Basic plan has no same-day data, and Starter/Developer are 15 min delayed |
| Backtest history, 60 days | yfinance | Free, but 5m bars reach back only 60 days |
| Backtest history, 2 years | Polygon.io / Massive **Basic (free)** | 2 years of minute aggregates; 5 calls/min |

Longer history is what turns "positive in one 60-day window" into evidence. To
set it up:

1. Put `POLYGON_API_KEY=...` in `.env`.
2. Run `python scripts/download_history.py` once. Two years is ~9 calls per
   ticker, so at 5 calls/min the first run takes ~1 hour for the watchlist and
   ~3.5 hours with the NASDAQ-100. The watchlist goes first. The download can be
   stopped and re-run; later runs only fetch new days.
3. Pass `--source polygon` to either backtester. They read the cache only and
   never call the API, so they can run while a download is in progress.

Hourly and daily bars are built from the regular-session 5m bars with Yahoo's
09:30 alignment, so the replay sees the same shapes the live scanner does. A
spot check on NVDA matched yfinance on all 312 overlapping bars (max price
difference 0.9 bps). The first 60 days of a Polygon window only warm up the
indicators.

Upgrade path: Advanced ($199/mo) is the first plan with **quotes** (to measure
real spreads instead of the estimated 3 bps) and **real-time** streaming (for
second-level alerts). Buy it only if the edge survives the 2-year backtests.

## Backtesting a rule change

```
python scripts/backtest.py                      # dashboard watchlist, ~2 min
python scripts/backtest.py --tickers AAPL NVDA  # a subset
python scripts/backtest.py --csv trades.csv     # dump the live variant's trades
```

The replay walks every completed 5m bar through `scanner.analyze_ticker`, the
function the live scan calls. Hourly and daily bars only become visible once
complete, so there is no look-ahead. It resolves each trade against the bars
that followed, flat at the session close, net of estimated cost, at 1.0R, 1.5R
and 2.0R targets, and reports each variant for the whole window and for each half.

**Judge on avg R, not win rate.** A 1.5R bracket breaks even at a 40% win rate and
a 1.0R bracket at 50%. Moving the target trades win rate for payoff without
changing the edge. A change is worth keeping only if avg R improves in *both*
halves.

### Why the pullback gate

In September 2026 the system was reviewed against its own records:

- **Live forward test, 1,645 trades (Jul–Sep):** 42.5% win rate, −0.06R/trade,
  about −100R in total. Nothing separated winners from losers: every logged
  feature, the total score and the signal tier had AUC 0.46–0.52, and the shadow
  model scored 0.48. The little signal there was pointed *against* the momentum
  the score rewards. Entries stretched above EMA20, with high RSI, or near the day
  high lost; entries on intraday weakness won.
- **Replay, 60 sessions × 34 tickers:** reproduced the live baseline (−0.063R
  over 3,160 trades). Entering only from the adverse quarter of the day's range:

  | 1.5R target | trades | win rate | avg R | 1st half | 2nd half |
  |---|---|---|---|---|---|
  | rules before the gate | 3,160 | 44.4% | −0.063 | −0.090 | −0.036 |
  | with pullback gate | 554 | 51.8% | **+0.080** | +0.037 | +0.122 |

  The gate was positive at every target size and across all 34 tickers, and the
  matching live bucket was +0.054R over 200 trades.
- **Two-year check (Polygon, Nov 2024–Sep 2026, same 34 tickers): the edge did
  not hold.**

  | 1.5R target | trades | win rate | avg R | 1st half | 2nd half |
  |---|---|---|---|---|---|
  | rules before the gate | 24,572 | 44.8% | −0.046 ± 0.006 | −0.063 | −0.029 |
  | with pullback gate | 4,381 | 46.8% | −0.010 ± 0.015 | −0.063 | +0.043 |
  | pullback gate + opt-in SPY>50d regime gate | 3,157 | 48.0% | +0.017 ± 0.017 | −0.042 | +0.077 |

  The opt-in regime gate (`STOCKS_MARKET_REGIME_GATE=1`) skips trades while SPY
  closed below its 50-day average. There, longs lost −0.069R and shorts −0.085R
  across 24,572 ungated trades, worse in both halves. It is a loss filter, not
  an edge: the combined row is within one standard error of zero.

  By quarter, the gated rules lost money in all five quarters from 2024 Q4 to
  2025 Q4 and made money in the last three (+0.02, +0.04, +0.09R). The 60-day
  window above sat in the best one.

**What this means:**
- The rules are not a proven money-maker, with or without the gate. Treat alerts
  as a screening aid, not trade instructions, and don't size real positions off
  them.
- The gate stays on because it loses about 4× less than the ungated rules, on
  about 6× fewer trades.
- Any future rule change should be judged on the 2-year `--source polygon`
  backtest, quarter by quarter, never on a 60-day window alone.

## Ranking (long/short) research

```
python scripts/rank_backtest.py                       # NASDAQ-100, 4 signals x 4 horizons, ~1.5 min
python scripts/rank_backtest.py --universe watchlist  # your dashboard watchlist instead
python scripts/rank_backtest.py --csv book.csv --signal reversal_60m --horizon 24
```

This is how systematic funds frame the problem. The scanner asks "will AAPL go
up?", which is mostly a question about the market. The ranking backtest asks
"will the top of the list beat the bottom?". At each rebalance it ranks the
universe on a signal, buys the top 20% and shorts the bottom 20% in equal
dollars, so the market move cancels out. It holds for 30m, 1h or 2h (or from
11:00 to the close) and charges estimated round-trip cost on both legs. It
reports:
- **IC**: the rank correlation between signal and next return. This is the
  number quant desks track; 0.02–0.05 that holds up is a good signal.
- **Net bps per rebalance.**
- **Breakeven cost**: the round-trip cost per name at which the signal stops
  paying.
- **Hit rate and day hit rate**: the portfolio version of "accuracy".

**First result (Sep 2026, 98 names, 60 sessions):**

- **Short-term reversal is real but costs eat it.** Ranking by the last
  30–60 minutes' vol-scaled return (buy the laggards, short the leaders) has
  IC +0.02 to +0.05, t ≈ 3, in every horizon from 30m to 2h. That is genuine,
  fund-grade stock-specific predictability.
- The gross spread is only 1–4 bps per name, though, at or below the estimated
  ~3 bps round trip. Net, it is about −4 bps at 30m and roughly breakeven at 2h
  (reversal_60m: +1.9 bps, t = 0.5; 51% hit rate, 55% of days up).
- **Day-level reversal and range position have no cross-sectional edge.** The
  scanner's pullback gate is a different signal: it only buys pullbacks inside
  a trend the rules have already identified. A beta decomposition of its
  replayed trades attributes +0.065R of its +0.080R to the stock, not the
  market.
- **The deciding number is real execution cost**, and yfinance can't measure
  it: there's no bid/ask. For mega caps the true spread is often about 1 bp,
  which would make the 2h reversal book profitable. Cross the spread at 3 bps
  and it isn't. Next steps: a quote feed (Polygon/Alpaca/Databento) to measure
  cost, and years of 1m history to confirm IC beyond one 60-day regime.
- Portfolio hit rates top out around 50–55%, as expected; there's no 60–70%
  anywhere in the table.

**Two-year result (Polygon, Sep 2024–Sep 2026, 99 names, 499 sessions): no
tradable edge.** The reversal effect is real but small. For 30m reversal held
30m, IC is +0.015 (t = 5.9 over 5,432 rebalances), but the gross long/short
spread is only +0.6 bps, a breakeven cost of 0.3 bps per name round trip, well
below even a mega-cap spread. Every signal × horizon lost money after costs in
every third of the window. The stronger 60-day numbers above were a favourable
stretch. Measuring real spreads (Polygon Advanced) cannot rescue a 0.3 bps edge,
so it is not worth buying for this.

Caveats: the universe is today's members (survivorship bias), ~16
signal × horizon tests were run, and nothing here is wired into live alerts
until a variant clears costs in every third of a longer sample.

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
5. **Shadow.** Before gating on it, run the candidate in **shadow**: it scores every
   directional setup and logs the probability, but vetoes nothing. This is the only
   way to learn what it would do to the trades it wants to block — once it is gating,
   those trades stop happening and stop being measurable. `scripts/model_report.py`
   then judges it on real resolved outcomes rather than on its own training split.
   Tracked rows record which mode produced their probability (`model_mode`), because
   gated rows are a censored sample and must never be pooled with shadow rows.
6. **Serve.** The active model scores each setup live. Below the cost-adjusted
   breakeven win rate plus a margin, an otherwise-actionable signal is downgraded
   to WATCH_ONLY with the reason shown. Only one model gates and at most one shadows;
   whenever something is gating, the shadow lane is ignored.

```
python scripts/train_model.py              # evaluate + save a candidate, do not activate
python scripts/train_model.py --activate   # promote it, if it clears the gate
python scripts/model_report.py             # judge a shadow model on resolved trades
python scripts/model_report.py --mode any  # include rows armed before mode tracking
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

- `STOCKS_ALERT_WEBHOOK_URL` — where to push new-signal alerts (see [Real-time alerts](#real-time-alerts)).
- `STOCKS_DB_PATH` — SQLite location. Defaults to `%LOCALAPPDATA%\StocksRealtimeScreening\stocks_screening.sqlite3`, deliberately outside synced folders (OneDrive + SQLite WAL files cause lock/sync conflicts).

## Project layout

```
app.py                    Streamlit UI (Scanner + Live Quotes pages; Model tab)
stocks/
  yf_client.py            Batched yfinance data client (forming-candle drop, TTL caches)
  scanner.py              Scan orchestration: fetch → analyze_ticker (pure, replayable) → persist → forward-test
  backtest.py             Historical replay of analyze_ticker, trade resolution, variant simulation
  cross_section.py        Vectorised long/short ranking backtest (signals, IC, costs)
  history.py              Cached history for both backtesters (yfinance 60d, Polygon 2y resumable)
  polygon_client.py       Polygon/Massive aggregates client: rate limit, pagination, Yahoo-aligned resampling
  universe.py             NASDAQ-100 approximation for ranking research
  alerts.py               Push alerts (ntfy / Discord / Slack / generic webhook) for newly armed signals
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
  model_report.py         Judge a shadow/active model on live resolved trades
  backfill_links.py       One-off link of legacy outcome rows to their tracking rows
  backtest.py             Replay ~60 sessions and compare rule variants
  download_history.py     Fill / top up the Polygon 2-year cache (watchlist first)
  rank_backtest.py        Rank the universe, long top / short bottom, report IC and net bps
  run_alerts.py           Headless scanner: scan on each 5m bar close and push new signals
tests/                    pytest suite (206 tests, all offline except one localhost check)
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
- The live forward tracker holds a trade for up to 8 wall-clock hours, often overnight, while the backtester exits at the session close. Live and replayed results are therefore close but not identical
- The backtester measures the rules only; a trained model's veto is not replayed
- Keep watchlists under ~50 tickers to stay friendly with Yahoo rate limits

> Not financial advice. Signals are heuristics for screening, not trade recommendations.
