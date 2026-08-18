from __future__ import annotations
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from .models import StockSnapshot, StockQuote, ScanSummary
from .timeutil import parse_ts

SQLITE_TIMEOUT = 30.0
SQLITE_BUSY_MS = 30000
MAX_SCAN_RUNS = 20


class Storage:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=SQLITE_TIMEOUT)
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA busy_timeout = {SQLITE_BUSY_MS}")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS stock_scan_runs (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at  TEXT    DEFAULT CURRENT_TIMESTAMP,
                    finished_at TEXT,
                    tickers_scanned INTEGER,
                    summary_json TEXT
                );

                CREATE TABLE IF NOT EXISTS stock_scan_logs (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    scan_id    INTEGER,
                    ticker     TEXT,
                    signal     TEXT,
                    error      TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS stock_snapshots (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    scan_id         INTEGER NOT NULL,
                    ticker          TEXT,
                    last            REAL,
                    prev_close      REAL,
                    avg_dollar_volume REAL,
                    open            REAL,
                    high            REAL,
                    low             REAL,
                    close           REAL,
                    day_change_pct  REAL,
                    rsi14           REAL,
                    ema9            REAL,
                    ema20           REAL,
                    ema50           REAL,
                    macd            REAL,
                    macd_signal     REAL,
                    macd_histogram  REAL,
                    atr14           REAL,
                    adx14           REAL,
                    bb_upper        REAL,
                    bb_middle       REAL,
                    bb_lower        REAL,
                    bb_width_pct    REAL,
                    market_phase    TEXT,
                    day_high        REAL,
                    day_low         REAL,
                    or_high         REAL,
                    or_low          REAL,
                    momentum_score  REAL,
                    reversion_score REAL,
                    breakout_score  REAL,
                    regime          TEXT,
                    total_score     REAL,
                    trade_signal    TEXT,
                    signal_reason   TEXT,
                    risk_notes      TEXT,
                    as_of           TEXT,
                    suggested_entry REAL,
                    suggested_stop  REAL,
                    suggested_target REAL,
                    stop_dollars    REAL,
                    target_dollars  REAL,
                    stop_pct        REAL,
                    rr_ratio        REAL,
                    hourly_direction TEXT,
                    daily_direction TEXT,
                    mtf_score       REAL DEFAULT 0,
                    mtf_confluence  TEXT,
                    nearest_support REAL,
                    nearest_resistance REAL,
                    sr_score        REAL DEFAULT 0,
                    at_key_level    INTEGER DEFAULT 0,
                    sr_levels_json  TEXT,
                    rs_vs_spy       REAL,
                    spy_change_pct  REAL,
                    rs_assessment   TEXT
                );

                CREATE TABLE IF NOT EXISTS stock_quotes (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker      TEXT,
                    last        REAL,
                    prev_close  REAL,
                    change_pct  REAL,
                    volume      INTEGER,
                    as_of       TEXT
                );

                CREATE TABLE IF NOT EXISTS stock_watchlist (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker        TEXT NOT NULL,
                    signal        TEXT,
                    entry_price   REAL,
                    target_price  REAL,
                    stop_price    REAL,
                    stop_dollars  REAL,
                    target_dollars REAL,
                    notes         TEXT,
                    status        TEXT DEFAULT 'watching',
                    created_at    TEXT DEFAULT CURRENT_TIMESTAMP,
                    closed_at     TEXT
                );

                CREATE TABLE IF NOT EXISTS stock_trade_outcomes (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    watchlist_id INTEGER NOT NULL,
                    ticker       TEXT,
                    signal       TEXT,
                    entry_price  REAL,
                    exit_price   REAL,
                    exit_dollars REAL,
                    exit_pct     REAL,
                    r_multiple   REAL,
                    outcome      TEXT,
                    hold_minutes INTEGER,
                    created_at   TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS stock_signal_tracking (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker        TEXT NOT NULL,
                    signal        TEXT,
                    direction     INTEGER,
                    entry_price   REAL,
                    stop_price    REAL,
                    target_price  REAL,
                    stop_dollars  REAL,
                    target_dollars REAL,
                    atr14         REAL,
                    entry_ts      TEXT,
                    status        TEXT DEFAULT 'open',
                    created_at    TEXT DEFAULT CURRENT_TIMESTAMP
                );

                -- Manual paper trading. Deliberately separate from
                -- stock_trade_outcomes: that table is the model's forward-test
                -- record, and the Performance tab counts every row in it as a
                -- prediction. User trades must never inflate those stats.
                CREATE TABLE IF NOT EXISTS stock_paper_account (
                    id               INTEGER PRIMARY KEY CHECK (id = 1),
                    starting_equity  REAL NOT NULL DEFAULT 100000.0,
                    created_at       TEXT DEFAULT CURRENT_TIMESTAMP,
                    reset_at         TEXT
                );

                CREATE TABLE IF NOT EXISTS stock_paper_positions (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker        TEXT NOT NULL,
                    direction     INTEGER NOT NULL,      -- +1 long / -1 short
                    qty           INTEGER NOT NULL,      -- open shares; 0 once fully closed
                    avg_entry     REAL NOT NULL,         -- weighted average cost
                    stop_price    REAL,
                    target_price  REAL,
                    realized_pnl  REAL DEFAULT 0.0,      -- accumulates across partial closes
                    signal        TEXT,                  -- trade_signal at open, for later review
                    notes         TEXT,
                    status        TEXT DEFAULT 'open',   -- 'open' | 'closed'
                    opened_at     TEXT DEFAULT CURRENT_TIMESTAMP,
                    closed_at     TEXT
                );

                -- Every fill is kept, so a position row is a running aggregate
                -- that can always be re-derived from its own history.
                CREATE TABLE IF NOT EXISTS stock_paper_fills (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    position_id  INTEGER NOT NULL,
                    kind         TEXT NOT NULL,          -- 'OPEN' | 'ADD' | 'CLOSE'
                    side         TEXT NOT NULL,          -- 'BUY' | 'SELL'
                    qty          INTEGER NOT NULL,
                    price        REAL NOT NULL,
                    realized_pnl REAL,                   -- NULL on OPEN/ADD
                    notes        TEXT,
                    fill_ts      TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS stock_performance_stats (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    computed_at     TEXT DEFAULT CURRENT_TIMESTAMP,
                    dimension       TEXT,
                    dimension_value TEXT,
                    trades          INTEGER,
                    wins            INTEGER,
                    win_rate        REAL,
                    avg_r           REAL,
                    expectancy      REAL
                );

                -- Trained direction models. Coefficients are stored inline so the
                -- scanner can serve predictions without a model file on disk, and so
                -- every historical model stays auditable against its own metrics.
                CREATE TABLE IF NOT EXISTS stock_models (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at       TEXT DEFAULT CURRENT_TIMESTAMP,
                    feature_version  INTEGER,
                    algo             TEXT,
                    n_train          INTEGER,
                    n_test           INTEGER,
                    auc              REAL,
                    brier            REAL,
                    top_decile_prec  REAL,
                    base_rate        REAL,
                    model_json       TEXT,
                    metrics_json     TEXT,
                    is_active        INTEGER DEFAULT 0,
                    notes            TEXT
                );
            """)

        # Additive column migrations, safe to re-run on every startup.
        migrations = {
            "stock_snapshots": [
                ("blocked_ahead", "INTEGER DEFAULT 0"),
                ("rel_volume", "REAL"),
                ("extension_atr", "REAL"),
                ("cost_pct", "REAL"),
                ("cost_ratio", "REAL"),
                ("model_prob", "REAL"),
                ("required_prob", "REAL"),
            ],
            # The feature snapshot at arm time. Without this the outcome rows are
            # unlearnable — every input was previously discarded the moment the
            # snapshot table was pruned, which is what blocked any model work.
            "stock_signal_tracking": [
                ("features_json", "TEXT"),
                ("feature_version", "INTEGER"),
                ("model_prob", "REAL"),
                ("required_prob", "REAL"),
                ("cost_pct", "REAL"),
                ("cost_ratio", "REAL"),
                ("total_score", "REAL"),
                ("adx14", "REAL"),
                ("regime", "TEXT"),
                ("market_phase", "TEXT"),
                ("rs_vs_spy", "REAL"),
                # Which mode produced model_prob. Rows scored by a gating model are a
                # censored sample (only trades it allowed exist), so they cannot be
                # pooled with shadow rows when judging the model.
                ("model_mode", "TEXT"),
            ],
            # Shadow mode: a model scores every setup and logs its probability but
            # never vetoes. An active model only ever gets outcomes for the trades it
            # allowed, so its own gate censors the evidence needed to judge it.
            "stock_models": [
                ("is_shadow", "INTEGER DEFAULT 0"),
            ],
            # tracking_id closes the loop: an outcome can now be joined back to the
            # exact feature vector that produced it.
            "stock_trade_outcomes": [
                ("tracking_id", "INTEGER"),
                ("gross_dollars", "REAL"),
                ("cost_dollars", "REAL"),
                ("net_dollars", "REAL"),
                ("exit_ts", "TEXT"),
                ("exit_reason", "TEXT"),
            ],
        }
        for table, cols in migrations.items():
            for col, typedef in cols:
                try:
                    with self._connect() as conn:
                        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typedef}")
                except sqlite3.OperationalError:
                    pass  # column already exists

        # Indexes go last — they reference columns added by the migrations above.
        with self._connect() as conn:
            conn.executescript("""
                CREATE INDEX IF NOT EXISTS idx_stock_tracking_status
                    ON stock_signal_tracking(status, ticker);
                CREATE INDEX IF NOT EXISTS idx_stock_outcomes_tracking
                    ON stock_trade_outcomes(tracking_id);
                CREATE INDEX IF NOT EXISTS idx_stock_models_active
                    ON stock_models(is_active, created_at);
                CREATE INDEX IF NOT EXISTS idx_stock_paper_pos_status
                    ON stock_paper_positions(status, ticker);
                CREATE INDEX IF NOT EXISTS idx_stock_paper_fills_pos
                    ON stock_paper_fills(position_id);
            """)

    # ── Scan run lifecycle ──────────────────────────────────────────────────

    def start_scan(self) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO stock_scan_runs (started_at) VALUES (CURRENT_TIMESTAMP)"
            )
            return cur.lastrowid

    def finish_scan(self, scan_id: int, summary: ScanSummary) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE stock_scan_runs SET finished_at=CURRENT_TIMESTAMP, "
                "tickers_scanned=?, summary_json=? WHERE id=?",
                (summary.tickers_scanned, summary.model_dump_json(), scan_id),
            )
            # Prune old runs
            conn.execute(
                "DELETE FROM stock_snapshots WHERE scan_id NOT IN "
                f"(SELECT id FROM stock_scan_runs ORDER BY id DESC LIMIT {MAX_SCAN_RUNS})"
            )
            conn.execute(
                "DELETE FROM stock_scan_logs WHERE scan_id NOT IN "
                f"(SELECT id FROM stock_scan_runs ORDER BY id DESC LIMIT {MAX_SCAN_RUNS})"
            )
            conn.execute(
                f"DELETE FROM stock_scan_runs WHERE id NOT IN "
                f"(SELECT id FROM stock_scan_runs ORDER BY id DESC LIMIT {MAX_SCAN_RUNS})"
            )

    def log_ticker(self, scan_id: int, ticker: str, signal: Optional[str], error: Optional[str]) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO stock_scan_logs (scan_id, ticker, signal, error) VALUES (?,?,?,?)",
                (scan_id, ticker, signal, error),
            )

    # ── Snapshots ───────────────────────────────────────────────────────────

    def save_snapshots(self, scan_id: int, snapshots: List[StockSnapshot]) -> None:
        rows = []
        for s in snapshots:
            rows.append((
                scan_id, s.ticker, s.last, s.prev_close, s.avg_dollar_volume,
                s.open, s.high, s.low, s.close, s.day_change_pct,
                s.rsi14, s.ema9, s.ema20, s.ema50,
                s.macd, s.macd_signal, s.macd_histogram,
                s.atr14, s.adx14, s.bb_upper, s.bb_middle, s.bb_lower, s.bb_width_pct,
                s.market_phase, s.day_high, s.day_low, s.or_high, s.or_low,
                s.momentum_score, s.reversion_score, s.breakout_score, s.regime,
                s.total_score, s.trade_signal, s.signal_reason, s.risk_notes, s.as_of,
                s.suggested_entry, s.suggested_stop, s.suggested_target,
                s.stop_dollars, s.target_dollars, s.stop_pct, s.rr_ratio,
                s.hourly_direction, s.daily_direction, s.mtf_score, s.mtf_confluence,
                s.nearest_support, s.nearest_resistance, s.sr_score,
                int(s.at_key_level), s.sr_levels_json,
                s.rs_vs_spy, s.spy_change_pct, s.rs_assessment,
                # Structure / volume / cost / model gating
                int(s.blocked_ahead), s.rel_volume, s.extension_atr,
                s.cost_pct, s.cost_ratio, s.model_prob, s.required_prob,
            ))
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO stock_snapshots "
                "(scan_id,ticker,last,prev_close,avg_dollar_volume,"
                "open,high,low,close,day_change_pct,"
                "rsi14,ema9,ema20,ema50,macd,macd_signal,macd_histogram,"
                "atr14,adx14,bb_upper,bb_middle,bb_lower,bb_width_pct,"
                "market_phase,day_high,day_low,or_high,or_low,"
                "momentum_score,reversion_score,breakout_score,regime,"
                "total_score,trade_signal,signal_reason,risk_notes,as_of,"
                "suggested_entry,suggested_stop,suggested_target,"
                "stop_dollars,target_dollars,stop_pct,rr_ratio,"
                "hourly_direction,daily_direction,mtf_score,mtf_confluence,"
                "nearest_support,nearest_resistance,sr_score,at_key_level,sr_levels_json,"
                "rs_vs_spy,spy_change_pct,rs_assessment,"
                "blocked_ahead,rel_volume,extension_atr,"
                "cost_pct,cost_ratio,model_prob,required_prob) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,"
                "?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,"
                "?,?,?,?,?,?,?)",
                rows,
            )

    def load_latest_snapshots(self) -> list:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM stock_snapshots WHERE scan_id = "
                "(SELECT MAX(id) FROM stock_scan_runs WHERE finished_at IS NOT NULL) "
                "ORDER BY total_score DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def load_scan_logs(self) -> list:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM stock_scan_logs WHERE scan_id = "
                "(SELECT MAX(id) FROM stock_scan_runs WHERE finished_at IS NOT NULL) "
                "ORDER BY created_at"
            ).fetchall()
        return [dict(r) for r in rows]

    def load_latest_scan_run(self) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM stock_scan_runs ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return dict(row) if row else None

    # ── Live quotes ─────────────────────────────────────────────────────────

    def save_quotes(self, quotes: List[StockQuote]) -> None:
        rows = [(q.ticker, q.last, q.prev_close, q.change_pct, q.volume, q.as_of) for q in quotes]
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO stock_quotes (ticker,last,prev_close,change_pct,volume,as_of) "
                "VALUES (?,?,?,?,?,?)",
                rows,
            )
            # Keep only last 500 rows
            conn.execute(
                "DELETE FROM stock_quotes WHERE id NOT IN "
                "(SELECT id FROM stock_quotes ORDER BY id DESC LIMIT 500)"
            )

    def load_latest_quotes(self) -> list:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT q.* FROM stock_quotes q "
                "INNER JOIN ("
                "  SELECT ticker, MAX(id) AS max_id FROM stock_quotes GROUP BY ticker"
                ") latest ON q.ticker=latest.ticker AND q.id=latest.max_id "
                "ORDER BY q.ticker"
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Watchlist ────────────────────────────────────────────────────────────

    def add_watchlist(
        self, ticker: str, signal: str, entry: float,
        target: float, stop: float, stop_dollars: float,
        target_dollars: float, notes: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO stock_watchlist "
                "(ticker,signal,entry_price,target_price,stop_price,stop_dollars,target_dollars,notes) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (ticker, signal, entry, target, stop, stop_dollars, target_dollars, notes),
            )

    def close_watchlist(self, row_id: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE stock_watchlist SET status='closed', closed_at=CURRENT_TIMESTAMP WHERE id=?",
                (row_id,),
            )

    def close_watchlist_with_outcome(self, row_id: int, exit_price: float) -> None:
        """Close a watchlist entry, compute P&L, and record the trade outcome."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM stock_watchlist WHERE id=?", (row_id,)
            ).fetchone()
        if not row:
            return
        row = dict(row)

        ticker = row.get("ticker", "")
        signal = row.get("signal") or ""
        entry_price = row.get("entry_price") or 0.0
        stop_dollars = row.get("stop_dollars") or 0.0
        created_at = row.get("created_at") or ""

        direction = -1 if signal in ("STRONG_SHORT", "SHORT_CANDIDATE") else 1
        exit_dollars = round((exit_price - entry_price) * direction, 4) if entry_price else 0.0
        exit_pct = round(exit_dollars / entry_price * 100, 3) if entry_price else 0.0
        r_multiple = round(exit_dollars / stop_dollars, 2) if stop_dollars and stop_dollars > 0 else None
        outcome = "WIN" if exit_dollars > 0 else ("LOSS" if exit_dollars < 0 else "BREAKEVEN")

        hold_minutes: Optional[int] = None
        try:
            created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            hold_minutes = int((datetime.now(timezone.utc) - created).total_seconds() / 60)
        except Exception:
            pass

        with self._connect() as conn:
            conn.execute(
                "INSERT INTO stock_trade_outcomes "
                "(watchlist_id,ticker,signal,entry_price,exit_price,exit_dollars,exit_pct,"
                "r_multiple,outcome,hold_minutes) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (row_id, ticker, signal, entry_price, exit_price, exit_dollars, exit_pct,
                 r_multiple, outcome, hold_minutes),
            )
            conn.execute(
                "UPDATE stock_watchlist SET status='closed', closed_at=CURRENT_TIMESTAMP WHERE id=?",
                (row_id,),
            )
        self.compute_and_save_performance()

    def compute_and_save_performance(self) -> None:
        """Recompute and save aggregated performance stats from all trade outcomes."""
        with self._connect() as conn:
            all_rows = [dict(r) for r in conn.execute("SELECT * FROM stock_trade_outcomes").fetchall()]
        if not all_rows:
            return

        def _stats(rows: list) -> Optional[dict]:
            if not rows:
                return None
            wins = sum(1 for r in rows if r["outcome"] == "WIN")
            n = len(rows)
            win_rate = round(wins / n, 3)
            # avg_r = mean R of winners; expectancy = mean R across ALL trades (per-trade edge)
            win_rs = [r["r_multiple"] for r in rows if r["outcome"] == "WIN" and r["r_multiple"] is not None]
            all_rs = [r["r_multiple"] for r in rows if r["r_multiple"] is not None]
            avg_r = round(sum(win_rs) / len(win_rs), 2) if win_rs else 0.0
            expectancy = round(sum(all_rs) / len(all_rs), 3) if all_rs else 0.0
            return {"trades": n, "wins": wins, "win_rate": win_rate, "avg_r": avg_r, "expectancy": expectancy}

        insert_rows = []
        for dimension, key in [("ticker", "ticker"), ("signal", "signal")]:
            for value in set(r[key] for r in all_rows if r.get(key)):
                subset = [r for r in all_rows if r.get(key) == value]
                s = _stats(subset)
                if s:
                    insert_rows.append((dimension, value, s["trades"], s["wins"], s["win_rate"], s["avg_r"], s["expectancy"]))
        s = _stats(all_rows)
        if s:
            insert_rows.append(("overall", "all", s["trades"], s["wins"], s["win_rate"], s["avg_r"], s["expectancy"]))

        with self._connect() as conn:
            conn.execute("DELETE FROM stock_performance_stats")
            conn.executemany(
                "INSERT INTO stock_performance_stats "
                "(dimension,dimension_value,trades,wins,win_rate,avg_r,expectancy) VALUES (?,?,?,?,?,?,?)",
                insert_rows,
            )

    def load_trade_outcomes(self, limit: int = 200) -> list:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM stock_trade_outcomes ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def load_performance_by_dimension(self, dimension: str) -> list:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM stock_performance_stats WHERE dimension=? ORDER BY win_rate DESC",
                (dimension,),
            ).fetchall()
        return [dict(r) for r in rows]

    def load_watchlist(self, status: str = "watching") -> list:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM stock_watchlist WHERE status=? ORDER BY created_at DESC",
                (status,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Manual paper trading ────────────────────────────────────────────────
    #
    # Nothing here writes to stock_trade_outcomes or calls
    # compute_and_save_performance(). Those belong to the model's forward test;
    # mixing user trades in would corrupt its calibration record.

    DEFAULT_STARTING_EQUITY = 100000.0

    def paper_account(self) -> dict:
        """The single account row, created on first use."""
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO stock_paper_account (id, starting_equity) VALUES (1, ?)",
                (self.DEFAULT_STARTING_EQUITY,),
            )
            row = conn.execute("SELECT * FROM stock_paper_account WHERE id=1").fetchone()
            realized = conn.execute(
                "SELECT COALESCE(SUM(realized_pnl), 0.0) FROM stock_paper_positions"
            ).fetchone()[0]
        account = dict(row)
        account["realized_pnl"] = round(realized or 0.0, 2)
        account["equity"] = round(account["starting_equity"] + account["realized_pnl"], 2)
        return account

    def set_paper_starting_equity(self, starting_equity: float) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO stock_paper_account (id, starting_equity) VALUES (1, ?) "
                "ON CONFLICT(id) DO UPDATE SET starting_equity=excluded.starting_equity",
                (float(starting_equity),),
            )

    def reset_paper_account(self, starting_equity: float) -> None:
        """Wipe all paper positions and fills and restart from a fresh balance."""
        with self._connect() as conn:
            conn.execute("DELETE FROM stock_paper_fills")
            conn.execute("DELETE FROM stock_paper_positions")
            conn.execute(
                "INSERT INTO stock_paper_account (id, starting_equity, reset_at) "
                "VALUES (1, ?, CURRENT_TIMESTAMP) "
                "ON CONFLICT(id) DO UPDATE SET starting_equity=excluded.starting_equity, "
                "reset_at=CURRENT_TIMESTAMP",
                (float(starting_equity),),
            )

    def open_paper_position(
        self, ticker: str, direction: int, qty: int, price: float,
        stop: Optional[float] = None, target: Optional[float] = None,
        signal: str = "", notes: str = "",
    ) -> int:
        """Open a position and record its first fill. Returns the position id."""
        side = "BUY" if direction > 0 else "SELL"
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO stock_paper_positions "
                "(ticker,direction,qty,avg_entry,stop_price,target_price,signal,notes) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (ticker.upper(), int(direction), int(qty), float(price),
                 stop, target, signal, notes),
            )
            position_id = int(cur.lastrowid)
            conn.execute(
                "INSERT INTO stock_paper_fills (position_id,kind,side,qty,price,notes) "
                "VALUES (?,'OPEN',?,?,?,?)",
                (position_id, side, int(qty), float(price), notes),
            )
        return position_id

    def add_to_paper_position(
        self, position_id: int, qty: int, price: float, notes: str = "",
    ) -> Optional[dict]:
        """Scale into an open position, re-averaging the entry cost."""
        from .paper import scale_in

        position = self.load_paper_position(position_id)
        if not position or position.get("status") != "open":
            return None

        new_qty, new_avg = scale_in(
            int(position["qty"]), float(position["avg_entry"]), int(qty), float(price)
        )
        side = "BUY" if int(position["direction"]) > 0 else "SELL"
        with self._connect() as conn:
            conn.execute(
                "UPDATE stock_paper_positions SET qty=?, avg_entry=? WHERE id=?",
                (new_qty, new_avg, position_id),
            )
            conn.execute(
                "INSERT INTO stock_paper_fills (position_id,kind,side,qty,price,notes) "
                "VALUES (?,'ADD',?,?,?,?)",
                (position_id, side, int(qty), float(price), notes),
            )
        return self.load_paper_position(position_id)

    def close_paper_position(
        self, position_id: int, qty: int, price: float, notes: str = "",
    ) -> Optional[float]:
        """Close all or part of a position. The position only flips to 'closed'
        once the last share is out, so partial exits need no separate path.
        Returns the dollar P&L realized by this close."""
        from .paper import realized_pnl as _realized

        position = self.load_paper_position(position_id)
        if not position or position.get("status") != "open":
            return None

        open_qty = int(position["qty"])
        close_qty = min(int(qty), open_qty)
        if close_qty <= 0:
            return None

        direction = int(position["direction"])
        pnl = _realized(direction, float(position["avg_entry"]), float(price), close_qty)
        remaining = open_qty - close_qty
        # Closing a long is a sell; closing a short is a buy-to-cover.
        side = "SELL" if direction > 0 else "BUY"

        with self._connect() as conn:
            conn.execute(
                "INSERT INTO stock_paper_fills "
                "(position_id,kind,side,qty,price,realized_pnl,notes) "
                "VALUES (?,'CLOSE',?,?,?,?,?)",
                (position_id, side, close_qty, float(price), pnl, notes),
            )
            if remaining > 0:
                conn.execute(
                    "UPDATE stock_paper_positions "
                    "SET qty=?, realized_pnl=COALESCE(realized_pnl,0)+? WHERE id=?",
                    (remaining, pnl, position_id),
                )
            else:
                conn.execute(
                    "UPDATE stock_paper_positions "
                    "SET qty=0, realized_pnl=COALESCE(realized_pnl,0)+?, "
                    "status='closed', closed_at=CURRENT_TIMESTAMP WHERE id=?",
                    (pnl, position_id),
                )
        return pnl

    def load_paper_position(self, position_id: int) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM stock_paper_positions WHERE id=?", (position_id,)
            ).fetchone()
        return dict(row) if row else None

    def load_paper_positions(self, status: str = "open") -> list:
        """Positions by status; status='all' returns every row."""
        with self._connect() as conn:
            if status == "all":
                rows = conn.execute(
                    "SELECT * FROM stock_paper_positions ORDER BY opened_at DESC"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM stock_paper_positions WHERE status=? ORDER BY opened_at DESC",
                    (status,),
                ).fetchall()
        return [dict(r) for r in rows]

    def load_open_paper_position(self, ticker: str) -> Optional[dict]:
        """The open position for a ticker, if any — drives the Buy/Sell vs
        Add/Close branch in the scanner grid."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM stock_paper_positions WHERE ticker=? AND status='open' "
                "ORDER BY opened_at DESC LIMIT 1",
                (ticker.upper(),),
            ).fetchone()
        return dict(row) if row else None

    def load_paper_fills(self, position_id: Optional[int] = None, limit: int = 200) -> list:
        with self._connect() as conn:
            if position_id is None:
                rows = conn.execute(
                    "SELECT f.*, p.ticker FROM stock_paper_fills f "
                    "LEFT JOIN stock_paper_positions p ON p.id = f.position_id "
                    "ORDER BY f.fill_ts DESC, f.id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT f.*, p.ticker FROM stock_paper_fills f "
                    "LEFT JOIN stock_paper_positions p ON p.id = f.position_id "
                    "WHERE f.position_id=? ORDER BY f.fill_ts DESC, f.id DESC LIMIT ?",
                    (position_id, limit),
                ).fetchall()
        return [dict(r) for r in rows]

    # ── Automatic signal tracking / calibration ─────────────────────────────

    REARM_COOLDOWN_MINUTES = 45

    def record_tracked_signal(
        self, ticker: str, signal: str, direction: int,
        entry: float, stop: float, target: float,
        stop_dollars: float, target_dollars: float, atr14: float, entry_ts: str,
        features: Optional[dict] = None,
        feature_version: Optional[int] = None,
        model_prob: Optional[float] = None,
        required_prob: Optional[float] = None,
        cost_pct: Optional[float] = None,
        cost_ratio: Optional[float] = None,
        total_score: Optional[float] = None,
        adx14: Optional[float] = None,
        regime: Optional[str] = None,
        market_phase: Optional[str] = None,
        rs_vs_spy: Optional[float] = None,
        model_mode: Optional[str] = None,
    ) -> Optional[int]:
        """
        Record an actionable signal for hands-off forward evaluation. Skips if an
        open signal already exists for this ticker+direction (avoids re-arming every
        scan), or if one was armed within the cooldown window — without this, every
        scan after a stop-out immediately re-enters the same chop and racks up
        correlated losses.

        ``features`` is the model feature vector captured **at arm time**. Storing it
        here (rather than recomputing later from a snapshot that has since been pruned)
        is what makes the outcome learnable. Returns the new tracking id, or None if
        the signal was suppressed by the dedupe/cooldown rule.
        """
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT 1 FROM stock_signal_tracking "
                "WHERE ticker=? AND direction=? AND ("
                "  status='open' OR created_at >= datetime('now', ?)"
                ") LIMIT 1",
                (ticker, direction, f"-{self.REARM_COOLDOWN_MINUTES} minutes"),
            ).fetchone()
            if existing:
                return None
            cur = conn.execute(
                "INSERT INTO stock_signal_tracking "
                "(ticker,signal,direction,entry_price,stop_price,target_price,"
                "stop_dollars,target_dollars,atr14,entry_ts,"
                "features_json,feature_version,model_prob,required_prob,"
                "cost_pct,cost_ratio,total_score,adx14,regime,market_phase,rs_vs_spy,"
                "model_mode) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (ticker, signal, direction, entry, stop, target,
                 stop_dollars, target_dollars, atr14, entry_ts,
                 json.dumps(features) if features else None, feature_version,
                 model_prob, required_prob, cost_pct, cost_ratio,
                 total_score, adx14, regime, market_phase, rs_vs_spy,
                 model_mode),
            )
            return cur.lastrowid

    def evaluate_tracked_signals(
        self, ticker: str, bars: List[dict], max_hold_hours: float = 8.0,
    ) -> int:
        """
        Resolve open tracked signals for ``ticker`` against forward 5m bars: a stop or
        target touch closes the trade; otherwise it times out at the last close after
        ``max_hold_hours``. Resolved trades feed stock_trade_outcomes (the same table
        the Performance tab reads), so win-rate-by-signal calibrates itself over time.
        Note: max_hold uses wall-clock age, so a signal armed near the close typically
        times out on the next morning's first scan — acceptable for calibration.
        Returns the number of signals resolved.

        Accounting is net of estimated transaction cost. Bars are last-trade prices,
        so a bracket that touches its printed target still costs the spread to get in
        and out — ``r_multiple`` is therefore computed from ``net_dollars``. Reporting
        gross R overstates every result, and at these stop sizes the overstatement is
        several percent of an R.
        """
        with self._connect() as conn:
            open_rows = [dict(r) for r in conn.execute(
                "SELECT * FROM stock_signal_tracking WHERE ticker=? AND status='open'", (ticker,)
            ).fetchall()]
        if not open_rows:
            return 0

        now = datetime.now(timezone.utc)
        resolved = 0

        for row in open_rows:
            entry_ts = row.get("entry_ts") or ""
            direction = row.get("direction") or 1
            entry_price = row.get("entry_price") or 0.0
            stop = row.get("stop_price") or 0.0
            target = row.get("target_price") or 0.0
            stop_dollars = row.get("stop_dollars") or 0.0

            # Forward bars only (client emits one uniform UTC-ISO format → string order is valid)
            forward = [b for b in bars if b.get("timestamp", "") > entry_ts]

            exit_price: Optional[float] = None
            outcome: Optional[str] = None
            exit_ts: Optional[str] = None
            exit_reason: Optional[str] = None
            for b in forward:
                hi, lo = b["high"], b["low"]
                if direction == 1:
                    if lo <= stop:        # stop checked first = conservative
                        exit_price, outcome, exit_reason = stop, "LOSS", "STOP"
                        exit_ts = b.get("timestamp")
                        break
                    if hi >= target:
                        exit_price, outcome, exit_reason = target, "WIN", "TARGET"
                        exit_ts = b.get("timestamp")
                        break
                else:
                    if hi >= stop:
                        exit_price, outcome, exit_reason = stop, "LOSS", "STOP"
                        exit_ts = b.get("timestamp")
                        break
                    if lo <= target:
                        exit_price, outcome, exit_reason = target, "WIN", "TARGET"
                        exit_ts = b.get("timestamp")
                        break

            created = self._parse_dt(row.get("created_at"))
            if outcome is None:
                # Timeout: close at last available close once held longer than max_hold
                aged_out = created is not None and (now - created).total_seconds() > max_hold_hours * 3600
                if aged_out and forward:
                    exit_price = forward[-1]["close"]
                    exit_ts = forward[-1].get("timestamp")
                    exit_reason = "TIMEOUT"
                else:
                    continue  # still live

            gross_dollars = round((exit_price - entry_price) * direction, 4) if entry_price else 0.0
            # Round-trip cost in dollars per share, from the cost tier recorded when
            # the signal was armed. Falls back to zero only for legacy rows that
            # predate cost logging.
            cost_dollars = round((row.get("cost_pct") or 0.0) / 100.0 * entry_price, 4) if entry_price else 0.0
            net_dollars = round(gross_dollars - cost_dollars, 4)
            if exit_reason == "TIMEOUT":
                outcome = "WIN" if net_dollars > 0 else ("LOSS" if net_dollars < 0 else "BREAKEVEN")
            exit_pct = round(net_dollars / entry_price * 100, 3) if entry_price else 0.0
            r_multiple = round(net_dollars / stop_dollars, 2) if stop_dollars and stop_dollars > 0 else None

            # Real trade duration: entry bar → resolving bar. Measuring this as
            # (now − created_at) instead reports how long until a scan happened to
            # evaluate the row, which makes wins and losses look identically long.
            entry_dt = self._parse_dt(entry_ts) or created
            exit_dt = self._parse_dt(exit_ts)
            if entry_dt and exit_dt:
                hold_minutes = max(0, int((exit_dt - entry_dt).total_seconds() / 60))
            else:
                hold_minutes = None

            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO stock_trade_outcomes "
                    "(watchlist_id,tracking_id,ticker,signal,entry_price,exit_price,"
                    "exit_dollars,exit_pct,gross_dollars,cost_dollars,net_dollars,"
                    "r_multiple,outcome,hold_minutes,exit_ts,exit_reason) "
                    "VALUES (0,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (row["id"], ticker, row.get("signal"), entry_price, exit_price,
                     net_dollars, exit_pct, gross_dollars, cost_dollars, net_dollars,
                     r_multiple, outcome, hold_minutes, exit_ts, exit_reason),
                )
                conn.execute(
                    "UPDATE stock_signal_tracking SET status='closed' WHERE id=?",
                    (row["id"],),
                )
            resolved += 1

        if resolved:
            self.compute_and_save_performance()
        return resolved

    _parse_dt = staticmethod(parse_ts)

    def load_tracked_signals(self, status: str = "open", limit: int = 200) -> list:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM stock_signal_tracking WHERE status=? "
                "ORDER BY created_at DESC LIMIT ?", (status, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Model store / training data ─────────────────────────────────────────

    def load_training_rows(self, feature_version: Optional[int] = None) -> list:
        """
        Resolved trades joined back to the feature vector captured at arm time.

        Only rows with a stored ``features_json`` are usable — trades recorded before
        feature logging existed are unlearnable and are excluded here rather than
        silently imputed, which would teach the model from fabricated inputs.
        Ordered oldest-first so a walk-forward split is just an index cut.
        """
        sql = (
            "SELECT t.id AS tracking_id, t.features_json, t.feature_version, t.ticker, "
            "       t.direction, t.signal, t.created_at, t.model_prob, "
            "       o.outcome, o.r_multiple, o.net_dollars, o.exit_reason "
            "FROM stock_signal_tracking t "
            "JOIN stock_trade_outcomes o ON o.tracking_id = t.id "
            "WHERE t.features_json IS NOT NULL AND o.outcome IS NOT NULL "
        )
        params: list = []
        if feature_version is not None:
            sql += "AND t.feature_version = ? "
            params.append(feature_version)
        sql += "ORDER BY t.created_at ASC, t.id ASC"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        out = []
        for r in rows:
            row = dict(r)
            try:
                row["features"] = json.loads(row.pop("features_json") or "{}")
            except (ValueError, TypeError):
                continue
            if not row["features"]:
                continue
            out.append(row)
        return out

    def save_model(self, model_json: str, metrics: dict, activate: bool = True,
                   notes: str = "") -> int:
        """Persist a trained model and optionally make it the one the scanner serves."""
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO stock_models "
                "(feature_version,algo,n_train,n_test,auc,brier,top_decile_prec,"
                " base_rate,model_json,metrics_json,is_active,notes) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,0,?)",
                (
                    metrics.get("feature_version"), metrics.get("algo"),
                    metrics.get("n_train"), metrics.get("n_test"),
                    metrics.get("auc"), metrics.get("brier"),
                    metrics.get("top_decile_prec"), metrics.get("base_rate"),
                    model_json, json.dumps(metrics), notes,
                ),
            )
            model_id = cur.lastrowid
            if activate:
                conn.execute("UPDATE stock_models SET is_active=0")
                conn.execute("UPDATE stock_models SET is_active=1 WHERE id=?", (model_id,))
        return model_id

    def activate_model(self, model_id: int) -> bool:
        """
        Promote one stored model to active, deactivating every other.

        Also used for rollback — promoting an older model id is a valid recovery
        path when a newly promoted one turns out to behave badly live.
        Returns False if the id does not exist.
        """
        with self._connect() as conn:
            exists = conn.execute(
                "SELECT 1 FROM stock_models WHERE id=?", (model_id,)
            ).fetchone()
            if not exists:
                return False
            conn.execute("UPDATE stock_models SET is_active=0")
            # A model cannot shadow and gate at once: shadow numbers are only an
            # honest preview while the model has no influence on what gets traded.
            conn.execute(
                "UPDATE stock_models SET is_active=1, is_shadow=0 WHERE id=?", (model_id,)
            )
        return True

    def deactivate_all_models(self) -> None:
        """Fall back to rules-only scanning without deleting any model."""
        with self._connect() as conn:
            conn.execute("UPDATE stock_models SET is_active=0")

    def shadow_model(self, model_id: int) -> bool:
        """
        Run one model in shadow: scored and logged on every setup, never vetoing.

        This is how a candidate earns promotion. An active model only ever sees
        outcomes for trades it let through, so its live win rate is measured on a
        censored sample and cannot say what the trades it blocked would have done.
        A shadow model is scored on every directional setup the rules propose, so
        its probabilities land on winners and losers alike and stay comparable.
        Returns False if the id does not exist.
        """
        with self._connect() as conn:
            exists = conn.execute(
                "SELECT 1 FROM stock_models WHERE id=?", (model_id,)
            ).fetchone()
            if not exists:
                return False
            conn.execute("UPDATE stock_models SET is_shadow=0")
            conn.execute(
                "UPDATE stock_models SET is_shadow=1, is_active=0 WHERE id=?", (model_id,)
            )
        return True

    def clear_shadow_model(self) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE stock_models SET is_shadow=0")

    def load_active_model_json(self) -> Optional[str]:
        return self._load_model_json("is_active")

    def load_shadow_model_json(self) -> Optional[str]:
        return self._load_model_json("is_shadow")

    def _load_model_json(self, flag: str) -> Optional[str]:
        if flag not in ("is_active", "is_shadow"):      # guards the interpolation
            raise ValueError(f"unknown model flag: {flag}")
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT model_json FROM stock_models WHERE {flag}=1 "
                "ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        return row["model_json"] if row else None

    def load_models(self, limit: int = 20) -> list:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id,created_at,algo,feature_version,n_train,n_test,auc,brier,"
                "top_decile_prec,base_rate,is_active,is_shadow,notes "
                "FROM stock_models ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]
