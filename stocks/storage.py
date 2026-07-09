from __future__ import annotations
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from .models import StockSnapshot, StockQuote, ScanSummary

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
                "rs_vs_spy,spy_change_pct,rs_assessment) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,"
                "?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
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

    # ── Automatic signal tracking / calibration ─────────────────────────────

    REARM_COOLDOWN_MINUTES = 45

    def record_tracked_signal(
        self, ticker: str, signal: str, direction: int,
        entry: float, stop: float, target: float,
        stop_dollars: float, target_dollars: float, atr14: float, entry_ts: str,
    ) -> None:
        """
        Record an actionable signal for hands-off forward evaluation. Skips if an
        open signal already exists for this ticker+direction (avoids re-arming every
        scan), or if one was armed within the cooldown window — without this, every
        scan after a stop-out immediately re-enters the same chop and racks up
        correlated losses.
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
                return
            conn.execute(
                "INSERT INTO stock_signal_tracking "
                "(ticker,signal,direction,entry_price,stop_price,target_price,"
                "stop_dollars,target_dollars,atr14,entry_ts) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (ticker, signal, direction, entry, stop, target,
                 stop_dollars, target_dollars, atr14, entry_ts),
            )

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
            for b in forward:
                hi, lo = b["high"], b["low"]
                if direction == 1:
                    if lo <= stop:        # stop checked first = conservative
                        exit_price, outcome = stop, "LOSS"
                        break
                    if hi >= target:
                        exit_price, outcome = target, "WIN"
                        break
                else:
                    if hi >= stop:
                        exit_price, outcome = stop, "LOSS"
                        break
                    if lo <= target:
                        exit_price, outcome = target, "WIN"
                        break

            if outcome is None:
                # Timeout: close at last available close once held longer than max_hold
                created = self._parse_dt(row.get("created_at"))
                aged_out = created is not None and (now - created).total_seconds() > max_hold_hours * 3600
                if aged_out and forward:
                    exit_price = forward[-1]["close"]
                    pnl = (exit_price - entry_price) * direction
                    outcome = "WIN" if pnl > 0 else ("LOSS" if pnl < 0 else "BREAKEVEN")
                else:
                    continue  # still live

            exit_dollars = round((exit_price - entry_price) * direction, 4) if entry_price else 0.0
            exit_pct = round(exit_dollars / entry_price * 100, 3) if entry_price else 0.0
            r_multiple = round(exit_dollars / stop_dollars, 2) if stop_dollars and stop_dollars > 0 else None
            created = self._parse_dt(row.get("created_at"))
            hold_minutes = int((now - created).total_seconds() / 60) if created else None

            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO stock_trade_outcomes "
                    "(watchlist_id,ticker,signal,entry_price,exit_price,exit_dollars,exit_pct,"
                    "r_multiple,outcome,hold_minutes) VALUES (0,?,?,?,?,?,?,?,?,?)",
                    (ticker, row.get("signal"), entry_price, exit_price, exit_dollars, exit_pct,
                     r_multiple, outcome, hold_minutes),
                )
                conn.execute(
                    "UPDATE stock_signal_tracking SET status='closed' WHERE id=?",
                    (row["id"],),
                )
            resolved += 1

        if resolved:
            self.compute_and_save_performance()
        return resolved

    @staticmethod
    def _parse_dt(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            return None

    def load_tracked_signals(self, status: str = "open", limit: int = 200) -> list:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM stock_signal_tracking WHERE status=? "
                "ORDER BY created_at DESC LIMIT ?", (status, limit),
            ).fetchall()
        return [dict(r) for r in rows]
