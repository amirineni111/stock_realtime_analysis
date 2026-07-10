from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from stocks.config import get_settings
from stocks.market_hours import current_market_phase, phase_badge_color
from stocks.models import ScanRequest
from stocks.tickers import parse_watchlist
from stocks.scanner import run_scan
from stocks.storage import Storage

st.set_page_config(
    page_title="Stock Screening Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Persistence helpers ──────────────────────────────────────────────────────

PREFS_PATH = Path("data/app_preferences.json")

DEFAULT_WATCHLIST = "AAPL\nMSFT\nNVDA\nTSLA\nAMD"


def _load_prefs() -> dict:
    if PREFS_PATH.exists():
        try:
            return json.loads(PREFS_PATH.read_text())
        except Exception:
            pass
    return {}


def _save_prefs(d: dict) -> None:
    prefs = _load_prefs()
    prefs.update(d)
    PREFS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREFS_PATH.write_text(json.dumps(prefs, indent=2))


# ── Session state init ───────────────────────────────────────────────────────

def _init_state() -> None:
    prefs = _load_prefs()
    defaults = {
        "watchlist_raw": prefs.get("watchlist_raw", DEFAULT_WATCHLIST),
        "min_dollar_volume_m": prefs.get("min_dollar_volume_m", 5.0),
        "auto_refresh": prefs.get("auto_refresh", False),
        "refresh_seconds": prefs.get("refresh_seconds", 60),
        "allow_offhours": prefs.get("allow_offhours", False),
        "auto_refresh_count_last": 0,
        "quotes_auto_refresh_count_last": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()

# ── Signal color ─────────────────────────────────────────────────────────────

SIGNAL_COLORS = {
    "STRONG_BUY": "🟢",
    "BUY_CANDIDATE": "🔵",
    "WATCH_ONLY": "🟡",
    "AVOID": "⚫",
    "SHORT_CANDIDATE": "🟠",
    "STRONG_SHORT": "🔴",
}


def _signal_badge(signal: str) -> str:
    return f"{SIGNAL_COLORS.get(signal, '⚫')} {signal}"


# ── Sidebar ──────────────────────────────────────────────────────────────────

def _render_sidebar() -> tuple:
    st.sidebar.title("📈 Stock Screener")

    phase = current_market_phase()
    badge = phase_badge_color(phase)
    st.sidebar.markdown(f"**Market:** {badge} {phase.replace('_', ' ')}")
    if phase == "CLOSED":
        st.sidebar.warning("US market is closed. Manual scans use the last session's bars.")

    st.sidebar.divider()

    # Watchlist
    st.sidebar.subheader("Watchlist")
    watchlist_raw = st.sidebar.text_area(
        "Tickers (one per line or comma-separated)",
        value=st.session_state.watchlist_raw,
        height=160,
        placeholder="AAPL\nMSFT\nNVDA",
    )
    st.session_state.watchlist_raw = watchlist_raw
    selected_tickers = parse_watchlist(watchlist_raw)
    st.sidebar.caption(f"{len(selected_tickers)} tickers: {', '.join(selected_tickers) or '—'}")
    if len(selected_tickers) > 50:
        st.sidebar.warning("Over ~50 tickers may hit Yahoo Finance rate limits.")

    st.sidebar.divider()

    # Scan settings
    st.sidebar.subheader("Scan Settings")
    min_dollar_volume_m = st.sidebar.number_input(
        "Min avg $ volume (millions/day, 0 = off)",
        min_value=0.0,
        max_value=10000.0,
        value=float(st.session_state.min_dollar_volume_m),
        step=1.0,
    )
    st.session_state.min_dollar_volume_m = min_dollar_volume_m
    signal_mode = st.sidebar.selectbox(
        "Signal mode",
        ["All", "Momentum", "Mean Reversion", "Day Breakout"],
    )

    st.sidebar.divider()

    # Auto-refresh
    st.sidebar.subheader("Auto-Refresh")
    auto_refresh = st.sidebar.checkbox("Auto-refresh scanner", value=st.session_state.auto_refresh)
    refresh_secs = st.sidebar.number_input(
        "Interval (seconds)",
        min_value=60,
        max_value=1800,
        value=max(60, int(st.session_state.refresh_seconds)),
        step=30,
        help="Minimum 60s to stay friendly with Yahoo Finance rate limits.",
    )
    allow_offhours = st.sidebar.checkbox(
        "Allow scans outside regular hours",
        value=st.session_state.allow_offhours,
        help="When off, auto-refresh pauses outside 09:30–16:00 ET to save API calls. Manual scans always work.",
    )
    st.session_state.auto_refresh = auto_refresh
    st.session_state.refresh_seconds = refresh_secs
    st.session_state.allow_offhours = allow_offhours

    _save_prefs({
        "watchlist_raw": watchlist_raw,
        "min_dollar_volume_m": min_dollar_volume_m,
        "auto_refresh": auto_refresh,
        "refresh_seconds": refresh_secs,
        "allow_offhours": allow_offhours,
    })

    return selected_tickers, min_dollar_volume_m * 1_000_000, signal_mode, auto_refresh, refresh_secs, allow_offhours, phase


# ── Scanner page ─────────────────────────────────────────────────────────────

def _page_scanner(
    settings,
    storage: Storage,
    selected_tickers: list,
    min_dollar_volume: float,
    signal_mode: str,
    auto_refresh: bool,
    refresh_secs: int,
    allow_offhours: bool,
    phase: str,
) -> None:
    st.title("Stock Scanner")

    scan_phase_ok = phase == "REGULAR" or allow_offhours

    # Auto-refresh wiring — suppressed off-hours unless the override is on.
    auto_count = None
    if auto_refresh and scan_phase_ok and selected_tickers:
        auto_count = st_autorefresh(interval=refresh_secs * 1000, key="scanner_autorefresh")

    auto_due = (
        auto_refresh
        and scan_phase_ok
        and bool(selected_tickers)
        and auto_count is not None
        and auto_count != st.session_state.auto_refresh_count_last
    )

    col1, col2 = st.columns([1, 6])
    with col1:
        run_now = st.button("▶ Run Scan", type="primary", disabled=not selected_tickers)
    with col2:
        if not selected_tickers:
            st.warning("Enter tickers in the sidebar watchlist.")
        elif auto_refresh and not scan_phase_ok:
            st.info("Auto-refresh paused — market not in regular hours. Enable the off-hours override or scan manually.")

    if run_now or auto_due:
        request = ScanRequest(
            tickers=selected_tickers,
            min_avg_dollar_volume=min_dollar_volume,
            signal_mode=signal_mode,
        )
        with st.spinner(f"Scanning {len(selected_tickers)} tickers…"):
            try:
                summary = run_scan(settings, storage, request)
                st.session_state.auto_refresh_count_last = auto_count or 0
                st.success(
                    f"Scan complete — {summary.tickers_scanned} tickers, "
                    f"{summary.signals_found} signals, {summary.errors} errors"
                )
            except Exception as exc:
                st.error(f"Scan failed: {exc}")

    tab_results, tab_watchlist, tab_perf, tab_logs, tab_settings = st.tabs(
        ["Results", "Watchlist", "Performance", "Scan Logs", "Settings"]
    )

    # ── Results tab ──────────────────────────────────────────────────────────
    with tab_results:
        rows = storage.load_latest_snapshots()
        if not rows:
            st.info("No scan results yet. Click 'Run Scan' to start.")
        else:
            df = pd.DataFrame(rows)

            # Relative Strength vs SPY
            if "rs_vs_spy" in df.columns and df["rs_vs_spy"].notna().any():
                with st.expander("Relative Strength vs SPY (day change % minus SPY's)", expanded=True):
                    rs_df = (
                        df[["ticker", "rs_vs_spy"]]
                        .dropna()
                        .rename(columns={"rs_vs_spy": "RS vs SPY (%)"})
                        .sort_values("RS vs SPY (%)", ascending=False)
                    )
                    st.bar_chart(rs_df.set_index("ticker"), height=220)
                    spy_chg = df["spy_change_pct"].dropna()
                    if len(spy_chg):
                        st.caption(f"SPY day change: {spy_chg.iloc[0]:+.2f}%. Positive RS = outperforming the market today.")

            # Metrics row
            m1, m2, m3, m4 = st.columns(4)
            last_run = storage.load_latest_scan_run()
            last_ts = last_run.get("finished_at", "—") if last_run else "—"
            m1.metric("Tickers Scanned", len(df))
            m2.metric("Actionable", len(df[df["trade_signal"].isin(["STRONG_BUY", "BUY_CANDIDATE", "STRONG_SHORT", "SHORT_CANDIDATE"])]))
            m3.metric("Avg Score", f"{df['total_score'].mean():.1f}")
            m4.metric("Last Scan (UTC)", last_ts)

            # Data staleness caption
            if "as_of" in df.columns and df["as_of"].notna().any():
                try:
                    newest = pd.to_datetime(df["as_of"], utc=True, format="ISO8601").max()
                    age_min = (datetime.now(timezone.utc) - newest.to_pydatetime()).total_seconds() / 60
                    st.caption(f"Last completed 5m bar: {newest.strftime('%Y-%m-%d %H:%M UTC')} ({age_min:.0f} min ago)")
                except Exception:
                    pass

            # Filters
            fc1, fc2, fc3 = st.columns(3)
            all_tickers = sorted(df["ticker"].unique().tolist())
            all_signals = sorted(df["trade_signal"].unique().tolist())
            ticker_filter = fc1.multiselect("Filter tickers", all_tickers, default=[])
            signal_filter = fc2.multiselect("Filter signals", all_signals, default=[])
            mtf_filter = fc3.selectbox("MTF Confluence", ["All", "Full Confluence Only", "Partial+"])

            filtered = df.copy()
            if ticker_filter:
                filtered = filtered[filtered["ticker"].isin(ticker_filter)]
            if signal_filter:
                filtered = filtered[filtered["trade_signal"].isin(signal_filter)]
            if mtf_filter == "Full Confluence Only" and "mtf_confluence" in filtered.columns:
                filtered = filtered[filtered["mtf_confluence"] == "FULL"]
            elif mtf_filter == "Partial+" and "mtf_confluence" in filtered.columns:
                filtered = filtered[filtered["mtf_confluence"].isin(["FULL", "PARTIAL"])]

            # Apply signal mode filter
            if signal_mode == "Momentum":
                filtered = filtered[filtered["momentum_score"] >= filtered["reversion_score"]]
            elif signal_mode == "Mean Reversion":
                filtered = filtered[filtered["reversion_score"] >= filtered["momentum_score"]]
            elif signal_mode == "Day Breakout":
                filtered = filtered[filtered["breakout_score"] > 0]

            display_cols = [
                "ticker", "trade_signal", "total_score", "regime",
                "momentum_score", "reversion_score", "breakout_score",
                "mtf_score", "mtf_confluence",
                "last", "day_change_pct", "rs_vs_spy", "rs_assessment",
                "suggested_entry", "suggested_stop", "suggested_target",
                "stop_dollars", "target_dollars", "stop_pct", "rr_ratio",
                "hourly_direction", "daily_direction",
                "sr_score", "at_key_level", "nearest_support", "nearest_resistance",
                "avg_dollar_volume",
                "rsi14", "adx14", "macd_histogram", "ema9", "ema20",
                "atr14", "bb_width_pct",
                "day_high", "day_low", "or_high", "or_low",
                "market_phase", "signal_reason", "as_of",
            ]
            available_cols = [c for c in display_cols if c in filtered.columns]

            # User-defined column order, persisted across sessions
            saved_order = _load_prefs().get("results_column_order") or available_cols
            saved_order = [c for c in saved_order if c in available_cols] or available_cols

            def _reset_column_order():
                st.session_state.results_col_order = available_cols

            with st.expander("⚙️ Arrange Columns"):
                st.caption(
                    "Columns display in the order you select them — remove one to hide it, "
                    "re-add it to move it to the end. Your arrangement is saved between sessions."
                )
                ordered_cols = st.multiselect(
                    "Visible columns (selection order = display order)",
                    options=available_cols,
                    default=saved_order,
                    key="results_col_order",
                )
                st.button("Reset to default order", on_click=_reset_column_order)

            if not ordered_cols:
                ordered_cols = available_cols
            if ordered_cols != saved_order:
                _save_prefs({"results_column_order": ordered_cols})

            data = filtered.reset_index(drop=True)
            display = data[ordered_cols].copy()
            if "trade_signal" in display.columns:
                display["trade_signal"] = display["trade_signal"].apply(_signal_badge)

            def _fmt2(x):
                try:
                    return f"{x:.2f}" if x is not None and x == x else ""
                except (TypeError, ValueError):
                    return ""

            def _fmt_mm(x):
                try:
                    return f"${x/1e6:,.0f}M" if x is not None and x == x else ""
                except (TypeError, ValueError):
                    return ""

            fmt_map = {c: _fmt2 for c in [
                "last", "close", "ema9", "ema20",
                "nearest_support", "nearest_resistance",
                "suggested_entry", "suggested_stop", "suggested_target",
                "stop_dollars", "target_dollars",
                "day_high", "day_low", "or_high", "or_low", "atr14",
            ] if c in display.columns}
            if "avg_dollar_volume" in display.columns:
                fmt_map["avg_dollar_volume"] = _fmt_mm

            # Selection made on the previous rerun — used to paint the full row
            selected_set: set = set()
            grid_state = st.session_state.get("results_grid")
            if grid_state:
                selected_set = {
                    r for r in grid_state.get("selection", {}).get("rows", [])
                    if r < len(display)
                }

            # Key-level flags come from the full data so the highlight survives
            # even when the at_key_level column itself is hidden by the user.
            key_flags = data["at_key_level"] if "at_key_level" in data.columns else None

            def _style_row(row):
                if row.name in selected_set:
                    return ["background-color: #cce0ff; color: #000000; font-weight: 600"] * len(row)
                flag = key_flags.iloc[row.name] if key_flags is not None else None
                if pd.notna(flag) and flag:
                    return ["background-color: #fff3b0; color: #000000"] * len(row)
                return [""] * len(row)

            styled = display.style.format(fmt_map, na_rep="").apply(_style_row, axis=1)

            event = st.dataframe(
                styled,
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                key="results_grid",
            )

            sel_rows = [r for r in (event.selection.rows if event else []) if r < len(data)]
            if sel_rows:
                sel = data.iloc[sel_rows[0]]

                def _num(v):
                    try:
                        return f"{float(v):.2f}"
                    except (TypeError, ValueError):
                        return "—"

                st.markdown(f"**Selected:** {_signal_badge(sel.get('trade_signal', ''))} — **{sel.get('ticker', '')}**")
                d1, d2, d3, d4, d5 = st.columns(5)
                d1.metric("Last", _num(sel.get("last")))
                d2.metric("Entry", _num(sel.get("suggested_entry")))
                d3.metric("Stop", _num(sel.get("suggested_stop")))
                d4.metric("Target", _num(sel.get("suggested_target")))
                d5.metric("R:R", _num(sel.get("rr_ratio")))
                if sel.get("signal_reason"):
                    st.caption(sel["signal_reason"])
            else:
                st.caption("Select a row (checkbox on the left) to highlight it and see its trade levels.")

            with st.expander("Column Guide"):
                st.markdown("""
| Column | Description |
|--------|-------------|
| trade_signal | Overall signal: STRONG_BUY, BUY_CANDIDATE, SHORT_CANDIDATE, STRONG_SHORT, WATCH_ONLY, AVOID |
| total_score | Combined score (base 0–100 + MTF 0–30 + S/R 0–25 + relative strength ±10) |
| regime | ADX-based: TREND (momentum weighted), RANGE (reversion weighted), MIXED, UNKNOWN |
| momentum_score | EMA/MACD/RSI momentum component (0–40), weighted by regime |
| reversion_score | RSI extreme/Bollinger Band component (0–40), weighted by regime |
| breakout_score | Day/opening-range breakout + liquidity window component (0–20) |
| last / day_change_pct | Last completed 5m close; change vs previous session close |
| rs_vs_spy / rs_assessment | Day change minus SPY's: OUTPERFORMING / IN_LINE / UNDERPERFORMING (±1%) |
| suggested_entry / stop / target | ATR-based levels (stop 2.5×ATR floor 0.50%, target 1.5R) |
| stop_dollars / target_dollars / stop_pct | Risk and reward in $ per share; stop as % of entry |
| mtf_score / mtf_confluence | Multi-timeframe agreement: 5m + hourly + daily (0/15/30) |
| hourly_direction / daily_direction | Higher-timeframe trend (LONG/SHORT/NEUTRAL) |
| sr_score / at_key_level | Support/resistance proximity bonus (0–25); highlighted at a key level |
| avg_dollar_volume | 20-day average of close×volume (liquidity gate) |
| rsi14 | RSI(14): <30 oversold, >70 overbought |
| adx14 | Trend strength: >25 trending, <18 ranging |
| bb_width_pct | Bollinger Band width % — squeeze detection |
| day_high/low, or_high/low | Day range since 09:30 ET and 09:30–10:00 opening range |
| market_phase | PRE_MARKET / REGULAR / AFTER_HOURS / CLOSED at scan time |
                """)

            st.download_button(
                "Export CSV",
                display.to_csv(index=False).encode(),
                file_name="stock_scan.csv",
                mime="text/csv",
            )

    # ── Watchlist tab ────────────────────────────────────────────────────────
    with tab_watchlist:
        st.subheader("Add to Watchlist")
        with st.form("add_watchlist"):
            wc1, wc2, wc3 = st.columns(3)
            w_ticker = wc1.text_input("Ticker (e.g. AAPL)")
            w_signal = wc2.selectbox(
                "Signal",
                ["STRONG_BUY", "BUY_CANDIDATE", "SHORT_CANDIDATE", "STRONG_SHORT", "WATCH_ONLY"],
            )
            w_entry = wc3.number_input("Entry price", min_value=0.0, format="%.2f")
            wc4, wc5, wc6 = st.columns(3)
            w_target = wc4.number_input("Target price", min_value=0.0, format="%.2f")
            w_stop = wc5.number_input("Stop price", min_value=0.0, format="%.2f")
            w_notes = wc6.text_input("Notes")
            w_stop_dollars = st.number_input("Stop distance ($/share)", min_value=0.0, value=1.0, format="%.2f")
            w_target_dollars = st.number_input("Target distance ($/share)", min_value=0.0, value=1.5, format="%.2f")
            submitted = st.form_submit_button("Add")
            if submitted and w_ticker:
                storage.add_watchlist(
                    w_ticker.strip().upper(), w_signal, w_entry,
                    w_target, w_stop, w_stop_dollars, w_target_dollars, w_notes,
                )
                st.success(f"Added {w_ticker.strip().upper()} to watchlist.")

        st.subheader("Active Watches")
        watching = storage.load_watchlist("watching")
        if not watching:
            st.info("No active watches.")
        else:
            wdf = pd.DataFrame(watching)
            if "signal" in wdf.columns:
                wdf["signal"] = wdf["signal"].apply(lambda s: _signal_badge(s) if s else s)
            st.dataframe(wdf, use_container_width=True, hide_index=True)

            cc1, cc2 = st.columns(2)
            close_id = cc1.number_input("Close watch ID", min_value=0, step=1, value=0)
            exit_price = cc2.number_input("Exit price (0 = skip outcome)", min_value=0.0, format="%.2f")
            if st.button("Close Watch") and close_id > 0:
                if exit_price > 0:
                    storage.close_watchlist_with_outcome(int(close_id), float(exit_price))
                    st.success(f"Closed watch ID {close_id} — outcome recorded.")
                else:
                    storage.close_watchlist(int(close_id))
                    st.success(f"Closed watch ID {close_id}.")
                st.rerun()

        with st.expander("Closed Watches"):
            closed = storage.load_watchlist("closed")
            if closed:
                st.dataframe(pd.DataFrame(closed), use_container_width=True, hide_index=True)

    # ── Performance tab ───────────────────────────────────────────────────────
    with tab_perf:
        outcomes = storage.load_trade_outcomes(limit=10000)
        if not outcomes:
            st.info(
                "No trade outcomes yet. Outcomes accrue automatically as each scan forward-tests "
                "actionable signals against their ATR stop/target — or close watchlist entries with "
                "an exit price to log manual trades."
            )
        else:
            odf = pd.DataFrame(outcomes)
            odf["date"] = pd.to_datetime(odf["created_at"]).dt.date.astype(str)

            all_dates = sorted(odf["date"].unique().tolist(), reverse=True)
            date_choice = st.selectbox("Filter by date", ["All dates"] + all_dates)
            fdf = odf if date_choice == "All dates" else odf[odf["date"] == date_choice]

            total_trades = len(fdf)
            wins = int((fdf["outcome"] == "WIN").sum())
            losses = int((fdf["outcome"] == "LOSS").sum())
            win_rate = wins / total_trades if total_trades else 0
            r_all = fdf["r_multiple"].dropna()
            r_wins = fdf.loc[fdf["outcome"] == "WIN", "r_multiple"].dropna()
            avg_win_r = r_wins.mean() if len(r_wins) else 0.0
            expectancy = r_all.mean() if len(r_all) else 0.0

            pm1, pm2, pm3, pm4, pm5 = st.columns(5)
            pm1.metric("Predictions", total_trades)
            pm2.metric("Wins / Losses", f"{wins} / {losses}")
            pm3.metric("Win Rate", f"{win_rate*100:.1f}%")
            pm4.metric("Avg Win R", f"{avg_win_r:.2f}R")
            pm5.metric("Expectancy", f"{expectancy:.3f}R/trade")
            if date_choice != "All dates":
                st.caption(f"{total_trades} predictions on {date_choice}: {wins} won, {losses} lost.")

            st.subheader("Results by Day")
            daily = (
                odf.assign(
                    win=(odf["outcome"] == "WIN").astype(int),
                    loss=(odf["outcome"] == "LOSS").astype(int),
                )
                .groupby("date")
                .agg(Predictions=("outcome", "size"), Wins=("win", "sum"), Losses=("loss", "sum"), R=("r_multiple", "sum"))
                .reset_index()
                .sort_values("date", ascending=False)
            )
            daily["Win Rate"] = (daily["Wins"] / daily["Predictions"] * 100).map("{:.1f}%".format)
            daily["R"] = daily["R"].round(1)
            daily.columns = ["Date", "Predictions", "Wins", "Losses", "Net R", "Win Rate"]
            st.dataframe(
                daily[["Date", "Predictions", "Wins", "Losses", "Win Rate", "Net R"]],
                use_container_width=True, hide_index=True,
            )

            if "r_multiple" in fdf.columns and len(fdf):
                st.subheader("Equity Curve (Cumulative R)")
                cum_r = fdf.sort_values("created_at")["r_multiple"].fillna(0).cumsum().reset_index(drop=True)
                st.line_chart(cum_r)

            def _dim_table(df: pd.DataFrame, col: str, label: str) -> None:
                if col not in df.columns or not len(df):
                    return
                grp = (
                    df.assign(win=(df["outcome"] == "WIN").astype(int))
                    .groupby(col)
                    .agg(Trades=("outcome", "size"), Wins=("win", "sum"), Expectancy=("r_multiple", "mean"))
                    .reset_index()
                    .sort_values("Trades", ascending=False)
                )
                grp["Win Rate"] = (grp["Wins"] / grp["Trades"] * 100).map("{:.1f}%".format)
                grp["Expectancy"] = grp["Expectancy"].round(3)
                grp.columns = [label, "Trades", "Wins", "Expectancy", "Win Rate"]
                st.subheader(f"Win Rate by {label}")
                st.dataframe(
                    grp[[label, "Trades", "Wins", "Win Rate", "Expectancy"]],
                    use_container_width=True, hide_index=True,
                )

            _dim_table(fdf, "ticker", "Ticker")
            _dim_table(fdf, "signal", "Signal")

            with st.expander("Trade History"):
                st.dataframe(fdf, use_container_width=True, hide_index=True)

        open_tracked = storage.load_tracked_signals("open")
        if open_tracked:
            with st.expander(f"Open Tracked Signals ({len(open_tracked)})"):
                tdf = pd.DataFrame(open_tracked)
                keep = [c for c in [
                    "ticker", "signal", "entry_price", "stop_price", "target_price",
                    "stop_dollars", "target_dollars", "entry_ts", "created_at",
                ] if c in tdf.columns]
                st.dataframe(tdf[keep], use_container_width=True, hide_index=True)
                st.caption("Each scan checks these against their ATR stop/target; a touch records a WIN/LOSS outcome.")

    # ── Scan Logs tab ─────────────────────────────────────────────────────────
    with tab_logs:
        logs = storage.load_scan_logs()
        if not logs:
            st.info("No scan logs yet.")
        else:
            st.dataframe(pd.DataFrame(logs), use_container_width=True, hide_index=True)

    # ── Settings tab ──────────────────────────────────────────────────────────
    with tab_settings:
        last_run = storage.load_latest_scan_run()
        if last_run:
            st.json(last_run)
        else:
            st.info("No scan runs recorded yet.")
        st.subheader("Current Request")
        st.json({
            "tickers": selected_tickers,
            "min_avg_dollar_volume": min_dollar_volume,
            "signal_mode": signal_mode,
            "market_phase": phase,
            "db_path": str(settings.db_path),
        })


# ── Live Quotes page ─────────────────────────────────────────────────────────

def _page_live_quotes(storage: Storage, selected_tickers: list, allow_offhours: bool, phase: str) -> None:
    st.title("Live Quotes")

    quotes_phase_ok = phase != "CLOSED" or allow_offhours

    auto_count = None
    if quotes_phase_ok:
        auto_count = st_autorefresh(interval=30_000, key="quotes_autorefresh")

    auto_due = (
        quotes_phase_ok
        and bool(selected_tickers)
        and auto_count is not None
        and auto_count != st.session_state.quotes_auto_refresh_count_last
    )

    col1, col2 = st.columns([1, 6])
    with col1:
        fetch_now = st.button("Fetch Quotes", disabled=not selected_tickers)
    with col2:
        badge = phase_badge_color(phase)
        suffix = "auto-refreshes every 30s" if quotes_phase_ok else "auto-refresh paused (market closed)"
        st.markdown(f"**Market:** {badge} {phase.replace('_', ' ')} — {suffix}")

    if fetch_now or auto_due:
        if selected_tickers:
            from stocks.yf_client import YFClient
            try:
                quotes = YFClient().get_quotes(selected_tickers)
                storage.save_quotes(quotes)
                st.session_state.quotes_auto_refresh_count_last = auto_count or 0
            except Exception as exc:
                st.error(f"Failed to fetch quotes: {exc}")

    quotes = storage.load_latest_quotes()
    if not quotes:
        st.info("No live quotes yet. Click 'Fetch Quotes' or wait for auto-refresh.")
    else:
        qdf = pd.DataFrame(quotes)

        display_cols = ["ticker", "last", "prev_close", "change_pct", "volume", "as_of"]
        display = qdf[[c for c in display_cols if c in qdf.columns]].reset_index(drop=True)

        selected_set: set = set()
        quotes_state = st.session_state.get("quotes_grid")
        if quotes_state:
            selected_set = {
                r for r in quotes_state.get("selection", {}).get("rows", [])
                if r < len(display)
            }

        def _row_color(row):
            if row.name in selected_set:
                return ["background-color: #cce0ff; color: #000000; font-weight: 600"] * len(row)
            try:
                chg = float(row.get("change_pct") or 0)
            except (TypeError, ValueError):
                chg = 0
            if chg > 0:
                return ["background-color: #d8f3d8; color: #000000"] * len(row)
            if chg < 0:
                return ["background-color: #f3d8d8; color: #000000"] * len(row)
            return [""] * len(row)

        fmt = {c: "{:.2f}" for c in ["last", "prev_close"] if c in display.columns}
        if "change_pct" in display.columns:
            fmt["change_pct"] = "{:+.2f}%"
        if "volume" in display.columns:
            fmt["volume"] = "{:,.0f}"
        st.dataframe(
            display.style.apply(_row_color, axis=1).format(fmt, na_rep=""),
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="quotes_grid",
        )

        st.caption(
            f"Market: {phase.replace('_', ' ')} | "
            f"Last refresh: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')} | "
            "Quotes from Yahoo Finance 1-minute bars (may lag ~1–2 min)."
        )


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    (selected_tickers, min_dollar_volume, signal_mode,
     auto_refresh, refresh_secs, allow_offhours, phase) = _render_sidebar()

    settings = get_settings()
    storage = Storage(settings.db_path)

    page = st.sidebar.radio(
        "Page",
        ["Stock Scanner", "Live Quotes"],
        label_visibility="collapsed",
    )

    if page == "Stock Scanner":
        _page_scanner(
            settings, storage, selected_tickers, min_dollar_volume,
            signal_mode, auto_refresh, refresh_secs, allow_offhours, phase,
        )
    else:
        _page_live_quotes(storage, selected_tickers, allow_offhours, phase)


if __name__ == "__main__":
    main()
