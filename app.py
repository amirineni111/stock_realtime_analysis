from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from stocks.config import get_settings
from stocks.features import FEATURE_VERSION
from stocks.indices import INDEX_SYMBOLS
from stocks.market_hours import current_market_phase, phase_badge_color
from stocks.model import MIN_TRAIN_SAMPLES
from stocks.models import ScanRequest
from stocks.tickers import parse_watchlist
from stocks.scanner import run_scan
from stocks.storage import Storage
from stocks.training import evaluate_and_fit, gate_summary, save_candidate

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
        "paper_dialog_open": False,
        "paper_flash": "",
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


# ── Market index header strip ────────────────────────────────────────────────

@st.cache_data(ttl=30, show_spinner=False)
def _index_quotes_cached() -> list:
    """Index levels for the page header. Streamlit reruns the whole script on
    every widget interaction, so cache to keep Yahoo requests at ~2/min.
    Failures return [] and are cached too — an outage must not re-hit the API
    on every rerun."""
    from stocks.indices import fetch_index_quotes
    try:
        return [q.model_dump() for q in fetch_index_quotes()]
    except Exception:
        return []


def _render_index_metrics(cols) -> None:
    """Render one st.metric per index — current level, with points and percent
    up/down as the delta. Streamlit colors the delta green/red from its sign."""
    quotes = {q["ticker"]: q for q in _index_quotes_cached()}
    for col, (sym, label) in zip(cols, INDEX_SYMBOLS):
        q = quotes.get(sym)
        if not q or q.get("last") is None:
            col.metric(label, "—", help="Index data unavailable")
            continue
        last, prev = q["last"], q.get("prev_close")
        delta = None
        if prev:
            delta = f"{last - prev:+,.2f} ({q['change_pct']:+.2f}%)"
        col.metric(label, f"{last:,.2f}", delta=delta)


def _page_header(title: str) -> None:
    """Page H1 with the index strip on the same row."""
    head_title, *head_idx = st.columns(
        [3] + [1] * len(INDEX_SYMBOLS), vertical_alignment="bottom"
    )
    with head_title:
        st.title(title)
    _render_index_metrics(head_idx)


# ── Paper trading ────────────────────────────────────────────────────────────

@st.cache_data(ttl=30, show_spinner=False)
def _position_quotes_cached(tickers: tuple) -> list:
    """Live prices for held tickers, cached like the index strip so mark-to-market
    doesn't re-hit Yahoo on every rerun. Takes a tuple — cache keys must hash."""
    from stocks.paper import fetch_position_quotes
    try:
        return [q.model_dump() for q in fetch_position_quotes(list(tickers))]
    except Exception:
        return []


def _last_prices(tickers) -> dict:
    """{ticker: last} for the given tickers, empty on any fetch failure."""
    unique = tuple(sorted({str(t).upper() for t in tickers if t}))
    return {q["ticker"]: q["last"] for q in _position_quotes_cached(unique)}


def _fnum(v, fmt: str = "{:,.2f}", dash: str = "—") -> str:
    """Format a possibly-missing number without blowing up on None/NaN."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return dash
    if f != f:  # NaN
        return dash
    return fmt.format(f)


def _dialog_open() -> None:
    """Auto-refresh reruns the whole script and would tear down an open modal,
    losing whatever the user has typed. Suppress it while a ticket is up."""
    st.session_state.paper_dialog_open = True


def _dialog_done() -> None:
    st.session_state.paper_dialog_open = False


@st.dialog("Order Ticket")
def _dialog_order_ticket(sel: dict, direction: int) -> None:
    """Place a paper order, prefilled from the scanner's suggested levels."""
    from stocks.paper import position_cost
    from stocks.signals import trade_levels

    ticker = str(sel.get("ticker", ""))
    side_label = "Buy" if direction > 0 else "Sell / Short"
    st.markdown(f"**{side_label} {ticker}**")

    suggested = sel.get("suggested_entry") or sel.get("last") or 0.0
    c1, c2 = st.columns(2)
    qty = c1.number_input("Quantity (shares)", min_value=1, step=1, value=100, key="ot_qty")
    entry = c2.number_input(
        "Entry price", min_value=0.01, format="%.2f",
        value=float(suggested) if suggested else 0.01, key="ot_entry",
    )

    # Re-derive stop/target from the same ATR math the scanner used, so changing
    # the entry keeps the levels coherent instead of stranding the suggestion.
    atr14 = sel.get("atr14")
    levels = trade_levels("LONG" if direction > 0 else "SHORT", entry, atr14)
    def_stop = levels.get("suggested_stop") or sel.get("suggested_stop") or 0.0
    def_target = levels.get("suggested_target") or sel.get("suggested_target") or 0.0

    c3, c4 = st.columns(2)
    stop = c3.number_input("Stop price (0 = none)", min_value=0.0, format="%.2f",
                           value=float(def_stop or 0.0), key="ot_stop")
    target = c4.number_input("Target price (0 = none)", min_value=0.0, format="%.2f",
                             value=float(def_target or 0.0), key="ot_target")

    equity = st.session_state.get("paper_equity", 0.0) or 0.0
    risk = abs(entry - stop) * qty if stop else 0.0
    reward = abs(target - entry) * qty if target else 0.0
    cost = position_cost(qty, entry)

    m1, m2, m3 = st.columns(3)
    m1.metric("Position cost", f"${cost:,.2f}")
    m2.metric(
        "Risk", f"${risk:,.2f}" if stop else "—",
        help=f"{risk / equity * 100:.2f}% of equity" if stop and equity else None,
    )
    m3.metric("R:R", f"{reward / risk:.2f}" if risk else "—")
    if stop and equity:
        st.caption(f"Risking {risk / equity * 100:.2f}% of ${equity:,.2f} equity.")

    # A stop on the wrong side isn't a stop — it would fill instantly.
    bad_stop = bool(stop) and (
        (direction > 0 and stop >= entry) or (direction < 0 and stop <= entry)
    )
    if bad_stop:
        side = "below" if direction > 0 else "above"
        st.error(f"Stop must be {side} the entry for a {side_label.lower()}.")

    notes = st.text_input("Notes", key="ot_notes")

    b1, b2 = st.columns(2)
    if b1.button("Cancel", use_container_width=True, key="ot_cancel"):
        _dialog_done()
        st.rerun()
    if b2.button("Place Order", type="primary", use_container_width=True,
                 disabled=bad_stop, key="ot_place"):
        st.session_state.paper_storage.open_paper_position(
            ticker=ticker, direction=direction, qty=int(qty), price=float(entry),
            stop=float(stop) or None, target=float(target) or None,
            signal=str(sel.get("trade_signal") or ""), notes=notes,
        )
        _dialog_done()
        st.session_state.paper_flash = (
            f"{side_label} {int(qty)} {ticker} @ {entry:,.2f} — position opened."
        )
        st.rerun()


@st.dialog("Add to Position")
def _dialog_add_to_position(position: dict, last) -> None:
    """Scale into an open position and preview the new average cost."""
    from stocks.paper import scale_in

    ticker = position["ticker"]
    direction = int(position["direction"])
    side = "LONG" if direction > 0 else "SHORT"
    st.markdown(
        f"**Add to {ticker} ({side})** — holding {int(position['qty'])} sh "
        f"@ {position['avg_entry']:,.2f}"
    )

    c1, c2 = st.columns(2)
    qty = c1.number_input("Add quantity", min_value=1, step=1, value=50, key="add_qty")
    price = c2.number_input(
        "Fill price", min_value=0.01, format="%.2f",
        value=float(last) if last else float(position["avg_entry"]), key="add_price",
    )

    new_qty, new_avg = scale_in(
        int(position["qty"]), float(position["avg_entry"]), int(qty), float(price)
    )
    m1, m2 = st.columns(2)
    m1.metric("New size", f"{new_qty:,} sh")
    m2.metric("New avg cost", f"{new_avg:,.2f}",
              delta=f"{new_avg - float(position['avg_entry']):+,.2f}",
              delta_color="off")

    notes = st.text_input("Notes", key="add_notes")

    b1, b2 = st.columns(2)
    if b1.button("Cancel", use_container_width=True, key="add_cancel"):
        _dialog_done()
        st.rerun()
    if b2.button("Add", type="primary", use_container_width=True, key="add_go"):
        st.session_state.paper_storage.add_to_paper_position(
            int(position["id"]), int(qty), float(price), notes
        )
        _dialog_done()
        st.session_state.paper_flash = (
            f"Added {int(qty)} {ticker} @ {price:,.2f} — now {new_qty:,} sh @ {new_avg:,.2f}."
        )
        st.rerun()


@st.dialog("Close Position")
def _dialog_close_position(position: dict, last) -> None:
    """Close all or part of a position, previewing the P&L before committing."""
    from stocks.paper import position_r_multiple, realized_pnl

    ticker = position["ticker"]
    direction = int(position["direction"])
    open_qty = int(position["qty"])
    avg = float(position["avg_entry"])
    st.markdown(f"**Close {ticker}** — {open_qty} sh @ {avg:,.2f}")

    c1, c2 = st.columns(2)
    qty = c1.number_input("Quantity to close", min_value=1, max_value=open_qty,
                          step=1, value=open_qty, key="cl_qty")
    price = c2.number_input("Exit price", min_value=0.01, format="%.2f",
                            value=float(last) if last else avg, key="cl_price")

    pnl = realized_pnl(direction, avg, float(price), int(qty))
    r = position_r_multiple(direction, avg, float(price), position.get("stop_price"))
    m1, m2, m3 = st.columns(3)
    m1.metric("Realized P&L", f"${pnl:,.2f}", delta=f"{pnl:+,.2f}")
    m2.metric("R multiple", f"{r:+.2f}R" if r is not None else "—")
    m3.metric("Remaining", f"{open_qty - int(qty):,} sh")
    if int(qty) < open_qty:
        st.caption("Partial close — the position stays open with the remaining shares.")

    notes = st.text_input("Notes", key="cl_notes")

    b1, b2 = st.columns(2)
    if b1.button("Cancel", use_container_width=True, key="cl_cancel"):
        _dialog_done()
        st.rerun()
    if b2.button("Close Position", type="primary", use_container_width=True, key="cl_go"):
        st.session_state.paper_storage.close_paper_position(
            int(position["id"]), int(qty), float(price), notes
        )
        _dialog_done()
        st.session_state.paper_flash = (
            f"Closed {int(qty)} {ticker} @ {price:,.2f} — P&L ${pnl:,.2f}."
        )
        st.rerun()


def _render_paper_tab(storage: Storage) -> None:
    """Account equity, open positions marked to market, and the trade history."""
    from stocks.paper import pnl_pct, position_r_multiple, unrealized_pnl

    account = storage.paper_account()
    open_positions = storage.load_paper_positions("open")

    marks = _last_prices([p["ticker"] for p in open_positions])
    open_pnl = 0.0
    exposure = 0.0
    for p in open_positions:
        mark = marks.get(p["ticker"]) or p["avg_entry"]
        open_pnl += unrealized_pnl(
            int(p["direction"]), float(p["avg_entry"]), float(mark), int(p["qty"])
        )
        exposure += float(mark) * int(p["qty"])

    total = account["equity"] + open_pnl
    start = account["starting_equity"]

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Starting equity", f"${start:,.2f}")
    k2.metric("Realized P&L", f"${account['realized_pnl']:,.2f}",
              delta=f"{account['realized_pnl']:+,.2f}")
    k3.metric("Open P&L", f"${open_pnl:,.2f}", delta=f"{open_pnl:+,.2f}")
    k4.metric("Total equity", f"${total:,.2f}",
              delta=f"{(total - start) / start * 100:+.2f}%" if start else None)
    k5.metric("Exposure", f"${exposure:,.2f}",
              help="Market value of open positions at the current mark")

    st.subheader("Open Positions")
    if not open_positions:
        st.info("No open positions. Select a row in the Results tab and click Buy or Sell/Short.")
    else:
        rows = []
        for p in open_positions:
            direction = int(p["direction"])
            avg = float(p["avg_entry"])
            qty = int(p["qty"])
            mark = marks.get(p["ticker"])
            mark_px = float(mark) if mark else avg
            rows.append({
                "id": p["id"],
                "ticker": p["ticker"],
                "side": "LONG" if direction > 0 else "SHORT",
                "qty": qty,
                "avg_entry": avg,
                "mark": mark_px if mark else None,
                "stop": p.get("stop_price"),
                "target": p.get("target_price"),
                "open_pnl": unrealized_pnl(direction, avg, mark_px, qty),
                "pnl_pct": pnl_pct(direction, avg, mark_px),
                "R": position_r_multiple(direction, avg, mark_px, p.get("stop_price")),
                "realized": p.get("realized_pnl") or 0.0,
                "opened_at": p.get("opened_at"),
            })
        pos_df = pd.DataFrame(rows)

        def _pnl_row(row):
            try:
                v = float(row.get("open_pnl") or 0)
            except (TypeError, ValueError):
                v = 0
            if v > 0:
                return ["background-color: #d8f3d8; color: #000000"] * len(row)
            if v < 0:
                return ["background-color: #f3d8d8; color: #000000"] * len(row)
            return [""] * len(row)

        st.dataframe(
            pos_df.style.apply(_pnl_row, axis=1).format({
                "avg_entry": "{:,.2f}", "mark": "{:,.2f}", "stop": "{:,.2f}",
                "target": "{:,.2f}", "open_pnl": "{:+,.2f}", "pnl_pct": "{:+.2f}%",
                "R": "{:+.2f}", "realized": "{:+,.2f}", "qty": "{:,.0f}",
            }, na_rep="—"),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Marks are live 1-minute prices (30s cache). Positions close only when "
            "you close them — nothing exits automatically."
        )

    st.subheader("Closed Trades")
    closed = storage.load_paper_positions("closed")
    if not closed:
        st.info("No closed trades yet.")
    else:
        cdf = pd.DataFrame([{
            "id": p["id"],
            "ticker": p["ticker"],
            "side": "LONG" if int(p["direction"]) > 0 else "SHORT",
            "avg_entry": p["avg_entry"],
            "realized": p.get("realized_pnl") or 0.0,
            "signal": p.get("signal"),
            "opened_at": p.get("opened_at"),
            "closed_at": p.get("closed_at"),
        } for p in closed])
        wins = int((cdf["realized"] > 0).sum())
        losses = int((cdf["realized"] < 0).sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Closed trades", len(cdf))
        c2.metric("Wins / Losses", f"{wins} / {losses}")
        c3.metric("Win rate", f"{wins / len(cdf) * 100:.1f}%" if len(cdf) else "—")
        st.dataframe(
            cdf.style.format({"avg_entry": "{:,.2f}", "realized": "{:+,.2f}"}, na_rep="—"),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Fill History"):
        fills = storage.load_paper_fills(limit=200)
        if not fills:
            st.caption("No fills yet.")
        else:
            fdf = pd.DataFrame(fills)[
                ["fill_ts", "ticker", "kind", "side", "qty", "price", "realized_pnl", "notes"]
            ]
            st.dataframe(
                fdf.style.format({"price": "{:,.2f}", "realized_pnl": "{:+,.2f}",
                                  "qty": "{:,.0f}"}, na_rep="—"),
                use_container_width=True,
                hide_index=True,
            )

    with st.expander("Account Settings"):
        st.caption(
            "Paper trades are stored separately from the model's forward-test record — "
            "they never affect the Performance tab or model training."
        )
        new_equity = st.number_input(
            "Starting equity ($)", min_value=100.0, step=1000.0,
            value=float(start), format="%.2f", key="paper_start_equity",
        )
        r1, r2 = st.columns([1, 3])
        if r1.button("Update starting equity"):
            storage.set_paper_starting_equity(float(new_equity))
            st.rerun()
        confirm = r2.checkbox("I understand this deletes all paper positions and fills",
                              key="paper_reset_confirm")
        if st.button("Reset paper account", type="secondary", disabled=not confirm):
            storage.reset_paper_account(float(new_equity))
            st.session_state.paper_flash = "Paper account reset."
            st.rerun()


def _render_trade_actions(storage: Storage, sel) -> None:
    """Buy/Sell when flat, Add/Close when a position is open. Sits under the
    selected row's trade levels in the Results grid."""
    ticker = str(sel.get("ticker") or "")
    if not ticker:
        return
    position = storage.load_open_paper_position(ticker)
    last = sel.get("last")

    if not position:
        b1, b2, _ = st.columns([1, 1, 4])
        if b1.button("🟢 Buy", key="act_buy", use_container_width=True):
            _dialog_open()
            _dialog_order_ticket(sel.to_dict(), 1)
        if b2.button("🔴 Sell / Short", key="act_sell", use_container_width=True):
            _dialog_open()
            _dialog_order_ticket(sel.to_dict(), -1)
        return

    from stocks.paper import unrealized_pnl

    direction = int(position["direction"])
    qty = int(position["qty"])
    avg = float(position["avg_entry"])
    mark = _last_prices([ticker]).get(ticker) or last or avg
    open_pnl = unrealized_pnl(direction, avg, float(mark), qty)

    b1, b2, b3 = st.columns([1, 1, 4])
    if b1.button("➕ Add", key="act_add", use_container_width=True):
        _dialog_open()
        _dialog_add_to_position(position, mark)
    if b2.button("✖ Close", key="act_close", use_container_width=True):
        _dialog_open()
        _dialog_close_position(position, mark)
    b3.markdown(
        f"**{'LONG' if direction > 0 else 'SHORT'}** {qty:,} sh @ {avg:,.2f} · "
        f"mark {float(mark):,.2f} · "
        f"{'🟢' if open_pnl >= 0 else '🔴'} **${open_pnl:,.2f}** open"
    )


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
    _page_header("Stock Scanner")

    scan_phase_ok = phase == "REGULAR" or allow_offhours

    # Auto-refresh wiring — suppressed off-hours unless the override is on, and
    # while an order ticket is open (a rerun would tear the modal down mid-entry).
    auto_count = None
    ticket_open = st.session_state.get("paper_dialog_open", False)
    if auto_refresh and scan_phase_ok and selected_tickers and not ticket_open:
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

    # A confirmation from a dialog that has since closed via st.rerun().
    if st.session_state.get("paper_flash"):
        st.success(st.session_state.paper_flash)
        st.session_state.paper_flash = ""

    (tab_results, tab_paper, tab_watchlist, tab_perf,
     tab_model, tab_logs, tab_settings) = st.tabs(
        ["Results", "Paper Trades", "Watchlist", "Performance", "Model", "Scan Logs", "Settings"]
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
                "model_prob", "required_prob",
                "momentum_score", "reversion_score", "breakout_score",
                "mtf_score", "mtf_confluence",
                "last", "day_change_pct", "rs_vs_spy", "rs_assessment",
                "suggested_entry", "suggested_stop", "suggested_target",
                "stop_dollars", "target_dollars", "stop_pct", "rr_ratio",
                "cost_pct", "cost_ratio",
                "hourly_direction", "daily_direction",
                "sr_score", "at_key_level", "blocked_ahead",
                "nearest_support", "nearest_resistance",
                "avg_dollar_volume", "rel_volume",
                "rsi14", "adx14", "macd_histogram", "ema9", "ema20",
                "atr14", "extension_atr", "bb_width_pct",
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

                _render_trade_actions(storage, sel)
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
| model_prob / required_prob | Trained model's P(target before stop), and the cost-adjusted breakeven it must clear. Blank until a model is activated or shadowing; a shadow model fills it in without vetoing anything |
| cost_pct / cost_ratio | Estimated round-trip cost as % of price, and as a fraction of the stop distance (>15% is vetoed) |
| sr_score / at_key_level / blocked_ahead | Direction-aware structure score (−25 to +25): rewards a level behind the trade, penalizes one blocking the target |
| rel_volume | Last 5m bar's volume vs the prior 20 bars' average (2.0 = twice normal) |
| extension_atr | ATRs from EMA20; STRONG signals beyond ±2.0 downgrade to WATCH_ONLY |
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

    # ── Paper Trades tab ─────────────────────────────────────────────────────
    with tab_paper:
        _render_paper_tab(storage)

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

    # ── Model tab ────────────────────────────────────────────────────────────
    with tab_model:
        st.subheader("Direction Model")
        st.caption(
            "The rules-based score proposes a setup; this model decides whether the "
            "measured odds justify paying the round-trip cost. Until one is trained "
            "and activated, the scanner runs rules-only and simply logs features."
        )

        training_rows = storage.load_training_rows(feature_version=FEATURE_VERSION)
        open_n = len(storage.load_tracked_signals("open"))
        needed = max(0, MIN_TRAIN_SAMPLES - len(training_rows))

        c1, c2, c3 = st.columns(3)
        c1.metric("Trainable trades", len(training_rows))
        c2.metric("Awaiting resolution", open_n)
        c3.metric("Needed to train", needed if needed else "ready")

        if needed:
            st.progress(min(1.0, len(training_rows) / MIN_TRAIN_SAMPLES))
            st.info(
                f"{needed} more resolved trades needed before the first train "
                f"(minimum {MIN_TRAIN_SAMPLES}). Every actionable signal now stores its "
                "feature vector, so this fills up as signals resolve."
            )

        models = storage.load_models()
        active = next((m for m in models if m["is_active"]), None)
        shadow = next((m for m in models if m.get("is_shadow")), None)

        if not models:
            st.warning("No model trained yet — the scanner is running rules-only.")
        elif active:
            a1, a2, a3, a4 = st.columns(4)
            a1.metric("OOS AUC", f"{active['auc']:.4f}" if active["auc"] else "—")
            a2.metric("Top-decile precision",
                      f"{active['top_decile_prec']:.3f}" if active["top_decile_prec"] else "—")
            a3.metric("Brier", f"{active['brier']:.4f}" if active["brier"] else "—")
            a4.metric("Trained on", active["n_train"] or "—")
            st.success(f"Model #{active['id']} is active — signals are being probability-gated.")
            if active["auc"] is not None and active["auc"] < 0.53:
                st.warning(
                    f"Active model's out-of-sample AUC is {active['auc']:.4f} — "
                    "close to the 0.50 no-skill line. Treat its vetoes as weak evidence."
                )
        elif shadow:
            st.info(
                f"Model #{shadow['id']} is shadowing — it scores every directional setup "
                "and logs the probability, but nothing is gated. Signals are rules-only. "
                "Check what it would have done with `python scripts/model_report.py`."
            )
        else:
            st.info("Models exist but none is active. The scanner is running rules-only.")

        # ── When to retrain ─────────────────────────────────────────────────
        with st.expander("When to retrain, and how to read the result"):
            st.markdown(f"""
**Required**

- The first time you cross **{MIN_TRAIN_SAMPLES}** resolved trades.
- After editing `FEATURE_NAMES` in `stocks/features.py` — bump `FEATURE_VERSION` and the
  scanner ignores the stale model (fails closed) until you retrain.
- After changing `_RR`, `_STOP_ATR_MULT` or `_MIN_STOP_PCT` in `stocks/signals.py`, or the
  cost tiers in `estimate_cost_pct`. Those change the stop/target geometry, so older
  labels describe a different bracket and the model would be fitting the wrong thing.

**Worth doing**

- Every ~50–100 newly resolved trades once past the first train.
- If the realised win rate on gated trades sits below the model's predicted rate for
  several weeks — that is calibration drift.

**Don't** retrain on a handful of new rows. With ~30 features you would be chasing
noise, and each retrain shifts the veto threshold under your live signals.

---

**Reading the result — three numbers, in priority order**

1. **Pooled OOS AUC** — 0.50 is no skill. Below ~0.53 the model has nothing and its
   vetoes are noise. This is the pass/fail.
2. **Expectancy at the decision threshold** — the money number. AUC can look
   respectable while expectancy stays negative. Compare the gated rows against the
   ungated baseline; if gating does not beat it, the model is not earning its keep.
3. **Calibration** — predicted vs actual should track. If it predicts 0.60 and delivers
   0.40, the probability gate is comparing against a breakeven number that means
   nothing, even with good AUC.

The gate enforces (1) and (2) automatically. Check (3) yourself — it is the one that
can be quietly wrong.

Also watch **fold AUC std dev**. One fold at 0.78 and the rest near 0.52 is a model
fitted to one regime, not an edge — more informative than the pooled number at this
sample size.

> Your effective sample is smaller than the row count. Equities are highly correlated
> intraday: simultaneous longs across AAPL/MSFT/NVDA are largely one beta bet, and on
> a strong tape almost everything resolves the same way. Treat a first-train AUC of
> 0.53–0.58 as encouraging but provisional.
""")

        # ── Retrain ─────────────────────────────────────────────────────────
        st.markdown("### Retrain")
        ready = len(training_rows) >= MIN_TRAIN_SAMPLES

        with st.expander("Advanced settings"):
            adv1, adv2, adv3 = st.columns(3)
            l2 = adv1.number_input("L2 penalty", 0.01, 100.0, 1.0, step=0.5,
                                   help="Higher = more shrinkage. Raise if coefficients "
                                        "swing wildly between folds.")
            folds = adv2.number_input("Walk-forward folds", 2, 12, 5, step=1)
            min_auc = adv3.number_input("Minimum AUC to pass", 0.50, 0.80, 0.53, step=0.01)
            notes = st.text_input("Note stored with the model", "",
                                  placeholder="e.g. after widening stops")

        if st.button("Run Retrain", type="primary", disabled=not ready,
                     help=None if ready else
                     f"Needs {MIN_TRAIN_SAMPLES} resolved trades; you have {len(training_rows)}."):
            with st.spinner("Walk-forward evaluating and fitting…"):
                try:
                    report = evaluate_and_fit(storage, l2=float(l2), folds=int(folds),
                                              min_auc=float(min_auc))
                    if report["error"]:
                        st.session_state.model_report = None
                        st.error(report["error"])
                    else:
                        # Saved inactive — promotion is a separate, deliberate click.
                        new_id = save_candidate(storage, report, notes=notes)
                        st.session_state.model_report = {
                            "id": new_id,
                            "passes": report["passes"],
                            "summary": gate_summary(report),
                            "metrics": report["metrics"],
                            "gated": report["gated"],
                            "calibration": report["calibration"],
                            "folds": report["folds"],
                            "coefficients": report["coefficients"],
                            "baseline": report["baseline_expectancy_r"],
                            "fold_auc_std": report["fold_auc_std"],
                        }
                except Exception as exc:
                    st.session_state.model_report = None
                    st.error(f"Retrain failed: {exc}")

        # ── Candidate review + promote ──────────────────────────────────────
        rep = st.session_state.get("model_report")
        if rep:
            st.markdown("---")
            st.markdown(f"### Candidate model #{rep['id']}")
            if rep["passes"]:
                st.success(rep["summary"])
            else:
                st.error(rep["summary"])

            m = rep["metrics"]
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("OOS AUC", f"{m['auc']:.4f}" if m["auc"] is not None else "—",
                      delta=f"{m['auc'] - 0.5:+.4f} vs no-skill" if m["auc"] is not None else None)
            r2.metric("Fold AUC std", f"{rep['fold_auc_std']:.4f}" if rep["fold_auc_std"] else "—")
            r3.metric("Top-decile prec",
                      f"{m['top_decile_prec']:.3f}" if m["top_decile_prec"] is not None else "—")
            r4.metric("Brier", f"{m['brier']:.4f}" if m["brier"] is not None else "—")

            st.markdown("**Expectancy at each decision threshold** "
                        f"(ungated baseline: `{rep['baseline']:+.4f}R`)")
            gdf = pd.DataFrame(rep["gated"])
            if not gdf.empty:
                gdf = gdf.rename(columns={
                    "threshold": "P(win) >=", "trades_taken": "Taken",
                    "trades_available": "Available", "selectivity": "Selectivity",
                    "expectancy_r": "Expectancy (R)", "total_r": "Total (R)",
                })
                keep = [c for c in ["P(win) >=", "Taken", "Available", "Selectivity",
                                    "Expectancy (R)", "Total (R)"] if c in gdf.columns]
                st.dataframe(gdf[keep], use_container_width=True, hide_index=True)

            cc1, cc2 = st.columns(2)
            with cc1:
                st.markdown("**Calibration** (predicted vs actual)")
                cdf = pd.DataFrame(rep["calibration"])
                if not cdf.empty:
                    st.dataframe(cdf, use_container_width=True, hide_index=True)
                    st.caption("These two columns should track each other.")
            with cc2:
                st.markdown("**Per-fold stability**")
                fdf2 = pd.DataFrame(rep["folds"])
                if not fdf2.empty:
                    st.dataframe(fdf2, use_container_width=True, hide_index=True)
                    st.caption("One strong fold among weak ones = regime-fitted, not an edge.")

            with st.expander("Largest standardised coefficients (what carries the edge)"):
                st.dataframe(
                    pd.DataFrame(rep["coefficients"], columns=["feature", "coefficient"]),
                    use_container_width=True, hide_index=True,
                )

            st.markdown("#### Run in shadow (recommended first step)")
            st.caption(
                "Shadow scores every setup and logs the probability, but never vetoes. "
                "It is the only way to find out what the model would do to the trades it "
                "wants to block — once it is gating, those trades stop happening and stop "
                "being measurable. Check progress with `python scripts/model_report.py`."
            )
            if st.button(f"Run model #{rep['id']} in shadow"):
                if storage.shadow_model(rep["id"]):
                    st.success(
                        f"Model #{rep['id']} is now shadowing. Nothing is gated; "
                        f"probabilities are logged against every tracked trade."
                    )
                    st.session_state.model_report = None
                    st.rerun()
                else:
                    st.error("Could not set shadow — model id not found.")

            st.markdown("#### Promote")
            if rep["passes"]:
                st.caption("This model clears the gate. Promoting makes it veto live signals "
                           "on the next scan.")
                if st.button(f"Promote model #{rep['id']} to active", type="primary"):
                    if storage.activate_model(rep["id"]):
                        st.success(f"Model #{rep['id']} is now active.")
                        st.session_state.model_report = None
                        st.rerun()
                    else:
                        st.error("Could not activate — model id not found.")
            else:
                st.warning(
                    "This model failed the quality gate. Promoting it would let a model "
                    "with no demonstrated edge suppress real setups. The usual answer is "
                    "to collect more resolved trades and retrain."
                )
                override = st.checkbox("I understand, promote it anyway")
                if st.button(f"Force-promote model #{rep['id']}", disabled=not override):
                    if storage.activate_model(rep["id"]):
                        st.warning(f"Model #{rep['id']} force-promoted despite failing the gate.")
                        st.session_state.model_report = None
                        st.rerun()

        # ── History / rollback ──────────────────────────────────────────────
        if models:
            st.markdown("---")
            st.markdown("### Model history")
            st.dataframe(pd.DataFrame(models), use_container_width=True, hide_index=True)

            def _label(i: int) -> str:
                m = next((x for x in models if x["id"] == i), {})
                tag = " (active)" if m.get("is_active") else (
                    " (shadow)" if m.get("is_shadow") else "")
                return f"#{i}{tag}"

            h1, h2, h3 = st.columns([3, 1, 1])
            options = [m["id"] for m in models]
            pick = h1.selectbox("Select a model", options, format_func=_label)
            if h2.button("Activate", key="rollback_activate"):
                if storage.activate_model(int(pick)):
                    st.success(f"Model #{pick} is now active and gating.")
                    st.rerun()
            if h3.button("Shadow", key="rollback_shadow"):
                if storage.shadow_model(int(pick)):
                    st.success(f"Model #{pick} is now shadowing (logging only).")
                    st.rerun()

            if active and st.button("Disable model (revert to rules-only)"):
                storage.deactivate_all_models()
                st.info("All models deactivated — the scanner is rules-only again.")
                st.rerun()

            if shadow and st.button("Stop shadowing"):
                storage.clear_shadow_model()
                st.info("Shadow cleared — no model is scoring.")
                st.rerun()

        st.caption(
            "Equivalent CLI, if you prefer it: `python scripts/train_model.py` to evaluate, "
            "`--activate` to promote in one step, and `python scripts/model_report.py` to "
            "judge a shadow model on resolved trades."
        )

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
    _page_header("Live Quotes")

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

    # Dialogs run outside the page functions' scope, so hand them the storage
    # handle and current equity through session state.
    st.session_state.paper_storage = storage
    st.session_state.paper_equity = storage.paper_account()["equity"]

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
