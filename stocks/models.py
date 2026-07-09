from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel


class StockBar(BaseModel):
    ticker: str
    timeframe: str
    timestamp: str  # UTC ISO-8601, uniform format so string ordering is chronological
    open: float
    high: float
    low: float
    close: float
    volume: int = 0


class StockQuote(BaseModel):
    ticker: str
    last: float
    prev_close: Optional[float] = None
    change_pct: Optional[float] = None
    volume: int = 0
    as_of: str = ""


class StockSnapshot(BaseModel):
    ticker: str
    last: Optional[float] = None
    prev_close: Optional[float] = None
    avg_dollar_volume: Optional[float] = None
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    day_change_pct: Optional[float] = None
    rsi14: Optional[float] = None
    ema9: Optional[float] = None
    ema20: Optional[float] = None
    ema50: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    atr14: Optional[float] = None
    adx14: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_width_pct: Optional[float] = None
    market_phase: Optional[str] = None
    day_high: Optional[float] = None
    day_low: Optional[float] = None
    or_high: Optional[float] = None
    or_low: Optional[float] = None
    momentum_score: float = 0.0
    reversion_score: float = 0.0
    breakout_score: float = 0.0
    regime: Optional[str] = None
    total_score: float = 0.0
    trade_signal: str = "AVOID"
    signal_reason: str = ""
    risk_notes: str = ""
    as_of: str = ""
    # Suggested trade levels (ATR-based, dollar-denominated)
    suggested_entry: Optional[float] = None
    suggested_stop: Optional[float] = None
    suggested_target: Optional[float] = None
    stop_dollars: Optional[float] = None
    target_dollars: Optional[float] = None
    stop_pct: Optional[float] = None
    rr_ratio: Optional[float] = None
    # Multi-timeframe confluence
    hourly_direction: Optional[str] = None
    daily_direction: Optional[str] = None
    mtf_score: float = 0.0
    mtf_confluence: Optional[str] = None
    # Support/Resistance
    nearest_support: Optional[float] = None
    nearest_resistance: Optional[float] = None
    sr_score: float = 0.0
    at_key_level: bool = False
    sr_levels_json: Optional[str] = None
    # Relative strength vs SPY
    rs_vs_spy: Optional[float] = None
    spy_change_pct: Optional[float] = None
    rs_assessment: Optional[str] = None


class ScanRequest(BaseModel):
    tickers: List[str]
    min_avg_dollar_volume: float = 0.0  # 0 disables the liquidity gate
    signal_mode: str = "All"


class ScanSummary(BaseModel):
    tickers_scanned: int = 0
    errors: int = 0
    signals_found: int = 0
