from __future__ import annotations
from typing import Optional, Tuple

# Replaces the forex currency-strength matrix with a simple relative-strength
# comparison against SPY: rs = ticker day-change% − SPY day-change%.

_RS_THRESHOLD = 1.0  # percentage points vs SPY


def calculate_rs(day_change_pct: Optional[float], spy_change_pct: Optional[float]) -> Optional[float]:
    if day_change_pct is None or spy_change_pct is None:
        return None
    return round(day_change_pct - spy_change_pct, 3)


def rs_assessment(rs: Optional[float]) -> Optional[str]:
    if rs is None:
        return None
    if rs >= _RS_THRESHOLD:
        return "OUTPERFORMING"
    if rs <= -_RS_THRESHOLD:
        return "UNDERPERFORMING"
    return "IN_LINE"


def rs_bonus(assessment: Optional[str], trade_signal: str) -> float:
    """Score adjustment: reward signals aligned with relative strength, lightly
    penalize signals fighting it. Applied post-classification (adjusts total_score only)."""
    if assessment is None:
        return 0.0
    is_long = trade_signal in ("STRONG_BUY", "BUY_CANDIDATE")
    is_short = trade_signal in ("STRONG_SHORT", "SHORT_CANDIDATE")
    if assessment == "OUTPERFORMING":
        if is_long:
            return 10.0
        if is_short:
            return -5.0
    elif assessment == "UNDERPERFORMING":
        if is_short:
            return 10.0
        if is_long:
            return -5.0
    return 0.0
