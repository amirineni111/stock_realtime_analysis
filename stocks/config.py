from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel
import os


class AppSettings(BaseModel):
    db_path: Path = Path("data/stocks_screening.sqlite3")
    request_timeout_seconds: float = 20.0
    # Push endpoint for new-signal alerts (ntfy topic URL, Discord/Slack webhook, or
    # any URL accepting a POST). Empty = in-dashboard toasts only.
    alert_webhook_url: str = ""
    # Polygon.io / Massive key for backtest history (research only; live scans stay
    # on yfinance). Free Basic plan = 5 calls/minute; set 0 for unlimited plans.
    polygon_api_key: str = ""
    polygon_calls_per_minute: float = 5.0
    # Opt-in: no new trades/alerts while SPY is below its 50-day average.
    market_regime_gate: bool = False


def _default_db_path() -> Path:
    # SQLite WAL files inside a OneDrive-synced folder cause lock/sync conflicts,
    # so the database defaults to LOCALAPPDATA instead of the project directory.
    local_app_data = os.getenv("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "StocksRealtimeScreening" / "stocks_screening.sqlite3"
    return Path("data/stocks_screening.sqlite3")


def _resolve_db_path(value: Optional[str] = None) -> Path:
    raw_path = value or str(_default_db_path())
    return Path(os.path.expanduser(os.path.expandvars(raw_path)))


def get_settings() -> AppSettings:
    load_dotenv()
    return AppSettings(
        db_path=_resolve_db_path(os.getenv("STOCKS_DB_PATH")),
        alert_webhook_url=(os.getenv("STOCKS_ALERT_WEBHOOK_URL") or "").strip(),
        polygon_api_key=(os.getenv("POLYGON_API_KEY") or "").strip(),
        polygon_calls_per_minute=float(os.getenv("POLYGON_CALLS_PER_MINUTE") or 5.0),
        market_regime_gate=(os.getenv("STOCKS_MARKET_REGIME_GATE") or "").strip().lower()
        in ("1", "true", "yes", "on"),
    )
