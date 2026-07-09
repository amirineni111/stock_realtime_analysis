from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel
import os


class AppSettings(BaseModel):
    db_path: Path = Path("data/stocks_screening.sqlite3")
    request_timeout_seconds: float = 20.0


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
    return AppSettings(db_path=_resolve_db_path(os.getenv("STOCKS_DB_PATH")))
