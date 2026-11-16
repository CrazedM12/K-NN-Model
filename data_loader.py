import os
import sqlite3
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DB_PATH = Path(os.getenv("NAS_DB_PATH", r"\\192.168.1.95\CS-CLASSROOM\intraday_5m.sqlite"))
DB_URI = "file:////" + DB_PATH.as_posix().lstrip("/") + "?mode=ro&immutable=1"


def load_bars(symbol, start, end):
    """Load 5-minute bars for one symbol. `end` is exclusive."""
    if not DB_PATH.is_file():
        raise FileNotFoundError(
            f"Cannot see {DB_PATH}. Connect to the NAS first."
        )

    with sqlite3.connect(DB_URI, uri=True) as conn:
        conn.execute("PRAGMA query_only = ON")
        df = pd.read_sql_query(
            """
            SELECT symbol, ts_utc, open, high, low, close, volume
            FROM bars_5m
            WHERE symbol = ? AND ts_utc >= ? AND ts_utc < ?
            ORDER BY ts_utc ASC
            """,
            conn,
            params=(symbol, start, end),
        )

    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    df["ts_et"] = df["ts_utc"].dt.tz_convert("America/New_York")
    return df.set_index("ts_utc")