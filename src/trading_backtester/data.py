from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

REQUIRED_PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def _normalize_columns(data: pd.DataFrame) -> pd.DataFrame:
    if isinstance(data.columns, pd.MultiIndex):
        data = data.copy()
        data.columns = [
            col[0] if isinstance(col, tuple) else col for col in data.columns
        ]
    return data


def validate_price_data(data: pd.DataFrame) -> pd.DataFrame:
    normalized = _normalize_columns(data)
    missing = [
        column for column in REQUIRED_PRICE_COLUMNS if column not in normalized.columns
    ]
    if missing:
        raise ValueError(f"Price data is missing required columns: {missing}")
    cleaned = normalized[REQUIRED_PRICE_COLUMNS].sort_index().dropna()
    if cleaned.empty:
        raise ValueError("Price data is empty after cleaning")
    return cleaned


def fetch_price_data(
    symbol: str,
    start_date: str | None,
    end_date: str | None,
    *,
    interval: str = "1d",
    cache_dir: str | Path = "data/raw",
    refresh: bool = False,
) -> pd.DataFrame:
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    file_path = cache_path / f"{symbol}_{interval}.csv"

    if file_path.exists() and not refresh:
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)
        return validate_price_data(data)

    data = yf.download(
        symbol,
        start=start_date,
        end=end_date,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )
    validated = validate_price_data(data)
    validated.to_csv(file_path)
    return validated


def load_price_data(path: str | Path) -> pd.DataFrame:
    return validate_price_data(pd.read_csv(path, index_col=0, parse_dates=True))
