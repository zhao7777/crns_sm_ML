from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .data import load_aligned_series, make_sequences


def prepare_sequences(
    data_dir: Path,
    years: Iterable[int],
    *,
    freq: str = "hourly",
    months: Iterable[int] | None = None,
    lookback: int = 72,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    """Returns X_seq, Y, ts_y (per sequence row), and full aligned y (for persistence)."""
    series = load_aligned_series(data_dir, years=years, freq=freq, months=months)
    X_seq, Y = make_sequences(series.x, series.y, lookback=lookback)
    ts_y = series.ts.iloc[lookback:].reset_index(drop=True).copy()
    return X_seq, Y, ts_y, series.y


def split_indices_by_year(
    ts_y: pd.DataFrame,
    train_years: Iterable[int],
    test_years: Iterable[int],
    val_frac_in_train: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Chronological split: train years -> last val_frac for val, remainder train;
    test years -> test.
    """
    years = ts_y["year"].astype(int).to_numpy()
    train_years_set = {int(y) for y in train_years}
    test_years_set = {int(y) for y in test_years}

    train_pool = np.where(np.isin(years, list(train_years_set)))[0]
    test_idx = np.where(np.isin(years, list(test_years_set)))[0]

    if len(train_pool) == 0:
        raise ValueError(f"No training samples for years {train_years_set}")
    if len(test_idx) == 0:
        raise ValueError(f"No test samples for years {test_years_set}")

    val_n = int(round(len(train_pool) * val_frac_in_train))
    val_n = max(1, min(val_n, len(train_pool) - 1))
    train_idx = train_pool[:-val_n]
    val_idx = train_pool[-val_n:]
    return train_idx, val_idx, test_idx


def split_indices_by_fraction(
    n: int,
    test_frac: float,
    val_frac_in_train: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    test_n = int(round(n * test_frac))
    test_n = max(1, min(test_n, n - 1))
    train_full_n = n - test_n

    val_n = int(round(train_full_n * val_frac_in_train))
    val_n = max(1, min(val_n, train_full_n - 1))
    train_n = train_full_n - val_n

    train_idx = np.arange(0, train_n)
    val_idx = np.arange(train_n, train_full_n)
    test_idx = np.arange(train_full_n, n)
    return train_idx, val_idx, test_idx
