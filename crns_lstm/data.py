from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AlignedSeries:
    ts: pd.DataFrame  # columns: year, month, day, hour
    x: np.ndarray  # shape (N,)
    y: np.ndarray  # shape (N,)


def _read_5th_col(file_path: Path) -> pd.DataFrame:
    """
    Reads a whitespace-separated file where the first 4 columns are
    year, month, day, hour; and the 5th column is the value.
    """
    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        engine="python",
    )
    if df.shape[1] < 5:
        raise ValueError(f"{file_path} has <5 columns (found {df.shape[1]}).")
    out = df.iloc[:, :5].copy()
    out.columns = ["year", "month", "day", "hour", "value"]
    return out


def load_aligned_series(
    data_dir: Path,
    years: Iterable[int] = (2022, 2023, 2024, 2025),
    freq: str = "hourly",
    months: Iterable[int] | None = None,
) -> AlignedSeries:
    data_dir = Path(data_dir)

    corec_files: list[Path] = []
    sm_files: list[Path] = []
    for y in years:
        # 2023 corec file name differs in this workspace; accept both.
        corec_candidates = [
            data_dir / f"CR_corec_{y}_1hour_avegNC.txt",
            data_dir / f"CR_corec_{y}_1hour_aveg_NC.txt",
        ]
        corec = next((p for p in corec_candidates if p.exists()), None)
        if corec is None:
            raise FileNotFoundError(f"Missing corec file for {y}: {corec_candidates}")
        sm = _sm_path(data_dir, y)

        corec_files.append(corec)
        sm_files.append(sm)

    corec_df = pd.concat([_read_5th_col(p) for p in corec_files], ignore_index=True)
    sm_df = pd.concat([_read_5th_col(p) for p in sm_files], ignore_index=True)

    if freq not in {"hourly", "daily"}:
        raise ValueError("freq must be 'hourly' or 'daily'")

    keys = ["year", "month", "day", "hour"]
    # Some yearly files may contain duplicate timestamps. Aggregate to a single value.
    corec_df = corec_df.groupby(keys, as_index=False)["value"].mean(numeric_only=True)
    sm_df = sm_df.groupby(keys, as_index=False)["value"].mean(numeric_only=True)

    merged = corec_df.merge(
        sm_df,
        on=keys,
        how="inner",
        suffixes=("_x", "_y"),
    ).sort_values(keys, kind="mergesort")

    # Drop NaNs in either X or y
    merged = merged.replace([np.inf, -np.inf], np.nan)
    merged = merged.dropna(subset=["value_x", "value_y"]).reset_index(drop=True)

    if months is not None:
        months_set = {int(m) for m in months}
        merged = merged[merged["month"].astype(int).isin(months_set)].reset_index(drop=True)

    if freq == "daily":
        day_keys = ["year", "month", "day"]
        merged = (
            merged.groupby(day_keys, as_index=False)[["value_x", "value_y"]]
            .mean(numeric_only=True)
            .sort_values(day_keys, kind="mergesort")
            .reset_index(drop=True)
        )
        merged["hour"] = 0
        ts = merged[["year", "month", "day", "hour"]].copy()
    else:
        ts = merged[keys].copy()

    x = merged["value_x"].to_numpy(dtype=np.float32)
    y = merged["value_y"].to_numpy(dtype=np.float32)
    return AlignedSeries(ts=ts, x=x, y=y)


def _corec_path(data_dir: Path, year: int) -> Path:
    candidates = [
        data_dir / f"CR_corec_{year}_1hour_avegNC.txt",
        data_dir / f"CR_corec_{year}_1hour_aveg_NC.txt",
    ]
    corec = next((p for p in candidates if p.exists()), None)
    if corec is None:
        raise FileNotFoundError(f"Missing corec file for {year}: {candidates}")
    return corec


def _sm_path(data_dir: Path, year: int) -> Path:
    # Prefer the updated SMNEWvertical targets if present.
    candidates = [
        data_dir / f"CR_{year}_1hour_SMNEWvertical.txt",
        data_dir / f"CR_{year}_1hour_SMvertical.txt",
    ]
    sm = next((p for p in candidates if p.exists()), None)
    if sm is None:
        raise FileNotFoundError(f"Missing SMvertical/SMNEWvertical file for {year}: {candidates}")
    return sm


def load_yearly_merged_for_plot(
    data_dir: Path,
    years: Iterable[int] = (2022, 2023, 2024, 2025),
    freq: str = "hourly",
    months: Iterable[int] | None = None,
) -> dict[int, pd.DataFrame]:
    """
    Returns {year: df} where df has columns:
      year, month, day, hour, datetime, corec_x, sm_y

    Unlike load_aligned_series, this uses an OUTER join (so years with missing
    aligned X–y pairs still produce plots), and it does not drop NaNs.
    """
    if freq not in {"hourly", "daily"}:
        raise ValueError("freq must be 'hourly' or 'daily'")

    data_dir = Path(data_dir)
    out: dict[int, pd.DataFrame] = {}

    for year in years:
        corec_df = _read_5th_col(_corec_path(data_dir, year))
        sm_df = _read_5th_col(_sm_path(data_dir, year))

        # Aggregate duplicates per timestamp first.
        keys_h = ["year", "month", "day", "hour"]
        corec_df = corec_df.groupby(keys_h, as_index=False)["value"].mean(numeric_only=True)
        sm_df = sm_df.groupby(keys_h, as_index=False)["value"].mean(numeric_only=True)

        if freq == "daily":
            keys = ["year", "month", "day"]
            corec_df = corec_df.groupby(keys, as_index=False)["value"].mean(numeric_only=True)
            sm_df = sm_df.groupby(keys, as_index=False)["value"].mean(numeric_only=True)
            corec_df["hour"] = 0
            sm_df["hour"] = 0
        else:
            keys = keys_h

        merged = corec_df.merge(sm_df, on=keys, how="outer", suffixes=("_x", "_y")).sort_values(
            keys, kind="mergesort"
        )
        merged = merged.replace([np.inf, -np.inf], np.nan)
        if "hour" not in merged.columns:
            merged["hour"] = 0
        if months is not None:
            months_set = {int(m) for m in months}
            merged = merged[merged["month"].astype(int).isin(months_set)].reset_index(drop=True)
        merged["datetime"] = pd.to_datetime(
            {
                "year": merged["year"].astype("Int64"),
                "month": merged["month"].astype("Int64"),
                "day": merged["day"].astype("Int64"),
                "hour": merged["hour"].astype("Int64"),
            },
            errors="coerce",
        )
        merged = merged.rename(columns={"value_x": "corec_x", "value_y": "sm_y"})
        out[int(year)] = merged.reset_index(drop=True)

    return out


def make_sequences(
    x: np.ndarray,
    y: np.ndarray,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    One-step prediction: for each t, predict y[t] from x[t-lookback:t].

    Returns:
      X_seq: (N-lookback, lookback, 1)
      y_t:   (N-lookback, 1)
    """
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    if len(x) != len(y):
        raise ValueError("x and y must be same length")
    n = len(x)
    if n <= lookback:
        raise ValueError(f"Not enough data ({n}) for lookback={lookback}")

    X = np.zeros((n - lookback, lookback, 1), dtype=np.float32)
    Y = np.zeros((n - lookback, 1), dtype=np.float32)
    for i in range(lookback, n):
        X[i - lookback, :, 0] = x[i - lookback : i]
        Y[i - lookback, 0] = y[i]
    return X, Y

