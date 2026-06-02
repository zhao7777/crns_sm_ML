#!/usr/bin/env python3
"""
8-panel figure: test-set time series + scatter for LSTM, RF, XGBoost, and Ridge.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from crns_lstm.metrics import mae, pearson_r, rmsd
from crns_lstm.splits import prepare_sequences, split_indices_by_year

FONT_SIZE = 8

PANEL_LABELS = ("a)", "b)", "c)", "d)")
MODEL_COLORS = {
    "LSTM": "tab:blue",
    "RF": "tab:orange",
    "XGB": "tab:green",
    "Ridge": "tab:red",
}


def _flatten_features(X_seq: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return X_seq[idx].reshape(len(idx), -1)


def _load_lstm_predictions(lstm_csv: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    df = pd.read_csv(lstm_csv)
    for col in ("y_true", "y_pred"):
        if col not in df.columns:
            raise ValueError(f"{lstm_csv} missing column {col!r}")
    ts = df[["year", "month", "day", "hour"]].copy()
    y_true = df["y_true"].to_numpy(dtype=np.float64)
    y_pred = df["y_pred"].to_numpy(dtype=np.float64)
    return y_true, y_pred, ts


def _fit_baseline_predictions(
    data_dir: Path,
    years: list[int],
    train_years: list[int],
    test_years: list[int],
    months: list[int],
    lookback: int,
    val_frac_in_train: float,
    seed: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    X_seq, Y, ts_y, _y_full = prepare_sequences(
        data_dir,
        years,
        freq="hourly",
        months=months,
        lookback=lookback,
    )
    train_idx, val_idx, test_idx = split_indices_by_year(
        ts_y,
        train_years=train_years,
        test_years=test_years,
        val_frac_in_train=val_frac_in_train,
    )

    X_train = _flatten_features(X_seq, train_idx)
    X_val = _flatten_features(X_seq, val_idx)
    X_test = _flatten_features(X_seq, test_idx)
    y_train = Y[train_idx].reshape(-1)
    y_val = Y[val_idx].reshape(-1)
    y_test = Y[test_idx].reshape(-1)

    x_scaler = StandardScaler()
    X_train_s = x_scaler.fit_transform(X_train)
    X_val_s = x_scaler.transform(X_val)
    X_test_s = x_scaler.transform(X_test)

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    ridge = Ridge(alpha=1.0, random_state=seed)
    ridge.fit(X_train_s, y_train)
    out["Ridge"] = (y_test, ridge.predict(X_test_s))

    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=seed,
    )
    rf.fit(X_train_s, y_train)
    out["RF"] = (y_test, rf.predict(X_test_s))

    from xgboost import XGBRegressor

    xgb = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=-1,
    )
    xgb.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)], verbose=False)
    out["XGB"] = (y_test, xgb.predict(X_test_s))

    return out


def _metrics_text(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    r = pearson_r(y_true, y_pred)
    return f"R={r:.3f}  RMSD={rmsd(y_true, y_pred):.4f}  MAE={mae(y_true, y_pred):.4f}"


def _plot_timeseries(
    ax: plt.Axes,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    panel_label: str,
    model_name: str,
    pred_color: str,
) -> None:
    t = np.arange(len(y_true))
    ax.plot(t, y_true, color="black", linewidth=0.5, label="observed")
    ax.plot(t, y_pred, color=pred_color, linewidth=0.5, label="predicted")
    ax.set_title(f"{panel_label} {model_name}", loc="left", fontsize=FONT_SIZE)
    ax.set_xlabel("index", fontsize=FONT_SIZE)
    ax.set_ylabel("SM", fontsize=FONT_SIZE)
    ax.tick_params(labelsize=FONT_SIZE)
    ax.legend(fontsize=FONT_SIZE, loc="upper right", frameon=False)


def _plot_scatter(
    ax: plt.Axes,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    pred_color: str,
    metrics: str,
) -> None:
    ax.scatter(
        y_true,
        y_pred,
        s=4,
        alpha=0.35,
        color=pred_color,
        linewidths=0,
        label=model_name,
    )
    mn = float(np.nanmin([y_true.min(), y_pred.min()]))
    mx = float(np.nanmax([y_true.max(), y_pred.max()]))
    ax.plot([mn, mx], [mn, mx], color="black", linewidth=0.8, linestyle="--", label="1:1")
    ax.set_title(metrics, fontsize=FONT_SIZE)
    ax.set_xlabel("observed SM", fontsize=FONT_SIZE)
    ax.set_ylabel("predicted SM", fontsize=FONT_SIZE)
    ax.tick_params(labelsize=FONT_SIZE)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=FONT_SIZE, loc="upper left", frameon=False)


def plot_test_panels(
    *,
    lstm_csv: Path,
    data_dir: Path,
    out_path: Path,
    years: list[int],
    train_years: list[int],
    test_years: list[int],
    months: list[int],
    lookback: int,
    val_frac_in_train: float,
    seed: int,
) -> None:
    y_true_lstm, y_pred_lstm, _ts = _load_lstm_predictions(lstm_csv)
    baseline_preds = _fit_baseline_predictions(
        data_dir=data_dir,
        years=years,
        train_years=train_years,
        test_years=test_years,
        months=months,
        lookback=lookback,
        val_frac_in_train=val_frac_in_train,
        seed=seed,
    )

    models: list[tuple[str, np.ndarray, np.ndarray]] = [
        ("LSTM", y_true_lstm, y_pred_lstm),
        ("RF", *baseline_preds["RF"]),
        ("XGB", *baseline_preds["XGB"]),
        ("Ridge", *baseline_preds["Ridge"]),
    ]

    for name, yt, yp in models[1:]:
        if len(yt) != len(y_true_lstm):
            raise ValueError(
                f"{name} test length {len(yt)} != LSTM test length {len(y_true_lstm)}"
            )
        if not np.allclose(yt, y_true_lstm, rtol=0, atol=1e-5, equal_nan=True):
            raise ValueError(f"{name} y_true differs from LSTM test_predictions y_true")

    fig, axes = plt.subplots(4, 2, figsize=(10, 10), dpi=200)
    for row, (name, y_true, y_pred) in enumerate(models):
        color = MODEL_COLORS[name]
        metrics = _metrics_text(y_true, y_pred)
        _plot_timeseries(
            axes[row, 0],
            y_true,
            y_pred,
            PANEL_LABELS[row],
            name,
            color,
        )
        _plot_scatter(axes[row, 1], y_true, y_pred, name, color, metrics)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="8-panel test plots: time series + scatter for LSTM/RF/XGB/Ridge."
    )
    p.add_argument("--data-dir", type=Path, default=Path("."))
    p.add_argument(
        "--lstm-csv",
        type=Path,
        default=Path("artifacts_train2024_test2025/lstm/test_predictions.csv"),
        help="LSTM test_predictions.csv (year, month, day, hour, y_true, y_pred).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("test_panels_lstm_rf_xgb_ridge.png"),
    )
    p.add_argument("--years", type=int, nargs="+", default=[2022, 2023, 2024, 2025])
    p.add_argument("--train-years", type=int, nargs="+", default=[2022, 2023, 2024])
    p.add_argument("--test-years", type=int, nargs="+", default=[2025])
    p.add_argument("--months", type=int, nargs="+", default=[3, 4, 5, 6, 7, 8, 9, 10])
    p.add_argument("--lookback", type=int, default=72)
    p.add_argument("--val-frac-in-train", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    plot_test_panels(
        lstm_csv=args.lstm_csv,
        data_dir=args.data_dir,
        out_path=args.out,
        years=args.years,
        train_years=args.train_years,
        test_years=args.test_years,
        months=args.months,
        lookback=args.lookback,
        val_frac_in_train=args.val_frac_in_train,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
