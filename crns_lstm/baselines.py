from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .metrics import mae, pearson_r, rmsd
from .splits import prepare_sequences, split_indices_by_year


def _persistence_predict(y_full: np.ndarray, lookback: int, test_idx: np.ndarray) -> np.ndarray:
    """y(t) predicted by y(t-1) in the aligned hourly series."""
    preds = np.empty(len(test_idx), dtype=np.float64)
    for j, i in enumerate(test_idx):
        t_global = lookback + int(i)
        preds[j] = float(y_full[t_global - 1])
    return preds


def _flatten_features(X_seq: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return X_seq[idx].reshape(len(idx), -1)


def main() -> None:
    p = argparse.ArgumentParser(description="Baselines with same split as LSTM.")
    p.add_argument("--data-dir", type=Path, default=Path("."))
    p.add_argument("--out-dir", type=Path, default=Path("artifacts_baselines"))
    p.add_argument("--years", type=int, nargs="+", default=[2022, 2023, 2024, 2025])
    p.add_argument("--train-years", type=int, nargs="+", default=[2022, 2023, 2024])
    p.add_argument("--test-years", type=int, nargs="+", default=[2025])
    p.add_argument("--months", type=int, nargs="+", default=[3, 4, 5, 6, 7, 8, 9, 10])
    p.add_argument("--lookback", type=int, default=72)
    p.add_argument("--val-frac-in-train", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    X_seq, Y, ts_y, y_full = prepare_sequences(
        args.data_dir,
        args.years,
        freq="hourly",
        months=args.months,
        lookback=args.lookback,
    )

    train_idx, val_idx, test_idx = split_indices_by_year(
        ts_y,
        train_years=args.train_years,
        test_years=args.test_years,
        val_frac_in_train=args.val_frac_in_train,
    )

    y_test = Y[test_idx].reshape(-1)
    ts_test = ts_y.iloc[test_idx].reset_index(drop=True)

    X_train = _flatten_features(X_seq, train_idx)
    X_val = _flatten_features(X_seq, val_idx)
    X_test = _flatten_features(X_seq, test_idx)
    y_train = Y[train_idx].reshape(-1)
    y_val = Y[val_idx].reshape(-1)

    x_scaler = StandardScaler()
    X_train_s = x_scaler.fit_transform(X_train)
    X_val_s = x_scaler.transform(X_val)
    X_test_s = x_scaler.transform(X_test)

    rows: list[dict] = []

    # 1) Persistence
    y_pred = _persistence_predict(y_full, args.lookback, test_idx)
    rows.append(_row("persistence", y_test, y_pred, ts_test))

    # 2) Linear (Ridge)
    lin = Ridge(alpha=1.0, random_state=args.seed)
    lin.fit(X_train_s, y_train)
    y_pred = lin.predict(X_test_s)
    rows.append(_row("linear_ridge", y_test, y_pred, ts_test))

    # 3) Random Forest
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=args.seed,
    )
    rf.fit(X_train_s, y_train)
    y_pred = rf.predict(X_test_s)
    rows.append(_row("random_forest", y_test, y_pred, ts_test))

    # 4) XGBoost
    try:
        from xgboost import XGBRegressor

        xgb = XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=args.seed,
            n_jobs=-1,
        )
        xgb.fit(
            X_train_s,
            y_train,
            eval_set=[(X_val_s, y_val)],
            verbose=False,
        )
        y_pred = xgb.predict(X_test_s)
        rows.append(_row("xgboost", y_test, y_pred, ts_test))
    except ImportError:
        rows.append(
            {
                "model": "xgboost",
                "R": float("nan"),
                "RMSD": float("nan"),
                "MAE": float("nan"),
                "note": "xgboost not installed",
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)

    split_info = {
        "train_years": args.train_years,
        "test_years": args.test_years,
        "months": args.months,
        "lookback": args.lookback,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
    }
    (out_dir / "split_info.json").write_text(json.dumps(split_info, indent=2))
    print(summary.to_string(index=False))


def _row(name: str, y_true: np.ndarray, y_pred: np.ndarray, ts_test: pd.DataFrame) -> dict:
    return {
        "model": name,
        "R": pearson_r(y_true, y_pred),
        "RMSD": rmsd(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "n_test": len(y_true),
        "test_year_min": int(ts_test["year"].min()),
        "test_year_max": int(ts_test["year"].max()),
    }


if __name__ == "__main__":
    main()
