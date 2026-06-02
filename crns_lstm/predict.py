from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import load_aligned_series, make_sequences
from .model import LSTMRegressor


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("."))
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--years", type=int, nargs="+", default=[2022, 2023, 2024, 2025])
    p.add_argument("--freq", type=str, default="hourly", choices=["hourly", "daily"])
    p.add_argument(
        "--months",
        type=int,
        nargs="*",
        default=None,
        help="Optional month filter (e.g. --months 3 4 5 6 7 8 9 10).",
    )
    p.add_argument("--lookback", type=int, default=None, help="Override training lookback if needed.")
    p.add_argument("--out-csv", type=Path, required=True)
    args = p.parse_args()

    model_dir = args.model_dir
    scalers = np.load(model_dir / "scalers.npz")
    cfg_path = model_dir / "config.json"
    train_cfg = {}
    if cfg_path.exists():
        import json

        train_cfg = json.loads(cfg_path.read_text())
    lookback = args.lookback if args.lookback is not None else int(train_cfg.get("lookback") or 24)

    series = load_aligned_series(args.data_dir, years=args.years, freq=args.freq, months=args.months)
    X_seq, Y = make_sequences(series.x, series.y, lookback=lookback)
    ts_y = series.ts.iloc[lookback:].reset_index(drop=True).copy()

    # Rebuild scalers (StandardScaler params) from saved arrays
    x_mean = np.asarray(scalers["x_mean"], dtype=np.float32).reshape(1)
    x_scale = np.asarray(scalers["x_scale"], dtype=np.float32).reshape(1)
    y_mean = np.asarray(scalers["y_mean"], dtype=np.float32).reshape(1)
    y_scale = np.asarray(scalers["y_scale"], dtype=np.float32).reshape(1)

    X_s = ((X_seq.reshape(-1, 1) - x_mean) / x_scale).reshape(X_seq.shape).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMRegressor(
        input_size=1,
        hidden_size=int(train_cfg.get("hidden_size") or 64),
        num_layers=int(train_cfg.get("num_layers") or 2),
        dropout=float(train_cfg.get("dropout") or 0.1),
    ).to(device)
    state = torch.load(model_dir / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    preds = []
    with torch.no_grad():
        xb = torch.from_numpy(X_s).to(device)
        # Predict in manageable chunks
        bs = 4096
        for i in range(0, len(xb), bs):
            preds.append(model(xb[i : i + bs]).detach().cpu().numpy())
    y_pred_s = np.vstack(preds)
    y_pred = (y_pred_s * y_scale) + y_mean

    out = ts_y.copy()
    out["y_true"] = Y.reshape(-1)
    out["y_pred"] = y_pred.reshape(-1)
    out.to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    main()

