from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import load_yearly_merged_for_plot
from .metrics import mae, pearson_r, rmsd
from .model import LSTMRegressor
from .splits import prepare_sequences, split_indices_by_fraction, split_indices_by_year


@dataclass(frozen=True)
class TrainConfig:
    lookback: int
    test_frac: float
    val_frac_in_train: float
    freq: str
    batch_size: int
    hidden_size: int
    num_layers: int
    dropout: float
    lr: float
    weight_decay: float
    epochs: int
    patience: int
    min_delta: float
    seed: int


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _to_datetime_index(ts: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.to_datetime(
        {
            "year": ts["year"].astype(int),
            "month": ts["month"].astype(int),
            "day": ts["day"].astype(int),
            "hour": ts["hour"].astype(int),
        },
        errors="coerce",
    )


def _save_figures(
    out_dir: Path,
    history: pd.DataFrame,
    test_df: pd.DataFrame,
    r_value: float,
    rmsd_value: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1) Loss curves
    fig = plt.figure(figsize=(8, 4.5), dpi=160)
    plt.plot(history["epoch"], history["train_mse_scaled"], label="train MSE (scaled)")
    if "train_eval_mse_scaled" in history.columns:
        plt.plot(history["epoch"], history["train_eval_mse_scaled"], label="train(eval) MSE (scaled)")
    plt.plot(history["epoch"], history["val_mse_scaled"], label="val MSE (scaled)")
    if "test_mae" in history.columns:
        plt.plot(history["epoch"], history["test_mae"], label="test MAE (original units)", linestyle="--")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training / validation / test metrics")
    plt.grid(True, alpha=0.3)
    plt.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curves.png")
    plt.close(fig)

    # 2) Test time series (use a readable index on x-axis)
    fig = plt.figure(figsize=(10, 4.5), dpi=160)
    t = np.arange(len(test_df))
    plt.plot(t, test_df["y_true"].to_numpy(), label="y_true", linewidth=1.2)
    plt.plot(t, test_df["y_pred"].to_numpy(), label="y_pred", linewidth=1.2)
    plt.xlabel("test sample index (chronological)")
    plt.ylabel("SM (5th col)")
    plt.title("Test set time series")
    plt.grid(True, alpha=0.3)
    plt.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "test_timeseries.png")
    plt.close(fig)

    # 3) Scatter: y_true vs y_pred
    fig = plt.figure(figsize=(5.5, 5.5), dpi=160)
    yt = test_df["y_true"].to_numpy()
    yp = test_df["y_pred"].to_numpy()
    plt.scatter(yt, yp, s=6, alpha=0.35)
    mn = float(np.nanmin([yt.min(), yp.min()]))
    mx = float(np.nanmax([yt.max(), yp.max()]))
    plt.plot([mn, mx], [mn, mx], color="black", linewidth=1, alpha=0.7)
    plt.xlabel("y_true")
    plt.ylabel("y_pred")
    plt.title(f"Test scatter (r={r_value:.3f}, RMSD={rmsd_value:.4f})")
    plt.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "test_scatter.png")
    plt.close(fig)


def _save_yearly_dual_axis_plots(
    out_dir: Path,
    yearly: dict[int, pd.DataFrame],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for year, g in yearly.items():
        if "datetime" not in g.columns:
            continue
        g = g.dropna(subset=["datetime"]).sort_values(["year", "month", "day", "hour"], kind="mergesort")
        if len(g) < 2:
            continue
        if g["corec_x"].notna().sum() == 0 and g["sm_y"].notna().sum() == 0:
            continue

        fig, ax1 = plt.subplots(figsize=(11, 4.5), dpi=160)
        ax2 = ax1.twinx()

        ax1.plot(g["datetime"], g["corec_x"], color="tab:blue", linewidth=1.0, label="corec (X, 5th col)")
        ax2.plot(g["datetime"], g["sm_y"], color="tab:orange", linewidth=1.0, label="SM (y, 5th col)")

        ax1.set_xlabel("time")
        ax1.set_ylabel("corec (X)")
        ax2.set_ylabel("SM (y)")
        ax1.set_title(f"corec and SM time series ({int(year)})")
        ax1.grid(True, alpha=0.25)

        lines = ax1.get_lines() + ax2.get_lines()
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper right")

        fig.tight_layout()
        fig.savefig(out_dir / f"corec_sm_dualaxis_{int(year)}.png")
        plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("."))
    p.add_argument("--out-dir", type=Path, default=Path("artifacts_lstm_corec"))
    p.add_argument("--years", type=int, nargs="+", default=[2022, 2023, 2024, 2025])
    p.add_argument(
        "--months",
        type=int,
        nargs="*",
        default=None,
        help="Optional month filter (e.g. --months 3 4 5 6 7 8 9 10).",
    )

    p.add_argument("--freq", type=str, default="hourly", choices=["hourly", "daily"])
    p.add_argument("--lookback", type=int, default=24)
    p.add_argument("--test-frac", type=float, default=0.30)
    p.add_argument("--val-frac-in-train", type=float, default=0.20)
    p.add_argument(
        "--train-years",
        type=int,
        nargs="*",
        default=None,
        help="If set, use year-based split (e.g. 2022 2023 2024).",
    )
    p.add_argument(
        "--test-years",
        type=int,
        nargs="*",
        default=None,
        help="Test years when using --train-years (e.g. 2025).",
    )
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--shuffle-train", action="store_true", help="Shuffle training batches (no effect on val/test).")
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--min-delta", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cfg = TrainConfig(
        lookback=args.lookback,
        test_frac=args.test_frac,
        val_frac_in_train=args.val_frac_in_train,
        freq=args.freq,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        seed=args.seed,
    )

    _set_seed(cfg.seed)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    mpl_dir = out_dir / ".mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))

    yearly_for_plot = load_yearly_merged_for_plot(args.data_dir, years=args.years, freq=cfg.freq, months=args.months)

    X_seq, Y, ts_y, _y_full = prepare_sequences(
        args.data_dir,
        args.years,
        freq=cfg.freq,
        months=args.months,
        lookback=cfg.lookback,
    )

    n = len(Y)
    if args.train_years is not None and args.test_years is not None:
        train_idx, val_idx, test_idx = split_indices_by_year(
            ts_y,
            train_years=args.train_years,
            test_years=args.test_years,
            val_frac_in_train=cfg.val_frac_in_train,
        )
    else:
        train_idx, val_idx, test_idx = split_indices_by_fraction(
            n, cfg.test_frac, cfg.val_frac_in_train
        )

    X_train, X_val, X_test = X_seq[train_idx], X_seq[val_idx], X_seq[test_idx]
    y_train, y_val, y_test = Y[train_idx], Y[val_idx], Y[test_idx]
    ts_test = ts_y.iloc[test_idx].reset_index(drop=True)

    # Scale using training only
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_2d = X_train.reshape(-1, 1)
    x_scaler.fit(X_train_2d)
    X_train_s = x_scaler.transform(X_train_2d).reshape(X_train.shape).astype(np.float32)
    X_val_s = x_scaler.transform(X_val.reshape(-1, 1)).reshape(X_val.shape).astype(np.float32)
    X_test_s = x_scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape).astype(np.float32)

    y_scaler.fit(y_train)
    y_train_s = y_scaler.transform(y_train).astype(np.float32)
    y_val_s = y_scaler.transform(y_val).astype(np.float32)
    y_test_s = y_scaler.transform(y_test).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMRegressor(
        input_size=1,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)

    train_ds = TensorDataset(
        torch.from_numpy(X_train_s),
        torch.from_numpy(y_train_s),
    )
    val_ds = TensorDataset(torch.from_numpy(X_val_s), torch.from_numpy(y_val_s))
    test_ds = TensorDataset(torch.from_numpy(X_test_s), torch.from_numpy(y_test_s))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)
    if args.shuffle_train:
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=max(2, cfg.patience // 4), min_lr=1e-6
    )

    best_state = None
    best_val = float("inf")
    no_improve = 0
    history: list[dict] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        # Validation loss (eval mode)
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        scheduler.step(val_loss)

        # Train loss in eval mode (dropout OFF) to compare fairly with validation
        train_eval_losses = []
        with torch.no_grad():
            for xb, yb in DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=False):
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                train_eval_losses.append(loss.item())
        train_eval_loss = float(np.mean(train_eval_losses)) if train_eval_losses else float("nan")

        # Test MAE in original SM units (not used for early stopping)
        test_mae_val = float("nan")
        with torch.no_grad():
            test_preds_s = []
            for xb, _ in DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False):
                xb = xb.to(device)
                test_preds_s.append(model(xb).detach().cpu().numpy())
            if test_preds_s:
                y_pred_test = y_scaler.inverse_transform(np.vstack(test_preds_s))
                test_mae_val = mae(y_test, y_pred_test)

        history.append(
            {
                "epoch": epoch,
                "lr": float(opt.param_groups[0]["lr"]),
                "train_mse_scaled": train_loss,
                "train_eval_mse_scaled": train_eval_loss,
                "val_mse_scaled": val_loss,
                "test_mae": test_mae_val,
            }
        )

        if val_loss + cfg.min_delta < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Predict on test set (unscale to original y units)
    model.eval()
    preds_s = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            preds_s.append(model(xb).detach().cpu().numpy())
    y_pred_s = np.vstack(preds_s)
    y_pred = y_scaler.inverse_transform(y_pred_s)
    y_true = y_test

    R = pearson_r(y_true, y_pred)
    RMSD = rmsd(y_true, y_pred)
    MAE = mae(y_true, y_pred)

    history_df = pd.DataFrame(history)
    history_df.to_csv(out_dir / "history.csv", index=False)
    pred_df = ts_test.copy()
    pred_df["y_true"] = y_true.reshape(-1)
    pred_df["y_pred"] = y_pred.reshape(-1)
    pred_df.to_csv(out_dir / "test_predictions.csv", index=False)

    split_info = {
        "train_years": args.train_years,
        "test_years": args.test_years,
        "months": args.months,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
    }
    (out_dir / "split_info.json").write_text(json.dumps(split_info, indent=2))

    (out_dir / "metrics.txt").write_text(f"R={R:.6f}\nRMSD={RMSD:.6f}\nMAE={MAE:.6f}\n")
    _save_figures(out_dir, history_df, pred_df, R, RMSD)
    _save_yearly_dual_axis_plots(out_dir, yearly_for_plot)

    torch.save(model.state_dict(), out_dir / "model.pt")
    np.savez(
        out_dir / "scalers.npz",
        x_mean=x_scaler.mean_,
        x_scale=x_scaler.scale_,
        y_mean=y_scaler.mean_,
        y_scale=y_scaler.scale_,
    )
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))


if __name__ == "__main__":
    main()

