from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _parse_metrics(metrics_path: Path) -> dict[str, float]:
    txt = metrics_path.read_text()
    out: dict[str, float] = {}
    for line in txt.splitlines():
        m = re.match(r"^\s*([A-Za-z0-9_]+)\s*=\s*([+-]?[0-9]*\.?[0-9]+([eE][+-]?[0-9]+)?)\s*$", line)
        if not m:
            continue
        out[m.group(1)] = float(m.group(2))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("."))
    p.add_argument("--out-root", type=Path, default=Path("artifacts_sweep_daily"))
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    lookbacks = [3, 7, 10, 14]
    batch_sizes = [4, 8]

    # settings
    dropout = 0.1
    lr = 5e-4
    weight_decay = 1e-5
    patience = 10
    test_frac = 0.2  # 80% train, 20% test
    val_frac_in_train = 0.2

    args.out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for lb in lookbacks:
        for bs in batch_sizes:
            out_dir = args.out_root / f"lb{lb}_bs{bs}"
            out_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                sys.executable,
                "-m",
                "crns_lstm.train",
                "--data-dir",
                str(args.data_dir),
                "--out-dir",
                str(out_dir),
                "--years",
                "2022",
                "2023",
                "2024",
                "2025",
                "--freq",
                "daily",
                "--lookback",
                str(lb),
                "--test-frac",
                str(test_frac),
                "--val-frac-in-train",
                str(val_frac_in_train),
                "--dropout",
                str(dropout),
                "--lr",
                str(lr),
                "--weight-decay",
                str(weight_decay),
                "--batch-size",
                str(bs),
                "--patience",
                str(patience),
                "--epochs",
                str(args.epochs),
                "--seed",
                str(args.seed),
                "--shuffle-train",
            ]

            subprocess.run(cmd, check=True)
            metrics = _parse_metrics(out_dir / "metrics.txt")
            rows.append(
                {
                    "lookback_days": lb,
                    "batch_size": bs,
                    "dropout": dropout,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "patience": patience,
                    "test_frac": test_frac,
                    "R": metrics.get("R"),
                    "RMSD": metrics.get("RMSD"),
                    "out_dir": str(out_dir),
                }
            )

    df = pd.DataFrame(rows).sort_values(["R"], ascending=False)
    df.to_csv(args.out_root / "summary.csv", index=False)


if __name__ == "__main__":
    main()

