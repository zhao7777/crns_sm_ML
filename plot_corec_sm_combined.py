#!/usr/bin/env python3
"""
Plot corec and SM on one dual-axis figure for 2022–2025, with vertical lines between years.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

FONT_SIZE = 8

from crns_lstm.data import load_yearly_merged_for_plot


def _concat_years(yearly: dict[int, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for year in sorted(yearly):
        g = yearly[year]
        if "datetime" not in g.columns:
            continue
        g = g.dropna(subset=["datetime"]).sort_values(
            ["year", "month", "day", "hour"], kind="mergesort"
        )
        if len(g) == 0:
            continue
        frames.append(g)
    if not frames:
        raise ValueError("No data to plot after loading yearly frames.")
    return pd.concat(frames, ignore_index=True)


def plot_corec_sm_combined(
    data_dir: Path,
    out_path: Path,
    years: list[int],
    freq: str = "hourly",
    months: list[int] | None = None,
) -> None:
    yearly = load_yearly_merged_for_plot(
        data_dir=data_dir,
        years=years,
        freq=freq,
        months=months,
    )
    g = _concat_years(yearly)
    if g["corec_x"].notna().sum() == 0 and g["sm_y"].notna().sum() == 0:
        raise ValueError("Both corec and SM are empty; nothing to plot.")

    fig, ax1 = plt.subplots(figsize=(12, 4), dpi=200)
    ax2 = ax1.twinx()

    ax1.plot(
        g["datetime"],
        g["corec_x"],
        color="gray",
        linewidth=0.5,
        label="CRNS",
    )
    ax2.plot(
        g["datetime"],
        g["sm_y"],
        color="black",
        linewidth=0.5,
        label="SM",
    )

    year_min, year_max = min(years), max(years)
    for y in range(year_min + 1, year_max + 1):
        boundary = pd.Timestamp(year=y, month=1, day=1)
        ax1.axvline(
            boundary,
            color="black",
            linewidth=1.0,
            linestyle="dashed",
            zorder=5,
            label="_nolegend_",
        )

    t_min = g["datetime"].min()
    t_max = g["datetime"].max()
    ax1.set_xlim(t_min, t_max)
    ax1.margins(x=0)

    ax1.set_xlabel("time", fontsize=FONT_SIZE)
    ax1.set_ylabel("CRNS", fontsize=FONT_SIZE)
    ax2.set_ylabel("SM", fontsize=FONT_SIZE)
    ax1.tick_params(axis="both", labelsize=FONT_SIZE)
    ax2.tick_params(axis="y", labelsize=FONT_SIZE)

    legend_handles = [
        Line2D([0], [0], color="gray", linewidth=0.5, label="CRNS"),
        Line2D([0], [0], color="black", linewidth=0.5, label="SM"),
    ]
    ax1.legend(handles=legend_handles, loc="upper right", fontsize=FONT_SIZE)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Plot corec and SM in one figure (2022–2025) with year separators."
    )
    p.add_argument("--data-dir", type=Path, default=Path("."))
    p.add_argument(
        "--out",
        type=Path,
        default=Path("corec_sm_dualaxis_2022_2025.png"),
        help="Output PNG path.",
    )
    p.add_argument("--years", type=int, nargs="+", default=[2022, 2023, 2024, 2025])
    p.add_argument("--freq", type=str, default="hourly", choices=["hourly", "daily"])
    p.add_argument(
        "--months",
        type=int,
        nargs="*",
        default=None,
        help="Optional month filter (e.g. --months 3 4 5 6 7 8 9 10).",
    )
    args = p.parse_args()
    months = None if args.months is None or len(args.months) == 0 else args.months

    plot_corec_sm_combined(
        data_dir=args.data_dir,
        out_path=args.out,
        years=args.years,
        freq=args.freq,
        months=months,
    )


if __name__ == "__main__":
    main()
