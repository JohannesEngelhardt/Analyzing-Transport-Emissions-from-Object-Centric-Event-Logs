from __future__ import annotations

import argparse
import os
from pathlib import Path


os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_DELAY_SUMMARY = Path(
    "trip_delay_changes_2026_05_04_new_e2o_no_layover_summary.csv"
)
DEFAULT_OCCUPANCY_SUMMARY = Path(
    "trip_occupancy_changes_2026_05_04_new_e2o_no_layover_summary.csv"
)
DEFAULT_OUTPUT_DIR = Path("unique_values_per_trip_charts")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the number of unique delay and occupancy values per trip."
        )
    )
    parser.add_argument(
        "--delay-summary",
        type=Path,
        default=DEFAULT_DELAY_SUMMARY,
        help=f"Trip delay summary CSV. Default: {DEFAULT_DELAY_SUMMARY}",
    )
    parser.add_argument(
        "--occupancy-summary",
        type=Path,
        default=DEFAULT_OCCUPANCY_SUMMARY,
        help=f"Trip occupancy summary CSV. Default: {DEFAULT_OCCUPANCY_SUMMARY}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for chart files. Default: {DEFAULT_OUTPUT_DIR}",
    )
    return parser.parse_args()


def load_series(summary_file: Path, column: str) -> pd.Series:
    df = pd.read_csv(summary_file, usecols=[column])
    series = pd.to_numeric(df[column], errors="coerce").dropna().astype(int)
    if series.empty:
        raise ValueError(f"No values found in column {column!r} of {summary_file}")
    return series


def short_number(value):
    if pd.isna(value):
        return ""
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f} M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f} k"
    if value.is_integer():
        return f"{int(value)}"
    return f"{value:.1f}"


def add_short_labels(ax: plt.Axes) -> None:
    for container in ax.containers:
        labels = [short_number(value) for value in container.datavalues]
        ax.bar_label(container, labels=labels, padding=3, fontsize=8)


def save_discrete_count_plot(
    series: pd.Series,
    title: str,
    xlabel: str,
    output_base: Path,
    color: str,
) -> None:
    counts = series.value_counts().sort_index()
    plot_df = counts.reset_index()
    plot_df.columns = ["unique_values", "trip_count"]

    plt.figure(figsize=(14, 6))
    ax = sns.barplot(
        data=plot_df,
        x="unique_values",
        y="trip_count",
        color=color,
        edgecolor="#1f2937",
        linewidth=0.7,
    )
    add_short_labels(ax)

    plt.xticks(rotation=60, ha="right", fontsize=8)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Number of Trips")
    plt.margins(y=0.12)

    plt.tight_layout()
    plt.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    delay_unique_values = load_series(args.delay_summary, "unique_delay_values")
    occupancy_unique_values = load_series(
        args.occupancy_summary,
        "unique_occupancy_values",
    )

    delay_output = args.output_dir / "unique_delay_values_per_trip"
    occupancy_output = args.output_dir / "unique_occupancy_values_per_trip"

    save_discrete_count_plot(
        delay_unique_values,
        "Number of Unique Delay Values per Trip",
        "Unique delay values per trip",
        delay_output,
        "indianred",
    )
    save_discrete_count_plot(
        occupancy_unique_values,
        "Number of Unique Occupancy Values per Trip",
        "Unique occupancy values per trip",
        occupancy_output,
        "mediumpurple",
    )

    print(f"Wrote {delay_output.with_suffix('.png')}")
    print(f"Wrote {delay_output.with_suffix('.svg')}")
    print(f"Wrote {occupancy_output.with_suffix('.png')}")
    print(f"Wrote {occupancy_output.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
