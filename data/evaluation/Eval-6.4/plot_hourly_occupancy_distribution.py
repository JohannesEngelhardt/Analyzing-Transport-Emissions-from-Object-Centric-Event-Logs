from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DATASET_DATE = "2026_05_04"
INPUT_FILE = Path(f"aux_event_log_overview_kodak_{DATASET_DATE}.csv")
OUTPUT_DIR = Path(f"hourly_occupancy_distribution_{DATASET_DATE}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_occupancy_events(input_file: Path) -> pd.DataFrame:
    df = pd.read_csv(
        input_file,
        usecols=[
            "timestamp",
            "occupancy_status",
            "route_type",
            "route_short_name",
            "trip_id",
            "segment_id",
            "stop_name",
            "activity_type",
        ],
        dtype={
            "occupancy_status": "string",
            "route_type": "string",
            "route_short_name": "string",
            "trip_id": "string",
            "segment_id": "string",
            "stop_name": "string",
            "activity_type": "string",
        },
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["occupancy_status"] = pd.to_numeric(df["occupancy_status"], errors="coerce")
    df = df.dropna(subset=["timestamp", "occupancy_status"]).copy()
    df["occupancy_status"] = df["occupancy_status"].astype(int)
    df["hour"] = df["timestamp"].dt.hour
    return df


def join_unique(values: pd.Series) -> str:
    unique_values = values.dropna().astype(str).unique()
    return ", ".join(unique_values[:8])


def build_hourly_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = (
        df.groupby(["hour", "occupancy_status"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    counts = counts.reindex(index=range(24), fill_value=0)
    counts = counts.reindex(columns=range(4), fill_value=0)
    counts.columns = [f"occupancy_{column}" for column in counts.columns]
    counts.index.name = "hour"
    counts = counts.astype(int)

    percentages = counts.div(counts.sum(axis=1).replace(0, pd.NA), axis=0) * 100
    percentages = percentages.fillna(0).astype(float)
    return counts, percentages


def build_hourly_spread_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("hour", as_index=False)
        .agg(
            event_count=("occupancy_status", "size"),
            mean_occupancy=("occupancy_status", "mean"),
            std_occupancy=("occupancy_status", "std"),
            min_occupancy=("occupancy_status", "min"),
            max_occupancy=("occupancy_status", "max"),
            unique_occupancy_values=("occupancy_status", "nunique"),
        )
        .fillna({"std_occupancy": 0})
    )
    summary["occupancy_range"] = summary["max_occupancy"] - summary["min_occupancy"]
    return summary.sort_values(
        ["std_occupancy", "occupancy_range", "event_count"],
        ascending=[False, False, False],
    )


def build_high_spread_segments(
    df: pd.DataFrame,
    selected_hours: list[int] | None,
    min_events_per_segment: int,
    top_n_per_hour: int,
) -> pd.DataFrame:
    segment_df = df.dropna(subset=["segment_id"]).copy()
    if selected_hours is not None:
        segment_df = segment_df[segment_df["hour"].isin(selected_hours)]

    grouped = (
        segment_df.groupby(["hour", "segment_id"], as_index=False)
        .agg(
            event_count=("occupancy_status", "size"),
            mean_occupancy=("occupancy_status", "mean"),
            std_occupancy=("occupancy_status", "std"),
            min_occupancy=("occupancy_status", "min"),
            max_occupancy=("occupancy_status", "max"),
            unique_occupancy_values=("occupancy_status", "nunique"),
            route_short_names=("route_short_name", join_unique),
            route_types=("route_type", join_unique),
            trip_ids=("trip_id", join_unique),
            stop_names=("stop_name", join_unique),
            activity_types=("activity_type", join_unique),
            first_timestamp=("timestamp", "min"),
            last_timestamp=("timestamp", "max"),
        )
        .fillna({"std_occupancy": 0})
    )
    grouped["occupancy_range"] = grouped["max_occupancy"] - grouped["min_occupancy"]

    occupancy_counts = (
        segment_df.groupby(["hour", "segment_id", "occupancy_status"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=range(4), fill_value=0)
    )
    occupancy_counts.columns = [f"occupancy_{column}_count" for column in occupancy_counts.columns]
    occupancy_counts = occupancy_counts.reset_index()

    grouped = grouped.merge(
        occupancy_counts,
        on=["hour", "segment_id"],
        how="left",
    )
    grouped = grouped[
        (grouped["event_count"] >= min_events_per_segment)
        & (grouped["unique_occupancy_values"] > 1)
    ].copy()

    if grouped.empty:
        return grouped

    grouped["coefficient_of_variation"] = (
        grouped["std_occupancy"] / grouped["mean_occupancy"].replace(0, np.nan)
    ).fillna(0)
    grouped = grouped.sort_values(
        [
            "hour",
            "std_occupancy",
            "occupancy_range",
            "unique_occupancy_values",
            "event_count",
        ],
        ascending=[True, False, False, False, False],
    )
    return grouped.groupby("hour", as_index=False, group_keys=False).head(top_n_per_hour)


def save_stacked_percentage_plot(percentages: pd.DataFrame, output_file: Path) -> None:
    colors = ["#6c757d", "#4c78a8", "#f58518", "#e45756"]
    ax = percentages.plot(
        kind="bar",
        stacked=True,
        figsize=(13, 6),
        color=colors,
        width=0.85,
    )
    ax.set_title("Hourly occupancy distribution")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Share of events (%)")
    ax.set_ylim(0, 100)
    ax.legend(
        title="Occupancy status",
        labels=["0", "1", "2", "3"],
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
    )
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def save_count_heatmap(counts: pd.DataFrame, output_file: Path) -> None:
    plt.figure(figsize=(8, 10))
    sns.heatmap(
        counts,
        annot=True,
        fmt=".0f",
        cmap="YlGnBu",
        linewidths=0.4,
        cbar_kws={"label": "Event count"},
    )
    plt.title("Hourly occupancy counts")
    plt.xlabel("Occupancy status")
    plt.ylabel("Hour of day")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def save_percentage_heatmap(percentages: pd.DataFrame, output_file: Path) -> None:
    plt.figure(figsize=(8, 10))
    sns.heatmap(
        percentages,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        linewidths=0.4,
        cbar_kws={"label": "Share of events (%)"},
    )
    plt.title("Hourly occupancy shares")
    plt.xlabel("Occupancy status")
    plt.ylabel("Hour of day")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot hourly occupancy distributions and list segments with high "
            "occupancy spread for selected hours."
        )
    )
    parser.add_argument("--hour", type=int, action="append", help="Hour to inspect, 0-23. Can be repeated.")
    parser.add_argument("--min-events-per-segment", type=int, default=4)
    parser.add_argument("--top-n-per-hour", type=int, default=25)
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_occupancy_events(INPUT_FILE)
    counts, percentages = build_hourly_tables(df)
    hourly_spread = build_hourly_spread_summary(df)
    selected_hours = sorted(set(args.hour)) if args.hour else None
    invalid_hours = [hour for hour in selected_hours or [] if hour < 0 or hour > 23]
    if invalid_hours:
        raise ValueError(f"Invalid hour values: {invalid_hours}. Use 0-23.")
    high_spread_segments = build_high_spread_segments(
        df,
        selected_hours=selected_hours,
        min_events_per_segment=args.min_events_per_segment,
        top_n_per_hour=args.top_n_per_hour,
    )

    counts.to_csv(OUTPUT_DIR / "hourly_occupancy_counts.csv")
    percentages.to_csv(OUTPUT_DIR / "hourly_occupancy_percentages.csv")
    hourly_spread.to_csv(OUTPUT_DIR / "hourly_occupancy_spread_summary.csv", index=False)
    if selected_hours is None:
        high_spread_file = OUTPUT_DIR / "high_spread_segments_all_hours.csv"
    else:
        hour_label = "_".join(f"{hour:02d}" for hour in selected_hours)
        high_spread_file = OUTPUT_DIR / f"high_spread_segments_hour_{hour_label}.csv"
    high_spread_segments.to_csv(high_spread_file, index=False)

    if not args.skip_plots:
        save_stacked_percentage_plot(
            percentages,
            OUTPUT_DIR / "hourly_occupancy_distribution_stacked_percent.png",
        )
        save_count_heatmap(
            counts,
            OUTPUT_DIR / "hourly_occupancy_counts_heatmap.png",
        )
        save_percentage_heatmap(
            percentages,
            OUTPUT_DIR / "hourly_occupancy_percentages_heatmap.png",
        )

    print(f"Events with occupancy: {len(df):,}")
    print(f"Written output directory: {OUTPUT_DIR}")
    print(f"High-spread segment output: {high_spread_file}")
    print("\nHours with highest occupancy spread:")
    print(hourly_spread.head(10).to_string(index=False))
    if not high_spread_segments.empty:
        print("\nTop high-spread segments:")
        print(
            high_spread_segments.head(20)[
                [
                    "hour",
                    "segment_id",
                    "event_count",
                    "std_occupancy",
                    "occupancy_range",
                    "route_short_names",
                    "route_types",
                    "trip_ids",
                    "stop_names",
                ]
            ].to_string(index=False)
        )
    print(counts.to_string())


if __name__ == "__main__":
    main()
