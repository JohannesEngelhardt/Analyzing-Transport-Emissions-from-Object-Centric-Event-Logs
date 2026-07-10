from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT_FILE = Path(
    "aux_event_log_overview_kodak_2026_05_04_new_e2o_no_layover.csv"
)
DEFAULT_OUTPUT_PREFIX = Path("trip_occupancy_changes_2026_05_04_new_e2o_no_layover")
OCCUPANCY_LEVELS = [0, 1, 2, 3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze whether occupancy_status changes within each trip over the "
            "ordered segment sequence."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help=f"Input aux event overview CSV. Default: {DEFAULT_INPUT_FILE}",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=DEFAULT_OUTPUT_PREFIX,
        help=(
            "Prefix for output CSV files. The script writes "
            "<prefix>_summary.csv and <prefix>_segment_sequence.csv."
        ),
    )
    parser.add_argument(
        "--include-event-sequence",
        action="store_true",
        help=(
            "Also write an event-level sequence file using every row with "
            "trip_id and occupancy_status, not only segment rows."
        ),
    )
    return parser.parse_args()


def read_input(input_file: Path) -> pd.DataFrame:
    required_columns = [
        "timestamp",
        "trip_id",
        "trip_id_org",
        "old_trip_id",
        "route_short_name",
        "route_id",
        "direction_id",
        "vehicle_id",
        "activity_type",
        "stop_sequence",
        "stop_id",
        "stop_name",
        "occupancy_status",
        "segment_id",
        "segment_departure_stop_id",
        "segment_arrive_stop_id",
        "cumulative_road_distance_km",
    ]
    df = pd.read_csv(
        input_file,
        usecols=lambda column: column in required_columns,
        dtype={
            "trip_id": "string",
            "trip_id_org": "string",
            "old_trip_id": "string",
            "route_short_name": "string",
            "route_id": "string",
            "direction_id": "string",
            "vehicle_id": "string",
            "activity_type": "string",
            "stop_id": "string",
            "stop_name": "string",
            "segment_id": "string",
            "segment_departure_stop_id": "string",
            "segment_arrive_stop_id": "string",
        },
    )

    missing_columns = sorted(set(required_columns) - set(df.columns))
    if missing_columns:
        raise ValueError(
            f"Missing required input columns in {input_file}: {missing_columns}"
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["occupancy_status"] = pd.to_numeric(
        df["occupancy_status"], errors="coerce"
    )
    df["stop_sequence"] = pd.to_numeric(df["stop_sequence"], errors="coerce")
    df["cumulative_road_distance_km"] = pd.to_numeric(
        df["cumulative_road_distance_km"], errors="coerce"
    )
    return df


def most_common_occupancy(values: pd.Series) -> float:
    counts = values.dropna().value_counts()
    if counts.empty:
        return np.nan
    return counts.sort_index().idxmax()


def first_non_null(values: pd.Series):
    non_null_values = values.dropna()
    if non_null_values.empty:
        return pd.NA
    return non_null_values.iloc[0]


def collapse_to_trip_segments(df: pd.DataFrame) -> pd.DataFrame:
    segment_events = df.dropna(
        subset=["trip_id", "segment_id", "occupancy_status"]
    ).copy()
    segment_events["occupancy_status"] = segment_events["occupancy_status"].astype(int)

    sort_columns = ["trip_id", "timestamp", "stop_sequence", "segment_id"]
    segment_events = segment_events.sort_values(sort_columns)

    segment_sequence = (
        segment_events.groupby(["trip_id", "segment_id"], as_index=False)
        .agg(
            first_timestamp=("timestamp", "min"),
            last_timestamp=("timestamp", "max"),
            event_count=("occupancy_status", "size"),
            occupancy_status=("occupancy_status", most_common_occupancy),
            first_occupancy_status=("occupancy_status", "first"),
            last_occupancy_status=("occupancy_status", "last"),
            route_short_name=("route_short_name", first_non_null),
            route_id=("route_id", first_non_null),
            direction_id=("direction_id", first_non_null),
            vehicle_id=("vehicle_id", first_non_null),
            trip_id_org=("trip_id_org", first_non_null),
            old_trip_id=("old_trip_id", first_non_null),
            activity_types=("activity_type", lambda values: "|".join(
                values.dropna().astype(str).unique()
            )),
            first_stop_sequence=("stop_sequence", "min"),
            first_stop_name=("stop_name", first_non_null),
            segment_departure_stop_id=("segment_departure_stop_id", first_non_null),
            segment_arrive_stop_id=("segment_arrive_stop_id", first_non_null),
            cumulative_road_distance_km=("cumulative_road_distance_km", "min"),
        )
    )
    segment_sequence["occupancy_status"] = (
        segment_sequence["occupancy_status"].astype(int)
    )
    segment_sequence = segment_sequence.sort_values(
        [
            "trip_id",
            "first_timestamp",
            "first_stop_sequence",
            "cumulative_road_distance_km",
            "segment_id",
        ]
    ).copy()
    segment_sequence["segment_order_in_trip"] = (
        segment_sequence.groupby("trip_id").cumcount() + 1
    )
    segment_sequence["previous_occupancy_status"] = (
        segment_sequence.groupby("trip_id")["occupancy_status"].shift()
    )
    segment_sequence["occupancy_delta_from_previous"] = (
        segment_sequence["occupancy_status"]
        - segment_sequence["previous_occupancy_status"]
    )
    segment_sequence["occupancy_changed_from_previous"] = (
        segment_sequence["occupancy_delta_from_previous"].fillna(0).ne(0)
    )
    return segment_sequence


def build_trip_summary(segment_sequence: pd.DataFrame) -> pd.DataFrame:
    if segment_sequence.empty:
        return pd.DataFrame()

    occupancy_counts = (
        segment_sequence.groupby(["trip_id", "occupancy_status"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=OCCUPANCY_LEVELS, fill_value=0)
    )
    occupancy_counts.columns = [
        f"segments_with_occupancy_{level}" for level in occupancy_counts.columns
    ]

    trip_summary = (
        segment_sequence.groupby("trip_id", as_index=False)
        .agg(
            route_short_name=("route_short_name", first_non_null),
            route_id=("route_id", first_non_null),
            direction_id=("direction_id", first_non_null),
            vehicle_id=("vehicle_id", first_non_null),
            trip_id_org=("trip_id_org", first_non_null),
            old_trip_id=("old_trip_id", first_non_null),
            first_timestamp=("first_timestamp", "min"),
            last_timestamp=("last_timestamp", "max"),
            segment_count=("segment_id", "size"),
            start_occupancy=("occupancy_status", "first"),
            end_occupancy=("occupancy_status", "last"),
            min_occupancy=("occupancy_status", "min"),
            max_occupancy=("occupancy_status", "max"),
            mean_occupancy=("occupancy_status", "mean"),
            median_occupancy=("occupancy_status", "median"),
            unique_occupancy_values=("occupancy_status", "nunique"),
            occupancy_sequence=(
                "occupancy_status",
                lambda values: " -> ".join(values.astype(str)),
            ),
            segment_sequence=("segment_id", lambda values: " -> ".join(
                values.astype(str)
            )),
        )
    )
    trip_summary["occupancy_range"] = (
        trip_summary["max_occupancy"] - trip_summary["min_occupancy"]
    )
    trip_summary["occupancy_changed"] = (
        trip_summary["unique_occupancy_values"] > 1
    )

    change_stats = (
        segment_sequence.assign(
            is_increase=lambda frame: frame["occupancy_delta_from_previous"] > 0,
            is_decrease=lambda frame: frame["occupancy_delta_from_previous"] < 0,
        )
        .groupby("trip_id")
        .agg(
            occupancy_change_count=("occupancy_changed_from_previous", "sum"),
            occupancy_increase_count=("is_increase", "sum"),
            occupancy_decrease_count=("is_decrease", "sum"),
            max_step_increase=("occupancy_delta_from_previous", "max"),
            max_step_decrease=("occupancy_delta_from_previous", "min"),
        )
        .fillna(
            {
                "max_step_increase": 0,
                "max_step_decrease": 0,
            }
        )
        .reset_index()
    )

    trip_summary = trip_summary.merge(change_stats, on="trip_id", how="left")
    trip_summary = trip_summary.merge(
        occupancy_counts.reset_index(), on="trip_id", how="left"
    )

    for level in OCCUPANCY_LEVELS:
        count_column = f"segments_with_occupancy_{level}"
        share_column = f"share_occupancy_{level}"
        trip_summary[count_column] = trip_summary[count_column].fillna(0).astype(int)
        trip_summary[share_column] = (
            trip_summary[count_column] / trip_summary["segment_count"]
        )

    trip_summary["mean_occupancy"] = trip_summary["mean_occupancy"].round(3)
    trip_summary["median_occupancy"] = trip_summary["median_occupancy"].round(3)
    trip_summary = trip_summary.sort_values(
        [
            "occupancy_changed",
            "occupancy_change_count",
            "occupancy_range",
            "segment_count",
        ],
        ascending=[False, False, False, False],
    )
    return trip_summary


def build_event_sequence(df: pd.DataFrame) -> pd.DataFrame:
    event_sequence = df.dropna(subset=["trip_id", "occupancy_status"]).copy()
    event_sequence["occupancy_status"] = event_sequence["occupancy_status"].astype(int)
    event_sequence = event_sequence.sort_values(
        ["trip_id", "timestamp", "stop_sequence", "segment_id"]
    )
    event_sequence["event_order_in_trip"] = event_sequence.groupby("trip_id").cumcount() + 1
    event_sequence["previous_occupancy_status"] = (
        event_sequence.groupby("trip_id")["occupancy_status"].shift()
    )
    event_sequence["occupancy_delta_from_previous"] = (
        event_sequence["occupancy_status"]
        - event_sequence["previous_occupancy_status"]
    )
    event_sequence["occupancy_changed_from_previous"] = (
        event_sequence["occupancy_delta_from_previous"].fillna(0).ne(0)
    )
    return event_sequence


def print_console_summary(trip_summary: pd.DataFrame, segment_sequence: pd.DataFrame) -> None:
    total_trips = len(trip_summary)
    changed_trips = int(trip_summary["occupancy_changed"].sum()) if total_trips else 0
    unchanged_trips = total_trips - changed_trips
    changed_share = changed_trips / total_trips if total_trips else 0

    print("\nTrip occupancy change analysis")
    print("=" * 31)
    print(f"Trips with segment occupancy data: {total_trips}")
    print(f"Trips with occupancy changes:      {changed_trips} ({changed_share:.1%})")
    print(f"Trips without occupancy changes:   {unchanged_trips}")
    print(f"Segment observations analyzed:      {len(segment_sequence)}")

    if total_trips:
        print("\nOccupancy change count per trip:")
        print(trip_summary["occupancy_change_count"].describe().round(2).to_string())

        print("\nTop trips with the most occupancy changes:")
        display_columns = [
            "trip_id",
            "route_short_name",
            "segment_count",
            "occupancy_change_count",
            "start_occupancy",
            "end_occupancy",
            "min_occupancy",
            "max_occupancy",
            "occupancy_sequence",
        ]
        print(trip_summary[display_columns].head(15).to_string(index=False))


def main() -> None:
    args = parse_args()
    df = read_input(args.input)
    segment_sequence = collapse_to_trip_segments(df)
    trip_summary = build_trip_summary(segment_sequence)

    summary_file = args.output_prefix.with_name(
        f"{args.output_prefix.name}_summary.csv"
    )
    sequence_file = args.output_prefix.with_name(
        f"{args.output_prefix.name}_segment_sequence.csv"
    )
    trip_summary.to_csv(summary_file, index=False)
    segment_sequence.to_csv(sequence_file, index=False)

    print_console_summary(trip_summary, segment_sequence)
    print(f"\nWrote trip summary:     {summary_file}")
    print(f"Wrote segment sequence: {sequence_file}")

    if args.include_event_sequence:
        event_sequence = build_event_sequence(df)
        event_sequence_file = args.output_prefix.with_name(
            f"{args.output_prefix.name}_event_sequence.csv"
        )
        event_sequence.to_csv(event_sequence_file, index=False)
        print(f"Wrote event sequence:   {event_sequence_file}")


if __name__ == "__main__":
    main()
