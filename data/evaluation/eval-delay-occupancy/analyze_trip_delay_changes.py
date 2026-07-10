from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT_FILE = Path(
    "aux_event_log_overview_kodak_2026_05_04_new_e2o_no_layover.csv"
)
DEFAULT_OUTPUT_PREFIX = Path("trip_delay_changes_2026_05_04_new_e2o_no_layover")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze how delay changes within each trip over the ordered "
            "segment sequence."
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
        "--change-threshold-seconds",
        type=float,
        default=0.0,
        help=(
            "Only count delay changes whose absolute delta is larger than this "
            "threshold. Default: 0, meaning every exact change counts."
        ),
    )
    parser.add_argument(
        "--include-event-sequence",
        action="store_true",
        help=(
            "Also write an event-level sequence file using every row with "
            "trip_id and delay, not only segment rows."
        ),
    )
    return parser.parse_args()


def first_non_null(values: pd.Series):
    non_null_values = values.dropna()
    if non_null_values.empty:
        return pd.NA
    return non_null_values.iloc[0]


def last_non_null(values: pd.Series):
    non_null_values = values.dropna()
    if non_null_values.empty:
        return pd.NA
    return non_null_values.iloc[-1]


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
        "delay",
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
    df["delay"] = pd.to_numeric(df["delay"], errors="coerce")
    df["stop_sequence"] = pd.to_numeric(df["stop_sequence"], errors="coerce")
    df["cumulative_road_distance_km"] = pd.to_numeric(
        df["cumulative_road_distance_km"], errors="coerce"
    )
    return df


def add_delay_change_columns(
    sequence: pd.DataFrame,
    order_column: str,
    change_threshold_seconds: float,
) -> pd.DataFrame:
    sequence = sequence.sort_values(
        ["trip_id", order_column, "timestamp_sort", "segment_id_sort"]
    ).copy()
    sequence["previous_delay"] = sequence.groupby("trip_id")["delay"].shift()
    sequence["delay_delta_from_previous"] = sequence["delay"] - sequence["previous_delay"]
    sequence["delay_changed_from_previous"] = (
        sequence["delay_delta_from_previous"]
        .abs()
        .fillna(0)
        .gt(change_threshold_seconds)
    )
    return sequence


def collapse_to_trip_segments(
    df: pd.DataFrame,
    change_threshold_seconds: float,
) -> pd.DataFrame:
    segment_events = df.dropna(subset=["trip_id", "segment_id", "delay"]).copy()
    segment_events = segment_events.sort_values(
        ["trip_id", "timestamp", "stop_sequence", "segment_id"]
    )

    segment_sequence = (
        segment_events.groupby(["trip_id", "segment_id"], as_index=False)
        .agg(
            first_timestamp=("timestamp", "min"),
            last_timestamp=("timestamp", "max"),
            event_count=("delay", "size"),
            delay=("delay", last_non_null),
            first_delay=("delay", first_non_null),
            last_delay=("delay", last_non_null),
            mean_delay=("delay", "mean"),
            min_delay_within_segment=("delay", "min"),
            max_delay_within_segment=("delay", "max"),
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
    segment_sequence["timestamp_sort"] = segment_sequence["first_timestamp"]
    segment_sequence["segment_id_sort"] = segment_sequence["segment_id"]
    segment_sequence = add_delay_change_columns(
        segment_sequence,
        "segment_order_in_trip",
        change_threshold_seconds,
    )
    return segment_sequence.drop(columns=["timestamp_sort", "segment_id_sort"])


def build_trip_summary(segment_sequence: pd.DataFrame) -> pd.DataFrame:
    if segment_sequence.empty:
        return pd.DataFrame()

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
            start_delay=("delay", "first"),
            end_delay=("delay", "last"),
            min_delay=("delay", "min"),
            max_delay=("delay", "max"),
            mean_delay=("delay", "mean"),
            median_delay=("delay", "median"),
            unique_delay_values=("delay", "nunique"),
            delay_sequence=("delay", lambda values: " -> ".join(
                values.round(0).astype(int).astype(str)
            )),
            segment_sequence=("segment_id", lambda values: " -> ".join(
                values.astype(str)
            )),
        )
    )
    trip_summary["delay_range"] = trip_summary["max_delay"] - trip_summary["min_delay"]
    trip_summary["net_delay_change"] = (
        trip_summary["end_delay"] - trip_summary["start_delay"]
    )
    trip_summary["delay_changed"] = trip_summary["unique_delay_values"] > 1

    change_stats = (
        segment_sequence.assign(
            is_delay_increase=lambda frame: frame["delay_delta_from_previous"] > 0,
            is_delay_decrease=lambda frame: frame["delay_delta_from_previous"] < 0,
        )
        .groupby("trip_id")
        .agg(
            delay_change_count=("delay_changed_from_previous", "sum"),
            delay_increase_count=("is_delay_increase", "sum"),
            delay_decrease_count=("is_delay_decrease", "sum"),
            max_step_delay_increase=("delay_delta_from_previous", "max"),
            max_step_delay_decrease=("delay_delta_from_previous", "min"),
        )
        .fillna(
            {
                "max_step_delay_increase": 0,
                "max_step_delay_decrease": 0,
            }
        )
        .reset_index()
    )

    trip_summary = trip_summary.merge(change_stats, on="trip_id", how="left")
    rounded_columns = [
        "start_delay",
        "end_delay",
        "min_delay",
        "max_delay",
        "mean_delay",
        "median_delay",
        "delay_range",
        "net_delay_change",
        "max_step_delay_increase",
        "max_step_delay_decrease",
    ]
    trip_summary[rounded_columns] = trip_summary[rounded_columns].round(2)
    trip_summary = trip_summary.sort_values(
        ["delay_change_count", "delay_range", "segment_count"],
        ascending=[False, False, False],
    )
    return trip_summary


def build_event_sequence(
    df: pd.DataFrame,
    change_threshold_seconds: float,
) -> pd.DataFrame:
    event_sequence = df.dropna(subset=["trip_id", "delay"]).copy()
    event_sequence = event_sequence.sort_values(
        ["trip_id", "timestamp", "stop_sequence", "segment_id"]
    )
    event_sequence["event_order_in_trip"] = event_sequence.groupby("trip_id").cumcount() + 1
    event_sequence["timestamp_sort"] = event_sequence["timestamp"]
    event_sequence["segment_id_sort"] = event_sequence["segment_id"]
    event_sequence = add_delay_change_columns(
        event_sequence,
        "event_order_in_trip",
        change_threshold_seconds,
    )
    return event_sequence.drop(columns=["timestamp_sort", "segment_id_sort"])


def print_console_summary(trip_summary: pd.DataFrame, segment_sequence: pd.DataFrame) -> None:
    total_trips = len(trip_summary)
    changed_trips = int(trip_summary["delay_changed"].sum()) if total_trips else 0
    unchanged_trips = total_trips - changed_trips
    changed_share = changed_trips / total_trips if total_trips else 0

    print("\nTrip delay change analysis")
    print("=" * 26)
    print(f"Trips with segment delay data: {total_trips}")
    print(f"Trips with delay changes:      {changed_trips} ({changed_share:.1%})")
    print(f"Trips without delay changes:   {unchanged_trips}")
    print(f"Segment observations analyzed: {len(segment_sequence)}")

    if total_trips:
        print("\nDelay change count per trip:")
        print(trip_summary["delay_change_count"].describe().round(2).to_string())

        print("\nTop trips with the most delay changes:")
        display_columns = [
            "trip_id",
            "route_short_name",
            "segment_count",
            "delay_change_count",
            "start_delay",
            "end_delay",
            "min_delay",
            "max_delay",
            "delay_range",
            "net_delay_change",
        ]
        print(trip_summary[display_columns].head(15).to_string(index=False))


def main() -> None:
    args = parse_args()
    df = read_input(args.input)
    segment_sequence = collapse_to_trip_segments(
        df,
        args.change_threshold_seconds,
    )
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
        event_sequence = build_event_sequence(df, args.change_threshold_seconds)
        event_sequence_file = args.output_prefix.with_name(
            f"{args.output_prefix.name}_event_sequence.csv"
        )
        event_sequence.to_csv(event_sequence_file, index=False)
        print(f"Wrote event sequence:   {event_sequence_file}")


if __name__ == "__main__":
    main()
