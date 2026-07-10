from __future__ import annotations

from pathlib import Path

import pandas as pd


DATASET_DATE = "2026_05_04"
OUTPUT_DIR = Path("hourly_occupancy_distribution_2026_05_04")
SEGMENTS_FILE = (
    OUTPUT_DIR
    / f"iterative_severity_merge_trip_path_segments_hour_06_{DATASET_DATE}.csv"
)
PATHS_FILE = OUTPUT_DIR / f"iterative_severity_merge_trip_paths_hour_06_{DATASET_DATE}.csv"
SOURCE_AUX_EVENT_LOG = Path(f"aux_event_log_kodak_{DATASET_DATE}.csv")
SOURCE_OVERVIEW_LOG = Path(f"aux_event_log_overview_kodak_{DATASET_DATE}.csv")

FULL_OUTPUT = Path(
    f"aux_event_log_emission_improved_iterative_severity_merge_{DATASET_DATE}.csv"
)
OVERVIEW_OUTPUT = Path(
    f"aux_event_log_overview_emission_improved_iterative_severity_merge_{DATASET_DATE}.csv"
)

EXTRA_COLUMNS = [
    "path_id",
    "source_trip_id",
    "path_color",
    "original_start_timestamp",
    "iterative_segment_order",
    "iterative_source_segment_id",
    "iterative_path_status",
    "iterative_split_reason",
    "merged_into_trip_ids",
    "merged_into_colors",
    "removed_nodes",
]


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    segments = pd.read_csv(SEGMENTS_FILE, dtype="string")
    paths = pd.read_csv(PATHS_FILE, dtype="string")
    aux = pd.read_csv(SOURCE_AUX_EVENT_LOG, dtype="string")
    overview_columns = list(pd.read_csv(SOURCE_OVERVIEW_LOG, nrows=0).columns)
    return segments, paths, aux, overview_columns


def parse_timestamps(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    parsed = df.copy()
    for column in columns:
        if column in parsed.columns:
            parsed[f"__{column}_dt"] = pd.to_datetime(parsed[column], errors="coerce")
    return parsed


def event_sort_key(activity_type: str) -> int:
    if activity_type == "departure_stop":
        return 0
    if activity_type == "arrive_stop":
        return 1
    return 2


def select_event(
    aux: pd.DataFrame,
    source_trip_id: str,
    segment_id: str,
    activity_type: str,
    timestamp: pd.Timestamp,
    stop_name: str,
) -> pd.Series:
    candidates = aux[
        (aux["trip_id"] == source_trip_id)
        & (aux["segment_id"] == segment_id)
        & (aux["activity_type"] == activity_type)
    ].copy()
    if candidates.empty:
        raise ValueError(
            f"No {activity_type} event for {source_trip_id} / {segment_id}"
        )

    timestamp_column = "__timestamp_dt"
    exact_time = candidates[candidates[timestamp_column] == timestamp]
    if not exact_time.empty:
        candidates = exact_time

    exact_stop = candidates[candidates["stop_name"] == stop_name]
    if not exact_stop.empty:
        candidates = exact_stop

    return candidates.sort_values(timestamp_column).iloc[0]


def build_full_event_log(
    segments: pd.DataFrame,
    paths: pd.DataFrame,
    aux: pd.DataFrame,
) -> pd.DataFrame:
    aux = parse_timestamps(aux, ["timestamp", "arrival_time_rt"])
    path_meta = paths.set_index("path_id", drop=False).to_dict(orient="index")

    output_rows = []
    event_counter = 1
    for segment in segments.itertuples(index=False):
        departure_timestamp = pd.to_datetime(segment.departure_timestamp, errors="coerce")
        arrive_timestamp = pd.to_datetime(segment.arrive_timestamp, errors="coerce")
        meta = path_meta[str(segment.path_id)]

        event_specs = [
            (
                "departure_stop",
                departure_timestamp,
                str(segment.departure_stop_name),
            ),
            (
                "arrive_stop",
                arrive_timestamp,
                str(segment.arrive_stop_name),
            ),
        ]
        for activity_type, timestamp, stop_name in event_specs:
            source_event = select_event(
                aux,
                source_trip_id=str(segment.source_trip_id),
                segment_id=str(segment.segment_id),
                activity_type=activity_type,
                timestamp=timestamp,
                stop_name=stop_name,
            )
            row = {
                column: source_event[column]
                for column in aux.columns
                if not column.startswith("__")
            }
            row["event_id"] = f"ism_{event_counter}"
            row["trip_id"] = str(segment.path_id)
            if "trip_id_unique" in row:
                row["trip_id_unique"] = str(segment.path_id)
            row["occupancy_status"] = str(segment.occupancy_status)
            row["path_id"] = str(segment.path_id)
            row["source_trip_id"] = str(segment.source_trip_id)
            row["path_color"] = str(segment.path_color)
            row["original_start_timestamp"] = str(segment.original_start_timestamp)
            row["iterative_segment_order"] = str(segment.segment_order)
            row["iterative_source_segment_id"] = str(segment.segment_id)
            row["iterative_path_status"] = str(meta.get("status", ""))
            row["iterative_split_reason"] = str(meta.get("split_reason", ""))
            row["merged_into_trip_ids"] = str(meta.get("merged_into_trip_ids", ""))
            row["merged_into_colors"] = str(meta.get("merged_into_colors", ""))
            row["removed_nodes"] = str(meta.get("removed_nodes", ""))
            output_rows.append(row)
            event_counter += 1

    event_log = pd.DataFrame(output_rows)
    event_log["__timestamp_dt"] = pd.to_datetime(event_log["timestamp"], errors="coerce")
    event_log["__original_start_dt"] = pd.to_datetime(
        event_log["original_start_timestamp"],
        errors="coerce",
    )
    event_log["__activity_order"] = event_log["activity_type"].map(event_sort_key)
    event_log["__segment_order"] = pd.to_numeric(
        event_log["iterative_segment_order"],
        errors="coerce",
    )
    event_log = event_log.sort_values(
        [
            "__original_start_dt",
            "trip_id",
            "__segment_order",
            "__timestamp_dt",
            "__activity_order",
        ]
    ).drop(
        columns=[
            "__timestamp_dt",
            "__original_start_dt",
            "__activity_order",
            "__segment_order",
        ]
    )
    original_columns = [
        column
        for column in aux.columns
        if not column.startswith("__")
    ]
    return event_log[original_columns + EXTRA_COLUMNS].reset_index(drop=True)


def build_overview_event_log(
    full_event_log: pd.DataFrame,
    overview_columns: list[str],
) -> pd.DataFrame:
    overview = full_event_log.copy()
    if "old_trip_id" in overview_columns and "old_trip_id" not in overview.columns:
        overview["old_trip_id"] = overview["source_trip_id"]

    for column in overview_columns:
        if column not in overview.columns:
            overview[column] = pd.NA

    return overview[overview_columns + EXTRA_COLUMNS].copy()


def main() -> None:
    segments, paths, aux, overview_columns = read_inputs()
    full_event_log = build_full_event_log(segments, paths, aux)
    overview_event_log = build_overview_event_log(full_event_log, overview_columns)

    full_event_log.to_csv(FULL_OUTPUT, index=False)
    overview_event_log.to_csv(OVERVIEW_OUTPUT, index=False)

    expected_events = len(segments) * 2
    print(f"Segments used: {len(segments)}")
    print(f"Expected events: {expected_events}")
    print(f"Full event log rows: {len(full_event_log)}")
    print(f"Overview event log rows: {len(overview_event_log)}")
    print(f"Cases: {full_event_log['trip_id'].nunique()}")
    print(f"Full output: {FULL_OUTPUT}")
    print(f"Overview output: {OVERVIEW_OUTPUT}")


if __name__ == "__main__":
    main()
