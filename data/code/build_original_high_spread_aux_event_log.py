from __future__ import annotations

from pathlib import Path

import pandas as pd

from build_iterative_severity_merge_aux_event_log import (
    EXTRA_COLUMNS,
    SOURCE_AUX_EVENT_LOG,
    SOURCE_OVERVIEW_LOG,
    build_full_event_log,
    build_overview_event_log,
)
from split_green_trip_paths_by_neighbors import load_selected_trip_segments


DATASET_DATE = "2026_05_04"
FULL_OUTPUT = Path(f"aux_event_log_emission_improved_high_spread_original_{DATASET_DATE}.csv")
OVERVIEW_OUTPUT = Path(
    f"aux_event_log_overview_emission_improved_high_spread_original_{DATASET_DATE}.csv"
)


def build_original_segments_and_paths() -> tuple[pd.DataFrame, pd.DataFrame]:
    segments = load_selected_trip_segments().copy()
    segments = segments.sort_values(
        ["trip_id", "departure_stop_sequence", "departure_timestamp"]
    ).reset_index(drop=True)
    segments["path_id"] = segments["trip_id"].astype(str)
    segments["source_trip_id"] = segments["trip_id"].astype(str)
    segments["path_color"] = (
        segments.groupby("trip_id")["occupancy_status"]
        .transform("max")
        .map({0: "green", 1: "yellow", 2: "orange", 3: "red"})
        .fillna("unknown")
    )
    segments["original_start_timestamp"] = segments.groupby("trip_id")[
        "departure_timestamp"
    ].transform("min")
    segments["segment_order"] = segments.groupby("trip_id").cumcount() + 1

    paths = (
        segments.groupby(["path_id", "source_trip_id", "path_color"], as_index=False)
        .agg(
            original_start_timestamp=("original_start_timestamp", "first"),
            start_timestamp=("departure_timestamp", "min"),
            end_timestamp=("arrive_timestamp", "max"),
            segment_count=("segment_id", "count"),
        )
        .sort_values(["original_start_timestamp", "path_id"])
        .reset_index(drop=True)
    )
    paths["status"] = "original_high_spread"
    paths["split_reason"] = "original_high_spread_no_iterative_merge"
    paths["merged_into_trip_ids"] = ""
    paths["merged_into_colors"] = ""
    paths["removed_nodes"] = ""
    return segments, paths


def main() -> None:
    segments, paths = build_original_segments_and_paths()
    aux = pd.read_csv(SOURCE_AUX_EVENT_LOG, dtype="string")
    overview_columns = list(pd.read_csv(SOURCE_OVERVIEW_LOG, nrows=0).columns)

    full_event_log = build_full_event_log(segments, paths, aux)
    overview_event_log = build_overview_event_log(full_event_log, overview_columns)

    full_event_log.to_csv(FULL_OUTPUT, index=False)
    overview_event_log.to_csv(OVERVIEW_OUTPUT, index=False)

    expected_events = len(segments) * 2
    print(f"Original high-spread segments used: {len(segments)}")
    print(f"Expected events: {expected_events}")
    print(f"Full event log rows: {len(full_event_log)}")
    print(f"Overview event log rows: {len(overview_event_log)}")
    print(f"Cases: {full_event_log['trip_id'].nunique()}")
    print(f"Full output: {FULL_OUTPUT}")
    print(f"Overview output: {OVERVIEW_OUTPUT}")
    print(f"Extra columns: {', '.join(EXTRA_COLUMNS)}")


if __name__ == "__main__":
    main()
