from __future__ import annotations

from pathlib import Path

import pandas as pd

from build_high_spread_trip_process_paths import (
    DATASET_DATE,
    DEFAULT_OUTPUT_DIR,
    build_pm4py_event_log,
    build_trip_path_graph,
    sort_trips_by_start_timestamp,
)
from render_split_green_process_paths import sort_event_df_by_path_start
from split_green_trip_paths_by_neighbors import (
    HOUR,
    TripPath,
    activity_key,
    build_trip_path,
    load_selected_trip_segments,
)


PAIR_RULES = [
    ("tr03_7353", "tr03_3878"),
    ("tr03_6882", "tr03_6074"),
]


def overlap_suffix_start(green_path: TripPath, reference_path: TripPath) -> int | None:
    reference_keys = set(reference_path.node_keys)
    overlap_indexes = [
        index
        for index, node_key in enumerate(green_path.node_keys)
        if node_key in reference_keys
    ]
    if not overlap_indexes:
        return None

    expected_suffix = list(range(overlap_indexes[0], len(green_path.node_keys)))
    if overlap_indexes != expected_suffix:
        return None
    return overlap_indexes[0]


def unchanged_record(path: TripPath, reason: str) -> dict:
    return {
        "path_id": path.trip_id,
        "source_trip_id": path.trip_id,
        "path_color": path.color,
        "max_occupancy": path.max_occupancy,
        "start_timestamp": path.start_timestamp,
        "end_timestamp": path.segments["arrive_timestamp"].max(),
        "segment_count": len(path.segments),
        "node_count": len(path.node_names),
        "status": "unchanged",
        "split_reason": reason,
        "neighbor_trip_ids": "",
        "neighbor_colors": "",
        "removed_nodes": "",
        "node_sequence": " -> ".join(path.node_names),
        "segments": path.segments.copy(),
    }


def split_record(
    path_id: str,
    source_path: TripPath,
    segments: pd.DataFrame,
    reference_path: TripPath,
    removed_nodes: list[str],
) -> dict:
    split_path = build_trip_path(source_path.trip_id, segments)
    return {
        "path_id": path_id,
        "source_trip_id": source_path.trip_id,
        "path_color": source_path.color,
        "max_occupancy": source_path.max_occupancy,
        "start_timestamp": split_path.start_timestamp,
        "end_timestamp": segments["arrive_timestamp"].max(),
        "segment_count": len(segments),
        "node_count": len(split_path.node_names),
        "status": "split",
        "split_reason": "specific_pair_suffix_overlap_removed",
        "neighbor_trip_ids": reference_path.trip_id,
        "neighbor_colors": reference_path.color,
        "removed_nodes": " | ".join(removed_nodes),
        "node_sequence": " -> ".join(split_path.node_names),
        "segments": segments.copy(),
    }


def apply_pair_rules(paths: list[TripPath]) -> tuple[list[dict], list[str]]:
    paths_by_id = {path.trip_id: path for path in paths}
    replaced_trip_ids: set[str] = set()
    split_records: dict[str, dict] = {}
    notes: list[str] = []

    for green_trip_id, reference_trip_id in PAIR_RULES:
        green_path = paths_by_id.get(green_trip_id)
        reference_path = paths_by_id.get(reference_trip_id)
        if green_path is None or reference_path is None:
            missing = [
                trip_id
                for trip_id, path in [
                    (green_trip_id, green_path),
                    (reference_trip_id, reference_path),
                ]
                if path is None
            ]
            notes.append(
                f"Skipped {green_trip_id}/{reference_trip_id}: missing {', '.join(missing)}"
            )
            continue

        suffix_start = overlap_suffix_start(green_path, reference_path)
        if suffix_start is None:
            notes.append(
                f"Skipped {green_trip_id}/{reference_trip_id}: overlap is not a suffix"
            )
            continue

        keep_segment_count = max(suffix_start - 1, 0)
        kept_segments = green_path.segments.iloc[:keep_segment_count].copy().reset_index(drop=True)
        removed_nodes = green_path.node_names[suffix_start:]
        if kept_segments.empty:
            notes.append(
                f"Skipped {green_trip_id}/{reference_trip_id}: suffix removal leaves no path"
            )
            continue

        split_records[green_trip_id] = split_record(
            f"{green_trip_id}_splittednew1",
            green_path,
            kept_segments,
            reference_path,
            removed_nodes,
        )
        replaced_trip_ids.add(green_trip_id)
        notes.append(
            f"Applied {green_trip_id}/{reference_trip_id}: removed suffix from "
            f"{removed_nodes[0]} to {removed_nodes[-1]}"
        )

    records = []
    for path in paths:
        if path.trip_id in replaced_trip_ids:
            records.append(split_records[path.trip_id])
        else:
            records.append(unchanged_record(path, "specific_pair_rule_not_applied"))
    return records, notes


def write_pair_outputs(records: list[dict], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = sorted(records, key=lambda record: (pd.Timestamp(record["start_timestamp"]), record["path_id"]))
    path_rows = []
    segment_rows = []

    for row_order, record in enumerate(records, start=1):
        path_rows.append({key: value for key, value in record.items() if key != "segments"} | {"row_order": row_order})
        for segment_order, segment in enumerate(record["segments"].itertuples(index=False), start=1):
            segment_rows.append(
                {
                    "path_id": record["path_id"],
                    "source_trip_id": record["source_trip_id"],
                    "path_color": record["path_color"],
                    "segment_order": segment_order,
                    "segment_id": segment.segment_id,
                    "departure_timestamp": segment.departure_timestamp,
                    "arrive_timestamp": segment.arrive_timestamp,
                    "departure_stop_name": segment.departure_stop_name,
                    "arrive_stop_name": segment.arrive_stop_name,
                    "occupancy_status": segment.occupancy_status,
                    "direction_id": getattr(segment, "direction_id", ""),
                    "route_short_name": segment.route_short_name,
                    "route_type": segment.route_type,
                }
            )

    path_output = output_dir / f"specific_pair_split_trip_paths_hour_{HOUR:02d}_{DATASET_DATE}.csv"
    segment_output = output_dir / f"specific_pair_split_trip_path_segments_hour_{HOUR:02d}_{DATASET_DATE}.csv"
    pd.DataFrame(path_rows).to_csv(path_output, index=False)
    pd.DataFrame(segment_rows).to_csv(segment_output, index=False)
    return path_output, segment_output


def load_pair_segments(segment_output: Path) -> pd.DataFrame:
    segments = pd.read_csv(segment_output, dtype={"path_id": "string", "source_trip_id": "string"})
    segments["departure_timestamp"] = pd.to_datetime(segments["departure_timestamp"], errors="coerce")
    segments["arrive_timestamp"] = pd.to_datetime(segments["arrive_timestamp"], errors="coerce")
    segments["segment_order"] = pd.to_numeric(segments["segment_order"], errors="coerce")
    segments["occupancy_status"] = pd.to_numeric(segments["occupancy_status"], errors="coerce").fillna(0).astype(int)
    graph_segments = segments.rename(
        columns={
            "path_id": "trip_id",
            "source_trip_id": "trip_id_org",
            "segment_order": "departure_stop_sequence",
        }
    ).copy()
    graph_segments["arrive_stop_sequence"] = graph_segments["departure_stop_sequence"] + 1
    return graph_segments.dropna(
        subset=["trip_id", "departure_timestamp", "arrive_timestamp", "departure_stop_sequence"]
    ).sort_values(["trip_id", "departure_stop_sequence", "departure_timestamp"]).reset_index(drop=True)


def render_outputs(segment_output: Path, output_dir: Path) -> None:
    segments = sort_trips_by_start_timestamp(load_pair_segments(segment_output))
    event_df, event_log = build_pm4py_event_log(segments)
    event_df = sort_event_df_by_path_start(event_df, segments)

    output_prefix = output_dir / f"specific_pair_split_process_paths_hour_{HOUR:02d}"
    build_trip_path_graph(
        segments,
        output_prefix,
        title="Specific pair suffix split process paths",
        sort_mode="start_timestamp",
    )
    event_output = output_dir / f"specific_pair_split_process_events_hour_{HOUR:02d}.csv"
    event_df.to_csv(event_output, index=False)

    print(f"PNG written: {output_prefix}.png")
    print(f"Event table written: {event_output}")
    print(f"PM4Py traces: {len(event_log)}")

    for direction_id, direction_segments in segments.groupby("direction_id", sort=True):
        direction_segments = sort_trips_by_start_timestamp(direction_segments)
        safe_direction = str(direction_id).replace(".", "_").replace(" ", "_")
        direction_output_prefix = (
            output_dir / f"specific_pair_split_process_paths_hour_{HOUR:02d}_direction_{safe_direction}"
        )
        direction_event_df, direction_event_log = build_pm4py_event_log(direction_segments)
        direction_event_df = sort_event_df_by_path_start(direction_event_df, direction_segments)
        build_trip_path_graph(
            direction_segments,
            direction_output_prefix,
            title=f"Specific pair suffix split process paths (direction {direction_id})",
            sort_mode="start_timestamp",
        )
        direction_event_output = (
            output_dir / f"specific_pair_split_process_events_hour_{HOUR:02d}_direction_{safe_direction}.csv"
        )
        direction_event_df.to_csv(direction_event_output, index=False)
        print(
            f"Direction {direction_id}: {direction_segments['trip_id'].nunique()} paths, "
            f"{len(direction_segments)} segments, {len(direction_event_log)} PM4Py traces, "
            f"PNG written: {direction_output_prefix}.png"
        )


def main() -> None:
    segments = load_selected_trip_segments()
    paths = [
        build_trip_path(trip_id, trip_segments)
        for trip_id, trip_segments in segments.groupby("trip_id", sort=False)
    ]
    paths = sorted(paths, key=lambda path: (path.start_timestamp, path.trip_id))

    records, notes = apply_pair_rules(paths)
    path_output, segment_output = write_pair_outputs(records, DEFAULT_OUTPUT_DIR)
    render_outputs(segment_output, DEFAULT_OUTPUT_DIR)

    summary = pd.DataFrame(
        {
            "path_id": record["path_id"],
            "source_trip_id": record["source_trip_id"],
            "path_color": record["path_color"],
            "status": record["status"],
            "segment_count": record["segment_count"],
            "split_reason": record["split_reason"],
            "neighbor_trip_ids": record["neighbor_trip_ids"],
            "removed_nodes": record["removed_nodes"],
        }
        for record in records
    )
    print()
    print(summary[summary["source_trip_id"].isin(["tr03_7353", "tr03_3878", "tr03_6882", "tr03_6074"])].to_string(index=False))
    print()
    print("\n".join(notes))
    print(f"Path output: {path_output}")
    print(f"Segment output: {segment_output}")


if __name__ == "__main__":
    main()
