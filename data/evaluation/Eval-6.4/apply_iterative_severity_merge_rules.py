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
    COLOR_BY_OCCUPANCY,
    HOUR,
    TripPath,
    activity_key,
    build_trip_path,
    load_selected_trip_segments,
)


SEVERITY_BY_COLOR = {
    "green": 0,
    "yellow": 1,
    "orange": 2,
    "red": 3,
}


def allowed_to_merge(lower_path: TripPath, higher_path: TripPath) -> bool:
    return SEVERITY_BY_COLOR[lower_path.color] < SEVERITY_BY_COLOR[higher_path.color]


def allowed_green_to_green_merge(lower_path: TripPath, reference_path: TripPath) -> bool:
    return lower_path.color == "green" and reference_path.color == "green"


def node_overlap_flags(path: TripPath, reference_path: TripPath) -> list[bool]:
    reference_keys = set(reference_path.node_keys)
    return [node_key in reference_keys for node_key in path.node_keys]


def kept_node_runs(overlap_node: list[bool]) -> list[tuple[int, int]]:
    runs = []
    run_start = None
    for index, is_overlap in enumerate(overlap_node):
        if not is_overlap and run_start is None:
            run_start = index
        elif is_overlap and run_start is not None:
            if index - run_start >= 2:
                runs.append((run_start, index - 1))
            run_start = None

    if run_start is not None and len(overlap_node) - run_start >= 2:
        runs.append((run_start, len(overlap_node) - 1))
    return runs


def segment_slice_for_node_run(path: TripPath, run_start: int, run_end: int) -> pd.DataFrame:
    return path.segments.iloc[run_start:run_end].copy().reset_index(drop=True)


def remove_overlap_from_piece(piece: TripPath, reference_path: TripPath) -> tuple[list[pd.DataFrame], list[str]]:
    overlap_node = node_overlap_flags(piece, reference_path)
    if not any(overlap_node):
        return [piece.segments.copy()], []

    removed_nodes = [
        node_name
        for node_name, is_overlap in zip(piece.node_names, overlap_node)
        if is_overlap
    ]
    kept_segments = [
        segment_slice_for_node_run(piece, run_start, run_end)
        for run_start, run_end in kept_node_runs(overlap_node)
    ]
    kept_segments = [segments for segments in kept_segments if not segments.empty]
    return kept_segments, removed_nodes


def make_path(
    trip_id: str,
    color: str,
    max_occupancy: int,
    segments: pd.DataFrame,
) -> TripPath:
    path = build_trip_path(trip_id, segments)
    return TripPath(
        trip_id=trip_id,
        color=color,
        max_occupancy=max_occupancy,
        start_timestamp=path.start_timestamp,
        segments=path.segments,
        node_names=path.node_names,
        node_timestamps=path.node_timestamps,
    )


def unchanged_record(path: TripPath, reason: str) -> dict:
    return {
        "path_id": path.trip_id,
        "source_trip_id": path.trip_id,
        "path_color": path.color,
        "max_occupancy": path.max_occupancy,
        "original_start_timestamp": path.start_timestamp,
        "start_timestamp": path.start_timestamp,
        "end_timestamp": path.segments["arrive_timestamp"].max(),
        "segment_count": len(path.segments),
        "node_count": len(path.node_names),
        "status": "unchanged",
        "split_reason": reason,
        "merged_into_trip_ids": "",
        "merged_into_colors": "",
        "removed_nodes": "",
        "node_sequence": " -> ".join(path.node_names),
        "segments": path.segments.copy(),
    }


def split_record(
    path_id: str,
    source_trip_id: str,
    color: str,
    max_occupancy: int,
    original_start_timestamp: pd.Timestamp,
    segments: pd.DataFrame,
    merged_into_trip_ids: list[str],
    merged_into_colors: list[str],
    removed_nodes: list[str],
) -> dict:
    path = make_path(source_trip_id, color, max_occupancy, segments)
    return {
        "path_id": path_id,
        "source_trip_id": source_trip_id,
        "path_color": color,
        "max_occupancy": max_occupancy,
        "original_start_timestamp": original_start_timestamp,
        "start_timestamp": path.start_timestamp,
        "end_timestamp": path.segments["arrive_timestamp"].max(),
        "segment_count": len(path.segments),
        "node_count": len(path.node_names),
        "status": "split",
        "split_reason": "iterative_overlap_merged_into_higher_severity",
        "merged_into_trip_ids": ", ".join(dict.fromkeys(merged_into_trip_ids)),
        "merged_into_colors": ", ".join(dict.fromkeys(merged_into_colors)),
        "removed_nodes": " | ".join(dict.fromkeys(removed_nodes)),
        "node_sequence": " -> ".join(path.node_names),
        "segments": path.segments.copy(),
    }


def removed_record(
    path: TripPath,
    merged_into_trip_ids: list[str],
    merged_into_colors: list[str],
    removed_nodes: list[str],
) -> dict:
    return {
        "path_id": path.trip_id,
        "source_trip_id": path.trip_id,
        "path_color": path.color,
        "max_occupancy": path.max_occupancy,
        "original_start_timestamp": path.start_timestamp,
        "start_timestamp": path.start_timestamp,
        "end_timestamp": path.segments["arrive_timestamp"].max(),
        "segment_count": 0,
        "node_count": 0,
        "status": "removed",
        "split_reason": "fully_merged_into_higher_severity",
        "merged_into_trip_ids": ", ".join(dict.fromkeys(merged_into_trip_ids)),
        "merged_into_colors": ", ".join(dict.fromkeys(merged_into_colors)),
        "removed_nodes": " | ".join(dict.fromkeys(removed_nodes)),
        "node_sequence": "",
        "segments": path.segments.iloc[0:0].copy(),
    }


def apply_reference_to_pieces(
    source_path: TripPath,
    pieces: list[TripPath],
    reference_path: TripPath,
    merged_into_trip_ids: list[str],
    merged_into_colors: list[str],
    removed_nodes: list[str],
) -> list[TripPath]:
    next_pieces = []
    for piece in pieces:
        kept_segments, piece_removed_nodes = remove_overlap_from_piece(
            piece,
            reference_path,
        )
        if piece_removed_nodes:
            merged_into_trip_ids.append(reference_path.trip_id)
            merged_into_colors.append(reference_path.color)
            removed_nodes.extend(piece_removed_nodes)
        for kept in kept_segments:
            next_pieces.append(
                make_path(
                    source_path.trip_id,
                    source_path.color,
                    source_path.max_occupancy,
                    kept,
                )
            )
    return next_pieces


def apply_iterative_merges(paths: list[TripPath]) -> list[dict]:
    records: list[dict] = []

    for index, path in enumerate(paths):
        if path.color == "red":
            records.append(unchanged_record(path, "highest_severity"))
            continue

        pieces = [path]
        merged_into_trip_ids: list[str] = []
        merged_into_colors: list[str] = []
        removed_nodes: list[str] = []

        for reference_path in paths[index + 1 :]:
            if not allowed_to_merge(path, reference_path):
                continue

            pieces = apply_reference_to_pieces(
                path,
                pieces,
                reference_path,
                merged_into_trip_ids,
                merged_into_colors,
                removed_nodes,
            )
            if not pieces:
                break

        if pieces and path.color == "green":
            for reference_path in paths[index + 1 :]:
                if not allowed_green_to_green_merge(path, reference_path):
                    continue

                pieces = apply_reference_to_pieces(
                    path,
                    pieces,
                    reference_path,
                    merged_into_trip_ids,
                    merged_into_colors,
                    removed_nodes,
                )
                if not pieces:
                    break

        if not merged_into_trip_ids:
            records.append(unchanged_record(path, "no_overlap_with_higher_later_trip"))
            continue

        if not pieces:
            records.append(
                removed_record(path, merged_into_trip_ids, merged_into_colors, removed_nodes)
            )
            continue

        for piece_index, piece in enumerate(pieces, start=1):
            records.append(
                split_record(
                    f"{path.trip_id}_splittednew{piece_index}",
                    path.trip_id,
                    path.color,
                    path.max_occupancy,
                    path.start_timestamp,
                    piece.segments,
                    merged_into_trip_ids,
                    merged_into_colors,
                    removed_nodes,
                )
            )

    return records


def write_outputs(records: list[dict], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = sorted(
        records,
        key=lambda record: (
            pd.Timestamp(record["original_start_timestamp"]),
            record["source_trip_id"],
            record["path_id"],
        ),
    )
    path_rows = []
    segment_rows = []

    for row_order, record in enumerate(records, start=1):
        path_rows.append(
            {key: value for key, value in record.items() if key != "segments"}
            | {"row_order": row_order}
        )
        for segment_order, segment in enumerate(record["segments"].itertuples(index=False), start=1):
            segment_rows.append(
                {
                    "path_id": record["path_id"],
                    "source_trip_id": record["source_trip_id"],
                    "path_color": record["path_color"],
                    "original_start_timestamp": record["original_start_timestamp"],
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

    path_output = output_dir / f"iterative_severity_merge_trip_paths_hour_{HOUR:02d}_{DATASET_DATE}.csv"
    segment_output = output_dir / f"iterative_severity_merge_trip_path_segments_hour_{HOUR:02d}_{DATASET_DATE}.csv"
    pd.DataFrame(path_rows).to_csv(path_output, index=False)
    pd.DataFrame(segment_rows).to_csv(segment_output, index=False)
    return path_output, segment_output


def load_graph_segments(segment_output: Path) -> pd.DataFrame:
    segments = pd.read_csv(segment_output, dtype={"path_id": "string", "source_trip_id": "string"})
    segments["departure_timestamp"] = pd.to_datetime(segments["departure_timestamp"], errors="coerce")
    segments["arrive_timestamp"] = pd.to_datetime(segments["arrive_timestamp"], errors="coerce")
    segments["original_start_timestamp"] = pd.to_datetime(segments["original_start_timestamp"], errors="coerce")
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
    graph_segments = graph_segments.dropna(
        subset=["trip_id", "departure_timestamp", "arrive_timestamp", "departure_stop_sequence"]
    ).sort_values(
        [
            "original_start_timestamp",
            "trip_id",
            "departure_stop_sequence",
            "departure_timestamp",
        ]
    ).reset_index(drop=True)
    return graph_segments


def render_outputs(segment_output: Path, output_dir: Path) -> None:
    segments = load_graph_segments(segment_output)
    event_df, event_log = build_pm4py_event_log(segments)
    event_df = sort_event_df_by_path_start(event_df, segments)

    output_prefix = output_dir / f"iterative_severity_merge_process_paths_hour_{HOUR:02d}"
    build_trip_path_graph(
        segments,
        output_prefix,
        title="Iterative severity merge process paths",
        sort_mode="start_timestamp",
    )
    event_output = output_dir / f"iterative_severity_merge_process_events_hour_{HOUR:02d}.csv"
    event_df.to_csv(event_output, index=False)

    print(f"PNG written: {output_prefix}.png")
    print(f"Event table written: {event_output}")
    print(f"PM4Py traces: {len(event_log)}")

    for direction_id, direction_segments in segments.groupby("direction_id", sort=True):
        direction_segments = direction_segments.sort_values(
            [
                "original_start_timestamp",
                "trip_id",
                "departure_stop_sequence",
                "departure_timestamp",
            ]
        ).reset_index(drop=True)
        safe_direction = str(direction_id).replace(".", "_").replace(" ", "_")
        direction_output_prefix = (
            output_dir / f"iterative_severity_merge_process_paths_hour_{HOUR:02d}_direction_{safe_direction}"
        )
        direction_event_df, direction_event_log = build_pm4py_event_log(direction_segments)
        direction_event_df = sort_event_df_by_path_start(direction_event_df, direction_segments)
        build_trip_path_graph(
            direction_segments,
            direction_output_prefix,
            title=f"Iterative severity merge process paths (direction {direction_id})",
            sort_mode="start_timestamp",
        )
        direction_event_output = (
            output_dir / f"iterative_severity_merge_process_events_hour_{HOUR:02d}_direction_{safe_direction}.csv"
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

    records = apply_iterative_merges(paths)
    path_output, segment_output = write_outputs(records, DEFAULT_OUTPUT_DIR)
    render_outputs(segment_output, DEFAULT_OUTPUT_DIR)

    summary = pd.DataFrame(
        {
            "path_id": record["path_id"],
            "source_trip_id": record["source_trip_id"],
            "path_color": record["path_color"],
            "status": record["status"],
            "segment_count": record["segment_count"],
            "merged_into_trip_ids": record["merged_into_trip_ids"],
            "removed_nodes": record["removed_nodes"],
        }
        for record in records
    )
    changed = summary[summary["status"].isin(["split", "removed"])]
    print()
    print(changed.to_string(index=False))
    print()
    print(f"Original trips: {len(paths)}")
    print(f"Resulting paths: {len(records)}")
    print(f"Split paths: {(summary['status'] == 'split').sum()}")
    print(f"Removed paths: {(summary['status'] == 'removed').sum()}")
    print(f"Path output: {path_output}")
    print(f"Segment output: {segment_output}")


if __name__ == "__main__":
    main()
