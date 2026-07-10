from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from build_high_spread_trip_process_paths import (
    DATASET_DATE,
    DEFAULT_AUX_EVENTS_FILE,
    DEFAULT_HIGH_SPREAD_FILE,
    DEFAULT_OUTPUT_DIR,
    build_trip_segments_from_aux_events,
    get_interval_start,
    load_aux_events,
    load_high_spread_segment_ids,
    select_trip_ids_by_high_spread_departure,
)


HOUR = 6
COLOR_BY_OCCUPANCY = {
    0: "green",
    1: "yellow",
    2: "orange",
    3: "red",
}
NEIGHBOR_PRIORITY = ["red", "orange", "yellow"]
RED_SEGMENTS_TO_SUPPRESS: dict[str, set[str]] = {}


@dataclass
class TripPath:
    trip_id: str
    color: str
    max_occupancy: int
    start_timestamp: pd.Timestamp
    segments: pd.DataFrame
    node_names: list[str]
    node_timestamps: list[pd.Timestamp]

    @property
    def node_keys(self) -> list[str]:
        return [activity_key(node_name) for node_name in self.node_names]


@dataclass
class GreenPiece:
    source_trip_id: str
    color: str
    max_occupancy: int
    segments: pd.DataFrame
    removed_nodes: set[str]
    split_reasons: list[str]
    neighbor_trip_ids: list[str]
    neighbor_colors: list[str]


def activity_key(value: str) -> str:
    return " ".join(str(value).strip().casefold().split())


def build_trip_path(trip_id: str, trip_segments: pd.DataFrame) -> TripPath:
    trip_segments = trip_segments.sort_values(
        ["departure_stop_sequence", "departure_timestamp"]
    ).reset_index(drop=True)

    first = trip_segments.iloc[0]
    node_names = [str(first["departure_stop_name"])]
    node_timestamps = [first["departure_timestamp"]]

    for row in trip_segments.itertuples(index=False):
        node_names.append(str(row.arrive_stop_name))
        node_timestamps.append(row.arrive_timestamp)

    max_occupancy = int(trip_segments["occupancy_status"].max())
    return TripPath(
        trip_id=trip_id,
        color=COLOR_BY_OCCUPANCY.get(max_occupancy, "unknown"),
        max_occupancy=max_occupancy,
        start_timestamp=trip_segments["departure_timestamp"].min(),
        segments=trip_segments,
        node_names=node_names,
        node_timestamps=node_timestamps,
    )


def load_selected_trip_segments() -> pd.DataFrame:
    high_spread_segment_ids = load_high_spread_segment_ids(DEFAULT_HIGH_SPREAD_FILE, None)
    events = load_aux_events(DEFAULT_AUX_EVENTS_FILE)
    hour_start = get_interval_start(events, HOUR)
    hour_end = hour_start + pd.Timedelta(hours=1)
    trip_ids = select_trip_ids_by_high_spread_departure(
        events,
        high_spread_segment_ids=high_spread_segment_ids,
        interval_start=hour_start,
        interval_end=hour_end,
    )
    return build_trip_segments_from_aux_events(events, trip_ids=trip_ids)


def contiguous_runs(keep_node: list[bool]) -> list[tuple[int, int]]:
    runs = []
    start = None
    for index, keep in enumerate(keep_node):
        if keep and start is None:
            start = index
        elif not keep and start is not None:
            if index - start >= 2:
                runs.append((start, index - 1))
            start = None
    if start is not None and len(keep_node) - start >= 2:
        runs.append((start, len(keep_node) - 1))
    return runs


def true_runs(flags: list[bool]) -> list[tuple[int, int]]:
    runs = []
    start = None
    for index, flag in enumerate(flags):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            if index - start >= 2:
                runs.append((start, index - 1))
            start = None
    if start is not None and len(flags) - start >= 2:
        runs.append((start, len(flags) - 1))
    return runs


def overlap_is_lower(path: TripPath, overlap_keys: set[str]) -> bool:
    overlap_indexes = [
        index for index, node_key in enumerate(path.node_keys) if node_key in overlap_keys
    ]
    if not overlap_indexes:
        return False
    midpoint = (len(path.node_keys) - 1) / 2
    return (sum(overlap_indexes) / len(overlap_indexes)) < midpoint


def non_overlap_runs_with_boundaries(overlap_node: list[bool]) -> list[tuple[int, int]]:
    runs = []
    index = 0
    while index < len(overlap_node):
        if overlap_node[index]:
            index += 1
            continue

        run_start = index
        while index < len(overlap_node) and not overlap_node[index]:
            index += 1
        run_end = index - 1

        # Keep the entry/exit boundary node so the prefix/suffix still shows where
        # the green path touches the removed neighbor overlap.
        if run_start > 0 and overlap_node[run_start - 1]:
            run_start -= 1
        if run_end + 1 < len(overlap_node) and overlap_node[run_end + 1]:
            run_end += 1

        if run_end - run_start + 1 >= 2:
            runs.append((run_start, run_end))

    return runs


def segment_slice_for_run(trip_path: TripPath, run_start: int, run_end: int) -> pd.DataFrame:
    # Segment i connects node i -> node i + 1.
    return trip_path.segments.iloc[run_start:run_end].copy().reset_index(drop=True)


def build_piece_path(piece: GreenPiece) -> TripPath:
    return build_trip_path(piece.source_trip_id, piece.segments)


def suppress_red_neighbor_segments(
    red_neighbors: list[TripPath],
    overlap_keys: set[str],
) -> None:
    for neighbor in red_neighbors:
        for segment in neighbor.segments.itertuples(index=False):
            from_key = activity_key(segment.departure_stop_name)
            to_key = activity_key(segment.arrive_stop_name)
            if from_key in overlap_keys and to_key in overlap_keys:
                RED_SEGMENTS_TO_SUPPRESS.setdefault(neighbor.trip_id, set()).add(
                    str(segment.segment_id)
                )


def split_piece_by_lower_red_overlap(
    piece: GreenPiece,
    red_neighbors: list[TripPath],
    overlap_keys: set[str],
) -> list[GreenPiece]:
    piece_path = build_piece_path(piece)
    overlap_node = [node_key in overlap_keys for node_key in piece_path.node_keys]
    pieces: list[GreenPiece] = []

    removed_nodes = {
        node_name
        for node_name in piece_path.node_names
        if activity_key(node_name) in overlap_keys
    }
    reason = "green_overlap_recolored_red_with_red_neighbor"
    neighbor_trip_ids = [neighbor.trip_id for neighbor in red_neighbors]
    neighbor_colors = [neighbor.color for neighbor in red_neighbors]

    for run_start, run_end in contiguous_runs([not flag for flag in overlap_node]):
        split_segments = segment_slice_for_run(piece_path, run_start, run_end)
        pieces.append(
            GreenPiece(
                source_trip_id=piece.source_trip_id,
                color=piece.color,
                max_occupancy=piece.max_occupancy,
                segments=split_segments,
                removed_nodes=piece.removed_nodes.union(removed_nodes),
                split_reasons=piece.split_reasons + [reason],
                neighbor_trip_ids=piece.neighbor_trip_ids + neighbor_trip_ids,
                neighbor_colors=piece.neighbor_colors + neighbor_colors,
            )
        )

    for run_start, run_end in true_runs(overlap_node):
        red_segments = segment_slice_for_run(piece_path, run_start, run_end)
        red_segments["occupancy_status"] = 3
        pieces.append(
            GreenPiece(
                source_trip_id=piece.source_trip_id,
                color="red",
                max_occupancy=3,
                segments=red_segments,
                removed_nodes=piece.removed_nodes.union(removed_nodes),
                split_reasons=piece.split_reasons + [reason],
                neighbor_trip_ids=piece.neighbor_trip_ids + neighbor_trip_ids,
                neighbor_colors=piece.neighbor_colors + neighbor_colors,
            )
        )

    suppress_red_neighbor_segments(red_neighbors, overlap_keys)
    return pieces


def split_piece_by_color(
    piece: GreenPiece,
    color: str,
    color_neighbors: list[TripPath],
) -> list[GreenPiece]:
    if not color_neighbors or piece.segments.empty:
        return [piece]

    piece_path = build_piece_path(piece)
    neighbor_keys = set()
    for neighbor in color_neighbors:
        neighbor_keys.update(neighbor.node_keys)

    overlap_keys = set(piece_path.node_keys).intersection(neighbor_keys)
    if not overlap_keys:
        return [piece]

    if color == "red" and overlap_is_lower(piece_path, overlap_keys):
        return split_piece_by_lower_red_overlap(piece, color_neighbors, overlap_keys)

    overlap_node = [node_key in overlap_keys for node_key in piece_path.node_keys]
    runs = non_overlap_runs_with_boundaries(overlap_node)
    if not runs:
        return []

    removed_nodes = {
        node_name
        for node_name in piece_path.node_names
        if activity_key(node_name) in overlap_keys
    }
    split_pieces = []
    for run_start, run_end in runs:
        split_segments = segment_slice_for_run(piece_path, run_start, run_end)
        split_pieces.append(
            GreenPiece(
                source_trip_id=piece.source_trip_id,
                color=piece.color,
                max_occupancy=piece.max_occupancy,
                segments=split_segments,
                removed_nodes=piece.removed_nodes.union(removed_nodes),
                split_reasons=piece.split_reasons + [f"removed_overlap_with_{color}_neighbor"],
                neighbor_trip_ids=piece.neighbor_trip_ids
                + [neighbor.trip_id for neighbor in color_neighbors],
                neighbor_colors=piece.neighbor_colors
                + [neighbor.color for neighbor in color_neighbors],
            )
        )
    return split_pieces


def split_green_trip(
    trip_path: TripPath,
    comparison_paths: list[TripPath],
) -> list[dict]:
    colored_neighbors = [
        neighbor for neighbor in comparison_paths if neighbor.color in NEIGHBOR_PRIORITY
    ]
    if not colored_neighbors:
        return [unchanged_path_record(trip_path, "no_colored_neighbor", [])]

    pieces = [
        GreenPiece(
            source_trip_id=trip_path.trip_id,
            color=trip_path.color,
            max_occupancy=trip_path.max_occupancy,
            segments=trip_path.segments.copy(),
            removed_nodes=set(),
            split_reasons=[],
            neighbor_trip_ids=[],
            neighbor_colors=[],
        )
    ]

    for color in NEIGHBOR_PRIORITY:
        color_neighbors = [neighbor for neighbor in colored_neighbors if neighbor.color == color]
        next_pieces = []
        for piece in pieces:
            if piece.color == "green":
                next_pieces.extend(split_piece_by_color(piece, color, color_neighbors))
            else:
                next_pieces.append(piece)
        pieces = next_pieces
        if not pieces:
            return [
                removed_path_record(
                    trip_path,
                    f"fully_removed_after_{color}_comparison",
                    color_neighbors,
                    set(trip_path.node_keys),
                )
            ]

    if not any(piece.split_reasons for piece in pieces):
        return [unchanged_path_record(trip_path, "no_overlap_with_colored_neighbor", colored_neighbors)]

    split_records = []
    for split_index, piece in enumerate(pieces, start=1):
        piece_path = build_piece_path(piece)
        path_suffix = "splittedred" if piece.color == "red" else "splittednew"
        split_records.append(
            {
                "path_id": f"{trip_path.trip_id}_{path_suffix}{split_index}",
                "source_trip_id": trip_path.trip_id,
                "path_color": piece.color,
                "max_occupancy": piece.max_occupancy,
                "start_timestamp": piece_path.start_timestamp,
                "end_timestamp": piece_path.segments["arrive_timestamp"].max(),
                "segment_count": len(piece.segments),
                "node_count": len(piece_path.node_names),
                "status": "split",
                "split_reason": " + ".join(dict.fromkeys(piece.split_reasons)),
                "neighbor_trip_ids": ", ".join(dict.fromkeys(piece.neighbor_trip_ids)),
                "neighbor_colors": ", ".join(dict.fromkeys(piece.neighbor_colors)),
                "removed_nodes": " | ".join(sorted(piece.removed_nodes)),
                "node_sequence": " -> ".join(piece_path.node_names),
                "segments": piece.segments.copy(),
            }
        )
    return split_records


def unchanged_path_record(
    trip_path: TripPath,
    reason: str,
    neighbors: list[TripPath],
) -> dict:
    return {
        "path_id": trip_path.trip_id,
        "source_trip_id": trip_path.trip_id,
        "path_color": trip_path.color,
        "max_occupancy": trip_path.max_occupancy,
        "start_timestamp": trip_path.start_timestamp,
        "end_timestamp": trip_path.segments["arrive_timestamp"].max(),
        "segment_count": len(trip_path.segments),
        "node_count": len(trip_path.node_names),
        "status": "unchanged",
        "split_reason": reason,
        "neighbor_trip_ids": ", ".join(neighbor.trip_id for neighbor in neighbors),
        "neighbor_colors": ", ".join(neighbor.color for neighbor in neighbors),
        "removed_nodes": "",
        "node_sequence": " -> ".join(trip_path.node_names),
        "segments": trip_path.segments.copy(),
    }


def removed_path_record(
    trip_path: TripPath,
    reason: str,
    neighbors: list[TripPath],
    overlap_keys: set[str],
) -> dict:
    return {
        "path_id": trip_path.trip_id,
        "source_trip_id": trip_path.trip_id,
        "path_color": trip_path.color,
        "max_occupancy": trip_path.max_occupancy,
        "start_timestamp": trip_path.start_timestamp,
        "end_timestamp": trip_path.segments["arrive_timestamp"].max(),
        "segment_count": 0,
        "node_count": 0,
        "status": "removed",
        "split_reason": reason,
        "neighbor_trip_ids": ", ".join(neighbor.trip_id for neighbor in neighbors),
        "neighbor_colors": ", ".join(neighbor.color for neighbor in neighbors),
        "removed_nodes": " | ".join(
            sorted(
                {
                    node_name
                    for node_name in trip_path.node_names
                    if activity_key(node_name) in overlap_keys
                }
            )
        ),
        "node_sequence": "",
        "segments": trip_path.segments.iloc[0:0].copy(),
    }


def split_record_from_segments(
    path_id: str,
    source_trip_id: str,
    path_color: str,
    max_occupancy: int,
    segments: pd.DataFrame,
    reason: str,
    neighbor: TripPath,
    removed_nodes: set[str],
) -> dict:
    piece_path = build_trip_path(source_trip_id, segments)
    return {
        "path_id": path_id,
        "source_trip_id": source_trip_id,
        "path_color": path_color,
        "max_occupancy": max_occupancy,
        "start_timestamp": piece_path.start_timestamp,
        "end_timestamp": segments["arrive_timestamp"].max(),
        "segment_count": len(segments),
        "node_count": len(piece_path.node_names),
        "status": "split",
        "split_reason": reason,
        "neighbor_trip_ids": neighbor.trip_id,
        "neighbor_colors": neighbor.color,
        "removed_nodes": " | ".join(sorted(removed_nodes)),
        "node_sequence": " -> ".join(piece_path.node_names),
        "segments": segments.copy(),
    }


def split_green_with_next_red(trip_path: TripPath, red_neighbor: TripPath) -> list[dict]:
    overlap_keys = set(trip_path.node_keys).intersection(set(red_neighbor.node_keys))
    if not overlap_keys:
        return [unchanged_path_record(trip_path, "no_overlap_with_next_red", [red_neighbor])]

    overlap_node = [node_key in overlap_keys for node_key in trip_path.node_keys]
    overlap_runs = true_runs(overlap_node)
    if not overlap_runs:
        return [unchanged_path_record(trip_path, "no_overlap_with_next_red", [red_neighbor])]

    first_start, first_end = overlap_runs[0]
    removed_nodes = {
        node_name
        for node_name in trip_path.node_names
        if activity_key(node_name) in overlap_keys
    }

    records = []
    if first_start == 0:
        red_segments = segment_slice_for_run(trip_path, first_start, first_end)
        red_segments["occupancy_status"] = 3
        records.append(
            split_record_from_segments(
                f"{trip_path.trip_id}_splittedred1",
                trip_path.trip_id,
                "red",
                3,
                red_segments,
                "start_overlap_green_part_recolored_red_next_red",
                red_neighbor,
                removed_nodes,
            )
        )

        if first_end < len(trip_path.node_names) - 1:
            green_segments = segment_slice_for_run(
                trip_path,
                first_end,
                len(trip_path.node_names) - 1,
            )
            records.append(
                split_record_from_segments(
                    f"{trip_path.trip_id}_splittednew1",
                    trip_path.trip_id,
                    "green",
                    0,
                    green_segments,
                    "remaining_green_after_start_overlap_next_red",
                    red_neighbor,
                    removed_nodes,
                )
            )
        suppress_red_neighbor_segments([red_neighbor], overlap_keys)
        return records

    lower_segments = segment_slice_for_run(trip_path, 0, first_start)
    if lower_segments.empty:
        return [
            removed_path_record(
                trip_path,
                "upper_overlap_removed_green_no_lower_prefix",
                [red_neighbor],
                overlap_keys,
            )
        ]

    return [
        split_record_from_segments(
            f"{trip_path.trip_id}_splittednew1",
            trip_path.trip_id,
            "green",
            0,
            lower_segments,
            "upper_overlap_removed_green_prefix_kept_next_red",
            red_neighbor,
            removed_nodes,
        )
    ]


def split_green_with_next_colored(trip_path: TripPath, neighbor: TripPath) -> list[dict]:
    if neighbor.color == "red":
        return split_green_with_next_red(trip_path, neighbor)

    piece = GreenPiece(
        source_trip_id=trip_path.trip_id,
        color=trip_path.color,
        max_occupancy=trip_path.max_occupancy,
        segments=trip_path.segments.copy(),
        removed_nodes=set(),
        split_reasons=[],
        neighbor_trip_ids=[],
        neighbor_colors=[],
    )
    pieces = split_piece_by_color(piece, neighbor.color, [neighbor])
    if len(pieces) == 1 and not pieces[0].split_reasons:
        return [unchanged_path_record(trip_path, f"no_overlap_with_next_{neighbor.color}", [neighbor])]
    if not pieces:
        return [
            removed_path_record(
                trip_path,
                f"fully_removed_after_next_{neighbor.color}_comparison",
                [neighbor],
                set(trip_path.node_keys),
            )
        ]

    records = []
    for index, split_piece in enumerate(pieces, start=1):
        records.append(
            split_record_from_segments(
                f"{trip_path.trip_id}_splittednew{index}",
                trip_path.trip_id,
                split_piece.color,
                split_piece.max_occupancy,
                split_piece.segments,
                " + ".join(dict.fromkeys(split_piece.split_reasons)),
                neighbor,
                split_piece.removed_nodes,
            )
        )
    return records


def split_green_paths(paths: list[TripPath]) -> list[dict]:
    RED_SEGMENTS_TO_SUPPRESS.clear()
    result_records = []
    for index, trip_path in enumerate(paths):
        if trip_path.color != "green":
            result_records.append(unchanged_path_record(trip_path, "not_green", []))
            continue

        next_path = paths[index + 1] if index + 1 < len(paths) else None
        if next_path is None or next_path.color not in NEIGHBOR_PRIORITY:
            result_records.append(unchanged_path_record(trip_path, "no_colored_next_neighbor", []))
            continue

        result_records.extend(split_green_with_next_colored(trip_path, next_path))
    return apply_red_suppression(result_records)


def apply_red_suppression(records: list[dict]) -> list[dict]:
    if not RED_SEGMENTS_TO_SUPPRESS:
        return records

    updated_records = []
    for record in records:
        suppressed_segment_ids = RED_SEGMENTS_TO_SUPPRESS.get(record["source_trip_id"], set())
        if record["path_color"] != "red" or not suppressed_segment_ids:
            updated_records.append(record)
            continue

        kept_segments = record["segments"][
            ~record["segments"]["segment_id"].astype(str).isin(suppressed_segment_ids)
        ].copy()
        if kept_segments.empty:
            updated = record.copy()
            updated["segments"] = kept_segments
            updated["segment_count"] = 0
            updated["node_count"] = 0
            updated["status"] = "removed"
            updated["split_reason"] = "red_overlap_replaced_by_green_red_piece"
            updated["node_sequence"] = ""
            updated_records.append(updated)
            continue

        updated_path = build_trip_path(record["source_trip_id"], kept_segments)
        updated = record.copy()
        updated["segments"] = kept_segments
        updated["start_timestamp"] = updated_path.start_timestamp
        updated["end_timestamp"] = kept_segments["arrive_timestamp"].max()
        updated["segment_count"] = len(kept_segments)
        updated["node_count"] = len(updated_path.node_names)
        updated["split_reason"] = (
            f"{record['split_reason']} + red_overlap_replaced_by_green_red_piece"
        )
        updated["node_sequence"] = " -> ".join(updated_path.node_names)
        updated_records.append(updated)

    return updated_records


def write_outputs(records: list[dict], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = sorted(
        records,
        key=lambda record: (
            pd.Timestamp(record["start_timestamp"]),
            record["path_id"],
        ),
    )
    path_rows = []
    segment_rows = []

    for row_order, record in enumerate(records, start=1):
        path_rows.append(
            {
                key: value
                for key, value in record.items()
                if key != "segments"
            }
            | {"row_order": row_order}
        )

        segments = record["segments"]
        for segment_order, segment in enumerate(segments.itertuples(index=False), start=1):
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

    path_output = output_dir / f"split_green_trip_paths_hour_{HOUR:02d}_{DATASET_DATE}.csv"
    segment_output = output_dir / f"split_green_trip_path_segments_hour_{HOUR:02d}_{DATASET_DATE}.csv"
    pd.DataFrame(path_rows).to_csv(path_output, index=False)
    pd.DataFrame(segment_rows).to_csv(segment_output, index=False)
    return path_output, segment_output


def main() -> None:
    segments = load_selected_trip_segments()
    paths = [
        build_trip_path(trip_id, trip_segments)
        for trip_id, trip_segments in segments.groupby("trip_id", sort=False)
    ]
    paths = sorted(paths, key=lambda path: (path.start_timestamp, path.trip_id))

    records = split_green_paths(paths)
    path_output, segment_output = write_outputs(records, DEFAULT_OUTPUT_DIR)

    summary = pd.DataFrame(
        {
            "path_id": record["path_id"],
            "source_trip_id": record["source_trip_id"],
            "path_color": record["path_color"],
            "status": record["status"],
            "segment_count": record["segment_count"],
            "split_reason": record["split_reason"],
            "neighbor_trip_ids": record["neighbor_trip_ids"],
        }
        for record in records
    )
    print(summary.to_string(index=False))
    print()
    print(f"Original trips: {len(paths)}")
    print(f"Resulting paths: {len(records)}")
    print(f"Split paths: {(summary['status'] == 'split').sum()}")
    print(f"Removed paths: {(summary['status'] == 'removed').sum()}")
    print(f"Path output: {path_output}")
    print(f"Segment output: {segment_output}")


if __name__ == "__main__":
    main()
