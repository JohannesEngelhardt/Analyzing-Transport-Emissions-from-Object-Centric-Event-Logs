from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from pathlib import Path

import numpy as np
import pandas as pd

from find_nearby_occupancy_contrast_segments import (
    build_low_segment_index,
    haversine_m,
    interval_gap_minutes,
    load_segments,
)


DATASET_DATE = "2026_05_04"
DEFAULT_SEGMENTS_FILE = Path(f"trip_segment_summary_{DATASET_DATE}.csv")
DEFAULT_COORDS_FILE = Path(f"segment_overview_kodak_{DATASET_DATE}.csv")
DEFAULT_OUTPUT_FILE = Path(f"nearby_route_type_700_contrast_segments_{DATASET_DATE}.csv")


def endpoint_alignment_distances(base, other) -> tuple[float, float, float]:
    same_start_m = haversine_m(
        base.segment_departure_stop_lat,
        base.segment_departure_stop_lon,
        other.segment_departure_stop_lat,
        other.segment_departure_stop_lon,
    )
    same_end_m = haversine_m(
        base.segment_arrive_stop_lat,
        base.segment_arrive_stop_lon,
        other.segment_arrive_stop_lat,
        other.segment_arrive_stop_lon,
    )
    opposite_start_m = haversine_m(
        base.segment_departure_stop_lat,
        base.segment_departure_stop_lon,
        other.segment_arrive_stop_lat,
        other.segment_arrive_stop_lon,
    )
    opposite_end_m = haversine_m(
        base.segment_arrive_stop_lat,
        base.segment_arrive_stop_lon,
        other.segment_departure_stop_lat,
        other.segment_departure_stop_lon,
    )
    same_direction_endpoint_distance_m = max(same_start_m, same_end_m)
    opposite_direction_endpoint_distance_m = max(opposite_start_m, opposite_end_m)
    endpoint_alignment_distance_m = min(
        same_direction_endpoint_distance_m,
        opposite_direction_endpoint_distance_m,
    )
    return (
        same_direction_endpoint_distance_m,
        opposite_direction_endpoint_distance_m,
        endpoint_alignment_distance_m,
    )


def passes_spatial_mode(
    midpoint_distance_m: float,
    endpoint_alignment_distance_m: float,
    max_space_m: float,
    spatial_mode: str,
) -> bool:
    if spatial_mode == "endpoints":
        return endpoint_alignment_distance_m <= max_space_m
    if spatial_mode == "midpoint":
        return midpoint_distance_m <= max_space_m
    return midpoint_distance_m <= max_space_m or endpoint_alignment_distance_m <= max_space_m


def find_nearby_route_type_contrasts(
    segments: pd.DataFrame,
    base_route_type: str,
    max_space_m: float,
    max_time_min: float,
    spatial_mode: str,
    allow_same_trip: bool,
) -> pd.DataFrame:
    base_segments = segments[segments["route_type"].astype(str) == base_route_type].copy()
    other_segments = segments[segments["route_type"].astype(str) != base_route_type].copy()

    by_cell, lat_cell_degrees, lon_cell_degrees = build_low_segment_index(
        other_segments,
        max_space_m,
    )
    max_midpoint_delta_ns = int(max_time_min * 60 * 1_000_000_000)

    base_segments["lat_cell"] = np.floor(base_segments["mid_lat"] / lat_cell_degrees).astype(int)
    base_segments["lon_cell"] = np.floor(base_segments["mid_lon"] / lon_cell_degrees).astype(int)

    results = []
    for base in base_segments.itertuples(index=False):
        lower_time = base.mid_timestamp_ns - max_midpoint_delta_ns
        upper_time = base.mid_timestamp_ns + max_midpoint_delta_ns

        for lat_offset in (-1, 0, 1):
            for lon_offset in (-1, 0, 1):
                cell = (base.lat_cell + lat_offset, base.lon_cell + lon_offset)
                other_bucket = by_cell.get(cell)
                if other_bucket is None:
                    continue

                times = other_bucket["time"]
                start_pos = bisect_left(times, lower_time)
                end_pos = bisect_right(times, upper_time)
                if start_pos == end_pos:
                    continue

                candidates = other_bucket["rows"].iloc[start_pos:end_pos]
                midpoint_distances = haversine_m(
                    base.mid_lat,
                    base.mid_lon,
                    candidates["mid_lat"].to_numpy(),
                    candidates["mid_lon"].to_numpy(),
                )
                spatial_candidates = candidates.loc[midpoint_distances <= max_space_m].copy()
                if spatial_candidates.empty:
                    continue
                spatial_candidates["midpoint_distance_m"] = midpoint_distances[
                    midpoint_distances <= max_space_m
                ]

                for other in spatial_candidates.itertuples(index=False):
                    if not allow_same_trip and base.trip_id == other.trip_id:
                        continue

                    time_gap_min = interval_gap_minutes(
                        base.departure_timestamp,
                        base.arrive_timestamp,
                        other.departure_timestamp,
                        other.arrive_timestamp,
                    )
                    if time_gap_min > max_time_min:
                        continue

                    (
                        same_direction_endpoint_distance_m,
                        opposite_direction_endpoint_distance_m,
                        endpoint_alignment_distance_m,
                    ) = endpoint_alignment_distances(base, other)
                    if not passes_spatial_mode(
                        other.midpoint_distance_m,
                        endpoint_alignment_distance_m,
                        max_space_m,
                        spatial_mode,
                    ):
                        continue

                    results.append(
                        {
                            "base_segment_id": base.segment_id,
                            "base_trip_id": base.trip_id,
                            "base_trip_id_org": base.trip_id_org,
                            "base_route_short_name": base.route_short_name,
                            "base_route_type": base.route_type,
                            "base_vehicle_type": base.vehicle_type,
                            "base_from_stop": base.departure_stop_name,
                            "base_to_stop": base.arrive_stop_name,
                            "base_departure_timestamp": base.departure_timestamp,
                            "base_arrive_timestamp": base.arrive_timestamp,
                            "base_occupancy_status": base.occupancy_status,
                            "base_duration_penalty": base.duration_penalty,
                            "other_segment_id": other.segment_id,
                            "other_trip_id": other.trip_id,
                            "other_trip_id_org": other.trip_id_org,
                            "other_route_short_name": other.route_short_name,
                            "other_route_type": other.route_type,
                            "other_vehicle_type": other.vehicle_type,
                            "other_from_stop": other.departure_stop_name,
                            "other_to_stop": other.arrive_stop_name,
                            "other_departure_timestamp": other.departure_timestamp,
                            "other_arrive_timestamp": other.arrive_timestamp,
                            "other_occupancy_status": other.occupancy_status,
                            "other_duration_penalty": other.duration_penalty,
                            "time_gap_min": time_gap_min,
                            "midpoint_time_diff_min": abs(
                                base.mid_timestamp_ns - other.mid_timestamp_ns
                            ) / 1_000_000_000 / 60,
                            "midpoint_distance_m": other.midpoint_distance_m,
                            "same_direction_endpoint_distance_m": same_direction_endpoint_distance_m,
                            "opposite_direction_endpoint_distance_m": opposite_direction_endpoint_distance_m,
                            "endpoint_alignment_distance_m": endpoint_alignment_distance_m,
                        }
                    )

    result_df = pd.DataFrame(results)
    if result_df.empty:
        return result_df

    return result_df.sort_values(
        [
            "time_gap_min",
            "endpoint_alignment_distance_m",
            "midpoint_distance_m",
            "midpoint_time_diff_min",
        ],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find route_type 700 segments that are spatially and temporally close "
            "to segments from another route_type."
        )
    )
    parser.add_argument("--segments", type=Path, default=DEFAULT_SEGMENTS_FILE)
    parser.add_argument("--coords", type=Path, default=DEFAULT_COORDS_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--base-route-type", default="700")
    parser.add_argument("--max-space-m", type=float, default=250.0)
    parser.add_argument("--max-time-min", type=float, default=10.0)
    parser.add_argument("--limit", type=int, default=1000, help="Use 0 for all rows.")
    parser.add_argument("--allow-same-trip", action="store_true")
    parser.add_argument(
        "--spatial-mode",
        choices=["endpoints", "midpoint", "either"],
        default="endpoints",
        help=(
            "endpoints: start/end stops must align within max-space-m; "
            "midpoint: only segment midpoints must be close; "
            "either: accept either criterion."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    segments = load_segments(args.segments, args.coords)
    result = find_nearby_route_type_contrasts(
        segments=segments,
        base_route_type=str(args.base_route_type),
        max_space_m=args.max_space_m,
        max_time_min=args.max_time_min,
        spatial_mode=args.spatial_mode,
        allow_same_trip=args.allow_same_trip,
    )

    output = result if args.limit == 0 else result.head(args.limit)
    output.to_csv(args.output, index=False)

    print(f"Loaded segments with coordinates: {len(segments):,}")
    print(f"Base route_type {args.base_route_type} segments: {(segments['route_type'].astype(str) == str(args.base_route_type)).sum():,}")
    print(f"Other route_type segments: {(segments['route_type'].astype(str) != str(args.base_route_type)).sum():,}")
    print(f"Matching segment pairs found: {len(result):,}")
    print(f"Matching segment pairs written: {len(output):,}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
