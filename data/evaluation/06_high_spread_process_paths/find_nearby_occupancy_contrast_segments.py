from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from pathlib import Path

import numpy as np
import pandas as pd


DATASET_DATE = "2026_05_04"
DEFAULT_SEGMENTS_FILE = Path(f"trip_segment_summary_{DATASET_DATE}.csv")
DEFAULT_COORDS_FILE = Path(f"segment_overview_kodak_{DATASET_DATE}.csv")
DEFAULT_OUTPUT_FILE = Path(f"nearby_occupancy_contrast_segments_{DATASET_DATE}.csv")
DEFAULT_TRIP_PAIR_OUTPUT_FILE = Path(
    f"nearby_occupancy_contrast_trip_pairs_{DATASET_DATE}.csv"
)

EARTH_RADIUS_M = 6_371_000


def normalize_stop_id(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def haversine_m(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def interval_gap_minutes(start_a, end_a, start_b, end_b) -> float:
    if start_a <= end_b and start_b <= end_a:
        return 0.0
    if end_a < start_b:
        return (start_b - end_a).total_seconds() / 60
    return (start_a - end_b).total_seconds() / 60


def load_segments(segments_file: Path, coords_file: Path) -> pd.DataFrame:
    segment_columns = [
        "segment_id",
        "trip_id",
        "trip_id_org",
        "departure_timestamp",
        "arrive_timestamp",
        "departure_stop_sequence",
        "departure_stop_id",
        "departure_stop_name",
        "arrive_stop_sequence",
        "arrive_stop_id",
        "arrive_stop_name",
        "route_id",
        "route_short_name",
        "route_type",
        "vehicle_id",
        "vehicle_type",
        "occupancy_status",
        "duration_min",
        "duration_penalty",
        "distance_diff_km",
    ]
    coord_columns = [
        "segment_departure_stop_id",
        "segment_departure_stop_lat",
        "segment_departure_stop_lon",
        "segment_arrive_stop_id",
        "segment_arrive_stop_lat",
        "segment_arrive_stop_lon",
    ]

    segments = pd.read_csv(
        segments_file,
        usecols=segment_columns,
        dtype={
            "segment_id": "string",
            "trip_id": "string",
            "trip_id_org": "string",
            "departure_stop_id": "string",
            "arrive_stop_id": "string",
            "route_id": "string",
            "route_short_name": "string",
            "route_type": "string",
            "vehicle_id": "string",
            "vehicle_type": "string",
        },
    )
    coords = pd.read_csv(
        coords_file,
        usecols=coord_columns,
        dtype={
            "segment_departure_stop_id": "string",
            "segment_arrive_stop_id": "string",
        },
    )

    segments["departure_stop_id_norm"] = normalize_stop_id(segments["departure_stop_id"])
    segments["arrive_stop_id_norm"] = normalize_stop_id(segments["arrive_stop_id"])
    coords["departure_stop_id_norm"] = normalize_stop_id(coords["segment_departure_stop_id"])
    coords["arrive_stop_id_norm"] = normalize_stop_id(coords["segment_arrive_stop_id"])

    coords = coords.drop_duplicates(
        ["departure_stop_id_norm", "arrive_stop_id_norm"]
    )
    segments = segments.merge(
        coords[
            [
                "departure_stop_id_norm",
                "arrive_stop_id_norm",
                "segment_departure_stop_lat",
                "segment_departure_stop_lon",
                "segment_arrive_stop_lat",
                "segment_arrive_stop_lon",
            ]
        ],
        on=["departure_stop_id_norm", "arrive_stop_id_norm"],
        how="left",
    )

    numeric_columns = [
        "occupancy_status",
        "duration_min",
        "duration_penalty",
        "distance_diff_km",
        "segment_departure_stop_lat",
        "segment_departure_stop_lon",
        "segment_arrive_stop_lat",
        "segment_arrive_stop_lon",
    ]
    for column in numeric_columns:
        segments[column] = pd.to_numeric(segments[column], errors="coerce")

    segments["departure_timestamp"] = pd.to_datetime(
        segments["departure_timestamp"],
        errors="coerce",
    )
    segments["arrive_timestamp"] = pd.to_datetime(
        segments["arrive_timestamp"],
        errors="coerce",
    )
    segments = segments.dropna(
        subset=[
            "departure_timestamp",
            "arrive_timestamp",
            "occupancy_status",
            "segment_departure_stop_lat",
            "segment_departure_stop_lon",
            "segment_arrive_stop_lat",
            "segment_arrive_stop_lon",
        ]
    ).copy()

    segments["mid_lat"] = (
        segments["segment_departure_stop_lat"] + segments["segment_arrive_stop_lat"]
    ) / 2
    segments["mid_lon"] = (
        segments["segment_departure_stop_lon"] + segments["segment_arrive_stop_lon"]
    ) / 2
    segments["mid_timestamp"] = (
        segments["departure_timestamp"]
        + (segments["arrive_timestamp"] - segments["departure_timestamp"]) / 2
    )
    segments["mid_timestamp_ns"] = segments["mid_timestamp"].astype("int64")
    return segments.reset_index(drop=True)


def build_low_segment_index(low_segments: pd.DataFrame, cell_size_m: float):
    lat_cell_degrees = cell_size_m / 111_320
    lon_cell_degrees = cell_size_m / (
        111_320 * np.cos(np.radians(low_segments["mid_lat"].median()))
    )

    low_segments = low_segments.copy()
    low_segments["lat_cell"] = np.floor(low_segments["mid_lat"] / lat_cell_degrees).astype(int)
    low_segments["lon_cell"] = np.floor(low_segments["mid_lon"] / lon_cell_degrees).astype(int)

    by_cell = {}
    for cell, group in low_segments.groupby(["lat_cell", "lon_cell"], sort=False):
        ordered = group.sort_values("mid_timestamp_ns").reset_index(drop=True)
        by_cell[cell] = {
            "time": ordered["mid_timestamp_ns"].to_numpy(),
            "rows": ordered,
        }
    return by_cell, lat_cell_degrees, lon_cell_degrees


def find_nearby_contrasts(
    segments: pd.DataFrame,
    max_space_m: float,
    max_time_min: float,
    high_occupancy_min: float,
    low_occupancy_max: float,
    allow_same_trip: bool,
    spatial_mode: str,
) -> pd.DataFrame:
    high_segments = segments[segments["occupancy_status"] >= high_occupancy_min].copy()
    low_segments = segments[segments["occupancy_status"] <= low_occupancy_max].copy()

    by_cell, lat_cell_degrees, lon_cell_degrees = build_low_segment_index(
        low_segments,
        max_space_m,
    )

    max_midpoint_delta_ns = int(max_time_min * 60 * 1_000_000_000)
    results = []
    high_segments["lat_cell"] = np.floor(high_segments["mid_lat"] / lat_cell_degrees).astype(int)
    high_segments["lon_cell"] = np.floor(high_segments["mid_lon"] / lon_cell_degrees).astype(int)

    for high in high_segments.itertuples(index=False):
        high_mid_ns = high.mid_timestamp_ns
        lower_time = high_mid_ns - max_midpoint_delta_ns
        upper_time = high_mid_ns + max_midpoint_delta_ns

        for lat_offset in (-1, 0, 1):
            for lon_offset in (-1, 0, 1):
                cell = (high.lat_cell + lat_offset, high.lon_cell + lon_offset)
                low_bucket = by_cell.get(cell)
                if low_bucket is None:
                    continue

                times = low_bucket["time"]
                start_pos = bisect_left(times, lower_time)
                end_pos = bisect_right(times, upper_time)
                if start_pos == end_pos:
                    continue

                candidates = low_bucket["rows"].iloc[start_pos:end_pos]
                midpoint_distances = haversine_m(
                    high.mid_lat,
                    high.mid_lon,
                    candidates["mid_lat"].to_numpy(),
                    candidates["mid_lon"].to_numpy(),
                )
                spatial_candidates = candidates.loc[midpoint_distances <= max_space_m].copy()
                if spatial_candidates.empty:
                    continue
                spatial_candidates["midpoint_distance_m"] = midpoint_distances[
                    midpoint_distances <= max_space_m
                ]

                for low in spatial_candidates.itertuples(index=False):
                    if not allow_same_trip and high.trip_id == low.trip_id:
                        continue

                    time_gap_min = interval_gap_minutes(
                        high.departure_timestamp,
                        high.arrive_timestamp,
                        low.departure_timestamp,
                        low.arrive_timestamp,
                    )
                    if time_gap_min > max_time_min:
                        continue

                    same_start_m = haversine_m(
                        high.segment_departure_stop_lat,
                        high.segment_departure_stop_lon,
                        low.segment_departure_stop_lat,
                        low.segment_departure_stop_lon,
                    )
                    same_end_m = haversine_m(
                        high.segment_arrive_stop_lat,
                        high.segment_arrive_stop_lon,
                        low.segment_arrive_stop_lat,
                        low.segment_arrive_stop_lon,
                    )
                    opposite_start_m = haversine_m(
                        high.segment_departure_stop_lat,
                        high.segment_departure_stop_lon,
                        low.segment_arrive_stop_lat,
                        low.segment_arrive_stop_lon,
                    )
                    opposite_end_m = haversine_m(
                        high.segment_arrive_stop_lat,
                        high.segment_arrive_stop_lon,
                        low.segment_departure_stop_lat,
                        low.segment_departure_stop_lon,
                    )
                    same_direction_endpoint_distance_m = max(same_start_m, same_end_m)
                    opposite_direction_endpoint_distance_m = max(opposite_start_m, opposite_end_m)
                    endpoint_alignment_distance_m = min(
                        same_direction_endpoint_distance_m,
                        opposite_direction_endpoint_distance_m,
                    )
                    if spatial_mode == "endpoints":
                        if endpoint_alignment_distance_m > max_space_m:
                            continue
                    elif spatial_mode == "either":
                        if (
                            low.midpoint_distance_m > max_space_m
                            and endpoint_alignment_distance_m > max_space_m
                        ):
                            continue

                    results.append(
                        {
                            "high_segment_id": high.segment_id,
                            "high_trip_id": high.trip_id,
                            "high_route_short_name": high.route_short_name,
                            "high_route_type": high.route_type,
                            "high_vehicle_type": high.vehicle_type,
                            "high_from_stop": high.departure_stop_name,
                            "high_to_stop": high.arrive_stop_name,
                            "high_departure_timestamp": high.departure_timestamp,
                            "high_arrive_timestamp": high.arrive_timestamp,
                            "high_occupancy_status": high.occupancy_status,
                            "low_segment_id": low.segment_id,
                            "low_trip_id": low.trip_id,
                            "low_route_short_name": low.route_short_name,
                            "low_route_type": low.route_type,
                            "low_vehicle_type": low.vehicle_type,
                            "low_from_stop": low.departure_stop_name,
                            "low_to_stop": low.arrive_stop_name,
                            "low_departure_timestamp": low.departure_timestamp,
                            "low_arrive_timestamp": low.arrive_timestamp,
                            "low_occupancy_status": low.occupancy_status,
                            "occupancy_difference": high.occupancy_status - low.occupancy_status,
                            "time_gap_min": time_gap_min,
                            "midpoint_time_diff_min": abs(
                                high.mid_timestamp_ns - low.mid_timestamp_ns
                            ) / 1_000_000_000 / 60,
                            "midpoint_distance_m": low.midpoint_distance_m,
                            "same_direction_endpoint_distance_m": same_direction_endpoint_distance_m,
                            "opposite_direction_endpoint_distance_m": opposite_direction_endpoint_distance_m,
                            "endpoint_alignment_distance_m": endpoint_alignment_distance_m,
                        }
                    )

    result_df = pd.DataFrame(results)
    if result_df.empty:
        return result_df

    result_df = result_df.sort_values(
        [
            "occupancy_difference",
            "time_gap_min",
            "midpoint_distance_m",
            "endpoint_alignment_distance_m",
        ],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)

    return result_df


def summarize_trip_pair_overlap(
    segment_pairs: pd.DataFrame,
    segments: pd.DataFrame,
    limit: int | None,
) -> pd.DataFrame:
    if segment_pairs.empty:
        return pd.DataFrame()

    trip_totals = (
        segments
        .groupby("trip_id", as_index=False)
        .agg(
            total_segments=("segment_id", "nunique"),
            route_short_name=("route_short_name", "first"),
            route_type=("route_type", "first"),
            vehicle_type=("vehicle_type", "first"),
            trip_start=("departure_timestamp", "min"),
            trip_end=("arrive_timestamp", "max"),
            avg_occupancy_status=("occupancy_status", "mean"),
            max_occupancy_status=("occupancy_status", "max"),
        )
    )

    high_totals = trip_totals.add_prefix("high_")
    low_totals = trip_totals.add_prefix("low_")

    summary = (
        segment_pairs
        .groupby(["high_trip_id", "low_trip_id"], as_index=False)
        .agg(
            nearby_segment_pair_count=("high_segment_id", "size"),
            high_nearby_segment_count=("high_segment_id", "nunique"),
            low_nearby_segment_count=("low_segment_id", "nunique"),
            max_occupancy_difference=("occupancy_difference", "max"),
            mean_occupancy_difference=("occupancy_difference", "mean"),
            min_time_gap_min=("time_gap_min", "min"),
            mean_time_gap_min=("time_gap_min", "mean"),
            mean_midpoint_distance_m=("midpoint_distance_m", "mean"),
            mean_endpoint_alignment_distance_m=("endpoint_alignment_distance_m", "mean"),
        )
    )
    summary = summary.merge(high_totals, on="high_trip_id", how="left")
    summary = summary.merge(low_totals, on="low_trip_id", how="left")

    summary["high_trip_nearby_segment_share_percent"] = (
        summary["high_nearby_segment_count"] / summary["high_total_segments"] * 100
    )
    summary["low_trip_nearby_segment_share_percent"] = (
        summary["low_nearby_segment_count"] / summary["low_total_segments"] * 100
    )
    summary["smaller_trip_nearby_segment_share_percent"] = (
        np.minimum(
            summary["high_nearby_segment_count"],
            summary["low_nearby_segment_count"],
        )
        / np.minimum(summary["high_total_segments"], summary["low_total_segments"])
        * 100
    )
    summary["larger_trip_nearby_segment_share_percent"] = (
        np.maximum(
            summary["high_nearby_segment_count"],
            summary["low_nearby_segment_count"],
        )
        / np.maximum(summary["high_total_segments"], summary["low_total_segments"])
        * 100
    )
    summary["matched_segment_balance_percent"] = (
        np.minimum(
            summary["high_nearby_segment_count"],
            summary["low_nearby_segment_count"],
        )
        / np.maximum(
            summary["high_nearby_segment_count"],
            summary["low_nearby_segment_count"],
        )
        * 100
    )

    summary = summary.sort_values(
        [
            "smaller_trip_nearby_segment_share_percent",
            "high_trip_nearby_segment_share_percent",
            "low_trip_nearby_segment_share_percent",
            "nearby_segment_pair_count",
            "mean_endpoint_alignment_distance_m",
            "mean_time_gap_min",
        ],
        ascending=[False, False, False, False, True, True],
    ).reset_index(drop=True)

    if limit is not None:
        summary = summary.head(limit)
    return summary


def add_trip_pair_overlap_info(
    segment_pairs: pd.DataFrame,
    trip_pair_summary: pd.DataFrame,
) -> pd.DataFrame:
    if segment_pairs.empty or trip_pair_summary.empty:
        return segment_pairs

    trip_pair_columns = [
        "high_trip_id",
        "low_trip_id",
        "trip_segment_overlap_percent",
    ]
    trip_pair_summary = trip_pair_summary.copy()
    trip_pair_summary["trip_segment_overlap_percent"] = (
        trip_pair_summary["smaller_trip_nearby_segment_share_percent"]
    )
    return segment_pairs.merge(
        trip_pair_summary[trip_pair_columns],
        on=["high_trip_id", "low_trip_id"],
        how="left",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find pairs of trip segments that are spatially and temporally close "
            "but have contrasting occupancy statuses."
        )
    )
    parser.add_argument("--segments", type=Path, default=DEFAULT_SEGMENTS_FILE)
    parser.add_argument("--coords", type=Path, default=DEFAULT_COORDS_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--trip-pair-output", type=Path, default=DEFAULT_TRIP_PAIR_OUTPUT_FILE)
    parser.add_argument("--max-space-m", type=float, default=250.0)
    parser.add_argument("--max-time-min", type=float, default=10.0)
    parser.add_argument("--high-occupancy-min", type=float, default=1.0)
    parser.add_argument("--low-occupancy-max", type=float, default=0.0)
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Maximum number of segment-pair rows written. Use 0 for all rows.",
    )
    parser.add_argument(
        "--trip-pair-limit",
        type=int,
        default=1000,
        help="Maximum number of trip-pair summary rows written. Use 0 for all rows.",
    )
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
    segment_pairs = find_nearby_contrasts(
        segments=segments,
        max_space_m=args.max_space_m,
        max_time_min=args.max_time_min,
        high_occupancy_min=args.high_occupancy_min,
        low_occupancy_max=args.low_occupancy_max,
        allow_same_trip=args.allow_same_trip,
        spatial_mode=args.spatial_mode,
    )

    full_trip_pair_summary = summarize_trip_pair_overlap(
        segment_pairs,
        segments,
        limit=None,
    )
    segment_pairs_with_overlap = add_trip_pair_overlap_info(
        segment_pairs,
        full_trip_pair_summary,
    )

    segment_pair_limit = None if args.limit == 0 else args.limit
    segment_pair_output = (
        segment_pairs_with_overlap
        if segment_pair_limit is None
        else segment_pairs_with_overlap.head(segment_pair_limit)
    )
    segment_pair_output.to_csv(args.output, index=False)

    trip_pair_limit = None if args.trip_pair_limit == 0 else args.trip_pair_limit
    trip_pair_summary = (
        full_trip_pair_summary
        if trip_pair_limit is None
        else full_trip_pair_summary.head(trip_pair_limit)
    )
    trip_pair_summary.to_csv(args.trip_pair_output, index=False)

    print(f"Loaded segments with coordinates: {len(segments):,}")
    print(f"Matching segment pairs found: {len(segment_pairs):,}")
    print(f"Matching segment pairs written: {len(segment_pair_output):,}")
    print(f"Trip-pair summaries written: {len(trip_pair_summary):,}")
    print(f"Segment-pair output: {args.output}")
    print(f"Trip-pair output: {args.trip_pair_output}")


if __name__ == "__main__":
    main()
