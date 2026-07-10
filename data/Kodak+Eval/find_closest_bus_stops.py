from __future__ import annotations

import argparse
import heapq
import math
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path("GTFS-OTRAF-2026-05-04/stops.txt")
DEFAULT_OUTPUT = Path("closest_bus_stops.csv")
DEFAULT_GTFS_DIR = Path("GTFS-OTRAF-2026-05-04")


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_m = 6_371_000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    return 2 * radius_m * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def read_stops(path: Path, stop_location_type: int | None) -> pd.DataFrame:
    stops = pd.read_csv(path, dtype={"stop_id": "string", "parent_station": "string"})
    required_columns = {"stop_id", "stop_name", "stop_lat", "stop_lon"}
    missing_columns = required_columns - set(stops.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing_columns)}")

    stops["stop_lat"] = pd.to_numeric(stops["stop_lat"], errors="coerce")
    stops["stop_lon"] = pd.to_numeric(stops["stop_lon"], errors="coerce")
    stops = stops.dropna(subset=["stop_id", "stop_lat", "stop_lon"]).copy()

    if stop_location_type is not None and "location_type" in stops.columns:
        stops["location_type"] = pd.to_numeric(stops["location_type"], errors="coerce")
        stops = stops[stops["location_type"].fillna(0).astype(int) == stop_location_type].copy()

    return stops.reset_index(drop=True)


def routes_by_stop(gtfs_dir: Path) -> pd.DataFrame:
    stop_times_path = gtfs_dir / "stop_times.txt"
    trips_path = gtfs_dir / "trips.txt"
    routes_path = gtfs_dir / "routes.txt"

    if not (stop_times_path.exists() and trips_path.exists() and routes_path.exists()):
        return pd.DataFrame(columns=[
            "stop_id",
            "route_types",
            "route_short_names",
            "stop_sequences_by_route",
        ])

    stop_times = pd.read_csv(
        stop_times_path,
        usecols=["trip_id", "stop_id", "stop_sequence"],
        dtype={"trip_id": "string", "stop_id": "string"},
    ).drop_duplicates()
    stop_times["stop_sequence"] = pd.to_numeric(
        stop_times["stop_sequence"], errors="coerce"
    ).astype("Int64")
    trips = pd.read_csv(
        trips_path,
        usecols=["trip_id", "route_id"],
        dtype={"trip_id": "string", "route_id": "string"},
    ).drop_duplicates()
    routes = pd.read_csv(
        routes_path,
        usecols=["route_id", "route_short_name", "route_type"],
        dtype={"route_id": "string"},
    ).drop_duplicates()
    routes["route_type"] = pd.to_numeric(routes["route_type"], errors="coerce").astype("Int64")
    routes["route_short_name"] = routes["route_short_name"].astype("string")

    stop_routes = (
        stop_times.merge(trips, on="trip_id", how="inner")
        .merge(routes, on="route_id", how="inner")
        .drop_duplicates(subset=["stop_id", "route_type", "route_short_name"])
        .sort_values(["stop_id", "route_type", "route_short_name"])
    )

    stop_route_types = (
        stop_routes.dropna(subset=["route_type"])
        .groupby("stop_id")["route_type"]
        .apply(lambda values: "|".join(str(int(value)) for value in sorted(values.unique())))
        .rename("route_types")
    )
    stop_route_short_names = (
        stop_routes.dropna(subset=["route_short_name"])
        .groupby("stop_id")["route_short_name"]
        .apply(lambda values: "|".join(sorted(str(value) for value in values.unique())))
        .rename("route_short_names")
    )
    stop_sequences_by_route = (
        stop_routes.dropna(subset=["route_short_name", "stop_sequence"])
        .groupby("stop_id")
        .apply(format_stop_sequences_by_route)
        .rename("stop_sequences_by_route")
    )

    return (
        pd.concat(
            [stop_route_types, stop_route_short_names, stop_sequences_by_route],
            axis=1,
        )
        .reset_index()
    )


def format_stop_sequences_by_route(stop_routes: pd.DataFrame) -> str:
    parts = []
    for route_short_name, route_rows in stop_routes.groupby("route_short_name"):
        sequences = sorted(
            int(value)
            for value in route_rows["stop_sequence"].dropna().unique()
        )
        if sequences:
            parts.append(
                f"{route_short_name}:{'|'.join(str(value) for value in sequences)}"
            )
    return "; ".join(parts)


def split_route_types(route_types: object) -> set[str]:
    if pd.isna(route_types):
        return set()
    return {value for value in str(route_types).split("|") if value}


def split_pipe_values(values: object) -> set[str]:
    if pd.isna(values):
        return set()
    return {value for value in str(values).split("|") if value}


def sort_route_short_names(values: set[str]) -> list[str]:
    def sort_key(value: str) -> tuple[int, int | str]:
        if value.isdigit():
            return (0, int(value))
        return (1, value)

    return sorted(values, key=sort_key)


def find_closest_pairs(
    stops: pd.DataFrame,
    top_n: int,
    ignore_same_parent: bool,
    require_common_route_type: bool,
) -> pd.DataFrame:
    best_pairs: list[tuple[float, dict]] = []
    stop_records = stops.to_dict(orient="records")
    pair_counter = 0

    for i, stop_a in enumerate(stop_records):
        for stop_b in stop_records[i + 1 :]:
            pair_counter += 1
            parent_a = stop_a.get("parent_station")
            parent_b = stop_b.get("parent_station")
            if ignore_same_parent and pd.notna(parent_a) and parent_a == parent_b:
                continue

            route_types_a = split_route_types(stop_a.get("route_types"))
            route_types_b = split_route_types(stop_b.get("route_types"))
            common_route_types = sorted(route_types_a & route_types_b, key=int)
            if require_common_route_type and not common_route_types:
                continue
            route_short_names_a = split_pipe_values(stop_a.get("route_short_names"))
            route_short_names_b = split_pipe_values(stop_b.get("route_short_names"))
            common_route_short_names = sort_route_short_names(
                route_short_names_a & route_short_names_b
            )

            distance_m = haversine_m(
                float(stop_a["stop_lat"]),
                float(stop_a["stop_lon"]),
                float(stop_b["stop_lat"]),
                float(stop_b["stop_lon"]),
            )
            row = {
                "distance_m": distance_m,
                "stop_id_a": stop_a["stop_id"],
                "stop_name_a": stop_a["stop_name"],
                "stop_lat_a": stop_a["stop_lat"],
                "stop_lon_a": stop_a["stop_lon"],
                "parent_station_a": parent_a,
                "route_types_a": "|".join(sorted(route_types_a, key=int)),
                "route_short_names_a": "|".join(sort_route_short_names(route_short_names_a)),
                "stop_sequences_by_route_a": stop_a.get("stop_sequences_by_route"),
                "stop_id_b": stop_b["stop_id"],
                "stop_name_b": stop_b["stop_name"],
                "stop_lat_b": stop_b["stop_lat"],
                "stop_lon_b": stop_b["stop_lon"],
                "parent_station_b": parent_b,
                "route_types_b": "|".join(sorted(route_types_b, key=int)),
                "route_short_names_b": "|".join(sort_route_short_names(route_short_names_b)),
                "stop_sequences_by_route_b": stop_b.get("stop_sequences_by_route"),
                "common_route_types": "|".join(common_route_types),
                "common_route_short_names": "|".join(common_route_short_names),
            }

            # heapq is a max heap here by storing negative distances.
            heap_item = (-distance_m, pair_counter, row)
            if len(best_pairs) < top_n:
                heapq.heappush(best_pairs, heap_item)
            elif distance_m < -best_pairs[0][0]:
                heapq.heapreplace(best_pairs, heap_item)

    result_rows = [row for _, _, row in best_pairs]
    result = pd.DataFrame(result_rows)
    if result.empty:
        return result

    return result.sort_values("distance_m").reset_index(drop=True)


def build_pair_trip_stop_sequence_details(
    pairs: pd.DataFrame,
    gtfs_dir: Path,
) -> pd.DataFrame:
    stop_times_path = gtfs_dir / "stop_times.txt"
    trips_path = gtfs_dir / "trips.txt"
    routes_path = gtfs_dir / "routes.txt"

    if pairs.empty or not (
        stop_times_path.exists() and trips_path.exists() and routes_path.exists()
    ):
        return pd.DataFrame()

    relevant_stop_ids = set(pairs["stop_id_a"].astype(str)) | set(
        pairs["stop_id_b"].astype(str)
    )
    stop_times = pd.read_csv(
        stop_times_path,
        usecols=["trip_id", "stop_id", "stop_sequence"],
        dtype={"trip_id": "string", "stop_id": "string"},
    )
    stop_times = stop_times[stop_times["stop_id"].isin(relevant_stop_ids)].copy()
    stop_times["stop_sequence"] = pd.to_numeric(
        stop_times["stop_sequence"], errors="coerce"
    ).astype("Int64")
    trips = pd.read_csv(
        trips_path,
        usecols=["trip_id", "route_id", "direction_id"],
        dtype={"trip_id": "string", "route_id": "string"},
    ).drop_duplicates()
    routes = pd.read_csv(
        routes_path,
        usecols=["route_id", "route_short_name", "route_type"],
        dtype={"route_id": "string"},
    ).drop_duplicates()
    routes["route_type"] = pd.to_numeric(routes["route_type"], errors="coerce").astype("Int64")
    routes["route_short_name"] = routes["route_short_name"].astype("string")

    stop_trip_sequences = (
        stop_times.merge(trips, on="trip_id", how="inner")
        .merge(routes, on="route_id", how="inner")
        .dropna(subset=["stop_sequence"])
    )

    detail_rows = []
    for pair_rank, pair in pairs.reset_index(drop=True).iterrows():
        stop_a_rows = stop_trip_sequences[
            stop_trip_sequences["stop_id"].eq(str(pair["stop_id_a"]))
        ][["trip_id", "route_id", "route_short_name", "route_type", "direction_id", "stop_sequence"]]
        stop_b_rows = stop_trip_sequences[
            stop_trip_sequences["stop_id"].eq(str(pair["stop_id_b"]))
        ][["trip_id", "route_id", "route_short_name", "route_type", "direction_id", "stop_sequence"]]

        shared_trip_rows = stop_a_rows.merge(
            stop_b_rows,
            on=["trip_id", "route_id", "route_short_name", "route_type", "direction_id"],
            how="inner",
            suffixes=("_a", "_b"),
        )

        for row in shared_trip_rows.sort_values(
            ["route_short_name", "direction_id", "trip_id", "stop_sequence_a", "stop_sequence_b"]
        ).itertuples(index=False):
            detail_rows.append({
                "pair_rank": pair_rank + 1,
                "distance_m": pair["distance_m"],
                "stop_id_a": pair["stop_id_a"],
                "stop_name_a": pair["stop_name_a"],
                "stop_id_b": pair["stop_id_b"],
                "stop_name_b": pair["stop_name_b"],
                "trip_id": row.trip_id,
                "route_id": row.route_id,
                "route_short_name": row.route_short_name,
                "route_type": row.route_type,
                "direction_id": row.direction_id,
                "stop_sequence_a": row.stop_sequence_a,
                "stop_sequence_b": row.stop_sequence_b,
                "sequence_relation": (
                    "A before B"
                    if row.stop_sequence_a < row.stop_sequence_b
                    else "B before A"
                    if row.stop_sequence_b < row.stop_sequence_a
                    else "same sequence"
                ),
            })

    return pd.DataFrame(detail_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find the closest pair(s) of bus stops based on stop_lat/stop_lon."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to GTFS stops.txt or a CSV with stop_id, stop_name, stop_lat, stop_lon. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"CSV output path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of closest stop pairs to export. Default: 10",
    )
    parser.add_argument(
        "--location-type",
        type=int,
        default=0,
        help="Filter GTFS location_type. Use 0 for stops/platforms, 1 for stations, or -1 for all. Default: 0",
    )
    parser.add_argument(
        "--ignore-same-parent",
        action="store_true",
        help="Ignore stop pairs that belong to the same parent_station.",
    )
    parser.add_argument(
        "--gtfs-dir",
        type=Path,
        default=DEFAULT_GTFS_DIR,
        help=f"GTFS folder containing stop_times.txt, trips.txt and routes.txt. Default: {DEFAULT_GTFS_DIR}",
    )
    parser.add_argument(
        "--require-common-route-type",
        action="store_true",
        help="Only keep stop pairs that are served by at least one common route_type.",
    )
    parser.add_argument(
        "--trip-sequence-output",
        type=Path,
        default=None,
        help="Optional CSV with one row per closest-stop pair and shared trip, including stop_sequence_a and stop_sequence_b.",
    )
    args = parser.parse_args()

    if args.top_n < 1:
        raise ValueError("--top-n must be at least 1")

    stop_location_type = None if args.location_type == -1 else args.location_type
    stops = read_stops(args.input, stop_location_type)
    stop_routes = routes_by_stop(args.gtfs_dir)
    if not stop_routes.empty:
        stops = stops.merge(stop_routes, on="stop_id", how="left")
    else:
        stops["route_types"] = pd.NA
        stops["route_short_names"] = pd.NA

    result = find_closest_pairs(
        stops,
        args.top_n,
        args.ignore_same_parent,
        args.require_common_route_type,
    )

    if result.empty:
        print("No stop pair found.")
        return

    result.to_csv(args.output, index=False)
    if args.trip_sequence_output is not None:
        trip_sequence_details = build_pair_trip_stop_sequence_details(
            result,
            args.gtfs_dir,
        )
        trip_sequence_details.to_csv(args.trip_sequence_output, index=False)

    closest = result.iloc[0]
    print(f"Analyzed stops: {len(stops):,}")
    print(f"Closest pair distance: {closest['distance_m']:.2f} m")
    print(
        f"Stop A: {closest['stop_name_a']} ({closest['stop_id_a']}) "
        f"[{closest['stop_lat_a']}, {closest['stop_lon_a']}], "
        f"route_type(s): {closest['route_types_a']}, route_short_name(s): {closest['route_short_names_a']}"
    )
    print(
        f"Stop B: {closest['stop_name_b']} ({closest['stop_id_b']}) "
        f"[{closest['stop_lat_b']}, {closest['stop_lon_b']}], "
        f"route_type(s): {closest['route_types_b']}, route_short_name(s): {closest['route_short_names_b']}"
    )
    print(f"Common route_type(s): {closest['common_route_types']}")
    print(f"Common route_short_name(s): {closest['common_route_short_names']}")
    print(f"Top {len(result)} pairs written to: {args.output}")
    if args.trip_sequence_output is not None:
        print(f"Shared-trip stop sequences written to: {args.trip_sequence_output}")


if __name__ == "__main__":
    main()
