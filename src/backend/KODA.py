import json
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import pdfRepr

try:
    from tabulate import tabulate
except ModuleNotFoundError:
    def tabulate(data, *args, **kwargs):
        return str(data)

ACTIVITY_ORDER = {
    "begin_shift": 0,
    "direction_change": 1,
    "end_layover": 2,
    "arrive_stop": 3,
    "departure_stop": 4,
    "begin_layover": 5,
    "parking": 6,
}

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parents[1]
DEFAULT_PIPELINE_DIR = PROJECT_DIR / "koda_pipeline_downloads" / "koda_otraf_2026_05_04"
PIPELINE_DIR_FROM_ENV = "KODA_PIPELINE_DIR" in os.environ
PIPELINE_INPUT_DIR = Path(
    os.environ.get("KODA_PIPELINE_DIR", DEFAULT_PIPELINE_DIR)
).expanduser()
GTFS_INPUT_DIR = Path(
    os.environ.get("KODA_GTFS_INPUT_DIR", PIPELINE_INPUT_DIR / "extracted" / "gtfs_static")
).expanduser()
TRIP_UPDATES_INPUT_FILE = Path(
    os.environ.get(
        "KODA_TRIP_UPDATES_INPUT_FILE",
        PIPELINE_INPUT_DIR / "csv" / "trip_updates" / "all_trip_updates.csv",
    )
).expanduser()
VEHICLE_POSITIONS_INPUT_DIR = Path(
    os.environ.get(
        "KODA_VEHICLE_POSITIONS_INPUT_DIR",
        PIPELINE_INPUT_DIR / "csv" / "vehicle_positions",
    )
).expanduser()
OCCUPANCY_MODE = os.environ.get("KODA_OCCUPANCY_MODE", "snapshots")
TRIP_OCCUPANCY_INPUT_FILE = Path(
    os.environ.get(
        "KODA_TRIP_OCCUPANCY_INPUT_FILE",
        VEHICLE_POSITIONS_INPUT_DIR / "trip_occupancy_ultra_fastlane.csv",
    )
).expanduser()
VEHICLE_POSITION_DTYPE = {
    "trip_id": "string",
    "vehicle_id": "string",
    "stop_id": "string",
    "route_id": "string",
}
VEHICLE_POSITION_USECOLS = [
    "route_id",
    "trip_id",
    "vehicle_id",
    "direction_id",
    "latitude",
    "longitude",
    "start_date",
    "timestamp",
    "occupancy_status",
    "speed",
]


def existing_input_path(path, fallback):
    path = Path(path).expanduser()
    if path.exists():
        return path
    fallback = Path(fallback).expanduser()
    if fallback.exists():
        return fallback
    return path


if not PIPELINE_DIR_FROM_ENV:
    GTFS_INPUT_DIR = existing_input_path(
        GTFS_INPUT_DIR,
        PROJECT_DIR / "GTFS-OTRAF-2026-05-04",
    )
    TRIP_UPDATES_INPUT_FILE = existing_input_path(
        TRIP_UPDATES_INPUT_FILE,
        PROJECT_DIR / "tripupdates_output3-2026-05-04" / "all_trip_updates.csv",
    )


def read_vehicle_positions_updates(input_dir):
    csv_file = input_dir / "all_vehicle_positions.csv"
    if not csv_file.exists():
        raise FileNotFoundError(
            f"No VehiclePositions CSV file found: {csv_file}. "
            "Run Preprocessing first to create it from all available PB files."
        )

    df_vehicle_positions_updates = pd.read_csv(
        csv_file,
        sep=",",
        usecols=lambda column: column in VEHICLE_POSITION_USECOLS,
        dtype=VEHICLE_POSITION_DTYPE,
    )
    #print(df_vehicle_positions_updates.columns)
    return df_vehicle_positions_updates


def read_trip_occupancy_by_trip(csv_file):
    csv_file = Path(csv_file).expanduser()
    if not csv_file.exists():
        raise FileNotFoundError(
            f"No Ultra Fastlane occupancy table found: {csv_file}. "
            "Run preprocessing first to create trip_occupancy_ultra_fastlane.csv."
        )

    return pd.read_csv(
        csv_file,
        sep=",",
        usecols=["trip_id", "occupancy_status"],
        dtype={"trip_id": "string"},
    ).drop_duplicates(subset=["trip_id"], keep="first")


def haversine_distance_m(lat1, lon1, lat2, lon2):
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return math.inf
    radius_m = 6371000
    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    delta_phi = math.radians(float(lat2) - float(lat1))
    delta_lambda = math.radians(float(lon2) - float(lon1))
    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    return 2 * radius_m * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def resolve_service_date():
    env_date = os.environ.get("KODA_SERVICE_DATE")
    if env_date:
        service_date = pd.to_datetime(env_date, errors="coerce")
        if pd.notna(service_date):
            return service_date.normalize()

    match = re.search(r"(\d{4})[_-](\d{2})[_-](\d{2})", str(PIPELINE_INPUT_DIR))
    if match:
        service_date = pd.to_datetime("-".join(match.groups()), errors="coerce")
        if pd.notna(service_date):
            return service_date.normalize()

    return None


def active_service_ids_for_date(df_calendar, df_calendar_dates, service_date):
    if service_date is None:
        return set()

    weekday_column = service_date.day_name().lower()
    active_service_ids = set()

    if not df_calendar.empty and weekday_column in df_calendar.columns:
        calendar = df_calendar.copy()
        calendar["start_date"] = pd.to_datetime(
            calendar["start_date"].astype(str),
            format="%Y%m%d",
            errors="coerce",
        ).dt.normalize()
        calendar["end_date"] = pd.to_datetime(
            calendar["end_date"].astype(str),
            format="%Y%m%d",
            errors="coerce",
        ).dt.normalize()
        weekday_active = pd.to_numeric(calendar[weekday_column], errors="coerce").fillna(0).eq(1)
        in_date_range = calendar["start_date"].le(service_date) & calendar["end_date"].ge(service_date)
        active_service_ids.update(
            calendar.loc[weekday_active & in_date_range, "service_id"].dropna().astype(str)
        )

    if not df_calendar_dates.empty:
        calendar_dates = df_calendar_dates.copy()
        calendar_dates["date"] = pd.to_datetime(
            calendar_dates["date"].astype(str),
            format="%Y%m%d",
            errors="coerce",
        ).dt.normalize()
        day_exceptions = calendar_dates[calendar_dates["date"].eq(service_date)]
        added = day_exceptions.loc[
            pd.to_numeric(day_exceptions["exception_type"], errors="coerce").eq(1),
            "service_id",
        ].dropna().astype(str)
        removed = day_exceptions.loc[
            pd.to_numeric(day_exceptions["exception_type"], errors="coerce").eq(2),
            "service_id",
        ].dropna().astype(str)
        active_service_ids.update(added)
        active_service_ids.difference_update(removed)

    return active_service_ids


def nearest_vehicle_snapshots_by_time(
    departure_events,
    vehicle_positions,
    stop_key_columns,
    vehicle_snapshot_columns,
):
    departures = departure_events.dropna(subset=["trip_id", "departure_time"]).copy()
    snapshots = vehicle_positions.dropna(subset=["trip_id", "timestamp"]).copy()
    if departures.empty or snapshots.empty:
        return pd.DataFrame(columns=stop_key_columns + vehicle_snapshot_columns)

    snapshot_groups = {
        str(trip_id): group.sort_values("timestamp").reset_index(drop=True)
        for trip_id, group in snapshots.groupby("trip_id", sort=False)
    }

    matched_groups = []
    for trip_id, departure_group in departures.groupby("trip_id", sort=False):
        snapshot_group = snapshot_groups.get(str(trip_id))
        if snapshot_group is None or snapshot_group.empty:
            continue

        snapshot_times = snapshot_group["timestamp"].to_numpy(dtype="datetime64[ns]")
        departure_times = departure_group["departure_time"].to_numpy(dtype="datetime64[ns]")
        insertion_points = np.searchsorted(snapshot_times, departure_times)

        nearest_positions = []
        for insertion_point, departure_time in zip(insertion_points, departure_times):
            candidates = []
            if insertion_point > 0:
                candidates.append(insertion_point - 1)
            if insertion_point < len(snapshot_times):
                candidates.append(insertion_point)
            if not candidates:
                continue
            nearest_positions.append(
                min(
                    candidates,
                    key=lambda candidate: abs(snapshot_times[candidate] - departure_time),
                )
            )

        if not nearest_positions:
            continue

        matched_keys = departure_group[stop_key_columns].reset_index(drop=True)
        matched_snapshots = (
            snapshot_group.iloc[nearest_positions][vehicle_snapshot_columns]
            .reset_index(drop=True)
        )
        matched_groups.append(pd.concat([matched_keys, matched_snapshots], axis=1))

    if not matched_groups:
        return pd.DataFrame(columns=stop_key_columns + vehicle_snapshot_columns)

    return (
        pd.concat(matched_groups, ignore_index=True)
        .drop_duplicates(subset=stop_key_columns, keep="first")
    )


def sort_by_activity_order(df, timestamp_col, activity_col):
    ordered_df = df.copy()
    ordered_df["_timestamp_sort"] = pd.to_datetime(ordered_df[timestamp_col])
    ordered_df["_activity_order"] = (
        ordered_df[activity_col].map(ACTIVITY_ORDER).fillna(len(ACTIVITY_ORDER))
    )
    ordered_df = ordered_df.sort_values(
        ["_timestamp_sort", "_activity_order", activity_col]
    )
    return ordered_df.drop(
        columns=["_timestamp_sort", "_activity_order"]
    ).reset_index(drop=True)


def derive_activity_type(activity):
    if pd.isna(activity):
        return activity
    activity = str(activity)
    if activity.startswith("arrive_stop"):
        return "arrive_stop"
    if activity.startswith("departure_stop"):
        return "departure_stop"
    if activity in ACTIVITY_ORDER:
        return activity
    return activity


def route_type_category(route_type):
    if pd.isna(route_type):
        return pd.NA
    try:
        route_type_number = int(float(route_type))
    except (TypeError, ValueError):
        return pd.NA

    if 100 <= route_type_number <= 117:
        return "Railway"
    if 200 <= route_type_number <= 209:
        return "Coach"
    if 400 <= route_type_number <= 405:
        return "Urban Railway"
    if 700 <= route_type_number <= 716:
        return "Bus"
    if route_type_number == 800:
        return "Trolleybus"
    if 900 <= route_type_number <= 906:
        return "Tram"
    if route_type_number == 1000:
        return "Water Transport"
    if route_type_number == 1100:
        return "Aeroplane"
    if route_type_number == 1200:
        return "Ferry"
    if 1300 <= route_type_number <= 1307:
        return "Aerial Lift"
    if route_type_number == 1400:
        return "Funicular"
    if 1500 <= route_type_number <= 1507:
        return "Taxi"
    if 1700 <= route_type_number <= 1702:
        return "Miscellaneous vehicle"
    return pd.NA


def add_segment_ids(aux_table):
    aux_table = aux_table.copy()
    segment_columns = [
        "segment_id",
        "segment_departure_stop_id",
        "segment_departure_stop_lat",
        "segment_departure_stop_lon",
        "segment_arrive_stop_id",
        "segment_arrive_stop_lat",
        "segment_arrive_stop_lon",
    ]
    for column in segment_columns:
        aux_table[column] = pd.NA

    stop_events = aux_table[
        aux_table["activity_type"].isin(["departure_stop", "arrive_stop"])
    ].copy()
    stop_events["_row_index"] = stop_events.index
    stop_events["_timestamp_sort"] = pd.to_datetime(
        stop_events["timestamp"], errors="coerce"
    )
    stop_events["_stop_sequence_sort"] = pd.to_numeric(
        stop_events["stop_sequence"], errors="coerce"
    )
    stop_events = stop_events.sort_values(
        ["trip_id", "_timestamp_sort", "_stop_sequence_sort", "_row_index"]
    )

    grouped = stop_events.groupby("trip_id", sort=False)
    stop_events["next_row_index"] = grouped["_row_index"].shift(-1)
    stop_events["next_activity_type"] = grouped["activity_type"].shift(-1)
    stop_events["next_stop_id"] = grouped["stop_id"].shift(-1)
    stop_events["next_stop_lat"] = grouped["stop_lat"].shift(-1)
    stop_events["next_stop_lon"] = grouped["stop_lon"].shift(-1)

    segments = stop_events[
        (stop_events["activity_type"] == "departure_stop")
        & (stop_events["next_activity_type"] == "arrive_stop")
    ].copy()
    if segments.empty:
        return aux_table

    pair_columns = [
        "stop_id",
        "stop_lat",
        "stop_lon",
        "next_stop_id",
        "next_stop_lat",
        "next_stop_lon",
    ]
    segments["segment_pair_key"] = (
        segments[pair_columns]
        .astype("string")
        .fillna("<NA>")
        .agg("|".join, axis=1)
    )
    segment_id_by_pair = {
        key: f"seg_{idx}"
        for idx, key in enumerate(sorted(segments["segment_pair_key"].unique()), start=1)
    }
    segments["segment_id"] = segments["segment_pair_key"].map(segment_id_by_pair)

    for _, segment in segments.iterrows():
        row_indexes = [
            int(segment["_row_index"]),
            int(segment["next_row_index"]),
        ]
        aux_table.loc[row_indexes, "segment_id"] = segment["segment_id"]
        aux_table.loc[row_indexes, "segment_departure_stop_id"] = segment["stop_id"]
        aux_table.loc[row_indexes, "segment_departure_stop_lat"] = segment["stop_lat"]
        aux_table.loc[row_indexes, "segment_departure_stop_lon"] = segment["stop_lon"]
        aux_table.loc[row_indexes, "segment_arrive_stop_id"] = segment["next_stop_id"]
        aux_table.loc[row_indexes, "segment_arrive_stop_lat"] = segment["next_stop_lat"]
        aux_table.loc[row_indexes, "segment_arrive_stop_lon"] = segment["next_stop_lon"]

    return aux_table


def build_segment_overview(aux_table):
    segment_columns = [
        "segment_id",
        "segment_departure_stop_id",
        "segment_departure_stop_lat",
        "segment_departure_stop_lon",
        "segment_arrive_stop_id",
        "segment_arrive_stop_lat",
        "segment_arrive_stop_lon",
    ]
    return (
        aux_table.dropna(subset=["segment_id"])
        [segment_columns]
        .drop_duplicates(subset=["segment_id"])
        .sort_values("segment_id")
        .reset_index(drop=True)
    )


def attach_segment_distances(segment_overview, distances_path):
    segment_overview = segment_overview.copy()
    distance_columns = [
        "segment_id",
        "air_distance_m",
        "road_distance_m",
        "road_distance_status",
    ]

    try:
        segment_distances = pd.read_csv(
            distances_path,
            dtype={"segment_id": "string"},
        )
    except FileNotFoundError:
        segment_overview["air_distance_m"] = pd.NA
        segment_overview["road_distance_m"] = pd.NA
        segment_overview["road_distance_status"] = "segment_distances_file_missing"
        return segment_overview

    segment_distances = (
        segment_distances[distance_columns]
        .drop_duplicates(subset=["segment_id"])
    )
    return segment_overview.merge(
        segment_distances,
        on="segment_id",
        how="left",
    )


def add_cumulative_road_distance(aux_table):
    aux_table = aux_table.copy()
    aux_table["_row_index"] = aux_table.index
    aux_table["_timestamp_sort"] = pd.to_datetime(
        aux_table["timestamp"], errors="coerce"
    )
    aux_table["_stop_sequence_sort"] = pd.to_numeric(
        aux_table["stop_sequence"], errors="coerce"
    )
    aux_table["_road_distance_contribution_m"] = pd.to_numeric(
        aux_table["road_distance_m"], errors="coerce"
    ).fillna(0)
    aux_table.loc[
        aux_table["activity_type"] != "arrive_stop",
        "_road_distance_contribution_m",
    ] = 0

    sorted_index = aux_table.sort_values(
        ["trip_id", "_timestamp_sort", "_stop_sequence_sort", "_row_index"]
    ).index
    aux_table.loc[sorted_index, "cumulative_road_distance_m"] = (
        aux_table.loc[sorted_index]
        .groupby("trip_id")["_road_distance_contribution_m"]
        .cumsum()
    )
    aux_table["cumulative_road_distance_km"] = (
        aux_table["cumulative_road_distance_m"] / 1000
    )

    return aux_table.drop(
        columns=[
            "_row_index",
            "_timestamp_sort",
            "_stop_sequence_sort",
            "_road_distance_contribution_m",
        ]
    )


def split_trip_ids_on_day_sequence_reset(aux_table):
    aux_table = aux_table.copy()
    aux_table["_row_index"] = aux_table.index
    aux_table["_timestamp_sort"] = pd.to_datetime(
        aux_table["timestamp"], errors="coerce"
    )
    aux_table["_stop_sequence_sort"] = pd.to_numeric(
        aux_table["stop_sequence"], errors="coerce"
    )

    sortable = aux_table.dropna(
        subset=["trip_id", "_timestamp_sort", "_stop_sequence_sort"]
    ).copy()
    sortable = sortable.sort_values(
        ["trip_id", "_timestamp_sort", "_stop_sequence_sort", "_row_index"]
    )

    grouped = sortable.groupby("trip_id", sort=False)
    previous_stop_sequence = grouped["_stop_sequence_sort"].shift(1)

    sortable["_trip_split_number"] = (
        (sortable["_stop_sequence_sort"] < previous_stop_sequence)
        .groupby(sortable["trip_id"])
        .cumsum()
    )

    existing_trip_ids = set(aux_table["trip_id"].dropna().astype(str))
    split_trip_id_cache = {}

    def make_split_trip_id(row):
        base_trip_id = str(row["trip_id"])
        split_number = int(row["_trip_split_number"])
        if split_number == 0:
            return base_trip_id

        cache_key = (base_trip_id, split_number)
        if cache_key in split_trip_id_cache:
            return split_trip_id_cache[cache_key]

        suffix_number = split_number + 1
        candidate = f"{base_trip_id}__split_{suffix_number}"
        while candidate in existing_trip_ids:
            suffix_number += 1
            candidate = f"{base_trip_id}__split_{suffix_number}"

        existing_trip_ids.add(candidate)
        split_trip_id_cache[cache_key] = candidate
        return candidate

    sortable["_new_trip_id"] = sortable.apply(make_split_trip_id, axis=1)
    aux_table.loc[sortable["_row_index"], "trip_id"] = sortable["_new_trip_id"].values
    split_count = int((sortable["_trip_split_number"] > 0).sum())
    #print(f"trip_id split rows after stop_sequence reset: {split_count}")

    return aux_table.drop(
        columns=["_row_index", "_timestamp_sort", "_stop_sequence_sort"]
    )


def normalize_json_value(value):
    if isinstance(value, dict):
        return {k: normalize_json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize_json_value(v) for v in value]
    if isinstance(value, tuple):
        return [normalize_json_value(v) for v in value]
    if pd.isna(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def dedupe_exported_objects(table, table_name):
    dedupe_signature = table.apply(
        lambda row: json.dumps(
            normalize_json_value(
                {
                    "id": row["id"],
                    "type": row["type"],
                    "attributes": row["attributes"],
                    "relationships": row["relationships"],
                }
            ),
            sort_keys=True,
            ensure_ascii=False,
        ),
        axis=1,
    )

    duplicate_count = int(dedupe_signature.duplicated().sum())
    deduped_table = (
        table.assign(_dedupe_signature=dedupe_signature)
        .drop_duplicates(subset="_dedupe_signature", keep="first")
        .drop(columns="_dedupe_signature")
        .reset_index(drop=True)
    )

    #print(f"{table_name}: {duplicate_count} duplicate objects removed")
    return deduped_table, duplicate_count
"""
Inital Loads

"""

df_stops = pd.read_csv(
    GTFS_INPUT_DIR / "stops.txt",
    sep=",",
    dtype={"stop_id": "string"}
)

df_routes = pd.read_csv(
    GTFS_INPUT_DIR / "routes.txt",
    encoding="latin1",
    sep=",",
    dtype={"route_id": "string"}
)

df_agency = pd.read_csv(
    GTFS_INPUT_DIR / "agency.txt",
    sep=",",
    dtype={"agency_id": "string"}
)

df_stop_times_plan = pd.read_csv(
    GTFS_INPUT_DIR / "stop_times.txt",
    sep=",",
    dtype={
        "trip_id": "string",
        "stop_id": "string",
    }
)

df_stop_times_rt = pd.read_csv(
    TRIP_UPDATES_INPUT_FILE,
    encoding="latin1",
    sep=",",
    dtype={
        "trip_id": "string",
        "vehicle_id": "string",
        "stop_id": "string",
    }
)
df_trips = pd.read_csv(
    GTFS_INPUT_DIR / "trips.txt",
    sep=",",
    dtype={
        "trip_id": "string",
        "vehicle_id": "string",
        "route_id": "string",
        "service_id": "string",
        "shape_id": "string"
    }
)
if OCCUPANCY_MODE == "trip_constant":
    df_trip_occupancy = read_trip_occupancy_by_trip(TRIP_OCCUPANCY_INPUT_FILE)
    df_vehicle_positions = pd.DataFrame(
        columns=[
            "route_id",
            "trip_id",
            "vehicle_id",
            "direction_id",
            "latitude",
            "longitude",
            "start_date",
            "timestamp",
            "occupancy_status",
            "speed",
        ]
    )
else:
    df_trip_occupancy = pd.DataFrame(columns=["trip_id", "occupancy_status"])
    df_vehicle_positions = read_vehicle_positions_updates(VEHICLE_POSITIONS_INPUT_DIR)

calendar_file = GTFS_INPUT_DIR / "calendar.txt"
if calendar_file.exists():
    df_calendar = pd.read_csv(
        calendar_file,
        sep=",",
        dtype={"service_id": "string"},
    )
else:
    df_calendar = pd.DataFrame()

df_calendar_dates = pd.read_csv(
    GTFS_INPUT_DIR / "calendar_dates.txt",
    sep=",",
    dtype={"service_id": "string"},
)

service_date = resolve_service_date()
active_service_ids = active_service_ids_for_date(
    df_calendar,
    df_calendar_dates,
    service_date,
)

df_trips["trip_id"] = df_trips["trip_id"].astype("string")
df_trips["service_id"] = df_trips["service_id"].astype("string")
df_stop_times_plan["trip_id"] = df_stop_times_plan["trip_id"].astype("string")
df_stop_times_rt["trip_id"] = df_stop_times_rt["trip_id"].astype("string")
df_vehicle_positions["trip_id"] = df_vehicle_positions["trip_id"].astype("string")

day_trip_ids = set(
    df_trips.loc[
        df_trips["service_id"].astype(str).isin(active_service_ids),
        "trip_id",
    ].dropna().astype(str)
)
rt_trip_ids = set(df_stop_times_rt["trip_id"].dropna().astype(str))
allowed_trip_ids = day_trip_ids
day_trips_with_rt = day_trip_ids & rt_trip_ids
rt_trips_not_in_calendar_day = rt_trip_ids - day_trip_ids

df_trips = df_trips[df_trips["trip_id"].astype(str).isin(allowed_trip_ids)].copy()
df_stop_times_plan = df_stop_times_plan[
    df_stop_times_plan["trip_id"].astype(str).isin(allowed_trip_ids)
].copy()
df_stop_times_rt = df_stop_times_rt[
    df_stop_times_rt["trip_id"].astype(str).isin(allowed_trip_ids)
].copy()
df_vehicle_positions = df_vehicle_positions[
    df_vehicle_positions["trip_id"].astype(str).isin(allowed_trip_ids)
].copy()
df_trip_occupancy = df_trip_occupancy[
    df_trip_occupancy["trip_id"].astype(str).isin(allowed_trip_ids)
].copy()

print(
    "Trip day filter:",
    f"service_date={service_date.date() if service_date is not None else 'unknown'}",
    f"active_services={len(active_service_ids)}",
    f"day_trips={len(day_trip_ids)}",
    f"rt_trips={len(rt_trip_ids)}",
    f"day_trips_with_rt={len(day_trips_with_rt)}",
    f"day_trips_without_rt={len(day_trip_ids - rt_trip_ids)}",
    f"rt_trips_not_in_calendar_day={len(rt_trips_not_in_calendar_day)}",
    f"used_trips={len(allowed_trip_ids)}",
)

df_vehicle_positions["speed"] = pd.to_numeric(
    df_vehicle_positions["speed"],
    errors="coerce"
)


trip_speed_summary = (
    df_vehicle_positions.dropna(subset=["trip_id", "speed"])
    .groupby("trip_id", as_index=False)
    .agg(
        avg_speed=("speed", "mean"),
        median_speed=("speed", "median"),
        speed_q25=("speed", lambda x: x.quantile(0.25)),
        speed_q58=("speed", lambda x: x.quantile(0.58)),
        speed_q60=("speed", lambda x: x.quantile(0.60)),
        speed_q625=("speed", lambda x: x.quantile(0.625)),
        speed_q75=("speed", lambda x: x.quantile(0.75)),
        speed_count=("speed", "count")
    )
    .sort_values("avg_speed", ascending=False)
)

trip_speed_summary.to_csv("trip_speed_summary_2026_05_04.csv")



df_shapes = pd.read_csv(GTFS_INPUT_DIR / "shapes.txt",
                        sep=",",
                        dtype={"shape_id": "string"})

df_stop_times_rt.drop(columns=["feed_timestamp", "source_file"], inplace=True)



# Keep all identifier columns aligned across GTFS/RT sources before merging.
df_routes["route_id"] = df_routes["route_id"].astype("string")
df_routes["agency_id"] = df_routes["agency_id"].astype("string")
df_agency["agency_id"] = df_agency["agency_id"].astype("string")
df_trips["trip_id"] = df_trips["trip_id"].astype("string")
df_trips["route_id"] = df_trips["route_id"].astype("string")
df_trips["service_id"] = df_trips["service_id"].astype("string")
df_vehicle_positions["trip_id"] = df_vehicle_positions["trip_id"].astype("string")
df_vehicle_positions["route_id"] = df_vehicle_positions["route_id"].astype("string")
df_vehicle_positions["vehicle_id"] = df_vehicle_positions["vehicle_id"].astype("string")




#df_stop_times_rt.drop(columns=["feed_timestamp", "source_file"], inplace=True)

#print(tabulate(df_stop_times_plan[df_stop_times_plan["trip_id"]=='55700000081956048'].head(10), headers="keys", tablefmt="psql"))

#print(tabulate(df_stop_times_rt[df_stop_times_rt["trip_id"]=='55700000081956048'].head(10), headers="keys", tablefmt="psql"))


df_vehicle_positions = df_vehicle_positions[["route_id", "trip_id", "direction_id", "latitude", "longitude", "start_date", "timestamp", "occupancy_status"]]

#df_vehicle_positions = df_vehicle_positions[df_vehicle_positions["trip_id"]==55700000080585229]


#df_stop_times_rt["vehicle_id"] = "veh_" + pd.factorize(df_stop_times_rt["vehicle_id"])[0].astype(str)

#print(df_stop_times_rt["trip_id"].unique())

"""
Event Table - Creation of Events

"""


df_trips = df_trips.reset_index(drop=True)
df_trips["trip_id_unique"] = "tr03_" + (df_trips.index + 1).astype(str)



event_log_arrival_plan = df_stop_times_plan.sort_values("stop_sequence").merge(df_stops, on="stop_id", how="inner")

event_log_arrival_rt = df_stop_times_rt.sort_values("stop_sequence").merge(df_stops, on="stop_id", how="inner")
#print(tabulate(event_log_arrival, headers="keys", tablefmt="psql", showindex=False))

event_log_departure_plan = df_stop_times_plan.sort_values("stop_sequence").merge(df_stops, on="stop_id", how="inner")

event_log_departure_rt = df_stop_times_rt.sort_values("stop_sequence").merge(df_stops, on="stop_id", how="inner")
#print(tabulate(event_log_arrival, headers="keys", tablefmt="psql", showindex=False))



event_log_arrival_plan = event_log_arrival_plan.drop(columns=["departure_time"])

event_log_departure_plan = event_log_departure_plan.drop(columns=["arrival_time"])



event_log_arrival_plan["activity"] = (
    "arrive_stop_" + event_log_arrival_plan["stop_name"].fillna("unknown").astype(str)
)

event_log_arrival_rt["activity"] = (
    "arrive_stop_" + event_log_arrival_rt["stop_name"].fillna("unknown").astype(str)
)

#print(tabulate(event_log_arrival_plan.head(10), headers="keys",tablefmt="psql"))

event_log_departure_plan["activity"] = (
    "departure_stop_" + event_log_departure_plan["stop_name"].fillna("unknown").astype(str)
)

event_log_departure_rt["activity"] = (
    "departure_stop_" + event_log_departure_rt["stop_name"].fillna("unknown").astype(str)
)

event_log_arrival_rt = event_log_arrival_rt.drop(columns=["departure_time"])

event_log_departure_rt = event_log_departure_rt.drop(columns=["arrival_time"])



event_log_arrival_plan["delay"] = None

event_log_arrival_plan["vehicle_id"] = None

event_log_departure_plan["delay"] = None

event_log_departure_plan["vehicle_id"] = None

event_log_arrival_rt["service_date_rt"] = pd.to_datetime(
    event_log_arrival_rt["arrival_time"],
    errors="coerce"
).dt.normalize()

event_log_departure_rt["service_date_rt"] = pd.to_datetime(
    event_log_departure_rt["departure_time"],
    errors="coerce"
).dt.normalize()

event_log_arrival_rt_projection = event_log_arrival_rt[
    ["trip_id", "vehicle_id", "arrival_time", "delay", "stop_sequence", "service_date_rt"]
]

event_log_arrival_rt_projection = event_log_arrival_rt_projection.rename(columns={
    "arrival_time": "arrival_time_rt",
    "delay": "delay_rt",
    "vehicle_id": "vehicle_id_rt"
})

event_log_departure_rt_projection = event_log_departure_rt[
    ["trip_id", "vehicle_id", "departure_time", "delay", "stop_sequence", "service_date_rt"]
]

event_log_departure_rt_projection = event_log_departure_rt_projection.rename(columns={
    "departure_time": "departure_time_rt",
    "delay": "delay_rt",
    "vehicle_id": "vehicle_id_rt"
})

keys = ["trip_id", "stop_sequence"]

event_log_arrival = event_log_arrival_plan.merge(event_log_arrival_rt_projection, on = keys, how = "left")
#print("event_log_arrival")
#print(tabulate(event_log_arrival[event_log_arrival["trip_id"]=='55700000081974302'].head(10), headers="keys",tablefmt="psql"))
#print("event_log_tet2")
#print(tabulate(event_log_arrival[event_log_arrival["pickup_type"]==3].head(10), headers="keys",tablefmt="psql"))

event_log_arrival["arrival_time"] = event_log_arrival["arrival_time_rt"].combine_first(event_log_arrival["arrival_time"])
event_log_arrival["delay"] = event_log_arrival["delay_rt"].combine_first(event_log_arrival["delay"])
event_log_arrival["vehicle_id"] = event_log_arrival["vehicle_id_rt"].combine_first(event_log_arrival["vehicle_id"])

event_log_departure = event_log_departure_plan.merge(event_log_departure_rt_projection, on = keys, how = "left")

event_log_departure["departure_time"] = event_log_departure["departure_time_rt"].combine_first(event_log_departure["departure_time"])
event_log_departure["delay"] = event_log_departure["delay_rt"].combine_first(event_log_departure["delay"])
event_log_departure["vehicle_id"] = event_log_departure["vehicle_id_rt"].combine_first(event_log_departure["vehicle_id"])




#print("tet3")
#print(tabulate(event_log_departure_plan[(event_log_departure_plan["trip_id"]=='55700000081974302')].head(50), headers="keys", tablefmt="psql"))

#print(tabulate(event_log_departure_rt[(event_log_departure_rt["trip_id"]=='55700000078905434')].head(50), headers="keys", tablefmt="psql"))




event_log_arrival = event_log_arrival.drop_duplicates(
    subset=[
        "trip_id",
        "stop_id",
        "stop_sequence",
        "arrival_time",
        "stop_name",
        "stop_lat",
        "stop_lon"
    ],
    keep="first"
).reset_index(drop=True)

event_log_departure = event_log_departure.drop_duplicates(
    subset=[
        "trip_id",
        "stop_id",
        "stop_sequence",
        "departure_time",
        "stop_name",
        "stop_lat",
        "stop_lon"
    ],
    keep="first"
).reset_index(drop=True)


#print(tabulate(event_log_departure[(event_log_departure["trip_id"]==55700000080585229) | (event_log_departure["trip_id"]==55700000082890402)| (event_log_departure["trip_id"]==55700000080524085)].head(50), headers="keys", tablefmt="psql"))
if OCCUPANCY_MODE != "trip_constant":
    df_vehicle_positions["latitude"]=df_vehicle_positions["latitude"].round(4)

    df_vehicle_positions["longitude"]=df_vehicle_positions["longitude"].round(4)

    df_vehicle_positions = df_vehicle_positions.drop_duplicates(
        subset=[
            "trip_id",
            "direction_id",
            "latitude",
            "longitude",
            "timestamp",
            "occupancy_status"
        ],
        keep="first"
    ).reset_index(drop=True)

    df_vehicle_positions["timestamp"] = pd.to_datetime(
        df_vehicle_positions["timestamp"],
        errors="coerce",
    )

event_log_arrival["service_date"] = service_date
event_log_arrival["service_date"] = event_log_arrival["service_date_rt"].combine_first(
    event_log_arrival["service_date"]
)

event_log_departure["service_date"] = service_date
event_log_departure["service_date"] = event_log_departure["service_date_rt"].combine_first(
    event_log_departure["service_date"]
)


event_log_arrival["arrival_time_raw"] = event_log_arrival["arrival_time"].astype(str).str.strip()
event_log_departure["departure_time_raw"] = event_log_departure["departure_time"].astype(str).str.strip()

arrival_time_as_datetime = pd.to_datetime(
    event_log_arrival["arrival_time_raw"],
    format="%Y-%m-%d %H:%M:%S",
    errors="coerce"
)
arrival_time_as_offset = (
    event_log_arrival["service_date"]
    + pd.to_timedelta(event_log_arrival["arrival_time_raw"], errors="coerce")
)
event_log_arrival["arrival_time"] = arrival_time_as_datetime.combine_first(
    arrival_time_as_offset
)

departure_time_as_datetime = pd.to_datetime(
    event_log_departure["departure_time_raw"],
    format="%Y-%m-%d %H:%M:%S",
    errors="coerce"
)
departure_time_as_offset = (
    event_log_departure["service_date"]
    + pd.to_timedelta(event_log_departure["departure_time_raw"], errors="coerce")
)
event_log_departure["departure_time"] = departure_time_as_datetime.combine_first(
    departure_time_as_offset
)

#print("after combine first test")
#print(tabulate(event_log_departure[event_log_departure["trip_id"]=='55700000081974302'].head(10), headers="keys", tablefmt="psql"))

#print("after combine first tim hellp")
#print(tabulate(event_log_departure[event_log_departure["trip_id"]=='55700000081974302'].head(10), headers="keys", tablefmt="psql"))

stop_key_columns = ["trip_id", "stop_id", "stop_sequence"]
vehicle_snapshot_columns = [
    "route_id",
    "direction_id",
    "latitude",
    "longitude",
    "start_date",
    "timestamp",
    "occupancy_status",
]

departure_base = (
    event_log_departure
    .drop(columns=vehicle_snapshot_columns + ["tim_help"], errors="ignore")
    .drop_duplicates(subset=stop_key_columns, keep="first")
)
if OCCUPANCY_MODE == "trip_constant":
    departure_vehicle_snapshot = departure_base[stop_key_columns].merge(
        df_trip_occupancy[["trip_id", "occupancy_status"]],
        on="trip_id",
        how="left",
    )
    for column in vehicle_snapshot_columns:
        if column not in departure_vehicle_snapshot.columns:
            departure_vehicle_snapshot[column] = pd.NA
else:
    departure_vehicle_snapshot = nearest_vehicle_snapshots_by_time(
        departure_base,
        df_vehicle_positions,
        stop_key_columns,
        vehicle_snapshot_columns,
    )

arrival_base = (
    event_log_arrival
    .drop(columns=vehicle_snapshot_columns + ["tim_help"], errors="ignore")
    .drop_duplicates(subset=stop_key_columns, keep="first")
)

combined_stop_base = pd.concat(
    [
        arrival_base.assign(_stop_event_kind="arrival"),
        departure_base.assign(_stop_event_kind="departure"),
    ],
    ignore_index=True,
)
combined_stop_events = combined_stop_base.merge(
    departure_vehicle_snapshot,
    on=stop_key_columns,
    how="left",
)
event_log_arrival = (
    combined_stop_events[combined_stop_events["_stop_event_kind"] == "arrival"]
    .drop(columns="_stop_event_kind")
    .reset_index(drop=True)
)
event_log_departure = (
    combined_stop_events[combined_stop_events["_stop_event_kind"] == "departure"]
    .drop(columns="_stop_event_kind")
    .reset_index(drop=True)
)


event_log_arrival = event_log_arrival.rename(columns= {"timestamp": "timestamp_vehicle_position"})
event_log_arrival = event_log_arrival.rename(columns= {"arrival_time": "timestamp"})
event_log_departure = event_log_departure.rename(columns= {"timestamp": "timestamp_vehicle_position"})
event_log_departure = event_log_departure.rename(columns= {"departure_time": "timestamp"})
#print("second test after df vehicle position")
#print(tabulate(event_log_departure[event_log_departure["trip_id"]=='55700000081974302'].head(10), headers="keys", tablefmt="psql"))

event_log = pd.concat([event_log_arrival, event_log_departure], ignore_index=True)
#print("event_Log")
#print(tabulate(event_log[event_log["trip_id"]=='55700000081974302'].head(10), headers="keys", tablefmt="psql"))

event_log = sort_by_activity_order(event_log, "timestamp", "activity")
event_log = event_log.dropna(subset=["timestamp"])
event_log["occupancy_status"] = event_log.groupby("trip_id")["occupancy_status"].bfill()

#print(tabulate(event_log[(event_log["vehicle_id"].isna()) & (event_log["trip_id"]==55700000082924596)].head(1000), headers="keys", tablefmt="psql"))

df_attributions = pd.read_csv(
    GTFS_INPUT_DIR / "attributions.txt",
    dtype={"trip_id": "string", "organization_name": "string"}
)

organization_ids = (
    df_attributions[["organization_name"]]
    .dropna()
    .drop_duplicates()
    .reset_index(drop=True)
)
organization_ids["organization_id"] = "org_" + (organization_ids.index + 1).astype(str)
df_attributions = df_attributions.merge(organization_ids, on="organization_name", how="left")

aux_event_log = event_log.merge(df_trips, on="trip_id", how="left")
aux_event_log = aux_event_log.rename(columns= {"route_id_y": "route_id"})

aux_event_log = aux_event_log.merge(df_attributions, on="trip_id", how="left")
#print(tabulate(aux_event_log.head(10), headers="keys", tablefmt="psql"))

aux_event_log = aux_event_log.reset_index(drop=True)
aux_event_log["event_id"] = "ad_" + (aux_event_log.index + 1).astype(str)

cols = ["event_id"] + [c for c in aux_event_log.columns if c != "event_id"]
aux_event_log = aux_event_log[cols]


aux_event_log = aux_event_log.merge(df_routes, on="route_id", how="left")


#aux_event_log = aux_event_log.merge(df_shapes, on="shape_id", how="left")
#aux_event_log = aux_event_log.rename(columns={"shape_dist_traveled_x": "shape_dist_traveled_st"})
#aux_event_log = aux_event_log.rename(columns={"shape_dist_traveled_y": "shape_dist_traveled_sh"})
#print(tabulate(aux_event_log.head(100), headers="keys", tablefmt="psql"))

#print(tabulate(aux_event_log[aux_event_log["trip_id"]=='55700200077432135'].head(100), headers="keys", tablefmt="psql"))
aux_event_log["trip_id_org"] = aux_event_log["trip_id"]
aux_event_log["trip_id"] = aux_event_log["trip_id_unique"]
trip_objects_before_split = int(aux_event_log["trip_id"].dropna().nunique())
aux_event_log = split_trip_ids_on_day_sequence_reset(aux_event_log)
trip_objects_after_split = int(aux_event_log["trip_id"].dropna().nunique())
additional_split_parts = trip_objects_after_split - trip_objects_before_split
print(
    "Trip split evaluation:",
    f"trip_objects_before_split={trip_objects_before_split}",
    f"final_trip_objects={trip_objects_after_split}",
    f"additional_split_parts={additional_split_parts}",
    f"final_minus_additional_split_parts={trip_objects_after_split - additional_split_parts}",
)

"""s
for i in range(0,len(aux_event_log)):
    aux_event_log[i, "trip_id"] ==
"""

aux_event_log = aux_event_log.merge(df_agency, on="agency_id", how="left")
#print(tabulate(aux_event_log[aux_event_log["trip_id"]=='55700000078905434'].head(100), headers="keys", tablefmt="psql"))

aux_event_log = aux_event_log.rename(columns= {"direction_id_y": "direction_id"})

aux_event_log["end_time"]=None
aux_event_log["start_time"]=None

aux_event_log["activity_type"] = aux_event_log["activity"].apply(derive_activity_type)
aux_event_log = add_segment_ids(aux_event_log)

segment_overview_kodak = build_segment_overview(aux_event_log)
segment_overview_kodak = attach_segment_distances(
    segment_overview_kodak,
    "segment_distances_kodak_2026_05_04.csv",
)
segment_overview_kodak.to_csv("segment_overview_kodak_2026_05_04.csv", index=False)

aux_event_log = aux_event_log.merge(
    segment_overview_kodak[
        [
            "segment_id",
            "air_distance_m",
            "road_distance_m",
            "road_distance_status",
        ]
    ],
    on="segment_id",
    how="left",
)
aux_event_log = add_cumulative_road_distance(aux_event_log)

#print("neuer aux_event_log")
#print(tabulate(aux_event_log[aux_event_log["trip_id"]=='tr03_9797'].head(100), headers="keys", tablefmt="psql"))
#print(tabulate(aux_event_log[aux_event_log["trip_id"]=='tr03_9797'].head(100), headers="keys", tablefmt="psql"))

aux_event_log.to_csv("aux_event_log_kodak_2026_05_04_no_layover.csv", index=False)



#print("aux_event_log")
#print(tabulate(aux_event_log[aux_event_log["trip_id"]=='tr03_4539'].head(50), headers="keys", tablefmt="psql"))

df = aux_event_log
#[
    ##(aux_event_log["route_short_name"] == 10) |
    #(
        #(aux_event_log["vehicle_id"] == 'veh_35') &
        #(aux_event_log["activity"] == 'arrive_stop')
    #)
#]

cols = [
    "event_id", "trip_id","trip_id_org", "activity", "timestamp","route_short_name", "stop_id", "stop_sequence","vehicle_id", "route_type","direction_id", "organization_name", "organization_id",
    "stop_name", "stop_lat", "stop_lon", "start_time", "end_time", "stop_headsign", "pickup_type",

    "route_id", "service_id", "agency_id",
  #  "shape_dist_traveled_st",  "shape_dist_traveled_sh",
    "shape_dist_traveled",
    "platform_code", "delay", "occupancy_status",
    "shape_id"]






df = df[cols]

#print("df_cols")
#print(tabulate(df[df["vehicle_id"]=='9031005920505750'].head(200), headers="keys", tablefmt="psql"))


#per trip we have the lowest and the hightes timestapm per trip
# Indizes für min/max timestamp pro trip
idx_min = df.groupby("trip_id")["timestamp"].idxmin()
idx_max = df.groupby("trip_id")["timestamp"].idxmax()

# Zeilen holen
df_result = pd.concat([df.loc[idx_min], df.loc[idx_max]]) \
   .sort_values(["trip_id", "timestamp"])

#print(tabulate(df_result[df_result["vehicle_id"]=='veh_43'].head(50), headers="keys", tablefmt="psql"))
#print(tabulate(df_result[(df_result["vehicle_id"].isna()) & (df_result["trip_id"]==55700000082795840)].sort_values("timestamp").head(50), headers="keys", tablefmt="psql"))
#print(tabulate(df_result[(df_result["trip_id"]==55700000082795840)].sort_values("timestamp").head(50), headers="keys", tablefmt="psql"))
#print(tabulate(df_result[(df_result["vehicle_id"].isna()) ].sort_values("timestamp").head(1000), headers="keys", tablefmt="psql"))





a_event_rows = []

df_base = df_result.copy()
df_base["timestamp_dt"] = pd.to_datetime(df_base["timestamp"])
df_base["_vehicle_route_fallback_group"] = df_base["vehicle_id"].where(
    df_base["vehicle_id"].notna(),
    "missing_vehicle_route_" + df_base["route_short_name"].astype("string").fillna("unknown"),
)

first_stop_events = df_base.copy()
first_stop_events["_stop_sequence_sort"] = pd.to_numeric(
    first_stop_events["stop_sequence"], errors="coerce"
)
first_stop_events = first_stop_events.sort_values(
    ["trip_id", "timestamp_dt", "_stop_sequence_sort"]
)
first_stop_by_trip = first_stop_events.groupby("trip_id", as_index=False).first().set_index("trip_id")

for (_, route_short_name), temp_table in df_base.groupby(
    ["_vehicle_route_fallback_group", "route_short_name"],
    dropna=False,
):
    temp_table = temp_table.sort_values("timestamp_dt").reset_index(drop=True)

    for i in range(len(temp_table) - 1):
        current_row = temp_table.iloc[i]
        next_row = temp_table.iloc[i + 1]

        if current_row["trip_id"] == next_row["trip_id"]:
            continue

        current_date = current_row["timestamp_dt"]
        next_date = next_row["timestamp_dt"]





        gap = next_date - current_date

        #können hier gerade noch ohne current date vergleich arbeiten weil wir alle am 2026-05-04 arbeiten
        #if current_date == next_date and pd.Timedelta(0) <= gap < pd.Timedelta(minutes=30):
        if pd.Timedelta(0) <= gap < pd.Timedelta(hours=1):

            if next_row["trip_id"] in first_stop_by_trip.index:
                direction_change_row = first_stop_by_trip.loc[next_row["trip_id"]]
            else:
                direction_change_row = next_row

            a_event_rows.append({
                "old_trip_id": current_row["trip_id"],
                "trip_id": next_row["trip_id"],
                "activity": "direction_change",
                "timestamp": direction_change_row["timestamp"],
                "stop_id": direction_change_row["stop_id"],
                "stop_sequence": direction_change_row["stop_sequence"],
                "stop_name": direction_change_row["stop_name"],
                "direction_id": next_row["direction_id"],
                "vehicle_id": current_row["vehicle_id"],
                "route_type": current_row["route_type"],
                "route_short_name": route_short_name,
                "route_id": next_row["route_id"],
                "service_id": next_row["service_id"],
                "agency_id": next_row["agency_id"],
                "organization_id": current_row["organization_id"],
                "operator": current_row["organization_name"]
            })
        else:

            a_event_rows.append({
                "old_trip_id": current_row["trip_id"],
                "trip_id": None,
                "activity": "parking",
                "timestamp": current_date,
                "stop_id": None,
                "stop_sequence": None,
                "stop_name": None,
                "direction_id": current_row["direction_id"],
                "vehicle_id": current_row["vehicle_id"],
                "route_type": current_row["route_type"],
                "route_short_name": current_row["route_short_name"],
                "route_id": current_row["route_id"],
                "service_id": current_row["service_id"],
                "agency_id": current_row["agency_id"],
                "organization_id": current_row["organization_id"],
                "operator": current_row["organization_name"]
            })

a_events = pd.DataFrame(a_event_rows)

if not a_events.empty:
    a_events = a_events.reset_index(drop=True)
    a_events["event_id"] = "cdp_" + (a_events.index + 1).astype(str)

    a_events = a_events[[
        "event_id",
        "trip_id",
        "old_trip_id",
        "activity",
        "timestamp",
        "stop_id",
        "stop_sequence",
        "stop_name",
        "direction_id",
        "vehicle_id",
        "route_type",
        "route_short_name",
        "route_id",
        "service_id",
        "agency_id",
        "organization_id",
        "operator"

    ]]
else:
    a_events = pd.DataFrame(columns=[
        "event_id",
        "trip_id",
        "old_trip_id",
        "activity",
        "timestamp",
        "stop_id",
        "stop_sequence",
        "stop_name",
        "direction_id",
        "vehicle_id",
        "route_type",
        "route_short_name",
        "route_id",
        "service_id",
        "agency_id",
        "organization_id",
        "operator"
    ])


#print(tabulate(a_events.head(10), headers="keys", tablefmt="psql"))

"""
begin shift activity 


"""
begin_shift_rows = []

df_begin_base = df_result.copy()
df_begin_base["timestamp_dt"] = pd.to_datetime(df_begin_base["timestamp"])
df_begin_base["_vehicle_fallback_group"] = df_begin_base["vehicle_id"].where(
    df_begin_base["vehicle_id"].notna(),
    "missing_vehicle_route_" + df_begin_base["route_short_name"].astype("string").fillna("unknown"),
)

# Fuer begin_shift betrachten wir den Fahrzeugumlauf; fehlt die vehicle_id,
# nutzen wir route_short_name als Fallback-Gruppe.
df_begin_base = df_begin_base.sort_values(["_vehicle_fallback_group", "timestamp_dt"]).reset_index(drop=True)
df_begin_base["delay_td"] = pd.to_timedelta(df_begin_base["delay"], unit="s")
for _, temp_table in df_begin_base.groupby("_vehicle_fallback_group", dropna=False):
    temp_table = temp_table.sort_values("timestamp_dt").reset_index(drop=True)

    for i in range(len(temp_table)):
        current_row = temp_table.iloc[i]
        current_date_compare = current_row["timestamp_dt"].date()
        current_date = current_row["timestamp_dt"]
        current_delay = current_row["delay"]
        current_delay_td = pd.to_timedelta(
            current_delay if pd.notna(current_delay) else 0,
            unit="s"
        )

        # Fall 1: kein Vorgaenger fuer dieses Fahrzeug
        if i == 0:
            begin_shift_rows.append({
                "trip_id": current_row["trip_id"],
                "activity": "begin_shift",
                "timestamp": current_date,
                "stop_id": current_row["stop_id"],
                "stop_sequence": current_row["stop_sequence"],
                "stop_name": current_row["stop_name"],
                "stop_lat": current_row["stop_lat"],
                "stop_lon": current_row["stop_lon"],
                "direction_id": current_row["direction_id"],
                "vehicle_id": current_row["vehicle_id"],
                "route_type": current_row["route_type"],
                "route_short_name": current_row["route_short_name"],
                "route_id": current_row["route_id"],
                "service_id": current_row["service_id"],
                "agency_id": current_row["agency_id"],
                "organization_id": current_row["organization_id"],
                "operator": current_row["organization_name"]
            })
            continue

        # Fall 2: Vorgaenger existiert, aber Luecke > 1 Stunde
        prev_row = temp_table.iloc[i - 1]
        prev_date_compare = prev_row["timestamp_dt"].date()
        prev_date = prev_row["timestamp_dt"]



        gap = current_date - prev_date

        if current_row["trip_id"] != prev_row["trip_id"] and gap > pd.Timedelta(hours=1):
            begin_shift_rows.append({
                "trip_id": current_row["trip_id"],
                "activity": "begin_shift",
                "timestamp": current_date,
                "stop_id": current_row["stop_id"],
                "stop_sequence": current_row["stop_sequence"],
                "stop_name": current_row["stop_name"],
                "stop_lat": current_row["stop_lat"],
                "stop_lon": current_row["stop_lon"],
                "direction_id": current_row["direction_id"],
                "vehicle_id": current_row["vehicle_id"],
                "route_type": current_row["route_type"],
                "route_short_name": current_row["route_short_name"],
                "route_id": current_row["route_id"],
                "service_id": current_row["service_id"],
                "agency_id": current_row["agency_id"],
                "organization_id": current_row["organization_id"],
                "operator": current_row["organization_name"]
            })

begin_shift_events = pd.DataFrame(begin_shift_rows)

if not begin_shift_events.empty:
    begin_shift_events = begin_shift_events.reset_index(drop=True)
    begin_shift_events["event_id"] = "bs_" + (begin_shift_events.index + 1).astype(str)

    begin_shift_events = begin_shift_events[[
        "event_id",
        "trip_id",
        "activity",
        "timestamp",
        "stop_id",
        "stop_sequence",
        "stop_name",
        "stop_lat",
        "stop_lon",
        "direction_id",
        "vehicle_id",
        "route_type",
        "route_short_name",
        "route_id",
        "service_id",
        "agency_id",
        "organization_id",
        "operator"
    ]]
else:
    begin_shift_events = pd.DataFrame(columns=[
        "event_id",
        "trip_id",
        "activity",
        "timestamp",
        "stop_id",
        "stop_sequence",
        "stop_name",
        "stop_lat",
        "stop_lon",
        "direction_id",
        "vehicle_id",
        "route_type",
        "route_short_name",
        "route_id",
        "service_id",
        "agency_id",
        "organization_id",
        "operator"
    ])


def normalize_trip_transition_boundaries(transition_events, begin_events, event_table):
    transition_events = transition_events.copy()
    begin_events = begin_events.copy()
    event_table = event_table.copy()

    transition_columns = transition_events.columns.tolist()
    begin_columns = begin_events.columns.tolist()
    event_table["timestamp_dt"] = pd.to_datetime(event_table["timestamp"], errors="coerce")
    event_table = event_table.dropna(subset=["trip_id", "timestamp_dt"])

    first_events = (
        event_table.sort_values(["trip_id", "timestamp_dt", "event_id"])
        .groupby("trip_id", as_index=False)
        .first()
    )
    last_events = (
        event_table.sort_values(["trip_id", "timestamp_dt", "event_id"])
        .groupby("trip_id", as_index=False)
        .last()
    )

    if not transition_events.empty:
        transition_events["_timestamp_sort"] = pd.to_datetime(
            transition_events["timestamp"], errors="coerce"
        )
        transition_events["_activity_priority"] = transition_events["activity"].map(
            {"direction_change": 0, "parking": 1}
        ).fillna(9)

        transition_events = transition_events.drop_duplicates(
            subset=[
                "activity",
                "trip_id",
                "old_trip_id",
                "timestamp",
                "vehicle_id",
                "route_short_name",
            ],
            keep="first",
        ).sort_values(
            ["_timestamp_sort", "_activity_priority", "event_id"]
        ).reset_index(drop=True)

        used_start_trips = set()
        used_end_trips = set()
        keep_indexes = []

        for row in transition_events.itertuples():
            start_trip = (
                str(row.trip_id)
                if row.activity == "direction_change" and pd.notna(row.trip_id)
                else None
            )
            end_trip = str(row.old_trip_id) if pd.notna(row.old_trip_id) else None

            if start_trip is not None and start_trip in used_start_trips:
                continue
            if end_trip is not None and end_trip in used_end_trips:
                continue

            keep_indexes.append(row.Index)
            if start_trip is not None:
                used_start_trips.add(start_trip)
            if end_trip is not None:
                used_end_trips.add(end_trip)

        transition_events = transition_events.loc[keep_indexes].copy()
        transition_events = transition_events.sort_values(
            ["_timestamp_sort", "_activity_priority", "event_id"]
        ).reset_index(drop=True)
        transition_events["event_id"] = "cdp_" + (transition_events.index + 1).astype(str)
        transition_events = transition_events[transition_columns]
    else:
        used_end_trips = set()
        used_start_trips = set()

    if not begin_events.empty:
        begin_events["_timestamp_sort"] = pd.to_datetime(
            begin_events["timestamp"], errors="coerce"
        )
        begin_events = begin_events.drop_duplicates(
            subset=["trip_id", "timestamp", "vehicle_id", "route_short_name"],
            keep="first",
        )
        begin_events = begin_events[
            ~begin_events["trip_id"].astype(str).isin(used_start_trips)
        ].copy()
        begin_events = begin_events.sort_values(
            ["trip_id", "_timestamp_sort", "event_id"]
        ).drop_duplicates(subset=["trip_id"], keep="first")
        begin_events = begin_events.sort_values(
            ["_timestamp_sort", "event_id"]
        ).reset_index(drop=True)
        begin_events["event_id"] = "bs_" + (begin_events.index + 1).astype(str)
        begin_events = begin_events[begin_columns]

    used_start_trips = set(
        transition_events.loc[
            transition_events["activity"].eq("direction_change")
            & transition_events["trip_id"].notna(),
            "trip_id",
        ].astype(str)
    )
    used_start_trips.update(
        begin_events.loc[begin_events["trip_id"].notna(), "trip_id"].astype(str)
    )

    missing_start_events = first_events[
        ~first_events["trip_id"].astype(str).isin(used_start_trips)
    ].copy()
    if not missing_start_events.empty:
        fallback_begin_events = pd.DataFrame({
            "trip_id": missing_start_events["trip_id"],
            "activity": "begin_shift",
            "timestamp": missing_start_events["timestamp_dt"],
            "stop_id": missing_start_events["stop_id"],
            "stop_sequence": missing_start_events["stop_sequence"],
            "stop_name": missing_start_events["stop_name"],
            "stop_lat": missing_start_events["stop_lat"],
            "stop_lon": missing_start_events["stop_lon"],
            "direction_id": missing_start_events["direction_id"],
            "vehicle_id": missing_start_events["vehicle_id"],
            "route_type": missing_start_events["route_type"],
            "route_short_name": missing_start_events["route_short_name"],
            "route_id": missing_start_events["route_id"],
            "service_id": missing_start_events["service_id"],
            "agency_id": missing_start_events["agency_id"],
            "organization_id": missing_start_events["organization_id"],
            "operator": missing_start_events["organization_name"],
        })
        fallback_begin_events["event_id"] = pd.NA
        begin_events = pd.concat(
            [begin_events, fallback_begin_events[begin_columns]],
            ignore_index=True,
        )

    used_end_trips = set(
        transition_events.loc[
            transition_events["old_trip_id"].notna(), "old_trip_id"
        ].astype(str)
    )

    missing_end_events = last_events[
        ~last_events["trip_id"].astype(str).isin(used_end_trips)
    ].copy()
    if not missing_end_events.empty:
        fallback_parking_events = pd.DataFrame({
            "trip_id": pd.NA,
            "old_trip_id": missing_end_events["trip_id"],
            "activity": "parking",
            "timestamp": missing_end_events["timestamp_dt"] + pd.Timedelta(microseconds=1),
            "stop_id": missing_end_events["stop_id"],
            "stop_sequence": missing_end_events["stop_sequence"],
            "stop_name": missing_end_events["stop_name"],
            "direction_id": missing_end_events["direction_id"],
            "vehicle_id": missing_end_events["vehicle_id"],
            "route_type": missing_end_events["route_type"],
            "route_short_name": missing_end_events["route_short_name"],
            "route_id": missing_end_events["route_id"],
            "service_id": missing_end_events["service_id"],
            "agency_id": missing_end_events["agency_id"],
            "organization_id": missing_end_events["organization_id"],
            "operator": missing_end_events["organization_name"],
        })
        fallback_parking_events["event_id"] = pd.NA
        transition_events = pd.concat(
            [transition_events, fallback_parking_events[transition_columns]],
            ignore_index=True,
        )

    if not transition_events.empty:
        transition_events["_timestamp_sort"] = pd.to_datetime(
            transition_events["timestamp"], errors="coerce"
        )
        transition_events["_activity_priority"] = transition_events["activity"].map(
            {"direction_change": 0, "parking": 1}
        ).fillna(9)
        transition_events = transition_events.sort_values(
            ["_timestamp_sort", "_activity_priority", "activity"]
        ).reset_index(drop=True)
        transition_events["event_id"] = "cdp_" + (transition_events.index + 1).astype(str)
        transition_events = transition_events[transition_columns]

    if not begin_events.empty:
        begin_events["_timestamp_sort"] = pd.to_datetime(
            begin_events["timestamp"], errors="coerce"
        )
        begin_events = begin_events.sort_values(
            ["_timestamp_sort", "activity"]
        ).reset_index(drop=True)
        begin_events["event_id"] = "bs_" + (begin_events.index + 1).astype(str)
        begin_events = begin_events[begin_columns]

    return transition_events, begin_events


a_events, begin_shift_events = normalize_trip_transition_boundaries(
    a_events,
    begin_shift_events,
    df_result,
)


def build_complete_trip_boundary_events(event_table, max_transition_gap=pd.Timedelta(hours=1)):
    base = event_table.copy()
    base["timestamp_dt"] = pd.to_datetime(base["timestamp"], errors="coerce")
    base = base.dropna(subset=["trip_id", "timestamp_dt"])
    base["_boundary_vehicle_id"] = base["vehicle_id"].where(
        base["vehicle_id"].notna(),
        "missing_vehicle_" + base["trip_id"].astype(str),
    )

    idx_first = base.groupby("trip_id")["timestamp_dt"].idxmin()
    idx_last = base.groupby("trip_id")["timestamp_dt"].idxmax()
    first_by_trip = base.loc[idx_first].copy()
    last_by_trip = base.loc[idx_last].copy()

    trip_bounds = first_by_trip.merge(
        last_by_trip[[
            "trip_id",
            "timestamp_dt",
            "timestamp",
            "stop_id",
            "stop_sequence",
            "stop_name",
            "direction_id",
            "vehicle_id",
            "_boundary_vehicle_id",
            "route_type",
            "route_short_name",
            "route_id",
            "service_id",
            "agency_id",
            "organization_id",
            "organization_name",
        ]],
        on="trip_id",
        how="inner",
        suffixes=("_first", "_last"),
    )
    trip_bounds = trip_bounds.sort_values(
        ["_boundary_vehicle_id_first", "timestamp_dt_first", "timestamp_dt_last", "trip_id"]
    ).reset_index(drop=True)

    begin_rows = []
    transition_rows = []

    for _, vehicle_trips in trip_bounds.groupby("_boundary_vehicle_id_first", sort=False):
        vehicle_trips = vehicle_trips.sort_values(
            ["timestamp_dt_first", "timestamp_dt_last", "trip_id"]
        ).reset_index(drop=True)

        for i, current_row in vehicle_trips.iterrows():
            previous_row = vehicle_trips.iloc[i - 1] if i > 0 else None
            next_row = vehicle_trips.iloc[i + 1] if i < len(vehicle_trips) - 1 else None

            if previous_row is None:
                start_gap = pd.NaT
            else:
                start_gap = current_row["timestamp_dt_first"] - previous_row["timestamp_dt_last"]

            if previous_row is None or pd.isna(start_gap) or start_gap > max_transition_gap:
                begin_rows.append({
                    "trip_id": current_row["trip_id"],
                    "activity": "begin_shift",
                    "timestamp": current_row["timestamp_dt_first"],
                    "stop_id": current_row["stop_id_first"],
                    "stop_sequence": current_row["stop_sequence_first"],
                    "stop_name": current_row["stop_name_first"],
                    "stop_lat": current_row["stop_lat"],
                    "stop_lon": current_row["stop_lon"],
                    "direction_id": current_row["direction_id_first"],
                    "vehicle_id": current_row["vehicle_id_first"],
                    "route_type": current_row["route_type_first"],
                    "route_short_name": current_row["route_short_name_first"],
                    "route_id": current_row["route_id_first"],
                    "service_id": current_row["service_id_first"],
                    "agency_id": current_row["agency_id_first"],
                    "organization_id": current_row["organization_id_first"],
                    "operator": current_row["organization_name_first"],
                })
            else:
                transition_rows.append({
                    "old_trip_id": pd.NA,
                    "trip_id": current_row["trip_id"],
                    "activity": "direction_change",
                    "timestamp": current_row["timestamp_dt_first"],
                    "stop_id": current_row["stop_id_first"],
                    "stop_sequence": current_row["stop_sequence_first"],
                    "stop_name": current_row["stop_name_first"],
                    "direction_id": current_row["direction_id_first"],
                    "vehicle_id": current_row["vehicle_id_first"],
                    "route_type": current_row["route_type_first"],
                    "route_short_name": current_row["route_short_name_first"],
                    "route_id": current_row["route_id_first"],
                    "service_id": current_row["service_id_first"],
                    "agency_id": current_row["agency_id_first"],
                    "organization_id": current_row["organization_id_first"],
                    "operator": current_row["organization_name_first"],
                })

            if next_row is None:
                end_gap = pd.NaT
            else:
                end_gap = next_row["timestamp_dt_first"] - current_row["timestamp_dt_last"]

            if next_row is not None and pd.notna(end_gap) and end_gap <= max_transition_gap:
                transition_rows.append({
                    "old_trip_id": pd.NA,
                    "trip_id": current_row["trip_id"],
                    "activity": "direction_change",
                    "timestamp": current_row["timestamp_dt_last"] + pd.Timedelta(microseconds=1),
                    "stop_id": current_row["stop_id_last"],
                    "stop_sequence": current_row["stop_sequence_last"],
                    "stop_name": current_row["stop_name_last"],
                    "direction_id": current_row["direction_id_last"],
                    "vehicle_id": current_row["vehicle_id_first"],
                    "route_type": current_row["route_type_last"],
                    "route_short_name": current_row["route_short_name_last"],
                    "route_id": current_row["route_id_last"],
                    "service_id": current_row["service_id_last"],
                    "agency_id": current_row["agency_id_last"],
                    "organization_id": current_row["organization_id_last"],
                    "operator": current_row["organization_name_last"],
                })
            else:
                transition_rows.append({
                    "old_trip_id": pd.NA,
                    "trip_id": current_row["trip_id"],
                    "activity": "parking",
                    "timestamp": current_row["timestamp_dt_last"] + pd.Timedelta(microseconds=1),
                    "stop_id": current_row["stop_id_last"],
                    "stop_sequence": current_row["stop_sequence_last"],
                    "stop_name": current_row["stop_name_last"],
                    "direction_id": current_row["direction_id_last"],
                    "vehicle_id": current_row["vehicle_id_first"],
                    "route_type": current_row["route_type_last"],
                    "route_short_name": current_row["route_short_name_last"],
                    "route_id": current_row["route_id_last"],
                    "service_id": current_row["service_id_last"],
                    "agency_id": current_row["agency_id_last"],
                    "organization_id": current_row["organization_id_last"],
                    "operator": current_row["organization_name_last"],
                })

    begin_columns = [
        "event_id",
        "trip_id",
        "activity",
        "timestamp",
        "stop_id",
        "stop_sequence",
        "stop_name",
        "stop_lat",
        "stop_lon",
        "direction_id",
        "vehicle_id",
        "route_type",
        "route_short_name",
        "route_id",
        "service_id",
        "agency_id",
        "organization_id",
        "operator",
    ]
    transition_columns = [
        "event_id",
        "trip_id",
        "old_trip_id",
        "activity",
        "timestamp",
        "stop_id",
        "stop_sequence",
        "stop_name",
        "direction_id",
        "vehicle_id",
        "route_type",
        "route_short_name",
        "route_id",
        "service_id",
        "agency_id",
        "organization_id",
        "operator",
    ]

    complete_begin_shift_events = pd.DataFrame(begin_rows)
    if complete_begin_shift_events.empty:
        complete_begin_shift_events = pd.DataFrame(columns=begin_columns)
    else:
        complete_begin_shift_events = complete_begin_shift_events.reset_index(drop=True)
        complete_begin_shift_events["event_id"] = (
            "bs_" + (complete_begin_shift_events.index + 1).astype(str)
        )
        complete_begin_shift_events = complete_begin_shift_events[begin_columns]

    complete_transition_events = pd.DataFrame(transition_rows)
    if complete_transition_events.empty:
        complete_transition_events = pd.DataFrame(columns=transition_columns)
    else:
        complete_transition_events = complete_transition_events.reset_index(drop=True)
        complete_transition_events["event_id"] = (
            "bd_" + (complete_transition_events.index + 1).astype(str)
        )
        complete_transition_events = complete_transition_events[transition_columns]

    return complete_transition_events, complete_begin_shift_events


#print("grouped trip transition boundary events")
#print("direction_change:", (a_events["activity"] == "direction_change").sum())
#print("parking:", (a_events["activity"] == "parking").sum())
#print("begin_shift:", len(begin_shift_events))

aux_e2o_begin_shift_trip = begin_shift_events[["event_id", "activity", "timestamp"]].copy()
aux_e2o_begin_shift_trip["oid"] = begin_shift_events["trip_id"]
aux_e2o_begin_shift_trip["type"] = "trip"
aux_e2o_begin_shift_trip["qualifier"] = "conduct trip"

aux_e2o_begin_shift_route = begin_shift_events[["event_id", "activity", "timestamp"]].copy()
aux_e2o_begin_shift_route["oid"] = begin_shift_events["route_id"]
aux_e2o_begin_shift_route["type"] = "route"
aux_e2o_begin_shift_route["qualifier"] = "conduct route"

aux_e2o_begin_shift_service = begin_shift_events[["event_id", "activity", "timestamp"]].copy()
aux_e2o_begin_shift_service["oid"] = begin_shift_events["service_id"]
aux_e2o_begin_shift_service["type"] = "service"
aux_e2o_begin_shift_service["qualifier"] = "conduct service"

aux_e2o_begin_shift_stop = begin_shift_events[["event_id", "activity", "timestamp"]].copy()
aux_e2o_begin_shift_stop["oid"] = begin_shift_events["stop_id"]
aux_e2o_begin_shift_stop["type"] = "stop"
aux_e2o_begin_shift_stop["qualifier"] = "used bus stop"

aux_e2o_begin_shift_agency = begin_shift_events[["event_id", "activity", "timestamp"]].copy()
aux_e2o_begin_shift_agency["oid"] = begin_shift_events["agency_id"]
aux_e2o_begin_shift_agency["type"] = "agency"
aux_e2o_begin_shift_agency["qualifier"] = "used transport agency"

aux_e2o_begin_shift_vehicle = begin_shift_events[["event_id", "activity", "timestamp"]].copy()
aux_e2o_begin_shift_vehicle["oid"] = begin_shift_events["vehicle_id"]
aux_e2o_begin_shift_vehicle["type"] = "vehicle"
aux_e2o_begin_shift_vehicle["qualifier"] = "used vehicle"

aux_e2o_begin_shift_operator = begin_shift_events[["event_id", "activity", "timestamp"]].copy()
aux_e2o_begin_shift_operator["oid"] = begin_shift_events["organization_id"]
aux_e2o_begin_shift_operator["type"] = "operator"
aux_e2o_begin_shift_operator["qualifier"] = "used operator"

e2o_begin_shift_table = pd.concat(
    [
        aux_e2o_begin_shift_trip,
        aux_e2o_begin_shift_route,
        aux_e2o_begin_shift_service,
        aux_e2o_begin_shift_stop,
        aux_e2o_begin_shift_agency,
        aux_e2o_begin_shift_vehicle,
        aux_e2o_begin_shift_operator
    ],
    ignore_index=True
).sort_values("timestamp").reset_index(drop=True)
"""

begin layer activity



aux_arrive = aux_event_log[
    aux_event_log["activity"].str.startswith("arrive_stop", na=False)
].copy()
aux_arrive["timestamp_dt"] = pd.to_datetime(aux_arrive["timestamp"])

idx_last_departure = aux_arrive.groupby("trip_id")["timestamp_dt"].idxmax()  ### HIER FEHLER!!!! TRIP_ID NICHT EINDEUTIG - problem gelöst

begin_layover_events = aux_arrive.loc[idx_last_departure].copy()
begin_layover_events["activity"] = "begin_layover"
begin_layover_events = begin_layover_events.drop(columns=["timestamp_dt"])

begin_layover_events = begin_layover_events.reset_index(drop=True)
begin_layover_events["event_id"] = "bl_" + (begin_layover_events.index + 1).astype(str)

#print(tabulate( begin_layover_events.head(10),headers="keys",tablefmt="psql",showindex=False))

aux_e2o_layover_trip = begin_layover_events[["event_id", "activity", "timestamp"]].copy()
aux_e2o_layover_trip["oid"] = begin_layover_events["trip_id"]
aux_e2o_layover_trip["type"] = "trip"
aux_e2o_layover_trip["qualifier"] = "conduct trip"

aux_e2o_layover_route = begin_layover_events[["event_id", "activity", "timestamp"]].copy()
aux_e2o_layover_route["oid"] = begin_layover_events["route_id"]
aux_e2o_layover_route["type"] = "route"
aux_e2o_layover_route["qualifier"] = "conduct route"

aux_e2o_layover_service = begin_layover_events[["event_id", "activity", "timestamp"]].copy()
aux_e2o_layover_service["oid"] = begin_layover_events["service_id"]
aux_e2o_layover_service["type"] = "service"
aux_e2o_layover_service["qualifier"] = "conduct service"

aux_e2o_layover_stop = begin_layover_events[["event_id", "activity", "timestamp"]].copy()
aux_e2o_layover_stop["oid"] = begin_layover_events["stop_id"]
aux_e2o_layover_stop["type"] = "stop"
aux_e2o_layover_stop["qualifier"] = "used bus stop"

aux_e2o_layover_agency = begin_layover_events[["event_id", "activity", "timestamp"]].copy()
aux_e2o_layover_agency["oid"] = begin_layover_events["agency_id"]
aux_e2o_layover_agency["type"] = "agency"
aux_e2o_layover_agency["qualifier"] = "used transport agency"

aux_e2o_layover_vehicle = begin_layover_events[["event_id", "activity", "timestamp"]].copy()
aux_e2o_layover_vehicle["oid"] = begin_layover_events["vehicle_id"]
aux_e2o_layover_vehicle["type"] = "vehicle"
aux_e2o_layover_vehicle["qualifier"] = "used vehicle"

aux_e2o_layover_operator = begin_layover_events[["event_id", "activity", "timestamp"]].copy()
aux_e2o_layover_operator["oid"] = begin_layover_events["organization_id"]
aux_e2o_layover_operator["type"] = "operator"
aux_e2o_layover_operator["qualifier"] = "used operator"

e2o_layover_table = pd.concat(
    [
        aux_e2o_layover_trip,
      #  aux_e2o_layover_route,
       # aux_e2o_layover_service,
     #   aux_e2o_layover_stop,
    #    aux_e2o_layover_agency,
        aux_e2o_layover_vehicle,
        #aux_e2o_layover_operator
    ],
    ignore_index=True
)

e2o_layover_table = e2o_layover_table.sort_values("timestamp").reset_index(drop=True)



end layover activity



aux_arrive = aux_event_log[
    aux_event_log["activity"].str.startswith("arrive_stop", na=False)
].copy()
aux_arrive["timestamp_dt"] = pd.to_datetime(aux_arrive["timestamp"])

idx_first_arrive = aux_arrive.groupby("trip_id")["timestamp_dt"].idxmin()

end_layover_events = aux_arrive.loc[idx_first_arrive].copy()
end_layover_events["activity"] = "end_layover"
end_layover_events = end_layover_events.drop(columns=["timestamp_dt"])

end_layover_events = end_layover_events.reset_index(drop=True)
end_layover_events["event_id"] = "el_" + (end_layover_events.index + 1).astype(str)

#print(tabulate(end_layover_events.head(100),headers="keys",tablefmt="psql",showindex=False))

aux_e2o_end_layover_trip = end_layover_events[["event_id", "activity", "timestamp"]].copy()
aux_e2o_end_layover_trip["oid"] = end_layover_events["trip_id"]
aux_e2o_end_layover_trip["type"] = "trip"
aux_e2o_end_layover_trip["qualifier"] = "conduct trip"

aux_e2o_end_layover_route = end_layover_events[["event_id", "activity", "timestamp"]].copy()
aux_e2o_end_layover_route["oid"] = end_layover_events["route_id"]
aux_e2o_end_layover_route["type"] = "route"
aux_e2o_end_layover_route["qualifier"] = "conduct route"

aux_e2o_end_layover_service = end_layover_events[["event_id", "activity", "timestamp"]].copy()
aux_e2o_end_layover_service["oid"] = end_layover_events["service_id"]
aux_e2o_end_layover_service["type"] = "service"
aux_e2o_end_layover_service["qualifier"] = "conduct service"

aux_e2o_end_layover_stop = end_layover_events[["event_id", "activity", "timestamp"]].copy()
aux_e2o_end_layover_stop["oid"] = end_layover_events["stop_id"]
aux_e2o_end_layover_stop["type"] = "stop"
aux_e2o_end_layover_stop["qualifier"] = "used bus stop"

aux_e2o_end_layover_agency = end_layover_events[["event_id", "activity", "timestamp"]].copy()
aux_e2o_end_layover_agency["oid"] = end_layover_events["agency_id"]
aux_e2o_end_layover_agency["type"] = "agency"
aux_e2o_end_layover_agency["qualifier"] = "used transport agency"


aux_e2o_end_layover_vehicle = end_layover_events[["event_id", "activity", "timestamp"]].copy()
aux_e2o_end_layover_vehicle["oid"] = end_layover_events["vehicle_id"]
aux_e2o_end_layover_vehicle["type"] = "vehicle"
aux_e2o_end_layover_vehicle["qualifier"] = "used vehicle"

aux_e2o_end_layover_operator = end_layover_events[["event_id", "activity", "timestamp"]].copy()
aux_e2o_end_layover_operator["oid"] = end_layover_events["organization_id"]
aux_e2o_end_layover_operator["type"] = "operator"
aux_e2o_end_layover_operator["qualifier"] = "used operator"

e2o_end_layover_table = pd.concat(
    [
        aux_e2o_end_layover_trip,
        #aux_e2o_end_layover_route,
        #aux_e2o_end_layover_service,
        #aux_e2o_end_layover_stop,
        #aux_e2o_end_layover_agency,
        aux_e2o_end_layover_vehicle,
        #aux_e2o_end_layover_operator
    ],
    ignore_index=True
).sort_values("timestamp").reset_index(drop=True)

events_end_layover = pd.DataFrame([
    {
        "id": e.event_id,
        "type": e.activity,
        "time": e.timestamp,
        "attributes": [
            {"name": "latitude", "value": e.stop_lat},
            {"name": "longitude", "value": e.stop_lon},
            {"name": "stop_sequence", "value": e.stop_sequence},
            {"name": "route_short_name", "value": e.route_short_name},
        ],
        "relationships": [
            {"objectId": e.trip_id, "qualifier": "conduct trip"},
            {"objectId": e.stop_id, "qualifier": "used bus stop"},
            {"objectId": e.service_id, "qualifier": "conduct service"},
            {"objectId": e.agency_id, "qualifier": "used transport agency"},
            {"objectId": e.route_id, "qualifier": "conduct route"}  ,
                            {
                                 "objectId": e.vehicle_id,
                                 "qualifier": "used vehicle"
                             }
        ],
    }
    for e in end_layover_events.itertuples(index=False)
])




"""
#Event-to-Object Table Creation

#!!!Achtung!!! hier ist mal nur die Objekte genommen die wir als Tabelle haben


#von aux_event_log aus arrive_stop/departure_stop



aux_e2o_trip = aux_event_log[["event_id", "activity", "timestamp"] ]

aux_e2o_trip["oid"] = aux_event_log["trip_id"]

aux_e2o_trip["type"] = "trip"

aux_e2o_trip["qualifier"] = "conduct trip"

aux_e2o_route = aux_event_log[["event_id", "activity", "timestamp"] ]

aux_e2o_route["oid"] = aux_event_log["route_id"]

aux_e2o_route["type"] = "route"

aux_e2o_route["qualifier"] = "conduct route"

aux_e2o_service = aux_event_log[["event_id", "activity", "timestamp"] ]

aux_e2o_service["oid"] = aux_event_log["service_id"]

aux_e2o_service["type"] = "service"

aux_e2o_service["qualifier"] = "conduct service"

aux_e2o_stops = aux_event_log[["event_id", "activity", "timestamp"] ]

aux_e2o_stops["oid"] = aux_event_log["stop_id"]

aux_e2o_stops["type"] = "stop"

aux_e2o_stops["qualifier"] = "used bus stop"

aux_e2o_agency = aux_event_log[["event_id", "activity", "timestamp"] ]

aux_e2o_agency["oid"] = aux_event_log["agency_id"]

aux_e2o_agency["type"] = "agency"

aux_e2o_agency["qualifier"] = "used transport agency"

aux_e2o_vehicle = aux_event_log[["event_id", "activity", "timestamp"] ]

aux_e2o_vehicle["oid"] = aux_event_log["vehicle_id"]

aux_e2o_vehicle["type"] = "vehicle"

aux_e2o_vehicle["qualifier"] = "used vehicle"

aux_e2o_operator = aux_event_log[["event_id", "activity", "timestamp"] ]

aux_e2o_operator["oid"] = aux_event_log["organization_id"]

aux_e2o_operator["type"] = "operator"

aux_e2o_operator["qualifier"] = "used operator"

e2o_table = pd.concat([
                       aux_e2o_stops,
                       aux_e2o_route,
                       aux_e2o_agency,
                       aux_e2o_service,
                       aux_e2o_trip,
                       aux_e2o_vehicle,
                       aux_e2o_operator
                        ],ignore_index=True)
e2o_table = e2o_table.sort_values(["timestamp"]).reset_index(drop=True)

#print(tabulate(e2o_table, headers="keys", tablefmt="psql", showindex=False))

"""
Aux_e2o Aktivität A: direction change

"""

activity_a_events = a_events[a_events["activity"] == "direction_change"].copy()

aux_e2o_A_trip = activity_a_events[["event_id", "activity", "timestamp"]]

aux_e2o_A_trip["oid"] = activity_a_events["trip_id"]

aux_e2o_A_trip["type"] = "trip"

aux_e2o_A_trip["qualifier"] = "conduct trip"

aux_e2o_A_old_trip = activity_a_events[["event_id", "activity", "timestamp"]].copy()

aux_e2o_A_old_trip["oid"] = activity_a_events["old_trip_id"]

aux_e2o_A_old_trip["type"] = "trip"

aux_e2o_A_old_trip["qualifier"] = "recently conducted trip"

aux_e2o_A_route = activity_a_events[["event_id", "activity", "timestamp"]]

aux_e2o_A_route["oid"] = activity_a_events["route_id"]

aux_e2o_A_route["type"] = "route"

aux_e2o_A_route["qualifier"] = "conduct route"

aux_e2o_A_service = activity_a_events[["event_id", "activity", "timestamp"] ]

aux_e2o_A_service["oid"] = activity_a_events["service_id"]

aux_e2o_A_service["type"] = "service"

aux_e2o_A_service["qualifier"] = "conduct service"

aux_e2o_A_agency = activity_a_events[["event_id", "activity", "timestamp"] ]

aux_e2o_A_agency["oid"] = activity_a_events["agency_id"]

aux_e2o_A_agency["type"] = "agency"

aux_e2o_A_agency["qualifier"] = "used transport agency"

aux_e2o_A_vehicle = activity_a_events[["event_id", "activity", "timestamp"] ]

aux_e2o_A_vehicle["oid"] = activity_a_events["vehicle_id"]

aux_e2o_A_vehicle["type"] = "vehicle"

aux_e2o_A_vehicle["qualifier"] = "used vehicle"

aux_e2o_A_operator = activity_a_events[["event_id", "activity", "timestamp"] ]

aux_e2o_A_operator["oid"] = activity_a_events["organization_id"]

aux_e2o_A_operator["type"] = "operator"

aux_e2o_A_operator["qualifier"] = "used operator"

e2o_a_table = pd.concat([
    aux_e2o_A_trip,
    aux_e2o_A_old_trip,
    aux_e2o_A_route,
    aux_e2o_A_service,
    aux_e2o_A_agency,
    aux_e2o_A_vehicle,
    aux_e2o_A_operator
    ]
    ,ignore_index=True)
e2o_a_table = e2o_a_table.sort_values(["timestamp"]).reset_index(drop=True)

#print(tabulate(e2o_a_table.head(1000), headers="keys", tablefmt="psql"))

"""
Aktivität B: parking
"""
activity_b_events = a_events[a_events["activity"] == "parking"].copy()

aux_e2o_B_trip = activity_b_events[["event_id", "activity", "timestamp"]].copy()
aux_e2o_B_trip["oid"] = activity_b_events["old_trip_id"]
aux_e2o_B_trip["type"] = "trip"
aux_e2o_B_trip["qualifier"] = "recently conducted trip"

aux_e2o_B_route = activity_b_events[["event_id", "activity", "timestamp"]].copy()
aux_e2o_B_route["oid"] = activity_b_events["route_id"]
aux_e2o_B_route["type"] = "route"
aux_e2o_B_route["qualifier"] = "recently conducted route"

aux_e2o_B_service = activity_b_events[["event_id", "activity", "timestamp"]].copy()
aux_e2o_B_service["oid"] = activity_b_events["service_id"]
aux_e2o_B_service["type"] = "service"
aux_e2o_B_service["qualifier"] = "conduct service"





aux_e2o_B_agency = activity_b_events[["event_id", "activity", "timestamp"]].copy()
aux_e2o_B_agency["oid"] = activity_b_events["agency_id"]
aux_e2o_B_agency["type"] = "agency"
aux_e2o_B_agency["qualifier"] = "used transport agency"

aux_e2o_B_vehicle = activity_b_events[["event_id", "activity", "timestamp"]].copy()
aux_e2o_B_vehicle["oid"] = activity_b_events["vehicle_id"]
aux_e2o_B_vehicle["type"] = "vehicle"
aux_e2o_B_vehicle["qualifier"] = "used vehicle"

aux_e2o_B_operator = activity_b_events[["event_id", "activity", "timestamp"]].copy()
aux_e2o_B_operator["oid"] = activity_b_events["organization_id"]
aux_e2o_B_operator["type"] = "operator"
aux_e2o_B_operator["qualifier"] = "used operator"

e2o_b_table = pd.concat(
    [
        aux_e2o_B_trip,
        aux_e2o_B_route,
        aux_e2o_B_service,
        aux_e2o_B_agency,
        aux_e2o_B_vehicle,
        aux_e2o_B_operator
    ],
    ignore_index=True
)

e2o_b_table = e2o_b_table.dropna(subset=["oid"])
e2o_b_table = e2o_b_table.sort_values("timestamp").reset_index(drop=True)

#print(tabulate(e2o_b_table.head(1000), headers="keys", tablefmt="psql"))

e2o_cols = ["event_id", "activity", "timestamp", "oid", "type", "qualifier"]

e2o_table = e2o_table[e2o_cols]
e2o_a_table = e2o_a_table[e2o_cols]
e2o_b_table = e2o_b_table[e2o_cols]
e2o_begin_shift_table = e2o_begin_shift_table[e2o_cols]

e2o_complete_table = pd.concat(
    [e2o_table, e2o_a_table, e2o_b_table, e2o_begin_shift_table
     #,e2o_end_layover_table,e2o_layover_table],
],ignore_index=True
)

e2o_complete_table = e2o_complete_table.dropna(subset=["oid"])
e2o_complete_table = e2o_complete_table.sort_values("timestamp").reset_index(drop=True)
"""
print(tabulate(
    e2o_complete_table.head(1000),
    headers="keys",
    tablefmt="psql",
    showindex=False
))
"""



"""
#Object Table Creation

"""

df_stop_times_rt_tmp = df_stop_times_rt[["trip_id", "vehicle_id"]]

df_trips = df_trips.merge(df_stop_times_rt, on="trip_id", how="left")
#df_trips = df_trips.dropna(subset=["timestamp"])
df_trips = df_trips.dropna(subset=["arrival_time", "departure_time"])

#print("test2")
#print(tabulate(df_trips[df_trips["trip_id"]=='55700000078905434'].head(10), headers="keys", tablefmt="psql"))


#print(tabulate(df_trips.head(100),headers="keys",tablefmt="psql",showindex=False))

#df_trip = df_trips.drop_duplicates(subset=["trip_id"], keep="first")

"""
object_table_trip = df_trips.copy()
object_table_trip = object_table_trip.drop_duplicates(subset="trip_id_unique", keep="first")

print(tabulate(object_table_trip.head(100),headers="keys",tablefmt="psql"))


#object_table_trip = df_trips.drop_duplicates(
 #   subset=["trip_id", "direction_id", "vehicle_id"], keep="first"
#).copy()

object_table_trip["trip_id_org"] = object_table_trip["trip_id"]


object_table_trip["oid"] = object_table_trip["trip_id_unique"]

object_table_trip["type"] = "trip"


object_table_trip = object_table_trip.drop(columns=["route_id", "service_id"])

"""

used_trip_ids = aux_event_log["trip_id"].dropna().unique()

df_trips = df_trips.merge(trip_speed_summary, on="trip_id", how="left")

object_table_trip = (
    aux_event_log[
        [
            "trip_id",
            "trip_id_org",
            "direction_id",
            "vehicle_id",
            "organization_name"
        ]
    ]
    .dropna(subset=["trip_id"])
    .drop_duplicates(subset="trip_id", keep="first")
    .copy()
)
object_table_trip = object_table_trip.rename(columns={"trip_id": "oid"})
object_table_trip = object_table_trip.merge(
    trip_speed_summary.rename(columns={"trip_id": "trip_id_org"}),
    on="trip_id_org",
    how="left",
)
object_table_trip["type"] = "trip"




object_table_route =  df_routes.copy()



used_route_ids = aux_event_log["route_id"].dropna().unique()

object_table_route = df_routes[df_routes["route_id"].isin(used_route_ids)].copy()

object_table_route["type"] = "route"
object_table_route["oid"] = object_table_route["route_id"]
object_table_route = object_table_route.drop(columns=["route_id", "agency_id"])


#print(tabulate(object_table_route, headers="keys", tablefmt="psql", showindex=False))


used_stop_ids = aux_event_log["stop_id"].dropna().unique()

object_table_stops = df_stops[df_stops["stop_id"].isin(used_stop_ids)].copy()
object_table_stops["type"] = "stops"
object_table_stops["oid"] = object_table_stops["stop_id"]
object_table_stops = object_table_stops.drop(columns=["stop_id"])



used_agency_ids = aux_event_log["agency_id"].dropna().unique()

object_table_agency = df_agency[df_agency["agency_id"].isin(used_agency_ids)].copy()
object_table_agency["type"] = "agency"
object_table_agency["oid"] = object_table_agency["agency_id"]
object_table_agency = object_table_agency.drop(columns=["agency_id"])

used_operator_ids = aux_event_log["organization_id"].dropna().unique()

object_table_operator = df_attributions[
    df_attributions["organization_id"].isin(used_operator_ids)
].copy()
object_table_operator = object_table_operator.drop_duplicates(
    subset=["organization_id"], keep="first"
).copy()
object_table_operator["type"] = "operator"
object_table_operator["oid"] = object_table_operator["organization_id"]
object_table_operator = object_table_operator.drop(columns=["trip_id", "organization_id"])



# = aux_event_log["vehicle_id"].dropna().unique()

#print(tabulate(used_vehicle_ids, headers="keys", tablefmt="psql"))


#object_table_vehicle = df_stop_times_rt[df_stop_times_rt["vehicle_id"].isin(used_vehicle_ids)].copy()
#object_table_vehicle = aux_event_log[aux_event_log["vehicle_id"].isin(used_vehicle_ids)].copy()
used_vehicle_ids = aux_event_log["vehicle_id"].dropna().unique()


object_table_vehicle = pd.DataFrame({
    "oid": used_vehicle_ids
})

object_table_vehicle["oid"] = object_table_vehicle["oid"].astype(str)
vehicle_route_types = aux_event_log.dropna(subset=["vehicle_id", "route_type"]).copy()
vehicle_route_types["vehicle_id"] = vehicle_route_types["vehicle_id"].astype(str)
vehicle_route_type_map = (
    vehicle_route_types.groupby(["vehicle_id", "route_type"])
    .size()
    .reset_index(name="count")
    .sort_values(["vehicle_id", "count"], ascending=[True, False])
    .drop_duplicates(subset=["vehicle_id"], keep="first")
    [["vehicle_id", "route_type"]]
)

object_table_vehicle = object_table_vehicle.merge(
    vehicle_route_type_map,
    left_on="oid",
    right_on="vehicle_id",
    how="left"
).drop(columns=["vehicle_id"])
object_table_vehicle["route_type"] = pd.to_numeric(
    object_table_vehicle["route_type"], errors="coerce"
).astype("Int64")
object_table_vehicle["vehicle_type"] = object_table_vehicle["route_type"].apply(route_type_category)
object_table_vehicle["route_type"] = object_table_vehicle["route_type"].astype("string")


object_table_vehicle["type"] = "vehicle"

#print("test")
#print(tabulate(object_table_vehicle.head(10), headers="keys", tablefmt="psql"))

#object_table_vehicle = used_vehicle_ids
#object_table = aux_event_log["vehicle_id"]
object_table_vehicle["type"] = "vehicle"
#object_table_vehicle = object_table_vehicle.drop(columns=["vehicle_id"])

#print("vehicle_id table")
#print(tabulate(object_table_vehicle.head(10), headers="keys", tablefmt="psql"))
used_service_ids = aux_event_log["service_id"].dropna().unique()


""""
object_table_calendar_dates = df_calendar_dates[
    df_calendar_dates["service_id"].isin(used_service_ids)
].copy()

object_table_calendar_dates = object_table_calendar_dates.drop_duplicates(
    subset=["service_id"], keep="first"
).copy()

object_table_calendar_dates["type"] = "service"
object_table_calendar_dates["oid"] = object_table_calendar_dates["service_id"]
object_table_calendar_dates = object_table_calendar_dates.drop(columns=["service_id"])

"""



object_table = pd.concat([object_table_trip, object_table_stops, object_table_agency, object_table_operator, object_table_route,
                          #object_table_calendar_dates,
                          object_table_vehicle])

#object_table = pd.concat([object_table_trip, object_table_stops, object_table_agency, object_table_route])

#print(print(tabulate(object_table.head(40), headers="keys", tablefmt="psql", showindex=False)))

#print(object_table.columns)

#print(list(object_table.columns))
#print((object_table["vehicle_id"].nunique()))

"""
#O2O Object Table

"""

aux_route_trip = aux_event_log[["route_id", "trip_id"]].rename(
    columns={"trip_id": "ocel_source_id", "route_id": "ocel_target_id"}
).drop_duplicates()

aux_route_trip["oid_qualifier"] = "trip belongs to route"


#print(tabulate(aux_route_trip, headers="keys", tablefmt="psql", showindex=False))

aux_trip_stops = aux_event_log[["trip_id", "stop_id"]].rename(
    columns={"trip_id": "ocel_source_id", "stop_id": "ocel_target_id"}
).drop_duplicates()

aux_trip_stops["oid_qualifier"] = "stop belongs to trip"

aux_trip_vehicle = aux_event_log[["trip_id", "vehicle_id"]].rename(
    columns={"trip_id": "ocel_source_id", "vehicle_id": "ocel_target_id"}
).drop_duplicates()

aux_trip_vehicle["oid_qualifier"] = "trip is conducted by vehicle"


aux_vehicle_stops = aux_event_log[["stop_id", "vehicle_id"]].rename(
    columns={"stop_id": "ocel_source_id", "vehicle_id": "ocel_target_id"}
).drop_duplicates()

aux_vehicle_stops["oid_qualifier"] = "vehicle stops at stop"

aux_agency_vehicle = aux_event_log[["agency_id", "vehicle_id"]].rename(
    columns={"agency_id": "ocel_source_id", "vehicle_id": "ocel_target_id"}
).drop_duplicates()

aux_agency_vehicle["oid_qualifier"] = "used vehicle by agency"


aux_trip_operator = aux_event_log[["trip_id", "organization_id"]].rename(
    columns={"trip_id": "ocel_source_id", "organization_id": "ocel_target_id"}
).drop_duplicates()

aux_trip_operator["oid_qualifier"]= "trip is conducted by operator"




#print(tabulate(aux_trip_stops, headers="keys", tablefmt="psql", showindex=False))

aux_agency_routes = aux_event_log[["route_id", "agency_id"]].rename(
    columns={"route_id": "ocel_source_id", "agency_id": "ocel_target_id"}
).drop_duplicates()

aux_agency_routes["oid_qualifier"] = "agency conducts route"

#print(tabulate(aux_agency_routes, headers="keys", tablefmt="psql", showindex=False))

aux_trips_calendar = aux_event_log[["trip_id", "service_id"]].rename(
    columns={"trip_id": "ocel_source_id", "service_id": "ocel_target_id"}
).drop_duplicates()

aux_trips_calendar["oid_qualifier"] = "trip belongs to service"

#print(tabulate(aux_trips_calendar, headers="keys", tablefmt="psql", showindex=False))

#("test23")
#print(tabulate(aux_trip_vehicle.head(10), headers="keys", tablefmt="psql"))

source_target = pd.concat([aux_route_trip, aux_trip_stops, aux_agency_routes, aux_trips_calendar, aux_trip_vehicle, aux_agency_vehicle,aux_vehicle_stops, aux_trip_operator], ignore_index=True)
relationships_by_source = {
    source_id: [
        {
            "objectId": row.ocel_target_id,
            "qualifier": row.oid_qualifier,
        }
        for row in group.itertuples(index=False)
    ]
    for source_id, group in source_target.groupby("ocel_source_id", sort=False)
}

#print(tabulate(source_target, headers="keys", tablefmt="psql", showindex=False))

activity_a_events = a_events[a_events["activity"] == "direction_change"].copy()
activity_b_events = a_events[a_events["activity"] == "parking"].copy()

#print(tabulate(activity_b_events.head(1000), headers="keys",tablefmt="psql", showindex=False))



stop_event_attributes = [
    {"name": "latitude", "type": "float"},
    {"name": "longitude", "type": "float"},
    {"name": "stop_sequence", "type": "integer"},
    {"name": "route_short_name", "type": "string"},
    {"name": "route_type", "type": "integer"},
    {"name": "delay", "type": "float"},
    {"name": "occupancy_status", "type": "integer"},
    {"name": "dist", "type": "float"},
]

begin_shift_attributes = [
    {"name": "latitude", "type": "float"},
    {"name": "longitude", "type": "float"},
    {"name": "stop_sequence", "type": "integer"},
    {"name": "route_short_name", "type": "string"},
    {"name": "route_type", "type": "integer"},
]

route_change_attributes = [
    {"name": "route_type", "type": "integer"},
]

event_type_records = []
for activity in sorted(aux_event_log["activity"].dropna().astype(str).unique()):
    if activity.startswith(("arrive_stop", "departure_stop")):
        event_type_records.append({
            "name": activity,
            "attributes": [dict(attribute) for attribute in stop_event_attributes],
        })

event_type_records.extend([
    {"name": "begin_shift", "attributes": begin_shift_attributes},
    {"name": "direction_change", "attributes": route_change_attributes},
    {"name": "parking", "attributes": route_change_attributes},
])

eventTypes = pd.DataFrame(event_type_records)
#print(aux_event_log["shape_dist_traveled"])

events = pd.DataFrame([ {"id": e.event_id,
                        "type": e.activity,
                        "time": e.timestamp,
                         "attributes": [
                             {
                            "name": "latitude",
                            "value": e.stop_lat
                             },
                            {
                            "name": "longitude",
                            "value": e.stop_lon
                            },
                            {
                            "name": "stop_sequence",
                            "value": e.stop_sequence
                            },
                             {

                            "name": "route_short_name",
                             "value": e.route_short_name
                             },
                             {
                            "name": "route_type",
                             "value": e.route_type
                             },
                             {"name": "delay",
                              "value": e.delay},
                             {"name": "occupancy_status",
                              "value": e.occupancy_status},
                            {"name": "dist",
                            "value": e.shape_dist_traveled},
                            ],
                         "relationships": [
                             {"objectId": e.trip_id, "qualifier": "conduct trip"},
                             {"objectId": e.stop_id, "qualifier": "used bus stop"},
                             {"objectId": e.service_id, "qualifier": "conduct service"},
                             {"objectId": e.agency_id, "qualifier": "used transport agency"},
                             {"objectId": e.route_id, "qualifier": "conduct route"},
                             {"objectId": e.vehicle_id, "qualifier": "used vehicle"},
                             {"objectId": e.organization_id, "qualifier": "used operator"}
                         ]
                        }
                       for e in aux_event_log.itertuples(index=False)])

events_a = pd.DataFrame([
                        {
                            "id": e.event_id,
                            "type": e.activity,
                            "time": e.timestamp,
                            "attributes": [
                                {"name": "route_type", "value": e.route_type}
                            ],
                            "relationships": [
                                {"objectId": e.trip_id, "qualifier": "conduct trip"},
                                {"objectId": e.old_trip_id, "qualifier": "recently conducted trip"},
                                {"objectId": e.service_id, "qualifier": "conduct service"},
                                {"objectId": e.agency_id, "qualifier": "used transport agency"},
                                {"objectId": e.route_id, "qualifier": "conduct route"},
                                {"objectId": e.vehicle_id, "qualifier": "used vehicle"},
                                {"objectId": e.organization_id, "qualifier": "used operator"}
                            ],
                        }
                        for e in activity_a_events.itertuples(index=False)
                    ])


events_b = pd.DataFrame([
                        {
                            "id": e.event_id,
                            "type": e.activity,
                            "time": e.timestamp,
                            "attributes": [
                                {"name": "route_type", "value": e.route_type}
                            ],
                            "relationships": [
                                {"objectId": e.old_trip_id, "qualifier": "recently conducted trip"},
                                {"objectId": e.service_id, "qualifier": "conduct service"},
                                {"objectId": e.agency_id, "qualifier": "used transport agency"},
                                {"objectId": e.route_id, "qualifier": "recently conducted route"},
                                {"objectId": e.vehicle_id, "qualifier": "used vehicle"},
                                {"objectId": e.organization_id, "qualifier": "used operator"}
                            ],
                        }
                        for e in activity_b_events.itertuples(index=False)
                    ])



events_begin_shift = pd.DataFrame([
                    {
                        "id": e.event_id,
                        "type": e.activity,
                        "time": e.timestamp,
                        "attributes": [
                            {"name": "latitude", "value": e.stop_lat},
                            {"name": "longitude", "value": e.stop_lon},
                            {"name": "stop_sequence", "value": e.stop_sequence},
                            {"name": "route_short_name", "value": e.route_short_name},
                            {"name": "route_type", "value": e.route_type},
                        ],
                        "relationships": [
                            {"objectId": e.trip_id, "qualifier": "conduct trip"},
                            {"objectId": e.stop_id, "qualifier": "used bus stop"},
                            {"objectId": e.service_id, "qualifier": "conduct service"},
                            {"objectId": e.agency_id, "qualifier": "used transport agency"},
                            {"objectId": e.route_id, "qualifier": "conduct route"},
                            {"objectId": e.vehicle_id, "qualifier": "used vehicle"},
                            {"objectId": e.organization_id, "qualifier": "used operator"}
                        ],
                    }
                    for e in begin_shift_events.itertuples(index=False)
                ])




events_complete = pd.concat(
                    [events, events_a, events_b
                    # events_layover, events_end_layover
                        , events_begin_shift],
                    ignore_index=True
                )


## aux event log hilfstabelle für flat event log
aux_event_log_overview = pd.concat(
    [
        aux_event_log,
        begin_shift_events,
       # begin_layover_events,
        activity_a_events,
        activity_b_events,
        #end_layover_events
    ],
    ignore_index=True
)





events_complete = sort_by_activity_order(events_complete, "time", "type")

aux_event_log_overview = sort_by_activity_order(aux_event_log_overview, "timestamp", "activity")

#aux_event_log_overview["trip_id"] = aux_event_log_overview["trip_id"].astype("Int64")aux_event_log_overview["activity_type"] = aux_event_log_overview["activity"]

aux_event_log_overview["activity_type"] = aux_event_log_overview["activity"].apply(derive_activity_type)
#print("auflistung")
#print("Anzahl unterschiedlicher Typen:", aux_event_log_overview["activity_type"].nunique())
#print(aux_event_log_overview["activity_type"].value_counts())

#print(tabulate(events_complete.head(50), headers="keys", tablefmt="psql"))

aux_event_log_overview = aux_event_log_overview[["event_id", "timestamp", "activity","arrival_time_rt", "trip_id", "stop_sequence", "stop_id", "stop_name", "stop_lat", "stop_lon","direction_id", "vehicle_id", "route_short_name", "delay", "occupancy_status" , "route_id", "route_type", "stop_headsign", "route_desc", "activity_type","trip_id_org","old_trip_id", "shape_dist_traveled", "segment_id", "segment_departure_stop_id", "segment_departure_stop_lat", "segment_departure_stop_lon", "segment_arrive_stop_id", "segment_arrive_stop_lat", "segment_arrive_stop_lon", "cumulative_road_distance_m", "cumulative_road_distance_km" ]]
aux_event_log_overview_kodak = aux_event_log_overview
aux_event_log_overview_kodak.to_csv("aux_event_log_overview_kodak_2026_05_04_new_e2o_no_layover.csv", index=False)
#("flat event log")
#print(tabulate(aux_event_log_overview_kodak.head(100), headers="keys", tablefmt="psql"))

#print(tabulate(aux_event_log_overview[aux_event_log_overview["timestamp"]>'2026-05-17 00:00:00'].head(1000), headers="keys", tablefmt="psql"))
#print(tabulate(aux_event_log_overview[aux_event_log_overview["trip_id"]=='55700000078906084'].head(1000), headers="keys", tablefmt="psql"))
#print(tabulate(aux_event_log_overview[(aux_event_log_overview["trip_id"]=='55700000081940174') | (aux_event_log_overview["trip_id"]=='55700000085877395')
 #              ].head(3000), headers="keys", tablefmt="psql"))

#& (aux_event_log_overview["timestamp"]>'2026-05-04 12:00:00')
#print(tabulate(aux_event_log_overview[(aux_event_log_overview["route_id"]=='9011005008300000') ].head(100000), headers="keys", tablefmt="psql"))
#print(tabulate(aux_event_log_overview[(aux_event_log_overview["event_id"]=='ad_67361')  ].head(100000), headers="keys", tablefmt="psql"))
#print(tabulate(aux_event_log_overview[(aux_event_log_overview["event_id"]=='ad_67486')  ].head(100000), headers="keys", tablefmt="psql"))


#print(tabulate(aux_event_log_overview[aux_event_log_overview["route_short_name"]=='39'].head(50), headers="keys", tablefmt="psql"))
#print(tabulate(aux_event_log_overview[(aux_event_log_overview["vehicle_id"]=='9031005920804816')  ].head(10000), headers="keys", tablefmt="psql"))

#print(tabulate(aux_event_log_overview[aux_event_log_overview["trip_id"]==55700000078905434].head(50), headers="keys", tablefmt="psql"))


object_table_routes = pd.DataFrame([{"id": o.oid,
                        "type": o.type,
                        "attributes": [
                            {
                                "name": "route_long_name",
                                "value": o.route_long_name
                        },
                            {
                                "name": "route_short_name",
                                "value": o.route_short_name
                            },
                            {
                                "name": "route_type",
                                "value": o.route_type
                            }
                            ,
{
                                "name": "route_id",
                                "value": o.oid
                            }

                        ],
                        "relationships": relationships_by_source.get(o.oid, [])



                        }
                       for o in object_table_route.itertuples(index=False)])

object_table_trip = pd.DataFrame([{"id": o.oid,
                        "type": o.type,
                        "attributes": [
                            {
                                "name": "direction_id",
                                "value": o.direction_id
                        },

                            {
                                "name": "vehicle_id",
                                "value": o.vehicle_id
                            },
                            {
                                "name": "trip_id",
                                "value": o.trip_id_org
                            }
                            ,
                            {
                                "name": "organization_name",
                                "value": o.organization_name
                            }
                            ,
                            {
                                "name": "avg_speed",
                                "value": o.avg_speed
                            }
                            ,
                            {
                                "name": "median_speed",
                                "value": o.median_speed
                            }

                        ],
                        "relationships": relationships_by_source.get(o.oid, [])
                        }
                       for o in object_table_trip.itertuples(index=False)])
#print("test3")
#print(tabulate(object_table_trip.head(10), headers="keys", tablefmt="psql"))
object_table_vehicle = pd.DataFrame([{"id": o.oid,
                        "type": o.type,
                        "attributes": [
                            {"name": "vehicle_id_object",
                             "value": o.oid},
                            {"name": "vehicle_type",
                             "value": o.vehicle_type}

                        ],
                        "relationships": relationships_by_source.get(o.oid, [])
                        }
                       for o in object_table_vehicle.itertuples(index=False)])

#("test4")
#print(tabulate(object_table_vehicle[object_table_vehicle["id"]=='9031005920505505'].head(10), headers="keys", tablefmt="psql"))

object_table_agency = pd.DataFrame([{"id": o.oid,
                        "type": o.type,
                        "attributes": [
                            {
                                "name": "agency_name",
                                "value": o.agency_name
                        },
                            {
                                "name": "agency_url",
                                "value": o.agency_url
                            },
                            {
                                "name": "agency_timezone",
                                "value": o.agency_timezone
                            }
                            ,
                            {
                                "name": "agency_id",
                                "value": o.oid
                            }


                        ],
                        "relationships": relationships_by_source.get(o.oid, [])



                        }
                       for o in object_table_agency.itertuples(index=False)])

object_table_operator = pd.DataFrame([{"id": o.oid,
                        "type": o.type,
                        "attributes": [
                            {
                                "name": "organization_name",
                                "value": o.organization_name
                            },
                            {
                                "name": "organization_id",
                                "value": o.oid
                            },
                            {
                                "name": "is_operator",
                                "value": o.is_operator
                            }
                        ],
                        "relationships": relationships_by_source.get(o.oid, [])
                        }
                       for o in object_table_operator.itertuples(index=False)])


object_table_stops = pd.DataFrame([{"id": o.oid,
                        "type": o.type,
                        "attributes": [
                            {
                                "name": "stop_name",
                                "value": o.stop_name
                        },
                            {
                                "name": "latitude",
                                "value": o.stop_lat
                            },
                            {
                                "name": "longitude",
                                "value": o.stop_lon
                            },
                            {
                                "name": "stop_id",
                                "value": o.oid
                            }


                        ],
                        "relationships": relationships_by_source.get(o.oid, [])



                        }
                       for o in object_table_stops.itertuples(index=False)])


"""
object_table_calendar_dates = pd.DataFrame([{"id": o.oid,
                        "type": o.type,
                        "attributes": [
                            {
                                "name": "service_id",
                                "value": o.oid
                            },
                            {
                                "name": "date",
                                "value": o.date
                        },
                            {
                                "name": "exception_type",
                                "value": o.exception_type
                            }


                        ],
                        "relationships": [
                        {
                            "objectId": row.ocel_target_id,
                            "qualifier": row.oid_qualifier
                                }
                            for row in source_target[source_target["ocel_source_id"] == o.oid].itertuples(index=False)


                        ]



                        }
                       for o in object_table_calendar_dates.itertuples(index=False)])

"""


object_table_routes, route_duplicates_removed = dedupe_exported_objects(
    object_table_routes, "object_table_routes"
)
object_table_trip, trip_duplicates_removed = dedupe_exported_objects(
    object_table_trip, "object_table_trip"
)
object_table_agency, agency_duplicates_removed = dedupe_exported_objects(
    object_table_agency, "object_table_agency"
)
object_table_operator, operator_duplicates_removed = dedupe_exported_objects(
    object_table_operator, "object_table_operator"
)
object_table_stops, stop_duplicates_removed = dedupe_exported_objects(
    object_table_stops, "object_table_stops"
)
#object_table_calendar_dates, service_duplicates_removed = dedupe_exported_objects(
 #   object_table_calendar_dates, "object_table_calendar_dates"
#)
object_table_vehicle, vehicle_duplicates_removed = dedupe_exported_objects(
    object_table_vehicle, "object_table_vehicle"
)

#print("used_route_ids:", len(used_route_ids))
#print("object_table_route:", len(object_table_route))
#print("object_table_routes:", len(object_table_routes))

#print("used_agency_ids:", len(used_agency_ids))
#print("object_table_agency:", len(object_table_agency))


objects_export = pd.concat(
    [
        object_table_routes,
        object_table_trip,
        object_table_agency,
        object_table_operator,
        object_table_stops,
      #  object_table_calendar_dates,
        object_table_vehicle
    ],
    ignore_index=True,

)


objects_export, total_duplicates_removed = dedupe_exported_objects(
    objects_export, "objects_export_total"
)



print(
    "Duplicate objects removed overall:",
    route_duplicates_removed
    + trip_duplicates_removed
    + agency_duplicates_removed
    + operator_duplicates_removed
    + stop_duplicates_removed
  #  + service_duplicates_removed
    + total_duplicates_removed,
)



#print(tabulate(object_table_routes, headers="keys", tablefmt="psql"))


#print(tabulate(object_table_trip.head(20), headers="keys", tablefmt="psql"))


object_types = pd.DataFrame([{"name": "trip",
                        "attributes": [
                            {
                                "name": "direction_id",
                                "type": "integer"
                        },


                            {
                                "name": "vehicle_id",
                                "type": "string"
                            },
                            {
                                "name": "trip_id",
                                "type": "string"
                            },
                            {
                                "name": "organization_name",
                                "type": "string"
                            },
                            {
                                "name": "avg_speed",
                                "type": "float"
                            },
                            {
                                "name": "med_speed",
                                "type": "float"
                            }



                        ]



                        },
                    {"name": "vehicle",
                        "attributes": [

                            {"name": "vehicle_id_object",
                             "type": "string"},
                            {"name": "vehicle_type",
                             "type": "string"}

                        ]



                        },
                        {"name": "route",
                        "attributes": [
                            {
                                "name": "route_long_name",
                                "type": "string"
                            },
                            {
                                "name": "route_short_name",
                                "type": "string"
                            },
                            {
                                "name": "route_type",
                                "type": "integer"
                            },
                            {
                                "name": "route_id",
                                "type": "string"
                            }


                        ]},
                        {"name": "agency",
                        "attributes": [
                            {
                                "name": "agency_name",
                                "type": "string"
                            },
                            {
                                "name": "agency_url",
                                "type": "string"
                            },
                            {
                                "name": "agency_timezone",
                                "type": "string"
                            }
                            ,
                            {
                                "name": "agency_id",
                                "type": "string"
                            }



                        ]},
                        {"name": "operator",
                        "attributes": [
                            {
                                "name": "organization_name",
                                "type": "string"
                            },
                            {
                                "name": "organization_id",
                                "type": "string"
                            },
                            {
                                "name": "is_operator",
                                "type": "integer"
                            }



                        ]}
                                #,
                     #   {"name": "service",
                      #  "attributes": [
                       #     {
                        #        "name": "service_id",
                         #       "type": "string"
                          #  },
                           # {
                            #    "name": "date",
                    #         #   "type": "string"
                     #       },
                      #      {
                       #         "name": "exception_type",
                        #        "type": "integer"
                         #   }
                      #  ]}
,
                        {"name": "stops",
                        "attributes": [
                            {
                                "name": "stop_name",
                                "type": "string"
                            },
                            {
                                "name": "latitude",
                                "type": "float"
                            },
                            {
                                "name": "longitude",
                                "type": "float"
                            }
                            ,
                            {
                                "name": "stop_id",
                                "type": "string"
                            }


                        ]}


                             ])
ocel = {
    "eventTypes": eventTypes.to_dict(orient="records"),
    "objectTypes": object_types.to_dict(orient="records"),
    "events": events_complete.to_dict(orient="records"),
    "objects": objects_export.to_dict(orient="records")
}

result = (
    objects_export.groupby("type")["id"]
    .nunique()
    .reset_index()
)

#print(tabulate(result, headers="keys", tablefmt="psql", showindex=False))

ocel = normalize_json_value(ocel)



with open("ocel_koda_2026_05_04_w_stop_names_no_layover.json", "w", encoding="utf-8") as f:
    json.dump(ocel, f, ensure_ascii=False, indent=2, default=str, allow_nan=False)
