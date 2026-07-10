import math
from pathlib import Path

import pandas as pd


INPUT_CSV = Path(
    "aux_event_log_overview_kodak_2026_05_04_new_e2o_no_layover_shift_position_work.csv"
)
OUTPUT_CSV = Path(
    "h5_last_to_first_stop_sequence_transition_distances_2026_05_04_distance_change_rule.csv"
)
SUMMARY_CSV = Path(
    "h5_last_to_first_stop_sequence_transition_distance_summary_2026_05_04_distance_change_rule.csv"
)
ROUTE_CHANGE_OUTPUT_CSV = Path(
    "h5_route_change_last_to_first_stop_sequence_transition_distances_2026_05_04_distance_change_rule.csv"
)
ROUTE_CHANGE_SUMMARY_CSV = Path(
    "h5_route_change_last_to_first_stop_sequence_transition_distance_summary_2026_05_04_distance_change_rule.csv"
)
ROUTE_CHANGE_NONZERO_OUTPUT_CSV = Path(
    "h5_route_change_nonzero_last_to_first_stop_sequence_transition_distances_2026_05_04_distance_change_rule.csv"
)
ROUTE_CHANGE_NONZERO_SUMMARY_CSV = Path(
    "h5_route_change_nonzero_last_to_first_stop_sequence_transition_distance_summary_2026_05_04_distance_change_rule.csv"
)
CONSECUTIVE_OUTPUT_CSV = Path(
    "consecutive_trip_last_to_first_stop_sequence_transition_distances_2026_05_04_distance_change_rule.csv"
)
CONSECUTIVE_SUMMARY_CSV = Path(
    "consecutive_trip_last_to_first_stop_sequence_transition_distance_summary_2026_05_04_distance_change_rule.csv"
)


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


def first_non_null(values: pd.Series):
    values = values.dropna()
    return values.iloc[0] if not values.empty else pd.NA


def prepare_trip_boundaries(event_log: pd.DataFrame) -> pd.DataFrame:
    stop_events = event_log[
        event_log["activity_type"].isin(["arrive_stop", "departure_stop"])
        & event_log["trip_id"].notna()
        & event_log["stop_sequence"].notna()
    ].copy()

    stop_events["timestamp_dt"] = pd.to_datetime(
        stop_events["timestamp"], errors="coerce"
    )
    stop_events["stop_sequence_num"] = pd.to_numeric(
        stop_events["stop_sequence"], errors="coerce"
    )
    stop_events = stop_events.dropna(
        subset=[
            "stop_sequence_num",
            "stop_lat",
            "stop_lon",
        ]
    ).copy()

    first_stop = (
        stop_events.sort_values(["trip_id", "stop_sequence_num", "timestamp_dt", "event_id"])
        .groupby("trip_id", as_index=False)
        .first()
    )
    last_stop = (
        stop_events.sort_values(["trip_id", "stop_sequence_num", "timestamp_dt", "event_id"])
        .groupby("trip_id", as_index=False)
        .last()
    )

    boundary_columns = [
        "trip_id",
        "event_id",
        "timestamp",
        "timestamp_dt",
        "stop_sequence_num",
        "stop_id",
        "stop_name",
        "stop_lat",
        "stop_lon",
        "vehicle_id",
        "route_short_name",
        "route_id",
        "route_type",
        "direction_id",
    ]

    boundaries = first_stop[boundary_columns].merge(
        last_stop[boundary_columns],
        on="trip_id",
        how="inner",
        suffixes=("_first", "_last"),
    )

    boundaries["vehicle_group"] = boundaries["vehicle_id_first"].where(
        boundaries["vehicle_id_first"].notna(),
        "missing_vehicle_route_" + boundaries["route_short_name_first"].astype(str),
    )
    return boundaries


def build_transition_rows(boundaries: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, group in boundaries.groupby("vehicle_group", dropna=False):
        group = group.sort_values(
            ["timestamp_dt_first", "timestamp_dt_last", "trip_id"]
        ).reset_index(drop=True)

        for i in range(len(group) - 1):
            old_trip = group.iloc[i]
            next_trip = group.iloc[i + 1]

            distance_m = haversine_m(
                float(old_trip["stop_lat_last"]),
                float(old_trip["stop_lon_last"]),
                float(next_trip["stop_lat_first"]),
                float(next_trip["stop_lon_first"]),
            )
            rows.append({
                "vehicle_group": old_trip["vehicle_group"],
                "vehicle_id": old_trip["vehicle_id_first"],
                "old_trip_id": old_trip["trip_id"],
                "next_trip_id": next_trip["trip_id"],
                "old_route_short_name": old_trip["route_short_name_last"],
                "next_route_short_name": next_trip["route_short_name_first"],
                "route_short_name_changed": (
                    str(old_trip["route_short_name_last"])
                    != str(next_trip["route_short_name_first"])
                ),
                "old_route_id": old_trip["route_id_last"],
                "next_route_id": next_trip["route_id_first"],
                "old_route_type": old_trip["route_type_last"],
                "next_route_type": next_trip["route_type_first"],
                "old_direction_id": old_trip["direction_id_last"],
                "next_direction_id": next_trip["direction_id_first"],
                "old_last_event_id": old_trip["event_id_last"],
                "next_first_event_id": next_trip["event_id_first"],
                "old_last_timestamp": old_trip["timestamp_last"],
                "next_first_timestamp": next_trip["timestamp_first"],
                "time_gap_minutes": (
                    next_trip["timestamp_dt_first"] - old_trip["timestamp_dt_last"]
                ).total_seconds() / 60,
                "old_last_stop_sequence": int(old_trip["stop_sequence_num_last"]),
                "next_first_stop_sequence": int(next_trip["stop_sequence_num_first"]),
                "old_last_stop_id": old_trip["stop_id_last"],
                "old_last_stop_name": old_trip["stop_name_last"],
                "old_last_stop_lat": old_trip["stop_lat_last"],
                "old_last_stop_lon": old_trip["stop_lon_last"],
                "next_first_stop_id": next_trip["stop_id_first"],
                "next_first_stop_name": next_trip["stop_name_first"],
                "next_first_stop_lat": next_trip["stop_lat_first"],
                "next_first_stop_lon": next_trip["stop_lon_first"],
                "transition_distance_m": distance_m,
                "transition_distance_km": distance_m / 1000,
                "derived_transition_activity": (
                    "stationary_pause"
                    if pd.notna(distance_m) and distance_m == 0
                    else "repositioning_pause"
                ),
            })

    output_columns = [
        "derived_transition_activity",
        "old_trip_id",
        "next_trip_id",
        "old_route_short_name",
        "next_route_short_name",
        "route_short_name_changed",
        "old_direction_id",
        "next_direction_id",
        "time_gap_minutes",
        "old_last_stop_sequence",
        "next_first_stop_sequence",
        "old_last_stop_name",
        "next_first_stop_name",
        "old_last_stop_lat",
        "old_last_stop_lon",
        "transition_distance_m",
        "transition_distance_km",
    ]
    return pd.DataFrame(rows)[output_columns].sort_values(
        ["transition_distance_m", "old_trip_id", "next_trip_id"]
    ).reset_index(drop=True)


def build_h5_transition_rows(event_log: pd.DataFrame, boundaries: pd.DataFrame) -> pd.DataFrame:
    transition_events = event_log[
        event_log["activity"].isin(["stationary_pause", "repositioning_pause"])
        & event_log["old_trip_id"].notna()
        & event_log["trip_id"].notna()
    ].copy()
    transition_events = transition_events.drop_duplicates("event_id")

    old_columns = [
        "trip_id",
        "event_id_last",
        "timestamp_last",
        "timestamp_dt_last",
        "stop_sequence_num_last",
        "stop_id_last",
        "stop_name_last",
        "stop_lat_last",
        "stop_lon_last",
        "vehicle_id_last",
        "route_short_name_last",
        "route_id_last",
        "route_type_last",
        "direction_id_last",
    ]
    next_columns = [
        "trip_id",
        "event_id_first",
        "timestamp_first",
        "timestamp_dt_first",
        "stop_sequence_num_first",
        "stop_id_first",
        "stop_name_first",
        "stop_lat_first",
        "stop_lon_first",
        "vehicle_id_first",
        "route_short_name_first",
        "route_id_first",
        "route_type_first",
        "direction_id_first",
    ]

    rows = (
        transition_events[
            ["event_id", "timestamp", "activity", "old_trip_id", "trip_id", "vehicle_id"]
        ]
        .rename(columns={
            "event_id": "transition_event_id",
            "timestamp": "transition_timestamp",
            "activity": "h5_activity",
            "trip_id": "next_trip_id",
            "vehicle_id": "transition_vehicle_id",
        })
        .merge(
            boundaries[old_columns].rename(columns={"trip_id": "old_trip_id"}),
            on="old_trip_id",
            how="left",
        )
        .merge(
            boundaries[next_columns].rename(columns={"trip_id": "next_trip_id"}),
            on="next_trip_id",
            how="left",
        )
    )

    rows = rows.dropna(
        subset=[
            "stop_lat_last",
            "stop_lon_last",
            "stop_lat_first",
            "stop_lon_first",
            "stop_sequence_num_last",
            "stop_sequence_num_first",
        ]
    ).copy()

    distances_m = []
    for _, row in rows.iterrows():
        distances_m.append(
            haversine_m(
                float(row["stop_lat_last"]),
                float(row["stop_lon_last"]),
                float(row["stop_lat_first"]),
                float(row["stop_lon_first"]),
            )
        )
    rows["transition_distance_m"] = distances_m
    rows["transition_distance_km"] = rows["transition_distance_m"] / 1000
    rows["distance_based_activity"] = rows.apply(
        lambda row: (
            "stationary_pause"
            if pd.notna(row["transition_distance_m"])
            and row["transition_distance_m"] == 0
            else "repositioning_pause"
        ),
        axis=1,
    )
    rows["h5_matches_distance_rule"] = (
        rows["h5_activity"] == rows["distance_based_activity"]
    )
    rows["time_gap_minutes"] = (
        rows["timestamp_dt_first"] - rows["timestamp_dt_last"]
    ).dt.total_seconds() / 60
    rows["route_short_name_changed"] = (
        rows["route_short_name_last"].astype(str)
        != rows["route_short_name_first"].astype(str)
    )

    rows = rows.rename(columns={
        "event_id_last": "old_last_event_id",
        "event_id_first": "next_first_event_id",
        "timestamp_last": "old_last_timestamp",
        "timestamp_first": "next_first_timestamp",
        "stop_sequence_num_last": "old_last_stop_sequence",
        "stop_sequence_num_first": "next_first_stop_sequence",
        "stop_id_last": "old_last_stop_id",
        "stop_name_last": "old_last_stop_name",
        "stop_lat_last": "old_last_stop_lat",
        "stop_lon_last": "old_last_stop_lon",
        "stop_id_first": "next_first_stop_id",
        "stop_name_first": "next_first_stop_name",
        "stop_lat_first": "next_first_stop_lat",
        "stop_lon_first": "next_first_stop_lon",
        "vehicle_id_last": "old_vehicle_id",
        "vehicle_id_first": "next_vehicle_id",
        "route_short_name_last": "old_route_short_name",
        "route_short_name_first": "next_route_short_name",
        "route_id_last": "old_route_id",
        "route_id_first": "next_route_id",
        "route_type_last": "old_route_type",
        "route_type_first": "next_route_type",
        "direction_id_last": "old_direction_id",
        "direction_id_first": "next_direction_id",
    })

    output_columns = [
        "h5_activity",
        "distance_based_activity",
        "h5_matches_distance_rule",
        "old_trip_id",
        "next_trip_id",
        "old_route_short_name",
        "next_route_short_name",
        "route_short_name_changed",
        "old_direction_id",
        "next_direction_id",
        "time_gap_minutes",
        "old_last_stop_sequence",
        "next_first_stop_sequence",
        "old_last_stop_name",
        "next_first_stop_name",
        "old_last_stop_lat",
        "old_last_stop_lon",
        "transition_distance_m",
        "transition_distance_km",
    ]
    return rows[output_columns].sort_values(
        ["transition_distance_m", "old_trip_id", "next_trip_id"]
    ).reset_index(drop=True)


def build_summary(transitions: pd.DataFrame) -> pd.DataFrame:
    if transitions.empty:
        return pd.DataFrame([{
            "activity": "overall",
            "transitions": 0,
            "min_distance_m": pd.NA,
            "median_distance_m": pd.NA,
            "mean_distance_m": pd.NA,
            "max_distance_m": pd.NA,
        }])

    activity_column = (
        "h5_activity"
        if "h5_activity" in transitions.columns
        else "derived_transition_activity"
    )
    summary = transitions.groupby(activity_column).agg(
        transitions=("transition_distance_m", "size"),
        min_distance_m=("transition_distance_m", "min"),
        median_distance_m=("transition_distance_m", "median"),
        mean_distance_m=("transition_distance_m", "mean"),
        max_distance_m=("transition_distance_m", "max"),
    ).reset_index()
    summary = summary.rename(columns={activity_column: "activity"})

    overall = pd.DataFrame([{
        "activity": "overall",
        "transitions": len(transitions),
        "min_distance_m": transitions["transition_distance_m"].min(),
        "median_distance_m": transitions["transition_distance_m"].median(),
        "mean_distance_m": transitions["transition_distance_m"].mean(),
        "max_distance_m": transitions["transition_distance_m"].max(),
    }])
    return pd.concat([summary, overall], ignore_index=True)


def main() -> None:
    event_log = pd.read_csv(
        INPUT_CSV,
        dtype={
            "trip_id": "string",
            "old_trip_id": "string",
            "vehicle_id": "string",
            "route_short_name": "string",
            "route_id": "string",
        },
        low_memory=False,
    )
    boundaries = prepare_trip_boundaries(event_log)
    h5_transitions = build_h5_transition_rows(event_log, boundaries)
    h5_summary = build_summary(h5_transitions)
    h5_route_change_transitions = h5_transitions[
        h5_transitions["route_short_name_changed"]
        & (h5_transitions["transition_distance_m"] > 0)
    ].copy()
    h5_route_change_summary = build_summary(h5_route_change_transitions)
    h5_route_change_nonzero_transitions = h5_route_change_transitions.copy()
    h5_route_change_nonzero_summary = build_summary(
        h5_route_change_nonzero_transitions
    )
    consecutive_transitions = build_transition_rows(boundaries)
    consecutive_summary = build_summary(consecutive_transitions)

    h5_transitions.to_csv(OUTPUT_CSV, index=False)
    h5_summary.to_csv(SUMMARY_CSV, index=False)
    h5_route_change_transitions.to_csv(ROUTE_CHANGE_OUTPUT_CSV, index=False)
    h5_route_change_summary.to_csv(ROUTE_CHANGE_SUMMARY_CSV, index=False)
    h5_route_change_nonzero_transitions.to_csv(
        ROUTE_CHANGE_NONZERO_OUTPUT_CSV, index=False
    )
    h5_route_change_nonzero_summary.to_csv(
        ROUTE_CHANGE_NONZERO_SUMMARY_CSV, index=False
    )
    consecutive_transitions.to_csv(CONSECUTIVE_OUTPUT_CSV, index=False)
    consecutive_summary.to_csv(CONSECUTIVE_SUMMARY_CSV, index=False)

    print(f"Trip boundaries: {len(boundaries):,}")
    print(f"H5 transitions: {len(h5_transitions):,}")
    print(h5_summary.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print()
    print(f"H5 route-short-name changes: {len(h5_route_change_transitions):,}")
    print(h5_route_change_summary.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print()
    print(
        "H5 route-short-name changes with distance > 0 m: "
        f"{len(h5_route_change_nonzero_transitions):,}"
    )
    print(h5_route_change_nonzero_summary.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print()
    print(f"Consecutive trip transitions: {len(consecutive_transitions):,}")
    print(consecutive_summary.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print(f"Written: {OUTPUT_CSV}")
    print(f"Written: {SUMMARY_CSV}")
    print(f"Written: {ROUTE_CHANGE_OUTPUT_CSV}")
    print(f"Written: {ROUTE_CHANGE_SUMMARY_CSV}")
    print(f"Written: {ROUTE_CHANGE_NONZERO_OUTPUT_CSV}")
    print(f"Written: {ROUTE_CHANGE_NONZERO_SUMMARY_CSV}")
    print(f"Written: {CONSECUTIVE_OUTPUT_CSV}")
    print(f"Written: {CONSECUTIVE_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
