##Emission calculation

from pathlib import Path
import os

import pandas as pd
from tabulate import tabulate

MPL_CACHE_DIR = Path(".matplotlib-cache")
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR.resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CACHE_DIR.resolve()))

import seaborn as sns
import matplotlib.pyplot as plt


DATASET_DATE = "2026_05_04"
OUTPUT_TAG = f"{DATASET_DATE}_high_spread_original"
GTFS_INPUT_DIR = Path(f"GTFS-OTRAF-{DATASET_DATE.replace('_', '-')}")
EMISSION_OUTPUT_DIR = Path(
    "thesis-main/figures/Emission_Experiment_2026_05_04_high_spread_original"
)
EMISSION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data.
aux_event_log_overview = pd.read_csv(
    f"aux_event_log_overview_emission_improved_high_spread_original_{DATASET_DATE}.csv",
    dtype={
        "vehicle_id": "string",
        "trip_id": "string",
        "trip_id_org": "string",
        "stop_sequence": "Int64"
    }
)

trip_speed_summary = pd.read_csv(
    f"trip_speed_summary_{DATASET_DATE}.csv",
    dtype={"trip_id": "string"}
)

stop_times = pd.read_csv(
    GTFS_INPUT_DIR / "stop_times.txt",
    dtype={
        "trip_id": "string",
        "arrival_time": "string",
        "departure_time": "string",
        "stop_sequence": "Int64",
    },
)

# Normalize trip ID column names when the speed summary uses trip_id for the original ID.
trip_speed_summary = trip_speed_summary.rename(columns={"trip_id": "trip_id_org"})

# Route-type-specific emission assumptions.
# Electric formula:
# distance_km / 100 * consumption_per_100km * emission_factor
# Biogas bus formula:
# distance_km * emission_factor_vol * (consumption_kg_per_100km / 100) / density
# Replace these assumptions with your final sources.
ROUTE_TYPE_FACTORS = {
    "100": {
        "vehicle_type": "Electric railway",
        "energy_source": "electricity",
        "consumption_per_100km": 400.0,
        "consumption_unit": "kWh/100km",
        "emission_factor": 0.00767,
        "emission_factor_unit": "kg CO2e/kWh",
    },
    "700": {
        "vehicle_type": "Biogas bus",
        "energy_source": "biogas",
        "consumption_per_100km": 45.0,
        "consumption_unit": "kg biogas/100km",
        "emission_factor": 0.304,
        "emission_factor_unit": "kg CO2e/m3 biogas",
        "biogas_density_kg_per_m3": 1.2,
    },
    "900": {
        "vehicle_type": "Electric tram",
        "energy_source": "electricity",
        "consumption_per_100km": 296.0,
        "consumption_unit": "kWh/100km",
        "emission_factor": 0.00767,
        "emission_factor_unit": "kg CO2e/kWh",
    },
}

BIOGAS_ROUTE_TYPE = "700"
BIOGAS_FACTOR = ROUTE_TYPE_FACTORS[BIOGAS_ROUTE_TYPE]
BIOGAS_CONSUMPTION_PER_100KM = BIOGAS_FACTOR["consumption_per_100km"]
BIOGAS_EMISSION_FACTOR = BIOGAS_FACTOR["emission_factor"]
BIOGAS_DENSITY_KG_PER_M3 = BIOGAS_FACTOR["biogas_density_kg_per_m3"]
SINGLE_FACTOR_CONSUMPTION_PER_100KM = 100.0
SINGLE_FACTOR_EMISSION_FACTOR = 1.0
SINGLE_FACTOR_EMISSION_PER_KM = (
    SINGLE_FACTOR_CONSUMPTION_PER_100KM / 100 *
    SINGLE_FACTOR_EMISSION_FACTOR
)

OCCUPANCY_LEVEL_PASSENGERS = {
    "100": {
        0: 0,
        1: 58,
        2: 174,
        3: 232,
    },
    "700": {
        0: 0,
        1: 8,
        2: 30,
        3: 34,
    },
    "900": {
        0: 0,
        1: 16,
        2: 48,
        3: 64,
    },
}

PASSENGER_KM_EMISSION_FACTORS = {
    "100": {
        "passenger_km_factor_source_category": "Regional railway",
        "passenger_km_emission_factor_g_per_pkm": 12.9,
    },
    "700": {
        "passenger_km_factor_source_category": "Local bus",
        "passenger_km_emission_factor_g_per_pkm": 40.0,
    },
    "900": {
        "passenger_km_factor_source_category": "Tram/metro",
        "passenger_km_emission_factor_g_per_pkm": 28.61,
    },
}


def normalize_route_type(route_type):
    if pd.isna(route_type):
        return pd.NA
    try:
        return str(int(float(route_type)))
    except (TypeError, ValueError):
        return str(route_type)


def gtfs_time_to_seconds(value):
    if pd.isna(value):
        return pd.NA
    parts = str(value).split(":")
    if len(parts) != 3:
        return pd.NA
    try:
        hours, minutes, seconds = [int(part) for part in parts]
    except ValueError:
        return pd.NA
    return hours * 3600 + minutes * 60 + seconds


def add_route_type_factors(df):
    factor_df = (
        pd.DataFrame.from_dict(ROUTE_TYPE_FACTORS, orient="index")
        .reset_index(names="route_type")
    )
    return df.merge(factor_df, on="route_type", how="left")


def add_passenger_km_emission_factors(df):
    factor_df = (
        pd.DataFrame.from_dict(PASSENGER_KM_EMISSION_FACTORS, orient="index")
        .reset_index(names="route_type")
    )
    return df.merge(factor_df, on="route_type", how="left")


def get_assumed_passengers(route_type, occupancy_status):
    if pd.isna(route_type) or pd.isna(occupancy_status):
        return pd.NA
    try:
        occupancy_level = int(float(occupancy_status))
    except (TypeError, ValueError):
        return pd.NA
    return OCCUPANCY_LEVEL_PASSENGERS.get(str(route_type), {}).get(occupancy_level, pd.NA)


def divide_by_positive(numerator, denominator):
    return numerator.div(denominator.where(denominator > 0))


def calculate_vehicle_emission(df, consumption_column):
    emission = df[consumption_column] * df["emission_factor"]
    if "biogas_density_kg_per_m3" not in df.columns:
        return emission

    density = pd.to_numeric(df["biogas_density_kg_per_m3"], errors="coerce")
    biogas_mask = density.notna()
    emission.loc[biogas_mask] = (
        df.loc[biogas_mask, consumption_column]
        / density.loc[biogas_mask]
        * df.loc[biogas_mask, "emission_factor"]
    )
    return emission


def positive_z_score(series):
    numeric_series = pd.to_numeric(series, errors="coerce")
    std = numeric_series.std()
    if pd.isna(std) or std == 0:
        return pd.Series(0, index=numeric_series.index, dtype="float64")
    return ((numeric_series - numeric_series.mean()) / std).clip(lower=0)


def latex_escape(value):
    if pd.isna(value):
        return ""
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

# Clean data types.
aux_event_log_overview["timestamp"] = pd.to_datetime(
    aux_event_log_overview["timestamp"],
    errors="coerce"
)

aux_event_log_overview["shape_dist_traveled"] = pd.to_numeric(
    aux_event_log_overview["shape_dist_traveled"],
    errors="coerce"
)

aux_event_log_overview["route_type"] = aux_event_log_overview["route_type"].apply(normalize_route_type)

aux_event_log_overview["occupancy_status"] = pd.to_numeric(
    aux_event_log_overview["occupancy_status"],
    errors="coerce"
).astype("Int64")

aux_event_log_overview["stop_sequence"] = pd.to_numeric(
    aux_event_log_overview["stop_sequence"],
    errors="coerce"
).astype("Int64")

trip_speed_summary["avg_speed"] = pd.to_numeric(
    trip_speed_summary["avg_speed"],
    errors="coerce"
)

trip_speed_summary["median_speed"] = pd.to_numeric(
    trip_speed_summary["median_speed"],
    errors="coerce"
)

stop_times["stop_sequence"] = pd.to_numeric(
    stop_times["stop_sequence"],
    errors="coerce",
).astype("Int64")

stop_times["arrival_time_sec"] = pd.to_numeric(
    stop_times["arrival_time"].apply(gtfs_time_to_seconds),
    errors="coerce",
)
stop_times["departure_time_sec"] = pd.to_numeric(
    stop_times["departure_time"].apply(gtfs_time_to_seconds),
    errors="coerce",
)

scheduled_departures = stop_times[[
    "trip_id",
    "stop_sequence",
    "departure_time",
    "departure_time_sec",
]].rename(columns={
    "trip_id": "scheduled_trip_id",
    "stop_sequence": "scheduled_departure_stop_sequence",
    "departure_time": "scheduled_departure_time",
    "departure_time_sec": "scheduled_departure_time_sec",
})

scheduled_arrivals = stop_times[[
    "trip_id",
    "stop_sequence",
    "arrival_time",
    "arrival_time_sec",
]].rename(columns={
    "trip_id": "scheduled_trip_id",
    "stop_sequence": "scheduled_arrive_stop_sequence",
    "arrival_time": "scheduled_arrive_time",
    "arrival_time_sec": "scheduled_arrive_time_sec",
})

# Keep only relevant events.
segment_df = aux_event_log_overview[
    aux_event_log_overview["activity_type"].isin(["departure_stop", "arrive_stop"])
].copy()

segment_df = segment_df.dropna(
    subset=["trip_id", "trip_id_org", "timestamp", "shape_dist_traveled"]
).sort_values(["trip_id", "timestamp"]).reset_index(drop=True)

# Determine the next event within the same trip.
segment_df["next_activity_type"] = segment_df.groupby("trip_id")["activity_type"].shift(-1)
segment_df["next_timestamp"] = segment_df.groupby("trip_id")["timestamp"].shift(-1)
segment_df["next_shape_dist_traveled"] = segment_df.groupby("trip_id")["shape_dist_traveled"].shift(-1)
segment_df["next_stop_sequence"] = segment_df.groupby("trip_id")["stop_sequence"].shift(-1)
segment_df["next_stop_name"] = segment_df.groupby("trip_id")["stop_name"].shift(-1)
segment_df["next_stop_id"] = segment_df.groupby("trip_id")["stop_id"].shift(-1)

# Keep only departure_stop events directly followed by an arrive_stop event.
segments = segment_df[
    (segment_df["activity_type"] == "departure_stop") &
    (segment_df["next_activity_type"] == "arrive_stop")
].copy()

# Calculate distance and duration.
segments["distance_diff"] = (
    segments["next_shape_dist_traveled"] - segments["shape_dist_traveled"]
)

segments["duration_sec"] = (
    segments["next_timestamp"] - segments["timestamp"]
).dt.total_seconds()

segments["duration_min"] = segments["duration_sec"] / 60

# Keep only plausible values.
segments = segments[
    (segments["distance_diff"] >= 0) &
    (segments["duration_sec"] > 0)
].copy()

# Build segment ID from trip_id and departure stop sequence.
segments["segment_id"] = (
    segments["trip_id"].astype(str) + "_" +
    segments["stop_sequence"].astype("Int64").astype(str)
)

# Build helper table.
trip_segment_summary = segments[[
    "segment_id",
    "trip_id",
    "trip_id_org",
    "timestamp",
    "next_timestamp",
    "stop_sequence",
    "stop_id",
    "stop_name",
    "next_stop_sequence",
    "next_stop_id",
    "next_stop_name",
    "shape_dist_traveled",
    "next_shape_dist_traveled",
    "distance_diff",
    "duration_sec",
    "duration_min",
    "route_id",
    "route_short_name",
    "route_type",
    "vehicle_id",
    "occupancy_status"
]].rename(columns={
    "timestamp": "departure_timestamp",
    "next_timestamp": "arrive_timestamp",
    "stop_sequence": "departure_stop_sequence",
    "stop_id": "departure_stop_id",
    "stop_name": "departure_stop_name",
    "next_stop_sequence": "arrive_stop_sequence",
    "next_stop_id": "arrive_stop_id",
    "next_stop_name": "arrive_stop_name",
    "shape_dist_traveled": "departure_shape_dist_traveled",
    "next_shape_dist_traveled": "arrive_shape_dist_traveled"
})

trip_segment_summary["departure_stop_sequence"] = pd.to_numeric(
    trip_segment_summary["departure_stop_sequence"],
    errors="coerce",
).astype("Int64")

trip_segment_summary["arrive_stop_sequence"] = pd.to_numeric(
    trip_segment_summary["arrive_stop_sequence"],
    errors="coerce",
).astype("Int64")

trip_segment_summary = trip_segment_summary.merge(
    scheduled_departures,
    left_on=["trip_id_org", "departure_stop_sequence"],
    right_on=["scheduled_trip_id", "scheduled_departure_stop_sequence"],
    how="left",
).drop(columns=["scheduled_trip_id", "scheduled_departure_stop_sequence"])

trip_segment_summary = trip_segment_summary.merge(
    scheduled_arrivals,
    left_on=["trip_id_org", "arrive_stop_sequence"],
    right_on=["scheduled_trip_id", "scheduled_arrive_stop_sequence"],
    how="left",
).drop(columns=["scheduled_trip_id", "scheduled_arrive_stop_sequence"])

trip_segment_summary["duration_expected_sec"] = (
    trip_segment_summary["scheduled_arrive_time_sec"] -
    trip_segment_summary["scheduled_departure_time_sec"]
)
negative_scheduled_duration = trip_segment_summary["duration_expected_sec"] < 0
trip_segment_summary.loc[negative_scheduled_duration, "duration_expected_sec"] = (
    trip_segment_summary.loc[negative_scheduled_duration, "duration_expected_sec"] +
    24 * 3600
)
trip_segment_summary.loc[
    trip_segment_summary["duration_expected_sec"] <= 0,
    "duration_expected_sec",
] = pd.NA

# Join on trip_id_org.
trip_segment_summary = trip_segment_summary.merge(
    trip_speed_summary[["trip_id_org", "avg_speed", "median_speed"]],
    on="trip_id_org",
    how="left"
)

# Actual distance in kilometres.
trip_segment_summary["distance_diff_km"] = (
    trip_segment_summary["distance_diff"] / 1000
)

trip_segment_summary = add_route_type_factors(trip_segment_summary)
trip_segment_summary = add_passenger_km_emission_factors(trip_segment_summary)

trip_segment_summary["assumed_passengers"] = trip_segment_summary.apply(
    lambda row: get_assumed_passengers(row["route_type"], row["occupancy_status"]),
    axis=1,
)

trip_segment_summary["assumed_passengers"] = pd.to_numeric(
    trip_segment_summary["assumed_passengers"],
    errors="coerce"
)

# Estimated distance from speed and time.
trip_segment_summary["distance_from_avg_speed"] = (
    trip_segment_summary["avg_speed"] * trip_segment_summary["duration_sec"]
)

trip_segment_summary["distance_from_median_speed"] = (
    trip_segment_summary["median_speed"] * trip_segment_summary["duration_sec"]
)

trip_segment_summary["distance_from_avg_speed_km"] = (
    trip_segment_summary["distance_from_avg_speed"] / 1000
)

trip_segment_summary["distance_from_median_speed_km"] = (
    trip_segment_summary["distance_from_median_speed"] / 1000
)

# Difference between actual and estimated distance.
trip_segment_summary["distance_error_avg_speed"] = (
    trip_segment_summary["distance_diff"] - trip_segment_summary["distance_from_avg_speed"]
)

trip_segment_summary["distance_error_median_speed"] = (
    trip_segment_summary["distance_diff"] - trip_segment_summary["distance_from_median_speed"]
)

trip_segment_summary["distance_error_avg_speed_km"] = (
    trip_segment_summary["distance_error_avg_speed"] / 1000
)

trip_segment_summary["distance_error_median_speed_km"] = (
    trip_segment_summary["distance_error_median_speed"] / 1000
)

# Segment-level consumption.
trip_segment_summary["consumption_actual"] = (
    trip_segment_summary["distance_diff_km"] / 100 * trip_segment_summary["consumption_per_100km"]
)

trip_segment_summary["consumption_avg_speed"] = (
    trip_segment_summary["distance_from_avg_speed_km"] / 100 * trip_segment_summary["consumption_per_100km"]
)

trip_segment_summary["consumption_median_speed"] = (
    trip_segment_summary["distance_from_median_speed_km"] / 100 * trip_segment_summary["consumption_per_100km"]
)

# Consumption differences.
trip_segment_summary["consumption_error_avg_speed"] = (
    trip_segment_summary["consumption_actual"] -
    trip_segment_summary["consumption_avg_speed"]
)

trip_segment_summary["consumption_error_median_speed"] = (
    trip_segment_summary["consumption_actual"] -
    trip_segment_summary["consumption_median_speed"]
)

# Segment-level emissions.
trip_segment_summary["emission_actual"] = calculate_vehicle_emission(
    trip_segment_summary,
    "consumption_actual",
)

trip_segment_summary["emission_avg_speed"] = calculate_vehicle_emission(
    trip_segment_summary,
    "consumption_avg_speed",
)

trip_segment_summary["emission_median_speed"] = calculate_vehicle_emission(
    trip_segment_summary,
    "consumption_median_speed",
)

CONSTANT_SPEED_KMH_VALUES = [20, 40, 60]

for speed_kmh in CONSTANT_SPEED_KMH_VALUES:
    speed_ms = speed_kmh / 3.6
    distance_column = f"distance_from_{speed_kmh}_kmh_km"
    consumption_column = f"consumption_{speed_kmh}_kmh"
    emission_column = f"emission_{speed_kmh}_kmh"

    trip_segment_summary[distance_column] = (
        speed_ms * trip_segment_summary["duration_sec"] / 1000
    )
    trip_segment_summary[consumption_column] = (
        trip_segment_summary[distance_column] / 100 *
        trip_segment_summary["consumption_per_100km"]
    )
    trip_segment_summary[emission_column] = calculate_vehicle_emission(
        trip_segment_summary,
        consumption_column,
    )

trip_segment_summary["duration_rule_20_40_speed_kmh"] = 40
trip_segment_summary.loc[
    trip_segment_summary["duration_min"] < 5,
    "duration_rule_20_40_speed_kmh",
] = 20

trip_segment_summary["distance_from_duration_rule_20_40_km"] = (
    trip_segment_summary["duration_rule_20_40_speed_kmh"] / 3.6 *
    trip_segment_summary["duration_sec"] / 1000
)
trip_segment_summary["consumption_duration_rule_20_40"] = (
    trip_segment_summary["distance_from_duration_rule_20_40_km"] / 100 *
    trip_segment_summary["consumption_per_100km"]
)
trip_segment_summary["emission_duration_rule_20_40"] = calculate_vehicle_emission(
    trip_segment_summary,
    "consumption_duration_rule_20_40",
)

# Emission differences.
trip_segment_summary["emission_error_avg_speed"] = (
    trip_segment_summary["emission_actual"] -
    trip_segment_summary["emission_avg_speed"]
)

trip_segment_summary["emission_error_median_speed"] = (
    trip_segment_summary["emission_actual"] -
    trip_segment_summary["emission_median_speed"]
)

# Occupancy-based passenger emission indicators.
# The vehicle emission stays unchanged; these values allocate it over assumed
# passengers and passenger-kilometres.
trip_segment_summary["passenger_km"] = (
    trip_segment_summary["distance_diff_km"] * trip_segment_summary["assumed_passengers"]
)

trip_segment_summary["passenger_km_avg_speed"] = (
    trip_segment_summary["distance_from_avg_speed_km"] * trip_segment_summary["assumed_passengers"]
)

trip_segment_summary["passenger_km_median_speed"] = (
    trip_segment_summary["distance_from_median_speed_km"] * trip_segment_summary["assumed_passengers"]
)

trip_segment_summary["emission_per_passenger_actual"] = divide_by_positive(
    trip_segment_summary["emission_actual"],
    trip_segment_summary["assumed_passengers"]
)

trip_segment_summary["emission_per_passenger_avg_speed"] = divide_by_positive(
    trip_segment_summary["emission_avg_speed"],
    trip_segment_summary["assumed_passengers"]
)

trip_segment_summary["emission_per_passenger_median_speed"] = divide_by_positive(
    trip_segment_summary["emission_median_speed"],
    trip_segment_summary["assumed_passengers"]
)

trip_segment_summary["emission_per_passenger_km_actual"] = divide_by_positive(
    trip_segment_summary["emission_actual"],
    trip_segment_summary["passenger_km"]
)

trip_segment_summary["emission_per_passenger_km_avg_speed"] = divide_by_positive(
    trip_segment_summary["emission_avg_speed"],
    trip_segment_summary["passenger_km_avg_speed"]
)

trip_segment_summary["emission_per_passenger_km_median_speed"] = divide_by_positive(
    trip_segment_summary["emission_median_speed"],
    trip_segment_summary["passenger_km_median_speed"]
)

# Person-kilometre based reference method:
# emissions [kg CO2e] = passenger_km * factor [g CO2e/Pkm] / 1000
trip_segment_summary["emission_passenger_km_factor_actual"] = (
    trip_segment_summary["passenger_km"] *
    trip_segment_summary["passenger_km_emission_factor_g_per_pkm"] /
    1000
)

trip_segment_summary["emission_passenger_km_factor_avg_speed"] = (
    trip_segment_summary["passenger_km_avg_speed"] *
    trip_segment_summary["passenger_km_emission_factor_g_per_pkm"] /
    1000
)

trip_segment_summary["emission_passenger_km_factor_median_speed"] = (
    trip_segment_summary["passenger_km_median_speed"] *
    trip_segment_summary["passenger_km_emission_factor_g_per_pkm"] /
    1000
)

trip_segment_summary["emission_passenger_km_factor_diff_actual"] = (
    trip_segment_summary["emission_actual"] -
    trip_segment_summary["emission_passenger_km_factor_actual"]
)

trip_segment_summary["emission_residual_ratio"] = divide_by_positive(
    trip_segment_summary["emission_avg_speed"] - trip_segment_summary["emission_actual"],
    trip_segment_summary["emission_actual"]
)

trip_segment_summary["occupancy_factor"] = (
    1 - pd.to_numeric(
        trip_segment_summary["occupancy_status"],
        errors="coerce",
    ).fillna(0) / 10
)

trip_segment_summary["occupancy_aware_emission"] = (
    trip_segment_summary["emission_actual"] *
    trip_segment_summary["occupancy_factor"]
)

trip_segment_summary["duration_residual_ratio"] = divide_by_positive(
    trip_segment_summary["duration_sec"] - trip_segment_summary["duration_expected_sec"],
    trip_segment_summary["duration_expected_sec"]
)

trip_segment_summary["duration_penalty"] = divide_by_positive(
    trip_segment_summary["duration_sec"],
    trip_segment_summary["duration_expected_sec"]
).clip(lower=1)
trip_segment_summary.loc[
    trip_segment_summary["route_type"].astype(str) != "700",
    "duration_penalty",
] = 1.0

trip_segment_summary["delay_aware_emission_score"] = (
    trip_segment_summary["occupancy_aware_emission"] *
    trip_segment_summary["duration_penalty"]
)

trip_segment_summary["score_abs_emission_component"] = positive_z_score(
    trip_segment_summary["emission_actual"]
)

trip_segment_summary["score_duration_residual_component"] = positive_z_score(
    trip_segment_summary["duration_residual_ratio"]
)

trip_segment_summary["segment_anomaly_score"] = (
    trip_segment_summary["score_abs_emission_component"] +
    trip_segment_summary["score_duration_residual_component"]
)

trip_segment_summary["emission_combined_actual"] = (
    trip_segment_summary["emission_actual"] +
    trip_segment_summary["emission_passenger_km_factor_actual"]
)

trip_segment_summary["emission_combined_avg_speed"] = (
    trip_segment_summary["emission_avg_speed"] +
    trip_segment_summary["emission_passenger_km_factor_avg_speed"]
)

trip_segment_summary["emission_combined_median_speed"] = (
    trip_segment_summary["emission_median_speed"] +
    trip_segment_summary["emission_passenger_km_factor_median_speed"]
)

trip_segment_summary["vehicle_km_emission_share_actual"] = (
    divide_by_positive(
        trip_segment_summary["emission_actual"],
        trip_segment_summary["emission_combined_actual"]
    ) * 100
)

trip_segment_summary["passenger_km_emission_share_actual"] = (
    divide_by_positive(
        trip_segment_summary["emission_passenger_km_factor_actual"],
        trip_segment_summary["emission_combined_actual"]
    ) * 100
)

trip_segment_summary["vehicle_km_emission_share_avg_speed"] = (
    divide_by_positive(
        trip_segment_summary["emission_avg_speed"],
        trip_segment_summary["emission_combined_avg_speed"]
    ) * 100
)

trip_segment_summary["passenger_km_emission_share_avg_speed"] = (
    divide_by_positive(
        trip_segment_summary["emission_passenger_km_factor_avg_speed"],
        trip_segment_summary["emission_combined_avg_speed"]
    ) * 100
)

trip_segment_summary["vehicle_km_emission_share_median_speed"] = (
    divide_by_positive(
        trip_segment_summary["emission_median_speed"],
        trip_segment_summary["emission_combined_median_speed"]
    ) * 100
)

trip_segment_summary["passenger_km_emission_share_median_speed"] = (
    divide_by_positive(
        trip_segment_summary["emission_passenger_km_factor_median_speed"],
        trip_segment_summary["emission_combined_median_speed"]
    ) * 100
)

# Trip-level aggregation.
trip_segment_ordered = trip_segment_summary.sort_values(
    ["trip_id", "departure_stop_sequence", "departure_timestamp"]
)

trip_endpoint_summary = (
    trip_segment_ordered.groupby("trip_id", as_index=False, dropna=False)
    .agg(
        trip_start_time=("departure_timestamp", "min"),
        trip_end_time=("arrive_timestamp", "max"),
        trip_from_stop=("departure_stop_name", "first"),
        trip_to_stop=("arrive_stop_name", "last"),
    )
)

trip_consumption_emission_summary = (
    trip_segment_summary.groupby(
        [
            "trip_id",
            "trip_id_org",
            "route_id",
            "route_short_name",
            "route_type",
            "vehicle_type",
            "energy_source",
            "consumption_unit",
            "emission_factor_unit",
            "passenger_km_factor_source_category",
        ],
        as_index=False,
        dropna=False,
    )
    .agg(
        segment_count=("segment_id", "nunique"),
        total_duration_sec=("duration_sec", "sum"),
        total_duration_min=("duration_min", "sum"),
        total_duration_expected_sec=("duration_expected_sec", lambda values: values.sum(min_count=1)),
        total_distance_m=("distance_diff", "sum"),
        total_distance_km=("distance_diff_km", "sum"),

        avg_speed=("avg_speed", "first"),
        median_speed=("median_speed", "first"),
        consumption_per_100km=("consumption_per_100km", "first"),
        emission_factor=("emission_factor", "first"),
        passenger_km_emission_factor_g_per_pkm=("passenger_km_emission_factor_g_per_pkm", "first"),

        total_consumption_actual=("consumption_actual", "sum"),
        total_consumption_avg_speed=("consumption_avg_speed", "sum"),
        total_consumption_median_speed=("consumption_median_speed", "sum"),

        total_emission_actual=("emission_actual", "sum"),
        total_emission_avg_speed=("emission_avg_speed", "sum"),
        total_emission_median_speed=("emission_median_speed", "sum"),
        total_occupancy_aware_emission=("occupancy_aware_emission", "sum"),
        total_delay_aware_emission_score=("delay_aware_emission_score", "sum"),
        avg_duration_penalty=("duration_penalty", "mean"),
        max_duration_penalty=("duration_penalty", "max"),

        avg_assumed_passengers=("assumed_passengers", "mean"),
        total_passenger_km=("passenger_km", "sum"),
        total_passenger_km_avg_speed=("passenger_km_avg_speed", "sum"),
        total_passenger_km_median_speed=("passenger_km_median_speed", "sum"),
        total_emission_passenger_km_factor_actual=("emission_passenger_km_factor_actual", "sum"),
        total_emission_passenger_km_factor_avg_speed=("emission_passenger_km_factor_avg_speed", "sum"),
        total_emission_passenger_km_factor_median_speed=("emission_passenger_km_factor_median_speed", "sum"),
    )
)

trip_consumption_emission_summary = trip_consumption_emission_summary.merge(
    trip_endpoint_summary,
    on="trip_id",
    how="left",
)

trip_consumption_emission_summary["total_duration_expected_min"] = (
    trip_consumption_emission_summary["total_duration_expected_sec"] / 60
)

trip_consumption_emission_summary["duration_residual_ratio"] = divide_by_positive(
    trip_consumption_emission_summary["total_duration_sec"] -
    trip_consumption_emission_summary["total_duration_expected_sec"],
    trip_consumption_emission_summary["total_duration_expected_sec"],
)

trip_consumption_emission_summary["trip_duration_penalty"] = divide_by_positive(
    trip_consumption_emission_summary["total_duration_sec"],
    trip_consumption_emission_summary["total_duration_expected_sec"],
).clip(lower=1)
trip_consumption_emission_summary.loc[
    trip_consumption_emission_summary["route_type"].astype(str) != "700",
    "trip_duration_penalty",
] = 1.0

trip_consumption_emission_summary["trip_delay_aware_emission_score"] = (
    trip_consumption_emission_summary["total_occupancy_aware_emission"] *
    trip_consumption_emission_summary["trip_duration_penalty"]
)

trip_consumption_emission_summary["score_abs_emission_component"] = positive_z_score(
    trip_consumption_emission_summary["total_emission_actual"]
)

trip_consumption_emission_summary["score_duration_residual_component"] = positive_z_score(
    trip_consumption_emission_summary["duration_residual_ratio"]
)

trip_consumption_emission_summary["trip_anomaly_score"] = (
    trip_consumption_emission_summary["score_abs_emission_component"] +
    trip_consumption_emission_summary["score_duration_residual_component"]
)

trip_consumption_emission_summary["consumption_diff_avg_speed"] = (
    trip_consumption_emission_summary["total_consumption_actual"] -
    trip_consumption_emission_summary["total_consumption_avg_speed"]
)

trip_consumption_emission_summary["consumption_diff_median_speed"] = (
    trip_consumption_emission_summary["total_consumption_actual"] -
    trip_consumption_emission_summary["total_consumption_median_speed"]
)

trip_consumption_emission_summary["emission_diff_avg_speed"] = (
    trip_consumption_emission_summary["total_emission_actual"] -
    trip_consumption_emission_summary["total_emission_avg_speed"]
)

trip_consumption_emission_summary["emission_diff_median_speed"] = (
    trip_consumption_emission_summary["total_emission_actual"] -
    trip_consumption_emission_summary["total_emission_median_speed"]
)

trip_consumption_emission_summary["emission_per_passenger_km_actual"] = divide_by_positive(
    trip_consumption_emission_summary["total_emission_actual"],
    trip_consumption_emission_summary["total_passenger_km"]
)

trip_consumption_emission_summary["emission_per_passenger_km_avg_speed"] = divide_by_positive(
    trip_consumption_emission_summary["total_emission_avg_speed"],
    trip_consumption_emission_summary["total_passenger_km_avg_speed"]
)

trip_consumption_emission_summary["emission_per_passenger_km_median_speed"] = divide_by_positive(
    trip_consumption_emission_summary["total_emission_median_speed"],
    trip_consumption_emission_summary["total_passenger_km_median_speed"]
)

trip_consumption_emission_summary["total_emission_combined_actual"] = (
    trip_consumption_emission_summary["total_emission_actual"] +
    trip_consumption_emission_summary["total_emission_passenger_km_factor_actual"]
)

trip_consumption_emission_summary["total_emission_combined_avg_speed"] = (
    trip_consumption_emission_summary["total_emission_avg_speed"] +
    trip_consumption_emission_summary["total_emission_passenger_km_factor_avg_speed"]
)

trip_consumption_emission_summary["total_emission_combined_median_speed"] = (
    trip_consumption_emission_summary["total_emission_median_speed"] +
    trip_consumption_emission_summary["total_emission_passenger_km_factor_median_speed"]
)

trip_consumption_emission_summary["vehicle_km_emission_share_actual"] = (
    divide_by_positive(
        trip_consumption_emission_summary["total_emission_actual"],
        trip_consumption_emission_summary["total_emission_combined_actual"]
    ) * 100
)

trip_consumption_emission_summary["passenger_km_emission_share_actual"] = (
    divide_by_positive(
        trip_consumption_emission_summary["total_emission_passenger_km_factor_actual"],
        trip_consumption_emission_summary["total_emission_combined_actual"]
    ) * 100
)

trip_consumption_emission_summary["vehicle_km_emission_share_avg_speed"] = (
    divide_by_positive(
        trip_consumption_emission_summary["total_emission_avg_speed"],
        trip_consumption_emission_summary["total_emission_combined_avg_speed"]
    ) * 100
)

trip_consumption_emission_summary["passenger_km_emission_share_avg_speed"] = (
    divide_by_positive(
        trip_consumption_emission_summary["total_emission_passenger_km_factor_avg_speed"],
        trip_consumption_emission_summary["total_emission_combined_avg_speed"]
    ) * 100
)

trip_consumption_emission_summary["vehicle_km_emission_share_median_speed"] = (
    divide_by_positive(
        trip_consumption_emission_summary["total_emission_median_speed"],
        trip_consumption_emission_summary["total_emission_combined_median_speed"]
    ) * 100
)

trip_consumption_emission_summary["passenger_km_emission_share_median_speed"] = (
    divide_by_positive(
        trip_consumption_emission_summary["total_emission_passenger_km_factor_median_speed"],
        trip_consumption_emission_summary["total_emission_combined_median_speed"]
    ) * 100
)

trip_biogas_experiment_source = (
    trip_consumption_emission_summary
    .sort_values("total_emission_actual", ascending=False)
    .copy()
)
trip_biogas_experiment_source["biogas_consumption_kg"] = (
    trip_biogas_experiment_source["total_distance_km"] / 100 *
    BIOGAS_CONSUMPTION_PER_100KM
)
trip_biogas_experiment_source["biogas_emission_kg"] = (
    trip_biogas_experiment_source["biogas_consumption_kg"] /
    BIOGAS_DENSITY_KG_PER_M3 *
    BIOGAS_EMISSION_FACTOR
)
trip_biogas_experiment_source["emission_group_kg"] = (
    trip_biogas_experiment_source["total_emission_actual"].round(6)
)
top10_trips_biogas_experiment = (
    trip_biogas_experiment_source
    .groupby(
        ["route_type", "route_short_name", "emission_group_kg"],
        as_index=False,
        dropna=False,
    )
    .agg(
        representative_trip_id=("trip_id", "first"),
        trip_count=("trip_id", "nunique"),
        max_distance_km=("total_distance_km", "max"),
        max_duration_min=("total_duration_min", "max"),
        original_emission_kg=("total_emission_actual", "max"),
        biogas_consumption_kg=("biogas_consumption_kg", "max"),
        biogas_emission_kg=("biogas_emission_kg", "max"),
    )
    .sort_values("original_emission_kg", ascending=False)
    .head(10)
    .reset_index(drop=True)
)
top10_trips_biogas_experiment["biogas_emission_per_km"] = divide_by_positive(
    top10_trips_biogas_experiment["biogas_emission_kg"],
    top10_trips_biogas_experiment["max_distance_km"],
)

top10_performing_trips_biogas_experiment = (
    trip_biogas_experiment_source
    .groupby(
        ["route_type", "route_short_name", "emission_group_kg"],
        as_index=False,
        dropna=False,
    )
    .agg(
        representative_trip_id=("trip_id", "first"),
        trip_count=("trip_id", "nunique"),
        min_distance_km=("total_distance_km", "min"),
        min_duration_min=("total_duration_min", "min"),
        original_emission_kg=("total_emission_actual", "min"),
        biogas_consumption_kg=("biogas_consumption_kg", "min"),
        biogas_emission_kg=("biogas_emission_kg", "min"),
    )
    .sort_values("original_emission_kg", ascending=True)
    .head(10)
    .reset_index(drop=True)
)
top10_performing_trips_biogas_experiment["biogas_emission_per_km"] = divide_by_positive(
    top10_performing_trips_biogas_experiment["biogas_emission_kg"],
    top10_performing_trips_biogas_experiment["min_distance_km"],
)

route_consumption_emission_summary = (
    trip_consumption_emission_summary.groupby(
        [
            "route_id",
            "route_short_name",
            "route_type",
            "vehicle_type",
            "energy_source",
            "consumption_unit",
            "emission_factor_unit",
            "passenger_km_factor_source_category",
        ],
        as_index=False,
        dropna=False,
    )
    .agg(
        trip_count=("trip_id", "nunique"),
        segment_count=("segment_count", "sum"),
        total_duration_sec=("total_duration_sec", "sum"),
        total_duration_min=("total_duration_min", "sum"),
        total_distance_m=("total_distance_m", "sum"),
        total_distance_km=("total_distance_km", "sum"),
        consumption_per_100km=("consumption_per_100km", "first"),
        emission_factor=("emission_factor", "first"),
        passenger_km_emission_factor_g_per_pkm=("passenger_km_emission_factor_g_per_pkm", "first"),
        total_consumption_actual=("total_consumption_actual", "sum"),
        total_consumption_avg_speed=("total_consumption_avg_speed", "sum"),
        total_consumption_median_speed=("total_consumption_median_speed", "sum"),
        total_emission_actual=("total_emission_actual", "sum"),
        total_emission_avg_speed=("total_emission_avg_speed", "sum"),
        total_emission_median_speed=("total_emission_median_speed", "sum"),
        avg_assumed_passengers=("avg_assumed_passengers", "mean"),
        total_passenger_km=("total_passenger_km", "sum"),
        total_passenger_km_avg_speed=("total_passenger_km_avg_speed", "sum"),
        total_passenger_km_median_speed=("total_passenger_km_median_speed", "sum"),
        total_emission_passenger_km_factor_actual=("total_emission_passenger_km_factor_actual", "sum"),
        total_emission_passenger_km_factor_avg_speed=("total_emission_passenger_km_factor_avg_speed", "sum"),
        total_emission_passenger_km_factor_median_speed=("total_emission_passenger_km_factor_median_speed", "sum"),
    )
)

route_consumption_emission_summary["consumption_diff_avg_speed"] = (
    route_consumption_emission_summary["total_consumption_actual"] -
    route_consumption_emission_summary["total_consumption_avg_speed"]
)

route_consumption_emission_summary["consumption_diff_median_speed"] = (
    route_consumption_emission_summary["total_consumption_actual"] -
    route_consumption_emission_summary["total_consumption_median_speed"]
)

route_consumption_emission_summary["emission_diff_avg_speed"] = (
    route_consumption_emission_summary["total_emission_actual"] -
    route_consumption_emission_summary["total_emission_avg_speed"]
)

route_consumption_emission_summary["emission_diff_median_speed"] = (
    route_consumption_emission_summary["total_emission_actual"] -
    route_consumption_emission_summary["total_emission_median_speed"]
)

route_consumption_emission_summary["emission_per_passenger_km_actual"] = divide_by_positive(
    route_consumption_emission_summary["total_emission_actual"],
    route_consumption_emission_summary["total_passenger_km"]
)

route_consumption_emission_summary["emission_per_passenger_km_avg_speed"] = divide_by_positive(
    route_consumption_emission_summary["total_emission_avg_speed"],
    route_consumption_emission_summary["total_passenger_km_avg_speed"]
)

route_consumption_emission_summary["emission_per_passenger_km_median_speed"] = divide_by_positive(
    route_consumption_emission_summary["total_emission_median_speed"],
    route_consumption_emission_summary["total_passenger_km_median_speed"]
)

route_consumption_emission_summary["total_emission_combined_actual"] = (
    route_consumption_emission_summary["total_emission_actual"] +
    route_consumption_emission_summary["total_emission_passenger_km_factor_actual"]
)

route_consumption_emission_summary["total_emission_combined_avg_speed"] = (
    route_consumption_emission_summary["total_emission_avg_speed"] +
    route_consumption_emission_summary["total_emission_passenger_km_factor_avg_speed"]
)

route_consumption_emission_summary["total_emission_combined_median_speed"] = (
    route_consumption_emission_summary["total_emission_median_speed"] +
    route_consumption_emission_summary["total_emission_passenger_km_factor_median_speed"]
)

route_consumption_emission_summary["vehicle_km_emission_share_actual"] = (
    divide_by_positive(
        route_consumption_emission_summary["total_emission_actual"],
        route_consumption_emission_summary["total_emission_combined_actual"]
    ) * 100
)

route_consumption_emission_summary["passenger_km_emission_share_actual"] = (
    divide_by_positive(
        route_consumption_emission_summary["total_emission_passenger_km_factor_actual"],
        route_consumption_emission_summary["total_emission_combined_actual"]
    ) * 100
)

route_consumption_emission_summary["vehicle_km_emission_share_avg_speed"] = (
    divide_by_positive(
        route_consumption_emission_summary["total_emission_avg_speed"],
        route_consumption_emission_summary["total_emission_combined_avg_speed"]
    ) * 100
)

route_consumption_emission_summary["passenger_km_emission_share_avg_speed"] = (
    divide_by_positive(
        route_consumption_emission_summary["total_emission_passenger_km_factor_avg_speed"],
        route_consumption_emission_summary["total_emission_combined_avg_speed"]
    ) * 100
)

route_consumption_emission_summary["vehicle_km_emission_share_median_speed"] = (
    divide_by_positive(
        route_consumption_emission_summary["total_emission_median_speed"],
        route_consumption_emission_summary["total_emission_combined_median_speed"]
    ) * 100
)

route_consumption_emission_summary["passenger_km_emission_share_median_speed"] = (
    divide_by_positive(
        route_consumption_emission_summary["total_emission_passenger_km_factor_median_speed"],
        route_consumption_emission_summary["total_emission_combined_median_speed"]
    ) * 100
)

top_segment_emission_summary = (
    trip_segment_summary[[
        "segment_id",
        "emission_actual",
        "emission_per_passenger_actual",
        "emission_per_passenger_km_actual",
        "emission_passenger_km_factor_actual",
        "emission_passenger_km_factor_diff_actual",
        "emission_combined_actual",
        "occupancy_factor",
        "occupancy_aware_emission",
        "duration_penalty",
        "delay_aware_emission_score",
        "vehicle_km_emission_share_actual",
        "passenger_km_emission_share_actual",
        "consumption_actual",
        "distance_diff_km",
        "passenger_km",
        "duration_min",
        "trip_id",
        "trip_id_org",
        "route_id",
        "route_short_name",
        "vehicle_id",
        "route_type",
        "vehicle_type",
        "energy_source",
        "occupancy_status",
        "assumed_passengers",
        "departure_stop_sequence",
        "departure_stop_id",
        "departure_stop_name",
        "arrive_stop_sequence",
        "arrive_stop_id",
        "arrive_stop_name",
        "consumption_per_100km",
        "consumption_unit",
        "emission_factor",
        "emission_factor_unit",
        "passenger_km_factor_source_category",
        "passenger_km_emission_factor_g_per_pkm",
    ]]
    .sort_values("emission_actual", ascending=False)
    .reset_index(drop=True)
)

segment_biogas_experiment_source = top_segment_emission_summary.copy()
segment_biogas_experiment_source["biogas_consumption_kg"] = (
    segment_biogas_experiment_source["distance_diff_km"] / 100 *
    BIOGAS_CONSUMPTION_PER_100KM
)
segment_biogas_experiment_source["biogas_emission_kg"] = (
    segment_biogas_experiment_source["biogas_consumption_kg"] /
    BIOGAS_DENSITY_KG_PER_M3 *
    BIOGAS_EMISSION_FACTOR
)
segment_biogas_experiment_source["emission_group_kg"] = (
    segment_biogas_experiment_source["emission_actual"].round(6)
)
top10_segments_biogas_experiment = (
    segment_biogas_experiment_source
    .groupby(
        [
            "route_type",
            "route_short_name",
            "departure_stop_name",
            "arrive_stop_name",
            "emission_group_kg",
        ],
        as_index=False,
        dropna=False,
    )
    .agg(
        representative_trip_id=("trip_id", "first"),
        segment_count=("segment_id", "nunique"),
        trip_count=("trip_id", "nunique"),
        max_distance_km=("distance_diff_km", "max"),
        max_duration_min=("duration_min", "max"),
        original_emission_kg=("emission_actual", "max"),
        biogas_consumption_kg=("biogas_consumption_kg", "max"),
        biogas_emission_kg=("biogas_emission_kg", "max"),
    )
    .sort_values("original_emission_kg", ascending=False)
    .head(10)
    .reset_index(drop=True)
)
top10_segments_biogas_experiment["biogas_emission_per_km"] = divide_by_positive(
    top10_segments_biogas_experiment["biogas_emission_kg"],
    top10_segments_biogas_experiment["max_distance_km"],
)

top10_performing_segments_biogas_experiment = (
    segment_biogas_experiment_source
    .groupby(
        [
            "route_type",
            "route_short_name",
            "departure_stop_name",
            "arrive_stop_name",
            "emission_group_kg",
        ],
        as_index=False,
        dropna=False,
    )
    .agg(
        representative_trip_id=("trip_id", "first"),
        segment_count=("segment_id", "nunique"),
        trip_count=("trip_id", "nunique"),
        min_distance_km=("distance_diff_km", "min"),
        min_duration_min=("duration_min", "min"),
        original_emission_kg=("emission_actual", "min"),
        biogas_consumption_kg=("biogas_consumption_kg", "min"),
        biogas_emission_kg=("biogas_emission_kg", "min"),
    )
    .sort_values("original_emission_kg", ascending=True)
    .head(10)
    .reset_index(drop=True)
)
top10_performing_segments_biogas_experiment["biogas_emission_per_km"] = divide_by_positive(
    top10_performing_segments_biogas_experiment["biogas_emission_kg"],
    top10_performing_segments_biogas_experiment["min_distance_km"],
)

segment_mode_experiment_source = top_segment_emission_summary.copy()
segment_mode_experiment_source["emission_group_kg"] = (
    segment_mode_experiment_source["emission_actual"].round(6)
)

top10_grouped_segments_mode_experiment = (
    segment_mode_experiment_source
    .groupby(
        [
            "route_type",
            "vehicle_type",
            "route_short_name",
            "departure_stop_name",
            "arrive_stop_name",
            "emission_group_kg",
        ],
        as_index=False,
        dropna=False,
    )
    .agg(
        representative_trip_id=("trip_id", "first"),
        segment_count=("segment_id", "nunique"),
        trip_count=("trip_id", "nunique"),
        max_distance_km=("distance_diff_km", "max"),
        max_duration_min=("duration_min", "max"),
        emission_kg=("emission_actual", "max"),
    )
    .sort_values("emission_kg", ascending=False)
    .head(10)
    .reset_index(drop=True)
)
top10_grouped_segments_mode_experiment["emission_per_km"] = divide_by_positive(
    top10_grouped_segments_mode_experiment["emission_kg"],
    top10_grouped_segments_mode_experiment["max_distance_km"],
)

top10_performing_grouped_segments_mode_experiment = (
    segment_mode_experiment_source
    .groupby(
        [
            "route_type",
            "vehicle_type",
            "route_short_name",
            "departure_stop_name",
            "arrive_stop_name",
            "emission_group_kg",
        ],
        as_index=False,
        dropna=False,
    )
    .agg(
        representative_trip_id=("trip_id", "first"),
        segment_count=("segment_id", "nunique"),
        trip_count=("trip_id", "nunique"),
        min_distance_km=("distance_diff_km", "min"),
        min_duration_min=("duration_min", "min"),
        emission_kg=("emission_actual", "min"),
    )
    .sort_values("emission_kg", ascending=True)
    .head(10)
    .reset_index(drop=True)
)
top10_performing_grouped_segments_mode_experiment["emission_per_km"] = divide_by_positive(
    top10_performing_grouped_segments_mode_experiment["emission_kg"],
    top10_performing_grouped_segments_mode_experiment["min_distance_km"],
)

trip_mode_experiment_source = trip_consumption_emission_summary.copy()
trip_mode_experiment_source["emission_group_kg"] = (
    trip_mode_experiment_source["total_emission_actual"].round(6)
)

top10_grouped_trips_mode_experiment = (
    trip_mode_experiment_source
    .groupby(
        ["route_type", "vehicle_type", "route_short_name", "emission_group_kg"],
        as_index=False,
        dropna=False,
    )
    .agg(
        representative_trip_id=("trip_id", "first"),
        trip_count=("trip_id", "nunique"),
        max_distance_km=("total_distance_km", "max"),
        max_duration_min=("total_duration_min", "max"),
        emission_kg=("total_emission_actual", "max"),
    )
    .sort_values("emission_kg", ascending=False)
    .head(10)
    .reset_index(drop=True)
)
top10_grouped_trips_mode_experiment["emission_per_km"] = divide_by_positive(
    top10_grouped_trips_mode_experiment["emission_kg"],
    top10_grouped_trips_mode_experiment["max_distance_km"],
)

top10_performing_grouped_trips_mode_experiment = (
    trip_mode_experiment_source
    .groupby(
        ["route_type", "vehicle_type", "route_short_name", "emission_group_kg"],
        as_index=False,
        dropna=False,
    )
    .agg(
        representative_trip_id=("trip_id", "first"),
        trip_count=("trip_id", "nunique"),
        min_distance_km=("total_distance_km", "min"),
        min_duration_min=("total_duration_min", "min"),
        emission_kg=("total_emission_actual", "min"),
    )
    .sort_values("emission_kg", ascending=True)
    .head(10)
    .reset_index(drop=True)
)
top10_performing_grouped_trips_mode_experiment["emission_per_km"] = divide_by_positive(
    top10_performing_grouped_trips_mode_experiment["emission_kg"],
    top10_performing_grouped_trips_mode_experiment["min_distance_km"],
)

single_factor_segment_source = top_segment_emission_summary.copy()
single_factor_segment_source["single_factor_consumption"] = (
    single_factor_segment_source["distance_diff_km"] / 100 *
    SINGLE_FACTOR_CONSUMPTION_PER_100KM
)
single_factor_segment_source["single_factor_emission"] = (
    single_factor_segment_source["single_factor_consumption"] *
    SINGLE_FACTOR_EMISSION_FACTOR
)
single_factor_segment_source["emission_group_kg"] = (
    single_factor_segment_source["single_factor_emission"].round(6)
)

top10_grouped_segments_single_factor_experiment = (
    single_factor_segment_source
    .groupby(
        [
            "route_type",
            "vehicle_type",
            "route_short_name",
            "departure_stop_name",
            "arrive_stop_name",
            "emission_group_kg",
        ],
        as_index=False,
        dropna=False,
    )
    .agg(
        representative_trip_id=("trip_id", "first"),
        segment_count=("segment_id", "nunique"),
        trip_count=("trip_id", "nunique"),
        max_distance_km=("distance_diff_km", "max"),
        max_duration_min=("duration_min", "max"),
        consumption=("single_factor_consumption", "max"),
        emission_kg=("single_factor_emission", "max"),
    )
    .sort_values("emission_kg", ascending=False)
    .head(10)
    .reset_index(drop=True)
)
top10_grouped_segments_single_factor_experiment["emission_per_km"] = divide_by_positive(
    top10_grouped_segments_single_factor_experiment["emission_kg"],
    top10_grouped_segments_single_factor_experiment["max_distance_km"],
)

top10_performing_grouped_segments_single_factor_experiment = (
    single_factor_segment_source
    .groupby(
        [
            "route_type",
            "vehicle_type",
            "route_short_name",
            "departure_stop_name",
            "arrive_stop_name",
            "emission_group_kg",
        ],
        as_index=False,
        dropna=False,
    )
    .agg(
        representative_trip_id=("trip_id", "first"),
        segment_count=("segment_id", "nunique"),
        trip_count=("trip_id", "nunique"),
        min_distance_km=("distance_diff_km", "min"),
        min_duration_min=("duration_min", "min"),
        consumption=("single_factor_consumption", "min"),
        emission_kg=("single_factor_emission", "min"),
    )
    .sort_values("emission_kg", ascending=True)
    .head(10)
    .reset_index(drop=True)
)
top10_performing_grouped_segments_single_factor_experiment["emission_per_km"] = divide_by_positive(
    top10_performing_grouped_segments_single_factor_experiment["emission_kg"],
    top10_performing_grouped_segments_single_factor_experiment["min_distance_km"],
)

single_factor_trip_source = trip_consumption_emission_summary.copy()
single_factor_trip_source["single_factor_consumption"] = (
    single_factor_trip_source["total_distance_km"] / 100 *
    SINGLE_FACTOR_CONSUMPTION_PER_100KM
)
single_factor_trip_source["single_factor_emission"] = (
    single_factor_trip_source["single_factor_consumption"] *
    SINGLE_FACTOR_EMISSION_FACTOR
)
single_factor_trip_source["emission_group_kg"] = (
    single_factor_trip_source["single_factor_emission"].round(6)
)

top10_grouped_trips_single_factor_experiment = (
    single_factor_trip_source
    .groupby(
        ["route_type", "vehicle_type", "route_short_name", "emission_group_kg"],
        as_index=False,
        dropna=False,
    )
    .agg(
        representative_trip_id=("trip_id", "first"),
        trip_count=("trip_id", "nunique"),
        max_distance_km=("total_distance_km", "max"),
        max_duration_min=("total_duration_min", "max"),
        consumption=("single_factor_consumption", "max"),
        emission_kg=("single_factor_emission", "max"),
    )
    .sort_values("emission_kg", ascending=False)
    .head(10)
    .reset_index(drop=True)
)
top10_grouped_trips_single_factor_experiment["emission_per_km"] = divide_by_positive(
    top10_grouped_trips_single_factor_experiment["emission_kg"],
    top10_grouped_trips_single_factor_experiment["max_distance_km"],
)

top10_performing_grouped_trips_single_factor_experiment = (
    single_factor_trip_source
    .groupby(
        ["route_type", "vehicle_type", "route_short_name", "emission_group_kg"],
        as_index=False,
        dropna=False,
    )
    .agg(
        representative_trip_id=("trip_id", "first"),
        trip_count=("trip_id", "nunique"),
        min_distance_km=("total_distance_km", "min"),
        min_duration_min=("total_duration_min", "min"),
        consumption=("single_factor_consumption", "min"),
        emission_kg=("single_factor_emission", "min"),
    )
    .sort_values("emission_kg", ascending=True)
    .head(10)
    .reset_index(drop=True)
)
top10_performing_grouped_trips_single_factor_experiment["emission_per_km"] = divide_by_positive(
    top10_performing_grouped_trips_single_factor_experiment["emission_kg"],
    top10_performing_grouped_trips_single_factor_experiment["min_distance_km"],
)

occupancy_mode_segment_source = top_segment_emission_summary.copy()
occupancy_mode_segment_source["occupancy_factor"] = (
    1 - pd.to_numeric(
        occupancy_mode_segment_source["occupancy_status"],
        errors="coerce",
    ).fillna(0) / 10
)
occupancy_mode_segment_source["occupancy_aware_emission"] = (
    occupancy_mode_segment_source["emission_actual"] *
    occupancy_mode_segment_source["occupancy_factor"]
)
occupancy_mode_segment_source["emission_group_kg"] = (
    occupancy_mode_segment_source["occupancy_aware_emission"].round(6)
)

top10_grouped_segments_occupancy_mode_experiment = (
    occupancy_mode_segment_source
    .groupby(
        [
            "route_type",
            "vehicle_type",
            "route_short_name",
            "departure_stop_name",
            "arrive_stop_name",
            "emission_group_kg",
        ],
        as_index=False,
        dropna=False,
    )
    .agg(
        representative_trip_id=("trip_id", "first"),
        segment_count=("segment_id", "nunique"),
        trip_count=("trip_id", "nunique"),
        max_distance_km=("distance_diff_km", "max"),
        max_duration_min=("duration_min", "max"),
        emission_kg=("occupancy_aware_emission", "max"),
    )
    .sort_values("emission_kg", ascending=False)
    .head(10)
    .reset_index(drop=True)
)
top10_grouped_segments_occupancy_mode_experiment["emission_per_km"] = divide_by_positive(
    top10_grouped_segments_occupancy_mode_experiment["emission_kg"],
    top10_grouped_segments_occupancy_mode_experiment["max_distance_km"],
)

top10_performing_grouped_segments_occupancy_mode_experiment = (
    occupancy_mode_segment_source
    .groupby(
        [
            "route_type",
            "vehicle_type",
            "route_short_name",
            "departure_stop_name",
            "arrive_stop_name",
            "emission_group_kg",
        ],
        as_index=False,
        dropna=False,
    )
    .agg(
        representative_trip_id=("trip_id", "first"),
        segment_count=("segment_id", "nunique"),
        trip_count=("trip_id", "nunique"),
        min_distance_km=("distance_diff_km", "min"),
        min_duration_min=("duration_min", "min"),
        emission_kg=("occupancy_aware_emission", "min"),
    )
    .sort_values("emission_kg", ascending=True)
    .head(10)
    .reset_index(drop=True)
)
top10_performing_grouped_segments_occupancy_mode_experiment["emission_per_km"] = divide_by_positive(
    top10_performing_grouped_segments_occupancy_mode_experiment["emission_kg"],
    top10_performing_grouped_segments_occupancy_mode_experiment["min_distance_km"],
)

occupancy_mode_trip_source = trip_segment_summary.copy()
occupancy_mode_trip_source["occupancy_factor"] = (
    1 - pd.to_numeric(
        occupancy_mode_trip_source["occupancy_status"],
        errors="coerce",
    ).fillna(0) / 10
)
occupancy_mode_trip_source["occupancy_aware_emission"] = (
    occupancy_mode_trip_source["emission_actual"] *
    occupancy_mode_trip_source["occupancy_factor"]
)
occupancy_mode_trip_summary = (
    occupancy_mode_trip_source
    .groupby(
        ["trip_id", "route_type", "vehicle_type", "route_short_name"],
        as_index=False,
        dropna=False,
    )
    .agg(
        total_distance_km=("distance_diff_km", "sum"),
        total_duration_min=("duration_min", "sum"),
        total_occupancy_aware_emission=("occupancy_aware_emission", "sum"),
    )
)
occupancy_mode_trip_summary["emission_group_kg"] = (
    occupancy_mode_trip_summary["total_occupancy_aware_emission"].round(6)
)

top10_grouped_trips_occupancy_mode_experiment = (
    occupancy_mode_trip_summary
    .groupby(
        ["route_type", "vehicle_type", "route_short_name", "emission_group_kg"],
        as_index=False,
        dropna=False,
    )
    .agg(
        representative_trip_id=("trip_id", "first"),
        trip_count=("trip_id", "nunique"),
        max_distance_km=("total_distance_km", "max"),
        max_duration_min=("total_duration_min", "max"),
        emission_kg=("total_occupancy_aware_emission", "max"),
    )
    .sort_values("emission_kg", ascending=False)
    .head(10)
    .reset_index(drop=True)
)
top10_grouped_trips_occupancy_mode_experiment["emission_per_km"] = divide_by_positive(
    top10_grouped_trips_occupancy_mode_experiment["emission_kg"],
    top10_grouped_trips_occupancy_mode_experiment["max_distance_km"],
)

top10_performing_grouped_trips_occupancy_mode_experiment = (
    occupancy_mode_trip_summary
    .groupby(
        ["route_type", "vehicle_type", "route_short_name", "emission_group_kg"],
        as_index=False,
        dropna=False,
    )
    .agg(
        representative_trip_id=("trip_id", "first"),
        trip_count=("trip_id", "nunique"),
        min_distance_km=("total_distance_km", "min"),
        min_duration_min=("total_duration_min", "min"),
        emission_kg=("total_occupancy_aware_emission", "min"),
    )
    .sort_values("emission_kg", ascending=True)
    .head(10)
    .reset_index(drop=True)
)
top10_performing_grouped_trips_occupancy_mode_experiment["emission_per_km"] = divide_by_positive(
    top10_performing_grouped_trips_occupancy_mode_experiment["emission_kg"],
    top10_performing_grouped_trips_occupancy_mode_experiment["min_distance_km"],
)

delay_aware_segment_source = trip_segment_summary.copy()
delay_aware_segment_source["score_group_kg"] = (
    delay_aware_segment_source["delay_aware_emission_score"].round(6)
)

top10_grouped_segments_delay_aware_experiment = (
    delay_aware_segment_source
    .groupby(
        [
            "route_type",
            "vehicle_type",
            "route_short_name",
            "departure_stop_name",
            "arrive_stop_name",
            "score_group_kg",
        ],
        as_index=False,
        dropna=False,
    )
    .agg(
        representative_trip_id=("trip_id", "first"),
        segment_count=("segment_id", "nunique"),
        trip_count=("trip_id", "nunique"),
        max_distance_km=("distance_diff_km", "max"),
        max_duration_min=("duration_min", "max"),
        max_duration_penalty=("duration_penalty", "max"),
        delay_aware_emission_score=("delay_aware_emission_score", "max"),
    )
    .sort_values("delay_aware_emission_score", ascending=False)
    .head(10)
    .reset_index(drop=True)
)
top10_grouped_segments_delay_aware_experiment["delay_aware_score_per_km"] = divide_by_positive(
    top10_grouped_segments_delay_aware_experiment["delay_aware_emission_score"],
    top10_grouped_segments_delay_aware_experiment["max_distance_km"],
)

top100_segments_delay_aware_experiment = (
    delay_aware_segment_source
    .sort_values("delay_aware_emission_score", ascending=False)
    .head(100)
    [[
        "segment_id",
        "trip_id",
        "trip_id_org",
        "departure_timestamp",
        "arrive_timestamp",
        "route_type",
        "vehicle_type",
        "route_short_name",
        "departure_stop_sequence",
        "departure_stop_name",
        "arrive_stop_sequence",
        "arrive_stop_name",
        "distance_diff_km",
        "duration_min",
        "duration_penalty",
        "occupancy_status",
        "occupancy_factor",
        "occupancy_aware_emission",
        "delay_aware_emission_score",
    ]]
    .reset_index(drop=True)
)
top100_segments_delay_aware_experiment["delay_aware_score_per_km"] = divide_by_positive(
    top100_segments_delay_aware_experiment["delay_aware_emission_score"],
    top100_segments_delay_aware_experiment["distance_diff_km"],
)

top10_performing_grouped_segments_delay_aware_experiment = (
    delay_aware_segment_source
    .groupby(
        [
            "route_type",
            "vehicle_type",
            "route_short_name",
            "departure_stop_name",
            "arrive_stop_name",
            "score_group_kg",
        ],
        as_index=False,
        dropna=False,
    )
    .agg(
        representative_trip_id=("trip_id", "first"),
        segment_count=("segment_id", "nunique"),
        trip_count=("trip_id", "nunique"),
        min_distance_km=("distance_diff_km", "min"),
        min_duration_min=("duration_min", "min"),
        min_duration_penalty=("duration_penalty", "min"),
        delay_aware_emission_score=("delay_aware_emission_score", "min"),
    )
    .sort_values("delay_aware_emission_score", ascending=True)
    .head(10)
    .reset_index(drop=True)
)
top10_performing_grouped_segments_delay_aware_experiment["delay_aware_score_per_km"] = divide_by_positive(
    top10_performing_grouped_segments_delay_aware_experiment["delay_aware_emission_score"],
    top10_performing_grouped_segments_delay_aware_experiment["min_distance_km"],
)

delay_aware_trip_source = trip_consumption_emission_summary.copy()
delay_aware_trip_source["score_group_kg"] = (
    delay_aware_trip_source["total_delay_aware_emission_score"].round(6)
)

top10_grouped_trips_delay_aware_experiment = (
    delay_aware_trip_source
    .groupby(
        ["route_type", "vehicle_type", "route_short_name", "score_group_kg"],
        as_index=False,
        dropna=False,
    )
    .agg(
        representative_trip_id=("trip_id", "first"),
        trip_count=("trip_id", "nunique"),
        max_distance_km=("total_distance_km", "max"),
        max_duration_min=("total_duration_min", "max"),
        max_duration_penalty=("max_duration_penalty", "max"),
        delay_aware_emission_score=("total_delay_aware_emission_score", "max"),
    )
    .sort_values("delay_aware_emission_score", ascending=False)
    .head(10)
    .reset_index(drop=True)
)
top10_grouped_trips_delay_aware_experiment["delay_aware_score_per_km"] = divide_by_positive(
    top10_grouped_trips_delay_aware_experiment["delay_aware_emission_score"],
    top10_grouped_trips_delay_aware_experiment["max_distance_km"],
)

top10_performing_grouped_trips_delay_aware_experiment = (
    delay_aware_trip_source
    .groupby(
        ["route_type", "vehicle_type", "route_short_name", "score_group_kg"],
        as_index=False,
        dropna=False,
    )
    .agg(
        representative_trip_id=("trip_id", "first"),
        trip_count=("trip_id", "nunique"),
        min_distance_km=("total_distance_km", "min"),
        min_duration_min=("total_duration_min", "min"),
        min_duration_penalty=("max_duration_penalty", "min"),
        delay_aware_emission_score=("total_delay_aware_emission_score", "min"),
    )
    .sort_values("delay_aware_emission_score", ascending=True)
    .head(10)
    .reset_index(drop=True)
)
top10_performing_grouped_trips_delay_aware_experiment["delay_aware_score_per_km"] = divide_by_positive(
    top10_performing_grouped_trips_delay_aware_experiment["delay_aware_emission_score"],
    top10_performing_grouped_trips_delay_aware_experiment["min_distance_km"],
)

segment_anomaly_score_summary = (
    trip_segment_summary[[
        "segment_id",
        "segment_anomaly_score",
        "score_abs_emission_component",
        "score_duration_residual_component",
        "emission_actual",
        "occupancy_factor",
        "occupancy_aware_emission",
        "duration_sec",
        "duration_min",
        "duration_expected_sec",
        "duration_residual_ratio",
        "duration_penalty",
        "delay_aware_emission_score",
        "distance_diff_km",
        "trip_id",
        "trip_id_org",
        "departure_timestamp",
        "arrive_timestamp",
        "route_id",
        "route_short_name",
        "vehicle_id",
        "route_type",
        "vehicle_type",
        "departure_stop_sequence",
        "departure_stop_id",
        "departure_stop_name",
        "arrive_stop_sequence",
        "arrive_stop_id",
        "arrive_stop_name",
    ]]
    .sort_values("segment_anomaly_score", ascending=False)
    .reset_index(drop=True)
)

trip_segment_summary.to_csv(f"trip_segment_summary_{OUTPUT_TAG}.csv", index=False)
trip_consumption_emission_summary.to_csv(
    f"trip_consumption_emission_summary_{OUTPUT_TAG}.csv",
    index=False
)
route_consumption_emission_summary.to_csv(
    f"route_consumption_emission_summary_{OUTPUT_TAG}.csv",
    index=False
)
top_segment_emission_summary.to_csv(
    f"top_segment_emission_summary_{OUTPUT_TAG}.csv",
    index=False
)
top10_segments_biogas_experiment.to_csv(
    f"top10_segments_biogas_experiment_{OUTPUT_TAG}.csv",
    index=False,
)
top10_performing_segments_biogas_experiment.to_csv(
    f"top10_performing_segments_biogas_experiment_{OUTPUT_TAG}.csv",
    index=False,
)
top10_trips_biogas_experiment.to_csv(
    f"top10_trips_biogas_experiment_{OUTPUT_TAG}.csv",
    index=False,
)
top10_performing_trips_biogas_experiment.to_csv(
    f"top10_performing_trips_biogas_experiment_{OUTPUT_TAG}.csv",
    index=False,
)
top10_grouped_segments_mode_experiment.to_csv(
    f"top10_grouped_segments_mode_experiment_{OUTPUT_TAG}.csv",
    index=False,
)
top10_performing_grouped_segments_mode_experiment.to_csv(
    f"top10_performing_grouped_segments_mode_experiment_{OUTPUT_TAG}.csv",
    index=False,
)
top10_grouped_trips_mode_experiment.to_csv(
    f"top10_grouped_trips_mode_experiment_{OUTPUT_TAG}.csv",
    index=False,
)
top10_performing_grouped_trips_mode_experiment.to_csv(
    f"top10_performing_grouped_trips_mode_experiment_{OUTPUT_TAG}.csv",
    index=False,
)
top10_grouped_segments_single_factor_experiment.to_csv(
    f"top10_grouped_segments_single_factor_experiment_{OUTPUT_TAG}.csv",
    index=False,
)
top10_performing_grouped_segments_single_factor_experiment.to_csv(
    f"top10_performing_grouped_segments_single_factor_experiment_{OUTPUT_TAG}.csv",
    index=False,
)
top10_grouped_trips_single_factor_experiment.to_csv(
    f"top10_grouped_trips_single_factor_experiment_{OUTPUT_TAG}.csv",
    index=False,
)
top10_performing_grouped_trips_single_factor_experiment.to_csv(
    f"top10_performing_grouped_trips_single_factor_experiment_{OUTPUT_TAG}.csv",
    index=False,
)
top10_grouped_segments_occupancy_mode_experiment.to_csv(
    f"top10_grouped_segments_occupancy_mode_experiment_{OUTPUT_TAG}.csv",
    index=False,
)
top10_performing_grouped_segments_occupancy_mode_experiment.to_csv(
    f"top10_performing_grouped_segments_occupancy_mode_experiment_{OUTPUT_TAG}.csv",
    index=False,
)
top10_grouped_trips_occupancy_mode_experiment.to_csv(
    f"top10_grouped_trips_occupancy_mode_experiment_{OUTPUT_TAG}.csv",
    index=False,
)
top10_performing_grouped_trips_occupancy_mode_experiment.to_csv(
    f"top10_performing_grouped_trips_occupancy_mode_experiment_{OUTPUT_TAG}.csv",
    index=False,
)
top10_grouped_segments_delay_aware_experiment.to_csv(
    f"top10_grouped_segments_delay_aware_experiment_{OUTPUT_TAG}.csv",
    index=False,
)
top100_segments_delay_aware_experiment.to_csv(
    f"top100_segments_delay_aware_experiment_{OUTPUT_TAG}.csv",
    index=False,
)
top10_performing_grouped_segments_delay_aware_experiment.to_csv(
    f"top10_performing_grouped_segments_delay_aware_experiment_{OUTPUT_TAG}.csv",
    index=False,
)
top10_grouped_trips_delay_aware_experiment.to_csv(
    f"top10_grouped_trips_delay_aware_experiment_{OUTPUT_TAG}.csv",
    index=False,
)
top10_performing_grouped_trips_delay_aware_experiment.to_csv(
    f"top10_performing_grouped_trips_delay_aware_experiment_{OUTPUT_TAG}.csv",
    index=False,
)
segment_anomaly_score_summary.to_csv(
    f"segment_anomaly_score_summary_{OUTPUT_TAG}.csv",
    index=False
)

trip_anomaly_score_summary = (
    trip_consumption_emission_summary.sort_values(
        "trip_anomaly_score",
        ascending=False,
    )
    .reset_index(drop=True)
)

trip_anomaly_score_summary.to_csv(
    f"trip_anomaly_score_summary_{OUTPUT_TAG}.csv",
    index=False,
)

top_segment_cluster_source = top_segment_emission_summary.copy()
top_segment_cluster_source["distance_group_km"] = (
    top_segment_cluster_source["distance_diff_km"].round(2)
)
top_segment_cluster_source["emission_group_kg"] = (
    top_segment_cluster_source["emission_actual"].round(2)
)

segment_cluster_summary = (
    top_segment_cluster_source.groupby(
        [
            "vehicle_type",
            "route_short_name",
            "departure_stop_name",
            "arrive_stop_name",
            "distance_group_km",
            "emission_group_kg",
        ],
        as_index=False,
        dropna=False,
    )
    .agg(
        trips=("trip_id", lambda values: list(dict.fromkeys(values))),
        segment_count=("segment_id", "nunique"),
        duration_min_min=("duration_min", "min"),
        duration_min_max=("duration_min", "max"),
    )
    .sort_values(
        ["emission_group_kg", "segment_count"],
        ascending=[False, False],
    )
    .reset_index(drop=True)
)

segment_cluster_summary["emission_per_km"] = divide_by_positive(
    segment_cluster_summary["emission_group_kg"],
    segment_cluster_summary["distance_group_km"],
)


def build_segment_cluster_latex_table(cluster_df, caption, label):
    table_rows = []
    for rank, row in cluster_df.iterrows():
        if round(row["duration_min_min"], 2) == round(row["duration_min_max"], 2):
            duration = f"{row['duration_min_min']:.2f}"
        else:
            duration = f"{row['duration_min_min']:.2f}--{row['duration_min_max']:.2f}"

        table_rows.append(
            " & ".join([
                str(rank + 1),
                latex_escape(row["vehicle_type"]),
                latex_escape(row["route_short_name"]),
                str(len(row["trips"])),
                latex_escape(row["departure_stop_name"]),
                latex_escape(row["arrive_stop_name"]),
                f"{row['distance_group_km']:.2f}",
                duration,
                f"{row['emission_group_kg']:.2f}",
                f"{row['emission_per_km']:.3f}",
            ]) + r" \\"
        )

    return rf"""\begin{{table}}[H]
\centering
\scriptsize
\setlength{{\tabcolsep}}{{2pt}}
\caption{{{caption}}}
\label{{{label}}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{r p{{1.7cm}} r r p{{2.7cm}} p{{2.7cm}} r r r r}}
\toprule
Rank & Mode & Route & Trips & From Stop & To Stop &
\begin{{tabular}}{{@{{}}c@{{}}}}Distance\\(km)\end{{tabular}} &
\begin{{tabular}}{{@{{}}c@{{}}}}Duration\\(min)\end{{tabular}} &
\begin{{tabular}}{{@{{}}c@{{}}}}Emission\\(kg CO$_2$e)\end{{tabular}} &
\begin{{tabular}}{{@{{}}c@{{}}}}Emission per km\\(kg CO$_2$e/km)\end{{tabular}} \\
\midrule
{chr(10).join(table_rows)}
\bottomrule
\end{{tabular}}
}}
\end{{table}}
"""


top10_segment_cluster_summary = segment_cluster_summary.head(10).copy()

top10_segment_emission_per_km_summary = (
    segment_cluster_summary.sort_values(
        ["emission_per_km", "emission_group_kg", "segment_count"],
        ascending=[False, False, False],
    )
    .head(10)
    .reset_index(drop=True)
)

bottom10_segment_cluster_summary = (
    segment_cluster_summary.sort_values(
        ["emission_group_kg", "segment_count"],
        ascending=[True, False],
    )
    .head(10)
    .reset_index(drop=True)
)

top10_segment_latex = build_segment_cluster_latex_table(
    top10_segment_cluster_summary,
    "Top 10 segment groups by absolute segment emissions.",
    "tab:top10_segment_emissions",
)

top10_segment_emission_per_km_latex = build_segment_cluster_latex_table(
    top10_segment_emission_per_km_summary,
    "Top 10 segment groups by emissions per travelled kilometre.",
    "tab:top10_segment_emissions_per_km",
)

bottom10_segment_latex = build_segment_cluster_latex_table(
    bottom10_segment_cluster_summary,
    "Bottom 10 segment groups by absolute segment emissions.",
    "tab:bottom10_segment_emissions",
)


def count_duration_occurrences(row, source_df):
    same_segment_type = (
        (source_df["vehicle_type"] == row["vehicle_type"]) &
        (source_df["route_short_name"] == row["route_short_name"]) &
        (source_df["departure_stop_name"] == row["departure_stop_name"]) &
        (source_df["arrive_stop_name"] == row["arrive_stop_name"])
    )
    at_least_as_long = source_df["duration_sec"] >= row["duration_sec"]
    return int((same_segment_type & at_least_as_long).sum())


def format_single_case_time(row, occurrence_count):
    if occurrence_count != 1 or pd.isna(row["departure_timestamp"]):
        return "--"
    return row["departure_timestamp"].strftime("%H:%M:%S")


top10_segment_score_rows = []
for rank, row in segment_anomaly_score_summary.head(10).iterrows():
    duration_occurrences = count_duration_occurrences(row, trip_segment_summary)
    emission_per_km = row["emission_actual"] / row["distance_diff_km"]
    top10_segment_score_rows.append(
        " & ".join([
            str(rank + 1),
            latex_escape(row["vehicle_type"]),
            latex_escape(row["route_short_name"]),
            latex_escape(row["departure_stop_name"]),
            latex_escape(row["arrive_stop_name"]),
            f"{row['distance_diff_km']:.2f}",
            f"{row['duration_min']:.2f}",
            f"{row['emission_actual']:.2f}",
            f"{emission_per_km:.3f}",
            f"{row['segment_anomaly_score']:.2f}",
            str(duration_occurrences),
            latex_escape(format_single_case_time(row, duration_occurrences)),
        ]) + r" \\"
    )

top10_segment_score_latex = r"""\begin{table}[H]
\centering
\scriptsize
\setlength{\tabcolsep}{1.4pt}
\caption{Top 10 segments by combined segment anomaly score.}
\label{tab:top10_segment_anomaly_score}
\resizebox{\textwidth}{!}{%
\begin{tabular}{r p{1.5cm} r p{2.1cm} p{2.1cm} r r r r r r p{1.6cm}}
\toprule
Rank & Mode & Route & From Stop & To Stop &
\begin{tabular}{@{}c@{}}Distance\\(km)\end{tabular} &
\begin{tabular}{@{}c@{}}Duration\\(min)\end{tabular} &
\begin{tabular}{@{}c@{}}Emission\\(kg CO$_2$e)\end{tabular} &
\begin{tabular}{@{}c@{}}Emission per km\\(kg CO$_2$e/km)\end{tabular} &
Score &
Occ. &
\begin{tabular}{@{}c@{}}Single-case\\time\end{tabular} \\
\midrule
""" + "\n".join(top10_segment_score_rows) + r"""
\bottomrule
\end{tabular}
}
\end{table}
"""


def build_top10_biogas_segment_latex_table(
    segment_df,
    caption,
    label,
    distance_column,
    duration_column,
    distance_label,
    duration_label,
):
    table_rows = []
    for rank, row in segment_df.iterrows():
        table_rows.append(
            " & ".join([
                str(rank + 1),
                latex_escape(row["route_type"]),
                latex_escape(row["route_short_name"]),
                latex_escape(row["departure_stop_name"]),
                latex_escape(row["arrive_stop_name"]),
                latex_escape(row["representative_trip_id"]),
                f"{row['segment_count']:.0f}",
                f"{row['trip_count']:.0f}",
                f"{row[distance_column]:.2f}",
                f"{row[duration_column]:.2f}",
                f"{row['original_emission_kg']:.2f}",
                f"{row['biogas_consumption_kg']:.2f}",
                f"{row['biogas_emission_kg']:.2f}",
                f"{row['biogas_emission_per_km']:.3f}",
            ]) + r" \\"
        )

    return rf"""\begin{{table}}[H]
\centering
\scriptsize
\setlength{{\tabcolsep}}{{1.5pt}}
\caption{{{caption}}}
\label{{{label}}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{r r r p{{2.3cm}} p{{2.3cm}} l r r r r r r r r}}
\toprule
Rank & Original RT & Route & From Stop & To Stop & Representative trip & Segments & Trips &
\begin{{tabular}}{{@{{}}c@{{}}}}{distance_label}\\(km)\end{{tabular}} &
\begin{{tabular}}{{@{{}}c@{{}}}}{duration_label}\\(min)\end{{tabular}} &
\begin{{tabular}}{{@{{}}c@{{}}}}Original emission\\(kg CO$_2$e)\end{{tabular}} &
\begin{{tabular}}{{@{{}}c@{{}}}}Biogas consumption\\(kg)\end{{tabular}} &
\begin{{tabular}}{{@{{}}c@{{}}}}Biogas emission\\(kg CO$_2$e)\end{{tabular}} &
\begin{{tabular}}{{@{{}}c@{{}}}}Biogas emission per km\\(kg CO$_2$e/km)\end{{tabular}} \\
\midrule
{chr(10).join(table_rows)}
\bottomrule
\end{{tabular}}
}}
\end{{table}}
"""


top10_biogas_segment_latex = build_top10_biogas_segment_latex_table(
    top10_segments_biogas_experiment,
    "Top 10 grouped segment emission results under a biogas-only what-if assumption.",
    "tab:top10_segment_biogas_experiment",
    "max_distance_km",
    "max_duration_min",
    "Max distance",
    "Max duration",
)

top10_performing_biogas_segment_latex = build_top10_biogas_segment_latex_table(
    top10_performing_segments_biogas_experiment,
    "Top 10 performing grouped segment emission results under a biogas-only what-if assumption.",
    "tab:top10_performing_segment_biogas_experiment",
    "min_distance_km",
    "min_duration_min",
    "Min distance",
    "Min duration",
)


def build_top10_biogas_trip_latex_table(
    trip_df,
    caption,
    label,
    distance_column,
    duration_column,
    distance_label,
    duration_label,
):
    table_rows = []
    for rank, row in trip_df.iterrows():
        table_rows.append(
            " & ".join([
                str(rank + 1),
                latex_escape(row["representative_trip_id"]),
                latex_escape(row["route_type"]),
                latex_escape(row["route_short_name"]),
                f"{row['trip_count']:.0f}",
                f"{row[distance_column]:.2f}",
                f"{row[duration_column]:.2f}",
                f"{row['original_emission_kg']:.2f}",
                f"{row['biogas_consumption_kg']:.2f}",
                f"{row['biogas_emission_kg']:.2f}",
                f"{row['biogas_emission_per_km']:.3f}",
            ]) + r" \\"
        )

    return rf"""\begin{{table}}[H]
\centering
\scriptsize
\setlength{{\tabcolsep}}{{1.6pt}}
\caption{{{caption}}}
\label{{{label}}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{r l r r r r r r r r r}}
\toprule
Rank & Representative trip & Original RT & Route & Trips &
\begin{{tabular}}{{@{{}}c@{{}}}}{distance_label}\\(km)\end{{tabular}} &
\begin{{tabular}}{{@{{}}c@{{}}}}{duration_label}\\(min)\end{{tabular}} &
\begin{{tabular}}{{@{{}}c@{{}}}}Original emission\\(kg CO$_2$e)\end{{tabular}} &
\begin{{tabular}}{{@{{}}c@{{}}}}Biogas consumption\\(kg)\end{{tabular}} &
\begin{{tabular}}{{@{{}}c@{{}}}}Biogas emission\\(kg CO$_2$e)\end{{tabular}} &
\begin{{tabular}}{{@{{}}c@{{}}}}Biogas emission per km\\(kg CO$_2$e/km)\end{{tabular}} \\
\midrule
{chr(10).join(table_rows)}
\bottomrule
\end{{tabular}}
}}
\end{{table}}
"""


top10_biogas_trip_latex = build_top10_biogas_trip_latex_table(
    top10_trips_biogas_experiment,
    "Top 10 grouped trip emission results under a biogas-only what-if assumption.",
    "tab:top10_trip_biogas_experiment",
    "max_distance_km",
    "max_duration_min",
    "Max distance",
    "Max duration",
)

top10_performing_biogas_trip_latex = build_top10_biogas_trip_latex_table(
    top10_performing_trips_biogas_experiment,
    "Top 10 performing grouped trip emission results under a biogas-only what-if assumption.",
    "tab:top10_performing_trip_biogas_experiment",
    "min_distance_km",
    "min_duration_min",
    "Min distance",
    "Min duration",
)


def build_mode_segment_latex_table(
    segment_df,
    caption,
    label,
    distance_column,
    duration_column,
    distance_label,
    duration_label,
):
    table_rows = []
    for rank, row in segment_df.iterrows():
        table_rows.append(
            " & ".join([
                str(rank + 1),
                latex_escape(row["vehicle_type"]),
                latex_escape(row["route_type"]),
                latex_escape(row["route_short_name"]),
                latex_escape(row["departure_stop_name"]),
                latex_escape(row["arrive_stop_name"]),
                latex_escape(row["representative_trip_id"]),
                f"{row['segment_count']:.0f}",
                f"{row['trip_count']:.0f}",
                f"{row[distance_column]:.2f}",
                f"{row[duration_column]:.2f}",
                f"{row['emission_kg']:.2f}",
                f"{row['emission_per_km']:.3f}",
            ]) + r" \\"
        )

    return rf"""\begin{{table}}[H]
\centering
\scriptsize
\setlength{{\tabcolsep}}{{1.4pt}}
\caption{{{caption}}}
\label{{{label}}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{r p{{1.6cm}} r r p{{2.2cm}} p{{2.2cm}} l r r r r r r}}
\toprule
Rank & Mode & RT & Route & From Stop & To Stop & Representative trip & Segments & Trips &
\begin{{tabular}}{{@{{}}c@{{}}}}{distance_label}\\(km)\end{{tabular}} &
\begin{{tabular}}{{@{{}}c@{{}}}}{duration_label}\\(min)\end{{tabular}} &
\begin{{tabular}}{{@{{}}c@{{}}}}Emission\\(kg CO$_2$e)\end{{tabular}} &
\begin{{tabular}}{{@{{}}c@{{}}}}Emission per km\\(kg CO$_2$e/km)\end{{tabular}} \\
\midrule
{chr(10).join(table_rows)}
\bottomrule
\end{{tabular}}
}}
\end{{table}}
"""


top10_mode_segment_latex = build_mode_segment_latex_table(
    top10_grouped_segments_mode_experiment,
    "Top 10 grouped segment emissions using route-type-specific transport mode factors.",
    "tab:top10_mode_segment_experiment",
    "max_distance_km",
    "max_duration_min",
    "Max distance",
    "Max duration",
)

top10_performing_mode_segment_latex = build_mode_segment_latex_table(
    top10_performing_grouped_segments_mode_experiment,
    "Top 10 performing grouped segment emissions using route-type-specific transport mode factors.",
    "tab:top10_performing_mode_segment_experiment",
    "min_distance_km",
    "min_duration_min",
    "Min distance",
    "Min duration",
)


def build_mode_trip_latex_table(
    trip_df,
    caption,
    label,
    distance_column,
    duration_column,
    distance_label,
    duration_label,
):
    table_rows = []
    for rank, row in trip_df.iterrows():
        table_rows.append(
            " & ".join([
                str(rank + 1),
                latex_escape(row["representative_trip_id"]),
                latex_escape(row["vehicle_type"]),
                latex_escape(row["route_type"]),
                latex_escape(row["route_short_name"]),
                f"{row['trip_count']:.0f}",
                f"{row[distance_column]:.2f}",
                f"{row[duration_column]:.2f}",
                f"{row['emission_kg']:.2f}",
                f"{row['emission_per_km']:.3f}",
            ]) + r" \\"
        )

    return rf"""\begin{{table}}[H]
\centering
\scriptsize
\setlength{{\tabcolsep}}{{1.6pt}}
\caption{{{caption}}}
\label{{{label}}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{r l p{{1.8cm}} r r r r r r r}}
\toprule
Rank & Representative trip & Mode & RT & Route & Trips &
\begin{{tabular}}{{@{{}}c@{{}}}}{distance_label}\\(km)\end{{tabular}} &
\begin{{tabular}}{{@{{}}c@{{}}}}{duration_label}\\(min)\end{{tabular}} &
\begin{{tabular}}{{@{{}}c@{{}}}}Emission\\(kg CO$_2$e)\end{{tabular}} &
\begin{{tabular}}{{@{{}}c@{{}}}}Emission per km\\(kg CO$_2$e/km)\end{{tabular}} \\
\midrule
{chr(10).join(table_rows)}
\bottomrule
\end{{tabular}}
}}
\end{{table}}
"""


top10_mode_trip_latex = build_mode_trip_latex_table(
    top10_grouped_trips_mode_experiment,
    "Top 10 grouped trip emissions using route-type-specific transport mode factors.",
    "tab:top10_mode_trip_experiment",
    "max_distance_km",
    "max_duration_min",
    "Max distance",
    "Max duration",
)

top10_performing_mode_trip_latex = build_mode_trip_latex_table(
    top10_performing_grouped_trips_mode_experiment,
    "Top 10 performing grouped trip emissions using route-type-specific transport mode factors.",
    "tab:top10_performing_mode_trip_experiment",
    "min_distance_km",
    "min_duration_min",
    "Min distance",
    "Min duration",
)


def build_single_factor_segment_latex_table(
    segment_df,
    caption,
    label,
    distance_column,
    duration_column,
    distance_label,
    duration_label,
):
    table_rows = []
    for rank, row in segment_df.iterrows():
        table_rows.append(
            " & ".join([
                str(rank + 1),
                latex_escape(row["vehicle_type"]),
                latex_escape(row["route_type"]),
                latex_escape(row["route_short_name"]),
                latex_escape(row["departure_stop_name"]),
                latex_escape(row["arrive_stop_name"]),
                latex_escape(row["representative_trip_id"]),
                f"{row['segment_count']:.0f}",
                f"{row['trip_count']:.0f}",
                f"{row[distance_column]:.2f}",
                f"{row[duration_column]:.2f}",
                f"{row['consumption']:.2f}",
                f"{row['emission_kg']:.2f}",
                f"{row['emission_per_km']:.3f}",
            ]) + r" \\"
        )

    return rf"""\begin{{table}}[H]
\centering
\scriptsize
\setlength{{\tabcolsep}}{{1.2pt}}
\caption{{{caption}}}
\label{{{label}}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{r p{{1.5cm}} r r p{{2.1cm}} p{{2.1cm}} l r r r r r r r}}
\toprule
Rank & Original mode & RT & Route & From Stop & To Stop & Representative trip & Segments & Trips &
\begin{{tabular}}{{@{{}}c@{{}}}}{distance_label}\\(km)\end{{tabular}} &
\begin{{tabular}}{{@{{}}c@{{}}}}{duration_label}\\(min)\end{{tabular}} &
\begin{{tabular}}{{@{{}}c@{{}}}}Consumption\end{{tabular}} &
\begin{{tabular}}{{@{{}}c@{{}}}}Emission\\(kg CO$_2$e)\end{{tabular}} &
\begin{{tabular}}{{@{{}}c@{{}}}}Emission per km\\(kg CO$_2$e/km)\end{{tabular}} \\
\midrule
{chr(10).join(table_rows)}
\bottomrule
\end{{tabular}}
}}
\end{{table}}
"""


top10_single_factor_segment_latex = build_single_factor_segment_latex_table(
    top10_grouped_segments_single_factor_experiment,
    "Top 10 grouped segment emissions using a single distance-based consumption and emission factor.",
    "tab:top10_single_factor_segment_experiment",
    "max_distance_km",
    "max_duration_min",
    "Max distance",
    "Max duration",
)

top10_performing_single_factor_segment_latex = build_single_factor_segment_latex_table(
    top10_performing_grouped_segments_single_factor_experiment,
    "Top 10 performing grouped segment emissions using a single distance-based consumption and emission factor.",
    "tab:top10_performing_single_factor_segment_experiment",
    "min_distance_km",
    "min_duration_min",
    "Min distance",
    "Min duration",
)


def build_single_factor_trip_latex_table(
    trip_df,
    caption,
    label,
    distance_column,
    duration_column,
    distance_label,
    duration_label,
):
    table_rows = []
    for rank, row in trip_df.iterrows():
        table_rows.append(
            " & ".join([
                str(rank + 1),
                latex_escape(row["representative_trip_id"]),
                latex_escape(row["vehicle_type"]),
                latex_escape(row["route_type"]),
                latex_escape(row["route_short_name"]),
                f"{row['trip_count']:.0f}",
                f"{row[distance_column]:.2f}",
                f"{row[duration_column]:.2f}",
                f"{row['consumption']:.2f}",
                f"{row['emission_kg']:.2f}",
                f"{row['emission_per_km']:.3f}",
            ]) + r" \\"
        )

    return rf"""\begin{{table}}[H]
\centering
\scriptsize
\setlength{{\tabcolsep}}{{1.4pt}}
\caption{{{caption}}}
\label{{{label}}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{r l p{{1.6cm}} r r r r r r r r}}
\toprule
Rank & Representative trip & Original mode & RT & Route & Trips &
\begin{{tabular}}{{@{{}}c@{{}}}}{distance_label}\\(km)\end{{tabular}} &
\begin{{tabular}}{{@{{}}c@{{}}}}{duration_label}\\(min)\end{{tabular}} &
\begin{{tabular}}{{@{{}}c@{{}}}}Consumption\end{{tabular}} &
\begin{{tabular}}{{@{{}}c@{{}}}}Emission\\(kg CO$_2$e)\end{{tabular}} &
\begin{{tabular}}{{@{{}}c@{{}}}}Emission per km\\(kg CO$_2$e/km)\end{{tabular}} \\
\midrule
{chr(10).join(table_rows)}
\bottomrule
\end{{tabular}}
}}
\end{{table}}
"""


top10_single_factor_trip_latex = build_single_factor_trip_latex_table(
    top10_grouped_trips_single_factor_experiment,
    "Top 10 grouped trip emissions using a single distance-based consumption and emission factor.",
    "tab:top10_single_factor_trip_experiment",
    "max_distance_km",
    "max_duration_min",
    "Max distance",
    "Max duration",
)

top10_performing_single_factor_trip_latex = build_single_factor_trip_latex_table(
    top10_performing_grouped_trips_single_factor_experiment,
    "Top 10 performing grouped trip emissions using a single distance-based consumption and emission factor.",
    "tab:top10_performing_single_factor_trip_experiment",
    "min_distance_km",
    "min_duration_min",
    "Min distance",
    "Min duration",
)

top10_occupancy_mode_segment_latex = build_mode_segment_latex_table(
    top10_grouped_segments_occupancy_mode_experiment,
    "Top 10 grouped occupancy-aware segment emissions using route-type-specific transport mode factors.",
    "tab:top10_occupancy_mode_segment_experiment",
    "max_distance_km",
    "max_duration_min",
    "Max distance",
    "Max duration",
)

top10_performing_occupancy_mode_segment_latex = build_mode_segment_latex_table(
    top10_performing_grouped_segments_occupancy_mode_experiment,
    "Top 10 performing grouped occupancy-aware segment emissions using route-type-specific transport mode factors.",
    "tab:top10_performing_occupancy_mode_segment_experiment",
    "min_distance_km",
    "min_duration_min",
    "Min distance",
    "Min duration",
)

top10_occupancy_mode_trip_latex = build_mode_trip_latex_table(
    top10_grouped_trips_occupancy_mode_experiment,
    "Top 10 grouped occupancy-aware trip emissions using route-type-specific transport mode factors.",
    "tab:top10_occupancy_mode_trip_experiment",
    "max_distance_km",
    "max_duration_min",
    "Max distance",
    "Max duration",
)

top10_performing_occupancy_mode_trip_latex = build_mode_trip_latex_table(
    top10_performing_grouped_trips_occupancy_mode_experiment,
    "Top 10 performing grouped occupancy-aware trip emissions using route-type-specific transport mode factors.",
    "tab:top10_performing_occupancy_mode_trip_experiment",
    "min_distance_km",
    "min_duration_min",
    "Min distance",
    "Min duration",
)


def build_delay_aware_segment_latex_table(
    segment_df,
    caption,
    label,
    distance_column,
    duration_column,
    penalty_column,
    distance_label,
    duration_label,
):
    table_rows = []
    for rank, row in segment_df.iterrows():
        table_rows.append(
            " & ".join([
                str(rank + 1),
                latex_escape(row["vehicle_type"]),
                latex_escape(row["route_type"]),
                latex_escape(row["route_short_name"]),
                latex_escape(row["departure_stop_name"]),
                latex_escape(row["arrive_stop_name"]),
                latex_escape(row["representative_trip_id"]),
                f"{row['segment_count']:.0f}",
                f"{row['trip_count']:.0f}",
                f"{row[distance_column]:.2f}",
                f"{row[duration_column]:.2f}",
                f"{row[penalty_column]:.2f}",
                f"{row['delay_aware_emission_score']:.2f}",
                f"{row['delay_aware_score_per_km']:.3f}",
            ]) + r" \\"
        )

    return rf"""\begin{{table}}[H]
\centering
\scriptsize
\setlength{{\tabcolsep}}{{1.2pt}}
\caption{{{caption}}}
\label{{{label}}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{r p{{1.5cm}} r r p{{2.1cm}} p{{2.1cm}} l r r r r r r r}}
\toprule
Rank & Mode & RT & Route & From Stop & To Stop & Representative trip & Segments & Trips &
\begin{{tabular}}{{@{{}}c@{{}}}}{distance_label}\\(km)\end{{tabular}} &
\begin{{tabular}}{{@{{}}c@{{}}}}{duration_label}\\(min)\end{{tabular}} &
\begin{{tabular}}{{@{{}}c@{{}}}}Duration penalty\end{{tabular}} &
\begin{{tabular}}{{@{{}}c@{{}}}}Delay-aware emission score\end{{tabular}} &
\begin{{tabular}}{{@{{}}c@{{}}}}Delay-aware score per km\end{{tabular}} \\
\midrule
{chr(10).join(table_rows)}
\bottomrule
\end{{tabular}}
}}
\end{{table}}
"""


top10_delay_aware_segment_latex = build_delay_aware_segment_latex_table(
    top10_grouped_segments_delay_aware_experiment,
    "Top 10 grouped segment delay-aware emission scores using occupancy-aware emissions.",
    "tab:top10_delay_aware_segment_experiment",
    "max_distance_km",
    "max_duration_min",
    "max_duration_penalty",
    "Max distance",
    "Max duration",
)

top10_performing_delay_aware_segment_latex = build_delay_aware_segment_latex_table(
    top10_performing_grouped_segments_delay_aware_experiment,
    "Top 10 performing grouped segment delay-aware emission scores using occupancy-aware emissions.",
    "tab:top10_performing_delay_aware_segment_experiment",
    "min_distance_km",
    "min_duration_min",
    "min_duration_penalty",
    "Min distance",
    "Min duration",
)


def build_normal_delay_aware_segment_latex_table(segment_df, caption, label):
    table_rows = []
    for rank, row in segment_df.iterrows():
        table_rows.append(
            " & ".join([
                str(rank + 1),
                latex_escape(row["trip_id"]),
                latex_escape(row["segment_id"]),
                latex_escape(row["vehicle_type"]),
                latex_escape(row["route_type"]),
                latex_escape(row["route_short_name"]),
                latex_escape(row["departure_stop_name"]),
                latex_escape(row["arrive_stop_name"]),
                latex_escape(row["departure_timestamp"]),
                latex_escape(row["arrive_timestamp"]),
                f"{row['distance_diff_km']:.2f}",
                f"{row['duration_min']:.2f}",
                f"{row['duration_penalty']:.2f}",
                f"{row['delay_aware_emission_score']:.2f}",
                f"{row['delay_aware_score_per_km']:.3f}",
            ]) + r" \\"
        )

    return rf"""\begin{{table}}[p]
\centering
\scriptsize
\setlength{{\tabcolsep}}{{1.0pt}}
\caption{{{caption}}}
\label{{{label}}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{r l l p{{1.5cm}} r r p{{2.1cm}} p{{2.1cm}} l l r r r r r}}
\toprule
Rank & Trip & Segment & Mode & RT & Route & From Stop & To Stop & Departure & Arrival &
\begin{{tabular}}{{@{{}}c@{{}}}}Distance\\(km)\end{{tabular}} &
\begin{{tabular}}{{@{{}}c@{{}}}}Duration\\(min)\end{{tabular}} &
\begin{{tabular}}{{@{{}}c@{{}}}}Duration penalty\end{{tabular}} &
\begin{{tabular}}{{@{{}}c@{{}}}}Delay-aware emission score\end{{tabular}} &
\begin{{tabular}}{{@{{}}c@{{}}}}Delay-aware score per km\end{{tabular}} \\
\midrule
{chr(10).join(table_rows)}
\bottomrule
\end{{tabular}}
}}
\end{{table}}
"""


top100_delay_aware_segment_latex = build_normal_delay_aware_segment_latex_table(
    top100_segments_delay_aware_experiment,
    "Top 100 individual segment delay-aware emission scores using occupancy-aware emissions.",
    "tab:top100_delay_aware_segment_experiment",
)


def build_delay_aware_trip_latex_table(
    trip_df,
    caption,
    label,
    distance_column,
    duration_column,
    penalty_column,
    distance_label,
    duration_label,
):
    table_rows = []
    for rank, row in trip_df.iterrows():
        table_rows.append(
            " & ".join([
                str(rank + 1),
                latex_escape(row["representative_trip_id"]),
                latex_escape(row["vehicle_type"]),
                latex_escape(row["route_type"]),
                latex_escape(row["route_short_name"]),
                f"{row['trip_count']:.0f}",
                f"{row[distance_column]:.2f}",
                f"{row[duration_column]:.2f}",
                f"{row[penalty_column]:.2f}",
                f"{row['delay_aware_emission_score']:.2f}",
                f"{row['delay_aware_score_per_km']:.3f}",
            ]) + r" \\"
        )

    return rf"""\begin{{table}}[H]
\centering
\scriptsize
\setlength{{\tabcolsep}}{{1.4pt}}
\caption{{{caption}}}
\label{{{label}}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{r l p{{1.6cm}} r r r r r r r r}}
\toprule
Rank & Representative trip & Mode & RT & Route & Trips &
\begin{{tabular}}{{@{{}}c@{{}}}}{distance_label}\\(km)\end{{tabular}} &
\begin{{tabular}}{{@{{}}c@{{}}}}{duration_label}\\(min)\end{{tabular}} &
\begin{{tabular}}{{@{{}}c@{{}}}}Duration penalty\end{{tabular}} &
\begin{{tabular}}{{@{{}}c@{{}}}}Delay-aware emission score\end{{tabular}} &
\begin{{tabular}}{{@{{}}c@{{}}}}Delay-aware score per km\end{{tabular}} \\
\midrule
{chr(10).join(table_rows)}
\bottomrule
\end{{tabular}}
}}
\end{{table}}
"""


top10_delay_aware_trip_latex = build_delay_aware_trip_latex_table(
    top10_grouped_trips_delay_aware_experiment,
    "Top 10 grouped trip delay-aware emission scores using occupancy-aware emissions.",
    "tab:top10_delay_aware_trip_experiment",
    "max_distance_km",
    "max_duration_min",
    "max_duration_penalty",
    "Max distance",
    "Max duration",
)

top10_performing_delay_aware_trip_latex = build_delay_aware_trip_latex_table(
    top10_performing_grouped_trips_delay_aware_experiment,
    "Top 10 performing grouped trip delay-aware emission scores using occupancy-aware emissions.",
    "tab:top10_performing_delay_aware_trip_experiment",
    "min_distance_km",
    "min_duration_min",
    "min_duration_penalty",
    "Min distance",
    "Min duration",
)


def build_trip_emission_latex_table(trip_df, caption, label):
    table_rows = []
    for rank, row in trip_df.iterrows():
        emission_per_km = row["total_emission_actual"] / row["total_distance_km"]
        table_rows.append(
            " & ".join([
                str(rank + 1),
                latex_escape(row["vehicle_type"]),
                latex_escape(row["route_short_name"]),
                latex_escape(row["trip_id"]),
                f"{row['total_distance_km']:.2f}",
                f"{row['total_duration_min']:.2f}",
                f"{row['total_emission_actual']:.2f}",
                f"{emission_per_km:.3f}",
            ]) + r" \\"
        )

    return rf"""\begin{{table}}[H]
\centering
\scriptsize
\setlength{{\tabcolsep}}{{2pt}}
\caption{{{caption}}}
\label{{{label}}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{r p{{1.8cm}} r l r r r r}}
\toprule
Rank & Mode & Route & Trip ID &
\begin{{tabular}}{{@{{}}c@{{}}}}Distance\\(km)\end{{tabular}} &
\begin{{tabular}}{{@{{}}c@{{}}}}Duration\\(min)\end{{tabular}} &
\begin{{tabular}}{{@{{}}c@{{}}}}Emission\\(kg CO$_2$e)\end{{tabular}} &
\begin{{tabular}}{{@{{}}c@{{}}}}Emission per km\\(kg CO$_2$e/km)\end{{tabular}} \\
\midrule
{chr(10).join(table_rows)}
\bottomrule
\end{{tabular}}
}}
\end{{table}}
"""


def build_trip_score_latex_table(trip_df, caption, label):
    table_rows = []
    for rank, row in trip_df.iterrows():
        emission_per_km = row["total_emission_actual"] / row["total_distance_km"]
        table_rows.append(
            " & ".join([
                str(rank + 1),
                latex_escape(row["vehicle_type"]),
                latex_escape(row["route_short_name"]),
                latex_escape(row["trip_id"]),
                f"{row['total_distance_km']:.2f}",
                f"{row['total_duration_min']:.2f}",
                f"{row['total_emission_actual']:.2f}",
                f"{emission_per_km:.3f}",
                f"{row['trip_anomaly_score']:.2f}",
            ]) + r" \\"
        )

    return rf"""\begin{{table}}[H]
\centering
\scriptsize
\setlength{{\tabcolsep}}{{1.8pt}}
\caption{{{caption}}}
\label{{{label}}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{r p{{1.8cm}} r l r r r r r}}
\toprule
Rank & Mode & Route & Trip ID &
\begin{{tabular}}{{@{{}}c@{{}}}}Distance\\(km)\end{{tabular}} &
\begin{{tabular}}{{@{{}}c@{{}}}}Duration\\(min)\end{{tabular}} &
\begin{{tabular}}{{@{{}}c@{{}}}}Emission\\(kg CO$_2$e)\end{{tabular}} &
\begin{{tabular}}{{@{{}}c@{{}}}}Emission per km\\(kg CO$_2$e/km)\end{{tabular}} &
Score \\
\midrule
{chr(10).join(table_rows)}
\bottomrule
\end{{tabular}}
}}
\end{{table}}
"""


def format_modes(values):
    return ", ".join(sorted({str(value) for value in values if pd.notna(value)}))


def build_route_emission_latex_table(route_df, caption, label):
    table_rows = []
    for rank, row in route_df.iterrows():
        table_rows.append(
            " & ".join([
                str(rank + 1),
                latex_escape(row["route_short_name"]),
                latex_escape(row["modes"]),
                f"{row['total_emission_actual']:.2f}",
            ]) + r" \\"
        )

    return rf"""\begin{{table}}[H]
\centering
\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\caption{{{caption}}}
\label{{{label}}}
\begin{{tabular}}{{r r p{{4cm}} r}}
\toprule
Rank & Route & Mode(s) &
\begin{{tabular}}{{@{{}}c@{{}}}}Emission\\(kg CO$_2$e)\end{{tabular}} \\
\midrule
{chr(10).join(table_rows)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def build_route_emission_per_km_latex_table(route_df, caption, label):
    table_rows = []
    for rank, row in route_df.iterrows():
        table_rows.append(
            " & ".join([
                str(rank + 1),
                latex_escape(row["route_short_name"]),
                latex_escape(row["modes"]),
                f"{row['total_distance_km']:.2f}",
                f"{row['total_emission_actual']:.2f}",
                f"{row['emission_per_km']:.3f}",
            ]) + r" \\"
        )

    return rf"""\begin{{table}}[H]
\centering
\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\caption{{{caption}}}
\label{{{label}}}
\begin{{tabular}}{{r r p{{3.3cm}} r r r}}
\toprule
Rank & Route & Mode(s) &
\begin{{tabular}}{{@{{}}c@{{}}}}Distance\\(km)\end{{tabular}} &
\begin{{tabular}}{{@{{}}c@{{}}}}Emission\\(kg CO$_2$e)\end{{tabular}} &
\begin{{tabular}}{{@{{}}c@{{}}}}Emission per km\\(kg CO$_2$e/km)\end{{tabular}} \\
\midrule
{chr(10).join(table_rows)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


bus_tram_trip_summary = trip_consumption_emission_summary[
    trip_consumption_emission_summary["route_type"].astype(str).isin(["700", "900"])
].copy()

bus_tram_trip_summary["emission_per_km"] = divide_by_positive(
    bus_tram_trip_summary["total_emission_actual"],
    bus_tram_trip_summary["total_distance_km"],
)

bus_tram_trip_score_summary = trip_anomaly_score_summary[
    trip_anomaly_score_summary["route_type"].astype(str).isin(["700", "900"])
].reset_index(drop=True)

railway_trip_summary = trip_consumption_emission_summary[
    trip_consumption_emission_summary["route_type"].astype(str) == "100"
].copy()

railway_trip_summary["emission_per_km"] = divide_by_positive(
    railway_trip_summary["total_emission_actual"],
    railway_trip_summary["total_distance_km"],
)

top10_trip_emission_latex = build_trip_emission_latex_table(
    bus_tram_trip_summary.sort_values(
        "total_emission_actual",
        ascending=False,
    ).head(10).reset_index(drop=True),
    "Top 10 bus and tram trips by absolute trip emissions.",
    "tab:top10_trip_emissions",
)

top10_trip_emission_per_km_latex = build_trip_emission_latex_table(
    bus_tram_trip_summary.sort_values(
        ["emission_per_km", "total_emission_actual"],
        ascending=[False, False],
    ).head(10).reset_index(drop=True),
    "Top 10 bus and tram trips by emissions per travelled kilometre.",
    "tab:top10_trip_emissions_per_km",
)

bottom10_trip_emission_latex = build_trip_emission_latex_table(
    bus_tram_trip_summary.sort_values(
        "total_emission_actual",
        ascending=True,
    ).head(10).reset_index(drop=True),
    "Bottom 10 bus and tram trips by absolute trip emissions.",
    "tab:bottom10_trip_emissions",
)

top10_trip_score_latex = build_trip_score_latex_table(
    bus_tram_trip_score_summary.head(10),
    "Top 10 bus and tram trips by combined trip anomaly score.",
    "tab:top10_trip_anomaly_score",
)

top10_railway_trip_emission_latex = build_trip_emission_latex_table(
    railway_trip_summary.sort_values(
        "total_emission_actual",
        ascending=False,
    ).head(10).reset_index(drop=True),
    "Top 10 railway trips by absolute trip emissions.",
    "tab:top10_railway_trip_emissions",
)

route_emission_latex_source = (
    route_consumption_emission_summary.groupby(
        ["route_id", "route_short_name"],
        as_index=False,
        dropna=False,
    )
    .agg(
        modes=("vehicle_type", format_modes),
        total_distance_km=("total_distance_km", "sum"),
        total_emission_actual=("total_emission_actual", "sum"),
    )
)

route_emission_latex_source["emission_per_km"] = divide_by_positive(
    route_emission_latex_source["total_emission_actual"],
    route_emission_latex_source["total_distance_km"],
)

railway_route_emission_latex_source = (
    route_consumption_emission_summary[
        route_consumption_emission_summary["route_type"].astype(str) == "100"
    ]
    .groupby(
        ["route_id", "route_short_name"],
        as_index=False,
        dropna=False,
    )
    .agg(
        modes=("vehicle_type", format_modes),
        total_distance_km=("total_distance_km", "sum"),
        total_emission_actual=("total_emission_actual", "sum"),
    )
)

railway_route_emission_latex_source["emission_per_km"] = divide_by_positive(
    railway_route_emission_latex_source["total_emission_actual"],
    railway_route_emission_latex_source["total_distance_km"],
)

top10_route_emission_latex = build_route_emission_latex_table(
    route_emission_latex_source.sort_values(
        "total_emission_actual",
        ascending=False,
    ).head(10).reset_index(drop=True),
    "Top 10 routes by absolute route emissions.",
    "tab:top10_route_emissions",
)

bottom10_route_emission_latex = build_route_emission_latex_table(
    route_emission_latex_source.sort_values(
        "total_emission_actual",
        ascending=True,
    ).head(10).reset_index(drop=True),
    "Bottom 10 routes by absolute route emissions.",
    "tab:bottom10_route_emissions",
)

top10_route_emission_per_km_latex = build_route_emission_per_km_latex_table(
    route_emission_latex_source.sort_values(
        "emission_per_km",
        ascending=False,
    ).head(10).reset_index(drop=True),
    "Top 10 routes by emissions per kilometre.",
    "tab:top10_route_emissions_per_km",
)

top10_railway_route_emission_latex = build_route_emission_latex_table(
    railway_route_emission_latex_source.sort_values(
        "total_emission_actual",
        ascending=False,
    ).head(10).reset_index(drop=True),
    "Top 10 railway routes by absolute route emissions.",
    "tab:top10_railway_route_emissions",
)

top10_railway_route_emission_per_km_latex = build_route_emission_per_km_latex_table(
    railway_route_emission_latex_source.sort_values(
        "emission_per_km",
        ascending=False,
    ).head(10).reset_index(drop=True),
    "Top 10 railway routes by emissions per kilometre.",
    "tab:top10_railway_route_emissions_per_km",
)

segment_emission_per_km_tables_latex = f"""% Auto-generated by Emission_Calculation.py
{top10_segment_emission_per_km_latex}
"""

trip_emission_per_km_tables_latex = f"""% Auto-generated by Emission_Calculation.py
{top10_trip_emission_per_km_latex}
"""

railway_tables_latex = f"""% Auto-generated by Emission_Calculation.py
{top10_railway_trip_emission_latex}

{top10_railway_route_emission_latex}

{top10_railway_route_emission_per_km_latex}
"""

segment_level_tables_latex = f"""% Auto-generated by Emission_Calculation.py
{top10_segment_latex}

{top10_segment_emission_per_km_latex}

{top10_biogas_segment_latex}

{top10_performing_biogas_segment_latex}

{top10_mode_segment_latex}

{top10_performing_mode_segment_latex}

{top10_single_factor_segment_latex}

{top10_performing_single_factor_segment_latex}

{top10_occupancy_mode_segment_latex}

{top10_performing_occupancy_mode_segment_latex}

{top10_delay_aware_segment_latex}

{top10_performing_delay_aware_segment_latex}

{bottom10_segment_latex}

{top10_segment_score_latex}
"""

trip_level_tables_latex = f"""% Auto-generated by Emission_Calculation.py
{top10_trip_emission_latex}

{top10_trip_emission_per_km_latex}

{top10_biogas_trip_latex}

{top10_performing_biogas_trip_latex}

{top10_mode_trip_latex}

{top10_performing_mode_trip_latex}

{top10_single_factor_trip_latex}

{top10_performing_single_factor_trip_latex}

{top10_occupancy_mode_trip_latex}

{top10_performing_occupancy_mode_trip_latex}

{top10_delay_aware_trip_latex}

{top10_performing_delay_aware_trip_latex}

{bottom10_trip_emission_latex}

{top10_trip_score_latex}
"""

route_level_tables_latex = f"""% Auto-generated by Emission_Calculation.py
{top10_route_emission_latex}

{bottom10_route_emission_latex}

{top10_route_emission_per_km_latex}
"""

emission_analysis_tables_only_latex = f"""% Auto-generated by Emission_Calculation.py
{segment_level_tables_latex}

{trip_level_tables_latex}

{route_level_tables_latex}

{railway_tables_latex}
"""

segment_emission_analysis_latex = f"""% Auto-generated by Emission_Calculation.py
First, a general overview of segment emissions is provided, for example using a histogram or a boxplot. The segment emission is denoted as $E_{{\\mathrm{{segment}}}}$ and reported in kilograms of CO$_2$ equivalents per segment. For electric vehicles, emissions are computed from distance, energy consumption, and the electricity emission factor:

\\[
E_{{\\mathrm{{segment,electric}}}}
= \\frac{{d_{{\\mathrm{{segment}}}}}}{{100}}
\\cdot c_{{\\mathrm{{vehicle}}}}
\\cdot f_{{\\mathrm{{emission}}}}
\\]

For the biogas bus, the mass-based consumption is converted into volume using the assumed biogas density. The corresponding segment emission formula is:

\\[
E_{{\\mathrm{{segment,bus}}}}
= d_{{\\mathrm{{segment}}}}
\\cdot EF_{{\\mathrm{{vol}}}}
\\cdot \\frac{{c_{{\\mathrm{{mass}}}} / 100}}{{\\rho_{{\\mathrm{{biogas}}}}}}
\\]

with $EF_{{\\mathrm{{vol}}}} = 0.304$ kg CO$_2$e/m$^3$, $c_{{\\mathrm{{mass}}}} = 45$ kg/100 km, and $\\rho_{{\\mathrm{{biogas}}}} = 1.2$ kg/m$^3$. This corresponds to approximately $0.114$ kg CO$_2$e per kilometre for the biogas bus.

This allows the reader to see how strongly segment emissions in kg CO$_2$e per segment vary and whether a small number of very high values is present. This forms the descriptive entry point into the segment-level analysis.

For the segment-level anomaly analysis, a combined score is defined that captures two complementary perspectives: the absolute segment emission and the deviation between observed and expected travel time. The absolute emission component is based on the previously computed segment emission $E_{{\\mathrm{{segment}}}}$, which uses travelled distance as input and assumes constant vehicle-type-specific consumption and emission factors.

\\[
\\mathrm{{score}}_{{\\mathrm{{segment}}}}
= C^{{\\mathrm{{abs}}}}_{{\\mathrm{{segment}}}}
+ C^{{\\mathrm{{duration\\_res}}}}_{{\\mathrm{{segment}}}}
\\]

The two components are defined as:

\\[
C^{{\\mathrm{{abs}}}}_{{\\mathrm{{segment}}}}
= z^{{+}}\\left(E_{{\\mathrm{{segment}}}}\\right)
\\]

\\[
C^{{\\mathrm{{duration\\_res}}}}_{{\\mathrm{{segment}}}}
= z^{{+}}\\left(
\\frac{{
\\mathrm{{duration}}_{{\\mathrm{{actual}},\\mathrm{{segment}}}}
- \\mathrm{{duration}}_{{\\mathrm{{expected}},\\mathrm{{segment}}}}
}}{{
\\mathrm{{duration}}_{{\\mathrm{{expected}},\\mathrm{{segment}}}}
}}
\\right)
\\]

Here, $z^{{+}}(\\cdot)$ denotes the positive standardized value of a component. Thus, only unusually high absolute emissions or unusually high positive duration residuals increase the segment score. Since both components are weighted equally, the score can be written as an unweighted sum; multiplying each term by $\\frac{{1}}{{2}}$ would not change the resulting ranking.

The resulting ranking is shown at the end of this section in Table~\\ref{{tab:top10_segment_anomaly_score}}. It lists the 10 individual segments with the highest combined anomaly score and decomposes the score into its two components. The occurrence column counts how often the same mode, route, departure stop, and arrival stop occurred with at least the same observed duration. If this happened only once, the segment start time is reported as a single-case time.

Next, a table with the top 10 segment groups by absolute emissions is shown. Segments with the same route, mode, departure stop, arrival stop, rounded distance, and rounded emission are grouped together; the number of associated trips is reported as a compact count column. This table identifies which segment types cause the highest emissions overall and is therefore particularly relevant from a pure emissions perspective.

\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=0.85\\textwidth]{{thesis-main/figures/Emission_Experiment_2026_05_04_high_spread_original/segment_emission_histogram_{OUTPUT_TAG}.png}}
    \\caption{{Distribution of segment emissions in kg CO$_2$e per segment.}}
    \\label{{fig:segment_emission_histogram}}
\\end{{figure}}

\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=0.85\\textwidth]{{thesis-main/figures/Emission_Experiment_2026_05_04_high_spread_original/segment_emission_boxplot_{OUTPUT_TAG}.png}}
    \\caption{{Boxplot of segment emissions in kg CO$_2$e per segment.}}
    \\label{{fig:segment_emission_boxplot}}
\\end{{figure}}

{top10_segment_latex}

The following table ranks segment groups by emissions per travelled kilometre. This normalizes emissions by distance and therefore highlights the most emission-intensive segment groups independent of their absolute length.

{top10_segment_emission_per_km_latex}

In addition, the following table shows the 10 segment groups with the lowest absolute segment emissions. This highlights which short or low-emission segment types form the lower end of the distribution.

{bottom10_segment_latex}

Finally, the top 10 table of the combined segment score is shown. It combines absolute segment emissions with the deviation between observed and expected segment duration.

{top10_segment_score_latex}

At trip level, each trip is considered as its own observation. This shows which complete trips cause the highest and lowest total emissions.

{top10_trip_emission_latex}

The following trip-level table uses the same emission values but divides them by the travelled trip distance. It therefore shows the highest trip-level emission intensity in kg CO$_2$e per kilometre.

{top10_trip_emission_per_km_latex}

{bottom10_trip_emission_latex}

Analogous to the segment-level analysis, an additional trip score is computed. Here, $C^{{\\mathrm{{abs}}}}$ is applied to the absolute trip emission, while $C^{{\\mathrm{{duration\\_res}}}}$ compares the observed trip duration with the expected trip duration. The expected trip duration is determined at trip level from trip distance and average trip speed:

\\[
\\mathrm{{score}}_{{\\mathrm{{trip}}}}
= z^{{+}}\\left(E_{{\\mathrm{{trip}}}}\\right)
+ z^{{+}}\\left(
\\frac{{
\\mathrm{{duration}}_{{\\mathrm{{actual}},\\mathrm{{trip}}}}
- \\mathrm{{duration}}_{{\\mathrm{{expected}},\\mathrm{{trip}}}}
}}{{
\\mathrm{{duration}}_{{\\mathrm{{expected}},\\mathrm{{trip}}}}
}}
\\right)
\\]

{top10_trip_score_latex}

For railway services, the following table separately reports the railway trips with the highest absolute emissions.

{top10_railway_trip_emission_latex}

At route level, emissions are aggregated by route. If a route is associated with more than one mode, all observed modes are shown in the mode column. The following tables therefore focus only on route-level emissions.

{top10_route_emission_latex}

{bottom10_route_emission_latex}

As a final route-level perspective, emissions are divided by the total travelled route distance. This gives the emission intensity in kg CO$_2$e per kilometre.

{top10_route_emission_per_km_latex}

The railway routes are additionally reported separately to make the train-specific route emissions visible.

{top10_railway_route_emission_latex}

{top10_railway_route_emission_per_km_latex}
"""

(EMISSION_OUTPUT_DIR / f"segment_emission_analysis_{OUTPUT_TAG}.tex").write_text(
    segment_emission_analysis_latex,
    encoding="utf-8",
)

(EMISSION_OUTPUT_DIR / f"emission_analysis_tables_only_{OUTPUT_TAG}.tex").write_text(
    emission_analysis_tables_only_latex,
    encoding="utf-8",
)

(EMISSION_OUTPUT_DIR / f"segment_level_tables_only_{OUTPUT_TAG}.tex").write_text(
    segment_level_tables_latex,
    encoding="utf-8",
)

(EMISSION_OUTPUT_DIR / f"trip_level_tables_only_{OUTPUT_TAG}.tex").write_text(
    trip_level_tables_latex,
    encoding="utf-8",
)

(EMISSION_OUTPUT_DIR / f"top10_grouped_segments_mode_experiment_{OUTPUT_TAG}.tex").write_text(
    top10_mode_segment_latex,
    encoding="utf-8",
)

(EMISSION_OUTPUT_DIR / f"top10_performing_grouped_segments_mode_experiment_{OUTPUT_TAG}.tex").write_text(
    top10_performing_mode_segment_latex,
    encoding="utf-8",
)

(EMISSION_OUTPUT_DIR / f"top10_grouped_trips_mode_experiment_{OUTPUT_TAG}.tex").write_text(
    top10_mode_trip_latex,
    encoding="utf-8",
)

(EMISSION_OUTPUT_DIR / f"top10_performing_grouped_trips_mode_experiment_{OUTPUT_TAG}.tex").write_text(
    top10_performing_mode_trip_latex,
    encoding="utf-8",
)

(EMISSION_OUTPUT_DIR / f"top10_grouped_segments_single_factor_experiment_{OUTPUT_TAG}.tex").write_text(
    top10_single_factor_segment_latex,
    encoding="utf-8",
)

(EMISSION_OUTPUT_DIR / f"top10_performing_grouped_segments_single_factor_experiment_{OUTPUT_TAG}.tex").write_text(
    top10_performing_single_factor_segment_latex,
    encoding="utf-8",
)

(EMISSION_OUTPUT_DIR / f"top10_grouped_trips_single_factor_experiment_{OUTPUT_TAG}.tex").write_text(
    top10_single_factor_trip_latex,
    encoding="utf-8",
)

(EMISSION_OUTPUT_DIR / f"top10_performing_grouped_trips_single_factor_experiment_{OUTPUT_TAG}.tex").write_text(
    top10_performing_single_factor_trip_latex,
    encoding="utf-8",
)

(EMISSION_OUTPUT_DIR / f"top10_grouped_segments_occupancy_mode_experiment_{OUTPUT_TAG}.tex").write_text(
    top10_occupancy_mode_segment_latex,
    encoding="utf-8",
)

(EMISSION_OUTPUT_DIR / f"top10_performing_grouped_segments_occupancy_mode_experiment_{OUTPUT_TAG}.tex").write_text(
    top10_performing_occupancy_mode_segment_latex,
    encoding="utf-8",
)

(EMISSION_OUTPUT_DIR / f"top10_grouped_trips_occupancy_mode_experiment_{OUTPUT_TAG}.tex").write_text(
    top10_occupancy_mode_trip_latex,
    encoding="utf-8",
)

(EMISSION_OUTPUT_DIR / f"top10_performing_grouped_trips_occupancy_mode_experiment_{OUTPUT_TAG}.tex").write_text(
    top10_performing_occupancy_mode_trip_latex,
    encoding="utf-8",
)

(EMISSION_OUTPUT_DIR / f"top10_grouped_segments_delay_aware_experiment_{OUTPUT_TAG}.tex").write_text(
    top10_delay_aware_segment_latex,
    encoding="utf-8",
)

(EMISSION_OUTPUT_DIR / f"top100_segments_delay_aware_experiment_{OUTPUT_TAG}.tex").write_text(
    top100_delay_aware_segment_latex,
    encoding="utf-8",
)

(EMISSION_OUTPUT_DIR / f"top10_performing_grouped_segments_delay_aware_experiment_{OUTPUT_TAG}.tex").write_text(
    top10_performing_delay_aware_segment_latex,
    encoding="utf-8",
)

(EMISSION_OUTPUT_DIR / f"top10_grouped_trips_delay_aware_experiment_{OUTPUT_TAG}.tex").write_text(
    top10_delay_aware_trip_latex,
    encoding="utf-8",
)

(EMISSION_OUTPUT_DIR / f"top10_performing_grouped_trips_delay_aware_experiment_{OUTPUT_TAG}.tex").write_text(
    top10_performing_delay_aware_trip_latex,
    encoding="utf-8",
)

(EMISSION_OUTPUT_DIR / f"route_level_tables_only_{OUTPUT_TAG}.tex").write_text(
    route_level_tables_latex,
    encoding="utf-8",
)

(EMISSION_OUTPUT_DIR / f"segment_emission_per_km_tables_only_{OUTPUT_TAG}.tex").write_text(
    segment_emission_per_km_tables_latex,
    encoding="utf-8",
)

(EMISSION_OUTPUT_DIR / f"trip_emission_per_km_tables_only_{OUTPUT_TAG}.tex").write_text(
    trip_emission_per_km_tables_latex,
    encoding="utf-8",
)

(EMISSION_OUTPUT_DIR / f"railway_emission_tables_only_{OUTPUT_TAG}.tex").write_text(
    railway_tables_latex,
    encoding="utf-8",
)

print("Segment-Level Summary:")
print(tabulate(
    trip_segment_summary.head(20),
    headers="keys",
    tablefmt="psql",
    showindex=False
))

print("\nRoute-Type Emission Parameters:")
print(tabulate(
    pd.DataFrame.from_dict(ROUTE_TYPE_FACTORS, orient="index").reset_index(names="route_type"),
    headers="keys",
    tablefmt="psql",
    showindex=False,
))

print("\nTrip-Level Consumption and Emission Summary:")
print(tabulate(
    trip_consumption_emission_summary.head(20),
    headers="keys",
    tablefmt="psql",
    showindex=False
))

print("\nRoute-Level Consumption and Emission Summary:")
print(tabulate(
    route_consumption_emission_summary.sort_values("total_emission_actual", ascending=False).head(20),
    headers="keys",
    tablefmt="psql",
    showindex=False
))

print("\nTop Segment Emission Summary:")
print(tabulate(
    top_segment_emission_summary.head(20),
    headers="keys",
    tablefmt="psql",
    showindex=False
))

print("\nTop 10 Grouped Segment Biogas-Only Experiment:")
print(tabulate(
    top10_segments_biogas_experiment,
    headers="keys",
    tablefmt="psql",
    showindex=False
))

print("\nTop 10 Performing Grouped Segment Biogas-Only Experiment:")
print(tabulate(
    top10_performing_segments_biogas_experiment,
    headers="keys",
    tablefmt="psql",
    showindex=False
))

print("\nTop 10 Trip Biogas-Only Experiment:")
print(tabulate(
    top10_trips_biogas_experiment,
    headers="keys",
    tablefmt="psql",
    showindex=False
))

print("\nTop 10 Performing Trip Biogas-Only Experiment:")
print(tabulate(
    top10_performing_trips_biogas_experiment,
    headers="keys",
    tablefmt="psql",
    showindex=False
))

print("\nTop Segment Anomaly Score Summary:")
print(tabulate(
    segment_anomaly_score_summary.head(20),
    headers="keys",
    tablefmt="psql",
    showindex=False
))

print("\nOriginal high-spread emission calculation tables written.")
raise SystemExit(0)

sns.set_style("whitegrid")



segment_plot_df = trip_segment_summary.copy()
trip_plot_df = trip_consumption_emission_summary.copy()

segment_plot_df["emission_error_avg_speed_abs"] = segment_plot_df["emission_error_avg_speed"].abs()
segment_plot_df["emission_error_median_speed_abs"] = segment_plot_df["emission_error_median_speed"].abs()

trip_plot_df["emission_diff_avg_speed_abs"] = trip_plot_df["emission_diff_avg_speed"].abs()
trip_plot_df["emission_diff_median_speed_abs"] = trip_plot_df["emission_diff_median_speed"].abs()

segment_emission_xmax = segment_plot_df["emission_actual"].quantile(0.95)

#
# 0. Descriptive overview of segment emissions
#

plt.figure(figsize=(10, 6))
sns.histplot(
    data=segment_plot_df,
    x="emission_actual",
    binwidth=0.005,
    binrange=(0, segment_emission_xmax),
    kde=True
)
plt.xlim(0, segment_emission_xmax)
plt.title("Distribution of Segment Emissions (Zoomed to 95th Percentile)")
plt.xlabel("Segment emission (kg CO2e per segment)")
plt.ylabel("Segment count")
plt.tight_layout()
plt.savefig(
    EMISSION_OUTPUT_DIR / f"segment_emission_histogram_{OUTPUT_TAG}.png",
    dpi=300
)
plt.close()

plt.figure(figsize=(10, 4))
sns.boxplot(
    data=segment_plot_df,
    x="emission_actual"
)
plt.xlim(0, segment_emission_xmax)
plt.title("Boxplot of Segment Emissions (Zoomed to 95th Percentile)")
plt.xlabel("Segment emission (kg CO2e per segment)")
plt.tight_layout()
plt.savefig(
    EMISSION_OUTPUT_DIR / f"segment_emission_boxplot_{OUTPUT_TAG}.png",
    dpi=300
)
plt.close()

#
# 1. Compare average vs median emissions directly.
#

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=segment_plot_df,
    x="emission_avg_speed",
    y="emission_median_speed",
    alpha=0.35
)
plt.plot(
    [segment_plot_df["emission_avg_speed"].min(), segment_plot_df["emission_avg_speed"].max()],
    [segment_plot_df["emission_avg_speed"].min(), segment_plot_df["emission_avg_speed"].max()],
    color="red",
    linestyle="--",
    linewidth=1.5
)
plt.title("Average-Speed-Based vs Median-Speed-Based Emissions (Segment Level)")
plt.xlabel("Emission based on Average Speed")
plt.ylabel("Emission based on Median Speed")
plt.tight_layout()
plt.show()

#
# 2. Actual vs estimated emissions at segment level.
#

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

sns.scatterplot(
    data=segment_plot_df,
    x="emission_actual",
    y="emission_avg_speed",
    alpha=0.35,
    ax=axes[0]
)
axes[0].plot(
    [segment_plot_df["emission_actual"].min(), segment_plot_df["emission_actual"].max()],
    [segment_plot_df["emission_actual"].min(), segment_plot_df["emission_actual"].max()],
    color="red",
    linestyle="--",
    linewidth=1.5
)
axes[0].set_title("Actual vs Avg-Speed-Based Emissions")
axes[0].set_xlabel("Actual Emission")
axes[0].set_ylabel("Estimated Emission")

sns.scatterplot(
    data=segment_plot_df,
    x="emission_actual",
    y="emission_median_speed",
    alpha=0.35,
    ax=axes[1]
)
axes[1].plot(
    [segment_plot_df["emission_actual"].min(), segment_plot_df["emission_actual"].max()],
    [segment_plot_df["emission_actual"].min(), segment_plot_df["emission_actual"].max()],
    color="red",
    linestyle="--",
    linewidth=1.5
)
axes[1].set_title("Actual vs Median-Speed-Based Emissions")
axes[1].set_xlabel("Actual Emission")
axes[1].set_ylabel("")

plt.tight_layout()
plt.show()

#
# 3. Error distribution at segment level.
#

error_long_segment = segment_plot_df[[
    "emission_error_avg_speed",
    "emission_error_median_speed"
]].rename(columns={
    "emission_error_avg_speed": "Avg Speed Error",
    "emission_error_median_speed": "Median Speed Error"
}).melt(
    var_name="Method",
    value_name="Emission Error"
)

plt.figure(figsize=(10, 6))
sns.boxplot(
    data=error_long_segment,
    x="Method",
    y="Emission Error"
)
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.title("Distribution of Emission Errors (Segment Level)")
plt.xlabel("Method")
plt.ylabel("Emission Error")
plt.tight_layout()
plt.show()

#
# 4. Error vs duration.
#

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

sns.scatterplot(
    data=segment_plot_df,
    x="duration_min",
    y="emission_error_avg_speed",
    alpha=0.35,
    ax=axes[0]
)
axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
axes[0].set_title("Emission Error vs Duration (Avg Speed)")
axes[0].set_xlabel("Duration (minutes)")
axes[0].set_ylabel("Emission Error")

sns.scatterplot(
    data=segment_plot_df,
    x="duration_min",
    y="emission_error_median_speed",
    alpha=0.35,
    ax=axes[1]
)
axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
axes[1].set_title("Emission Error vs Duration (Median Speed)")
axes[1].set_xlabel("Duration (minutes)")
axes[1].set_ylabel("")

plt.tight_layout()
plt.show()

#
# 5. Trip level: actual vs estimated emissions.
#

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

sns.scatterplot(
    data=trip_plot_df,
    x="total_emission_actual",
    y="total_emission_avg_speed",
    alpha=0.6,
    ax=axes[0]
)
axes[0].plot(
    [trip_plot_df["total_emission_actual"].min(), trip_plot_df["total_emission_actual"].max()],
    [trip_plot_df["total_emission_actual"].min(), trip_plot_df["total_emission_actual"].max()],
    color="red",
    linestyle="--",
    linewidth=1.5
)
axes[0].set_title("Trip-Level: Actual vs Avg-Speed-Based Emissions")
axes[0].set_xlabel("Actual Emission")
axes[0].set_ylabel("Estimated Emission")

sns.scatterplot(
    data=trip_plot_df,
    x="total_emission_actual",
    y="total_emission_median_speed",
    alpha=0.6,
    ax=axes[1]
)
axes[1].plot(
    [trip_plot_df["total_emission_actual"].min(), trip_plot_df["total_emission_actual"].max()],
    [trip_plot_df["total_emission_actual"].min(), trip_plot_df["total_emission_actual"].max()],
    color="red",
    linestyle="--",
    linewidth=1.5
)
axes[1].set_title("Trip-Level: Actual vs Median-Speed-Based Emissions")
axes[1].set_xlabel("Actual Emission")
axes[1].set_ylabel("")

plt.tight_layout()
plt.show()


# 6. Top trips with the largest deviations.


top_n = 15
top_trip_errors = trip_plot_df.copy()
top_trip_errors["max_abs_error"] = top_trip_errors[[
    "emission_diff_avg_speed_abs",
    "emission_diff_median_speed_abs"
]].max(axis=1)

top_trip_errors = top_trip_errors.sort_values("max_abs_error", ascending=False).head(top_n)

top_trip_errors_long = top_trip_errors[[
    "trip_id",
    "emission_diff_avg_speed",
    "emission_diff_median_speed"
]].rename(columns={
    "emission_diff_avg_speed": "Avg Speed Error",
    "emission_diff_median_speed": "Median Speed Error"
}).melt(
    id_vars="trip_id",
    var_name="Method",
    value_name="Emission Difference"
)

plt.figure(figsize=(14, 7))
sns.barplot(
    data=top_trip_errors_long,
    x="trip_id",
    y="Emission Difference",
    hue="Method"
)
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.xticks(rotation=45, ha="right")
plt.title(f"Top {top_n} Trips with Largest Emission Differences")
plt.xlabel("Trip ID")
plt.ylabel("Emission Difference")
plt.tight_layout()
plt.show()


# 7. Histogram of absolute errors.

abs_error_long_trip = trip_plot_df[[
    "emission_diff_avg_speed_abs",
    "emission_diff_median_speed_abs"
]].rename(columns={
    "emission_diff_avg_speed_abs": "Avg Speed Absolute Error",
    "emission_diff_median_speed_abs": "Median Speed Absolute Error"
}).melt(
    var_name="Method",
    value_name="Absolute Emission Difference"
)

plt.figure(figsize=(10, 6))
sns.histplot(
    data=abs_error_long_trip,
    x="Absolute Emission Difference",
    hue="Method",
    bins=30,
    element="step",
    stat="density",
    common_norm=False
)
plt.title("Distribution of Absolute Emission Differences (Trip Level)")
plt.xlabel("Absolute Emission Difference")
plt.ylabel("Density")
plt.tight_layout()
plt.show()


# 8. Constant-speed segment emission comparison.

constant_speed_methods = [
    ("20 km/h", "emission_20_kmh"),
    ("40 km/h", "emission_40_kmh"),
    ("60 km/h", "emission_60_kmh"),
    ("Segment duration <5 min: 20 km/h, otherwise 40 km/h", "emission_duration_rule_20_40"),
]

fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
constant_speed_axes = axes.flatten()

constant_speed_max = pd.concat(
    [segment_plot_df["emission_actual"]] +
    [segment_plot_df[column] for _, column in constant_speed_methods],
    ignore_index=True,
).quantile(0.99)

for ax, (method_label, emission_column) in zip(constant_speed_axes, constant_speed_methods):
    sns.scatterplot(
        data=segment_plot_df,
        x="emission_actual",
        y=emission_column,
        alpha=0.25,
        s=15,
        ax=ax,
    )
    ax.plot(
        [0, constant_speed_max],
        [0, constant_speed_max],
        color="red",
        linestyle="--",
        linewidth=1.2,
    )
    ax.set_xlim(0, constant_speed_max)
    ax.set_ylim(0, constant_speed_max)
    ax.set_title(method_label)
    ax.set_xlabel("Shape-based emission (kg CO2e)")
    ax.set_ylabel("Speed-based emission (kg CO2e)")

for ax in constant_speed_axes[len(constant_speed_methods):]:
    ax.axis("off")

plt.suptitle("Segment Emissions: Shape-Based vs Constant-Speed Estimates", y=1.02)
plt.tight_layout()
plt.savefig(
    EMISSION_OUTPUT_DIR / f"constant_speed_emission_comparison_{OUTPUT_TAG}.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()


# 9. Constant-speed percentage errors by distance bin.

CONSTANT_SPEED_DISTANCE_BIN_EDGES = [step / 1000 for step in range(0, 3001, 200)] + [float("inf")]
CONSTANT_SPEED_DISTANCE_BIN_LABELS = [
    f"{CONSTANT_SPEED_DISTANCE_BIN_EDGES[index]:.1f}-{CONSTANT_SPEED_DISTANCE_BIN_EDGES[index + 1]:.1f}"
    for index in range(len(CONSTANT_SPEED_DISTANCE_BIN_EDGES) - 2)
] + [">=3.0"]

constant_speed_error_df = segment_plot_df[
    ["segment_id", "distance_diff_km", "emission_actual"] +
    [column for _, column in constant_speed_methods]
].copy()

constant_speed_error_df = constant_speed_error_df[
    constant_speed_error_df["emission_actual"] > 0
].copy()

constant_speed_error_df["distance_bin"] = pd.cut(
    constant_speed_error_df["distance_diff_km"],
    bins=CONSTANT_SPEED_DISTANCE_BIN_EDGES,
    labels=CONSTANT_SPEED_DISTANCE_BIN_LABELS,
    include_lowest=True,
    right=False,
)

constant_speed_error_long_rows = []
for method_label, emission_column in constant_speed_methods:
    method_df = constant_speed_error_df[[
        "segment_id",
        "distance_bin",
        "emission_actual",
        emission_column,
    ]].copy()
    method_df["Method"] = method_label
    method_df["Signed percentage error"] = (
        (method_df[emission_column] - method_df["emission_actual"]) /
        method_df["emission_actual"] * 100
    )
    method_df["Absolute percentage error"] = method_df["Signed percentage error"].abs()
    constant_speed_error_long_rows.append(method_df)

constant_speed_error_long = pd.concat(
    constant_speed_error_long_rows,
    ignore_index=True,
)

constant_speed_error_bin_summary = (
    constant_speed_error_long
    .groupby(["distance_bin", "Method"], observed=False)
    .agg(
        segment_count=("segment_id", "count"),
        mean_absolute_percentage_error=("Absolute percentage error", "mean"),
        mean_signed_percentage_error=("Signed percentage error", "mean"),
        signed_percentage_error_std=("Signed percentage error", "std"),
    )
    .reset_index()
)

constant_speed_error_bin_summary.to_csv(
    EMISSION_OUTPUT_DIR / f"constant_speed_percentage_error_by_distance_bin_{OUTPUT_TAG}.csv",
    index=False,
)

constant_speed_error_metrics = [
    (
        "mean_absolute_percentage_error",
        "Mean absolute percentage error (%)",
        "Mean Absolute Percentage Error by Distance Bin",
    ),
    (
        "mean_signed_percentage_error",
        "Mean signed percentage error (%)",
        "Mean Signed Percentage Error by Distance Bin",
    ),
    (
        "signed_percentage_error_std",
        "Std. dev. of signed percentage error (%)",
        "Standard Deviation of Percentage Error by Distance Bin",
    ),
]

fig, axes = plt.subplots(
    len(constant_speed_error_metrics),
    1,
    figsize=(18, 16),
    sharex=True,
)

for ax, (metric_column, y_label, title) in zip(axes, constant_speed_error_metrics):
    sns.barplot(
        data=constant_speed_error_bin_summary,
        x="distance_bin",
        y=metric_column,
        hue="Method",
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(y_label)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(title="Method", ncol=2, fontsize=8, title_fontsize=9)

axes[-1].set_xlabel("Shape-based segment distance bin (km)")
axes[-1].tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig(
    EMISSION_OUTPUT_DIR / f"constant_speed_percentage_error_by_distance_bin_{OUTPUT_TAG}.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()


# 10. Trip-level constant-speed percentage errors across all Kodak datasets.

ALL_KODAK_DATASET_DATES = [
    "2025_08_04",
    "2025_11_04",
    "2026_02_04",
    "2026_05_04",
]

ALL_KODAK_OUTPUT_DIR = Path("thesis-main/figures/Emission_Experiment_All_Kodak")
ALL_KODAK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def format_kodak_dataset_label(dataset_date):
    year, month, day = dataset_date.split("_")
    return f"kodak_{year}-{month}_{day}"


def build_constant_speed_error_tables_for_dataset(dataset_date):
    event_log_path = Path(f"aux_event_log_overview_kodak_{dataset_date}.csv")
    speed_summary_path = Path(f"trip_speed_summary_{dataset_date}.csv")

    if not event_log_path.exists() or not speed_summary_path.exists():
        return pd.DataFrame(), pd.DataFrame()

    events = pd.read_csv(
        event_log_path,
        dtype={
            "vehicle_id": "string",
            "trip_id": "string",
            "trip_id_org": "string",
            "stop_sequence": "Int64",
        },
    )

    speed_summary = pd.read_csv(
        speed_summary_path,
        dtype={"trip_id": "string"},
    ).rename(columns={"trip_id": "trip_id_org"})

    events["timestamp"] = pd.to_datetime(events["timestamp"], errors="coerce")
    events["shape_dist_traveled"] = pd.to_numeric(
        events["shape_dist_traveled"],
        errors="coerce",
    )
    events["route_type"] = events["route_type"].apply(normalize_route_type)
    events["stop_sequence"] = pd.to_numeric(
        events["stop_sequence"],
        errors="coerce",
    ).astype("Int64")

    speed_summary["avg_speed"] = pd.to_numeric(
        speed_summary["avg_speed"],
        errors="coerce",
    )
    speed_summary["median_speed"] = pd.to_numeric(
        speed_summary["median_speed"],
        errors="coerce",
    )

    dataset_segments = events[
        events["activity_type"].isin(["departure_stop", "arrive_stop"])
    ].copy()

    dataset_segments = dataset_segments.dropna(
        subset=["trip_id", "trip_id_org", "timestamp", "shape_dist_traveled"]
    ).sort_values(["trip_id", "timestamp"]).reset_index(drop=True)

    dataset_segments["next_activity_type"] = (
        dataset_segments.groupby("trip_id")["activity_type"].shift(-1)
    )
    dataset_segments["next_timestamp"] = (
        dataset_segments.groupby("trip_id")["timestamp"].shift(-1)
    )
    dataset_segments["next_shape_dist_traveled"] = (
        dataset_segments.groupby("trip_id")["shape_dist_traveled"].shift(-1)
    )

    dataset_segments = dataset_segments[
        (dataset_segments["activity_type"] == "departure_stop") &
        (dataset_segments["next_activity_type"] == "arrive_stop")
    ].copy()

    dataset_segments["distance_diff"] = (
        dataset_segments["next_shape_dist_traveled"] -
        dataset_segments["shape_dist_traveled"]
    )
    dataset_segments["duration_sec"] = (
        dataset_segments["next_timestamp"] -
        dataset_segments["timestamp"]
    ).dt.total_seconds()
    dataset_segments["duration_min"] = dataset_segments["duration_sec"] / 60

    dataset_segments = dataset_segments[
        (dataset_segments["distance_diff"] >= 0) &
        (dataset_segments["duration_sec"] > 0)
    ].copy()

    dataset_segments["distance_diff_km"] = dataset_segments["distance_diff"] / 1000
    dataset_segments = add_route_type_factors(dataset_segments)

    dataset_segments["consumption_actual"] = (
        dataset_segments["distance_diff_km"] / 100 *
        dataset_segments["consumption_per_100km"]
    )
    dataset_segments["emission_actual"] = calculate_vehicle_emission(
        dataset_segments,
        "consumption_actual",
    )

    for speed_kmh in CONSTANT_SPEED_KMH_VALUES:
        speed_ms = speed_kmh / 3.6
        distance_column = f"distance_from_{speed_kmh}_kmh_km"
        consumption_column = f"consumption_{speed_kmh}_kmh"
        emission_column = f"emission_{speed_kmh}_kmh"

        dataset_segments[distance_column] = (
            speed_ms * dataset_segments["duration_sec"] / 1000
        )
        dataset_segments[consumption_column] = (
            dataset_segments[distance_column] / 100 *
            dataset_segments["consumption_per_100km"]
        )
        dataset_segments[emission_column] = calculate_vehicle_emission(
            dataset_segments,
            consumption_column,
        )

    dataset_segments["duration_rule_20_40_speed_kmh"] = 40
    dataset_segments.loc[
        dataset_segments["duration_min"] < 5,
        "duration_rule_20_40_speed_kmh",
    ] = 20

    dataset_segments["distance_from_duration_rule_20_40_km"] = (
        dataset_segments["duration_rule_20_40_speed_kmh"] / 3.6 *
        dataset_segments["duration_sec"] / 1000
    )
    dataset_segments["consumption_duration_rule_20_40"] = (
        dataset_segments["distance_from_duration_rule_20_40_km"] / 100 *
        dataset_segments["consumption_per_100km"]
    )
    dataset_segments["emission_duration_rule_20_40"] = calculate_vehicle_emission(
        dataset_segments,
        "consumption_duration_rule_20_40",
    )

    segment_method_columns = [
        ("20 km/h", "emission_20_kmh"),
        ("40 km/h", "emission_40_kmh"),
        ("60 km/h", "emission_60_kmh"),
        ("Segment duration <5 min: 20 km/h, otherwise 40 km/h", "emission_duration_rule_20_40"),
    ]

    dataset_segments = dataset_segments[
        dataset_segments["emission_actual"] > 0
    ].copy()
    dataset_segments["segment_row_id"] = (
        dataset_date + "_" + dataset_segments.index.astype(str)
    )

    segment_method_rows = []
    for method_label, emission_column in segment_method_columns:
        method_df = dataset_segments[[
            "segment_row_id",
            "emission_actual",
            emission_column,
        ]].copy()
        method_df["Dataset"] = format_kodak_dataset_label(dataset_date)
        method_df["Method"] = method_label
        method_df["Signed percentage error"] = (
            (method_df[emission_column] - method_df["emission_actual"]) /
            method_df["emission_actual"] * 100
        )
        method_df["Absolute percentage error"] = method_df["Signed percentage error"].abs()
        segment_method_rows.append(method_df)

    segment_error_long = pd.concat(segment_method_rows, ignore_index=True)

    trip_emissions = (
        dataset_segments.groupby("trip_id", as_index=False)
        .agg(
            total_emission_actual=("emission_actual", "sum"),
            total_emission_20_kmh=("emission_20_kmh", "sum"),
            total_emission_40_kmh=("emission_40_kmh", "sum"),
            total_emission_60_kmh=("emission_60_kmh", "sum"),
            total_emission_duration_rule_20_40=("emission_duration_rule_20_40", "sum"),
        )
    )

    trip_emissions = trip_emissions[
        trip_emissions["total_emission_actual"] > 0
    ].copy()

    method_columns = [
        ("20 km/h", "total_emission_20_kmh"),
        ("40 km/h", "total_emission_40_kmh"),
        ("60 km/h", "total_emission_60_kmh"),
        ("Segment duration <5 min: 20 km/h, otherwise 40 km/h", "total_emission_duration_rule_20_40"),
    ]

    method_rows = []
    for method_label, emission_column in method_columns:
        method_df = trip_emissions[["trip_id", "total_emission_actual", emission_column]].copy()
        method_df["Dataset"] = format_kodak_dataset_label(dataset_date)
        method_df["Method"] = method_label
        method_df["Signed percentage error"] = (
            (method_df[emission_column] - method_df["total_emission_actual"]) /
            method_df["total_emission_actual"] * 100
        )
        method_df["Absolute percentage error"] = method_df["Signed percentage error"].abs()
        method_rows.append(method_df)

    trip_error_long = pd.concat(method_rows, ignore_index=True)

    return segment_error_long, trip_error_long


all_kodak_error_tables = [
    build_constant_speed_error_tables_for_dataset(dataset_date)
    for dataset_date in ALL_KODAK_DATASET_DATES
]

all_kodak_segment_error_long = pd.concat(
    [segment_error_long for segment_error_long, _ in all_kodak_error_tables],
    ignore_index=True,
)

all_kodak_trip_error_long = pd.concat(
    [trip_error_long for _, trip_error_long in all_kodak_error_tables],
    ignore_index=True,
)

all_kodak_segment_error_summary = (
    all_kodak_segment_error_long
    .groupby(["Dataset", "Method"], as_index=False)
    .agg(
        segment_count=("segment_row_id", "nunique"),
        mean_absolute_percentage_error=("Absolute percentage error", "mean"),
        mean_signed_percentage_error=("Signed percentage error", "mean"),
        signed_percentage_error_std=("Signed percentage error", "std"),
    )
)

all_kodak_trip_error_summary = (
    all_kodak_trip_error_long
    .groupby(["Dataset", "Method"], as_index=False)
    .agg(
        trip_count=("trip_id", "nunique"),
        mean_absolute_percentage_error=("Absolute percentage error", "mean"),
        mean_signed_percentage_error=("Signed percentage error", "mean"),
        signed_percentage_error_std=("Signed percentage error", "std"),
    )
)

all_kodak_segment_error_summary.to_csv(
    ALL_KODAK_OUTPUT_DIR / "constant_speed_segment_percentage_error_by_method_all_kodak.csv",
    index=False,
)

all_kodak_trip_error_summary.to_csv(
    ALL_KODAK_OUTPUT_DIR / "constant_speed_trip_percentage_error_by_method_all_kodak.csv",
    index=False,
)

all_kodak_segment_mean_percentage_error_table = all_kodak_segment_error_summary[
    ["Dataset", "Method", "segment_count", "mean_signed_percentage_error"]
].copy()
all_kodak_segment_mean_percentage_error_table.to_csv(
    ALL_KODAK_OUTPUT_DIR / "constant_speed_segment_mean_percentage_error_table_all_kodak.csv",
    index=False,
)
all_kodak_segment_mean_percentage_error_table.to_latex(
    ALL_KODAK_OUTPUT_DIR / "constant_speed_segment_mean_percentage_error_table_all_kodak.tex",
    index=False,
    float_format="%.2f",
    caption="Segment-level mean percentage error by constant-speed method.",
    label="tab:constant_speed_segment_mpe_all_kodak",
)

all_kodak_trip_mean_percentage_error_table = all_kodak_trip_error_summary[
    ["Dataset", "Method", "trip_count", "mean_signed_percentage_error"]
].copy()
all_kodak_trip_mean_percentage_error_table.to_csv(
    ALL_KODAK_OUTPUT_DIR / "constant_speed_trip_mean_percentage_error_table_all_kodak.csv",
    index=False,
)
all_kodak_trip_mean_percentage_error_table.to_latex(
    ALL_KODAK_OUTPUT_DIR / "constant_speed_trip_mean_percentage_error_table_all_kodak.tex",
    index=False,
    float_format="%.2f",
    caption="Trip-level mean percentage error by constant-speed method.",
    label="tab:constant_speed_trip_mpe_all_kodak",
)

method_order = [
    "20 km/h",
    "40 km/h",
    "60 km/h",
    "Segment duration <5 min: 20 km/h, otherwise 40 km/h",
]

segment_error_metrics = [
    (
        "mean_absolute_percentage_error",
        "Mean absolute percentage error (%)",
        "Segment-Level MAPE by Constant-Speed Method",
        "constant_speed_segment_mape_by_method_all_kodak.png",
    ),
    (
        "mean_signed_percentage_error",
        "Mean signed percentage error (%)",
        "Segment-Level Mean Signed Percentage Error by Constant-Speed Method",
        "constant_speed_segment_mean_signed_error_by_method_all_kodak.png",
    ),
    (
        "signed_percentage_error_std",
        "Std. dev. of signed percentage error (%)",
        "Segment-Level Percentage Error Standard Deviation by Constant-Speed Method",
        "constant_speed_segment_signed_error_std_by_method_all_kodak.png",
    ),
]

trip_error_metrics = [
    (
        "mean_absolute_percentage_error",
        "Mean absolute percentage error (%)",
        "Trip-Level MAPE by Constant-Speed Method",
        "constant_speed_trip_mape_by_method_all_kodak.png",
    ),
    (
        "mean_signed_percentage_error",
        "Mean signed percentage error (%)",
        "Trip-Level Mean Signed Percentage Error by Constant-Speed Method",
        "constant_speed_trip_mean_signed_error_by_method_all_kodak.png",
    ),
    (
        "signed_percentage_error_std",
        "Std. dev. of signed percentage error (%)",
        "Trip-Level Percentage Error Standard Deviation by Constant-Speed Method",
        "constant_speed_trip_signed_error_std_by_method_all_kodak.png",
    ),
]

def save_constant_speed_error_charts(summary_df, metrics, combined_filename):
    fig, axes = plt.subplots(
        len(metrics),
        1,
        figsize=(14, 16),
        sharex=True,
    )

    for ax, (metric_column, y_label, title, _) in zip(axes, metrics):
        sns.barplot(
            data=summary_df,
            x="Method",
            y=metric_column,
            hue="Dataset",
            order=method_order,
            ax=ax,
        )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel(y_label)
        ax.legend(title="Dataset", ncol=2, fontsize=8, title_fontsize=9)

    axes[-1].set_xlabel("Constant-speed method")
    axes[-1].tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(
        ALL_KODAK_OUTPUT_DIR / combined_filename,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    for metric_column, y_label, title, filename in metrics:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(
            data=summary_df,
            x="Method",
            y=metric_column,
            hue="Dataset",
            order=method_order,
            ax=ax,
        )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("Constant-speed method")
        ax.set_ylabel(y_label)
        ax.tick_params(axis="x", rotation=20)
        ax.legend(title="Dataset", ncol=2, fontsize=8, title_fontsize=9)
        plt.tight_layout()
        plt.savefig(
            ALL_KODAK_OUTPUT_DIR / filename,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


save_constant_speed_error_charts(
    all_kodak_segment_error_summary,
    segment_error_metrics,
    "constant_speed_segment_percentage_error_by_method_all_kodak.png",
)

save_constant_speed_error_charts(
    all_kodak_trip_error_summary,
    trip_error_metrics,
    "constant_speed_trip_percentage_error_by_method_all_kodak.png",
)
