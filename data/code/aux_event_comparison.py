from pathlib import Path
import os
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate


# Input files

DATASETS = [
    {
        "name": "kodak_2026_02_04",
        "aux_file": Path("aux_event_log_overview_kodak_2026_02_04.csv"),
        "speed_file": Path("trip_speed_summary_2026_02_04.csv"),
    },
    {
        "name": "kodak_2026_05_04",
        "aux_file": Path("aux_event_log_overview_kodak_2026_05_04.csv"),
        "speed_file": Path("trip_speed_summary_2026_05_04.csv"),
    },
    {
        "name": "kodak_2025_08_04",
        "aux_file": Path("aux_event_log_overview_kodak_2025_08_04.csv"),
        "speed_file": Path("trip_speed_summary_2025_08_04.csv"),
    },
    {
        "name": "kodak_2025_11_04",
        "aux_file": Path("aux_event_log_overview_kodak_2025_11_04.csv"),
        "speed_file": Path("trip_speed_summary_2025_11_04.csv"),
    }
]

OUTPUT_DIR = Path("distance_speed_duration_emission_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Constants

CONSUMPTION_PER_KM = 2.5
EMISSION_FACTOR = 0.4

# Use "m_per_s" if avg_speed * duration_sec gives meters.
# Use "km_per_h" if avg_speed is km/h.
SPEED_UNIT = "m_per_s"


# Helper functions

def estimate_distance_km(speed, duration_sec):
    if SPEED_UNIT == "m_per_s":
        return (speed * duration_sec) / 1000

    if SPEED_UNIT == "km_per_h":
        return speed * (duration_sec / 3600)

    raise ValueError("SPEED_UNIT must be 'm_per_s' or 'km_per_h'")


def safe_pct_error(estimated, actual):
    return ((estimated - actual) / actual) * 100


def load_aux_event_log(file_path):
    df = pd.read_csv(
        file_path,
        dtype={
            "vehicle_id": "string",
            "trip_id": "string",
            "trip_id_org": "string",
            "stop_sequence": "Int64",
        },
        low_memory=False,
    )

    df.columns = df.columns.str.strip()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    df["shape_dist_traveled"] = pd.to_numeric(
        df["shape_dist_traveled"],
        errors="coerce",
    )

    df["stop_sequence"] = pd.to_numeric(
        df["stop_sequence"],
        errors="coerce",
    ).astype("Int64")

    df = df.dropna(
        subset=[
            "trip_id",
            "trip_id_org",
            "timestamp",
            "shape_dist_traveled",
            "activity_type",
        ]
    ).copy()

    df["trip_id"] = df["trip_id"].astype("string")
    df["trip_id_org"] = df["trip_id_org"].astype("string")

    return df


def load_trip_speed_summary(file_path):
    df = pd.read_csv(file_path, dtype={"trip_id": "string"}, low_memory=False)
    df.columns = df.columns.str.strip()

    if "trip_id_org" not in df.columns and "trip_id" in df.columns:
        df = df.rename(columns={"trip_id": "trip_id_org"})

    if "median_speed" not in df.columns and "med_speed" in df.columns:
        df = df.rename(columns={"med_speed": "median_speed"})

    required_cols = [
        "trip_id_org",
        "avg_speed",
        "median_speed",
        "speed_q25",
        "speed_q58",
        "speed_q60",
        "speed_q625",
        "speed_q75",
    ]
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise KeyError(
            f"In {file_path} fehlen diese Spalten: {missing}\n"
            f"Vorhandene Spalten: {df.columns.tolist()}"
        )

    df["trip_id_org"] = df["trip_id_org"].astype("string")
    df["avg_speed"] = pd.to_numeric(df["avg_speed"], errors="coerce")
    df["median_speed"] = pd.to_numeric(df["median_speed"], errors="coerce")
    df["speed_q25"] = pd.to_numeric(df["speed_q25"], errors="coerce")
    df["speed_q58"] = pd.to_numeric(df["speed_q58"], errors="coerce")
    df["speed_q60"] = pd.to_numeric(df["speed_q60"], errors="coerce")
    df["speed_q625"] = pd.to_numeric(df["speed_q625"], errors="coerce")
    df["speed_q75"] = pd.to_numeric(df["speed_q75"], errors="coerce")

    return df


def build_segments(aux_df, speed_df):
    segment_df = aux_df[
        aux_df["activity_type"].isin(["departure_stop", "arrive_stop"])
    ].copy()

    segment_df = segment_df.sort_values(
        ["trip_id", "timestamp"]
    ).reset_index(drop=True)

    segment_df["next_activity_type"] = (
        segment_df.groupby("trip_id")["activity_type"].shift(-1)
    )

    segment_df["next_timestamp"] = (
        segment_df.groupby("trip_id")["timestamp"].shift(-1)
    )

    segment_df["next_shape_dist_traveled"] = (
        segment_df.groupby("trip_id")["shape_dist_traveled"].shift(-1)
    )

    segment_df["next_stop_sequence"] = (
        segment_df.groupby("trip_id")["stop_sequence"].shift(-1)
    )

    segment_df["next_stop_name"] = (
        segment_df.groupby("trip_id")["stop_name"].shift(-1)
    )

    segments = segment_df[
        (segment_df["activity_type"] == "departure_stop")
        & (segment_df["next_activity_type"] == "arrive_stop")
    ].copy()

    segments["duration_sec"] = (
        segments["next_timestamp"] - segments["timestamp"]
    ).dt.total_seconds()

    segments["duration_min"] = segments["duration_sec"] / 60

    segments["distance_diff_m"] = (
        segments["next_shape_dist_traveled"]
        - segments["shape_dist_traveled"]
    )

    segments["distance_diff_km"] = segments["distance_diff_m"] / 1000

    segments = segments[
        (segments["duration_sec"] > 0)
        & (segments["distance_diff_km"] > 0)
    ].copy()

    segments["segment_id"] = (
        segments["trip_id"].astype(str)
        + "_"
        + segments["stop_sequence"].astype("Int64").astype(str)
    )

    segment_summary = segments[
        [
            "segment_id",
            "trip_id",
            "trip_id_org",
            "vehicle_id",
            "route_id",
            "route_short_name",
            "stop_sequence",
            "stop_name",
            "next_stop_sequence",
            "next_stop_name",
            "shape_dist_traveled",
            "next_shape_dist_traveled",
            "distance_diff_m",
            "distance_diff_km",
            "duration_sec",
            "duration_min",
        ]
    ].rename(
        columns={
            "stop_sequence": "departure_stop_sequence",
            "stop_name": "departure_stop_name",
            "next_stop_sequence": "arrive_stop_sequence",
            "next_stop_name": "arrive_stop_name",
            "shape_dist_traveled": "departure_shape_dist_traveled",
            "next_shape_dist_traveled": "arrive_shape_dist_traveled",
        }
    )

    segment_summary = segment_summary.merge(
        speed_df[
            [
                "trip_id_org",
                "avg_speed",
                "median_speed",
                "speed_q25",
                "speed_q58",
                "speed_q60",
                "speed_q625",
                "speed_q75",
            ]
        ],
        on="trip_id_org",
        how="left",
    )

    segment_summary = segment_summary.dropna(
        subset=[
            "avg_speed",
            "median_speed",
            "speed_q25",
            "speed_q58",
            "speed_q60",
            "speed_q625",
            "speed_q75",
        ]
    ).copy()

    return segment_summary


def calculate_distance_and_emissions(segment_summary):
    df = segment_summary.copy()

    df["distance_from_avg_speed_km"] = estimate_distance_km(
        df["avg_speed"],
        df["duration_sec"],
    )

    df["distance_from_median_speed_km"] = estimate_distance_km(
        df["median_speed"],
        df["duration_sec"],
    )

    df["distance_from_q25_speed_km"] = estimate_distance_km(
        df["speed_q25"],
        df["duration_sec"],
    )

    df["distance_from_q58_speed_km"] = estimate_distance_km(
        df["speed_q58"],
        df["duration_sec"],
    )

    df["distance_from_q60_speed_km"] = estimate_distance_km(
        df["speed_q60"],
        df["duration_sec"],
    )

    df["distance_from_q625_speed_km"] = estimate_distance_km(
        df["speed_q625"],
        df["duration_sec"],
    )

    df["distance_from_q75_speed_km"] = estimate_distance_km(
        df["speed_q75"],
        df["duration_sec"],
    )

    df["distance_error_avg_speed_km"] = (
        df["distance_from_avg_speed_km"] - df["distance_diff_km"]
    )

    df["distance_error_median_speed_km"] = (
        df["distance_from_median_speed_km"] - df["distance_diff_km"]
    )

    df["distance_error_q25_speed_km"] = (
        df["distance_from_q25_speed_km"] - df["distance_diff_km"]
    )

    df["distance_error_q58_speed_km"] = (
        df["distance_from_q58_speed_km"] - df["distance_diff_km"]
    )

    df["distance_error_q60_speed_km"] = (
        df["distance_from_q60_speed_km"] - df["distance_diff_km"]
    )

    df["distance_error_q625_speed_km"] = (
        df["distance_from_q625_speed_km"] - df["distance_diff_km"]
    )

    df["distance_error_q75_speed_km"] = (
        df["distance_from_q75_speed_km"] - df["distance_diff_km"]
    )

    df["distance_abs_error_avg_speed_km"] = (
        df["distance_error_avg_speed_km"].abs()
    )

    df["distance_abs_error_median_speed_km"] = (
        df["distance_error_median_speed_km"].abs()
    )

    df["distance_abs_error_q25_speed_km"] = (
        df["distance_error_q25_speed_km"].abs()
    )

    df["distance_abs_error_q58_speed_km"] = (
        df["distance_error_q58_speed_km"].abs()
    )

    df["distance_abs_error_q60_speed_km"] = (
        df["distance_error_q60_speed_km"].abs()
    )

    df["distance_abs_error_q625_speed_km"] = (
        df["distance_error_q625_speed_km"].abs()
    )

    df["distance_abs_error_q75_speed_km"] = (
        df["distance_error_q75_speed_km"].abs()
    )

    df["distance_pct_error_avg_speed"] = safe_pct_error(
        df["distance_from_avg_speed_km"],
        df["distance_diff_km"],
    )

    df["distance_pct_error_median_speed"] = safe_pct_error(
        df["distance_from_median_speed_km"],
        df["distance_diff_km"],
    )

    df["distance_pct_error_q25_speed"] = safe_pct_error(
        df["distance_from_q25_speed_km"],
        df["distance_diff_km"],
    )

    df["distance_pct_error_q58_speed"] = safe_pct_error(
        df["distance_from_q58_speed_km"],
        df["distance_diff_km"],
    )

    df["distance_pct_error_q60_speed"] = safe_pct_error(
        df["distance_from_q60_speed_km"],
        df["distance_diff_km"],
    )

    df["distance_pct_error_q625_speed"] = safe_pct_error(
        df["distance_from_q625_speed_km"],
        df["distance_diff_km"],
    )

    df["distance_pct_error_q75_speed"] = safe_pct_error(
        df["distance_from_q75_speed_km"],
        df["distance_diff_km"],
    )

    df["distance_abs_pct_error_avg_speed"] = (
        df["distance_pct_error_avg_speed"].abs()
    )

    df["distance_abs_pct_error_median_speed"] = (
        df["distance_pct_error_median_speed"].abs()
    )

    df["distance_abs_pct_error_q25_speed"] = (
        df["distance_pct_error_q25_speed"].abs()
    )

    df["distance_abs_pct_error_q58_speed"] = (
        df["distance_pct_error_q58_speed"].abs()
    )

    df["distance_abs_pct_error_q60_speed"] = (
        df["distance_pct_error_q60_speed"].abs()
    )

    df["distance_abs_pct_error_q625_speed"] = (
        df["distance_pct_error_q625_speed"].abs()
    )

    df["distance_abs_pct_error_q75_speed"] = (
        df["distance_pct_error_q75_speed"].abs()
    )

    df["emission_actual"] = (
        df["distance_diff_km"] * CONSUMPTION_PER_KM * EMISSION_FACTOR
    )

    df["emission_avg_speed"] = (
        df["distance_from_avg_speed_km"] * CONSUMPTION_PER_KM * EMISSION_FACTOR
    )

    df["emission_median_speed"] = (
        df["distance_from_median_speed_km"] * CONSUMPTION_PER_KM * EMISSION_FACTOR
    )

    df["emission_q25_speed"] = (
        df["distance_from_q25_speed_km"] * CONSUMPTION_PER_KM * EMISSION_FACTOR
    )

    df["emission_q58_speed"] = (
        df["distance_from_q58_speed_km"] * CONSUMPTION_PER_KM * EMISSION_FACTOR
    )

    df["emission_q60_speed"] = (
        df["distance_from_q60_speed_km"] * CONSUMPTION_PER_KM * EMISSION_FACTOR
    )

    df["emission_q625_speed"] = (
        df["distance_from_q625_speed_km"] * CONSUMPTION_PER_KM * EMISSION_FACTOR
    )

    df["emission_q75_speed"] = (
        df["distance_from_q75_speed_km"] * CONSUMPTION_PER_KM * EMISSION_FACTOR
    )

    df["emission_error_avg_speed"] = (
        df["emission_avg_speed"] - df["emission_actual"]
    )

    df["emission_error_median_speed"] = (
        df["emission_median_speed"] - df["emission_actual"]
    )

    df["emission_error_q25_speed"] = (
        df["emission_q25_speed"] - df["emission_actual"]
    )

    df["emission_error_q58_speed"] = (
        df["emission_q58_speed"] - df["emission_actual"]
    )

    df["emission_error_q60_speed"] = (
        df["emission_q60_speed"] - df["emission_actual"]
    )

    df["emission_error_q625_speed"] = (
        df["emission_q625_speed"] - df["emission_actual"]
    )

    df["emission_error_q75_speed"] = (
        df["emission_q75_speed"] - df["emission_actual"]
    )

    df["emission_abs_error_avg_speed"] = (
        df["emission_error_avg_speed"].abs()
    )

    df["emission_abs_error_median_speed"] = (
        df["emission_error_median_speed"].abs()
    )

    df["emission_abs_error_q25_speed"] = (
        df["emission_error_q25_speed"].abs()
    )

    df["emission_abs_error_q58_speed"] = (
        df["emission_error_q58_speed"].abs()
    )

    df["emission_abs_error_q60_speed"] = (
        df["emission_error_q60_speed"].abs()
    )

    df["emission_abs_error_q625_speed"] = (
        df["emission_error_q625_speed"].abs()
    )

    df["emission_abs_error_q75_speed"] = (
        df["emission_error_q75_speed"].abs()
    )

    df["emission_pct_error_avg_speed"] = safe_pct_error(
        df["emission_avg_speed"],
        df["emission_actual"],
    )

    df["emission_pct_error_median_speed"] = safe_pct_error(
        df["emission_median_speed"],
        df["emission_actual"],
    )

    df["emission_pct_error_q25_speed"] = safe_pct_error(
        df["emission_q25_speed"],
        df["emission_actual"],
    )

    df["emission_pct_error_q58_speed"] = safe_pct_error(
        df["emission_q58_speed"],
        df["emission_actual"],
    )

    df["emission_pct_error_q60_speed"] = safe_pct_error(
        df["emission_q60_speed"],
        df["emission_actual"],
    )

    df["emission_pct_error_q625_speed"] = safe_pct_error(
        df["emission_q625_speed"],
        df["emission_actual"],
    )

    df["emission_pct_error_q75_speed"] = safe_pct_error(
        df["emission_q75_speed"],
        df["emission_actual"],
    )

    df["emission_abs_pct_error_avg_speed"] = (
        df["emission_pct_error_avg_speed"].abs()
    )

    df["emission_abs_pct_error_median_speed"] = (
        df["emission_pct_error_median_speed"].abs()
    )

    df["emission_abs_pct_error_q25_speed"] = (
        df["emission_pct_error_q25_speed"].abs()
    )

    df["emission_abs_pct_error_q58_speed"] = (
        df["emission_pct_error_q58_speed"].abs()
    )

    df["emission_abs_pct_error_q60_speed"] = (
        df["emission_pct_error_q60_speed"].abs()
    )

    df["emission_abs_pct_error_q625_speed"] = (
        df["emission_pct_error_q625_speed"].abs()
    )

    df["emission_abs_pct_error_q75_speed"] = (
        df["emission_pct_error_q75_speed"].abs()
    )

    df = df.replace([float("inf"), float("-inf")], pd.NA)

    return df


def build_trip_summary(segment_df):
    trip_df = (
        segment_df.groupby(["trip_id", "trip_id_org"], as_index=False)
        .agg(
            segment_count=("segment_id", "count"),
            total_duration_sec=("duration_sec", "sum"),
            total_duration_min=("duration_min", "sum"),
            total_distance_actual_km=("distance_diff_km", "sum"),
            total_distance_avg_speed_km=("distance_from_avg_speed_km", "sum"),
            total_distance_median_speed_km=("distance_from_median_speed_km", "sum"),
            total_distance_q25_speed_km=("distance_from_q25_speed_km", "sum"),
            total_distance_q58_speed_km=("distance_from_q58_speed_km", "sum"),
            total_distance_q60_speed_km=("distance_from_q60_speed_km", "sum"),
            total_distance_q625_speed_km=("distance_from_q625_speed_km", "sum"),
            total_distance_q75_speed_km=("distance_from_q75_speed_km", "sum"),
            total_emission_actual=("emission_actual", "sum"),
            total_emission_avg_speed=("emission_avg_speed", "sum"),
            total_emission_median_speed=("emission_median_speed", "sum"),
            total_emission_q25_speed=("emission_q25_speed", "sum"),
            total_emission_q58_speed=("emission_q58_speed", "sum"),
            total_emission_q60_speed=("emission_q60_speed", "sum"),
            total_emission_q625_speed=("emission_q625_speed", "sum"),
            total_emission_q75_speed=("emission_q75_speed", "sum"),
        )
    )

    trip_df["distance_error_avg_speed_km"] = (
        trip_df["total_distance_avg_speed_km"]
        - trip_df["total_distance_actual_km"]
    )

    trip_df["distance_error_median_speed_km"] = (
        trip_df["total_distance_median_speed_km"]
        - trip_df["total_distance_actual_km"]
    )

    trip_df["distance_error_q25_speed_km"] = (
        trip_df["total_distance_q25_speed_km"]
        - trip_df["total_distance_actual_km"]
    )

    trip_df["distance_error_q58_speed_km"] = (
        trip_df["total_distance_q58_speed_km"]
        - trip_df["total_distance_actual_km"]
    )

    trip_df["distance_error_q60_speed_km"] = (
        trip_df["total_distance_q60_speed_km"]
        - trip_df["total_distance_actual_km"]
    )

    trip_df["distance_error_q625_speed_km"] = (
        trip_df["total_distance_q625_speed_km"]
        - trip_df["total_distance_actual_km"]
    )

    trip_df["distance_error_q75_speed_km"] = (
        trip_df["total_distance_q75_speed_km"]
        - trip_df["total_distance_actual_km"]
    )

    trip_df["distance_pct_error_avg_speed"] = safe_pct_error(
        trip_df["total_distance_avg_speed_km"],
        trip_df["total_distance_actual_km"],
    )

    trip_df["distance_pct_error_median_speed"] = safe_pct_error(
        trip_df["total_distance_median_speed_km"],
        trip_df["total_distance_actual_km"],
    )

    trip_df["distance_pct_error_q25_speed"] = safe_pct_error(
        trip_df["total_distance_q25_speed_km"],
        trip_df["total_distance_actual_km"],
    )

    trip_df["distance_pct_error_q58_speed"] = safe_pct_error(
        trip_df["total_distance_q58_speed_km"],
        trip_df["total_distance_actual_km"],
    )

    trip_df["distance_pct_error_q60_speed"] = safe_pct_error(
        trip_df["total_distance_q60_speed_km"],
        trip_df["total_distance_actual_km"],
    )

    trip_df["distance_pct_error_q625_speed"] = safe_pct_error(
        trip_df["total_distance_q625_speed_km"],
        trip_df["total_distance_actual_km"],
    )

    trip_df["distance_pct_error_q75_speed"] = safe_pct_error(
        trip_df["total_distance_q75_speed_km"],
        trip_df["total_distance_actual_km"],
    )

    trip_df["emission_error_avg_speed"] = (
        trip_df["total_emission_avg_speed"]
        - trip_df["total_emission_actual"]
    )

    trip_df["emission_error_median_speed"] = (
        trip_df["total_emission_median_speed"]
        - trip_df["total_emission_actual"]
    )

    trip_df["emission_error_q25_speed"] = (
        trip_df["total_emission_q25_speed"]
        - trip_df["total_emission_actual"]
    )

    trip_df["emission_error_q58_speed"] = (
        trip_df["total_emission_q58_speed"]
        - trip_df["total_emission_actual"]
    )

    trip_df["emission_error_q60_speed"] = (
        trip_df["total_emission_q60_speed"]
        - trip_df["total_emission_actual"]
    )

    trip_df["emission_error_q625_speed"] = (
        trip_df["total_emission_q625_speed"]
        - trip_df["total_emission_actual"]
    )

    trip_df["emission_error_q75_speed"] = (
        trip_df["total_emission_q75_speed"]
        - trip_df["total_emission_actual"]
    )

    trip_df["emission_pct_error_avg_speed"] = safe_pct_error(
        trip_df["total_emission_avg_speed"],
        trip_df["total_emission_actual"],
    )

    trip_df["emission_pct_error_median_speed"] = safe_pct_error(
        trip_df["total_emission_median_speed"],
        trip_df["total_emission_actual"],
    )

    trip_df["emission_pct_error_q25_speed"] = safe_pct_error(
        trip_df["total_emission_q25_speed"],
        trip_df["total_emission_actual"],
    )

    trip_df["emission_pct_error_q58_speed"] = safe_pct_error(
        trip_df["total_emission_q58_speed"],
        trip_df["total_emission_actual"],
    )

    trip_df["emission_pct_error_q60_speed"] = safe_pct_error(
        trip_df["total_emission_q60_speed"],
        trip_df["total_emission_actual"],
    )

    trip_df["emission_pct_error_q625_speed"] = safe_pct_error(
        trip_df["total_emission_q625_speed"],
        trip_df["total_emission_actual"],
    )

    trip_df["emission_pct_error_q75_speed"] = safe_pct_error(
        trip_df["total_emission_q75_speed"],
        trip_df["total_emission_actual"],
    )

    trip_df = trip_df.replace([float("inf"), float("-inf")], pd.NA)

    return trip_df


def percentage_with_abs_pct_error_above(df, column, threshold=1.0):
    values = df[column].dropna().abs()

    if values.empty:
        return pd.NA

    return (values > threshold).mean() * 100


def summarize_deviation(df, dataset_name, level):
    return {
        "dataset": dataset_name,
        "level": level,
        "rows": len(df),

        "actual_distance_km_sum": df["distance_diff_km"].sum()
        if level == "segment" else df["total_distance_actual_km"].sum(),

        "avg_speed_distance_km_sum": df["distance_from_avg_speed_km"].sum()
        if level == "segment" else df["total_distance_avg_speed_km"].sum(),

        "median_speed_distance_km_sum": df["distance_from_median_speed_km"].sum()
        if level == "segment" else df["total_distance_median_speed_km"].sum(),

        "q25_speed_distance_km_sum": df["distance_from_q25_speed_km"].sum()
        if level == "segment" else df["total_distance_q25_speed_km"].sum(),

        "q58_speed_distance_km_sum": df["distance_from_q58_speed_km"].sum()
        if level == "segment" else df["total_distance_q58_speed_km"].sum(),

        "q60_speed_distance_km_sum": df["distance_from_q60_speed_km"].sum()
        if level == "segment" else df["total_distance_q60_speed_km"].sum(),

        "q625_speed_distance_km_sum": df["distance_from_q625_speed_km"].sum()
        if level == "segment" else df["total_distance_q625_speed_km"].sum(),

        "q75_speed_distance_km_sum": df["distance_from_q75_speed_km"].sum()
        if level == "segment" else df["total_distance_q75_speed_km"].sum(),

        "avg_speed_distance_pct_error_mean": df["distance_pct_error_avg_speed"].mean(),
        "avg_speed_distance_pct_error_median": df["distance_pct_error_avg_speed"].median(),
        "avg_speed_distance_pct_error_std": df["distance_pct_error_avg_speed"].std(),
        "avg_speed_distance_abs_pct_error_mean": df["distance_pct_error_avg_speed"].abs().mean(),
        "avg_speed_distance_abs_pct_error_gt_1pct_share": percentage_with_abs_pct_error_above(
            df,
            "distance_pct_error_avg_speed",
        ),

        "median_speed_distance_pct_error_mean": df["distance_pct_error_median_speed"].mean(),
        "median_speed_distance_pct_error_median": df["distance_pct_error_median_speed"].median(),
        "median_speed_distance_pct_error_std": df["distance_pct_error_median_speed"].std(),
        "median_speed_distance_abs_pct_error_mean": df["distance_pct_error_median_speed"].abs().mean(),
        "median_speed_distance_abs_pct_error_gt_1pct_share": percentage_with_abs_pct_error_above(
            df,
            "distance_pct_error_median_speed",
        ),

        "q25_speed_distance_pct_error_mean": df["distance_pct_error_q25_speed"].mean(),
        "q25_speed_distance_pct_error_median": df["distance_pct_error_q25_speed"].median(),
        "q25_speed_distance_pct_error_std": df["distance_pct_error_q25_speed"].std(),
        "q25_speed_distance_abs_pct_error_mean": df["distance_pct_error_q25_speed"].abs().mean(),
        "q25_speed_distance_abs_pct_error_gt_1pct_share": percentage_with_abs_pct_error_above(
            df,
            "distance_pct_error_q25_speed",
        ),

        "q58_speed_distance_pct_error_mean": df["distance_pct_error_q58_speed"].mean(),
        "q58_speed_distance_pct_error_median": df["distance_pct_error_q58_speed"].median(),
        "q58_speed_distance_pct_error_std": df["distance_pct_error_q58_speed"].std(),
        "q58_speed_distance_abs_pct_error_mean": df["distance_pct_error_q58_speed"].abs().mean(),
        "q58_speed_distance_abs_pct_error_gt_1pct_share": percentage_with_abs_pct_error_above(
            df,
            "distance_pct_error_q58_speed",
        ),

        "q60_speed_distance_pct_error_mean": df["distance_pct_error_q60_speed"].mean(),
        "q60_speed_distance_pct_error_median": df["distance_pct_error_q60_speed"].median(),
        "q60_speed_distance_pct_error_std": df["distance_pct_error_q60_speed"].std(),
        "q60_speed_distance_abs_pct_error_mean": df["distance_pct_error_q60_speed"].abs().mean(),
        "q60_speed_distance_abs_pct_error_gt_1pct_share": percentage_with_abs_pct_error_above(
            df,
            "distance_pct_error_q60_speed",
        ),

        "q625_speed_distance_pct_error_mean": df["distance_pct_error_q625_speed"].mean(),
        "q625_speed_distance_pct_error_median": df["distance_pct_error_q625_speed"].median(),
        "q625_speed_distance_pct_error_std": df["distance_pct_error_q625_speed"].std(),
        "q625_speed_distance_abs_pct_error_mean": df["distance_pct_error_q625_speed"].abs().mean(),
        "q625_speed_distance_abs_pct_error_gt_1pct_share": percentage_with_abs_pct_error_above(
            df,
            "distance_pct_error_q625_speed",
        ),

        "q75_speed_distance_pct_error_mean": df["distance_pct_error_q75_speed"].mean(),
        "q75_speed_distance_pct_error_median": df["distance_pct_error_q75_speed"].median(),
        "q75_speed_distance_pct_error_std": df["distance_pct_error_q75_speed"].std(),
        "q75_speed_distance_abs_pct_error_mean": df["distance_pct_error_q75_speed"].abs().mean(),
        "q75_speed_distance_abs_pct_error_gt_1pct_share": percentage_with_abs_pct_error_above(
            df,
            "distance_pct_error_q75_speed",
        ),

        "actual_emission_sum": df["emission_actual"].sum()
        if level == "segment" else df["total_emission_actual"].sum(),

        "avg_speed_emission_sum": df["emission_avg_speed"].sum()
        if level == "segment" else df["total_emission_avg_speed"].sum(),

        "median_speed_emission_sum": df["emission_median_speed"].sum()
        if level == "segment" else df["total_emission_median_speed"].sum(),

        "q25_speed_emission_sum": df["emission_q25_speed"].sum()
        if level == "segment" else df["total_emission_q25_speed"].sum(),

        "q58_speed_emission_sum": df["emission_q58_speed"].sum()
        if level == "segment" else df["total_emission_q58_speed"].sum(),

        "q60_speed_emission_sum": df["emission_q60_speed"].sum()
        if level == "segment" else df["total_emission_q60_speed"].sum(),

        "q625_speed_emission_sum": df["emission_q625_speed"].sum()
        if level == "segment" else df["total_emission_q625_speed"].sum(),

        "q75_speed_emission_sum": df["emission_q75_speed"].sum()
        if level == "segment" else df["total_emission_q75_speed"].sum(),

        "avg_speed_emission_pct_error_mean": df["emission_pct_error_avg_speed"].mean(),
        "avg_speed_emission_pct_error_median": df["emission_pct_error_avg_speed"].median(),
        "avg_speed_emission_pct_error_std": df["emission_pct_error_avg_speed"].std(),
        "avg_speed_emission_abs_pct_error_mean": df["emission_pct_error_avg_speed"].abs().mean(),
        "avg_speed_emission_abs_pct_error_gt_1pct_share": percentage_with_abs_pct_error_above(
            df,
            "emission_pct_error_avg_speed",
        ),

        "median_speed_emission_pct_error_mean": df["emission_pct_error_median_speed"].mean(),
        "median_speed_emission_pct_error_median": df["emission_pct_error_median_speed"].median(),
        "median_speed_emission_pct_error_std": df["emission_pct_error_median_speed"].std(),
        "median_speed_emission_abs_pct_error_mean": df["emission_pct_error_median_speed"].abs().mean(),
        "median_speed_emission_abs_pct_error_gt_1pct_share": percentage_with_abs_pct_error_above(
            df,
            "emission_pct_error_median_speed",
        ),

        "q25_speed_emission_pct_error_mean": df["emission_pct_error_q25_speed"].mean(),
        "q25_speed_emission_pct_error_median": df["emission_pct_error_q25_speed"].median(),
        "q25_speed_emission_pct_error_std": df["emission_pct_error_q25_speed"].std(),
        "q25_speed_emission_abs_pct_error_mean": df["emission_pct_error_q25_speed"].abs().mean(),
        "q25_speed_emission_abs_pct_error_gt_1pct_share": percentage_with_abs_pct_error_above(
            df,
            "emission_pct_error_q25_speed",
        ),

        "q58_speed_emission_pct_error_mean": df["emission_pct_error_q58_speed"].mean(),
        "q58_speed_emission_pct_error_median": df["emission_pct_error_q58_speed"].median(),
        "q58_speed_emission_pct_error_std": df["emission_pct_error_q58_speed"].std(),
        "q58_speed_emission_abs_pct_error_mean": df["emission_pct_error_q58_speed"].abs().mean(),
        "q58_speed_emission_abs_pct_error_gt_1pct_share": percentage_with_abs_pct_error_above(
            df,
            "emission_pct_error_q58_speed",
        ),

        "q60_speed_emission_pct_error_mean": df["emission_pct_error_q60_speed"].mean(),
        "q60_speed_emission_pct_error_median": df["emission_pct_error_q60_speed"].median(),
        "q60_speed_emission_pct_error_std": df["emission_pct_error_q60_speed"].std(),
        "q60_speed_emission_abs_pct_error_mean": df["emission_pct_error_q60_speed"].abs().mean(),
        "q60_speed_emission_abs_pct_error_gt_1pct_share": percentage_with_abs_pct_error_above(
            df,
            "emission_pct_error_q60_speed",
        ),

        "q625_speed_emission_pct_error_mean": df["emission_pct_error_q625_speed"].mean(),
        "q625_speed_emission_pct_error_median": df["emission_pct_error_q625_speed"].median(),
        "q625_speed_emission_pct_error_std": df["emission_pct_error_q625_speed"].std(),
        "q625_speed_emission_abs_pct_error_mean": df["emission_pct_error_q625_speed"].abs().mean(),
        "q625_speed_emission_abs_pct_error_gt_1pct_share": percentage_with_abs_pct_error_above(
            df,
            "emission_pct_error_q625_speed",
        ),

        "q75_speed_emission_pct_error_mean": df["emission_pct_error_q75_speed"].mean(),
        "q75_speed_emission_pct_error_median": df["emission_pct_error_q75_speed"].median(),
        "q75_speed_emission_pct_error_std": df["emission_pct_error_q75_speed"].std(),
        "q75_speed_emission_abs_pct_error_mean": df["emission_pct_error_q75_speed"].abs().mean(),
        "q75_speed_emission_abs_pct_error_gt_1pct_share": percentage_with_abs_pct_error_above(
            df,
            "emission_pct_error_q75_speed",
        ),
    }


def add_total_percentage_errors(summary_row):
    actual_distance = summary_row["actual_distance_km_sum"]
    actual_emission = summary_row["actual_emission_sum"]

    summary_row["avg_speed_total_distance_pct_error"] = (
        (summary_row["avg_speed_distance_km_sum"] - actual_distance)
        / actual_distance
    ) * 100

    summary_row["median_speed_total_distance_pct_error"] = (
        (summary_row["median_speed_distance_km_sum"] - actual_distance)
        / actual_distance
    ) * 100

    summary_row["q25_speed_total_distance_pct_error"] = (
        (summary_row["q25_speed_distance_km_sum"] - actual_distance)
        / actual_distance
    ) * 100

    summary_row["q58_speed_total_distance_pct_error"] = (
        (summary_row["q58_speed_distance_km_sum"] - actual_distance)
        / actual_distance
    ) * 100

    summary_row["q60_speed_total_distance_pct_error"] = (
        (summary_row["q60_speed_distance_km_sum"] - actual_distance)
        / actual_distance
    ) * 100

    summary_row["q625_speed_total_distance_pct_error"] = (
        (summary_row["q625_speed_distance_km_sum"] - actual_distance)
        / actual_distance
    ) * 100

    summary_row["q75_speed_total_distance_pct_error"] = (
        (summary_row["q75_speed_distance_km_sum"] - actual_distance)
        / actual_distance
    ) * 100

    summary_row["avg_speed_total_emission_pct_error"] = (
        (summary_row["avg_speed_emission_sum"] - actual_emission)
        / actual_emission
    ) * 100

    summary_row["median_speed_total_emission_pct_error"] = (
        (summary_row["median_speed_emission_sum"] - actual_emission)
        / actual_emission
    ) * 100

    summary_row["q25_speed_total_emission_pct_error"] = (
        (summary_row["q25_speed_emission_sum"] - actual_emission)
        / actual_emission
    ) * 100

    summary_row["q58_speed_total_emission_pct_error"] = (
        (summary_row["q58_speed_emission_sum"] - actual_emission)
        / actual_emission
    ) * 100

    summary_row["q60_speed_total_emission_pct_error"] = (
        (summary_row["q60_speed_emission_sum"] - actual_emission)
        / actual_emission
    ) * 100

    summary_row["q625_speed_total_emission_pct_error"] = (
        (summary_row["q625_speed_emission_sum"] - actual_emission)
        / actual_emission
    ) * 100

    summary_row["q75_speed_total_emission_pct_error"] = (
        (summary_row["q75_speed_emission_sum"] - actual_emission)
        / actual_emission
    ) * 100

    return summary_row


def export_distribution_stats(segment_df, trip_df, dataset_output_dir):
    segment_metrics = [
        "distance_diff_km",
        "duration_min",
        "distance_from_avg_speed_km",
        "distance_from_median_speed_km",
        "distance_from_q25_speed_km",
        "distance_from_q58_speed_km",
        "distance_from_q60_speed_km",
        "distance_from_q625_speed_km",
        "distance_from_q75_speed_km",
        "distance_pct_error_avg_speed",
        "distance_pct_error_median_speed",
        "distance_pct_error_q25_speed",
        "distance_pct_error_q58_speed",
        "distance_pct_error_q60_speed",
        "distance_pct_error_q625_speed",
        "distance_pct_error_q75_speed",
        "emission_actual",
        "emission_avg_speed",
        "emission_median_speed",
        "emission_q25_speed",
        "emission_q58_speed",
        "emission_q60_speed",
        "emission_q625_speed",
        "emission_q75_speed",
        "emission_pct_error_avg_speed",
        "emission_pct_error_median_speed",
        "emission_pct_error_q25_speed",
        "emission_pct_error_q58_speed",
        "emission_pct_error_q60_speed",
        "emission_pct_error_q625_speed",
        "emission_pct_error_q75_speed",
    ]

    trip_metrics = [
        "total_distance_actual_km",
        "total_duration_min",
        "total_distance_avg_speed_km",
        "total_distance_median_speed_km",
        "total_distance_q25_speed_km",
        "total_distance_q58_speed_km",
        "total_distance_q60_speed_km",
        "total_distance_q625_speed_km",
        "total_distance_q75_speed_km",
        "distance_pct_error_avg_speed",
        "distance_pct_error_median_speed",
        "distance_pct_error_q25_speed",
        "distance_pct_error_q58_speed",
        "distance_pct_error_q60_speed",
        "distance_pct_error_q625_speed",
        "distance_pct_error_q75_speed",
        "total_emission_actual",
        "total_emission_avg_speed",
        "total_emission_median_speed",
        "total_emission_q25_speed",
        "total_emission_q58_speed",
        "total_emission_q60_speed",
        "total_emission_q625_speed",
        "total_emission_q75_speed",
        "emission_pct_error_avg_speed",
        "emission_pct_error_median_speed",
        "emission_pct_error_q25_speed",
        "emission_pct_error_q58_speed",
        "emission_pct_error_q60_speed",
        "emission_pct_error_q625_speed",
        "emission_pct_error_q75_speed",
    ]

    segment_df[segment_metrics].describe(
        percentiles=[0.05, 0.25, 0.50, 0.75, 0.95]
    ).T.to_csv(dataset_output_dir / "segment_distribution_statistics.csv")

    trip_df[trip_metrics].describe(
        percentiles=[0.05, 0.25, 0.50, 0.75, 0.95]
    ).T.to_csv(dataset_output_dir / "trip_distribution_statistics.csv")


def export_distance_bin_stats(segment_df, dataset_output_dir):
    bin_edges = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, float("inf")]
    bin_labels = [
        "0-1 km", "1-2 km", "2-3 km", "3-4 km", "4-5 km",
        "5-6 km", "6-7 km", "7-8 km", "8-9 km", "9-10 km", ">10 km",
    ]

    segment_df = segment_df.copy()

    segment_df["distance_bin"] = pd.cut(
        segment_df["distance_diff_km"],
        bins=bin_edges,
        labels=bin_labels,
        right=True,
        include_lowest=True,
    )

    bin_summary = (
        segment_df.groupby("distance_bin", observed=False)
        .agg(
            count=("duration_min", "count"),
            duration_mean=("duration_min", "mean"),
            duration_median=("duration_min", "median"),
            duration_std=("duration_min", "std"),
            distance_mean=("distance_diff_km", "mean"),
            avg_speed_abs_pct_error_mean=("distance_abs_pct_error_avg_speed", "mean"),
            median_speed_abs_pct_error_mean=("distance_abs_pct_error_median_speed", "mean"),
            q25_speed_abs_pct_error_mean=("distance_abs_pct_error_q25_speed", "mean"),
            q58_speed_abs_pct_error_mean=("distance_abs_pct_error_q58_speed", "mean"),
            q60_speed_abs_pct_error_mean=("distance_abs_pct_error_q60_speed", "mean"),
            q625_speed_abs_pct_error_mean=("distance_abs_pct_error_q625_speed", "mean"),
            q75_speed_abs_pct_error_mean=("distance_abs_pct_error_q75_speed", "mean"),
            avg_speed_emission_abs_pct_error_mean=("emission_abs_pct_error_avg_speed", "mean"),
            median_speed_emission_abs_pct_error_mean=("emission_abs_pct_error_median_speed", "mean"),
            q25_speed_emission_abs_pct_error_mean=("emission_abs_pct_error_q25_speed", "mean"),
            q58_speed_emission_abs_pct_error_mean=("emission_abs_pct_error_q58_speed", "mean"),
            q60_speed_emission_abs_pct_error_mean=("emission_abs_pct_error_q60_speed", "mean"),
            q625_speed_emission_abs_pct_error_mean=("emission_abs_pct_error_q625_speed", "mean"),
            q75_speed_emission_abs_pct_error_mean=("emission_abs_pct_error_q75_speed", "mean"),
        )
        .reset_index()
    )

    bin_summary.to_csv(dataset_output_dir / "distance_bin_summary.csv", index=False)

    print("\nDistance-bin summary:")
    print(tabulate(bin_summary, headers="keys", tablefmt="psql", showindex=False))

    return segment_df, bin_summary


def save_plot(path):
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def export_charts(segment_df, trip_df, bin_summary, dataset_name, dataset_output_dir):
    sns.set_style("whitegrid")

    plt.figure(figsize=(14, 7))
    sns.boxplot(data=segment_df, x="distance_bin", y="duration_min")
    plt.xticks(rotation=45, ha="right")
    plt.title(f"{dataset_name}: Travel time distribution by distance bin")
    plt.xlabel("Distance bin")
    plt.ylabel("Travel time between stops (minutes)")
    save_plot(dataset_output_dir / "duration_by_distance_bin_boxplot.png")

    plt.figure(figsize=(14, 6))
    sns.barplot(data=bin_summary, x="distance_bin", y="duration_std")
    plt.xticks(rotation=45, ha="right")
    plt.title(f"{dataset_name}: Standard deviation of travel time by distance bin")
    plt.xlabel("Distance bin")
    plt.ylabel("Std travel time (minutes)")
    save_plot(dataset_output_dir / "duration_std_by_distance_bin.png")

    error_long = segment_df[
        [
            "distance_pct_error_avg_speed",
            "distance_pct_error_median_speed",
            "distance_pct_error_q25_speed",
            "distance_pct_error_q58_speed",
            "distance_pct_error_q60_speed",
            "distance_pct_error_q625_speed",
            "distance_pct_error_q75_speed",
        ]
    ].rename(
        columns={
            "distance_pct_error_avg_speed": "Avg speed",
            "distance_pct_error_median_speed": "Median speed",
            "distance_pct_error_q25_speed": "Q25 speed",
            "distance_pct_error_q58_speed": "Q58 speed",
            "distance_pct_error_q60_speed": "Q60 speed",
            "distance_pct_error_q625_speed": "Q62.5 speed",
            "distance_pct_error_q75_speed": "Q75 speed",
        }
    ).melt(var_name="Method", value_name="Distance percentage error")

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=error_long, x="Method", y="Distance percentage error")
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.title(f"{dataset_name}: Distance percentage error")
    save_plot(dataset_output_dir / "distance_percentage_error_boxplot.png")

    emission_error_long = segment_df[
        [
            "emission_pct_error_avg_speed",
            "emission_pct_error_median_speed",
            "emission_pct_error_q25_speed",
            "emission_pct_error_q58_speed",
            "emission_pct_error_q60_speed",
            "emission_pct_error_q625_speed",
            "emission_pct_error_q75_speed",
        ]
    ].rename(
        columns={
            "emission_pct_error_avg_speed": "Avg speed",
            "emission_pct_error_median_speed": "Median speed",
            "emission_pct_error_q25_speed": "Q25 speed",
            "emission_pct_error_q58_speed": "Q58 speed",
            "emission_pct_error_q60_speed": "Q60 speed",
            "emission_pct_error_q625_speed": "Q62.5 speed",
            "emission_pct_error_q75_speed": "Q75 speed",
        }
    ).melt(var_name="Method", value_name="Emission percentage error")

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=emission_error_long, x="Method", y="Emission percentage error")
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.title(f"{dataset_name}: Emission percentage error")
    save_plot(dataset_output_dir / "emission_percentage_error_boxplot.png")

    fig, axes = plt.subplots(4, 2, figsize=(14, 20), sharey=True)
    axes = axes.flatten()

    distance_scatter_specs = [
        ("distance_from_avg_speed_km", "Actual vs avg-speed distance"),
        ("distance_from_median_speed_km", "Actual vs median-speed distance"),
        ("distance_from_q25_speed_km", "Actual vs q25-speed distance"),
        ("distance_from_q58_speed_km", "Actual vs q58-speed distance"),
        ("distance_from_q60_speed_km", "Actual vs q60-speed distance"),
        ("distance_from_q625_speed_km", "Actual vs q62.5-speed distance"),
        ("distance_from_q75_speed_km", "Actual vs q75-speed distance"),
    ]
    for ax, (column, title) in zip(axes, distance_scatter_specs):
        sns.scatterplot(
            data=segment_df,
            x="distance_diff_km",
            y=column,
            alpha=0.3,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("Actual distance (km)")
        ax.set_ylabel("Estimated distance (km)")

    max_distance = segment_df[
        [
            "distance_diff_km",
            "distance_from_avg_speed_km",
            "distance_from_median_speed_km",
            "distance_from_q25_speed_km",
            "distance_from_q58_speed_km",
            "distance_from_q60_speed_km",
            "distance_from_q625_speed_km",
            "distance_from_q75_speed_km",
        ]
    ].max().max()

    for ax in axes[:len(distance_scatter_specs)]:
        ax.plot([0, max_distance], [0, max_distance], color="red", linestyle="--")

    for ax in axes[len(distance_scatter_specs):]:
        ax.set_visible(False)

    save_plot(dataset_output_dir / "actual_vs_estimated_distance_scatter.png")

    fig, axes = plt.subplots(4, 2, figsize=(14, 20), sharey=True)
    axes = axes.flatten()

    emission_scatter_specs = [
        ("total_emission_avg_speed", "Trip actual vs avg-speed emission"),
        ("total_emission_median_speed", "Trip actual vs median-speed emission"),
        ("total_emission_q25_speed", "Trip actual vs q25-speed emission"),
        ("total_emission_q58_speed", "Trip actual vs q58-speed emission"),
        ("total_emission_q60_speed", "Trip actual vs q60-speed emission"),
        ("total_emission_q625_speed", "Trip actual vs q62.5-speed emission"),
        ("total_emission_q75_speed", "Trip actual vs q75-speed emission"),
    ]
    for ax, (column, title) in zip(axes, emission_scatter_specs):
        sns.scatterplot(
            data=trip_df,
            x="total_emission_actual",
            y=column,
            alpha=0.5,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("Actual emission")
        ax.set_ylabel("Estimated emission")

    max_emission = trip_df[
        [
            "total_emission_actual",
            "total_emission_avg_speed",
            "total_emission_median_speed",
            "total_emission_q25_speed",
            "total_emission_q58_speed",
            "total_emission_q60_speed",
            "total_emission_q625_speed",
            "total_emission_q75_speed",
        ]
    ].max().max()

    for ax in axes[:len(emission_scatter_specs)]:
        ax.plot([0, max_emission], [0, max_emission], color="red", linestyle="--")

    for ax in axes[len(emission_scatter_specs):]:
        ax.set_visible(False)

    save_plot(dataset_output_dir / "trip_actual_vs_estimated_emission_scatter.png")


# Main analysis

all_summary_rows = []

for dataset in DATASETS:
    dataset_name = dataset["name"]
    dataset_output_dir = OUTPUT_DIR / dataset_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing {dataset_name}")

    aux_df = load_aux_event_log(dataset["aux_file"])
    speed_df = load_trip_speed_summary(dataset["speed_file"])

    segment_df = build_segments(aux_df, speed_df)
    segment_df = calculate_distance_and_emissions(segment_df)

    trip_df = build_trip_summary(segment_df)

    segment_df, bin_summary = export_distance_bin_stats(
        segment_df,
        dataset_output_dir,
    )

    segment_df.to_csv(
        dataset_output_dir / "trip_segment_summary.csv",
        index=False,
    )

    trip_df.to_csv(
        dataset_output_dir / "trip_consumption_emission_summary.csv",
        index=False,
    )

    export_distribution_stats(
        segment_df,
        trip_df,
        dataset_output_dir,
    )

    export_charts(
        segment_df,
        trip_df,
        bin_summary,
        dataset_name,
        dataset_output_dir,
    )

    segment_summary_row = summarize_deviation(
        segment_df,
        dataset_name,
        "segment",
    )

    trip_summary_row = summarize_deviation(
        trip_df,
        dataset_name,
        "trip",
    )

    all_summary_rows.append(add_total_percentage_errors(segment_summary_row))
    all_summary_rows.append(add_total_percentage_errors(trip_summary_row))


all_datasets_summary = pd.DataFrame(all_summary_rows)

abs_pct_error_mean_columns = [
    "dataset",
    "level",
    "avg_speed_distance_abs_pct_error_mean",
    "median_speed_distance_abs_pct_error_mean",
    "q25_speed_distance_abs_pct_error_mean",
    "q58_speed_distance_abs_pct_error_mean",
    "q60_speed_distance_abs_pct_error_mean",
    "q625_speed_distance_abs_pct_error_mean",
    "q75_speed_distance_abs_pct_error_mean",
    "avg_speed_emission_abs_pct_error_mean",
    "median_speed_emission_abs_pct_error_mean",
    "q25_speed_emission_abs_pct_error_mean",
    "q58_speed_emission_abs_pct_error_mean",
    "q60_speed_emission_abs_pct_error_mean",
    "q625_speed_emission_abs_pct_error_mean",
    "q75_speed_emission_abs_pct_error_mean",
]

abs_pct_error_mean_summary = all_datasets_summary[abs_pct_error_mean_columns]

abs_pct_error_mean_summary.to_csv(
    OUTPUT_DIR / "all_datasets_deviation_summary.csv",
    index=False,
)

abs_pct_error_mean_summary.to_csv(
    OUTPUT_DIR / "all_datasets_abs_pct_error_mean_summary.csv",
    index=False,
)

above_1pct_share_columns = [
    "dataset",
    "level",
    "avg_speed_distance_abs_pct_error_gt_1pct_share",
    "median_speed_distance_abs_pct_error_gt_1pct_share",
    "q25_speed_distance_abs_pct_error_gt_1pct_share",
    "q58_speed_distance_abs_pct_error_gt_1pct_share",
    "q60_speed_distance_abs_pct_error_gt_1pct_share",
    "q625_speed_distance_abs_pct_error_gt_1pct_share",
    "q75_speed_distance_abs_pct_error_gt_1pct_share",
    "avg_speed_emission_abs_pct_error_gt_1pct_share",
    "median_speed_emission_abs_pct_error_gt_1pct_share",
    "q25_speed_emission_abs_pct_error_gt_1pct_share",
    "q58_speed_emission_abs_pct_error_gt_1pct_share",
    "q60_speed_emission_abs_pct_error_gt_1pct_share",
    "q625_speed_emission_abs_pct_error_gt_1pct_share",
    "q75_speed_emission_abs_pct_error_gt_1pct_share",
]

above_1pct_share_summary = all_datasets_summary[above_1pct_share_columns]

above_1pct_share_summary.to_csv(
    OUTPUT_DIR / "all_datasets_abs_pct_error_gt_1pct_share_summary.csv",
    index=False,
)

all_datasets_summary.to_csv(
    OUTPUT_DIR / "all_datasets_deviation_summary_full.csv",
    index=False,
)

print("\nAll datasets mean absolute percentage error summary:")
print(tabulate(
    abs_pct_error_mean_summary,
    headers="keys",
    tablefmt="psql",
    showindex=False,
))

distance_abs_pct_error_table = abs_pct_error_mean_summary[
    [
        "dataset",
        "level",
        "avg_speed_distance_abs_pct_error_mean",
        "median_speed_distance_abs_pct_error_mean",
        "q25_speed_distance_abs_pct_error_mean",
        "q58_speed_distance_abs_pct_error_mean",
        "q60_speed_distance_abs_pct_error_mean",
        "q625_speed_distance_abs_pct_error_mean",
        "q75_speed_distance_abs_pct_error_mean",
    ]
].rename(
    columns={
        "dataset": "Dataset",
        "level": "Level",
        "avg_speed_distance_abs_pct_error_mean": "Avg. speed",
        "median_speed_distance_abs_pct_error_mean": "Median speed",
        "q25_speed_distance_abs_pct_error_mean": "Q25",
        "q58_speed_distance_abs_pct_error_mean": "Q58",
        "q60_speed_distance_abs_pct_error_mean": "Q60",
        "q625_speed_distance_abs_pct_error_mean": "Q62.5",
        "q75_speed_distance_abs_pct_error_mean": "Q75",
    }
).round(2)

emission_abs_pct_error_table = abs_pct_error_mean_summary[
    [
        "dataset",
        "level",
        "avg_speed_emission_abs_pct_error_mean",
        "median_speed_emission_abs_pct_error_mean",
        "q25_speed_emission_abs_pct_error_mean",
        "q58_speed_emission_abs_pct_error_mean",
        "q60_speed_emission_abs_pct_error_mean",
        "q625_speed_emission_abs_pct_error_mean",
        "q75_speed_emission_abs_pct_error_mean",
    ]
].rename(
    columns={
        "dataset": "Dataset",
        "level": "Level",
        "avg_speed_emission_abs_pct_error_mean": "Avg. speed",
        "median_speed_emission_abs_pct_error_mean": "Median speed",
        "q25_speed_emission_abs_pct_error_mean": "Q25",
        "q58_speed_emission_abs_pct_error_mean": "Q58",
        "q60_speed_emission_abs_pct_error_mean": "Q60",
        "q625_speed_emission_abs_pct_error_mean": "Q62.5",
        "q75_speed_emission_abs_pct_error_mean": "Q75",
    }
).round(2)

print("\nDistance MAPE (%) by dataset and level:")
print(tabulate(
    distance_abs_pct_error_table,
    headers="keys",
    tablefmt="psql",
    showindex=False,
))

print("\nCO2 emission MAPE (%) by dataset and level:")
print(tabulate(
    emission_abs_pct_error_table,
    headers="keys",
    tablefmt="psql",
    showindex=False,
))

emission_pct_error_direction_table = all_datasets_summary[
    [
        "dataset",
        "level",
        "avg_speed_emission_pct_error_mean",
        "median_speed_emission_pct_error_mean",
        "q25_speed_emission_pct_error_mean",
        "q58_speed_emission_pct_error_mean",
        "q60_speed_emission_pct_error_mean",
        "q625_speed_emission_pct_error_mean",
        "q75_speed_emission_pct_error_mean",
    ]
].rename(
    columns={
        "dataset": "Dataset",
        "level": "Level",
        "avg_speed_emission_pct_error_mean": "Avg. speed",
        "median_speed_emission_pct_error_mean": "Median speed",
        "q25_speed_emission_pct_error_mean": "Q25",
        "q58_speed_emission_pct_error_mean": "Q58",
        "q60_speed_emission_pct_error_mean": "Q60",
        "q625_speed_emission_pct_error_mean": "Q62.5",
        "q75_speed_emission_pct_error_mean": "Q75",
    }
).round(2)

emission_pct_error_direction_table.to_csv(
    OUTPUT_DIR / "all_datasets_emission_pct_error_direction_summary.csv",
    index=False,
)

print("\nCO2 emission mean percentage error by dataset and level (%):")
print("Positive values indicate overestimation; negative values indicate underestimation.")
print(tabulate(
    emission_pct_error_direction_table,
    headers="keys",
    tablefmt="psql",
    showindex=False,
))

distance_above_1pct_table = above_1pct_share_summary[
    [
        "dataset",
        "level",
        "avg_speed_distance_abs_pct_error_gt_1pct_share",
        "median_speed_distance_abs_pct_error_gt_1pct_share",
        "q25_speed_distance_abs_pct_error_gt_1pct_share",
        "q58_speed_distance_abs_pct_error_gt_1pct_share",
        "q60_speed_distance_abs_pct_error_gt_1pct_share",
        "q625_speed_distance_abs_pct_error_gt_1pct_share",
        "q75_speed_distance_abs_pct_error_gt_1pct_share",
    ]
].rename(
    columns={
        "dataset": "Dataset",
        "level": "Level",
        "avg_speed_distance_abs_pct_error_gt_1pct_share": "Avg. speed",
        "median_speed_distance_abs_pct_error_gt_1pct_share": "Median speed",
        "q25_speed_distance_abs_pct_error_gt_1pct_share": "Q25",
        "q58_speed_distance_abs_pct_error_gt_1pct_share": "Q58",
        "q60_speed_distance_abs_pct_error_gt_1pct_share": "Q60",
        "q625_speed_distance_abs_pct_error_gt_1pct_share": "Q62.5",
        "q75_speed_distance_abs_pct_error_gt_1pct_share": "Q75",
    }
).round(2)

emission_above_1pct_table = above_1pct_share_summary[
    [
        "dataset",
        "level",
        "avg_speed_emission_abs_pct_error_gt_1pct_share",
        "median_speed_emission_abs_pct_error_gt_1pct_share",
        "q25_speed_emission_abs_pct_error_gt_1pct_share",
        "q58_speed_emission_abs_pct_error_gt_1pct_share",
        "q60_speed_emission_abs_pct_error_gt_1pct_share",
        "q625_speed_emission_abs_pct_error_gt_1pct_share",
        "q75_speed_emission_abs_pct_error_gt_1pct_share",
    ]
].rename(
    columns={
        "dataset": "Dataset",
        "level": "Level",
        "avg_speed_emission_abs_pct_error_gt_1pct_share": "Avg. speed",
        "median_speed_emission_abs_pct_error_gt_1pct_share": "Median speed",
        "q25_speed_emission_abs_pct_error_gt_1pct_share": "Q25",
        "q58_speed_emission_abs_pct_error_gt_1pct_share": "Q58",
        "q60_speed_emission_abs_pct_error_gt_1pct_share": "Q60",
        "q625_speed_emission_abs_pct_error_gt_1pct_share": "Q62.5",
        "q75_speed_emission_abs_pct_error_gt_1pct_share": "Q75",
    }
).round(2)

print("\nDistance cases with absolute percentage error > 1% (%):")
print(tabulate(
    distance_above_1pct_table,
    headers="keys",
    tablefmt="psql",
    showindex=False,
))

print("\nCO2 emission cases with absolute percentage error > 1% (%):")
print(tabulate(
    emission_above_1pct_table,
    headers="keys",
    tablefmt="psql",
    showindex=False,
))


# Cross-dataset comparison charts

emission_mape_columns = [
    "dataset",
    "level",
    "avg_speed_emission_abs_pct_error_mean",
    "median_speed_emission_abs_pct_error_mean",
    "q25_speed_emission_abs_pct_error_mean",
    "q58_speed_emission_abs_pct_error_mean",
    "q60_speed_emission_abs_pct_error_mean",
    "q625_speed_emission_abs_pct_error_mean",
    "q75_speed_emission_abs_pct_error_mean",
]

emission_mape_column_names = {
    "avg_speed_emission_abs_pct_error_mean": "Avg. speed",
    "median_speed_emission_abs_pct_error_mean": "Median speed",
    "q25_speed_emission_abs_pct_error_mean": "Q25",
    "q58_speed_emission_abs_pct_error_mean": "Q58",
    "q60_speed_emission_abs_pct_error_mean": "Q60",
    "q625_speed_emission_abs_pct_error_mean": "Q62.5",
    "q75_speed_emission_abs_pct_error_mean": "Q75",
}


def export_co2_emission_mape_chart(level, output_name):
    plot_df = all_datasets_summary[
        all_datasets_summary["level"] == level
    ][emission_mape_columns].rename(
        columns=emission_mape_column_names
    ).melt(
        id_vars=["dataset", "level"],
        var_name="Method",
        value_name="CO2 emission MAPE (%)",
    )

    plt.figure(figsize=(13, 6.5))
    sns.barplot(
        data=plot_df,
        x="Method",
        y="CO2 emission MAPE (%)",
        hue="dataset",
    )
    plt.xticks(rotation=35, ha="right")
    plt.title(f"CO2 Emission MAPE by Dataset ({level} level)")
    plt.xlabel("Estimation method")
    plt.ylabel("CO2 emission MAPE (%)")
    plt.legend(title="Dataset", loc="upper left", bbox_to_anchor=(1.01, 1.0))
    save_plot(OUTPUT_DIR / output_name)


export_co2_emission_mape_chart(
    "segment",
    "co2_emission_mape_segment_level.png",
)
export_co2_emission_mape_chart(
    "trip",
    "co2_emission_mape_trip_level.png",
)

plt.figure(figsize=(12, 6))
sns.barplot(
    data=all_datasets_summary[all_datasets_summary["level"] == "trip"],
    x="dataset",
    y="avg_speed_total_emission_pct_error",
)
plt.xticks(rotation=45, ha="right")
plt.title("Total emission percentage error using avg_speed")
plt.xlabel("Dataset")
plt.ylabel("Total emission error (%)")
save_plot(OUTPUT_DIR / "comparison_avg_speed_total_emission_pct_error.png")

plt.figure(figsize=(12, 6))
sns.barplot(
    data=all_datasets_summary[all_datasets_summary["level"] == "trip"],
    x="dataset",
    y="median_speed_total_emission_pct_error",
)
plt.xticks(rotation=45, ha="right")
plt.title("Total emission percentage error using median_speed")
plt.xlabel("Dataset")
plt.ylabel("Total emission error (%)")
save_plot(OUTPUT_DIR / "comparison_median_speed_total_emission_pct_error.png")

plt.figure(figsize=(12, 6))
comparison_long = all_datasets_summary[
    all_datasets_summary["level"] == "trip"
][
    [
        "dataset",
        "avg_speed_emission_abs_pct_error_mean",
        "median_speed_emission_abs_pct_error_mean",
        "q25_speed_emission_abs_pct_error_mean",
        "q58_speed_emission_abs_pct_error_mean",
        "q60_speed_emission_abs_pct_error_mean",
        "q625_speed_emission_abs_pct_error_mean",
        "q75_speed_emission_abs_pct_error_mean",
    ]
].rename(
    columns={
        "avg_speed_emission_abs_pct_error_mean": "Avg speed",
        "median_speed_emission_abs_pct_error_mean": "Median speed",
        "q25_speed_emission_abs_pct_error_mean": "Q25 speed",
        "q58_speed_emission_abs_pct_error_mean": "Q58 speed",
        "q60_speed_emission_abs_pct_error_mean": "Q60 speed",
        "q625_speed_emission_abs_pct_error_mean": "Q62.5 speed",
        "q75_speed_emission_abs_pct_error_mean": "Q75 speed",
    }
).melt(
    id_vars="dataset",
    var_name="Method",
    value_name="Mean absolute emission percentage error",
)

sns.barplot(
    data=comparison_long,
    x="dataset",
    y="Mean absolute emission percentage error",
    hue="Method",
)
plt.xticks(rotation=45, ha="right")
plt.title("Mean absolute emission percentage error by dataset")
plt.xlabel("Dataset")
plt.ylabel("Mean absolute error (%)")
save_plot(OUTPUT_DIR / "comparison_mean_abs_emission_pct_error.png")

print(f"\nDone. Outputs written to: {OUTPUT_DIR.resolve()}")
