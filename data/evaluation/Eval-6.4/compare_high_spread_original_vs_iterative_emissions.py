from __future__ import annotations

from pathlib import Path

import pandas as pd


DATASET_DATE = "2026_05_04"
ORIGINAL_TAG = f"{DATASET_DATE}_high_spread_original"
ITERATIVE_TAG = f"{DATASET_DATE}_iterative_severity_merge"
ITERATIVE_PATHS = (
    Path("hourly_occupancy_distribution_2026_05_04")
    / f"iterative_severity_merge_trip_paths_hour_06_{DATASET_DATE}.csv"
)


def read_trip_summary(tag: str) -> pd.DataFrame:
    return pd.read_csv(f"trip_consumption_emission_summary_{tag}.csv")


def read_segment_summary(tag: str) -> pd.DataFrame:
    return pd.read_csv(f"trip_segment_summary_{tag}.csv")


def metric_row(label: str, trip_df: pd.DataFrame, segment_df: pd.DataFrame) -> dict:
    return {
        "dataset": label,
        "trip_count": trip_df["trip_id"].nunique(),
        "segment_count": len(segment_df),
        "total_distance_km": trip_df["total_distance_km"].sum(),
        "total_duration_min": trip_df["total_duration_min"].sum(),
        "total_emission_actual": trip_df["total_emission_actual"].sum(),
        "total_occupancy_aware_emission": trip_df[
            "total_occupancy_aware_emission"
        ].sum(),
        "total_delay_aware_emission_score": trip_df[
            "total_delay_aware_emission_score"
        ].sum(),
    }


def add_delta_columns(df: pd.DataFrame, key_columns: list[str]) -> pd.DataFrame:
    metric_columns = [
        column
        for column in df.columns
        if column not in key_columns and column.endswith(("_original", "_iterative"))
    ]
    original_bases = sorted(
        {
            column.removesuffix("_original")
            for column in metric_columns
            if column.endswith("_original")
        }
    )
    for base in original_bases:
        original = f"{base}_original"
        iterative = f"{base}_iterative"
        if original not in df.columns or iterative not in df.columns:
            continue
        df[f"{base}_delta"] = df[iterative] - df[original]
        df[f"{base}_remaining_pct"] = df[iterative].div(
            df[original].where(df[original] != 0)
        ) * 100
    return df


def build_overall_comparison(
    original_trip: pd.DataFrame,
    original_segment: pd.DataFrame,
    iterative_trip: pd.DataFrame,
    iterative_segment: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        metric_row("high_spread_original", original_trip, original_segment),
        metric_row("iterative_severity_merge", iterative_trip, iterative_segment),
    ]
    comparison = pd.DataFrame(rows)
    original = comparison.iloc[0]
    iterative = comparison.iloc[1]
    delta = {"dataset": "delta_iterative_minus_original"}
    remaining = {"dataset": "remaining_pct_iterative_of_original"}
    for column in comparison.columns:
        if column == "dataset":
            continue
        delta[column] = iterative[column] - original[column]
        remaining[column] = iterative[column] / original[column] * 100 if original[column] else pd.NA
    return pd.concat([comparison, pd.DataFrame([delta, remaining])], ignore_index=True)


def build_route_comparison(original_trip: pd.DataFrame, iterative_trip: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["route_type", "route_short_name"]
    metric_aggs = {
        "trip_id": "nunique",
        "segment_count": "sum",
        "total_distance_km": "sum",
        "total_duration_min": "sum",
        "total_emission_actual": "sum",
        "total_occupancy_aware_emission": "sum",
        "total_delay_aware_emission_score": "sum",
    }
    original = (
        original_trip.groupby(group_columns, as_index=False)
        .agg(metric_aggs)
        .rename(columns={"trip_id": "trip_count"})
    )
    iterative = (
        iterative_trip.groupby(group_columns, as_index=False)
        .agg(metric_aggs)
        .rename(columns={"trip_id": "trip_count"})
    )
    merged = original.merge(
        iterative,
        on=group_columns,
        how="outer",
        suffixes=("_original", "_iterative"),
    ).fillna(0)
    return add_delta_columns(merged, group_columns)


def build_source_trip_comparison(
    original_trip: pd.DataFrame,
    iterative_trip: pd.DataFrame,
) -> pd.DataFrame:
    path_source = pd.read_csv(ITERATIVE_PATHS, dtype="string")[
        ["path_id", "source_trip_id", "path_color", "status"]
    ].drop_duplicates()
    iterative_with_source = iterative_trip.merge(
        path_source,
        left_on="trip_id",
        right_on="path_id",
        how="left",
    )
    iterative_with_source["source_trip_id"] = iterative_with_source[
        "source_trip_id"
    ].fillna(iterative_with_source["trip_id"])

    metric_columns = [
        "segment_count",
        "total_distance_km",
        "total_duration_min",
        "total_emission_actual",
        "total_occupancy_aware_emission",
        "total_delay_aware_emission_score",
    ]
    original = original_trip.rename(columns={"trip_id": "source_trip_id"})[
        ["source_trip_id", "route_type", "route_short_name"] + metric_columns
    ]
    iterative = (
        iterative_with_source.groupby("source_trip_id", as_index=False)
        .agg(
            route_type=("route_type", "first"),
            route_short_name=("route_short_name", "first"),
            iterative_path_count=("trip_id", "nunique"),
            iterative_path_ids=("trip_id", lambda values: ", ".join(sorted(map(str, values)))),
            **{column: (column, "sum") for column in metric_columns},
        )
    )
    merged = original.merge(
        iterative,
        on=["source_trip_id", "route_type", "route_short_name"],
        how="outer",
        suffixes=("_original", "_iterative"),
    ).fillna(
        {
            "iterative_path_count": 0,
            "iterative_path_ids": "",
        }
    )
    metric_fill_columns = [
        column
        for column in merged.columns
        if column.endswith("_original") or column.endswith("_iterative")
    ]
    merged[metric_fill_columns] = merged[metric_fill_columns].fillna(0)
    return add_delta_columns(
        merged,
        ["source_trip_id", "route_type", "route_short_name", "iterative_path_count", "iterative_path_ids"],
    ).sort_values("total_emission_actual_delta")


def main() -> None:
    original_trip = read_trip_summary(ORIGINAL_TAG)
    original_segment = read_segment_summary(ORIGINAL_TAG)
    iterative_trip = read_trip_summary(ITERATIVE_TAG)
    iterative_segment = read_segment_summary(ITERATIVE_TAG)

    overall = build_overall_comparison(
        original_trip,
        original_segment,
        iterative_trip,
        iterative_segment,
    )
    route = build_route_comparison(original_trip, iterative_trip)
    source_trip = build_source_trip_comparison(original_trip, iterative_trip)

    overall_output = f"high_spread_original_vs_iterative_emission_overall_{DATASET_DATE}.csv"
    route_output = f"high_spread_original_vs_iterative_emission_by_route_{DATASET_DATE}.csv"
    source_trip_output = (
        f"high_spread_original_vs_iterative_emission_by_source_trip_{DATASET_DATE}.csv"
    )
    overall.to_csv(overall_output, index=False)
    route.to_csv(route_output, index=False)
    source_trip.to_csv(source_trip_output, index=False)

    print(overall.to_string(index=False))
    print(f"Overall output: {overall_output}")
    print(f"Route output: {route_output}")
    print(f"Source trip output: {source_trip_output}")


if __name__ == "__main__":
    main()
