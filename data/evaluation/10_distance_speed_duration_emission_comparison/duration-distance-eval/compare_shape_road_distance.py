import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tabulate import tabulate


DEFAULT_AUX_FILE = "aux_event_log_overview_kodak_2025_08_04.csv"
DEFAULT_SEGMENT_DISTANCE_FILE = "segment_distances_kodak_2025_08_04.csv"
DEFAULT_OUTPUT_DIR = "thesis-main/figures/Eval_Shape_vs_Road_Distance_08_04"

KODAK_DATASETS = [
    {
        "name": "Kodak 2026-02-04",
        "aux_file": "aux_event_log_overview_kodak_2026_02_04.csv",
        "segment_distance_file": "segment_distances_kodak_2026_02_04.csv",
        "output_dir": "thesis-main/figures/Eval_Shape_vs_Road_Distance_02_04",
    },
    {
        "name": "Kodak 2026-05-04",
        "aux_file": "aux_event_log_overview_kodak_2026_05_04.csv",
        "segment_distance_file": "segment_distances_kodak_2026_05_04.csv",
        "output_dir": "thesis-main/figures/Eval_Shape_vs_Road_Distance_05_04",
    },
    {
        "name": "Kodak 2025-08-04",
        "aux_file": "aux_event_log_overview_kodak_2025_08_04.csv",
        "segment_distance_file": "segment_distances_kodak_2025_08_04.csv",
        "output_dir": "thesis-main/figures/Eval_Shape_vs_Road_Distance_08_04",
    },
    {
        "name": "Kodak 2025-11-04",
        "aux_file": "aux_event_log_overview_kodak_2025_11_04.csv",
        "segment_distance_file": "segment_distances_kodak_2025_11_04.csv",
        "output_dir": "thesis-main/figures/Eval_Shape_vs_Road_Distance_11_04",
    },
]

BIN_EDGES = [step / 1000 for step in range(0, 3001, 200)]
BIN_LABELS = [
    f"{start}-{end} m"
    for start, end in zip(range(0, 3000, 200), range(200, 3200, 200))
]
SCATTER_MAX_DISTANCE_KM = 3


def load_aux_with_road_distance(aux_file, segment_distance_file):
    aux = pd.read_csv(
        aux_file,
        dtype={
            "vehicle_id": "string",
            "trip_id": "string",
            "trip_id_org": "string",
            "segment_id": "string",
        },
    )

    if "road_distance_m" not in aux.columns:
        segment_distances = pd.read_csv(
            segment_distance_file,
            dtype={"segment_id": "string"},
        )
        aux = aux.merge(
            segment_distances[["segment_id", "road_distance_m", "road_distance_status"]],
            on="segment_id",
            how="left",
        )

    return aux


def build_arrive_segment_comparison(aux):
    aux["timestamp"] = pd.to_datetime(aux["timestamp"], errors="coerce")
    aux["shape_dist_traveled"] = pd.to_numeric(
        aux["shape_dist_traveled"], errors="coerce"
    )
    aux["road_distance_m"] = pd.to_numeric(aux["road_distance_m"], errors="coerce")

    aux = aux.dropna(subset=["trip_id", "timestamp", "shape_dist_traveled"]).copy()
    aux["trip_id"] = aux["trip_id"].astype(str)

    arrive_df = aux[aux["activity_type"] == "arrive_stop"].copy()
    arrive_df = arrive_df.sort_values(["trip_id", "timestamp"]).reset_index(drop=True)

    arrive_df["prev_timestamp"] = arrive_df.groupby("trip_id")["timestamp"].shift(1)
    arrive_df["prev_shape_dist_traveled"] = arrive_df.groupby("trip_id")[
        "shape_dist_traveled"
    ].shift(1)
    arrive_df["prev_stop_name"] = arrive_df.groupby("trip_id")["stop_name"].shift(1)
    arrive_df["prev_stop_id"] = arrive_df.groupby("trip_id")["stop_id"].shift(1)

    arrive_df["duration_sec"] = (
        arrive_df["timestamp"] - arrive_df["prev_timestamp"]
    ).dt.total_seconds()
    arrive_df["duration_min"] = arrive_df["duration_sec"] / 60
    arrive_df["shape_distance_m"] = (
        arrive_df["shape_dist_traveled"] - arrive_df["prev_shape_dist_traveled"]
    )
    arrive_df["shape_distance_km"] = arrive_df["shape_distance_m"] / 1000
    arrive_df["road_distance_km"] = arrive_df["road_distance_m"] / 1000
    arrive_df["distance_delta_m"] = (
        arrive_df["road_distance_m"] - arrive_df["shape_distance_m"]
    )
    arrive_df["distance_delta_km"] = arrive_df["distance_delta_m"] / 1000
    arrive_df["road_shape_ratio"] = (
        arrive_df["road_distance_m"] / arrive_df["shape_distance_m"]
    )

    comparison = arrive_df.dropna(
        subset=[
            "prev_timestamp",
            "prev_shape_dist_traveled",
            "duration_sec",
            "shape_distance_km",
            "road_distance_km",
        ]
    ).copy()

    comparison = comparison[
        (comparison["duration_sec"] > 0)
        & (comparison["shape_distance_km"] > 0)
        & (comparison["road_distance_km"] > 0)
    ].copy()

    return comparison


def add_distance_bins(comparison):
    comparison["shape_distance_bin"] = pd.cut(
        comparison["shape_distance_km"],
        bins=BIN_EDGES,
        labels=BIN_LABELS,
        right=True,
        include_lowest=True,
    )
    return comparison


def create_summary(comparison):
    return (
        comparison.groupby("shape_distance_bin", observed=False)
        .agg(
            count=("duration_min", "count"),
            shape_distance_mean_km=("shape_distance_km", "mean"),
            road_distance_mean_km=("road_distance_km", "mean"),
            delta_mean_km=("distance_delta_km", "mean"),
            delta_median_km=("distance_delta_km", "median"),
            delta_std_km=("distance_delta_km", "std"),
            ratio_mean=("road_shape_ratio", "mean"),
            ratio_std=("road_shape_ratio", "std"),
            duration_mean_min=("duration_min", "mean"),
            duration_std_min=("duration_min", "std"),
        )
        .reset_index()
    )


def save_plots(comparison, summary, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(14, 6))
    sns.barplot(
        data=summary,
        x="shape_distance_bin",
        y="delta_std_km",
        order=BIN_LABELS,
    )
    plt.xticks(rotation=45, ha="right")
    plt.title("Standard Deviation of Difference: road_distance - shape_distance")
    plt.xlabel("Distance-Bin")
    plt.ylabel("Standard Deviation of GPS Road Distance")
    plt.tight_layout()
    plt.savefig(output_dir / "STD_Road_vs_Shape_Distance_by_Bin.png", dpi=300)
    plt.close()

    scatter_data = comparison[
        (comparison["shape_distance_km"] <= SCATTER_MAX_DISTANCE_KM)
        & (comparison["road_distance_km"] <= SCATTER_MAX_DISTANCE_KM)
    ].copy()
    if len(scatter_data) > 100_000:
        scatter_data = scatter_data.sample(100_000, random_state=42)

    plt.figure(figsize=(9, 8))
    sns.scatterplot(
        data=scatter_data,
        x="shape_distance_km",
        y="road_distance_km",
        alpha=0.35,
        s=18,
        edgecolor=None,
    )
    plt.plot(
        [0, SCATTER_MAX_DISTANCE_KM],
        [0, SCATTER_MAX_DISTANCE_KM],
        linestyle="--",
        color="black",
    )
    plt.xlim(0, SCATTER_MAX_DISTANCE_KM)
    plt.ylim(0, SCATTER_MAX_DISTANCE_KM)
    plt.title("Shape-Distance vs. GPS Road Distance (0-3 km)")
    plt.xlabel("Shape-Distance between arrive_stops (km)")
    plt.ylabel("Road Distance / OSRM (km)")
    plt.tight_layout()
    plt.savefig(output_dir / "Scatter_Road_vs_Shape_Distance.png", dpi=300)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Vergleicht shape_dist_traveled-Differenz mit road_distance_m."
    )
    parser.add_argument("--aux-file", default=DEFAULT_AUX_FILE)
    parser.add_argument("--segment-distance-file", default=DEFAULT_SEGMENT_DISTANCE_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--summary-output",
        default="shape_vs_road_distance_bin_summary.csv",
    )
    parser.add_argument(
        "--comparison-output",
        default="shape_vs_road_distance_segments.csv",
    )
    parser.add_argument(
        "--single-dataset",
        action="store_true",
        help="Nur die mit --aux-file/--segment-distance-file angegebene Datei auswerten.",
    )
    return parser.parse_args()


def process_dataset(aux_file, segment_distance_file, output_dir, summary_output, comparison_output, dataset_name):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    aux = load_aux_with_road_distance(aux_file, segment_distance_file)
    comparison = build_arrive_segment_comparison(aux)
    comparison = add_distance_bins(comparison)
    summary = create_summary(comparison)

    comparison.to_csv(output_dir / comparison_output, index=False)
    summary.to_csv(output_dir / summary_output, index=False)
    save_plots(comparison, summary, output_dir)

    print(f"Vergleich Shape-Distanz vs. Road-Distance: {dataset_name}")
    print(f"Segmente: {len(comparison)}")
    print(tabulate(summary, headers="keys", tablefmt="psql", showindex=False))
    print(f"Outputs written to: {output_dir}")


def main():
    args = parse_args()

    if args.single_dataset:
        datasets = [
            {
                "name": "Single dataset",
                "aux_file": args.aux_file,
                "segment_distance_file": args.segment_distance_file,
                "output_dir": args.output_dir,
            }
        ]
    else:
        datasets = KODAK_DATASETS

    for dataset in datasets:
        process_dataset(
            dataset["aux_file"],
            dataset["segment_distance_file"],
            dataset["output_dir"],
            args.summary_output,
            args.comparison_output,
            dataset["name"],
        )


if __name__ == "__main__":
    main()
