from pathlib import Path
import os

import pandas as pd


os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


OUTPUT_DIR = Path("thesis-main/figures/Eval_Shape_vs_Road_Distance_Comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "gps_distance_delta_std_by_distance_bin_all_datasets.png"
OUTPUT_CSV = OUTPUT_DIR / "gps_distance_delta_std_by_distance_bin_all_datasets.csv"
MAPE_OUTPUT_FILE = OUTPUT_DIR / "gps_distance_mape_by_distance_bin_all_datasets.png"
MAPE_OUTPUT_CSV = OUTPUT_DIR / "gps_distance_mape_by_distance_bin_all_datasets.csv"
SIGNED_ERROR_OUTPUT_FILE = (
    OUTPUT_DIR / "gps_distance_signed_pct_error_by_distance_bin_all_datasets.png"
)
SIGNED_ERROR_OUTPUT_CSV = (
    OUTPUT_DIR / "gps_distance_signed_pct_error_by_distance_bin_all_datasets.csv"
)

DATASETS = [
    (
        Path("thesis-main/figures/Eval_Shape_vs_Road_Distance_02_04/shape_vs_road_distance_segments.csv"),
        "Kodak 2026-02-04",
    ),
    (
        Path("thesis-main/figures/Eval_Shape_vs_Road_Distance_05_04/shape_vs_road_distance_segments.csv"),
        "Kodak 2026-05-04",
    ),
    (
        Path("thesis-main/figures/Eval_Shape_vs_Road_Distance_08_04/shape_vs_road_distance_segments.csv"),
        "Kodak 2025-08-04",
    ),
    (
        Path("thesis-main/figures/Eval_Shape_vs_Road_Distance_11_04/shape_vs_road_distance_segments.csv"),
        "Kodak 2025-11-04",
    ),
]

BIN_EDGES = [step / 1000 for step in range(0, 3001, 200)]
BIN_ORDER = [
    f"{start}-{end} m"
    for start, end in zip(range(0, 3000, 200), range(200, 3200, 200))
]


def load_gps_distance_metrics():
    rows = []

    for path, dataset in DATASETS:
        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")

        df = pd.read_csv(
            path,
            usecols=["shape_distance_km", "distance_delta_km"],
        )
        df["shape_distance_km"] = pd.to_numeric(
            df["shape_distance_km"],
            errors="coerce",
        )
        df["distance_delta_km"] = pd.to_numeric(
            df["distance_delta_km"],
            errors="coerce",
        )
        df = df.dropna(subset=["shape_distance_km", "distance_delta_km"])
        df = df[df["shape_distance_km"] > 0].copy()
        df["absolute_percentage_error"] = (
            df["distance_delta_km"].abs() / df["shape_distance_km"]
        ) * 100
        df["signed_percentage_error"] = (
            df["distance_delta_km"] / df["shape_distance_km"]
        ) * 100

        df["Distance bin"] = pd.cut(
            df["shape_distance_km"],
            bins=BIN_EDGES,
            labels=BIN_ORDER,
            right=True,
            include_lowest=True,
        )

        summary = (
            df.groupby("Distance bin", observed=False)
            .agg(
                **{
                    "STD of GPS distance difference (km)": (
                        "distance_delta_km",
                        "std",
                    ),
                    "GPS distance MAPE (%)": (
                        "absolute_percentage_error",
                        "mean",
                    ),
                    "GPS distance mean percentage error (%)": (
                        "signed_percentage_error",
                        "mean",
                    ),
                    "Segment count": ("distance_delta_km", "count"),
                }
            )
            .reindex(BIN_ORDER)
            .reset_index()
        )
        summary["Dataset"] = dataset
        rows.append(summary)

    combined = pd.concat(rows, ignore_index=True)
    combined["Distance bin"] = pd.Categorical(
        combined["Distance bin"],
        categories=BIN_ORDER,
        ordered=True,
    )
    return combined.sort_values(["Distance bin", "Dataset"])


metrics_df = load_gps_distance_metrics()
std_df = metrics_df[
    ["Distance bin", "STD of GPS distance difference (km)", "Segment count", "Dataset"]
].copy()
mape_df = metrics_df[
    ["Distance bin", "GPS distance MAPE (%)", "Segment count", "Dataset"]
].copy()
signed_error_df = metrics_df[
    [
        "Distance bin",
        "GPS distance mean percentage error (%)",
        "Segment count",
        "Dataset",
    ]
].copy()

std_df.to_csv(OUTPUT_CSV, index=False)
mape_df.to_csv(MAPE_OUTPUT_CSV, index=False)
signed_error_df.to_csv(SIGNED_ERROR_OUTPUT_CSV, index=False)

sns.set_theme(style="whitegrid", context="paper")

plt.figure(figsize=(14, 7))
ax = sns.barplot(
    data=std_df,
    x="Distance bin",
    y="STD of GPS distance difference (km)",
    hue="Dataset",
    order=BIN_ORDER,
)

ax.set_title("Standard Deviation of GPS Distance Difference by Distance Bin")
ax.set_xlabel("Shape distance bin")
ax.set_ylabel("STD of road distance minus shape distance (km)")
ax.tick_params(axis="x", rotation=35)
ax.legend(title="Dataset", loc="upper left", bbox_to_anchor=(1.01, 1.0))

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=200, bbox_inches="tight")
plt.close()

plt.figure(figsize=(14, 7))
ax = sns.barplot(
    data=mape_df,
    x="Distance bin",
    y="GPS distance MAPE (%)",
    hue="Dataset",
    order=BIN_ORDER,
)

ax.set_title("GPS Distance MAPE by Distance Bin")
ax.set_xlabel("Shape distance bin")
ax.set_ylabel("Mean absolute percentage error (%)")
ax.tick_params(axis="x", rotation=35)
ax.legend(title="Dataset", loc="upper left", bbox_to_anchor=(1.01, 1.0))

plt.tight_layout()
plt.savefig(MAPE_OUTPUT_FILE, dpi=200, bbox_inches="tight")
plt.close()

plt.figure(figsize=(14, 7))
ax = sns.barplot(
    data=signed_error_df,
    x="Distance bin",
    y="GPS distance mean percentage error (%)",
    hue="Dataset",
    order=BIN_ORDER,
)

ax.axhline(0, color="black", linewidth=1)
ax.set_title("GPS Distance Mean Percentage Error by Distance Bin")
ax.set_xlabel("Shape distance bin")
ax.set_ylabel("Mean percentage error (%)")
ax.tick_params(axis="x", rotation=35)
ax.legend(title="Dataset", loc="upper left", bbox_to_anchor=(1.01, 1.0))

plt.tight_layout()
plt.savefig(SIGNED_ERROR_OUTPUT_FILE, dpi=200, bbox_inches="tight")
plt.close()

print(f"Wrote chart to: {OUTPUT_FILE.resolve()}")
print(f"Wrote values to: {OUTPUT_CSV.resolve()}")
print(f"Wrote MAPE chart to: {MAPE_OUTPUT_FILE.resolve()}")
print(f"Wrote MAPE values to: {MAPE_OUTPUT_CSV.resolve()}")
print(f"Wrote signed error chart to: {SIGNED_ERROR_OUTPUT_FILE.resolve()}")
print(f"Wrote signed error values to: {SIGNED_ERROR_OUTPUT_CSV.resolve()}")
