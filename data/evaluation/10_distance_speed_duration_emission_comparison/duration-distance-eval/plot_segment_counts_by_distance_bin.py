from pathlib import Path
import os

import pandas as pd


os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


INPUT_DIR = Path("distance_speed_duration_emission_comparison")
OUTPUT_FILE = INPUT_DIR / "segment_counts_by_distance_bin.png"
OUTPUT_CSV = INPUT_DIR / "segment_counts_by_distance_bin.csv"
TRAVEL_TIME_OUTPUT_FILE = INPUT_DIR / "average_travel_time_by_distance_bin.png"
TRAVEL_TIME_OUTPUT_CSV = INPUT_DIR / "average_travel_time_by_distance_bin.csv"
TRAVEL_TIME_STD_OUTPUT_FILE = INPUT_DIR / "travel_time_std_by_distance_bin.png"
TRAVEL_TIME_STD_OUTPUT_CSV = INPUT_DIR / "travel_time_std_by_distance_bin.csv"

DATASETS = [
    ("kodak_2026_02_04", "Kodak 2026-02-04"),
    ("kodak_2026_05_04", "Kodak 2026-05-04"),
    ("kodak_2025_08_04", "Kodak 2025-08-04"),
    ("kodak_2025_11_04", "Kodak 2025-11-04"),
]

BIN_EDGES = [step / 1000 for step in range(0, 3001, 200)] + [float("inf")]
BIN_LABELS = [
    f"{start}-{end} m"
    for start, end in zip(range(0, 3000, 200), range(200, 3200, 200))
] + [">3 km"]


def load_segment_counts():
    count_rows = []
    travel_time_rows = []
    travel_time_std_rows = []

    for dataset_dir, dataset_label in DATASETS:
        path = INPUT_DIR / dataset_dir / "trip_segment_summary.csv"

        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")

        df = pd.read_csv(path, usecols=["distance_diff_km", "duration_min"])
        df["distance_diff_km"] = pd.to_numeric(
            df["distance_diff_km"],
            errors="coerce",
        )
        df["duration_min"] = pd.to_numeric(
            df["duration_min"],
            errors="coerce",
        )
        df = df.dropna(subset=["distance_diff_km", "duration_min"])
        df = df[df["distance_diff_km"] > 0].copy()
        df = df[df["duration_min"] > 0].copy()

        df["Distance bin"] = pd.cut(
            df["distance_diff_km"],
            bins=BIN_EDGES,
            labels=BIN_LABELS,
            right=True,
            include_lowest=True,
        )

        counts = (
            df.groupby("Distance bin", observed=False)
            .size()
            .reindex(BIN_LABELS, fill_value=0)
            .reset_index(name="Segment count")
        )
        counts["Dataset"] = dataset_label
        count_rows.append(counts)

        travel_times = (
            df.groupby("Distance bin", observed=False)["duration_min"]
            .mean()
            .reindex(BIN_LABELS)
            .reset_index(name="Average travel time (min)")
        )
        travel_times["Dataset"] = dataset_label
        travel_time_rows.append(travel_times)

        travel_time_stds = (
            df.groupby("Distance bin", observed=False)["duration_min"]
            .std()
            .reindex(BIN_LABELS)
            .reset_index(name="Travel time std (min)")
        )
        travel_time_stds["Dataset"] = dataset_label
        travel_time_std_rows.append(travel_time_stds)

    return (
        pd.concat(count_rows, ignore_index=True),
        pd.concat(travel_time_rows, ignore_index=True),
        pd.concat(travel_time_std_rows, ignore_index=True),
    )


counts_df, travel_time_df, travel_time_std_df = load_segment_counts()
counts_df.to_csv(OUTPUT_CSV, index=False)
travel_time_df.to_csv(TRAVEL_TIME_OUTPUT_CSV, index=False)
travel_time_std_df.to_csv(TRAVEL_TIME_STD_OUTPUT_CSV, index=False)

sns.set_theme(style="whitegrid", context="paper")

plt.figure(figsize=(14, 7))
ax = sns.barplot(
    data=counts_df,
    x="Distance bin",
    y="Segment count",
    hue="Dataset",
    order=BIN_LABELS,
)

ax.set_title("Segment Count by Distance Bin and Kodak Dataset")
ax.set_xlabel("Distance bin")
ax.set_ylabel("Number of segments")
ax.tick_params(axis="x", rotation=35)
ax.legend(title="Dataset", loc="upper left", bbox_to_anchor=(1.01, 1.0))

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=200, bbox_inches="tight")
plt.close()

plt.figure(figsize=(14, 7))
ax = sns.barplot(
    data=travel_time_df,
    x="Distance bin",
    y="Average travel time (min)",
    hue="Dataset",
    order=BIN_LABELS,
)

ax.set_title("Average Travel Time by Distance Bin and Kodak Dataset")
ax.set_xlabel("Distance bin")
ax.set_ylabel("Average travel time (minutes)")
ax.tick_params(axis="x", rotation=35)
ax.legend(title="Dataset", loc="upper left", bbox_to_anchor=(1.01, 1.0))

plt.tight_layout()
plt.savefig(TRAVEL_TIME_OUTPUT_FILE, dpi=200, bbox_inches="tight")
plt.close()

plt.figure(figsize=(14, 7))
ax = sns.barplot(
    data=travel_time_std_df,
    x="Distance bin",
    y="Travel time std (min)",
    hue="Dataset",
    order=BIN_LABELS,
)

ax.set_title("Standard Deviation of Travel Time by Distance Bin and Kodak Dataset")
ax.set_xlabel("Distance bin")
ax.set_ylabel("Travel time standard deviation (minutes)")
ax.tick_params(axis="x", rotation=35)
ax.legend(title="Dataset", loc="upper left", bbox_to_anchor=(1.01, 1.0))

plt.tight_layout()
plt.savefig(TRAVEL_TIME_STD_OUTPUT_FILE, dpi=200, bbox_inches="tight")
plt.close()

print(f"Wrote chart to: {OUTPUT_FILE.resolve()}")
print(f"Wrote counts to: {OUTPUT_CSV.resolve()}")
print(f"Wrote travel time chart to: {TRAVEL_TIME_OUTPUT_FILE.resolve()}")
print(f"Wrote travel times to: {TRAVEL_TIME_OUTPUT_CSV.resolve()}")
print(f"Wrote travel time std chart to: {TRAVEL_TIME_STD_OUTPUT_FILE.resolve()}")
print(f"Wrote travel time stds to: {TRAVEL_TIME_STD_OUTPUT_CSV.resolve()}")
