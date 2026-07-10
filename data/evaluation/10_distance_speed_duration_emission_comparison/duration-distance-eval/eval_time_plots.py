#### Distance bins und duration aber mit plots


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate

sns.set_style("whitegrid")

# Load data
aux_event_log_overview = pd.read_csv(
    "aux_event_log_overview_kodak_2026_05_04.csv",
    dtype={"vehicle_id": "string", "trip_id": "string", "trip_id_org": "string"}
)

aux_event_log_overview["timestamp"] = pd.to_datetime(
    aux_event_log_overview["timestamp"],
    errors="coerce"
)

aux_event_log_overview["shape_dist_traveled"] = pd.to_numeric(
    aux_event_log_overview["shape_dist_traveled"],
    errors="coerce"
)

aux_event_log_overview = aux_event_log_overview.dropna(
    subset=["trip_id", "timestamp", "shape_dist_traveled", "activity_type"]
).copy()

aux_event_log_overview["trip_id"] = aux_event_log_overview["trip_id"].astype(str)

# Keep only relevant events
segment_df = aux_event_log_overview[
    aux_event_log_overview["activity_type"].isin(["departure_stop", "arrive_stop"])
].copy()

segment_df = segment_df.sort_values(["trip_id", "timestamp"]).reset_index(drop=True)

# Determine next event within the same trip
segment_df["next_activity_type"] = segment_df.groupby("trip_id")["activity_type"].shift(-1)
segment_df["next_timestamp"] = segment_df.groupby("trip_id")["timestamp"].shift(-1)
segment_df["next_shape_dist_traveled"] = segment_df.groupby("trip_id")["shape_dist_traveled"].shift(-1)
segment_df["next_stop_name"] = segment_df.groupby("trip_id")["stop_name"].shift(-1)

# Keep only departure_stop -> subsequent arrive_stop pairs
segments = segment_df[
    (segment_df["activity_type"] == "departure_stop") &
    (segment_df["next_activity_type"] == "arrive_stop")
].copy()

# Calculate duration and distance
segments["duration_sec"] = (
    segments["next_timestamp"] - segments["timestamp"]
).dt.total_seconds()

segments["duration_min"] = segments["duration_sec"] / 60

segments["distance_diff"] = (
    segments["next_shape_dist_traveled"] - segments["shape_dist_traveled"]
)

# If shape_dist_traveled is stored in meters
segments["distance_diff_km"] = segments["distance_diff"] / 1000

# Keep only valid values
segments = segments[
    (segments["duration_min"] > 0) &
    (segments["distance_diff_km"] > 0)
].copy()

print(tabulate(
    segments[
        ["trip_id", "stop_name", "next_stop_name", "distance_diff_km", "duration_min"]
    ].head(50),
    headers="keys",
    tablefmt="psql",
    showindex=False
))

# Define distance bins
bin_edges = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, float("inf")]
bin_labels = [
    "0-1 km", "1-2 km", "2-3 km", "3-4 km", "4-5 km",
    "5-6 km", "6-7 km", "7-8 km", "8-9 km", "9-10 km", ">10 km"
]

segments["distance_bin"] = pd.cut(
    segments["distance_diff_km"],
    bins=bin_edges,
    labels=bin_labels,
    include_lowest=True
)

# Statistics per bin
bin_summary = (
    segments.groupby("distance_bin", observed=False)["duration_min"]
    .agg(["count", "mean", "median", "std", "min", "max"])
    .reset_index()
)

print("\nStatistics per distance bin:")
print(tabulate(
    bin_summary,
    headers="keys",
    tablefmt="psql",
    showindex=False
))

# Boxplot
plt.figure(figsize=(14, 7))
sns.boxplot(
    data=segments,
    x="distance_bin",
    y="duration_min",
    color="lightblue"
)
plt.xticks(rotation=45, ha="right")
plt.title("Distribution of Travel Time by Distance Bin")
plt.xlabel("Distance Bin")
plt.ylabel("Travel Time (minutes)")
plt.tight_layout()
plt.show()

# Violin plot
plt.figure(figsize=(14, 7))
sns.violinplot(
    data=segments,
    x="distance_bin",
    y="duration_min",
    inner="quartile",
    color="lightgreen"
)
plt.xticks(rotation=45, ha="right")
plt.title("Travel Time Distribution by Distance Bin")
plt.xlabel("Distance Bin")
plt.ylabel("Travel Time (minutes)")
plt.tight_layout()
plt.show()

# Mean travel time per bin
plt.figure(figsize=(14, 6))
ax = sns.barplot(
    data=bin_summary,
    x="distance_bin",
    y="mean",
    color="steelblue"
)
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", padding=3)
plt.xticks(rotation=45, ha="right")
plt.title("Average Travel Time by Distance Bin")
plt.xlabel("Distance Bin")
plt.ylabel("Average Travel Time (minutes)")
plt.tight_layout()
plt.show()

# Standard deviation per bin
plt.figure(figsize=(14, 6))
ax = sns.barplot(
    data=bin_summary,
    x="distance_bin",
    y="std",
    color="salmon"
)
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", padding=3)
plt.xticks(rotation=45, ha="right")
plt.title("Standard Deviation of Travel Time by Distance Bin")
plt.xlabel("Distance Bin")
plt.ylabel("Standard Deviation of Travel Time (minutes)")
plt.tight_layout()
plt.show()

# Continuous scatterplot
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=segments,
    x="distance_diff_km",
    y="duration_min",
    alpha=0.4
)
plt.title("Distance vs. Travel Time Between departure_stop and Subsequent arrive_stop")
plt.xlabel("Distance Difference (km)")
plt.ylabel("Travel Time (minutes)")
plt.tight_layout()
plt.show()

# Continuous scatterplot with linear trend line
plt.figure(figsize=(10, 6))
sns.regplot(
    data=segments,
    x="distance_diff_km",
    y="duration_min",
    scatter_kws={"alpha": 0.3},
    line_kws={"color": "red"}
)
plt.title("Distance vs. Travel Time with Linear Trend Line")
plt.xlabel("Distance Difference (km)")
plt.ylabel("Travel Time (minutes)")
plt.tight_layout()
plt.show()

# Continuous scatterplot with smoothed trend curve
plt.figure(figsize=(10, 6))
sns.regplot(
    data=segments,
    x="distance_diff_km",
    y="duration_min",

    scatter_kws={"alpha": 0.25},
    line_kws={"color": "darkred", "linewidth": 2}
)
plt.title("Distance vs. Travel Time with Smoothed Trend Curve")
plt.xlabel("Distance Difference (km)")
plt.ylabel("Travel Time (minutes)")
plt.tight_layout()
plt.show()


"""
# Density-based plot for many points
plt.figure(figsize=(10, 6))
plt.hexbin(
    segments["distance_diff_km"],
    segments["duration_min"],
    gridsize=35,
    cmap="Blues",
    mincnt=1
)
plt.colorbar(label="Number of Segments")
plt.title("Density Plot of Distance and Travel Time")
plt.xlabel("Distance Difference (km)")
plt.ylabel("Travel Time (minutes)")
plt.tight_layout()
plt.show()
"""