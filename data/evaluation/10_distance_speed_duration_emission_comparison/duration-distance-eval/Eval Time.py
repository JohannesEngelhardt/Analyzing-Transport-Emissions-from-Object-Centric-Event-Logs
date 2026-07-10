### kilometer bin uns duration verteilung

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate

# Daten laden
aux_event_log_overview = pd.read_csv(
    "aux_event_log_overview_kodak_2026_05_04.csv",
    dtype={"vehicle_id": "string", "trip_id": "string", "trip_id_org": "string"}
)
print("testtest")
print(tabulate(aux_event_log_overview[(aux_event_log_overview["vehicle_id"]=='9031005920804452')].head(500),headers="keys", showindex=False))
#print(tabulate(aux_event_log_overview[(aux_event_log_overview["trip_id"]=='tr03_9781') |  (aux_event_log_overview["trip_id"]=='tr03_9796')].head(100),headers="keys", showindex=False))

aux_event_log_overview["timestamp"] = pd.to_datetime(
    aux_event_log_overview["timestamp"],
    errors="coerce"
)

aux_event_log_overview["shape_dist_traveled"] = pd.to_numeric(
    aux_event_log_overview["shape_dist_traveled"],
    errors="coerce"
)

aux_event_log_overview = aux_event_log_overview.dropna(
    subset=["trip_id", "timestamp", "shape_dist_traveled"]
).copy()

aux_event_log_overview["trip_id"] = aux_event_log_overview["trip_id"].astype(str)

# Nur arrive_stop behalten
arrive_df = aux_event_log_overview[
    aux_event_log_overview["activity_type"] == "arrive_stop"
].copy()

# Nach Trip und Zeit sortieren
arrive_df = arrive_df.sort_values(["trip_id", "timestamp"]).reset_index(drop=True)

# Differenzen innerhalb desselben Trips berechnen
arrive_df["prev_timestamp"] = arrive_df.groupby("trip_id")["timestamp"].shift(1)
arrive_df["prev_shape_dist_traveled"] = arrive_df.groupby("trip_id")["shape_dist_traveled"].shift(1)
arrive_df["prev_stop_name"] = arrive_df.groupby("trip_id")["stop_name"].shift(1)

# Zeitdifferenz und Distanzdifferenz berechnen
arrive_df["duration_sec"] = (
    arrive_df["timestamp"] - arrive_df["prev_timestamp"]
).dt.total_seconds()

arrive_df["duration_min"] = arrive_df["duration_sec"] / 60

arrive_df["distance_diff"] = (
    arrive_df["shape_dist_traveled"] - arrive_df["prev_shape_dist_traveled"]
)

# Falls shape_dist_traveled in Metern vorliegt:
arrive_df["distance_diff_km"] = arrive_df["distance_diff"] / 1000

# Nur sinnvolle Paare behalten
arrive_segments = arrive_df.dropna(
    subset=["prev_timestamp", "prev_shape_dist_traveled", "duration_sec", "distance_diff_km"]
).copy()

arrive_segments = arrive_segments[
    (arrive_segments["duration_sec"] > 0) &
    (arrive_segments["distance_diff_km"] > 0)
].copy()

print(tabulate(
    arrive_segments[
        ["trip_id", "prev_stop_name", "stop_name", "distance_diff_km", "duration_min"]
    ].head(50),
    headers="keys",
    tablefmt="psql",
    showindex=False
))

# Distanz-Bins definieren
bin_edges = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, float("inf")]
bin_labels = [
    "0-1 km", "1-2 km", "2-3 km", "3-4 km", "4-5 km",
    "5-6 km", "6-7 km", "7-8 km", "8-9 km", "9-10 km", ">10 km"
]

arrive_segments["distance_bin"] = pd.cut(
    arrive_segments["distance_diff_km"],
    bins=bin_edges,
    labels=bin_labels,
    right=True,
    include_lowest=True
)

# Statistik pro Bin
bin_summary = (
    arrive_segments.groupby("distance_bin", observed=False)["duration_min"]
    .agg(["count", "mean", "median", "std", "min", "max"])
    .reset_index()
)

print("\nStatistik pro Distanz-Bin:")
print(tabulate(
    bin_summary,
    headers="keys",
    tablefmt="psql",
    showindex=False
))

# Boxplot = Streuungsdarstellung pro Bin
plt.figure(figsize=(14, 7))
sns.boxplot(
    data=arrive_segments,
    x="distance_bin",
    y="duration_min"
)
plt.xticks(rotation=45, ha="right")
plt.title("Streuung der Fahrtdauer zwischen zwei arrive_stops pro Distanz-Bin")
plt.xlabel("Distanz-Bin")
plt.ylabel("Dauer zwischen arrive_stops (Minuten)")
plt.tight_layout()
plt.show()

# Optional zusätzlich: Standardabweichung pro Bin als Balkendiagramm
plt.figure(figsize=(14, 6))
sns.barplot(
    data=bin_summary,
    x="distance_bin",
    y="std"
)
plt.xticks(rotation=45, ha="right")
plt.title("Standardabweichung der Dauer pro Distanz-Bin")
plt.xlabel("Distanz-Bin")
plt.ylabel("Std der Dauer (Minuten)")
plt.tight_layout()
plt.show()