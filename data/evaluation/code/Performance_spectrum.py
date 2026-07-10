import pandas as pd
import pm4py
from tabulate import tabulate


ACTIVITIES = [ "arrive_stop", "departure_stop"]


df = pd.read_csv("aux_event_log_overview_kodak.csv")
df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    format="%Y-%m-%d %H:%M:%S"
)
# PM4Py erwartet intern Nanosekundenauflösung und rechnet sonst falsch auf Epoch-Sekunden.
df["timestamp"] = df["timestamp"].astype("datetime64[ns]")
# Nur gueltige Cases behalten und Datentypen fuer PM4Py sauber setzen.
df = df.dropna(subset=["trip_id", "activity", "timestamp"]).copy()
df["trip_id"] = df["trip_id"].astype("Int64").astype(str)

print("Verfuegbare Aktivitaeten:")
print(df["activity"].value_counts())

filtered_df = df[df["activity"].isin(ACTIVITIES)].copy()

print("\nBeispieldaten fuer das Performance Spectrum:")
print(
    tabulate(
        filtered_df.sort_values(["trip_id", "timestamp"]).head(20),
        headers="keys",
        tablefmt="psql",
    )
)

valid_trips = (
    filtered_df.groupby("trip_id")["activity"]
    .apply(lambda x: set(ACTIVITIES).issubset(set(x)))
)
valid_trip_ids = valid_trips[valid_trips].index

filtered_df = filtered_df[filtered_df["trip_id"].isin(valid_trip_ids)].copy()

print(f"\nAnzahl Trips mit allen Aktivitaeten {ACTIVITIES}: {len(valid_trip_ids)}")

if filtered_df.empty:
    raise ValueError(
        f"Keine Daten fuer das Performance Spectrum gefunden. "
        f"Pruefe die Aktivitaeten {ACTIVITIES}."
    )

filtered_df = pm4py.format_dataframe(
    filtered_df,
    case_id="trip_id",
    activity_key="activity",
    timestamp_key="timestamp",
)

print(tabulate(filtered_df.head(10), headers="keys", tablefmt="psql"))

pm4py.view_performance_spectrum(
    filtered_df.sort_values("timestamp").head(100),
    ACTIVITIES,
    activity_key="activity",
    timestamp_key="timestamp",
    case_id_key="trip_id",
)
