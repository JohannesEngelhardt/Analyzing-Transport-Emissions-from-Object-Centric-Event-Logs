import pandas as pd
import pm4py

df = pd.read_csv("aux_event_log_overview.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])
df["trip_id"] = df["trip_id"].astype(str)

df = pm4py.format_dataframe(
    df,
    case_id="trip_id",
    activity_key="activity",
    timestamp_key="timestamp",
)

variants = pm4py.get_variants_as_tuples(df)

print("Anzahl Varianten:", len(variants))

for variant, cases in list(variants.items()):
    print(variant, "->", cases)

