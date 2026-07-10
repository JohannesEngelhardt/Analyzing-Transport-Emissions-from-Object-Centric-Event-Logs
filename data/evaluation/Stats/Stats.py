import json
from functools import cache
import pm4py
@cache
def load_ocel(path):
    print("Lade Datei...")
    return pm4py.read_ocel2_json(path)




import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
import pandas as pd
import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import deque
from typing import Callable, Dict, Hashable, Iterable, List, Optional, Tuple
import math
from pm4py.algo.filtering.ocel.event_attributes import apply_timestamp
from tabulate import tabulate




#ocel_koda_2026_02_04_w_stop_names.json
#file_path = ("/Users/johannesengelhardt/PycharmProjects/Masterthesis3/ocel_koda.json")
#file_path = ("ocel_koda_2026_05_04_w_stop_names_no_layover.json")
file_path = ("/Users/johannesengelhardt/PycharmProjects/Masterthesis3/ocel_koda_2026_05_04_w_stop_names_no_layover_shift_position_work.json")
# file_path = "/Users/johannesengelhardt/PycharmProjects/Masterthesis/data/pallet-logistics-v0.9.sqlite" #Happy Scenario
# file_path = "/Users/johannesengelhardt/PycharmProjects/Masterthesis/data/order-management.sqlite"
#ocel = pm4py.read_ocel2_json(file_path)
ocel = load_ocel(file_path)
print(ocel.events.columns)
#print(tabulate(ocel.events.head(100), headers="keys"))
length_ocel = len(ocel.events)
print(length_ocel)
print(ocel.events["ocel:activity"].unique())
print(ocel.events["ocel:activity"].nunique())

print(ocel.objects["ocel:oid"].nunique())
print(len(ocel.objects))
print(ocel.objects["ocel:type"].unique())

print(ocel.o2o.nunique())
print(len(ocel.objects))


flat_log = pm4py.ocel_flattening(ocel, "trip")
print("\nFlat log auf Objekt trip:")
print(tabulate(flat_log.head(10), headers="keys", tablefmt="psql"))
print(flat_log["case:concept:name"].head(10))

flat_trace_lengths = (
    flat_log.groupby("case:concept:name")
    .size()
    .reset_index(name="trace_length")
)

print("\nTrace-Längen im geflatteten Event Log über Objekt trip:")
print(tabulate(
    flat_trace_lengths["trace_length"].describe().reset_index(),
    headers=["Metric", "Value"],
    tablefmt="psql",
    showindex=False
))
object_counts_unique = (
    ocel.objects.groupby("ocel:type")["ocel:oid"]
    .nunique()
    .reset_index(name="anzahl_objekte")
    .sort_values("anzahl_objekte", ascending=False)
)


object_counts_all = (
    ocel.objects.groupby("ocel:type")
    .size()
    .reset_index(name="anzahl_objekte_nicht_unique")
    .sort_values("anzahl_objekte_nicht_unique", ascending=False)
)

object_counts = object_counts_unique.merge(object_counts_all, on="ocel:type", how="outer")

print("\nAnzahl Objekte pro Objekttyp:")
print(tabulate(object_counts, headers="keys", tablefmt="psql", showindex=False))

def get_relation_table(ocel):
    if hasattr(ocel, "relations"):
        return ocel.relations.copy()
    if hasattr(ocel, "relations_table"):
        return ocel.relations_table.copy()
    raise AttributeError("Keine Relationstabelle im OCEL gefunden.")


def compute_event_log_stats(ocel, dataset_name, trace_object_type="trip", flat_log=None):
    timestamp_col = "ocel:timestamp"

    events = ocel.events.copy()
    objects = ocel.objects.copy()

    events[timestamp_col] = pd.to_datetime(
        events[timestamp_col],
        errors="coerce"
    )

    total_events = len(events)
    activities = events["ocel:activity"].nunique()
    objects_unique = objects["ocel:oid"].nunique()
    object_types = objects["ocel:type"].nunique()

    min_timestamp = events[timestamp_col].min()
    max_timestamp = events[timestamp_col].max()

    if flat_log is None:
        flat_log = pm4py.ocel_flattening(ocel, trace_object_type)

    flat_log = flat_log.copy()
    flat_timestamp_col = "time:timestamp"
    flat_activity_col = "concept:name"
    flat_case_col = "case:concept:name"

    if flat_timestamp_col in flat_log.columns:
        flat_log[flat_timestamp_col] = pd.to_datetime(
            flat_log[flat_timestamp_col],
            errors="coerce"
        )

    trace_lengths = flat_log.groupby(flat_case_col).size()

    min_trace_length = trace_lengths.min()
    avg_trace_length = trace_lengths.mean()
    max_trace_length = trace_lengths.max()

    sort_columns = [flat_case_col]
    if flat_timestamp_col in flat_log.columns:
        sort_columns.append(flat_timestamp_col)

    trace_events = flat_log.sort_values(sort_columns)

    def activity_type(activity):
        if str(activity).startswith("arrive_stop_"):
            return "arrive_stop"
        if str(activity).startswith("departure_stop_"):
            return "departure_stop"
        return activity

    trace_events["activity_type"] = trace_events[flat_activity_col].apply(activity_type)

    variants = (
        trace_events.groupby(flat_case_col)[flat_activity_col]
        .apply(lambda activities: tuple(activities))
        .nunique()
    )

    variants_activity_type = (
        trace_events.groupby(flat_case_col)["activity_type"]
        .apply(lambda activities: tuple(activities))
        .nunique()
    )

    trace_length_summary = pd.DataFrame(
        {
            "Metric": [
                "Min. Trace",
                "Avg. Trace",
                "Max. Trace",
                "Variants",
                "Variants Activity Type",
            ],
            "Value": [
                min_trace_length,
                round(avg_trace_length, 2),
                max_trace_length,
                variants,
                variants_activity_type,
            ],
        }
    )

    print(f"\nTrace statistics from flattened log over object type '{trace_object_type}':")
    print(tabulate(
        trace_length_summary,
        headers="keys",
        tablefmt="psql",
        showindex=False
    ))

    return {
        "Dataset": dataset_name,
        "Events": total_events,
        "Activities": activities,
        "Objects": objects_unique,
        "Object Types": object_types,
        "Min. Timestamp": min_timestamp,
        "Max. Timestamp": max_timestamp,
        "Min. Trace": min_trace_length,
        "Avg. Trace": round(avg_trace_length, 2),
        "Max. Trace": max_trace_length,
        "Variants": variants,
        "Variants Activity Type": variants_activity_type,
    }

stats = compute_event_log_stats(
    ocel,
    dataset_name="Kodak OTRAF ",
    trace_object_type="trip",
    flat_log=flat_log
)

stats_df = pd.DataFrame([stats])

print(tabulate(
    stats_df,
    headers="keys",
    tablefmt="psql",
    showindex=False
))


def compute_activity_type_counts(ocel):
    events = ocel.events.copy()

    def activity_type(activity):
        activity = str(activity)
        if activity.startswith("arrive_stop_"):
            return "arrive_stop"
        if activity.startswith("departure_stop_"):
            return "departure_stop"
        return activity

    events["activity_type"] = events["ocel:activity"].apply(activity_type)

    activity_type_counts = (
        events.groupby("activity_type")
        .size()
        .reset_index(name="Number of Events")
        .sort_values("Number of Events", ascending=False)
    )

    print("\nNumber of events per activity type:")
    print(tabulate(
        activity_type_counts,
        headers="keys",
        tablefmt="psql",
        showindex=False
    ))

    return activity_type_counts


activity_type_counts = compute_activity_type_counts(ocel)

print("\nNumber of distinct activity types:")
print(activity_type_counts["activity_type"].nunique())