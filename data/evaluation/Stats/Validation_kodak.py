import pm4py

import pandas as pd
from fontTools.pens.basePen import NullPen
from tabulate import tabulate

file_path = ("/Users/johannesengelhardt/PycharmProjects/Masterthesis3/ocel_koda_2026_05_04_combined.json")

ocel = pm4py.read_ocel2_json(file_path)

#print(tabulate(ocel.events.head(1000) , headers="keys",
 #   tablefmt="psql",
  #  showindex=False))

ocel_event = ocel.events
print(ocel_event.columns)
print(ocel.events["ocel:activity"].unique())
#print(tabulate(ocel_event.head(10000), headers="keys", tablefmt="psql"))

ocel_e2o = ocel.relations
print(ocel_e2o.columns)
print("e2o")
print(tabulate(ocel_e2o[ocel_e2o["ocel:activity"]=='parking'].head(20), headers="keys", tablefmt="psql"))


ocel_object = ocel.objects
print(ocel_object.columns)
print("object")
print(tabulate(ocel_object.head(20), headers="keys", tablefmt="psql"))




ocel_e2o_object = ocel_e2o.merge(ocel_object, on="ocel:oid", how="left")



#print(tabulate(ocel_e2o_object.head(100), headers="keys", tablefmt="psql", showindex=False))


tmp_ocel_e2o = None

#print(ocel_e2o_object["ocel:type_x"].unique())
""""
for type in ocel_e2o_object["ocel:type_x"].unique():
    if tmp_ocel_e2o is None:
        tmp_ocel_e2o = ocel_e2o_object[ocel_e2o_object["ocel:type_x"] == type].drop(columns=["ocel:type_x", "ocel:type_y"])
    else:
        tmp_ocel_e2o = tmp_ocel_e2o.merge(ocel_e2o_object[ocel_e2o_object["ocel:type_x"] == type].drop(columns=["ocel:type_x", "ocel:type_y"]), on='ocel:eid', how="left").reindex(tmp_ocel_e2o.index)

print(tabulate(tmp_ocel_e2o.head(100), headers="keys", tablefmt="psql", showindex=False))


"""

tmp_ocel_e2o = Nonetmp_ocel_e2o = pd.DataFrame({
    "ocel:eid": ocel_e2o_object["ocel:eid"].drop_duplicates()
})

for obj_type in ocel_e2o_object["ocel:type_x"].dropna().unique():
    right = ocel_e2o_object.loc[
        ocel_e2o_object["ocel:type_x"] == obj_type
    ].copy()

    right = right.drop(columns=["ocel:type_x", "ocel:type_y"], errors="ignore")
    right = right.groupby("ocel:eid", as_index=False).first()

    tmp_ocel_e2o = tmp_ocel_e2o.merge(
        right,
        on="ocel:eid",
        how="left",
        suffixes=("", "_new"),
        sort=False
    )

    for col in right.columns:
        if col == "ocel:eid":
            continue

        new_col = f"{col}_new"

        if col in tmp_ocel_e2o.columns and new_col in tmp_ocel_e2o.columns:
            tmp_ocel_e2o[col] = tmp_ocel_e2o[col].combine_first(tmp_ocel_e2o[new_col])
            tmp_ocel_e2o = tmp_ocel_e2o.drop(columns=[new_col])
        elif new_col in tmp_ocel_e2o.columns:
            tmp_ocel_e2o = tmp_ocel_e2o.rename(columns={new_col: col})




tmp_ocel_e2o = ocel_e2o_object.drop(
    columns=["ocel:oid", "ocel:type_x", "ocel:type_y", "ocel:qualifier"],
    errors="ignore"
)

tmp_ocel_e2o = tmp_ocel_e2o.groupby(
    ["ocel:eid", "ocel:activity", "ocel:timestamp"],
    as_index=False
).first()

"""
print(tabulate(
    tmp_ocel_e2o[tmp_ocel_e2o["route_short_name"]==10].sort_values(["ocel:timestamp", "ocel:activity"]).head(100),
   # tmp_ocel_e2o.sort_values(["ocel:timestamp", "ocel:activity"]).head(100),

    headers="keys",
    tablefmt="psql",
    showindex=False
))
"""
overlap = [
    col for col in tmp_ocel_e2o.columns
    if col in ocel_event.columns and col != "ocel:eid"
]

tmp_ocel_e2o_clean = tmp_ocel_e2o.drop(columns=overlap)

#tmp_ocel_e2o_clean = [["event_id", "trip_id", "timestamp", "activity","stop_id", "stop_sequence", "stop-headsign", "stop_name", "vehicle_id", "direction_id", "route_short_name"]]

final_ocel = ocel_event.merge(
    tmp_ocel_e2o_clean,
    on="ocel:eid",
    how="left",
    sort=False
)

final_ocel = final_ocel[["ocel:eid", "ocel:timestamp", "ocel:activity", "vehicle_id","route_short_name","delay", "occupancy_status", "stop_sequence", "direction_id", "trip_id", "stop_name", "stop_id","dist","vehicle_id_object"]]
final_ocel["vehicle_id"] = final_ocel["vehicle_id"].astype("string")


print(tabulate(
    #final_ocel[(final_ocel["route_short_name"]==39) & (final_ocel["vehicle_id"]=='9031005920804816')].sort_values(["ocel:timestamp", "ocel:activity"]).head(2000),
    #final_ocel[(final_ocel["ocel:timestamp"]>'2026-05-17 00:00:00') ].sort_values(["ocel:timestamp", "ocel:activity"]).head(2000),
    #final_ocel[(final_ocel["trip_id"]=='55700000078905684')].sort_values(["ocel:timestamp", "ocel:activity"]).head(2000),
    #final_ocel[final_ocel["trip_id"]=='55700100077432135'].sort_values(["ocel:timestamp", "ocel:activity"]).head(2000),
    final_ocel.tail(100),
    #55700000078905684
    #final_ocel[(final_ocel["route_short_name"]==39)].sort_values(["ocel:timestamp", "ocel:activity"]).head(1000),
    #final_ocel[ (final_ocel["ocel:activity"]=='begin_shift')].sort_values(["ocel:timestamp", "ocel:activity"]).head(1000),

    # tmp_ocel_e2o.sort_values(["ocel:timestamp", "ocel:activity"]).head(100),

    headers="keys",
    tablefmt="psql",
    showindex=False
))

