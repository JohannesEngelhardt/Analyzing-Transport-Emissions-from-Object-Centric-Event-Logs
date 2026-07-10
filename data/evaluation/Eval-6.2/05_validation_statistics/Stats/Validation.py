import pandas as pd
import pm4py
from tabulate import tabulate


def id_to_string(value):
    if pd.isna(value):
        return pd.NA

    if isinstance(value, float) and value.is_integer():
        return str(int(value))

    return str(value)


def convert_id_columns_to_string(*dataframes):
    id_columns = [
        "ocel:eid",
        "ocel:oid",
        "ocel:oid_2",
        "ocel:oid1",
        "ocel:oid2",
        "ocel:source_oid",
        "ocel:target_oid",
        "route_id",
        "trip_id",
        "vehicle_id",
        "service_id",
        "agency_id",
        "stop_id",
        "old_trip_id",
        "trip_id_org",
        "date",
        "vehicle_id_object"
    ]

    for df in dataframes:
        for col in id_columns:
            if col in df.columns:
                df[col] = df[col].apply(id_to_string).astype("string")


def get_o2o_endpoint_columns(o2o_table):
    endpoint_candidates = [
        ("ocel:oid", "ocel:oid_2"),
        ("ocel:oid1", "ocel:oid2"),
        ("ocel:source_oid", "ocel:target_oid"),
    ]

    for source_col, target_col in endpoint_candidates:
        if source_col in o2o_table.columns and target_col in o2o_table.columns:
            return source_col, target_col

    raise ValueError(
        "Could not detect O2O endpoint columns. "
        f"Available columns: {o2o_table.columns.to_list()}"
    )


OBJECT_TYPE_ORDER = {
    "agency": 0,
    "operator": 1,
    "route": 2,
    "trip": 3,
    "vehicle": 4,
    "stops": 5,
    "stop": 5,
    "service": 6,
}

CONTEXT_O2O_QUALIFIERS = {
    "trip belongs to route",
    "trip is conducted by operator",
    "trip is preceded by trip",
}

ROUTE_CONTEXT_O2O_QUALIFIERS = {
    "agency conducts route",
}

TRAVERSAL_SOURCE_TO_TARGET_QUALIFIERS = {
    "stop belongs to trip",
    "trip is conducted by vehicle",
    "trip is conducted by operator",
    "trip is preceded by trip",
    "route belongs to trip",
}

TRAVERSAL_TARGET_TO_SOURCE_QUALIFIERS = {
    "route belongs to trip",
    "agency belongs to route",
    "trip is preceded by trip",
}

MAX_O2O_TRAVERSAL_ROUNDS = 6

INFER_EXAMPLE_CONTEXT_RELATIONS = True
AUX_EVENT_LOG_PATH = "aux_event_log_overview_kodak_2026_05_04_new_e2o_no_layover.csv"
GTFS_ROUTES_PATH = "GTFS-OTRAF-2026-05-04/routes.txt"
GTFS_AGENCY_PATH = "GTFS-OTRAF-2026-05-04/agency.txt"


def traverse_o2o_context(o2o_table, seed_ids, source_col, target_col):
    reached_ids = pd.Series(seed_ids, dtype="string").dropna().drop_duplicates()
    selected_o2o = pd.DataFrame(columns=o2o_table.columns)

    for _ in range(MAX_O2O_TRAVERSAL_ROUNDS):
        reached_before = set(reached_ids.astype(str))

        source_to_target_mask = (
            o2o_table[source_col].isin(reached_ids)
            & o2o_table["ocel:qualifier"].isin(TRAVERSAL_SOURCE_TO_TARGET_QUALIFIERS)
        )
        target_to_source_mask = (
            o2o_table[target_col].isin(reached_ids)
            & o2o_table["ocel:qualifier"].isin(TRAVERSAL_TARGET_TO_SOURCE_QUALIFIERS)
        )

        step_o2o = o2o_table[source_to_target_mask | target_to_source_mask].copy()
        selected_o2o = (
            pd.concat([selected_o2o, step_o2o], ignore_index=True)
            .drop_duplicates()
            .reset_index(drop=True)
        )

        step_object_ids = pd.concat(
            [step_o2o[source_col], step_o2o[target_col]],
            ignore_index=True,
        ).dropna().astype("string")

        reached_ids = (
            pd.concat([reached_ids, step_object_ids], ignore_index=True)
            .drop_duplicates()
            .astype("string")
        )

        if set(reached_ids.astype(str)) == reached_before:
            break

    internal_o2o_mask = (
        o2o_table[source_col].isin(reached_ids)
        & o2o_table[target_col].isin(reached_ids)
    )
    selected_o2o = (
        pd.concat([selected_o2o, o2o_table[internal_o2o_mask]], ignore_index=True)
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return selected_o2o, reached_ids


#file_path = "/Users/johannesengelhardt/PycharmProjects/Masterthesis3/ocel_koda_2026_05_04_w_stop_names_no_layover.json"
file_path = "/Users/johannesengelhardt/PycharmProjects/Masterthesis3/ocel_koda_2026_05_04_w_stop_names_no_layover_shift_position_work.json"

ocel = pm4py.read_ocel2_json(file_path)

event_id = "ad_1032"
num_events = 4

ocel_event = ocel.events.copy()
ocel_e2o = ocel.relations.copy()
ocel_object = ocel.objects.copy()
ocel_o2o = ocel.o2o.copy()

convert_id_columns_to_string(ocel_event, ocel_e2o, ocel_object, ocel_o2o)

event_id = str(event_id)

event_index = ocel_event.index[ocel_event["ocel:eid"] == event_id][0]

filtered_events = ocel_event.loc[event_index:event_index + num_events - 1]

filtered_relations = ocel_e2o[
    ocel_e2o["ocel:eid"].isin(filtered_events["ocel:eid"])
]

related_object_ids = (
    filtered_relations["ocel:oid"]
    .dropna()
    .astype("string")
    .unique()
)

direct_trip_ids = (
    filtered_relations.loc[filtered_relations["ocel:type"] == "trip", "ocel:oid"]
    .dropna()
    .astype("string")
    .unique()
)

if ocel_o2o.empty:
    filtered_o2o = ocel_o2o.copy()
    output_object_ids = related_object_ids
else:
    o2o_source_col, o2o_target_col = get_o2o_endpoint_columns(ocel_o2o)
    direct_object_ids = pd.Series(related_object_ids, dtype="string")
    direct_trip_ids = pd.Series(direct_trip_ids, dtype="string")

    internal_o2o_mask = (
        ocel_o2o[o2o_source_col].isin(direct_object_ids)
        & ocel_o2o[o2o_target_col].isin(direct_object_ids)
    )

    direct_trip_context_mask = (
        (
            ocel_o2o[o2o_source_col].isin(direct_trip_ids)
            | ocel_o2o[o2o_target_col].isin(direct_trip_ids)
        )
        & ocel_o2o["ocel:qualifier"].isin(CONTEXT_O2O_QUALIFIERS)
    )

    first_hop_o2o = ocel_o2o[internal_o2o_mask | direct_trip_context_mask].copy()
    first_hop_object_ids = pd.concat(
        [
            first_hop_o2o[o2o_source_col],
            first_hop_o2o[o2o_target_col],
        ],
        ignore_index=True,
    ).dropna().astype("string").drop_duplicates()

    # Add route-agency context for routes reached from selected trips.
    second_hop_o2o_mask = (
        (
            ocel_o2o[o2o_source_col].isin(first_hop_object_ids)
            | ocel_o2o[o2o_target_col].isin(first_hop_object_ids)
        )
        & ocel_o2o["ocel:qualifier"].isin(ROUTE_CONTEXT_O2O_QUALIFIERS)
    )

    filtered_o2o = (
        pd.concat([first_hop_o2o, ocel_o2o[second_hop_o2o_mask]], ignore_index=True)
        .drop_duplicates()
        .reset_index(drop=True)
    )

    inferred_context_o2o = pd.DataFrame(columns=filtered_o2o.columns)
    inferred_context_objects = pd.DataFrame(columns=ocel_object.columns)

    if INFER_EXAMPLE_CONTEXT_RELATIONS and len(direct_trip_ids) > 0:
        aux_context = pd.read_csv(
            AUX_EVENT_LOG_PATH,
            usecols=["trip_id", "route_id", "old_trip_id"],
            dtype="string",
        )
        routes_context = pd.read_csv(
            GTFS_ROUTES_PATH,
            usecols=["route_id", "agency_id", "route_short_name", "route_type"],
            dtype="string",
        )
        agency_context = pd.read_csv(
            GTFS_AGENCY_PATH,
            dtype="string",
        )

        selected_trip_context = (
            aux_context[aux_context["trip_id"].isin(direct_trip_ids)]
            .drop_duplicates()
            .dropna(subset=["trip_id"])
        )
        selected_route_ids = selected_trip_context["route_id"].dropna().drop_duplicates()

        inferred_rows = []
        for row in selected_trip_context.itertuples(index=False):
            if pd.notna(row.route_id):
                inferred_rows.append(
                    {
                        o2o_source_col: row.route_id,
                        o2o_target_col: row.trip_id,
                        "ocel:qualifier": "route belongs to trip",
                    }
                )
            if pd.notna(row.old_trip_id):
                inferred_rows.append(
                    {
                        o2o_source_col: row.trip_id,
                        o2o_target_col: row.old_trip_id,
                        "ocel:qualifier": "trip is preceded by trip",
                    }
                )

        selected_routes_context = routes_context[
            routes_context["route_id"].isin(selected_route_ids)
        ].drop_duplicates()
        for row in selected_routes_context.itertuples(index=False):
            if pd.notna(row.agency_id):
                inferred_rows.append(
                    {
                        o2o_source_col: row.agency_id,
                        o2o_target_col: row.route_id,
                        "ocel:qualifier": "agency belongs to route",
                    }
                )

        if inferred_rows:
            inferred_context_o2o = pd.DataFrame(inferred_rows)
            filtered_o2o = (
                pd.concat([filtered_o2o, inferred_context_o2o], ignore_index=True)
                .drop_duplicates()
                .reset_index(drop=True)
            )

        inferred_object_rows = []
        for row in selected_routes_context.itertuples(index=False):
            inferred_object_rows.append(
                {
                    "ocel:oid": row.route_id,
                    "ocel:type": "route",
                    "route_short_name": row.route_short_name,
                    "route_type": row.route_type,
                    "route_id": row.route_id,
                }
            )

        selected_agency_ids = selected_routes_context["agency_id"].dropna().drop_duplicates()
        selected_agency_context = agency_context[
            agency_context["agency_id"].isin(selected_agency_ids)
        ].drop_duplicates()
        for row in selected_agency_context.itertuples(index=False):
            inferred_object_rows.append(
                {
                    "ocel:oid": row.agency_id,
                    "ocel:type": "agency",
                    "agency_name": getattr(row, "agency_name", pd.NA),
                    "agency_url": getattr(row, "agency_url", pd.NA),
                    "agency_timezone": getattr(row, "agency_timezone", pd.NA),
                    "agency_id": row.agency_id,
                }
            )

        if inferred_object_rows:
            inferred_context_objects = pd.DataFrame(inferred_object_rows)
            for column in ocel_object.columns:
                if column not in inferred_context_objects.columns:
                    inferred_context_objects[column] = pd.NA
            inferred_context_objects = inferred_context_objects[ocel_object.columns]

    o2o_for_traversal = (
        pd.concat([ocel_o2o, inferred_context_o2o], ignore_index=True)
        .drop_duplicates()
        .reset_index(drop=True)
    )
    filtered_o2o, traversed_object_ids = traverse_o2o_context(
        o2o_for_traversal,
        direct_trip_ids,
        o2o_source_col,
        o2o_target_col,
    )

    o2o_related_object_ids = pd.concat(
        [
            filtered_o2o[o2o_source_col],
            filtered_o2o[o2o_target_col],
        ],
        ignore_index=True,
    ).dropna().astype("string")

    output_object_ids = pd.Series(related_object_ids, dtype="string")
    output_object_ids = pd.concat(
        [output_object_ids, traversed_object_ids, o2o_related_object_ids],
        ignore_index=True,
    ).drop_duplicates().to_numpy()

filtered_objects = ocel_object[
    ocel_object["ocel:oid"].isin(output_object_ids)
]

if "inferred_context_objects" in locals() and not inferred_context_objects.empty:
    filtered_objects = pd.concat(
        [filtered_objects, inferred_context_objects],
        ignore_index=True,
    ).drop_duplicates(subset=["ocel:oid"], keep="first")

filtered_objects = (
    filtered_objects
    .set_index("ocel:oid")
    .reindex(output_object_ids)
    .dropna(how="all")
    .reset_index()
)

filtered_objects_for_output = filtered_objects.copy()
if "ocel:type" in filtered_objects_for_output.columns:
    filtered_objects_for_output["_type_order"] = (
        filtered_objects_for_output["ocel:type"]
        .map(OBJECT_TYPE_ORDER)
        .fillna(len(OBJECT_TYPE_ORDER))
    )
    filtered_objects_for_output = (
        filtered_objects_for_output
        .sort_values(["_type_order", "ocel:type", "ocel:oid"], na_position="last")
        .drop(columns=["_type_order"])
        .reset_index(drop=True)
    )

print("Events:")
print(tabulate(
    filtered_events,
    headers="keys",
    tablefmt="psql",
    showindex=False
))

print("\nRelations:")
print(tabulate(
    filtered_relations,
    headers="keys",
    tablefmt="psql",
    showindex=False
))

print("\nObject-to-object relations:")
print(tabulate(
    filtered_o2o,
    headers="keys",
    tablefmt="psql",
    showindex=False
))

print("\nObjects:")
print(tabulate(
    filtered_objects_for_output,
    headers="keys",
    tablefmt="psql",
    showindex=False
))

print("\nObject counts by type:")
print(tabulate(
    filtered_objects_for_output.groupby("ocel:type", dropna=False)["ocel:oid"]
    .nunique()
    .reset_index(name="count"),
    headers="keys",
    tablefmt="psql",
    showindex=False
))



print("\nDifferent route_type values:")
print(
    ocel_object["route_type"]
    .dropna()
    .astype("string")
    .drop_duplicates()
    .sort_values()
    .to_list()
)
"""
ocel_e2o_object = ocel_e2o.merge(ocel_object, on="ocel:oid", how="left")

#print(tabulate(ocel_e2o_object.head(100), headers="keys", tablefmt="psql", showindex=False))


tmp_ocel_e2o = None

#print(ocel_e2o_object["ocel:type_x"].unique())

for type in ocel_e2o_object["ocel:type_x"].unique():
    if tmp_ocel_e2o is None:
        tmp_ocel_e2o = ocel_e2o_object[ocel_e2o_object["ocel:type_x"] == type].drop(columns=["ocel:type_x", "ocel:type_y"])
    else:
        tmp_ocel_e2o = tmp_ocel_e2o.merge(ocel_e2o_object[ocel_e2o_object["ocel:type_x"] == type].drop(columns=["ocel:type_x", "ocel:type_y"]), on='ocel:eid', how="left").reindex(tmp_ocel_e2o.index)

print(tabulate(tmp_ocel_e2o.head(100), headers="keys", tablefmt="psql", showindex=False))



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


print(tabulate(
    tmp_ocel_e2o[tmp_ocel_e2o["route_short_name"]==10].sort_values(["ocel:timestamp", "ocel:activity"]).head(100),
   # tmp_ocel_e2o.sort_values(["ocel:timestamp", "ocel:activity"]).head(100),

    headers="keys",
    tablefmt="psql",
    showindex=False
))

overlap = [
    col for col in tmp_ocel_e2o.columns
    if col in ocel_event.columns and col != "ocel:eid"
]

tmp_ocel_e2o_clean = tmp_ocel_e2o.drop(columns=overlap)

final_ocel = ocel_event.merge(
    tmp_ocel_e2o_clean,
    on="ocel:eid",
    how="left",
    sort=False
)

print(tabulate(
    #final_ocel[(final_ocel["route_short_name"]==10) & (final_ocel["vehicle_id"]=='veh_0') & (final_ocel["ocel:activity"]=='parking')].sort_values(["ocel:timestamp", "ocel:activity"]).head(2000),
    #final_ocel[ (final_ocel["vehicle_id"]=='veh_35')].sort_values(["ocel:timestamp", "ocel:activity"]).head(2000),
    final_ocel[ (final_ocel["route_short_name"]=='2')].sort_values(["ocel:timestamp", "ocel:activity"]).head(2000),
    # tmp_ocel_e2o.sort_values(["ocel:timestamp", "ocel:activity"]).head(100),

    headers="keys",
    tablefmt="psql",
    showindex=False
))

final_ocel.to_csv("final_ocel_log.csv", index=False)

"""
