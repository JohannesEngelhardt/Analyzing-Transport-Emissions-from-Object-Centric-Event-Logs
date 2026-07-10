import pandas as pd

from tabulate import tabulate

#

ACTIVITY_ORDER = {
    "begin_shift": 0,
    "direction_change": 1,
    "end_layover": 2,
    "arrive_stop": 3,
    "departure_stop": 4,
    "begin_layover": 5,
    "parking": 6,
}


def sort_by_activity_order(df, timestamp_col, activity_col):
    ordered_df = df.copy()
    ordered_df["_timestamp_sort"] = pd.to_datetime(ordered_df[timestamp_col])
    ordered_df["_activity_order"] = (
        ordered_df[activity_col].map(ACTIVITY_ORDER).fillna(len(ACTIVITY_ORDER))
    )
    ordered_df = ordered_df.sort_values(
        ["_timestamp_sort", "_activity_order", activity_col]
    )
    return ordered_df.drop(
        columns=["_timestamp_sort", "_activity_order"]
    ).reset_index(drop=True)

"""
Inital Loads

"""
df_stops = pd.read_csv('/Users/johannesengelhardt/PycharmProjects/Masterthesis3/gtfs_data/stops.txt', sep="\t")
df_stop_times = pd.read_csv('/Users/johannesengelhardt/PycharmProjects/Masterthesis3/gtfs_data/stop_times.txt', sep="\t")
df_routes = pd.read_csv('/Users/johannesengelhardt/PycharmProjects/Masterthesis3/gtfs_data/routes.txt',  encoding="latin1", sep="\t")
df_agency = pd.read_csv('/Users/johannesengelhardt/PycharmProjects/Masterthesis3/gtfs_data/agency.txt', sep="\t")
df_calendar_dates = pd.read_csv('/Users/johannesengelhardt/PycharmProjects/Masterthesis3/gtfs_data/calendar_dates.txt', sep="\t")
df_trips = pd.read_csv('/Users/johannesengelhardt/PycharmProjects/Masterthesis3/gtfs_data/trips.txt', sep="\t")

#print(df_trips["vehicle_id"].nunique())

#print(df_routes["route_type"].nunique())

df_trips["vehicle_id"] = "veh_" + pd.factorize(df_trips["vehicle_id"])[0].astype(str)

#print(tabulate(df_trips.sort_values([ "service_id", "start_time", "end_time"]).head(100), headers="keys", tablefmt="psql", showindex=False))
#print(tabulate(df_stops.head(10), headers="keys", tablefmt="psql", showindex=False))

#print(tabulate(df_stop_times[df_stop_times['trip_id']].head(10), headers="keys", tablefmt="psql", showindex=False))
#print(tabulate(df_stop_times[df_stop_times["trip_id"]==10406].sort_values("stop_sequence"),headers="keys"))
#print(df_stop_times.columns)

"""
Event Table - Creation of Events

"""

event_log_arrival = df_stop_times.sort_values("stop_sequence").merge(df_stops, on="stop_id", how="left")
#print(tabulate(event_log_arrival, headers="keys", tablefmt="psql", showindex=False))

event_log_departure = df_stop_times.sort_values("stop_sequence").merge(df_stops, on="stop_id", how="left")
#print(tabulate(event_log_arrival, headers="keys", tablefmt="psql", showindex=False))



event_log_arrival = event_log_arrival.drop(columns=["departure_time"])

event_log_departure = event_log_departure.drop(columns=["arrival_time"])

event_log_arrival["activity"]= "arrive_stop"

event_log_departure["activity"]= "departure_stop"

#print(tabulate(event_log_arrival.head(10), headers="keys", tablefmt="psql", showindex=False))

#print(tabulate(event_log_departure, headers="keys", tablefmt="psql", showindex=False))

event_log_departure= event_log_departure[["trip_id","activity","departure_time", "stop_id", "stop_sequence", "stop_name","stop_lat", "stop_lon"]]

event_log_arrival= event_log_arrival[[ "trip_id","activity","arrival_time", "stop_id", "stop_sequence", "stop_name","stop_lat", "stop_lon"]]

event_log_arrival = event_log_arrival.rename(columns= {"arrival_time": "timestamp"})

event_log_departure = event_log_departure.rename(columns= {"departure_time": "timestamp"})

event_log = pd.concat([event_log_arrival, event_log_departure],ignore_index=True)
event_log = sort_by_activity_order(event_log, "timestamp", "activity")



"""

#Enrichment of Auxillary Event Log

"""


#aux_event_log = event_log.merge(df_trips[df_trips["vehicle_id"]=="7ba503cd-4593-444c-b868-2985edc5b1cb"], on="trip_id", how="left")
aux_event_log = event_log.merge(df_trips, on="trip_id", how="left")


aux_event_log = aux_event_log.merge(df_routes, on="route_id", how="left")
print(tabulate(aux_event_log.head(100), headers="keys", tablefmt="psql"))
#print(tabulate(aux_event_log.sort_values([ "service_id", "start_time", "end_time"]).head(100), headers="keys", tablefmt="psql", showindex=False))


aux_event_log = aux_event_log.merge(df_agency, on="agency_id", how="left")

aux_event_log = aux_event_log.merge(df_calendar_dates, on="service_id", how="left")

aux_event_log["timestamp_hourly"] = aux_event_log["timestamp"]

aux_event_log["timestamp"] = aux_event_log["date"] + ' ' + aux_event_log["timestamp_hourly"]

aux_event_log = aux_event_log.reset_index(drop=True)
aux_event_log["event_id"] = "ad_" + (aux_event_log.index + 1).astype(str)

cols = ["event_id"] + [c for c in aux_event_log.columns if c != "event_id"]
aux_event_log = aux_event_log[cols]





#print(tabulate(aux_event_log[aux_event_log["route_short_name"]==10].sort_values(["timestamp", "activity"]).head(1000), headers="keys", tablefmt="psql", showindex=False))
#print(tabulate(aux_event_log[(aux_event_log["route_type"]==3) | (aux_event_log["vehicle_id"]=='d6917582-ac81-4329-af24-ebba569e9323') & (aux_event_log["activity"]== 'departure_stop')].sort_values([ "timestamp"]).head(1000)[["event_id", "trip_id", "activity", "timestamp", "stop_sequence", "stop_name", "direction_id", "start_time", "end_time", "vehicle_id", "route_type"]], headers="keys", tablefmt="psql", showindex=False))

df = aux_event_log
#[
    ##(aux_event_log["route_short_name"] == 10) |
    #(
        #(aux_event_log["vehicle_id"] == 'veh_35') &
        #(aux_event_log["activity"] == 'arrive_stop')
    #)
#]

cols = [
    "event_id", "trip_id", "activity", "timestamp", "stop_id", "stop_sequence",
    "stop_name", "stop_lat", "stop_lon", "direction_id", "start_time", "end_time",
    "vehicle_id", "route_type", "route_short_name", "route_id", "service_id", "agency_id"
]


df = df[cols]

# Indizes für min/max timestamp pro trip
idx_min = df.groupby("trip_id")["timestamp"].idxmin()
idx_max = df.groupby("trip_id")["timestamp"].idxmax()

# Zeilen holen
df_result = pd.concat([df.loc[idx_min], df.loc[idx_max]]) \
   .sort_values(["trip_id", "timestamp"])

#df_result = df.loc[idx_min].sort_values(["trip_id", "timestamp"])


#print(tabulate(df_result[(df_result["route_short_name"]==10) & (df_result["vehicle_id"]=='veh_35')].sort_values("timestamp").head(1000), headers="keys", tablefmt="psql", showindex=False))
#print(tabulate(df_result[df_result["vehicle_id"]=='veh_35'].sort_values("timestamp").head(100), headers="keys", tablefmt="psql", showindex=False))

#print(tabulate(df_result.sort_values("timestamp").head(1000), headers="keys", tablefmt="psql", showindex=False))

# A-Events erzeugen: pro vehicle_id und route_short_name prüfen,
# ob der nächste Trip innerhalb von 30 Minuten startet.

a_event_rows = []

df_base = df_result.copy()
df_base["timestamp_dt"] = pd.to_datetime(df_base["timestamp"])

a_event_rows = []

df_base = df_result.copy()
df_base["timestamp_dt"] = pd.to_datetime(df_base["timestamp"])

for (vehicle_id, route_short_name), temp_table in df_base.groupby(["vehicle_id", "route_short_name"]):
    temp_table = temp_table.sort_values("timestamp_dt").reset_index(drop=True)

    for i in range(len(temp_table) - 1):
        current_row = temp_table.iloc[i]
        next_row = temp_table.iloc[i + 1]

        if current_row["trip_id"] == next_row["trip_id"]:
            continue

        current_date = current_row["timestamp_dt"].date()
        next_date = next_row["timestamp_dt"].date()

        current_end_dt = pd.to_datetime(
            str(current_date) + " " + str(current_row["end_time"])
        )

        next_start_dt = pd.to_datetime(
            str(next_date) + " " + str(next_row["start_time"])
        )

        gap = next_start_dt - current_end_dt

        if current_date == next_date and pd.Timedelta(0) <= gap < pd.Timedelta(minutes=30):
            a_event_rows.append({
                "old_trip_id": current_row["trip_id"],
                "trip_id": next_row["trip_id"],
                "activity": "direction_change",
                "timestamp": next_start_dt,
                "stop_id": None,
                "stop_sequence": None,
                "stop_name": None,
                "direction_id": next_row["direction_id"],
                "vehicle_id": vehicle_id,
                "route_type": current_row["route_type"],
                "route_short_name": route_short_name,
                "route_id": next_row["route_id"],
                "service_id": next_row["service_id"],
                "agency_id": next_row["agency_id"],
            })
        else:
            a_event_rows.append({
                "old_trip_id": current_row["trip_id"],
                "trip_id": None,
                "activity": "parking",
                "timestamp": current_end_dt,
                "stop_id": None,
                "stop_sequence": None,
                "stop_name": None,
                "direction_id": current_row["direction_id"],
                "vehicle_id": vehicle_id,
                "route_type": current_row["route_type"],
                "route_short_name": current_row["route_short_name"],
                "route_id": current_row["route_id"],
                "service_id": current_row["service_id"],
                "agency_id": current_row["agency_id"],
            })

a_events = pd.DataFrame(a_event_rows)

if not a_events.empty:
    a_events = a_events.reset_index(drop=True)
    a_events["event_id"] = "cdp_" + (a_events.index + 1).astype(str)

    a_events = a_events[[
        "event_id",
        "trip_id",
        "old_trip_id",
        "activity",
        "timestamp",
        "stop_id",
        "stop_sequence",
        "stop_name",
        "direction_id",
        "vehicle_id",
        "route_type",
        "route_short_name",
        "route_id",
        "service_id",
        "agency_id"
    ]]
else:
    a_events = pd.DataFrame(columns=[
        "event_id",
        "trip_id",
        "old_trip_id",
        "activity",
        "timestamp",
        "stop_id",
        "stop_sequence",
        "stop_name",
        "direction_id",
        "vehicle_id",
        "route_type",
        "route_short_name",
        "route_id",
        "service_id",
        "agency_id"
    ])

#print(tabulate(
 #   a_events.head(100),
  #  headers="keys",
   # tablefmt="psql",
    #showindex=False
#))


# begin_layover: letzte departure_stop pro trip

"""
begin shift activity 


"""
begin_shift_rows = []

df_begin_base = df_result.copy()
df_begin_base["timestamp_dt"] = pd.to_datetime(df_begin_base["timestamp"])

# Für begin_shift wollen wir auf den Start des aktuellen Trips schauen
df_begin_base = df_begin_base.sort_values(["vehicle_id", "route_short_name", "timestamp_dt"]).reset_index(drop=True)

for (vehicle_id, route_short_name), temp_table in df_begin_base.groupby(["vehicle_id", "route_short_name"]):
    temp_table = temp_table.sort_values("timestamp_dt").reset_index(drop=True)

    for i in range(len(temp_table)):
        current_row = temp_table.iloc[i]
        current_date = current_row["timestamp_dt"].date()

        current_start_dt = pd.to_datetime(
            str(current_date) + " " + str(current_row["start_time"])
        )

        # Fall 1: kein Vorgänger in dieser vehicle/route-Gruppe
        if i == 0:
            begin_shift_rows.append({
                "trip_id": current_row["trip_id"],
                "activity": "begin_shift",
                "timestamp": current_start_dt,
                "stop_id": current_row["stop_id"],
                "stop_sequence": current_row["stop_sequence"],
                "stop_name": current_row["stop_name"],
                "stop_lat": current_row["stop_lat"],
                "stop_lon": current_row["stop_lon"],
                "direction_id": current_row["direction_id"],
                "vehicle_id": vehicle_id,
                "route_type": current_row["route_type"],
                "route_short_name": route_short_name,
                "route_id": current_row["route_id"],
                "service_id": current_row["service_id"],
                "agency_id": current_row["agency_id"],
            })
            continue

        # Fall 2: Vorgänger existiert, aber Lücke > 30 Minuten
        prev_row = temp_table.iloc[i - 1]
        prev_date = prev_row["timestamp_dt"].date()

        prev_end_dt = pd.to_datetime(
            str(prev_date) + " " + str(prev_row["end_time"])
        )

        gap = current_start_dt - prev_end_dt

        if gap > pd.Timedelta(minutes=30):
            begin_shift_rows.append({
                "trip_id": current_row["trip_id"],
                "activity": "begin_shift",
                "timestamp": current_start_dt,
                "stop_id": current_row["stop_id"],
                "stop_sequence": current_row["stop_sequence"],
                "stop_name": current_row["stop_name"],
                "stop_lat": current_row["stop_lat"],
                "stop_lon": current_row["stop_lon"],
                "direction_id": current_row["direction_id"],
                "vehicle_id": vehicle_id,
                "route_type": current_row["route_type"],
                "route_short_name": route_short_name,
                "route_id": current_row["route_id"],
                "service_id": current_row["service_id"],
                "agency_id": current_row["agency_id"],
            })

begin_shift_events = pd.DataFrame(begin_shift_rows)

if not begin_shift_events.empty:
    begin_shift_events = begin_shift_events.reset_index(drop=True)
    begin_shift_events["event_id"] = "bs_" + (begin_shift_events.index + 1).astype(str)

    begin_shift_events = begin_shift_events[[
        "event_id",
        "trip_id",
        "activity",
        "timestamp",
        "stop_id",
        "stop_sequence",
        "stop_name",
        "stop_lat",
        "stop_lon",
        "direction_id",
        "vehicle_id",
        "route_type",
        "route_short_name",
        "route_id",
        "service_id",
        "agency_id",
    ]]
else:
    begin_shift_events = pd.DataFrame(columns=[
        "event_id",
        "trip_id",
        "activity",
        "timestamp",
        "stop_id",
        "stop_sequence",
        "stop_name",
        "stop_lat",
        "stop_lon",
        "direction_id",
        "vehicle_id",
        "route_type",
        "route_short_name",
        "route_id",
        "service_id",
        "agency_id",
    ])

print(tabulate(
    begin_shift_events.head(10),
    headers="keys",
    tablefmt="psql",
    showindex=False
))


"""

begin layer activity

"""

aux_departure = aux_event_log[aux_event_log["activity"] == "departure_stop"].copy()
aux_departure["timestamp_dt"] = pd.to_datetime(aux_departure["timestamp"])

idx_last_departure = aux_departure.groupby("trip_id")["timestamp_dt"].idxmax()

begin_layover_events = aux_departure.loc[idx_last_departure].copy()
begin_layover_events["activity"] = "begin_layover"
begin_layover_events = begin_layover_events.drop(columns=["timestamp_dt"])

begin_layover_events = begin_layover_events.reset_index(drop=True)
begin_layover_events["event_id"] = "bl_" + (begin_layover_events.index + 1).astype(str)

print(tabulate(
    begin_layover_events.head(10),
    headers="keys",
    tablefmt="psql",
    showindex=False
))

aux_e2o_layover_trip = begin_layover_events[["event_id", "activity", "timestamp"]].copy()
aux_e2o_layover_trip["oid"] = begin_layover_events["trip_id"]
aux_e2o_layover_trip["type"] = "trip"
aux_e2o_layover_trip["qualifier"] = "conduct trip"

aux_e2o_layover_route = begin_layover_events[["event_id", "activity", "timestamp"]].copy()
aux_e2o_layover_route["oid"] = begin_layover_events["route_id"]
aux_e2o_layover_route["type"] = "route"
aux_e2o_layover_route["qualifier"] = "conduct route"

aux_e2o_layover_service = begin_layover_events[["event_id", "activity", "timestamp"]].copy()
aux_e2o_layover_service["oid"] = begin_layover_events["service_id"]
aux_e2o_layover_service["type"] = "service"
aux_e2o_layover_service["qualifier"] = "conduct service"

aux_e2o_layover_stop = begin_layover_events[["event_id", "activity", "timestamp"]].copy()
aux_e2o_layover_stop["oid"] = begin_layover_events["stop_id"]
aux_e2o_layover_stop["type"] = "stop"
aux_e2o_layover_stop["qualifier"] = "used bus stop"

aux_e2o_layover_agency = begin_layover_events[["event_id", "activity", "timestamp"]].copy()
aux_e2o_layover_agency["oid"] = begin_layover_events["agency_id"]
aux_e2o_layover_agency["type"] = "agency"
aux_e2o_layover_agency["qualifier"] = "used transport agency"

e2o_layover_table = pd.concat(
    [
        aux_e2o_layover_trip,
        aux_e2o_layover_route,
        aux_e2o_layover_service,
        aux_e2o_layover_stop,
        aux_e2o_layover_agency,
    ],
    ignore_index=True
)

e2o_layover_table = e2o_layover_table.sort_values("timestamp").reset_index(drop=True)


"""
end layover activity

"""

# end_layover: erste arrive_stop pro trip

aux_arrive = aux_event_log[aux_event_log["activity"] == "arrive_stop"].copy()
aux_arrive["timestamp_dt"] = pd.to_datetime(aux_arrive["timestamp"])

idx_first_arrive = aux_arrive.groupby("trip_id")["timestamp_dt"].idxmin()

end_layover_events = aux_arrive.loc[idx_first_arrive].copy()
end_layover_events["activity"] = "end_layover"
end_layover_events = end_layover_events.drop(columns=["timestamp_dt"])

end_layover_events = end_layover_events.reset_index(drop=True)
end_layover_events["event_id"] = "el_" + (end_layover_events.index + 1).astype(str)

print(tabulate(
    end_layover_events.head(100),
    headers="keys",
    tablefmt="psql",
    showindex=False
))

aux_e2o_end_layover_trip = end_layover_events[["event_id", "activity", "timestamp"]].copy()
aux_e2o_end_layover_trip["oid"] = end_layover_events["trip_id"]
aux_e2o_end_layover_trip["type"] = "trip"
aux_e2o_end_layover_trip["qualifier"] = "conduct trip"

aux_e2o_end_layover_route = end_layover_events[["event_id", "activity", "timestamp"]].copy()
aux_e2o_end_layover_route["oid"] = end_layover_events["route_id"]
aux_e2o_end_layover_route["type"] = "route"
aux_e2o_end_layover_route["qualifier"] = "conduct route"

aux_e2o_end_layover_service = end_layover_events[["event_id", "activity", "timestamp"]].copy()
aux_e2o_end_layover_service["oid"] = end_layover_events["service_id"]
aux_e2o_end_layover_service["type"] = "service"
aux_e2o_end_layover_service["qualifier"] = "conduct service"

aux_e2o_end_layover_stop = end_layover_events[["event_id", "activity", "timestamp"]].copy()
aux_e2o_end_layover_stop["oid"] = end_layover_events["stop_id"]
aux_e2o_end_layover_stop["type"] = "stop"
aux_e2o_end_layover_stop["qualifier"] = "used bus stop"

aux_e2o_end_layover_agency = end_layover_events[["event_id", "activity", "timestamp"]].copy()
aux_e2o_end_layover_agency["oid"] = end_layover_events["agency_id"]
aux_e2o_end_layover_agency["type"] = "agency"
aux_e2o_end_layover_agency["qualifier"] = "used transport agency"

e2o_end_layover_table = pd.concat(
    [
        aux_e2o_end_layover_trip,
        aux_e2o_end_layover_route,
        aux_e2o_end_layover_service,
        aux_e2o_end_layover_stop,
        aux_e2o_end_layover_agency,
    ],
    ignore_index=True
).sort_values("timestamp").reset_index(drop=True)

events_end_layover = pd.DataFrame([
    {
        "id": e.event_id,
        "type": e.activity,
        "time": e.timestamp,
        "attributes": [
            {"name": "latitude", "value": e.stop_lat},
            {"name": "longitude", "value": e.stop_lon},
            {"name": "stop_sequence", "value": e.stop_sequence},
            {"name": "route_short_name", "value": e.route_short_name},
        ],
        "relationships": [
            {"objectId": e.trip_id, "qualifier": "conduct trip"},
            {"objectId": e.stop_id, "qualifier": "used bus stop"},
            {"objectId": e.service_id, "qualifier": "conduct service"},
            {"objectId": e.agency_id, "qualifier": "used transport agency"},
            {"objectId": e.route_id, "qualifier": "conduct route"},
        ],
    }
    for e in end_layover_events.itertuples(index=False)
])



"""

#Event-to-Object Table Creation

!!!Achtung!!! hier ist mal nur die Objekte genommen die wir als Tabelle haben

"""
aux_e2o_trip = aux_event_log[["event_id", "activity", "timestamp"] ]

aux_e2o_trip["oid"] = aux_event_log["trip_id"]

aux_e2o_trip["type"] = "trip"

aux_e2o_trip["qualifier"] = "conduct trip"

aux_e2o_route = aux_event_log[["event_id", "activity", "timestamp"] ]

aux_e2o_route["oid"] = aux_event_log["route_id"]

aux_e2o_route["type"] = "route"

aux_e2o_route["qualifier"] = "conduct route"

aux_e2o_service = aux_event_log[["event_id", "activity", "timestamp"] ]

aux_e2o_service["oid"] = aux_event_log["service_id"]

aux_e2o_service["type"] = "service"

aux_e2o_service["qualifier"] = "conduct service"

aux_e2o_stops = aux_event_log[["event_id", "activity", "timestamp"] ]

aux_e2o_stops["oid"] = aux_event_log["stop_id"]

aux_e2o_stops["type"] = "stop"

aux_e2o_stops["qualifier"] = "used bus stop"

aux_e2o_agency = aux_event_log[["event_id", "activity", "timestamp"] ]

aux_e2o_agency["oid"] = aux_event_log["agency_id"]

aux_e2o_agency["type"] = "agency"

aux_e2o_agency["qualifier"] = "used transport agency"

e2o_table = pd.concat([aux_e2o_stops, aux_e2o_route, aux_e2o_agency, aux_e2o_service, aux_e2o_trip],ignore_index=True)
e2o_table = e2o_table.sort_values(["timestamp"]).reset_index(drop=True)

#print(tabulate(e2o_table, headers="keys", tablefmt="psql", showindex=False))

"""
Aux_e2o Aktivität A

"""

activity_a_events = a_events[a_events["activity"] == "direction_change"].copy()

aux_e2o_A_trip = activity_a_events[["event_id", "activity", "timestamp"] ]

aux_e2o_A_trip["oid"] = activity_a_events["trip_id"]

aux_e2o_A_trip["type"] = "trip"

aux_e2o_A_trip["qualifier"] = "conduct trip"

aux_e2o_A_old_trip = activity_a_events[["event_id", "activity", "timestamp"]]

aux_e2o_A_old_trip["oid"] = activity_a_events["old_trip_id"]

aux_e2o_A_old_trip["type"] = "trip"

aux_e2o_A_old_trip["qualifier"] = "recently conducted trip"

aux_e2o_A_route = activity_a_events[["event_id", "activity", "timestamp"]]

aux_e2o_A_route["oid"] = activity_a_events["route_id"]

aux_e2o_A_route["type"] = "route"

aux_e2o_A_route["qualifier"] = "conduct route"

aux_e2o_A_service = activity_a_events[["event_id", "activity", "timestamp"] ]

aux_e2o_A_service["oid"] = activity_a_events["service_id"]

aux_e2o_A_service["type"] = "service"

aux_e2o_A_service["qualifier"] = "conduct service"

aux_e2o_A_agency = activity_a_events[["event_id", "activity", "timestamp"] ]

aux_e2o_A_agency["oid"] = activity_a_events["agency_id"]

aux_e2o_A_agency["type"] = "agency"

aux_e2o_A_agency["qualifier"] = "used transport agency"


e2o_a_table = pd.concat([aux_e2o_A_trip, aux_e2o_A_old_trip, aux_e2o_A_route, aux_e2o_A_service, aux_e2o_A_agency],ignore_index=True)
e2o_a_table = e2o_a_table.sort_values(["timestamp"]).reset_index(drop=True)

#print(tabulate(e2o_a_table.head(1000), headers="keys", tablefmt="psql"))

"""
Aktivität B
"""
activity_b_events = a_events[a_events["activity"] == "parking"].copy()

aux_e2o_B_old_trip = activity_b_events[["event_id", "activity", "timestamp"]].copy()
aux_e2o_B_old_trip["oid"] = activity_b_events["old_trip_id"]
aux_e2o_B_old_trip["type"] = "trip"
aux_e2o_B_old_trip["qualifier"] = "recently conducted trip"

aux_e2o_B_route = activity_b_events[["event_id", "activity", "timestamp"]].copy()
aux_e2o_B_route["oid"] = activity_b_events["route_id"]
aux_e2o_B_route["type"] = "route"
aux_e2o_B_route["qualifier"] = "recently conducted route"

aux_e2o_B_service = activity_b_events[["event_id", "activity", "timestamp"]].copy()
aux_e2o_B_service["oid"] = activity_b_events["service_id"]
aux_e2o_B_service["type"] = "service"
aux_e2o_B_service["qualifier"] = "conduct service"

aux_e2o_B_agency = activity_b_events[["event_id", "activity", "timestamp"]].copy()
aux_e2o_B_agency["oid"] = activity_b_events["agency_id"]
aux_e2o_B_agency["type"] = "agency"
aux_e2o_B_agency["qualifier"] = "used transport agency"

e2o_b_table = pd.concat(
    [
        aux_e2o_B_old_trip,
        aux_e2o_B_route,
        aux_e2o_B_service,
        aux_e2o_B_agency,
    ],
    ignore_index=True
)

e2o_b_table = e2o_b_table.dropna(subset=["oid"])
e2o_b_table = e2o_b_table.sort_values("timestamp").reset_index(drop=True)

#print(tabulate(e2o_b_table.head(1000), headers="keys", tablefmt="psql"))

e2o_cols = ["event_id", "activity", "timestamp", "oid", "type", "qualifier"]

e2o_table = e2o_table[e2o_cols]
e2o_a_table = e2o_a_table[e2o_cols]
e2o_b_table = e2o_b_table[e2o_cols]

e2o_complete_table = pd.concat(
    [e2o_table, e2o_a_table, e2o_b_table, e2o_end_layover_table,e2o_layover_table],
    ignore_index=True
)

#e2o_complete_table = e2o_complete_table.dropna(subset=["oid"])
#e2o_complete_table = e2o_complete_table.sort_values("timestamp").reset_index(drop=True)
"""
print(tabulate(
    e2o_complete_table.head(1000),
    headers="keys",
    tablefmt="psql",
    showindex=False
))
"""



"""
#Object Table Creation

"""

object_table_trip =  df_trips.copy()

object_table_trip["type"] = "trip"

object_table_trip["oid"] = df_trips["trip_id"]


object_table_trip["type"] = "trip"

object_table_trip = object_table_trip.drop(columns=["route_id", "service_id", "trip_id"])

object_table_route =  df_routes.copy()



object_table_route["type"] = "route"

object_table_route["oid"] = df_routes["route_id"]

object_table_route = object_table_route.drop(columns=["route_id", "agency_id"])

#print(tabulate(object_table_route, headers="keys", tablefmt="psql", showindex=False))


object_table_stops =  df_stops.copy()

object_table_stops["type"] = "stops"

object_table_stops["oid"] = df_stops["stop_id"]

object_table_stops = object_table_stops.drop(columns=["stop_id"])





object_table_agency = df_agency.copy()

object_table_agency["type"] = "agency"

object_table_agency["oid"] = df_agency["agency_id"]

object_table_agency = object_table_agency.drop(columns=["agency_id"])


object_table_calendar_dates = df_calendar_dates.copy()

object_table_calendar_dates["type"] = "service"

object_table_calendar_dates["oid"] = df_calendar_dates["service_id"]

object_table_calendar_dates = object_table_calendar_dates.drop(columns=["service_id"])

object_table = pd.concat([object_table_trip, object_table_stops, object_table_agency, object_table_route, object_table_calendar_dates])

#print(print(tabulate(object_table.head(40), headers="keys", tablefmt="psql", showindex=False)))

#print(object_table.columns)

#print(list(object_table.columns))
#print((object_table["vehicle_id"].nunique()))

"""
#O2O Object Table

"""

aux_route_trip = aux_event_log[["route_id", "trip_id"]].rename(
    columns={"route_id": "ocel_source_id", "trip_id": "ocel_target_id"}
).drop_duplicates()

aux_route_trip["oid_qualifier"] = "trip belongs to route"


aux_trip_trip = a_events[["trip_id", "old_trip_id"]].rename(columns={"trip_id": "ocel_source_id", "old_trip_id": "ocel_target_id"}
).drop_duplicates()

aux_trip_trip["oid_qualifier"] = "trip is preceded by trip"

#print(tabulate(aux_route_trip, headers="keys", tablefmt="psql", showindex=False))

aux_trip_stops = aux_event_log[["trip_id", "stop_id"]].rename(
    columns={"trip_id": "ocel_source_id", "stop_id": "ocel_target_id"}
).drop_duplicates()

aux_trip_stops["oid_qualifier"] = "stop belongs to trip"

#print(tabulate(aux_trip_stops, headers="keys", tablefmt="psql", showindex=False))

aux_agency_routes = aux_event_log[["trip_id", "agency_id"]].rename(
    columns={"trip_id": "ocel_source_id", "agency_id": "ocel_target_id"}
).drop_duplicates()

aux_agency_routes["oid_qualifier"] = "agency conducts route"

#print(tabulate(aux_agency_routes, headers="keys", tablefmt="psql", showindex=False))

aux_trips_calendar = aux_event_log[["trip_id", "service_id"]].rename(
    columns={"trip_id": "ocel_source_id", "service_id": "ocel_target_id"}
).drop_duplicates()

aux_trips_calendar["oid_qualifier"] = "trip belongs to service"

#print(tabulate(aux_trips_calendar, headers="keys", tablefmt="psql", showindex=False))

source_target = pd.concat([aux_route_trip,aux_trip_trip, aux_trip_stops, aux_agency_routes, aux_trips_calendar], ignore_index=True)

#print(tabulate(source_target, headers="keys", tablefmt="psql", showindex=False))

activity_a_events = a_events[a_events["activity"] == "direction_change"].copy()
activity_b_events = a_events[a_events["activity"] == "parking"].copy()

print(tabulate(activity_b_events.head(1000), headers="keys",tablefmt="psql", showindex=False))



eventTypes = pd.DataFrame([{"attributes":
                            [
                                {
                                    "name": "latitude",
                                    "type": "float"
                                },
                                {
                                    "name": "longitude",
                                    "type": "float"
                                }, {
                                    "name": "stop_sequence",
                                    "type": "int"
                                },
                                {  "name": "route_short_name",
                                 "type": "int"}


                            ],
                            "name": "arrive_stop"
                        },
                        {"attributes":
                            [
                                {
                                    "name": "latitude",
                                    "type": "float"
                                },
                                {
                                    "name": "longitude",
                                    "type": "float"
                                }, {
                                    "name": "stop_sequence",
                                    "type": "int"
                                },
                                {  "name": "route_short_name",
                                 "type": "int"}


                            ],
                            "name": "begin_shift"
                        },
                    {"attributes":
                        [
                            {
                                "name": "latitude",
                                "type": "float"
                            },
                            {
                                "name": "longitude",
                                "type": "float"
                            }, {
                            "name": "stop_sequence",
                            "type": "int"
                        },
                            {"name": "route_short_name",
                             "type": "int"}

                        ],
                        "name": "end_layover"
                    },
                            {"attributes":
                                [
                                    {
                                        "name": "latitude",
                                        "type": "float"
                                    },
                                    {
                                        "name": "longitude",
                                        "type": "float"
                                    }, {
                                    "name": "stop_sequence",
                                    "type": "int"
                                },
                                    {"name": "route_short_name",
                                     "type": "int"}

                                ],
                                "name": "begin_layover"
                            }
                                ,
                                {
                                        "attributes":
                                    [
                                        {
                                            "name": "latitude",
                                            "type": "float"
                                        },
                                        {
                                            "name": "longitude",
                                            "type": "float"
                                        }, {
                                            "name": "stop_sequence",
                                            "type": "int"
                                        },

                                        {  "name": "route_short_name",
                                         "type": "int"}


                                    ],
                                    "name": "departure_stop"},
                                    {
                                        "attributes":
                                    [ ],
                                    "name": "direction_change"},
                                    {
                                        "attributes":
                                    [ ],
                                    "name": "parking"}

])

events = pd.DataFrame([ {"id": e.event_id,
                        "type": e.activity,
                        "time": e.timestamp,
                         "attributes": [
                             {
                            "name": "latitude",
                            "value": e.stop_lat
                             },
                            {
                            "name": "longitude",
                            "value": e.stop_lon
                            },
                            {
                            "name": "stop_sequence",
                            "value": e.stop_sequence
                            },
                             {
                                 "name": "route_short_name",
                                 "value": e.route_short_name
                             }
                            ],
                         "relationships": [
                             {
                                 "objectId": e.trip_id,
                                 "qualifier": "conduct trip"
                             },
                             {
                                 "objectId": e.stop_id,
                                 "qualifier": "used bus stop"
                             },
                             {
                                 "objectId": e.service_id,
                                 "qualifier": "conduct service"
                             },
                             {
                                 "objectId": e.agency_id,
                                 "qualifier": "used transport agency"
                             },

                             {
                                 "objectId": e.route_id,
                                 "qualifier": "conduct route"
                             }
                         ]
                        }
                       for e in aux_event_log.itertuples(index=False)])

events_a = pd.DataFrame([
                        {
                            "id": e.event_id,
                            "type": e.activity,
                            "time": e.timestamp,
                            "attributes": [

                            ],
                            "relationships": [
                                {"objectId": e.trip_id, "qualifier": "conduct trip"},
                                {"objectId": e.old_trip_id, "qualifier": "recently conducted trip"},
                                {"objectId": e.service_id, "qualifier": "conduct service"},
                                {"objectId": e.agency_id, "qualifier": "used transport agency"},
                                {"objectId": e.route_id, "qualifier": "conduct route"},
                            ],
                        }
                        for e in activity_a_events.itertuples(index=False)
                    ])


events_b = pd.DataFrame([
                        {
                            "id": e.event_id,
                            "type": e.activity,
                            "time": e.timestamp,
                            "attributes": [

                            ],
                            "relationships": [
                                {"objectId": e.old_trip_id, "qualifier": "recently conducted trip"},
                                {"objectId": e.service_id, "qualifier": "conduct service"},
                                {"objectId": e.agency_id, "qualifier": "used transport agency"},
                                {"objectId": e.route_id, "qualifier": "recently conducted route"},
                            ],
                        }
                        for e in activity_b_events.itertuples(index=False)
                    ])

events_layover = pd.DataFrame([
                    {
                        "id": e.event_id,
                        "type": e.activity,
                        "time": e.timestamp,
                        "attributes": [
                            {"name": "latitude", "value": e.stop_lat},
                            {"name": "longitude", "value": e.stop_lon},
                            {"name": "stop_sequence", "value": e.stop_sequence},
                            {"name": "route_short_name", "value": e.route_short_name},
                        ],
                        "relationships": [
                            {"objectId": e.trip_id, "qualifier": "conduct trip"},
                            {"objectId": e.stop_id, "qualifier": "used bus stop"},
                            {"objectId": e.service_id, "qualifier": "conduct service"},
                            {"objectId": e.agency_id, "qualifier": "used transport agency"},
                            {"objectId": e.route_id, "qualifier": "conduct route"},
                        ],
                    }
                    for e in begin_layover_events.itertuples(index=False)
                ])

events_begin_shift = pd.DataFrame([
                    {
                        "id": e.event_id,
                        "type": e.activity,
                        "time": e.timestamp,
                        "attributes": [
                            {"name": "latitude", "value": e.stop_lat},
                            {"name": "longitude", "value": e.stop_lon},
                            {"name": "stop_sequence", "value": e.stop_sequence},
                            {"name": "route_short_name", "value": e.route_short_name},
                        ],
                        "relationships": [
                            {"objectId": e.trip_id, "qualifier": "conduct trip"},
                            {"objectId": e.stop_id, "qualifier": "used bus stop"},
                            {"objectId": e.service_id, "qualifier": "conduct service"},
                            {"objectId": e.agency_id, "qualifier": "used transport agency"},
                            {"objectId": e.route_id, "qualifier": "conduct route"},
                        ],
                    }
                    for e in begin_shift_events.itertuples(index=False)
                ])




events_complete = pd.concat(
                    [events, events_a, events_b, events_layover, events_end_layover, events_begin_shift],
                    ignore_index=True
                )


aux_event_log_overview = pd.concat(
    [
        aux_event_log,
        begin_shift_events,
        begin_layover_events,
        activity_a_events,
        activity_b_events,
        end_layover_events
    ],
    ignore_index=True
)

aux_event_log_overview = sort_by_activity_order(
    aux_event_log_overview, "timestamp", "activity"
)





events_complete = sort_by_activity_order(events_complete, "time", "type")




object_table_routes = pd.DataFrame([{"id": o.oid,
                        "type": o.type,
                        "attributes": [
                            {
                                "name": "route_long_name",
                                "value": o.route_long_name
                        },
                            {
                                "name": "route_short_name",
                                "value": o.route_short_name
                            },
                            {
                                "name": "route_type",
                                "value": o.route_type
                            }

                        ],
                        "relationships": [
{
                            "objectId": row.ocel_target_id,
                            "qualifier": row.oid_qualifier
                                }
                            for row in source_target[source_target["ocel_source_id"] == o.oid].itertuples(index=False)


                        ]



                        }
                       for o in object_table_route.itertuples(index=False)])

object_table_trip = pd.DataFrame([{"id": o.oid,
                        "type": o.type,
                        "attributes": [
                            {
                                "name": "direction_id",
                                "value": o.direction_id
                        },
                            {
                                "name": "start_time",
                                "value": o.start_time
                            },
                            {
                                "name": "end_time",
                                "value": o.end_time
                            }
                            ,
                            {
                                "name": "vehicle_id",
                                "value": o.vehicle_id
                            }

                        ],
                        "relationships": [
                            {
                            "objectId": row.ocel_target_id,
                            "qualifier": row.oid_qualifier
                                }
                            for row in source_target[source_target["ocel_source_id"] == o.oid].itertuples(index=False)
                        ]
                        }
                       for o in object_table_trip.itertuples(index=False)])

object_table_agency = pd.DataFrame([{"id": o.oid,
                        "type": o.type,
                        "attributes": [
                            {
                                "name": "agency_name",
                                "value": o.agency_name
                        },
                            {
                                "name": "agency_url",
                                "value": o.agency_url
                            },
                            {
                                "name": "agency_timezone",
                                "value": o.agency_timezone
                            }


                        ],
                        "relationships": [
                        {
                            "objectId": row.ocel_target_id,
                            "qualifier": row.oid_qualifier
                                }
                            for row in source_target[source_target["ocel_source_id"] == o.oid].itertuples(index=False)


                        ]



                        }
                       for o in object_table_agency.itertuples(index=False)])


object_table_stops = pd.DataFrame([{"id": o.oid,
                        "type": o.type,
                        "attributes": [
                            {
                                "name": "stop_name",
                                "value": o.stop_name
                        },
                            {
                                "name": "latitude",
                                "value": o.stop_lat
                            },
                            {
                                "name": "longitude",
                                "value": o.stop_lon
                            }


                        ],
                        "relationships": [
                            {
                            "objectId": row.ocel_target_id,
                            "qualifier": row.oid_qualifier
                                }
                            for row in source_target[source_target["ocel_source_id"] == o.oid].itertuples(index=False)


                        ]



                        }
                       for o in object_table_stops.itertuples(index=False)])


object_table_calendar_dates = pd.DataFrame([{"id": o.oid,
                        "type": o.type,
                        "attributes": [
                            {
                                "name": "date",
                                "value": o.date
                        },
                            {
                                "name": "exception_type",
                                "value": o.exception_type
                            }


                        ],
                        "relationships": [
                        {
                            "objectId": row.ocel_target_id,
                            "qualifier": row.oid_qualifier
                                }
                            for row in source_target[source_target["ocel_source_id"] == o.oid].itertuples(index=False)


                        ]



                        }
                       for o in object_table_calendar_dates.itertuples(index=False)])






print(tabulate(object_table_routes, headers="keys", tablefmt="psql"))


print(tabulate(object_table_trip.head(20), headers="keys", tablefmt="psql"))


object_types = pd.DataFrame([{"name": "trip",
                        "attributes": [
                            {
                                "name": "direction_id",
                                "type": "string"
                        },
                            {
                                "name": "start_time",
                                "type": "datetime"
                            },
                            {
                                "name": "end_time",
                                "type": "datetime"
                            },
                            {
                                "name": "vehicle_id",
                                "type": "string"
                            }


                        ]



                        },
                        {"name": "route",
                        "attributes": [
                            {
                                "name": "route_long_name",
                                "type": "string"
                            },
                            {
                                "name": "route_short_name",
                                "type": "string"
                            },
                            {
                                "name": "route_type",
                                "type": "int"
                            }


                        ]},
                        {"name": "agency",
                        "attributes": [
                            {
                                "name": "agency_name",
                                "type": "string"
                            },
                            {
                                "name": "angency_url",
                                "type": "string"
                            },
                            {
                                "name": "agency_timezone",
                                "type": "string"
                            }


                        ]},
                        {"name": "calendar_dates",
                        "attributes": [
                            {
                                "name": "date",
                                "type": "string"
                            },
                            {
                                "name": "exception_type",
                                "type": "string"
                            }


                        ]},
                        {"name": "stops",
                        "attributes": [
                            {
                                "name": "stop_name",
                                "type": "string"
                            },
                            {
                                "name": "latitude",
                                "type": "float"
                            },
                            {
                                "name": "longitude",
                                "type": "float"
                            }


                        ]}


                             ])

import json

ocel = {
    "eventTypes": eventTypes.to_dict(orient="records"),
    "objectTypes": object_types.to_dict(orient="records"),
    "events": events_complete.to_dict(orient="records"),
    "objects": pd.concat([
        object_table_routes,
        object_table_trip,
        object_table_agency,
        object_table_stops,
        object_table_calendar_dates
    ], ignore_index=True).to_dict(orient="records")
}

print(
    tabulate(
        aux_event_log_overview[
            aux_event_log_overview["vehicle_id"] == "veh_35"
        ][
            [
                "event_id",
                "trip_id",
                "activity",
                "timestamp",
                "stop_sequence",
                "stop_name",
                "vehicle_id"
            ]
        ]
        .sort_values(["timestamp", "activity"])
        .head(100),
        headers="keys",
        tablefmt="psql"
    )
)
"""

with open("ocel2.json", "w", encoding="utf-8") as f:
    json.dump(ocel, f, ensure_ascii=False, indent=2, default=str)

# lesen
with open("ocel2.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# falls events im JSON unter "events" liegen
events_df = pd.DataFrame(data["events"])

#print(tabulate(events_df.head(100), headers="keys", tablefmt="psql"))


"""

import seaborn as sns
import matplotlib.pyplot as plt


aux_event_log_overview["trip_id"] = (
    aux_event_log_overview["trip_id"]
    .astype(str)
)

aux_event_log_overview["activity"] = (
    aux_event_log_overview["activity"]
    .astype(str)
)

aux_event_log_overview["timestamp"] = pd.to_datetime(
    aux_event_log_overview["timestamp"]
)

trip_ids = (
    aux_event_log_overview["trip_id"]
    .dropna()
    .unique()[:50]
)

# Nur diese Trips behalten
filtered_df = aux_event_log_overview[
    aux_event_log_overview["trip_id"].isin(trip_ids)
]

# Plot
plt.figure(figsize=(18, 10))

sns.scatterplot(
    data=filtered_df[
            filtered_df["vehicle_id"] == "veh_35"
        ],
    x="timestamp",
    y="trip_id",
    hue="activity",
    s=80
)

plt.title("Dotted Chart der Trips")
plt.xlabel("Zeit")
plt.ylabel("Trip ID")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


aux_event_log_overview.to_csv("aux_event_log_overview.csv", index=False)
