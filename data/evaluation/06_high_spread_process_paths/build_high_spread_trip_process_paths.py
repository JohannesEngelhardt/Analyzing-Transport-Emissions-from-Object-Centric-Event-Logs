from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import pandas as pd
import pm4py
from graphviz import Digraph


DATASET_DATE = "2026_05_04"
DEFAULT_HIGH_SPREAD_FILE = Path(
    f"hourly_occupancy_distribution_{DATASET_DATE}/high_spread_segments_hour_06.csv"
)
DEFAULT_SEGMENTS_FILE = Path(f"trip_segment_summary_{DATASET_DATE}.csv")
DEFAULT_AUX_EVENTS_FILE = Path(f"aux_event_log_overview_kodak_{DATASET_DATE}.csv")
DEFAULT_OUTPUT_DIR = Path(f"hourly_occupancy_distribution_{DATASET_DATE}")

OCCUPANCY_COLORS = {
    0: "#2ca02c",  # green
    1: "#f2c94c",  # yellow
    2: "#f2994a",  # medium red/orange
    3: "#d62728",  # red
}


def parse_trip_ids(value: str) -> list[str]:
    return [
        trip_id.strip()
        for trip_id in str(value).split(",")
        if trip_id and trip_id.strip()
    ]


def load_high_spread_trip_ids(high_spread_file: Path, max_segments: int | None) -> list[str]:
    high_spread = pd.read_csv(high_spread_file, dtype={"trip_ids": "string"})
    if max_segments is not None:
        high_spread = high_spread.head(max_segments)

    trip_ids: list[str] = []
    for trip_id_cell in high_spread["trip_ids"].dropna():
        for trip_id in parse_trip_ids(trip_id_cell):
            if trip_id not in trip_ids:
                trip_ids.append(trip_id)
    return trip_ids


def load_high_spread_segment_ids(high_spread_file: Path, max_segments: int | None) -> list[str]:
    high_spread = pd.read_csv(high_spread_file, usecols=["segment_id"], dtype={"segment_id": "string"})
    if max_segments is not None:
        high_spread = high_spread.head(max_segments)
    return high_spread["segment_id"].dropna().astype(str).drop_duplicates().tolist()


def load_trip_segments(
    segments_file: Path,
    trip_ids: list[str],
    hour: int,
    only_hour_window: bool,
) -> pd.DataFrame:
    columns = [
        "segment_id",
        "trip_id",
        "trip_id_org",
        "departure_timestamp",
        "arrive_timestamp",
        "departure_stop_sequence",
        "departure_stop_name",
        "arrive_stop_sequence",
        "arrive_stop_name",
        "route_short_name",
        "route_type",
        "occupancy_status",
    ]
    segments = pd.read_csv(
        segments_file,
        usecols=columns,
        dtype={
            "segment_id": "string",
            "trip_id": "string",
            "trip_id_org": "string",
            "departure_stop_name": "string",
            "arrive_stop_name": "string",
            "route_short_name": "string",
            "route_type": "string",
        },
    )
    segments = segments[segments["trip_id"].isin(trip_ids)].copy()
    segments["departure_timestamp"] = pd.to_datetime(
        segments["departure_timestamp"],
        errors="coerce",
    )
    segments["arrive_timestamp"] = pd.to_datetime(
        segments["arrive_timestamp"],
        errors="coerce",
    )
    segments["departure_stop_sequence"] = pd.to_numeric(
        segments["departure_stop_sequence"],
        errors="coerce",
    )
    segments["occupancy_status"] = pd.to_numeric(
        segments["occupancy_status"],
        errors="coerce",
    ).fillna(0).astype(int)
    segments = segments.dropna(
        subset=["departure_timestamp", "arrive_timestamp", "departure_stop_sequence"]
    )

    if only_hour_window:
        hour_start = segments["departure_timestamp"].dt.normalize() + pd.to_timedelta(hour, unit="h")
        hour_end = hour_start + pd.Timedelta(hours=1)
        overlaps_hour = (
            (segments["departure_timestamp"] < hour_end)
            & (segments["arrive_timestamp"] >= hour_start)
        )
        segments = segments[overlaps_hour].copy()

    return segments.sort_values(["trip_id", "departure_stop_sequence"]).reset_index(drop=True)


def load_aux_high_spread_segments(
    aux_events_file: Path,
    high_spread_segment_ids: list[str],
    hour: int,
    occupancy_source: str = "departure",
) -> pd.DataFrame:
    if occupancy_source not in {"departure", "arrive"}:
        raise ValueError("occupancy_source must be 'departure' or 'arrive'")

    columns = [
        "timestamp",
        "arrival_time_rt",
        "trip_id",
        "stop_sequence",
        "stop_name",
        "route_short_name",
        "route_type",
        "direction_id",
        "activity_type",
        "trip_id_org",
        "segment_id",
        "occupancy_status",
    ]
    events = pd.read_csv(
        aux_events_file,
        usecols=columns,
        dtype={
            "trip_id": "string",
            "stop_name": "string",
            "route_short_name": "string",
            "route_type": "string",
            "direction_id": "string",
            "activity_type": "string",
            "trip_id_org": "string",
            "segment_id": "string",
        },
    )
    events = events[
        events["segment_id"].isin(high_spread_segment_ids)
        & events["activity_type"].isin(["departure_stop", "arrive_stop"])
    ].copy()
    events["timestamp"] = pd.to_datetime(events["timestamp"], errors="coerce")
    events["arrival_time_rt"] = pd.to_datetime(events["arrival_time_rt"], errors="coerce")
    events["stop_sequence"] = pd.to_numeric(events["stop_sequence"], errors="coerce")
    events["occupancy_status"] = pd.to_numeric(
        events["occupancy_status"],
        errors="coerce",
    ).fillna(0).astype(int)
    events = events.dropna(subset=["timestamp", "stop_sequence"])

    hour_start = events["timestamp"].dt.normalize() + pd.to_timedelta(hour, unit="h")
    hour_end = hour_start + pd.Timedelta(hours=1)
    events = events[(events["timestamp"] >= hour_start) & (events["timestamp"] < hour_end)].copy()

    segment_rows = []
    for (trip_id, segment_id), group in events.groupby(["trip_id", "segment_id"], sort=False):
        departures = group[group["activity_type"] == "departure_stop"].sort_values("timestamp")
        arrivals = group[group["activity_type"] == "arrive_stop"].sort_values("timestamp")
        if departures.empty or arrivals.empty:
            continue

        departure = departures.iloc[0]
        arrival = arrivals.iloc[-1]
        occupancy_event = arrival if occupancy_source == "arrive" else departure
        segment_rows.append(
            {
                "segment_id": str(segment_id),
                "trip_id": str(trip_id),
                "trip_id_org": str(departure["trip_id_org"]),
                "departure_timestamp": departure["timestamp"],
                "arrive_timestamp": arrival["arrival_time_rt"]
                if pd.notna(arrival["arrival_time_rt"])
                else arrival["timestamp"],
                "departure_stop_sequence": departure["stop_sequence"],
                "departure_stop_name": str(departure["stop_name"]),
                "arrive_stop_sequence": arrival["stop_sequence"],
                "arrive_stop_name": str(arrival["stop_name"]),
                "route_short_name": str(departure["route_short_name"]),
                "route_type": str(departure["route_type"]),
                "direction_id": str(departure["direction_id"]),
                "occupancy_status": int(occupancy_event["occupancy_status"]),
            }
        )

    segments = pd.DataFrame(segment_rows)
    if segments.empty:
        return segments
    return segments.sort_values(["trip_id", "departure_stop_sequence"]).reset_index(drop=True)


def load_aux_events(aux_events_file: Path) -> pd.DataFrame:
    columns = [
        "timestamp",
        "arrival_time_rt",
        "trip_id",
        "stop_sequence",
        "stop_name",
        "route_short_name",
        "route_type",
        "direction_id",
        "activity_type",
        "trip_id_org",
        "segment_id",
        "occupancy_status",
    ]
    events = pd.read_csv(
        aux_events_file,
        usecols=columns,
        dtype={
            "trip_id": "string",
            "stop_name": "string",
            "route_short_name": "string",
            "route_type": "string",
            "direction_id": "string",
            "activity_type": "string",
            "trip_id_org": "string",
            "segment_id": "string",
        },
    )
    events = events[events["activity_type"].isin(["departure_stop", "arrive_stop"])].copy()
    events["timestamp"] = pd.to_datetime(events["timestamp"], errors="coerce")
    events["arrival_time_rt"] = pd.to_datetime(events["arrival_time_rt"], errors="coerce")
    events["stop_sequence"] = pd.to_numeric(events["stop_sequence"], errors="coerce")
    events["occupancy_status"] = pd.to_numeric(
        events["occupancy_status"],
        errors="coerce",
    ).fillna(0).astype(int)
    events = events.dropna(subset=["timestamp", "stop_sequence", "segment_id"])
    return events


def get_interval_start(events: pd.DataFrame, hour: int) -> pd.Timestamp:
    dataset_day = pd.Timestamp(DATASET_DATE.replace("_", "-"))
    return dataset_day + pd.Timedelta(hours=hour)


def select_trip_ids_by_high_spread_departure(
    events: pd.DataFrame,
    high_spread_segment_ids: list[str],
    interval_start: pd.Timestamp,
    interval_end: pd.Timestamp,
) -> list[str]:
    starts_in_interval = events[
        (events["segment_id"].isin(high_spread_segment_ids))
        & (events["activity_type"] == "departure_stop")
        & (events["timestamp"] >= interval_start)
        & (events["timestamp"] < interval_end)
    ].sort_values(["timestamp", "trip_id"])
    return starts_in_interval["trip_id"].dropna().astype(str).drop_duplicates().tolist()


def build_trip_segments_from_aux_events(
    events: pd.DataFrame,
    trip_ids: list[str],
    occupancy_source: str = "departure",
) -> pd.DataFrame:
    if occupancy_source not in {"departure", "arrive"}:
        raise ValueError("occupancy_source must be 'departure' or 'arrive'")

    events = events[events["trip_id"].isin(trip_ids)].copy()

    segment_rows = []
    for (trip_id, segment_id), group in events.groupby(["trip_id", "segment_id"], sort=False):
        departures = group[group["activity_type"] == "departure_stop"].sort_values("timestamp")
        arrivals = group[group["activity_type"] == "arrive_stop"].sort_values("timestamp")
        if departures.empty or arrivals.empty:
            continue

        departure = departures.iloc[0]
        arrival = arrivals.iloc[-1]
        occupancy_event = arrival if occupancy_source == "arrive" else departure
        segment_rows.append(
            {
                "segment_id": str(segment_id),
                "trip_id": str(trip_id),
                "trip_id_org": str(departure["trip_id_org"]),
                "departure_timestamp": departure["timestamp"],
                "arrive_timestamp": arrival["arrival_time_rt"]
                if pd.notna(arrival["arrival_time_rt"])
                else arrival["timestamp"],
                "departure_stop_sequence": departure["stop_sequence"],
                "departure_stop_name": str(departure["stop_name"]),
                "arrive_stop_sequence": arrival["stop_sequence"],
                "arrive_stop_name": str(arrival["stop_name"]),
                "route_short_name": str(departure["route_short_name"]),
                "route_type": str(departure["route_type"]),
                "direction_id": str(departure["direction_id"]),
                "occupancy_status": int(occupancy_event["occupancy_status"]),
            }
        )

    segments = pd.DataFrame(segment_rows)
    if segments.empty:
        return segments
    return segments.sort_values(["trip_id", "departure_stop_sequence"]).reset_index(drop=True)


def filter_segments_to_interval(
    segments: pd.DataFrame,
    interval_start: pd.Timestamp,
    interval_end: pd.Timestamp,
) -> pd.DataFrame:
    if segments.empty:
        return segments
    overlaps_interval = (
        (segments["departure_timestamp"] < interval_end)
        & (segments["arrive_timestamp"] >= interval_start)
    )
    return segments[overlaps_interval].sort_values(
        ["trip_id", "departure_stop_sequence"]
    ).reset_index(drop=True)


def sort_trips_by_stop_timestamp(
    segments: pd.DataFrame,
    stop_name: str = "US Norra entrén",
) -> pd.DataFrame:
    if segments.empty:
        return segments

    target_stop_key = activity_key(stop_name)
    fallback_first_activity = segments.groupby("trip_id", sort=False)["departure_timestamp"].min()
    sort_rows = []

    for trip_id, trip_segments in segments.groupby("trip_id", sort=False):
        stop_timestamps = []
        departures_at_stop = trip_segments[
            trip_segments["departure_stop_name"].map(activity_key) == target_stop_key
        ]
        arrivals_at_stop = trip_segments[
            trip_segments["arrive_stop_name"].map(activity_key) == target_stop_key
        ]
        if not departures_at_stop.empty:
            stop_timestamps.extend(departures_at_stop["departure_timestamp"].tolist())
        if not arrivals_at_stop.empty:
            stop_timestamps.extend(arrivals_at_stop["arrive_timestamp"].tolist())

        if stop_timestamps:
            sort_rows.append(
                {
                    "trip_id": trip_id,
                    "has_target_stop": 0,
                    "sort_timestamp": min(stop_timestamps),
                    "fallback_timestamp": fallback_first_activity.loc[trip_id],
                }
            )
        else:
            sort_rows.append(
                {
                    "trip_id": trip_id,
                    "has_target_stop": 1,
                    "sort_timestamp": fallback_first_activity.loc[trip_id],
                    "fallback_timestamp": fallback_first_activity.loc[trip_id],
                }
            )

    trip_order = (
        pd.DataFrame(sort_rows)
        .sort_values(["has_target_stop", "sort_timestamp", "fallback_timestamp", "trip_id"])
        ["trip_id"]
        .tolist()
    )
    ordered = segments.copy()
    ordered["trip_id"] = pd.Categorical(
        ordered["trip_id"],
        categories=trip_order,
        ordered=True,
    )
    ordered = ordered.sort_values(
        ["trip_id", "departure_stop_sequence", "departure_timestamp"]
    ).reset_index(drop=True)
    ordered["trip_id"] = ordered["trip_id"].astype(str)
    return ordered


def sort_trips_by_start_timestamp(segments: pd.DataFrame) -> pd.DataFrame:
    if segments.empty:
        return segments

    trip_order = (
        segments.groupby("trip_id", sort=False)["departure_timestamp"]
        .min()
        .sort_values()
        .index
    )
    ordered = segments.copy()
    ordered["trip_id"] = pd.Categorical(
        ordered["trip_id"],
        categories=trip_order,
        ordered=True,
    )
    ordered = ordered.sort_values(
        ["trip_id", "departure_stop_sequence", "departure_timestamp"]
    ).reset_index(drop=True)
    ordered["trip_id"] = ordered["trip_id"].astype(str)
    return ordered


def build_pm4py_event_log(segments: pd.DataFrame):
    if segments.empty:
        return pd.DataFrame(), []

    events = []
    for row in segments.itertuples(index=False):
        events.append(
            {
                "case:concept:name": row.trip_id,
                "concept:name": str(row.departure_stop_name),
                "time:timestamp": row.departure_timestamp,
                "segment_id": row.segment_id,
                "occupancy_status": row.occupancy_status,
            }
        )
        events.append(
            {
                "case:concept:name": row.trip_id,
                "concept:name": str(row.arrive_stop_name),
                "time:timestamp": row.arrive_timestamp,
                "segment_id": row.segment_id,
                "occupancy_status": row.occupancy_status,
            }
        )

    event_df = pd.DataFrame(events).drop_duplicates(
        ["case:concept:name", "concept:name", "time:timestamp"]
    )
    event_df = pm4py.format_dataframe(
        event_df,
        case_id="case:concept:name",
        activity_key="concept:name",
        timestamp_key="time:timestamp",
    )
    event_log = pm4py.convert_to_event_log(event_df)
    return event_df, event_log


def short_label(value: str, max_len: int = 18) -> str:
    value = str(value)
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."


def activity_key(value: str) -> str:
    return " ".join(str(value).strip().casefold().split())


def format_clock(timestamp: pd.Timestamp) -> str:
    return f"{pd.Timestamp(timestamp):%H:%M}"


def format_time_range(start: pd.Timestamp, end: pd.Timestamp) -> str:
    return f"{format_clock(start)}-{format_clock(end)}"


def add_legend(graph: Digraph) -> None:
    with graph.subgraph(name="cluster_legend") as legend:
        legend.attr(label="Occupancy edge colors", style="rounded", color="#999999")
        legend.attr(rank="same")
        previous = None
        for occupancy, color in OCCUPANCY_COLORS.items():
            node_id = f"legend_{occupancy}"
            legend.node(
                node_id,
                f"occ {occupancy}",
                shape="box",
                style="filled",
                fillcolor="#ffffff",
                color=color,
                fontname="Helvetica",
            )
            if previous is not None:
                legend.edge(previous, node_id, style="invis")
            previous = node_id


def build_trip_path_graph(
    segments: pd.DataFrame,
    output_prefix: Path,
    title: str = "High-spread trip paths by occupancy status",
    sort_mode: str = "stop_timestamp",
) -> Digraph:
    if sort_mode == "input_order":
        segments = segments.copy()
    elif sort_mode == "start_timestamp":
        segments = sort_trips_by_start_timestamp(segments)
    else:
        segments = sort_trips_by_stop_timestamp(segments)

    graph = Digraph("high_spread_trip_paths", format="png")
    graph.attr(
        rankdir="BT",
        splines="ortho",
        overlap="false",
        bgcolor="white",
        fontname="Helvetica",
        labelloc="t",
        label=title,
    )
    graph.attr("node", shape="box", style="rounded,filled", fillcolor="#ffffff", fontname="Helvetica")
    graph.attr("edge", fontname="Helvetica", arrowsize="0.7")

    previous_anchor = None
    nodes_by_activity: dict[str, list[str]] = {}
    anchor_ids: list[str] = []
    for trip_index, (trip_id, trip_segments) in enumerate(segments.groupby("trip_id", sort=False)):
        anchor_id = f"anchor_{trip_index}"
        trip_group = f"trip_{trip_index}"
        anchor_ids.append(anchor_id)

        graph.node(
            anchor_id,
            str(trip_id),
            shape="box",
            style="rounded,filled",
            fillcolor="#f5f5f5",
            color="#bbbbbb",
            fontsize="18",
            margin="0.18,0.12",
            width="1.8",
            height="0.5",
            group=trip_group,
        )

        previous_node = anchor_id
        for segment_index, row in enumerate(trip_segments.itertuples(index=False)):
            from_id = f"{trip_id}_{segment_index}_from"
            to_id = f"{trip_id}_{segment_index}_to"
            if segment_index == 0:
                graph.node(
                    from_id,
                    short_label(row.departure_stop_name),
                    xlabel=format_clock(row.departure_timestamp),
                    group=trip_group,
                )
                nodes_by_activity.setdefault(activity_key(row.departure_stop_name), []).append(from_id)
                graph.edge(previous_node, from_id, style="invis", weight="20")
            else:
                from_id = previous_node

            graph.node(
                to_id,
                short_label(row.arrive_stop_name),
                xlabel=format_clock(row.arrive_timestamp),
                group=trip_group,
            )
            nodes_by_activity.setdefault(activity_key(row.arrive_stop_name), []).append(to_id)
            occupancy = int(row.occupancy_status)
            color = OCCUPANCY_COLORS.get(occupancy, "#999999")
            time_range = format_time_range(row.departure_timestamp, row.arrive_timestamp)
            graph.edge(
                from_id,
                to_id,
                color=color,
                penwidth=str(2.0 + occupancy),
                fontcolor=color,
                tooltip=(
                    f"{trip_id} | {row.segment_id} | "
                    f"{row.departure_stop_name} -> {row.arrive_stop_name} | "
                    f"{time_range} | occupancy {occupancy}"
                ),
            )
            previous_node = to_id

        if previous_anchor is not None:
            graph.edge(previous_anchor, anchor_id, style="invis", weight="50")
        previous_anchor = anchor_id

    with graph.subgraph(name="rank_trip_headers") as header_rank:
        header_rank.attr(rank="same")
        for anchor_id in anchor_ids:
            header_rank.node(anchor_id)

    for rank_index, node_ids in enumerate(nodes_by_activity.values()):
        if len(node_ids) < 2:
            continue
        with graph.subgraph(name=f"rank_activity_{rank_index}") as same_rank:
            same_rank.attr(rank="same")
            for node_id in node_ids:
                same_rank.node(node_id)

    add_legend(graph)
    graph.render(output_prefix, cleanup=True)
    return graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build side-by-side process paths for trips from high-spread occupancy "
            "segments and color edges by occupancy status."
        )
    )
    parser.add_argument("--high-spread-file", type=Path, default=DEFAULT_HIGH_SPREAD_FILE)
    parser.add_argument("--segments-file", type=Path, default=DEFAULT_SEGMENTS_FILE)
    parser.add_argument("--aux-events-file", type=Path, default=DEFAULT_AUX_EVENTS_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hour", type=int, default=6)
    parser.add_argument("--interval-minutes", type=int, default=10)
    parser.add_argument("--ten-minute-pngs", action="store_true")
    parser.add_argument("--direction-pngs", action="store_true")
    parser.add_argument("--max-high-spread-segments", type=int, default=None)
    parser.add_argument("--all-high-spread-segments", action="store_true")
    parser.add_argument("--full-trip", action="store_true")
    return parser.parse_args()


def render_ten_minute_pngs(args: argparse.Namespace, max_segments: int | None) -> None:
    high_spread_segment_ids = load_high_spread_segment_ids(
        args.high_spread_file,
        max_segments=max_segments,
    )
    events = load_aux_events(args.aux_events_file)
    hour_start = get_interval_start(events, args.hour)
    if events.empty:
        print("No segment observations found.")
        return

    interval_count = 60 // args.interval_minutes

    written_pngs = []
    for interval_index in range(interval_count):
        interval_start = hour_start + pd.Timedelta(
            minutes=interval_index * args.interval_minutes
        )
        interval_end = interval_start + pd.Timedelta(minutes=args.interval_minutes)
        trip_ids = select_trip_ids_by_high_spread_departure(
            events,
            high_spread_segment_ids=high_spread_segment_ids,
            interval_start=interval_start,
            interval_end=interval_end,
        )
        interval_segments = build_trip_segments_from_aux_events(events, trip_ids=trip_ids)
        if interval_segments.empty:
            print(
                f"{interval_start:%H:%M}-{interval_end:%H:%M}: "
                "no trips starting a high-spread segment"
            )
            continue

        event_df, event_log = build_pm4py_event_log(interval_segments)
        suffix = f"{interval_start:%H_%M}_{interval_end:%H_%M}"
        output_prefix = args.output_dir / f"high_spread_trip_process_paths_{suffix}"
        build_trip_path_graph(
            interval_segments,
            output_prefix,
            title=(
                "High-spread segment trip paths by occupancy status "
                f"({interval_start:%H:%M}-{interval_end:%H:%M})"
            ),
        )
        event_path = args.output_dir / f"high_spread_trip_process_events_{suffix}.csv"
        event_df.to_csv(event_path, index=False)
        written_pngs.append(f"{output_prefix}.png")
        print(
            f"{interval_start:%H:%M}-{interval_end:%H:%M}: "
            f"{interval_segments['trip_id'].nunique()} trips, "
            f"{len(interval_segments)} segments, "
            f"{len(event_log)} PM4Py traces, PNG written: {output_prefix}.png"
        )

    print(f"PNG files written: {len(written_pngs)}")


def render_direction_pngs(args: argparse.Namespace, max_segments: int | None) -> None:
    high_spread_segment_ids = load_high_spread_segment_ids(
        args.high_spread_file,
        max_segments=max_segments,
    )
    events = load_aux_events(args.aux_events_file)
    hour_start = get_interval_start(events, args.hour)
    hour_end = hour_start + pd.Timedelta(hours=1)
    trip_ids = select_trip_ids_by_high_spread_departure(
        events,
        high_spread_segment_ids=high_spread_segment_ids,
        interval_start=hour_start,
        interval_end=hour_end,
    )
    segments = build_trip_segments_from_aux_events(events, trip_ids=trip_ids)
    if segments.empty:
        print("No trips starting a high-spread segment found.")
        return

    written_pngs = []
    for direction_id, direction_segments in segments.groupby("direction_id", sort=True):
        safe_direction = str(direction_id).replace(".", "_").replace(" ", "_")
        output_prefix = (
            args.output_dir
            / f"high_spread_trip_process_paths_hour_{args.hour:02d}_direction_{safe_direction}"
        )
        event_df, event_log = build_pm4py_event_log(direction_segments)
        build_trip_path_graph(
            direction_segments,
            output_prefix,
            title=(
                "High-spread segment trip paths by occupancy status "
                f"(hour {args.hour:02d}, direction {direction_id})"
            ),
            sort_mode="start_timestamp",
        )
        event_path = (
            args.output_dir
            / f"high_spread_trip_process_events_hour_{args.hour:02d}_direction_{safe_direction}.csv"
        )
        event_df.to_csv(event_path, index=False)
        written_pngs.append(f"{output_prefix}.png")
        print(
            f"Direction {direction_id}: "
            f"{direction_segments['trip_id'].nunique()} trips, "
            f"{len(direction_segments)} segments, "
            f"{len(event_log)} PM4Py traces, PNG written: {output_prefix}.png"
        )

    print(f"Direction PNG files written: {len(written_pngs)}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    max_segments = None if args.all_high_spread_segments else args.max_high_spread_segments

    if args.ten_minute_pngs:
        render_ten_minute_pngs(args, max_segments=max_segments)
        return

    if args.direction_pngs:
        render_direction_pngs(args, max_segments=max_segments)
        return

    if args.full_trip:
        trip_ids = load_high_spread_trip_ids(args.high_spread_file, max_segments=max_segments)
        segments = load_trip_segments(
            args.segments_file,
            trip_ids=trip_ids,
            hour=args.hour,
            only_hour_window=False,
        )
    else:
        high_spread_segment_ids = load_high_spread_segment_ids(
            args.high_spread_file,
            max_segments=max_segments,
        )
        events = load_aux_events(args.aux_events_file)
        hour_start = get_interval_start(events, args.hour)
        hour_end = hour_start + pd.Timedelta(hours=1)
        trip_ids = select_trip_ids_by_high_spread_departure(
            events,
            high_spread_segment_ids=high_spread_segment_ids,
            interval_start=hour_start,
            interval_end=hour_end,
        )
        segments = build_trip_segments_from_aux_events(events, trip_ids=trip_ids)
    event_df, event_log = build_pm4py_event_log(segments)

    suffix = "full_trip" if args.full_trip else f"hour_{args.hour:02d}"
    output_prefix = args.output_dir / f"high_spread_trip_process_paths_{suffix}"
    build_trip_path_graph(segments, output_prefix)

    event_df.to_csv(args.output_dir / f"high_spread_trip_process_events_{suffix}.csv", index=False)

    print(f"Trips included: {len(trip_ids)}")
    print(f"Segments drawn: {len(segments)}")
    print(f"PM4Py traces: {len(event_log)}")
    print(f"PNG written: {output_prefix}.png")
    print(f"Event table written: {args.output_dir / f'high_spread_trip_process_events_{suffix}.csv'}")


if __name__ == "__main__":
    main()
