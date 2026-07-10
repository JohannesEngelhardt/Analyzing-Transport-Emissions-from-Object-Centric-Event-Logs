import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = "aux_event_log_overview_kodak_2026_05_04_new_e2o_no_layover.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute waiting times per vehicle from one trip to the next trip."
        )
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Input aux_event_log_overview CSV.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV. Defaults to <input stem>_trip_change_waiting_times.csv.",
    )
    parser.add_argument(
        "--plot",
        default=None,
        help="Optional output path for a histogram PNG.",
    )
    parser.add_argument(
        "--plot-by-route-change",
        default=None,
        help="Optional output path for a two-panel histogram split by route_short_name change.",
    )
    parser.add_argument(
        "--plot-near-route-changes",
        default=None,
        help=(
            "Optional output path for changed route_short_name transitions where "
            "old and new arrive_stop locations are close."
        ),
    )
    parser.add_argument(
        "--measure",
        choices=["last_arrive_to_first_arrive", "last_departure_to_first_arrive"],
        default="last_arrive_to_first_arrive",
        help=(
            "Waiting-time definition. Default measures from the last arrive_stop "
            "of the old trip to the first arrive_stop of the new trip."
        ),
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=2,
        help="Number of 1D clusters to mark in the plot and summary.",
    )
    parser.add_argument(
        "--cluster-max-minutes",
        type=float,
        default=None,
        help="Optional upper bound used only for cluster-center detection.",
    )
    parser.add_argument(
        "--plot-max-minutes",
        type=float,
        default=None,
        help="Optional upper bound used only for the histogram x-axis.",
    )
    parser.add_argument(
        "--log-y",
        action="store_true",
        help="Use a logarithmic y-axis for the histogram.",
    )
    parser.add_argument(
        "--max-hours",
        type=float,
        default=None,
        help="Optional upper bound for waiting times kept in the output.",
    )
    parser.add_argument(
        "--max-arrival-distance-meters",
        type=float,
        default=100.0,
        help=(
            "Maximum distance between old last arrive_stop and new first arrive_stop "
            "for the near-route-change analysis."
        ),
    )
    return parser.parse_args()


def haversine_meters(lat1, lon1, lat2, lon2):
    earth_radius_m = 6371000
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return earth_radius_m * c


def find_trip_change_waiting_times(df, measure="last_arrive_to_first_arrive"):
    required_columns = {
        "event_id",
        "timestamp",
        "activity",
        "trip_id",
        "vehicle_id",
        "route_short_name",
        "stop_id",
        "stop_name",
        "stop_lat",
        "stop_lon",
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "trip_id", "vehicle_id", "activity"])
    df = df.sort_values(["vehicle_id", "timestamp", "event_id"]).reset_index(drop=True)

    arrivals = df[df["activity"].astype(str).str.startswith("arrive_stop", na=False)].copy()
    departures = df[df["activity"].astype(str).str.startswith("departure_stop", na=False)].copy()

    if arrivals.empty:
        return pd.DataFrame()
    if measure == "last_departure_to_first_arrive" and departures.empty:
        return pd.DataFrame()

    first_arrival_idx = arrivals.groupby(["vehicle_id", "trip_id"])["timestamp"].idxmin()
    last_arrival_idx = arrivals.groupby(["vehicle_id", "trip_id"])["timestamp"].idxmax()
    last_departure_idx = departures.groupby(["vehicle_id", "trip_id"])["timestamp"].idxmax()

    first_arrivals = arrivals.loc[first_arrival_idx, [
        "vehicle_id",
        "trip_id",
        "event_id",
        "timestamp",
        "route_short_name",
        "stop_id",
        "stop_name",
        "stop_lat",
        "stop_lon",
    ]].rename(columns={
        "event_id": "arrival_event_id",
        "timestamp": "first_arrival_time",
        "route_short_name": "new_route_short_name",
        "stop_id": "arrival_stop_id",
        "stop_name": "arrival_stop_name",
        "stop_lat": "arrival_stop_lat",
        "stop_lon": "arrival_stop_lon",
    })

    last_arrivals = arrivals.loc[last_arrival_idx, [
        "vehicle_id",
        "trip_id",
        "event_id",
        "timestamp",
        "route_short_name",
        "stop_id",
        "stop_name",
        "stop_lat",
        "stop_lon",
    ]].rename(columns={
        "event_id": "old_arrival_event_id",
        "timestamp": "last_arrival_time",
        "route_short_name": "old_route_short_name",
        "stop_id": "old_arrival_stop_id",
        "stop_name": "old_arrival_stop_name",
        "stop_lat": "old_arrival_stop_lat",
        "stop_lon": "old_arrival_stop_lon",
    })

    if departures.empty:
        last_departures = pd.DataFrame(columns=[
            "vehicle_id",
            "trip_id",
            "departure_event_id",
            "last_departure_time",
            "old_departure_route_short_name",
            "departure_stop_id",
            "departure_stop_name",
        ])
    else:
        last_departures = departures.loc[last_departure_idx, [
        "vehicle_id",
        "trip_id",
        "event_id",
        "timestamp",
        "route_short_name",
        "stop_id",
        "stop_name",
        ]].rename(columns={
            "event_id": "departure_event_id",
            "timestamp": "last_departure_time",
            "route_short_name": "old_departure_route_short_name",
            "stop_id": "departure_stop_id",
            "stop_name": "departure_stop_name",
        })

    trip_bounds = first_arrivals.merge(
        last_arrivals,
        on=["vehicle_id", "trip_id"],
        how="inner",
    ).merge(
        last_departures,
        on=["vehicle_id", "trip_id"],
        how="left",
    )
    trip_bounds = trip_bounds.sort_values(
        ["vehicle_id", "first_arrival_time", "last_arrival_time", "trip_id"]
    ).reset_index(drop=True)

    rows = []

    for vehicle_id, vehicle_trips in trip_bounds.groupby("vehicle_id", sort=False):
        vehicle_trips = vehicle_trips.sort_values(
            ["first_arrival_time", "last_arrival_time", "trip_id"]
        ).reset_index(drop=True)

        for i in range(len(vehicle_trips) - 1):
            current_trip = vehicle_trips.iloc[i]
            current_end_time = (
                current_trip["last_departure_time"]
                if measure == "last_departure_to_first_arrive"
                else current_trip["last_arrival_time"]
            )
            if pd.isna(current_end_time):
                continue
            future_trips = vehicle_trips[
                (vehicle_trips.index > i)
                & (vehicle_trips["trip_id"].astype(str) != str(current_trip["trip_id"]))
                & (vehicle_trips["first_arrival_time"] >= current_end_time)
            ]

            if future_trips.empty:
                continue

            next_trip = future_trips.iloc[0]

            waiting_time = (
                next_trip["first_arrival_time"]
                - current_end_time
            )
            arrival_distance_m = haversine_meters(
                current_trip["old_arrival_stop_lat"],
                current_trip["old_arrival_stop_lon"],
                next_trip["arrival_stop_lat"],
                next_trip["arrival_stop_lon"],
            )

            rows.append(
                {
                    "vehicle_id": vehicle_id,
                    "old_trip_id": current_trip["trip_id"],
                    "new_trip_id": next_trip["trip_id"],
                    "old_arrival_event_id": current_trip["old_arrival_event_id"],
                    "departure_event_id": current_trip["departure_event_id"],
                    "arrival_event_id": next_trip["arrival_event_id"],
                    "measure": measure,
                    "old_last_arrival_time": current_trip["last_arrival_time"],
                    "departure_time": current_trip["last_departure_time"],
                    "new_first_arrival_time": next_trip["first_arrival_time"],
                    "waiting_seconds": waiting_time.total_seconds(),
                    "waiting_minutes": waiting_time.total_seconds() / 60,
                    "waiting_hours": waiting_time.total_seconds() / 3600,
                    "old_route_short_name": current_trip["old_route_short_name"],
                    "new_route_short_name": next_trip["new_route_short_name"],
                    "route_short_name_changed": (
                        str(current_trip["old_route_short_name"])
                        != str(next_trip["new_route_short_name"])
                    ),
                    "route_change_group": (
                        "changed_route_short_name"
                        if str(current_trip["old_route_short_name"])
                        != str(next_trip["new_route_short_name"])
                        else "same_route_short_name"
                    ),
                    "old_arrival_stop_id": current_trip["old_arrival_stop_id"],
                    "old_arrival_stop_name": current_trip["old_arrival_stop_name"],
                    "old_arrival_stop_lat": current_trip["old_arrival_stop_lat"],
                    "old_arrival_stop_lon": current_trip["old_arrival_stop_lon"],
                    "departure_stop_id": current_trip["departure_stop_id"],
                    "departure_stop_name": current_trip["departure_stop_name"],
                    "arrival_stop_id": next_trip["arrival_stop_id"],
                    "arrival_stop_name": next_trip["arrival_stop_name"],
                    "arrival_stop_lat": next_trip["arrival_stop_lat"],
                    "arrival_stop_lon": next_trip["arrival_stop_lon"],
                    "arrival_distance_m": arrival_distance_m,
                }
            )

    return pd.DataFrame(rows)


def find_1d_cluster_centers(values, n_clusters=2, max_minutes=None, max_iter=100):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if max_minutes is not None:
        values = values[values <= max_minutes]
    if values.size == 0:
        return pd.DataFrame(columns=["cluster", "center_minutes", "count", "share"])

    n_clusters = max(1, min(int(n_clusters), values.size))
    centers = np.percentile(values, np.linspace(10, 90, n_clusters))

    labels = np.zeros(values.size, dtype=int)
    for _ in range(max_iter):
        distances = np.abs(values[:, None] - centers[None, :])
        new_labels = distances.argmin(axis=1)

        new_centers = centers.copy()
        for cluster_idx in range(n_clusters):
            cluster_values = values[new_labels == cluster_idx]
            if cluster_values.size:
                new_centers[cluster_idx] = cluster_values.mean()

        if np.array_equal(labels, new_labels) and np.allclose(centers, new_centers):
            break

        labels = new_labels
        centers = new_centers

    order = np.argsort(centers)
    rows = []
    for new_idx, old_idx in enumerate(order, start=1):
        count = int((labels == old_idx).sum())
        rows.append({
            "cluster": new_idx,
            "center_minutes": float(centers[old_idx]),
            "count": count,
            "share": count / values.size,
        })
    return pd.DataFrame(rows)


def print_summary(waiting_times, cluster_summary=None):
    print(f"Transitions found: {len(waiting_times)}")

    if waiting_times.empty:
        return

    summary = waiting_times["waiting_minutes"].describe(
        percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    )
    print("\nWaiting time in minutes:")
    print(summary.to_string())

    if cluster_summary is not None and not cluster_summary.empty:
        print("\nCluster centers in minutes:")
        print(cluster_summary.to_string(index=False, formatters={
            "center_minutes": "{:.2f}".format,
            "share": "{:.1%}".format,
        }))

    if "route_change_group" in waiting_times.columns:
        print("\nBy route_short_name change:")
        for group_name, group_df in waiting_times.groupby("route_change_group"):
            group_summary = group_df["waiting_minutes"].describe(
                percentiles=[0.5, 0.75, 0.9, 0.95]
            )
            print(f"\n{group_name} ({len(group_df)} transitions)")
            print(group_summary.to_string())

    print("\nShortest transitions:")
    print(
        waiting_times.sort_values("waiting_minutes")
        .head(10)[
            [
                "vehicle_id",
                "old_trip_id",
                "new_trip_id",
                "waiting_minutes",
                "old_last_arrival_time",
                "new_first_arrival_time",
                "old_arrival_stop_name",
                "arrival_stop_name",
            ]
        ]
        .to_string(index=False)
    )

    print("\nLongest transitions:")
    print(
        waiting_times.sort_values("waiting_minutes", ascending=False)
        .head(10)[
            [
                "vehicle_id",
                "old_trip_id",
                "new_trip_id",
                "waiting_minutes",
                "old_last_arrival_time",
                "new_first_arrival_time",
                "old_arrival_stop_name",
                "arrival_stop_name",
            ]
        ]
        .to_string(index=False)
    )


def save_histogram(
    waiting_times,
    output_path,
    cluster_summary=None,
    plot_max_minutes=None,
    log_y=False,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if waiting_times.empty:
        print("No histogram written because no transitions were found.")
        return

    plot_data = waiting_times["waiting_minutes"].dropna()
    if plot_max_minutes is not None:
        plot_data = plot_data[plot_data <= plot_max_minutes]

    plt.figure(figsize=(10, 5))
    plt.hist(plot_data, bins=60, color="#d9d9d9", edgecolor="#333333")

    if cluster_summary is not None and not cluster_summary.empty:
        colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
        for idx, row in cluster_summary.iterrows():
            center = row["center_minutes"]
            if plot_max_minutes is not None and center > plot_max_minutes:
                continue
            color = colors[idx % len(colors)]
            plt.axvline(center, color=color, linewidth=2.5)
            plt.text(
                center,
                plt.ylim()[1] * 0.92,
                f"{center:.1f} min",
                color=color,
                rotation=90,
                va="top",
                ha="right",
            )

    plt.xlabel("Waiting time between trips (minutes)")
    plt.ylabel("Number of trip changes")
    if log_y:
        plt.yscale("log")
    plt.title("Waiting time from last arrive_stop to next trip's first arrive_stop")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Histogram written to: {Path(output_path).resolve()}")


def save_route_change_histograms(
    waiting_times,
    output_path,
    n_clusters=2,
    cluster_max_minutes=None,
    plot_max_minutes=None,
    log_y=False,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if waiting_times.empty:
        print("No route-change histogram written because no transitions were found.")
        return

    if "route_change_group" not in waiting_times.columns:
        print("No route-change histogram written because route_change_group is missing.")
        return

    groups = [
        ("same_route_short_name", "Same route_short_name"),
        ("changed_route_short_name", "Changed route_short_name"),
    ]
    colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, (group_key, title) in zip(axes, groups):
        group_df = waiting_times[waiting_times["route_change_group"] == group_key]
        plot_data = group_df["waiting_minutes"].dropna()
        if plot_max_minutes is not None:
            plot_data = plot_data[plot_data <= plot_max_minutes]

        ax.hist(plot_data, bins=45, color="#d9d9d9", edgecolor="#333333")
        centers = find_1d_cluster_centers(
            group_df["waiting_minutes"],
            n_clusters=n_clusters,
            max_minutes=cluster_max_minutes,
        )

        for idx, row in centers.iterrows():
            center = row["center_minutes"]
            if plot_max_minutes is not None and center > plot_max_minutes:
                continue
            color = colors[idx % len(colors)]
            ax.axvline(center, color=color, linewidth=2.5)
            ax.text(
                center,
                ax.get_ylim()[1] * 0.92,
                f"{center:.1f} min",
                color=color,
                rotation=90,
                va="top",
                ha="right",
            )

        ax.set_title(f"{title}\n(n={len(group_df)})")
        ax.set_xlabel("Waiting time (minutes)")
        if log_y:
            ax.set_yscale("log")

        print(f"\nCluster centers for {group_key}:")
        if centers.empty:
            print("No values.")
        else:
            print(centers.to_string(index=False, formatters={
                "center_minutes": "{:.2f}".format,
                "share": "{:.1%}".format,
            }))

    axes[0].set_ylabel("Number of trip changes")
    fig.suptitle("Waiting time split by route_short_name change")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Route-change histogram written to: {Path(output_path).resolve()}")


def save_near_route_change_histogram(
    waiting_times,
    output_path,
    max_arrival_distance_meters=100.0,
    n_clusters=2,
    cluster_max_minutes=None,
    plot_max_minutes=None,
    log_y=False,
):
    if waiting_times.empty:
        print("No near-route-change histogram written because no transitions were found.")
        return pd.DataFrame()

    near_changes = waiting_times[
        (waiting_times["route_short_name_changed"])
        & (waiting_times["arrival_distance_m"] <= max_arrival_distance_meters)
    ].copy()

    print(
        "\nChanged route_short_name and close arrival locations "
        f"(<= {max_arrival_distance_meters:.0f} m):"
    )
    print_summary(
        near_changes,
        find_1d_cluster_centers(
            near_changes["waiting_minutes"] if not near_changes.empty else [],
            n_clusters=n_clusters,
            max_minutes=cluster_max_minutes,
        ),
    )

    save_histogram(
        near_changes,
        output_path,
        cluster_summary=find_1d_cluster_centers(
            near_changes["waiting_minutes"] if not near_changes.empty else [],
            n_clusters=n_clusters,
            max_minutes=cluster_max_minutes,
        ),
        plot_max_minutes=plot_max_minutes,
        log_y=log_y,
    )
    return near_changes


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = (
        Path(args.output)
        if args.output
        else input_path.with_name(
            f"{input_path.stem}_trip_change_waiting_times_arrive_to_arrive.csv"
        )
    )

    df = pd.read_csv(input_path)
    waiting_times = find_trip_change_waiting_times(df, measure=args.measure)

    if args.max_hours is not None and not waiting_times.empty:
        waiting_times = waiting_times[waiting_times["waiting_hours"] <= args.max_hours].copy()

    waiting_times.to_csv(output_path, index=False)
    print(f"Waiting times written to: {output_path.resolve()}")
    cluster_summary = find_1d_cluster_centers(
        waiting_times["waiting_minutes"] if not waiting_times.empty else [],
        n_clusters=args.clusters,
        max_minutes=args.cluster_max_minutes,
    )
    print_summary(waiting_times, cluster_summary)

    if args.plot:
        save_histogram(
            waiting_times,
            args.plot,
            cluster_summary=cluster_summary,
            plot_max_minutes=args.plot_max_minutes,
            log_y=args.log_y,
        )

    if args.plot_by_route_change:
        save_route_change_histograms(
            waiting_times,
            args.plot_by_route_change,
            n_clusters=args.clusters,
            cluster_max_minutes=args.cluster_max_minutes,
            plot_max_minutes=args.plot_max_minutes,
            log_y=args.log_y,
        )

    if args.plot_near_route_changes:
        near_changes = save_near_route_change_histogram(
            waiting_times,
            args.plot_near_route_changes,
            max_arrival_distance_meters=args.max_arrival_distance_meters,
            n_clusters=args.clusters,
            cluster_max_minutes=args.cluster_max_minutes,
            plot_max_minutes=args.plot_max_minutes,
            log_y=args.log_y,
        )
        if not near_changes.empty:
            near_output_path = output_path.with_name(
                f"{output_path.stem}_changed_route_near_{int(args.max_arrival_distance_meters)}m.csv"
            )
            near_changes.to_csv(near_output_path, index=False)
            print(f"Near route changes written to: {near_output_path.resolve()}")


if __name__ == "__main__":
    main()
