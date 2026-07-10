###Methode Bilder

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from tabulate import tabulate
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches


def short_number(value):
    if pd.isna(value):
        return ""
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f} M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f} k"
    if value.is_integer():
        return f"{int(value)}"
    return f"{value:.1f}"


def add_short_labels(ax):
    for container in ax.containers:
        labels = [short_number(v) for v in container.datavalues]
        ax.bar_label(container, labels=labels, padding=3)


def latex_escape(value):
    text = "" if pd.isna(value) else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def write_operator_summary_latex(summary, output_path):
    rows = []
    for row in summary.itertuples(index=False):
        rows.append(
            "    "
            + " & ".join(
                [
                    latex_escape(row.organization_id),
                    latex_escape(row.organization_name),
                    f"{row.total_distance_km:.2f}",
                    f"{int(row.trip_count)}",
                    f"{int(row.route_count)}",
                    f"{row.avg_occupancy_status:.2f}",
                    f"{row.avg_delay_seconds:.2f}",
                ]
            )
            + r" \\"
        )

    table = "\n".join(
        [
            r"\begin{table}[t]",
            r"  \caption{Operator-level summary of distance, trips, routes, occupancy, and delay.}",
            r"  \label{tab:operator-summary-kodak-2026-05-04}",
            r"  \centering",
            r"  \scriptsize",
            r"  \begin{tabular}{llrrrrr}",
            r"    \toprule",
            r"    \textbf{Operator ID} & \textbf{Operator} & \textbf{Distance (km)} & \textbf{Trips} & \textbf{Routes} & \textbf{Occupancy} & \textbf{Delay (s)} \\",
            r"    \midrule",
            *rows,
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(table)


def add_reference_line(ax, value, label, color, linestyle="--"):
    ax.axhline(
        y=value,
        color=color,
        linestyle=linestyle,
        linewidth=1.5
    )
    ax.text(
        1.01,
        value,
        f" {label}: {value:.2f}",
        color=color,
        va="center",
        ha="left",
        fontsize=9,
        transform=ax.get_yaxis_transform(),
        clip_on=False
    )


def horizontal_legend_columns(values, max_columns=6):
    return max(1, min(max_columns, len(pd.Series(values).dropna().unique())))


def set_horizontal_route_legend(ax, title, values, max_columns=6):
    used_labels = set(pd.Series(values).dropna().astype(str).unique())
    handles, labels = ax.get_legend_handles_labels()
    legend_items = [
        (handle, label)
        for handle, label in zip(handles, labels)
        if label in used_labels
    ]

    if not legend_items:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        return

    handles, labels = zip(*legend_items)
    ax.legend(
        handles=handles,
        labels=labels,
        title=title,
        loc="upper right",
        ncol=horizontal_legend_columns(labels, max_columns=max_columns),
        frameon=True,
        fancybox=False,
        edgecolor="black",
    )


def plot_bar(
        df,
        x,
        y,
        title,
        xlabel,
        ylabel,
        top_n=None,
        rotate=60,
        color="steelblue",
        short_labels=False,
        show_reference_lines=True,
        save_path=None
):
    plot_df = df.copy().sort_values(y, ascending=False)
    average_value = plot_df[y].mean()
    median_value = plot_df[y].median()
    if top_n is not None:
        plot_df = plot_df.head(top_n)

    plt.figure(figsize=(14, 6))
    ax = sns.barplot(data=plot_df, x=x, y=y, color=color)
    plt.xticks(rotation=rotate, ha="right", fontsize=8)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if show_reference_lines:
        add_reference_line(ax, average_value, "AVG", "darkred")
        add_reference_line(ax, median_value, "MED", "navy", linestyle=":")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_trip_bar_with_route(df, x, y, title, ylabel, route_map, route_color_map, top_n=30, short_labels=False):
    plot_df = df.merge(route_map, on="trip_id", how="left")
    plot_df = plot_df.dropna(subset=["route_short_name"])
    plot_df["route_short_name"] = plot_df["route_short_name"].astype(str)
    plot_df = plot_df.sort_values(y, ascending=False).head(top_n)
    reference_df = df.copy()
    average_value = reference_df[y].mean()
    median_value = reference_df[y].median()

    plt.figure(figsize=(15, 6))
    ax = sns.barplot(
        data=plot_df,
        x=x,
        y=y,
        hue="route_short_name",
        dodge=False,
        palette=route_color_map
    )

    set_horizontal_route_legend(ax, "Route Short Name", plot_df["route_short_name"])

    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.xlabel("Trip ID")
    plt.ylabel(ylabel)

    add_reference_line(ax, average_value, "AVG", "darkred")
    add_reference_line(ax, median_value, "MED", "navy", linestyle=":")

    plt.tight_layout()
    plt.show()


def plot_bar_with_route(df, x, y, route_col, title, xlabel, ylabel, route_color_map, top_n=None, rotate=45, short_labels=False):
    plot_df = df.copy().sort_values(y, ascending=False)
    average_value = plot_df[y].mean()
    median_value = plot_df[y].median()
    if top_n is not None:
        plot_df = plot_df.head(top_n)

    plot_df = plot_df.dropna(subset=[route_col])
    plot_df[route_col] = plot_df[route_col].astype(str)

    plt.figure(figsize=(15, 6))
    ax = sns.barplot(
        data=plot_df,
        x=x,
        y=y,
        hue=route_col,
        dodge=False,
        palette=route_color_map
    )

    set_horizontal_route_legend(ax, route_col, plot_df[route_col])

    plt.xticks(rotation=rotate, ha="right")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    add_reference_line(ax, average_value, "AVG", "darkred")
    add_reference_line(ax, median_value, "MED", "navy", linestyle=":")

    plt.tight_layout()
    plt.show()


def plot_bar_by_route(df, x, y, title, xlabel, ylabel, route_color_map, top_n=None, rotate=60, short_labels=False):
    plot_df = df.copy().sort_values(y, ascending=False)
    average_value = plot_df[y].mean()
    median_value = plot_df[y].median()
    if top_n is not None:
        plot_df = plot_df.head(top_n)

    plot_df = plot_df.dropna(subset=[x])
    plot_df[x] = plot_df[x].astype(str)

    plt.figure(figsize=(14, 6))
    ax = sns.barplot(
        data=plot_df,
        x=x,
        y=y,
        hue=x,
        dodge=False,
        palette=route_color_map
    )

    set_horizontal_route_legend(ax, "Route Short Name", plot_df[x])

    plt.xticks(rotation=rotate, ha="right", fontsize=7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    add_reference_line(ax, average_value, "AVG", "darkred")
    add_reference_line(ax, median_value, "MED", "navy", linestyle=":")

    plt.subplots_adjust(bottom=0.18)
    plt.tight_layout()
    plt.show()


def plot_stacked_bar(df, index_col, stack_col, value_col, title, xlabel, ylabel, route_values, route_color_map, top_n=None):
    plot_df = df.copy()
    plot_df[stack_col] = plot_df[stack_col].astype(str)
    plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[value_col, stack_col])

    pivot_df = plot_df.pivot_table(
        index=index_col,
        columns=stack_col,
        values=value_col,
        aggfunc="sum",
        fill_value=0
    )

    if pivot_df.empty:
        print(f"Kein numerischer Plot moeglich fuer: {title}")
        return

    pivot_df = pivot_df.loc[pivot_df.sum(axis=1).sort_values(ascending=False).index]
    reference_totals = pivot_df.sum(axis=1)
    average_value = reference_totals.mean()
    median_value = reference_totals.median()

    if top_n is not None:
        pivot_df = pivot_df.head(top_n)

    ordered_cols = [c for c in route_values if c in pivot_df.columns]
    pivot_df = pivot_df[ordered_cols]
    used_cols = pivot_df.columns[pivot_df.sum(axis=0) > 0].tolist()
    pivot_df = pivot_df[used_cols]

    if pivot_df.empty or len(pivot_df.columns) == 0:
        print(f"Keine passenden Spalten fuer: {title}")
        return

    colors = [route_color_map[c] for c in used_cols]

    ax = pivot_df.plot(
        kind="bar",
        stacked=True,
        figsize=(15, 6),
        color=colors
    )

    add_reference_line(ax, average_value, "AVG", "darkred")
    add_reference_line(ax, median_value, "MED", "navy", linestyle=":")

    ax.legend(
        title="Route Short Name",
        loc="upper right",
        ncol=horizontal_legend_columns(used_cols),
        frameon=True,
        fancybox=False,
        edgecolor="black",
    )

    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


def trip_route_map(df):
    return (
        df.dropna(subset=["trip_id", "route_short_name"])
        .groupby("trip_id", as_index=False)["route_short_name"]
        .first()
    )


def route_type_label(route_type):
    if pd.isna(route_type):
        return "Unknown"

    try:
        route_type_number = int(float(route_type))
    except (TypeError, ValueError):
        return str(route_type)

    if 100 <= route_type_number <= 117:
        return f"{route_type_number} - Railway"
    if 200 <= route_type_number <= 209:
        return f"{route_type_number} - Coach"
    if 400 <= route_type_number <= 405:
        return f"{route_type_number} - Urban Railway"
    if 700 <= route_type_number <= 716:
        return f"{route_type_number} - Bus"
    if route_type_number == 800:
        return f"{route_type_number} - Trolleybus"
    if 900 <= route_type_number <= 906:
        return f"{route_type_number} - Tram"
    if route_type_number == 1000:
        return f"{route_type_number} - Water Transport"
    if route_type_number == 1100:
        return f"{route_type_number} - Air Service"
    if route_type_number == 1200:
        return f"{route_type_number} - Ferry"
    if 1300 <= route_type_number <= 1307:
        return f"{route_type_number} - Aerial Lift"
    if route_type_number == 1400:
        return f"{route_type_number} - Funicular"
    if 1500 <= route_type_number <= 1507:
        return f"{route_type_number} - Taxi"
    if 1700 <= route_type_number <= 1702:
        return f"{route_type_number} - Miscellaneous"
    return str(route_type_number)


aux_event_log_overview = pd.read_csv(
    "aux_event_log_overview_kodak_2026_05_04.csv",
    dtype={"vehicle_id": "string", "trip_id": "string", "trip_id_org": "string"}
)
aux_event_log_overview["timestamp"] = pd.to_datetime(aux_event_log_overview["timestamp"], errors="coerce")
aux_event_log_overview["shape_dist_traveled"] = pd.to_numeric(aux_event_log_overview["shape_dist_traveled"], errors="coerce")
aux_event_log_overview["occupancy_status"] = pd.to_numeric(aux_event_log_overview["occupancy_status"], errors="coerce")
aux_event_log_overview["delay"] = pd.to_numeric(aux_event_log_overview["delay"], errors="coerce")
if "cumulative_road_distance_km" in aux_event_log_overview.columns:
    aux_event_log_overview["cumulative_road_distance_km"] = pd.to_numeric(
        aux_event_log_overview["cumulative_road_distance_km"], errors="coerce"
    )
aux_event_log_overview = aux_event_log_overview.dropna(subset=["trip_id"]).copy()
aux_event_log_overview["trip_id"] = aux_event_log_overview["trip_id"].astype(str)

operator_lookup = pd.read_csv(
    "aux_event_log_kodak_2026_05_04.csv",
    dtype={
        "trip_id": "string",
        "trip_id_org": "string",
        "organization_id": "string",
        "organization_name": "string",
    }
)
operator_lookup = (
    operator_lookup[
        [
            "trip_id",
            "organization_id",
            "organization_name",
        ]
    ]
    .dropna(subset=["trip_id", "organization_id", "organization_name"])
    .drop_duplicates(subset=["trip_id"], keep="first")
)

if "organization_id" not in aux_event_log_overview.columns:
    aux_event_log_overview = aux_event_log_overview.merge(
        operator_lookup,
        on="trip_id",
        how="left",
    )

print(aux_event_log_overview.columns)
print(aux_event_log_overview["trip_id_org"].nunique())
print(tabulate(
    aux_event_log_overview[aux_event_log_overview["trip_id"] == "tr03_5828"].sort_values("timestamp").head(100),
    headers="keys",
    tablefmt="psql"
))

route_values = (
    aux_event_log_overview["route_short_name"]
    .dropna()
    .astype(str)
    .sort_values()
    .unique()
)
def build_route_palette(route_values):
    base_colors = [
        "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a",
        "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
        "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d",
        "#17becf", "#9edae5", "#393b79", "#637939", "#8c6d31", "#843c39"
    ]

    if len(route_values) <= len(base_colors):
        return base_colors[:len(route_values)]

    extra_colors = sns.color_palette("husl", n_colors=len(route_values) - len(base_colors)).as_hex()
    return base_colors + extra_colors


palette = build_route_palette(route_values)
route_color_map = dict(zip(route_values, palette))

# Gezielte Korrekturen fuer visuell zu aehnliche Routenfarben
route_color_map.update({
    "1": "#1f77b4",
    "10": "#aec7e8",
    "11": "#8B0000",
    "81": "#ff7f0e",
    "530": "#7f7f7f",
    "612": "#17becf"
})

route_map = trip_route_map(aux_event_log_overview)

route_type_counts = (
    aux_event_log_overview.dropna(subset=["route_type"])
    .assign(route_type_label=lambda df: df["route_type"].apply(route_type_label))
    .groupby("route_type_label")
    .size()
    .reset_index(name="event_count")
    .sort_values("event_count", ascending=False)
)

print(tabulate(
    route_type_counts,
    headers="keys",
    tablefmt="psql",
    showindex=False
))

plot_bar(
    route_type_counts,
    x="route_type_label",
    y="event_count",
    title="Frequency of Route Types",
    xlabel="Route Type",
    ylabel="Number of Events",
    rotate=30,
    color="darkcyan",
    short_labels=True,
    show_reference_lines=False
)

legend_handles = [
    mpatches.Patch(color=route_color_map[str(route)], label=str(route))
    for route in route_values
]
plt.figure(figsize=(12, max(4, len(route_values) * 0.25)))
plt.legend(
    handles=legend_handles,
    title="Route Short Name",
    loc="center",
    ncol=4,
    frameon=True,
    fancybox=False,
    edgecolor="black",
)
plt.axis("off")
plt.tight_layout()
plt.show()

activity_order = [
    "begin_shift",
    "direction_change",
    "end_layover",
    "arrive_stop",
    "departure_stop",
    "begin_layover",
    "parking"
]

activity_counts = (
    aux_event_log_overview["activity_type"]
    .value_counts()
    .reindex(activity_order, fill_value=0)
    .reset_index()
)
activity_counts.columns = ["activity_type", "count"]

plot_bar(
    activity_counts,
    x="activity_type",
    y="count",
    title="Frequency of event types",
    xlabel="Event Type",
    ylabel="Number of Events",
    short_labels=True
)
print(tabulate(activity_counts, headers="keys", tablefmt="psql", showindex=False))

operator_source = aux_event_log_overview.dropna(
    subset=["organization_id", "organization_name", "trip_id"]
).copy()

if "cumulative_road_distance_km" in operator_source.columns and operator_source["cumulative_road_distance_km"].notna().any():
    operator_source["trip_distance_km"] = operator_source["cumulative_road_distance_km"]
else:
    operator_source["trip_distance_km"] = operator_source["shape_dist_traveled"] / 1000

operator_trip_distance = (
    operator_source.dropna(subset=["trip_distance_km"])
    .groupby(["organization_id", "organization_name", "trip_id"], as_index=False)["trip_distance_km"]
    .max()
)

operator_distance = (
    operator_trip_distance
    .groupby(["organization_id", "organization_name"], as_index=False)["trip_distance_km"]
    .sum()
    .rename(columns={"trip_distance_km": "total_distance_km"})
)

operator_summary = (
    operator_source
    .groupby(["organization_id", "organization_name"], as_index=False)
    .agg(
        trip_count=("trip_id", "nunique"),
        route_count=("route_id", "nunique"),
        avg_occupancy_status=("occupancy_status", "mean"),
        avg_delay_seconds=("delay", "mean"),
    )
    .merge(operator_distance, on=["organization_id", "organization_name"], how="left")
)

operator_summary["total_distance_km"] = operator_summary["total_distance_km"].fillna(0)
operator_summary = operator_summary[
    [
        "organization_id",
        "organization_name",
        "total_distance_km",
        "trip_count",
        "route_count",
        "avg_occupancy_status",
        "avg_delay_seconds",
    ]
].sort_values("total_distance_km", ascending=False)

operator_summary.to_csv("operator_summary_kodak_2026_05_04.csv", index=False)
write_operator_summary_latex(operator_summary, "operator_summary_kodak_2026_05_04.tex")

print(tabulate(
    operator_summary,
    headers="keys",
    tablefmt="psql",
    showindex=False
))

plot_bar(
    operator_summary,
    x="organization_name",
    y="total_distance_km",
    title="Total Distance per Operator",
    xlabel="Operator",
    ylabel="Total Distance (km)",
    rotate=45,
    color="seagreen",
    short_labels=True,
    save_path="operator_total_distance_kodak_2026_05_04.png"
)

plot_bar(
    operator_summary,
    x="organization_name",
    y="trip_count",
    title="Trips per Operator",
    xlabel="Operator",
    ylabel="Number of Trips",
    rotate=45,
    color="steelblue",
    short_labels=True,
    save_path="operator_trip_count_kodak_2026_05_04.png"
)

plot_bar(
    operator_summary,
    x="organization_name",
    y="route_count",
    title="Routes per Operator",
    xlabel="Operator",
    ylabel="Number of Routes",
    rotate=45,
    color="darkcyan",
    short_labels=True,
    save_path="operator_route_count_kodak_2026_05_04.png"
)

plot_bar(
    operator_summary,
    x="organization_name",
    y="avg_occupancy_status",
    title="Average Occupancy Status per Operator",
    xlabel="Operator",
    ylabel="Average Occupancy Status",
    rotate=45,
    color="mediumpurple",
    short_labels=False,
    save_path="operator_average_occupancy_kodak_2026_05_04.png"
)

plot_bar(
    operator_summary,
    x="organization_name",
    y="avg_delay_seconds",
    title="Average Delay per Operator",
    xlabel="Operator",
    ylabel="Average Delay (seconds)",
    rotate=45,
    color="indianred",
    short_labels=False,
    save_path="operator_average_delay_kodak_2026_05_04.png"
)

trip_ids = aux_event_log_overview.sort_values("timestamp")["trip_id"].dropna().unique()
filtered_df = aux_event_log_overview[aux_event_log_overview["trip_id"].isin(trip_ids)].copy()

trip_order = (
    filtered_df.groupby("trip_id")["timestamp"]
    .min()
    .sort_values()
    .index
)
trip_id_mapping = {
    old_trip_id: new_trip_id
    for new_trip_id, old_trip_id in enumerate(trip_order, start=1)
}
filtered_df["trip_id_original"] = filtered_df["trip_id"]
filtered_df["trip_id"] = filtered_df["trip_id"].map(trip_id_mapping).astype("Int64")

print(tabulate(
    filtered_df[filtered_df["vehicle_id"] == "veh_28"].sort_values("timestamp").head(100),
    headers="keys",
    tablefmt="psql"
))

plt.figure(figsize=(18, 10))
ax = sns.scatterplot(
    data=aux_event_log_overview,
    x="timestamp",
    y="trip_id",
    hue="occupancy_status"
)
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.invert_yaxis()
plt.xticks(rotation=45)
plt.title("Dotted Chart of Trips")
plt.xlabel("Time")
plt.ylabel("Trip ID")
plt.tight_layout()
plt.show()

cmap = mcolors.LinearSegmentedColormap.from_list(
    "delay_map",
    ["#ffd6d6", "#ff4d4d", "#8b0000", "#000000"]
)
ab = sns.scatterplot(
    data=filtered_df[filtered_df["trip_id"].isin([1, 2, 3, 4, 5, 6, 8, 27, 33, 35, 36, 42, 43])].sort_values("timestamp"),
    x="timestamp",
    y="trip_id",
    hue="activity_type",
  #  palette=cmap,
   # hue_norm=(0, 500)
)
plt.xticks(rotation=45)
ab.invert_yaxis()
plt.title("Dotted Chart of Trips")
plt.xlabel("Time")
plt.ylabel("Trip ID")
plt.tight_layout()
plt.show()

print(tabulate(
    aux_event_log_overview.head(10),
    headers="keys",
    tablefmt="psql"
))

# Durchlaufzeit
begin_layover = aux_event_log_overview[
    aux_event_log_overview["activity_type"] == "begin_layover"
][["trip_id", "timestamp"]].rename(columns={"timestamp": "begin_layover_ts"})

end_layover = aux_event_log_overview[
    aux_event_log_overview["activity_type"] == "end_layover"
][["trip_id", "timestamp"]].rename(columns={"timestamp": "end_layover_ts"})

layover_times = begin_layover.merge(end_layover, on="trip_id", how="inner")
layover_times["layover_duration"] = layover_times["begin_layover_ts"] - layover_times["end_layover_ts"]
layover_times["layover_duration_min"] = layover_times["layover_duration"].dt.total_seconds() / 60

print(tabulate(
    layover_times.sort_values("layover_duration"),
    headers="keys",
    tablefmt="psql",
    showindex=False
))

plot_trip_bar_with_route(
    layover_times.rename(columns={"layover_duration_min": "value"}),
    x="trip_id",
    y="value",
    title="Throughput Time per Trip",
    ylabel="Throughput Time (minutes)",
    route_map=route_map,
    route_color_map=route_color_map,
    top_n=30,
    short_labels=True
)

layover_time_series = layover_times.copy()
layover_time_series["date"] = layover_time_series["begin_layover_ts"].dt.floor("h")
layover_time_series = layover_time_series.groupby("date", as_index=False)["layover_duration_min"].mean()

plt.figure(figsize=(15, 6))
sns.lineplot(data=layover_time_series, x="date", y="layover_duration_min", marker="o")
plt.xticks(rotation=45, ha="right")
plt.title("Throughput Time over Time")
plt.xlabel("Time")
plt.ylabel("Average Throughput Time (minutes)")
plt.tight_layout()
plt.show()

# Netto Fahrzeit
netto_df = aux_event_log_overview.copy().sort_values(["trip_id", "timestamp"]).reset_index(drop=True)
netto_df = netto_df[netto_df["activity_type"].isin(["departure_stop", "arrive_stop"])].copy()
netto_df["next_activity_type"] = netto_df.groupby("trip_id")["activity_type"].shift(-1)
netto_df["next_timestamp"] = netto_df.groupby("trip_id")["timestamp"].shift(-1)

netto_segments = netto_df[
    (netto_df["activity_type"] == "departure_stop") &
    (netto_df["next_activity_type"] == "arrive_stop")
].copy()

netto_segments["segment_duration"] = netto_segments["next_timestamp"] - netto_segments["timestamp"]
netto_segments["segment_duration_min"] = netto_segments["segment_duration"].dt.total_seconds() / 60

netto_fahrzeit_pro_trip = netto_segments.groupby("trip_id", as_index=False)["segment_duration"].sum()
netto_fahrzeit_pro_trip["netto_fahrzeit_min"] = netto_fahrzeit_pro_trip["segment_duration"].dt.total_seconds() / 60

print(tabulate(
    netto_fahrzeit_pro_trip.sort_values("netto_fahrzeit_min"),
    headers="keys",
    tablefmt="psql",
    showindex=False
))

plot_trip_bar_with_route(
    netto_fahrzeit_pro_trip.rename(columns={"netto_fahrzeit_min": "value"}),
    x="trip_id",
    y="value",
    title="Net Travel Time per Trip",
    ylabel="Net Travel Time (minutes)",
    route_map=route_map,
    route_color_map=route_color_map,
    top_n=50,
    short_labels=True
)

# Dwell time
halt_df = aux_event_log_overview.copy().sort_values(["trip_id", "timestamp"]).reset_index(drop=True)
halt_df = halt_df[halt_df["activity_type"].isin(["arrive_stop", "departure_stop"])].copy()
halt_df["next_activity_type"] = halt_df.groupby("trip_id")["activity_type"].shift(-1)
halt_df["next_timestamp"] = halt_df.groupby("trip_id")["timestamp"].shift(-1)

halt_segments = halt_df[
    (halt_df["activity_type"] == "arrive_stop") &
    (halt_df["next_activity_type"] == "departure_stop")
].copy()

halt_segments["halt_duration"] = halt_segments["next_timestamp"] - halt_segments["timestamp"]
halt_segments["halt_duration_min"] = halt_segments["halt_duration"].dt.total_seconds() / 60

haltzeit_pro_trip = halt_segments.groupby("trip_id", as_index=False)["halt_duration"].sum()
haltzeit_pro_trip["haltzeit_min"] = haltzeit_pro_trip["halt_duration"].dt.total_seconds() / 60

print(tabulate(
    haltzeit_pro_trip.sort_values("haltzeit_min"),
    headers="keys",
    tablefmt="psql",
    showindex=False
))

plot_trip_bar_with_route(
    haltzeit_pro_trip.rename(columns={"haltzeit_min": "value"}),
    x="trip_id",
    y="value",
    title="Dwell Time per Trip",
    ylabel="Dwell Time (minutes)",
    route_map=route_map,
    route_color_map=route_color_map,
    top_n=50,
    short_labels=True
)

halt_segments_time = halt_segments.copy()
halt_segments_time["date"] = halt_segments_time["timestamp"].dt.floor("h")
halt_time_series = halt_segments_time.groupby("date", as_index=False)["halt_duration_min"].mean()
dwell_time_average = halt_time_series["halt_duration_min"].mean()

plt.figure(figsize=(15, 6))
sns.lineplot(data=halt_time_series, x="date", y="halt_duration_min", marker="o", label="Average Dwell Time")
plt.axhline(
    y=dwell_time_average,
    color="darkred",
    linestyle="--",
    linewidth=1.5,
    label=f"Overall Average ({dwell_time_average:.2f} min)"
)
plt.xticks(rotation=45, ha="right")
plt.title("Dwell Time over Time")
plt.xlabel("Time")
plt.ylabel("Average Dwell Time (minutes)")
plt.legend(frameon=True, fancybox=False, edgecolor="black")
plt.tight_layout()
plt.show()

# Distance per trip
max_dist_per_trip = (
    aux_event_log_overview.groupby("trip_id", as_index=False)["shape_dist_traveled"]
    .max()
    .rename(columns={"shape_dist_traveled": "max_shape_dist_traveled"})
)

print(tabulate(
    max_dist_per_trip.sort_values("max_shape_dist_traveled", ascending=False).head(100),
    headers="keys",
    tablefmt="psql",
    showindex=False
))

plot_trip_bar_with_route(
    max_dist_per_trip.rename(columns={"max_shape_dist_traveled": "value"}),
    x="trip_id",
    y="value",
    title="Distance per Trip",
    ylabel="Maximum Distance",
    route_map=route_map,
    route_color_map=route_color_map,
    top_n=30,
    short_labels=True
)

# Average Distanz zwischen Haltestops
arrive_df = aux_event_log_overview[aux_event_log_overview["activity_type"] == "arrive_stop"].copy()
arrive_df = arrive_df.sort_values(["trip_id", "timestamp"])
arrive_df["distance_step"] = arrive_df.groupby("trip_id")["shape_dist_traveled"].diff()

avg_distance_per_trip = (
    arrive_df.groupby("trip_id", as_index=False)["distance_step"]
    .mean()
    .rename(columns={"distance_step": "avg_distance_between_stops"})
)

print(tabulate(
    avg_distance_per_trip,
    headers="keys",
    tablefmt="psql",
    showindex=False
))

plot_trip_bar_with_route(
    avg_distance_per_trip.rename(columns={"avg_distance_between_stops": "value"}),
    x="trip_id",
    y="value",
    title="Average Distance Between Stops per Trip",
    ylabel="Average Distance",
    route_map=route_map,
    route_color_map=route_color_map,
    top_n=30,
    short_labels=True
)

# Zeit Unterschiede zwischen arrive_stops
arrive_time_df = aux_event_log_overview[aux_event_log_overview["activity_type"] == "arrive_stop"].copy()
arrive_time_df = arrive_time_df.sort_values(["trip_id", "timestamp"])
arrive_time_df["time_step"] = arrive_time_df.groupby("trip_id")["timestamp"].diff()

avg_time_per_trip = (
    arrive_time_df.groupby("trip_id", as_index=False)["time_step"]
    .mean()
    .rename(columns={"time_step": "avg_time_between_arrive_stops"})
)
avg_time_per_trip["avg_time_between_arrive_stops_min"] = (
    avg_time_per_trip["avg_time_between_arrive_stops"].dt.total_seconds() / 60
)

print(tabulate(
    avg_time_per_trip,
    headers="keys",
    tablefmt="psql",
    showindex=False
))

# Average Occupancy Status
avg_occupancy_per_trip = (
    aux_event_log_overview.groupby("trip_id", as_index=False)["occupancy_status"]
    .mean()
    .rename(columns={"occupancy_status": "avg_occupancy_status"})
)

print(tabulate(
    avg_occupancy_per_trip,
    headers="keys",
    tablefmt="psql",
    showindex=False
))

plot_trip_bar_with_route(
    avg_occupancy_per_trip.rename(columns={"avg_occupancy_status": "value"}),
    x="trip_id",
    y="value",
    title="Average Occupancy Status per Trip",
    ylabel="Average Occupancy Status",
    route_map=route_map,
    route_color_map=route_color_map,
    top_n=30,
    short_labels=True
)

# Average Occupancy Status per route
avg_occupancy_per_route = (
    aux_event_log_overview.dropna(subset=["route_short_name"])
    .groupby("route_short_name", as_index=False)["occupancy_status"]
    .mean()
    .rename(columns={"occupancy_status": "avg_occupancy_status"})
    .sort_values("avg_occupancy_status", ascending=False)
)

print(tabulate(
    avg_occupancy_per_route,
    headers="keys",
    tablefmt="psql",
    showindex=False
))

plot_bar_by_route(
    avg_occupancy_per_route,
    x="route_short_name",
    y="avg_occupancy_status",
    title="Average Occupancy Status per Route",
    xlabel="Route Short Name",
    ylabel="Average Occupancy Status",
    route_color_map=route_color_map,
    top_n=25
)

# Average Occupancy Status over time
occupancy_time_series = aux_event_log_overview.dropna(
    subset=["timestamp", "occupancy_status"]
).copy()
occupancy_time_series["date"] = occupancy_time_series["timestamp"].dt.floor("h")
occupancy_time_series = (
    occupancy_time_series
    .groupby("date", as_index=False)["occupancy_status"]
    .mean()
)

plt.figure(figsize=(15, 6))
sns.lineplot(
    data=occupancy_time_series,
    x="date",
    y="occupancy_status",
    marker="o"
)
plt.xticks(rotation=45, ha="right")
plt.title("Occupancy Status over Time")
plt.xlabel("Time")
plt.ylabel("Average Occupancy Status")
plt.tight_layout()
plt.show()

# Average delay per trip
avg_delay_per_trip = (
    aux_event_log_overview.groupby("trip_id", as_index=False)["delay"]
    .mean()
    .rename(columns={"delay": "avg_delay"})
)

print(tabulate(
    avg_delay_per_trip,
    headers="keys",
    tablefmt="psql",
    showindex=False
))

plot_trip_bar_with_route(
    avg_delay_per_trip.rename(columns={"avg_delay": "value"}),
    x="trip_id",
    y="value",
    title="Average Delay per Trip",
    ylabel="Average Delay (seconds)",
    route_map=route_map,
    route_color_map=route_color_map,
    top_n=30,
    short_labels=True
)

delay_time_series = aux_event_log_overview.dropna(subset=["delay"]).copy()
delay_time_series["date"] = delay_time_series["timestamp"].dt.floor("h")
delay_time_series = delay_time_series.groupby("date", as_index=False)["delay"].mean()

plt.figure(figsize=(15, 6))
sns.lineplot(data=delay_time_series, x="date", y="delay", marker="o")
plt.xticks(rotation=45, ha="right")
plt.title("Delay over Time")
plt.xlabel("Time")
plt.ylabel("Average Delay (seconds)")
plt.tight_layout()
plt.show()

# Most used stops
most_used_stops = (
    aux_event_log_overview[aux_event_log_overview["activity_type"] == "arrive_stop"]
    .dropna(subset=["stop_name"])
    .groupby("stop_name")
    .size()
    .reset_index(name="arrive_stop_count")
    .sort_values("arrive_stop_count", ascending=False)
)

print(tabulate(
    most_used_stops,
    headers="keys",
    tablefmt="psql",
    showindex=False
))

stop_route_counts = (
    aux_event_log_overview[aux_event_log_overview["activity_type"] == "arrive_stop"]
    .dropna(subset=["stop_name", "route_short_name"])
    .groupby(["stop_name", "route_short_name"])
    .size()
    .reset_index(name="arrive_stop_count")
)

plot_stacked_bar(
    stop_route_counts,
    index_col="stop_name",
    stack_col="route_short_name",
    value_col="arrive_stop_count",
    title="Most Used Stops by Route",
    xlabel="Stop Name",
    ylabel="Number of Arrive Stops",
    route_values=route_values,
    route_color_map=route_color_map,
    top_n=20
)

plot_stacked_bar(
    stop_route_counts,
    index_col="stop_name",
    stack_col="route_short_name",
    value_col="arrive_stop_count",
    title="Most Used Stops by Route (Top 10)",
    xlabel="Stop Name",
    ylabel="Number of Arrive Stops",
    route_values=route_values,
    route_color_map=route_color_map,
    top_n=10
)

plot_stacked_bar(
    stop_route_counts,
    index_col="stop_name",
    stack_col="route_short_name",
    value_col="arrive_stop_count",
    title="Most Used Stops by Route (Top 5)",
    xlabel="Stop Name",
    ylabel="Number of Arrive Stops",
    route_values=route_values,
    route_color_map=route_color_map,
    top_n=5
)

# Most used routes
route_type_per_route = (
    aux_event_log_overview.dropna(subset=["route_short_name", "route_type"])
    .assign(
        route_short_name=lambda df: df["route_short_name"].astype(str),
        route_type=lambda df: df["route_type"].apply(short_number)
    )
    .groupby("route_short_name", as_index=False)["route_type"]
    .first()
)

trips_per_route = (
    aux_event_log_overview.dropna(subset=["trip_id", "route_short_name"])
    .assign(route_short_name=lambda df: df["route_short_name"].astype(str))
    .groupby("route_short_name")["trip_id"]
    .nunique()
    .reset_index()
    .rename(columns={"trip_id": "trip_count"})
    .merge(route_type_per_route, on="route_short_name", how="left")
    .sort_values("trip_count", ascending=False)
)

print(tabulate(
    trips_per_route,
    headers="keys",
    tablefmt="psql",
    showindex=False
))

most_used_routes_plot = trips_per_route.head(20).copy()

plt.figure(figsize=(14, 6))
ax = sns.barplot(
    data=most_used_routes_plot,
    x="route_short_name",
    y="trip_count",
    hue="route_short_name",
    dodge=False,
    palette=route_color_map
)
set_horizontal_route_legend(ax, "Route Short Name", most_used_routes_plot["route_short_name"])

visible_bars = [patch for patch in ax.patches if patch.get_height() > 0]
for patch, route_type_code in zip(visible_bars, most_used_routes_plot["route_type"].fillna("").astype(str)):
    ax.text(
        patch.get_x() + patch.get_width() / 2,
        patch.get_height(),
        route_type_code,
        ha="center",
        va="bottom",
        fontsize=8
    )

plt.xticks(rotation=60, ha="right", fontsize=7)
plt.title("Most Used Routes")
plt.xlabel("Route Short Name")
plt.ylabel("Number of Trips")
plt.subplots_adjust(bottom=0.18)
plt.tight_layout()
plt.show()

# Route - trips - distance
trip_distance = (
    aux_event_log_overview.dropna(subset=["trip_id", "route_short_name"])
    .groupby(["route_short_name", "trip_id"], as_index=False)["shape_dist_traveled"]
    .max()
    .rename(columns={"shape_dist_traveled": "trip_distance"})
)

route_summary = (
    trip_distance.groupby("route_short_name", as_index=False)
    .agg(
        trip_count=("trip_id", "nunique"),
        total_distance=("trip_distance", "sum"),
        avg_distance_per_trip=("trip_distance", "mean")
    )
    .sort_values("trip_count", ascending=False)
)

print(tabulate(
    route_summary,
    headers="keys",
    tablefmt="psql",
    showindex=False
))

plot_bar_by_route(
    route_summary,
    x="route_short_name",
    y="trip_count",
    title="Trips per Route",
    xlabel="Route Short Name",
    ylabel="Number of Trips",
    route_color_map=route_color_map,
    short_labels=True
)

plot_bar(
    route_summary.sort_values("total_distance", ascending=False),
    x="route_short_name",
    y="total_distance",
    title="Total Distance per Route",
    xlabel="Route Short Name",
    ylabel="Total Distance",
    short_labels=True
)

# Route type - trips - delay - occupancy - net duration - distance
route_type_trip_source = aux_event_log_overview.dropna(subset=["trip_id", "route_type"]).copy()
route_type_trip_source["route_type"] = route_type_trip_source["route_type"].astype(str)
route_type_trip_source["route_type_label"] = route_type_trip_source["route_type"].apply(route_type_label)

trips_per_route_type = (
    route_type_trip_source.groupby("route_type_label")["trip_id"]
    .nunique()
    .reset_index()
    .rename(columns={"trip_id": "trip_count"})
    .sort_values("trip_count", ascending=False)
)

print(tabulate(
    trips_per_route_type,
    headers="keys",
    tablefmt="psql",
    showindex=False
))

plot_bar(
    trips_per_route_type,
    x="route_type_label",
    y="trip_count",
    title="Trips by Route Type",
    xlabel="Route Type",
    ylabel="Number of Trips",
    show_reference_lines=False
)

route_type_delay_source = aux_event_log_overview.dropna(subset=["route_type", "delay"]).copy()
route_type_delay_source["route_type"] = route_type_delay_source["route_type"].astype(str)
route_type_delay_source["route_type_label"] = route_type_delay_source["route_type"].apply(route_type_label)

avg_delay_per_route_type = (
    route_type_delay_source.groupby("route_type_label", as_index=False)["delay"]
    .mean()
    .rename(columns={"delay": "avg_delay"})
    .sort_values("avg_delay", ascending=False)
)

print(tabulate(
    avg_delay_per_route_type,
    headers="keys",
    tablefmt="psql",
    showindex=False
))

plot_bar(
    avg_delay_per_route_type,
    x="route_type_label",
    y="avg_delay",
    title="Average Delay by Route Type",
    xlabel="Route Type",
    ylabel="Average Delay (seconds)",
    rotate=30,
    show_reference_lines=False
)

route_type_occupancy_source = aux_event_log_overview.dropna(
    subset=["route_type", "occupancy_status"]
).copy()
route_type_occupancy_source["route_type"] = route_type_occupancy_source["route_type"].astype(str)
route_type_occupancy_source["route_type_label"] = (
    route_type_occupancy_source["route_type"].apply(route_type_label)
)

avg_occupancy_per_route_type = (
    route_type_occupancy_source.groupby("route_type_label", as_index=False)["occupancy_status"]
    .mean()
    .rename(columns={"occupancy_status": "avg_occupancy_status"})
    .sort_values("avg_occupancy_status", ascending=False)
)

print(tabulate(
    avg_occupancy_per_route_type,
    headers="keys",
    tablefmt="psql",
    showindex=False
))

plot_bar(
    avg_occupancy_per_route_type,
    x="route_type_label",
    y="avg_occupancy_status",
    title="Average Occupancy Status by Route Type",
    xlabel="Route Type",
    ylabel="Average Occupancy Status",
    rotate=30,
    show_reference_lines=False
)

trip_route_type_map = (
    aux_event_log_overview.dropna(subset=["trip_id", "route_type"])
    .assign(route_type_label=lambda df: df["route_type"].apply(route_type_label))
    .groupby("trip_id", as_index=False)["route_type_label"]
    .first()
)

netto_duration_per_route_type = (
    netto_fahrzeit_pro_trip[["trip_id", "netto_fahrzeit_min"]]
    .merge(trip_route_type_map, on="trip_id", how="inner")
    .groupby("route_type_label", as_index=False)
    .agg(
        avg_netto_duration_min=("netto_fahrzeit_min", "mean"),
        trip_count=("trip_id", "nunique")
    )
    .sort_values("avg_netto_duration_min", ascending=False)
)

print(tabulate(
    netto_duration_per_route_type,
    headers="keys",
    tablefmt="psql",
    showindex=False
))

plot_bar(
    netto_duration_per_route_type,
    x="route_type_label",
    y="avg_netto_duration_min",
    title="Average Net Trip Duration by Route Type",
    xlabel="Route Type",
    ylabel="Average Net Trip Duration (minutes)",
    rotate=30,
    show_reference_lines=False
)

median_delay_per_route_type = (
    route_type_delay_source.groupby("route_type_label", as_index=False)["delay"]
    .median()
    .rename(columns={"delay": "median_delay"})
    .sort_values("median_delay", ascending=False)
)

print(tabulate(
    median_delay_per_route_type,
    headers="keys",
    tablefmt="psql",
    showindex=False
))

plot_bar(
    median_delay_per_route_type,
    x="route_type_label",
    y="median_delay",
    title="Median Delay by Route Type",
    xlabel="Route Type",
    ylabel="Median Delay (seconds)",
    rotate=30,
    show_reference_lines=False
)

route_type_distance_source = aux_event_log_overview.dropna(
    subset=["trip_id", "route_type", "shape_dist_traveled"]
).copy()
route_type_distance_source["route_type"] = route_type_distance_source["route_type"].astype(str)
route_type_distance_source["route_type_label"] = route_type_distance_source["route_type"].apply(route_type_label)

trip_distance_per_route_type = (
    route_type_distance_source.groupby(["route_type_label", "trip_id"], as_index=False)["shape_dist_traveled"]
    .max()
    .rename(columns={"shape_dist_traveled": "trip_distance"})
)

distance_per_route_type = (
    trip_distance_per_route_type.groupby("route_type_label", as_index=False)["trip_distance"]
    .mean()
    .rename(columns={"trip_distance": "avg_distance_per_trip"})
    .sort_values("avg_distance_per_trip", ascending=False)
)

print(tabulate(
    distance_per_route_type,
    headers="keys",
    tablefmt="psql",
    showindex=False
))

plot_bar(
    distance_per_route_type,
    x="route_type_label",
    y="avg_distance_per_trip",
    title="Average Distance per Trip by Route Type",
    xlabel="Route Type",
    ylabel="Average Distance per Trip",
    rotate=30,
    show_reference_lines=False
)

route_type_summary_table = (
    trips_per_route_type
    .merge(avg_delay_per_route_type, on="route_type_label", how="outer")
    .merge(avg_occupancy_per_route_type, on="route_type_label", how="outer")
    .merge(
        netto_duration_per_route_type[
            ["route_type_label", "avg_netto_duration_min"]
        ],
        on="route_type_label",
        how="outer"
    )
    .merge(distance_per_route_type, on="route_type_label", how="outer")
    .sort_values("trip_count", ascending=False)
)

route_type_summary_table = route_type_summary_table[
    [
        "route_type_label",
        "trip_count",
        "avg_delay",
        "avg_occupancy_status",
        "avg_netto_duration_min",
        "avg_distance_per_trip",
    ]
].round(2)

print("\nRoute type summary:")
print(tabulate(
    route_type_summary_table,
    headers="keys",
    tablefmt="psql",
    showindex=False
))

route_type_summary_table.to_csv(
    "route_type_summary_kodak_2025_05_04.csv",
    index=False
)


#Most used vehicles und Kilometeranzahl


trip_distance_per_vehicle = (
    aux_event_log_overview.dropna(subset=["vehicle_id", "trip_id", "route_short_name"])
    .groupby(["vehicle_id", "route_short_name", "trip_id"], as_index=False)["shape_dist_traveled"]
    .max()
    .rename(columns={"shape_dist_traveled": "trip_distance"})
)

vehicle_trip_counts = (
    trip_distance_per_vehicle.groupby("vehicle_id", as_index=False)
    .agg(
        trip_count=("trip_id", "nunique"),
        total_distance=("trip_distance", "sum"),
        avg_distance_per_trip=("trip_distance", "mean")
    )
    .sort_values("trip_count", ascending=False)
)

print(tabulate(
    vehicle_trip_counts,
    headers="keys",
    tablefmt="psql",
    showindex=False
))

# Trips pro Vehicle aufgeteilt nach Route
vehicle_route_trip_counts = (
    trip_distance_per_vehicle.groupby(["vehicle_id", "route_short_name"], as_index=False)
    .agg(trip_count=("trip_id", "nunique"))
)

plot_stacked_bar(
    vehicle_route_trip_counts,
    index_col="vehicle_id",
    stack_col="route_short_name",
    value_col="trip_count",
    title="Trips per Vehicle by Route",
    xlabel="Vehicle ID",
    ylabel="Number of Trips",
    route_values=route_values,
    route_color_map=route_color_map,
    top_n=20
)

plot_stacked_bar(
    vehicle_route_trip_counts,
    index_col="vehicle_id",
    stack_col="route_short_name",
    value_col="trip_count",
    title="Trips per Vehicle by Route (Top 10)",
    xlabel="Vehicle ID",
    ylabel="Number of Trips",
    route_values=route_values,
    route_color_map=route_color_map,
    top_n=10
)

plot_stacked_bar(
    vehicle_route_trip_counts,
    index_col="vehicle_id",
    stack_col="route_short_name",
    value_col="trip_count",
    title="Trips per Vehicle by Route (Top 5)",
    xlabel="Vehicle ID",
    ylabel="Number of Trips",
    route_values=route_values,
    route_color_map=route_color_map,
    top_n=5
)

# Gesamtdistanz pro Vehicle aufgeteilt nach Route
vehicle_route_distance = (
    trip_distance_per_vehicle.groupby(["vehicle_id", "route_short_name"], as_index=False)
    .agg(total_distance=("trip_distance", "sum"))
)

plot_stacked_bar(
    vehicle_route_distance,
    index_col="vehicle_id",
    stack_col="route_short_name",
    value_col="total_distance",
    title="Total Distance per Vehicle by Route",
    xlabel="Vehicle ID",
    ylabel="Total Distance",
    route_values=route_values,
    route_color_map=route_color_map,
    top_n=40
)

def plot_distribution_boxplot(df, column, title, ylabel):
    plot_df = df[[column]].dropna().copy()

    stats = plot_df[column].agg(["min", "max", "median", "mean", "std"])
    print(f"\nStatistics for {column}:")
    print(stats)

    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(
        data=plot_df,
        y=column,
        width=0.35,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "red",
            "markeredgecolor": "black",
            "markersize": 8
        },
        flierprops={
            "marker": "o",
            "markerfacecolor": "orange",
            "markeredgecolor": "black",
            "markersize": 5,
            "alpha": 0.7
        }
    )

    sns.stripplot(
        data=plot_df,
        y=column,
        color="gray",
        alpha=0.25,
        size=3,
        jitter=0.2
    )

    text = (
        f"Min: {stats['min']:.2f}\n"
        f"Max: {stats['max']:.2f}\n"
        f"Median: {stats['median']:.2f}\n"
        f"Mean: {stats['mean']:.2f}\n"
        f"Std: {stats['std']:.2f}"
    )

    ax.text(
        1.05, 0.5, text,
        transform=ax.transAxes,
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("")
    plt.tight_layout()
    plt.show()

plot_distribution_boxplot(
        aux_event_log_overview,
        column="occupancy_status",
        title="Distribution of Occupancy Status",
        ylabel="Occupancy Status"
    )

plot_distribution_boxplot(
        aux_event_log_overview,
        column="delay",
        title="Distribution of Delay",
        ylabel="Delay (seconds)"
    )
