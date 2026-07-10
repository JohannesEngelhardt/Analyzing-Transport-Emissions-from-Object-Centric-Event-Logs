from pathlib import Path
import os

MPLCONFIG_DIR = Path("speed_analysis") / ".matplotlib"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


OUTPUT_DIR = Path("speed_analysis")
SUMMARY_PATTERN = "trip_speed_summary_*.csv"


def dataset_label(path: Path) -> str:
    stem = path.stem.replace("trip_speed_summary_", "")
    parts = stem.split("_")
    if len(parts) == 3:
        return f"KODA {parts[0]}-{parts[1]}-{parts[2]}"
    return stem


def load_trip_speed_summaries() -> pd.DataFrame:
    paths = sorted(Path(".").glob(SUMMARY_PATTERN))
    if not paths:
        raise FileNotFoundError(f"No files found for pattern {SUMMARY_PATTERN}")

    frames = []
    for path in paths:
        df = pd.read_csv(path)
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        df["dataset"] = dataset_label(path)
        frames.append(df)

    speed = pd.concat(frames, ignore_index=True)
    numeric_cols = [
        "avg_speed",
        "median_speed",
        "speed_q25",
        "speed_q58",
        "speed_q60",
        "speed_q625",
        "speed_q75",
        "speed_count",
    ]
    for col in numeric_cols:
        if col in speed.columns:
            speed[col] = pd.to_numeric(speed[col], errors="coerce")

    required = ["trip_id", "dataset", "avg_speed", "median_speed", "speed_q25", "speed_q75"]
    speed = speed.dropna(subset=[col for col in required if col in speed.columns]).copy()
    speed["speed_iqr"] = speed["speed_q75"] - speed["speed_q25"]
    speed["abs_avg_median_diff"] = (speed["avg_speed"] - speed["median_speed"]).abs()
    speed["relative_iqr"] = speed["speed_iqr"] / speed["median_speed"].replace(0, pd.NA)
    speed["relative_avg_median_diff"] = (
        speed["abs_avg_median_diff"] / speed["median_speed"].replace(0, pd.NA)
    )
    return speed


def apply_plot_style() -> None:
    sns.set_theme(style="whitegrid", font="DejaVu Sans")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
        }
    )


def add_reference_line(ax, value: float, label: str, color: str, linestyle: str = "--") -> None:
    ax.axhline(y=value, color=color, linestyle=linestyle, linewidth=1.5)
    ax.text(
        1.01,
        value,
        f" {label}: {value:.2f}",
        color=color,
        va="center",
        ha="left",
        fontsize=9,
        transform=ax.get_yaxis_transform(),
        clip_on=False,
    )


def save_boxplot(
    data: pd.DataFrame,
    value_col: str,
    title: str,
    ylabel: str,
    filename_stem: str,
    showfliers: bool = False,
) -> None:
    plot_df = data.dropna(subset=["dataset", value_col]).copy()
    average_value = plot_df[value_col].mean()
    median_value = plot_df[value_col].median()

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(
        data=plot_df,
        x="dataset",
        y=value_col,
        color="darkcyan",
        showfliers=showfliers,
        width=0.55,
        linewidth=1.1,
        ax=ax,
    )

    add_reference_line(ax, average_value, "AVG", "darkred")
    add_reference_line(ax, median_value, "MED", "navy", linestyle=":")

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Dataset")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{filename_stem}.png", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{filename_stem}.svg", bbox_inches="tight")
    plt.close(fig)


def save_speed_level_boxplot(data: pd.DataFrame) -> None:
    plot_df = data.melt(
        id_vars=["dataset", "trip_id"],
        value_vars=["avg_speed", "median_speed"],
        var_name="metric",
        value_name="speed",
    )
    plot_df["metric"] = plot_df["metric"].replace(
        {"avg_speed": "Average speed", "median_speed": "Median speed"}
    )
    average_value = plot_df["speed"].mean()
    median_value = plot_df["speed"].median()

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(
        data=plot_df,
        x="dataset",
        y="speed",
        hue="metric",
        palette={"Average speed": "steelblue", "Median speed": "darkcyan"},
        showfliers=False,
        width=0.65,
        linewidth=1.1,
        ax=ax,
    )

    add_reference_line(ax, average_value, "AVG", "darkred")
    add_reference_line(ax, median_value, "MED", "navy", linestyle=":")

    ax.set_title("Aggregated speed values per trip")
    ax.set_ylabel("Speed")
    ax.set_xlabel("Dataset")
    plt.xticks(rotation=30, ha="right")
    legend = ax.legend(
        title="Speed metric",
        loc="upper right",
        frameon=True,
        fancybox=False,
        edgecolor="black",
    )
    if legend is not None:
        legend.get_frame().set_linewidth(1.0)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "trip_speed_avg_median_boxplot.png", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "trip_speed_avg_median_boxplot.svg", bbox_inches="tight")
    plt.close(fig)


def write_summary(data: pd.DataFrame) -> pd.DataFrame:
    summary = (
        data.groupby("dataset")
        .agg(
            trips=("trip_id", "nunique"),
            median_avg_speed=("avg_speed", "median"),
            median_median_speed=("median_speed", "median"),
            median_speed_iqr=("speed_iqr", "median"),
            p75_speed_iqr=("speed_iqr", lambda s: s.quantile(0.75)),
            p90_speed_iqr=("speed_iqr", lambda s: s.quantile(0.90)),
            median_relative_iqr=("relative_iqr", "median"),
            median_abs_avg_median_diff=("abs_avg_median_diff", "median"),
            median_speed_count=("speed_count", "median"),
        )
        .reset_index()
    )
    summary.to_csv(OUTPUT_DIR / "trip_speed_boxplot_summary.csv", index=False)

    lines = []
    for _, row in summary.iterrows():
        lines.append(
            f"{row['dataset']}: {int(row['trips'])} trips, "
            f"median(avg_speed)={row['median_avg_speed']:.2f}, "
            f"median(IQR)={row['median_speed_iqr']:.2f}, "
            f"p90(IQR)={row['p90_speed_iqr']:.2f}, "
            f"median(relative IQR)={row['median_relative_iqr']:.2%}."
        )
    (OUTPUT_DIR / "trip_speed_boxplot_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    apply_plot_style()

    speed = load_trip_speed_summaries()
    speed.to_csv(OUTPUT_DIR / "trip_speed_boxplot_data.csv", index=False)

    save_speed_level_boxplot(speed)
    save_boxplot(
        speed,
        "speed_iqr",
        "Speed variation within trips",
        "IQR of speed per trip (q75 - q25)",
        "trip_speed_iqr_boxplot",
    )
    save_boxplot(
        speed,
        "relative_iqr",
        "Relative speed variation within trips",
        "IQR / median speed",
        "trip_speed_relative_iqr_boxplot",
    )
    save_boxplot(
        speed,
        "abs_avg_median_diff",
        "Difference between average and median speed",
        "|average speed - median speed|",
        "trip_speed_avg_median_difference_boxplot",
    )

    summary = write_summary(speed)
    print(summary.to_string(index=False))
    print(f"\nSaved plots and tables to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
