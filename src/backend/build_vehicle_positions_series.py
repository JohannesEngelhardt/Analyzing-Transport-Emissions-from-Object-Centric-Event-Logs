from argparse import ArgumentParser
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd
from google.transit import gtfs_realtime_pb2


BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parents[1]
DEFAULT_PIPELINE_DIR = PROJECT_DIR / "koda_pipeline_downloads" / "koda_otraf_2026_05_04"
DEFAULT_INPUT_DIR = DEFAULT_PIPELINE_DIR / "extracted" / "vehicle_positions"
DEFAULT_OUTPUT_DIR = DEFAULT_PIPELINE_DIR / "csv" / "vehicle_positions"
DEFAULT_FASTLANE_SECONDS = [0, 9, 19, 29, 39, 49, 59]
TRIP_OCCUPANCY_ULTRA_FASTLANE_FILE = "trip_occupancy_ultra_fastlane.csv"


def pb_files_for_seconds(input_dir: Path, seconds: list[int]) -> list[Path]:
    # Fastlane reads only snapshots from selected seconds of each minute.
    pb_files_by_path = {}
    for second in seconds:
        second_label = f"{second:02d}"
        for pb_path in input_dir.rglob(f"*{second_label}Z.pb"):
            pb_files_by_path[pb_path] = pb_path
    return sorted(pb_files_by_path)


def vehicle_position_rows_from_pb(pb_path: Path) -> list[dict]:
    # Convert one GTFS-RT VehiclePositions protobuf file into flat CSV rows.
    feed = gtfs_realtime_pb2.FeedMessage()

    with open(pb_path, "rb") as f:
        feed.ParseFromString(f.read())

    rows = []

    for entity in feed.entity:
        if not entity.HasField("vehicle"):
            continue

        vp = entity.vehicle
        if not vp.HasField("trip"):
            continue

        trip_id = None
        route_id = None
        start_time = None
        start_date = None
        direction_id = None
        schedule_relationship = None

        if vp.trip.schedule_relationship is not None:
            trip_id = vp.trip.trip_id if vp.trip.trip_id else None
            route_id = vp.trip.route_id if vp.trip.route_id else None
            start_time = vp.trip.start_time if vp.trip.start_time else None
            start_date = vp.trip.start_date if vp.trip.start_date else None
            direction_id = (
                vp.trip.direction_id if vp.trip.direction_id is not None else None
            )
            schedule_relationship = vp.trip.schedule_relationship

        vehicle_id = None
        vehicle_label = None
        if vp.HasField("vehicle"):
            vehicle_id = vp.vehicle.id if vp.vehicle.id else None
            vehicle_label = vp.vehicle.label if vp.vehicle.label else None

        latitude = None
        longitude = None
        bearing = None
        speed = None
        if vp.HasField("position"):
            latitude = vp.position.latitude
            longitude = vp.position.longitude
            bearing = vp.position.bearing
            speed = vp.position.speed

        rows.append(
            {
                "source_file": pb_path.name,
                "entity_id": entity.id,
                "trip_id": trip_id,
                "route_id": route_id,
                "start_time": start_time,
                "start_date": start_date,
                "direction_id": direction_id,
                "schedule_relationship": schedule_relationship,
                "vehicle_id": vehicle_id,
                "vehicle_label": vehicle_label,
                "stop_id": vp.stop_id if vp.stop_id else None,
                "current_stop_sequence": (
                    vp.current_stop_sequence if vp.current_stop_sequence else None
                ),
                "current_status": (
                    vp.current_status if vp.current_status is not None else None
                ),
                "occupancy_status": (
                    vp.occupancy_status if vp.occupancy_status is not None else None
                ),
                "timestamp": vp.timestamp if vp.timestamp else None,
                "latitude": latitude,
                "longitude": longitude,
                "bearing": bearing,
                "speed": speed,
            }
        )

    return rows


def build_trip_occupancy_ultra_fastlane(
    vehicle_positions_csv: Path,
    output_dir: Path,
    chunksize: int = 500_000,
) -> Path | None:
    # Build a small trip-level table for trips where occupancy never changes.
    if not vehicle_positions_csv.exists():
        return None

    trip_stats: dict[str, dict] = {}
    total_rows = 0
    usable_rows = 0

    for chunk in pd.read_csv(
        vehicle_positions_csv,
        usecols=["trip_id", "occupancy_status"],
        dtype={"trip_id": "string"},
        chunksize=chunksize,
    ):
        total_rows += len(chunk)
        chunk = chunk.dropna(subset=["trip_id", "occupancy_status"]).copy()
        if chunk.empty:
            continue

        chunk["occupancy_status"] = pd.to_numeric(
            chunk["occupancy_status"],
            errors="coerce",
        )
        chunk = chunk.dropna(subset=["occupancy_status"])
        if chunk.empty:
            continue

        usable_rows += len(chunk)
        grouped = (
            chunk.groupby("trip_id", sort=False)["occupancy_status"]
            .agg(["min", "max", "count"])
            .reset_index()
        )

        for row in grouped.itertuples(index=False):
            trip_id = str(row.trip_id)
            stats = trip_stats.get(trip_id)
            if stats is None:
                trip_stats[trip_id] = {
                    "min": float(row.min),
                    "max": float(row.max),
                    "count": int(row.count),
                }
            else:
                stats["min"] = min(stats["min"], float(row.min))
                stats["max"] = max(stats["max"], float(row.max))
                stats["count"] += int(row.count)

    output_file = output_dir / TRIP_OCCUPANCY_ULTRA_FASTLANE_FILE
    constant_rows = [
        {
            "trip_id": trip_id,
            "occupancy_status": int(stats["min"]),
            "occupancy_sample_count": stats["count"],
        }
        for trip_id, stats in trip_stats.items()
        if stats["count"] > 0 and stats["min"] == stats["max"]
    ]
    spread_counts = {}
    for stats in trip_stats.values():
        if stats["count"] <= 0:
            continue
        spread = int(stats["max"] - stats["min"])
        spread_counts[str(spread)] = spread_counts.get(str(spread), 0) + 1

    pd.DataFrame(
        constant_rows,
        columns=["trip_id", "occupancy_status", "occupancy_sample_count"],
    ).to_csv(output_file, index=False)

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_csv": str(vehicle_positions_csv),
        "output_csv": str(output_file),
        "total_rows": total_rows,
        "usable_occupancy_rows": usable_rows,
        "trips_with_occupancy": len(trip_stats),
        "constant_occupancy_trips": len(constant_rows),
        "variable_occupancy_trips": len(trip_stats) - len(constant_rows),
        "occupancy_spread_counts": dict(sorted(spread_counts.items(), key=lambda item: int(item[0]))),
    }
    (output_dir / "trip_occupancy_ultra_fastlane_info.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return output_file


def update_trip_occupancy_stats(trip_stats: dict[str, dict], df: pd.DataFrame):
    # Track min/max occupancy per trip while streaming VehiclePositions.
    if df.empty or "trip_id" not in df.columns or "occupancy_status" not in df.columns:
        return 0

    occupancy = df[["trip_id", "occupancy_status"]].dropna(
        subset=["trip_id", "occupancy_status"]
    ).copy()
    if occupancy.empty:
        return 0

    occupancy["occupancy_status"] = pd.to_numeric(
        occupancy["occupancy_status"],
        errors="coerce",
    )
    occupancy = occupancy.dropna(subset=["occupancy_status"])
    if occupancy.empty:
        return 0

    grouped = (
        occupancy.groupby("trip_id", sort=False)["occupancy_status"]
        .agg(["min", "max", "count"])
        .reset_index()
    )

    for row in grouped.itertuples(index=False):
        trip_id = str(row.trip_id)
        stats = trip_stats.get(trip_id)
        if stats is None:
            trip_stats[trip_id] = {
                "min": float(row.min),
                "max": float(row.max),
                "count": int(row.count),
            }
        else:
            stats["min"] = min(stats["min"], float(row.min))
            stats["max"] = max(stats["max"], float(row.max))
            stats["count"] += int(row.count)

    return len(occupancy)


def write_trip_occupancy_ultra_fastlane(
    trip_stats: dict[str, dict],
    output_dir: Path,
    vehicle_positions_csv: Path,
    total_rows: int,
    usable_rows: int,
) -> Path:
    # Write only trips with constant occupancy; variable trips stay out of Ultra Fastlane.
    output_file = output_dir / TRIP_OCCUPANCY_ULTRA_FASTLANE_FILE
    constant_rows = [
        {
            "trip_id": trip_id,
            "occupancy_status": int(stats["min"]),
            "occupancy_sample_count": stats["count"],
        }
        for trip_id, stats in trip_stats.items()
        if stats["count"] > 0 and stats["min"] == stats["max"]
    ]
    spread_counts = {}
    for stats in trip_stats.values():
        if stats["count"] <= 0:
            continue
        spread = int(stats["max"] - stats["min"])
        spread_counts[str(spread)] = spread_counts.get(str(spread), 0) + 1

    pd.DataFrame(
        constant_rows,
        columns=["trip_id", "occupancy_status", "occupancy_sample_count"],
    ).to_csv(output_file, index=False)

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_csv": str(vehicle_positions_csv),
        "output_csv": str(output_file),
        "total_rows": total_rows,
        "usable_occupancy_rows": usable_rows,
        "trips_with_occupancy": len(trip_stats),
        "constant_occupancy_trips": len(constant_rows),
        "variable_occupancy_trips": len(trip_stats) - len(constant_rows),
        "occupancy_spread_counts": dict(sorted(spread_counts.items(), key=lambda item: int(item[0]))),
    }
    (output_dir / "trip_occupancy_ultra_fastlane_info.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return output_file


def build_vehicle_position_csvs(
    input_dir: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    mode: str = "complete",
    seconds: list[int] | None = None,
) -> Path | None:
    if mode == "fastlane":
        # Fastlane keeps fewer PB files to reduce runtime and memory use.
        selected_seconds = seconds or DEFAULT_FASTLANE_SECONDS
        pb_files = pb_files_for_seconds(input_dir, selected_seconds)
        source_label = ", ".join(f"{second:02d}Z" for second in selected_seconds)
    else:
        # Complete keeps every available VehiclePositions snapshot.
        pb_files = sorted(input_dir.rglob("*.pb"))
        source_label = "all available seconds"

    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "all_vehicle_positions.csv"
    if output_csv.exists():
        output_csv.unlink()

    written_rows = 0
    files_with_rows = 0
    trip_occupancy_stats: dict[str, dict] = {}
    usable_occupancy_rows = 0

    for file_index, pb_path in enumerate(pb_files, start=1):
        rows = vehicle_position_rows_from_pb(pb_path)
        if not rows:
            continue

        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")

        # Dedupe per PB file only; this keeps memory bounded for large feeds.
        df = df.drop_duplicates(
            subset=["timestamp", "trip_id", "vehicle_id", "latitude", "longitude"],
            keep="first",
        )
        if df.empty:
            continue

        # Collect Ultra Fastlane stats during the main pass to avoid reading the CSV again.
        usable_occupancy_rows += update_trip_occupancy_stats(trip_occupancy_stats, df)
        df.to_csv(output_csv, index=False, mode="a", header=written_rows == 0)
        written_rows += len(df)
        files_with_rows += 1

        if file_index % 1000 == 0:
            print(
                f"Processed {file_index}/{len(pb_files)} VehiclePosition PB files; "
                f"{written_rows} unique rows written."
            )

    if written_rows == 0:
        print(
            f"No VehiclePosition PB files with data found in {input_dir} "
            f"for {source_label}."
        )
        return None

    build_info = {
        "mode": mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "output_csv": str(output_csv),
        "pb_file_count": len(pb_files),
        "pb_files_with_rows": files_with_rows,
        "row_count": written_rows,
        "selected_seconds": selected_seconds if mode == "fastlane" else None,
        "source_label": source_label,
        "dedupe_scope": "per_file",
    }
    (output_dir / "build_info.json").write_text(
        json.dumps(build_info, indent=2),
        encoding="utf-8",
    )
    write_trip_occupancy_ultra_fastlane(
        trip_occupancy_stats,
        output_dir,
        output_csv,
        written_rows,
        usable_occupancy_rows,
    )

    print(
        f"{output_csv}: {written_rows} vehicle position updates "
        f"from {len(pb_files)} PB files ({source_label})"
    )

    return output_csv


def parse_args():
    parser = ArgumentParser(
        description=(
            "Creates a consolidated CSV file from VehiclePosition PB files. "
            "Complete mode processes all available seconds, while Fastlane "
            "processes only the selected seconds."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["complete", "fastlane"],
        default="complete",
        help="Complete processes all PB files, Fastlane processes only selected seconds.",
    )
    parser.add_argument(
        "--seconds",
        default="0,9,19,29,39,49,59",
        help="Comma-separated list of seconds for Fastlane mode, e.g. 0,9,19.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing VehiclePosition PB files. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for the consolidated CSV file. Default: {DEFAULT_OUTPUT_DIR}",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    selected_seconds = [
        int(value.strip())
        for value in args.seconds.split(",")
        if value.strip()
    ]

    written_file = build_vehicle_position_csvs(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        mode=args.mode,
        seconds=selected_seconds,
    )

    if written_file:
        print("Written CSV file:")
        print(written_file)
    else:
        print("No CSV file was written.")
