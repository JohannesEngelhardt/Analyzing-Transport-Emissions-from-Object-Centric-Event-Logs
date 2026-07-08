from __future__ import annotations

import json
import mimetypes
import os
import re
import shutil
import ssl
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen

import certifi
import pandas as pd
from google.transit import gtfs_realtime_pb2
from collections import Counter

try:
    from build_vehicle_positions_series import build_vehicle_position_csvs
except ModuleNotFoundError:
    from backend.build_vehicle_positions_series import build_vehicle_position_csvs


BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = Path(os.environ.get("KODA_PROJECT_DIR", BACKEND_DIR.parents[1])).expanduser()
FRONTEND_DIR = PROJECT_DIR / "src" / "frontend"
DOWNLOAD_ROOT = PROJECT_DIR / "koda_pipeline_downloads"
SERVER_HOST = os.environ.get("KODA_SERVER_HOST", "127.0.0.1")
SERVER_PORT = int(os.environ.get("KODA_SERVER_PORT", "8765"))

STATIC_URL = "https://api.koda.trafiklab.se/KoDa/api/v2/gtfs-static/{operator}"
REALTIME_URL = "https://api.koda.trafiklab.se/KoDa/api/v2/gtfs-rt/{operator}/{feed}"
REALTIME_FEEDS = {
    "trip_updates": "TripUpdates",
    "vehicle_positions": "VehiclePositions",
}


def resolve_html_file() -> Path:
    candidates = [
        Path(os.environ["KODA_FRONTEND_HTML"]).expanduser()
        for _ in [None]
        if os.environ.get("KODA_FRONTEND_HTML")
    ]
    candidates.extend(
        [
            FRONTEND_DIR / "koda_demo_frontend.html",
            Path.cwd() / "src" / "frontend" / "koda_demo_frontend.html",
            BACKEND_DIR.parent / "frontend" / "koda_demo_frontend.html",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


HTML_FILE = resolve_html_file()
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
JOBS: dict[str, dict] = {}
PROCESSES: dict[str, subprocess.Popen] = {}
JOBS_LOCK = threading.Lock()


def safe_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "-_." else "_" for char in value)
    return cleaned.strip("_") or "koda"


def request_url(template: str, operator: str, date: str, api_key: str, feed: str | None = None) -> str:
    values = {"operator": operator}
    if feed is not None:
        values["feed"] = feed
    return template.format(**values) + "?" + urlencode({"date": date, "key": api_key})


def content_disposition_filename(header_value: str | None) -> str | None:
    if not header_value:
        return None
    for part in header_value.split(";"):
        part = part.strip()
        if part.lower().startswith("filename="):
            return part.split("=", 1)[1].strip("\"'")
    return None


def download_file(url: str, target_dir: Path, fallback_name: str) -> tuple[Path, int, int]:
    target_dir.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "KODA local pipeline"})

    try:
        with urlopen(request, timeout=120, context=SSL_CONTEXT) as response:
            status = response.status
            filename = content_disposition_filename(response.headers.get("Content-Disposition"))
            if filename is None:
                extension = mimetypes.guess_extension(response.headers.get_content_type()) or ".bin"
                filename = fallback_name + extension
            output_file = target_dir / safe_name(filename)
            with open(output_file, "wb") as f:
                shutil.copyfileobj(response, f)
    except HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"Request rejected with HTTP {exc.code}: {message}") from exc
    except URLError as exc:
        raise RuntimeError(f"Request failed: {exc.reason}") from exc

    return output_file, status, output_file.stat().st_size


def extract_zip_if_needed(download_file_path: Path, extract_dir: Path) -> Path:
    if not zipfile.is_zipfile(download_file_path):
        return download_file_path.parent

    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(download_file_path) as archive:
        archive.extractall(extract_dir)
    return extract_dir


def is_7z_file(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(6) == b"7z\xbc\xaf\x27\x1c"
    except OSError:
        return False


def extract_7z_archive(archive_file: Path, extract_dir: Path):
    seven_zip = shutil.which("7z")
    if seven_zip is None:
        raise RuntimeError("7z archive found, but the '7z' command is not available.")

    subprocess.run(
        [seven_zip, "x", "-y", f"-o{extract_dir}", str(archive_file)],
        check=True,
        capture_output=True,
        text=True,
    )


def reset_directory(directory: Path):
    if directory.exists():
        stale_directory = directory.with_name(
            f"{directory.name}.stale-{uuid.uuid4().hex}"
        )
        try:
            directory.rename(stale_directory)
        except OSError:
            shutil.rmtree(directory, ignore_errors=True)
            if directory.exists():
                raise
        else:
            shutil.rmtree(stale_directory, ignore_errors=True)

    directory.mkdir(parents=True, exist_ok=True)


def prepare_realtime_input_from_raw(raw_dir: Path, extract_dir: Path, operator: str) -> tuple[Path, int]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw input folder not found: {raw_dir}")

    all_archive_files = [
        path
        for path in sorted(raw_dir.iterdir())
        if path.is_file() and (zipfile.is_zipfile(path) or is_7z_file(path))
    ]
    archive_files = [
        path for path in all_archive_files if archive_matches_operator(path, operator)
    ]

    if all_archive_files and not archive_files:
        if len(all_archive_files) == 1:
            archive_files = all_archive_files
        else:
            archive_names = ", ".join(path.name for path in all_archive_files)
            raise FileNotFoundError(
                f"No realtime archive for operator '{operator}' found in {raw_dir}. "
                f"Available archives: {archive_names}"
            )

    if archive_files:
        reset_directory(extract_dir)

        for archive_file in archive_files:
            if zipfile.is_zipfile(archive_file):
                with zipfile.ZipFile(archive_file) as archive:
                    archive.extractall(extract_dir)
            elif is_7z_file(archive_file):
                extract_7z_archive(archive_file, extract_dir)

        return extract_dir, len(archive_files)

    if any(raw_dir.rglob("*.pb")):
        return raw_dir, 0

    raise FileNotFoundError(f"No protobuf or zip files found in raw folder: {raw_dir}")


def archive_matches_operator(path: Path, operator: str) -> bool:
    return operator.lower() in path.name.lower()


def prepare_static_input_from_raw(raw_dir: Path, extract_dir: Path, operator: str) -> tuple[Path, int]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw GTFS Static folder not found: {raw_dir}")

    all_archive_files = [
        path
        for path in sorted(raw_dir.iterdir())
        if path.is_file() and (zipfile.is_zipfile(path) or is_7z_file(path))
    ]
    archive_files = [
        path for path in all_archive_files if archive_matches_operator(path, operator)
    ]

    if all_archive_files and not archive_files:
        if len(all_archive_files) == 1:
            archive_files = all_archive_files
        else:
            archive_names = ", ".join(path.name for path in all_archive_files)
            raise FileNotFoundError(
                f"No GTFS Static archive for operator '{operator}' found in {raw_dir}. "
                f"Available archives: {archive_names}"
            )

    if archive_files:
        reset_directory(extract_dir)

        for archive_file in archive_files:
            if zipfile.is_zipfile(archive_file):
                with zipfile.ZipFile(archive_file) as archive:
                    archive.extractall(extract_dir)
            elif is_7z_file(archive_file):
                extract_7z_archive(archive_file, extract_dir)

        return extract_dir, len(archive_files)

    if any(raw_dir.glob("*.txt")):
        return raw_dir, 0

    raise FileNotFoundError(f"No GTFS Static archive or txt files found in raw folder: {raw_dir}")


def build_trip_update_csv(input_dir: Path, output_dir: Path) -> Path | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "all_trip_updates.csv"
    if output_file.exists():
        output_file.unlink()

    seen_keys = set()
    written_rows = 0
    pb_files = sorted(input_dir.rglob("*.pb"))

    for file_index, pb_path in enumerate(pb_files, start=1):
        feed = gtfs_realtime_pb2.FeedMessage()

        try:
            with open(pb_path, "rb") as f:
                feed.ParseFromString(f.read())
        except Exception as exc:
            print(f"Skipping {pb_path}: {exc}", file=sys.stderr)
            continue

        rows = []
        for entity in feed.entity:
            if not entity.HasField("trip_update"):
                continue

            trip_update = entity.trip_update
            trip_id = trip_update.trip.trip_id
            vehicle_id = trip_update.vehicle.id if trip_update.HasField("vehicle") else None
            feed_timestamp = trip_update.timestamp

            for stop_time_update in trip_update.stop_time_update:
                rows.append(
                    {
                        "source_file": pb_path.name,
                        "trip_id": trip_id,
                        "vehicle_id": vehicle_id,
                        "stop_id": stop_time_update.stop_id,
                        "stop_sequence": stop_time_update.stop_sequence,
                        "arrival_time": (
                            stop_time_update.arrival.time
                            if stop_time_update.HasField("arrival")
                            else None
                        ),
                        "departure_time": (
                            stop_time_update.departure.time
                            if stop_time_update.HasField("departure")
                            else None
                        ),
                        "delay": (
                            stop_time_update.arrival.delay
                            if stop_time_update.HasField("arrival")
                            else None
                        ),
                        "feed_timestamp": feed_timestamp,
                    }
                )

        if not rows:
            continue

        df = pd.DataFrame(rows)
        df["arrival_time"] = pd.to_datetime(df["arrival_time"], unit="s", errors="coerce")
        df["departure_time"] = pd.to_datetime(df["departure_time"], unit="s", errors="coerce")
        df["feed_timestamp"] = pd.to_datetime(df["feed_timestamp"], unit="s", errors="coerce")
        row_keys = pd.util.hash_pandas_object(
            df[["trip_id", "stop_id", "stop_sequence"]],
            index=False,
        )
        keep_mask = []
        for row_key in row_keys:
            row_key = int(row_key)
            if row_key in seen_keys:
                keep_mask.append(False)
            else:
                seen_keys.add(row_key)
                keep_mask.append(True)

        df = df.loc[keep_mask]
        if df.empty:
            continue

        df.to_csv(output_file, index=False, mode="a", header=written_rows == 0)
        written_rows += len(df)

        if file_index % 1000 == 0:
            print(
                f"Processed {file_index}/{len(pb_files)} TripUpdates PB files; "
                f"{written_rows} unique rows written."
            )

    if written_rows == 0:
        return None

    return output_file


def collapse_download_root_segments(path: Path) -> Path:
    parts = path.parts
    download_root_indexes = [
        index for index, part in enumerate(parts) if part == DOWNLOAD_ROOT.name
    ]
    if not download_root_indexes:
        return path

    tail_parts = parts[download_root_indexes[-1] + 1:]
    if not tail_parts:
        return DOWNLOAD_ROOT
    return DOWNLOAD_ROOT.joinpath(*tail_parts)


def resolve_pipeline_folder(value: str) -> Path:
    run_dir = collapse_download_root_segments(Path(value).expanduser())

    candidates = []

    if run_dir.is_absolute():
        candidates.append(run_dir)
        candidates.append(DOWNLOAD_ROOT / run_dir.name)
    elif run_dir.parts and run_dir.parts[0] == DOWNLOAD_ROOT.name:
        candidates.extend(
            [
                PROJECT_DIR / run_dir,
                run_dir,
                DOWNLOAD_ROOT / Path(*run_dir.parts[1:]),
            ]
        )
    else:
        candidates.extend(
            [
                DOWNLOAD_ROOT / run_dir,
                PROJECT_DIR / run_dir,
                run_dir,
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[1] if run_dir.is_absolute() and len(candidates) > 1 else candidates[0]


def get_run_dir(config: dict, operator: str, date: str) -> Path:
    explicit_run_dir = str(
        config.get("run_dir") or config.get("preprocess_input_dir") or ""
    ).strip()
    if explicit_run_dir:
        run_dir = resolve_pipeline_folder(explicit_run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    run_name = safe_name(
        str(config.get("output_name") or f"{operator}_{date}_{time.strftime('%Y%m%d_%H%M%S')}")
    )
    run_dir = DOWNLOAD_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def update_job(job_id: str, message: str | None = None, progress: int | None = None, **extra):
    with JOBS_LOCK:
        job = JOBS[job_id]
        if message is not None:
            job["messages"].append(message)
            job["messages"] = job["messages"][-250:]
        if progress is not None:
            job["progress"] = progress
        job.update(extra)


def is_cancel_requested(job_id: str | None) -> bool:
    if job_id is None:
        return False
    with JOBS_LOCK:
        return bool(JOBS[job_id].get("cancel_requested"))


def check_cancelled(job_id: str | None):
    if is_cancel_requested(job_id):
        raise InterruptedError("Job aborted by user.")


def event_activity_type(event_type: str | None) -> str:
    event_type = str(event_type or "unknown")
    if event_type.startswith("arrive_stop"):
        return "arrive_stop"
    if event_type.startswith("departure_stop"):
        return "departure_stop"
    return event_type


def summarize_ocel_event_log(ocel_file: Path) -> dict:
    with ocel_file.open(encoding="utf-8") as f:
        ocel = json.load(f)

    events = ocel.get("events", [])
    objects = ocel.get("objects", [])
    activity_type_counts = Counter(
        event_activity_type(event.get("type"))
        for event in events
        if isinstance(event, dict)
    )
    event_type_counts = Counter(
        str(event.get("type") or "unknown")
        for event in events
        if isinstance(event, dict)
    )
    object_type_counts = Counter(
        str(obj.get("type") or "unknown")
        for obj in objects
        if isinstance(obj, dict)
    )
    relationship_count = sum(
        len(event.get("relationships", []))
        for event in events
        if isinstance(event, dict)
    )

    return {
        "file": str(ocel_file),
        "total_events": len(events),
        "total_objects": len(objects),
        "total_event_types": len(event_type_counts),
        "total_activity_types": len(activity_type_counts),
        "total_object_types": len(object_type_counts),
        "total_event_object_relationships": relationship_count,
        "activity_type_counts": dict(activity_type_counts.most_common()),
        "event_type_counts": dict(event_type_counts.most_common()),
        "object_type_counts": dict(object_type_counts.most_common()),
    }


def legacy_event_log_output_file(variant: str) -> Path:
    if variant == "plus":
        return PROJECT_DIR / "ocel_koda_2026_05_04_w_stop_names_no_layover_shift_position_work.json"
    return PROJECT_DIR / "ocel_koda_2026_05_04_w_stop_names_no_layover.json"


def event_log_output_file(variant: str, operator: str, date: str) -> Path:
    builder_name = "ocel_koda+" if variant == "plus" else "ocel_koda"
    operator_part = safe_name(operator.lower())
    date_part = safe_name(date)
    return PROJECT_DIR / f"{builder_name}_{operator_part}_{date_part}.json"


def read_json_file(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def table_summary(name: str, path: Path | None, row_count: int | None = None) -> dict | None:
    if path is None or not path.exists():
        return None
    return {
        "name": name,
        "path": str(path),
        "bytes": path.stat().st_size,
        "rows": row_count,
    }


def format_bytes(size: int | None) -> str:
    if size is None:
        return "-"
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    if size < 1024 * 1024 * 1024:
        return f"{size / 1024 / 1024:.1f} MB"
    return f"{size / 1024 / 1024 / 1024:.1f} GB"


def format_table_summaries(table_summaries: list[dict]) -> list[str]:
    if not table_summaries:
        return ["Table sizes:", "- no table summaries available"]

    lines = ["Table sizes:"]
    for table in table_summaries:
        row_count = table.get("rows")
        rows_label = f", {row_count:,} rows" if isinstance(row_count, int) else ""
        lines.append(
            f"- {table['name']}: {format_bytes(table.get('bytes'))}{rows_label}"
        )
    return lines


def format_occupancy_spread(info: dict | None) -> list[str]:
    if not info:
        return ["Occupancy spread:", "- no occupancy spread summary available"]

    spread_counts = info.get("occupancy_spread_counts") or {}
    lines = [
        "Occupancy spread:",
        f"- Trips with occupancy: {info.get('trips_with_occupancy', '-')}",
        f"- Trips with spread 0: {info.get('constant_occupancy_trips', '-')}",
        f"- Trips with variable occupancy: {info.get('variable_occupancy_trips', '-')}",
        "- Distribution:",
    ]
    lines.extend(
        f"  - spread {spread}: {count} trip(s)"
        for spread, count in spread_counts.items()
    )
    return lines


def list_pipeline_folders() -> list[dict]:
    if not DOWNLOAD_ROOT.exists():
        return []

    folders = []
    for path in sorted(
        (item for item in DOWNLOAD_ROOT.iterdir() if item.is_dir()),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    ):
        if path.name == DOWNLOAD_ROOT.name:
            continue

        raw_dir = path / "raw"
        csv_dir = path / "csv"
        metadata = infer_pipeline_metadata(path)
        folders.append(
            {
                "name": path.name,
                "path": str(path),
                "operator": metadata.get("operator"),
                "date": metadata.get("date"),
                "modified_at": path.stat().st_mtime,
                "has_raw": raw_dir.exists(),
                "has_trip_updates_raw": (raw_dir / "trip_updates").exists(),
                "has_vehicle_positions_raw": (raw_dir / "vehicle_positions").exists(),
                "has_gtfs_static_raw": (raw_dir / "gtfs_static").exists(),
                "has_csv": csv_dir.exists(),
                "has_trip_updates_csv": (
                    csv_dir / "trip_updates" / "all_trip_updates.csv"
                ).exists(),
                "has_complete_vehicle_positions": (
                    csv_dir / "vehicle_positions_complete" / "all_vehicle_positions.csv"
                ).exists(),
                "has_fastlane_vehicle_positions": (
                    csv_dir / "vehicle_positions_fastlane" / "all_vehicle_positions.csv"
                ).exists(),
                "has_ultra_fastlane": (
                    (csv_dir / "vehicle_positions_complete" / "trip_occupancy_ultra_fastlane.csv").exists()
                    or (csv_dir / "vehicle_positions_fastlane" / "trip_occupancy_ultra_fastlane.csv").exists()
                ),
            }
        )
    return folders


def infer_pipeline_metadata(path: Path) -> dict:
    patterns = [
        re.compile(r"GTFS-([A-Za-z0-9_]+)-(\d{4}-\d{2}-\d{2})", re.IGNORECASE),
        re.compile(r"([A-Za-z0-9_]+)-(?:TripUpdates|VehiclePositions)-(\d{4}-\d{2}-\d{2})", re.IGNORECASE),
    ]
    search_dirs = [
        path / "raw" / "gtfs_static",
        path / "raw" / "trip_updates",
        path / "raw" / "vehicle_positions",
    ]

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for file_path in sorted(search_dir.iterdir()):
            if not file_path.is_file():
                continue
            for pattern in patterns:
                match = pattern.search(file_path.name)
                if match:
                    return {
                        "operator": match.group(1).lower(),
                        "date": match.group(2),
                    }

    return {}


def create_zip_file(directory: Path) -> Path:
    temp_file = tempfile.NamedTemporaryFile(
        prefix=f"{safe_name(directory.name)}_",
        suffix=".zip",
        dir=directory.parent,
        delete=False,
    )
    zip_path = Path(temp_file.name)
    temp_file.close()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(directory))
    return zip_path


def download_koda_data(config: dict, job_id: str | None = None) -> dict:
    operator = safe_name(str(config.get("operator", "otraf")).lower())
    date = str(config.get("date", "")).strip()
    api_key = str(config.get("api_key", "")).strip()

    if not date:
        raise ValueError("Date is required.")
    if not api_key:
        raise ValueError("API key is required.")

    run_dir = get_run_dir(config, operator, date)
    raw_dir = run_dir / "raw"

    check_cancelled(job_id)
    if job_id:
        update_job(job_id, "Starting GTFS Static download...", progress=5)
    static_url = request_url(STATIC_URL, operator, date, api_key)
    static_file, static_status, static_size = download_file(
        static_url,
        raw_dir / "gtfs_static",
        "gtfs_static",
    )
    if job_id:
        update_job(
            job_id,
            f"GTFS Static downloaded: HTTP {static_status}, {static_size} bytes",
            progress=30,
        )

    check_cancelled(job_id)
    if job_id:
        update_job(job_id, "Starting TripUpdates download...", progress=35)
    trip_url = request_url(
        REALTIME_URL,
        operator,
        date,
        api_key,
        feed=REALTIME_FEEDS["trip_updates"],
    )
    trip_file, trip_status, trip_size = download_file(
        trip_url,
        raw_dir / "trip_updates",
        "trip_updates",
    )
    if job_id:
        update_job(
            job_id,
            f"TripUpdates downloaded: HTTP {trip_status}, {trip_size} bytes",
            progress=60,
        )

    check_cancelled(job_id)
    if job_id:
        update_job(job_id, "Starting VehiclePositions download...", progress=65)
    vehicle_url = request_url(
        REALTIME_URL,
        operator,
        date,
        api_key,
        feed=REALTIME_FEEDS["vehicle_positions"],
    )
    vehicle_file, vehicle_status, vehicle_size = download_file(
        vehicle_url,
        raw_dir / "vehicle_positions",
        "vehicle_positions",
    )
    if job_id:
        update_job(
            job_id,
            f"VehiclePositions downloaded: HTTP {vehicle_status}, {vehicle_size} bytes",
            progress=90,
        )

    check_cancelled(job_id)
    return {
        "run_dir": str(run_dir),
        "requests": {
            "gtfs_static": {"status": static_status, "bytes": static_size, "file": str(static_file)},
            "trip_updates": {"status": trip_status, "bytes": trip_size, "file": str(trip_file)},
            "vehicle_positions": {"status": vehicle_status, "bytes": vehicle_size, "file": str(vehicle_file)},
        },
        "raw": {
            "gtfs_static": str(raw_dir / "gtfs_static"),
            "trip_updates": str(raw_dir / "trip_updates"),
            "vehicle_positions": str(raw_dir / "vehicle_positions"),
        },
    }


def preprocess_koda_data(config: dict, job_id: str | None = None) -> dict:
    operator = safe_name(str(config.get("operator", "otraf")).lower())
    date = str(config.get("date", "")).strip()
    vehicle_position_mode = str(config.get("vehicle_position_mode", "complete"))
    vehicle_position_seconds = [
        int(second)
        for second in config.get("vehicle_position_seconds", [0, 9, 19, 29, 39, 49, 59])
    ]

    if not date:
        raise ValueError("Date is required.")
    if vehicle_position_mode not in ("complete", "fastlane"):
        raise ValueError("VehiclePosition mode must be complete or fastlane.")
    if vehicle_position_mode == "fastlane" and not vehicle_position_seconds:
        raise ValueError("Fastlane requires at least one selected second.")

    run_dir = get_run_dir(config, operator, date)
    raw_dir = run_dir / "raw"
    extracted_dir = run_dir / "extracted"
    csv_dir = run_dir / "csv"
    vehicle_csv_dir = csv_dir / f"vehicle_positions_{vehicle_position_mode}"
    static_raw_dir = raw_dir / "gtfs_static"
    trip_raw_dir = raw_dir / "trip_updates"
    vehicle_raw_dir = raw_dir / "vehicle_positions"

    check_cancelled(job_id)
    if job_id:
        update_job(job_id, "Preparing GTFS Static from raw folder...", progress=3)
    gtfs_input_dir, gtfs_archive_count = prepare_static_input_from_raw(
        static_raw_dir,
        extracted_dir / "gtfs_static",
        operator,
    )
    if job_id:
        update_job(
            job_id,
            f"GTFS Static input ready: {gtfs_archive_count} archive(s) extracted.",
            progress=8,
        )

    check_cancelled(job_id)
    if job_id:
        update_job(job_id, "Preparing TripUpdates from raw folder...", progress=5)
    trip_input_dir, trip_zip_count = prepare_realtime_input_from_raw(
        trip_raw_dir,
        extracted_dir / "trip_updates",
        operator,
    )
    if job_id:
        update_job(
            job_id,
            f"TripUpdates input ready: {trip_zip_count} archive(s) extracted.",
            progress=15,
        )

    check_cancelled(job_id)
    if job_id:
        update_job(job_id, "Preparing VehiclePositions from raw folder...", progress=20)
    vehicle_input_dir, vehicle_zip_count = prepare_realtime_input_from_raw(
        vehicle_raw_dir,
        extracted_dir / "vehicle_positions",
        operator,
    )
    if job_id:
        update_job(
            job_id,
            f"VehiclePositions input ready: {vehicle_zip_count} archive(s) extracted.",
            progress=30,
        )

    check_cancelled(job_id)
    if job_id:
        update_job(job_id, "Building TripUpdates CSV...", progress=35)
    trip_csv = build_trip_update_csv(trip_input_dir, csv_dir / "trip_updates")
    if job_id:
        update_job(job_id, "TripUpdates CSV finished.", progress=55)

    check_cancelled(job_id)
    if job_id:
        if vehicle_position_mode == "fastlane":
            seconds_label = ", ".join(f"{second:02d}Z" for second in vehicle_position_seconds)
            update_job(
                job_id,
                f"Building VehiclePositions Fastlane CSV for {seconds_label}...",
                progress=60,
            )
        else:
            update_job(
                job_id,
                "Building VehiclePositions Complete CSV from all available PB files...",
                progress=60,
            )
    vehicle_csv = build_vehicle_position_csvs(
        input_dir=vehicle_input_dir,
        output_dir=vehicle_csv_dir,
        mode=vehicle_position_mode,
        seconds=vehicle_position_seconds,
    )
    trip_occupancy_csv = (
        vehicle_csv_dir / "trip_occupancy_ultra_fastlane.csv"
        if vehicle_csv
        else None
    )
    trip_occupancy_info = read_json_file(
        vehicle_csv_dir / "trip_occupancy_ultra_fastlane_info.json"
    )
    vehicle_build_info = read_json_file(vehicle_csv_dir / "build_info.json")
    table_summaries = [
        table_summary("TripUpdates", trip_csv),
        table_summary(
            "VehiclePositions",
            vehicle_csv,
            vehicle_build_info.get("row_count") if vehicle_build_info else None,
        ),
        table_summary(
            "Ultra Fastlane occupancy",
            trip_occupancy_csv,
            (
                trip_occupancy_info.get("constant_occupancy_trips")
                if trip_occupancy_info
                else None
            ),
        ),
    ]
    table_summaries = [summary for summary in table_summaries if summary is not None]
    if job_id:
        update_job(job_id, "VehiclePositions CSV files finished.", progress=95)
        update_job(
            job_id,
            "\n".join(
                [
                    "Preprocessing summary:",
                    *format_table_summaries(table_summaries),
                    *format_occupancy_spread(trip_occupancy_info),
                ]
            ),
            progress=96,
        )

    check_cancelled(job_id)
    return {
        "run_dir": str(run_dir),
        "inputs": {
            "gtfs_static": str(gtfs_input_dir),
            "trip_updates": str(trip_input_dir),
            "vehicle_positions": str(vehicle_input_dir),
        },
        "csv": {
            "trip_updates": str(trip_csv) if trip_csv else None,
            "vehicle_positions": [str(vehicle_csv)] if vehicle_csv else [],
            "vehicle_positions_dir": str(vehicle_csv_dir),
            "trip_occupancy_ultra_fastlane": (
                str(trip_occupancy_csv)
                if trip_occupancy_csv and trip_occupancy_csv.exists()
                else None
            ),
            "trip_occupancy_ultra_fastlane_info": trip_occupancy_info,
            "table_summaries": table_summaries,
            "vehicle_position_mode": vehicle_position_mode,
            "vehicle_position_seconds": vehicle_position_seconds,
        },
    }


def build_event_log(config: dict, job_id: str | None = None) -> dict:
    variant = str(config.get("event_builder_variant", "standard"))
    if variant == "plus":
        script = BACKEND_DIR / "KODA_Robust.py"
        label = "Event Builder+"
    else:
        script = BACKEND_DIR / "KODA.py"
        label = "Event Builder"

    if not script.exists():
        raise FileNotFoundError(f"Event builder script not found: {script}")

    operator = safe_name(str(config.get("operator", "otraf")).lower())
    date = str(config.get("date", "")).strip()
    run_dir = get_run_dir(config, operator, date)
    vehicle_position_source = str(config.get("event_vehicle_position_source", "ultra_fastlane"))
    if vehicle_position_source not in ("complete", "fastlane", "ultra_fastlane", "legacy"):
        raise ValueError(
            "VehiclePosition source must be complete, fastlane, ultra_fastlane, or legacy."
        )
    if vehicle_position_source == "ultra_fastlane":
        vehicle_position_mode = str(config.get("vehicle_position_mode", "complete"))
        if vehicle_position_mode not in ("complete", "fastlane"):
            vehicle_position_mode = "complete"
        candidate_modes = [vehicle_position_mode] + [
            mode for mode in ("complete", "fastlane") if mode != vehicle_position_mode
        ]
        vehicle_position_dir = run_dir / "csv" / f"vehicle_positions_{candidate_modes[0]}"
        for candidate_mode in candidate_modes:
            candidate_dir = run_dir / "csv" / f"vehicle_positions_{candidate_mode}"
            if (candidate_dir / "trip_occupancy_ultra_fastlane.csv").exists():
                vehicle_position_dir = candidate_dir
                break
    else:
        vehicle_position_dir = run_dir / "csv" / f"vehicle_positions_{vehicle_position_source}"
    if vehicle_position_source == "legacy":
        vehicle_position_dir = run_dir / "csv" / "vehicle_positions"

    vehicle_position_csv = vehicle_position_dir / "all_vehicle_positions.csv"
    trip_occupancy_csv = vehicle_position_dir / "trip_occupancy_ultra_fastlane.csv"
    required_vehicle_file = (
        trip_occupancy_csv
        if vehicle_position_source == "ultra_fastlane"
        else vehicle_position_csv
    )
    if not required_vehicle_file.exists():
        legacy_dir = run_dir / "csv" / "vehicle_positions"
        legacy_csv = legacy_dir / "all_vehicle_positions.csv"
        if vehicle_position_source not in ("legacy", "ultra_fastlane") and legacy_csv.exists():
            vehicle_position_source = "legacy"
            vehicle_position_dir = legacy_dir
        else:
            raise FileNotFoundError(
                f"VehiclePositions CSV not found for source '{vehicle_position_source}': "
                f"{required_vehicle_file}. Run preprocessing for that source first."
            )

    env = os.environ.copy()
    env["KODA_PIPELINE_DIR"] = str(run_dir)
    env["KODA_SERVICE_DATE"] = date
    env["KODA_VEHICLE_POSITIONS_INPUT_DIR"] = str(vehicle_position_dir)
    if vehicle_position_source == "ultra_fastlane":
        env["KODA_OCCUPANCY_MODE"] = "trip_constant"
        env["KODA_TRIP_OCCUPANCY_INPUT_FILE"] = str(trip_occupancy_csv)

    if job_id:
        source_label = (
            f"Ultra Fastlane ({vehicle_position_dir.name})"
            if vehicle_position_source == "ultra_fastlane"
            else vehicle_position_source
        )
        update_job(
            job_id,
            f"Starting {label} with {script.name} using VehiclePositions source: {source_label}",
            progress=5,
        )

    process = subprocess.Popen(
        [sys.executable, str(script)],
        cwd=PROJECT_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    if job_id:
        with JOBS_LOCK:
            PROCESSES[job_id] = process
            JOBS[job_id]["process_pid"] = process.pid

    try:
        assert process.stdout is not None
        for line in process.stdout:
            clean_line = line.rstrip()
            if clean_line and job_id:
                update_job(job_id, clean_line, progress=50)

            if is_cancel_requested(job_id):
                process.terminate()
                raise InterruptedError("Job aborted by user.")

        return_code = process.wait()
        if return_code != 0:
            if return_code == -9:
                raise RuntimeError(
                    f"{script.name} was killed with exit code -9 while using "
                    f"VehiclePositions source '{vehicle_position_source}'. "
                    "This usually means the process ran out of memory."
                )
            raise RuntimeError(f"{script.name} failed with exit code {return_code}.")
    finally:
        if job_id:
            with JOBS_LOCK:
                PROCESSES.pop(job_id, None)

    if job_id:
        update_job(job_id, f"{label} finished.", progress=95)

    legacy_output = legacy_event_log_output_file(variant)
    primary_output = event_log_output_file(variant, operator, date)
    if legacy_output.exists() and legacy_output != primary_output:
        shutil.move(str(legacy_output), primary_output)

    standard_output = event_log_output_file("standard", operator, date)
    robust_output = event_log_output_file("plus", operator, date)
    expected_outputs = [primary_output]
    for output in (standard_output, robust_output):
        if output != primary_output:
            expected_outputs.append(output)

    existing_outputs = [path for path in expected_outputs if path.exists()]
    summary = None
    if primary_output.exists():
        if job_id:
            update_job(job_id, "Building event log summary...", progress=98)
        summary = summarize_ocel_event_log(primary_output)

    return {
        "variant": variant,
        "vehicle_position_source": vehicle_position_source,
        "vehicle_positions_input": str(vehicle_position_dir),
        "script": str(script),
        "outputs": [str(path) for path in existing_outputs],
        "summary": summary,
    }


def start_job(kind: str, config: dict) -> str:
    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "kind": kind,
            "status": "running",
            "progress": 0,
            "messages": ["Job started."],
            "result": None,
            "error": None,
            "cancel_requested": False,
        }

    def runner():
        try:
            if kind == "download":
                result = download_koda_data(config, job_id=job_id)
                update_job(job_id, "Download completed.", progress=100, status="done", result=result)
            elif kind == "preprocess":
                result = preprocess_koda_data(config, job_id=job_id)
                update_job(job_id, "Preprocessing completed.", progress=100, status="done", result=result)
            elif kind == "eventlog":
                result = build_event_log(config, job_id=job_id)
                update_job(job_id, "Event log build completed.", progress=100, status="done", result=result)
            else:
                raise ValueError(f"Unknown job kind: {kind}")
        except InterruptedError as exc:
            update_job(job_id, str(exc), status="aborted", error=str(exc))
        except Exception as exc:
            update_job(job_id, str(exc), status="error", error=str(exc))

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    return job_id


class KodaRequestHandler(BaseHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_GET(self):
        if self.path in ("/", "/koda_demo_frontend.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_FILE.read_bytes())
            return

        parsed_path = urlparse(self.path)

        if parsed_path.path == "/api/health":
            self.write_json({"ok": True, "download_root": str(DOWNLOAD_ROOT)})
            return

        if parsed_path.path == "/api/pipeline-folders":
            self.write_json(
                {
                    "ok": True,
                    "download_root": str(DOWNLOAD_ROOT),
                    "folders": list_pipeline_folders(),
                }
            )
            return

        if parsed_path.path == "/api/download-output":
            query = parse_qs(parsed_path.query)
            output_type = query.get("type", ["ocel"])[0]

            if output_type == "ocel":
                variant = query.get("variant", ["standard"])[0]
                operator = safe_name(str(query.get("operator", ["otraf"])[0]).lower())
                date = str(query.get("date", [""])[0]).strip()
                output_file = event_log_output_file(variant, operator, date)
                if not output_file.exists():
                    self.send_error(404, f"OCEL output not found: {output_file}")
                    return
                self.write_file(output_file)
                return

            if output_type == "csv":
                config = {key: values[0] for key, values in query.items()}
                operator = safe_name(str(config.get("operator", "otraf")).lower())
                date = str(config.get("date", "")).strip()
                csv_dir = get_run_dir(config, operator, date) / "csv"
                if not csv_dir.exists():
                    self.send_error(404, f"CSV folder not found: {csv_dir}")
                    return
                filename = f"{safe_name(csv_dir.parent.name)}_csv.zip"
                zip_path = create_zip_file(csv_dir)
                try:
                    self.write_file(zip_path, download_name=filename, content_type="application/zip")
                finally:
                    zip_path.unlink(missing_ok=True)
                return

            self.send_error(400, "Unknown download type.")
            return

        if parsed_path.path == "/api/job-status":
            query = parse_qs(parsed_path.query)
            job_id = query.get("id", [""])[0]
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                payload = dict(job) if job else None
            if payload is None:
                self.write_json({"ok": False, "error": "Job not found."}, status=404)
            else:
                self.write_json({"ok": True, "job": payload})
            return

        self.send_error(404)

    def do_POST(self):
        if self.path not in (
            "/api/download-data",
            "/api/preprocess-data",
            "/api/build-event-log",
            "/api/cancel-job",
        ):
            self.send_error(404)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = self.rfile.read(length)
            config = json.loads(payload.decode("utf-8"))
            if self.path == "/api/cancel-job":
                job_id = str(config.get("job_id", ""))
                with JOBS_LOCK:
                    job = JOBS.get(job_id)
                    if job is None:
                        self.write_json({"ok": False, "error": "Job not found."}, status=404)
                        return
                    job["cancel_requested"] = True
                    job["messages"].append("Abort requested.")
                    if job["status"] == "running":
                        job["status"] = "aborting"
                    process = PROCESSES.get(job_id)
                    if process is not None and process.poll() is None:
                        process.terminate()
                self.write_json({"ok": True})
                return
            if self.path == "/api/download-data":
                job_id = start_job("download", config)
            elif self.path == "/api/preprocess-data":
                job_id = start_job("preprocess", config)
            else:
                job_id = start_job("eventlog", config)
            self.write_json({"ok": True, "job_id": job_id})
        except Exception as exc:
            self.write_json({"ok": False, "error": str(exc)}, status=500)

    def write_json(self, payload: dict, status: int = 200):
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def write_file(
        self,
        path: Path,
        download_name: str | None = None,
        content_type: str | None = None,
    ):
        filename = download_name or path.name
        content_type = content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(path.stat().st_size))
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        with open(path, "rb") as f:
            shutil.copyfileobj(f, self.wfile, length=1024 * 1024)

def main():
    server = ThreadingHTTPServer((SERVER_HOST, SERVER_PORT), KodaRequestHandler)
    print(f"KODA pipeline server running at http://{SERVER_HOST}:{SERVER_PORT}/")
    print(f"Downloads will be written to {DOWNLOAD_ROOT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
