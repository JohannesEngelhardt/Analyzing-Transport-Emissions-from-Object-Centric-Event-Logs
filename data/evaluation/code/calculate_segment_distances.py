import argparse
from pathlib import Path
import json
import time
from math import atan2, cos, radians, sin, sqrt
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd


DEFAULT_INPUT = "segment_overview_kodak_2025_11_04.csv"


def haversine_m(lat1, lon1, lat2, lon2):
    earth_radius_m = 6_371_000
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = (
        sin(dlat / 2) ** 2
        + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    )
    return earth_radius_m * 2 * atan2(sqrt(a), sqrt(1 - a))


def osrm_route_distance_m(
    lat1,
    lon1,
    lat2,
    lon2,
    base_url,
    profile,
    timeout_seconds,
):
    coordinates = f"{lon1},{lat1};{lon2},{lat2}"
    query = urlencode({"overview": "false", "alternatives": "false", "steps": "false"})
    url = f"{base_url.rstrip('/')}/route/v1/{profile}/{coordinates}?{query}"

    with urlopen(url, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))

    if payload.get("code") != "Ok" or not payload.get("routes"):
        message = payload.get("message") or payload.get("code") or "no route returned"
        raise ValueError(message)

    return float(payload["routes"][0]["distance"])


def default_output_for(input_path):
    input_path = Path(input_path)
    output_name = input_path.name.replace("segment_overview", "segment_distances", 1)
    if output_name == input_path.name:
        output_name = f"{input_path.stem}_distances{input_path.suffix}"
    return input_path.with_name(output_name)


def calculate_distances(args, input_path=None, output_path=None):
    input_path = input_path or args.input
    output_path = output_path or args.output or default_output_for(input_path)

    segments = pd.read_csv(
        input_path,
        dtype={
            "segment_id": "string",
            "segment_departure_stop_id": "string",
            "segment_arrive_stop_id": "string",
        },
    )

    coordinate_columns = [
        "segment_departure_stop_lat",
        "segment_departure_stop_lon",
        "segment_arrive_stop_lat",
        "segment_arrive_stop_lon",
    ]
    for column in coordinate_columns:
        segments[column] = pd.to_numeric(segments[column], errors="coerce")

    segments["air_distance_m"] = segments.apply(
        lambda row: haversine_m(
            row["segment_departure_stop_lat"],
            row["segment_departure_stop_lon"],
            row["segment_arrive_stop_lat"],
            row["segment_arrive_stop_lon"],
        ),
        axis=1,
    )

    if args.road_distance:
        road_distances = []
        road_statuses = []
        total = len(segments)

        for index, row in segments.iterrows():
            try:
                road_distance_m = osrm_route_distance_m(
                    row["segment_departure_stop_lat"],
                    row["segment_departure_stop_lon"],
                    row["segment_arrive_stop_lat"],
                    row["segment_arrive_stop_lon"],
                    args.osrm_base_url,
                    args.profile,
                    args.timeout_seconds,
                )
                road_distances.append(road_distance_m)
                road_statuses.append("ok")
            except (HTTPError, URLError, TimeoutError, ValueError) as exc:
                road_distances.append(pd.NA)
                road_statuses.append(f"error: {exc}")

            if args.sleep_seconds > 0 and index + 1 < total:
                time.sleep(args.sleep_seconds)

            if args.progress_every and (index + 1) % args.progress_every == 0:
                print(f"processed {index + 1}/{total} segments")

        segments["road_distance_m"] = road_distances
        segments["road_distance_status"] = road_statuses
    else:
        if "road_distance_m" not in segments.columns:
            segments["road_distance_m"] = pd.NA
        if "road_distance_status" not in segments.columns:
            segments["road_distance_status"] = "not_requested"

    segments.to_csv(output_path, index=False)
    return segments


def iter_input_files(input_glob):
    paths = sorted(Path().glob(input_glob))
    if not paths:
        raise FileNotFoundError(f"No files matched --input-glob {input_glob!r}")
    return paths


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Berechnet Luftlinie und optional OSRM-Strassendistanz pro Segment."
        )
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument(
        "--output",
        default=None,
        help="Output-Datei. Wenn nicht gesetzt: segment_overview... wird zu segment_distances...",
    )
    parser.add_argument(
        "--input-glob",
        help="Berechnet mehrere Dateien, z.B. 'segment_overview_kodak_*.csv'.",
    )
    parser.add_argument(
        "--road-distance",
        action="store_true",
        help="Berechnet zusaetzlich die gefahrene Strassendistanz per OSRM.",
    )
    parser.add_argument(
        "--osrm-base-url",
        default="https://router.project-osrm.org",
        help="OSRM Basis-URL, z.B. http://localhost:5000 fuer lokalen OSRM.",
    )
    parser.add_argument("--profile", default="driving")
    parser.add_argument("--timeout-seconds", type=float, default=20)
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Pause zwischen OSRM-Anfragen, hilfreich fuer Public APIs.",
    )
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.input_glob:
        for input_path in iter_input_files(args.input_glob):
            output_path = default_output_for(input_path)
            print(f"processing {input_path} -> {output_path}")
            result = calculate_distances(args, input_path=input_path, output_path=output_path)
            print(f"written {len(result)} segment distances to {output_path}")
    else:
        output_path = args.output or default_output_for(args.input)
        result = calculate_distances(args, output_path=output_path)
        print(f"written {len(result)} segment distances to {output_path}")
