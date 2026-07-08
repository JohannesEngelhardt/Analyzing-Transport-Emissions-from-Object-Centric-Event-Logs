import argparse
from pathlib import Path

import pandas as pd


def active_service_ids_for_date(df_calendar, df_calendar_dates, service_date):
    service_date = pd.to_datetime(service_date, errors="coerce")
    if pd.isna(service_date):
        raise ValueError("service_date must be parseable, for example 2026-05-04")
    service_date = service_date.normalize()

    weekday_column = service_date.day_name().lower()
    active_services = set()

    if not df_calendar.empty and weekday_column in df_calendar.columns:
        calendar = df_calendar.copy()
        calendar["start_date"] = pd.to_datetime(
            calendar["start_date"].astype(str),
            format="%Y%m%d",
            errors="coerce",
        )
        calendar["end_date"] = pd.to_datetime(
            calendar["end_date"].astype(str),
            format="%Y%m%d",
            errors="coerce",
        )
        in_date_range = calendar["start_date"].le(service_date) & calendar["end_date"].ge(service_date)
        weekday_active = pd.to_numeric(calendar[weekday_column], errors="coerce").fillna(0).eq(1)
        active_services.update(
            calendar.loc[in_date_range & weekday_active, "service_id"]
            .dropna()
            .astype(str)
        )

    if not df_calendar_dates.empty:
        calendar_dates = df_calendar_dates.copy()
        calendar_dates["date"] = pd.to_datetime(
            calendar_dates["date"].astype(str),
            format="%Y%m%d",
            errors="coerce",
        )
        day_exceptions = calendar_dates[calendar_dates["date"].eq(service_date)]
        additions = set(
            day_exceptions.loc[day_exceptions["exception_type"].eq(1), "service_id"]
            .dropna()
            .astype(str)
        )
        removals = set(
            day_exceptions.loc[day_exceptions["exception_type"].eq(2), "service_id"]
            .dropna()
            .astype(str)
        )
        active_services.update(additions)
        active_services.difference_update(removals)

    return active_services


def read_trip_ids(path, column="trip_id"):
    if not path.exists():
        raise FileNotFoundError(path)
    return set(
        pd.read_csv(path, usecols=[column], dtype={column: "string"})[column]
        .dropna()
        .astype(str)
    )


def default_overview_file(project_dir, variant):
    if variant == "plus":
        return project_dir / "aux_event_log_overview_kodak_2026_05_04_new_e2o_no_layover_shift_position_work.csv"
    return project_dir / "aux_event_log_overview_kodak_2026_05_04_new_e2o_no_layover.csv"


def evaluate(pipeline_dir, service_date, overview_file):
    gtfs_dir = pipeline_dir / "extracted" / "gtfs_static"
    trip_updates_file = pipeline_dir / "csv" / "trip_updates" / "all_trip_updates.csv"

    df_calendar = pd.read_csv(gtfs_dir / "calendar.txt", dtype={"service_id": "string"})
    calendar_dates_path = gtfs_dir / "calendar_dates.txt"
    df_calendar_dates = (
        pd.read_csv(calendar_dates_path, dtype={"service_id": "string"})
        if calendar_dates_path.exists()
        else pd.DataFrame(columns=["service_id", "date", "exception_type"])
    )
    df_trips = pd.read_csv(
        gtfs_dir / "trips.txt",
        usecols=["service_id", "trip_id"],
        dtype={"service_id": "string", "trip_id": "string"},
    )

    active_service_ids = active_service_ids_for_date(df_calendar, df_calendar_dates, service_date)
    day_trip_ids = set(
        df_trips.loc[
            df_trips["service_id"].astype(str).isin(active_service_ids),
            "trip_id",
        ].dropna().astype(str)
    )
    rt_trip_ids = read_trip_ids(trip_updates_file)
    day_trips_with_rt = day_trip_ids & rt_trip_ids
    day_trips_without_rt = day_trip_ids - rt_trip_ids
    rt_trips_not_in_calendar_day = rt_trip_ids - day_trip_ids

    overview = pd.read_csv(
        overview_file,
        usecols=["trip_id", "trip_id_org"],
        dtype={"trip_id": "string", "trip_id_org": "string"},
    ).dropna(subset=["trip_id", "trip_id_org"])

    trip_mapping = overview[["trip_id_org", "trip_id"]].drop_duplicates()
    parts_by_original_trip = trip_mapping.groupby("trip_id_org")["trip_id"].nunique()
    original_trips_with_events = int(parts_by_original_trip.shape[0])
    final_trip_objects = int(trip_mapping["trip_id"].nunique())
    split_original_trips = int(parts_by_original_trip.gt(1).sum())
    additional_split_parts = int((parts_by_original_trip - 1).clip(lower=0).sum())
    final_minus_additional_split_parts = final_trip_objects - additional_split_parts

    original_trip_ids_with_events = set(parts_by_original_trip.index.astype(str))
    expected_input_trips = len(day_trip_ids)
    actual_input_trips_from_log = final_minus_additional_split_parts

    return {
        "service_date": str(pd.to_datetime(service_date).date()),
        "active_services": len(active_service_ids),
        "day_trips": len(day_trip_ids),
        "rt_trips": len(rt_trip_ids),
        "day_trips_with_rt": len(day_trips_with_rt),
        "day_trips_without_rt": len(day_trips_without_rt),
        "rt_trips_not_in_calendar_day": len(rt_trips_not_in_calendar_day),
        "expected_input_trips": expected_input_trips,
        "actual_input_trips_from_log": actual_input_trips_from_log,
        "original_trips_with_events": original_trips_with_events,
        "final_trip_objects": final_trip_objects,
        "split_original_trips": split_original_trips,
        "additional_split_parts": additional_split_parts,
        "final_minus_additional_split_parts": final_minus_additional_split_parts,
        "split_balance_ok": final_minus_additional_split_parts == original_trips_with_events,
        "expected_vs_actual_ok": expected_input_trips == actual_input_trips_from_log,
        "event_original_trips_not_in_day_trips": len(original_trip_ids_with_events - day_trip_ids),
        "day_trips_without_events": len(day_trip_ids - original_trip_ids_with_events),
        "day_trips_with_rt_without_events": len(day_trips_with_rt - original_trip_ids_with_events),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate KODA event log trip correctness.")
    parser.add_argument("--pipeline-dir", required=True, type=Path)
    parser.add_argument("--service-date", required=True)
    parser.add_argument("--overview-file", type=Path)
    parser.add_argument("--variant", choices=["standard", "plus"], default="plus")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parents[2]
    overview_file = args.overview_file or default_overview_file(project_dir, args.variant)
    result = evaluate(args.pipeline_dir.expanduser(), args.service_date, overview_file.expanduser())

    print("Event log correctness evaluation")
    print(f"Pipeline folder: {args.pipeline_dir.expanduser()}")
    print(f"Overview file: {overview_file.expanduser()}")
    print("")
    print("Trip day basis:")
    for key in (
        "service_date",
        "active_services",
        "day_trips",
        "rt_trips",
        "day_trips_with_rt",
        "day_trips_without_rt",
        "rt_trips_not_in_calendar_day",
    ):
        print(f"- {key}: {result[key]}")

    print("")
    print("Soll/Ist trip check:")
    print(f"- soll_input_trips: {result['expected_input_trips']}")
    print(f"- ist_input_trips_from_log: {result['actual_input_trips_from_log']}")
    print(f"- soll_ist_ok: {result['expected_vs_actual_ok']}")
    print("")
    print("Split calculation:")
    for key in (
        "original_trips_with_events",
        "final_trip_objects",
        "split_original_trips",
        "additional_split_parts",
        "final_minus_additional_split_parts",
        "split_balance_ok",
    ):
        print(f"- {key}: {result[key]}")

    print("")
    print("Coverage diagnostics:")
    for key in (
        "event_original_trips_not_in_day_trips",
        "day_trips_without_events",
        "day_trips_with_rt_without_events",
    ):
        print(f"- {key}: {result[key]}")

    if result["split_balance_ok"]:
        print("")
        print("OK: final_trip_objects - additional_split_parts equals original_trips_with_events.")
    else:
        print("")
        print("NOT OK: split balance does not match.")

    if result["expected_vs_actual_ok"]:
        print("OK: soll_input_trips equals ist_input_trips_from_log.")
    else:
        print("NOT OK: soll_input_trips differs from ist_input_trips_from_log.")


if __name__ == "__main__":
    main()
