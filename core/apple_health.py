"""Apple Health import, from the Health app's own export file.

There is no API. The only route is the export the Health app produces on the
phone (Health, then your profile, then Export All Health Data), which lands as
`export.zip` containing `export.xml`.

Two things shape this module:

  * That file is big — hundreds of megabytes for a few years of data, because
    every step sample is its own element. It is streamed with iterparse and each
    element is cleared after use; loading it into a tree would exhaust memory.

  * Most of it is already in Garmin. A Garmin watch syncs into Apple Health, so
    the same run exists in both. Anything imported has to be deduplicated against
    what is already stored, or every Garmin run gets counted twice.

What Apple genuinely adds is phone-measured step counts, which include walking
done without the watch, and history from before the watch was bought.
"""

from __future__ import annotations

import logging
import zipfile
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterator
from xml.etree import ElementTree as ET

log = logging.getLogger("aerobic_engine.apple")

# Apple's workout type names mapped onto this app's sports. Anything not listed
# is skipped rather than guessed at: a mis-mapped strength session would corrupt
# the load maths, and an unrecognised type is more likely to be a walk or a yoga
# class than a training session.
WORKOUT_SPORTS: dict[str, str] = {
    "HKWorkoutActivityTypeRunning": "run",
    "HKWorkoutActivityTypeCycling": "bike",
    "HKWorkoutActivityTypeSwimming": "swim",
    "HKWorkoutActivityTypeTraditionalStrengthTraining": "strength",
    "HKWorkoutActivityTypeFunctionalStrengthTraining": "strength",
    "HKWorkoutActivityTypeWalking": "walk",
    "HKWorkoutActivityTypeHiking": "walk",
}

# Daily quantities worth keeping. Steps are the point; the rest fills gaps from
# before the watch, and is only used where Garmin has nothing for that day.
DAILY_SUMS = {"HKQuantityTypeIdentifierStepCount": "apple_steps"}
DAILY_MEANS = {
    "HKQuantityTypeIdentifierRestingHeartRate": "resting_hr",
    "HKQuantityTypeIdentifierHeartRateVariabilitySDNN": "hrv_last_night",
    "HKQuantityTypeIdentifierVO2Max": "vo2max_run",
    "HKQuantityTypeIdentifierBodyMass": "weight_kg",
}

# Garmin writes into Apple Health, so a workout from a Garmin source is by
# definition already in the database via the API — with heart-rate streams and
# zones the export does not carry. Those are skipped outright.
GARMIN_SOURCES = ("garmin",)


def _parse_dt(raw: str | None) -> datetime | None:
    """Apple stamps are "2026-08-19 06:41:00 +0530"."""
    if not raw:
        return None
    try:
        return datetime.strptime(raw.strip(), "%Y-%m-%d %H:%M:%S %z")
    except ValueError:
        try:
            return datetime.fromisoformat(raw.strip())
        except ValueError:
            return None


def _f(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def open_export(path: str | Path) -> Iterator[bytes]:
    """Yield the export XML, from either the zip or an already-extracted file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"no such Apple Health export: {p}")
    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p) as z:
            names = [n for n in z.namelist()
                     if n.endswith("export.xml") and "cda" not in n.lower()]
            if not names:
                raise ValueError(
                    f"{p.name} has no export.xml. Export again from Health, "
                    f"profile, Export All Health Data."
                )
            with z.open(names[0]) as f:
                yield from iter(lambda: f.read(1 << 20), b"")
    else:
        with p.open("rb") as f:
            yield from iter(lambda: f.read(1 << 20), b"")


def _distance_km(elem: ET.Element) -> float | None:
    """Distance lives in an attribute on older exports, a child on newer ones."""
    unit = (elem.get("totalDistanceUnit") or "").lower()
    raw = _f(elem.get("totalDistance"))
    if raw is None:
        for stat in elem.findall("WorkoutStatistics"):
            if "Distance" in (stat.get("type") or ""):
                raw = _f(stat.get("sum"))
                unit = (stat.get("unit") or "").lower()
                break
    if raw is None:
        return None
    if unit in ("mi", "mile", "miles"):
        return raw * 1.609344
    if unit in ("m", "metre", "meters"):
        return raw / 1000.0
    return raw          # km


def parse(path: str | Path) -> dict[str, Any]:
    """Stream the export into workouts and per-day quantities.

    Returns raw findings; deduplication against Garmin happens in the importer,
    which is the only place that knows what is already stored.
    """
    workouts: list[dict[str, Any]] = []
    sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    means: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    skipped_types: dict[str, int] = defaultdict(int)

    parser = ET.XMLPullParser(("end",))
    for chunk in open_export(path):
        parser.feed(chunk)
        for _, elem in parser.read_events():
            if elem.tag == "Record":
                rtype = elem.get("type") or ""
                start = _parse_dt(elem.get("startDate"))
                value = _f(elem.get("value"))
                if start is not None and value is not None:
                    day = start.date().isoformat()
                    if rtype in DAILY_SUMS:
                        sums[day][DAILY_SUMS[rtype]] += value
                    elif rtype in DAILY_MEANS:
                        means[day][DAILY_MEANS[rtype]].append(value)
                elem.clear()
            elif elem.tag == "Workout":
                atype = elem.get("workoutActivityType") or ""
                sport = WORKOUT_SPORTS.get(atype)
                start = _parse_dt(elem.get("startDate"))
                if sport is None:
                    skipped_types[atype] += 1
                elif start is not None:
                    minutes = _f(elem.get("duration")) or 0.0
                    if (elem.get("durationUnit") or "min").lower().startswith("s"):
                        minutes /= 60.0
                    km = _distance_km(elem)
                    workouts.append({
                        "sport": sport,
                        "start": start,
                        "minutes": round(minutes, 1),
                        "km": round(km, 3) if km is not None else None,
                        "source_name": elem.get("sourceName") or "",
                    })
                elem.clear()

    daily = {}
    for day in set(sums) | set(means):
        row: dict[str, Any] = {"day": day}
        row.update({k: round(v, 1) for k, v in sums.get(day, {}).items()})
        for field, values in means.get(day, {}).items():
            if values:
                row[field] = round(sum(values) / len(values), 1)
        daily[day] = row

    return {
        "workouts": sorted(workouts, key=lambda w: w["start"]),
        "daily": daily,
        "skipped_workout_types": dict(skipped_types),
    }


def is_from_garmin(workout: dict[str, Any]) -> bool:
    name = (workout.get("source_name") or "").lower()
    return any(g in name for g in GARMIN_SOURCES)


def duplicate_of(
    workout: dict[str, Any],
    existing: list[dict[str, Any]],
    minutes_tolerance: int = 20,
) -> dict[str, Any] | None:
    """Find the stored activity this Apple workout is the same session as.

    Matched on start time and sport rather than on duration: the two platforms
    round durations differently, and a watch that pauses reports a shorter moving
    time than Apple's wall clock. The window is generous because a phone and a
    watch do not agree to the second.
    """
    start = workout["start"]
    if start.tzinfo is not None:
        start = start.replace(tzinfo=None)
    window = timedelta(minutes=minutes_tolerance)
    for a in existing:
        try:
            other = datetime.fromisoformat(str(a["start_time"])[:19])
        except (ValueError, TypeError, KeyError):
            continue
        if a.get("sport") != workout["sport"]:
            continue
        if abs(other - start) <= window:
            return a
    return None
