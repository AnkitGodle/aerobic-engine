"""Read a Strava bulk export and turn it into activities this app can store.

Garmin is the sensor layer and stays that way. This exists for the history that
predates the watch: a Strava account holds months of runs that are simply gone
otherwise, and "how far have I come" is a worse question to answer with half the
record missing.

Three things this is careful about, because each of them can quietly corrupt the
data it is trying to add:

  * **The export's clock is UTC.** Strava's `Activity Date` column is UTC while
    its own titles are local, which is how a 12:40 entry comes back named
    "Evening Run". Stored raw, an evening session in India lands on the wrong day
    a third of the time and every weekly total shifts with it.
  * **The recent weeks are already in Garmin.** Anything that synced from the
    watch to Strava appears in both, and counting it twice would inflate exactly
    the totals this is meant to complete. Matching is on sport and start time,
    within a window, because the two services round durations differently.
  * **Imported rows must not reach the AI layer.** Strava's API terms forbid
    their data being used with a language model, and every planning decision here
    goes through one. So every row carries `source="strava"`, and
    `Store.activities()` filters those out unless a caller explicitly asks —
    which only the lifetime totals and the log do. Nothing here writes heart
    rate into the analysis tables either: an export rarely has it, and a
    half-populated efficiency trend is worse than a short one.

This is a bulk export the athlete downloaded themselves, not the Strava API. It
is a one-off import, run from `scripts/import_strava.py`.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Iterable, Sequence
from zoneinfo import ZoneInfo

log = logging.getLogger("aerobic_engine.strava")

# The file inside the export that holds one row per activity. Strava has moved it
# between the archive root and a subdirectory, so it is found rather than assumed.
ACTIVITIES_CSV = "activities.csv"

# Strava's activity types against this app's sports. Walks and hikes are
# deliberately absent from the default set: they are not training here, and
# adding 24 walks to a lifetime aerobic total makes the number mean less, not
# more. `--sports walk` includes them for anyone who disagrees.
SPORT_BY_TYPE: dict[str, str] = {
    "Run": "run",
    "Trail Run": "run",
    "Treadmill Run": "run",
    "Virtual Run": "run",
    "Ride": "bike",
    "Virtual Ride": "bike",
    "Mountain Bike Ride": "bike",
    "Gravel Ride": "bike",
    "Swim": "swim",
    "Open Water Swim": "swim",
    "Weight Training": "strength",
    "Workout": "strength",
    "Walk": "walk",
    "Hike": "walk",
}
DEFAULT_SPORTS = ("run", "bike", "swim", "strength")

# How close two starts have to be to be the same session. Ten minutes, measured
# rather than guessed: every genuine overlap in this athlete's export matched the
# stored Garmin activity to the second, and a wider window started merging real
# sessions — two short runs 19 minutes apart on the same evening are two runs.
DUPLICATE_WINDOW_MIN = 10

# Columns whose names repeat in the export — Strava emits "Distance" twice, once
# in kilometres for display and once in metres from the file. csv.DictReader
# keeps the last, which is the metres one, and that is what this wants.
NUMERIC = ("Elapsed Time", "Moving Time", "Distance", "Average Speed",
           "Max Speed", "Elevation Gain", "Average Heart Rate",
           "Max Heart Rate", "Average Cadence", "Calories", "Average Watts")


@dataclass
class ImportPlan:
    """What an import would do, before anything is written."""

    to_insert: list[dict[str, Any]] = field(default_factory=list)
    duplicates: list[tuple[dict[str, Any], str]] = field(default_factory=list)
    skipped_sport: list[str] = field(default_factory=list)
    unparsed: list[str] = field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        return {"import": len(self.to_insert), "duplicates": len(self.duplicates),
                "other sports": len(self.skipped_sport),
                "unreadable": len(self.unparsed)}


def read_export(zip_path: str) -> list[dict[str, str]]:
    """The activity rows from a Strava export archive.

    Raises FileNotFoundError when the archive has no activities.csv, because an
    export without it is not an export and saying so beats importing nothing and
    reporting success.
    """
    with zipfile.ZipFile(zip_path) as archive:
        name = next((n for n in archive.namelist()
                     if n.rsplit("/", 1)[-1] == ACTIVITIES_CSV), None)
        if name is None:
            raise FileNotFoundError(
                f"{zip_path} has no {ACTIVITIES_CSV} — is it a Strava export?")
        with archive.open(name) as handle:
            text = io.TextIOWrapper(handle, encoding="utf-8-sig", newline="")
            return list(csv.DictReader(text))


def _number(row: dict[str, str], key: str) -> float | None:
    raw = (row.get(key) or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def parse_start(raw: str, tz: str = "Asia/Kolkata") -> datetime | None:
    """Strava's UTC timestamp as local time.

    The export writes `Aug 22, 2026, 12:40:43 PM` in UTC while naming the same
    activity "Evening Run". Without the conversion an 18:10 run in India is
    stored as a lunchtime one, and the ones after 18:30 land on the wrong day.
    """
    raw = (raw or "").strip().strip('"')
    if not raw:
        return None
    for fmt in ("%b %d, %Y, %I:%M:%S %p", "%b %d, %Y, %H:%M:%S",
                "%Y-%m-%d %H:%M:%S"):
        try:
            naive = datetime.strptime(raw, fmt)
        except ValueError:
            continue
        return (naive.replace(tzinfo=ZoneInfo("UTC"))
                .astimezone(ZoneInfo(tz)).replace(tzinfo=None))
    return None


def parse_row(row: dict[str, str], tz: str = "Asia/Kolkata") -> dict[str, Any] | None:
    """One export row as an activity, or None if it cannot be read.

    Heart rate is carried across when the export happens to have it, but nothing
    downstream computes efficiency from an imported row: `activity_metrics` is
    left alone, so these appear in totals and in the log and nowhere else.
    """
    activity_id = (row.get("Activity ID") or "").strip()
    start = parse_start(row.get("Activity Date") or "", tz)
    if not activity_id or start is None:
        return None
    sport = SPORT_BY_TYPE.get((row.get("Activity Type") or "").strip())
    if not sport:
        return None

    moving = _number(row, "Moving Time")
    elapsed = _number(row, "Elapsed Time")
    duration = moving or elapsed or 0.0
    distance = _number(row, "Distance") or 0.0
    speed = _number(row, "Average Speed")
    if not speed and distance and duration:
        speed = distance / duration
    return {
        # Prefixed, so an imported row can never collide with a Garmin id and is
        # obvious in the database.
        "activity_id": f"strava-{activity_id}",
        "sport": sport,
        "garmin_type": (row.get("Activity Type") or "").strip(),
        "name": (row.get("Activity Name") or "").strip() or None,
        "start_time": start.isoformat(timespec="seconds"),
        "start_date": start.date().isoformat(),
        "duration_s": duration,
        "moving_s": moving,
        "distance_m": distance,
        "avg_hr": _number(row, "Average Heart Rate"),
        "max_hr": _number(row, "Max Heart Rate"),
        "avg_speed_mps": speed,
        "avg_power_w": _number(row, "Average Watts"),
        "elevation_gain_m": _number(row, "Elevation Gain"),
        "calories": _number(row, "Calories"),
        "avg_cadence": _number(row, "Average Cadence"),
        "source": "strava",
        "raw_json": json.dumps({k: v for k, v in row.items()
                                if v not in (None, "")}, default=str),
        "ingested_at": datetime.now().isoformat(timespec="seconds"),
    }


def find_duplicate(
    candidate: dict[str, Any],
    existing: Sequence[dict[str, Any]],
    window_min: int = DUPLICATE_WINDOW_MIN,
) -> str | None:
    """The id of an activity already stored for the same session, if any.

    Sport and start time only. Distance is tempting as a tiebreak and wrong as
    one: the two services measure it differently enough that a genuine duplicate
    can differ by a hundred metres.
    """
    start = candidate.get("start_time")
    if not start:
        return None
    try:
        when = datetime.fromisoformat(str(start))
    except ValueError:
        return None
    window = timedelta(minutes=window_min)
    for row in existing:
        if (row.get("sport") or "") != candidate["sport"]:
            continue
        try:
            other = datetime.fromisoformat(str(row.get("start_time")))
        except (ValueError, TypeError):
            continue
        if abs(other - when) <= window:
            return str(row.get("activity_id"))
    return None


def plan_import(
    rows: Iterable[dict[str, str]],
    existing: Sequence[dict[str, Any]],
    sports: Sequence[str] = DEFAULT_SPORTS,
    tz: str = "Asia/Kolkata",
    window_min: int = DUPLICATE_WINDOW_MIN,
) -> ImportPlan:
    """Decide what to import, without writing anything.

    Duplicates are matched only against what is already stored. Rows within one
    export are not matched against each other: every Strava activity has its own
    id, so a genuinely repeated one collapses on upsert anyway, and comparing by
    time instead merged two short runs on the same evening into one.
    """
    wanted = {s.lower() for s in sports}
    plan = ImportPlan()

    for raw in rows:
        parsed = parse_row(raw, tz)
        if parsed is None:
            label = (raw.get("Activity Type") or "").strip()
            if label and label not in SPORT_BY_TYPE:
                plan.skipped_sport.append(label)
            else:
                plan.unparsed.append((raw.get("Activity ID") or "?").strip())
            continue
        if parsed["sport"] not in wanted:
            plan.skipped_sport.append(parsed["sport"])
            continue
        clash = find_duplicate(parsed, existing, window_min)
        if clash:
            plan.duplicates.append((parsed, clash))
            continue
        plan.to_insert.append(parsed)

    plan.to_insert.sort(key=lambda r: r["start_time"])
    return plan


def import_export(
    store: Any,
    zip_path: str,
    sports: Sequence[str] = DEFAULT_SPORTS,
    tz: str = "Asia/Kolkata",
    window_min: int = DUPLICATE_WINDOW_MIN,
    dry_run: bool = False,
) -> ImportPlan:
    """Read the export, work out what is new, and store it unless `dry_run`.

    Existing activities are read with `include_imported=True` so a second run
    recognises what the first one added and does nothing.
    """
    rows = read_export(zip_path)
    existing = store.activities(include_parents=True, include_imported=True)
    plan = plan_import(rows, existing, sports=sports, tz=tz,
                       window_min=window_min)
    if plan.to_insert and not dry_run:
        store.upsert_activities(plan.to_insert)
        log.info("Imported %d activities from %s", len(plan.to_insert), zip_path)
    return plan
