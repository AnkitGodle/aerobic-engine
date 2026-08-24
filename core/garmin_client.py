"""Garmin Connect data layer: cached login, incremental fetch, normalisation.

Notes on the library (verified against garminconnect 0.3.2):
  * `Garmin.login(tokenstore=<path>)` loads a cached session if one exists and
    otherwise logs in with credentials and writes the tokens to that path. So we
    log in once and reuse the session; Garmin flags repeated SSO logins.
  * MFA is handled by passing `prompt_mfa=<callable returning the code>`; the
    non-interactive path uses `return_on_mfa=True` + `resume_login(state, code)`.
  * 0.3.2 no longer depends on garth — token persistence is built into the client.

Garmin's JSON shapes move between account/device/firmware combinations, so every
extraction here goes through `dig()`, a recursive key search, rather than a
hard-coded path. Missing values come back as None, never as an exception.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable, Iterable
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from garminconnect import (
    Garmin,
    GarminConnectAuthenticationError,
    GarminConnectTooManyRequestsError,
)

from core.garmin_guard import GarminBlocked, GarminGuard

log = logging.getLogger("aerobic_engine.garmin")

# Garmin activityType.typeKey -> our sport buckets.
SPORT_MAP: dict[str, str] = {
    "running": "run",
    "treadmill_running": "run",
    "indoor_running": "run",
    "trail_running": "run",
    "track_running": "run",
    "virtual_run": "run",
    "obstacle_run": "run",
    "ultra_run": "run",
    "cycling": "bike",
    "road_biking": "bike",
    "mountain_biking": "bike",
    "gravel_cycling": "bike",
    "indoor_cycling": "bike",
    "virtual_ride": "bike",
    "cyclocross": "bike",
    "commuting": "bike",
    "lap_swimming": "swim",
    "open_water_swimming": "swim",
    "swimming": "swim",
    "strength_training": "strength",
    "indoor_cardio": "other",
    "walking": "other",
    "hiking": "other",
    "multi_sport": "brick",
    "triathlon": "brick",
}


def status_of(exc: BaseException) -> int | None:
    """Best-effort HTTP status from a garminconnect exception.

    The library wraps upstream errors in its own types and puts the status in the
    message, so this checks the response object first and falls back to the text.
    """
    resp = getattr(exc, "response", None)
    code = getattr(resp, "status_code", None)
    if isinstance(code, int):
        return code
    text = str(exc)
    for candidate in (429, 401, 403, 404, 500, 502, 503, 504):
        if str(candidate) in text:
            return candidate
    return None


def retry_after_of(exc: BaseException) -> float | None:
    """Seconds Garmin asked us to wait, if it said."""
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None) or {}
    for key in ("retry-after", "Retry-After"):
        if key in headers:
            try:
                return float(headers[key])
            except (TypeError, ValueError):
                pass
    match = re.search(r"'?retry_after'?\s*[:=]\s*(\d+)", str(exc))
    return float(match.group(1)) if match else None


def dig(obj: Any, *keys: str, cast: Callable[[Any], Any] | None = float) -> Any:
    """Search `obj` for `keys`, honouring the order they were given in.

    Garmin nests the same field under different parents depending on device and
    endpoint version, so this walks the whole tree rather than a fixed path. The
    key order is a priority order: `dig(hrv, "lastNightAvg", "weeklyAvg")` must
    return last night's value even though `weeklyAvg` sits earlier in the dict.
    """
    for key in keys:
        found = _find_scalar(obj, key.lower(), cast)
        if found is not None:
            return found
    return None


def _find_scalar(obj: Any, key_lower: str, cast: Callable[[Any], Any] | None) -> Any:
    """Breadth-first hunt for one scalar key anywhere in a nested structure."""
    stack: list[Any] = [obj]
    while stack:
        node = stack.pop(0)
        if isinstance(node, dict):
            for k, v in node.items():
                if (
                    k.lower() == key_lower
                    and v is not None
                    and not isinstance(v, (dict, list))
                ):
                    if cast is None:
                        return v
                    try:
                        return cast(v)
                    except (TypeError, ValueError):
                        continue
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return None


# Garmin writes these where a real status has not been established yet — the
# 265 reports "NONE" for HRV status during its multi-week onboarding period.
GARMIN_NULL_STRINGS = {"none", "unknown", "not_set", "no_status", ""}


def dig_str(obj: Any, *keys: str) -> str | None:
    v = dig(obj, *keys, cast=None)
    if v is None:
        return None
    text = str(v).strip()
    return None if text.lower() in GARMIN_NULL_STRINGS else text


def _stamp(entry: dict[str, Any], keys: tuple[str, ...]) -> str:
    for k in keys:
        if entry.get(k):
            return str(entry[k])
    return ""


def _entries(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return [e for e in data if isinstance(e, dict)]
    return []


TIME_KEYS = ("timestampLocal", "timestamp", "calendarDate")


def latest_entry(data: Any, *time_keys: str) -> Any:
    """Collapse an endpoint that returns several snapshots per day to the newest."""
    entries = _entries(data)
    if not entries:
        return None
    return max(entries, key=lambda e: _stamp(e, time_keys or TIME_KEYS))


def morning_entry(data: Any) -> Any:
    """The day's *waking* snapshot — the one worth planning from.

    `get_training_readiness` recalculates all day, including right after a hard
    session, where it collapses (a 78-minute multisport session took this account
    from 39 to 1 within the hour). That trough is real but transient, and treating
    it as the day's readiness would force a deload off a number that recovers
    overnight. Garmin's own widget shows the morning value, so we do too:
    the wake-up reset if present, else the earliest snapshot of the day.
    """
    entries = _entries(data)
    if not entries:
        return None
    wake = [
        e
        for e in entries
        if str(e.get("inputContext", "")).upper() == "AFTER_WAKEUP_RESET"
    ]
    pool = wake or entries
    return min(pool, key=lambda e: _stamp(e, TIME_KEYS))


class GarminClient:
    """Authenticated Garmin Connect session with a persisted token store."""

    def __init__(
        self,
        email: str | None = None,
        password: str | None = None,
        tokenstore: str | None = None,
        prompt_mfa: Callable[[], str] | None = None,
        guard: GarminGuard | None = None,
        allow_password_login: bool = True,
    ) -> None:
        self.email = email or os.getenv("GARMIN_EMAIL")
        # Read from the environment but not expected to be there. A cached
        # session covers every normal run, so the password is only needed to mint
        # tokens in the first place — and a credential that lives on disk to be
        # used a few times a year is a credential stored for no reason. Passed in
        # by scripts/export_tokens.py, which asks for it and keeps it in memory.
        self.password = password or os.getenv("GARMIN_PASSWORD")
        # A token blob (from scripts/export_tokens.py) can be supplied directly
        # instead of a path — that is how a hosted deployment authenticates
        # without ever performing an SSO login from a datacenter IP.
        self.tokenstore = tokenstore or os.getenv("GARMINTOKENS") or ".garmin_tokens"
        if len(self.tokenstore) <= 512:
            self.tokenstore = str(Path(self.tokenstore).expanduser())
        self.prompt_mfa = prompt_mfa
        self.guard = guard or GarminGuard()
        # Set False where an interactive login is impossible (a web host): better
        # to fail loudly than to hammer Garmin's SSO endpoint from the cloud.
        self.allow_password_login = allow_password_login
        self.api: Garmin | None = None

    # -- auth -----------------------------------------------------------
    def connect(self) -> Garmin:
        """Resume the cached session, falling back to a credential login once."""
        if self.api is not None:
            return self.api

        self.guard.check()
        api = Garmin(self.email, self.password, prompt_mfa=self.prompt_mfa)
        try:
            api.login(self.tokenstore)
            log.info("Garmin session resumed from cached tokens")
        except Exception as exc:
            status = status_of(exc)
            if status == 429 or isinstance(exc, GarminConnectTooManyRequestsError):
                self.guard.record_failure(429, retry_after_of(exc))
                raise
            if not self.allow_password_login:
                raise GarminConnectAuthenticationError(
                    "Cached Garmin session is unusable and password login is "
                    "disabled here. Re-run scripts/export_tokens.py locally and "
                    "update the stored token blob. A hosted app must never log in "
                    "to Garmin's SSO endpoint directly."
                ) from exc
            log.warning("Cached session unusable (%s); logging in fresh", exc)
            if not self.email or not self.password:
                raise GarminConnectAuthenticationError(
                    "No usable cached session, and no credentials were supplied "
                    "to mint a new one. Run scripts/export_tokens.py, which will "
                    "ask for your password and keep it in memory only."
                ) from exc
            try:
                api = Garmin(self.email, self.password, prompt_mfa=self.prompt_mfa)
                api.client.login(self.email, self.password, prompt_mfa=self.prompt_mfa)
                Path(self.tokenstore).mkdir(parents=True, exist_ok=True)
                api.client.dump(self.tokenstore)
                api.login(self.tokenstore)
            except Exception as inner:
                # A failed SSO login is the single most dangerous call we make.
                self.guard.record_failure(status_of(inner), retry_after_of(inner))
                raise
        self.guard.record_success()

        self.api = api
        return api

    @property
    def display_name(self) -> str:
        api = self.connect()
        return getattr(api, "full_name", None) or getattr(api, "display_name", "") or ""

    def _call(self, fn: Callable[..., Any], *args: Any, label: str = "") -> Any:
        """Invoke a Garmin endpoint through the guard.

        A single endpoint failing is normal — Garmin returns 404s and empty
        bodies for metrics an account has not earned yet — so those return None.
        A 429 is different: it propagates, and the guard trips the breaker.
        """
        self.guard.before_call()
        try:
            out = fn(*args)
        except GarminBlocked:
            raise
        except Exception as exc:
            status = status_of(exc)
            self.guard.record_failure(status, retry_after_of(exc))
            if status == 429 or isinstance(exc, GarminConnectTooManyRequestsError):
                log.error("Garmin rate-limited us (429) — stopping this sync")
                raise
            log.warning("Garmin call %s%s failed: %s", label or fn.__name__, args, exc)
            return None
        self.guard.record_success()
        return out

    # -- activities -----------------------------------------------------
    def fetch_activities(
        self,
        since: date | None = None,
        page_size: int = 50,
        max_activities: int = 2000,
    ) -> list[dict[str, Any]]:
        """Newest-first paging that stops as soon as it passes `since`.

        `since` is inclusive of the day, exclusive of already-stored activities —
        the caller de-duplicates on activity_id.
        """
        api = self.connect()
        out: list[dict[str, Any]] = []
        start = 0
        while start < max_activities:
            batch = self._call(api.get_activities, start, page_size, label="get_activities")
            if not batch:
                break
            if isinstance(batch, dict):
                batch = batch.get("activityList") or []
            stop = False
            for raw in batch:
                norm = normalize_activity(raw)
                if not norm:
                    continue
                if since and norm["start_date"] < since.isoformat():
                    stop = True
                    break
                out.append(norm)
            if stop or len(batch) < page_size:
                break
            start += page_size
        log.info("Fetched %d activity summaries", len(out))
        return out

    def fetch_stream(self, activity_id: str, max_points: int = 600) -> list[dict[str, Any]]:
        """HR / speed / power time series for one activity, downsampled."""
        api = self.connect()
        details = self._call(
            api.get_activity_details, str(activity_id), 2000, 0, label="get_activity_details"
        )
        return parse_stream(details, max_points=max_points)

    def fetch_multisport_children(self, parent_id: str) -> list[dict[str, Any]]:
        """The individual legs of a multisport activity.

        Garmin hides these from the activity list, and the parent carries no
        average heart rate, so without this a bike-to-run brick lands in the
        database as one HR-less blob and contributes nothing to either sport's
        volume, session count or efficiency trend.
        """
        api = self.connect()
        parent = self._call(api.get_activity, str(parent_id), label="get_activity")
        if not isinstance(parent, dict):
            return []
        meta = parent.get("metadataDTO") or {}
        child_ids = meta.get("childIds") or []
        out: list[dict[str, Any]] = []
        for cid in child_ids:
            child = self._call(api.get_activity, str(cid), label="get_activity")
            if not isinstance(child, dict):
                continue
            norm = normalize_activity(flatten_activity(child))
            if norm:
                norm["parent_activity_id"] = str(parent_id)
                norm["is_multisport_parent"] = 0
                out.append(norm)
        return out

    def fetch_zones(self, activity_id: str) -> list[dict[str, Any]]:
        """Seconds spent in each heart-rate zone for one activity."""
        api = self.connect()
        data = self._call(
            api.get_activity_hr_in_timezones,
            str(activity_id),
            label="get_activity_hr_in_timezones",
        )
        rows = []
        for z in data or []:
            if not isinstance(z, dict) or z.get("zoneNumber") is None:
                continue
            rows.append(
                {
                    "activity_id": str(activity_id),
                    "zone_number": int(z["zoneNumber"]),
                    "secs_in_zone": _f(z.get("secsInZone")) or 0.0,
                    "zone_low_bpm": _f(z.get("zoneLowBoundary")),
                }
            )
        return rows

    def fetch_exercise_sets(self, activity_id: str) -> list[dict[str, Any]]:
        """Per-exercise sets recorded by the watch's strength mode.

        Weights come back in grams; reps and duration are per set.
        """
        api = self.connect()
        data = self._call(
            api.get_activity_exercise_sets,
            str(activity_id),
            label="get_activity_exercise_sets",
        )
        sets = (data or {}).get("exerciseSets") if isinstance(data, dict) else None
        out: list[dict[str, Any]] = []
        for s in sets or []:
            if not isinstance(s, dict):
                continue
            if str(s.get("setType", "")).upper() not in ("ACTIVE", ""):
                continue  # skip REST sets
            exercises = s.get("exercises") or []
            first = exercises[0] if exercises and isinstance(exercises[0], dict) else {}
            grams = _f(s.get("weight"))
            out.append(
                {
                    "activity_id": str(activity_id),
                    "garmin_category": (first.get("category") or "") or None,
                    "garmin_name": (first.get("name") or "") or None,
                    "reps": _f(s.get("repetitionCount")),
                    "duration_s": _f(s.get("duration")),
                    "load_kg": round(grams / 1000.0, 1) if grams else None,
                }
            )
        return out

    # -- wellness -------------------------------------------------------
    def fetch_wellness_day(self, day: date, extras: bool = True) -> dict[str, Any]:
        """One row of daily recovery metrics. Every field is best-effort.

        `extras` adds three calls per day for stress, respiration and blood
        oxygen. They are worth it — stress and respiration are recovery signals
        the planner can use, and steps and intensity minutes are load that
        happens outside a logged session but still has to be recovered from —
        but a long backfill can switch them off to keep the request count down.
        """
        api = self.connect()
        cdate = day.isoformat()

        rhr = self._call(api.get_rhr_day, cdate, label="get_rhr_day")
        hrv = self._call(api.get_hrv_data, cdate, label="get_hrv_data")
        vo2 = self._call(api.get_max_metrics, cdate, label="get_max_metrics")
        readiness = morning_entry(
            self._call(api.get_training_readiness, cdate, label="get_training_readiness")
        )
        status = self._call(api.get_training_status, cdate, label="get_training_status")

        vo2_run = dig(vo2, "vo2MaxPreciseValue", "vo2MaxValue")
        vo2_bike = None
        if isinstance(vo2, list):
            for entry in vo2:
                cyc = entry.get("cycling") if isinstance(entry, dict) else None
                if cyc:
                    vo2_bike = dig(cyc, "vo2MaxPreciseValue", "vo2MaxValue")
                    break
        if vo2_bike is None:
            vo2_bike = dig(dig_nested(status, "cycling"), "vo2MaxPreciseValue", "vo2MaxValue")

        sleep = self._call(api.get_sleep_data, cdate, label="get_sleep_data")
        summary = resp = spo2 = None
        if extras:
            summary = self._call(api.get_user_summary, cdate, label="get_user_summary")
            resp = self._call(api.get_respiration_data, cdate, label="get_respiration")
            spo2 = self._call(api.get_spo2_data, cdate, label="get_spo2")
        battery = latest_entry(
            self._call(api.get_body_battery, cdate, label="get_body_battery")
        )

        row: dict[str, Any] = {
            "day": cdate,
            "resting_hr": dig(rhr, "restingHeartRate", "value", "wellnessRestingHeartRate"),
            "hrv_last_night": dig(hrv, "lastNightAvg", "weeklyAvg"),
            "hrv_7d_avg": dig(hrv, "weeklyAvg"),
            "hrv_status": dig_str(hrv, "status", "hrvStatus"),
            "vo2max_run": vo2_run,
            "vo2max_bike": vo2_bike,
            "training_readiness": dig(readiness, "score"),
            "readiness_level": dig_str(readiness, "level"),
            "training_status": dig_str(
                status, "trainingStatusFeedbackPhrase", "trainingStatus"
            ),
            "acute_load": dig(status, "dailyTrainingLoadAcute", "acuteTrainingLoad"),
            "chronic_load": dig(status, "dailyTrainingLoadChronic", "chronicTrainingLoad"),
            "load_ratio": dig(status, "acwrPercent", "loadRatio"),
            "sleep_score": dig(readiness, "sleepScore"),
            "body_battery_high": dig(readiness, "bodyBatteryHigh"),
            "body_battery_low": dig(readiness, "bodyBatteryLow"),
            "sleep_seconds": dig(sleep, "sleepTimeSeconds"),
            "nap_seconds": dig(sleep, "napTimeSeconds"),
            "battery_charged": dig(battery, "charged"),
            "battery_drained": dig(battery, "drained"),
            "steps": dig(summary, "totalSteps"),
            "stress_avg": dig(summary, "averageStressLevel"),
            "stress_max": dig(summary, "maxStressLevel"),
            "intensity_moderate_min": dig(summary, "moderateIntensityMinutes"),
            "intensity_vigorous_min": dig(summary, "vigorousIntensityMinutes"),
            "floors_climbed": dig(summary, "floorsAscended"),
            "active_calories": dig(summary, "activeKilocalories"),
            "respiration_avg": dig(resp, "avgWakingRespirationValue",
                                   "avgTomorrowSleepRespirationValue"),
            "respiration_sleep_avg": dig(resp, "avgSleepRespirationValue"),
            "spo2_avg": dig(spo2, "averageSpO2", "averageSpo2"),
            "spo2_lowest": dig(spo2, "lowestSpO2", "lowestSpo2"),
            "weight_kg": _weight_kg(dig(summary, "weight")),
            "ingested_at": datetime.now().isoformat(timespec="seconds"),
        }
        # Garmin reports a stress level of -1 or -2 for "not measured". Those are
        # sentinels, not low stress, and would drag any average down.
        for key in ("stress_avg", "stress_max"):
            if row.get(key) is not None and row[key] < 0:
                row[key] = None
        # Garmin reports ACWR as a percentage on some firmware; store it as a ratio.
        if row["load_ratio"] and row["load_ratio"] > 5:
            row["load_ratio"] = row["load_ratio"] / 100.0
        return row

    def fetch_wellness_range(
        self, days: Iterable[date], skip: set[str] | None = None,
        extras: bool = True,
    ) -> list[dict[str, Any]]:
        skip = skip or set()
        rows = []
        for d in days:
            if d.isoformat() in skip:
                continue
            rows.append(self.fetch_wellness_day(d, extras=extras))
        return rows

    def fetch_weather(self, activity_id: str) -> dict[str, Any] | None:
        """Conditions during one activity.

        Worth a request per outdoor session because heat and humidity raise
        heart rate at a given pace, and heart rate at a given pace is the whole
        basis of the efficiency chart. A humid fortnight otherwise reads as lost
        fitness.
        """
        api = self.connect()
        raw = self._call(api.get_activity_weather, activity_id, label="weather")
        if not isinstance(raw, dict) or not raw:
            return None
        return {
            "activity_id": str(activity_id),
            "temp_c": _to_celsius(raw.get("temp")),
            "apparent_c": _to_celsius(raw.get("apparentTemp")),
            "dew_point_c": _to_celsius(raw.get("dewPoint")),
            "humidity_pct": _f(raw.get("relativeHumidity")),
            "wind_kph": _f(raw.get("windSpeed")),
            "condition": dig_str(raw.get("weatherTypeDTO") or {}, "desc"),
            "fetched_at": datetime.now().isoformat(timespec="seconds"),
        }

    def scheduled_workouts(self, year: int, month: int) -> list[dict[str, Any]]:
        """Calendar entries for scheduled workouts in one month.

        Garmin's calendar month is zero-based in the response and one-based in
        the request, which is exactly the sort of thing to write down rather than
        rediscover.
        """
        data = self._call(self.connect().get_scheduled_workouts, int(year),
                          int(month), label="get_scheduled_workouts")
        items = (data or {}).get("calendarItems") if isinstance(data, dict) else None
        return [
            {
                "schedule_id": item.get("id"),
                "workout_id": item.get("workoutId"),
                "title": item.get("title") or "",
                "date": item.get("date"),
                "sport": item.get("sportTypeKey"),
                "protected": bool(item.get("protectedWorkoutSchedule")),
            }
            for item in (items or [])
            if isinstance(item, dict) and item.get("workoutId")
            and item.get("itemType") == "workout"
        ]

    def unschedule_workout(self, schedule_id: str) -> bool:
        """Take one entry off the training calendar.

        Separate from deleting the workout: the calendar entry is what shows up
        in Garmin Connect's plan, and deleting the workout alone leaves it there.
        """
        try:
            self._call(self.connect().unschedule_workout, str(schedule_id),
                       label="unschedule_workout")
            return True
        except Exception as exc:  # noqa: BLE001 - tidying never fails a sync
            if status_of(exc) in (404, 400):
                return True
            log.warning("Could not unschedule %s: %s", schedule_id, exc)
            return False

    def delete_workout(self, workout_id: str) -> bool:
        """Remove a workout from the account. True if Garmin accepted it.

        Used to clear a session off the watch once it has been done: a new one is
        pushed most days, and without this the saved-workout list fills up with
        every session of the last month.

        A workout that is already gone is a success, not a failure — the goal is
        "not on the watch any more", and someone deleting it by hand should not
        make the sync noisy.
        """
        try:
            self._call(self.connect().delete_workout, str(workout_id),
                       label="delete_workout")
            return True
        except Exception as exc:  # noqa: BLE001 - never worth failing a sync for
            status = status_of(exc)
            if status in (404, 400):
                log.info("Workout %s was already gone", workout_id)
                return True
            log.warning("Could not delete workout %s: %s", workout_id, exc)
            return False

    def fetch_laps(self, activity_id: str) -> list[dict[str, Any]]:
        """Per-lap heart rate and pace.

        Worth a request per session because it answers a question the session
        averages cannot: whether heart rate climbed while pace held. Aerobic
        drift computed from the stream needs a 60-minute session to split in
        half; auto-lap gives the same signal on a 45-minute one, one kilometre
        at a time.
        """
        api = self.connect()
        raw = self._call(api.get_activity_splits, activity_id, label="laps")
        laps = (raw or {}).get("lapDTOs") if isinstance(raw, dict) else None
        rows = []
        for lap in _entries(laps):
            index = lap.get("lapIndex")
            if index is None:
                continue
            rows.append({
                "activity_id": str(activity_id),
                "lap_index": int(index),
                "duration_s": _f(lap.get("duration")),
                "distance_m": _f(lap.get("distance")),
                "avg_hr": _f(lap.get("averageHR")),
                "max_hr": _f(lap.get("maxHR")),
                "avg_speed_mps": _f(lap.get("averageSpeed")),
                "avg_cadence": _f(lap.get("averageRunCadence")
                                  or lap.get("averageBikeCadence")),
                "avg_power_w": _f(lap.get("averagePower")),
                "elevation_gain_m": _f(lap.get("elevationGain")),
                # Garmin labels every auto-lap "INTERVAL" whether or not the
                # session was one, so this is stored as a fact about the file
                # rather than trusted as a description of the training.
                "intensity": dig_str(lap, "intensityType"),
                "fetched_at": datetime.now().isoformat(timespec="seconds"),
            })
        return rows

    def fetch_profile(self) -> dict[str, Any]:
        """Body and physiology constants: weight, height, age, threshold HR.

        Weight is what turns bike watts into watts per kilo, and the intensity
        zone numbers explain a discrepancy that is otherwise baffling — Garmin
        counts zone 3 as "moderate" and zone 4 up as "vigorous", so time spent
        at a deliberately raised aerobic ceiling never shows up as easy.
        """
        api = self.connect()
        raw = self._call(api.get_user_profile, label="get_user_profile") or {}
        data = raw.get("userData") or {}
        grams = _f(data.get("weight"))
        born = dig_str(data, "birthDate")
        age = None
        if born:
            try:
                b = date.fromisoformat(born[:10])
                today = date.today()
                age = today.year - b.year - ((today.month, today.day) < (b.month, b.day))
            except ValueError:
                age = None
        return {
            # Garmin stores weight in grams.
            "weight_kg": round(grams / 1000.0, 1) if grams else None,
            "height_cm": _f(data.get("height")),
            "birth_date": born,
            "age": age,
            "gender": dig_str(data, "gender"),
            "vo2max_run": _f(data.get("vo2MaxRunning")),
            "vo2max_bike": _f(data.get("vo2MaxCycling")),
            "threshold_hr": _f(data.get("lactateThresholdHeartRate")),
            "moderate_zone": _f(data.get("moderateIntensityMinutesHrZone")),
            "vigorous_zone": _f(data.get("vigorousIntensityMinutesHrZone")),
        }

    # Garmin's own numbering for personal records. Only the ones this athlete's
    # sports can produce are named; anything else is stored with its raw id
    # rather than guessed at.
    PR_LABELS: dict[int, tuple[str, str]] = {
        1: ("run", "Fastest 1 km"),
        2: ("run", "Fastest 1 mile"),
        3: ("run", "Fastest 5 km"),
        4: ("run", "Fastest 10 km"),
        5: ("run", "Fastest half marathon"),
        6: ("run", "Fastest marathon"),
        7: ("run", "Longest run"),
        8: ("bike", "Longest ride"),
        9: ("bike", "Biggest climb"),
        12: ("other", "Most steps in a day"),
        13: ("other", "Most steps in a week"),
        14: ("other", "Most steps in a month"),
        15: ("other", "Most floors in a day"),
    }

    def fetch_personal_records(self) -> list[dict[str, Any]]:
        api = self.connect()
        raw = self._call(api.get_personal_record, label="get_personal_record")
        rows = []
        for entry in _entries(raw):
            try:
                type_id = int(entry.get("typeId"))
            except (TypeError, ValueError):
                continue
            sport, label = self.PR_LABELS.get(type_id, ("other", f"Record {type_id}"))
            rows.append({
                "type_id": type_id,
                "sport": dig_str(entry, "activityType") or sport,
                "label": label,
                "value": _f(entry.get("value")),
                "achieved_at": as_iso_day(
                    dig_str(entry, "prStartTimeLocal",
                            "activityStartDateTimeLocal")),
                "activity_id": dig_str(entry, "activityId"),
                "fetched_at": datetime.now().isoformat(timespec="seconds"),
            })
        return rows

    def fetch_race_predictions(self, start: date, end: date) -> list[dict[str, Any]]:
        """Garmin's run-time predictions.

        The date-ranged call returns a row per day that is all-null unless the
        prediction actually changed that day, while the no-argument call returns
        the current prediction. We take both: history where it exists, and always
        today's standing numbers.
        """
        api = self.connect()
        rows: list[dict[str, Any]] = []
        ranged = self._call(
            api.get_race_predictions,
            start.isoformat(),
            end.isoformat(),
            "daily",
            label="get_race_predictions",
        )
        latest = self._call(api.get_race_predictions, label="get_race_predictions")

        for data in (ranged, latest):
            if not data:
                continue
            for e in data if isinstance(data, list) else [data]:
                if not isinstance(e, dict):
                    continue
                row = {
                    "day": (
                        dig_str(e, "calendarDate", "toCalendarDate") or end.isoformat()
                    )[:10],
                    "time_5k": dig(e, "time5K"),
                    "time_10k": dig(e, "time10K"),
                    "time_half": dig(e, "timeHalfMarathon"),
                    "time_marathon": dig(e, "timeMarathon"),
                }
                # Skip the empty placeholder rows the ranged call pads with.
                if any(row[k] for k in ("time_5k", "time_10k", "time_half", "time_marathon")):
                    rows.append(row)
        return rows


    def fetch_thresholds(self) -> dict[str, Any]:
        """Account-level performance thresholds — no date, just current values.

        These anchor the zone charts: without threshold HR, a zone number is
        just a number.
        """
        api = self.connect()
        lt = self._call(api.get_lactate_threshold, label="get_lactate_threshold")
        ftp = self._call(api.get_cycling_ftp, label="get_cycling_ftp")
        run_ftp = None
        if isinstance(lt, dict):
            power = lt.get("power") or {}
            if str(power.get("sport", "")).upper() == "RUNNING":
                run_ftp = _f(power.get("functionalThresholdPower"))
        return {
            "threshold_hr": dig(lt, "heartRate"),
            "threshold_speed": dig(lt, "speed"),
            "running_ftp": run_ftp,
            "cycling_ftp": dig(ftp, "functionalThresholdPower"),
        }


def as_iso_day(value: Any) -> str | None:
    """Normalise a Garmin date to an ISO day, whatever shape it arrived in.

    Personal records date themselves with `prStartTimeLocal`, which on this
    account is a Unix timestamp in milliseconds rather than a date string. Stored
    raw it displayed as 16 December 1787, because Python's `date.fromisoformat`
    is lenient enough to read the first eight digits of the epoch as a date
    instead of rejecting them. Normalising here means the database holds a real
    day, not just the display fixing it up.
    """
    if value in (None, ""):
        return None
    text = str(value).strip()
    if text.isdigit() and len(text) in (10, 13):
        seconds = int(text) / (1000.0 if len(text) == 13 else 1.0)
        try:
            return datetime.fromtimestamp(seconds, tz=timezone.utc).date().isoformat()
        except (OverflowError, OSError, ValueError):
            return text
    return text


def flatten_activity(detail: dict[str, Any]) -> dict[str, Any]:
    """`get_activity` nests the numbers in summaryDTO; the list endpoint doesn't.

    normalize_activity() expects the flat list shape, so lift the summary keys up
    and rebuild the activityType field the flat shape uses.
    """
    flat = dict(detail)
    flat.update(detail.get("summaryDTO") or {})
    atype = detail.get("activityTypeDTO") or detail.get("activityType") or {}
    if atype:
        flat["activityType"] = atype
    if detail.get("activityName"):
        flat["activityName"] = detail["activityName"]
    if flat.get("startTimeLocal"):
        flat["startTimeLocal"] = str(flat["startTimeLocal"]).replace("T", " ")[:19]
    flat["activityId"] = detail.get("activityId") or flat.get("activityId")
    return flat


def dig_nested(obj: Any, key: str) -> Any:
    """Return the first sub-dict stored under `key` anywhere in `obj`."""
    stack = [obj]
    while stack:
        node = stack.pop(0)
        if isinstance(node, dict):
            if key in node and isinstance(node[key], (dict, list)):
                return node[key]
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return None


def normalize_activity(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Garmin activity summary -> our `activities` row."""
    if not isinstance(raw, dict):
        return None
    aid = raw.get("activityId")
    start = raw.get("startTimeLocal") or raw.get("startTimeGMT")
    if aid is None or not start:
        return None

    type_key = (raw.get("activityType") or {}).get("typeKey", "") or ""
    parent = (raw.get("activityType") or {}).get("parentTypeId")
    sport = SPORT_MAP.get(type_key.lower())
    if sport is None:
        # Unmapped subtype — fall back to a substring match on the type key.
        lowered = type_key.lower()
        for needle, mapped in (
            ("run", "run"),
            ("cycl", "bike"),
            ("bik", "bike"),
            ("swim", "swim"),
            ("strength", "strength"),
        ):
            if needle in lowered:
                sport = mapped
                break
    sport = sport or "other"

    start_iso = str(start).replace(" ", "T")
    return {
        "activity_id": str(aid),
        "sport": sport,
        "garmin_type": type_key or (str(parent) if parent else None),
        "name": raw.get("activityName"),
        "start_time": start_iso,
        "start_date": start_iso[:10],
        "duration_s": _f(raw.get("duration")),
        "moving_s": _f(raw.get("movingDuration")),
        "distance_m": _f(raw.get("distance")),
        "avg_hr": _f(raw.get("averageHR")),
        "max_hr": _f(raw.get("maxHR")),
        "avg_speed_mps": _f(raw.get("averageSpeed")),
        "avg_power_w": _f(raw.get("avgPower") or raw.get("averagePower")),
        "norm_power_w": _f(raw.get("normPower")),
        # Running dynamics. Cadence and stride length are the two halves of
        # pace, and the interesting one is cadence: a low cadence means a long
        # stride, which means landing further in front of the body, which is the
        # knee and shin load the strength work exists to protect against.
        "avg_cadence": _f(raw.get("averageRunCadence")
                          or raw.get("averageRunningCadenceInStepsPerMinute")
                          or raw.get("averageBikingCadenceInRevPerMinute")),
        "max_cadence": _f(raw.get("maxRunCadence")
                          or raw.get("maxRunningCadenceInStepsPerMinute")
                          or raw.get("maxBikingCadenceInRevPerMinute")),
        "stride_length_cm": _f(raw.get("strideLength")),
        "ground_contact_ms": _f(raw.get("groundContactTime")),
        "vertical_osc_cm": _f(raw.get("verticalOscillation")),
        "vertical_ratio": _f(raw.get("verticalRatio")),
        "steps": _f(raw.get("steps")),
        "elevation_gain_m": _f(raw.get("elevationGain")),
        "calories": _f(raw.get("calories")),
        "training_load": _f(
            raw.get("activityTrainingLoad") or raw.get("trainingLoad")
        ),
        "aerobic_te": _f(raw.get("aerobicTrainingEffect")),
        "anaerobic_te": _f(raw.get("anaerobicTrainingEffect")),
        "rpe": _f(raw.get("rpe") or raw.get("perceivedExertion")),
        "pool_length_m": _f(raw.get("poolLength")),
        # The list endpoint says isParent/parent; the detail endpoint says
        # isMultiSportParent. The activity type is the backstop.
        "is_multisport_parent": int(
            bool(
                raw.get("isMultiSportParent")
                or raw.get("isParent")
                or raw.get("parent")
                or (type_key or "").lower() in ("multi_sport", "triathlon")
            )
        ),
        "raw_json": None,  # summaries are re-fetchable; keep the DB small
        "ingested_at": datetime.now().isoformat(timespec="seconds"),
    }


def parse_stream(details: Any, max_points: int = 600) -> list[dict[str, Any]]:
    """`get_activity_details` -> [{t_s, hr, speed_mps, power_w, altitude_m}].

    Downsampled: a two-hour ride is ~7000 samples and the maths does not need
    them all.
    """
    if not isinstance(details, dict):
        return []
    descriptors = details.get("metricDescriptors") or []
    metrics = details.get("activityDetailMetrics") or []
    if not descriptors or not metrics:
        return []

    idx: dict[str, int] = {}
    for d in descriptors:
        key = (d.get("key") or "").strip()
        if key and d.get("metricsIndex") is not None:
            idx[key] = int(d["metricsIndex"])

    def pick(row: list[Any], *keys: str) -> float | None:
        for k in keys:
            i = idx.get(k)
            if i is not None and i < len(row):
                return _f(row[i])
        return None

    step = max(1, len(metrics) // max_points)
    out: list[dict[str, Any]] = []
    for n, m in enumerate(metrics):
        if n % step:
            continue
        row = m.get("metrics") if isinstance(m, dict) else None
        if not row:
            continue
        t = pick(row, "sumDuration", "sumElapsedDuration", "directTimestamp")
        hr = pick(row, "directHeartRate")
        if t is None:
            t = float(n)
        out.append(
            {
                "t_s": t,
                "hr": hr,
                "speed_mps": pick(row, "directSpeed", "directGroundSpeed"),
                "power_w": pick(row, "directPower", "directBikePower"),
                "altitude_m": pick(row, "directElevation", "directAltitude",
                                   "directCorrectedElevation"),
                # directRunCadence is steps for one leg on some firmware and
                # both on others; directDoubleCadence is always both. Prefer the
                # unambiguous one so a 75 never gets read as a 150.
                "cadence": pick(row, "directDoubleCadence", "directRunCadence",
                                "directBikeCadence"),
                "stride_length_cm": pick(row, "directStrideLength"),
            }
        )
    # directTimestamp is epoch-ms; rebase to seconds from the start.
    if out and out[0]["t_s"] > 1e9:
        t0 = out[0]["t_s"]
        for s in out:
            s["t_s"] = (s["t_s"] - t0) / 1000.0
    return out


def _weight_kg(v: Any) -> float | None:
    """Garmin stores weight in grams; a bare 66000 in a kg column is nonsense."""
    n = _f(v)
    if n is None:
        return None
    return round(n / 1000.0, 1) if n > 500 else round(n, 1)


def _to_celsius(v: Any) -> float | None:
    """Garmin's weather endpoint reports Fahrenheit whatever the account says.

    Verified on this account: get_unit_system() returns "metric", yet the same
    activity came back as temp 74 / dewPoint 70 / humidity 88 — a 74C morning
    run being impossible. The threshold below is a guard rather than a guess: it
    only matters if Garmin ever starts honouring the setting, in which case a
    plausible Celsius reading passes straight through.
    """
    n = _f(v)
    if n is None:
        return None
    if n <= 45:
        return round(n, 1)          # already Celsius
    return round((n - 32.0) * 5.0 / 9.0, 1)


def _f(v: Any) -> float | None:
    try:
        if v is None or isinstance(v, bool):
            return None
        f = float(v)
        return f if f == f else None  # drop NaN
    except (TypeError, ValueError):
        return None


def date_range(start: date, end: date) -> list[date]:
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]
