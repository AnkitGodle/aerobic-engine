"""Sync orchestration: Garmin -> database. No Streamlit, no CLI concerns.

Lives in `core/` because two callers need it — `scripts/fetch.py` for a local or
scheduled run, and the dashboard's Refresh button. Every Garmin call inside here
goes through `GarminGuard`, and the whole sync takes the guard's single-flight
lock, so two callers cannot run at once.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from typing import Any, Callable

from core import applog, strength
from core.analysis import compute_activity_metrics
from core.garmin_client import GarminClient, date_range
from core.garmin_guard import GarminGuard
from core.store import Store, default_db

log = logging.getLogger("aerobic_engine.sync")

STREAM_SPORTS = {"run", "bike", "swim", "brick"}
# Pool swims and strength sessions happen indoors, so asking for weather would
# spend a request to learn nothing.
WEATHER_SPORTS = {"run", "bike", "brick"}
# Laps only mean something where distance is measured and paced.
LAP_SPORTS = {"run", "bike", "swim", "brick"}
ZONE_SPORTS = {"run", "bike", "swim", "strength", "brick"}


def prompt_mfa() -> str:
    """Interactive MFA. Only ever hit on the first login of a new session."""
    return input("Garmin MFA code: ").strip()


def import_multisport_children(store: Store, client: GarminClient) -> int:
    """Store each leg of a multisport activity as its own activity."""
    added = 0
    for parent in store.multisport_parents():
        if store.has_children(parent["activity_id"]):
            continue
        children = client.fetch_multisport_children(parent["activity_id"])
        if children:
            store.upsert_activities(children)
            added += len(children)
            log.info(
                "Split multisport %s into %d legs: %s",
                parent.get("name") or parent["activity_id"],
                len(children),
                ", ".join(f"{c['sport']} {c['duration_s'] / 60:.0f}min" for c in children),
            )
    return added


def remap_exercise_sets(store: Store) -> tuple[int, int]:
    """Re-derive every stored set's exercise, and rebuild the log rows from it.

    Needed whenever the mapping changes, and it changed for a good reason: the
    first real session came back with a pushed wall sit and split squat logged as
    goblet squats, and three sets of tibialis raises logged as calf raises,
    because a bare category used to be resolved by guessing. Nothing is refetched
    — Garmin's own category and name are already stored beside each set.

    Returns (sets re-mapped, log rows rebuilt). Sessions the athlete logged by
    hand are left alone: their rows have no activity_id.
    """
    changed = rebuilt = 0
    per_activity: dict[str, list[dict[str, Any]]] = {}
    for row in store.exercise_sets():
        per_activity.setdefault(str(row["activity_id"]), []).append(dict(row))

    for activity_id, rows in per_activity.items():
        moved = False
        for r in rows:
            if r.get("assigned_by_hand"):
                continue          # the athlete's own answer outranks the table
            before = r.get("exercise_id")
            r["exercise_id"] = strength.map_garmin_exercise(
                r.get("garmin_category"), r.get("garmin_name"),
                reps=r.get("reps"), duration_s=r.get("duration_s"))
            was_rest = r.get("is_rest")
            r["is_rest"] = int(strength.looks_like_rest(r))
            if r["is_rest"] != was_rest:
                moved = True
            if r["exercise_id"] != before:
                moved = True
                changed += 1
        if not moved:
            continue
        store.replace_exercise_sets(activity_id, rows)
        day = store.activity_day(activity_id)
        if not day:
            continue
        # Only the rows this activity produced are replaced; a hand-logged
        # session for the same day is left as the athlete entered it.
        store.delete_strength_log(activity_id=activity_id)
        fresh = strength.sets_to_log_rows(day, activity_id, rows)
        if fresh:
            rebuilt += store.log_strength(fresh)
    if changed:
        log.info("Re-mapped %d sets and rebuilt %d log rows", changed, rebuilt)
    return changed, rebuilt


def import_exercise_sets(store: Store, client: GarminClient) -> tuple[int, int]:
    """Pull watch-recorded strength sets and turn them into strength_log rows."""
    sets_stored = logged = 0
    already = store.strength_days_logged()
    for act in store.strength_activities_missing_sets():
        raw = client.fetch_exercise_sets(act["activity_id"])
        if not raw:
            continue
        for r in raw:
            # Reps and duration go in too: they are what separates a single-leg
            # calf raise from a single-leg calf hold, which Garmin gives the same
            # name.
            r["exercise_id"] = strength.map_garmin_exercise(
                r.get("garmin_category"), r.get("garmin_name"),
                reps=r.get("reps"), duration_s=r.get("duration_s"),
            )
            r["is_rest"] = int(strength.looks_like_rest(r))
        store.replace_exercise_sets(act["activity_id"], raw)
        sets_stored += len(raw)
        unmapped = [r for r in raw
                    if not r.get("exercise_id") and not r.get("is_rest")]
        if unmapped:
            log.info(
                "%d of %d sets on %s are outside the exercise library "
                "(assign them in the UI): %s",
                len(unmapped), len(raw), act["start_date"],
                sorted({r.get("garmin_name") or r.get("garmin_category") or "?"
                        for r in unmapped}),
            )
        # Don't overwrite a session the athlete already logged by hand.
        if act["start_date"] in already:
            continue
        rows = strength.sets_to_log_rows(act["start_date"], act["activity_id"], raw)
        if rows:
            logged += store.log_strength(rows)
            already.add(act["start_date"])
    return sets_stored, logged


def generate_ai_notes(db: str | None = None, today: date | None = None) -> int:
    """Write every page summary and chart reading into the database.

    Done here, once per sync, rather than on page render. Streamlit executes the
    body of every tab on every run, so inline AI calls fired ten times per page
    load and took it from 1.5 seconds to 92. Generating at sync time costs about
    ten calls after each Garmin refresh — comfortably inside a free tier — and
    the dashboard then reads plain text out of SQLite.
    """
    from core import ai, insights

    if not ai.available():
        log.info("No AI backend configured; skipping summaries")
        return 0

    today = today or date.today()
    # Free tiers cap requests per minute, and a dozen summaries fired back to back
    # trip that every time. This runs after a Garmin sync, not in front of a
    # waiting user, so it can simply go slowly. A rate limit pauses and retries
    # once rather than losing the summary.
    # AI_NOTES_BACKEND can aim the summaries at a different set of providers
    # from the planner; by default they share AI_BACKEND.
    backends = _note_backends()
    if not backends:
        log.info("No usable summary backend; skipping")
        return 0
    store = Store(db)
    try:
        data = {
            "activities": store.activities(),
            "all_activities": store.activities(include_parents=True),
            "wellness": store.wellness(), "zones": store.zones(),
            "strength": store.strength_log(), "sets": store.exercise_sets(),
            "race": store.race_predictions(),
            "records": store.personal_records(),
            "weather": store.weather(),
        }
        model = getattr(backends[0], "model", "")
        rows: list[dict[str, Any]] = []

        jobs: list[tuple[str, str, Any]] = [
            ("page", page, insights.for_page(page, data, today))
            for page in insights.PAGES
        ]
        jobs += [("chart", key, (title, payload))
                 for key, title, payload in _chart_inputs(data, today)]

        failed = 0
        # Round-robin across providers, run concurrently. Two providers at
        # roughly 20 requests a minute each comfortably absorb fourteen calls, so
        # no spacing is needed — spreading the load is what the spacing was
        # approximating, badly.
        live = [(i, k, key, a) for i, (k, key, a) in enumerate(jobs) if a is not None]
        workers = max(1, min(len(backends) * 3, len(live)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {}
            for n, (_, kind, key, arg) in enumerate(live):
                primary = backends[n % len(backends)]
                others = [b for b in backends if b is not primary]
                futures[pool.submit(_one_note, kind, key, arg, primary, others)] = (
                    kind, key)
            for future in as_completed(futures):
                kind, key = futures[future]
                text = future.result()
                if text:
                    rows.append({"key": key, "kind": kind, "text": text,
                                 "model": model})
                else:
                    failed += 1
        if failed:
            log.info("%d of %d summaries could not be generated this run",
                     failed, len(jobs))

        written = store.set_notes(rows)
        store.set_state("ai_notes_at", datetime.now().isoformat(timespec="seconds"))
        log.info("Wrote %d AI summaries", written)
        return written
    finally:
        store.close()


def _note_backends() -> list[Any]:
    """One backend object per fast provider, not a single fallback chain.

    Separate objects on purpose: the summaries are independent calls, so they can
    run at the same time and be spread across providers. That keeps each one
    under its own per-minute cap, which is what the spacing was for, and the
    wall-clock becomes the slowest single call rather than the sum of all of
    them plus the waiting.
    """
    from core import ai

    names = [n.strip().lower() for n in
             (os.getenv("AI_NOTES_BACKEND") or os.getenv("AI_BACKEND")
              or ai.DEFAULT_CHAIN).split(",") if n.strip()]
    out = []
    for name in names:
        try:
            out.append(ai._one_backend(name))
        except ai.AIUnavailable as exc:
            log.info("Notes backend %s unavailable (%s)", name, exc)
    return out


def _one_note(kind: str, key: str, arg: Any, backend: Any,
              alternates: Sequence[Any] = ()) -> str | None:
    """Generate one summary. A rate limit moves it to another provider.

    Moving beats waiting: with two providers configured, the second one's budget
    is untouched at the moment the first refuses.
    """
    from core import ai, insights

    for candidate in [backend, *alternates]:
        try:
            if kind == "page":
                return insights.narrate(key, arg, backend=candidate)
            title, payload = arg
            return insights.chart_note(title, payload, backend=candidate)
        except ai.AIUnavailable as exc:
            log.info("%s on %s (%s); trying another provider", key,
                     getattr(candidate, "name", "?"), str(exc)[:70])
            continue
        except Exception:  # noqa: BLE001 - one bad summary is not a failed sync
            return None
    return None


def _chart_inputs(data: dict[str, Any], today: date) -> list[tuple[str, str, Any]]:
    """The charts worth a sentence, and the numbers behind each."""
    from core.analysis import (
        ef_points, hr_points, load_ramp, polarisation, ramp_verdict, week_summaries,
        weather_effect, zone_distribution,
    )
    from core.schemas import ENDURANCE_SPORTS

    acts, zones = data["activities"], data["zones"]
    since = today - timedelta(days=28)
    out: list[tuple[str, str, Any]] = []

    hr = {sp: [{"date": str(p["date"]), "bpm": p["hr_at_reference"],
                "min": round(p["minutes"])}
               for p in hr_points(acts, sp) if p.get("hr_at_reference")]
          for sp in ENDURANCE_SPORTS}
    if any(hr.values()):
        out.append(("chart:training_hr",
                    "Heart rate at the athlete's usual pace, by sport "
                    "(bpm; falling is better)", hr))
        # The same series over all time rather than the recent window. Read
        # differently on purpose: the question there is the direction of travel
        # over months, not whether this block is working.
        out.append(("chart:lifetime_hr",
                    "Heart rate at the athlete's usual pace across their whole "
                    "history, by sport (bpm; falling is better). Comment on the "
                    "overall direction and on how much history there is to "
                    "judge it from", hr))

    wl = data.get("wellness") or []
    if wl:
        # Weekly means, not every day. Sending the raw record would grow this
        # prompt for the rest of the athlete's life — a year of daily rows is
        # about 8,000 tokens for a one-line caption — and the question being
        # asked is the long-run direction, which weekly means answer better than
        # daily noise does.
        buckets: dict[str, list[dict[str, Any]]] = {}
        for r in wl:
            try:
                monday = date.fromisoformat(str(r["day"])[:10])
            except (ValueError, TypeError, KeyError):
                continue
            monday -= timedelta(days=monday.weekday())
            buckets.setdefault(monday.isoformat(), []).append(r)

        def mean(rows: list[dict[str, Any]], key: str, scale: float = 1.0):
            vals = [r[key] for r in rows if r.get(key) is not None]
            return round(sum(vals) / len(vals) * scale, 1) if vals else None

        weekly = [{"week_of": wk,
                   "resting_hr": mean(rows, "resting_hr"),
                   "hrv": mean(rows, "hrv_last_night"),
                   "sleep_h": mean(rows, "sleep_seconds", 1 / 3600)}
                  for wk, rows in sorted(buckets.items())]
        out.append(("chart:lifetime_recovery",
                    "Weekly averages of resting heart rate, overnight HRV and "
                    "sleep across the whole record (resting HR falling and HRV "
                    "rising are both good). Say what the long-run direction is",
                    weekly[-52:]))

    if zones:
        out.append(("chart:intensity",
                    "Share of training time by intensity, last 28 days "
                    "(base target: 70%+ easy, under 15% hard)",
                    {"percent": polarisation(zones, since=since),
                     "by_sport_minutes": {sp: zone_distribution(zones, sport=sp,
                                                                since=since)
                                          for sp in ENDURANCE_SPORTS}}))

    ef = {sp: [{"date": str(p.date), "ef": round(p.ef, 3), "steady": p.is_steady}
               for p in ef_points(acts, sp)] for sp in ENDURANCE_SPORTS}
    if any(ef.values()):
        out.append(("chart:efficiency",
                    "Efficiency (speed or watts per heartbeat) as % change from "
                    "each sport's first session", ef))

    weeks = week_summaries(acts, weeks=12, as_of=today,
                           strength_rows=data["strength"])
    done = [{"week": str(w.week_start), "min": round(w.total_minutes)}
            for w in weeks if w.total_minutes > 0]
    if done:
        out.append(("chart:volume", "Weekly training minutes completed", done))

    ramp = load_ramp(acts, as_of=today, days=60)
    if len(ramp) >= 7:
        # Weekly samples of the ramp, not sixty daily rows: the question is the
        # direction of the ratio, and daily rows would spend most of the prompt
        # on load values the caption never mentions.
        out.append(("chart:load_ramp",
                    "Training load ramp: 7-day average against the 28-day base, "
                    "and the ratio between them (0.8-1.3 is productive, above "
                    "1.3 is ramping faster than the body absorbs)",
                    {"weekly": [{"day": str(r["day"]), "acute": r["acute"],
                                 "chronic": r["chronic"], "ratio": r["ratio"]}
                                for r in ramp[::7]],
                     "verdict": ramp_verdict(ramp)}))

    race = [r for r in (data.get("race") or [])
            if r.get("time_5k") or r.get("time_10k")]
    if len(race) >= 2:
        out.append(("chart:race",
                    "Garmin's own race-time predictions over time, in seconds "
                    "(falling is faster). Say whether they are improving and by "
                    "how much, and note that they are an estimate from recent "
                    "running, not a test",
                    [{"day": str(r["day"]), "s_5k": r.get("time_5k"),
                      "s_10k": r.get("time_10k"), "s_half": r.get("time_half"),
                      "s_marathon": r.get("time_marathon")} for r in race]))

    heat = weather_effect(acts, data.get("weather") or {}, sport="run")
    if heat["points"]:
        out.append(("chart:heat",
                    "Heart rate at the athlete's usual running pace against dew "
                    "point. Above 20C dew point sweat stops evaporating well and "
                    "heart rate rises for the same effort. Say whether the "
                    "weather explains the heart rates or not",
                    {"points": [{"date": str(p["date"]),
                                 "dew_point_c": p["dew_point_c"],
                                 "bpm_at_usual_pace": p["hr_at_reference"]}
                                for p in heat["points"]],
                     "bpm_per_degree": heat["bpm_per_deg"],
                     "share_above_20C": heat["hot_share"]}))
    return out



# Pushed workouts are recorded as `workout_pushed_<date>` for strength and
# `workout_pushed_<sport>_<date>` for a run or a ride, keyed by the day they were
# for.
PUSHED_PREFIX = "workout_pushed_"
# How long a pushed workout may sit on the watch unused before it is cleared. One
# day: the plan moves on, and yesterday's session is not what you want offered
# when you press START.
PUSHED_KEEP_DAYS = 1


def _pushed_key_parts(key: str) -> tuple[str | None, date | None]:
    """('run', 2026-08-24) or (None, 2026-08-24) for a strength push."""
    rest = key[len(PUSHED_PREFIX):]
    bits = rest.rsplit("_", 1)
    stamp = bits[-1]
    sport = bits[0] if len(bits) > 1 else None
    try:
        return sport, date.fromisoformat(stamp)
    except ValueError:
        return sport, None


def delete_finished_workouts(
    store: Store, client: Any, today: date | None = None,
    keep_days: int = PUSHED_KEEP_DAYS,
) -> int:
    """Take a pushed session off the watch and out of the training calendar.

    Two separate things have to go, which is why the first version of this looked
    like it worked and did not: `delete_workout` removes the workout from the
    saved list, and the calendar entry `schedule_workout` created stays exactly
    where it was — which is the copy that shows up in Garmin Connect under the
    plan.

    The calendar is also the better source of truth. Bookkeeping in sync_state
    only knows about pushes it saw, so a push whose state row was cleared leaves
    an orphan nothing would ever tidy. The sweep therefore runs over the
    calendar, filtered to workouts this app named, and the stored keys are
    cleaned up alongside it.

    A session goes when an activity of that sport exists on its day, or when the
    day is more than `keep_days` old. Workouts the athlete built themselves are
    never touched, whatever their age.
    """
    today = today or date.today()
    trained = _trained_by_day(store, today)

    def finished(day: date, sport: str | None) -> bool:
        sports = trained.get(day, set())
        done = (sport in sports) if sport else bool(sports)
        return done or (today - day).days > keep_days

    removed = 0
    handled: set[str] = set()
    for item in _calendar_pushes(client, today):
        if not finished(item["day"], item["sport"]):
            continue
        if item["protected"]:
            log.info("Leaving protected calendar entry %s alone",
                     item["schedule_id"])
            continue
        # Unschedule first. Deleting the workout while its calendar entry stands
        # leaves the entry pointing at nothing, which is worse than leaving both.
        if item["schedule_id"] and not client.unschedule_workout(
                item["schedule_id"]):
            continue
        workout_id = str(item["workout_id"])
        if client.delete_workout(workout_id):
            handled.add(workout_id)
            removed += 1
            log.info("Removed %s (%s) from the watch and the calendar",
                     item["title"], item["day"])

    # Then anything the bookkeeping knows about that the calendar did not cover:
    # a push that never made it onto the calendar, or one already unscheduled.
    for key, workout_id in store.states_with_prefix(PUSHED_PREFIX).items():
        sport, day = _pushed_key_parts(key)
        if day is None or not workout_id:
            store.delete_state(key)
            continue
        if not finished(day, sport or "strength"):
            continue
        if workout_id in handled:
            store.delete_state(key)
        elif client.delete_workout(workout_id):
            store.delete_state(key)
            removed += 1
    if removed:
        applog.event(default_db(),
                     f"Removed {removed} finished workout(s) from the watch")
    return removed


def _trained_by_day(store: Store, today: date) -> dict[date, set[str]]:
    """Which sports were actually done on each of the last 30 days."""
    out: dict[date, set[str]] = {}
    for a in store.activities(since=today - timedelta(days=30)):
        try:
            day = date.fromisoformat(str(a["start_date"])[:10])
        except (ValueError, TypeError):
            continue
        out.setdefault(day, set()).add(a.get("sport") or "")
    return out


# Garmin's calendar sport keys against the ones the activities table stores.
CALENDAR_SPORTS = {"strength_training": "strength", "running": "run",
                   "cycling": "bike", "swimming": "swim"}


def _calendar_pushes(client: Any, today: date) -> list[dict[str, Any]]:
    """This app's scheduled workouts from this month and last.

    Two months, because a session pushed near the end of a month is finished at
    the start of the next one and a single window would strand it.
    """
    from core.garmin_workout import APP_MARKER

    months = {(today.year, today.month)}
    previous = today.replace(day=1) - timedelta(days=1)
    months.add((previous.year, previous.month))

    out: list[dict[str, Any]] = []
    for year, month in sorted(months):
        try:
            items = client.scheduled_workouts(year, month)
        except Exception as exc:  # noqa: BLE001 - tidying never fails a sync
            log.warning("Could not read the workout calendar for %04d-%02d: %s",
                        year, month, exc)
            continue
        for item in items or []:
            if APP_MARKER not in (item.get("title") or ""):
                continue          # not ours; leave it alone
            try:
                day = date.fromisoformat(str(item.get("date"))[:10])
            except (ValueError, TypeError):
                continue
            sport_key = str(item.get("sport") or "")
            out.append({
                "schedule_id": item.get("schedule_id"),
                "workout_id": item.get("workout_id"),
                "title": item.get("title") or "",
                "day": day,
                "sport": CALENDAR_SPORTS.get(sport_key) or sport_key or None,
                "protected": bool(item.get("protected")),
            })
    return out



def recompute_metrics(
    store: Store, activity_ids: list[str] | None = None, force: bool = False
) -> int:
    """EF / steady flag / decoupling. Only the activities missing them, unless
    `force` — which is the point of `--metrics-only` after the maths changes."""
    if force:
        targets = [a["activity_id"] for a in store.activities()]
    else:
        targets = (
            activity_ids
            if activity_ids is not None
            else store.activities_missing_metrics()
        )
    if not targets:
        return 0
    wanted = set(targets)
    rows = []
    for a in store.activities():
        if a["activity_id"] not in wanted:
            continue
        stream = store.stream(a["activity_id"]) if store.has_stream(a["activity_id"]) else []
        zones = store.activity_zones(a["activity_id"])
        rows.append(compute_activity_metrics(a, stream, zones))
    return store.upsert_metrics(rows)


def _sync_locked(
    db: str | None = None,
    days: int | None = None,
    full: bool = False,
    streams: bool = True,
    wellness: bool = True,
    metrics_only: bool = False,
    stream_limit: int = 100,
    refresh_wellness: bool = False,
    prompt_mfa: Callable[[], str] | None = None,
    allow_password_login: bool = True,
    guard: GarminGuard | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    store = Store(db)
    stats: dict[str, int] = {}

    if metrics_only:
        stats["metrics"] = recompute_metrics(store, force=True)
        log.info("Recomputed metrics for %d activities", stats["metrics"])
        store.close()
        return stats

    client = GarminClient(
        prompt_mfa=prompt_mfa,
        guard=guard or GarminGuard(store),
        allow_password_login=allow_password_login,
    )
    client.connect()
    log.info("Connected to Garmin as %s", client.display_name or "(unknown)")

    # --- activities ----------------------------------------------------
    since: date | None = None
    if full:
        since = None
    elif days:
        since = date.today() - timedelta(days=days)
    else:
        latest = store.latest_activity_start()
        # Re-scan the last two days: Garmin backfills late-syncing activities.
        since = (latest.date() - timedelta(days=2)) if latest else date.today() - timedelta(days=180)

    if progress:
        progress("Fetching activities…")
    known = store.known_activity_ids()
    activities = client.fetch_activities(since=since)
    new = [a for a in activities if a["activity_id"] not in known]
    stats["activities_seen"] = len(activities)
    stats["activities_new"] = store.upsert_activities(activities)
    log.info("Stored %d activities (%d new)", len(activities), len(new))

    # Multisport legs are hidden from the activity list and the parent has no
    # heart rate, so split them out before anything is analysed.
    if progress:
        progress("Splitting multisport sessions…")
    stats["multisport_legs"] = import_multisport_children(store, client)

    # Time-in-zone: the strongest signal for whether a session was actually easy.
    if progress:
        progress("Fetching heart-rate zones…")
    zone_rows = 0
    pending_zones = store.activities_missing_zones(sorted(ZONE_SPORTS))
    for a in pending_zones[:stream_limit]:
        zone_rows += store.upsert_zones(client.fetch_zones(a["activity_id"]))
    stats["zone_rows"] = zone_rows

    # Strength sessions recorded on the watch, logged automatically.
    sets_stored, auto_logged = import_exercise_sets(store, client)
    stats["exercise_sets"] = sets_stored
    stats["strength_auto_logged"] = auto_logged

    # Conditions during each outdoor session. Backfilled from the database
    # rather than from what was new in this run, so an activity skipped by a
    # request cap or a transient error is picked up next time instead of being
    # left without weather permanently.
    if progress:
        progress("Fetching weather for outdoor sessions…")
    weather_rows = 0
    pending_weather = store.activities_missing_weather(sorted(WEATHER_SPORTS))
    for a in pending_weather[:stream_limit]:
        row = client.fetch_weather(a["activity_id"])
        if row:
            weather_rows += store.upsert_weather([row])
    stats["weather_rows"] = weather_rows

    # Laps, backfilled from the database like weather and streams, so a session
    # skipped by a request cap is picked up next run rather than left without.
    if progress:
        progress("Fetching laps…")
    lap_rows = 0
    for a in store.activities_missing_laps(sorted(LAP_SPORTS))[:stream_limit]:
        lap_rows += store.upsert_laps(client.fetch_laps(a["activity_id"]))
    stats["laps"] = lap_rows

    # Clear finished sessions off the watch. After the activity fetch on purpose:
    # "was it done" is answered by what was just pulled in.
    if progress:
        progress("Tidying pushed workouts…")
    try:
        stats["workouts_removed"] = delete_finished_workouts(store, client)
    except Exception as exc:  # noqa: BLE001 - tidying is never worth a failed sync
        log.warning("Could not tidy pushed workouts: %s", exc)
        stats["workouts_removed"] = 0

    # Body constants and lifetime records. One call each, so they are refreshed
    # every sync rather than tracked for staleness.
    profile = client.fetch_profile()
    for key, value in profile.items():
        if value is not None:
            store.set_state(f"profile_{key}", str(value))
    stats["personal_records"] = store.set_personal_records(
        client.fetch_personal_records())
    # Mirror the exercise library into the database so it is visible in the data
    # alongside what was actually logged against it.
    stats["exercise_library"] = store.sync_exercise_library(strength.library_rows())

    thresholds = client.fetch_thresholds()
    for k, v in thresholds.items():
        # The keys already read as names ("threshold_hr", "cycling_ftp"), so they
        # are stored as-is. They used to be prefixed again, producing
        # "threshold_threshold_hr" and a lookup nobody would guess.
        if v is not None:
            store.set_state(k, str(v))
    if client.display_name:
        store.set_state("athlete_name", client.display_name)

    # --- HR streams -----------------------------------------------------
    # Driven off what the DB is missing, not off what was new in this run, so a
    # previous --no-streams sync gets backfilled instead of staying blank.
    fetched_streams = 0
    if streams:
        pending = store.activities_missing_streams(sorted(STREAM_SPORTS))
        if len(pending) > stream_limit:
            log.info(
                "%d activities need HR streams; fetching the %d most recent "
                "(re-run to continue)",
                len(pending),
                stream_limit,
            )
            pending = pending[:stream_limit]
        for a in pending:
            samples = client.fetch_stream(a["activity_id"])
            if samples:
                store.replace_stream(a["activity_id"], samples)
                fetched_streams += 1
    stats["streams"] = fetched_streams
    log.info("Fetched %d HR streams", fetched_streams)

    # --- daily wellness ------------------------------------------------
    if wellness:
        if progress:
            progress("Fetching daily wellness (this is the slow part)…")
        start = since or (date.today() - timedelta(days=90))
        start = max(start, date.today() - timedelta(days=400))
        present = store.wellness_days_present()
        # Always refresh the last 3 days: Garmin finalises these late.
        refresh = {(date.today() - timedelta(days=i)).isoformat() for i in range(3)}
        wanted = list(date_range(start, date.today()))
        skip = set() if refresh_wellness else {p for p in present if p not in refresh}
        rows = client.fetch_wellness_range(wanted, skip=skip)
        stats["wellness_days"] = store.upsert_wellness(rows)
        log.info("Stored %d wellness days", stats["wellness_days"])

        preds = client.fetch_race_predictions(start, date.today())
        stats["race_predictions"] = store.upsert_race_predictions(preds)

    # --- derived metrics ------------------------------------------------
    recompute = [a["activity_id"] for a in new] + store.activities_missing_metrics()
    if fetched_streams:
        # A newly stored stream changes the steady verdict and adds decoupling.
        recompute += [a["activity_id"] for a in pending]
    if zone_rows:
        # Time-in-zone overrides the cruder steadiness heuristics.
        recompute += [a["activity_id"] for a in pending_zones[:stream_limit]]
    if stats.get("multisport_legs"):
        recompute += [a["activity_id"] for a in store.activities()]
    stats["metrics"] = recompute_metrics(store, list(dict.fromkeys(recompute)))
    store.set_state("last_sync", datetime.now().isoformat(timespec="seconds"))
    log.info("Sync complete: %s", stats)
    # Written to the database too: a scheduled sync has nobody watching stderr,
    # and "when did this last actually work" is the first question after a gap.
    applog.event(db or default_db(), "Garmin sync complete", **{
        k: v for k, v in stats.items() if isinstance(v, (int, float))})
    store.close()

    if progress:
        progress("Writing summaries…")
    try:
        stats["ai_notes"] = generate_ai_notes(db)
    except Exception as exc:  # noqa: BLE001 - summaries are never worth a failed sync
        log.warning("Could not write AI summaries: %s", exc)
        stats["ai_notes"] = 0
    return stats




def sync(
    db: str | None = None,
    days: int | None = None,
    full: bool = False,
    streams: bool = True,
    wellness: bool = True,
    metrics_only: bool = False,
    stream_limit: int = 100,
    refresh_wellness: bool = False,
    prompt_mfa: Callable[[], str] | None = None,
    allow_password_login: bool = True,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run a sync under the guard's single-flight lock and sync cooldown.

    Raises GarminBlocked when the guard refuses — a rate-limit cooldown, a spent
    request budget, or a sync that ran too recently. Callers should show that
    message rather than retrying.
    """
    if metrics_only:
        return _sync_locked(db=db, metrics_only=True)

    store = Store(db)
    guard = GarminGuard(store)
    try:
        guard.acquire_sync()
    except Exception:
        store.close()
        raise
    try:
        if progress:
            progress("Connecting to Garmin…")
        stats = _sync_locked(
            db=db, days=days, full=full, streams=streams, wellness=wellness,
            stream_limit=stream_limit, refresh_wellness=refresh_wellness,
            prompt_mfa=prompt_mfa, allow_password_login=allow_password_login,
            guard=guard, progress=progress,
        )
    finally:
        guard.release_sync()
        store.close()
    return stats


def guard_status(db: str | None = None) -> dict[str, Any]:
    with Store(db) as store:
        return GarminGuard(store).status()


def can_sync(db: str | None = None, store: Store | None = None) -> tuple[bool, str]:
    """Whether the rate guard would allow a sync right now.

    Accepts an open store because the dashboard asks this on every rerun, and
    opening a connection just to read the guard's counters cost more than the
    check itself once the database moved off local disk.
    """
    if store is not None:
        return GarminGuard(store).can_sync()
    with Store(db) as own:
        return GarminGuard(own).can_sync()
