"""Sync orchestration: Garmin -> database. No Streamlit, no CLI concerns.

Lives in `core/` because two callers need it — `scripts/fetch.py` for a local or
scheduled run, and the dashboard's Refresh button. Every Garmin call inside here
goes through `GarminGuard`, and the whole sync takes the guard's single-flight
lock, so two callers cannot run at once.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any, Callable

from core import strength
from core.analysis import compute_activity_metrics
from core.garmin_client import GarminClient, date_range
from core.garmin_guard import GarminGuard
from core.store import DEFAULT_DB, Store

log = logging.getLogger("iron_coach.sync")

STREAM_SPORTS = {"run", "bike", "swim", "brick"}
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


def import_exercise_sets(store: Store, client: GarminClient) -> tuple[int, int]:
    """Pull watch-recorded strength sets and turn them into strength_log rows."""
    sets_stored = logged = 0
    already = store.strength_days_logged()
    for act in store.strength_activities_missing_sets():
        raw = client.fetch_exercise_sets(act["activity_id"])
        if not raw:
            continue
        for r in raw:
            r["exercise_id"] = strength.map_garmin_exercise(
                r.get("garmin_category"), r.get("garmin_name")
            )
        store.replace_exercise_sets(act["activity_id"], raw)
        sets_stored += len(raw)
        unmapped = [r for r in raw if not r.get("exercise_id")]
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
    db: str = DEFAULT_DB,
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

    thresholds = client.fetch_thresholds()
    for k, v in thresholds.items():
        if v is not None:
            store.set_state(f"threshold_{k}", str(v))
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
    store.close()
    return stats




def sync(
    db: str = DEFAULT_DB,
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


def guard_status(db: str = DEFAULT_DB) -> dict[str, Any]:
    with Store(db) as store:
        return GarminGuard(store).status()


def can_sync(db: str = DEFAULT_DB) -> tuple[bool, str]:
    with Store(db) as store:
        return GarminGuard(store).can_sync()
