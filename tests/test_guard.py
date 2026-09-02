"""The Garmin guard is the thing standing between a Refresh button and a locked
account, so it gets tested like it matters."""

from __future__ import annotations

import pytest

import core.garmin_guard as gg
from core.garmin_guard import GarminBlocked, GarminGuard, SyncInProgress


class FakeStore:
    """Minimal get_state/set_state, so guard state survives 'restarts'."""

    def __init__(self) -> None:
        self.kv: dict[str, str] = {}

    def get_state(self, key: str, default: str | None = None) -> str | None:
        return self.kv.get(key, default)

    def set_state(self, key: str, value: str) -> None:
        self.kv[key] = value


def test_a_429_trips_the_breaker_and_blocks_further_calls():
    guard = GarminGuard(FakeStore())
    guard.record_failure(429)
    status = guard.status()
    assert status["breaker_open"]
    assert status["breaker_minutes_left"] > 0
    try:
        guard.check()
        raise AssertionError("guard should have refused the call")
    except GarminBlocked as exc:
        assert "paused" in str(exc)


def test_retry_after_never_shortens_the_cooldown():
    """Garmin suggesting 5 seconds does not mean 5 seconds is wise."""
    guard = GarminGuard(FakeStore())
    guard.record_failure(429, retry_after=5)
    assert guard.status()["breaker_minutes_left"] >= gg.BREAKER_COOLDOWN_S / 60 - 1


def test_a_longer_retry_after_is_honoured():
    guard = GarminGuard(FakeStore())
    guard.record_failure(429, retry_after=gg.BREAKER_COOLDOWN_S * 3)
    assert guard.status()["breaker_minutes_left"] > gg.BREAKER_COOLDOWN_S / 60


def test_breaker_state_survives_a_restart():
    store = FakeStore()
    GarminGuard(store).record_failure(429)
    reborn = GarminGuard(store)  # new process, same database
    assert reborn.status()["breaker_open"]


def test_repeated_failures_trip_the_breaker():
    guard = GarminGuard(FakeStore())
    for _ in range(gg.BREAKER_FAILURE_THRESHOLD - 1):
        guard.record_failure(500)
    assert not guard.status()["breaker_open"]
    guard.record_failure(500)
    assert guard.status()["breaker_open"]


def test_one_success_clears_the_failure_run():
    guard = GarminGuard(FakeStore())
    for _ in range(gg.BREAKER_FAILURE_THRESHOLD - 1):
        guard.record_failure(500)
    guard.record_success()
    assert guard.status()["consecutive_failures"] == 0


def test_hourly_budget_stops_a_runaway_loop():
    guard = GarminGuard(FakeStore())
    for _ in range(gg.MAX_CALLS_PER_HOUR):
        guard.record_success()
    try:
        guard.check()
        raise AssertionError("hourly budget should have stopped this")
    except GarminBlocked as exc:
        assert "hourly" in str(exc).lower()


def test_sync_cooldown_blocks_a_second_refresh():
    guard = GarminGuard(FakeStore())
    guard.acquire_sync()
    guard.release_sync()
    ok, why = guard.can_sync()
    assert not ok and "recently" in why


def test_single_flight_stops_two_concurrent_syncs():
    store = FakeStore()
    first = GarminGuard(store)
    first.acquire_sync()
    second = GarminGuard(store)  # e.g. a second browser tab
    ok, why = second.can_sync()
    assert not ok and "already running" in why
    try:
        second.acquire_sync()
        raise AssertionError("the second sync should have been refused")
    except (SyncInProgress, GarminBlocked):
        pass


def test_a_stale_lock_does_not_wedge_the_app_forever():
    """A crashed sync must not lock out every future one."""
    store = FakeStore()
    guard = GarminGuard(store)
    guard.acquire_sync()
    from datetime import datetime, timedelta, timezone

    stale = datetime.now(timezone.utc) - timedelta(seconds=gg.LOCK_STALE_S + 60)
    store.set_state(gg.LOCK_KEY, stale.isoformat())
    guard._state["last_sync"] = None  # pretend the cooldown has passed
    ok, _ = guard.can_sync()
    assert ok


def test_pacing_spaces_requests_out():
    import time

    guard = GarminGuard(FakeStore())
    guard.pace()
    start = time.monotonic()
    guard.pace()
    assert time.monotonic() - start >= gg.MIN_INTERVAL_S * 0.9


def test_reset_breaker_is_available_but_explicit():
    guard = GarminGuard(FakeStore())
    guard.record_failure(429)
    assert guard.status()["breaker_open"]
    guard.reset_breaker()
    assert not guard.status()["breaker_open"]


# --------------------------------------------------------------------------
# A failed sync must not lock the athlete out
# --------------------------------------------------------------------------


def test_a_failed_sync_does_not_start_the_cooldown(tmp_path, monkeypatch):
    """Reported as "it is not letting me sync from the UI".

    The cooldown paces syncs that worked. A sync that died on an expired session
    never fetched anything, and marking it complete meant every retry for the
    next fifteen minutes was refused — so one broken session became an hour of
    being unable to try again.
    """
    from core import sync as sync_mod
    from core.store import Store

    db = str(tmp_path / "guard.db")
    Store(db).close()

    def explode(**kwargs):
        raise RuntimeError("Failed to retrieve social profile")

    monkeypatch.setattr(sync_mod, "_sync_locked", explode)
    with pytest.raises(RuntimeError):
        sync_mod.sync(db=db)

    with Store(db) as store:
        guard = GarminGuard(store)
        assert guard.sync_cooldown_remaining() == 0
        ok, why = guard.can_sync()
        assert ok, why


def test_a_successful_sync_does_start_the_cooldown(tmp_path, monkeypatch):
    from core import sync as sync_mod
    from core.store import Store

    db = str(tmp_path / "guard2.db")
    Store(db).close()
    monkeypatch.setattr(sync_mod, "_sync_locked", lambda **kw: {"activities_new": 1})
    sync_mod.sync(db=db)

    with Store(db) as store:
        guard = GarminGuard(store)
        assert guard.sync_cooldown_remaining() > 0
        assert guard.can_sync()[0] is False


def test_a_failed_sync_still_releases_the_lock(tmp_path, monkeypatch):
    """Whatever else happens, the next attempt must not meet a held lock."""
    from core import sync as sync_mod
    from core.store import Store

    db = str(tmp_path / "guard3.db")
    Store(db).close()
    monkeypatch.setattr(sync_mod, "_sync_locked",
                        lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        sync_mod.sync(db=db)
    with Store(db) as store:
        assert "already running" not in GarminGuard(store).can_sync()[1]


# --------------------------------------------------------------------------
# A new week arrives with a plan in it
# --------------------------------------------------------------------------


def test_a_week_with_no_plan_gets_one(tmp_path, monkeypatch):
    """Reported as "the plan is also not coming automatically, this week was
    all blank". Nothing created it: every Monday the week arrived empty and
    stayed empty until someone pressed a button."""
    from conftest import TODAY, build_db
    from core import sync as sync_mod
    from core.store import week_start_of

    monkeypatch.setattr("core.ai.available", lambda: False)
    monkeypatch.setattr("core.clock.today", lambda tz=None: TODAY)
    store = build_db(str(tmp_path / "plan.db"), today=TODAY)
    store.execute("DELETE FROM plans")
    store.conn.commit()
    store.close()

    assert sync_mod.ensure_week_plan(str(tmp_path / "plan.db"), today=TODAY) is True
    from core.store import Store
    with Store(str(tmp_path / "plan.db")) as s:
        plan = s.latest_plan(week_start_of(TODAY))
        assert plan is not None
        sessions = [d for d in plan["plan"]["week_plan"] if d["duration_min"] > 0]
        assert sessions, "a plan with no sessions in it is not a plan"


def test_a_plan_the_athlete_edited_is_never_overwritten(tmp_path, monkeypatch):
    from conftest import TODAY, build_db
    from core import sync as sync_mod
    from core.store import Store, week_start_of

    monkeypatch.setattr("core.ai.available", lambda: False)
    path = str(tmp_path / "manual.db")
    store = build_db(path, today=TODAY)
    store.execute("DELETE FROM plans")
    store.save_plan(week_start_of(TODAY),
                    {"week_plan": [{"day": "Mon", "sport": "bike",
                                    "duration_min": 90, "target_zone": "Z2"}],
                     "flags": [], "adjustments_made": [], "source": "manual"},
                    "manual")
    store.conn.commit()
    store.close()

    assert sync_mod.ensure_week_plan(path, today=TODAY) is False
    with Store(path) as s:
        assert s.latest_plan(week_start_of(TODAY))["source"] == "manual"


def test_a_rules_only_plan_is_upgraded_when_the_model_comes_back(tmp_path, monkeypatch):
    """The fallback is a complete week, but it is still the fallback."""
    from conftest import TODAY, build_db
    from core import sync as sync_mod
    from core.store import Store, week_start_of

    path = str(tmp_path / "upgrade.db")
    monkeypatch.setattr("core.ai.available", lambda: False)
    store = build_db(path, today=TODAY)
    store.execute("DELETE FROM plans")
    store.conn.commit()
    store.close()
    sync_mod.ensure_week_plan(path, today=TODAY)
    with Store(path) as s:
        assert s.latest_plan(week_start_of(TODAY))["source"] == "rules"

    # With no AI it stays put; with AI available it is tried again.
    assert sync_mod.ensure_week_plan(path, today=TODAY) is False
    monkeypatch.setattr("core.ai.available", lambda: True)
    monkeypatch.setattr(
        "core.ai.plan_week",
        lambda payload, **kw: {"week_plan": [
            {"day": "Wed", "sport": "bike", "duration_min": 60,
             "target_zone": "Z2", "purpose": "aerobic base",
             "exercise_ids": [], "why": "from the model"}],
            "flags": [], "adjustments_made": []})
    assert sync_mod.ensure_week_plan(path, today=TODAY) is True
    with Store(path) as s:
        assert s.latest_plan(week_start_of(TODAY))["source"] != "rules"
