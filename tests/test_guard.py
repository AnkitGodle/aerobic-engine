"""The Garmin guard is the thing standing between a Refresh button and a locked
account, so it gets tested like it matters."""

from __future__ import annotations

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
