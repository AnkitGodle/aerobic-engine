"""Rate limiting, backoff and a circuit breaker for Garmin Connect.

Garmin has no published API and no published limits. What is known: it returns
429 on login attempts it dislikes, it is stricter with datacenter IPs than with
home ones, and accounts do get locked. This app is a personal-use client and has
to behave like a well-mannered one, especially once a "Refresh" button exists on
a web page where it can be clicked repeatedly.

Four protections, all with state persisted in the database so that a restart, a
second browser tab, or a redeploy cannot reset them:

  1. **Pacing** — a minimum gap between requests, so a sync is a trickle rather
     than a burst.
  2. **Quotas** — hourly and daily request ceilings. A runaway loop stops before
     Garmin notices it.
  3. **A circuit breaker** — after a 429 or a run of failures, all calls are
     refused for a long cooldown. `Retry-After` is honoured when Garmin sends it.
  4. **Single-flight** — one sync at a time, plus a cooldown between syncs, so
     double-clicking Refresh cannot start two.

None of this makes hammering Garmin safe. It makes accidental hammering hard.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

log = logging.getLogger("aerobic_engine.guard")

# Deliberately conservative. A full first sync of 180 days of wellness is about
# 900 calls, which at these limits spreads over a few hours across several runs —
# slower, but invisible to Garmin.
MIN_INTERVAL_S = float(os.getenv("GARMIN_MIN_INTERVAL", "1.2"))
MAX_CALLS_PER_HOUR = int(os.getenv("GARMIN_MAX_PER_HOUR", "400"))
MAX_CALLS_PER_DAY = int(os.getenv("GARMIN_MAX_PER_DAY", "2500"))
MIN_SYNC_INTERVAL_S = float(os.getenv("GARMIN_MIN_SYNC_INTERVAL", "900"))  # 15 min
BREAKER_COOLDOWN_S = float(os.getenv("GARMIN_BREAKER_COOLDOWN", "3600"))  # 1 hour
BREAKER_FAILURE_THRESHOLD = int(os.getenv("GARMIN_FAILURE_THRESHOLD", "8"))
LOCK_STALE_S = float(os.getenv("GARMIN_LOCK_STALE", "1800"))  # 30 min

STATE_KEY = "garmin_guard"
LOCK_KEY = "garmin_sync_lock"


class GuardStore(Protocol):
    """Just the two state methods — any Store satisfies this."""

    def get_state(self, key: str, default: str | None = None) -> str | None: ...
    def set_state(self, key: str, value: str) -> None: ...


class GarminBlocked(RuntimeError):
    """The guard refused the call. Never retry this in a loop — wait it out."""

    def __init__(self, message: str, retry_at: datetime | None = None) -> None:
        super().__init__(message)
        self.retry_at = retry_at


class SyncInProgress(RuntimeError):
    """Another sync holds the lock."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _parse(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


class GarminGuard:
    """Gatekeeper for every outbound Garmin request."""

    def __init__(self, store: GuardStore | None = None) -> None:
        self.store = store
        self._last_call = 0.0
        self._state = self._load()

    # -- persistence ----------------------------------------------------
    def _load(self) -> dict[str, Any]:
        blank = {
            "hour_bucket": "", "hour_count": 0,
            "day_bucket": "", "day_count": 0,
            "failures": 0, "breaker_until": None, "last_sync": None,
            "last_429": None, "total_calls": 0,
        }
        if self.store is None:
            return blank
        raw = self.store.get_state(STATE_KEY)
        if not raw:
            return blank
        try:
            loaded = json.loads(raw)
            return {**blank, **loaded} if isinstance(loaded, dict) else blank
        except json.JSONDecodeError:
            return blank

    def _save(self) -> None:
        if self.store is not None:
            self.store.set_state(STATE_KEY, json.dumps(self._state, default=str))

    def _roll_buckets(self) -> None:
        now = _now()
        hour, day = now.strftime("%Y-%m-%dT%H"), now.strftime("%Y-%m-%d")
        if self._state.get("hour_bucket") != hour:
            self._state["hour_bucket"], self._state["hour_count"] = hour, 0
        if self._state.get("day_bucket") != day:
            self._state["day_bucket"], self._state["day_count"] = day, 0

    # -- the gate -------------------------------------------------------
    def check(self) -> None:
        """Raise if a call must not be made right now."""
        self._roll_buckets()
        until = _parse(self._state.get("breaker_until"))
        if until and _now() < until:
            wait = (until - _now()).total_seconds()
            raise GarminBlocked(
                f"Garmin requests are paused for another {wait / 60:.0f} min "
                f"(circuit breaker open after a rate-limit or repeated failures). "
                f"Waiting is the correct response — retrying sooner is what gets "
                f"an account locked.",
                retry_at=until,
            )
        if self._state["hour_count"] >= MAX_CALLS_PER_HOUR:
            raise GarminBlocked(
                f"Hourly request budget spent ({MAX_CALLS_PER_HOUR}). "
                f"Resumes at the top of the hour."
            )
        if self._state["day_count"] >= MAX_CALLS_PER_DAY:
            raise GarminBlocked(
                f"Daily request budget spent ({MAX_CALLS_PER_DAY}). Resumes tomorrow."
            )

    def pace(self) -> None:
        """Sleep just enough to keep requests spaced out."""
        gap = time.monotonic() - self._last_call
        if self._last_call and gap < MIN_INTERVAL_S:
            time.sleep(MIN_INTERVAL_S - gap)
        self._last_call = time.monotonic()

    def before_call(self) -> None:
        self.check()
        self.pace()

    def record_success(self) -> None:
        self._roll_buckets()
        self._state["hour_count"] += 1
        self._state["day_count"] += 1
        self._state["total_calls"] = int(self._state.get("total_calls", 0)) + 1
        self._state["failures"] = 0
        self._save()

    def record_failure(self, status: int | None = None, retry_after: float | None = None) -> None:
        """Count a failure, and trip the breaker on a 429 or a run of errors."""
        self._roll_buckets()
        self._state["hour_count"] += 1
        self._state["day_count"] += 1
        self._state["failures"] = int(self._state.get("failures", 0)) + 1

        trip_for: float | None = None
        if status == 429:
            self._state["last_429"] = _iso(_now())
            # Honour Retry-After, but never back off for less than the cooldown:
            # a 429 from Garmin is a warning worth taking seriously.
            trip_for = max(BREAKER_COOLDOWN_S, retry_after or 0)
        elif self._state["failures"] >= BREAKER_FAILURE_THRESHOLD:
            trip_for = BREAKER_COOLDOWN_S

        if trip_for:
            until = _now() + timedelta(seconds=trip_for)
            self._state["breaker_until"] = _iso(until)
            log.warning(
                "Garmin circuit breaker OPEN until %s (status=%s, failures=%s)",
                until.isoformat(timespec="seconds"), status, self._state["failures"],
            )
        self._save()

    def reset_breaker(self) -> None:
        """Manual override — only for when you know the cause is fixed."""
        self._state["breaker_until"] = None
        self._state["failures"] = 0
        self._save()

    # -- whole-sync gating ----------------------------------------------
    def sync_cooldown_remaining(self) -> float:
        last = _parse(self._state.get("last_sync"))
        if not last:
            return 0.0
        return max(0.0, MIN_SYNC_INTERVAL_S - (_now() - last).total_seconds())

    def can_sync(self) -> tuple[bool, str]:
        """Cheap check for the UI, so the button can explain itself."""
        try:
            self.check()
        except GarminBlocked as exc:
            return False, str(exc)
        wait = self.sync_cooldown_remaining()
        if wait > 0:
            return False, f"Synced recently — next sync available in {wait / 60:.0f} min."
        if self._locked():
            return False, "A sync is already running."
        return True, "Ready to sync."

    def _locked(self) -> bool:
        if self.store is None:
            return False
        held = _parse(self.store.get_state(LOCK_KEY))
        if not held:
            return False
        if (_now() - held).total_seconds() > LOCK_STALE_S:
            log.warning("Clearing a stale Garmin sync lock from %s", held)
            self.store.set_state(LOCK_KEY, "")
            return False
        return True

    def acquire_sync(self) -> None:
        ok, why = self.can_sync()
        if not ok:
            raise SyncInProgress(why) if "already running" in why else GarminBlocked(why)
        if self.store is not None:
            self.store.set_state(LOCK_KEY, _iso(_now()))

    def release_sync(self, mark_complete: bool = True) -> None:
        if mark_complete:
            self._state["last_sync"] = _iso(_now())
            self._save()
        if self.store is not None:
            self.store.set_state(LOCK_KEY, "")

    # -- reporting ------------------------------------------------------
    def status(self) -> dict[str, Any]:
        self._roll_buckets()
        until = _parse(self._state.get("breaker_until"))
        open_for = (until - _now()).total_seconds() if until and _now() < until else 0
        return {
            "calls_this_hour": self._state["hour_count"],
            "hour_limit": MAX_CALLS_PER_HOUR,
            "calls_today": self._state["day_count"],
            "day_limit": MAX_CALLS_PER_DAY,
            "total_calls": self._state.get("total_calls", 0),
            "consecutive_failures": self._state.get("failures", 0),
            "breaker_open": open_for > 0,
            "breaker_minutes_left": round(open_for / 60, 1),
            "last_429": self._state.get("last_429"),
            "last_sync": self._state.get("last_sync"),
            "sync_cooldown_min": round(self.sync_cooldown_remaining() / 60, 1),
        }
