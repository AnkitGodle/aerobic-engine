"""Aerobic Engine — training dashboard.

Four pages, in the order you actually need them: what to do today, whether it is
working, what is coming, and the raw record. All logic lives in `core/`; this file
reads the database, draws, and collects input. Presentation primitives are in
`app/ui.py` so this file stays about structure rather than styling.

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import logging
import os
import sys
import sqlite3
import threading
import time
import zlib
from uuid import uuid4
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

# Before the first `core` import, and for a reason worth stating: a hosted
# container can pull a new revision, re-execute this script, and still hold the
# previous revision's `core` modules — so an import of a function that exists in
# the checked-out source fails and the app is down until someone reboots it. See
# app/freshness.py.
from app.freshness import purge_stale_modules, stamp_modules  # noqa: E402

# INFO, not WARNING: a successful self-heal is not a problem, and at WARNING it
# was stored in the database — which during a session of edits meant 47 rows of
# "reloading" burying everything else in the App log. The one row worth keeping
# is written as an event below, once the imports it needed have happened.
_reloaded = purge_stale_modules(log=logging.getLogger("aerobic_engine.ui").info)

import app.ui as ui  # noqa: E402
from core import (  # noqa: E402
    ai, applog, bugs as bugs_mod, goal as goal_mod, insights, planner,
    rules as rules_mod, strength, sync as sync_mod, visits as visits_mod,
)
from core.analysis import (  # noqa: E402
    ACWR_HIGH,
    ACWR_LOW,
    DEW_POINT_HARD_C,
    ZONE_LABELS,
    aerobic_ceiling_options,
    consistency,
    load_ramp,
    ramp_verdict,
    streak,
    weather_effect,
    weekly_zone_minutes_from_streams,
    zone_bounds,
    zone_bounds_with_ceiling,
    zone_distribution_from_streams,
    baseline_trend,
    hr_points,
    hr_trend,
    cadence_stats,
    lap_drift,
    lap_pace_spread,
    ef_data_status,
    ef_points,
    polarisation,
    polarisation_from_streams,
    recovery_signals,
    totals,
    volume_forecast,
    week_summaries,
    zone_distribution,
)
from core.auth import PinGate, session_expired  # noqa: E402
from core import garmin_workout  # noqa: E402
from core.garmin_client import GarminClient  # noqa: E402
from core.garmin_guard import GarminBlocked  # noqa: E402
from core.schemas import DAYS, ENDURANCE_SPORTS, SPORTS, Checkin, PlanDay, WeekPlan  # noqa: E402
from core.store import Store, default_db, is_postgres, week_start_of  # noqa: E402

# Stamped here, with every app module loaded and agreeing with the files it came
# from. That is what lets the guard above tell "changed since I imported it" from
# "first time I have looked at it".
stamp_modules()

try:  # psycopg is only installed where Postgres is used
    from psycopg import InterfaceError, OperationalError
except ImportError:  # pragma: no cover - SQLite-only environments
    class InterfaceError(Exception): ...
    class OperationalError(Exception): ...

load_dotenv()
log = logging.getLogger("aerobic_engine.ui")
# Warnings and above are written to the database as well as stderr. On the hosted
# app the on-screen message is redacted and the container log is behind another
# login, so without this a failure on the phone is unreadable exactly when it
# matters. Idempotent, and it fails quietly if the table is unreachable.
applog.install(default_db())
if _reloaded:
    # Reloading the modules is only half of it. st.cache_resource is keyed on a
    # function's module and name rather than its identity, so the cached
    # connection wrapper survives the reload — holding a Store built from the
    # *previous* revision's class. The hosted app died on exactly that: new
    # script calling s.activities(include_imported=True) against an old Store,
    # which is a TypeError, reported as "could not open the database".
    for cache in (st.cache_resource, st.cache_data):
        try:
            cache.clear()
        except Exception as exc:  # noqa: BLE001 - never worth failing a page load
            log.warning("Could not clear the %s cache: %s", cache, exc)
    # One row per deploy that actually needed the guard, with what moved.
    applog.event(default_db(), "Reloaded app modules after a deploy",
                 modules=", ".join(_reloaded))
st.set_page_config(page_title="Aerobic Engine", page_icon="📈", layout="wide",
                   initial_sidebar_state="collapsed")

EMOJI = ui.SPORT_EMOJI
SPORT_COLOR = ui.SPORT_COLOR
ZONE_COLOR = ui.ZONE_COLOR
TONE = ui.TONE_COLOR

COLUMN_NAMES = {
    "start_date": "Date", "sport": "Sport", "name": "Activity",
    "minutes": "Minutes", "km": "Distance (km)", "avg_pace": "Pace",
    "avg_hr": "Avg HR", "max_hr": "Max HR", "avg_power_w": "Power (W)",
    "training_load": "Load", "ef": "Efficiency", "is_steady": "Steady?",
    "steady_reason": "Why not", "day": "Date", "resting_hr": "Resting HR",
    "hrv_last_night": "HRV", "sleep_hours": "Sleep (h)",
    "training_readiness": "Readiness", "vo2max_run": "VO2max",
    "training_status": "Status", "exercise_id": "Exercise", "sets": "Sets",
    "reps": "Reps", "hold_s": "Hold (s)", "load_kg": "Load (kg)",
    "clean": "Completed", "pain": "Pain", "notes": "Notes", "sleep": "Sleep",
    "soreness": "Soreness", "motivation": "Motivation",
    "time_available_min": "Time (min)", "garmin_name": "Watch exercise",
    "garmin_category": "Category", "duration_s": "Seconds",
    "week": "Week", "total minutes": "Minutes", "load": "Load",
    "rest days": "Rest days", "strength": "Strength",
}
DROP_COLS = {
    "activity_id", "raw_json", "ingested_at", "computed_at", "parent_activity_id",
    "is_multisport_parent", "id", "created_at", "garmin_type", "moving_s",
    "distance_m", "norm_power_w", "aerobic_te", "anaerobic_te", "pool_length_m",
    "hr_samples", "ef_first_half", "ef_second_half", "rpe", "set_index",
    "ef_metric", "readiness_level", "decoupling_pct", "elevation_gain_m",
    "calories", "activity_id",
}


# Columns that hold a date and should be shown day-first rather than as the ISO
# string the database stores.
DATE_COLUMNS = {"start_date", "day", "date", "achieved_at", "start_time"}


def table(df: pd.DataFrame, **kw) -> None:
    out = df.drop(columns=[c for c in df.columns if c in DROP_COLS])
    # Formatted here, once, rather than at each call site: every table that
    # shows a date was showing 2026-08-24, and the ISO form is the storage
    # format, not the reading format.
    for column in out.columns:
        if column in DATE_COLUMNS:
            out[column] = out[column].map(
                lambda v: day_label(v) if v is not None and str(v).strip() else "")
    out = out.rename(columns={c: COLUMN_NAMES.get(c, c.replace("_", " ").capitalize())
                              for c in out.columns})
    st.dataframe(out, width="stretch", hide_index=True, **kw)


# --------------------------------------------------------------------------
# data + access
# --------------------------------------------------------------------------


def _secret(name: str, default: str = "") -> str:
    val = os.getenv(name, "")
    if val:
        return val
    try:
        return str(st.secrets.get(name, default))
    except Exception:
        return default


def db_path() -> str:
    # DATABASE_URL wins deliberately. A hosted deployment sets it, and a stray
    # AEROBIC_ENGINE_DB left in the environment must not quietly point the app
    # at an ephemeral local file instead — losing that database means re-pulling
    # months of Garmin history, which is the traffic that gets accounts flagged.
    return default_db()


def db_stamp() -> float:
    """Cache key for load(): must change whenever the data could have."""
    target = db_path()
    if is_postgres(target):
        # A remote database has no mtime, so a fingerprint of the data stands in:
        # the sync marker, the activity count and the newest ingest time. One
        # cheap query per rerun. The marker alone was not enough — it only moves
        # on a Garmin sync, so importing history or repairing rows from a script
        # left every open dashboard serving what it had already cached.
        try:
            marker = with_store(lambda s: s.data_stamp()) or ""
            return float(zlib.crc32(marker.encode("utf-8")))
        except Exception:  # noqa: BLE001 - a dead cache key beats a dead page
            log.warning("could not read the data stamp for the cache key")
            return 0.0
    p = Path(target)
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_data(show_spinner=False)
def load(stamp: float) -> dict:  # noqa: ARG001 - stamp is the cache key
    def read(s: Store) -> dict:
        state = s.get_states(("last_sync", "athlete_name", "threshold_hr",
                              "running_ftp", "cycling_ftp", "aerobic_ceiling_bpm"))
        # Body constants, stored one key per field by the sync.
        fields = ("weight_kg", "height_cm", "age", "gender", "vo2max_run",
                  "vo2max_bike", "threshold_hr", "moderate_zone", "vigorous_zone")
        raw = s.get_states(tuple(f"profile_{f}" for f in fields))
        profile = {f: raw.get(f"profile_{f}") for f in fields}
        return {
            "activities": s.activities(),
            "all_activities": s.activities(include_parents=True),
            # Garmin plus anything imported from a Strava export. Used by the
            # lifetime totals and the log, and by nothing that talks to the AI —
            # see core/strava_import.py for why that line exists.
            "history": s.activities(include_imported=True),
            "wellness": s.wellness(), "race": s.race_predictions(),
            "zones": s.zones(), "strength": s.strength_log(),
            "sets": s.exercise_sets(), "checkins": s.checkins(limit=60),
            "counts": s.counts(), "targets": s.targets(), "notes": s.notes(),
            "last_sync": state["last_sync"],
            "name": state["athlete_name"] or "",
            "thresholds": {k: state[k]
                           for k in ("threshold_hr", "running_ftp", "cycling_ftp")},
            "aerobic_ceiling": state["aerobic_ceiling_bpm"],
            "profile": profile,
            "records": s.personal_records(),
            "weather": s.weather(),
            "laps": s.laps(),
            "plan": s.latest_plan(week_start_of(date.today())),
        }

    return with_store(read)


def refresh() -> None:
    load.clear()


class _SharedStore:
    """One reused database connection for the whole app, behind a lock.

    Opening a connection per call was fine against a local file and expensive
    against a managed Postgres: about half a second each, and the PIN gate alone
    makes several, which is why unlocking felt broken.

    Two details make reuse safe. st.cache_resource is global rather than
    per-session and a psycopg connection must not be used from two threads at
    once, so every access is serialised — a fine trade for a personal dashboard.
    And an idle Neon project scales to zero, so a handle held between page views
    can come back dead; a failed call reopens once and retries rather than
    poisoning every read after it.
    """

    def __init__(self, target: str) -> None:
        self.target = target
        self._lock = threading.Lock()
        self._store: Store | None = None

    def _open(self) -> Store:
        if self._store is None:
            self._store = Store(self.target)
        return self._store

    def _end_transaction(self) -> None:
        """Close the read transaction this call opened.

        This is not tidiness. With autocommit off, the first SELECT opens a
        transaction that stays open until something ends it, so a reused
        connection sits "idle in transaction" between page views — holding a
        snapshot and an ACCESS SHARE lock on everything it touched.

        That is not a slow query, it is a full stop: an ALTER TABLE from a
        migration queues behind it, and because Postgres queues lock waiters in
        order, every later read of that table queues behind the ALTER. One idle
        read had blocked hr_streams for ten minutes and taken the Log page down
        with it. Writes have already committed inside tx(), so this only ever
        discards a read snapshot.
        """
        if self._store is None or not self._store.postgres:
            return
        try:
            self._store.conn.rollback()
        except Exception:  # noqa: BLE001 - a dead handle is reopened next call
            pass

    # Retried only for the failures reopening can actually fix. Retrying
    # everything meant a plain AttributeError in the read function came back as
    # "could not open the database", which sends you looking at DATABASE_URL for
    # a bug that is in the code.
    RETRYABLE = (OSError, sqlite3.Error, InterfaceError, OperationalError)

    def run(self, fn):
        with self._lock:
            try:
                try:
                    return fn(self._open())
                finally:
                    self._end_transaction()
            except self.RETRYABLE:
                log.info("database handle went stale; reopening")
                try:
                    if self._store is not None:
                        self._store.close()
                except Exception:  # noqa: BLE001 - already discarding it
                    pass
                self._store = None
                return fn(self._open())


@st.cache_resource(show_spinner=False)
def shared_store(target: str) -> _SharedStore:
    return _SharedStore(target)


def with_store(fn):
    """Run `fn(store)` on the shared connection."""
    return shared_store(db_path()).run(fn)


class _StateStore:
    """The small key/value interface PinGate needs, on the shared connection."""

    def get_state(self, key: str, default: str | None = None) -> str | None:
        return with_store(lambda s: s.get_state(key, default))

    def set_state(self, key: str, value: str) -> None:
        with_store(lambda s: s.set_state(key, value))


def pin_gate() -> PinGate:
    return PinGate(_StateStore(), pin_hash=_secret("REFRESH_PIN_HASH"),
                   salt=_secret("REFRESH_PIN_SALT"), plaintext=_secret("REFRESH_PIN"))


def writes_allowed() -> bool:
    gate = pin_gate()
    if not gate.configured:
        return True
    at = st.session_state.get("unlocked_at")
    if session_expired(at, time.time()):
        st.session_state.pop("unlocked_at", None)
        return False
    return True


def read_pin_gate() -> PinGate:
    """The gate on reading the dashboard at all.

    A separate PIN from the write one, and separately rate-limited: they are
    guessed at independently, and one counter would mean strangers failing at
    the front door locking the owner out of their own controls.

    This replaced a plaintext DASHBOARD_PASSWORD compared with
    compare_digest. Same PBKDF2 hashing as the write PIN, so the secret stored
    on the host is not the secret you type, and wrong guesses now cost time
    instead of nothing.
    """
    return PinGate(_StateStore(),
                   pin_hash=_secret("READ_PIN_HASH"),
                   salt=_secret("READ_PIN_SALT"),
                   plaintext=_secret("READ_PIN"),
                   attempts_key="read_pin_attempts")


def read_gate() -> bool:
    gate = read_pin_gate()
    if not gate.configured or st.session_state.get("authed"):
        return True

    ui.brand("Aerobic Engine", "Enter your PIN to continue.")
    wait = gate.lockout_remaining()
    if wait > 0:
        st.error(f"Too many wrong PINs. Try again in {wait:.0f}s.")
        return False
    with st.form("read_gate", clear_on_submit=True):
        pin = st.text_input("PIN", type="password", label_visibility="collapsed",
                            placeholder="PIN")
        if st.form_submit_button("Enter", type="primary"):
            ok, message = gate.verify(pin)
            del pin
            if ok:
                st.session_state["authed"] = True
                st.rerun()
            st.error(message)
    return False


# --------------------------------------------------------------------------
# small formatters
# --------------------------------------------------------------------------


# The athlete trains in India, and a cloud host runs on UTC — so the cutoff is
# anchored to a real timezone rather than to wherever the server happens to be.
LOCAL_TZ = ZoneInfo(os.getenv("LOCAL_TZ", "Asia/Kolkata"))

# The athlete's public Strava profile. A hyperlink and nothing else: no API, no
# import, no data crossing in either direction. Strava's API terms forbid using
# their data with a language model, and this app's whole planning layer is a
# language model — so the profile stays a link out, which those terms do not
# touch. Override with STRAVA_PROFILE_URL if the profile ever moves.
STRAVA_PROFILE_URL = os.getenv(
    "STRAVA_PROFILE_URL", "https://www.strava.com/athletes/71829400")
INSTAGRAM_PROFILE_URL = os.getenv(
    "INSTAGRAM_PROFILE_URL", "https://www.instagram.com/ankitgodle/")

# Label, URL, brand colour. Instagram's mark is a gradient; its pink is the one
# solid colour that still reads as Instagram at chip size.
REPO_URL = "https://github.com/AnkitGodle/aerobic-engine"

PROFILE_LINKS: tuple[tuple[str, str, str], ...] = (
    ("Strava", STRAVA_PROFILE_URL, "#FC5200"),
    ("Instagram", INSTAGRAM_PROFILE_URL, "#E1306C"),
    # The repo belongs beside them: it is the same kind of link out, and the
    # code being readable is part of what the page is claiming.
    ("GitHub", REPO_URL, "#8C9AA8"),
)
EVENING_CUTOFF_HOUR = int(os.getenv("EVENING_CUTOFF_HOUR", "19"))


def local_now() -> datetime:
    return datetime.now(LOCAL_TZ)


def training_focus(today: date) -> tuple[date, bool]:
    """Which day the athlete can still act on.

    Suggesting a 90-minute ride at half past nine at night is useless advice, so
    after the cutoff the focus rolls to tomorrow. Only the "what to do next" card
    moves; the week strips and every analysis still key off the real date.
    """
    now = local_now()
    if today == now.date() and now.hour >= EVENING_CUTOFF_HOUR:
        return today + timedelta(days=1), True
    return today, False


def hm(minutes: float) -> str:
    h, m = divmod(int(round(minutes or 0)), 60)
    return f"{h}h {m:02d}m" if h else f"{m}m"


def day_label(iso: object, year: bool = False) -> str:
    raw = str(iso)
    # Garmin dates a personal record with a Unix timestamp in milliseconds, and
    # Python 3.11's fromisoformat is lenient enough to read "1787121654" as
    # 16 December 1787 rather than rejecting it — so every personal record was
    # dated to the eighteenth century. Digits-only values are epochs, not dates.
    if raw.isdigit() and len(raw) in (10, 13):
        try:
            seconds = int(raw) / (1000.0 if len(raw) == 13 else 1.0)
            d = datetime.fromtimestamp(seconds, LOCAL_TZ).date()
            return d.strftime("%a %d-%m-%Y") if year else d.strftime("%a %d-%m")
        except (OverflowError, OSError, ValueError):
            return raw
    try:
        d = date.fromisoformat(raw[:10])
    except (TypeError, ValueError):
        return raw
    # Day-month-year and a 12-hour clock throughout, the way this athlete reads
    # dates. The weekday stays in front of it: training is planned by day of the
    # week, so "Mon" is the part that answers the question fastest.
    return d.strftime("%a %d-%m-%Y") if year else d.strftime("%a %d-%m")


def fmt_stamp(raw: object) -> str:
    """An ISO timestamp as "24-08-2026, 9:05 am", in your timezone.

    A timestamp with an offset is converted; one without is assumed to be local
    already. That split is what makes the sidebar honest: a sync run from
    Streamlit Cloud records UTC and a sync run from a laptop in India records
    IST, and read literally the same moment looked five and a half hours apart.
    Anything stored before that was fixed is naive and was written locally, so
    leaving those alone is also correct.
    """
    text = str(raw or "").strip()
    if not text:
        return "never"
    try:
        when = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text[:16].replace("T", " ")
    if when.tzinfo is not None:
        when = when.astimezone(LOCAL_TZ).replace(tzinfo=None)
    # The year is dropped in the current year. On a sync timestamp it is almost
    # always noise — you know what year it is — and it only earns its space when
    # the reading is genuinely old.
    pattern = "%d-%m, %I:%M %p" if when.year == date.today().year else \
              "%d-%m-%Y, %I:%M %p"
    return when.strftime(pattern).replace(" 0", " ").lower()


def pace_str(sport: str, dist_m: float | None, dur_s: float | None) -> str:
    if not dist_m or not dur_s or dist_m <= 0:
        return "—"
    if sport == "bike":
        return f"{dist_m / dur_s * 3.6:.1f} km/h"
    if sport == "swim":
        per = dur_s / (dist_m / 100.0)
        return f"{int(per // 60)}:{int(per % 60):02d}/100m"
    per = dur_s / (dist_m / 1000.0)
    return f"{int(per // 60)}:{int(per % 60):02d}/km"


def insight_banner(page: str, data: dict, today: date) -> None:
    """Deterministic headline and bullets, plus the AI paragraph if one was
    written at sync time. No live model call: see `core.sync.generate_ai_notes`
    for why (Streamlit runs every tab body on every render)."""
    ins = insights.for_page(page, data, today)
    if ins is None:
        return
    tone = {"success": "good", "warning": "caution", "error": "bad",
            "info": "neutral"}[ins.tone]
    prose = (data.get("notes") or {}).get(page, {}).get("text")
    # The written summary is visible, not folded away. The reason this app
    # generates it at all is so the charts are optional — a summary behind a
    # click is a summary nobody reads. The deterministic bullets stay collapsed:
    # they are the workings, and the paragraph is the answer.
    ui.banner(ins.headline, prose or "", tone)
    if ins.bullets:
        with st.expander("The numbers behind it"):
            for b in ins.bullets:
                st.markdown(f"- {b}")


def chart_ai_note(key: str, notes: dict) -> None:
    """The stored one-line reading for a chart, if the last sync wrote one."""
    text = (notes or {}).get(f"chart:{key}", {}).get("text")
    if text:
        st.caption(f"✦ {text}")


# --------------------------------------------------------------------------
# PAGE 1 — Today
# --------------------------------------------------------------------------


def exercise_howto(ex, prescription=None) -> None:
    """How to actually perform one exercise.

    A prescription without a technique is not something you can follow, and bad
    technique is how strength work causes the injury it was meant to prevent.
    """
    dose = ""
    if prescription is not None:
        if prescription.hold_s:
            dose = f"{prescription.sets} x {prescription.hold_s}s"
        else:
            dose = f"{prescription.sets} x {prescription.reps}"
        if prescription.load_kg:
            dose += f" @ {prescription.load_kg:g} kg"
    elif ex.rep_range:
        dose = f"{ex.sets} x {ex.rep_range[0]}–{ex.rep_range[1]}"
    elif ex.hold_range:
        dose = f"{ex.sets} x {ex.hold_range[0]}–{ex.hold_range[1]}s"

    head = f"**{ex.name}** — {dose}" if dose else f"**{ex.name}**"
    if ex.unilateral:
        head += "  ·  per side"
    st.markdown(head)
    meta = " · ".join(x for x in (ex.focus, ex.tempo and f"tempo {ex.tempo}") if x)
    if meta:
        st.caption(meta)
    # The dose stays on the page; the technique folds away. Five exercises spelled
    # out in full is 1,800px of instructions on the one page you open every day,
    # and by the fourth week you need the numbers, not the tutorial. One click
    # still brings it back, and the mistake to avoid is the line worth keeping.
    with st.expander("How to do it"):
        if ex.setup:
            st.markdown(f"**Set up:** {ex.setup}")
        if ex.steps:
            st.markdown("\n".join(f"{i}. {step}"
                                  for i, step in enumerate(ex.steps, 1)))
        if ex.why:
            st.caption(f"Why it matters: {ex.why}")
        if ex.load_note:
            st.caption(f"Progressing: {ex.load_note}")
    if ex.mistakes:
        st.caption(f"Avoid: {ex.mistakes}")


def send_to_watch(prescriptions: list, day: date, label: str) -> None:
    """Push the session to Garmin as a named workout.

    The point is not convenience. The watch counts reps well but usually does
    not know which exercise it is watching, so sets come back unnamed and have
    to be assigned by hand before the progression can use them. A named workout
    removes that step entirely.

    PIN-gated and once-per-day: this writes to the Garmin account, and pushing
    twice leaves two identical workouts on the watch.
    """
    key = f"workout_pushed_{day.isoformat()}"
    already = with_store(lambda st: st.get_state(key))
    unlocked = writes_allowed()

    if already:
        st.caption(f"✓ Sent to your watch — look for “{label}” under saved "
                   f"workouts. Starting it there names every set for you.")
        if not st.button("Send again", key=f"resend_{day}", disabled=not unlocked):
            return
    elif not st.button("Send this session to my watch", type="primary",
                       key=f"send_{day}", disabled=not unlocked,
                       help=None if unlocked else "Unlock with your PIN first."):
        if not unlocked:
            st.caption("🔒 Unlock with your PIN to send workouts to the watch.")
        return
    if not writes_allowed():
        return

    try:
        with st.spinner("Sending…"):
            # Never allow an SSO login from here. The dashboard may be hosted,
            # and a password login from a datacenter IP is the single riskiest
            # call this app can make. A stale token must fail loudly instead.
            client = GarminClient(allow_password_login=False)
            created = garmin_workout.push(
                client.connect(), prescriptions, label, on_date=day.isoformat())
        with_store(lambda st_: st_.set_state(key, str(created.get("workoutId", ""))))
        st.success(
            f"Sent. On the watch: START → Strength → it should offer “{label}”. "
            f"Every set arrives already named, so nothing needs assigning "
            f"afterwards. Once you have logged it, the next sync takes it back "
            f"off the watch so tomorrow's session is the one on offer.")
        refresh()
    except GarminBlocked as exc:
        st.warning(str(exc))
    except Exception as exc:  # noqa: BLE001
        log.exception("Could not send the workout")
        st.error(f"Could not send it ({type(exc).__name__}). The session is "
                 f"unchanged — log it by hand and nothing is lost.")


def send_session_to_watch(day_plan: dict, day: date) -> None:
    """Push a run or ride to the watch with its heart-rate range.

    Time-based rather than distance-based, because the ceiling is the point:
    chasing a distance is what turns an easy session into a moderate one.
    """
    sport = day_plan.get("sport")
    if sport not in garmin_workout.ENDURANCE_SPORT:
        if sport in ("swim", "brick"):
            st.caption(
                "Swims and bricks are not sent to the watch. A Garmin swim "
                "workout is built from pool length and stroke rather than "
                "minutes, and wrist heart rate in water is too unreliable to "
                "target; a brick is two sports in one session, which is a "
                "different kind of workout again.")
        return

    minutes = int(day_plan.get("duration_min") or 0)
    target = day_plan.get("target_hr") or ""
    if minutes <= 0:
        return
    band = garmin_workout.parse_hr_target(target)
    key = f"workout_pushed_{sport}_{day.isoformat()}"
    already = with_store(lambda st_: st_.get_state(key))
    unlocked = writes_allowed()
    label = f"{sport.title()} {minutes}m {garmin_workout.APP_MARKER}"

    if already:
        st.caption(f"✓ Sent — “{label}” is on your watch under {sport} workouts.")
        if not st.button("Send again", key=f"resend_{sport}_{day}",
                         disabled=not unlocked):
            return
    elif not st.button(f"Send this {sport} to my watch", key=f"send_{sport}_{day}",
                       disabled=not unlocked):
        if not unlocked:
            st.caption("🔒 Unlock with your PIN to send workouts to the watch.")
        return
    if not writes_allowed():
        return

    try:
        with st.spinner("Sending…"):
            created = garmin_workout.push_endurance(
                GarminClient(allow_password_login=False).connect(),
                sport, minutes, target, label,
                purpose=day_plan.get("purpose") or "", on_date=day.isoformat())
        with_store(lambda st_: st_.set_state(key, str(created.get("workoutId", ""))))
        st.success(
            f"Sent. On the watch: START → {sport.title()} → “{label}”. "
            + (f"It will hold you to {band[0]}-{band[1]} bpm and buzz if you "
               f"drift out." if band else
               "No heart-rate range was set for this session, so it is timed "
               "only.")
            + " It is removed from the watch once the session is logged.")
        refresh()
    except GarminBlocked as exc:
        st.warning(str(exc))
    except Exception as exc:  # noqa: BLE001
        log.exception("Could not send the endurance workout")
        st.error(f"Could not send it ({type(exc).__name__}). The session is "
                 f"unchanged.")


def strength_howto_block(exercise_ids: list[str], log_rows: list[dict],
                         session_index: int = 0, intensity: float = 1.0,
                         day: date | None = None) -> None:
    """The full session, with instructions, for the day it is scheduled.

    The prescription comes from the plan's own exercise list when it has one.
    It used to display the plan's exercises and send the template's, which is
    how a session on the watch could contain exercises that were never on the
    page — and at the template's volume rather than the plan's.
    """
    ids = [e for e in exercise_ids if e in strength.EXERCISES]
    if ids:
        prescriptions = [strength.next_prescription(e, log_rows, intensity)
                         for e in ids]
        label = f"Legs {garmin_workout.APP_MARKER}"
    else:
        # No list on the plan — fall back to where the cycle has got to.
        prescriptions = strength.build_session(log_rows,
                                               session_index=session_index,
                                               intensity=intensity)
        ids = [x.exercise_id for x in prescriptions]
        label = (f"Legs {chr(65 + session_index % 3)} "
                 f"{garmin_workout.APP_MARKER}")
    presc = {x.exercise_id: x for x in prescriptions}
    if not ids:
        return
    # The session's own length, which is not always the plan's nominal figure: a
    # deload trims the accessory work, and a hand-edited plan can carry any
    # number at all. Saying both beats silently contradicting the week strip.
    minutes = strength.session_minutes(prescriptions)
    ui.section(f"How to do today's session · {minutes} min",
               ("Reduced for a deload week. " if intensity < 1 else "")
               + "Slow and controlled beats heavy. Stop a set if something "
                 "sharp appears — soreness is fine, pain is not.")
    # Exactly what is listed below, in this order, at these numbers.
    send_to_watch(prescriptions, day or date.today(), label)
    for i, eid in enumerate(ids):
        with st.container(border=True):
            exercise_howto(strength.EXERCISES[eid], presc.get(eid))
        if i < len(ids) - 1:
            st.write("")


@st.cache_data(show_spinner=False, ttl=1800)
def route_points(stamp: float, activity_id: str,
                 keep: int = 160) -> list[tuple[float, float]]:  # noqa: ARG001
    """The route, thinned to something a browser can draw instantly.

    A thousand GPS points is a thousand points more than a 300px outline needs.
    Cached per activity, so opening the same session twice costs one query once.
    """
    try:
        stream = with_store(lambda st_: st_.stream(str(activity_id)))
    except Exception as exc:  # noqa: BLE001 - a missing route is not an error
        log.info("Could not read the route for %s: %s", activity_id, exc)
        return []
    coords = [(float(r["lat"]), float(r["lon"])) for r in stream
              if r.get("lat") is not None and r.get("lon") is not None]
    if len(coords) <= keep:
        return coords
    step = len(coords) / keep
    return [coords[int(i * step)] for i in range(keep)]


def route_figure(coords: list[tuple[float, float]]):
    """The shape of the route, as a line. No tiles, no WebGL, no map at all.

    `st.map` renders through deck.gl, which pulls WebGL and a basemap and locked
    the browser for several seconds on a page that might show six of them. The
    outline is what identifies a route at a glance — the streets underneath it
    are decoration you already know.
    """
    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]
    fig = go.Figure()
    fig.add_scatter(x=lons, y=lats, mode="lines",
                    line=dict(color=TONE["good"], width=2.4),
                    hoverinfo="skip")
    fig.add_scatter(x=[lons[0]], y=[lats[0]], mode="markers", name="start",
                    marker=dict(size=9, color=TONE["good"]), hoverinfo="skip")
    fig.add_scatter(x=[lons[-1]], y=[lats[-1]], mode="markers", name="finish",
                    marker=dict(size=9, color=TONE["bad"]), hoverinfo="skip")
    # Degrees of longitude shrink with latitude, so an unconstrained plot
    # stretches a route east-west and it stops looking like the place you ran.
    import math
    fig.update_yaxes(scaleanchor="x",
                     scaleratio=1.0 / max(math.cos(math.radians(lats[0])), 0.1))
    fig.update_layout(showlegend=False, xaxis=dict(visible=False),
                      yaxis=dict(visible=False), margin=dict(t=4, b=4, l=4, r=4),
                      height=230, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False,
                                                  "staticPlot": True})


def session_recap(act: dict, data: dict) -> None:
    """What a finished session was: the numbers, the route, the splits.

    On Today rather than only on the Log page, because the question "how did
    that go" arrives about ten seconds after finishing, and making someone
    navigate to a different page to answer it is how they stop asking.
    """
    minutes = (act.get("duration_s") or 0) / 60.0
    km = (act.get("distance_m") or 0) / 1000.0
    cells = [
        {"label": "Time", "value": hm(minutes)},
        {"label": "Distance", "value": f"{km:.2f} km" if km else "—"},
        {"label": "Pace", "value": pace_str(act["sport"], act.get("distance_m"),
                                            act.get("duration_s"))},
        {"label": "Heart rate",
         "value": f"{act['avg_hr']:.0f}" if act.get("avg_hr") else "—",
         "note": f"max {act['max_hr']:.0f}" if act.get("max_hr") else ""},
    ]
    ceiling = data.get("aerobic_ceiling")
    if act.get("avg_hr") and ceiling:
        over = float(act["avg_hr"]) - float(ceiling)
        cells.append({"label": "Against your ceiling",
                      "value": f"{over:+.0f} bpm",
                      "note": "easy" if over <= 0 else "harder than easy",
                      "tone": "good" if over <= 0 else "caution"})
    if act.get("training_load"):
        cells.append({"label": "Load", "value": f"{act['training_load']:.0f}",
                      "note": "Garmin's own"})
    ui.figures(cells)

    coords = route_points(db_stamp(), str(act["activity_id"]))
    left, right = st.columns([1.25, 1], gap="medium")
    with left:
        if coords:
            route_figure(coords)
        else:
            st.caption("No route for this one — indoors, or recorded before "
                       "coordinates were stored.")
    with right:
        bands = (session_zone_minutes(db_stamp(), str(act["activity_id"]),
                                      int(float(ceiling))) if ceiling else {})
        if sum(bands.values()) > 0:
            st.markdown("**Where the time went**")
            ui.proportion_bar([(ZONE_LABELS[z], bands.get(z, 0), ZONE_COLOR[z])
                               for z in range(1, 6)])
        wx = (data.get("weather") or {}).get(str(act["activity_id"])) or {}
        if wx.get("temp_c") is not None:
            bits = [f"{wx['temp_c']:.0f}°C"]
            if wx.get("dew_point_c") is not None:
                bits.append(f"dew point {wx['dew_point_c']:.0f}°C")
            if wx.get("condition"):
                bits.append(str(wx["condition"]).lower())
            st.caption(" · ".join(bits))
        if act.get("decoupling_pct") is not None:
            st.caption(f"Heart rate crept up {act['decoupling_pct']:.1f}% from "
                       f"the first half to the second.")
    laps = [l for l in (data.get("laps") or [])
            if str(l.get("activity_id")) == str(act["activity_id"])]
    if len(laps) >= 2:
        drift = lap_drift(laps)
        if drift.get("drift_bpm") is not None:
            st.caption(drift["message"])


def done_sessions_block(data: dict, today: date) -> None:
    """Every session finished this week, each one openable.

    The week strip says a day is done; this is where you find out how it went
    without leaving the page.
    """
    start = week_start_of(today)
    acts = [a for a in (data.get("activities") or [])
            if a.get("start_date")
            and start <= date.fromisoformat(str(a["start_date"])[:10]) <= today]
    if not acts:
        return
    ui.section("How this week has gone",
               "Pick a session for its numbers, its route and its splits.")
    ordered = sorted(acts, key=lambda a: str(a.get("start_time") or ""),
                     reverse=True)
    labels = {}
    for act in ordered:
        when = date.fromisoformat(str(act["start_date"])[:10])
        km = (act.get("distance_m") or 0) / 1000.0
        labels[f"{EMOJI.get(act['sport'], '•')} {day_label(when.isoformat())}"
               f" · {hm((act.get('duration_s') or 0) / 60)}"
               + (f" · {km:.1f} km" if km else "")] = act
    # A picker rather than a stack of expanders: Streamlit executes an
    # expander's body whether or not it is open, so six sessions meant six
    # stream reads and six charts built on every click of anything.
    picked = st.pills("Session", list(labels), selection_mode="single",
                      key="week_session", label_visibility="collapsed")
    if picked:
        session_recap(labels[picked], data)
    else:
        st.caption("Nothing selected, so nothing loaded — pick a day above.")


def page_today(data: dict, today: date) -> None:
    acts, wl = data["activities"], data["wellness"]
    sig = recovery_signals(wl, acts, as_of=today) if wl else None
    with Store(db_path()) as s:
        facts = planner.build_facts(s, today=today)
        env = planner.build_envelope(facts, s)
        verdict = planner.readiness_verdict(facts)

    plan = st.session_state.get("plan") or (data["plan"] or {}).get("plan")
    focus_day, rolled = training_focus(today)
    day_name = DAYS[focus_day.weekday()]
    source = plan
    if rolled and focus_day > today:
        # Tomorrow may belong to next week's provisional plan.
        source = plan if focus_day.isocalendar()[1] == today.isocalendar()[1] \
            else next_week_plan(today, data.get("scoped_to"))
    todo = [d for d in (source or {}).get("week_plan", [])
            if d["day"] == day_name and d.get("purpose") != "completed"]
    done_today = [d for d in (plan or {}).get("week_plan", [])
                  if d["day"] == DAYS[today.weekday()]
                  and d.get("purpose") == "completed"]

    # The written summary first, and full width. Every other page leads with it;
    # on Today it sat below the session card, which made this the one page where
    # the reading you actually want was under the thing it was about.
    insight_banner("Overview", data, today)

    heading = "Tomorrow" if rolled else "Today"
    ui.section(heading, f"It is past {EVENING_CUTOFF_HOUR}:00 — showing "
                        f"{focus_day.strftime('%A')} instead." if rolled else "")
    if not plan:
        ui.banner("No plan for this week yet",
                  "Open the Plan page and build one.", "neutral")
    elif todo:
        d = todo[0]
        bits = [f"{d['duration_min']} min"] if d["duration_min"] else []
        if d.get("target_hr"):
            # The number the athlete actually watches mid-session.
            bits.append(f"{d['target_hr']} ({d.get('target_zone', '')})".strip())
        elif d.get("target_zone") not in (None, "n/a", ""):
            bits.append(d["target_zone"])
        names = [strength.EXERCISES[e].name for e in d.get("exercise_ids", [])
                 if e in strength.EXERCISES]
        ui.today_card(d["sport"], " · ".join(bits) or "—",
                      "; ".join(names) if names else d.get("why", ""))
        if d["sport"] != "strength":
            send_session_to_watch(d, focus_day)
        for extra in todo[1:]:
            st.caption(f"also today: {EMOJI.get(extra['sport'], '')} "
                       f"{extra['sport']} · {extra['duration_min']} min")
    elif done_today:
        # The sport that was actually done, not "rest": a finished session is an
        # achievement, and labelling it as a rest day reads like the app missed it.
        first = done_today[0]
        rest = done_today[1:]
        ui.today_card(
            first["sport"],
            f"Done · {first['duration_min']} min"
            + (" · " + ", ".join(f"{d['sport']} {d['duration_min']} min"
                                 for d in rest) if rest else ""),
            (first.get("why") or "").replace("logged: ", "read back from your watch: ")
            or "Logged from your watch.")
    else:
        ui.today_card("rest", "Nothing scheduled",
                      "Rest is where the adaptation happens.")
    ask_block(data, today)

    # One column, not two. The right-hand stack of six cards was taller than the
    # plan beside it, so whichever side was shorter left a third of the screen
    # empty — first the right, then the left. The same six numbers read fine as a
    # strip, and the session card gets the width it deserves.
    tone = {"deload": "bad", "hold": "neutral", "build": "good"}[verdict["verdict"]]
    wk = week_summaries(acts, weeks=1, as_of=today,
                        strength_rows=data["strength"])[-1]
    left_min = max(0.0, env.max_week_minutes - wk.total_minutes)
    trained = len(facts.trained_days)
    ui.figures([
        {"label": "Verdict",
         "value": {"deload": "Back off", "hold": "Hold", "build": "Build"}[
             verdict["verdict"]],
         "note": " · ".join(verdict["reasons"])[:60]
                 or "nothing arguing either way", "tone": tone},
        {"label": "Readiness",
         "value": f"{sig.training_readiness:.0f}"
                  if sig and sig.training_readiness else "—",
         "note": (sig.training_status or "morning reading") if sig else "no data",
         "tone": "bad" if sig and sig.training_readiness
                 and sig.training_readiness < 35 else "neutral"},
        {"label": "Resting HR",
         "value": f"{sig.rhr_recent:.0f}" if sig and sig.rhr_recent else "—",
         "note": f"{sig.rhr_delta:+.1f} vs 28-day"
                 if sig and sig.rhr_delta is not None else "building a baseline",
         "tone": "bad" if sig and sig.rhr_delta and sig.rhr_delta > 3
                 else "neutral"},
        {"label": "Load ratio", "value": f"{sig.acwr:.2f}" if sig and sig.acwr
                                          else "—",
         "note": acwr_note(sig), "tone": acwr_tone(sig)},
        {"label": "Week so far", "value": hm(wk.total_minutes),
         "note": f"{hm(left_min)} left of {hm(env.max_week_minutes)}",
         "tone": "caution" if left_min <= 0 else "neutral"},
        {"label": "Days trained", "value": f"{trained}/7",
         "note": f"{wk.rest_days} rest "
                 f"{'day' if wk.rest_days == 1 else 'days'} left",
         "tone": "bad" if wk.rest_days == 0 else "neutral"},
    ])

    ui.section("This week",
               f"{hm(wk.total_minutes)} done of a {hm(env.max_week_minutes)} ceiling")
    ui.week_strip(week_cells(plan, today))

    done_sessions_block(data, today)

    nxt = next_week_plan(today, data.get("scoped_to"))
    if nxt:
        total = sum(d["duration_min"] for d in nxt.get("week_plan", []))
        # Collapsed: it is a preview of a week that has not started, and left open
        # it cost 340px above the strength instructions for today, which is the
        # thing actually being done in the next hour.
        with st.expander(
                f"Next week · "
                f"{day_label((week_start_of(today) + timedelta(weeks=1)).isoformat())}"
                f" onwards · {hm(total)} planned"):
            st.caption("Provisional, and re-derived when the week arrives.")
            ui.week_strip(week_cells(nxt, week_start_of(today) + timedelta(weeks=1),
                                     mark_today=False))

    # Today's strength session, spelled out. It is the one sport where knowing
    # what to do is not enough — the exercises are only protective if they are
    # done slowly and in the right position.
    legs_today = next((d for d in todo if d["sport"] == "strength"), None)
    if legs_today:
        strength_howto_block(
            list(legs_today.get("exercise_ids") or []), data["strength"],
            session_index=len({str(r["day"]) for r in data["strength"]}),
            # The same reduction the planner applies, so the page, the watch and
            # the plan agree on the numbers in a deload week.
            intensity=0.6 if env.deload else 1.0,
            day=focus_day)





@st.cache_data(show_spinner=False, ttl=900)
def _next_week_plan(stamp: float, iso_today: str,
                    sports: tuple[str, ...] = ()) -> dict | None:  # noqa: ARG001
    """Computed on demand and cached: it is a preview, not a saved plan.

    `sports` is part of the cache key as well as the plan: filtering to run and
    bike has to change the preview, not just the tabs around it.
    """
    try:
        with Store(db_path()) as s:
            p = planner.plan_next_week(s, today=date.fromisoformat(iso_today),
                                       use_ai=False, save=False,
                                       only_sports=list(sports) or None)
        return p.model_dump(mode="json")
    except Exception as exc:  # noqa: BLE001 - a preview must never break the page
        log.warning("Next-week preview failed: %s", exc)
        return None


def next_week_plan(today: date, sports: list[str] | None = None) -> dict | None:
    # Sorted so an unchanged selection reuses the cached preview regardless of
    # the order the toggles happened to be clicked in.
    return _next_week_plan(db_stamp(), today.isoformat(),
                           tuple(sorted(sports or ())))


def week_cells(plan: dict | None, today: date,
               mark_today: bool = True) -> list[dict]:
    """`mark_today=False` for a future week, which has no today to highlight."""
    start = week_start_of(today)
    entries = (plan or {}).get("week_plan", [])
    real_today = date.today()
    cells = []
    for i, name in enumerate(DAYS):
        d = start + timedelta(days=i)
        items = [
            {"sport": e["sport"], "minutes": e["duration_min"],
             "zone": "" if e.get("target_zone") in (None, "n/a", "") else e["target_zone"],
             "hr": e.get("target_hr") or "",
             "done": e.get("purpose") == "completed",
             # A day that has gone by with the session still only planned. Today
             # is not counted: the evening is still available.
             "missed": e.get("purpose") != "completed" and d < real_today}
            for e in entries if e["day"] == name and e["sport"] != "rest"
        ]
        cells.append({"name": name, "date": d.strftime("%d-%m"),
                      "today": mark_today and d == real_today, "items": items})
    return cells


# --------------------------------------------------------------------------
# PAGE 2 — Progress
# --------------------------------------------------------------------------


def acwr_note(sig) -> str:
    """Acute:chronic load, in words. The number alone means nothing to most people."""
    if not (sig and sig.acwr):
        return "needs 3 weeks"
    if sig.acwr > 1.3:
        return "ramping too fast"
    if sig.acwr < 0.8:
        return "losing fitness"
    return "sustainable"


def acwr_tone(sig) -> str:
    """Above 1.3 is the injury-risk zone; below 0.8 is detraining."""
    if not (sig and sig.acwr):
        return "neutral"
    if sig.acwr > 1.3:
        return "bad"
    if sig.acwr < 0.8:
        return "caution"
    return "good"


def week_targets(data: dict, today: date) -> dict[str, dict[str, int]]:
    """Sessions done this week against sessions asked for, per sport.

    The plan is the only thing that makes a session count good or bad — three
    runs is excellent against a target of three and a warning sign against one.
    """
    start = week_start_of(today)
    done: dict[str, int] = {}
    for a in data.get("activities") or []:
        try:
            day = date.fromisoformat(str(a["start_date"])[:10])
        except (KeyError, ValueError, TypeError):
            continue
        if start <= day <= today:
            done[a.get("sport")] = done.get(a.get("sport"), 0) + 1

    out: dict[str, dict[str, int]] = {}
    for sport, target in (data.get("targets") or {}).items():
        want = int(target.get("sessions") or 0)
        if want:
            out[sport] = {"want": want, "done": done.get(sport, 0)}
    # No explicit target: fall back to what the plan actually prescribes.
    plan = ((data.get("plan") or {}).get("plan")) or {}
    for d in plan.get("week_plan") or []:
        sp = d.get("sport")
        if sp in ("rest", "strength") or (d.get("duration_min") or 0) <= 0:
            continue
        if sp not in out:
            out[sp] = {"want": 0, "done": done.get(sp, 0)}
        out[sp]["want"] = max(out[sp]["want"],
                              sum(1 for x in plan["week_plan"]
                                  if x.get("sport") == sp
                                  and (x.get("duration_min") or 0) > 0))
    return out


def page_progress(data: dict, today: date) -> None:
    """Trend first, numbers second.

    The previous order was twelve bordered stat cards — about 330px of height —
    before the first chart. Numbers in boxes are the slowest possible way to
    answer "is it working": the answer is a direction, and a direction is a
    shape. So the headline chart leads at full width, and everything that used
    to be a card is now one dense strip beneath it.
    """
    acts, wl, zones = data["activities"], data["wellness"], data["zones"]
    tot = totals(acts)
    sig = recovery_signals(wl, acts, as_of=today) if wl else None

    insight_banner("Fitness", data, today)

    # The one chart the whole page exists for.
    ui.section("Heart rate at your usual pace",
               "Every session is shown as the heart rate it would have taken at "
               "your normal pace, so a fast day and a slow day can be compared. "
               "A line going down means you are getting fitter.")
    with ui.frame():
        training_hr_block(acts, today, data.get("notes"), data.get("weather"),
                          ceiling=data.get("aerobic_ceiling"))

    rhr = baseline_trend(wl, "resting_hr", as_of=today, lower_is_better=True)
    hrv = baseline_trend(wl, "hrv_last_night", as_of=today, lower_is_better=False)
    tones = {"improving": "good", "worsening": "bad", "steady": "neutral",
             "insufficient_data": "neutral"}

    def trend_note(t: dict, unit: str) -> str:
        if t["verdict"] == "insufficient_data":
            return t.get("needed", "more data")
        if t["per_week"] is None:
            return "steady"
        return f"{t['per_week']:+.2f} {unit}/wk"

    recent = [r for r in wl if r.get("day")
              and date.fromisoformat(str(r["day"])[:10]) >= today - timedelta(days=7)]

    def avg(field: str) -> float | None:
        vals = [float(r[field]) for r in recent if r.get(field) is not None]
        return sum(vals) / len(vals) if vals else None

    weekly_im = sum(float(r.get("intensity_moderate_min") or 0)
                    + 2 * float(r.get("intensity_vigorous_min") or 0)
                    for r in recent)
    steps, stress = avg("steps"), avg("stress_avg")
    cad = cadence_stats(acts)

    # One strip, and only entries that carry a number. A card reading "Swim 0"
    # spends a twelfth of the row saying nothing.
    figures = [
        {"label": "Resting HR",
         "value": f"{rhr['recent']:.0f}" if rhr["recent"] else "—",
         "note": trend_note(rhr, "bpm"), "tone": tones[rhr["verdict"]]},
        {"label": "HRV", "value": f"{hrv['recent']:.0f}" if hrv["recent"] else "—",
         "note": trend_note(hrv, "ms"), "tone": tones[hrv["verdict"]]},
        {"label": "VO2max",
         "value": f"{sig.vo2max_run:.1f}" if sig and sig.vo2max_run else "—",
         "note": "Garmin estimate"},
        {"label": "Load ratio",
         "value": f"{sig.acwr:.2f}" if sig and sig.acwr else "—",
         "note": acwr_note(sig), "tone": acwr_tone(sig)},
    ]
    if cad.get("avg"):
        figures.append({
            "label": "Cadence", "value": f"{cad['avg']:.0f}",
            "note": f"stride {cad['avg_stride_cm']:.0f} cm"
                    if cad.get("avg_stride_cm") else "steps/min",
            "tone": {"low": "bad", "fair": "caution",
                     "good": "good"}.get(cad["verdict"], "neutral")})
    # Per-sport counts are coloured against this week's plan, not against the
    # lifetime total: "3 runs" is only good or bad relative to what was asked for.
    planned = week_targets(data, today)
    for sport in shown_sports():
        row = tot["by_sport"].get(sport) or {}
        if not row.get("sessions"):
            continue
        done_this_week = planned.get(sport, {}).get("done", 0)
        want = planned.get(sport, {}).get("want", 0)
        note = f"{row.get('km', 0):.0f} km · {hm(row.get('minutes') or 0)}"
        tone = "neutral"
        if want:
            note = f"{done_this_week}/{want} this week · {note}"
            # Judged against how much of the week has gone. Zero sessions on a
            # Monday is not behind schedule; zero on a Saturday is.
            elapsed = (today.weekday() + 1) / 7
            expected = want * elapsed
            if done_this_week >= want:
                tone = "good"
            elif done_this_week >= expected - 0.5:
                tone = "neutral"
            elif elapsed > 0.6:
                tone = "bad"
            else:
                tone = "caution"
        figures.append({"label": sport.title(),
                        "value": f"{int(row['sessions'])}",
                        "note": note, "tone": tone})
    if steps:
        figures.append({"label": "Steps / day", "value": f"{steps:,.0f}",
                        "note": "7-day average"})
    if weekly_im:
        figures.append({"label": "Intensity min", "value": f"{weekly_im:.0f}",
                        "note": "7 days, goal 150",
                        "tone": "good" if weekly_im >= 150 else "neutral"})
    if stress:
        figures.append({"label": "Stress", "value": f"{stress:.0f}",
                        "note": "7-day average",
                        "tone": "bad" if stress >= 50 else "neutral"})
    ui.figures(figures)

    # Two per row, and the secondary work behind tabs rather than stacked. Six
    # panels down the page was 2354px of scrolling; three visible with the rest
    # one click away is the same information in a third of the height.
    a, b = st.columns(2, gap="medium")
    with a, ui.card("How hard your training has been",
                    "At this stage most of it should feel easy."):
        intensity_block(zones, today, data.get("notes"), data)
    with b, ui.card("Speed you get per heartbeat",
                    "How much further each beat carries you, compared with your "
                    "first session."):
        efficiency_block(acts, today, data.get("notes"), data.get("weather"))

    detail = st.tabs(["Hours a week", "How fast you are building",
                      "Easy vs hard, week by week", "Overnight numbers",
                      "Steps and stride", "Running form",
                      "Heart rate creep", "Heat"])
    with detail[0], ui.frame():
        st.caption("Never more than 10% more than last week, and every "
                   "fourth week is an easy one.")
        volume_chart(data, today, data.get("notes"))
    with detail[1], ui.frame():
        st.caption("Minutes say how much you did. Load says how much it took "
                   "out of you. Comparing the last week with the last month is "
                   "the best warning sign there is for getting hurt.")
        load_ramp_block(acts, today, data.get("notes"))
    with detail[2], ui.frame():
        st.caption("Easy training turns into hard training a few minutes at "
                   "a time, and this is where you would see it happening.")
        zone_trend_block(data, today)
    with detail[3], ui.frame():
        st.caption("What your watch measured while you slept, against your "
                   "own normal.")
        trend_chart(wl, today)
    with detail[4], ui.frame():
        cadence_block(acts, today)
    with detail[5], ui.frame():
        form_block(acts)
    with detail[6], ui.frame():
        drift_block(acts, data.get("laps"))
    with detail[7], ui.frame():
        weather_block(data, today, data.get("notes"))


def form_block(acts: list[dict]) -> None:
    """Ground contact, vertical oscillation and vertical ratio.

    Garmin measures all three and this app ignored them until now. Vertical
    ratio is the useful one: it is bounce as a share of stride, so it says how
    much of each step went upwards instead of forwards.
    """
    runs = [a for a in acts if a.get("sport") == "run"
            and (a.get("ground_contact_ms") or a.get("vertical_ratio"))]
    if not runs:
        # Accurate about why, now that the sync fetches these deliberately: they
        # are not in the activity list, so each run costs one extra request and
        # they are filled in a few at a time.
        st.caption(
            "No numbers yet. Your watch measures these, but Garmin only hands "
            "them over one run at a time, so the sync collects a few on each "
            "run — they will appear here shortly.")
        return

    def mean(field: str) -> float | None:
        vals = [float(a[field]) for a in runs if a.get(field)]
        return sum(vals) / len(vals) if vals else None

    gct, osc, ratio = (mean("ground_contact_ms"), mean("vertical_osc_cm"),
                       mean("vertical_ratio"))
    ui.rows([
        ("Time each foot is on the ground", f"{gct:.0f} ms" if gct else "—",
         "under 250 is quick, over 300 is slow"),
        ("How far you bounce each step", f"{osc:.1f} cm" if osc else "—",
         "up and down, not forwards"),
        ("Bounce compared with stride", f"{ratio:.1f}%" if ratio else "—",
         "under 8% is good"),
    ])
    if ratio and ratio > 9:
        st.caption(
            f"At {ratio:.1f}% a noticeable share of each step goes upwards "
            f"rather than forwards. A quicker cadence is the usual fix — it is "
            f"the same change that shortens the stride.")


def intensity_block(zones: list[dict], today: date,
                    notes: dict | None = None, data: dict | None = None) -> None:
    if not zones:
        st.caption("No zone data yet — sync from Garmin.")
        return
    since = today - timedelta(days=28)

    # Measured against the athlete's own aerobic ceiling where one is set, not
    # Garmin's fixed Z2 top. Otherwise every minute deliberately spent between
    # the two counts as "moderate", the easy share can never improve, and this
    # page contradicts the setting on the Plan page.
    pol = polarisation(zones, since=since)
    bounds = zone_bounds(zones)
    custom = None
    ceiling = (data or {}).get("aerobic_ceiling")
    if ceiling:
        custom = stream_polarisation(
            db_stamp(), today.isoformat(), int(float(ceiling)),
            int(bounds.get(4, (0, 0))[0] or 0), tuple(sorted(shown_sports())))

    live = bool(custom and custom.get("samples"))
    shown = custom if live else pol
    ui.proportion_bar([("easy", shown["easy"], TONE["good"]),
                       ("moderate", shown["moderate"], TONE["caution"]),
                       ("hard", shown["hard"], TONE["bad"])])
    verdict = ("On target — base phase wants 70%+ easy." if shown["easy"] >= 70
               else "Inverted: base phase wants roughly the reverse of this."
               if shown["hard"] >= 35 else "Drifting harder than base phase wants.")
    if live:
        st.caption(f"28 days, against your {custom['ceiling']} bpm ceiling. "
                   f"{verdict}")
        with st.expander("How this compares with Garmin's own zones"):
            st.markdown(
                f"Garmin stops easy at **{bounds.get(2, (0, 0))[1]} bpm** and "
                f"reads {pol['easy']:.0f}% easy / {pol['moderate']:.0f}% "
                f"moderate / {pol['hard']:.0f}% hard for the same period.\n\n"
                f"Hard starts at the same {custom['hard_floor']} bpm either way, "
                f"so the whole difference is your raised ceiling.")
    else:
        st.caption(f"28 days, on Garmin's zones. {verdict}")
    chart_ai_note("intensity", notes)
    with st.expander("Zone breakdown by sport"):
        if live:
            st.caption(
                f"Recounted from your stored heart-rate samples with zone 2 "
                f"topping out at {custom['ceiling']} bpm. Totals differ a little "
                f"from Garmin's own: those are computed on the watch from every "
                f"beat, and the samples stored here are thinned to 600 a session.")
        for sport in shown_sports():
            sp = (stream_zone_minutes(db_stamp(), today.isoformat(),
                                      int(float(ceiling)), sport) if live
                  else zone_distribution(zones, sport=sport, since=since))
            sp = {int(k): v for k, v in (sp or {}).items()}
            if sum(sp.values()) <= 0:
                continue
            st.markdown(f"<span style='font-size:.82rem;opacity:.8'>{EMOJI[sport]} "
                        f"{sport.title()} · {hm(sum(sp.values()))}</span>",
                        unsafe_allow_html=True)
            ui.proportion_bar([(ZONE_LABELS[z], sp.get(z, 0), ZONE_COLOR[z])
                               for z in range(1, 6)])


def efficiency_block(acts: list[dict], today: date,
                     notes: dict | None = None,
                     weather: dict | None = None) -> None:
    """One chart for all three sports.

    Efficiency is in different units per sport — watts per beat on the bike,
    speed per beat elsewhere — so the raw values cannot share an axis. Expressing
    each as a percentage change from its own earliest baseline can.
    """
    drawn = False
    fig = go.Figure()
    for sport in shown_sports():
        pts = [p for p in ef_points(acts, sport) if p.is_steady] or \
              ef_points(acts, sport)
        if len(pts) < 2:
            continue
        base = pts[0].ef
        if not base:
            continue
        drawn = True
        fig.add_scatter(
            x=[p.date for p in pts], y=[(p.ef / base - 1) * 100 for p in pts],
            mode="lines+markers", name=sport,
            line=dict(color=SPORT_COLOR[sport], width=2), marker=dict(size=8),
            hovertemplate="%{x|%a %d-%m-%Y}<br>%{y:+.1f}% vs first session"
                          f"<extra>{sport}</extra>")
    if not drawn:
        st.caption("Needs at least two sessions with heart rate in a sport. "
                   "Three steady sessions gives a direction; six makes it reliable.")
        return
    muggy = humid_days(acts, weather)
    if muggy:
        marks = sorted(muggy)
        fig.add_scatter(
            x=marks, y=[0] * len(marks), mode="markers", name="humid",
            marker=dict(size=11, symbol="triangle-up", color=TONE["caution"],
                        opacity=.75),
            customdata=[muggy[d] for d in marks],
            hovertemplate="%{customdata}<extra>humid — efficiency reads low</extra>")
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(140,158,176,.5)")
    fig.update_layout(yaxis_title="% vs first session")
    ui.chart(fig, 200, date_axis=True)
    chart_ai_note("efficiency", notes)
    statuses = [ef_data_status(acts, s) for s in shown_sports()]
    short = [s for s in statuses if s["total"] and s["needed_for_verdict"]]
    if short:
        st.caption(" · ".join(f"{s['sport']}: {s['steady']}/3 steady" for s in short))


def cadence_block(acts: list[dict], today: date) -> None:
    """Cadence and stride together, because neither means much alone.

    Cadence rises with pace on its own, so a faster session shows a higher
    number without the stride having changed. Stride length is what closes the
    loop: the same pace at a higher cadence is a shorter stride, which means the
    foot lands closer to underneath the body instead of out in front of it.
    """
    stats = cadence_stats(acts)
    pts = stats["points"]
    if not pts:
        st.caption(stats["message"])
        return

    tone = {"low": "bad", "fair": "caution", "good": "good"}.get(
        stats["verdict"], "neutral")
    ui.stats_row([
        {"label": "Cadence", "value": f"{stats['avg']:.0f} spm",
         "note": f"target {stats['target']:.0f}+", "tone": tone},
        {"label": "Stride", "value": f"{stats['avg_stride_cm']:.0f} cm"
         if stats.get("avg_stride_cm") else "—", "note": "distance per step"},
        {"label": "Sessions", "value": len(pts), "note": "with cadence recorded"},
    ])

    fig = go.Figure()
    fig.add_scatter(x=[p["date"] for p in pts], y=[p["cadence"] for p in pts],
                    mode="lines+markers" if len(pts) >= 3 else "markers",
                    name="cadence", line=dict(color=SPORT_COLOR["run"], width=2),
                    marker=dict(size=9),
                    customdata=[p["stride_cm"] for p in pts],
                    hovertemplate="%{x|%a %d-%m-%Y}<br>%{y:.0f} spm<br>"
                                  "stride %{customdata:.0f} cm<extra>run</extra>")
    fig.add_hline(y=stats["target"], line_dash="dot",
                  line_color="rgba(140,158,176,.6)",
                  annotation_text=f"{stats['target']:.0f} spm")
    fig.update_layout(yaxis_title="steps per minute")
    ui.chart(fig, 200, date_axis=True)
    st.caption(stats["message"])

    if stats["verdict"] != "good":
        with st.expander("How to raise it"):
            st.markdown(
                "Cadence is a habit your nervous system holds, so it changes "
                "with repetition rather than with effort. Two rules before "
                "anything else:\n\n"
                "1. **Add 5%, not 20%.** From "
                f"{stats['avg']:.0f} spm that is about "
                f"{stats['avg'] * 1.05:.0f} spm. A big jump just makes you "
                "bounce, which is worse than a long stride.\n"
                "2. **Hold the effort, let the pace drop.** Shorter steps at "
                "first mean a slower pace. That is the change working, not a "
                "loss of fitness — the pace comes back within a few weeks.\n\n"
                "Then, in this order:"
            )
            for drill in strength.DRILLS.values():
                if drill.get("focus") != "cadence" or "bike" in drill["name"].lower():
                    continue
                with st.container(border=True):
                    st.markdown(f"**{drill['name']}** — {drill['dose']}")
                    st.caption(drill["where"])
                    st.markdown(f"**Set up:** {drill['setup']}")
                    st.markdown("\n".join(f"{i}. {x}" for i, x
                                           in enumerate(drill["steps"], 1)))
                    st.markdown(f"**Common mistake:** {drill['mistakes']}")
                    st.caption(f"Why it matters: {drill['why']}")
            st.markdown(
                "**Stride length looks after itself.** It is pace divided by "
                "cadence, so at the same pace a quicker turnover shortens it "
                "automatically. Chasing a longer stride directly means reaching "
                "the foot out in front, which brakes and loads the knee — the "
                "opposite of what you want."
            )


def drift_block(acts: list[dict], laps: list[dict] | None = None) -> None:
    """Aerobic drift, from laps where the stream cannot give it.

    Decoupling needs a session long enough to split into two comparable halves —
    an hour in practice — so on 45-minute sessions this panel was permanently
    empty. Auto-lap answers the same question at kilometre resolution, and only
    compares laps run at the same pace, because heart rate rising with pace is
    effort rather than drift.
    """
    by_activity: dict[str, list[dict]] = {}
    for lap in laps or []:
        by_activity.setdefault(str(lap["activity_id"]), []).append(lap)

    rows = []
    for a in acts:
        session_laps = by_activity.get(str(a.get("activity_id")))
        if not session_laps:
            continue
        d = lap_drift(session_laps)
        if d.get("drift_bpm") is None:
            continue
        rows.append({"act": a, "drift": d,
                     "spread": lap_pace_spread(session_laps)})

    if rows:
        st.caption("Only kilometres you ran at the same pace are compared, so a "
                   "rise here means the session got harder, not faster.")
        table(pd.DataFrame([{
            "start_date": r["act"]["start_date"],
            "sport": r["act"]["sport"],
            "Heart rate rise": f"+{r['drift']['drift_bpm']:.0f} bpm",
            "From": f"{r['drift']['first_hr']} → {r['drift']['last_hr']}",
            "Kilometres compared": r["drift"]["laps_compared"],
            "How even the pace was": (f"{r['spread']:.1f}% apart"
                                      if r["spread"] else "—"),
        } for r in rows]))
        worst = max(rows, key=lambda r: r["drift"]["drift_bpm"])
        tone = {"flat": "good", "mild": "caution", "steep": "bad"}[
            worst["drift"]["verdict"]]
        ui.banner("Heart rate creep", worst["drift"]["message"], tone)

    drift = [a for a in acts if a.get("decoupling_pct") is not None]
    if not drift:
        if not rows:
            st.caption("Needs one session with at least three kilometres of "
                       "three minutes or more, or a session of an hour.")
        return
    df = pd.DataFrame([{"Date": a["start_date"], "Sport": a["sport"],
                        "Drift": round(a["decoupling_pct"], 2)} for a in drift])
    fig = px.bar(df, x="Date", y="Drift", color="Sport",
                 color_discrete_map=SPORT_COLOR)
    fig.add_hline(y=5, line_dash="dot", annotation_text="5%")
    fig.update_layout(yaxis_title="% drift")
    ui.chart(fig, 200, date_axis=True)


# Dew point above this limits evaporative cooling enough to cost several beats
# a minute at the same pace. Below it, conditions are not the explanation.
MUGGY_DEW_C = 18.0


def load_ramp_block(acts: list[dict], today: date,
                    notes: dict | None = None) -> None:
    """Daily load with its 7-day and 28-day averages, and the ratio between them.

    The dashboard already shows today's load ratio as one number. A number cannot
    say which way it is moving, and that is the part that decides whether to back
    off: 1.25 on the way down is a different week from 1.25 climbing.
    """
    ramp = load_ramp(acts, as_of=today, days=90)
    if len(ramp) < 2:
        st.caption("Not enough logged sessions yet to draw a ramp.")
        return
    days = [r["day"] for r in ramp]
    fig = go.Figure()
    fig.add_bar(x=days, y=[r["load"] for r in ramp], name="that day",
                marker=dict(color="rgba(140,158,176,.42)"),
                # Width in milliseconds, because the axis is dates. Plotly's
                # default splits the span between however many bars there are,
                # which drew day-and-a-half-wide bars across six days of data.
                width=0.8 * 86_400_000,
                hovertemplate="%{x|%a %d-%m-%Y}<br>%{y:.0f} load<extra></extra>")
    fig.add_scatter(x=days, y=[r["acute"] for r in ramp], mode="lines",
                    name="7-day average", line=dict(color=TONE["caution"], width=2.5),
                    hovertemplate="%{y:.1f}/day<extra>last 7 days</extra>")
    fig.add_scatter(x=days, y=[r["chronic"] for r in ramp], mode="lines",
                    name="28-day base", line=dict(color="#7FB6DC", width=2, dash="dot"),
                    hovertemplate="%{y:.1f}/day<extra>last 28 days</extra>")
    ratios = [r["ratio"] for r in ramp]
    live = [x for x in ratios if x is not None]
    if live:
        # Second axis, because a ratio around 1 and a load in the hundreds cannot
        # share a scale without one of them becoming a flat line.
        fig.add_scatter(x=days, y=ratios, mode="lines", name="load ratio",
                        yaxis="y2", line=dict(color=TONE["bad"], width=1.8),
                        hovertemplate="ratio %{y:.2f}<extra></extra>")
        fig.add_hrect(y0=ACWR_LOW, y1=ACWR_HIGH, yref="y2", line_width=0,
                      fillcolor=TONE["good"], opacity=.10)
        fig.update_layout(yaxis2=dict(
            overlaying="y", side="right", showgrid=False,
            range=[0, max(2.0, max(live) * 1.2)],
            tickfont=dict(color=TONE["bad"], size=10)))
    fig.update_layout(yaxis_title="load per day", hovermode="x unified")
    ui.chart(fig, 230, date_axis=True)
    verdict = ramp_verdict(ramp)
    if verdict["ratio"] is None:
        st.caption(f"Load ratio not shown yet — {verdict['note']}. The shaded band "
                   f"is where a ramp is usually productive ({ACWR_LOW}–{ACWR_HIGH}).")
    else:
        st.caption(f"Load ratio {verdict['ratio']:.2f} — {verdict['note']}. "
                   f"Productive band is {ACWR_LOW}–{ACWR_HIGH}.")
    chart_ai_note("load_ramp", notes)


def zone_trend_block(data: dict, today: date, weeks: int = 8) -> None:
    """Where each week's minutes went, week on week, plus the easy share.

    The 28-day split says what the mix is. This says whether it is drifting —
    which is the thing that creeps: a base block slides into accidental tempo
    work a few minutes at a time, and a single four-week average hides it.
    """
    ceiling = data.get("aerobic_ceiling")
    rows = (stream_zone_weeks(db_stamp(), today.isoformat(), int(float(ceiling)),
                              weeks, tuple(sorted(shown_sports())))
            if ceiling else [])
    if not rows:
        st.caption("Needs stored heart-rate samples and an aerobic ceiling — "
                   "set one on the Plan page." if not ceiling else
                   "No heart-rate samples in the last few weeks.")
        return
    # Category axis, not dates: a weekly bar has no width in time, and with one
    # or two weeks on record the date version repeated the same tick label three
    # times and stretched a single bar across the whole plot.
    weeks_x = [day_label(r["week_start"]) for r in rows]
    fig = go.Figure()
    for z in range(1, 6):
        fig.add_bar(x=weeks_x, y=[r.get(f"z{z}", 0) for r in rows],
                    name=ZONE_LABELS[z], marker=dict(color=ZONE_COLOR[z]),
                    hovertemplate="week of %{x}<br>%{y:.0f} min"
                                  f"<extra>{ZONE_LABELS[z]}</extra>")
    easy = [r.get("easy_pct") for r in rows]
    if any(e is not None for e in easy):
        fig.add_scatter(x=weeks_x, y=easy, mode="lines+markers", yaxis="y2",
                        name="easy share", line=dict(color=TONE["good"], width=2),
                        marker=dict(size=7),
                        hovertemplate="%{y:.0f}% easy<extra></extra>")
        fig.add_hline(y=70, yref="y2", line=dict(color=TONE["good"], width=1,
                                                 dash="dot"))
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", range=[0, 100],
                                      showgrid=False, ticksuffix="%",
                                      tickfont=dict(color=TONE["good"], size=10)))
    # traceorder normal, so the legend reads Z1 to Z5 the way the stack does.
    # bargap keeps a single week from becoming a bar the width of the panel.
    fig.update_layout(barmode="stack", yaxis_title="minutes", bargap=0.55,
                      legend=dict(traceorder="normal"))
    # Keep at least three slots on the axis, so a first week does not draw one
    # bar half the width of the panel.
    fig.update_xaxes(range=[-0.5, max(2.5, len(rows) - 0.5)])
    ui.chart(fig, 240)
    st.caption(f"Counted against your {int(float(ceiling))} bpm ceiling. The dotted "
               f"line is the 70% easy that base phase wants.")


def consistency_block(data: dict, today: date, weeks: int = 16) -> None:
    """A calendar of showing up. Endurance is mostly an attendance problem.

    A heatmap rather than a chart of minutes on purpose: the failure mode this
    catches is a gap, and a gap is a hole in a grid — it is invisible in a line
    that simply carries on from either side of it.
    """
    rows = consistency(data["activities"], as_of=today, weeks=weeks,
                       strength_rows=data.get("strength"))
    if not rows:
        st.caption("No sessions logged yet.")
        return
    week_cols = sorted({r["day"] - timedelta(days=r["day"].weekday()) for r in rows})
    at = {w: i for i, w in enumerate(week_cols)}
    grid = [[None] * len(week_cols) for _ in range(7)]
    label = [[""] * len(week_cols) for _ in range(7)]
    for r in rows:
        col = at[r["day"] - timedelta(days=r["day"].weekday())]
        row = r["day"].weekday()
        grid[row][col] = r["minutes"]
        sports = ", ".join(f"{EMOJI.get(sp, '')} {sp}" for sp in r["sports"])
        label[row][col] = (f"{day_label(r['day'].isoformat(), year=True)}<br>"
                           + (f"{sports}<br>{r['minutes']:.0f} min" if r["sports"]
                              else "rest"))
    top = max([r["minutes"] for r in rows] or [60]) or 60
    fig = go.Figure(go.Heatmap(
        z=grid, x=[w.strftime("%d-%m") for w in week_cols],
        y=[DAYS[i] for i in range(7)], text=label,
        hovertemplate="%{text}<extra></extra>",
        # A rest day has to read as empty, not as the bottom of a green scale,
        # so zero gets its own near-transparent step.
        colorscale=[[0, "rgba(140,158,176,.12)"], [0.001, "rgba(63,182,139,.30)"],
                    [1, TONE["good"]]],
        zmin=0, zmax=max(60, top), showscale=False, xgap=3, ygap=3))
    # Explicitly categorical. Left to itself Plotly reads "17-08" as a number and
    # drew the sixteen weeks along an axis labelled 2005 to 2025.
    fig.update_xaxes(type="category", tickangle=0)
    fig.update_yaxes(type="category", autorange="reversed")
    ui.chart(fig, 210)
    s = streak(rows)
    active = s["active_days"]
    ui.figures([
        {"label": "Current streak", "value": f"{s['current']} d",
         "note": "days with something logged",
         "tone": "good" if s["current"] >= 3 else "neutral"},
        {"label": "Longest streak", "value": f"{s['longest']} d",
         "note": f"in the last {weeks} weeks"},
        {"label": "Active days", "value": f"{active}/{s['days']}",
         "note": f"{active / max(s['days'], 1) * 100:.0f}% of days",
         "tone": "good" if active / max(s["days"], 1) >= 0.5 else "caution"},
        {"label": "Rest days", "value": f"{s['days'] - active}",
         "note": "recovery is where adaptation happens"},
    ])


def weather_block(data: dict, today: date, notes: dict | None = None) -> None:
    """Heart rate at a reference pace against dew point.

    Here because the efficiency charts treat every session as comparable and they
    are not. At a 21°C dew point sweat barely evaporates and the same pace costs
    several beats more; reading that as lost fitness is the easiest wrong
    conclusion this dashboard makes available.
    """
    eff = weather_effect(data["activities"], data.get("weather") or {}, sport="run")
    pts = eff["points"]
    if not pts:
        st.caption("No outdoor runs yet with both weather and a comparable pace.")
        return
    fig = go.Figure()
    fig.add_scatter(
        x=[p["dew_point_c"] for p in pts], y=[p["hr_at_reference"] for p in pts],
        mode="markers+text", name="runs",
        marker=dict(size=12, opacity=.85,
                    color=[TONE["bad"] if p["dew_point_c"] >= DEW_POINT_HARD_C
                           else TONE["caution"] for p in pts]),
        text=[day_label(p["date"].isoformat()) for p in pts],
        textposition="middle right", textfont=dict(size=10),
        customdata=[[p["condition"], p["temp_c"] or 0] for p in pts],
        hovertemplate="dew point %{x:.1f}°C · %{customdata[1]:.0f}°C air<br>"
                      "%{y:.0f} bpm at your usual pace<br>%{customdata[0]}"
                      "<extra></extra>")
    hottest = max(p["dew_point_c"] for p in pts)
    if hottest >= DEW_POINT_HARD_C:  # noqa: SIM102
        fig.add_vrect(x0=DEW_POINT_HARD_C, x1=hottest + 1.5, line_width=0,
                      fillcolor=TONE["bad"], opacity=.07)
    fig.update_layout(xaxis_title="dew point °C",
                      yaxis_title="bpm at your usual pace")
    lo = min(p["dew_point_c"] for p in pts)
    fig.update_xaxes(showgrid=False, range=[lo - 0.8, hottest + 2.2])
    ui.chart(fig, 230)
    bits = [f"{eff['hot_share']}% of these runs were above {DEW_POINT_HARD_C:.0f}°C "
            f"dew point, where evaporative cooling stops keeping up"]
    if eff["bpm_per_deg"] is not None:
        bits.append(f"and each degree costs about {eff['bpm_per_deg']:+.1f} bpm at "
                    f"the same pace")
    st.caption(". ".join(bits) + ". Read a bad session against this before "
               "reading it as lost fitness.")
    chart_ai_note("heat", notes)


@st.cache_data(show_spinner=False, ttl=1800)
def _ask_answer(stamp: float, iso_today: str, question: str) -> str:  # noqa: ARG001
    """Cached on the question and the data, so a rerun costs nothing."""
    try:
        def read(s: Store) -> str:
            payload = {
                "activities": s.activities(), "wellness": s.wellness(),
                "zones": s.zones(), "strength": s.strength_log(),
                "checkins": s.checkins(limit=8), "race": s.race_predictions(),
                "records": s.personal_records(), "targets": s.targets(),
                "aerobic_ceiling": s.get_state("aerobic_ceiling_bpm"),
                "plan": s.latest_plan(week_start_of(date.fromisoformat(iso_today))),
            }
            return insights.ask(question, payload,
                                date.fromisoformat(iso_today)) or ""
        return with_store(read)
    except Exception as exc:  # noqa: BLE001 - a failed answer is not a broken page
        log.warning("Coach answer failed: %s", exc)
        return ""


SUGGESTED_QUESTIONS = (
    "Why were my runs so hard?",
    "Am I actually getting fitter?",
    "Is my easy pace easy enough?",
    "Can I add a session this week?",
)


def ask_block(data: dict, today: date) -> None:
    """Free-text questions answered from the same facts the charts are drawn from.

    Deliberately read-only: it cannot change the plan. The planner is the only
    thing allowed to do that, because the planner is the layer the rules are
    enforced in — an answer that could quietly edit the week would route around
    every guardrail in the app.
    """
    if not ai.available():
        return
    with st.expander("Ask the coach", expanded=False):
        picked = st.pills("Common questions", SUGGESTED_QUESTIONS,
                          selection_mode="single", key="ask_pick",
                          label_visibility="collapsed")
        typed = st.text_input(
            "Your question", key="ask_q", label_visibility="collapsed",
            placeholder="Ask about your own numbers — “why was Wednesday so hard?”")
        question = (typed or picked or "").strip()
        go_ask = st.button("Ask", key="ask_go", disabled=not question)
        if go_ask and question:
            with st.spinner("Reading your data…"):
                answer = _ask_answer(db_stamp(), today.isoformat(), question)
            st.session_state["ask_answer"] = (question, answer)
        held = st.session_state.get("ask_answer")
        if held and held[1]:
            st.markdown(f"**{held[0]}**")
            st.markdown(held[1])
        elif held:
            st.caption("The AI backend did not answer. The charts below are "
                       "unaffected — they are computed locally.")
        st.caption("Answered from your stored sessions, recovery signals and this "
                   "week's plan. It cannot change the plan or override a rule.")


RACE_DISTANCES = (("time_5k", "5K", 5.0), ("time_10k", "10K", 10.0),
                  ("time_half", "Half", 21.0975), ("time_marathon", "Marathon",
                                                    42.195))


def clock(seconds: float | None) -> str:
    """h:mm:ss, or m:ss under an hour. Race times, not durations."""
    if not seconds:
        return "—"
    total = int(round(float(seconds)))
    h, rest = divmod(total, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def race_block(race: list[dict], notes: dict | None = None) -> None:
    """Garmin's race predictions, plotted as pace so the four are comparable.

    Stored since the first sync and never shown until now. Plotting the raw
    seconds would put a 29-minute 5K and a six-hour marathon on one axis, where
    the 5K becomes a flat line at the bottom — so each is divided by its own
    distance. Predicted pace per kilometre falling across all four distances is
    the same claim as "the engine is getting bigger", in a unit that can be run.
    """
    rows = [r for r in (race or [])
            if any(r.get(field) for field, _, _ in RACE_DISTANCES)]
    if not rows:
        st.caption("No race predictions stored yet — Garmin needs a few more runs.")
        return
    rows.sort(key=lambda r: str(r["day"]))
    fig = go.Figure()
    colours = {"5K": TONE["good"], "10K": "#7FB6DC", "Half": TONE["caution"],
               "Marathon": "#A98BD9"}
    for field, label, km in RACE_DISTANCES:
        xs = [r["day"] for r in rows if r.get(field)]
        ys = [float(r[field]) / 60.0 / km for r in rows if r.get(field)]
        if not xs:
            continue
        fig.add_scatter(x=xs, y=ys, mode="lines+markers", name=label,
                        line=dict(color=colours[label], width=2.2),
                        marker=dict(size=7),
                        hovertemplate="%{x|%a %d-%m-%Y}<br>%{y:.2f} min/km"
                                      f"<extra>{label}</extra>")
    fig.update_layout(yaxis_title="predicted pace (min/km)")
    ui.chart(fig, 220, date_axis=True)
    latest = rows[-1]
    first = rows[0]
    cells = []
    for field, label, _ in RACE_DISTANCES:
        if not latest.get(field):
            continue
        note, tone = "Garmin estimate", "neutral"
        if first.get(field) and first is not latest:
            delta = float(latest[field]) - float(first[field])
            # Faster is a negative delta, so the sign is flipped for reading.
            note = f"{'-' if delta < 0 else '+'}{clock(abs(delta))} since " \
                   f"{day_label(str(first['day']))}"
            tone = "good" if delta < 0 else "caution" if delta > 0 else "neutral"
        cells.append({"label": label, "value": clock(latest[field]),
                      "note": note, "tone": tone})
    ui.figures(cells)
    chart_ai_note("race", notes)


def lap_block(activity_id: str, laps: list[dict], sport: str) -> None:
    """Per-lap pace with heart rate over it, for one session.

    The single most useful chart for the problem this athlete actually has: the
    session summary says the average heart rate was 158, and this says whether
    that was 158 all the way or 140 climbing to 172 while the pace held. Auto-lap
    on the watch makes it a free kilometre-by-kilometre split.
    """
    rows = sorted((l for l in (laps or [])
                   if str(l.get("activity_id")) == str(activity_id)),
                  key=lambda l: l.get("lap_index") or 0)
    if len(rows) < 2:
        return
    ui.section("Splits", f"{len(rows)} laps, as the watch recorded them.")
    x = [f"{int(l.get('lap_index') or i + 1)}" for i, l in enumerate(rows)]
    fig = go.Figure()
    speeds = [float(l["avg_speed_mps"]) for l in rows if l.get("avg_speed_mps")]
    if len(speeds) == len(rows) and all(s > 0 for s in speeds):
        if sport == "swim":
            pace = [100.0 / s / 60.0 for s in speeds]      # min per 100 m
            unit = "min/100m"
        else:
            pace = [1000.0 / s / 60.0 for s in speeds]      # min per km
            unit = "min/km"
        fig.add_bar(x=x, y=pace, name=f"pace ({unit})",
                    marker=dict(color="rgba(127,182,220,.55)"),
                    hovertemplate="lap %{x}<br>%{y:.2f} " + unit + "<extra></extra>")
        # Reversed, so a taller bar is a faster lap rather than a slower one.
        fig.update_yaxes(autorange="reversed", title=unit)
    hrs = [l.get("avg_hr") for l in rows]
    if any(h for h in hrs):
        fig.add_scatter(x=x, y=hrs, mode="lines+markers", name="heart rate",
                        yaxis="y2", line=dict(color=TONE["bad"], width=2.2),
                        marker=dict(size=8),
                        hovertemplate="lap %{x}<br>%{y:.0f} bpm<extra></extra>")
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False,
                                      title="bpm",
                                      tickfont=dict(color=TONE["bad"], size=10)))
    fig.update_layout(xaxis_title="lap", hovermode="x unified", barmode="overlay")
    with ui.frame():
        ui.chart(fig, 240)
    drift = lap_drift(rows)
    spread = lap_pace_spread(rows)
    bits = []
    if drift.get("hr_drift_bpm") is not None:
        bits.append(f"Heart rate rose {drift['hr_drift_bpm']:+.0f} bpm from the "
                    f"first comparable lap to the last")
    if spread is not None:
        bits.append(f"pace varied {spread * 100:.1f}% across them")
    if bits:
        st.caption(". ".join(bits) + ". Heart rate climbing while the pace holds "
                   "is either a start that was too quick or the heat.")


def humid_days(acts: list[dict], weather: dict | None) -> dict[date, str]:
    """Days whose conditions were muggy enough to inflate heart rate.

    Returned so a chart can mark those points. A session that looks like lost
    fitness because it was 24C with a 22C dew point is the most common false
    alarm this app could raise, and the data to rule it out is already stored.
    """
    out: dict[date, str] = {}
    for a in acts:
        wx = (weather or {}).get(str(a.get("activity_id")))
        if not wx:
            continue
        dew = wx.get("dew_point_c")
        if dew is None or float(dew) < MUGGY_DEW_C:
            continue
        try:
            day = date.fromisoformat(str(a["start_date"])[:10])
        except (ValueError, KeyError, TypeError):
            continue
        label = f"{wx.get('temp_c', '?')}C, dew {float(dew):.0f}C"
        if wx.get("humidity_pct"):
            label += f", {float(wx['humidity_pct']):.0f}% humidity"
        out[day] = label
    return out


def training_hr_block(acts: list[dict], today: date,
                      notes: dict | None = None,
                      weather: dict | None = None,
                      ceiling: float | None = None) -> None:
    """All sports on one axis, so this is one chart rather than three."""
    # Right-aligned beside the heading rather than on a row of its own: a
    # two-option control does not deserve 40px of full-width page.
    _, ctrl = st.columns([3, 1.15], vertical_alignment="center")
    with ctrl:
        view = st.segmented_control(
            "View", ["Usual pace", "Raw"], default="Usual pace",
            key="hr_view", label_visibility="collapsed",
            help="Usual pace normalises each session to your median pace, so "
                 "a fast day and a slow day are comparable.")
    normalised = (view or "Usual pace").startswith("Usual")
    field = "hr_at_reference" if normalised else "avg_hr"

    fig = go.Figure()
    # `trend_notes`, not `notes`: the old name shadowed the stored-notes argument,
    # so the AI caption for this chart — the headline chart of the whole page —
    # was being handed a list of trend strings and quietly rendered nothing.
    drawn, trend_notes = False, []
    for sport in shown_sports():
        pts = [p for p in hr_points(acts, sport) if p.get(field)]
        if not pts:
            continue
        # Two sessions on one day would otherwise draw a vertical line, which
        # reads as a spike rather than as two data points.
        by_day: dict[date, list[dict]] = {}
        for p in pts:
            by_day.setdefault(p["date"], []).append(p)
        days = sorted(by_day)
        ys = [sum(q[field] for q in by_day[d]) / len(by_day[d]) for d in days]
        mins = [sum(q["minutes"] for q in by_day[d]) for d in days]
        raws = [sum(q["avg_hr"] for q in by_day[d]) / len(by_day[d]) for d in days]
        drawn = True
        # Two points get a line too. Three bike sessions on two days were drawn
        # as loose dots beside a run line, which reads as missing data rather
        # than as a sport with fewer sessions; the markers still show where the
        # actual sessions are.
        mode = "lines+markers" if len(days) >= 2 else "markers"
        fig.add_scatter(
            x=days, y=ys, mode=mode, name=sport,
            line=dict(color=SPORT_COLOR[sport], width=2),
            marker=dict(size=10, color=SPORT_COLOR[sport]),
            customdata=list(zip(mins, raws)),
            hovertemplate="%{x|%a %d-%m-%Y}<br>%{y:.0f} bpm<br>"
                          "%{customdata[0]:.0f} min · raw %{customdata[1]:.0f}"
                          f"<extra>{sport}</extra>")
        t = hr_trend(acts, sport, as_of=today, steady_only=False)
        if t["normalised_change_bpm"] is not None:
            trend_notes.append(f"{sport} {t['normalised_change_bpm']:+.1f} bpm")
    # The ceiling drawn on the chart, because it is the line the whole page is
    # asking about: below it was an easy session, above it was not, and without
    # the line the reader has to hold the number in their head.
    if ceiling and drawn:
        fig.add_hline(
            y=float(ceiling), line_dash="dash", line_color=TONE["good"],
            line_width=1.4,
            annotation_text=f"your easy ceiling · {float(ceiling):.0f} bpm",
            annotation_position="top left",
            annotation_font=dict(size=10, color=TONE["good"]))

    muggy = humid_days(acts, weather)
    if muggy and drawn:
        # Marked, not corrected. There is no defensible constant to subtract for
        # humidity, so the honest move is to say which points to distrust.
        rings = []
        for d in sorted(muggy):
            same = [p[field] for sp in shown_sports()
                    for p in hr_points(acts, sp)
                    if p.get(field) and p["date"] == d]
            if same:
                rings.append((d, sum(same) / len(same), muggy[d]))
        if rings:
            fig.add_scatter(
                x=[r[0] for r in rings], y=[r[1] for r in rings],
                mode="markers", name="humid",
                marker=dict(size=19, color="rgba(0,0,0,0)", symbol="circle-open",
                            line=dict(color=TONE["caution"], width=2)),
                customdata=[r[2] for r in rings],
                hovertemplate="%{customdata}<extra>humid — expect a few beats "
                              "more at the same pace</extra>")

    if not drawn:
        st.caption("No sessions with heart rate yet.")
        return
    fig.update_layout(yaxis_title="bpm")
    ui.chart(fig, 200, date_axis=True)
    chart_ai_note("training_hr", notes)
    if trend_notes:
        st.caption("Change at the same pace: " + " · ".join(trend_notes)
                   + " (going down is progress).")
    elif normalised:
        st.caption("Power-based bike sessions are excluded here: watts per beat "
                   "has no pace equivalent.")


def trend_chart(wl: list[dict], today: date) -> None:
    if not wl:
        st.caption("No wellness data yet.")
        return
    df = pd.DataFrame(wl)
    df["day"] = pd.to_datetime(df["day"])
    metric = st.radio(
        "Metric",
        ["Resting HR", "HRV", "Readiness", "Sleep", "Stress", "Respiration",
         "Blood oxygen", "Steps"],
        horizontal=True, index=0, label_visibility="collapsed")
    col, color = {
        "Resting HR": ("resting_hr", TONE["bad"]),
        "HRV": ("hrv_last_night", TONE["good"]),
        "Readiness": ("training_readiness", "#7FB6DC"),
        "Sleep": ("sleep_seconds", "#A98BD9"),
        # Stress and respiration are recovery signals in their own right, and
        # steps are the load that happens outside a session but still has to be
        # recovered from.
        "Stress": ("stress_avg", TONE["caution"]),
        "Respiration": ("respiration_avg", "#8FBF9F"),
        "Blood oxygen": ("spo2_avg", "#7FB6DC"),
        "Steps": ("steps", "#B79A6B"),
    }[metric]
    if col not in df or not df[col].notna().any():
        st.caption(f"No {metric.lower()} recorded yet.")
        return
    y = df[col] / 3600.0 if col == "sleep_seconds" else df[col]
    fig = go.Figure()
    fig.add_scatter(x=df["day"], y=y, mode="markers", name="daily",
                    marker=dict(size=6, color=color, opacity=.35),
                    hovertemplate="%{x|%a %d-%m-%Y}<br>%{y:.1f}<extra></extra>")
    fig.add_scatter(x=df["day"], y=y.rolling(7, min_periods=2).mean(), mode="lines",
                    name="7-day average", line=dict(color=color, width=2.5))
    if y.notna().sum() >= 20:
        fig.add_scatter(x=df["day"], y=y.rolling(28, min_periods=5).mean(),
                        mode="lines", name="28-day baseline",
                        line=dict(color=color, width=1.5, dash="dot"))
    fig.update_layout(yaxis_title="hours" if col == "sleep_seconds" else None)
    ui.chart(fig, 200, date_axis=True)


def volume_chart(data: dict, today: date,
                 notes: dict | None = None) -> None:
    weeks = week_summaries(data["activities"], weeks=12, as_of=today,
                           strength_rows=data["strength"])
    with Store(db_path()) as s:
        env = planner.build_envelope(planner.build_facts(s, today=today), s)
    fc = volume_forecast(weeks, ahead=4, week_index=env.week_index)
    done = [(w.week_start, w.total_minutes) for w in weeks if w.total_minutes > 0]
    if not done and not fc:
        st.caption("No volume logged yet.")
        return
    fig = go.Figure()
    if done:
        fig.add_scatter(x=[d for d, _ in done], y=[m for _, m in done],
                        mode="lines+markers", name="completed",
                        line=dict(color=TONE["good"], width=3),
                        marker=dict(size=8),
                        hovertemplate="week of %{x|%a %d-%m-%Y}<br>%{y:.0f} min"
                                      "<extra>completed</extra>")
    bridge = [done[-1]] if done else []
    pts = [(week_start_of(today) + timedelta(weeks=f["week_offset"]), f["minutes"],
            f["deload"]) for f in fc]
    fig.add_scatter(x=[d for d, _ in bridge] + [d for d, _, _ in pts],
                    y=[m for _, m in bridge] + [m for _, m, _ in pts],
                    mode="lines+markers", name="ceiling the rules allow",
                    line=dict(color="#7FB6DC", width=2.5, dash="dash"),
                    marker=dict(size=8),
                    hovertemplate="week of %{x|%a %d-%m-%Y}<br>%{y:.0f} min"
                                  "<extra>planned</extra>")
    dl = [(d, m) for d, m, is_dl in pts if is_dl]
    if dl:
        fig.add_scatter(x=[d for d, _ in dl], y=[m for _, m in dl], mode="markers",
                        name="deload week",
                        marker=dict(size=14, symbol="diamond", color=TONE["caution"]),
                        hovertemplate="week of %{x|%a %d-%m-%Y}<br>%{y:.0f} min"
                                      "<extra>deload</extra>")
    fig.update_layout(yaxis_title="minutes per week", hovermode="x unified")
    ui.chart(fig, 200, date_axis=True)
    chart_ai_note("volume", notes)

    rows = [{"week": day_label(w.week_start.isoformat()), "total minutes": w.total_minutes,
             "load": w.total_load, "rest days": w.rest_days,
             "strength": w.strength_sessions} for w in weeks if w.total_minutes > 0]
    if rows:
        with st.expander("Week by week"):
            table(pd.DataFrame(rows).iloc[::-1])


# --------------------------------------------------------------------------
# PAGE 3 — Plan
# --------------------------------------------------------------------------


def page_plan(data: dict, today: date) -> None:
    insight_banner("Plan", data, today)
    unlocked = writes_allowed()
    if not unlocked:
        st.caption("🔒 Read-only. Enter your PIN in the sidebar to make changes.")
    with Store(db_path()) as s:
        facts = planner.build_facts(s, today=today)
        env = planner.build_envelope(facts, s)
        verdict = planner.readiness_verdict(facts)

    tone = {"deload": "bad", "hold": "neutral", "build": "good"}[verdict["verdict"]]
    ui.banner(verdict["headline"], " · ".join(verdict["reasons"]), tone)
    # A band rather than four cards: this is context for the week below it, and
    # 330px of bordered boxes before the plan itself is the wrong emphasis.
    ui.figures([
        {"label": "Phase", "value": env.phase.title(),
         "note": (f"{env.weeks_to_race} weeks to go"
                  if env.weeks_to_race is not None else "no race set"),
         "tone": PHASE_TONE.get(env.phase, "neutral")},
        {"label": "Week ceiling", "value": hm(env.max_week_minutes),
         "note": "the most the rules allow"},
        {"label": "Done", "value": hm(facts.completed_this_week.total_minutes),
         "note": f"{len(facts.trained_days)} days trained"},
        {"label": "Left", "value": hm(planner.remaining_budget(facts, env)),
         "note": "still available"},
        {"label": "Hard sessions", "value": f"{env.max_quality_sessions}",
         "note": "allowed this week"},
    ])


    # The week is what this page is for, so it is drawn first — into a container
    # opened here and filled once the plan is known. A slot rather than a
    # reordering of the code: the check-in below can rebuild the plan, and code
    # that renders before it runs would be showing the previous answer.
    week_slot = st.container()

    ui.section("How do you feel?", "This shapes the week, within what the rules "
                                   "allow. Deload triggers come from data, not "
                                   "mood.")
    with st.form("checkin"):
        c = st.columns(4, wrap=True)
        sleep = c[0].slider("Sleep", 1, 5, 3)
        sore = c[1].slider("Soreness", 1, 5, 3)
        moti = c[2].slider("Motivation", 1, 5, 3)
        avail = c[3].slider("Time today (min)", 0, 300, 90, step=15)
        notes = st.text_input("Anything else", placeholder="knee feels off / "
                                                           "only 45 min tomorrow")
        use_ai = st.checkbox("Let the AI adjust it", value=ai.available())
        go_ = st.form_submit_button("Build my week", type="primary",
                                   disabled=not unlocked)
    if go_ and writes_allowed():
        with Store(db_path()) as s:
            s.save_checkin({"day": today.isoformat(), "sleep": sleep,
                            "soreness": sore, "motivation": moti,
                            "time_available_min": avail, "notes": notes})
            with st.spinner("Working it out…"):
                p = planner.plan_week(
                    s, checkin=Checkin(date=today, sleep=sleep, soreness=sore,
                                       motivation=moti, time_available_min=avail,
                                       notes=notes), today=today, use_ai=use_ai,
                    only_sports=data.get("scoped_to"))
        st.session_state["plan"] = p.model_dump(mode="json")
        st.session_state.pop("plan_editor", None)
        refresh()

    stored = st.session_state.get("plan") or (data["plan"] or {}).get("plan")
    if data.get("plan_repaired"):
        st.info("This plan was built before your current settings, so it has been "
                "re-checked against them — heart-rate targets, the spacing rule "
                "and the session floor all reflect what is saved now.")
    if not stored:
        if st.button("Build one from the rules only", disabled=not unlocked) \
                and writes_allowed():
            with Store(db_path()) as s:
                p = planner.plan_week(s, today=today, use_ai=False,
                                      only_sports=data.get("scoped_to"))
            st.session_state["plan"] = p.model_dump(mode="json")
            refresh()
            st.rerun()
        return

    origin = {"rules": "suggested by the rules",
              "ai": "suggested by the AI, inside the rules",
              "ai_repaired": "AI suggestion, corrected by the rules",
              "manual": "your own plan"}.get(stored.get("source", "rules"), "")
    with week_slot:
        ui.section("Your week", origin)
        ui.week_strip(week_cells(stored, today))
        for f in stored.get("flags", []):
            st.caption(("🤖 " if f.startswith("AI:") else "⚠ ") + f)
        if stored.get("adjustments_made"):
            with st.expander(
                    f"What the rules changed ({len(stored['adjustments_made'])})"):
                for a in stored["adjustments_made"]:
                    st.markdown(f"- {a}")

    done_rows = [d for d in stored.get("week_plan", [])
                 if d.get("purpose") == "completed"]
    ui.section("Change it", "Edit any row, add sessions, delete what you do not "
                            "want. Saving keeps exactly what you enter.")
    if done_rows:
        # Said out loud rather than just left out of the table: a session missing
        # from the editor looks like a bug until you know it is finished.
        st.caption(
            "Already done and not editable: "
            + ", ".join(f"{d['day']} {d['sport']} {d['duration_min']}′"
                        for d in done_rows)
            + ". Those are what the watch recorded, so they are the one part of "
              "the week that is not up for discussion.")
    editable = [d for d in stored.get("week_plan", [])
                if d.get("purpose") != "completed"]
    seed = pd.DataFrame([{"Day": d["day"], "Sport": d["sport"],
                          "Minutes": d["duration_min"],
                          "Zone": d.get("target_zone") or "Z2",
                          "Target HR": d.get("target_hr") or "",
                          "Note": (d.get("why") or "")[:80]} for d in editable]
                        or [{"Day": "Mon", "Sport": "bike", "Minutes": 60,
                             "Zone": "Z2", "Target HR": "", "Note": ""}])
    edited = st.data_editor(
        seed, num_rows="dynamic", hide_index=True, width="stretch",
        key="plan_editor", disabled=not unlocked,
        column_config={
            "Day": st.column_config.SelectboxColumn(options=list(DAYS), required=True),
            "Sport": st.column_config.SelectboxColumn(options=list(SPORTS),
                                                      required=True),
            "Minutes": st.column_config.NumberColumn(min_value=0, max_value=400,
                                                     step=5),
            "Zone": st.column_config.SelectboxColumn(
                options=["Z1", "Z2", "Z3", "Z4", "Z5", "technique", "n/a"]),
            "Target HR": st.column_config.TextColumn(
                width="small", disabled=True,
                help="Derived from your Garmin zones — set the zone and this follows."),
            "Note": st.column_config.TextColumn(width="large"),
        })
    b = st.columns([1, 1, 2])
    if b[0].button("Save my plan", type="primary", width="stretch",
                   disabled=not unlocked) and writes_allowed():
        days, rejected = [], []
        # The editor has no column for exercises, so a strength row would come
        # back with an empty list and the session would silently revert to
        # whatever the template said. Carried across from the stored plan by day.
        legs_before = {d["day"]: list(d.get("exercise_ids") or [])
                       for d in stored.get("week_plan", [])
                       if d.get("sport") == "strength"}
        for n, r in edited.iterrows():
            try:
                days.append(PlanDay(
                    day=str(r["Day"]), sport=str(r["Sport"]),
                    duration_min=max(0, min(400, int(r["Minutes"] or 0))),
                    target_zone=str(r["Zone"] or "Z2"),
                    exercise_ids=(legs_before.get(str(r["Day"]), [])
                                  if str(r["Sport"]) == "strength" else []),
                    purpose="chosen by athlete", why=str(r["Note"] or "")[:200]))
            except Exception as exc:  # noqa: BLE001
                rejected.append(f"row {int(n) + 1}: {type(exc).__name__}")
        if rejected:
            st.warning("Could not save: " + "; ".join(rejected))
        completed = [PlanDay.model_validate(d) for d in stored.get("week_plan", [])
                     if d.get("purpose") == "completed"]
        mine = WeekPlan(week_plan=completed + days, source="manual",
                        flags=["Your own plan — the rules were not applied."])
        with Store(db_path()) as s:
            # Your rows win, but a plan needs more than the editor can ask for:
            # the exercises for a leg session, the bpm range a zone means for
            # you, a reason for each session. Filled in here so the plan, the
            # page and the watch all carry the same thing.
            mine, _ = planner.enrich_manual(mine, s, today=today)
            s.save_plan(week_start_of(today), mine.model_dump(mode="json"), "manual")
        st.session_state["plan"] = mine.model_dump(mode="json")
        refresh()
        st.rerun()
    if b[1].button("Reset to suggestion", width="stretch"):
        st.session_state.pop("plan", None)
        st.session_state.pop("plan_editor", None)
        st.rerun()

    push = st.text_area("Or tell it what is wrong",
                        placeholder="Saturday is out, I'm travelling.")
    if (st.button("Re-suggest", disabled=not unlocked) and push.strip()
            and writes_allowed()):
        with Store(db_path()) as s:
            last = s.latest_checkin()
            ci = Checkin(date=today, sleep=(last or {}).get("sleep") or 3,
                         soreness=(last or {}).get("soreness") or 3,
                         motivation=(last or {}).get("motivation") or 3,
                         time_available_min=(last or {}).get("time_available_min") or 90,
                         notes=(last or {}).get("notes") or "") if last else None
            with st.spinner("Rethinking…"):
                p = planner.plan_week(s, checkin=ci, today=today, use_ai=True,
                                      pushback=push, previous_plan=stored,
                                      only_sports=data.get("scoped_to"))
        st.session_state["plan"] = p.model_dump(mode="json")
        st.session_state.pop("plan_editor", None)
        refresh()
        st.rerun()
    # Settings last, and deliberately. The week is what this page is for;
    # zones, the easy ceiling and weekly volume get set once and then left,
    # and having them first pushed the plan itself halfway down the page.
    st.caption("Zones, your easy ceiling, weekly targets and the training "
               "rules themselves live on the Rules page.")




# --------------------------------------------------------------------------
# PAGE 4 — Rules
#
# Everything that shapes a week, in one place and editable. It used to be an
# expander at the foot of the Plan page, which made the rules feel like a
# footnote to the plan when they are the thing that produces it — and left the
# athlete with no way to see that the endurance floor was three, never mind
# change it.
# --------------------------------------------------------------------------


PHASE_TONE = {"base": "neutral", "build": "good", "peak": "caution",
              "taper": "good"}


def goal_block(today: date, unlocked: bool) -> None:
    """The race, and the shape of the weeks between here and it.

    First on the page because it changes everything below it: with a date set,
    the growth cap, the hard-session allowance and the long sessions all move
    with how close the race is.
    """
    with Store(db_path()) as s:
        goal = goal_mod.load(s)
    ui.section("What you are training for",
               "Set a date and the plan stops treating every week the same.")
    if goal.set:
        weeks = goal.weeks_to_go(today) or 0
        phase = goal.phase(today)
        cells = [
            {"label": "Race", "value": goal.event or "your race",
             "note": day_label(goal.day.isoformat(), year=True)},
            {"label": "Weeks to go", "value": f"{max(weeks, 0)}",
             "note": "this week" if weeks <= 0 else ""},
            {"label": "Phase now", "value": phase.title(),
             "note": goal_mod.PHASE_NOTES[phase][:44],
             "tone": PHASE_TONE.get(phase, "neutral")},
        ]
        if goal.distance_km:
            cells.append({"label": "Distance",
                          "value": f"{goal.distance_km:g} km",
                          "note": f"{goal.taper_weeks()}-week taper"})
        ui.figures(cells)
        rows = goal.timeline(today)
        if rows:
            ui.rows([
                (("→ " if r["current"] else "") + r["phase"].title(),
                 (f"{r['from_weeks']} to {r['to_weeks']} weeks out"
                  if r["from_weeks"] != r["to_weeks"]
                  else f"{r['to_weeks']} weeks out"),
                 r["note"])
                for r in rows
            ])
    else:
        st.caption("No race set, so every week is a base week — easy volume, "
                   "intensity on a leash. That is the right answer while you are "
                   "building an engine, and the wrong one eight weeks out.")

    with st.expander("Set the race" if not goal.set else "Change the race"):
        with st.form("goal"):
            c = st.columns([2, 1.4, 1, 1], vertical_alignment="bottom")
            event = c[0].text_input("Race", value=goal.event,
                                    placeholder="Pune Half Marathon")
            when = c[1].date_input(
                "Date", value=goal.day or (today + timedelta(weeks=12)),
                min_value=today - timedelta(days=365),
                format="DD-MM-YYYY")
            sport = c[2].selectbox("Sport", ["run", "bike", "swim", "triathlon"],
                                   index=["run", "bike", "swim",
                                          "triathlon"].index(goal.sport)
                                   if goal.sport in ("run", "bike", "swim",
                                                     "triathlon") else 0)
            distance = c[3].number_input("km", 0.0, 300.0,
                                         float(goal.distance_km or 0.0), step=0.5)
            b = st.columns(2)
            save = b[0].form_submit_button("Save the race", type="primary",
                                           width="stretch", disabled=not unlocked)
            drop = b[1].form_submit_button("No race for now", width="stretch",
                                           disabled=not unlocked)
        if (save or drop) and writes_allowed():
            with Store(db_path()) as s:
                if drop:
                    goal_mod.clear(s)
                else:
                    goal_mod.save(s, event, when, sport=sport,
                                  distance_km=distance or None)
            refresh()
            st.rerun()
        st.caption("The distance sets the taper: three weeks for a marathon, two "
                   "for a half, one for anything shorter. A race in the past is "
                   "ignored, and no race means base phase.")


def page_rules(data: dict, today: date) -> None:
    unlocked = writes_allowed()
    with Store(db_path()) as s:
        current = rules_mod.load(s)
        facts = planner.build_facts(s, today=today)
        env = planner.build_envelope(facts, s)

    goal_block(today, unlocked)

    changed = rules_mod.changed_from_default(current)
    if changed:
        ui.banner(
            f"{len(changed)} rule{'s' if len(changed) != 1 else ''} changed from "
            f"the base-phase default",
            "; ".join(f"{name.replace('_', ' ')} {mine} (default {default})"
                      for name, (mine, default) in changed.items()),
            "caution")
    else:
        ui.banner("Running on the base-phase defaults",
                  "Three endurance sessions a week spaced a day apart, two leg "
                  "sessions, 10% growth, every fourth week easier.", "neutral")

    ui.section("What this week works out to",
               "The settings below, applied to where you are now.")
    ui.figures([
        {"label": "Week ceiling", "value": hm(env.max_week_minutes),
         "note": f"{env.progression_cap_pct:.0f}% over "
                 f"{hm(env.prev_week_minutes)} last week"},
        {"label": "Phase", "value": env.phase.title(),
         "note": (f"{env.weeks_to_race} weeks to {env.race_name or 'the race'}"
                  if env.weeks_to_race is not None else "no race set"),
         "tone": PHASE_TONE.get(env.phase, "neutral")},
        {"label": "Block week", "value": f"{env.week_index + 1}/{env.block_weeks}",
         "note": "last week of a block is a deload",
         "tone": "caution" if env.deload else "neutral"},
        {"label": "Endurance floor", "value": f"{env.min_endurance_sessions}",
         "note": "spaced a day apart" if env.space_endurance else "spacing off"},
        {"label": "Leg sessions", "value": f"{env.strength_sessions}",
         "note": "reduced in a deload" if env.deload else "a week"},
        {"label": "Rest days", "value": f"{env.min_rest_days}",
         "note": "minimum"},
        {"label": "Hard sessions", "value": f"{env.max_quality_sessions}",
         "note": "allowed this week",
         "tone": "caution" if env.max_quality_sessions == 0 else "neutral"},
    ])

    ui.section("Your settings", "Change one and the next plan follows it. "
                                "Each has a sensible range you cannot go past.")
    with st.form("rules"):
        entered: dict[str, object] = {}
        rows = rules_mod.describe(current)
        cols = st.columns(2, gap="large")
        for n, row in enumerate(rows):
            with cols[n % 2]:
                # Whole numbers stay whole. A session count rendered as "3.00"
                # reads like a measurement rather than a choice.
                whole = float(row["value"]).is_integer() and float(row["step"]) >= 1
                entered[row["name"]] = st.number_input(
                    row["label"],
                    min_value=int(row["min"]) if whole else float(row["min"]),
                    max_value=int(row["max"]) if whole else float(row["max"]),
                    value=int(row["value"]) if whole else float(row["value"]),
                    step=int(row["step"]) if whole else float(row["step"]),
                    key=f"rule_{row['name']}",
                    help=f"{row['why']} Default {row['default']}.")
        entered["space_endurance"] = st.checkbox(
            "Keep a clear day between endurance sessions",
            value=current.space_endurance,
            help="Leg strength is exempt — it belongs in the gaps.")
        b = st.columns(2)
        save = b[0].form_submit_button("Save rules", type="primary",
                                      width="stretch", disabled=not unlocked)
        reset = b[1].form_submit_button("Back to defaults", width="stretch",
                                       disabled=not unlocked)
    if (save or reset) and writes_allowed():
        with Store(db_path()) as s:
            saved = rules_mod.reset(s) if reset else rules_mod.save(s, entered)
            # Changing a rule is a planning decision, so the week is rebuilt
            # against it immediately rather than waiting to be asked.
            stored = (data.get("plan") or {}).get("plan") or {}
            if stored and stored.get("source") != "manual":
                fixed, _ = planner.reapply_rules(
                    s, stored, today=today, only_sports=data.get("scoped_to"))
                s.save_plan(week_start_of(today), fixed.model_dump(mode="json"),
                            fixed.source)
                st.session_state["plan"] = fixed.model_dump(mode="json")
        refresh()
        st.toast(f"Saved. Endurance floor {saved.min_endurance_sessions}, "
                 f"{saved.progression_cap_pct:.0f}% growth cap.")
        st.rerun()

    ui.section("When it makes you take an easy week",
               "Any one of these forces an easy week, whatever you tell it that "
               "morning. That is the point: feeling good is not evidence that "
               "you have recovered.")
    sig = recovery_signals(data["wellness"], data["activities"], as_of=today) \
        if data.get("wellness") else None
    live = []
    if sig:
        live = [
            ("HRV vs 28-day baseline",
             f"{sig.hrv_delta_pct:+.1f}%" if sig.hrv_delta_pct is not None else "—",
             f"deload at {planner.HRV_DROP_PCT:.0f}%"),
            ("Resting HR vs baseline",
             f"{sig.rhr_delta:+.1f} bpm" if sig.rhr_delta is not None else "—",
             f"deload at +{planner.RHR_RISE_BPM:.0f} bpm"),
            ("Training Readiness",
             f"{sig.training_readiness:.0f}" if sig.training_readiness else "—",
             f"deload under {planner.READINESS_FLOOR:.0f}"),
            ("Load ratio (7 vs 28 days)",
             f"{sig.acwr:.2f}" if sig.acwr else "—",
             f"deload over {planner.ACWR_CEILING:.2f}"),
        ]
    if live:
        ui.rows(live)
    else:
        st.caption("No recovery data yet, so nothing to compare against.")
    if env.deload_reasons:
        st.caption("Firing right now: " + "; ".join(env.deload_reasons))

    ui.section("Other things you cannot change",
               "Not numbers, but not yours to move either.")
    ui.rows([(what, "fixed", why) for what, why in rules_mod.FIXED])

    ui.section("Your heart rate and your targets",
               "What counts as easy for you, and how much of each sport you "
               "want in a week.")

    with st.expander("Heart-rate zones and what counts as easy for you"):
        with Store(db_path()) as _s:
            bounds = zone_bounds(_s.zones())
        if bounds:
            with Store(db_path()) as _s:
                lthr = _s.get_state("threshold_hr")
                saved_ceiling = _s.get_state("aerobic_ceiling_bpm")
            lthr_f = float(lthr) if lthr else None
            opts = aerobic_ceiling_options(bounds, lthr_f)
            ceiling = int(float(saved_ceiling)) if saved_ceiling else opts["garmin_z2_top"]

            ui.section("Your heart-rate zones",
                       "From Garmin, so they match the watch. Note Z5 starts exactly "
                       "at your threshold heart rate — these zones are anchored to "
                       "threshold, not to an estimated maximum.")
            ui.stats_row([
                {"label": f"Z{z}", "value": (f"{lo}–{hi}" if hi else f"{lo}+"),
                 "note": {1: "recovery", 2: "aerobic base", 3: "tempo",
                          4: "threshold", 5: "VO2max"}.get(z, ""),
                 "tone": "good" if z == 2 else ("bad" if z >= 4 else "neutral")}
                for z, (lo, hi) in sorted(bounds.items())
            ])

            if opts["candidates"]:
                ui.section("What counts as easy for you",
                           f"Garmin's Z2 ceiling is {opts['garmin_z2_top']} bpm — "
                           f"{opts['garmin_z2_top'] / lthr_f * 100:.0f}% of your "
                           f"{int(lthr_f)} bpm threshold, which is conservative next to "
                           f"the common threshold-anchored schemes below. If Z2 feels "
                           f"absurdly slow, you are probably right; set your own "
                           f"ceiling and every Z2 target follows it.")
                ui.stats_row([
                    {"label": c["label"], "value": f"{c['bpm']} bpm", "note": c["note"],
                     "tone": "good" if c["bpm"] == ceiling else "neutral"}
                    for c in opts["candidates"]
                ])
                with st.form("ceiling"):
                    cc = st.columns([2, 1], vertical_alignment="bottom")
                    new_ceiling = cc[0].slider(
                        "Your aerobic-base ceiling (bpm)",
                        int(bounds[2][0]) + 5, int(lthr_f) - 5, int(ceiling),
                        help="The real test is speech: the highest heart rate at which "
                             "you can still talk in full sentences. Nothing in the "
                             "data can tell you that, so this is your call.")
                    if cc[1].form_submit_button("Save ceiling", type="primary",
                                               width="stretch",
                                               disabled=not unlocked) and writes_allowed():
                        with Store(db_path()) as _s:
                            _s.set_state("aerobic_ceiling_bpm", str(int(new_ceiling)))
                            # Every Z2 target is derived from this, so a saved ceiling
                            # that leaves the plan untouched looks like it did nothing.
                            with st.spinner("Restamping your targets…"):
                                rebuilt = planner.plan_week(
                                    _s, today=today, use_ai=False,
                                    only_sports=data.get("scoped_to"))
                        st.session_state["plan"] = rebuilt.model_dump(mode="json")
                        st.session_state.pop("plan_editor", None)
                        refresh()
                        st.rerun()
                if ceiling and ceiling > (bounds[2][1] or 0):
                    st.caption(
                        f"At {ceiling} bpm you are above Garmin's Z2, so Garmin itself "
                        f"will call that time 'moderate'. The Intensity page measures "
                        f"against this ceiling instead, from your stored heart-rate "
                        f"samples, and shows both numbers side by side."
                    )

    ui.section("Your weekly targets",
               "How much of each sport you want. The scheduler builds around this; "
               "the safety rules still cap the total.")
    existing = data["targets"]
    suggestions = suggested_targets(today, data.get("scoped_to"))
    with st.form("targets"):
        st.caption("Pre-filled from your own recent weeks — edit anything you "
                   "disagree with. Use the header toggle to drop a sport "
                   "entirely; it then gets no sessions and no long-session "
                   "requirement, and its share of the week goes to the rest.")
        rows = []
        for sport in shown_sports():
            cur = existing.get(sport) or {}
            hint = suggestions.get(sport) or {}
            # An empty box asks the athlete to guess a number the app already has
            # the evidence for, so a saved value wins and a suggestion fills the
            # gap. The reasoning is shown either way.
            default_sessions = int(cur.get("sessions") or hint.get("sessions") or 0)
            default_minutes = int(cur.get("minutes") or hint.get("minutes") or 0)
            c = st.columns([1.35, 1, 1], vertical_alignment="center")
            c[0].markdown(f"{EMOJI[sport]} **{sport.title()}**")
            if hint.get("basis"):
                c[0].caption(("saved" if cur.get("sessions") else "suggested")
                             + f" · {hint['basis']}")
            rows.append({
                "sport": sport,
                # Preserved, not re-decided here: the header toggle is the one
                # place that answers "is this sport on?".
                "enabled": int(bool(cur.get("enabled", 1))),
                "sessions": c[1].number_input("sessions", 0, 7, default_sessions,
                                              key=f"ts_{sport}"),
                "minutes": c[2].number_input("minutes", 0, 900, default_minutes,
                                             step=15, key=f"tm_{sport}"),
            })
        b = st.columns(2)
        save = b[0].form_submit_button("Save targets", type="primary",
                                       width="stretch", disabled=not unlocked)
        clear = b[1].form_submit_button("Clear", width="stretch",
                                        disabled=not unlocked)
    if (save or clear) and writes_allowed():
        # Changing which sports are on IS a planning decision, so rebuild
        # immediately rather than making the athlete find a second button.
        with Store(db_path()) as s:
            if clear:
                s.clear_targets()
            else:
                s.set_targets(rows)
            last = s.latest_checkin()
            ci = Checkin(
                date=today, sleep=(last or {}).get("sleep") or 3,
                soreness=(last or {}).get("soreness") or 3,
                motivation=(last or {}).get("motivation") or 3,
                time_available_min=(last or {}).get("time_available_min") or 90,
                notes=(last or {}).get("notes") or "") if last else None
            with st.spinner("Rebuilding your week around that…"):
                plan = planner.plan_week(s, checkin=ci, today=today,
                                         use_ai=ai.available(),
                                         only_sports=data.get("scoped_to"))
        st.session_state["plan"] = plan.model_dump(mode="json")
        st.session_state.pop("plan_editor", None)
        refresh()
        st.rerun()

# --------------------------------------------------------------------------
# PAGE 5 — Log
# --------------------------------------------------------------------------


def pr_value(row: dict) -> str:
    """Garmin stores a record as a bare number whose meaning depends on the type.

    Distance records are metres and time records are seconds, and nothing in the
    payload says which — so the label decides. Showing "401.9" for a 6:42 kilometre
    is worse than showing nothing.
    """
    v = row.get("value")
    if v is None:
        return "—"
    label = (row.get("label") or "").lower()
    if label.startswith("fastest"):
        total = int(round(float(v)))
        h, rem = divmod(total, 3600)
        m, sec = divmod(rem, 60)
        return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"
    if label.startswith("longest") or "climb" in label:
        return f"{float(v) / 1000:.2f} km" if float(v) > 100 else f"{float(v):.0f} m"
    return f"{float(v):,.0f}"


@st.cache_data(show_spinner=False, ttl=900)
def _conformed_plan(stamp: float, iso_today: str, raw: str,
                    sports: tuple[str, ...] = ()) -> tuple[dict, bool]:  # noqa: ARG001
    """Push a stored plan back through the current rules. Cached on its content."""
    import json as _json
    plan = _json.loads(raw)
    try:
        def conform(s: Store) -> tuple[dict, bool]:
            today = date.fromisoformat(iso_today)
            fixed, changed = planner.reapply_rules(
                s, plan, today=today, only_sports=list(sports) or None)
            # Completions first, whatever the plan's source: a hand-edited week
            # is exempt from being re-shaped by the rules, but not from the fact
            # that Monday's session already happened. It also has to come before
            # the fill-in below, so a finished day is not handed the next
            # session in the strength cycle.
            facts = planner.build_facts(s, today=today)
            marked, moved = planner.refresh_completions(fixed, facts, s)
            if marked.source == "manual":
                # Not re-shaping — filling in. A hand-edited week keeps its days,
                # sports and minutes; what it gains is the detail the editor has
                # no column for.
                marked, filled = planner.enrich_manual(marked, s, today=today)
                moved = moved or filled
            return marked.model_dump(mode="json"), changed or moved
        return with_store(conform)
    except Exception as exc:  # noqa: BLE001 - showing the saved plan beats an error
        log.warning("Could not re-apply rules to the stored plan: %s", exc)
        return plan, False


def conformed_plan(plan: dict | None, today: date,
                   sports: list[str] | None = None) -> tuple[dict | None, bool]:
    """This week's plan as the *current* rules would have it.

    Plans are stored as data, so one built before a rule changed keeps the old
    shape — which is indistinguishable from the new rule not working. Re-running
    enforce() over it is cheap, deterministic, and uses the same code path that
    checked it in the first place.
    """
    if not plan or not plan.get("week_plan"):
        return plan, False
    import json as _json
    return _conformed_plan(db_stamp(), today.isoformat(),
                           _json.dumps(plan, sort_keys=True, default=str),
                           tuple(sorted(sports or ())))


@st.cache_data(show_spinner=False, ttl=900)
def session_zone_minutes(stamp: float, activity_id: str,
                         ceiling: int) -> dict:  # noqa: ARG001
    """One session's minutes per zone, against the athlete's own zone 2 top."""
    try:
        def read(s: Store) -> dict:
            acts = [a for a in s.activities(include_parents=True)
                    if str(a.get("activity_id")) == str(activity_id)]
            bounds = zone_bounds_with_ceiling(zone_bounds(s.zones()), ceiling)
            return zone_distribution_from_streams(
                {str(activity_id): s.stream(activity_id)}, acts, bounds)
        return {int(k): v for k, v in (with_store(read) or {}).items()}
    except Exception as exc:  # noqa: BLE001
        log.warning("Session zone minutes failed: %s", exc)
        return {}


@st.cache_data(show_spinner=False, ttl=900)
def stream_zone_minutes(stamp: float, iso_today: str, ceiling: int,
                        sport: str = "", days: int = 28) -> dict:  # noqa: ARG001
    """Minutes per zone from the stored samples, against the athlete's ceiling.

    Cached: it reads every stream, which is the heaviest query on the page.
    """
    try:
        def read(s: Store) -> dict:
            acts = s.activities()
            streams = {a["activity_id"]: s.stream(a["activity_id"]) for a in acts}
            bounds = zone_bounds_with_ceiling(zone_bounds(s.zones()), ceiling)
            return zone_distribution_from_streams(
                streams, acts, bounds, sport=sport or None,
                since=date.fromisoformat(iso_today) - timedelta(days=days))
        return with_store(read)
    except Exception as exc:  # noqa: BLE001 - fall back to Garmin's own buckets
        log.warning("Stream zone minutes failed: %s", exc)
        return {}


@st.cache_data(show_spinner=False, ttl=900)
def stream_zone_weeks(stamp: float, iso_today: str, ceiling: int, weeks: int = 8,
                      sports: tuple[str, ...] = ()) -> list:  # noqa: ARG001
    """Minutes per zone per week, against the athlete's ceiling. Cached: it reads
    every stored heart-rate stream, the same heavy query as the 28-day version."""
    try:
        def read(s: Store) -> list:
            acts = [a for a in s.activities()
                    if not sports or a.get("sport") in set(sports)]
            streams = {a["activity_id"]: s.stream(a["activity_id"]) for a in acts}
            bounds = zone_bounds_with_ceiling(zone_bounds(s.zones()), ceiling)
            rows = weekly_zone_minutes_from_streams(
                streams, acts, bounds, weeks=weeks,
                as_of=date.fromisoformat(iso_today))
            # Dates out, ISO strings back: the cache stores what this returns.
            return [{**r, "week_start": r["week_start"].isoformat()} for r in rows]
        return with_store(read) or []
    except Exception as exc:  # noqa: BLE001 - an empty panel beats a broken page
        log.warning("Weekly zone minutes failed: %s", exc)
        return []


@st.cache_data(show_spinner=False, ttl=900)
def stream_polarisation(stamp: float, iso_today: str, ceiling: int,
                        hard_floor: int,
                        sports: tuple[str, ...] = ()) -> dict:  # noqa: ARG001
    """Easy/moderate/hard from the stored HR samples rather than Garmin's zone
    rows. Cached, because it reads every stream — the one genuinely heavy query
    on the page."""
    try:
        def read(s: Store) -> dict:
            acts = [a for a in s.activities()
                    if not sports or a.get("sport") in set(sports)]
            streams = {a["activity_id"]: s.stream(a["activity_id"]) for a in acts}
            return polarisation_from_streams(
                streams, acts, ceiling=ceiling, hard_floor=hard_floor or None,
                since=date.fromisoformat(iso_today) - timedelta(days=28))
        return with_store(read)
    except Exception as exc:  # noqa: BLE001 - fall back to Garmin's own buckets
        log.warning("Stream polarisation failed: %s", exc)
        return {}


@st.cache_data(show_spinner=False, ttl=900)
def _suggested_targets(stamp: float, iso_today: str,
                       sports: tuple[str, ...] = ()) -> dict:  # noqa: ARG001
    try:
        return with_store(lambda s: planner.suggest_targets(
            s, today=date.fromisoformat(iso_today),
            only_sports=list(sports) or None))
    except Exception as exc:  # noqa: BLE001 - an empty form beats a broken page
        log.warning("Could not derive target suggestions: %s", exc)
        return {}


def suggested_targets(today: date, sports: list[str] | None = None) -> dict:
    return _suggested_targets(db_stamp(), today.isoformat(),
                              tuple(sorted(sports or ())))


def page_lifetime(data: dict, today: date) -> None:
    """Everything that is not about this week.

    A separate page because the framing is different: a weekly dashboard answers
    "am I on track", and these numbers answer "how far have I come", which is the
    question that actually keeps people training.

    Deliberately typographic rather than card-based. On a page that is almost
    entirely numbers, a grid of bordered boxes is decoration — it triples the
    vertical space and adds nothing a hairline rule does not.
    """
    acts = data["activities"]
    # The whole record here, imported history included: this is the page that
    # answers "how far have I come", and answering it with only the months since
    # the watch arrived is answering a different question.
    history = data.get("history") or acts
    if not history:
        st.caption("No activities yet — sync from the sidebar.")
        return
    tot = totals(history)
    imported = [a for a in history if (a.get("source") or "garmin") != "garmin"]
    body = data.get("profile") or {}

    span = "no dated sessions"
    if tot["first_day"]:
        span = (f"{day_label(tot['first_day'].isoformat(), year=True)} — "
                f"{day_label(tot['last_day'].isoformat(), year=True)}")
    ui.page_title("Lifetime", span)
    insight_banner("Lifetime", data, today)
    if imported:
        st.caption(
            f"Totals and the calendar include {len(imported)} imported "
            f"session(s) from before the watch. The charts need heart rate, so "
            f"they show the Garmin record only.")

    ui.figures([
        {"label": "Sessions", "value": f"{tot['sessions']:,}"},
        {"label": "Moving time", "value": hm(tot["minutes"])},
        {"label": "Distance", "value": f"{tot['km']:,.0f} km"},
        {"label": "Weeks", "value": f"{tot['weeks']:.0f}",
         "note": f"{tot['sessions'] / max(tot['weeks'], 1):.1f} / week"},
    ])

    left, right = st.columns([1.15, 1], gap="large")
    with left:
        ui.section("By sport")
        sport_rows = []
        for sp in shown_sports():
            row = tot["by_sport"].get(sp) or {}  # from the full record
            n = int(row.get("sessions") or 0)
            if not n:
                sport_rows.append((f"{EMOJI.get(sp, '')} {sp.title()}", "—",
                                   "nothing logged"))
                continue
            km = row.get("km") or 0
            sport_rows.append((
                f"{EMOJI.get(sp, '')} {sp.title()}",
                f"{km:,.0f} km" if km else hm(row.get("minutes") or 0),
                f"{n} session{'s' if n != 1 else ''} · {hm(row.get('minutes') or 0)}",
            ))
        legs = tot["by_sport"].get("strength") or {}
        if legs.get("sessions"):
            sport_rows.append((f"{EMOJI['strength']} Strength",
                               f"{int(legs['sessions'])} sessions",
                               hm(legs.get("minutes") or 0)))
        ui.rows(sport_rows)

    with right:
        ui.section("Body and physiology", "From your Garmin profile.")
        best = max((float(a.get("norm_power_w") or a.get("avg_power_w") or 0)
                    for a in acts), default=0.0)
        wkg = (best / float(body["weight_kg"])
               if best and body.get("weight_kg") else None)
        ui.rows([
            ("Age", body.get("age") or "—"),
            ("Best power", f"{best:.0f} W" if best else "—",
             f"{wkg:.2f} W/kg" if wkg else ""),
            ("Weight", f"{body['weight_kg']} kg" if body.get("weight_kg") else "—"),
            ("Height", f"{float(body['height_cm']):.0f} cm"
             if body.get("height_cm") else "—"),
            ("VO2max (run)", body.get("vo2max_run") or "—", "Garmin estimate"),
            ("Threshold HR", f"{float(body['threshold_hr']):.0f} bpm"
             if body.get("threshold_hr") else "—", "lactate threshold"),
        ])

    ui.section("What Garmin thinks you could race",
               "Garmin's own guesses, shown as pace per kilometre so all four "
               "distances fit one chart. Lower is faster.")
    with ui.frame():
        race_block(data.get("race") or [], data.get("notes"))

    records = data.get("records") or []
    if records:
        ui.section("Personal records", "Garmin's own, not recomputed here.")
        pr_rows = [(r["label"], pr_value(r),
                    day_label(r["achieved_at"]) if r.get("achieved_at") else "")
                   for r in records]
        half = (len(pr_rows) + 1) // 2
        cols = st.columns(2, gap="large")
        with cols[0]:
            ui.rows(pr_rows[:half])
        with cols[1]:
            ui.rows(pr_rows[half:])

    ui.section("Showing up",
               "Every day of the last four months. The darker the square, the "
               "longer you trained. The empty ones are the part worth looking "
               "at.")
    with ui.frame():
        consistency_block({**data, "activities": history}, today)

    ui.section("Heart rate at your usual pace",
               "The one that matters most. The same pace needing fewer beats "
               "means you are fitter. Every session on record.")
    with ui.frame():
        training_hr_block(acts, today, data.get("notes"), data.get("weather"),
                          ceiling=data.get("aerobic_ceiling"))
        chart_ai_note("lifetime_hr", data.get("notes"))

    ui.section("Resting heart rate, recovery and sleep",
               "Everything on record, not just the last few weeks. HRV is how "
               "much your heartbeat varies overnight — higher usually means "
               "better recovered.")
    with ui.frame():
        trend_chart(data["wellness"], today)
        chart_ai_note("lifetime_recovery", data.get("notes"))


def page_about(data: dict, today: date) -> None:
    """A poster: one idea, a diagram, and the numbers inside it.

    Four rewrites got here. The first read as documentation, the second as a
    spec sheet of tables and stat cards, the third said the same things at three
    times the length. This one keeps the claim, the picture and the proof, and
    cuts the rest — every figure still read live from the database, so it stays
    a description of what the app has done rather than a brochure.
    """
    counts = data.get("counts_all") or data.get("counts") or {}
    history = data.get("history") or data.get("activities") or []
    tot = totals(history) if history else {}
    notes = data.get("notes") or {}
    chart_notes = len([k for k in notes if str(k).startswith("chart:")])
    ceiling = data.get("aerobic_ceiling")
    ceiling_txt = f"{int(float(ceiling))} bpm" if ceiling else "a ceiling you set"
    with Store(db_path()) as s:
        goal = goal_mod.load(s)
    phase = goal.phase(today)
    weeks = goal.weeks_to_go(today)

    ui.hero(
        "Aerobic Engine",
        "An AI coach that cannot talk you into a stupid week.",
        f"It reads your Garmin, works out whether you are getting fitter or "
        f"just getting tired, and writes the week that follows — swim, bike, "
        f"run and leg strength as one plan. Then it sends each session to your "
        f"watch. {tot.get('sessions', 0):,} sessions on record, "
        f"{tot.get('km', 0):,.0f} km.")

    ui.flow([
        ("⌚", "Your watch", "what you actually did"),
        ("📥", "Sync", f"{counts.get('hr_streams', 0):,} heart-rate samples"),
        ("📊", "Analysis", "efficiency, drift, load — arithmetic"),
        ("🤖", "The AI", "volume, intensity, which day"),
        ("🛡", "The rules", "every limit re-checked in code"),
        ("⌚", "Back to the watch", "named sets, or a bpm range"),
    ], accent="The AI", guard="The rules")

    ui.pull(
        "A model asked how your training should go will agree with you.",
        "Say you feel strong and it offers a bigger week. That is why its "
        "answer is checked again by code that cannot be flattered.")

    left, right = st.columns(2, gap="large")
    with left:
        ui.section("The AI decides")
        st.markdown(
            f"- **Which day** the long ride lands on\n"
            f"- **How long** each session runs, inside the cap\n"
            f"- **Why**, in one line per session\n"
            f"- **What the charts say** — {chart_notes} readings written on the "
            f"last sync\n"
            f"- **Your questions**, answered from your own data")
    with right:
        ui.section("The rules decide")
        st.markdown(
            f"- **Easy** means {ceiling_txt} — your number\n"
            f"- **Growth** never more than the weekly cap\n"
            f"- **Deloads** come from recovery data, not mood\n"
            f"- **Exercises** from a fixed {len(strength.EXERCISES)}, never "
            f"invented\n"
            f"- **Weights** rise one rep, then one step, when clean")

    if goal.set and weeks is not None and weeks >= 0:
        ui.pull(f"{goal.event or 'Your race'} in {weeks} weeks — {phase} phase.",
                goal_mod.PHASE_NOTES[phase])
    else:
        ui.pull("No race set, so every week is a base week.",
                "Right while you build an engine, wrong eight weeks out from "
                "something. Set a date on the Rules page.")

    ui.section("A Monday, from the inside")
    ui.prose(
        "The top of the page says <b>strength, 18 minutes</b> — cut down "
        "because readiness came in at 31. One tap puts it on your watch with "
        "every set named. You lift, the watch counts, the next sync reads the "
        "sets back and works out what weight comes next. The workout deletes "
        "itself because it is done. Tuesday's ride arrives with a heart-rate "
        "range, and the watch buzzes if you drift above it.")

    ui.pull("Free to run, end to end.",
            "Streamlit hosting, Neon Postgres, Gemini with Groq behind it — "
            "every one a free tier. Switch the AI off and the rules still write "
            "a full week.")

    ui.section("Where to look")
    ui.flow([
        ("📍", "Today", "what to do now"),
        ("📈", "Progress", "is it working"),
        ("🗓", "Plan", "the week, and how to change it"),
        ("⚙️", "Rules", "your race, your limits"),
        ("🏔", "Lifetime", "how far you have come"),
        ("📚", "Log", "every session, every bug"),
    ])

    ui.section("Logging leg work", "The one thing it cannot do for you.")
    st.markdown(
        "- **Send it from Today**, then START → Strength → pick it. Reps "
        "counted, sets named.\n"
        "- **Or log it on the watch** — turn on rep counting first. Name any "
        "stray sets on the Log page.\n"
        "- **Or type it in** on the Log page.")
    ui.prose(
        "Any of the three. None of them is not: a session the watch never saw "
        "counts as a rest day, and tomorrow's plan is built on a week that did "
        "not happen.")

    ui.link_chips(PROFILE_LINKS)
    st.caption("Not medical advice. Tendon pain that keeps coming back is a "
               "physio, not a training problem.")


def page_log(data: dict, today: date) -> None:
    tabs = st.tabs(["Sessions", "Strength", "Data", "App log", "Bugs"])
    with tabs[0]:
        log_sessions(data, today)
    with tabs[1]:
        log_strength(data, today)
    with tabs[2]:
        log_data(data)
    with tabs[3]:
        log_app(data)
    with tabs[4]:
        log_bugs()


LOG_TONE = {"ERROR": "bad", "CRITICAL": "bad", "WARNING": "caution",
            "EVENT": "good", "INFO": "neutral"}


def log_bugs() -> None:
    """Everything reported, and what happened to it."""
    counts = with_store(bugs_mod.counts)
    ui.figures([
        {"label": "Open", "value": f"{counts.get(bugs_mod.OPEN, 0)}",
         "note": "waiting to be fixed",
         "tone": "caution" if counts.get(bugs_mod.OPEN) else "good"},
        {"label": "Fixed", "value": f"{counts.get(bugs_mod.FIXED, 0)}"},
        {"label": "Not fixing", "value": f"{counts.get(bugs_mod.WONTFIX, 0)}",
         "note": "looked at and left"},
    ])
    picked = st.segmented_control(
        "Status", ["Open", "Fixed", "Not fixing", "All"], default="Open",
        key="bug_status", label_visibility="collapsed") or "Open"
    status = {"Open": bugs_mod.OPEN, "Fixed": bugs_mod.FIXED,
              "Not fixing": bugs_mod.WONTFIX, "All": None}[picked]
    rows = with_store(lambda s: bugs_mod.listing(s, status=status, limit=200))
    if not rows:
        st.caption("Nothing here. Report one from the sidebar when something "
                   "looks wrong.")
        return
    table(pd.DataFrame([{
        "#": r["id"],
        "reported": fmt_stamp(r["reported_at"]),
        "page": r["page"] or "",
        "what happened": r["text"],
        "status": r["status"],
        "fixed": fmt_stamp(r["resolved_at"]) if r.get("resolved_at") else "",
        "what was done": r.get("resolution") or "",
    } for r in rows]))
    st.caption("Reports are fixed from the command line — `python scripts/bugs.py` "
               "lists them and marks them done — so the note about what was done "
               "is written by whoever fixed it.")


def log_app(data: dict) -> None:  # noqa: ARG001 - symmetry with the other tabs
    """What the app itself has been doing, out of the database.

    Here because the hosted dashboard redacts its own error messages and the
    container log is behind another login — so this is the only place a failure
    that happened on the phone can actually be read.
    """
    target = db_path()
    tallies = applog.counts(target, days=7)
    if tallies:
        ui.figures([
            {"label": level.title(), "value": f"{count}",
             "note": "last 7 days",
             "tone": LOG_TONE.get(level, "neutral")}
            for level, count in sorted(tallies.items())
        ])
    levels = ["ALL", "ERROR", "WARNING", "EVENT", "INFO"]
    picked = st.segmented_control("Level", levels, default="ALL",
                                 key="applog_level",
                                 label_visibility="collapsed") or "ALL"
    rows = applog.recent(target, limit=300, level=picked)
    if not rows:
        st.caption("Nothing logged yet. Warnings, errors and milestones land "
                   "here; ordinary page loads do not.")
        return
    table(pd.DataFrame([{
        "when": fmt_stamp(r["at"]), "level": r["level"],
        "where": (r["logger"] or "").replace("aerobic_engine.", ""),
        "message": r["message"],
        "detail": (r["context"] or "")[:300],
    } for r in rows]))
    st.caption(f"Newest first, {len(rows)} shown. The table keeps the most "
               f"recent {applog.KEEP_ROWS} entries and prunes itself.")


def log_sessions(data: dict, today: date) -> None:
    # The log is a record, so it shows the whole record — imported sessions
    # included, with a column saying which is which.
    acts = data.get("history") or data["activities"]
    if not acts:
        st.caption("No activities yet — sync from the sidebar.")
        return
    insight_banner("Activities", data, today)
    sports = sorted({a["sport"] for a in acts})
    chosen = st.multiselect("Sport", sports, default=sports,
                            label_visibility="collapsed")
    view = [a for a in acts if a["sport"] in chosen]
    table(pd.DataFrame([{
        "start_date": day_label(a["start_date"]), "sport": a["sport"],
        "name": a.get("name") or "", "minutes": round((a.get("duration_s") or 0) / 60),
        "km": round((a.get("distance_m") or 0) / 1000, 2),
        "avg_pace": pace_str(a["sport"], a.get("distance_m"), a.get("duration_s")),
        "avg_hr": a.get("avg_hr"), "training_load": a.get("training_load"),
        "ef": round(a["ef"], 3) if a.get("ef") else None,
        "is_steady": "yes" if a.get("is_steady") else "no",
        "steady_reason": "" if a.get("is_steady") else (a.get("steady_reason") or ""),
        "source": a.get("source") or "garmin",
    } for a in view]).iloc[::-1])

    if not view:
        return
    ui.section("Session detail")
    labels = {f"{day_label(a['start_date'])} · {EMOJI.get(a['sport'], '')} "
              f"{a['sport']} · {(a.get('duration_s') or 0) / 60:.0f} min":
              a["activity_id"] for a in reversed(view)}
    aid = labels[st.selectbox("Session", list(labels), label_visibility="collapsed")]
    act = next(a for a in view if a["activity_id"] == aid)
    ui.stats_row([
        {"label": "Duration", "value": hm((act.get("duration_s") or 0) / 60)},
        {"label": "Distance", "value": f"{(act.get('distance_m') or 0) / 1000:.2f} km"},
        {"label": "Pace", "value": pace_str(act["sport"], act.get("distance_m"),
                                            act.get("duration_s"))},
        {"label": "Avg HR", "value": f"{act['avg_hr']:.0f}" if act.get("avg_hr") else "—",
         "note": f"max {act['max_hr']:.0f}" if act.get("max_hr") else ""},
    ])
    if act.get("is_steady"):
        ui.banner("Steady aerobic work", "This session feeds the efficiency trend.",
                  "good")
    else:
        ui.banner("Not counted as steady", f"{act.get('steady_reason')}. It still "
                  f"appears on the efficiency chart, marked as a harder effort.",
                  "caution")
    # Watts per kilo, from the profile weight. Garmin's running power is an
    # estimate derived from pace and elevation rather than a measurement, so it
    # is shown as context, not promoted into the efficiency trend where it would
    # break comparability with everything already stored.
    weight = (data.get("profile") or {}).get("weight_kg")
    power = act.get("norm_power_w") or act.get("avg_power_w")
    if power and weight:
        try:
            wkg = float(power) / float(weight)
            ui.section("Power")
            ui.rows([
                ("Average power", f"{float(act.get('avg_power_w') or 0):.0f} W"),
                ("Normalised power",
                 f"{float(act['norm_power_w']):.0f} W" if act.get("norm_power_w")
                 else "—"),
                ("Watts per kilo", f"{wkg:.2f} W/kg",
                 f"at {float(weight):.0f} kg"),
                ("Watts per beat",
                 f"{float(power) / float(act['avg_hr']):.2f}"
                 if act.get("avg_hr") else "—",
                 "power divided by heart rate"),
            ])
        except (TypeError, ValueError, ZeroDivisionError):
            pass

    wx = (data.get("weather") or {}).get(aid)
    if wx and wx.get("temp_c") is not None:
        dew = wx.get("dew_point_c")
        bits = [f"{wx['temp_c']:.0f}°C"]
        if wx.get("humidity_pct"):
            bits.append(f"{wx['humidity_pct']:.0f}% humidity")
        if dew is not None:
            bits.append(f"dew point {dew:.0f}°C")
        if wx.get("wind_kph"):
            bits.append(f"wind {wx['wind_kph']:.0f} kph")
        if wx.get("condition"):
            bits.append(str(wx["condition"]).lower())
        ui.section("Conditions", " · ".join(bits))
        # Dew point, not temperature, is what limits evaporative cooling — which
        # is why a humid 24C session costs more beats than a dry 30C one.
        if dew is not None:
            if dew >= 21:
                st.warning(
                    f"A dew point of {dew:.0f}°C badly limits cooling: expect "
                    f"5-10 bpm more at the same pace, and read this session's "
                    f"efficiency as weather rather than fitness.")
            elif dew >= 16:
                st.info(
                    f"A dew point of {dew:.0f}°C costs a few beats a minute at "
                    f"the same pace. Worth remembering before reading anything "
                    f"into this session.")
            else:
                st.caption("Cool and dry enough that the numbers here are "
                           "comparable with other sessions.")

    zrows = [z for z in data["zones"] if z["activity_id"] == aid]
    ceiling = data.get("aerobic_ceiling")
    if ceiling:
        # Recounted against this athlete's zone 2 top rather than Garmin's, so
        # this bar agrees with the rest of the dashboard.
        bands = session_zone_minutes(db_stamp(), str(aid), int(float(ceiling)))
        if sum(bands.values()) > 0:
            st.markdown(f"**Time in each zone** — easy up to "
                        f"{int(float(ceiling))} bpm")
            ui.proportion_bar([(ZONE_LABELS[z], bands.get(z, 0), ZONE_COLOR[z])
                               for z in range(1, 6)])
            zrows = []
    if zrows:
        st.markdown("**Time in each heart-rate zone** — Garmin's own bands")
        ui.proportion_bar([(ZONE_LABELS[int(z["zone_number"])],
                            float(z["secs_in_zone"] or 0) / 60,
                            ZONE_COLOR[int(z["zone_number"])])
                           for z in sorted(zrows, key=lambda r: r["zone_number"])])
    with Store(db_path()) as s:
        stream = s.stream(aid)
    if stream:
        sdf = pd.DataFrame(stream)
        sdf["minutes"] = sdf["t_s"] / 60.0
        has_alt = "altitude_m" in sdf and sdf["altitude_m"].notna().any()

        # Elevation as a filled area behind the traces, not another line. It is
        # context for the other two rather than a measurement in its own right:
        # heart rate climbing while speed drops is only a bad sign if the ground
        # was flat.
        fig = go.Figure()
        if has_alt:
            fig.add_scatter(
                x=sdf["minutes"], y=sdf["altitude_m"], name="Elevation (m)",
                mode="lines", line=dict(width=0.8, color="rgba(140,158,176,.55)"),
                fill="tozeroy", fillcolor="rgba(140,158,176,.13)", yaxis="y3",
                hovertemplate="%{y:.0f} m<extra>elevation</extra>")
        for col, nm, colr in (("hr", "Heart rate", TONE["bad"]),
                              ("speed_mps", "Speed (m/s)", "#7FB6DC"),
                              ("power_w", "Power (W)", TONE["caution"])):
            if col in sdf and sdf[col].notna().any():
                fig.add_scatter(x=sdf["minutes"], y=sdf[col], mode="lines", name=nm,
                                line=dict(width=1.6, color=colr),
                                yaxis="y" if col == "hr" else "y2")
        # Zone bands behind the trace, against this athlete's own ceiling. The
        # zone bar above says how long was spent in each; this says *when*, which
        # is the difference between a steady session and a fast start.
        if "hr" in sdf and sdf["hr"].notna().any():
            bands = zone_bounds_with_ceiling(
                zone_bounds(data["zones"]),
                float(ceiling) if ceiling else None)
            top = float(sdf["hr"].max())
            for number, (low, high) in sorted(bands.items()):
                if low > top:
                    continue
                fig.add_hrect(y0=low, y1=min(high or top + 5, top + 5),
                              line_width=0, fillcolor=ZONE_COLOR[number],
                              opacity=.09, layer="below")
        layout = dict(xaxis_title="minutes into the session",
                      yaxis=dict(title="heart rate"),
                      yaxis2=dict(overlaying="y", side="right", showgrid=False),
                      hovermode="x unified")
        if has_alt:
            # Squeezed into the bottom third and unlabelled, so the terrain reads
            # as a backdrop instead of competing with heart rate.
            lo, hi = float(sdf["altitude_m"].min()), float(sdf["altitude_m"].max())
            pad = max(4.0, (hi - lo) * 0.25)
            layout["yaxis3"] = dict(overlaying="y", side="right",
                                    range=[lo - pad, lo + (hi - lo + pad) * 3.2],
                                    showgrid=False, showticklabels=False,
                                    visible=False)
        fig.update_layout(**layout)
        with ui.frame():
            ui.chart(fig, 280)
        if has_alt:
            gain = float(sdf["altitude_m"].diff().clip(lower=0).sum())
            st.caption(
                f"Climbed about {gain:.0f} m across the session "
                f"({float(sdf['altitude_m'].min()):.0f}–"
                f"{float(sdf['altitude_m'].max()):.0f} m). Heart rate rising on a "
                f"climb is terrain, not fatigue.")
        elif act.get("elevation_gain_m"):
            st.caption(
                f"{float(act['elevation_gain_m']):.0f} m of climbing in total. "
                f"The per-second elevation trace needs a re-fetch of this "
                f"session's stream.")
    if act.get("decoupling_pct") is not None:
        st.caption(
            f"Your heart rate crept up {act['decoupling_pct']:.1f}% between the "
            f"first half of this session and the second, at the same effort. "
            f"Under 5% means you held it together well.")
    lap_block(aid, data.get("laps") or [], act["sport"])


def assign_sets_block(unmapped: list[dict], unlocked: bool) -> None:
    """Let the athlete name the sets the watch could not.

    The watch records what it recognises. A pushed session now names every
    exercise, but it still drops a name occasionally — a 30-second hold comes
    back as a bare SQUAT — and a session pushed before that fix has none at all.
    Left alone those sets are dropped from the log, so the progression never sees
    the work. This is the one place a guess is acceptable, because the person
    making it was there.
    """
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for row in unmapped:
        key = (str(row.get("activity_id")),
               str(row.get("garmin_category") or "—"),
               str(row.get("garmin_name") or ""))
        groups.setdefault(key, []).append(row)

    with st.expander(f"{len(unmapped)} recorded set(s) the watch could not name"):
        st.caption("Assigning them adds the work to your log, so the weights "
                   "progress from what you actually did.")
        names = {eid: ex.name for eid, ex in strength.EXERCISES.items()}
        for (activity_id, category, name), rows in sorted(groups.items()):
            reps = [r.get("reps") or 0 for r in rows]
            secs = [round(r.get("duration_s") or 0) for r in rows]
            label = f"{category}{' · ' + name if name else ''}"
            c = st.columns([2, 2, 1], vertical_alignment="bottom")
            c[0].markdown(
                f"**{label}**  \n<span style='opacity:.6;font-size:.8em'>"
                f"{len(rows)} set(s) · "
                f"{'reps ' + '/'.join(str(int(x)) for x in reps) if any(reps) else ''}"
                f"{' · ' if any(reps) else ''}"
                f"{'holds ' + '/'.join(f'{x}s' for x in secs)}</span>",
                unsafe_allow_html=True)
            pick = c[1].selectbox(
                "Exercise", ["—", *names], format_func=lambda k: names.get(k, k),
                key=f"assign_{activity_id}_{category}_{name}",
                label_visibility="collapsed")
            if c[2].button("Assign", key=f"do_{activity_id}_{category}_{name}",
                           disabled=not unlocked or pick == "—"):
                if writes_allowed() and pick != "—":
                    indices = [r["set_index"] for r in rows]
                    with Store(db_path()) as store:
                        store.assign_exercise_sets(activity_id, pick, indices)
                        # Rebuild this activity's log rows from the sets, so the
                        # newly named work reaches the progression.
                        fresh = store.exercise_sets(activity_id)
                        day = store.activity_day(activity_id)
                        store.delete_strength_log(activity_id=activity_id)
                        rebuilt = strength.sets_to_log_rows(day, activity_id, fresh)
                        if rebuilt:
                            store.log_strength(rebuilt)
                    st.session_state["strength_flash"] = (
                        f"Assigned {len(indices)} set(s) to {names[pick]}.")
                    refresh()
                    st.rerun()


def log_strength(data: dict, today: date) -> None:
    unlocked = writes_allowed()
    log_rows = data["strength"]
    if flash := st.session_state.pop("strength_flash", None):
        st.success(flash)
    insight_banner("Strength", data, today)
    if strength.needs_physio_note(log_rows):
        st.warning(strength.PHYSIO_NOTE)
    # Rests excluded: the watch records the gap after every step as a set of its
    # own, and counting those made a clean session read as a dozen strays.
    unmapped = [s for s in data["sets"]
                if not s.get("exercise_id") and not s.get("is_rest")]
    if unmapped:
        assign_sets_block(unmapped, unlocked)
    if not unlocked:
        st.caption("🔒 Read-only. Enter your PIN in the sidebar to log a session.")

    idx = len({r["day"] for r in log_rows})
    presc = strength.build_session(log_rows, session_index=idx)
    ui.section(f"Next session ({'A' if idx % 2 == 0 else 'B'})",
               f"About {strength.session_minutes(presc)} minutes. Load rises one step "
               f"only after a session completed cleanly and pain-free.")
    with st.form("slog"):
        rows = []
        for p in presc:
            c = st.columns([3, 1, 1, 1, 1, 1], vertical_alignment="center")
            c[0].markdown(f"**{p.name}**  \n<span style='opacity:.6;font-size:.8em'>"
                          f"{ui.esc(p.note)}</span>", unsafe_allow_html=True)
            sets = c[1].number_input("Sets", 1, 8, p.sets, key=f"s{p.exercise_id}")
            reps = (c[2].number_input("Reps", 1, 30, p.reps, key=f"r{p.exercise_id}")
                    if p.reps else None)
            hold = (c[2].number_input("Hold s", 5, 120, p.hold_s,
                                      key=f"h{p.exercise_id}") if p.hold_s else None)
            kg = c[3].number_input("kg", 0.0, 200.0, float(p.load_kg or 0), step=0.5,
                                   key=f"l{p.exercise_id}")
            clean = c[4].checkbox("Done", True, key=f"c{p.exercise_id}")
            pain = c[5].checkbox("Pain", False, key=f"p{p.exercise_id}")
            rows.append({"day": today.isoformat(), "exercise_id": p.exercise_id,
                         "sets": int(sets), "reps": int(reps) if reps else None,
                         "hold_s": int(hold) if hold else None,
                         "load_kg": float(kg) or None, "clean": int(clean),
                         "pain": int(pain)})
        if st.form_submit_button("Log this session", type="primary",
                                 disabled=not unlocked) and writes_allowed():
            with Store(db_path()) as s:
                s.log_strength(rows)
            refresh()
            st.session_state["strength_flash"] = (
                "Logged. Recording it on the watch too keeps training load accurate.")
            st.rerun()

    with st.expander("Exercise guide — how to do all of them"):
        st.caption(
            "Every exercise the planner can pick from, grouped by what it "
            "protects. The list is closed on purpose: the AI can choose from it "
            "and change the sets, but it cannot invent an exercise."
        )
        by_focus: dict[str, list] = {}
        for e in strength.EXERCISES.values():
            by_focus.setdefault(e.focus or "other", []).append(e)
        # Calves and knees first: they are what actually limit run volume.
        order = ["calf / Achilles", "quad / knee", "glute / hip", "glute / drive",
                 "hamstring / glute", "shin"]
        for focus in sorted(by_focus, key=lambda f: (order.index(f)
                                                     if f in order else 99, f)):
            st.markdown(f"#### {focus.title()}")
            for e in sorted(by_focus[focus], key=lambda x: x.name):
                with st.container(border=True):
                    exercise_howto(e)

        ui.section("Cadence work",
                   "Not strength, and not part of a leg session — these go inside "
                   "an easy run or ride. Nothing here jumps: the ban on "
                   "plyometrics in base phase is about impact, and cadence is "
                   "raised with quicker, shorter steps rather than hops.")
        for drill in strength.DRILLS.values():
            with st.container(border=True):
                st.markdown(f"**{drill['name']}** — {drill['dose']}")
                st.caption(drill["where"])
                st.markdown(f"**Set up:** {drill['setup']}")
                st.markdown("\n".join(f"{i}. {x}"
                                       for i, x in enumerate(drill["steps"], 1)))
                st.markdown(f"**Common mistake:** {drill['mistakes']}")
                st.caption(f"Why it matters: {drill['why']}")
    if log_rows:
        with st.expander("History"):
            h = pd.DataFrame(log_rows)[["day", "exercise_id", "sets", "reps", "hold_s",
                                        "load_kg", "clean", "pain"]].copy()
            h["day"] = h["day"].map(day_label)
            table(h.iloc[::-1])


def log_data(data: dict) -> None:
    status = sync_mod.guard_status(db_path())
    ui.stats_row([
        {"label": "Last sync",
         "value": (data["last_sync"] or "never")[:10],
         "note": (data["last_sync"] or "")[11:16]},
        {"label": "Activities", "value": data["counts"]["activities"]},
        {"label": "Wellness days", "value": data["counts"]["daily_wellness"]},
        {"label": "Zone records", "value": data["counts"]["activity_zones"]},
    ])
    ui.section("Garmin request budget",
               "Deliberately conservative — Garmin publishes no limits and does lock "
               "accounts. The breaker trips on a single 429 and stays shut for an hour.")
    ui.stats_row([
        {"label": "This hour",
         "value": f"{status['calls_this_hour']}/{status['hour_limit']}"},
        {"label": "Today", "value": f"{status['calls_today']}/{status['day_limit']}"},
        {"label": "Breaker", "value": "open" if status["breaker_open"] else "closed",
         "note": (f"{status['breaker_minutes_left']:.0f} min left"
                  if status["breaker_open"] else "requests allowed"),
         "tone": "bad" if status["breaker_open"] else "good"},
        {"label": "Next sync",
         "value": f"{status['sync_cooldown_min']:.0f} min" if status["sync_cooldown_min"]
                  else "ready", "note": "cooldown between syncs"},
    ])
    if data["wellness"]:
        with st.expander("Daily wellness"):
            w = pd.DataFrame(data["wellness"])
            keep = [c for c in ("day", "resting_hr", "hrv_last_night",
                                "training_readiness", "vo2max_run", "sleep_seconds",
                                "training_status") if c in w]
            w = w[keep].copy()
            if "sleep_seconds" in w:
                w["sleep_hours"] = (w.pop("sleep_seconds") / 3600).round(1)
            w["day"] = w["day"].map(day_label)
            table(w.iloc[::-1])
    if data["checkins"]:
        with st.expander("Your check-ins"):
            c = pd.DataFrame(data["checkins"])
            c["day"] = c["day"].map(day_label)
            table(c)
    if data["sets"]:
        with st.expander("Watch-recorded strength sets"):
            table(pd.DataFrame(data["sets"]))


# --------------------------------------------------------------------------
# sidebar + main
# --------------------------------------------------------------------------


def bug_form() -> None:
    """Report something broken, from wherever you noticed it.

    In the sidebar rather than on a page of its own because that is where you
    are when something looks wrong, and it records which page you were on so the
    report does not have to say.
    """
    counts = with_store(bugs_mod.counts)
    open_count = counts.get(bugs_mod.OPEN, 0)
    # "Or suggest something": most of what gets typed in here is not a bug, and
    # a box labelled only for bugs quietly discourages the other half.
    label = (f"Report a bug or suggest something ({open_count} open)"
             if open_count else "Report a bug or suggest something")
    with st.expander(label):
        with st.form("bug_report", clear_on_submit=True):
            text = st.text_area(
                "What happened, or what would help?", height=110,
                label_visibility="collapsed",
                placeholder="Something broken, or something you wish it did. "
                            "The page you are on is recorded for you.")
            sent = st.form_submit_button("Send", type="primary")
        if sent:
            bug_id = with_store(lambda s: bugs_mod.report(
                s, text, page=st.session_state.get("page")))
            if bug_id:
                st.success(f"Saved as #{bug_id}. It stays in the database until "
                           f"it is fixed.")
            else:
                st.caption("Nothing to save." if not (text or "").strip()
                           else "Could not save that — try again in a moment.")
        if open_count:
            for row in with_store(lambda s: bugs_mod.listing(s, limit=5)):
                st.caption(f"#{row['id']} · {fmt_stamp(row['reported_at'])}"
                           + (f" · {row['page']}" if row["page"] else "")
                           + f" — {row['text'][:90]}")


def count_visit() -> dict[str, int]:
    """Record this session once, and return the running totals.

    Once per session, not once per rerun: Streamlit re-executes the script on
    every interaction, and counting those would report how often a slider moved.
    Streamlit strips <script> out of markdown — verified in a browser — so a
    third-party analytics snippet would not run anyway, and an image-based hit
    counter would tell someone else's server every time this page is opened.
    """
    if "visit_counted" not in st.session_state:
        st.session_state["visit_counted"] = True
        try:
            headers = dict(st.context.headers or {})
        except Exception:  # noqa: BLE001 - no request context in a headless run
            headers = {}
        key = st.session_state.get("visit_key") or uuid4().hex
        st.session_state["visit_key"] = key
        # The user agent only. The forwarding address changes per websocket
        # connection on Community Cloud, which counted one visitor as five.
        with_store(lambda s: visits_mod.record(
            s, key, user_agent=headers.get("User-Agent"),
            url=str(getattr(st.context, "url", "") or "")))
    return with_store(lambda s: visits_mod.summary(s))


def sidebar(data: dict) -> None:
    with st.sidebar:
        # No logo or app name here: the top bar carries both, and a third copy
        # in the sidebar is just noise.
        st.subheader(data["name"] or "Athlete", anchor=False)
        # Marks only here. The logos say which is which, and the labels were
        # three words each on the narrowest column in the app.
        ui.link_chips(PROFILE_LINKS, labels=False)
        # Which store is in use, always visible. Neon is the real database and
        # the SQLite file is a local fallback, and the two cost a whole evening
        # once already: the CLI wrote to the file while the dashboard read Neon
        # and nothing errored, the numbers just disagreed.
        on_neon = is_postgres(db_path())
        backends = os.getenv("AI_BACKEND", ai.DEFAULT_CHAIN)
        ready = ai.available()
        # The phase came from a hardcoded "Base" until races existed, which was
        # true for as long as nothing knew about a date and wrong the moment one
        # was set. And the AI line was a caption at the foot of the sidebar: it
        # is the thing that makes this more than a dashboard, so it belongs in
        # the table with everything else that says what you are working with.
        with Store(db_path()) as _s:
            goal = goal_mod.load(_s)
        phase = goal.phase(date.today()).title()
        weeks = goal.weeks_to_go(date.today())
        ui.rows([
            ("Watch", "Forerunner 265"),
            ("Phase", phase,
             f"{weeks} weeks to {goal.event or 'race'}"
             if weeks is not None and weeks >= 0 else "no race set"),
            ("AI", backends if ready else "off",
             "planning and summaries" if ready else "rules-only plan"),
            ("Database", "Neon Postgres" if on_neon else "local file",
             "" if on_neon else "not the hosted store"),
            ("Last synced",
             fmt_stamp(data["last_sync"])),
            ("On record", f"{data['counts']['activities']} activities",
             f"{data['counts']['daily_wellness']} days"),
        ])
        if not on_neon:
            st.warning(
                f"Reading **{db_path()}**, not Neon. Anything saved here — a "
                f"plan, a rule, a logged session — stays on this machine and "
                f"will not appear on the hosted app. Set DATABASE_URL to point "
                f"at Neon.")
        st.divider()
        unlock_control()
        sync_control()
        st.divider()
        if st.button("Reload page data", width="stretch"):
            refresh()
            st.rerun()
        bug_form()
        seen = count_visit()
        if seen.get("visits"):
            # All three spans, each labelled. One number with "12 today" beside
            # it read as though the 12 were the total.
            # Page loads are deliberately absent: the visit is recorded once per
            # session, so that figure would only ever equal the visit count and
            # look like a second, more precise measurement.
            ui.rows([
                ("Visits, all time", f"{seen['visits']:,}",
                 f"{seen['devices']:,} device{'' if seen['devices'] == 1 else 's'}"),
                ("This week", f"{seen['recent']:,}"),
                ("Today", f"{seen['today']:,}"),
            ])

        st.caption("Not medical advice. Persistent tendon pain is a physio visit.")


# The sports a filter can actually select: what the athlete does, not what the
# planner can schedule (which also has "rest" and the composite "brick").
FILTER_SPORTS = ("run", "bike", "swim", "strength")


PAGES = ("Today", "Progress", "Plan", "Rules", "Lifetime", "Log",
         "About")


def nav() -> str:
    """Top-level navigation, and the single biggest thing keeping the app quick.

    This used to be st.tabs, which executes every tab body on every rerun
    whether or not you are looking at it. Four pages of charts cost about five
    seconds a click; one page costs a little over one. Anything that reruns the
    script — unlocking, moving a slider — paid for all four.

    Kept in session state so a rerun does not bounce you back to Today.
    """
    current = st.session_state.get("page")
    picked = st.segmented_control(
        "Page", PAGES, default=current if current in PAGES else "Today",
        key="nav", label_visibility="collapsed")
    page = picked or current or "Today"
    st.session_state["page"] = page
    return page


def filter_summary(today: date, sports: list[str]) -> str:
    """The popover's label — short, and only as long as it needs to be.

    "This week · All sports" is the default state, so spelling it out spends the
    widest label in the header saying nothing is filtered. The sports half
    appears only when it is actually narrowing something.
    """
    monday = week_start_of(today)
    if monday == week_start_of(date.today()):
        span = "This week"
    elif monday == week_start_of(date.today()) + timedelta(weeks=1):
        span = "Next week"
    else:
        span = monday.strftime("%d-%m")
    if set(sports) >= set(FILTER_SPORTS):
        return span
    return f"{span} · " + ", ".join(sp[:4].title() for sp in sports)


def filter_popover(data: dict, today: date) -> tuple[date, list[str]]:
    """Week and sport filters behind one trigger.

    One copy for the whole app, not one per page: Streamlit executes every tab
    body on every rerun, so per-page copies would need distinct widget keys and
    would then disagree about what the page is showing.
    """
    remembered = st.session_state.get("week_choice")
    label = filter_summary(
        remembered if isinstance(remembered, date) else today,
        active_sports(data))
    with st.popover(f"⚙ {label}"):
        today = week_picker(today)
        sports = sport_filter(data)
        if set(sports) != set(FILTER_SPORTS):
            st.caption("Recovery, HRV and sleep still cover everything — they "
                       "are not attributable to one sport, and the written "
                       "summary describes all of your training.")
    return today, sports


def header_controls(data: dict, today: date) -> tuple[str, date, list[str]]:
    """Title, navigation and filters. One place, so no page can disagree."""
    # Three full-width rows, in reading order: what this is, where you are, what
    # you are looking at. Sharing a row was the mistake — six tabs plus a week
    # selector plus four pills do not fit across one column, so the nav wrapped
    # and the filters ended up beside the wrap.
    #
    # The filters are left-aligned under the nav rather than pushed right: a
    # right-aligned control with an empty half-page to its left reads as
    # floating, and this is a toolbar, which belongs under the thing it filters.
    # A keyed container, because CSS needs something stable to pin. Streamlit
    # renders `key` as a class on the wrapper, which is the only reliable handle
    # it offers — the emotion class names change between releases, and the
    # brand's own vertical block is the whole page, so sticking that would pin
    # everything.
    with st.container(key="topbar"):
        # Name only. Who the athlete is, which watch, which phase and when it
        # last synced are all reference detail — true, rarely needed, and they
        # were costing a wrapped line at the top of every page. They live in the
        # sidebar now.
        ui.brand("Aerobic Engine")

        # The filters live in a popover rather than laid out beside the tabs.
        # Column ratios only ever "fit" a window you happened to test: a week
        # selector plus four pills next to six tabs wraps as soon as the
        # viewport, the font size or the number of sports changes. A single
        # trigger button always fits, at any width, and its label carries the
        # current state so nothing is hidden.
        nav_col, filter_col = st.columns([7, 2], vertical_alignment="center")
        with nav_col:
            page = nav()
        with filter_col:
            today, sports = filter_popover(data, today)
    if set(sports) != set(FILTER_SPORTS):
        st.caption("Recovery, HRV and sleep still cover everything — they are "
                   "not attributable to one sport, and the written summary "
                   "describes all of your training.")
    return page, today, sports


def shown_sports() -> tuple[str, ...]:
    """Endurance sports currently in scope, in their canonical order.

    Read from session state rather than threaded through every chart helper:
    the scope is decided once per run, before any tab renders, and passing it
    down six call chains would be a lot of plumbing for one tuple.
    """
    picked = st.session_state.get("scope") or FILTER_SPORTS
    return tuple(sp for sp in ENDURANCE_SPORTS if sp in set(picked))


def active_sports(data: dict) -> list[str]:
    """Sports currently switched on. No saved rows at all means everything is on."""
    targets = data.get("targets") or {}
    off = {sp for sp, t in targets.items() if not t.get("enabled", 1)}
    return [sp for sp in FILTER_SPORTS if sp not in off]


def sport_filter(data: dict) -> list[str]:
    """Which sports the whole dashboard is about — the plan included.

    Saved rather than per-session, because "I am only doing run and bike right
    now" is a standing decision, not a passing view preference. It writes
    weekly_targets.enabled — the same flag the planner already reads — so there
    is one answer to "is swim on?" rather than two that can disagree.

    A locked visitor still gets the filter, but only for their own session: the
    saved default belongs to the owner, and reading a shared dashboard should not
    let a stranger change what it plans.
    """
    saved = active_sports(data)
    picked = st.pills(
        "Sports shown", FILTER_SPORTS, selection_mode="multi", default=saved,
        format_func=lambda sp: f"{EMOJI.get(sp, '•')} {sp.title()}",
        key="sport_pills",
        help="Filters every tab and the plan. Saved until you change it.")

    # Deselecting everything means "show me everything" rather than an empty
    # dashboard — and a week with every sport disabled is not a plan the planner
    # can honour anyway.
    chosen = list(picked) if picked else list(FILTER_SPORTS)
    if set(chosen) == set(saved):
        return chosen

    if not writes_allowed():
        st.caption("🔒 This filter applies to your view only — unlock to save it "
                   "as the default and re-plan around it.")
        return chosen

    with Store(db_path()) as store:
        store.set_sport_enabled({sp: sp in chosen for sp in FILTER_SPORTS})
    refresh()
    st.rerun()
    return chosen  # unreachable; rerun() raises


def scope_to_sports(data: dict, sports: list[str]) -> dict:
    """Return `data` with the activity-shaped entries narrowed to `sports`.

    Whole-body signals — resting HR, HRV, readiness, sleep — are deliberately
    left alone: they are not attributable to one sport, and filtering them would
    invent a "running resting HR" that does not exist.
    """
    st.session_state["scope"] = list(sports or FILTER_SPORTS)
    if not sports or set(sports) >= set(FILTER_SPORTS):
        return dict(data, scoped_to=list(FILTER_SPORTS),
                counts_all=dict(data["counts"]))
    keep = set(sports)
    # A brick counts as both its parts, so it survives a bike-only or run-only
    # filter rather than vanishing from the record.
    if keep & {"bike", "run"}:
        keep.add("brick")

    scoped = dict(data)
    for key in ("activities", "all_activities", "history"):
        scoped[key] = [a for a in data.get(key) or [] if a.get("sport") in keep]
    ids = {a.get("activity_id") for a in scoped["all_activities"]}
    scoped["zones"] = [z for z in data.get("zones") or []
                       if z.get("activity_id") in ids]
    if "strength" not in keep:
        scoped["strength"] = []
        scoped["sets"] = []
    else:
        scoped["sets"] = [x for x in data.get("sets") or []
                          if x.get("activity_id") in ids or not x.get("activity_id")]
    scoped["counts"] = dict(data["counts"], activities=len(scoped["activities"]))
    # The whole record, kept alongside the filtered view. The About page is
    # describing the app rather than this view of it, and "6 sessions on record"
    # under a filter that hides two sports is simply wrong there.
    scoped["counts_all"] = dict(data["counts"])
    # The stored plan was built when the filter may have been wider, so filter it
    # for display too — otherwise a run-and-bike dashboard still lists a swim.
    plan = data.get("plan")
    if plan and isinstance(plan.get("plan"), dict):
        inner = dict(plan["plan"])
        inner["week_plan"] = [d for d in inner.get("week_plan") or []
                              if d.get("sport") in keep | {"rest"}]
        # An adjustment about a sport that is no longer shown ("dropped 21 min
        # swim") explains a decision the athlete can no longer see, so it reads
        # as a contradiction rather than a reason.
        dropped = set(FILTER_SPORTS) - keep
        inner["adjustments_made"] = [
            a for a in inner.get("adjustments_made") or []
            if not any(sp in str(a).lower() for sp in dropped)]
        scoped["plan"] = dict(plan, plan=inner)
    scoped["scoped_to"] = sorted(sports)
    scoped["plan_repaired"] = data.get("plan_repaired", False)
    return scoped


def week_picker(today: date) -> date:
    """Pick a whole Monday-to-Sunday week.

    Streamlit's date_input renders a calendar that starts its weeks on Sunday and
    offers no way to change that, which contradicted the Monday-anchored weeks
    used everywhere else. Choosing the week directly removes the ambiguity.
    """
    this_monday = week_start_of(date.today())
    options = [this_monday - timedelta(weeks=i) for i in range(-1, 12)]
    labels = {}
    for mon in options:
        sun = mon + timedelta(days=6)
        span = f"{mon.strftime('%d-%m')} – {sun.strftime('%d-%m')}"
        if mon == this_monday:
            span += "  (this week)"
        elif mon > this_monday:
            span += "  (next week)"
        labels[span] = mon

    keys = list(labels)
    # Remembered, like the sport toggle: having the week snap back to "this week"
    # on every rerun made planning a past or future week impossible.
    remembered = st.session_state.get("week_choice")
    current = remembered if remembered in labels.values() else week_start_of(today)
    index = next((i for i, k in enumerate(keys) if labels[k] == current), 1)
    picked = labels[st.selectbox(
        "Week (Mon–Sun)", keys, index=index, key="week_select",
        help="Monday to Sunday. Everything on the page follows this week.")]
    st.session_state["week_choice"] = picked

    if picked == this_monday:
        return date.today()          # keep "today" real for the current week
    return picked + timedelta(days=6)  # otherwise show the whole week, ending Sunday


def unlock_control() -> None:
    gate = pin_gate()
    for w in gate.warnings():
        st.warning(w)
    if not gate.configured:
        return
    if writes_allowed():
        st.success("🔓 Unlocked")
        if st.button("Lock", width="stretch"):
            st.session_state.pop("unlocked_at", None)
            st.rerun()
        return
    wait = gate.lockout_remaining()
    if wait > 0:
        st.error(f"Locked for {wait:.0f}s")
        return
    with st.form("unlock", clear_on_submit=True):
        pin = st.text_input("PIN", type="password", label_visibility="collapsed",
                            placeholder="PIN to make changes")
        if st.form_submit_button("Unlock", type="primary", width="stretch"):
            ok, msg = gate.verify(pin)
            del pin
            if ok:
                st.session_state["unlocked_at"] = time.time()
                st.rerun()
            st.error(msg)


def sync_control() -> None:
    ok, why = with_store(lambda s: sync_mod.can_sync(store=s))
    unlocked = writes_allowed()
    st.caption(why if unlocked else "Locked — enter your PIN to sync.")
    if st.button("Refresh from Garmin", type="primary", width="stretch",
                 disabled=not (ok and unlocked)) and writes_allowed():
        box = st.empty()
        try:
            with st.spinner("Syncing…"):
                stats = sync_mod.sync(db=db_path(), streams=True, wellness=True,
                                      allow_password_login=False,
                                      progress=lambda m: box.caption(m))
            refresh()
            box.success(f"{stats.get('activities_new', 0)} new activities, "
                        f"{stats.get('wellness_days', 0)} wellness days.")
            st.rerun()
        except GarminBlocked as exc:
            box.warning(str(exc))
        except Exception as exc:  # noqa: BLE001
            log.exception("Sync failed")
            box.error(f"Sync failed ({type(exc).__name__}). See the terminal.")


def main() -> None:
    ui.load_css()
    if not read_gate():
        return
    today = date.today()
    target = db_path()
    # Only a local file can be "missing"; a Postgres URL either connects or
    # raises, and Path(url).exists() is always False — which used to stop the
    # hosted app dead before it ever tried to connect.
    if not is_postgres(target) and not Path(target).exists():
        ui.page_title("Aerobic Engine")
        st.error("No database yet. Run `python scripts/fetch.py` first.")
        return
    try:
        data = load(db_stamp())
    except Exception as exc:  # noqa: BLE001
        ui.page_title("Aerobic Engine")
        log.exception("Could not open the database")
        st.error(f"Could not open the database ({type(exc).__name__}). "
                 "Check DATABASE_URL and that the host is reachable.")
        return

    sidebar(data)
    page, today, sports = header_controls(data, today)

    # Re-check the stored plan against the current rules here rather than in each
    # page. Doing it per page meant every page had to remember to, and the Today
    # week strip did not — so the dashboard showed a plan that predated the rules
    # it claimed to follow.
    stored_row = data.get("plan") or {}
    fresh, repaired = conformed_plan(stored_row.get("plan"), today, sports)
    if stored_row and fresh is not None:
        data = dict(data, plan=dict(stored_row, plan=fresh),
                   plan_repaired=repaired)

    data = scope_to_sports(data, sports)

    {"Today": page_today, "Progress": page_progress, "Plan": page_plan,
      "Rules": page_rules,
     "Lifetime": page_lifetime, "Log": page_log,
     "About": page_about}[page](data, today)


main()
