"""Aerobic Engine — training dashboard.

Four pages, in the order you actually need them: what to do today, whether it is
working, what is coming, and the raw record. All logic lives in `core/`; this file
reads the database, draws, and collects input. Presentation primitives are in
`app/ui.py` so this file stays about structure rather than styling.

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import hmac
import logging
import os
import sys
import sqlite3
import threading
import time
import zlib
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

import app.ui as ui  # noqa: E402
from core import ai, insights, planner, strength, sync as sync_mod  # noqa: E402
from core.analysis import (  # noqa: E402
    ZONE_LABELS,
    aerobic_ceiling_options,
    zone_bounds,
    baseline_trend,
    hr_points,
    hr_trend,
    cadence_stats,
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

try:  # psycopg is only installed where Postgres is used
    from psycopg import InterfaceError, OperationalError
except ImportError:  # pragma: no cover - SQLite-only environments
    class InterfaceError(Exception): ...
    class OperationalError(Exception): ...

load_dotenv()
log = logging.getLogger("aerobic_engine.ui")
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


def table(df: pd.DataFrame, **kw) -> None:
    out = df.drop(columns=[c for c in df.columns if c in DROP_COLS])
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
        # A remote database has no mtime, so the sync marker stands in. One
        # cheap query per rerun, and unlike a constant it also notices a sync
        # run from the command line rather than from this page.
        try:
            marker = with_store(lambda s: s.get_state("last_sync")) or ""
            return float(zlib.crc32(marker.encode("utf-8")))
        except Exception:  # noqa: BLE001 - a dead cache key beats a dead page
            log.warning("could not read the sync marker for the cache key")
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


def read_gate() -> bool:
    expected = _secret("DASHBOARD_PASSWORD")
    if not expected or st.session_state.get("authed"):
        return True
    ui.page_title("Aerobic Engine")
    with st.form("gate"):
        pw = st.text_input("Password", type="password")
        if st.form_submit_button("Enter", type="primary"):
            if hmac.compare_digest(pw.encode("utf-8"), expected.encode("utf-8")):
                st.session_state["authed"] = True
                st.rerun()
            st.error("Incorrect.")
    return False


# --------------------------------------------------------------------------
# small formatters
# --------------------------------------------------------------------------


# The athlete trains in India, and a cloud host runs on UTC — so the cutoff is
# anchored to a real timezone rather than to wherever the server happens to be.
LOCAL_TZ = ZoneInfo(os.getenv("LOCAL_TZ", "Asia/Kolkata"))
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
    try:
        d = date.fromisoformat(str(iso)[:10])
    except (TypeError, ValueError):
        return str(iso)
    return d.strftime("%a %d %b %Y") if year else d.strftime("%a %d %b")


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
    # Headline only, with the prose and the numbers folded into one expander.
    # Printing the headline and then a paragraph restating it, followed by a
    # separate collapsed row, cost about 130px at the top of every page to say
    # one sentence twice.
    ui.banner(ins.headline, "", tone)
    if prose or ins.bullets:
        with st.expander("Why, and the numbers behind it"):
            if prose:
                st.markdown(prose)
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
    if ex.setup:
        st.markdown(f"**Set up:** {ex.setup}")
    if ex.steps:
        st.markdown("\n".join(f"{i}. {step}" for i, step in enumerate(ex.steps, 1)))
    if ex.mistakes:
        st.markdown(f"**Common mistake:** {ex.mistakes}")
    if ex.why:
        st.caption(f"Why it matters: {ex.why}")
    if ex.load_note:
        st.caption(f"Progressing: {ex.load_note}")


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
            f"afterwards.")
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
    label = f"{sport.title()} {minutes}m · Aerobic Engine"

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
               "No heart-rate range was set for this session, so it is timed only."))
        refresh()
    except GarminBlocked as exc:
        st.warning(str(exc))
    except Exception as exc:  # noqa: BLE001
        log.exception("Could not send the endurance workout")
        st.error(f"Could not send it ({type(exc).__name__}). The session is "
                 f"unchanged.")


def strength_howto_block(exercise_ids: list[str], log_rows: list[dict],
                         session_index: int = 0) -> None:
    """The full session, with instructions, for the day it is scheduled."""
    presc = {x.exercise_id: x for x in
             strength.build_session(log_rows, session_index=session_index)}
    ids = [e for e in exercise_ids if e in strength.EXERCISES] or list(presc)
    if not ids:
        return
    ui.section("How to do today's session",
               "Slow and controlled beats heavy. Stop a set if something sharp "
               "appears — soreness is fine, pain is not.")
    send_to_watch(list(presc.values()), date.today(),
                  f"Legs {chr(65 + session_index % 3)} · Aerobic Engine")
    for i, eid in enumerate(ids):
        with st.container(border=True):
            exercise_howto(strength.EXERCISES[eid], presc.get(eid))
        if i < len(ids) - 1:
            st.write("")


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

    left, right = st.columns([2, 1], gap="medium")
    with left:
        heading = "Tomorrow" if rolled else "Today"
        ui.section(heading, f"It is past {EVENING_CUTOFF_HOUR}:00 — showing "
                            f"{focus_day.strftime('%A')} instead."
                   if rolled else "")
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
            ui.today_card("rest", "Session already logged today",
                          " · ".join(f"{d['sport']} {d['duration_min']} min"
                                     for d in done_today))
        else:
            ui.today_card("rest", "Nothing scheduled",
                          "Rest is where the adaptation happens.")
        # Sits here rather than at the foot of the page: it fills the column
        # beside the stat cards, and it is the most useful thing on the screen.
        insight_banner("Overview", data, today)
    with right:
        tone = {"deload": "bad", "hold": "neutral", "build": "good"}[verdict["verdict"]]
        ui.section("Right now")
        ui.stat("Verdict", verdict["headline"].rstrip("."),
                " · ".join(verdict["reasons"])[:90] or "nothing arguing either way",
                tone, small=True)
        ui.stat("Readiness",
                f"{sig.training_readiness:.0f}" if sig and sig.training_readiness else "—",
                (sig.training_status or "morning reading") if sig else "no data",
                "bad" if sig and sig.training_readiness and sig.training_readiness < 35
                else "neutral")
        ui.stat("Resting HR",
                f"{sig.rhr_recent:.0f} bpm" if sig and sig.rhr_recent else "—",
                f"{sig.rhr_delta:+.1f} vs 28-day" if sig and sig.rhr_delta is not None
                else "building a baseline",
                "bad" if sig and sig.rhr_delta and sig.rhr_delta > 3 else "neutral",
                small=True)

    wk = week_summaries(acts, weeks=1, as_of=today,
                        strength_rows=data["strength"])[-1]
    ui.section("This week",
               f"{hm(wk.total_minutes)} done of a {hm(env.max_week_minutes)} ceiling")
    ui.week_strip(week_cells(plan, today))

    nxt = next_week_plan(today, data.get("scoped_to"))
    if nxt:
        total = sum(d["duration_min"] for d in nxt.get("week_plan", []))
        ui.section(f"Next week · {day_label((week_start_of(today) + timedelta(weeks=1)).isoformat())} onwards",
                   f"{hm(total)} planned — provisional, and re-derived when the week "
                   f"arrives.")
        ui.week_strip(week_cells(nxt, week_start_of(today) + timedelta(weeks=1),
                                 mark_today=False))

    # Today's strength session, spelled out. It is the one sport where knowing
    # what to do is not enough — the exercises are only protective if they are
    # done slowly and in the right position.
    legs_today = next((d for d in todo if d["sport"] == "strength"), None)
    if legs_today:
        strength_howto_block(
            list(legs_today.get("exercise_ids") or []), data["strength"],
            session_index=len({str(r["day"]) for r in data["strength"]}))





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
             "done": e.get("purpose") == "completed"}
            for e in entries if e["day"] == name and e["sport"] != "rest"
        ]
        cells.append({"name": name, "date": d.strftime("%d %b"),
                      "today": mark_today and d == real_today, "items": items})
    return cells


# --------------------------------------------------------------------------
# PAGE 2 — Progress
# --------------------------------------------------------------------------


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
               "Each point is what this session's efficiency implies at your "
               "median pace, so sessions of different speeds are comparable. "
               "Down is progress.")
    training_hr_block(acts, today, data.get("notes"), data.get("weather"))

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
         "note": "needs 3 weeks" if not (sig and sig.acwr) else "acute vs chronic"},
    ]
    if cad.get("avg"):
        figures.append({
            "label": "Cadence", "value": f"{cad['avg']:.0f}",
            "note": f"stride {cad['avg_stride_cm']:.0f} cm"
                    if cad.get("avg_stride_cm") else "steps/min",
            "tone": {"low": "bad", "fair": "caution",
                     "good": "good"}.get(cad["verdict"], "neutral")})
    for sport in shown_sports():
        row = tot["by_sport"].get(sport) or {}
        if not row.get("sessions"):
            continue
        figures.append({
            "label": sport.title(), "value": f"{int(row['sessions'])}",
            "note": f"{row.get('km', 0):.0f} km · {hm(row.get('minutes') or 0)}"})
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
    with a, ui.card("Where your effort goes",
                    "Base phase wants most time easy."):
        intensity_block(zones, today, data.get("notes"), data)
    with b, ui.card("Efficiency against your own baseline",
                    "Percent change in speed or watts per heartbeat."):
        efficiency_block(acts, today, data.get("notes"), data.get("weather"))

    detail = st.tabs(["Volume", "Daily signals", "Cadence and stride",
                      "Running form", "Aerobic drift"])
    with detail[0]:
        st.caption("At most 10% growth a week, every fourth week easier.")
        volume_chart(data, today, data.get("notes"))
    with detail[1]:
        st.caption("Overnight measurements against your own baseline.")
        trend_chart(wl, today)
    with detail[2]:
        cadence_block(acts, today)
    with detail[3]:
        form_block(acts)
    with detail[4]:
        st.caption("Efficiency in the second half versus the first. Under 5% is "
                   "good durability.")
        drift_block(acts)


def form_block(acts: list[dict]) -> None:
    """Ground contact, vertical oscillation and vertical ratio.

    Garmin measures all three and this app ignored them until now. Vertical
    ratio is the useful one: it is bounce as a share of stride, so it says how
    much of each step went upwards instead of forwards.
    """
    runs = [a for a in acts if a.get("sport") == "run"
            and (a.get("ground_contact_ms") or a.get("vertical_ratio"))]
    if not runs:
        st.caption(
            "Ground contact and bounce are not in the activity list Garmin "
            "returns — they arrive with the next full sync of each run.")
        return

    def mean(field: str) -> float | None:
        vals = [float(a[field]) for a in runs if a.get(field)]
        return sum(vals) / len(vals) if vals else None

    gct, osc, ratio = (mean("ground_contact_ms"), mean("vertical_osc_cm"),
                       mean("vertical_ratio"))
    ui.rows([
        ("Ground contact", f"{gct:.0f} ms" if gct else "—",
         "under 250 ms is quick, 300+ is long"),
        ("Vertical bounce", f"{osc:.1f} cm" if osc else "—",
         "how far the body rises each step"),
        ("Bounce as % of stride", f"{ratio:.1f}%" if ratio else "—",
         "under 8% is efficient"),
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
        for sport in shown_sports():
            sp = zone_distribution(zones, sport=sport, since=since)
            if sum(sp.values()) <= 0:
                continue
            st.markdown(f"<span style='font-size:.82rem;opacity:.8'>{EMOJI[sport]} "
                        f"{sport.title()} · {hm(sum(sp.values()))}</span>",
                        unsafe_allow_html=True)
            ui.proportion_bar([(ZONE_LABELS[z], sp[z], ZONE_COLOR[z])
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
            hovertemplate="%{x|%a %d %b}<br>%{y:+.1f}% vs first session"
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
                    hovertemplate="%{x|%a %d %b}<br>%{y:.0f} spm<br>"
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


def drift_block(acts: list[dict]) -> None:
    drift = [a for a in acts if a.get("decoupling_pct") is not None]
    if not drift:
        st.caption("No session long enough yet — drift needs 60+ minutes with a "
                   "heart-rate stream.")
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
                      weather: dict | None = None) -> None:
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
    drawn, notes = False, []
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
        # A line through two points implies a trend that is not there.
        mode = "lines+markers" if len(days) >= 3 else "markers"
        fig.add_scatter(
            x=days, y=ys, mode=mode, name=sport,
            line=dict(color=SPORT_COLOR[sport], width=2),
            marker=dict(size=10, color=SPORT_COLOR[sport]),
            customdata=list(zip(mins, raws)),
            hovertemplate="%{x|%a %d %b}<br>%{y:.0f} bpm<br>"
                          "%{customdata[0]:.0f} min · raw %{customdata[1]:.0f}"
                          f"<extra>{sport}</extra>")
        t = hr_trend(acts, sport, as_of=today, steady_only=False)
        if t["normalised_change_bpm"] is not None:
            notes.append(f"{sport} {t['normalised_change_bpm']:+.1f} bpm")
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
    if notes:
        st.caption("Change at the same pace: " + " · ".join(notes)
                   + " (negative is progress).")
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
                    hovertemplate="%{x|%a %d %b}<br>%{y:.1f}<extra></extra>")
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
                        hovertemplate="week of %{x|%a %d %b}<br>%{y:.0f} min"
                                      "<extra>completed</extra>")
    bridge = [done[-1]] if done else []
    pts = [(week_start_of(today) + timedelta(weeks=f["week_offset"]), f["minutes"],
            f["deload"]) for f in fc]
    fig.add_scatter(x=[d for d, _ in bridge] + [d for d, _, _ in pts],
                    y=[m for _, m in bridge] + [m for _, m, _ in pts],
                    mode="lines+markers", name="ceiling the rules allow",
                    line=dict(color="#7FB6DC", width=2.5, dash="dash"),
                    marker=dict(size=8),
                    hovertemplate="week of %{x|%a %d %b}<br>%{y:.0f} min"
                                  "<extra>planned</extra>")
    dl = [(d, m) for d, m, is_dl in pts if is_dl]
    if dl:
        fig.add_scatter(x=[d for d, _ in dl], y=[m for _, m in dl], mode="markers",
                        name="deload week",
                        marker=dict(size=14, symbol="diamond", color=TONE["caution"]),
                        hovertemplate="week of %{x|%a %d %b}<br>%{y:.0f} min"
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
    unlocked = writes_allowed()
    if not unlocked:
        st.caption("🔒 Read-only. Enter your PIN in the sidebar to make changes.")
    with Store(db_path()) as s:
        facts = planner.build_facts(s, today=today)
        env = planner.build_envelope(facts, s)
        verdict = planner.readiness_verdict(facts)

    tone = {"deload": "bad", "hold": "neutral", "build": "good"}[verdict["verdict"]]
    ui.banner(verdict["headline"], " · ".join(verdict["reasons"]), tone)
    ui.stats_row([
        {"label": "Week ceiling", "value": hm(env.max_week_minutes),
         "note": "the most the rules allow"},
        {"label": "Done", "value": hm(facts.completed_this_week.total_minutes),
         "note": f"{len(facts.trained_days)} days trained"},
        {"label": "Left", "value": hm(planner.remaining_budget(facts, env)),
         "note": "still available"},
        {"label": "Hard sessions", "value": env.max_quality_sessions,
         "note": "allowed this week"},
    ])

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

    ui.section("How do you feel?", "This shapes the week inside what the rules allow. "
                                   "Deload triggers come from data, not mood.")
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
    ui.section("Your week", origin)
    ui.week_strip(week_cells(stored, today))
    for f in stored.get("flags", []):
        st.caption(("🤖 " if f.startswith("AI:") else "⚠ ") + f)
    if stored.get("adjustments_made"):
        with st.expander(f"What the rules changed ({len(stored['adjustments_made'])})"):
            for a in stored["adjustments_made"]:
                st.markdown(f"- {a}")

    ui.section("Change it", "Edit any row, add sessions, delete what you do not want. "
                            "Saving keeps exactly what you enter.")
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
        for n, r in edited.iterrows():
            try:
                days.append(PlanDay(
                    day=str(r["Day"]), sport=str(r["Sport"]),
                    duration_min=max(0, min(400, int(r["Minutes"] or 0))),
                    target_zone=str(r["Zone"] or "Z2"),
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


# --------------------------------------------------------------------------
# PAGE 4 — Log
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
        fixed, changed = with_store(lambda s: planner.reapply_rules(
            s, plan, today=date.fromisoformat(iso_today),
            only_sports=list(sports) or None))
        return fixed.model_dump(mode="json"), changed
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
    if not acts:
        st.caption("No activities yet — sync from the sidebar.")
        return
    tot = totals(acts)
    body = data.get("profile") or {}

    span = "no dated sessions"
    if tot["first_day"]:
        span = (f"{day_label(tot['first_day'].isoformat(), year=True)} — "
                f"{day_label(tot['last_day'].isoformat(), year=True)}")
    ui.page_title("Lifetime", span)
    insight_banner("Lifetime", data, today)

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
            row = tot["by_sport"].get(sp) or {}
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

    ui.section("Heart rate at your usual pace",
               "The headline trend: the same pace costing fewer beats is fitness. "
               "Every session on record.")
    with ui.frame():
        training_hr_block(acts, today, data.get("notes"),
                          data.get("weather"))
        chart_ai_note("lifetime_hr", data.get("notes"))

    ui.section("Resting heart rate, HRV and sleep",
               "All time, not a rolling window.")
    with ui.frame():
        trend_chart(data["wellness"], today)
        chart_ai_note("lifetime_recovery", data.get("notes"))


def page_about(data: dict, today: date) -> None:
    """Short, plain, and honest about what it does.

    An earlier version listed every rule and every limit, which read as
    documentation for the code rather than an answer to "what is this". This
    leads with the loop the athlete actually lives in and keeps the caveats to
    the ones that change what they would do.
    """
    # No brand block here: the sticky header already carries the name, and
    # printing it twice on the one page that talks about the app looks like a
    # mistake.
    st.markdown(
        "### AI plans your week. The plan goes to your watch.\n\n"
        "Your watch collects the data. This reads it, works out whether your "
        "fitness is rising while your heart rate falls, and builds the week "
        "that follows — swim, bike, run and leg strength, together."
    )

    ui.figures([
        {"label": "It plans", "value": "4 sports",
         "note": "as one week, not four apps"},
        {"label": "It sends", "value": "to your watch",
         "note": "runs, rides and leg sessions"},
        {"label": "It targets", "value": "your bpm",
         "note": "your ceiling, not Garmin's"},
        {"label": "It checks", "value": "in code",
         "note": "every limit, after the AI answers"},
    ])

    ui.section("The loop")
    ui.rows([
        ("1. It plans the week", "AI",
         "inside limits it is not allowed to cross"),
        ("2. You send it to the watch", "one tap",
         "named exercises, or a heart-rate range"),
        ("3. You train", "the watch guides",
         "it buzzes if you drift out of the range"),
        ("4. It reads what happened", "on Refresh",
         "and rebuilds the rest of the week"),
    ])

    ui.section("What goes to the watch")
    st.markdown(
        "- **Runs and rides** — timed, with your heart-rate range attached. The "
        "watch holds you to it, which is the whole point of setting a ceiling.\n"
        "- **Leg sessions** — every exercise, set, rep, hold and weight, named. "
        "Sets come back matched, and the weights move up on their own.\n"
        "- **Swims and bricks stay off the watch.** Garmin builds swim workouts "
        "from pool length and stroke rather than minutes, and wrist heart rate "
        "in water is not reliable enough to hold you to."
    )
    st.markdown("On the watch: **START → the sport → pick the session.**")

    ui.section("Why not just use Garmin?",
               "Garmin's numbers are good. This does the part it does not.")
    ui.rows([
        ("Your watch has no triathlon coach", "so", "nothing plans four sports"),
        ("Garmin's easy zone is fixed", "so",
         "yours is 137 bpm, and every target follows it"),
        ("Garmin logs strength", "but", "it does not decide the weights"),
        ("Garmin shows the weather", "but", "it does not say a humid day "
                                            "explains your heart rate"),
        ("Garmin waits 28 days for a trend", "this", "answers in about two"),
    ])

    ui.section("What the AI is not allowed to do",
               "It writes and it places sessions. It does not do the arithmetic.")
    ui.rows([
        ("Invent an exercise", "blocked", "it picks from a fixed 22"),
        ("Cancel an easy week", "blocked", "your recovery data decides"),
        ("Add more than 10% volume", "blocked", "that is how injuries start"),
        ("Choose a heart-rate number", "blocked", "those come from your zones"),
        ("Set how much weight", "blocked", "one rep, then one step"),
    ])
    st.caption(
        "A model asked how your training should go will agree with you — say you "
        "feel strong and it offers a bigger week, say you are tired and it "
        "cancels one. So every limit is re-checked in code after it answers, and "
        "the plan says ai_repaired when that changed something."
    )

    ui.section("Logging your leg sessions", "Two ways.")
    st.markdown(
        "**Send it from Today**, then on the watch **START → Strength → pick "
        "it**. Reps are counted for you and every set arrives named.\n\n"
        "**Or record it yourself:** START → Strength → lift → save. Turn on rep "
        "counting and rest detection in that activity's settings first. Sets "
        "arrive counted but sometimes unnamed, and the Log page lets you assign "
        "them.\n\n"
        "**Or by hand** on the Log page, if you forgot the watch entirely.\n\n"
        "Either way, record it somehow. A leg session the watch never saw counts "
        "as a rest day, and then tomorrow's readiness — and this plan — are "
        "built on a week that did not happen."
    )

    ui.section("Straight answers")
    st.markdown(
        "- **Your real easy limit** is whatever heart rate you can still talk in "
        "full sentences at. No watch knows that; the number here is a starting "
        "point you can change.\n"
        "- **Efficiency needs three steady sessions per sport** before it means "
        "anything. Until then it says how many more you need.\n"
        "- **Not medical advice.** Tendon pain that keeps coming back is a "
        "physio, not a training problem."
    )

    ui.section("Your data")
    ui.rows([
        ("Where it lives", "one database", "shared with nobody"),
        ("Garmin sign-in", "never from the web", "a saved session, copied by hand"),
        ("Changing anything", "needs your PIN", "stored scrambled, never as text"),
        ("Reading", "open if you share the link", "closeable with a password"),
    ])
    st.markdown(
        "Code: [github.com/AnkitGodle/aerobic-engine]"
        "(https://github.com/AnkitGodle/aerobic-engine). Your training data is "
        "not in it."
    )
    st.caption(
        "Built on the unofficial Garmin Connect API for personal use. Not "
        "connected to Garmin. Not medical advice."
    )


def page_log(data: dict, today: date) -> None:
    tabs = st.tabs(["Sessions", "Strength", "Data"])
    with tabs[0]:
        log_sessions(data, today)
    with tabs[1]:
        log_strength(data, today)
    with tabs[2]:
        log_data(data)


def log_sessions(data: dict, today: date) -> None:
    acts = data["activities"]
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
    if zrows:
        st.markdown("**Time in each heart-rate zone**")
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
            base = float(sdf["altitude_m"].min())
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
        st.caption(f"Aerobic drift across this session: {act['decoupling_pct']:.1f}% "
                   f"(under 5% is good durability).")


def log_strength(data: dict, today: date) -> None:
    unlocked = writes_allowed()
    log_rows = data["strength"]
    if flash := st.session_state.pop("strength_flash", None):
        st.success(flash)
    insight_banner("Strength", data, today)
    if strength.needs_physio_note(log_rows):
        st.warning(strength.PHYSIO_NOTE)
    unmapped = [s for s in data["sets"] if not s.get("exercise_id")]
    if unmapped:
        st.caption(f"{len(unmapped)} watch-recorded set(s) sit outside this exercise "
                   f"list and were not logged: " + ", ".join(sorted(
                       {s.get("garmin_name") or s.get("garmin_category") or "?"
                        for s in unmapped})))
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


def sidebar(data: dict) -> None:
    with st.sidebar:
        st.markdown(
            f'<div class="ic-sidebrand">{ui.logo(24)}'
            f'<div class="ic-sidebrand-name">Aerobic Engine</div></div>',
            unsafe_allow_html=True)
        st.subheader(data["name"] or "Athlete", anchor=False)
        st.caption(f"{data['counts']['activities']} activities · "
                   f"{data['counts']['daily_wellness']} days of wellness")
        st.divider()
        unlock_control()
        sync_control()
        st.divider()
        if st.button("Reload page data", width="stretch"):
            refresh()
            st.rerun()
        st.caption(f"AI: {os.getenv('AI_BACKEND', 'anthropic')} "
                   f"({'ready' if ai.available() else 'off'})")
        st.caption("Not medical advice. Persistent tendon pain is a physio visit.")


# The sports a filter can actually select: what the athlete does, not what the
# planner can schedule (which also has "rest" and the composite "brick").
FILTER_SPORTS = ("run", "bike", "swim", "strength")


PAGES = ("Today", "Progress", "Plan", "Lifetime", "Log", "About")


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
    """The popover's label. It has to say what is filtered, or the filter hides."""
    monday = week_start_of(today)
    span = f"{monday.strftime('%d %b')} – {(monday + timedelta(days=6)).strftime('%d %b')}"
    if monday == week_start_of(date.today()):
        span = "This week"
    picked = "All sports" if set(sports) >= set(FILTER_SPORTS) else ", ".join(
        sp.title() for sp in sports)
    return f"{span} · {picked}"


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
    with st.popover(f"⚙ {label}", width="stretch"):
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
        ui.brand("Aerobic Engine", data["subtitle"])

        # The filters live in a popover rather than laid out beside the tabs.
        # Column ratios only ever "fit" a window you happened to test: a week
        # selector plus four pills next to six tabs wraps as soon as the
        # viewport, the font size or the number of sports changes. A single
        # trigger button always fits, at any width, and its label carries the
        # current state so nothing is hidden.
        nav_col, filter_col = st.columns([5, 2], vertical_alignment="center")
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
        return dict(data, scoped_to=list(FILTER_SPORTS))
    keep = set(sports)
    # A brick counts as both its parts, so it survives a bike-only or run-only
    # filter rather than vanishing from the record.
    if keep & {"bike", "run"}:
        keep.add("brick")

    scoped = dict(data)
    for key in ("activities", "all_activities"):
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
        span = f"{mon.strftime('%d %b')} – {sun.strftime('%d %b')}"
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
    data["subtitle"] = (
        (f"{data['name']} · " if data["name"] else "")
        + "Garmin Forerunner 265 · base phase · synced "
        + (data["last_sync"] or "never")[:16].replace("T", " "))
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
     "Lifetime": page_lifetime, "Log": page_log,
     "About": page_about}[page](data, today)


main()
