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
import time
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
    baseline_trend,
    hr_points,
    hr_trend,
    ef_data_status,
    ef_points,
    ols_slope,
    polarisation,
    recovery_signals,
    totals,
    volume_forecast,
    week_summaries,
    zone_distribution,
)
from core.auth import PinGate, session_expired  # noqa: E402
from core.garmin_guard import GarminBlocked  # noqa: E402
from core.schemas import DAYS, ENDURANCE_SPORTS, SPORTS, Checkin, PlanDay, WeekPlan  # noqa: E402
from core.store import DEFAULT_DB, Store, week_start_of  # noqa: E402

load_dotenv()
log = logging.getLogger("iron_coach.ui")
st.set_page_config(page_title="Aerobic Engine", page_icon="🏊", layout="wide",
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
    return os.getenv("IRON_COACH_DB", DEFAULT_DB)


def db_stamp() -> float:
    p = Path(db_path())
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_data(show_spinner=False)
def load(stamp: float) -> dict:
    with Store(db_path()) as s:
        return {
            "activities": s.activities(),
            "all_activities": s.activities(include_parents=True),
            "wellness": s.wellness(), "race": s.race_predictions(),
            "zones": s.zones(), "strength": s.strength_log(),
            "sets": s.exercise_sets(), "checkins": s.checkins(limit=60),
            "counts": s.counts(), "targets": s.targets(),
            "last_sync": s.get_state("last_sync"),
            "name": s.get_state("athlete_name") or "",
            "thresholds": {k: s.get_state(f"threshold_{k}")
                           for k in ("threshold_hr", "running_ftp", "cycling_ftp")},
            "plan": s.latest_plan(week_start_of(date.today())),
        }


def refresh() -> None:
    load.clear()


class _StateStore:
    """Opens a short-lived connection per call: Streamlit reruns constantly, and a
    handle held across a rerun ends up used after it was closed."""

    def get_state(self, key: str, default: str | None = None) -> str | None:
        with Store(db_path()) as s:
            return s.get_state(key, default)

    def set_state(self, key: str, value: str) -> None:
        with Store(db_path()) as s:
            s.set_state(key, value)


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
    """The one thing worth reading on this page, in words."""
    ins = insights.for_page(page, data, today)
    if ins is None:
        return
    tone = {"success": "good", "warning": "caution", "error": "bad",
            "info": "neutral"}[ins.tone]
    body = ""
    if ai.available():
        try:
            import json as _json
            body = _narrate(page, _json.dumps(ins.as_facts(), sort_keys=True)) or ""
        except Exception:
            body = ""
    if not body:
        body = " ".join(b.replace("**", "") for b in ins.bullets[:2])
    ui.banner(ins.headline, body, tone)
    remaining = ins.bullets[2:]
    if remaining:
        # Narrow column so it reads as a footnote to the banner rather than a
        # full-width bar of its own.
        with st.columns([2, 1])[0], st.expander("More detail"):
            for b in ins.bullets:
                st.markdown(f"- {b}")


@st.cache_data(show_spinner=False, ttl=1800)
def _narrate(page: str, facts_json: str) -> str | None:
    import json as _json

    from core.insights import PageInsight, narrate
    f = _json.loads(facts_json)
    return narrate(page, PageInsight(f["headline"], f["points"]))


# --------------------------------------------------------------------------
# PAGE 1 — Today
# --------------------------------------------------------------------------


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
            else next_week_plan(today)
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
            if d.get("target_zone") not in (None, "n/a", ""):
                bits.append(d["target_zone"])
            names = [strength.EXERCISES[e].name for e in d.get("exercise_ids", [])
                     if e in strength.EXERCISES]
            ui.today_card(d["sport"], " · ".join(bits) or "—",
                          "; ".join(names) if names else d.get("why", ""))
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

    nxt = next_week_plan(today)
    if nxt:
        total = sum(d["duration_min"] for d in nxt.get("week_plan", []))
        ui.section(f"Next week · {day_label((week_start_of(today) + timedelta(weeks=1)).isoformat())} onwards",
                   f"{hm(total)} planned — provisional, and re-derived when the week "
                   f"arrives.")
        ui.week_strip(week_cells(nxt, week_start_of(today) + timedelta(weeks=1),
                                 mark_today=False))




@st.cache_data(show_spinner=False, ttl=900)
def _next_week_plan(stamp: float, iso_today: str) -> dict | None:
    """Computed on demand and cached: it is a preview, not a saved plan."""
    try:
        with Store(db_path()) as s:
            p = planner.plan_next_week(s, today=date.fromisoformat(iso_today),
                                       use_ai=False, save=False)
        return p.model_dump(mode="json")
    except Exception as exc:  # noqa: BLE001 - a preview must never break the page
        log.warning("Next-week preview failed: %s", exc)
        return None


def next_week_plan(today: date) -> dict | None:
    return _next_week_plan(db_stamp(), today.isoformat())


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
    acts, wl, zones = data["activities"], data["wellness"], data["zones"]
    tot = totals(acts)
    sig = recovery_signals(wl, acts, as_of=today) if wl else None

    insight_banner("Fitness", data, today)

    ui.stats_row([
        {"label": "Activities", "value": tot["sessions"],
         "note": f"{tot['km']:.0f} km · {hm(tot['minutes'])}"},
        *[{"label": s.title(),
           "value": f"{int(tot['by_sport'].get(s, {}).get('sessions', 0))}",
           "note": f"{tot['by_sport'].get(s, {}).get('km', 0):.0f} km · "
                   f"{hm(tot['by_sport'].get(s, {}).get('minutes', 0))}"}
          for s in ENDURANCE_SPORTS],
    ])

    # --- the engine, in four numbers ----------------------------------
    rhr = baseline_trend(wl, "resting_hr", as_of=today, lower_is_better=True)
    hrv = baseline_trend(wl, "hrv_last_night", as_of=today, lower_is_better=False)
    tones = {"improving": "good", "worsening": "bad", "steady": "neutral",
             "insufficient_data": "neutral"}

    def trend_note(t: dict, unit: str) -> str:
        if t["verdict"] == "insufficient_data":
            return f"needs {t.get('needed', 'more data')}"
        if t["per_week"] is None:
            return "holding steady"
        arrow = "improving" if t["verdict"] == "improving" else (
            "worsening" if t["verdict"] == "worsening" else "holding steady")
        return f"{t['per_week']:+.2f} {unit}/week — {arrow}"
    ui.stats_row([
        {"label": "Resting HR",
         "value": f"{rhr['recent']:.0f} bpm" if rhr["recent"] else "—",
         "note": trend_note(rhr, "bpm"), "tone": tones[rhr["verdict"]]},
        {"label": "HRV", "value": f"{hrv['recent']:.0f} ms" if hrv["recent"] else "—",
         "note": trend_note(hrv, "ms"), "tone": tones[hrv["verdict"]]},
        {"label": "VO2max",
         "value": f"{sig.vo2max_run:.1f}" if sig and sig.vo2max_run else "—",
         "note": "Garmin running estimate"},
        {"label": "Load ratio", "value": f"{sig.acwr:.2f}" if sig and sig.acwr else "—",
         "note": "needs ~3 weeks of history"},
    ])

    # --- two panels per row -------------------------------------------
    a, b = st.columns(2, gap="medium")
    with a, ui.card("Heart rate during training",
                    "Each point is the heart rate this session's efficiency implies "
                    "at your usual pace. Down is progress."):
        training_hr_block(acts, today)
    with b, ui.card("Where your effort goes",
                    "Base phase wants most time easy. Hard work costs recovery "
                    "without adding much base."):
        intensity_block(zones, today)

    c, d = st.columns(2, gap="medium")
    with c, ui.card("Efficiency against your own baseline",
                    "Percent change in speed or watts per heartbeat, so all three "
                    "sports share one axis."):
        efficiency_block(acts, today)
    with d, ui.card("Volume, and the ceiling ahead",
                    "At most 10% growth a week, every fourth week a deload."):
        volume_chart(data, today)

    e, f = st.columns(2, gap="medium")
    with e, ui.card("Daily signals", "Overnight measurements, against baseline."):
        trend_chart(wl, today)
    with f, ui.card("Aerobic drift on long sessions",
                    "Efficiency in the second half versus the first. Under 5% is "
                    "good durability."):
        drift_block(acts)


def intensity_block(zones: list[dict], today: date) -> None:
    if not zones:
        st.caption("No zone data yet — sync from Garmin.")
        return
    since = today - timedelta(days=28)
    pol = polarisation(zones, since=since)
    ui.proportion_bar([("easy", pol["easy"], TONE["good"]),
                       ("moderate", pol["moderate"], TONE["caution"]),
                       ("hard", pol["hard"], TONE["bad"])])
    verdict = ("On target — base phase wants 70%+ easy." if pol["easy"] >= 70
               else "Inverted: base phase wants roughly the reverse of this."
               if pol["hard"] >= 35 else "Drifting harder than base phase wants.")
    st.caption(f"Last 28 days. {verdict}")
    with st.expander("Zone breakdown by sport"):
        for sport in ENDURANCE_SPORTS:
            sp = zone_distribution(zones, sport=sport, since=since)
            if sum(sp.values()) <= 0:
                continue
            st.markdown(f"<span style='font-size:.82rem;opacity:.8'>{EMOJI[sport]} "
                        f"{sport.title()} · {hm(sum(sp.values()))}</span>",
                        unsafe_allow_html=True)
            ui.proportion_bar([(ZONE_LABELS[z], sp[z], ZONE_COLOR[z])
                               for z in range(1, 6)])


def efficiency_block(acts: list[dict], today: date) -> None:
    """One chart for all three sports.

    Efficiency is in different units per sport — watts per beat on the bike,
    speed per beat elsewhere — so the raw values cannot share an axis. Expressing
    each as a percentage change from its own earliest baseline can.
    """
    drawn = False
    fig = go.Figure()
    for sport in ENDURANCE_SPORTS:
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
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(140,158,176,.5)")
    fig.update_layout(yaxis_title="% vs first session")
    ui.chart(fig, 200)
    statuses = [ef_data_status(acts, s) for s in ENDURANCE_SPORTS]
    short = [s for s in statuses if s["total"] and s["needed_for_verdict"]]
    if short:
        st.caption(" · ".join(f"{s['sport']}: {s['steady']}/3 steady" for s in short))


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
    ui.chart(fig, 200)


def training_hr_block(acts: list[dict], today: date) -> None:
    """All sports on one axis, so this is one chart rather than three."""
    view = st.radio("View", ["At your usual pace", "Raw average"], horizontal=True,
                    index=0, label_visibility="collapsed", key="hr_view")
    normalised = view.startswith("At")
    field = "hr_at_reference" if normalised else "avg_hr"

    fig = go.Figure()
    drawn, notes = False, []
    for sport in ENDURANCE_SPORTS:
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
    if not drawn:
        st.caption("No sessions with heart rate yet.")
        return
    fig.update_layout(yaxis_title="bpm")
    ui.chart(fig, 200)
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
    metric = st.radio("Metric", ["Resting HR", "HRV", "Readiness", "Sleep"],
                      horizontal=True, index=0, label_visibility="collapsed")
    col, color = {
        "Resting HR": ("resting_hr", TONE["bad"]),
        "HRV": ("hrv_last_night", TONE["good"]),
        "Readiness": ("training_readiness", "#7FB6DC"),
        "Sleep": ("sleep_seconds", "#A98BD9"),
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
    ui.chart(fig, 200)


def ef_chart(acts: list[dict], steady_only: bool) -> None:
    any_drawn = False
    for sport in ENDURANCE_SPORTS:
        pts = ef_points(acts, sport)
        if len(pts) < 2:
            continue
        any_drawn = True
        df = pd.DataFrame([{"date": p.date, "EF": p.ef, "hr": p.avg_hr,
                            "min": round(p.duration_min),
                            "kind": "steady aerobic" if p.is_steady else "harder effort"}
                           for p in pts])
        fig = go.Figure()
        for kind, colr in (("steady aerobic", SPORT_COLOR[sport]),
                           ("harder effort", ui.TONE_COLOR["neutral"])):
            sub = df[df["kind"] == kind]
            if sub.empty:
                continue
            fig.add_scatter(x=sub["date"], y=sub["EF"], mode="markers", name=kind,
                            marker=dict(size=11, color=colr,
                                        line=dict(width=0)),
                            customdata=sub[["hr", "min"]],
                            hovertemplate="%{x|%a %d %b}<br>EF %{y:.3f}<br>"
                                          "HR %{customdata[0]:.0f} · "
                                          "%{customdata[1]} min<extra></extra>")
        tdf = df if not steady_only else df[df["kind"] == "steady aerobic"]
        if len(tdf) >= 3:
            x = (pd.to_datetime(tdf["date"]) - pd.to_datetime(tdf["date"]).min()).dt.days
            slope = ols_slope(x.tolist(), tdf["EF"].tolist())
            if slope is not None:
                fig.add_scatter(x=tdf["date"],
                                y=tdf["EF"].mean() + slope * (x - x.mean()),
                                mode="lines", name="trend",
                                line=dict(color=SPORT_COLOR[sport], width=2,
                                          dash="dash"))
        st.markdown(f"**{EMOJI[sport]} {sport.title()}**")
        ui.chart(fig, 230)
    if not any_drawn:
        st.caption("Not enough sessions with heart rate yet. Three steady aerobic "
                   "sessions in a sport gives a direction; six makes it reliable.")


def volume_chart(data: dict, today: date) -> None:
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
    ui.chart(fig, 200)

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

    ui.section("Your weekly targets",
               "How much of each sport you want. The scheduler builds around this; "
               "the safety rules still cap the total.")
    existing = data["targets"]
    with st.form("targets"):
        rows = []
        for sport in ENDURANCE_SPORTS:
            cur = existing.get(sport) or {}
            c = st.columns([1, 1, 1], vertical_alignment="center")
            c[0].markdown(f"{EMOJI[sport]} **{sport.title()}**")
            rows.append({
                "sport": sport,
                "sessions": c[1].number_input("sessions", 0, 7,
                                              int(cur.get("sessions") or 0),
                                              key=f"ts_{sport}"),
                "minutes": c[2].number_input("minutes", 0, 900,
                                             int(cur.get("minutes") or 0), step=15,
                                             key=f"tm_{sport}"),
            })
        b = st.columns(2)
        save = b[0].form_submit_button("Save targets", type="primary",
                                       width="stretch", disabled=not unlocked)
        clear = b[1].form_submit_button("Clear", width="stretch",
                                        disabled=not unlocked)
    if save and writes_allowed():
        with Store(db_path()) as s:
            s.set_targets(rows)
        refresh()
        st.rerun()
    if clear and writes_allowed():
        with Store(db_path()) as s:
            s.clear_targets()
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
                                       notes=notes), today=today, use_ai=use_ai)
        st.session_state["plan"] = p.model_dump(mode="json")
        st.session_state.pop("plan_editor", None)
        refresh()

    stored = st.session_state.get("plan") or (data["plan"] or {}).get("plan")
    if not stored:
        if st.button("Build one from the rules only", disabled=not unlocked) \
                and writes_allowed():
            with Store(db_path()) as s:
                p = planner.plan_week(s, today=today, use_ai=False)
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
                          "Note": (d.get("why") or "")[:80]} for d in editable]
                        or [{"Day": "Mon", "Sport": "bike", "Minutes": 60,
                             "Zone": "Z2", "Note": ""}])
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
                                      pushback=push, previous_plan=stored)
        st.session_state["plan"] = p.model_dump(mode="json")
        st.session_state.pop("plan_editor", None)
        refresh()
        st.rerun()


# --------------------------------------------------------------------------
# PAGE 4 — Log
# --------------------------------------------------------------------------


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
        fig = go.Figure()
        for col, nm, colr in (("hr", "Heart rate", TONE["bad"]),
                              ("speed_mps", "Speed (m/s)", "#7FB6DC"),
                              ("power_w", "Power (W)", TONE["caution"])):
            if col in sdf and sdf[col].notna().any():
                fig.add_scatter(x=sdf["minutes"], y=sdf[col], mode="lines", name=nm,
                                line=dict(width=1.6, color=colr),
                                yaxis="y" if col == "hr" else "y2")
        fig.update_layout(xaxis_title="minutes into the session",
                          yaxis=dict(title="heart rate"),
                          yaxis2=dict(overlaying="y", side="right", showgrid=False))
        ui.chart(fig, 260)
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

    with st.expander("The full exercise list"):
        table(pd.DataFrame([
            {"Exercise": e.name, "Type": e.kind, "Targets": e.target, "Sets": e.sets,
             "Reps": f"{e.rep_range[0]}–{e.rep_range[1]}" if e.rep_range else "",
             "Hold": f"{e.hold_range[0]}–{e.hold_range[1]}s" if e.hold_range else "",
             "Step (kg)": e.load_step_kg, "Cue": e.cue}
            for e in strength.EXERCISES.values()]))
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


def sidebar(data: dict, today: date) -> date:
    with st.sidebar:
        st.subheader(data["name"] or "Athlete", anchor=False)
        st.caption(f"{data['counts']['activities']} activities · "
                   f"{data['counts']['daily_wellness']} days of wellness")
        st.divider()
        unlock_control()
        sync_control()
        st.divider()
        today = week_picker(today)
        if st.button("Reload page data", width="stretch"):
            refresh()
            st.rerun()
        st.caption(f"AI: {os.getenv('AI_BACKEND', 'anthropic')} "
                   f"({'ready' if ai.available() else 'off'})")
        st.caption("Not medical advice. Persistent tendon pain is a physio visit.")
    return today


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

    current = week_start_of(today)
    keys = list(labels)
    index = next((i for i, k in enumerate(keys) if labels[k] == current), 1)
    picked = labels[st.selectbox("Week (Mon–Sun)", keys, index=index)]

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
    ok, why = sync_mod.can_sync(db_path())
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
    if not Path(db_path()).exists():
        ui.page_title("Aerobic Engine")
        st.error("No database yet. Run `python scripts/fetch.py` first.")
        return

    data = load(db_stamp())
    today = sidebar(data, today)
    ui.page_title(
        "Aerobic Engine",
        (f"{data['name']} · " if data["name"] else "")
        + "Garmin Forerunner 265 · base phase · synced "
        + (data["last_sync"] or "never")[:16].replace("T", " "))

    today_tab, progress_tab, plan_tab, log_tab = st.tabs(
        ["Today", "Progress", "Plan", "Log"])
    with today_tab:
        page_today(data, today)
    with progress_tab:
        page_progress(data, today)
    with plan_tab:
        page_plan(data, today)
    with log_tab:
        page_log(data, today)


main()
