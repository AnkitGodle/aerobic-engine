"""The planner: facts, then a rules envelope, then the AI inside that envelope.

Layer 1 (`build_facts`)     — what actually happened, from analysis.py.
Layer 2 (`build_envelope`)  — the non-negotiables: progression cap, deload
                              triggers, session counts, required long sessions.
Layer 3 (`ai.plan_week`)    — adjusts volume, intensity and placement, and
                              explains itself.

`enforce()` sits between layers 3 and the user and is the reason this is safe to
use: it re-checks every constraint in code. A raw LLM is sycophantic — tell it
you feel great and it hands you a reckless week; tell it you are tired and it
cancels everything. The rules hold the line; the AI negotiates around it.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import date, timedelta
from typing import Any

from core import ai, strength
from core.analysis import (
    all_ef_trends,
    recovery_signals,
    week_summaries,
    week_summary,
)
from core.schemas import (
    DAYS,
    Checkin,
    Envelope,
    PlanDay,
    PlannerFacts,
    PlanPayload,
    SportEnvelope,
    WeekPlan,
)
from core.store import Store, week_start_of

log = logging.getLogger("iron_coach.planner")

# --- base-phase shape -----------------------------------------------------
# Bike-heavy: it buys the most aerobic volume per unit of tissue damage and is
# itself a race discipline. Running stays frequent but conservative to protect
# tendons. Swimming is mostly technique.
SPORT_SHARE = {"bike": 0.50, "run": 0.29, "swim": 0.21}

SESSION_BOUNDS: dict[tuple[str, str], tuple[int, int]] = {
    ("swim", "technique"): (30, 60),
    ("bike", "easy"): (45, 75),
    ("bike", "endurance"): (60, 105),
    ("bike", "long"): (100, 240),
    ("run", "easy"): (35, 60),
    ("run", "long"): (55, 105),
}

# Hard per-session ceilings. Nothing — rules or AI — gets past these.
SESSION_CEILING = {
    "swim": 120,
    "bike": 300,
    "run": 180,
    "strength": 45,
    "brick": 330,
    "rest": 0,
}

# Below these a session stops being worth doing — the planner drops it instead
# of shrinking everything into a week of pointless 20-minute stubs.
SESSION_FLOOR = {"swim": 25, "bike": 40, "run": 25, "strength": 15, "brick": 70}

# What survives when time runs out, worst first. The long ride is the week's
# aerobic anchor and the last thing to go; swims are the first.
DROP_PRIORITY = {"swim": 0, "run": 1, "bike": 2, "strength": 3, "brick": 5}

ZONE_FOR_ROLE = {
    "technique": "technique",
    "easy": "Z2",
    "endurance": "Z2",
    "long": "Z2",
    "strength": "n/a",
    "rest": "n/a",
}

PURPOSE_FOR_ROLE = {
    "technique": "swim technique and feel for the water",
    "easy": "frequency without tissue cost",
    "endurance": "aerobic base",
    "long": "aerobic durability",
    "strength": "tendon and muscle resilience to protect run volume",
    "rest": "full rest day",
}

# (day, sport, role, optional)
DAY_TEMPLATE: tuple[tuple[str, str, str, bool], ...] = (
    ("Mon", "swim", "technique", False),
    ("Mon", "strength", "strength", False),
    ("Tue", "bike", "endurance", False),
    ("Tue", "swim", "technique", True),
    ("Wed", "run", "easy", False),
    ("Wed", "strength", "strength", False),
    ("Thu", "swim", "technique", False),
    ("Thu", "bike", "easy", True),
    ("Fri", "rest", "rest", False),
    ("Sat", "run", "long", False),
    ("Sun", "bike", "long", False),
)

QUALITY_ZONES = {"Z3", "Z4", "Z5", "mixed"}

# --- deload triggers (rules force these regardless of mood) ---------------
HRV_DROP_PCT = -5.0      # 7-day HRV this far below the 28-day baseline
RHR_RISE_BPM = 5.0       # resting HR this far above baseline
READINESS_FLOOR = 35.0   # Garmin Training Readiness
ACWR_CEILING = 1.3       # acute:chronic workload ratio
BAD_HRV_STATUS = {"unbalanced", "low", "poor"}
BAD_STATUS_WORDS = ("overreaching", "unproductive", "detraining", "strained")

DELOAD_FACTOR = 0.65     # ~35% volume cut
PROGRESSION_CAP_PCT = 10.0
STARTING_WEEK_MIN = 360.0  # sane first week when there is no history
ABSOLUTE_WEEK_CEILING = 900.0
BLOCK_WEEKS = 4


# --------------------------------------------------------------------------
# Layer 1 — facts
# --------------------------------------------------------------------------


def build_facts(
    store: Store, today: date | None = None, history_weeks: int = 12
) -> PlannerFacts:
    """Ground truth for the week: what is done, how recovery looks, EF direction."""
    today = today or date.today()
    ws = week_start_of(today)
    lookback = ws - timedelta(weeks=history_weeks)

    activities = store.activities(since=lookback - timedelta(days=60))
    strength_rows = store.strength_log(since=lookback)
    wellness = store.wellness(since=today - timedelta(days=60))

    weeks = week_summaries(
        activities, weeks=history_weeks, as_of=today, strength_rows=strength_rows
    )
    this_week = week_summary(activities, ws, strength_rows)

    checkins = []
    for row in store.checkins(since=today - timedelta(days=14), limit=14):
        try:
            checkins.append(
                Checkin(
                    date=date.fromisoformat(row["day"]),
                    sleep=row.get("sleep") or 3,
                    soreness=row.get("soreness") or 3,
                    motivation=row.get("motivation") or 3,
                    time_available_min=row.get("time_available_min") or 90,
                    notes=row.get("notes") or "",
                )
            )
        except (ValueError, TypeError):
            continue

    trained = sorted(
        {
            DAYS[date.fromisoformat(a["start_date"]).weekday()]
            for a in activities
            if ws <= date.fromisoformat(a["start_date"]) <= today
            and (a.get("duration_s") or 0) > 0
        },
        key=DAYS.index,
    )

    return PlannerFacts(
        week_start=ws,
        today=today,
        completed_this_week=this_week,
        previous_weeks=[w for w in weeks if w.week_start < ws],
        recovery=recovery_signals(wellness, activities, as_of=today),
        ef_trends=all_ef_trends(activities, as_of=today),
        recent_checkins=checkins,
        days_remaining=list(DAYS[today.weekday() :]),
        trained_days=trained,
    )


# --------------------------------------------------------------------------
# Layer 2 — the envelope
# --------------------------------------------------------------------------


def deload_triggers(facts: PlannerFacts) -> list[str]:
    """Recovery-driven reasons the week must be capped. Mood is not one of them."""
    r = facts.recovery
    out: list[str] = []
    if r.hrv_delta_pct is not None and r.hrv_delta_pct <= HRV_DROP_PCT:
        out.append(f"HRV {abs(r.hrv_delta_pct):.0f}% below 28-day baseline")
    if r.hrv_status and r.hrv_status.lower() in BAD_HRV_STATUS:
        out.append(f"Garmin HRV status: {r.hrv_status}")
    if r.rhr_delta is not None and r.rhr_delta >= RHR_RISE_BPM:
        out.append(f"resting HR +{r.rhr_delta:.0f} bpm over baseline")
    if r.training_readiness is not None and r.training_readiness < READINESS_FLOOR:
        out.append(f"Training Readiness {r.training_readiness:.0f}")
    if r.acwr is not None and r.acwr > ACWR_CEILING:
        out.append(f"acute:chronic load ratio {r.acwr:.2f}")
    if r.training_status and any(
        w in r.training_status.lower() for w in BAD_STATUS_WORDS
    ):
        out.append(f"Garmin training status: {r.training_status}")
    return out


# Recovery signals good enough to justify taking the full progression, and to
# offer a quality session. The envelope only ever capped the week before this —
# but an athlete who is absorbing the load should be told so, not left guessing.
HRV_STRONG_PCT = 3.0       # 7-day HRV this far above the 28-day baseline
RHR_STRONG_BPM = -1.0      # resting HR at least this far below baseline
READINESS_STRONG = 65.0
ACWR_HEADROOM = 1.05       # acute load not yet ahead of chronic


def build_signals(facts: PlannerFacts) -> list[str]:
    """Reasons the athlete looks ready for more, not less."""
    r = facts.recovery
    out: list[str] = []
    if r.hrv_delta_pct is not None and r.hrv_delta_pct >= HRV_STRONG_PCT:
        out.append(f"HRV {r.hrv_delta_pct:.0f}% above baseline")
    if r.rhr_delta is not None and r.rhr_delta <= RHR_STRONG_BPM:
        out.append(f"resting HR {abs(r.rhr_delta):.0f} bpm below baseline")
    if r.training_readiness is not None and r.training_readiness >= READINESS_STRONG:
        out.append(f"Training Readiness {r.training_readiness:.0f}")
    if r.acwr is not None and r.acwr <= ACWR_HEADROOM:
        out.append(f"load ratio {r.acwr:.2f} — room to add")
    if r.training_status and "productive" in r.training_status.lower():
        out.append(f"Garmin training status: {r.training_status}")
    return out


def readiness_verdict(facts: PlannerFacts) -> dict[str, Any]:
    """Should this week hold, back off, or build?

    Deload always wins: a single red flag outranks any number of green ones,
    because the cost of being wrong is asymmetric.
    """
    down = deload_triggers(facts)
    up = build_signals(facts)
    if down:
        return {
            "verdict": "deload", "reasons": down, "positives": up,
            "headline": "Back off — recovery data says so.",
        }
    if len(up) >= 2:
        return {
            "verdict": "build", "reasons": up, "positives": up,
            "headline": "You are absorbing the training — take the full step up.",
        }
    return {
        "verdict": "hold", "reasons": up,
        "positives": up,
        "headline": "Steady as planned — nothing arguing for more or less.",
    }


def week_index(facts: PlannerFacts, store: Store | None = None) -> int:
    """Position in the 4-week block, anchored on the first week of training."""
    anchor: date | None = None
    if store is not None:
        raw = store.get_state("block_anchor")
        if raw:
            try:
                anchor = date.fromisoformat(raw)
            except ValueError:
                anchor = None
        if anchor is None:
            # Anchor on the first training week ever recorded and persist it —
            # deriving it from a rolling history window would make the deload
            # week drift every time the lookback changed.
            first = store.earliest_activity_date()
            anchor = week_start_of(first) if first else facts.week_start
            store.set_state("block_anchor", anchor.isoformat())
    anchor = anchor or facts.week_start
    return max(0, (facts.week_start - anchor).days // 7) % BLOCK_WEEKS


def build_envelope(
    facts: PlannerFacts,
    store: Store | None = None,
    targets: dict[str, dict[str, Any]] | None = None,
) -> Envelope:
    """The bounds the AI plans inside. Deterministic, and it always wins.

    `targets` is the athlete's own weekly intent (sessions and minutes per sport).
    It shapes the week — how many swims, how the minutes divide — but it cannot
    lift the ceiling: the progression cap and the deload rules still apply on top,
    because those are the parts that stop an ambitious target from becoming an
    injury.
    """
    if targets is None and store is not None:
        targets = store.targets()
    targets = targets or {}
    idx = week_index(facts, store)
    triggers = deload_triggers(facts)
    scheduled_deload = idx == BLOCK_WEEKS - 1
    if scheduled_deload:
        triggers.insert(0, f"scheduled deload (week {idx + 1} of {BLOCK_WEEKS})")
    deload = bool(triggers)

    recent = [w for w in facts.previous_weeks[-3:] if w.total_minutes > 0]
    prev = facts.previous_weeks[-1].total_minutes if facts.previous_weeks else 0.0
    typical = max((w.total_minutes for w in recent), default=0.0)

    verdict = readiness_verdict(facts)["verdict"]
    if deload:
        budget = (prev or typical or STARTING_WEEK_MIN) * DELOAD_FACTOR
    elif prev < 120:
        # No meaningful history (new user, illness, holiday) — start sensibly
        # rather than applying a percentage to near-zero.
        budget = min(STARTING_WEEK_MIN, max(typical * (1 + PROGRESSION_CAP_PCT / 100), 240))
    else:
        # "hold" takes a conservative half-step; "build" takes the full cap.
        step = PROGRESSION_CAP_PCT if verdict == "build" else PROGRESSION_CAP_PCT / 2
        budget = prev * (1 + step / 100)
    budget = round(min(budget, ABSOLUTE_WEEK_CEILING))

    # A target's minutes decide how the week divides between sports; without
    # targets, fall back to the bike-heavy base-phase default.
    target_minutes = {
        sport: float(t.get("minutes") or 0) for sport, t in targets.items()
    }
    total_target = sum(v for k, v in target_minutes.items() if k in SPORT_SHARE)
    if total_target > 0:
        shares = {
            sport: (target_minutes.get(sport, 0.0) / total_target)
            for sport in SPORT_SHARE
        }
    else:
        shares = dict(SPORT_SHARE)

    # An explicit total target may lower the week but never raise it past the cap.
    if total_target > 0:
        budget = round(min(budget, total_target * (DELOAD_FACTOR if deload else 1.0)))

    scale = 0.7 if deload else 1.0
    by_sport: dict[str, SportEnvelope] = {}
    for sport, share in shares.items():
        long_floor = {"bike": 100.0, "run": 55.0, "swim": 0.0}[sport]
        tgt = targets.get(sport) or {}
        want = int(tgt.get("sessions") or 0)
        default_min = 2 if not deload else 1
        min_sessions = min(want, default_min) if want else default_min
        max_sessions = max(want, 1) if want else 3
        by_sport[sport] = SportEnvelope(
            sport=sport,
            min_sessions=min_sessions,
            max_sessions=min(max_sessions, 7),
            max_minutes=round(budget * share * 1.15),
            long_session_min=0.0 if (deload and sport != "bike") else round(long_floor * scale),
            notes=(
                "bike is the aerobic volume tool and a race discipline"
                if sport == "bike"
                else "keep frequent but conservative — tendons"
                if sport == "run"
                else "mostly technique"
            ),
        )

    return Envelope(
        week_start=facts.week_start,
        phase="base",
        week_index=idx,
        deload=deload,
        deload_reasons=triggers,
        max_week_minutes=budget,
        prev_week_minutes=prev,
        progression_cap_pct=PROGRESSION_CAP_PCT,
        min_rest_days=2 if deload else 1,
        strength_sessions=1 if deload else 2,
        brick_required=(not deload) and idx % 2 == 1,
        max_quality_sessions=0 if deload else (2 if verdict == "build" else 1),
        readiness_verdict=verdict,
        build_signals=readiness_verdict(facts)["positives"],
        by_sport=by_sport,
    )


def remaining_budget(facts: PlannerFacts, envelope: Envelope) -> float:
    """Minutes still available this week after what has already been done."""
    done = facts.completed_this_week.total_minutes
    return max(0.0, envelope.max_week_minutes - done)


def completed_entries(facts: PlannerFacts, store: Store) -> list[PlanDay]:
    """Past days of this week, rendered as read-only plan rows for the UI."""
    out: list[PlanDay] = []
    acts = store.activities(since=facts.week_start)
    for a in acts:
        d = date.fromisoformat(a["start_date"])
        if d > facts.today or d < facts.week_start:
            continue
        sport = a["sport"] if a["sport"] in SESSION_CEILING else "rest"
        out.append(
            PlanDay(
                day=DAYS[d.weekday()],
                sport=sport,
                duration_min=int((a.get("duration_s") or 0) / 60),
                target_zone="n/a",
                purpose="completed",
                why=f"logged: {a.get('name') or sport}",
            )
        )
    return out


# --------------------------------------------------------------------------
# Layer 2b — the rules-only plan (also the fallback when the AI is unusable)
# --------------------------------------------------------------------------


def rules_plan(
    facts: PlannerFacts,
    envelope: Envelope,
    strength_log: Sequence[dict[str, Any]] = (),
    checkin: Checkin | None = None,
) -> WeekPlan:
    """A defensible week with no AI involved at all."""
    remaining = set(facts.days_remaining)
    budget = remaining_budget(facts, envelope)
    done = facts.completed_this_week.by_sport
    deload = envelope.deload

    slots: list[tuple[str, str, str]] = []
    for day, sport, role, optional in DAY_TEMPLATE:
        if day not in remaining:
            continue
        if day in facts.trained_days:
            continue  # already trained today — don't stack a second session on it
        if sport == "rest":
            slots.append((day, sport, role))
            continue
        if sport == "strength":
            slots.append((day, sport, role))
            continue
        done_n = done[sport].sessions if sport in done else 0
        planned_n = sum(1 for s in slots if s[1] == sport)
        cap = envelope.by_sport[sport].max_sessions
        if done_n + planned_n >= cap:
            continue
        if optional and (deload or budget < 300):
            continue
        slots.append((day, sport, role))

    # Trim strength to the allowed count, keeping the earliest slots.
    strength_done = done["strength"].sessions if "strength" in done else 0
    allowed_strength = max(0, envelope.strength_sessions - strength_done)
    kept, seen_strength = [], 0
    for slot in slots:
        if slot[1] == "strength":
            if seen_strength >= allowed_strength:
                continue
            seen_strength += 1
        kept.append(slot)
    slots = kept

    # Provisional durations, then scale the endurance work to the budget.
    scale = 0.7 if deload else 1.0
    provisional: list[tuple[str, str, str, float]] = []
    for day, sport, role in slots:
        if sport in ("rest", "strength"):
            provisional.append((day, sport, role, 0.0))
            continue
        lo, hi = SESSION_BOUNDS[(sport, role)]
        target = (lo + hi) / 2 * scale
        if role == "long":
            target = max(target, envelope.by_sport[sport].long_session_min)
        provisional.append((day, sport, role, target))

    strength_minutes = 0
    prescription = strength.build_session(
        strength_log, session_index=strength_done, intensity=0.6 if deload else 1.0
    )
    if any(s[1] == "strength" for s in slots):
        strength_minutes = strength.session_minutes(prescription)

    endurance_total = sum(p[3] for p in provisional)
    endurance_budget = max(0.0, budget - strength_minutes * seen_strength)
    factor = 1.0
    if endurance_total > 0 and endurance_budget > 0:
        factor = min(1.15, max(0.55, endurance_budget / endurance_total))
    elif endurance_total > 0:
        factor = 0.55

    plan: list[PlanDay] = []
    session_idx = strength_done
    for day, sport, role, target in provisional:
        if sport == "rest":
            plan.append(
                PlanDay(
                    day=day,
                    sport="rest",
                    duration_min=0,
                    target_zone="n/a",
                    purpose=PURPOSE_FOR_ROLE["rest"],
                    why="full rest day — adaptation happens here",
                )
            )
            continue
        if sport == "strength":
            presc = strength.build_session(
                strength_log, session_index=session_idx, intensity=0.6 if deload else 1.0
            )
            session_idx += 1
            plan.append(
                PlanDay(
                    day=day,
                    sport="strength",
                    duration_min=strength.session_minutes(presc),
                    target_zone="n/a",
                    purpose=PURPOSE_FOR_ROLE["strength"],
                    exercise_ids=[p.exercise_id for p in presc],
                    why=(
                        "lightened — recovery signals are down"
                        if deload
                        else "slow, heavy, no jumping; placed away from the long run"
                    ),
                )
            )
            continue

        lo, hi = SESSION_BOUNDS[(sport, role)]
        minutes = int(round(min(hi, max(lo * scale, target * factor))))
        plan.append(
            PlanDay(
                day=day,
                sport=sport,
                duration_min=minutes,
                target_zone=ZONE_FOR_ROLE[role],
                purpose=PURPOSE_FOR_ROLE[role],
                why=_rules_why(sport, role, envelope),
            )
        )

    if envelope.brick_required:
        plan = _make_brick(plan)

    flags = list(envelope.deload_reasons)
    if envelope.deload:
        flags.insert(0, "DELOAD week — volume capped, quality removed")
    if checkin and checkin.time_available_min < 45:
        flags.append("Check-in reports very little time today; rules plan is unadjusted")
    if strength.needs_physio_note(strength_log):
        flags.append(strength.PHYSIO_NOTE)

    return WeekPlan(
        week_plan=plan,
        flags=flags,
        adjustments_made=[
            f"week budget {envelope.max_week_minutes:.0f} min "
            f"(previous week {envelope.prev_week_minutes:.0f} min)"
        ],
        source="rules",
    )


def _rules_why(sport: str, role: str, envelope: Envelope) -> str:
    if role == "long" and sport == "bike":
        return "the week's aerobic anchor — long, low-impact, race-specific"
    if role == "long" and sport == "run":
        return "long run kept conservative on purpose; tendons adapt slower than fitness"
    if sport == "swim":
        return "technique volume — cheap aerobic work, no impact"
    if sport == "bike":
        return "midweek aerobic volume at conversational effort"
    if envelope.deload:
        return "easy by design this week"
    return "steady aerobic frequency"


def _make_brick(plan: list[PlanDay]) -> list[PlanDay]:
    """Turn the long ride into a bike-to-run brick and trim the long run."""
    long_bike = max(
        (d for d in plan if d.sport == "bike"), key=lambda d: d.duration_min, default=None
    )
    if long_bike is None or long_bike.duration_min < 60:
        return plan
    run_off = 20 if long_bike.duration_min < 120 else 25
    long_run = max(
        (d for d in plan if d.sport == "run"), key=lambda d: d.duration_min, default=None
    )
    if long_run is not None and long_run.duration_min > 45:
        long_run.duration_min = int(long_run.duration_min * 0.85)
        long_run.why += "; trimmed because this week carries a brick"
    long_bike.sport = "brick"
    long_bike.purpose = "race-specific: running on bike legs"
    long_bike.why = (
        f"{long_bike.duration_min} min ride straight into a {run_off} min easy run"
    )
    long_bike.duration_min += run_off
    return plan


# --------------------------------------------------------------------------
# Enforcement — the reason the AI is safe to use
# --------------------------------------------------------------------------


def enforce(
    plan: WeekPlan,
    facts: PlannerFacts,
    envelope: Envelope,
    strength_log: Sequence[dict[str, Any]] = (),
) -> WeekPlan:
    """Re-check every constraint in code. Returns a plan that cannot break them."""
    notes: list[str] = []
    remaining = list(facts.days_remaining)
    days: list[PlanDay] = []

    # 1. Only the remaining days, only known sports/days.
    for d in plan.week_plan:
        if d.purpose == "completed":
            continue
        if d.day not in remaining:
            notes.append(f"dropped {d.sport} on {d.day}: not a remaining day")
            continue
        days.append(d.model_copy(deep=True))

    # 2. Per-session ceilings.
    for d in days:
        ceiling = SESSION_CEILING.get(d.sport, 180)
        if d.duration_min > ceiling:
            notes.append(f"{d.day} {d.sport} cut {d.duration_min}->{ceiling} min (ceiling)")
            d.duration_min = ceiling

    # 3. Deload overrides mood and model alike.
    if envelope.deload:
        for d in days:
            if d.target_zone in QUALITY_ZONES:
                notes.append(f"{d.day} {d.sport}: {d.target_zone} -> Z2 (deload week)")
                d.why = _annotate(d.why, "held at Z2 — deload week")
                d.target_zone = "Z2"
            if d.sport == "brick":
                notes.append(f"{d.day}: brick removed (deload week)")
                d.sport = "bike"
                d.duration_min = int(d.duration_min * 0.8)
    else:
        quality = [d for d in days if d.target_zone in QUALITY_ZONES]
        for extra in sorted(quality, key=lambda d: -d.duration_min)[
            envelope.max_quality_sessions :
        ]:
            notes.append(
                f"{extra.day} {extra.sport}: {extra.target_zone} -> Z2 "
                f"(max {envelope.max_quality_sessions} quality session in base)"
            )
            extra.why = _annotate(extra.why, "held at Z2 — base phase quality limit")
            extra.target_zone = "Z2"

    # 4. Session counts per sport, completed sessions included.
    done = facts.completed_this_week.by_sport
    for sport, se in envelope.by_sport.items():
        done_n = done[sport].sessions if sport in done else 0
        planned = [d for d in days if d.sport == sport and d.duration_min > 0]
        overflow = (done_n + len(planned)) - se.max_sessions
        for extra in sorted(planned, key=lambda d: d.duration_min)[:max(0, overflow)]:
            notes.append(
                f"dropped {extra.duration_min} min {sport} on {extra.day}: "
                f"over {se.max_sessions} sessions/week"
            )
            days.remove(extra)

    # 5. Strength: library-only exercises, deterministic prescription, placement.
    strength_done = done["strength"].sessions if "strength" in done else 0
    allowed = max(0, envelope.strength_sessions - strength_done)
    s_days = [d for d in days if d.sport == "strength"]
    for extra in s_days[allowed:]:
        notes.append(
            f"dropped strength on {extra.day}: {strength_done} of "
            f"{envelope.strength_sessions} allowed sessions already logged this week"
        )
        days.remove(extra)
    for i, d in enumerate(d for d in days if d.sport == "strength"):
        presc = strength.build_session(
            strength_log,
            session_index=strength_done + i,
            intensity=0.6 if envelope.deload else 1.0,
        )
        valid = strength.validate_exercise_ids(d.exercise_ids)
        invented = [x for x in d.exercise_ids if x.strip().lower() not in strength.LIBRARY_IDS]
        if invented:
            notes.append(f"{d.day}: removed exercises outside the library: {invented}")
        if not valid:
            valid = [p.exercise_id for p in presc]
        d.exercise_ids = valid
        d.duration_min = min(
            SESSION_CEILING["strength"],
            max(15, d.duration_min or strength.session_minutes(presc)),
        )
        d.target_zone = "n/a"

    days = _fix_strength_placement(days, notes)

    # 6. Total volume must fit the remaining budget.
    days = _fit_budget(days, remaining_budget(facts, envelope), envelope, notes)

    # 7. Required long sessions must not silently disappear two weeks running.
    days = _ensure_long_sessions(days, facts, envelope, notes)

    # 8. Minimum rest days across the whole week.
    days = _ensure_rest_days(days, facts, envelope, notes)

    # 9. Descriptions must match the numbers that survived steps 1-8.
    _relabel(days, envelope)

    flags = list(plan.flags)
    for reason in envelope.deload_reasons:
        if reason not in flags:
            flags.append(reason)
    if envelope.deload and not any("deload" in f.lower() for f in flags):
        flags.insert(0, "DELOAD week — volume capped, quality removed")
    if strength.needs_physio_note(strength_log) and strength.PHYSIO_NOTE not in flags:
        flags.append(strength.PHYSIO_NOTE)

    order = {d: i for i, d in enumerate(DAYS)}
    days.sort(key=lambda d: (order.get(d.day, 9), -d.duration_min))
    return WeekPlan(
        week_plan=days,
        flags=flags,
        adjustments_made=plan.adjustments_made + notes,
        source=plan.source if not notes else ("ai_repaired" if plan.source == "ai" else plan.source),
    )


def _fix_strength_placement(days: list[PlanDay], notes: list[str]) -> list[PlanDay]:
    """Legs never land on, or the day before, a long run or a quality bike."""
    order = {d: i for i, d in enumerate(DAYS)}
    hard = {
        d.day
        for d in days
        if (d.sport in ("run", "brick") and d.duration_min >= 70)
        or (d.sport == "bike" and (d.duration_min >= 120 or d.target_zone in QUALITY_ZONES))
    }
    day_before_long_run = {
        DAYS[order[d.day] - 1]
        for d in days
        if d.sport in ("run", "brick") and d.duration_min >= 70 and order.get(d.day, 0) > 0
    }
    blocked = hard | day_before_long_run
    present = {d.day for d in days}

    for d in list(days):
        if d.sport != "strength" or d.day not in blocked:
            continue
        candidates = [
            c
            for c in present
            if c not in blocked
            and not any(x.sport == "strength" and x.day == c for x in days)
        ]
        if candidates:
            new_day = min(candidates, key=lambda c: order[c])
            notes.append(
                f"moved leg strength {d.day} -> {new_day}: too close to a long run "
                f"or quality bike"
            )
            d.day = new_day
        else:
            notes.append(f"dropped leg strength on {d.day}: nowhere safe to put it")
            days.remove(d)
    return days


def _ensure_long_sessions(
    days: list[PlanDay], facts: PlannerFacts, envelope: Envelope, notes: list[str]
) -> list[PlanDay]:
    order = {d: i for i, d in enumerate(DAYS)}
    for sport, se in envelope.by_sport.items():
        if se.long_session_min <= 0:
            continue
        done_long = (
            facts.completed_this_week.by_sport[sport].longest_min
            if sport in facts.completed_this_week.by_sport
            else 0.0
        )
        if done_long >= se.long_session_min:
            continue
        # A brick is a long ride with a short run tacked on: it satisfies the
        # long bike requirement, never the long run one.
        counts_as = (sport, "brick") if sport == "bike" else (sport,)
        planned = [d for d in days if d.sport in counts_as and d.duration_min > 0]
        if planned and max(d.duration_min for d in planned) >= se.long_session_min:
            continue
        if not planned:
            notes.append(
                f"no long {sport} this week and none completed — "
                f"add one when a day frees up"
            )
            continue
        target = max(planned, key=lambda d: (order.get(d.day, 0), d.duration_min))
        notes.append(
            f"{target.day} {sport} raised {target.duration_min} -> "
            f"{se.long_session_min:.0f} min: the week needs one long {sport}"
        )
        target.duration_min = int(se.long_session_min)
        target.purpose = PURPOSE_FOR_ROLE["long"]
    return days


def _annotate(why: str, note: str) -> str:
    """Attach a rules override to a session's rationale.

    Without this, a model's "you said you feel great" survives verbatim onto a
    session the deload rules just cut in half.
    """
    why = (why or "").strip().rstrip(".")
    if note in why:
        return why
    return f"{why} [{note}]" if why else f"[{note}]"


def _relabel(days: list[PlanDay], envelope: Envelope) -> None:
    """Keep `purpose` and `why` honest after the constraints have had their way.

    A session described as the week's long run reads as a lie once the volume cap
    has cut it to 25 minutes, and a brick's written split has to add up to the
    duration actually prescribed.
    """
    for d in days:
        if d.sport == "brick" and d.duration_min > 0:
            run_off = 20 if d.duration_min < 140 else 25
            ride = max(30, d.duration_min - run_off)
            d.why = (
                f"{ride} min ride straight into a {d.duration_min - ride} min easy run"
            )
            continue
        if d.purpose != PURPOSE_FOR_ROLE["long"]:
            continue
        long_floor = (
            envelope.by_sport[d.sport].long_session_min
            if d.sport in envelope.by_sport
            else 0.0
        )
        if d.duration_min < max(long_floor, 50):
            d.purpose = PURPOSE_FOR_ROLE["endurance"]
            d.why = f"cut back this week — {d.why}"


def _fit_budget(
    days: list[PlanDay], budget: float, envelope: Envelope, notes: list[str]
) -> list[PlanDay]:
    """Bring the week inside its minutes budget.

    Scaling alone would turn a tight week into a set of useless 20-minute stubs,
    so a session that would fall below its floor is dropped instead — lowest
    training priority first.
    """

    def rank(d: PlanDay) -> tuple[int, int]:
        base = DROP_PRIORITY.get(d.sport, 1)
        long_floor = (
            envelope.by_sport[d.sport].long_session_min
            if d.sport in envelope.by_sport
            else 0.0
        )
        if long_floor and d.duration_min >= long_floor:
            base += 2  # a required long session outranks its everyday siblings
        return (base, d.duration_min)

    while True:
        movable = [d for d in days if d.sport != "rest" and d.duration_min > 0]
        total = sum(d.duration_min for d in movable)
        if not movable or total <= budget:
            return days

        if budget < min(SESSION_FLOOR.get(d.sport, 25) for d in movable):
            notes.append(
                f"weekly volume budget is spent ({budget:.0f} min left) — "
                f"the rest of the week is rest"
            )
            kept = [d for d in days if d.sport == "rest"]
            covered = {d.day for d in kept}
            for d in movable:
                if d.day in covered:
                    continue
                covered.add(d.day)
                kept.append(
                    PlanDay(
                        day=d.day,
                        sport="rest",
                        duration_min=0,
                        target_zone="n/a",
                        purpose=PURPOSE_FOR_ROLE["rest"],
                        why="the week's volume cap is already spent",
                    )
                )
            return kept

        factor = budget / total
        if all(
            d.duration_min * factor >= SESSION_FLOOR.get(d.sport, 25) for d in movable
        ):
            notes.append(
                f"scaled remaining volume {total:.0f} -> {budget:.0f} min "
                f"(progression cap {envelope.progression_cap_pct:.0f}%/week)"
            )
            for d in movable:
                if d.sport == "strength":
                    continue
                if factor < 0.85:
                    d.why = _annotate(d.why, "trimmed to fit the week's volume cap")
                d.duration_min = int(d.duration_min * factor)
            return days

        victim = min(movable, key=rank)
        notes.append(
            f"dropped {victim.duration_min} min {victim.sport} on {victim.day}: "
            f"does not fit the week's volume cap"
        )
        days.remove(victim)


def _ensure_rest_days(
    days: list[PlanDay], facts: PlannerFacts, envelope: Envelope, notes: list[str]
) -> list[PlanDay]:
    """At least `min_rest_days` days with nothing on them, across the whole week."""
    planned_days = {d.day for d in days if d.duration_min > 0}
    rest_days = {
        day for day in DAYS if day not in facts.trained_days and day not in planned_days
    }
    shortfall = envelope.min_rest_days - len(rest_days)
    if shortfall <= 0:
        return days

    order = {d: i for i, d in enumerate(DAYS)}
    by_day: dict[str, int] = {}
    for d in days:
        by_day[d.day] = by_day.get(d.day, 0) + d.duration_min
    # Clear the lightest remaining days — never one that has already been trained.
    candidates = sorted(
        (
            day
            for day in facts.days_remaining
            if by_day.get(day, 0) > 0 and day not in facts.trained_days
        ),
        key=lambda day: (by_day.get(day, 0), order[day]),
    )
    for day in candidates[:shortfall]:
        notes.append(
            f"{day} cleared to a full rest day: the week needs {envelope.min_rest_days}"
        )
        days = [d for d in days if d.day != day]
        days.append(
            PlanDay(
                day=day,
                sport="rest",
                duration_min=0,
                target_zone="n/a",
                purpose=PURPOSE_FOR_ROLE["rest"],
                why="rules require a full rest day",
            )
        )
    return days


# --------------------------------------------------------------------------
# Layer 3 — orchestration
# --------------------------------------------------------------------------


def build_payload(
    facts: PlannerFacts,
    envelope: Envelope,
    strength_log: Sequence[dict[str, Any]] = (),
    checkin: Checkin | None = None,
) -> PlanPayload:
    """Exactly what the model sees. Facts and bounds — no free-form instructions."""
    strength_done = (
        facts.completed_this_week.by_sport["strength"].sessions
        if "strength" in facts.completed_this_week.by_sport
        else 0
    )
    state = strength.strength_state(
        strength_log,
        session_index=strength_done,
        intensity=0.6 if envelope.deload else 1.0,
    )
    state_dict = state.model_dump(mode="json")
    state_dict["allowed_exercise_ids"] = sorted(strength.LIBRARY_IDS)
    state_dict["sessions_remaining_this_week"] = max(
        0, envelope.strength_sessions - strength_done
    )

    env_dict = envelope.model_dump(mode="json")
    env_dict["days_remaining"] = facts.days_remaining
    env_dict["remaining_minutes_budget"] = round(remaining_budget(facts, envelope))

    return PlanPayload(
        completed_this_week=facts.completed_this_week.model_dump(mode="json"),
        recovery_signals=facts.recovery.model_dump(mode="json"),
        envelope=env_dict,
        strength_state=state_dict,
        checkin=(checkin.model_dump(mode="json") if checkin else {}),
        history={
            "previous_weeks": [
                w.model_dump(mode="json") for w in facts.previous_weeks[-4:]
            ],
            "ef_trends": [t.model_dump(mode="json") for t in facts.ef_trends],
            "recent_checkins": [c.model_dump(mode="json") for c in facts.recent_checkins],
        },
    )


def plan_week(
    store: Store,
    checkin: Checkin | None = None,
    today: date | None = None,
    use_ai: bool = True,
    pushback: str | None = None,
    previous_plan: dict[str, Any] | None = None,
    save: bool = True,
) -> WeekPlan:
    """Facts -> envelope -> AI (optional) -> enforcement -> saved plan."""
    facts = build_facts(store, today=today)
    envelope = build_envelope(facts, store)
    targets = store.targets()
    strength_log = store.strength_log(since=facts.week_start - timedelta(days=120))

    fallback = enforce(
        rules_plan(facts, envelope, strength_log, checkin), facts, envelope, strength_log
    )
    payload = build_payload(facts, envelope, strength_log, checkin)
    if targets:
        payload.envelope["athlete_weekly_targets"] = targets

    plan = fallback
    if use_ai:
        try:
            raw = ai.plan_week(
                payload.model_dump(mode="json"),
                user_pushback=pushback,
                previous_plan=previous_plan,
            )
            candidate = WeekPlan(
                week_plan=[
                    d
                    for d in (_coerce_day(x) for x in raw.get("week_plan", []))
                    if d is not None
                ],
                # Prefixed so the UI never presents model commentary as a rule.
                flags=[f"AI: {f}" for f in raw.get("flags", [])][:8],
                adjustments_made=[f"AI: {a}" for a in raw.get("adjustments_made", [])][:8],
                source="ai",
            )
            if not candidate.week_plan:
                raise ai.AIUnavailable("AI returned an empty week")
            plan = enforce(candidate, facts, envelope, strength_log)
        except ai.AIUnavailable as exc:
            log.info("AI layer unavailable (%s) — using the rules plan", exc)
            plan.flags.append(f"AI layer unavailable ({exc}); this is the rules-only plan")
        except Exception as exc:  # noqa: BLE001 - never let the planner fail hard
            log.warning("AI layer failed (%s) — using the rules plan", exc)
            plan.flags.append(f"AI layer errored ({type(exc).__name__}); rules-only plan")

    completed = completed_entries(facts, store)
    plan.week_plan = completed + plan.week_plan

    if save:
        store.save_plan(
            facts.week_start,
            plan.model_dump(mode="json"),
            plan.source,
            payload.model_dump(mode="json"),
        )
    return plan


def _coerce_day(raw: Any) -> PlanDay | None:
    try:
        return PlanDay.model_validate(raw)
    except Exception:  # noqa: BLE001 - a bad row is dropped, not fatal
        return None
