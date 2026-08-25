"""The rules must beat the AI, every time.

These tests exist because the whole design rests on one claim: a sycophantic or
broken model cannot produce a dangerous week. `enforce()` is fed a deliberately
reckless plan and the output is checked against every constraint.
"""

from __future__ import annotations

import json
from datetime import timedelta

import pytest

from conftest import TODAY

from core import ai, planner, strength
from core.schemas import DAYS, Checkin, PlanDay, WeekPlan

# Everything a raw LLM gets wrong when told "I feel amazing": too much volume,
# intervals during a deload, invented plyometric exercises, three strength days,
# no rest day, and a malformed row for good measure.
ROGUE_PLAN = {
    "week_plan": [
        {"day": "Mon", "sport": "run", "duration_min": 200, "target_zone": "Z4",
         "purpose": "crush it", "exercise_ids": [], "why": "you said you feel great"},
        {"day": "Tue", "sport": "bike", "duration_min": 400, "target_zone": "Z5",
         "purpose": "big day", "exercise_ids": [], "why": "more is more"},
        {"day": "Tue", "sport": "strength", "duration_min": 60, "target_zone": "Z4",
         "purpose": "legs", "exercise_ids": ["box_jumps", "depth_drops", "squat_1rm"],
         "why": "plyos build power"},
        {"day": "Wed", "sport": "run", "duration_min": 120, "target_zone": "Z4",
         "purpose": "intervals", "exercise_ids": [], "why": "speed week"},
        {"day": "Thu", "sport": "swim", "duration_min": 90, "target_zone": "Z4",
         "purpose": "swim hard", "exercise_ids": [], "why": "why not"},
        {"day": "Fri", "sport": "run", "duration_min": 150, "target_zone": "Z3",
         "purpose": "more running", "exercise_ids": [], "why": "fourth run"},
        {"day": "Sat", "sport": "brick", "duration_min": 300, "target_zone": "Z4",
         "purpose": "epic brick", "exercise_ids": [], "why": "race sim"},
        {"day": "Sun", "sport": "strength", "duration_min": 60, "target_zone": "n/a",
         "purpose": "legs again", "exercise_ids": ["split_squat"], "why": "third day"},
        {"day": "Quidditch", "sport": "nonsense", "duration_min": 45,
         "target_zone": "Z9", "purpose": "?", "exercise_ids": [], "why": "malformed"},
        {"day": "Sun", "sport": "run", "duration_min": 60, "target_zone": "Z2",
         "purpose": "double", "exercise_ids": [], "why": "no rest day anywhere"},
    ],
    "flags": ["athlete feels amazing, ignoring the deload"],
    "adjustments_made": ["added volume because motivation is high"],
}


class RogueBackend:
    name = "rogue"

    # json_mode is part of the backend protocol: the planner asks for guaranteed
    # JSON where a provider supports it.
    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        return json.dumps(ROGUE_PLAN)


class BabblingBackend:
    name = "babbling"

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        return "I'd love to help you train! Here are some thoughts..."


def enforced_rogue(store):
    facts = planner.build_facts(store, today=TODAY)
    envelope = planner.build_envelope(facts, store)
    raw = ai.plan_week({}, backend=RogueBackend())
    candidate = WeekPlan(
        week_plan=[
            d for d in (planner._coerce_day(x) for x in raw["week_plan"]) if d
        ],
        flags=raw["flags"],
        adjustments_made=raw["adjustments_made"],
        source="ai",
    )
    plan = planner.enforce(candidate, facts, envelope, store.strength_log())
    return facts, envelope, plan


# --- the exercise library is closed --------------------------------------


def test_invented_exercises_never_survive(wrecked):
    _, _, plan = enforced_rogue(wrecked)
    used = {e for d in plan.week_plan for e in d.exercise_ids}
    assert used <= strength.LIBRARY_IDS
    assert "box_jumps" not in used


def test_validate_exercise_ids_is_a_whitelist():
    assert strength.validate_exercise_ids(["rdl", "box_jumps", "RDL", "wall_sit"]) == [
        "rdl",
        "wall_sit",
    ]


# --- deload is not negotiable -------------------------------------------


def test_recovery_signals_force_a_deload(wrecked):
    facts = planner.build_facts(wrecked, today=TODAY)
    envelope = planner.build_envelope(facts, wrecked)
    assert envelope.deload
    joined = " ".join(envelope.deload_reasons).lower()
    assert "hrv" in joined and "resting hr" in joined and "readiness" in joined


def test_deload_strips_intensity_and_bricks(wrecked):
    _, _, plan = enforced_rogue(wrecked)
    assert not [d for d in plan.week_plan if d.target_zone in planner.QUALITY_ZONES]
    assert not [d for d in plan.week_plan if d.sport == "brick"]


def test_a_great_mood_cannot_cancel_a_deload(wrecked):
    """The check-in is an input to the AI, never to the envelope."""
    facts = planner.build_facts(wrecked, today=TODAY)
    ecstatic = planner.build_envelope(facts, wrecked)
    plan = planner.plan_week(
        wrecked,
        checkin=Checkin(
            date=TODAY, sleep=5, soreness=1, motivation=5, time_available_min=300,
            notes="I feel unstoppable, give me a huge week",
        ),
        today=TODAY,
        use_ai=False,
    )
    assert ecstatic.deload
    planned = [d for d in plan.week_plan if d.purpose != "completed"]
    assert sum(d.duration_min for d in planned) <= planner.remaining_budget(
        facts, ecstatic
    ) + 60  # a required long session may overshoot, nothing else may


# --- volume and session caps --------------------------------------------


def test_progression_cap_bounds_the_week(healthy):
    facts = planner.build_facts(healthy, today=TODAY)
    envelope = planner.build_envelope(facts, healthy)
    prev = facts.previous_weeks[-1].total_minutes
    assert envelope.max_week_minutes <= prev * 1.1 + 1


def test_session_counts_stay_inside_the_envelope(wrecked):
    facts, envelope, plan = enforced_rogue(wrecked)
    for sport, se in envelope.by_sport.items():
        done = (
            facts.completed_this_week.by_sport[sport].sessions
            if sport in facts.completed_this_week.by_sport
            else 0
        )
        planned = len([d for d in plan.week_plan if d.sport == sport])
        assert done + planned <= se.max_sessions, sport


def test_strength_frequency_is_capped(wrecked):
    """Only the remaining allowance is plannable.

    Sessions already logged cannot be un-done, so when the athlete has already
    trained legs twice and a deload then cuts the week's allowance to one, the
    correct behaviour is to plan zero more — not to go negative.
    """
    facts, envelope, plan = enforced_rogue(wrecked)
    done = (
        facts.completed_this_week.by_sport["strength"].sessions
        if "strength" in facts.completed_this_week.by_sport
        else 0
    )
    planned = len([d for d in plan.week_plan if d.sport == "strength"])
    assert planned <= max(0, envelope.strength_sessions - done)
    assert planned == 0, "two sessions were already logged this week"


def test_minimum_rest_days_survive(wrecked):
    facts, envelope, plan = enforced_rogue(wrecked)
    busy = {d.day for d in plan.week_plan if d.duration_min > 0}
    free = [d for d in DAYS if d not in busy and d not in facts.trained_days]
    assert len(free) >= envelope.min_rest_days


def test_per_session_ceilings_hold(wrecked):
    _, _, plan = enforced_rogue(wrecked)
    for d in plan.week_plan:
        assert d.duration_min <= planner.SESSION_CEILING[d.sport], d


# --- failure modes fall back, never crash --------------------------------


def test_unparseable_model_output_falls_back_to_rules(healthy, monkeypatch):
    monkeypatch.setattr(ai, "get_backend", lambda name=None: BabblingBackend())
    plan = planner.plan_week(healthy, today=TODAY, use_ai=True)
    assert plan.source == "rules"
    assert any("unavailable" in f.lower() for f in plan.flags)
    assert plan.week_plan  # still a usable week


def test_missing_ai_config_still_produces_a_plan(healthy, monkeypatch):
    monkeypatch.setenv("AI_BACKEND", "none")
    plan = planner.plan_week(healthy, today=TODAY, use_ai=True)
    assert plan.source == "rules"
    assert plan.week_plan


def test_empty_database_does_not_explode(tmp_path):
    from core.store import Store

    with Store(str(tmp_path / "empty.db")) as store:
        plan = planner.plan_week(store, today=TODAY, use_ai=False)
    assert plan.week_plan
    assert plan.minutes() > 0


# --- the rationale must match the numbers --------------------------------


def test_overridden_sessions_say_so(wrecked):
    _, _, plan = enforced_rogue(wrecked)
    downgraded = [d for d in plan.week_plan if "deload" in d.why.lower()]
    assert downgraded, "a session cut by the deload rules should say it was cut"
    assert plan.source == "ai_repaired"
    assert plan.adjustments_made


def test_brick_description_matches_its_duration(healthy):
    facts = planner.build_facts(healthy, today=TODAY)
    envelope = planner.build_envelope(facts, healthy)
    envelope.brick_required = True
    plan = planner.enforce(
        planner.rules_plan(facts, envelope, healthy.strength_log()),
        facts,
        envelope,
        healthy.strength_log(),
    )
    for d in plan.week_plan:
        if d.sport != "brick":
            continue
        ride, run = (int(n) for n in __import__("re").findall(r"(\d+) min", d.why)[:2])
        assert ride + run == d.duration_min


# --- leg strength placement ---------------------------------------------


def _plan_for(store, today):
    facts = planner.build_facts(store, today=today)
    env = planner.build_envelope(facts, store)
    return facts, env, planner.enforce(
        planner.rules_plan(facts, env, store.strength_log()), facts, env,
        store.strength_log())


def test_leg_strength_never_shares_a_day_with_a_run_or_ride(healthy):
    """The athlete's rule: legs go on days free of running and riding."""
    from datetime import date

    _, _, plan = _plan_for(healthy, date(2026, 8, 24))   # a Monday, week wide open
    strength_days = {d.day for d in plan.week_plan if d.sport == "strength"}
    for day in strength_days:
        clash = [d for d in plan.week_plan
                 if d.day == day and d.sport in planner.LEG_CONFLICT_SPORTS
                 and d.duration_min > 0]
        assert not clash, f"legs on {day} alongside {[c.sport for c in clash]}"


def test_leg_strength_is_never_the_day_before_a_long_run(healthy):
    from datetime import date

    from core.schemas import DAYS

    _, _, plan = _plan_for(healthy, date(2026, 8, 24))
    long_runs = [d for d in plan.week_plan
                 if d.sport in ("run", "brick") and d.duration_min >= 70]
    for lr in long_runs:
        i = DAYS.index(lr.day)
        if i == 0:
            continue
        before = DAYS[i - 1]
        assert not [d for d in plan.week_plan
                    if d.day == before and d.sport == "strength"], \
            f"legs on {before}, the day before a long run on {lr.day}"


def test_a_strength_only_day_still_leaves_a_genuinely_blank_rest_day(healthy):
    """Putting legs on an otherwise-free day must not consume the last rest day."""
    from datetime import date

    from core.schemas import DAYS

    facts, env, plan = _plan_for(healthy, date(2026, 8, 24))
    busy = {d.day for d in plan.week_plan if d.duration_min > 0}
    blank = [d for d in DAYS if d not in busy and d not in facts.trained_days]
    assert len(blank) >= env.min_rest_days, \
        f"only {len(blank)} blank day(s), envelope wants {env.min_rest_days}"


def test_an_ai_plan_stacking_legs_onto_a_ride_gets_moved(wrecked):
    """Enforcement, not just the template, has to hold this line."""
    from datetime import date

    from core.schemas import WeekPlan

    facts = planner.build_facts(wrecked, today=date(2026, 8, 24))
    env = planner.build_envelope(facts, wrecked)
    rogue = WeekPlan(week_plan=[
        planner.PlanDay(day="Tue", sport="bike", duration_min=90, target_zone="Z2"),
        planner.PlanDay(day="Tue", sport="strength", duration_min=30,
                        target_zone="n/a", exercise_ids=["rdl"]),
    ], source="ai")
    out = planner.enforce(rogue, facts, env, wrecked.strength_log())
    legs = [d for d in out.week_plan if d.sport == "strength"]
    for leg in legs:
        assert not [d for d in out.week_plan
                    if d.day == leg.day and d.sport in planner.LEG_CONFLICT_SPORTS
                    and d.duration_min > 0]


# --------------------------------------------------------------------------
# The dashboard-wide sport filter. It reaches the planner, not just the charts:
# a page filtered to run and bike that still prescribes a swim contradicts
# itself, and the athlete would reasonably trust the plan over the filter.
# --------------------------------------------------------------------------


def prescribed(plan):
    """Sessions the plan is asking for, as opposed to ones already logged.

    A completed session stays in the week whatever the filter says — it happened,
    and a planner that quietly deleted training history would be lying about the
    week. Only the prescriptions are the filter's business.
    """
    return [d for d in plan.week_plan
            if d.duration_min > 0 and d.purpose != "completed"]


def test_only_sports_removes_the_others_from_the_plan(healthy):
    plan = planner.plan_week(healthy, today=TODAY, use_ai=False,
                             only_sports=["run", "bike"])
    trained = {d.sport for d in prescribed(plan)}
    assert "swim" not in trained
    # A brick is a ride and a run in one session, so it stays legitimate here.
    assert trained <= {"run", "bike", "brick"}


def test_only_sports_can_switch_strength_off(healthy):
    plan = planner.plan_week(healthy, today=TODAY, use_ai=False,
                             only_sports=["run", "bike"])
    assert not [d for d in prescribed(plan) if d.sport == "strength"]


def test_strength_survives_when_it_is_selected(healthy):
    """Asserted on the envelope, not the plan.

    The plan is the wrong place to check: this fixture has already logged its two
    strength sessions for the week, so the frequency cap correctly prescribes no
    more. The envelope is where "is strength on at all" actually lives.
    """
    facts = planner.build_facts(healthy, today=TODAY)
    on = planner.build_envelope(facts, healthy,
                                only_sports=["run", "bike", "strength"])
    off = planner.build_envelope(facts, healthy, only_sports=["run", "bike"])
    assert on.strength_sessions > 0
    assert off.strength_sessions == 0


def test_a_filtered_week_still_obeys_the_volume_cap(healthy):
    """Narrowing the sports must not become a way to smuggle volume in.

    The share a dropped sport held is redistributed, so the risk is real: three
    sports' minutes landing on two.
    """
    facts = planner.build_facts(healthy, today=TODAY)
    envelope = planner.build_envelope(facts, healthy, only_sports=["run", "bike"])
    plan = planner.plan_week(healthy, today=TODAY, use_ai=False,
                             only_sports=["run", "bike"])
    planned = sum(d.duration_min for d in plan.week_plan)
    assert planned <= envelope.max_week_minutes * 1.10


def test_no_selection_at_all_plans_everything(healthy):
    """only_sports=None is "no opinion", never "nothing enabled"."""
    plan = planner.plan_week(healthy, today=TODAY, use_ai=False, only_sports=None)
    assert {d.sport for d in prescribed(plan)}


# --------------------------------------------------------------------------
# Endurance sessions keep a clear day between them. Strength is exempt: it is
# the low-impact work that belongs in the gaps.
# --------------------------------------------------------------------------

DAY_INDEX = {d: i for i, d in enumerate(DAYS)}


def endurance_days(plan) -> list[int]:
    return sorted(DAY_INDEX[d.day] for d in plan.week_plan
                  if d.sport in planner.SPACED_SPORTS and d.duration_min > 0)


def test_the_rules_plan_never_stacks_endurance_on_consecutive_days(healthy):
    plan = planner.plan_next_week(healthy, today=TODAY, use_ai=False, save=False)
    days = endurance_days(plan)
    assert not [(a, b) for a, b in zip(days, days[1:]) if b - a <= 1], days


def test_a_model_proposing_back_to_back_sessions_is_repaired(healthy):
    """The whole point of enforce(): the model cannot quietly ignore the rule."""
    facts = planner.build_facts(healthy, today=TODAY)
    envelope = planner.build_envelope(facts, healthy)
    crammed = WeekPlan(week_plan=[
        PlanDay(day="Mon", sport="run", duration_min=40, target_zone="Z2"),
        PlanDay(day="Tue", sport="bike", duration_min=40, target_zone="Z2"),
        PlanDay(day="Wed", sport="swim", duration_min=30, target_zone="Z2"),
        PlanDay(day="Thu", sport="run", duration_min=40, target_zone="Z2"),
        PlanDay(day="Fri", sport="bike", duration_min=60, target_zone="Z2"),
    ], flags=[], adjustments_made=[])
    plan = planner.enforce(crammed, facts, envelope, healthy.strength_log())
    days = endurance_days(plan)
    assert not [(a, b) for a, b in zip(days, days[1:]) if b - a <= 1], days
    assert plan.source in ("ai_repaired", "rules")


def test_strength_is_exempt_from_the_spacing_rule(healthy):
    """Strength on the day after a ride is the intended arrangement, not a bug."""
    plan = planner.plan_next_week(healthy, today=TODAY, use_ai=False, save=False)
    strength = [DAY_INDEX[d.day] for d in plan.week_plan
                if d.sport == "strength" and d.duration_min > 0]
    endurance = endurance_days(plan)
    # Every strength day sits next to an endurance day precisely because it fills
    # the gaps between them.
    assert strength
    assert all(any(abs(s - e) == 1 for e in endurance) for s in strength)


def test_spacing_never_stacks_two_sessions_on_one_day(healthy):
    """Dropping is preferred to stacking, which would defeat the rule."""
    facts = planner.build_facts(healthy, today=TODAY)
    envelope = planner.build_envelope(facts, healthy)
    crammed = WeekPlan(week_plan=[
        PlanDay(day=d, sport="run", duration_min=30, target_zone="Z2")
        for d in DAYS
    ], flags=[], adjustments_made=[])
    plan = planner.enforce(crammed, facts, envelope, healthy.strength_log())
    per_day: dict[str, int] = {}
    for d in plan.week_plan:
        if d.sport in planner.SPACED_SPORTS and d.duration_min > 0:
            per_day[d.day] = per_day.get(d.day, 0) + 1
    assert not [k for k, v in per_day.items() if v > 1], per_day


def test_a_tight_budget_cannot_drop_below_three_endurance_sessions(healthy):
    """The floor and the progression cap genuinely conflict on a light week.

    A near-empty previous week makes the +10% budget small enough that
    _fit_budget drops whole sessions. Three short easy sessions is not what the
    cap exists to prevent, so the floor wins and the overshoot is declared.
    """
    plan = planner.plan_week(healthy, today=TODAY, use_ai=False, save=False)
    live = [d for d in plan.week_plan
            if d.sport in planner.SPACED_SPORTS and d.duration_min > 0]
    assert len(live) >= planner.MIN_ENDURANCE_SESSIONS, [
        (d.day, d.sport, d.duration_min) for d in plan.week_plan]


def test_restored_sessions_still_obey_spacing_and_the_sport_filter(healthy):
    """Whatever the floor puts back must satisfy every other rule too."""
    plan = planner.plan_week(healthy, today=TODAY, use_ai=False, save=False,
                             only_sports=["bike", "strength"])
    # Prescriptions only. A completed run stays in the week whatever the filter
    # says, and two sessions that already happened on adjacent days cannot be
    # spaced retrospectively.
    live = [d for d in prescribed(plan) if d.sport in planner.SPACED_SPORTS]
    assert {d.sport for d in live} <= {"bike", "brick"}, [
        (d.day, d.sport) for d in live]


def test_the_floor_declares_any_overshoot(healthy):
    """Going over the budget silently would be the actual problem."""
    facts = planner.build_facts(healthy, today=TODAY)
    envelope = planner.build_envelope(facts, healthy)
    plan = planner.plan_week(healthy, today=TODAY, use_ai=False, save=False)
    planned = sum(d.duration_min for d in plan.week_plan
                  if d.purpose != "completed")
    if planned > envelope.max_week_minutes:
        assert any("floor" in a or "long" in a for a in plan.adjustments_made), \
            plan.adjustments_made


# --------------------------------------------------------------------------
# Watch-recorded strength sets. If a session logged on the 265 does not map
# back into the library, it never reaches the progression, and the athlete's
# next session is built on a session that looks like it never happened.
# --------------------------------------------------------------------------

GARMIN_CASES = [
    # (category, name, expected library id)
    ("CALF_RAISE", "STANDING_CALF_RAISE", "calf_raise_straight"),
    ("CALF_RAISE", "SEATED_CALF_RAISE", "calf_raise_bent"),
    ("CALF_RAISE", "SINGLE_LEG_CALF_RAISE", "calf_raise_single_leg"),
    ("SQUAT", "SPLIT_SQUAT", "split_squat"),
    ("SQUAT", "BULGARIAN_SPLIT_SQUAT", "split_squat"),
    ("SQUAT", "SPANISH_SQUAT", "spanish_squat"),
    ("SQUAT", "WALL_SIT", "wall_sit"),
    ("SQUAT", "GOBLET_SQUAT", "goblet_squat"),
    ("SQUAT", "SINGLE_LEG_STEP_DOWN", "step_down"),
    ("DEADLIFT", "ROMANIAN_DEADLIFT", "rdl"),
    ("DEADLIFT", "SINGLE_LEG_ROMANIAN_DEADLIFT", "single_leg_rdl"),
    ("LUNGE", "REVERSE_LUNGE", "reverse_lunge"),
    ("STEP_UP", "BOX_STEP_UP", "step_up"),
    ("HIP_RAISE", "GLUTE_BRIDGE", "glute_bridge"),
    ("HIP_RAISE", "BARBELL_HIP_THRUST", "hip_thrust"),
    ("HIP_STABILITY", "SIDE_LYING_LEG_RAISE", "side_lying_abduction"),
    ("HIP_STABILITY", "LATERAL_BAND_WALK", "band_monster_walk"),
    ("PLANK", "SIDE_PLANK", "side_plank_hip_lift"),
    ("PLANK", "SIDE_PLANK_HIP_ADDUCTION", "copenhagen_plank"),
    ("LEG_CURL", "NORDIC_HAMSTRING_CURL", "nordic_curl_assisted"),
    ("TIBIALIS_RAISE", "TOE_RAISE", "tib_raise"),
]


@pytest.mark.parametrize("category,name,expected", GARMIN_CASES)
def test_garmin_exercises_map_into_the_library(category, name, expected):
    assert strength.map_garmin_exercise(category, name) == expected


def test_a_specific_name_beats_a_generic_category():
    """Garmin sends both, and the category is also a key in the table.

    Without name-before-category precedence, HIP_STABILITY / LATERAL_BAND_WALK
    logged a banded walk as a side-lying leg raise — a real mis-attribution that
    would then drive the wrong exercise's progression.
    """
    assert strength.map_garmin_exercise(
        "HIP_STABILITY", "LATERAL_BAND_WALK") == "band_monster_walk"
    assert strength.map_garmin_exercise("SQUAT", "SPANISH_SQUAT") == "spanish_squat"


def test_every_library_exercise_is_reachable_from_garmin():
    """Otherwise a prescribed exercise could never be auto-logged from the watch."""
    reachable = set(strength.GARMIN_EXERCISE_MAP.values())
    # The isometric calf hold is the one exception: the watch records it as a
    # calf raise, because it cannot tell a hold from a rep.
    missing = set(strength.EXERCISES) - reachable - {"single_leg_calf_hold"}
    assert not missing, sorted(missing)


def test_an_unknown_exercise_is_left_unmapped_not_guessed():
    """A wrong guess corrupts whatever it is mistaken for; None asks the athlete."""
    assert strength.map_garmin_exercise("BENCH_PRESS", "BARBELL_BENCH_PRESS") is None
    assert strength.map_garmin_exercise("SHOULDER_PRESS", "OVERHEAD_PRESS") is None
    assert strength.map_garmin_exercise(None, None) is None


# --------------------------------------------------------------------------
# Pushing a session to the watch. It writes to the Garmin account, so the
# payload has to be right before it is sent, not after.
# --------------------------------------------------------------------------


def test_the_workout_payload_names_the_right_sport():
    """sportTypeId 5 is strength_training. 13, the plausible guess, is rucking —
    established by uploading both to the live account and reading them back."""
    from core import garmin_workout

    presc = strength.build_session([], session_index=0)
    w = garmin_workout.build(presc, "Legs A")
    assert w["sportType"]["sportTypeKey"] == "strength_training"
    assert w["sportType"]["sportTypeId"] == 5
    assert w["workoutSegments"][0]["sportType"] == w["sportType"]


def test_every_library_exercise_can_be_sent():
    """An unmapped exercise would silently reach the watch as a bare SQUAT."""
    from core import garmin_workout

    missing = set(strength.EXERCISES) - set(garmin_workout.GARMIN_TARGET)
    assert not missing, sorted(missing)


def test_a_set_becomes_one_step_each_with_rest_between():
    from core import garmin_workout

    presc = strength.build_session([], session_index=0)
    steps = garmin_workout.build(presc, "Legs A")["workoutSegments"][0]["workoutSteps"]
    work = [s for s in steps if s["stepType"]["stepTypeKey"] == "interval"]
    assert len(work) == sum(max(1, p.sets) for p in presc)
    # Rest between every pair of working sets, and none trailing at the end.
    assert steps[-1]["stepType"]["stepTypeKey"] == "interval"
    assert all(s["stepOrder"] == i + 1 for i, s in enumerate(steps))


def test_isometric_holds_go_as_time_not_reps():
    """A 30-second wall sit sent as "30 reps" would be nonsense on the watch."""
    from core import garmin_workout

    presc = strength.build_session([], session_index=0)
    steps = garmin_workout.build(presc, "Legs A")["workoutSegments"][0]["workoutSteps"]
    holds = [p for p in presc if p.hold_s]
    assert holds, "session A should contain an isometric"
    timed = [s for s in steps if s["stepType"]["stepTypeKey"] == "interval"
             and s["endCondition"]["conditionTypeKey"] == "time"]
    assert len(timed) == sum(max(1, p.sets) for p in holds)
    assert all(s["endConditionValue"] > 0 for s in timed)


def test_every_library_exercise_reaches_the_watch_by_name():
    """It used to be that anything Garmin had no name for was sent as a bare
    category, and the watch labelled a tibialis raise "Calf raise" and a wall sit
    "Squat" — the right exercise under a different muscle's name. Every id now
    maps to a name that was round-tripped against the account, and the real name
    still goes in the description."""
    from core import garmin_workout

    for eid, exercise in strength.EXERCISES.items():
        category, name = garmin_workout.GARMIN_TARGET[eid]
        assert category in garmin_workout.VALID_CATEGORIES, (eid, category)
        assert name, f"{eid} would show as a bare {category}"
        assert name in garmin_workout.VERIFIED_NAMES, (eid, name)
        presc = strength.next_prescription(eid, [], 1.0)
        step = garmin_workout._step(1, presc, exercise)
        assert step["exerciseName"] == name
        assert exercise.name in step["description"]


def test_no_exercise_is_filed_under_the_wrong_muscle():
    """The specific bug: a tibialis raise is a shin exercise and was going up as
    a calf raise, which is the muscle it exists to balance."""
    from core import garmin_workout

    assert garmin_workout.GARMIN_TARGET["tib_raise"][0] != "CALF_RAISE"
    assert "DORSIFLEXION" in garmin_workout.GARMIN_TARGET["tib_raise"][1]
    assert "WALL" in garmin_workout.GARMIN_TARGET["wall_sit"][1]


def test_an_empty_session_is_refused_rather_than_uploaded():
    """Sending an empty workout would leave a useless entry on the watch."""
    from core import garmin_workout

    with pytest.raises(ValueError):
        garmin_workout.push(object(), [], "Legs A")


def test_every_mapped_category_is_one_garmin_accepts():
    """An unknown category is HTTP 400 and rejects the entire session. STEP_UP
    looked obvious and is not a category; step-ups live under SQUAT."""
    from core import garmin_workout

    for eid, (category, _) in garmin_workout.GARMIN_TARGET.items():
        assert category in garmin_workout.VALID_CATEGORIES, (eid, category)


def test_only_names_garmin_actually_keeps_are_sent():
    """An invalid name is accepted and silently dropped, so the upload looks
    fine and the watch shows a bare category. Of thirty probed, nineteen went
    that way."""
    from core import garmin_workout

    for eid, (_, name) in garmin_workout.GARMIN_TARGET.items():
        if name is not None:
            assert name in garmin_workout.VERIFIED_NAMES, (eid, name)


def test_all_three_sessions_build_without_raising():
    """The guard in _step turns a bad mapping into a test failure rather than a
    400 at upload time or a blank label on the watch."""
    from core import garmin_workout

    for index in range(3):
        presc = strength.build_session([], session_index=index)
        steps = garmin_workout.build(presc, f"Legs {index}")
        assert steps["workoutSegments"][0]["workoutSteps"]


def test_endurance_hr_target_is_parsed_from_the_plan_text():
    """The plan carries "112-137 bpm"; the watch needs two numbers."""
    from core import garmin_workout

    assert garmin_workout.parse_hr_target("112-137 bpm") == (112, 137)
    assert garmin_workout.parse_hr_target("Z2") is None
    assert garmin_workout.parse_hr_target(None) is None
    # Nonsense ranges are refused rather than sent to a watch that will buzz.
    assert garmin_workout.parse_hr_target("300-400 bpm") is None
    assert garmin_workout.parse_hr_target("137-112 bpm") is None


def test_a_long_session_gets_a_warmup_and_a_short_one_does_not():
    """Ten minutes of warm-up out of a twenty-minute run leaves nothing to warm
    up for."""
    from core import garmin_workout

    long_steps = garmin_workout.build_endurance(
        "bike", 70, "112-137 bpm", "Bike")["workoutSegments"][0]["workoutSteps"]
    assert [s["stepType"]["stepTypeKey"] for s in long_steps] == [
        "warmup", "interval", "cooldown"]
    assert sum(s["endConditionValue"] for s in long_steps) == 70 * 60

    short_steps = garmin_workout.build_endurance(
        "run", 25, "112-137 bpm", "Run")["workoutSegments"][0]["workoutSteps"]
    assert len(short_steps) == 1


def test_the_hr_range_reaches_the_working_step_only():
    """A warm-up held to the same ceiling is a warm-up you cannot do."""
    from core import garmin_workout

    steps = garmin_workout.build_endurance(
        "run", 70, "112-137 bpm", "Run")["workoutSegments"][0]["workoutSteps"]
    work = [s for s in steps if s["stepType"]["stepTypeKey"] == "interval"]
    assert work[0]["targetValueOne"] == 112 and work[0]["targetValueTwo"] == 137
    assert all(s["targetType"]["workoutTargetTypeKey"] == "no.target"
               for s in steps if s["stepType"]["stepTypeKey"] != "interval")


def test_swims_and_bricks_are_refused_rather_than_mangled():
    """A Garmin swim workout is pool length and stroke, not minutes."""
    from core import garmin_workout

    for sport in ("swim", "brick", "strength"):
        with pytest.raises(ValueError):
            garmin_workout.build_endurance(sport, 30, "112-137 bpm", "x")


def test_a_hand_edited_week_can_be_saved(healthy):
    """The Plan page's "Save my plan" wrote source="manual", which the schema
    never listed — so every attempt to save an edited week raised a
    ValidationError instead of saving it."""
    from core.schemas import PlanDay, WeekPlan

    mine = WeekPlan(
        week_plan=[PlanDay(day="Mon", sport="bike", duration_min=60)],
        source="manual", flags=["Your own plan — the rules were not applied."])
    assert mine.source == "manual"
    # It has to survive the round trip through the database as well.
    assert WeekPlan.model_validate(mine.model_dump(mode="json")).source == "manual"


def test_an_unknown_source_is_still_refused():
    import pytest
    from pydantic import ValidationError

    from core.schemas import WeekPlan

    with pytest.raises(ValidationError):
        WeekPlan(source="whatever-the-model-said")


def test_a_hand_edited_week_is_left_exactly_as_saved(healthy):
    """Everything else gets pushed back through enforce() on every page load. A
    manual week does not: the button promises what you entered is what you get,
    and its own flag records that the rules were not applied."""
    from core import planner
    from core.schemas import PlanDay, WeekPlan

    # Deliberately reckless: three sessions on one day, Z5 in a base week.
    mine = WeekPlan(
        week_plan=[PlanDay(day="Sat", sport="run", duration_min=95,
                           target_zone="Z5"),
                   PlanDay(day="Sat", sport="bike", duration_min=180,
                           target_zone="Z4"),
                   PlanDay(day="Sat", sport="swim", duration_min=60)],
        source="manual")
    after, changed = planner.reapply_rules(healthy, mine.model_dump(mode="json"),
                                          today=TODAY)
    assert changed is False
    assert [d.duration_min for d in after.week_plan] == [95, 180, 60]
    assert [d.target_zone for d in after.week_plan] == ["Z5", "Z4", "Z2"]


def test_the_endurance_floor_counts_sessions_already_done(healthy):
    """The floor is three sessions a week, not three still to come. Counting only
    the remaining days added a fourth session on a Wednesday that already had a
    swim, a ride and a run behind it — and put it on the same day as the run."""
    from core import planner

    facts = planner.build_facts(healthy, today=TODAY)
    assert len(facts.endurance_days) >= 3, facts.endurance_days

    plan = planner.plan_week(healthy, today=TODAY, use_ai=False, save=False)
    assert not any("floor" in note for note in plan.adjustments_made), \
        plan.adjustments_made


def test_nothing_is_scheduled_onto_a_day_that_already_trained(healthy):
    """Spacing used to compare proposals only against each other. Completed rows
    are merged into the plan after enforce() runs, so a ride could land on a day
    that already had a run on it — the one thing the rule exists to prevent."""
    from core import planner

    facts = planner.build_facts(healthy, today=TODAY)
    plan = planner.plan_week(healthy, today=TODAY, use_ai=False, save=False)
    proposed = [d for d in plan.week_plan
                if d.purpose != "completed" and d.duration_min > 0
                and d.sport in planner.SPACED_SPORTS]
    order = {d: i for i, d in enumerate(DAYS)}
    busy = {order[d] for d in facts.endurance_days}
    clashes = [d.day for d in proposed
               if any(abs(order[d.day] - b) <= 1 for b in busy)]
    assert not clashes, clashes


def test_the_floor_is_satisfied_by_sessions_already_completed(healthy):
    """Directly, because the spacing rule can mask this: with three endurance
    days already behind you and empty days ahead, the floor must not add more."""
    from core import planner

    facts = planner.build_facts(healthy, today=TODAY)
    facts = facts.model_copy(update={
        "endurance_days": ["Mon", "Tue", "Wed"],
        "days_remaining": ["Wed", "Thu", "Fri", "Sat", "Sun"],
    })
    envelope = planner.build_envelope(facts, healthy)
    notes: list[str] = []
    days = planner._ensure_minimum_endurance([], facts, envelope, notes)
    assert days == []
    assert notes == []

    # And the other way: nothing done yet, so the floor does add sessions.
    empty = facts.model_copy(update={"endurance_days": []})
    added = planner._ensure_minimum_endurance([], empty, envelope, [])
    assert len(added) == envelope.min_endurance_sessions


class _FakeGarmin:
    """Records what would have been deleted or unscheduled, and can refuse."""

    def __init__(self, refuse: set[str] = frozenset(),
                 calendar: list[dict] | None = None,
                 refuse_unschedule: set[str] = frozenset()):
        self.deleted: list[str] = []
        self.unscheduled: list[str] = []
        self.refuse = set(refuse)
        self.refuse_unschedule = set(refuse_unschedule)
        self.calendar = calendar or []

    def delete_workout(self, workout_id: str) -> bool:
        if workout_id in self.refuse:
            return False
        self.deleted.append(str(workout_id))
        return True

    def unschedule_workout(self, schedule_id: str) -> bool:
        if str(schedule_id) in self.refuse_unschedule:
            return False
        self.unscheduled.append(str(schedule_id))
        return True

    def scheduled_workouts(self, year: int, month: int) -> list[dict]:
        return [c for c in self.calendar
                if str(c.get("date", "")).startswith(f"{year:04d}-{month:02d}")]


def test_a_finished_session_is_taken_off_the_watch(healthy):
    """A workout is pushed most days and Garmin keeps every one of them, so
    within a fortnight pressing START offers a fortnight of history."""
    from core import sync as sync_mod

    # Wednesday is TODAY in the fixtures, and it has a completed run on it.
    healthy.set_state(f"workout_pushed_run_{TODAY.isoformat()}", "111")
    client = _FakeGarmin()
    assert sync_mod.delete_finished_workouts(healthy, client, today=TODAY) == 1
    assert client.deleted == ["111"]
    # And the bookkeeping goes with it, so the page offers to send again.
    assert healthy.get_state(f"workout_pushed_run_{TODAY.isoformat()}") is None


def test_todays_unfinished_session_stays_on_the_watch(healthy):
    from core import sync as sync_mod

    key = f"workout_pushed_swim_{TODAY.isoformat()}"
    healthy.set_state(key, "222")
    client = _FakeGarmin()
    # No swim on record today, so it has not been done.
    assert sync_mod.delete_finished_workouts(healthy, client, today=TODAY) == 0
    assert client.deleted == []
    assert healthy.get_state(key) == "222"


def test_yesterdays_session_goes_whether_or_not_it_was_done(healthy):
    """The plan has moved on, and an unused workout from yesterday is exactly
    what you do not want offered when you press START."""
    from core import sync as sync_mod

    key = f"workout_pushed_swim_{(TODAY - timedelta(days=2)).isoformat()}"
    healthy.set_state(key, "333")
    client = _FakeGarmin()
    assert sync_mod.delete_finished_workouts(healthy, client, today=TODAY) == 1
    assert client.deleted == ["333"]


def test_a_strength_push_matches_a_strength_activity(healthy):
    """The strength key carries no sport, so it must not be cleared by a run."""
    from core import sync as sync_mod

    key = f"workout_pushed_{TODAY.isoformat()}"
    healthy.set_state(key, "444")
    client = _FakeGarmin()
    removed = sync_mod.delete_finished_workouts(healthy, client, today=TODAY)
    # The fixture week logs strength on Monday and Wednesday, and TODAY is a
    # Wednesday, so this one is done.
    assert removed == 1
    assert client.deleted == ["444"]


def test_a_refused_delete_keeps_the_bookkeeping(healthy):
    """Otherwise the workout stays on the watch and nothing remembers to retry."""
    from core import sync as sync_mod

    key = f"workout_pushed_run_{TODAY.isoformat()}"
    healthy.set_state(key, "555")
    client = _FakeGarmin(refuse={"555"})
    assert sync_mod.delete_finished_workouts(healthy, client, today=TODAY) == 0
    assert healthy.get_state(key) == "555"


def test_bookkeeping_with_no_workout_id_is_just_cleared(healthy):
    from core import sync as sync_mod

    healthy.set_state(f"workout_pushed_run_{TODAY.isoformat()}", "")
    client = _FakeGarmin()
    assert sync_mod.delete_finished_workouts(healthy, client, today=TODAY) == 0
    assert client.deleted == []
    assert healthy.get_state(f"workout_pushed_run_{TODAY.isoformat()}") is None


def test_every_pushed_name_comes_back_as_the_same_exercise():
    """The round trip has to close: what the watch is sent must map back to the
    exercise it was sent for, or the progression logs the wrong movement."""
    from core import garmin_workout

    for exercise_id, (category, name) in garmin_workout.GARMIN_TARGET.items():
        exercise = strength.EXERCISES[exercise_id]
        # A hold is sent as the same Garmin name as its rep version, so the shape
        # of the set is what separates them coming back.
        hold = exercise.kind == strength.ISOMETRIC
        got = strength.map_garmin_exercise(
            category, name,
            reps=None if hold else 8,
            duration_s=40.0 if hold else 25.0)
        assert got == exercise_id, f"{name} came back as {got}, not {exercise_id}"


def test_a_bare_ambiguous_category_maps_to_nothing():
    """This is the bug that logged a pushed wall sit and split squat as goblet
    squats, and three sets of tibialis raises as calf raises: a category with no
    name used to be resolved by guessing the most common exercise in it."""
    assert strength.map_garmin_exercise("SQUAT", None) is None
    assert strength.map_garmin_exercise("CALF_RAISE", None) is None
    assert strength.map_garmin_exercise("DEADLIFT", None) is None
    assert strength.map_garmin_exercise("UNKNOWN", None) is None
    # An unmapped set is surfaced for manual assignment, which is honest.
    # Where the library has exactly one exercise in a category, the category
    # does identify it and is used.
    assert strength.map_garmin_exercise("WARM_UP", None) == "tib_raise"
    assert strength.map_garmin_exercise("LUNGE", None) == "reverse_lunge"


def test_a_calf_hold_and_a_calf_raise_are_told_apart_by_the_set(healthy):  # noqa: ARG001
    """Garmin has one single-leg calf name for both. The hold records seconds and
    no reps, which is the only thing that distinguishes them."""
    name = "SINGLE_LEG_STANDING_CALF_RAISE"
    assert strength.map_garmin_exercise("CALF_RAISE", name, reps=8,
                                        duration_s=25.0) == "calf_raise_single_leg"
    assert strength.map_garmin_exercise("CALF_RAISE", name, reps=0,
                                        duration_s=40.0) == "single_leg_calf_hold"
    # A short zero-rep set is a rest interval, not a 40-second hold.
    assert strength.map_garmin_exercise("CALF_RAISE", name, reps=0,
                                        duration_s=5.0) == "calf_raise_single_leg"


def test_a_named_set_still_maps_even_from_a_hand_logged_session():
    """The watch names sets itself when you pick an exercise on it, and those
    names are not the ones this app sends."""
    for name, expected in (
        ("BULGARIAN_SPLIT_SQUAT", "split_squat"),
        ("WEIGHTED_STANDING_CALF_RAISE", "calf_raise_straight"),
        ("SEATED_CALF_RAISE", "calf_raise_bent"),
        ("STRAIGHT_LEG_DEADLIFT", "rdl"),
        ("GOBLET_SQUAT", "goblet_squat"),
    ):
        assert strength.map_garmin_exercise(None, name) == expected, name


def test_a_rest_interval_is_not_a_missing_exercise(healthy):
    """The watch records the gap after every pushed step as a set of its own:
    active, no category, no reps. Counting them as unidentified made a clean
    session read as a dozen strays."""
    rest = {"garmin_category": "UNKNOWN", "garmin_name": None, "reps": 0.0,
            "duration_s": 45.0}
    work = {"garmin_category": "SQUAT", "garmin_name": None, "reps": 5.0,
            "duration_s": 30.0}
    hold = {"garmin_category": "SQUAT", "garmin_name": None, "reps": 0.0,
            "duration_s": 30.0}
    assert strength.looks_like_rest(rest) is True
    # Real work the watch could not categorise is not a rest, and neither is a
    # hold that reports a category.
    assert strength.looks_like_rest(work) is False
    assert strength.looks_like_rest(hold) is False


def test_assigning_a_set_by_hand_reaches_the_log(healthy):
    """The recovery path for sets the watch could not name: the athlete says what
    it was, and the work has to reach the progression."""
    from core import sync as sync_mod

    activity_id = "test-strength-1"
    healthy.upsert_activities([{
        "activity_id": activity_id, "sport": "strength", "name": "Legs",
        "start_time": f"{TODAY.isoformat()}T18:00:00",
        "start_date": TODAY.isoformat(), "duration_s": 1500.0,
        "ingested_at": f"{TODAY.isoformat()}T19:00:00"}])
    healthy.replace_exercise_sets(activity_id, [
        {"garmin_category": "SQUAT", "garmin_name": None, "reps": 0.0,
         "duration_s": 30.0, "exercise_id": None, "is_rest": 0},
        {"garmin_category": "SQUAT", "garmin_name": None, "reps": 0.0,
         "duration_s": 30.0, "exercise_id": None, "is_rest": 0},
    ])
    assert healthy.assign_exercise_sets(activity_id, "wall_sit", [0, 1]) == 2
    sets = healthy.exercise_sets(activity_id)
    assert [s["exercise_id"] for s in sets] == ["wall_sit", "wall_sit"]

    rows = strength.sets_to_log_rows(TODAY.isoformat(), activity_id, sets)
    assert rows and rows[0]["exercise_id"] == "wall_sit"
    assert rows[0]["sets"] == 2

    # A re-map must not undo it: the athlete was there, the mapping table was not.
    sync_mod.remap_exercise_sets(healthy)
    after = healthy.exercise_sets(activity_id)
    assert [s["exercise_id"] for s in after] == ["wall_sit", "wall_sit"]

    # Nor may a re-import from the watch, which rewrites every row.
    healthy.replace_exercise_sets(activity_id, [
        {"garmin_category": "SQUAT", "garmin_name": None, "reps": 0.0,
         "duration_s": 30.0, "exercise_id": None, "is_rest": 0},
        {"garmin_category": "SQUAT", "garmin_name": None, "reps": 0.0,
         "duration_s": 30.0, "exercise_id": None, "is_rest": 0},
    ])
    reimported = healthy.exercise_sets(activity_id)
    assert [s["exercise_id"] for s in reimported] == ["wall_sit", "wall_sit"]


def test_a_session_that_has_happened_stops_being_planned(healthy):
    """A plan is stored data and the activities are what happened. Merging the
    completed sessions only when the plan was built meant training on Monday
    evening left Monday's session sitting there as something still to do."""
    from core import planner
    from core.schemas import PlanDay, WeekPlan

    facts = planner.build_facts(healthy, today=TODAY)
    # The fixture week logs a swim on Monday and a strength session on Monday.
    plan = WeekPlan(week_plan=[
        PlanDay(day="Mon", sport="swim", duration_min=45, purpose="aerobic base"),
        PlanDay(day="Mon", sport="strength", duration_min=28, purpose="legs"),
        PlanDay(day="Sun", sport="run", duration_min=70, purpose="long run"),
    ], source="rules")
    marked, changed = planner.refresh_completions(plan, facts, healthy)
    assert changed is True
    monday = [d for d in marked.week_plan if d.day == "Mon"]
    assert monday, "Monday's work should still appear, as completed"
    assert all(d.purpose == "completed" for d in monday), monday
    # Sunday has not happened yet, so it stays planned.
    sunday = [d for d in marked.week_plan if d.day == "Sun"]
    assert [d.purpose for d in sunday] == ["long run"]


def test_a_different_sport_on_the_same_day_is_still_planned(healthy):
    """Riding on Monday does not mean Monday's swim happened."""
    from core import planner
    from core.schemas import PlanDay, WeekPlan

    facts = planner.build_facts(healthy, today=TODAY)
    plan = WeekPlan(week_plan=[
        PlanDay(day="Tue", sport="run", duration_min=40, purpose="aerobic base"),
    ], source="rules")
    marked, _ = planner.refresh_completions(plan, facts, healthy)
    still_planned = [d for d in marked.week_plan
                     if d.day == "Tue" and d.sport == "run"
                     and d.purpose != "completed"]
    # The fixture logs a ride on Tuesday, not a run.
    assert still_planned


def test_completed_rows_are_rebuilt_rather_than_duplicated(healthy):
    """Running twice must not leave two copies of every finished session."""
    from core import planner
    from core.schemas import WeekPlan

    facts = planner.build_facts(healthy, today=TODAY)
    once, _ = planner.refresh_completions(WeekPlan(source="rules"), facts, healthy)
    twice, changed = planner.refresh_completions(once, facts, healthy)
    assert len(twice.week_plan) == len(once.week_plan)
    assert changed is False


def test_a_brick_needs_both_halves_before_it_counts_as_done(healthy):
    from core import planner
    from core.schemas import PlanDay, WeekPlan

    facts = planner.build_facts(healthy, today=TODAY)
    # Tuesday has a ride only, so a brick planned for Tuesday is not done.
    plan = WeekPlan(week_plan=[
        PlanDay(day="Tue", sport="brick", duration_min=90, purpose="brick"),
    ], source="rules")
    marked, _ = planner.refresh_completions(plan, facts, healthy)
    assert [d.sport for d in marked.week_plan if d.purpose == "brick"] == ["brick"]


def _calendar_entry(day, workout_id="900", schedule_id="90",
                    title="Legs A · Aerobic Engine",
                    sport="strength_training", protected=False):
    return {"schedule_id": schedule_id, "workout_id": workout_id, "title": title,
            "date": day.isoformat(), "sport": sport, "protected": protected}


def test_a_finished_session_leaves_the_training_calendar_too(healthy):
    """Deleting the workout is only half of it. `schedule_workout` puts an entry
    on the calendar, and that entry is the copy you see in Garmin Connect under
    the plan — it survives the workout being deleted."""
    from core import sync as sync_mod

    client = _FakeGarmin(calendar=[_calendar_entry(TODAY)])
    removed = sync_mod.delete_finished_workouts(healthy, client, today=TODAY)
    assert removed == 1
    assert client.unscheduled == ["90"]
    assert client.deleted == ["900"]


def test_a_workout_the_athlete_built_is_never_touched(healthy):
    """A tidy-up that deletes someone's own workouts is not a tidy-up."""
    from core import sync as sync_mod

    client = _FakeGarmin(calendar=[
        _calendar_entry(TODAY, workout_id="800", schedule_id="80",
                        title="Benchmark Run", sport="running")])
    assert sync_mod.delete_finished_workouts(healthy, client, today=TODAY) == 0
    assert client.deleted == [] and client.unscheduled == []


def test_an_orphan_push_is_still_cleared(healthy):
    """The bookkeeping only knows about pushes it saw. The calendar knows about
    all of them, which is why the sweep goes over the calendar."""
    from core import sync as sync_mod

    assert healthy.states_with_prefix("workout_pushed_") == {}
    client = _FakeGarmin(calendar=[
        _calendar_entry(TODAY - timedelta(days=3), workout_id="700",
                        schedule_id="70")])
    assert sync_mod.delete_finished_workouts(healthy, client, today=TODAY) == 1
    assert client.deleted == ["700"]


def test_a_protected_calendar_entry_is_left_alone(healthy):
    """Garmin marks entries belonging to one of its own plans as protected."""
    from core import sync as sync_mod

    client = _FakeGarmin(calendar=[
        _calendar_entry(TODAY - timedelta(days=3), protected=True)])
    assert sync_mod.delete_finished_workouts(healthy, client, today=TODAY) == 0
    assert client.deleted == []


def test_the_workout_stays_if_it_cannot_be_unscheduled(healthy):
    """Otherwise the calendar keeps an entry pointing at a workout that is gone."""
    from core import sync as sync_mod

    client = _FakeGarmin(calendar=[_calendar_entry(TODAY)],
                         refuse_unschedule={"90"})
    assert sync_mod.delete_finished_workouts(healthy, client, today=TODAY) == 0
    assert client.deleted == []


def test_a_calendar_entry_for_an_unfinished_day_stays(healthy):
    from core import sync as sync_mod

    client = _FakeGarmin(calendar=[
        _calendar_entry(TODAY, title="Swim 40m · Aerobic Engine",
                        sport="swimming")])
    # No swim on record today, and the day is not stale.
    assert sync_mod.delete_finished_workouts(healthy, client, today=TODAY) == 0
    assert client.deleted == []


def test_a_hand_edited_week_is_filled_in_without_being_overruled(healthy):
    """Manual edits outrank the rules — but "what I typed" and "a complete plan"
    are not the same thing. The editor has columns for a day, a sport, minutes
    and a zone; it has none for the exercises in a leg session or the bpm range a
    zone means for this athlete, so those arrived empty."""
    from core import planner
    from core.schemas import PlanDay, WeekPlan

    # The synthetic fixture has no zone rows, and without them there is no bpm
    # range to stamp — which is correct behaviour, so give it some.
    activity = healthy.activities()[-1]["activity_id"]
    healthy.upsert_zones([
        {"activity_id": activity, "zone_number": z, "secs_in_zone": 60.0,
         "zone_low_bpm": low}
        for z, low in ((1, 93), (2, 112), (3, 138), (4, 149), (5, 167))
    ])
    typed = WeekPlan(week_plan=[
        PlanDay(day="Sat", sport="bike", duration_min=95, target_zone="Z2",
                purpose="", why=""),
        PlanDay(day="Sun", sport="strength", duration_min=30, target_zone="",
                purpose="", why=""),
    ], source="manual")
    filled, changed = planner.enrich_manual(typed, healthy, today=TODAY)
    assert changed is True

    ride = next(d for d in filled.week_plan if d.sport == "bike")
    legs = next(d for d in filled.week_plan if d.sport == "strength")
    # Nothing entered is touched.
    assert (ride.day, ride.duration_min, ride.target_zone) == ("Sat", 95, "Z2")
    assert (legs.day, legs.duration_min) == ("Sun", 30)
    # What was missing is filled: a real range, real exercises, a reason.
    assert "bpm" in ride.target_hr
    assert legs.exercise_ids
    assert all(e in strength.EXERCISES for e in legs.exercise_ids)
    assert legs.target_zone == "n/a"
    assert ride.purpose and legs.purpose and ride.why and legs.why


def test_filling_in_is_idempotent(healthy):
    """It runs on every page load."""
    from core import planner
    from core.schemas import PlanDay, WeekPlan

    typed = WeekPlan(week_plan=[
        PlanDay(day="Sun", sport="strength", duration_min=30)], source="manual")
    once, _ = planner.enrich_manual(typed, healthy, today=TODAY)
    twice, changed = planner.enrich_manual(once, healthy, today=TODAY)
    assert changed is False
    assert twice.week_plan[0].exercise_ids == once.week_plan[0].exercise_ids


def test_a_finished_day_does_not_take_the_next_strength_session(healthy):
    """A completed Monday used to be handed the next session in the cycle, which
    pushed the day still to come one further along and prescribed the wrong
    session for it."""
    from core import planner
    from core.schemas import PlanDay, WeekPlan

    facts = planner.build_facts(healthy, today=TODAY)
    plan = WeekPlan(week_plan=[
        PlanDay(day="Mon", sport="strength", duration_min=28, purpose="legs"),
        PlanDay(day="Fri", sport="strength", duration_min=28, purpose="legs"),
    ], source="manual")
    marked, _ = planner.refresh_completions(plan, facts, healthy)
    filled, _ = planner.enrich_manual(marked, healthy, today=TODAY)

    monday = [d for d in filled.week_plan if d.day == "Mon"]
    friday = next(d for d in filled.week_plan if d.day == "Fri")
    # Monday is on record as done, so it keeps its completed row and no
    # prescription; Friday gets the next session in the cycle.
    assert all(d.purpose == "completed" for d in monday)
    assert friday.exercise_ids
    logged_days = len({str(r["day"]) for r in healthy.strength_log()})
    assert friday.exercise_ids == [
        p.exercise_id for p in strength.build_session(
            healthy.strength_log(), session_index=logged_days, intensity=1.0)]


def test_a_day_already_gone_is_not_prescribed_for(healthy):
    """There is nothing to hand someone for a Monday when it is Wednesday."""
    from core import planner
    from core.schemas import PlanDay, WeekPlan

    plan = WeekPlan(week_plan=[
        PlanDay(day="Tue", sport="strength", duration_min=28, purpose="legs"),
    ], source="manual")
    filled, _ = planner.enrich_manual(plan, healthy, today=TODAY)  # a Wednesday
    assert filled.week_plan[0].exercise_ids == []


def test_a_day_already_trained_is_not_planned_again(healthy):
    """Re-planning used to hand a finished Monday a different session. The
    athlete has done the work; the plan has no business rewriting it."""
    from core import planner

    facts = planner.build_facts(healthy, today=TODAY)
    assert facts.trained_days, "the fixture week has sessions behind it"
    assert not set(facts.days_remaining) & set(facts.trained_days)
    # And today is still available when nothing has been logged on it.
    quiet = planner.build_facts(healthy, today=TODAY + timedelta(days=1))
    assert DAYS[(TODAY + timedelta(days=1)).weekday()] in quiet.days_remaining \
        or DAYS[(TODAY + timedelta(days=1)).weekday()] in quiet.trained_days


def test_replanning_leaves_finished_sessions_alone(healthy):
    from core import planner

    facts = planner.build_facts(healthy, today=TODAY)
    plan = planner.plan_week(healthy, today=TODAY, use_ai=False, save=False)
    planned_days = {d.day for d in plan.week_plan
                    if d.purpose != "completed" and d.duration_min > 0}
    assert not planned_days & set(facts.trained_days), (
        f"planned on days already trained: "
        f"{sorted(planned_days & set(facts.trained_days))}")
