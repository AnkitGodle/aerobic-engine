"""The rules must beat the AI, every time.

These tests exist because the whole design rests on one claim: a sycophantic or
broken model cannot produce a dangerous week. `enforce()` is fed a deliberately
reckless plan and the output is checked against every constraint.
"""

from __future__ import annotations

import json

from conftest import TODAY

from core import ai, planner, strength
from core.schemas import DAYS, Checkin, WeekPlan

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
