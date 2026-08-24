"""The editable rules layer: stored, clamped, and actually obeyed by the planner."""

from __future__ import annotations

from conftest import TODAY

from core import planner, rules
from core.schemas import DAYS


def test_defaults_are_the_base_phase_design():
    r = rules.DEFAULTS
    assert (r.min_endurance_sessions, r.strength_sessions, r.min_rest_days) == (3, 2, 1)
    assert r.progression_cap_pct == 10.0
    assert r.block_weeks == 4
    assert r.space_endurance is True


def test_an_unset_store_reads_as_the_defaults(healthy):
    assert rules.load(healthy) == rules.DEFAULTS


def test_saving_then_loading_round_trips(healthy):
    saved = rules.save(healthy, {"min_endurance_sessions": 4,
                                 "strength_sessions": 1,
                                 "space_endurance": False,
                                 "progression_cap_pct": 7})
    assert saved.min_endurance_sessions == 4
    assert saved.space_endurance is False
    again = rules.load(healthy)
    assert again == saved
    assert again.progression_cap_pct == 7.0


def test_values_are_clamped_not_trusted(healthy):
    """A settings page is a way to talk yourself into a 40% jump. The bounds are
    the point of having them in code rather than in a text box."""
    saved = rules.save(healthy, {"min_endurance_sessions": 99,
                                 "progression_cap_pct": -5,
                                 "deload_cut_pct": 500})
    assert saved.min_endurance_sessions == 6      # BOUNDS caps it
    assert saved.progression_cap_pct == 0.0
    assert saved.deload_cut_pct == 60.0


def test_junk_falls_back_to_the_default(healthy):
    healthy.set_state("rule_min_endurance_sessions", "three")
    assert rules.load(healthy).min_endurance_sessions == 3


def test_reset_forgets_every_override(healthy):
    rules.save(healthy, {"min_endurance_sessions": 5})
    assert rules.load(healthy).min_endurance_sessions == 5
    assert rules.reset(healthy) == rules.DEFAULTS
    assert rules.load(healthy) == rules.DEFAULTS


def test_the_envelope_is_built_from_the_saved_rules(healthy):
    rules.save(healthy, {"min_endurance_sessions": 4, "strength_sessions": 0,
                         "min_rest_days": 2, "progression_cap_pct": 5,
                         "brick_every_weeks": 0})
    facts = planner.build_facts(healthy, today=TODAY)
    env = planner.build_envelope(facts, healthy)
    assert env.min_endurance_sessions == 4
    assert env.strength_sessions == 0
    assert env.min_rest_days >= 2
    assert env.progression_cap_pct == 5.0
    assert env.brick_required is False


def test_a_raised_floor_changes_the_plan(healthy):
    """Not just the envelope — the week that comes out of it."""
    rules.save(healthy, {"min_endurance_sessions": 4})
    plan = planner.plan_week(healthy, today=TODAY, use_ai=False, save=False)
    endurance = [d for d in plan.week_plan
                 if d.sport in planner.SPACED_SPORTS and d.duration_min > 0]
    assert len(endurance) >= 4


def test_turning_spacing_off_stops_the_planner_moving_sessions(healthy):
    rules.save(healthy, {"space_endurance": False})
    plan = planner.plan_week(healthy, today=TODAY, use_ai=False, save=False)
    assert not any("clear day" in note for note in plan.adjustments_made)


def test_spacing_on_is_still_enforced(healthy):
    """The other half of the previous test: the rule has to bite by default.

    Only the days still to come are checked. Sessions already completed are
    immovable — two of them back to back is history, not a planning error.
    """
    rules.save(healthy, {"space_endurance": True})
    plan = planner.plan_week(healthy, today=TODAY, use_ai=False, save=False)
    order = {d: i for i, d in enumerate(DAYS)}
    planned = sorted(order[d.day] for d in plan.week_plan
                     if d.sport in planner.SPACED_SPORTS and d.duration_min > 0
                     and d.purpose != "completed")
    assert all(b - a > 1 for a, b in zip(planned, planned[1:])), planned
    done = {order[d.day] for d in plan.week_plan
            if d.sport in planner.SPACED_SPORTS and d.purpose == "completed"}
    assert not [i for i in planned if any(abs(i - j) <= 1 for j in done)]


def test_no_strength_means_no_strength_days(healthy):
    rules.save(healthy, {"strength_sessions": 0})
    plan = planner.plan_week(healthy, today=TODAY, use_ai=False, save=False)
    # Completed sessions stay on the week whatever the rule says: they happened.
    assert not [d for d in plan.week_plan
                if d.sport == "strength" and d.duration_min > 0
                and d.purpose != "completed"]


def test_the_summary_reads_as_sentences():
    lines = list(rules.summary(rules.DEFAULTS))
    assert any("3 endurance sessions" in line for line in lines)
    assert any("10% a week" in line for line in lines)


def test_what_the_athlete_changed_is_reportable():
    changed = rules.changed_from_default(
        rules.Rules(min_endurance_sessions=5, strength_sessions=2))
    assert changed == {"min_endurance_sessions": (5, 3)}
