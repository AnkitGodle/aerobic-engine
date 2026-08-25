"""A race, a date, and the phases between here and it."""

from __future__ import annotations

from datetime import date, timedelta

from conftest import TODAY
from core import goal as goal_mod
from core import planner


def _goal(weeks_away: int, km: float = 21.0) -> goal_mod.Goal:
    return goal_mod.Goal(event="Pune Half", day=TODAY + timedelta(weeks=weeks_away),
                        sport="run", distance_km=km)


def test_no_goal_is_base_forever():
    """The behaviour before goals existed, and the right answer for someone
    building an engine with no race in mind."""
    assert goal_mod.NO_GOAL.set is False
    assert goal_mod.NO_GOAL.phase(TODAY) == goal_mod.BASE
    assert goal_mod.NO_GOAL.weeks_to_go(TODAY) is None
    assert "No race set" in goal_mod.describe(goal_mod.NO_GOAL, TODAY)


def test_the_phases_run_backwards_from_the_race():
    far, build, peak, taper = _goal(20), _goal(10), _goal(4), _goal(1)
    assert far.phase(TODAY) == goal_mod.BASE
    assert build.phase(TODAY) == goal_mod.BUILD
    assert peak.phase(TODAY) == goal_mod.PEAK
    assert taper.phase(TODAY) == goal_mod.TAPER


def test_the_taper_is_as_long_as_the_distance_needs():
    """Three weeks for a marathon, one for a 5K: the taper exists to absorb the
    training, and there is less of it to absorb."""
    assert _goal(2, km=42.2).taper_weeks() == 3
    assert _goal(2, km=21.1).taper_weeks() == 2
    assert _goal(2, km=10.0).taper_weeks() == 1
    assert _goal(2, km=5.0).taper_weeks() == 1
    # So the same two-weeks-out date is a taper for a marathon and a peak for a 5K.
    assert _goal(2, km=42.2).phase(TODAY) == goal_mod.TAPER
    assert _goal(2, km=5.0).phase(TODAY) == goal_mod.PEAK


def test_a_race_that_has_passed_goes_back_to_base():
    gone = goal_mod.Goal(event="Last month's 10K", day=TODAY - timedelta(days=30),
                        distance_km=10.0)
    assert gone.phase(TODAY) == goal_mod.BASE
    assert "been and gone" in goal_mod.describe(gone, TODAY)


def test_the_timeline_says_where_you_are():
    """Only the phases that are left, and the first one starts now rather than
    at some week that has already gone."""
    rows = _goal(10).timeline(TODAY)
    assert [r["phase"] for r in rows] == ["build", "peak", "taper"]
    assert rows[0]["from_weeks"] == 10
    current = [r for r in rows if r["current"]]
    assert len(current) == 1 and current[0]["phase"] == goal_mod.BUILD


def test_a_long_run_up_still_has_a_base_phase():
    rows = _goal(30).timeline(TODAY)
    assert [r["phase"] for r in rows] == ["base", "build", "peak", "taper"]
    assert rows[0]["from_weeks"] == 30 and rows[0]["current"] is True


def test_a_goal_round_trips_through_the_database(healthy):
    saved = goal_mod.save(healthy, "Pune Half", TODAY + timedelta(weeks=9),
                          sport="run", distance_km=21.1)
    again = goal_mod.load(healthy)
    assert again == saved
    assert again.phase(TODAY) == goal_mod.BUILD
    assert goal_mod.clear(healthy).set is False
    assert goal_mod.load(healthy).set is False


def test_the_envelope_follows_the_phase(healthy):
    """Build allows more hard work than base; a taper takes the volume away."""
    facts = planner.build_facts(healthy, today=TODAY)

    goal_mod.clear(healthy)
    base = planner.build_envelope(facts, healthy)
    goal_mod.save(healthy, "Race", TODAY + timedelta(weeks=10), distance_km=21.1)
    build = planner.build_envelope(facts, healthy)
    goal_mod.save(healthy, "Race", TODAY + timedelta(weeks=1), distance_km=21.1)
    taper = planner.build_envelope(facts, healthy)

    assert (base.phase, build.phase, taper.phase) == ("base", "build", "taper")
    assert build.max_quality_sessions >= base.max_quality_sessions
    assert taper.max_week_minutes < base.max_week_minutes * 0.7
    assert taper.min_rest_days > base.min_rest_days
    assert build.weeks_to_race == 10 and build.race_name == "Race"
    goal_mod.clear(healthy)


def test_a_deload_still_outranks_a_peak(wrecked):
    """The whole design rests on the backstop being unconditional."""
    goal_mod.save(wrecked, "Race", TODAY + timedelta(weeks=3), distance_km=21.1)
    facts = planner.build_facts(wrecked, today=TODAY)
    env = planner.build_envelope(facts, wrecked)
    assert env.phase == goal_mod.PEAK
    assert env.deload is True
    assert env.max_quality_sessions == 0, "a peak may not restore hard sessions"
    goal_mod.clear(wrecked)


def test_the_taper_keeps_its_sharpness(healthy):
    """A taper is not a deload: the volume goes and the quality stays."""
    goal_mod.save(healthy, "Race", TODAY + timedelta(weeks=1), distance_km=21.1)
    env = planner.build_envelope(planner.build_facts(healthy, today=TODAY), healthy)
    assert env.max_quality_sessions >= 1
    goal_mod.clear(healthy)
