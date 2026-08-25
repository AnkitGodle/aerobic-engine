"""The database's zone buckets must match the Python ones, exactly.

The app used to pull every stored heart-rate sample into the process to work out
how many minutes went into each zone — a few thousand dicts to produce five
numbers, and the single heaviest thing on the page. The same arithmetic now runs
in SQL. That is only worth doing if the answers are identical, so these tests
compare the two implementations rather than asserting numbers of their own: the
stream version stays as the reference.
"""

from __future__ import annotations

from datetime import date, timedelta

from core.analysis import (
    polarisation_from_streams,
    weekly_zone_minutes_from_rows,
    weekly_zone_minutes_from_streams,
    zone_distribution_from_streams,
)

BOUNDS = {1: (90, 120), 2: (121, 135), 3: (136, 150), 4: (151, 165), 5: (166, None)}
TODAY = date(2026, 8, 19)


def _streams(store):
    acts = store.activities()
    return {a["activity_id"]: store.stream(a["activity_id"]) for a in acts}, acts


def test_totals_match_the_stream_version(healthy):
    streams, acts = _streams(healthy)
    assert healthy.sample_minutes(BOUNDS) == \
        zone_distribution_from_streams(streams, acts, BOUNDS)


def test_a_date_filter_matches(healthy):
    streams, acts = _streams(healthy)
    since = TODAY - timedelta(days=28)
    assert healthy.sample_minutes(BOUNDS, since=since) == \
        zone_distribution_from_streams(streams, acts, BOUNDS, since=since)


def test_a_sport_filter_matches(healthy):
    streams, acts = _streams(healthy)
    assert healthy.sample_minutes(BOUNDS, sports=["bike"]) == \
        zone_distribution_from_streams(streams, acts, BOUNDS, sport="bike")


def test_one_activity_matches(healthy):
    with_stream = [a for a in healthy.activities()
                   if healthy.stream(a["activity_id"])]
    assert with_stream, "the fixture should store some streams"
    one = with_stream[0]
    assert healthy.sample_minutes(BOUNDS, activity_id=one["activity_id"]) == \
        zone_distribution_from_streams(
            {one["activity_id"]: healthy.stream(one["activity_id"])},
            [one], BOUNDS)


def test_the_weekly_table_matches(healthy):
    streams, acts = _streams(healthy)
    rows = healthy.sample_minutes_by_date(BOUNDS)
    assert weekly_zone_minutes_from_rows(rows, weeks=8, as_of=TODAY) == \
        weekly_zone_minutes_from_streams(streams, acts, BOUNDS, weeks=8,
                                         as_of=TODAY)


def test_the_weekly_table_respects_the_window(healthy):
    rows = healthy.sample_minutes_by_date(BOUNDS)
    weeks = weekly_zone_minutes_from_rows(rows, weeks=3, as_of=TODAY)
    this_week = TODAY - timedelta(days=TODAY.weekday())
    assert len(weeks) <= 3
    # Only weeks inside the window, and none of them in the future.
    assert all(this_week - timedelta(weeks=2) <= w["week_start"] <= this_week
               for w in weeks)
    assert weeks == sorted(weeks, key=lambda w: w["week_start"])


def test_the_easy_hard_split_matches(healthy):
    streams, acts = _streams(healthy)
    sql = healthy.sample_split(135, 151, since=TODAY - timedelta(days=28))
    ref = polarisation_from_streams(streams, acts, ceiling=135, hard_floor=151,
                                   since=TODAY - timedelta(days=28))
    for key in ("easy", "moderate", "hard", "samples"):
        assert sql[key] == ref[key], key


def test_no_bounds_means_no_answer(healthy):
    assert healthy.sample_minutes({}) == {}
    assert healthy.sample_minutes_by_date({}) == []


def test_nothing_in_range_is_zeroed_not_missing(healthy):
    empty = healthy.sample_minutes(BOUNDS, since=TODAY + timedelta(days=7))
    assert empty == {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}
    blank = healthy.sample_split(135, 151, since=TODAY + timedelta(days=7))
    assert blank["samples"] == 0 and blank["easy"] == 0.0


def test_a_multisport_parent_is_not_counted_twice(healthy):
    """A parent row's stream would double-count its own legs, so it is excluded."""
    kid = next(a for a in healthy.activities() if healthy.stream(a["activity_id"]))
    healthy.upsert_activities([{
        "activity_id": "parent-1", "sport": "brick", "garmin_type": "multi_sport",
        "name": "brick", "start_time": f"{kid['start_date']}T06:00:00",
        "start_date": kid["start_date"], "duration_s": 3600,
        "avg_hr": 140, "is_multisport_parent": 1,
        "ingested_at": f"{kid['start_date']}T12:00:00",
    }])
    healthy.replace_stream("parent-1", [{"t_s": i * 30, "hr": 145}
                                        for i in range(60)])
    counted = healthy.sample_minutes(BOUNDS)
    # `activities()` already hides parents, so the reference built from it is the
    # answer with the parent excluded — which is what the SQL must agree with.
    without_parent = zone_distribution_from_streams(
        {a["activity_id"]: healthy.stream(a["activity_id"])
         for a in healthy.activities()},
        healthy.activities(), BOUNDS)
    assert counted == without_parent
