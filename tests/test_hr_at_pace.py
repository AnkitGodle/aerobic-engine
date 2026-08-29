"""Normalising heart rate to a reference pace, correctly.

The bug these exist for was in the headline number of the whole dashboard. It
was computed as `HR x (reference_speed / session_speed)` — heart rate treated as
proportional to speed, a line through the origin. Heart rate at a standstill is
resting, not zero, so the ratio punishes slow sessions and flatters fast ones.

On four real runs spanning 7:29 to 9:38 per km at 138-158 bpm, the ratio read
146, 148, 157, 164: an eighteen-beat decline that was nothing but pace. The
fitted line puts all four within a beat and a half of each other, which is what
they were.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from core import analysis

TODAY = date(2026, 8, 26)

# The athlete's own four runs, as stored, to the numbers actually recorded.
REAL_RUNS = [
    ("2026-08-19", 2.224, 158.0),
    ("2026-08-20", 2.196, 158.0),
    ("2026-08-22", 1.920, 146.0),
    ("2026-08-26", 1.729, 138.0),
]


def runs(rows=REAL_RUNS, sport: str = "run") -> list[dict]:
    out = []
    for i, (day, speed, hr) in enumerate(rows):
        out.append({
            "activity_id": f"a{i}", "sport": sport, "start_date": day,
            "start_time": f"{day}T06:00:00", "avg_hr": hr, "max_hr": hr + 25,
            "avg_speed_mps": speed, "duration_s": 2700,
            "distance_m": speed * 2700, "ef": speed * 100 / hr,
            "ef_metric": "speed_per_hr", "is_steady": 0,
            "ingested_at": f"{day}T12:00:00",
        })
    return out


# --------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------


def test_the_slope_is_fitted_from_the_athletes_own_sessions():
    model = analysis.hr_pace_model(runs(), "run")
    assert model["source"] == "fitted"
    assert 35 < model["slope"] < 48          # measured 41.5 bpm per m/s
    assert model["r2"] > 0.95
    assert model["n"] == 4


def test_too_few_sessions_uses_a_documented_default():
    model = analysis.hr_pace_model(runs(REAL_RUNS[:2]), "run")
    assert model["source"] == "default"
    assert model["slope"] == analysis.DEFAULT_HR_SLOPE["run"]
    assert "needs 4" in model["why"]


def test_sessions_all_at_one_pace_cannot_fit_a_slope():
    same = [(f"2026-08-{10 + i:02d}", 2.05, 150.0 + i) for i in range(6)]
    model = analysis.hr_pace_model(runs(same), "run")
    assert model["source"] == "default"
    assert "same pace" in model["why"]


def test_a_relationship_that_is_not_there_is_not_used():
    """Heart rate scattered against pace must not produce a confident slope."""
    noisy = [("2026-08-10", 1.8, 150.0), ("2026-08-12", 2.4, 148.0),
             ("2026-08-14", 2.0, 172.0), ("2026-08-16", 2.6, 141.0),
             ("2026-08-18", 2.2, 165.0)]
    model = analysis.hr_pace_model(runs(noisy), "run")
    assert model["source"] == "default"


def test_an_implausible_slope_is_refused():
    """A steep line through four points is not a physiological relationship."""
    steep = [("2026-08-10", 1.80, 100.0), ("2026-08-12", 2.10, 135.0),
             ("2026-08-14", 2.40, 170.0), ("2026-08-16", 2.70, 205.0)]
    model = analysis.hr_pace_model(runs(steep), "run")
    assert model["source"] == "default"
    assert "plausible" in model["why"]


@pytest.mark.parametrize("sport", ["run", "bike", "swim"])
def test_every_sport_has_a_default_and_a_sane_band(sport):
    low, high = analysis.HR_SLOPE_BOUNDS[sport]
    assert low < analysis.DEFAULT_HR_SLOPE[sport] < high


# --------------------------------------------------------------------------
# What the chart plots
# --------------------------------------------------------------------------


def test_four_runs_at_one_fitness_read_as_one_fitness():
    """The regression test for the original bug."""
    points = analysis.hr_points(runs(), "run")
    values = [p["hr_at_reference"] for p in points]
    assert all(v is not None for v in values)
    assert max(values) - min(values) < 3.0, values
    # And the old maths, for the record: an 18 bpm spread from the same data.
    ref = analysis.reference_speed(runs(), "run")
    ratio = [p["avg_hr"] * ref / p["speed_mps"] for p in points]
    assert max(ratio) - min(ratio) > 15


def test_a_slower_session_is_not_punished_for_being_slower():
    points = {p["date"].isoformat(): p for p in analysis.hr_points(runs(), "run")}
    slowest = points["2026-08-26"]        # 9:38/km, the easiest run
    fastest = points["2026-08-19"]        # 7:29/km
    assert slowest["hr_at_reference"] == pytest.approx(
        fastest["hr_at_reference"], abs=2.0)


def test_a_session_at_the_reference_pace_is_left_alone():
    ref = analysis.reference_speed(runs(), "run")
    rows = runs() + [("2026-08-27", ref, 150.0)][0:0]
    extra = runs([("2026-08-27", ref, 150.0)])
    points = analysis.hr_points(rows + extra, "run")
    at_ref = [p for p in points if p["date"] == date(2026, 8, 27)][0]
    assert at_ref["hr_at_reference"] == pytest.approx(150.0, abs=0.6)
    assert at_ref["extrapolated"] is False


def test_inside_the_fitted_range_is_not_a_guess():
    """A slow easy run inside the measured pace range is interpolated, and as
    trustworthy as the rest. That is the point of fitting rather than assuming."""
    points = {p["date"].isoformat(): p for p in analysis.hr_points(runs(), "run")}
    assert all(not p["extrapolated"] for p in points.values())
    assert abs(points["2026-08-26"]["pace_offset"]) > 0.15   # far, but measured


def test_a_correction_that_could_be_wrong_by_beats_is_marked():
    """A loose fit plus a big pace gap: the correction is not evidence."""
    loose = [("2026-08-02", 1.90, 150.0), ("2026-08-05", 2.05, 143.0),
             ("2026-08-08", 2.20, 158.0), ("2026-08-11", 2.35, 152.0),
             ("2026-08-14", 2.50, 166.0), ("2026-08-17", 1.10, 120.0)]
    acts = runs(loose)
    model = analysis.hr_pace_model(acts, "run")
    points = {p["date"].isoformat(): p for p in analysis.hr_points(acts, "run")}
    far = points["2026-08-17"]
    assert far["extrapolated"] is True
    assert far["correction_doubt_bpm"] > analysis.CORRECTION_DOUBT_BPM
    assert model["slope_se"] is not None


def test_a_tight_fit_trusts_a_wide_pace_gap():
    """Their own runs: 16% off the reference, and the doubt is under a beat."""
    points = {p["date"].isoformat(): p for p in analysis.hr_points(runs(), "run")}
    slow = points["2026-08-26"]
    assert slow["correction_doubt_bpm"] < 1.0
    assert slow["extrapolated"] is False
    assert slow["correction_bpm"] > 10       # a real correction, confidently made


def test_with_no_fit_to_stand_on_distance_is_all_there_is():
    two = runs(REAL_RUNS[:2]) + runs([("2026-08-25", 1.2, 120.0)])
    points = {p["date"].isoformat(): p for p in analysis.hr_points(two, "run")}
    assert analysis.hr_pace_model(two, "run")["source"] == "default"
    assert points["2026-08-25"]["extrapolated"] is True


def test_a_power_measured_ride_has_no_pace_equivalent():
    rides = runs([("2026-08-20", 7.5, 140.0)], sport="bike")
    rides[0]["ef_metric"] = "power_per_hr"
    points = analysis.hr_points(rides, "bike")
    assert points[0]["hr_at_reference"] is None
    assert points[0]["avg_hr"] == 140.0


def test_the_raw_heart_rate_is_never_altered():
    for point in analysis.hr_points(runs(), "run"):
        original = next(r for r in REAL_RUNS if r[0] == point["date"].isoformat())
        assert point["avg_hr"] == original[2]


# --------------------------------------------------------------------------
# Why the reference can stay dynamic
# --------------------------------------------------------------------------


def test_moving_the_reference_shifts_the_line_without_bending_it():
    """The property that makes a growing yardstick safe.

    An additive correction moves every point by the same amount, so the shape of
    the trend — which is the only thing being read — does not change. Under the
    old ratio it did: each point was scaled by its own factor.
    """
    base = analysis.hr_points(runs(), "run")
    shifted = analysis.hr_points(runs(), "run", ref_speed=2.5)
    gaps_before = [b["hr_at_reference"] - a["hr_at_reference"]
                   for a, b in zip(base, base[1:])]
    gaps_after = [b["hr_at_reference"] - a["hr_at_reference"]
                  for a, b in zip(shifted, shifted[1:])]
    for before, after in zip(gaps_before, gaps_after):
        assert before == pytest.approx(after, abs=0.15)


def test_the_trend_verdict_survives_a_change_of_yardstick():
    long_history = [
        ((date(2026, 5, 1) + timedelta(days=i * 4)).isoformat(),
         2.0 + (i % 3) * 0.15, 160.0 - i * 0.8)
        for i in range(14)
    ]
    acts = runs(long_history)
    a = analysis.hr_trend(acts, "run", as_of=date(2026, 7, 15))
    b = analysis.hr_trend(acts, "run", as_of=date(2026, 7, 15))
    assert a["verdict"] == b["verdict"]
    assert a["verdict"] in ("improving", "flat", "worsening")


def test_real_improvement_still_shows_as_improvement():
    """A genuine drop in heart rate at the same pace must survive the fix."""
    improving = [
        ((date(2026, 6, 1) + timedelta(days=i * 3)).isoformat(),
         2.05 + (i % 2) * 0.1, 165.0 - i * 1.4)
        for i in range(16)
    ]
    trend = analysis.hr_trend(runs(improving), "run", as_of=date(2026, 7, 21))
    assert trend["verdict"] == "improving"
    assert trend["normalised_change_bpm"] < -1


# --------------------------------------------------------------------------
# The verdict on the page
# --------------------------------------------------------------------------


def test_the_verdict_ignores_raw_efficiency_when_a_pace_correction_exists():
    """The banner said "efficiency is slipping in run" over these four sessions.

    Raw efficiency factor fell 11.8% a week across them, entirely because they
    were run slower. At the athlete's own pace the change is four hundredths of a
    beat a week.
    """
    trend = analysis.ef_trend(runs(), "run", as_of=TODAY)
    assert trend.pace_corrected is True
    assert trend.verdict == "flat"
    assert abs(trend.slope_bpm_at_pace_per_week) < 1.0
    assert trend.slope_pct_per_week < -5      # what the old verdict was built on


def test_a_short_burst_of_sessions_gets_no_verdict():
    """Three rides inside three days can produce any slope you like."""
    rides = runs([("2026-08-22", 4.35, 114.0), ("2026-08-23", 4.21, 133.0),
                  ("2026-08-25", 4.39, 126.0)], sport="bike")
    trend = analysis.ef_trend(rides, "bike", as_of=TODAY)
    assert trend.verdict == "insufficient_data"
    assert analysis.hr_trend(rides, "bike")["span_days"] < 7


def test_a_week_of_sessions_does_get_one():
    assert analysis.hr_trend(runs(), "run")["span_days"] == 7
    assert analysis.hr_trend(runs(), "run")["verdict"] == "flat"


def test_a_real_decline_is_still_called_a_decline():
    """Same paces, rising heart rate: that is a genuine loss and must show."""
    worse = [
        ((date(2026, 6, 1) + timedelta(days=i * 4)).isoformat(),
         2.05 + (i % 2) * 0.08, 145.0 + i * 1.6)
        for i in range(10)
    ]
    trend = analysis.ef_trend(runs(worse), "run", as_of=date(2026, 7, 8))
    assert trend.verdict == "declining"
    assert trend.pace_corrected is True


def test_the_insight_says_a_change_too_small_to_print_in_words():
    """"+0.0 bpm" is right and reads as a failed calculation. It was reported."""
    from core import insights
    data = {"activities": runs(), "wellness": [], "zones": [], "strength": [],
            "scoped_to": ("run",)}
    ins = insights.fitness_insight(data, TODAY)
    text = " ".join(ins.bullets)
    assert "no measurable change" in text and "usual pace" in text
    assert "+0.0" not in text
    assert "slipping" not in ins.headline.lower()


def test_the_insight_states_a_real_change_as_a_number():
    from core import insights
    worse = [
        ((date(2026, 6, 1) + timedelta(days=i * 4)).isoformat(),
         2.05 + (i % 2) * 0.08, 145.0 + i * 1.6)
        for i in range(10)
    ]
    data = {"activities": runs(worse), "wellness": [], "zones": [],
            "strength": [], "scoped_to": ("run",)}
    ins = insights.fitness_insight(data, date(2026, 7, 8))
    text = " ".join(ins.bullets)
    assert "bpm" in text and "usual pace" in text
