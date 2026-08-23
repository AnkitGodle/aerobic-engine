"""Training heart rate: the "same pace, fewer beats" signal.

Raw average heart rate is not progress on its own — a hard session is high
because it was hard. These check that the normalised figure isolates fitness.
"""

from __future__ import annotations

from datetime import timedelta

from conftest import TODAY

from core.analysis import hr_points, hr_trend, reference_speed


def run(day, speed, hr, steady=1, aid=None):
    ef = speed * 100.0 / hr
    return {
        "activity_id": aid or f"a{day}", "sport": "run",
        "start_date": day.isoformat(), "duration_s": 2700, "moving_s": 2700,
        "distance_m": speed * 2700, "avg_hr": hr, "max_hr": hr * 1.06,
        "avg_speed_mps": speed, "ef": ef, "ef_metric": "speed_per_hr",
        "is_steady": steady, "steady_reason": "steady" if steady else "too hard",
    }


def test_reference_speed_is_the_athletes_own_median():
    acts = [run(TODAY - timedelta(days=d), 2.0 + d * 0.1, 150) for d in range(5)]
    ref = reference_speed(acts, "run")
    assert 2.0 <= ref <= 2.4


def test_same_pace_fewer_beats_reads_as_improving():
    """Pace held constant, heart rate falling: unambiguous aerobic progress."""
    acts = []
    for i in range(10):
        day = TODAY - timedelta(days=70 - i * 7)
        acts.append(run(day, 2.60, 160 - i))       # 160 bpm down to 151
    t = hr_trend(acts, "run", as_of=TODAY)
    assert t["normalised_change_bpm"] < -1
    assert t["verdict"] == "improving"
    assert t["recent_hr"] < t["baseline_hr"]


def test_going_faster_at_the_same_heart_rate_also_reads_as_improving():
    """The other shape of the same progress: HR flat, pace rising."""
    acts = []
    for i in range(10):
        day = TODAY - timedelta(days=70 - i * 7)
        acts.append(run(day, 2.40 + i * 0.03, 150))
    t = hr_trend(acts, "run", as_of=TODAY)
    assert t["normalised_change_bpm"] < -1, "faster at the same HR must count"
    assert t["verdict"] == "improving"


def test_a_hard_session_does_not_look_like_lost_fitness():
    """A slow, high-HR session should not be flattered by its raw average, and a
    fast hard session should not be punished for having a high one."""
    ref = 2.5
    slow_but_easy_hr = hr_points([run(TODAY, 1.8, 140)], "run", ref)[0]
    fast_hard = hr_points([run(TODAY, 3.2, 175)], "run", ref)[0]
    # Raw: the slow session looks better. Normalised: it is worse.
    assert slow_but_easy_hr["avg_hr"] < fast_hard["avg_hr"]
    assert slow_but_easy_hr["hr_at_reference"] > fast_hard["hr_at_reference"]


def test_steady_only_filters_hard_sessions_out():
    acts = [run(TODAY - timedelta(days=d * 7), 2.5, 150, steady=1) for d in range(6)]
    acts += [run(TODAY - timedelta(days=1), 2.5, 178, steady=0, aid="hard")]
    everything = hr_trend(acts, "run", as_of=TODAY, steady_only=False)
    clean = hr_trend(acts, "run", as_of=TODAY, steady_only=True)
    assert everything["n_sessions"] == 7
    assert clean["n_sessions"] == 6
    assert clean["recent_hr"] < everything["recent_hr"]


def test_power_based_bike_sessions_get_no_pace_normalisation():
    """Watts per beat cannot be restated as a heart rate at a pace."""
    acts = [{
        "activity_id": "b1", "sport": "bike", "start_date": TODAY.isoformat(),
        "duration_s": 3600, "distance_m": 30000, "avg_hr": 140,
        "avg_speed_mps": 8.3, "avg_power_w": 200, "ef": 200 / 140,
        "ef_metric": "power_per_hr", "is_steady": 1,
    }]
    pts = hr_points(acts, "bike")
    assert pts[0]["avg_hr"] == 140
    assert pts[0]["hr_at_reference"] is None


def test_no_heart_rate_means_no_point():
    acts = [{"activity_id": "x", "sport": "run", "start_date": TODAY.isoformat(),
             "duration_s": 1800, "avg_hr": None}]
    assert hr_points(acts, "run") == []


def test_empty_history_does_not_explode():
    t = hr_trend([], "run", as_of=TODAY)
    assert t["verdict"] == "insufficient_data" and t["n_sessions"] == 0
