"""Analysis maths: efficiency factor, the steady-session gate, decoupling."""

from __future__ import annotations

from datetime import date, timedelta

from conftest import TODAY

from core.analysis import (
    acwr_from_activities,
    decoupling,
    ef_trend,
    efficiency_factor,
    recovery_signals,
    steady_check,
    week_summary,
)


def act(**kw):
    base = {
        "activity_id": "1",
        "sport": "run",
        "name": "easy run",
        "start_date": TODAY.isoformat(),
        "duration_s": 3600,
        "moving_s": 3600,
        "distance_m": 10000,
        "avg_hr": 145,
        "max_hr": 155,
        "avg_speed_mps": 2.78,
        "anaerobic_te": 0.5,
    }
    return {**base, **kw}


def test_ef_prefers_power_on_the_bike():
    ef, metric = efficiency_factor(act(sport="bike", avg_power_w=200, avg_hr=140))
    assert metric == "power_per_hr"
    assert ef == 200 / 140


def test_ef_falls_back_to_speed_without_power():
    _, metric = efficiency_factor(act(sport="bike", avg_power_w=None))
    assert metric == "speed_per_hr"


def test_ef_needs_heart_rate():
    assert efficiency_factor(act(avg_hr=None)) == (None, "none")


def test_intervals_are_excluded_from_the_trend():
    ok, _ = steady_check(act())
    assert ok
    assert not steady_check(act(name="5x1k intervals"))[0]
    assert not steady_check(act(name="Parkrun race"))[0]
    assert not steady_check(act(anaerobic_te=2.5))[0]
    assert not steady_check(act(max_hr=185))[0]        # HR spread too wide
    assert not steady_check(act(duration_s=600))[0]    # too short
    assert not steady_check(act(sport="strength"))[0]


def test_a_ragged_hr_stream_fails_the_steady_gate():
    steady = [{"t_s": i * 30, "hr": 145 + (i % 3)} for i in range(60)]
    surges = [{"t_s": i * 30, "hr": 120 if i % 2 else 175} for i in range(60)]
    assert steady_check(act(), steady)[0]
    assert not steady_check(act(), surges)[0]


def test_rising_ef_reads_as_improving():
    acts = []
    for i in range(12):
        d = TODAY - timedelta(days=7 * (12 - i))
        acts.append(
            act(
                activity_id=str(i),
                start_date=d.isoformat(),
                avg_speed_mps=2.6 * (1 + 0.01 * i),
                ef=(2.6 * (1 + 0.01 * i) * 100) / 145,
                ef_metric="speed_per_hr",
                is_steady=1,
            )
        )
    trend = ef_trend(acts, "run", as_of=TODAY)
    assert trend.verdict == "improving"
    assert trend.slope_pct_per_week > 0
    assert trend.n_sessions == 12


def test_decoupling_detects_drift():
    # Same speed throughout, heart rate climbing: output per beat falls.
    stream = [
        {"t_s": i * 60, "hr": 130 + 20 * i / 120, "speed_mps": 2.8}
        for i in range(120)
    ]
    drift, first, second = decoupling(act(duration_s=7200), stream)
    assert drift is not None and drift > 5
    assert first > second


def test_decoupling_ignores_short_sessions():
    stream = [{"t_s": i * 30, "hr": 140, "speed_mps": 2.8} for i in range(40)]
    assert decoupling(act(duration_s=1800), stream) == (None, None, None)


def test_week_summary_counts_rest_days():
    ws = date(2026, 8, 17)
    acts = [
        act(activity_id="a", start_date="2026-08-17", duration_s=3600, sport="bike"),
        act(activity_id="b", start_date="2026-08-19", duration_s=1800, sport="swim"),
    ]
    w = week_summary(acts, ws)
    assert w.by_sport["bike"].minutes == 60
    assert w.by_sport["swim"].sessions == 1
    assert w.rest_days == 5
    assert w.total_minutes == 90


def test_acwr_flags_a_spike():
    acts = [
        act(activity_id=str(i), start_date=(TODAY - timedelta(days=i)).isoformat(),
            training_load=20)
        for i in range(28)
    ]
    steady = acwr_from_activities(acts, TODAY)
    assert 0.9 < steady < 1.1
    spike = acts + [
        act(activity_id=f"x{i}", start_date=(TODAY - timedelta(days=i)).isoformat(),
            training_load=200)
        for i in range(3)
    ]
    assert acwr_from_activities(spike, TODAY) > 1.3


def test_recovery_signals_tolerate_empty_data():
    sig = recovery_signals([], [], as_of=TODAY)
    assert sig.rhr_recent is None and sig.acwr is None
