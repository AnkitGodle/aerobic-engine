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


# --------------------------------------------------------------------------
# Lap-based drift. Decoupling needs an hour-long session to split in half, so
# on this athlete's 45-minute sessions it never computed once. Laps answer the
# same question per kilometre.
# --------------------------------------------------------------------------


def _lap(index, seconds=450, metres=1000, hr=150, speed=2.2):
    return {"activity_id": "a1", "lap_index": index, "duration_s": seconds,
            "distance_m": metres, "avg_hr": hr, "avg_speed_mps": speed}


def test_drift_is_measured_only_across_laps_at_the_same_pace():
    """Heart rate rising with pace is effort, not drift. Speeding up must not
    be reported as poor durability."""
    from core.analysis import lap_drift

    # Steady pace, heart rate climbing: real drift.
    steady = [_lap(i, hr=140 + i * 5) for i in range(1, 5)]
    assert lap_drift(steady)["drift_bpm"] == 15
    assert lap_drift(steady)["verdict"] == "steep"

    # Pace rising with heart rate: the later laps fall outside the tolerance and
    # are excluded, so there is nothing honest to report.
    faster = [_lap(i, hr=140 + i * 5, speed=2.2 + i * 0.3) for i in range(1, 5)]
    assert faster and lap_drift(faster)["drift_bpm"] is None


def test_short_laps_are_ignored():
    """A 40-second lap is a stop or a warm-up segment; its average heart rate
    has not settled and would drag the comparison."""
    from core.analysis import lap_drift

    laps = [_lap(1, seconds=40, hr=110)] + [_lap(i, hr=150) for i in range(2, 5)]
    result = lap_drift(laps)
    assert result["laps_compared"] == 3
    assert result["drift_bpm"] == 0


def test_flat_heart_rate_reads_as_holding_together():
    """The message is read on the page, so it says what happened in plain words
    rather than naming the physiology."""
    from core.analysis import lap_drift

    result = lap_drift([_lap(i, hr=148 + (i % 2)) for i in range(1, 6)])
    assert result["verdict"] == "flat"
    assert "holding it together" in result["message"]
    assert "durability" not in result["message"]


def test_too_few_usable_laps_says_so_rather_than_guessing():
    from core.analysis import lap_drift

    assert lap_drift([])["drift_bpm"] is None
    assert lap_drift([_lap(1), _lap(2)])["drift_bpm"] is None
    assert "three laps" in lap_drift([_lap(1)])["message"]


def test_pace_spread_separates_steady_from_intervals():
    """A more direct reading than heart-rate spread, which also moves with heat,
    fatigue and hills."""
    from core.analysis import lap_pace_spread

    steady = [_lap(i, speed=2.20 + (i % 2) * 0.01) for i in range(1, 7)]
    intervals = [_lap(i, speed=2.2 if i % 2 else 3.6) for i in range(1, 7)]
    assert lap_pace_spread(steady) < 1.0
    assert lap_pace_spread(intervals) > 15.0
    assert lap_pace_spread([_lap(1), _lap(2)]) is None


# --------------------------------------------------------------------------
# Load ramp, consistency and weather
# --------------------------------------------------------------------------


def _day(offset: int) -> str:
    return (TODAY - timedelta(days=offset)).isoformat()


def test_load_ramp_starts_where_the_data_starts():
    """Not `days` ago. A three-week-old account drawing two months of empty axis
    reads as missing data rather than as data that does not exist yet."""
    from core.analysis import load_ramp

    ramp = load_ramp([act(start_date=_day(3), training_load=100)],
                     as_of=TODAY, days=90)
    assert [r["day"] for r in ramp] == [TODAY - timedelta(days=i)
                                        for i in (3, 2, 1, 0)]


def test_load_ramp_averages_and_withholds_the_ratio_until_there_is_history():
    from core.analysis import load_ramp

    acts = [act(activity_id=str(i), start_date=_day(i), training_load=70)
            for i in range(0, 10)]
    ramp = load_ramp(acts, as_of=TODAY, days=90)
    today_row = ramp[-1]
    # Seven days of 70 spread over seven days.
    assert today_row["acute"] == 70.0
    # Ten days of load averaged over 28 gives a much lower chronic base.
    assert today_row["chronic"] == 25.0
    # Ten days of history is not enough for the ratio to mean anything.
    assert today_row["ratio"] is None


def test_load_ramp_ratio_appears_once_there_is_enough_history():
    from core.analysis import load_ramp, ramp_verdict

    acts = [act(activity_id=str(i), start_date=_day(i), training_load=50)
            for i in range(0, 30)]
    ramp = load_ramp(acts, as_of=TODAY, days=90)
    assert ramp[-1]["ratio"] is not None
    # Steady daily load: the last week and the last month agree.
    assert 0.9 <= ramp[-1]["ratio"] <= 1.1
    assert ramp_verdict(ramp)["verdict"] == "productive"


def test_ramp_verdict_calls_out_a_spike():
    from core.analysis import load_ramp, ramp_verdict

    acts = [act(activity_id=str(i), start_date=_day(i), training_load=20)
            for i in range(7, 40)]
    acts += [act(activity_id=f"hard{i}", start_date=_day(i), training_load=200)
             for i in range(0, 7)]
    verdict = ramp_verdict(load_ramp(acts, as_of=TODAY))
    assert verdict["verdict"] == "high"
    assert verdict["ratio"] > 1.3


def test_load_ramp_falls_back_to_minutes_when_garmin_scored_nothing():
    """Load is missing on a new account, and dropping those sessions would
    understate the ramp."""
    from core.analysis import load_ramp

    ramp = load_ramp([act(start_date=_day(0), training_load=None,
                          duration_s=3600)], as_of=TODAY)
    assert ramp[-1]["load"] == 60.0


def test_weekly_zone_minutes_follow_the_athletes_ceiling():
    """The whole point of recounting from samples: raising the ceiling has to move
    minutes from moderate into easy."""
    from core.analysis import weekly_zone_minutes_from_streams

    acts = [act(activity_id="a", start_date=_day(0), duration_s=3600)]
    # Half the session at 120 bpm, half at 133.
    streams = {"a": [{"hr": 120}] * 10 + [{"hr": 133}] * 10}
    garmin = {1: (90, 111), 2: (112, 125), 3: (126, 140), 4: (141, 160),
              5: (161, None)}
    raised = {**garmin, 2: (112, 137), 3: (138, 140)}

    low = weekly_zone_minutes_from_streams(streams, acts, garmin, as_of=TODAY)
    high = weekly_zone_minutes_from_streams(streams, acts, raised, as_of=TODAY)
    assert low[0]["z2"] == 30.0 and low[0]["z3"] == 30.0
    assert high[0]["z2"] == 60.0 and high[0]["z3"] == 0.0
    assert high[0]["easy_pct"] == 100.0
    # Duration-scaled, not sample-counted: the week totals the real hour.
    assert low[0]["total"] == 60.0


def test_consistency_marks_every_day_including_the_empty_ones():
    from core.analysis import consistency, streak

    acts = [act(activity_id="a", start_date=_day(1), duration_s=1800),
            act(activity_id="b", start_date=_day(0), duration_s=2400)]
    rows = consistency(acts, as_of=TODAY, weeks=2)
    assert len(rows) == 10          # two Mondays back to a Wednesday
    assert rows[-1]["minutes"] == 40.0
    assert rows[-1]["sports"] == ["run"]
    assert rows[0]["sessions"] == 0
    assert streak(rows)["current"] == 2
    assert streak(rows)["active_days"] == 2


def test_streak_survives_a_rest_day_today():
    """A streak should not read as zero because today's session has not happened
    yet at nine in the morning."""
    from core.analysis import consistency, streak

    acts = [act(activity_id=str(i), start_date=_day(i)) for i in (1, 2, 3)]
    rows = consistency(acts, as_of=TODAY, weeks=2)
    assert rows[-1]["sessions"] == 0
    assert streak(rows)["current"] == 3


def test_strength_days_count_towards_showing_up():
    from core.analysis import consistency

    rows = consistency([], as_of=TODAY, weeks=1,
                       strength_rows=[{"day": _day(0)}])
    assert rows[-1]["sports"] == ["strength"]


def test_weather_effect_pairs_dew_point_with_heart_rate_at_one_pace():
    from core.analysis import weather_effect

    acts = [act(activity_id="a", start_date=_day(2), avg_hr=150,
                avg_speed_mps=2.5, ef=2.5 * 100 / 150, ef_metric="speed_per_hr"),
            act(activity_id="b", start_date=_day(0), avg_hr=140,
                avg_speed_mps=2.5, ef=2.5 * 100 / 140, ef_metric="speed_per_hr")]
    weather = {"a": {"dew_point_c": 22.0, "temp_c": 28.0, "condition": "Humid"},
               "b": {"dew_point_c": 12.0, "temp_c": 20.0, "condition": "Clear"}}
    out = weather_effect(acts, weather)
    assert [p["dew_point_c"] for p in out["points"]] == [22.0, 12.0]
    assert out["hot_sessions"] == 1
    assert out["hot_share"] == 50


def test_weather_effect_skips_sessions_with_no_weather_row():
    from core.analysis import weather_effect

    acts = [act(activity_id="a", ef=1.7, ef_metric="speed_per_hr")]
    assert weather_effect(acts, {})["points"] == []


def test_no_module_defines_the_same_name_twice():
    """A scripted edit once inserted a helper twice and left a 400-line block
    duplicated in core/sync.py. Python takes the last definition, so the tests
    passed while the code that ran was the old copy. Cheap to rule out."""
    import ast
    from collections import Counter
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for path in sorted((root / "core").glob("*.py")) + \
            sorted((root / "app").glob("*.py")) + \
            sorted((root / "scripts").glob("*.py")):
        tree = ast.parse(path.read_text())
        names = [n.name for n in tree.body
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                                   ast.ClassDef))]
        repeated = [n for n, c in Counter(names).items() if c > 1]
        assert not repeated, f"{path.name} defines {repeated} more than once"
