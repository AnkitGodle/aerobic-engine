"""Analysis maths: efficiency factor, the steady-session gate, decoupling."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from conftest import TODAY
from core import analysis

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


def test_no_module_is_shadowed_by_a_function_of_the_same_name():
    """`def clock(...)` in the app shadowed the `core.clock` module it imports,
    and every page died with "'function' object has no attribute 'today'". The
    tests passed: nothing but rendering the app would have caught it."""
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for path in sorted((root / "core").glob("*.py")) + \
            sorted((root / "app").glob("*.py")) + \
            sorted((root / "scripts").glob("*.py")):
        tree = ast.parse(path.read_text())
        imported = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add((alias.asname or alias.name).split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imported.add(alias.asname or alias.name)
        defined = {n.name for n in tree.body
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                                     ast.ClassDef))}
        clash = sorted(imported & defined)
        assert not clash, f"{path.name} defines {clash}, which it also imports"


# --------------------------------------------------------------------------
# Garmin's status vocabulary, and keeping it off the page
# --------------------------------------------------------------------------


@pytest.mark.parametrize("raw,shown", [
    ("PRODUCTIVE_3", "Productive"),
    ("PRODUCTIVE_2", "Productive"),
    ("MAINTAINING_1", "Maintaining"),
    ("HIGH_STRAIN_2", "High Strain"),
    ("OVERREACHING_1", "Overreaching"),
    ("Productive", "Productive"),
    ("overreaching", "Overreaching"),
    ("PEAKING", "Peaking"),
])
def test_a_status_reaches_the_page_as_words(raw, shown):
    """The number in PRODUCTIVE_3 picks Garmin's wording, not a level."""
    assert analysis.clean_status(raw) == shown


@pytest.mark.parametrize("raw", [
    "NO_STATUS_1", "NO_STATUS_2", "no_status", "ONBOARDING_1", "UNKNOWN",
    "none", "NOT_SET", "", None, "   ",
])
def test_a_placeholder_is_not_a_status(raw):
    assert analysis.clean_status(raw) is None
    assert analysis.status_meaning(raw) is None


def test_every_status_garmin_documents_has_a_meaning():
    for key in analysis.STATUS_MEANING:
        assert analysis.status_meaning(f"{key.upper()}_2")
        assert analysis.status_meaning(key)


def test_an_unknown_status_still_shows_but_says_nothing_it_cannot():
    """A status Garmin adds later must not vanish, and must not be glossed."""
    assert analysis.clean_status("SOMETHING_NEW_4") == "Something New"
    assert analysis.status_meaning("SOMETHING_NEW_4") is None


def test_the_deload_trigger_still_reads_a_cleaned_status():
    """The guardrail matches words inside the status, so tidying it must not
    stop a bad status from forcing an easy week."""
    from core.planner import BAD_STATUS_WORDS
    for bad in ("OVERREACHING_1", "UNPRODUCTIVE_2", "DETRAINING_1", "STRAINED_3"):
        cleaned = analysis.clean_status(bad)
        assert any(w in cleaned.lower() for w in BAD_STATUS_WORDS), cleaned


def test_a_good_status_does_not_trip_the_deload_words():
    from core.planner import BAD_STATUS_WORDS
    for good in ("PRODUCTIVE_3", "MAINTAINING_1", "PEAKING_2", "RECOVERY_1"):
        cleaned = analysis.clean_status(good)
        assert not any(w in cleaned.lower() for w in BAD_STATUS_WORDS), cleaned


# --------------------------------------------------------------------------
# Every session counts
# --------------------------------------------------------------------------


def _hard_runs(days: int = 4) -> list[dict]:
    """Runs that the steady gate would reject: fast, spiky, short."""
    out = []
    for i in range(days):
        day = TODAY - timedelta(days=i * 3)
        out.append({
            "activity_id": f"hard-{i}", "sport": "run", "garmin_type": "run",
            "name": f"run {day}", "start_time": f"{day}T06:00:00",
            "start_date": day.isoformat(), "duration_s": 1800,
            "moving_s": 1800, "distance_m": 5000 + i * 100,
            "avg_hr": 160 + i, "max_hr": 195, "avg_speed_mps": 2.8 + i * 0.05,
            "ef": 1.7 + i * 0.02, "ef_metric": "speed_per_hr",
            "is_steady": 0, "steady_reason": "85% of time in Z4-Z5",
            "ingested_at": f"{TODAY}T12:00:00",
        })
    return out


def test_a_trend_counts_hard_sessions_too():
    """The athlete asked for this directly: four hard runs are four runs."""
    acts = _hard_runs()
    trend = analysis.ef_trend(acts, "run", as_of=TODAY)
    assert trend.n_sessions == 4
    assert analysis.hr_trend(acts, "run", as_of=TODAY)["n_sessions"] == 4
    assert analysis.all_ef_trends(acts, as_of=TODAY)[0].n_sessions >= 0


def test_the_steady_filter_is_still_available_to_a_caller():
    acts = _hard_runs()
    assert analysis.ef_trend(acts, "run", as_of=TODAY, steady_only=True).n_sessions == 0
    assert analysis.ef_trend(acts, "run", as_of=TODAY).n_sessions == 4


def test_the_status_never_says_a_session_was_excluded():
    status = analysis.ef_data_status(_hard_runs(), "run")
    assert status["sessions"] == 4
    assert status["rejected_reasons"] == {}
    text = status["message"].lower()
    assert "exclud" not in text and "does not count" not in text
    # It may say the sessions were hard — that is information, not a refusal.
    assert "4 run sessions" in text


def test_the_status_says_when_hard_sessions_make_the_line_noisy():
    assert "jump" in analysis.ef_data_status(_hard_runs(), "run")["message"]


def test_a_page_insight_gives_a_verdict_from_hard_sessions(healthy):
    from core import insights
    data = {"activities": _hard_runs(6), "wellness": [], "zones": [],
            "strength": [], "scoped_to": ("run",)}
    ins = insights.fitness_insight(data, TODAY)
    assert "not enough" not in ins.headline.lower()
    assert "excluded" not in " ".join(ins.bullets).lower()


# --------------------------------------------------------------------------
# Static checks over the app, for the mistakes a test suite cannot see
# --------------------------------------------------------------------------


def _app_tree():
    import ast
    import pathlib
    source = (pathlib.Path(__file__).resolve().parents[1]
              / "app" / "streamlit_app.py").read_text()
    return ast.parse(source)


def test_no_call_argues_with_itself_about_a_parameter():
    """The bug this exists for: a parameter was removed from training_hr_block
    and one of its two call sites still passed the old argument positionally, so
    the fourth positional landed on `ceiling` — which the same call also passed
    by keyword. The Lifetime page died with a TypeError. Nothing in the suite
    touched it, because the suite does not render pages.

    Checks both halves: too many positional arguments, and a positional argument
    that lands on a parameter the call also names.
    """
    import ast

    tree = _app_tree()
    params = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            args = node.args
            if args.vararg:
                continue                      # *args takes anything
            params[node.name] = [a.arg for a in args.posonlyargs + args.args]

    problems = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        names = params.get(node.func.id)
        if names is None or any(isinstance(a, ast.Starred) for a in node.args):
            continue
        given = len(node.args)
        if given > len(names):
            problems.append(
                f"{node.func.id}() takes {len(names)} positional arguments, "
                f"called with {given} at line {node.lineno}")
            continue
        taken = set(names[:given])
        for kw in node.keywords:
            if kw.arg and kw.arg in taken:
                problems.append(
                    f"{node.func.id}() got {kw.arg} both positionally and by "
                    f"keyword at line {node.lineno}")
    assert not problems, "; ".join(problems)


def test_no_call_uses_a_keyword_the_function_does_not_have():
    import ast

    tree = _app_tree()
    known = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            args = node.args
            if args.kwarg:
                continue                      # **kwargs takes anything
            known[node.name] = {a.arg for a in
                                args.posonlyargs + args.args + args.kwonlyargs}

    problems = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        allowed = known.get(node.func.id)
        if allowed is None:
            continue
        for kw in node.keywords:
            if kw.arg and kw.arg not in allowed:
                problems.append(f"{node.func.id}(... {kw.arg}=) at line {node.lineno}")
    assert not problems, "; ".join(problems)


# --------------------------------------------------------------------------
# Cadence: the last run, not only the lifetime mean
# --------------------------------------------------------------------------


def _runs_with_cadence(values):
    out = []
    for i, spm in enumerate(values):
        day = (TODAY - timedelta(days=(len(values) - i) * 2)).isoformat()
        out.append({
            "activity_id": f"c{i}", "sport": "run", "start_date": day,
            "start_time": f"{day}T06:00:00", "avg_hr": 150.0,
            "avg_speed_mps": 2.6, "avg_cadence": spm, "duration_s": 2400,
            "distance_m": 6240, "ingested_at": f"{day}T12:00:00",
        })
    return out


def test_a_single_good_session_shows_even_when_the_average_barely_moves():
    """Reported as "my cadence was good today and the KPI did not update".

    It had: three runs at 148.6 plus one at 151.6 moves an all-time mean by 0.8
    of a step. The number the athlete looks at is now the run they just did.
    """
    # The athlete's own four runs, to the precision Garmin recorded them.
    stats = analysis.cadence_stats(
        _runs_with_cadence([151.3125, 150.59375, 143.9375, 151.5625]))
    assert stats["latest"] == 151.6
    assert stats["earlier_avg"] == 148.6
    # 151.5625 against an earlier mean of 148.614: 2.9, shown as "+3".
    assert stats["change_vs_earlier"] == 2.9
    assert stats["avg"] == 149.4          # the mean that hardly moved
    assert stats["sessions"] == 4


def test_the_latest_is_the_most_recent_session_not_the_last_row_given():
    rows = _runs_with_cadence([150.0, 160.0])
    rows.reverse()                        # order of arrival must not matter
    assert analysis.cadence_stats(rows)["latest"] == 160.0


def test_one_session_has_nothing_to_compare_against():
    stats = analysis.cadence_stats(_runs_with_cadence([155.0]))
    assert stats["latest"] == 155.0
    assert stats["earlier_avg"] is None
    assert stats["change_vs_earlier"] is None


def test_no_cadence_anywhere_is_not_an_error():
    stats = analysis.cadence_stats([])
    assert stats["latest"] is None and stats["sessions"] == 0
    assert stats["verdict"] == "no_data"


def test_stride_comes_back_for_the_latest_run():
    stats = analysis.cadence_stats(_runs_with_cadence([150.0, 168.0]))
    # 2.6 m/s at 168 spm is a 92.9 cm stride, derived exactly rather than guessed.
    assert stats["latest_stride_cm"] == pytest.approx(92.9, abs=0.2)
