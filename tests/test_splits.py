"""Even splits, cut from the samples rather than from the lap button.

The bug these exist for: a 10 km ride was shown as five splits numbered 1 to 5,
because the watch had five laps — 203 m, 5 km, 1.9 km, 1.3 km and 1.6 km. The
rows were labelled like kilometres and were nothing of the kind.
"""

from __future__ import annotations

import math

from core.analysis import MIN_PARTIAL_SPLIT, km_splits

# A metre of latitude, near enough, for building a straight synthetic course.
DEG_PER_M = 1.0 / 111_320.0


def straight_line(metres_per_second: float, seconds: int, hr: float = 140.0,
                  step: int = 5, climb_per_sample: float = 0.0) -> list[dict]:
    """A course heading due north at a constant speed, one fix every `step`."""
    out = []
    for i in range(0, seconds + step, step):
        out.append({
            "t_s": float(i),
            "hr": hr,
            "speed_mps": metres_per_second,
            "lat": 18.5 + metres_per_second * i * DEG_PER_M,
            "lon": 73.9,
            "altitude_m": 500.0 + climb_per_sample * (i / step),
        })
    return out


def test_ten_kilometres_makes_ten_splits():
    stream = straight_line(4.0, 2500)          # 10 km at 4 m/s
    rows = km_splits(stream, 1000.0, total_m=10_000.0)
    assert len(rows) == 10
    assert [r["index"] for r in rows] == list(range(1, 11))
    assert all(not r["partial"] for r in rows)
    assert all(abs(r["distance_m"] - 1000.0) < 1 for r in rows)


def test_each_split_is_the_time_it_took():
    stream = straight_line(4.0, 2500)
    rows = km_splits(stream, 1000.0, total_m=10_000.0)
    for row in rows:
        assert math.isclose(row["seconds"], 250.0, abs_tol=6.0)
    assert math.isclose(sum(r["seconds"] for r in rows), 2500.0, abs_tol=15.0)


def test_the_remainder_is_its_own_row_and_says_so():
    stream = straight_line(4.0, 1437)          # 5.75 km
    rows = km_splits(stream, 1000.0, total_m=5750.0)
    assert len(rows) == 6
    assert rows[-1]["partial"] is True
    assert 700 < rows[-1]["distance_m"] < 800
    assert all(not r["partial"] for r in rows[:-1])


def test_a_sliver_of_a_split_is_not_shown():
    """40 m left over is not a row. It would sit beside ten real kilometres."""
    stream = straight_line(4.0, 2510)          # 10.04 km
    rows = km_splits(stream, 1000.0, total_m=10_040.0)
    assert len(rows) == 10
    assert rows[-1]["partial"] is False


def test_the_partial_threshold_is_the_one_documented():
    stream = straight_line(4.0, 2500 + int(1000 * MIN_PARTIAL_SPLIT / 4) + 5)
    rows = km_splits(stream, 1000.0)
    assert rows[-1]["partial"] is True


def test_heart_rate_is_the_average_over_that_split_alone():
    stream = straight_line(4.0, 2500)
    for row in stream:
        # Second half ten beats higher, so the split averages must differ.
        if row["t_s"] > 1250:
            row["hr"] = 150.0
    rows = km_splits(stream, 1000.0, total_m=10_000.0)
    assert rows[0]["avg_hr"] == 140.0
    assert rows[-1]["avg_hr"] == 150.0


def test_climbing_is_counted_per_split_not_for_the_session():
    stream = straight_line(4.0, 2500, climb_per_sample=1.0)   # 1 m every 5 s
    rows = km_splits(stream, 1000.0, total_m=10_000.0)
    assert all(45 < r["elev_gain_m"] < 55 for r in rows)


def test_downhill_does_not_subtract_from_the_climb():
    stream = straight_line(4.0, 2500, climb_per_sample=-1.0)
    rows = km_splits(stream, 1000.0, total_m=10_000.0)
    assert all(r["elev_gain_m"] == 0 for r in rows)


def test_a_pool_swim_has_no_gps_so_speed_carries_it():
    stream = [{"t_s": float(i), "hr": 130.0, "speed_mps": 1.0}
              for i in range(0, 601, 5)]
    rows = km_splits(stream, 100.0)             # 600 m in 100 m lengths
    assert len(rows) == 6
    assert all(abs(r["distance_m"] - 100.0) < 1 for r in rows)
    assert all(math.isclose(r["seconds"], 100.0, abs_tol=6) for r in rows)


def test_the_recorded_distance_wins_over_gps_drift():
    """GPS over a thinned stream lands short; the watch's distance is the truth."""
    stream = straight_line(4.0, 2500)
    honest = km_splits(stream, 1000.0)
    scaled = km_splits(stream, 1000.0, total_m=11_000.0)
    assert len(scaled) > len(honest)
    assert len(scaled) == 11


def test_one_sample_crossing_two_boundaries_still_makes_two_splits():
    """A 30-second gap at 70 km/h covers 580 m; a 2-minute gap covers two units."""
    stream = [
        {"t_s": 0.0, "hr": 120.0, "speed_mps": 20.0, "lat": 18.5, "lon": 73.9},
        {"t_s": 120.0, "hr": 130.0, "speed_mps": 20.0,
         "lat": 18.5 + 2400 * DEG_PER_M, "lon": 73.9},
        {"t_s": 240.0, "hr": 135.0, "speed_mps": 20.0,
         "lat": 18.5 + 4800 * DEG_PER_M, "lon": 73.9},
        {"t_s": 250.0, "hr": 135.0, "speed_mps": 20.0,
         "lat": 18.5 + 5000 * DEG_PER_M, "lon": 73.9},
    ]
    rows = km_splits(stream, 1000.0, total_m=5000.0)
    assert len(rows) == 5
    assert [r["index"] for r in rows] == [1, 2, 3, 4, 5]


def test_no_samples_means_no_splits():
    assert km_splits([], 1000.0) == []
    assert km_splits([{"t_s": 0.0, "hr": 120.0}], 1000.0) == []


def test_a_stationary_stream_makes_no_splits():
    """Strength work has a stream and no distance. It must not produce rows."""
    stream = [{"t_s": float(i), "hr": 110.0} for i in range(0, 600, 5)]
    assert km_splits(stream, 1000.0) == []


def test_samples_out_of_order_are_sorted_first():
    stream = straight_line(4.0, 2000)
    shuffled = stream[::-1]
    assert km_splits(shuffled, 1000.0, total_m=8000.0) == \
        km_splits(stream, 1000.0, total_m=8000.0)
