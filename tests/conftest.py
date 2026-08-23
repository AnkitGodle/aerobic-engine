"""Synthetic fixtures. No Garmin account and no network needed to run the suite."""

from __future__ import annotations

import random
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.analysis import compute_activity_metrics  # noqa: E402
from core.store import Store, week_start_of  # noqa: E402

# A Wednesday, so "part of the week is already done" is the default situation.
TODAY = date(2026, 8, 19)

WEEK_TEMPLATE = [
    (0, "swim", 45),
    (1, "bike", 80),
    (2, "run", 50),
    (3, "swim", 45),
    (5, "run", 70),
    (6, "bike", 130),
    (0, "strength", 28),
    (2, "strength", 28),
]
HR = {"swim": 132, "bike": 138, "run": 146, "strength": 105}
SPEED = {"swim": 0.95, "bike": 7.2, "run": 2.75, "strength": 0.0}


def build_db(
    path: str, weeks: int = 14, today: date = TODAY, bad_recovery: bool = False
) -> Store:
    """Weeks of steadily improving training, optionally with recovery in the bin."""
    rng = random.Random(7)
    store = Store(path)
    start = week_start_of(today) - timedelta(weeks=weeks - 1)
    activities, wellness = [], []
    aid = 1000

    for w in range(weeks):
        ws = start + timedelta(weeks=w)
        gain = 1 + 0.006 * w  # efficiency improves ~0.6%/week
        vol = 0.75 + 0.02 * w
        for dow, sport, mins in WEEK_TEMPLATE:
            d = ws + timedelta(days=dow)
            if d > today:
                continue
            aid += 1
            minutes = mins * vol * rng.uniform(0.95, 1.05)
            speed = SPEED[sport] * gain * rng.uniform(0.98, 1.02)
            activities.append(
                {
                    "activity_id": str(aid),
                    "sport": sport,
                    "garmin_type": sport,
                    "name": f"{sport} {d}",
                    "start_time": f"{d}T07:00:00",
                    "start_date": d.isoformat(),
                    "duration_s": minutes * 60,
                    "moving_s": minutes * 60,
                    "distance_m": speed * minutes * 60,
                    "avg_hr": HR[sport] * rng.uniform(0.99, 1.01),
                    "max_hr": HR[sport] * 1.09,
                    "avg_speed_mps": speed or None,
                    "avg_power_w": 185 * gain if sport == "bike" else None,
                    "elevation_gain_m": 100 if sport == "bike" else 20,
                    "calories": minutes * 9,
                    "training_load": minutes * 1.6,
                    "aerobic_te": 2.6,
                    "anaerobic_te": 0.4,
                    "ingested_at": f"{today}T12:00:00",
                }
            )
        for dow in range(7):
            d = ws + timedelta(days=dow)
            if d > today:
                continue
            recent = bad_recovery and (today - d).days < 7
            wellness.append(
                {
                    "day": d.isoformat(),
                    "resting_hr": 52 - 0.9 * w / weeks + (7 if recent else 0),
                    "hrv_last_night": 68 + 4 * w / weeks - (14 if recent else 0),
                    "hrv_status": "unbalanced" if recent else "balanced",
                    "vo2max_run": 48 + 3 * w / weeks,
                    "vo2max_bike": 50 + 3 * w / weeks,
                    "training_readiness": 28 if recent else 68,
                    "training_status": "Overreaching" if recent else "Productive",
                    "acute_load": 420,
                    "chronic_load": 300 if recent else 400,
                    "load_ratio": 1.4 if recent else 1.05,
                    "sleep_score": 78,
                    "ingested_at": f"{today}T12:00:00",
                }
            )

    store.upsert_activities(activities)
    store.upsert_wellness(wellness)

    # HR streams on the long rides, so decoupling has something to chew on.
    for a in activities:
        if a["sport"] == "bike" and a["duration_s"] > 100 * 60:
            n, dur = 120, a["duration_s"]
            store.replace_stream(
                a["activity_id"],
                [
                    {
                        "t_s": i * dur / n,
                        "hr": a["avg_hr"] * (0.97 + 0.06 * i / n),
                        "speed_mps": a["avg_speed_mps"],
                        "power_w": 185,
                    }
                    for i in range(n)
                ],
            )
    store.upsert_metrics(
        [
            compute_activity_metrics(a, store.stream(a["activity_id"]))
            for a in store.activities()
        ]
    )
    return store


@pytest.fixture
def healthy(tmp_path) -> Store:
    store = build_db(str(tmp_path / "healthy.db"))
    yield store
    store.close()


@pytest.fixture
def wrecked(tmp_path) -> Store:
    """Same training, but HRV down, RHR up, readiness low, ACWR 1.4."""
    store = build_db(str(tmp_path / "wrecked.db"), bad_recovery=True)
    yield store
    store.close()
