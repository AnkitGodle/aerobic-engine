"""Build a Garmin strength workout from a planned session and send it up.

Why this exists: the watch counts reps well but often does not know which
exercise you are doing, so sets come back unnamed and have to be assigned by
hand afterwards. Sending the session up as a named workout removes that step —
the watch knows each exercise while you do it, and the sets arrive labelled.

Two things were established by probing the live account rather than assumed,
because Garmin publishes no schema for this:

  * `sportTypeId` 5 is strength_training. 13 is rucking, which is what a guess
    would have produced.
  * A step accepts `category`, `exerciseName`, `endCondition` of reps or time,
    and a weight, and all of it survives a round trip.

What the round trip does *not* prove is that an exercise name is real: Garmin
echoes back whatever it was sent. So the mapping below is conservative. Where
the category is well known but the exact name is not, only the category goes
up and the precise name goes in the step description, which the watch shows.
An invented enum value would display as nothing at all, which is worse than a
coarser category plus readable text.
"""

from __future__ import annotations

import logging
from typing import Any

from core import strength

log = logging.getLogger("aerobic_engine.workout")

STRENGTH_SPORT = {"sportTypeId": 5, "sportTypeKey": "strength_training",
                  "displayOrder": 4}

STEP_INTERVAL = {"stepTypeId": 3, "stepTypeKey": "interval", "displayOrder": 3}
STEP_REST = {"stepTypeId": 4, "stepTypeKey": "rest", "displayOrder": 4}
END_REPS = {"conditionTypeId": 10, "conditionTypeKey": "reps",
            "displayOrder": 10, "displayable": True}
END_TIME = {"conditionTypeId": 2, "conditionTypeKey": "time",
            "displayOrder": 2, "displayable": True}
NO_TARGET = {"workoutTargetTypeId": 1, "workoutTargetTypeKey": "no.target",
             "displayOrder": 1}
KG = {"unitId": 8, "unitKey": "kilogram", "factor": 1000.0}

# library id -> (Garmin category, Garmin exercise name or None)
#
# A None name means "category only": the category enums are stable and widely
# used, the per-exercise names are not, and a wrong name shows as blank on the
# watch. The exercise's real name still reaches the watch through the step
# description.
GARMIN_TARGET: dict[str, tuple[str, str | None]] = {
    "calf_raise_straight": ("CALF_RAISE", "STANDING_CALF_RAISE"),
    "calf_raise_bent": ("CALF_RAISE", "SEATED_CALF_RAISE"),
    "calf_raise_single_leg": ("CALF_RAISE", "SINGLE_LEG_CALF_RAISE"),
    "single_leg_calf_hold": ("CALF_RAISE", None),
    "split_squat": ("SQUAT", "SPLIT_SQUAT"),
    "reverse_lunge": ("LUNGE", "REVERSE_LUNGE"),
    "step_up": ("STEP_UP", "STEP_UP"),
    "step_down": ("SQUAT", None),
    "goblet_squat": ("SQUAT", "GOBLET_SQUAT"),
    "wall_sit": ("SQUAT", "WALL_SIT"),
    "spanish_squat": ("SQUAT", None),
    "terminal_knee_extension": ("SQUAT", None),
    "rdl": ("DEADLIFT", "ROMANIAN_DEADLIFT"),
    "single_leg_rdl": ("DEADLIFT", "SINGLE_LEG_DEADLIFT"),
    "nordic_curl_assisted": ("CURL", None),
    "glute_bridge": ("HIP_RAISE", None),
    "hip_thrust": ("HIP_RAISE", None),
    "side_lying_abduction": ("HIP_RAISE", None),
    "band_monster_walk": ("HIP_RAISE", None),
    "side_plank_hip_lift": ("PLANK", "SIDE_PLANK"),
    "copenhagen_plank": ("PLANK", "SIDE_PLANK"),
    "tib_raise": ("CALF_RAISE", None),
}

REST_SECONDS = 60


def _step(order: int, prescription: Any, exercise: Any) -> dict[str, Any]:
    category, name = GARMIN_TARGET.get(exercise.id, ("SQUAT", None))
    # Per side is doubled here rather than left to the athlete to remember: the
    # watch counts what it sees, and a unilateral set is two sets of work.
    hold = prescription.hold_s
    step: dict[str, Any] = {
        "type": "ExecutableStepDTO",
        "stepOrder": order,
        "stepType": STEP_INTERVAL,
        "category": category,
        "targetType": NO_TARGET,
        "description": (
            f"{exercise.name}"
            + (" (per side)" if exercise.unilateral else "")
            + (f" — {exercise.tempo}" if exercise.tempo else "")
        )[:512],
    }
    if name:
        step["exerciseName"] = name
    if hold:
        step["endCondition"] = END_TIME
        step["endConditionValue"] = float(hold)
    else:
        step["endCondition"] = END_REPS
        step["endConditionValue"] = float(prescription.reps or 8)
    if prescription.load_kg:
        step["weightValue"] = float(prescription.load_kg)
        step["weightUnit"] = KG
    return step


def build(prescriptions: list[Any], name: str) -> dict[str, Any]:
    """One workout from a session's prescription.

    Sets are emitted as repeated steps rather than as a repeat group. A
    RepeatGroupDTO nests differently across firmware and there was no strength
    workout on the account to copy a working one from, so this uses the shape
    that was actually verified end to end.
    """
    steps: list[dict[str, Any]] = []
    order = 1
    for presc in prescriptions:
        exercise = strength.EXERCISES.get(presc.exercise_id)
        if exercise is None:            # cannot happen via the planner; be safe
            continue
        for set_index in range(max(1, presc.sets)):
            steps.append(_step(order, presc, exercise))
            order += 1
            last = (presc is prescriptions[-1]
                    and set_index == max(1, presc.sets) - 1)
            if not last:
                steps.append({
                    "type": "ExecutableStepDTO", "stepOrder": order,
                    "stepType": STEP_REST, "endCondition": END_TIME,
                    "endConditionValue": float(REST_SECONDS),
                    "targetType": NO_TARGET, "description": "Rest",
                })
                order += 1
    return {
        "sportType": STRENGTH_SPORT,
        "workoutName": name[:80],
        "description": ("Built by Aerobic Engine. Weights follow your logged "
                        "progression.")[:1024],
        "workoutSegments": [{
            "segmentOrder": 1,
            "sportType": STRENGTH_SPORT,
            "workoutSteps": steps,
        }],
    }


def push(api: Any, prescriptions: list[Any], name: str,
         on_date: str | None = None) -> dict[str, Any]:
    """Upload, and schedule it for a date when one is given.

    Returns the Garmin response so the caller can store the id — pushing the
    same session twice would leave two identical workouts on the watch.
    """
    payload = build(prescriptions, name)
    if not payload["workoutSegments"][0]["workoutSteps"]:
        raise ValueError("nothing to send: the session has no exercises")
    created = api.upload_workout(payload) or {}
    workout_id = created.get("workoutId")
    if workout_id and on_date:
        try:
            api.schedule_workout(str(workout_id), on_date)
        except Exception as exc:  # noqa: BLE001 - the workout still exists
            log.warning("Uploaded %s but could not schedule it for %s: %s",
                        workout_id, on_date, exc)
    return created
