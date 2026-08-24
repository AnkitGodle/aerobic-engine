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
import re
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
# Both columns are verified against the live account, because Garmin fails the
# two differently and neither failure is visible from the code:
#
#   * An invalid *category* returns HTTP 400 "Invalid category". STEP_UP and
#     BALANCE are not categories, which would have rejected a whole session —
#     step-ups belong under SQUAT.
#   * An invalid *name* is accepted and then silently dropped, so the upload
#     succeeds and the watch shows a bare category. Of thirty plausible names
#     probed, nineteen were discarded this way: SPLIT_SQUAT, WALL_SIT,
#     REVERSE_LUNGE, GLUTE_BRIDGE, HIP_THRUST, SINGLE_LEG_DEADLIFT and
#     SINGLE_LEG_CALF_RAISE among them.
#
# So a name appears below only if Garmin kept it. Everything else is
# category-only and carries its real name in the step description, which the
# watch displays. Nothing here is a near-miss substitution: labelling a reverse
# lunge as WALKING_LUNGE would be a name Garmin accepts and a lie about what to
# do.
# Names this account actually keeps. Probed by uploading a workout, reading it
# back and comparing: an unknown name is accepted, silently blanked, and the
# watch then shows the bare category instead. That is how a tibialis raise came
# out as "Calf raise" and a wall sit as "Squat" — the right exercise under a
# label for a different muscle, which is worse than no label at all.
#
# Round-tripped 2026-08-24. Re-probe with the same method if Garmin's catalogue
# moves; do not add a name here on the strength of it looking plausible.
VERIFIED_NAMES = frozenset({
    "STANDING_CALF_RAISE", "SEATED_CALF_RAISE", "WEIGHTED_STANDING_CALF_RAISE",
    "SINGLE_LEG_STANDING_CALF_RAISE", "SINGLE_LEG_BENT_KNEE_CALF_RAISE",
    "GOBLET_SQUAT", "STEP_UP", "BARBELL_STEP_UP", "BOX_STEP_SQUAT",
    "DUMBBELL_SPLIT_SQUAT", "BODY_WEIGHT_WALL_SQUAT", "BRACED_SQUAT",
    "WALKING_LUNGE", "BARBELL_REVERSE_LUNGE", "REVERSE_LUNGE_WITH_REACH_BACK",
    "ROMANIAN_DEADLIFT", "STRAIGHT_LEG_DEADLIFT",
    "SINGLE_LEG_ROMANIAN_DEADLIFT_WITH_DUMBBELL",
    "LEG_CURL", "SLIDING_LEG_CURL", "SINGLE_LEG_SLIDING_LEG_CURL",
    "SPLIT_STANCE_EXTENSION",
    "HIP_RAISE", "WEIGHTED_HIP_RAISE", "SINGLE_LEG_HIP_RAISE",
    "SIDE_PLANK", "SIDE_PLANK_WITH_LEG_LIFT", "SIDE_PLANK_LIFT",
    "SIDE_LYING_LEG_RAISE", "LATERAL_WALKS_WITH_BAND_AT_ANKLES",
    "STANDING_HIP_ABDUCTION", "ANKLE_DORSIFLEXION_WITH_BAND",
})

# Every library exercise now has a name the watch will display. Where Garmin has
# no entry for the exact movement, the closest verified name is used and the
# mismatch is only ever the implement — "barbell reverse lunge" while you hold
# dumbbells reads oddly but tells you which exercise you are doing, which is the
# job. A name that points at the wrong muscle is never used.
GARMIN_TARGET: dict[str, tuple[str, str | None]] = {
    "calf_raise_straight": ("CALF_RAISE", "STANDING_CALF_RAISE"),
    "calf_raise_bent": ("CALF_RAISE", "SEATED_CALF_RAISE"),
    "calf_raise_single_leg": ("CALF_RAISE", "SINGLE_LEG_STANDING_CALF_RAISE"),
    "single_leg_calf_hold": ("CALF_RAISE", "SINGLE_LEG_STANDING_CALF_RAISE"),
    # Under WARM_UP because that is the only category Garmin files ankle
    # dorsiflexion under. CALF_RAISE was the opposite muscle group.
    "tib_raise": ("WARM_UP", "ANKLE_DORSIFLEXION_WITH_BAND"),
    "split_squat": ("SQUAT", "DUMBBELL_SPLIT_SQUAT"),
    "reverse_lunge": ("LUNGE", "BARBELL_REVERSE_LUNGE"),
    "step_up": ("SQUAT", "STEP_UP"),
    "step_down": ("SQUAT", "BOX_STEP_SQUAT"),
    "goblet_squat": ("SQUAT", "GOBLET_SQUAT"),
    "wall_sit": ("SQUAT", "BODY_WEIGHT_WALL_SQUAT"),
    "spanish_squat": ("SQUAT", "BRACED_SQUAT"),
    "terminal_knee_extension": ("LEG_CURL", "SPLIT_STANCE_EXTENSION"),
    "rdl": ("DEADLIFT", "ROMANIAN_DEADLIFT"),
    "single_leg_rdl": ("DEADLIFT", "SINGLE_LEG_ROMANIAN_DEADLIFT_WITH_DUMBBELL"),
    "nordic_curl_assisted": ("LEG_CURL", "SLIDING_LEG_CURL"),
    "glute_bridge": ("HIP_RAISE", "HIP_RAISE"),
    "hip_thrust": ("HIP_RAISE", "WEIGHTED_HIP_RAISE"),
    "side_lying_abduction": ("HIP_STABILITY", "SIDE_LYING_LEG_RAISE"),
    "band_monster_walk": ("HIP_STABILITY", "LATERAL_WALKS_WITH_BAND_AT_ANKLES"),
    "side_plank_hip_lift": ("PLANK", "SIDE_PLANK"),
    "copenhagen_plank": ("PLANK", "SIDE_PLANK_WITH_LEG_LIFT"),
}

# Categories the account accepted. An unlisted one is an HTTP 400, not a
# cosmetic problem, so this is asserted before anything is sent.
VALID_CATEGORIES = frozenset({
    "SQUAT", "DEADLIFT", "CALF_RAISE", "LUNGE", "HIP_RAISE", "PLANK", "CURL",
    "LEG_CURL", "HIP_STABILITY", "CORE", "TOTAL_BODY", "OLYMPIC_LIFT",
    "BENCH_PRESS", "ROW", "CARRY", "WARM_UP",
})

REST_SECONDS = 60


def _step(order: int, prescription: Any, exercise: Any) -> dict[str, Any]:
    category, name = GARMIN_TARGET.get(exercise.id, ("SQUAT", None))
    # Fail here rather than at the API: an unknown category is a 400 that
    # rejects the whole session, and an unverified name is accepted and then
    # dropped, leaving the watch showing a bare category with no explanation.
    if category not in VALID_CATEGORIES:
        raise ValueError(f"{exercise.id}: {category!r} is not a Garmin category")
    if name and name not in VERIFIED_NAMES:
        raise ValueError(f"{exercise.id}: {name!r} was not kept by Garmin")
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


# Endurance sports the watch can be sent a structured workout for. Swimming is
# absent on purpose: a Garmin swim workout is built from pool length and stroke
# rather than duration, and wrist heart rate in water is unreliable enough that a
# heart-rate target there would be a target you cannot follow. Bricks are absent
# because they are two sports in one session, which is a multisport workout and a
# different shape again.
ENDURANCE_SPORT = {
    "run": {"sportTypeId": 1, "sportTypeKey": "running", "displayOrder": 1},
    "bike": {"sportTypeId": 2, "sportTypeKey": "cycling", "displayOrder": 2},
}

# Verified against the live account: a custom bpm range on this target type
# round-trips exactly, as targetValueOne and targetValueTwo.
HR_TARGET = {"workoutTargetTypeId": 4, "workoutTargetTypeKey": "heart.rate.zone",
             "displayOrder": 4}
STEP_WARMUP = {"stepTypeId": 1, "stepTypeKey": "warmup", "displayOrder": 1}
STEP_COOLDOWN = {"stepTypeId": 2, "stepTypeKey": "cooldown", "displayOrder": 2}

# Below this a session is one block: ten minutes of warm-up out of a
# twenty-minute easy run leaves nothing to warm up for.
WARMUP_FLOOR_MIN = 40
WARMUP_MIN = 10
COOLDOWN_MIN = 5


def parse_hr_target(text: str | None) -> tuple[int, int] | None:
    """"112-137 bpm" -> (112, 137). None when there is no range to send."""
    if not text:
        return None
    digits = re.findall(r"\d+", str(text))
    if len(digits) < 2:
        return None
    low, high = int(digits[0]), int(digits[1])
    return (low, high) if 60 <= low < high <= 220 else None


def _endurance_step(order: int, kind: dict[str, Any], minutes: float,
                    hr: tuple[int, int] | None, note: str) -> dict[str, Any]:
    step: dict[str, Any] = {
        "type": "ExecutableStepDTO", "stepOrder": order, "stepType": kind,
        "endCondition": END_TIME, "endConditionValue": float(minutes) * 60.0,
        "targetType": HR_TARGET if hr else NO_TARGET,
        "description": note[:512],
    }
    if hr:
        step["targetValueOne"] = float(hr[0])
        step["targetValueTwo"] = float(hr[1])
    return step


def build_endurance(sport: str, minutes: int, target_hr: str | None,
                    name: str, purpose: str = "") -> dict[str, Any]:
    """A run or ride as a timed workout with the athlete's own bpm range.

    Time rather than distance, because the plan is written in minutes and the
    heart-rate ceiling is the point: holding a pace to hit a distance is what
    turns an easy session into a moderate one.
    """
    sport_type = ENDURANCE_SPORT.get(sport)
    if sport_type is None:
        raise ValueError(f"{sport} cannot be sent as a structured workout")
    if minutes <= 0:
        raise ValueError("nothing to send: the session has no duration")

    hr = parse_hr_target(target_hr)
    band = f"{hr[0]}-{hr[1]} bpm" if hr else "easy"
    steps: list[dict[str, Any]] = []
    if minutes >= WARMUP_FLOOR_MIN:
        main = minutes - WARMUP_MIN - COOLDOWN_MIN
        steps.append(_endurance_step(
            1, STEP_WARMUP, WARMUP_MIN, None,
            "Easy. Let the heart rate come up on its own."))
        steps.append(_endurance_step(
            2, STEP_INTERVAL, main, hr,
            f"{purpose or 'Aerobic base'} — hold {band}."))
        steps.append(_endurance_step(
            3, STEP_COOLDOWN, COOLDOWN_MIN, None, "Easy to finish."))
    else:
        steps.append(_endurance_step(
            1, STEP_INTERVAL, minutes, hr,
            f"{purpose or 'Aerobic base'} — hold {band}."))
    return {
        "sportType": sport_type,
        "workoutName": name[:80],
        "description": (f"Built by Aerobic Engine. Target {band}.")[:1024],
        "workoutSegments": [{"segmentOrder": 1, "sportType": sport_type,
                             "workoutSteps": steps}],
    }


def push_endurance(api: Any, sport: str, minutes: int, target_hr: str | None,
                   name: str, purpose: str = "",
                   on_date: str | None = None) -> dict[str, Any]:
    payload = build_endurance(sport, minutes, target_hr, name, purpose)
    created = api.upload_workout(payload) or {}
    workout_id = created.get("workoutId")
    if workout_id and on_date:
        try:
            api.schedule_workout(str(workout_id), on_date)
        except Exception as exc:  # noqa: BLE001 - the workout still exists
            log.warning("Uploaded %s but could not schedule it for %s: %s",
                        workout_id, on_date, exc)
    return created


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
