"""Leg strength: a fixed exercise library and deterministic progression.

Two hard rules live here, and neither is negotiable by the AI layer:

  1. The library is closed. `plan_week` may pick exercise IDs from it and change
     sets/placement; it may not invent an exercise. `EXERCISES` is the allowlist
     that the planner validates against.
  2. Progression is arithmetic, not judgement. Load or hold time goes up only
     after a session that was completed cleanly and pain-free, one step at a
     time. Tendon injuries come from load *jumps*, not from load — so no
     plyometrics or jumping anywhere in base phase.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

from core.schemas import StrengthPrescription, StrengthState

STRENGTH = "strength"
ISOMETRIC = "isometric"


class Exercise:
    """One library entry with its own progression bounds."""

    def __init__(
        self,
        id: str,
        name: str,
        kind: str,
        target: str,
        sets: int,
        rep_range: tuple[int, int] | None = None,
        hold_range: tuple[int, int] | None = None,
        load_step_kg: float = 2.0,
        tempo: str = "",
        unilateral: bool = False,
        cue: str = "",
        progress_to: str | None = None,
    ) -> None:
        self.id = id
        self.name = name
        self.kind = kind
        self.target = target
        self.sets = sets
        self.rep_range = rep_range
        self.hold_range = hold_range
        self.load_step_kg = load_step_kg
        self.tempo = tempo
        self.unilateral = unilateral
        self.cue = cue
        self.progress_to = progress_to  # harder variant once the top is reached


# --- the library. Closed set; the AI cannot add to it. --------------------
EXERCISES: dict[str, Exercise] = {
    e.id: e
    for e in [
        Exercise(
            "calf_raise_straight",
            "Straight-leg calf raise",
            STRENGTH,
            "gastrocnemius / Achilles",
            sets=4,
            rep_range=(5, 8),
            load_step_kg=2.5,
            tempo="3s down, 1s pause",
            cue="Full range off a step; control the lowering.",
            progress_to="calf_raise_single_leg",
        ),
        Exercise(
            "calf_raise_bent",
            "Bent-knee calf raise (soleus)",
            STRENGTH,
            "soleus",
            sets=3,
            rep_range=(6, 10),
            load_step_kg=2.5,
            tempo="3s down",
            cue="Knee bent ~30 degrees throughout — this is the runner's calf.",
        ),
        Exercise(
            "calf_raise_single_leg",
            "Single-leg calf raise",
            STRENGTH,
            "Achilles (unilateral)",
            sets=3,
            rep_range=(5, 8),
            load_step_kg=2.0,
            tempo="3s down",
            unilateral=True,
            cue="Per side. Add load only once bodyweight reps are clean.",
        ),
        Exercise(
            "split_squat",
            "Split squat",
            STRENGTH,
            "quad / glute (unilateral)",
            sets=3,
            rep_range=(5, 8),
            load_step_kg=2.0,
            tempo="2s down",
            unilateral=True,
            cue="Per side. Torso tall, front shin near vertical.",
        ),
        Exercise(
            "reverse_lunge",
            "Reverse lunge",
            STRENGTH,
            "quad / glute (unilateral)",
            sets=3,
            rep_range=(6, 8),
            load_step_kg=2.0,
            tempo="controlled",
            unilateral=True,
            cue="Per side. Step back, not forward — kinder on the knee.",
        ),
        Exercise(
            "rdl",
            "Romanian deadlift",
            STRENGTH,
            "posterior chain",
            sets=3,
            rep_range=(5, 8),
            load_step_kg=5.0,
            tempo="3s down",
            cue="Hinge at the hip, flat back, feel the hamstring.",
            progress_to="single_leg_rdl",
        ),
        Exercise(
            "single_leg_rdl",
            "Single-leg RDL",
            STRENGTH,
            "posterior chain (unilateral)",
            sets=3,
            rep_range=(5, 8),
            load_step_kg=2.0,
            tempo="3s down",
            unilateral=True,
            cue="Per side. Hips square; balance is part of the exercise.",
        ),
        Exercise(
            "step_up",
            "Step-up",
            STRENGTH,
            "run-specific unilateral drive",
            sets=3,
            rep_range=(6, 8),
            load_step_kg=2.0,
            tempo="no push off the trailing leg",
            unilateral=True,
            cue="Per side. Knee-height step; drive through the top leg only.",
        ),
        Exercise(
            "wall_sit",
            "Wall sit",
            ISOMETRIC,
            "knee tendon",
            sets=5,
            hold_range=(30, 45),
            cue="Thighs parallel to the floor. Breathe.",
            progress_to="spanish_squat",
        ),
        Exercise(
            "spanish_squat",
            "Spanish squat (band)",
            ISOMETRIC,
            "patellar tendon",
            sets=5,
            hold_range=(30, 45),
            cue="Band behind the knees, sit back into it, shins vertical.",
        ),
        Exercise(
            "single_leg_calf_hold",
            "Single-leg calf-raise hold",
            ISOMETRIC,
            "Achilles",
            sets=4,
            hold_range=(30, 45),
            unilateral=True,
            cue="Per side. Top of the range, heel high, still.",
        ),
        Exercise(
            "tib_raise",
            "Tibialis raise",
            STRENGTH,
            "tibialis anterior (shin-splint insurance)",
            sets=3,
            rep_range=(10, 15),
            load_step_kg=1.0,
            tempo="slow both ways",
            cue="Back against a wall, toes up. Cheap insurance.",
        ),
    ]
}

LIBRARY_IDS: frozenset[str] = frozenset(EXERCISES)

# Two alternating templates: A is calf/quad dominant, B posterior/unilateral.
SESSION_A: tuple[str, ...] = (
    "calf_raise_straight",
    "split_squat",
    "rdl",
    "wall_sit",
    "tib_raise",
)
SESSION_B: tuple[str, ...] = (
    "calf_raise_bent",
    "step_up",
    "single_leg_rdl",
    "single_leg_calf_hold",
    "tib_raise",
)

# Exercises to drop first when readiness is low or a joint is complaining.
DELOAD_DROP_ORDER: tuple[str, ...] = ("tib_raise", "step_up", "rdl", "single_leg_rdl")

PHYSIO_NOTE = (
    "Pain (not muscle soreness) logged on the same exercise twice or more. "
    "This app is not medical advice — persistent tendon pain is a physio visit, "
    "not a programming problem."
)


# Garmin's strength mode records its own exercise taxonomy. Mapping it onto the
# library lets a session logged on the watch populate the progression state
# automatically — the AI still cannot introduce anything outside the library,
# because this maps *into* it and drops whatever does not fit.
GARMIN_EXERCISE_MAP: dict[str, str] = {
    # calves
    "STANDING_CALF_RAISE": "calf_raise_straight",
    "CALF_RAISE": "calf_raise_straight",
    "DOUBLE_CALF_RAISE": "calf_raise_straight",
    "WEIGHTED_STANDING_CALF_RAISE": "calf_raise_straight",
    "SEATED_CALF_RAISE": "calf_raise_bent",
    "WEIGHTED_SEATED_CALF_RAISE": "calf_raise_bent",
    "SINGLE_LEG_CALF_RAISE": "calf_raise_single_leg",
    "SINGLE_LEG_STANDING_CALF_RAISE": "calf_raise_single_leg",
    # single-leg squat patterns
    "SPLIT_SQUAT": "split_squat",
    "BULGARIAN_SPLIT_SQUAT": "split_squat",
    "WEIGHTED_SPLIT_SQUAT": "split_squat",
    "REVERSE_LUNGE": "reverse_lunge",
    "LUNGE": "reverse_lunge",
    "WEIGHTED_LUNGE": "reverse_lunge",
    "DUMBBELL_LUNGE": "reverse_lunge",
    # posterior chain
    "ROMANIAN_DEADLIFT": "rdl",
    "STRAIGHT_LEG_DEADLIFT": "rdl",
    "DEADLIFT": "rdl",
    "DUMBBELL_DEADLIFT": "rdl",
    "SINGLE_LEG_DEADLIFT": "single_leg_rdl",
    "SINGLE_LEG_ROMANIAN_DEADLIFT": "single_leg_rdl",
    # unilateral drive
    "STEP_UP": "step_up",
    "WEIGHTED_STEP_UP": "step_up",
    "BOX_STEP_UP": "step_up",
    # isometrics
    "WALL_SIT": "wall_sit",
    "WEIGHTED_WALL_SIT": "wall_sit",
    "SPANISH_SQUAT": "spanish_squat",
    # shins
    "TIBIALIS_RAISE": "tib_raise",
    "TOE_RAISE": "tib_raise",
    "ANKLE_DORSIFLEXION": "tib_raise",
}


def map_garmin_exercise(category: str | None, name: str | None) -> str | None:
    """Best-effort match from Garmin's taxonomy into the library.

    Tries the specific exercise name first, then the broader category, then a
    substring match. Returns None rather than guessing — an unmapped set is
    surfaced in the UI for manual assignment instead of being silently binned.
    """
    for raw in (name, category):
        if not raw:
            continue
        key = str(raw).strip().upper().replace(" ", "_").replace("-", "_")
        if key in GARMIN_EXERCISE_MAP:
            return GARMIN_EXERCISE_MAP[key]
    haystack = (
        f"{name or ''} {category or ''}".upper().replace(" ", "_").replace("-", "_")
    )
    for needle, exercise_id in (
        ("SINGLE_LEG_CALF", "calf_raise_single_leg"),
        ("SEATED_CALF", "calf_raise_bent"),
        ("CALF", "calf_raise_straight"),
        ("SINGLE_LEG_DEAD", "single_leg_rdl"),
        ("DEADLIFT", "rdl"),
        ("SPLIT_SQUAT", "split_squat"),
        ("LUNGE", "reverse_lunge"),
        ("STEP_UP", "step_up"),
        ("WALL_SIT", "wall_sit"),
        ("SPANISH", "spanish_squat"),
        ("TIBIALIS", "tib_raise"),
    ):
        if needle in haystack:
            return exercise_id
    return None


def sets_to_log_rows(
    day: str, activity_id: str, sets: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Collapse per-set watch records into one strength_log row per exercise.

    The log stores a session's work per exercise, so N sets of an exercise become
    one row: set count, the top weight used, and the modal rep count.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for s in sets:
        ex = s.get("exercise_id")
        if ex:
            grouped.setdefault(ex, []).append(s)

    rows = []
    for exercise_id, items in grouped.items():
        reps = [int(i["reps"]) for i in items if i.get("reps")]
        loads = [float(i["load_kg"]) for i in items if i.get("load_kg")]
        holds = [float(i["duration_s"]) for i in items if i.get("duration_s")]
        is_iso = EXERCISES[exercise_id].kind == ISOMETRIC
        rows.append(
            {
                "day": day,
                "activity_id": activity_id,
                "exercise_id": exercise_id,
                "sets": len(items),
                "reps": max(set(reps), key=reps.count) if reps and not is_iso else None,
                "hold_s": int(max(holds)) if holds and is_iso else None,
                "load_kg": max(loads) if loads else None,
                # The watch cannot know whether it hurt; assume clean and let the
                # athlete flag pain in the UI.
                "clean": 1,
                "pain": 0,
                "notes": "imported from watch strength mode",
            }
        )
    return rows


def validate_exercise_ids(ids: Sequence[str]) -> list[str]:
    """Drop anything the AI invented. Order preserved, duplicates removed."""
    seen: set[str] = set()
    out: list[str] = []
    for i in ids:
        key = str(i).strip().lower()
        if key in LIBRARY_IDS and key not in seen:
            seen.add(key)
            out.append(key)
    return out


# --------------------------------------------------------------------------
# Deterministic progression
# --------------------------------------------------------------------------


def last_entries(
    log: Sequence[dict[str, Any]], exercise_id: str, n: int = 3
) -> list[dict[str, Any]]:
    rows = [r for r in log if r.get("exercise_id") == exercise_id]
    rows.sort(key=lambda r: (str(r.get("day")), r.get("id") or 0))
    return rows[-n:]


def next_prescription(
    exercise_id: str, log: Sequence[dict[str, Any]], intensity: float = 1.0
) -> StrengthPrescription:
    """One step up if the last session was clean and pain-free; else hold or back off.

    `intensity` (0.5-1.0) is the planner's dial for a low-readiness day: it cuts
    sets, never load, so the tissue still sees the stimulus it has adapted to.
    """
    ex = EXERCISES[exercise_id]
    history = last_entries(log, exercise_id, n=2)
    last = history[-1] if history else None

    sets = max(2, round(ex.sets * intensity))
    reps = ex.rep_range[0] if ex.rep_range else None
    hold = ex.hold_range[0] if ex.hold_range else None
    load = None
    note = "First session at this exercise — establish a baseline, stay comfortable."

    if last:
        prev_reps = last.get("reps") or reps
        prev_hold = last.get("hold_s") or hold
        prev_load = last.get("load_kg")
        clean = bool(last.get("clean", 1)) and not bool(last.get("pain", 0))

        if not clean:
            reps, hold, load = prev_reps, prev_hold, prev_load
            note = (
                "Last session was flagged incomplete or painful — repeat it at the "
                "same load before adding anything."
            )
            if last.get("pain"):
                load = _back_off(prev_load)
                note = "Pain flagged last time — back off ~10% and reassess."
        elif ex.rep_range:
            lo, hi = ex.rep_range
            if prev_reps is not None and prev_reps < hi:
                reps, load = prev_reps + 1, prev_load
                note = f"Clean last time — add one rep (target {reps}/set)."
            else:
                reps = lo
                load = (prev_load or 0.0) + ex.load_step_kg
                note = (
                    f"Top of the rep range hit — add {ex.load_step_kg:g} kg and "
                    f"reset to {lo} reps."
                )
        elif ex.hold_range:
            lo, hi = ex.hold_range
            if prev_hold is not None and prev_hold < hi:
                hold, load = min(hi, prev_hold + 5), prev_load
                note = f"Clean last time — hold {hold}s."
            else:
                hold = lo
                load = (prev_load or 0.0) + ex.load_step_kg
                note = (
                    f"Top of the hold range reached — add {ex.load_step_kg:g} kg and "
                    f"reset to {lo}s."
                )

    return StrengthPrescription(
        exercise_id=ex.id,
        name=ex.name,
        sets=sets,
        reps=reps,
        hold_s=hold,
        load_kg=round(load, 1) if load else load,
        tempo=ex.tempo,
        note=f"{ex.cue} {note}".strip(),
    )


def _back_off(load: float | None) -> float | None:
    if not load:
        return load
    return round(load * 0.9, 1)


def pain_flags(log: Sequence[dict[str, Any]], lookback: int = 6) -> list[str]:
    """Exercises with pain logged more than once in the recent history."""
    counts: dict[str, int] = {}
    for row in sorted(log, key=lambda r: str(r.get("day")))[-lookback * 6 :]:
        if row.get("pain"):
            counts[row["exercise_id"]] = counts.get(row["exercise_id"], 0) + 1
    return sorted([k for k, v in counts.items() if v >= 2])


def session_template(session_index: int) -> tuple[str, ...]:
    """Alternate A/B so both calf variants and both hinge patterns get worked."""
    return SESSION_A if session_index % 2 == 0 else SESSION_B


def build_session(
    log: Sequence[dict[str, Any]],
    session_index: int = 0,
    intensity: float = 1.0,
    avoid: Sequence[str] = (),
) -> list[StrengthPrescription]:
    """The prescription for one strength session."""
    avoid = set(avoid) | set(pain_flags(log))
    ids = [i for i in session_template(session_index) if i not in avoid]

    if intensity < 0.7:
        # Low readiness: shed accessory work, keep the tendon isometrics.
        for drop in DELOAD_DROP_ORDER:
            if len(ids) <= 3:
                break
            if drop in ids and EXERCISES[drop].kind != ISOMETRIC:
                ids.remove(drop)

    return [next_prescription(i, log, intensity) for i in ids]


def strength_state(
    log: Sequence[dict[str, Any]],
    session_index: int = 0,
    intensity: float = 1.0,
) -> StrengthState:
    """What the planner and the AI layer see. Read-only for the AI."""
    days = sorted({str(r["day"]) for r in log})
    return StrengthState(
        sessions_logged=len(days),
        last_session_date=datetime.fromisoformat(days[-1]).date() if days else None,
        pain_flagged=pain_flags(log),
        prescription=build_session(log, session_index, intensity),
    )


def session_minutes(prescriptions: Sequence[StrengthPrescription]) -> int:
    """Rough wall-clock for a prescription, clamped to the 20-30 min brief."""
    total = 4.0  # warm-up
    for p in prescriptions:
        per_set = (p.hold_s or 40) / 60.0 + 0.75 if p.hold_s else 1.2
        # Per-side work is not quite double: the other leg rests during it.
        total += p.sets * per_set * (1.7 if EXERCISES[p.exercise_id].unilateral else 1)
    return int(min(32, max(18, round(total))))


def needs_physio_note(log: Sequence[dict[str, Any]]) -> bool:
    return bool(pain_flags(log))
