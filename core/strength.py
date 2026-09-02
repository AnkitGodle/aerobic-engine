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

from collections.abc import Mapping, Sequence
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
        focus: str = "",
        setup: str = "",
        steps: tuple[str, ...] = (),
        mistakes: str = "",
        why: str = "",
        load_note: str = "",
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
        # What the exercise is for, and how to actually do it. Held here rather
        # than in the UI because a prescription without a technique is not
        # something you can follow, and bad technique is how strength work causes
        # the injury it was meant to prevent.
        self.focus = focus
        self.setup = setup
        self.steps = steps
        self.mistakes = mistakes
        self.why = why
        self.load_note = load_note


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
            focus="calf / Achilles",
            setup="Stand on the edge of a step with the balls of both feet on it and heels hanging free. Fingertips on a wall or rail for balance only.",
            steps=(
                "Rise as high as you can onto your toes. Pause one second at the top.",
                "Lower over three slow seconds until your heels are below the step and you feel a stretch through the calf and Achilles.",
                "Pause briefly at the bottom, then rise again.",
            ),
            mistakes="Bouncing out of the bottom, or cutting the range short at either end. The slow lowering is the part the tendon responds to.",
            why="The Achilles takes several times your bodyweight on every running stride. This is the single most protective exercise for a runner.",
            load_note="Hold a dumbbell or a loaded rucksack once 4x8 bodyweight reps feel controlled.",
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
            focus="calf / Achilles",
            setup="Same step position, but bend your knees to about 30 degrees and keep them bent throughout. Sitting on a chair with weight on your thighs works too.",
            steps=(
                "With knees held bent, push through the balls of your feet to raise your heels as high as they will go.",
                "Lower over three seconds, knees still bent.",
            ),
            mistakes="Letting the knees straighten as you rise — that shifts the work back to the gastrocnemius and misses the point.",
            why="Bending the knee takes the gastrocnemius out of the movement and isolates the soleus, which carries most of the load in distance running and is usually the weak link.",
            load_note="Weight across the thighs if seated, or a rucksack if standing.",
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
            focus="calf / Achilles",
            setup="One foot on the step edge, the other foot lifted clear. Fingertips on a wall for balance.",
            steps=(
                "Rise onto the toes of the working leg as high as possible.",
                "Pause one second at the top.",
                "Lower over three seconds to a full stretch below the step.",
                "Complete all reps, then switch legs.",
            ),
            mistakes="Pushing through the wall with your arms, or letting the hip drop on the working side.",
            why="Running is a single-leg activity, and side-to-side differences are common and invisible until they cause an injury.",
            load_note="Bodyweight is plenty at first. Add a light dumbbell only when all reps on the weaker side are clean.",
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
            focus="quad / knee",
            setup="Stand in a long stride, front foot flat, back heel raised. Feet about hip-width apart, not on a tightrope.",
            steps=(
                "Lower straight down over two seconds until the back knee is just above the floor.",
                "Keep your torso tall and the front shin close to vertical.",
                "Drive up through the front foot. Do not push off the back toes.",
                "Complete all reps, then switch legs.",
            ),
            mistakes="Leaning forward, letting the front knee travel past the toes, or using the back leg to help.",
            why="Builds single-leg quad and glute strength, which controls how much the knee collapses inward on landing.",
            load_note="A dumbbell in each hand, or one held at the chest.",
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
            focus="quad / knee",
            setup="Stand tall, feet hip-width.",
            steps=(
                "Step one foot back and lower until both knees are bent about 90 degrees.",
                "Keep your weight over the front foot throughout.",
                "Push through the front foot to return to standing.",
                "Complete all reps on one side, then switch.",
            ),
            mistakes="Stepping forward instead of back, which loads the knee harder, and letting the front heel lift.",
            why="Same benefit as the split squat with a balance demand and less knee stress, which is why it is the reverse rather than forward version.",
            load_note="Dumbbells at your sides.",
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
            focus="hamstring / glute",
            setup="Stand holding a weight in front of your thighs, feet hip-width, knees softly bent.",
            steps=(
                "Push your hips backwards, letting the weight travel down your legs.",
                "Keep your back flat and your shins nearly vertical.",
                "Stop when you feel a strong stretch in the hamstrings, usually around mid-shin. Depth is not the goal.",
                "Drive your hips forward to stand up, squeezing the glutes.",
            ),
            mistakes="Rounding the back, squatting instead of hinging, or chasing depth past the point where your back can stay flat.",
            why="Strong hamstrings and glutes reduce load on the calf and Achilles and protect the hamstring itself, a common runner's injury.",
            load_note="Dumbbells, a barbell, or a rucksack held to the chest.",
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
            focus="hamstring / glute",
            setup="Stand on one leg holding a weight in the opposite hand.",
            steps=(
                "Hinge at the hip, letting the free leg extend straight behind you as a counterweight.",
                "Keep your hips square to the floor — the biggest challenge here.",
                "Lower until you feel the hamstring load, then return to standing.",
                "Complete all reps, then switch.",
            ),
            mistakes="Letting the hip of the free leg rotate open towards the ceiling.",
            why="Trains the hamstring and glute while the hip resists rotation, which is exactly what they do during the stance phase of running.",
            load_note="Light. Balance should be the limit before load is.",
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
            focus="glute / drive",
            setup="Stand facing a knee-height step or bench.",
            steps=(
                "Place one whole foot on the step.",
                "Drive up through that leg until you are standing on the step.",
                "Do not push off the bottom foot — it should feel almost passive.",
                "Lower slowly under control. Complete all reps, then switch.",
            ),
            mistakes="Pushing off the trailing foot, or hopping up. If you cannot rise without help, the step is too high.",
            why="Closest strength movement to the drive phase of running, and it builds the glute strength that keeps the pelvis level.",
            load_note="Dumbbells at your sides once bodyweight reps are easy.",
        ),
        Exercise(
            "wall_sit",
            "Wall sit",
            ISOMETRIC,
            "knee tendon",
            sets=5,
            hold_range=(30, 45),
            cue="Thighs parallel to the floor. Breathe.",
            focus="quad / knee",
            setup="Back flat against a wall, feet a stride out in front, hip-width.",
            steps=(
                "Slide down until your thighs are parallel to the floor and your knees are above your ankles.",
                "Hold. Breathe normally rather than holding your breath.",
                "Stand up between sets and rest about a minute.",
            ),
            mistakes="Sitting too shallow to be hard, or letting the knees drift past the toes.",
            why="A long isometric hold loads the quad tendon heavily with no impact at all, which is why it is safe to do near hard running days.",
            load_note="Add time before adding weight. A plate on the thighs comes later.",
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
            focus="quad / knee",
            setup="Loop a resistance band around a solid post at knee height and around the back of both knees. Step back until the band is taut.",
            steps=(
                "Let the band pull your knees back as you sit down into a squat.",
                "Keep your shins vertical — the band makes this possible.",
                "Hold at about parallel, chest up.",
            ),
            mistakes="Too little band tension, which turns it into an ordinary squat and loses the point.",
            why="Loads the patellar tendon hard while keeping the shins vertical, so the knee joint itself stays comfortable. The standard rehab exercise for runner's knee pain.",
            load_note="A heavier band, or hold a weight at the chest.",
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
            focus="calf / Achilles",
            setup="Stand on one foot on flat ground or a step, fingertips on a wall.",
            steps=(
                "Rise to the very top of the calf raise, heel as high as it goes.",
                "Hold still at the top. Do not let the heel sink.",
                "Complete the hold, then switch legs.",
            ),
            mistakes="Slowly sinking during the hold. When the heel drops, the set is over, even if the clock has not run out.",
            why="Isometric holds reduce tendon pain and build capacity without the repeated loading of reps, which makes this the go-to when the Achilles is grumbling.",
            load_note="Hold a dumbbell once 4x45s per side is comfortable.",
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
            focus="shin",
            setup="Stand with your back and hips against a wall, heels about a foot from it, legs straight.",
            steps=(
                "Pull your toes up towards your shins as far as they will go.",
                "Lower slowly, feeling the front of the shin work.",
            ),
            mistakes="Bending the knees or shuffling the feet closer to the wall to make it easier.",
            why="The tibialis anterior decelerates your foot on every landing. It is the muscle behind most shin splints, and almost nobody trains it.",
            load_note="Bodyweight is enough for a long time; a light ankle weight later.",
        ),
    ]
}


# --- glute, knee and hamstring work ---------------------------------------
# Added because the original library was calf-and-hinge heavy. Weak glutes let
# the pelvis drop and the knee fall inward on every stride, which is the usual
# mechanism behind runner's knee and IT band pain; the knee-specific entries load
# the patellar tendon with the shin kept vertical so the joint stays comfortable.
# Everything here is slow and loaded. Nothing jumps: tendon injuries come from
# load jumps, not from load, so base phase has no plyometrics.
EXERCISES.update({e.id: e for e in [
    Exercise(
        "glute_bridge", "Glute bridge", STRENGTH, "glutes",
        sets=3, rep_range=(8, 12), load_step_kg=5.0, tempo="2s up, 1s squeeze",
        cue="Ribs down, squeeze at the top rather than arching the back.",
        focus="glute / hip",
        progress_to="hip_thrust",
        setup="Lie on your back, knees bent, heels about a hand's length from "
              "your backside, arms flat at your sides.",
        steps=(
            "Press through your heels and lift your hips until your body makes a "
            "straight line from knees to shoulders.",
            "Squeeze the glutes hard for a second at the top.",
            "Lower under control without letting the hips touch down between reps.",
        ),
        mistakes="Arching the lower back to get higher, which turns it into a "
                 "back exercise, or pushing through the toes instead of the heels.",
        why="The glutes extend the hip on every stride and every pedal push. When "
            "they are weak the hamstrings and lower back take over.",
        load_note="A weight across the hips once 3x12 bodyweight is easy.",
    ),
    Exercise(
        "hip_thrust", "Hip thrust", STRENGTH, "glutes (loaded)",
        sets=3, rep_range=(6, 10), load_step_kg=5.0, tempo="2s up, 1s squeeze",
        cue="Shoulders on a bench or sofa edge; chin tucked, ribs down.",
        focus="glute / hip",
        setup="Sit on the floor with your upper back against a sofa or bench, "
              "knees bent, feet flat and hip-width. Rest a weight across your hips.",
        steps=(
            "Drive through your heels to lift your hips until your torso is "
            "parallel to the floor.",
            "Squeeze the glutes for a second at the top with your chin tucked.",
            "Lower under control to just above the floor.",
        ),
        mistakes="Letting the ribs flare and the back arch at the top, or letting "
                 "the knees fall inward.",
        why="The heaviest glute exercise you can do with minimal kit, and stronger "
            "glutes are what keep the pelvis level when you get tired.",
        load_note="Dumbbell or loaded rucksack across the hips; add 5 kg at a time.",
    ),
    Exercise(
        "side_lying_abduction", "Side-lying leg raise", STRENGTH,
        "gluteus medius (unilateral)",
        sets=3, rep_range=(10, 15), load_step_kg=1.0, tempo="slow, controlled",
        unilateral=True,
        cue="Per side. Toes pointed slightly down, not up.",
        focus="glute / hip",
        setup="Lie on your side in a straight line, bottom knee bent for balance, "
              "top leg straight.",
        steps=(
            "Raise the top leg about 45 degrees, leading with the heel.",
            "Rotate the toes slightly towards the floor as you lift — that is what "
            "targets the gluteus medius rather than the hip flexor.",
            "Lower slowly. Complete all reps, then switch sides.",
        ),
        mistakes="Rolling the hips backwards, or leading with the toes, both of "
                 "which hand the work to the wrong muscle.",
        why="The gluteus medius stops the pelvis dropping and the knee collapsing "
            "inward on landing. It is the single most common weakness in runners "
            "with knee pain.",
        load_note="An ankle weight once 3x15 is easy.",
    ),
    Exercise(
        "band_monster_walk", "Banded lateral walk", STRENGTH,
        "gluteus medius / hip stability",
        sets=3, rep_range=(10, 15), load_step_kg=0.0, tempo="controlled",
        cue="Steps per direction. Keep tension on the band throughout.",
        focus="glute / hip",
        setup="Loop a resistance band around both legs just above the knees. "
              "Stand in a quarter squat, feet hip-width, chest up.",
        steps=(
            "Step sideways against the band, keeping the knees pushed out.",
            "Bring the trailing foot in without letting the band go slack.",
            "Complete the reps in one direction, then come back the other way.",
        ),
        mistakes="Standing upright, letting the knees drop inward, or bobbing up "
                 "and down between steps.",
        why="Trains the hip to resist the knee falling inward while you are "
            "moving, which is closer to what happens when you run than a static "
            "hold is.",
        load_note="A stiffer band. Do not chase reps past 15.",
    ),
    Exercise(
        "terminal_knee_extension", "Terminal knee extension (band)", STRENGTH,
        "quad / patellar tendon",
        sets=3, rep_range=(10, 15), load_step_kg=0.0, tempo="2s, 1s squeeze",
        cue="Band pulling the knee forward; straighten against it.",
        focus="quad / knee",
        setup="Anchor a band at knee height in front of you and loop it around the "
              "back of one knee. Step back until it is taut, that foot flat.",
        steps=(
            "Let the band pull your knee into a slight bend.",
            "Straighten the knee fully against the band and squeeze the quad.",
            "Release slowly. Complete all reps, then switch legs.",
        ),
        mistakes="Locking out by leaning back rather than by using the quad.",
        why="Strengthens the last few degrees of knee extension, which is where "
            "the quad is weakest and where the patellar tendon is most loaded.",
        load_note="A stiffer band, or stand further back.",
    ),
    Exercise(
        "step_down", "Step-down", STRENGTH, "quad / knee control (unilateral)",
        sets=3, rep_range=(6, 10), load_step_kg=2.0, tempo="3s down",
        unilateral=True,
        cue="Per side. Lower slowly; the descent is the exercise.",
        focus="quad / knee",
        setup="Stand on a step on one leg, the other foot hanging off the side.",
        steps=(
            "Bend the standing knee and lower the free heel slowly towards the "
            "floor over three seconds.",
            "Tap lightly and press back up, keeping the standing knee tracking "
            "over the middle of the foot.",
            "Complete all reps, then switch legs.",
        ),
        mistakes="Dropping quickly, or letting the standing knee dive inward. If "
                 "the knee cannot stay in line, use a lower step.",
        why="Trains the eccentric control that decelerates you on every landing "
            "and on every downhill, which is where knee pain usually starts.",
        load_note="A lower step first, then a higher one, then hold a dumbbell.",
    ),
    Exercise(
        "nordic_curl_assisted", "Assisted Nordic curl", STRENGTH,
        "hamstring (eccentric)",
        sets=3, rep_range=(4, 6), load_step_kg=0.0, tempo="as slow as possible",
        cue="Lower as slowly as you can; use your hands to catch yourself.",
        focus="hamstring / glute",
        setup="Kneel on something padded with your heels held down — under a sofa, "
              "or by a partner. Hands ready in front of your chest.",
        steps=(
            "Keeping your hips extended and body straight, lower your chest "
            "towards the floor as slowly as you can control.",
            "Catch yourself with your hands when you lose control.",
            "Push back up with your hands, then reset.",
        ),
        mistakes="Bending at the hips, which makes it much easier and misses the "
                 "hamstring, or starting this while a hamstring is sore.",
        why="The strongest known protection against hamstring strain. Start with "
            "very few reps: this causes real soreness for a week or two.",
        load_note="Add range and reps before anything else. Cap at 6 reps.",
    ),
    Exercise(
        "goblet_squat", "Goblet squat", STRENGTH, "quad / glute",
        sets=3, rep_range=(6, 10), load_step_kg=2.5, tempo="2s down",
        cue="Weight at the chest, elbows inside the knees at the bottom.",
        focus="quad / knee",
        setup="Hold a dumbbell or kettlebell against your chest, feet a little "
              "wider than hip-width, toes slightly out.",
        steps=(
            "Sit down between your feet, keeping your chest up and heels planted.",
            "Descend until your thighs are at least parallel, elbows tracking "
            "inside the knees.",
            "Drive up through the whole foot.",
        ),
        mistakes="Heels lifting, or the knees collapsing inward on the way up.",
        why="The most efficient way to load both legs at once, and the base of "
            "cycling power as well as running strength.",
        load_note="Add 2.5 kg once 3x10 is controlled.",
    ),
    Exercise(
        "side_plank_hip_lift", "Side plank with hip lift", ISOMETRIC,
        "hip stability (unilateral)",
        sets=3, hold_range=(20, 40), tempo="hold still", unilateral=True,
        cue="Per side. Body in one straight line; do not let the hip sag.",
        focus="glute / hip",
        setup="Lie on your side propped on a forearm, elbow under the shoulder, "
              "knees bent or legs straight for a harder version.",
        steps=(
            "Lift your hips until your body is a straight line from shoulder to "
            "knee or ankle.",
            "Hold, breathing normally, without letting the hip drift down or "
            "the shoulders rotate.",
            "Complete the hold, then switch sides.",
        ),
        mistakes="Letting the top shoulder roll forward, or the hip sinking as the "
                 "hold goes on.",
        why="The muscles down the side of the hip are what stop your torso "
            "swaying with each stride. Sway wastes energy and loads the knee.",
        load_note="Straighten the legs, then add time. Weight comes last.",
    ),
    Exercise(
        "copenhagen_plank", "Copenhagen plank (short)", ISOMETRIC,
        "adductor / groin (unilateral)",
        sets=3, hold_range=(10, 25), tempo="hold still", unilateral=True,
        cue="Per side. Start with the bottom knee down and short holds.",
        focus="glute / hip",
        setup="Side plank on a forearm with the top leg resting on a chair or "
              "bench at about knee height. Bottom knee on the floor to start.",
        steps=(
            "Press the top leg down into the bench to lift your hips.",
            "Hold with the body in a straight line.",
            "Complete the hold, then switch sides.",
        ),
        mistakes="Starting with the full straight-leg version. This one causes "
                 "groin strains when rushed — build up over months, not weeks.",
        why="The adductors stabilise the hip and are a common cycling and running "
            "weakness. Short holds are deliberate here.",
        load_note="Extend the bottom leg only once 3x25s with the knee down is easy.",
    ),
]})

# Cadence work, kept out of EXERCISES on purpose. These are neuromuscular drills
# with no external load, so they must not go through the sets-and-reps
# progression or count towards a strength session's minutes. They belong inside a
# run or a ride, as a finisher.
#
# Nothing here bounces or jumps: the base-phase ban on plyometrics is about
# impact, and raising cadence is done by taking quicker, shorter steps rather
# than by hopping.
DRILLS: dict[str, dict[str, Any]] = {
    # How to raise cadence, in the order that works. Deliberately not "just run
    # faster": cadence rises with pace on its own, so a faster session shows a
    # higher number without the stride having changed at all. The point is a
    # quicker turnover at the *same* pace.
    "run_cadence_strides": {
        "name": "Cadence strides",
        "focus": "cadence",
        "where": "End of an easy run",
        "dose": "4-6 x 20 seconds, walking back to full recovery between",
        "setup": "Flat ground, after an easy run. Set a metronome app to 5 spm "
                 "above your usual cadence.",
        "steps": (
            "Run 20 seconds taking quicker, shorter steps in time with the beat.",
            "Keep the effort easy — this is about foot speed, not pace.",
            "Walk until you are fully recovered, then repeat.",
        ),
        "mistakes": "Reaching further forward with each step. Higher cadence means "
                    "the foot lands closer to underneath you, not further ahead.",
        "why": "Most runners land with the foot too far in front, which brakes and "
               "loads the knee. Raising cadence by 5-10% shortens the stride and "
               "reduces load per step without any loss of speed.",
    },
    "metronome_easy_run": {
        "name": "Metronome easy run",
        "focus": "cadence",
        "where": "A whole easy run, once a week",
        "dose": "10-20 minutes of the run, at 5% above your usual cadence",
        "setup": "Find your normal cadence from a recent easy run, add 5%, and "
                 "set a metronome app or your watch alert to that.",
        "steps": (
            "Run at your usual easy effort, not your usual pace — the pace will "
            "drop at first and that is correct.",
            "Take shorter, quicker steps in time with the beat. Let the foot land "
            "under your hip rather than out in front.",
            "Do it for 10 minutes at first. Stop matching the beat if your "
            "breathing rises: this is a technique change, not a workout.",
        ),
        "mistakes": "Jumping straight to 180. A 5% step is what the body adapts "
                    "to; a 20% jump just makes you bounce.",
        "why": "Cadence is a habit held by the nervous system, so it changes with "
               "repetition rather than with effort. One deliberate run a week "
               "moves it in a few weeks, and it holds.",
    },
    "hill_walk_backs": {
        "name": "Short hill repeats, walk down",
        "focus": "cadence",
        "where": "Instead of one easy run, every other week",
        "dose": "6-8 x 30 seconds uphill, walking all the way back down",
        "setup": "A gentle hill, 30 seconds of running long. Warm up first.",
        "steps": (
            "Run up at a strong but controlled effort, taking short quick steps.",
            "Walk back down. The walk is not optional — it is what keeps this out "
            "of interval territory.",
        ),
        "mistakes": "Running the descent. Downhill running is the highest-impact "
                    "thing a runner can do and has no place in base phase.",
        "why": "A hill forces a short stride and a quick turnover without you "
               "having to think about it, and the incline removes most of the "
               "landing impact.",
    },
    "bike_spin_ups": {
        "name": "Spin-ups",
        "focus": "cadence",
        "where": "Inside an easy ride",
        "dose": "4 x 1 minute at 100-110 rpm, 2 minutes easy between",
        "setup": "An easy gear, flat road or trainer, already warmed up.",
        "steps": (
            "Spin up to 100-110 rpm in a gear light enough that it stays easy.",
            "Hold it for a minute with a still upper body and no bouncing.",
            "Return to your normal cadence for two minutes, then repeat.",
        ),
        "mistakes": "Using a heavy gear, which turns a cadence drill into an "
                    "interval and makes the ride no longer easy.",
        "why": "A higher self-selected cadence shifts load off the muscles and "
               "onto the cardiovascular system, which is exactly what base phase "
               "is trying to develop.",
    },
    "single_leg_pedalling": {
        "name": "Single-leg pedalling",
        "focus": "cadence",
        "where": "Inside an easy ride, on a trainer",
        "dose": "3 x 30 seconds per leg, easy spinning between",
        "setup": "On a turbo trainer in a light gear. Unclip one foot and rest it "
                 "on a box or the frame.",
        "steps": (
            "Pedal with one leg only, aiming for a smooth circle rather than "
            "stamping down.",
            "Listen for the dead spot at the top and bottom of the stroke and try "
            "to smooth it out.",
            "Swap legs. Spin easy with both legs between sets.",
        ),
        "mistakes": "Grinding a heavy gear, or continuing once the stroke has gone "
                    "lumpy — the point is the smoothness, so stop when it goes.",
        "why": "Exposes side-to-side differences and the dead spots in your pedal "
               "stroke, which is where cycling efficiency is lost.",
    },
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
# The glute and knee session. A third template rather than more exercises in the
# first two: five movements is what fits the 20-30 minute brief, and a session
# that overruns is one that gets skipped. Two sessions a week means this cycle
# covers everything about every ten days.
SESSION_C: tuple[str, ...] = (
    "glute_bridge",
    "side_lying_abduction",
    "step_down",
    "spanish_squat",
    "tib_raise",
)

# Exercises to drop first when readiness is low or a joint is complaining.
# Dropped first when readiness is low or a joint is complaining. Accessory work
# goes before anything that loads a tendon, and the isometrics never go at all.
DELOAD_DROP_ORDER: tuple[str, ...] = (
    "tib_raise", "band_monster_walk", "side_lying_abduction", "step_up",
    "step_down", "glute_bridge", "rdl", "single_leg_rdl", "nordic_curl_assisted",
)

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
    "SINGLE_LEG_BENT_KNEE_CALF_RAISE": "calf_raise_bent",
    # Garmin files ankle dorsiflexion under WARM_UP; it is the only name it has
    # for the front of the shin, and it is what this app sends for a tibialis
    # raise. Without this line the set came back as a calf raise — the muscle a
    # tibialis raise exists to balance.
    "ANKLE_DORSIFLEXION_WITH_BAND": "tib_raise",
    "ANKLE_DORSIFLEXION": "tib_raise",
    # single-leg squat patterns
    "SPLIT_SQUAT": "split_squat",
    "DUMBBELL_SPLIT_SQUAT": "split_squat",
    "BULGARIAN_SPLIT_SQUAT": "split_squat",
    # The rest of what this app now sends, so a session comes back as the
    # exercises it was pushed as. Every one of these was round-tripped against
    # the account; see core/garmin_workout.GARMIN_TARGET.
    "BODY_WEIGHT_WALL_SQUAT": "wall_sit",
    "BRACED_SQUAT": "spanish_squat",
    "BOX_STEP_SQUAT": "step_down",
    "BARBELL_STEP_UP": "step_up",
    "BARBELL_REVERSE_LUNGE": "reverse_lunge",
    "REVERSE_LUNGE_WITH_REACH_BACK": "reverse_lunge",
    "SINGLE_LEG_ROMANIAN_DEADLIFT_WITH_DUMBBELL": "single_leg_rdl",
    "SPLIT_STANCE_EXTENSION": "terminal_knee_extension",
    "SLIDING_LEG_CURL": "nordic_curl_assisted",
    "SINGLE_LEG_SLIDING_LEG_CURL": "nordic_curl_assisted",
    "WEIGHTED_HIP_RAISE": "hip_thrust",
    "SINGLE_LEG_HIP_RAISE": "glute_bridge",
    "LATERAL_WALKS_WITH_BAND_AT_ANKLES": "band_monster_walk",
    "STANDING_HIP_ABDUCTION": "side_lying_abduction",
    "SIDE_PLANK_WITH_LEG_LIFT": "copenhagen_plank",
    "SIDE_PLANK_LIFT": "copenhagen_plank",
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
    # glutes and hips. Garmin files most of these under the HIP_RAISE and
    # HIP_STABILITY categories, so the category alone resolves a set even when
    # the watch records no specific exercise name.
    "HIP_RAISE": "glute_bridge",
    "GLUTE_BRIDGE": "glute_bridge",
    "BRIDGE": "glute_bridge",
    "WEIGHTED_GLUTE_BRIDGE": "glute_bridge",
    "SINGLE_LEG_GLUTE_BRIDGE": "glute_bridge",
    "HIP_THRUST": "hip_thrust",
    "BARBELL_HIP_THRUST": "hip_thrust",
    "WEIGHTED_HIP_THRUST": "hip_thrust",
    "HIP_STABILITY": "side_lying_abduction",
    "SIDE_LYING_LEG_RAISE": "side_lying_abduction",
    "HIP_ABDUCTION": "side_lying_abduction",
    "SIDE_LYING_HIP_ABDUCTION": "side_lying_abduction",
    "CLAMSHELL": "side_lying_abduction",
    "LATERAL_WALK": "band_monster_walk",
    "MONSTER_WALK": "band_monster_walk",
    "BAND_WALK": "band_monster_walk",
    "SIDE_PLANK": "side_plank_hip_lift",
    "SIDE_PLANK_HIP_ADDUCTION": "copenhagen_plank",
    "COPENHAGEN_PLANK": "copenhagen_plank",
    "HIP_ADDUCTION": "copenhagen_plank",
    # knee and quad
    "GOBLET_SQUAT": "goblet_squat",
    "SQUAT": "goblet_squat",
    "DUMBBELL_SQUAT": "goblet_squat",
    "KETTLEBELL_GOBLET_SQUAT": "goblet_squat",
    "STEP_DOWN": "step_down",
    "SINGLE_LEG_STEP_DOWN": "step_down",
    "LATERAL_STEP_DOWN": "step_down",
    "TERMINAL_KNEE_EXTENSION": "terminal_knee_extension",
    "KNEE_EXTENSION": "terminal_knee_extension",
    "LEG_EXTENSION": "terminal_knee_extension",
    # hamstring
    "NORDIC_HAMSTRING_CURL": "nordic_curl_assisted",
    "NORDIC_CURL": "nordic_curl_assisted",
    "LEG_CURL": "nordic_curl_assisted",
    "HAMSTRING_CURL": "nordic_curl_assisted",
}


def _substring_match(text: str) -> str | None:
    """Ordered longest-and-most-specific first, because these overlap.

    COPENHAGEN and the adduction planks contain SIDE_PLANK; SPLIT_SQUAT and
    SPANISH_SQUAT contain SQUAT. Whichever is tested first wins, so the order
    here is the rule, not a preference.
    """
    for needle, exercise_id in (
        ("SINGLE_LEG_CALF", "calf_raise_single_leg"),
        ("SEATED_CALF", "calf_raise_bent"),
        ("CALF", "calf_raise_straight"),
        ("SINGLE_LEG_DEAD", "single_leg_rdl"),
        ("SINGLE_LEG_ROMANIAN", "single_leg_rdl"),
        ("DEADLIFT", "rdl"),
        ("SPLIT_SQUAT", "split_squat"),
        ("SPANISH", "spanish_squat"),
        ("WALL_SIT", "wall_sit"),
        ("LUNGE", "reverse_lunge"),
        ("STEP_DOWN", "step_down"),
        ("STEP_UP", "step_up"),
        ("TIBIALIS", "tib_raise"),
        ("COPENHAGEN", "copenhagen_plank"),
        ("ADDUCTION", "copenhagen_plank"),
        ("SIDE_PLANK", "side_plank_hip_lift"),
        ("HIP_THRUST", "hip_thrust"),
        ("GLUTE", "glute_bridge"),
        ("BRIDGE", "glute_bridge"),
        ("MONSTER", "band_monster_walk"),
        ("BAND_WALK", "band_monster_walk"),
        ("LATERAL_WALK", "band_monster_walk"),
        ("ABDUCTION", "side_lying_abduction"),
        ("CLAMSHELL", "side_lying_abduction"),
        ("NORDIC", "nordic_curl_assisted"),
        ("HAMSTRING", "nordic_curl_assisted"),
        ("KNEE_EXTENSION", "terminal_knee_extension"),
        ("GOBLET", "goblet_squat"),
    ):
        if needle in text:
            return exercise_id
    return None


def _norm(raw: object) -> str:
    return str(raw or "").strip().upper().replace(" ", "_").replace("-", "_")


def _unambiguous_categories() -> dict[str, str]:
    """Categories where the library has exactly one exercise, so a bare category
    identifies it.

    Derived rather than listed, so adding an exercise cannot leave a stale
    mapping behind. The ambiguous ones deliberately map to nothing: a bare SQUAT
    used to become a goblet squat, which is how a pushed wall sit and split squat
    both came back logged as goblet squats and the progression started adding
    weight to an exercise that was never done.
    """
    from core.garmin_workout import GARMIN_TARGET

    per_category: dict[str, set[str]] = {}
    for exercise_id, (category, _name) in GARMIN_TARGET.items():
        per_category.setdefault(category, set()).add(exercise_id)
    return {c: next(iter(ids)) for c, ids in per_category.items()
            if len(ids) == 1}


# A set with no reps and at least this long is a hold, not a rep-counted set.
HOLD_SECONDS_MIN = 20.0

# Where Garmin has one name for both a rep exercise and its isometric version,
# the shape of the set tells them apart: {name: (rep version, hold version)}.
HOLD_VARIANTS: dict[str, tuple[str, str]] = {
    "SINGLE_LEG_STANDING_CALF_RAISE": ("calf_raise_single_leg",
                                       "single_leg_calf_hold"),
    "SINGLE_LEG_CALF_RAISE": ("calf_raise_single_leg", "single_leg_calf_hold"),
}


def looks_like_rest(row: Mapping[str, Any]) -> bool:
    """A recorded "set" that is really the gap between two.

    The watch emits one of these after every step of a pushed workout: no
    category, no exercise name, no reps, just a duration. They are not work, and
    counting them as unidentified sets made a clean session look like a mess.
    """
    category = _norm(row.get("garmin_category"))
    if _norm(row.get("garmin_name")):
        return False
    if category not in ("", "UNKNOWN", "REST"):
        return False
    return not (row.get("reps") or 0)


def map_garmin_exercise(category: str | None, name: str | None,
                        reps: float | None = None,
                        duration_s: float | None = None) -> str | None:
    """Best-effort match from Garmin's taxonomy into the library.

    Precedence is specific-before-generic, in four passes: the exact exercise
    name, then a substring match on that name, then the exact category, then a
    substring match on both together.

    The middle pass matters. Garmin often records a broad category alongside a
    specific name — HIP_STABILITY / LATERAL_BAND_WALK — and the category is an
    exact key in the table, so without it the generic category would win and a
    banded walk would be logged as a side-lying leg raise.

    Returns None rather than guessing: an unmapped set is surfaced in the UI for
    manual assignment instead of being silently binned or mis-attributed, which
    would corrupt the progression for whatever it was mistaken for.
    """
    nkey, ckey = _norm(name), _norm(category)
    # Garmin has one single-leg calf name and this library has two exercises
    # using it — the raise and the isometric hold. Nothing in the name separates
    # them, but the set does: a hold records seconds and no reps.
    variants = HOLD_VARIANTS.get(nkey)
    if variants:
        held = (not reps) and (duration_s or 0) >= HOLD_SECONDS_MIN
        return variants[1] if held else variants[0]
    if nkey and nkey in GARMIN_EXERCISE_MAP:
        return GARMIN_EXERCISE_MAP[nkey]
    if nkey:
        hit = _substring_match(nkey)
        if hit:
            return hit
    if ckey:
        # A category on its own only identifies an exercise when the library has
        # one in that category. Otherwise it is a guess, and a wrong guess feeds
        # the progression for an exercise that was never done.
        solo = _unambiguous_categories().get(ckey)
        if solo:
            return solo
    hit = _substring_match(f"{nkey}_{ckey}")
    # The same rule applies to the last pass: it must not resolve a bare
    # ambiguous category through a substring either.
    if hit and not nkey and ckey not in _unambiguous_categories():
        generic = _substring_match(ckey)
        if generic == hit:
            return None
    return hit


def sets_to_log_rows(
    day: str, activity_id: str, sets: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Collapse per-set watch records into one strength_log row per exercise.

    The log stores a session's work per exercise, so N sets of an exercise become
    one row: set count, the top weight used, and the modal rep count.

    Unilateral work is halved, because a set means the same thing on both sides
    of this app: the prescription asks for three sets of a single-leg RDL meaning
    three per leg, and the watch records six. Storing six would make the history
    read as double the prescription for ever. Rounded up, so a session where one
    side was skipped still shows as the sets that happened.
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
        ex = EXERCISES[exercise_id]
        is_iso = ex.kind == ISOMETRIC
        sets = -(-len(items) // 2) if ex.unilateral else len(items)
        rows.append(
            {
                "day": day,
                "activity_id": activity_id,
                "exercise_id": exercise_id,
                "sets": sets,
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


SESSION_CYCLE: tuple[tuple[str, ...], ...] = (SESSION_A, SESSION_B, SESSION_C)


def session_template(session_index: int) -> tuple[str, ...]:
    """Rotate A/B/C so both calf variants, both hinge patterns and the glute and
    knee work all get covered."""
    return SESSION_CYCLE[session_index % len(SESSION_CYCLE)]


def library_rows() -> list[dict[str, Any]]:
    """The library flattened for storage. Code stays the source of truth."""
    import json
    from datetime import datetime as _dt

    now = _dt.now().isoformat(timespec="seconds")
    rows = []
    for e in EXERCISES.values():
        low, high = e.rep_range or (None, None)
        hlow, hhigh = e.hold_range or (None, None)
        rows.append({
            "exercise_id": e.id, "name": e.name, "kind": e.kind,
            "focus": e.focus, "target": e.target, "sets": e.sets,
            "rep_low": low, "rep_high": high,
            "hold_low": hlow, "hold_high": hhigh,
            "unilateral": int(e.unilateral), "tempo": e.tempo, "cue": e.cue,
            "setup": e.setup, "steps": json.dumps(list(e.steps)),
            "mistakes": e.mistakes, "why": e.why, "load_note": e.load_note,
            "progress_to": e.progress_to, "synced_at": now,
        })
    return rows


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
