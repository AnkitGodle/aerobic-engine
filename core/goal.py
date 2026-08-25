"""A race, a date, and the phases between here and it.

Until now every week was generated as though the horizon were infinite:
`phase` was the string "base" and nothing in the app knew an event existed. That
is the ceiling on how far the planning can go, because almost every decision a
coach makes is really a decision about *when* — a hard session eight weeks out is
preparation, the same session in the last week is damage.

With a date, the envelope becomes a curve:

    base    easy volume, tendons first, intensity kept on a leash
    build   quality earns its way in, long sessions grow
    peak    the hardest weeks, volume steady rather than rising
    taper   volume falls hard, sharpness kept, rest days added

Two deliberate limits. The phase never *relaxes* a safety rule — a deload
triggered by recovery data still cuts a peak week, because the point of the
backstop is that nothing outranks it. And with no goal set, everything behaves
exactly as it did before: base phase, forever, which is the right answer for
someone building an engine with no race in mind.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any

log = logging.getLogger("aerobic_engine.goal")

KEYS = ("goal_event", "goal_date", "goal_sport", "goal_distance_km")

BASE, BUILD, PEAK, TAPER = "base", "build", "peak", "taper"
PHASES = (BASE, BUILD, PEAK, TAPER)

# How long each phase lasts, counted back from the event. The taper depends on
# the distance because that is what it is for: a marathon needs three weeks to
# absorb the training, a 5K is sharp again in one.
TAPER_WEEKS = ((42.0, 3), (21.0, 2), (10.0, 1), (0.0, 1))
PEAK_WEEKS = 3
BUILD_WEEKS = 8

PHASE_NOTES = {
    BASE: "Easy volume, tendons first. Intensity stays on a leash.",
    BUILD: "Quality earns its way in, and the long sessions grow.",
    PEAK: "The hardest weeks. Volume holds rather than rising.",
    TAPER: "Volume drops, sharpness stays, and you rest more than feels right.",
}


@dataclass(frozen=True)
class Goal:
    """What you are training for."""

    event: str = ""
    day: date | None = None
    sport: str = "run"
    distance_km: float | None = None

    @property
    def set(self) -> bool:
        return bool(self.day)

    def taper_weeks(self) -> int:
        km = float(self.distance_km or 0)
        for threshold, weeks in TAPER_WEEKS:
            if km >= threshold:
                return weeks
        return 1

    def weeks_to_go(self, today: date | None = None) -> int | None:
        """Whole weeks left, or None with no date. Negative once it has passed."""
        if not self.day:
            return None
        today = today or date.today()
        return (self.day - today).days // 7

    def phase(self, today: date | None = None) -> str:
        """Which phase today falls in. Base with no goal, and base again after."""
        weeks = self.weeks_to_go(today)
        if weeks is None or weeks < 0:
            return BASE
        taper = self.taper_weeks()
        if weeks < taper:
            return TAPER
        if weeks < taper + PEAK_WEEKS:
            return PEAK
        if weeks < taper + PEAK_WEEKS + BUILD_WEEKS:
            return BUILD
        return BASE

    def timeline(self, today: date | None = None) -> list[dict[str, Any]]:
        """Each phase with the weeks it covers, for showing the shape of the plan.

        Phases that have already gone by are marked done rather than dropped: the
        point of the picture is where you are in it.
        """
        weeks = self.weeks_to_go(today)
        if weeks is None:
            return []
        taper = self.taper_weeks()
        spans = [
            (BASE, weeks, taper + PEAK_WEEKS + BUILD_WEEKS),
            (BUILD, taper + PEAK_WEEKS + BUILD_WEEKS - 1, taper + PEAK_WEEKS),
            (PEAK, taper + PEAK_WEEKS - 1, taper),
            (TAPER, taper - 1, 0),
        ]
        here = self.phase(today)
        out = []
        for name, starts_at, ends_at in spans:
            if starts_at < ends_at:
                continue          # no room for this phase in a short run-up
            # Clamped to the weeks that actually exist: with ten weeks to go the
            # build phase starts now, not in twelve weeks' time.
            begins = min(starts_at, weeks)
            if begins < ends_at:
                continue
            out.append({
                "phase": name,
                "from_weeks": begins,
                "to_weeks": ends_at,
                "current": name == here,
                "note": PHASE_NOTES[name],
            })
        return out


NO_GOAL = Goal()


def load(store: Any) -> Goal:
    """The stored goal, or an empty one. Never raises: no goal is a valid answer."""
    try:
        state = store.get_states(KEYS)
    except Exception:  # noqa: BLE001 - the planner must still produce a week
        return NO_GOAL
    day = None
    raw = state.get("goal_date")
    if raw:
        try:
            day = date.fromisoformat(str(raw)[:10])
        except ValueError:
            day = None
    distance = None
    if state.get("goal_distance_km"):
        try:
            distance = float(state["goal_distance_km"])
        except (TypeError, ValueError):
            distance = None
    return Goal(event=str(state.get("goal_event") or ""), day=day,
                sport=str(state.get("goal_sport") or "run"),
                distance_km=distance)


def save(store: Any, event: str, day: date | None, sport: str = "run",
         distance_km: float | None = None) -> Goal:
    """Store the goal. A missing date clears it, because a goal without one is
    a wish rather than something a plan can be built backwards from."""
    goal = Goal(event=(event or "").strip()[:80], day=day,
                sport=(sport or "run").strip().lower(),
                distance_km=float(distance_km) if distance_km else None)
    store.set_state("goal_event", goal.event)
    store.set_state("goal_date", goal.day.isoformat() if goal.day else "")
    store.set_state("goal_sport", goal.sport)
    store.set_state("goal_distance_km",
                    str(goal.distance_km) if goal.distance_km else "")
    return goal


def clear(store: Any) -> Goal:
    for key in KEYS:
        store.set_state(key, "")
    return NO_GOAL


def describe(goal: Goal, today: date | None = None) -> str:
    """One line for the top of a page."""
    if not goal.set:
        return "No race set — building the engine, base phase."
    weeks = goal.weeks_to_go(today)
    name = goal.event or "your race"
    if weeks is None:
        return name
    if weeks < 0:
        return f"{name} has been and gone. Back to base."
    phase = goal.phase(today)
    when = "this week" if weeks == 0 else \
        "next week" if weeks == 1 else f"in {weeks} weeks"
    return f"{name} {when} · {phase} phase"


# How each phase moves the envelope. Multipliers on volume and the long-session
# floor, plus how many hard sessions a week may hold. Nothing here can lift a
# limit the recovery rules have already imposed — see planner.build_envelope.
PHASE_SHAPE: dict[str, dict[str, float]] = {
    BASE: {"volume": 1.0, "long": 1.0, "quality": 0},
    BUILD: {"volume": 1.0, "long": 1.1, "quality": 1},
    PEAK: {"volume": 0.95, "long": 1.15, "quality": 2},
    # A taper is not a deload: the volume goes but the sharpness stays, which is
    # why quality is not zeroed and the cut is steeper than a deload's.
    TAPER: {"volume": 0.55, "long": 0.6, "quality": 1},
}


def shape(phase: str) -> dict[str, float]:
    return PHASE_SHAPE.get(phase, PHASE_SHAPE[BASE])
