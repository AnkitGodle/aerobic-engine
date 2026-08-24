"""The training rules, as data the athlete can see and change.

Every number in here was a constant buried in `planner.py`. That was fine while
they were the same for everyone who used this — which is one person — but it made
the rules invisible: the plan said "added a session to reach the 3-session floor"
and there was nowhere to see that the floor was 3, let alone make it 4.

So the tunables live here, are stored one key per rule, and are clamped to a
range on the way in and on the way out. What is *not* here matters just as much:

  * The **deload triggers** (HRV below baseline, resting HR above it, Training
    Readiness under 35, load ratio over 1.3) stay fixed in `planner.py`. They are
    the backstop that makes the whole design work — a mood-driven "I feel fine"
    must not be able to widen them, and neither should a settings page on a bad
    day.
  * The **strength library** stays fixed in `strength.py`, for the same reason:
    tendon injuries come from load jumps, and an exercise picker is exactly how
    a jump gets introduced.

Both are shown on the Rules page as read-only, because "you cannot change this"
is itself information.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Iterable

# Stored one key per rule, prefixed so they cannot collide with sync bookkeeping.
KEY_PREFIX = "rule_"


@dataclass(frozen=True)
class Rules:
    """The editable half of the envelope. Defaults are the base-phase design."""

    # At least this many swim/bike/run/brick sessions a week. Below three, a base
    # block stops being a base block.
    min_endurance_sessions: int = 3
    # Keep a clear day between endurance sessions. Strength is exempt: it is
    # low-impact tendon work and belongs in the gaps.
    space_endurance: bool = True
    strength_sessions: int = 2
    min_rest_days: int = 1
    # Volume growth ceiling per week. Ten percent is the conventional number and
    # the reason this app exists is that ambition ignores it.
    progression_cap_pct: float = 10.0
    # Every Nth week is a deload, cutting volume by `deload_cut_pct`.
    block_weeks: int = 4
    deload_cut_pct: float = 35.0
    # A bike-to-run brick every Nth week. 0 turns them off.
    brick_every_weeks: int = 2

    def as_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


DEFAULTS = Rules()

# name -> (low, high, step, label, why it matters)
BOUNDS: dict[str, tuple[float, float, float, str, str]] = {
    "min_endurance_sessions": (
        2, 6, 1, "Endurance sessions a week",
        "The floor. The planner adds a session rather than let a week fall "
        "below it."),
    "strength_sessions": (
        0, 4, 1, "Leg strength sessions a week",
        "Tendons adapt slowly and protect run volume. Two is the design."),
    "min_rest_days": (
        1, 4, 1, "Full rest days a week",
        "Days with nothing on them. A deload week gets one more automatically."),
    "progression_cap_pct": (
        0, 20, 1, "Weekly volume growth cap (%)",
        "The most this week may exceed last week. Raising it is how people get "
        "hurt."),
    "block_weeks": (
        2, 8, 1, "Weeks per block",
        "The last week of each block is a scheduled deload, whatever the "
        "signals say."),
    "deload_cut_pct": (
        20, 60, 5, "Deload volume cut (%)",
        "How much a deload week takes off."),
    "brick_every_weeks": (
        0, 6, 1, "Brick every N weeks",
        "A bike-to-run session. 0 turns them off."),
}

# Rules the athlete may not move, and why. Shown alongside the editable ones so
# the page is the whole picture rather than the convenient half of it.
#
# The numeric deload triggers are not in here: the UI renders those with the
# athlete's current reading next to each threshold, which says far more than the
# threshold alone. This is what is left — the qualitative backstops.
FIXED: tuple[tuple[str, str], ...] = (
    ("A bad Garmin HRV or training status",
     "forces a deload — unbalanced, low, overreaching, unproductive, strained"),
    ("Strength exercises come from a fixed library",
     "the AI may pick from it and set volume, never add to it"),
    ("Hard sessions are capped, and dropped entirely in a deload",
     "base phase is built on easy volume"),
    ("No plyometrics or jumping in base",
     "tendon injuries come from load jumps, not load"),
    ("A deload cannot be overruled by how you feel",
     "the trigger is the data; the check-in adjusts what is left"),
)


def _clamp(name: str, value: Any) -> Any:
    """Coerce and clamp one rule. Anything unusable falls back to the default."""
    default = getattr(DEFAULTS, name)
    if isinstance(default, bool):
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:                       # NaN
        return default
    low, high, *_ = BOUNDS.get(name, (number, number, 1, "", ""))
    number = max(low, min(high, number))
    return int(round(number)) if isinstance(default, int) else round(number, 1)


def from_mapping(values: dict[str, Any] | None) -> Rules:
    """Build Rules from stored strings, clamping each field."""
    values = values or {}
    kept: dict[str, Any] = {}
    for f in fields(Rules):
        raw = values.get(f.name, values.get(f"{KEY_PREFIX}{f.name}"))
        if raw is not None and raw != "":
            kept[f.name] = _clamp(f.name, raw)
    return Rules(**kept)


def load(store: Any) -> Rules:
    """Read the athlete's rules. Any store failure yields the defaults.

    Never raises: the planner has to produce a week even if the settings table is
    unreachable, and the defaults are the design rather than a fallback.
    """
    try:
        keys = tuple(f"{KEY_PREFIX}{f.name}" for f in fields(Rules))
        return from_mapping(store.get_states(keys))
    except Exception:  # noqa: BLE001 - defaults are always a valid answer
        return DEFAULTS


def save(store: Any, values: dict[str, Any]) -> Rules:
    """Clamp and persist. Returns what was actually stored, not what was asked for."""
    saved = from_mapping({**load(store).as_dict(), **values})
    for name, value in saved.as_dict().items():
        store.set_state(f"{KEY_PREFIX}{name}",
                        "1" if value is True else "0" if value is False
                        else str(value))
    return saved


def reset(store: Any) -> Rules:
    """Forget every override, back to the base-phase design."""
    for f in fields(Rules):
        store.set_state(f"{KEY_PREFIX}{f.name}", "")
    return DEFAULTS


def changed_from_default(rules: Rules) -> dict[str, tuple[Any, Any]]:
    """What the athlete has moved, as name -> (theirs, default)."""
    out = {}
    for name, value in rules.as_dict().items():
        default = getattr(DEFAULTS, name)
        if value != default:
            out[name] = (value, default)
    return out


def describe(rules: Rules) -> list[dict[str, Any]]:
    """Rows for the UI: label, value, bounds and why the rule exists."""
    rows = []
    for name, (low, high, step, label, why) in BOUNDS.items():
        rows.append({
            "name": name, "label": label, "why": why,
            "value": getattr(rules, name), "min": low, "max": high, "step": step,
            "default": getattr(DEFAULTS, name),
        })
    return rows


def summary(rules: Rules) -> Iterable[str]:
    """One line per rule, for a plan's flags or an AI payload."""
    yield (f"at least {rules.min_endurance_sessions} endurance sessions a week"
           + (", spaced a day apart" if rules.space_endurance else ""))
    yield f"{rules.strength_sessions} leg strength session(s) a week"
    yield f"at least {rules.min_rest_days} full rest day(s)"
    yield f"volume grows at most {rules.progression_cap_pct:.0f}% a week"
    yield (f"every {rules.block_weeks} weeks is a deload, cutting "
           f"{rules.deload_cut_pct:.0f}%")
    yield (f"a brick every {rules.brick_every_weeks} weeks"
           if rules.brick_every_weeks else "no bricks")
