"""Typed payloads shared across the data, analysis, planner and AI layers.

No Streamlit, no Garmin, no Anthropic imports here — these are the contracts
every other layer builds on, which is what keeps them testable in isolation.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

Sport = Literal["swim", "bike", "run", "strength", "brick", "rest"]
SPORTS: tuple[Sport, ...] = ("swim", "bike", "run", "strength", "brick", "rest")
ENDURANCE_SPORTS: tuple[str, ...] = ("swim", "bike", "run")
DAYS: tuple[str, ...] = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

Zone = Literal["Z1", "Z2", "Z3", "Z4", "Z5", "mixed", "technique", "n/a"]


# --------------------------------------------------------------------------
# Analysis outputs
# --------------------------------------------------------------------------


class EFPoint(BaseModel):
    """One steady-session efficiency-factor observation."""

    activity_id: str
    date: date
    sport: str
    ef: float
    metric: str  # "speed_per_hr" | "power_per_hr"
    avg_hr: float
    duration_min: float
    decoupling_pct: float | None = None
    is_steady: bool = True
    steady_reason: str = ""


class EFTrend(BaseModel):
    """Direction of travel for a sport's efficiency factor."""

    sport: str
    metric: str
    n_sessions: int
    recent_mean: float | None = None
    baseline_mean: float | None = None
    change_pct: float | None = None  # recent vs baseline
    slope_pct_per_week: float | None = None
    verdict: Literal["improving", "flat", "declining", "insufficient_data"] = (
        "insufficient_data"
    )


class RecoverySignals(BaseModel):
    """Rolling recovery picture. Everything optional — Garmin gaps are normal."""

    as_of: date
    rhr_recent: float | None = None
    rhr_baseline: float | None = None
    rhr_delta: float | None = None  # recent - baseline, bpm (positive = worse)
    hrv_recent: float | None = None
    hrv_baseline: float | None = None
    hrv_delta_pct: float | None = None  # positive = above baseline (good)
    hrv_status: str | None = None
    training_readiness: float | None = None
    training_status: str | None = None
    acute_load: float | None = None
    chronic_load: float | None = None
    acwr: float | None = None
    vo2max_run: float | None = None
    vo2max_bike: float | None = None
    sleep_score: float | None = None


class SportWeek(BaseModel):
    """Completed volume for one sport in one week."""

    sport: str
    sessions: int = 0
    minutes: float = 0.0
    distance_km: float = 0.0
    load: float = 0.0
    longest_min: float = 0.0


class WeekSummary(BaseModel):
    week_start: date
    by_sport: dict[str, SportWeek] = Field(default_factory=dict)
    total_minutes: float = 0.0
    total_load: float = 0.0
    rest_days: int = 0
    strength_sessions: int = 0


# --------------------------------------------------------------------------
# Planner inputs
# --------------------------------------------------------------------------


class Checkin(BaseModel):
    """How the athlete feels today. Drives the AI layer, never the safety rules."""

    date: date
    sleep: int = Field(3, ge=1, le=5)
    soreness: int = Field(3, ge=1, le=5)  # 1 = fresh, 5 = wrecked
    motivation: int = Field(3, ge=1, le=5)
    time_available_min: int = Field(90, ge=0, le=480)
    notes: str = ""


class SportEnvelope(BaseModel):
    """Per-sport bounds the AI must plan inside."""

    sport: str
    enabled: bool = True
    min_sessions: int
    max_sessions: int
    max_minutes: float
    long_session_min: float = 0.0  # 0 = no long session required
    notes: str = ""


class Envelope(BaseModel):
    """The rules backstop. Produced deterministically; the AI cannot widen it."""

    week_start: date
    phase: str = "base"
    week_index: int = 0  # position in the 4-week block
    deload: bool = False
    deload_reasons: list[str] = Field(default_factory=list)
    max_week_minutes: float = 0.0
    prev_week_minutes: float = 0.0
    progression_cap_pct: float = 10.0
    min_rest_days: int = 1
    strength_sessions: int = 2
    brick_required: bool = False
    max_quality_sessions: int = 2
    # deload | hold | build — whether recovery argues for less, the same, or more
    readiness_verdict: str = "hold"
    build_signals: list[str] = Field(default_factory=list)
    by_sport: dict[str, SportEnvelope] = Field(default_factory=dict)


class StrengthPrescription(BaseModel):
    exercise_id: str
    name: str
    sets: int
    reps: int | None = None
    hold_s: int | None = None
    load_kg: float | None = None
    tempo: str = ""
    note: str = ""


class StrengthState(BaseModel):
    """Deterministic progression state — the AI reads this, never writes it."""

    sessions_logged: int = 0
    last_session_date: date | None = None
    pain_flagged: list[str] = Field(default_factory=list)
    prescription: list[StrengthPrescription] = Field(default_factory=list)


class PlannerFacts(BaseModel):
    """Ground truth handed to the envelope and then to the AI."""

    week_start: date
    today: date
    completed_this_week: WeekSummary
    previous_weeks: list[WeekSummary] = Field(default_factory=list)
    recovery: RecoverySignals
    ef_trends: list[EFTrend] = Field(default_factory=list)
    recent_checkins: list[Checkin] = Field(default_factory=list)
    days_remaining: list[str] = Field(default_factory=list)
    trained_days: list[str] = Field(default_factory=list)  # already active this week


# --------------------------------------------------------------------------
# AI contract (strict)
# --------------------------------------------------------------------------


class PlanDay(BaseModel):
    day: str
    sport: str
    duration_min: int = Field(0, ge=0, le=480)
    target_zone: str = "Z2"
    # A real bpm range, e.g. "112-129 bpm". Filled in by planner.enforce() from
    # the athlete's own Garmin zone boundaries — never written by the model, which
    # has no business inventing heart-rate numbers.
    target_hr: str = ""
    purpose: str = ""
    exercise_ids: list[str] = Field(default_factory=list)
    why: str = ""

    @field_validator("day")
    @classmethod
    def _valid_day(cls, v: str) -> str:
        v = v.strip()[:3].title()
        if v not in DAYS:
            raise ValueError(f"day must be one of {DAYS}")
        return v

    @field_validator("sport")
    @classmethod
    def _valid_sport(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in SPORTS:
            raise ValueError(f"sport must be one of {SPORTS}")
        return v


class WeekPlan(BaseModel):
    week_plan: list[PlanDay] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    adjustments_made: list[str] = Field(default_factory=list)
    # "manual" is the athlete's own edited week, saved from the Plan page.
    source: Literal["rules", "ai", "ai_repaired", "manual"] = "rules"

    def minutes(self) -> float:
        return float(sum(d.duration_min for d in self.week_plan))


class PlanPayload(BaseModel):
    """Exactly what goes to the model — see Section 10 of the spec."""

    completed_this_week: dict[str, Any]
    recovery_signals: dict[str, Any]
    envelope: dict[str, Any]
    strength_state: dict[str, Any]
    checkin: dict[str, Any]
    history: dict[str, Any]
