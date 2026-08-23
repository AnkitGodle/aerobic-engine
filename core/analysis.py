"""Deterministic training analysis. No AI, no Streamlit, no Garmin.

The headline question this module answers: is output rising at the same heart
rate? That is Efficiency Factor (EF) trended over steady aerobic sessions only —
mixing intervals and races into the trend makes it noise.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from datetime import date, datetime, timedelta
from statistics import fmean, pstdev
from typing import Any

from core.schemas import (
    ENDURANCE_SPORTS,
    EFPoint,
    EFTrend,
    RecoverySignals,
    SportWeek,
    WeekSummary,
)

# --- steady-session gate ---------------------------------------------------
MIN_STEADY_MIN = {"run": 25.0, "bike": 35.0, "swim": 15.0}
MAX_HR_RATIO = 1.18          # max_hr / avg_hr above this => surges/intervals
MAX_ANAEROBIC_TE = 2.0       # Garmin anaerobic training effect
MAX_HR_CV = 0.12             # HR coefficient of variation from the stream
QUALITY_WORDS = (
    "interval", "race", "threshold", "tempo", "vo2", "track", "test", "ttt",
    "fartlek", "hill", "sprint", "time trial", "5k", "10k", "parkrun", "brick",
)
MIN_DECOUPLE_MIN = 60.0      # only long sessions get a decoupling number

# Garmin's own time-in-zone is a far better steadiness test than avg/max HR: a
# run can sit at a modest average while spending half its time above threshold.
HARD_ZONES = (4, 5)
MAX_HARD_ZONE_FRACTION = 0.20  # more than this above Z3 is a quality session


# --------------------------------------------------------------------------
# Per-activity metrics
# --------------------------------------------------------------------------


def efficiency_factor(activity: dict[str, Any]) -> tuple[float | None, str]:
    """Aerobic output per heartbeat.

    Bike prefers Pw:HR (power / HR) when a power meter was present; everything
    else uses speed / HR. Swim speed is derived from distance/time because the
    watch reports pool speed inconsistently.
    """
    sport = activity.get("sport")
    hr = _pos(activity.get("avg_hr"))
    if not hr:
        return None, "none"

    if sport == "bike":
        power = _pos(activity.get("norm_power_w")) or _pos(activity.get("avg_power_w"))
        if power:
            return power / hr, "power_per_hr"

    speed = _pos(activity.get("avg_speed_mps")) or _derived_speed(activity)
    if not speed:
        return None, "none"
    # Scaled so the numbers read like 0.5–2.0 rather than 0.01.
    return (speed * 100.0) / hr, "speed_per_hr"


def _derived_speed(activity: dict[str, Any]) -> float | None:
    dist = _pos(activity.get("distance_m"))
    dur = _pos(activity.get("moving_s")) or _pos(activity.get("duration_s"))
    return dist / dur if dist and dur else None


def hard_zone_fraction(zones: Sequence[dict[str, Any]]) -> float | None:
    """Share of recorded time spent in zone 4 or 5."""
    total = sum(float(z.get("secs_in_zone") or 0) for z in zones)
    if total <= 0:
        return None
    hard = sum(
        float(z.get("secs_in_zone") or 0)
        for z in zones
        if int(z.get("zone_number") or 0) in HARD_ZONES
    )
    return hard / total


def steady_check(
    activity: dict[str, Any],
    stream: Sequence[dict[str, Any]] | None = None,
    zones: Sequence[dict[str, Any]] | None = None,
) -> tuple[bool, str]:
    """Is this a steady aerobic session that belongs in an EF trend?"""
    sport = activity.get("sport")
    if sport not in ENDURANCE_SPORTS:
        return False, "not an endurance sport"
    if not _pos(activity.get("avg_hr")):
        return False, "no HR"

    minutes = (_pos(activity.get("duration_s")) or 0) / 60.0
    floor = MIN_STEADY_MIN.get(sport, 25.0)
    if minutes < floor:
        return False, f"too short ({minutes:.0f} < {floor:.0f} min)"

    name = (activity.get("name") or "").lower()
    for word in QUALITY_WORDS:
        if word in name:
            return False, f"name flags quality session ({word!r})"

    an_te = activity.get("anaerobic_te")
    if an_te is not None and an_te >= MAX_ANAEROBIC_TE:
        return False, f"anaerobic TE {an_te:.1f}"

    avg_hr, max_hr = _pos(activity.get("avg_hr")), _pos(activity.get("max_hr"))
    if avg_hr and max_hr and max_hr / avg_hr > MAX_HR_RATIO:
        return False, f"HR spread {max_hr / avg_hr:.2f}"

    if zones:
        frac = hard_zone_fraction(zones)
        if frac is not None and frac > MAX_HARD_ZONE_FRACTION:
            return False, f"{frac * 100:.0f}% of time in Z4-Z5"

    if stream:
        hrs = [s["hr"] for s in stream if _pos(s.get("hr"))]
        if len(hrs) > 20:
            cv = pstdev(hrs) / fmean(hrs)
            if cv > MAX_HR_CV:
                return False, f"HR variability {cv:.2f}"

    return True, "steady"


def decoupling(
    activity: dict[str, Any], stream: Sequence[dict[str, Any]]
) -> tuple[float | None, float | None, float | None]:
    """Aerobic drift: EF of the first half vs the second half, in percent.

    Returns (decoupling_pct, ef_first, ef_second). Positive = output fell
    relative to HR in the back half. Under ~5% is good aerobic durability.
    """
    minutes = (_pos(activity.get("duration_s")) or 0) / 60.0
    if minutes < MIN_DECOUPLE_MIN or len(stream) < 20:
        return None, None, None

    use_power = activity.get("sport") == "bike" and any(
        _pos(s.get("power_w")) for s in stream
    )
    usable = [
        s
        for s in stream
        if _pos(s.get("hr"))
        and _pos(s.get("power_w") if use_power else s.get("speed_mps"))
    ]
    if len(usable) < 20:
        return None, None, None

    mid = usable[len(usable) // 2]["t_s"]
    first = [s for s in usable if s["t_s"] <= mid]
    second = [s for s in usable if s["t_s"] > mid]
    if len(first) < 10 or len(second) < 10:
        return None, None, None

    def half_ef(rows: list[dict[str, Any]]) -> float | None:
        key = "power_w" if use_power else "speed_mps"
        out = fmean(r[key] for r in rows)
        hr = fmean(r["hr"] for r in rows)
        if not hr:
            return None
        return out / hr if use_power else (out * 100.0) / hr

    ef1, ef2 = half_ef(first), half_ef(second)
    if not ef1 or not ef2:
        return None, ef1, ef2
    return (ef1 - ef2) / ef1 * 100.0, ef1, ef2


def compute_activity_metrics(
    activity: dict[str, Any],
    stream: Sequence[dict[str, Any]] | None = None,
    zones: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """The row that goes into `activity_metrics`."""
    ef, metric = efficiency_factor(activity)
    is_steady, reason = steady_check(activity, stream, zones)
    drift, ef1, ef2 = (None, None, None)
    if stream:
        drift, ef1, ef2 = decoupling(activity, stream)
    return {
        "activity_id": activity["activity_id"],
        "ef": ef,
        "ef_metric": metric,
        "ef_first_half": ef1,
        "ef_second_half": ef2,
        "decoupling_pct": drift,
        "is_steady": int(is_steady),
        "steady_reason": reason,
        "hr_samples": len(stream) if stream else 0,
        "computed_at": datetime.now().isoformat(timespec="seconds"),
    }


# --------------------------------------------------------------------------
# Trends
# --------------------------------------------------------------------------


def ef_points(
    activities: Iterable[dict[str, Any]], sport: str, steady_only: bool = False
) -> list[EFPoint]:
    """EF observations for one sport, oldest first.

    By default every session with heart rate is returned, with `is_steady`
    carried through so the caller can show which ones are clean enough to trend.
    Pass steady_only=True for the statistically tidy subset: hard sessions have a
    lower EF by nature, so a mixed series moves with how hard you trained as much
    as with fitness.
    """
    out: list[EFPoint] = []
    for a in activities:
        if a.get("sport") != sport or not _pos(a.get("ef")):
            continue
        if steady_only and not a.get("is_steady"):
            continue
        out.append(
            EFPoint(
                activity_id=a["activity_id"],
                date=_as_date(a["start_date"]),
                sport=sport,
                ef=float(a["ef"]),
                metric=a.get("ef_metric") or "speed_per_hr",
                avg_hr=float(a["avg_hr"]),
                duration_min=(a.get("duration_s") or 0) / 60.0,
                decoupling_pct=a.get("decoupling_pct"),
                is_steady=bool(a.get("is_steady")),
                steady_reason=a.get("steady_reason") or "",
            )
        )
    return sorted(out, key=lambda p: p.date)


def ef_trend(
    activities: Iterable[dict[str, Any]],
    sport: str,
    recent_days: int = 21,
    baseline_days: int = 56,
    as_of: date | None = None,
    steady_only: bool = True,
) -> EFTrend:
    """Recent EF mean vs the preceding baseline, plus a least-squares slope.

    Mixing metrics (Pw:HR and speed:HR) in one trend would be meaningless, so
    when a bike history spans both we keep only the dominant metric.
    """
    pts = ef_points(activities, sport, steady_only=steady_only)
    metric = "none"
    if pts:
        metrics = [p.metric for p in pts]
        metric = max(set(metrics), key=metrics.count)
        pts = [p for p in pts if p.metric == metric]

    trend = EFTrend(sport=sport, metric=metric, n_sessions=len(pts))
    if len(pts) < 3:
        return trend

    today = as_of or pts[-1].date
    recent_cut = today - timedelta(days=recent_days)
    base_cut = today - timedelta(days=baseline_days)
    recent = [p.ef for p in pts if p.date > recent_cut]
    baseline = [p.ef for p in pts if base_cut <= p.date <= recent_cut]

    trend.recent_mean = round(fmean(recent), 4) if recent else None
    trend.baseline_mean = round(fmean(baseline), 4) if baseline else None
    if trend.recent_mean and trend.baseline_mean:
        trend.change_pct = round(
            (trend.recent_mean - trend.baseline_mean) / trend.baseline_mean * 100.0, 2
        )

    window = [p for p in pts if p.date >= base_cut]
    if len(window) >= 3:
        t0 = window[0].date
        xs = [(p.date - t0).days / 7.0 for p in window]
        ys = [p.ef for p in window]
        slope = ols_slope(xs, ys)
        mean_y = fmean(ys)
        if slope is not None and mean_y:
            trend.slope_pct_per_week = round(slope / mean_y * 100.0, 3)

    signal = trend.change_pct if trend.change_pct is not None else (
        trend.slope_pct_per_week
    )
    if signal is None or len(window) < 3:
        trend.verdict = "insufficient_data"
    elif signal > 1.0:
        trend.verdict = "improving"
    elif signal < -1.0:
        trend.verdict = "declining"
    else:
        trend.verdict = "flat"
    return trend


def all_ef_trends(
    activities: Iterable[dict[str, Any]],
    as_of: date | None = None,
    steady_only: bool = True,
) -> list[EFTrend]:
    acts = list(activities)
    return [
        ef_trend(acts, s, as_of=as_of, steady_only=steady_only) for s in ENDURANCE_SPORTS
    ]


def rolling(
    rows: Sequence[dict[str, Any]], field: str, days: int, as_of: date
) -> float | None:
    """Mean of `field` over the `days` ending at `as_of` (inclusive)."""
    cut = as_of - timedelta(days=days - 1)
    vals = [
        float(r[field])
        for r in rows
        if r.get(field) is not None and cut <= _as_date(r["day"]) <= as_of
    ]
    return fmean(vals) if vals else None


def recovery_signals(
    wellness: Sequence[dict[str, Any]],
    activities: Sequence[dict[str, Any]] | None = None,
    as_of: date | None = None,
    recent_days: int = 7,
    baseline_days: int = 28,
) -> RecoverySignals:
    """7-day picture against a 28-day baseline, plus load ratio.

    Garmin's own acute:chronic ratio is used when present; otherwise it is
    computed from stored per-activity training load.
    """
    wellness = sorted(wellness, key=lambda r: r["day"])
    as_of = as_of or (_as_date(wellness[-1]["day"]) if wellness else date.today())
    sig = RecoverySignals(as_of=as_of)
    if not wellness:
        return sig

    sig.rhr_recent = _r(rolling(wellness, "resting_hr", recent_days, as_of))
    sig.rhr_baseline = _r(rolling(wellness, "resting_hr", baseline_days, as_of))
    if sig.rhr_recent and sig.rhr_baseline:
        sig.rhr_delta = round(sig.rhr_recent - sig.rhr_baseline, 2)

    sig.hrv_recent = _r(rolling(wellness, "hrv_last_night", recent_days, as_of))
    sig.hrv_baseline = _r(rolling(wellness, "hrv_last_night", baseline_days, as_of))
    if sig.hrv_recent and sig.hrv_baseline:
        sig.hrv_delta_pct = round(
            (sig.hrv_recent - sig.hrv_baseline) / sig.hrv_baseline * 100.0, 2
        )

    latest = _latest_with(wellness, ("hrv_status",))
    sig.hrv_status = (latest or {}).get("hrv_status")
    sig.training_status = (_latest_with(wellness, ("training_status",)) or {}).get(
        "training_status"
    )
    sig.training_readiness = _r(rolling(wellness, "training_readiness", 3, as_of))
    sig.sleep_score = _r(rolling(wellness, "sleep_score", recent_days, as_of))
    sig.vo2max_run = _r(_last_value(wellness, "vo2max_run"), 1)
    sig.vo2max_bike = _r(_last_value(wellness, "vo2max_bike"), 1)

    sig.acute_load = _r(_last_value(wellness, "acute_load"))
    sig.chronic_load = _r(_last_value(wellness, "chronic_load"))
    garmin_ratio = _last_value(wellness, "load_ratio")
    if garmin_ratio:
        sig.acwr = round(float(garmin_ratio), 2)
    elif sig.acute_load and sig.chronic_load:
        sig.acwr = round(sig.acute_load / sig.chronic_load, 2)
    elif activities:
        sig.acwr = acwr_from_activities(activities, as_of)
    return sig


MIN_ACWR_HISTORY_DAYS = 21


def acwr_from_activities(
    activities: Sequence[dict[str, Any]],
    as_of: date | None = None,
    min_history_days: int = MIN_ACWR_HISTORY_DAYS,
) -> float | None:
    """Acute:chronic workload ratio from stored load (or minutes as a proxy).

    Returns None until there is enough history for the chronic side to mean
    anything. With one week of data the chronic average is just the acute week
    divided by four, which reads as a ratio near 4.0 and would force a deload on
    every new account — a number that is arithmetically correct and completely
    uninformative.
    """
    as_of = as_of or date.today()
    dated = [_as_date(a["start_date"]) for a in activities if a.get("start_date")]
    if not dated or (as_of - min(dated)).days < min_history_days:
        return None

    def load_between(start: date, end: date) -> float:
        total = 0.0
        for a in activities:
            d = _as_date(a["start_date"])
            if start <= d <= end:
                total += float(
                    a.get("training_load") or (a.get("duration_s") or 0) / 60.0
                )
        return total

    acute = load_between(as_of - timedelta(days=6), as_of)
    chronic = load_between(as_of - timedelta(days=27), as_of) / 4.0
    if chronic <= 0:
        return None
    return round(acute / chronic, 2)


ZONE_LABELS = {
    1: "Z1 recovery",
    2: "Z2 aerobic base",
    3: "Z3 tempo",
    4: "Z4 threshold",
    5: "Z5 VO2max",
}


def zone_distribution(
    zone_rows: Sequence[dict[str, Any]],
    sport: str | None = None,
    since: date | None = None,
) -> dict[int, float]:
    """Total minutes per heart-rate zone, optionally filtered."""
    out: dict[int, float] = {z: 0.0 for z in range(1, 6)}
    for r in zone_rows:
        if sport and r.get("sport") != sport:
            continue
        if since and _as_date(r["start_date"]) < since:
            continue
        z = int(r.get("zone_number") or 0)
        if z in out:
            out[z] += float(r.get("secs_in_zone") or 0) / 60.0
    return {z: round(m, 1) for z, m in out.items()}


def polarisation(zone_rows: Sequence[dict[str, Any]], **kw: Any) -> dict[str, float]:
    """Easy / moderate / hard split, as percentages of recorded time.

    Iron Man base wants the large majority easy. A "moderate" bulge — the
    junk-mile zone — is the classic way to accumulate fatigue without adaptation.
    """
    dist = zone_distribution(zone_rows, **kw)
    total = sum(dist.values())
    if total <= 0:
        return {"easy": 0.0, "moderate": 0.0, "hard": 0.0}
    easy = dist[1] + dist[2]
    moderate = dist[3]
    hard = dist[4] + dist[5]
    return {
        "easy": round(easy / total * 100, 1),
        "moderate": round(moderate / total * 100, 1),
        "hard": round(hard / total * 100, 1),
    }


# How much data the EF trend actually needs, so the UI can say so plainly
# instead of showing an empty chart.
EF_MIN_FOR_CHART = 2
EF_MIN_FOR_VERDICT = 3
EF_GOOD_SAMPLE = 6


def ef_data_status(
    activities: Sequence[dict[str, Any]], sport: str
) -> dict[str, Any]:
    """What the athlete needs to do before the EF trend for `sport` means anything."""
    steady = ef_points(activities, sport, steady_only=True)
    considered = [a for a in activities if a.get("sport") == sport]
    rejected: dict[str, int] = {}
    for a in considered:
        if a.get("is_steady"):
            continue
        reason = a.get("steady_reason") or "unknown"
        rejected[reason] = rejected.get(reason, 0) + 1

    n = len(steady)
    if n >= EF_GOOD_SAMPLE:
        message = f"{n} steady sessions — the trend is meaningful."
    elif n >= EF_MIN_FOR_VERDICT:
        message = (
            f"{n} steady sessions — enough for a direction, "
            f"{EF_GOOD_SAMPLE - n} more for a trend you can lean on."
        )
    elif n >= EF_MIN_FOR_CHART:
        message = (
            f"{n} steady sessions — {EF_MIN_FOR_VERDICT - n} more before this "
            f"reads as improving or declining."
        )
    else:
        message = (
            f"{n} steady {sport} session{'s' if n != 1 else ''}. "
            f"Needs {EF_MIN_FOR_VERDICT} to show a direction — "
            f"aim for a 30-60 min conversational effort with heart rate."
        )
    return {
        "sport": sport,
        "steady": n,
        "total": len(considered),
        "needed_for_verdict": max(0, EF_MIN_FOR_VERDICT - n),
        "rejected_reasons": dict(sorted(rejected.items(), key=lambda kv: -kv[1])),
        "message": message,
    }


# --------------------------------------------------------------------------
# Volume / load
# --------------------------------------------------------------------------


def week_summary(
    activities: Sequence[dict[str, Any]],
    week_start: date,
    strength_rows: Sequence[dict[str, Any]] | None = None,
) -> WeekSummary:
    """Completed volume for the Mon-Sun week beginning `week_start`."""
    week_end = week_start + timedelta(days=6)
    summary = WeekSummary(week_start=week_start)
    active_days: set[date] = set()

    for a in activities:
        d = _as_date(a["start_date"])
        if not (week_start <= d <= week_end):
            continue
        sport = a.get("sport") or "other"
        sw = summary.by_sport.setdefault(sport, SportWeek(sport=sport))
        minutes = (a.get("duration_s") or 0) / 60.0
        sw.sessions += 1
        sw.minutes += minutes
        sw.distance_km += (a.get("distance_m") or 0) / 1000.0
        sw.load += float(a.get("training_load") or 0)
        sw.longest_min = max(sw.longest_min, minutes)
        summary.total_minutes += minutes
        summary.total_load += float(a.get("training_load") or 0)
        active_days.add(d)

    for row in strength_rows or []:
        d = _as_date(row["day"])
        if week_start <= d <= week_end:
            active_days.add(d)

    strength = summary.by_sport.get("strength")
    summary.strength_sessions = strength.sessions if strength else 0
    summary.rest_days = 7 - len(active_days)

    for sw in summary.by_sport.values():
        sw.minutes = round(sw.minutes, 1)
        sw.distance_km = round(sw.distance_km, 2)
        sw.load = round(sw.load, 1)
        sw.longest_min = round(sw.longest_min, 1)
    summary.total_minutes = round(summary.total_minutes, 1)
    summary.total_load = round(summary.total_load, 1)
    return summary


def week_summaries(
    activities: Sequence[dict[str, Any]],
    weeks: int = 12,
    as_of: date | None = None,
    strength_rows: Sequence[dict[str, Any]] | None = None,
) -> list[WeekSummary]:
    """Oldest-first list of the last `weeks` weekly summaries."""
    as_of = as_of or date.today()
    this_week = as_of - timedelta(days=as_of.weekday())
    return [
        week_summary(activities, this_week - timedelta(weeks=i), strength_rows)
        for i in range(weeks - 1, -1, -1)
    ]


def totals(activities: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Lifetime headline numbers, overall and per sport."""
    per: dict[str, dict[str, float]] = {}
    for a in activities:
        sport = a.get("sport") or "other"
        row = per.setdefault(sport, {"sessions": 0, "minutes": 0.0, "km": 0.0, "load": 0.0})
        row["sessions"] += 1
        row["minutes"] += (a.get("duration_s") or 0) / 60.0
        row["km"] += (a.get("distance_m") or 0) / 1000.0
        row["load"] += float(a.get("training_load") or 0)
    for row in per.values():
        row["minutes"] = round(row["minutes"], 1)
        row["km"] = round(row["km"], 1)
        row["load"] = round(row["load"], 1)
    dated = sorted(_as_date(a["start_date"]) for a in activities if a.get("start_date"))
    return {
        "sessions": len(activities),
        "minutes": round(sum(r["minutes"] for r in per.values()), 1),
        "km": round(sum(r["km"] for r in per.values()), 1),
        "by_sport": per,
        "first_day": dated[0] if dated else None,
        "last_day": dated[-1] if dated else None,
        "weeks": (
            max(1, ((dated[-1] - dated[0]).days // 7) + 1) if dated else 0
        ),
    }


def baseline_trend(
    wellness: Sequence[dict[str, Any]],
    field: str = "resting_hr",
    as_of: date | None = None,
    lower_is_better: bool = True,
) -> dict[str, Any]:
    """Is a daily metric's baseline moving the right way?

    Compares the most recent 28 days against the 28 before them, which is long
    enough to see through night-to-night noise.
    """
    wellness = sorted(wellness, key=lambda r: r["day"])
    as_of = as_of or (_as_date(wellness[-1]["day"]) if wellness else date.today())
    recent = rolling(wellness, field, 28, as_of)
    prior = rolling(wellness, field, 28, as_of - timedelta(days=28))
    out: dict[str, Any] = {
        "field": field, "recent": _r(recent), "prior": _r(prior),
        "change": None, "verdict": "insufficient_data",
    }
    if recent is None or prior is None:
        return out
    change = recent - prior
    out["change"] = round(change, 2)
    improving = change < 0 if lower_is_better else change > 0
    if abs(change) < (0.5 if lower_is_better else max(0.5, abs(prior) * 0.02)):
        out["verdict"] = "steady"
    else:
        out["verdict"] = "improving" if improving else "worsening"
    return out


def volume_forecast(
    weeks: Sequence[WeekSummary],
    ahead: int = 4,
    cap_pct: float = 10.0,
    deload_factor: float = 0.65,
    block_weeks: int = 4,
    week_index: int = 0,
) -> list[dict[str, Any]]:
    """Project the next few weeks under the progression cap and deload cycle.

    This is what the rules will allow, not a promise — a recovery-triggered
    deload can cap any week at short notice.
    """
    loaded = [w.total_minutes for w in weeks if w.total_minutes > 0]
    current = loaded[-1] if loaded else 0.0
    out = []
    idx = week_index
    for i in range(1, ahead + 1):
        idx = (idx + 1) % block_weeks
        deload = idx == block_weeks - 1
        current = (
            current * deload_factor if deload else current * (1 + cap_pct / 100.0)
        )
        out.append(
            {
                "week_offset": i,
                "minutes": round(current),
                "deload": deload,
                "label": "deload" if deload else f"+{cap_pct:.0f}%",
            }
        )
    return out


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def ols_slope(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx, my = fmean(xs), fmean(ys)
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom


def _pos(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f > 0 and not math.isnan(f) else None


def _r(v: float | None, digits: int = 2) -> float | None:
    return round(float(v), digits) if v is not None else None


def _as_date(v: Any) -> date:
    if isinstance(v, date):
        return v
    return datetime.fromisoformat(str(v)[:10]).date()


def _latest_with(rows: Sequence[dict[str, Any]], fields: tuple[str, ...]) -> dict | None:
    for r in reversed(rows):
        if any(r.get(f) for f in fields):
            return r
    return None


def _last_value(rows: Sequence[dict[str, Any]], field: str) -> float | None:
    for r in reversed(rows):
        if r.get(field) is not None:
            return float(r[field])
    return None
