"""Deterministic training analysis. No AI, no Streamlit, no Garmin.

The headline question this module answers: is output rising at the same heart
rate? That is Efficiency Factor (EF) trended over steady aerobic sessions only —
mixing intervals and races into the trend makes it noise.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
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
    steady_only: bool = False,
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


# Every session, by default. The steady gate exists to keep interval work out of
# a fitness trend, and it is still available to any caller that wants it — but a
# trend the athlete cannot see because their sessions were hard is worse than a
# noisy one they can. The athlete asked for this directly and they are right:
# telling someone "0 of your 4 runs counted" is not analysis, it is a refusal.
def all_ef_trends(
    activities: Iterable[dict[str, Any]],
    as_of: date | None = None,
    steady_only: bool = False,
) -> list[EFTrend]:
    acts = list(activities)
    return [
        ef_trend(acts, s, as_of=as_of, steady_only=steady_only) for s in ENDURANCE_SPORTS
    ]


# --------------------------------------------------------------------------
# Training heart rate — the same question as efficiency factor, but in bpm
# --------------------------------------------------------------------------


def reference_speed(activities: Sequence[dict[str, Any]], sport: str) -> float | None:
    """A stable, personal yardstick pace for one sport: the median of what you
    actually do. Anchoring to a fixed textbook pace would make the number say more
    about the athlete's level than about their progress."""
    speeds = sorted(
        s for s in (
            _pos(a.get("avg_speed_mps")) or _derived_speed(a)
            for a in activities
            if a.get("sport") == sport and _pos(a.get("avg_hr"))
        ) if s
    )
    if not speeds:
        return None
    mid = len(speeds) // 2
    return speeds[mid] if len(speeds) % 2 else (speeds[mid - 1] + speeds[mid]) / 2


def hr_points(
    activities: Sequence[dict[str, Any]], sport: str, ref_speed: float | None = None
) -> list[dict[str, Any]]:
    """Per-session training heart rate, plus the same thing normalised to a
    reference pace.

    Raw average heart rate alone is not progress: a hard session is high because
    it was hard, not because fitness fell. `hr_at_reference` removes that by
    asking what heart rate this session's efficiency implies at one fixed pace —
    so a falling line means the same pace now costs fewer beats. It is efficiency
    factor turned back into a unit you can feel.
    """
    ref = ref_speed if ref_speed is not None else reference_speed(activities, sport)
    out: list[dict[str, Any]] = []
    for a in activities:
        if a.get("sport") != sport:
            continue
        hr = _pos(a.get("avg_hr"))
        if not hr:
            continue
        ef = _pos(a.get("ef"))
        normalised = None
        # Only meaningful for the speed-based metric: watts per beat cannot be
        # turned into "heart rate at a pace".
        if ef and ref and (a.get("ef_metric") or "speed_per_hr") == "speed_per_hr":
            normalised = round(ref * 100.0 / ef, 1)
        out.append(
            {
                "activity_id": a["activity_id"],
                "date": _as_date(a["start_date"]),
                "avg_hr": round(hr, 1),
                "max_hr": _pos(a.get("max_hr")),
                "minutes": round((a.get("duration_s") or 0) / 60.0, 1),
                "speed_mps": _pos(a.get("avg_speed_mps")) or _derived_speed(a),
                "hr_at_reference": normalised,
                "is_steady": bool(a.get("is_steady")),
                "steady_reason": a.get("steady_reason") or "",
            }
        )
    return sorted(out, key=lambda p: p["date"])


def hr_trend(
    activities: Sequence[dict[str, Any]],
    sport: str,
    recent_days: int = 28,
    baseline_days: int = 56,
    as_of: date | None = None,
    steady_only: bool = False,
) -> dict[str, Any]:
    """Is the heart rate you train at coming down for the same pace?

    Reported in bpm, and negative is good — the opposite sign convention to
    efficiency factor, which is exactly why it is easier to read.
    """
    ref = reference_speed(activities, sport)
    pts = hr_points(activities, sport, ref)
    if steady_only:
        pts = [p for p in pts if p["is_steady"]]
    out: dict[str, Any] = {
        "sport": sport, "reference_speed_mps": ref, "n_sessions": len(pts),
        "recent_hr": None, "baseline_hr": None, "change_bpm": None,
        "recent_normalised": None, "baseline_normalised": None,
        "normalised_change_bpm": None, "verdict": "insufficient_data",
    }
    if not pts:
        return out
    today = as_of or pts[-1]["date"]
    rec_cut = today - timedelta(days=recent_days)
    base_cut = today - timedelta(days=baseline_days)

    def mean_of(rows: list[dict[str, Any]], field: str) -> float | None:
        vals = [r[field] for r in rows if r.get(field)]
        return round(fmean(vals), 1) if vals else None

    recent = [p for p in pts if p["date"] > rec_cut]
    baseline = [p for p in pts if base_cut <= p["date"] <= rec_cut]
    out["recent_hr"] = mean_of(recent, "avg_hr")
    out["baseline_hr"] = mean_of(baseline, "avg_hr")
    out["recent_normalised"] = mean_of(recent, "hr_at_reference")
    out["baseline_normalised"] = mean_of(baseline, "hr_at_reference")
    if out["recent_hr"] is not None and out["baseline_hr"] is not None:
        out["change_bpm"] = round(out["recent_hr"] - out["baseline_hr"], 1)
    if out["recent_normalised"] is not None and out["baseline_normalised"] is not None:
        out["normalised_change_bpm"] = round(
            out["recent_normalised"] - out["baseline_normalised"], 1
        )

    signal = out["normalised_change_bpm"]
    if signal is None or len(pts) < 3:
        out["verdict"] = "insufficient_data"
    elif signal <= -1.0:
        out["verdict"] = "improving"     # same pace, fewer beats
    elif signal >= 1.0:
        out["verdict"] = "worsening"
    else:
        out["verdict"] = "flat"
    return out


# What an easy base block should look like, and the bands each driver is judged
# against. All of them are the same numbers the planner and the Rules page use;
# none is invented here.
EASY_SHARE_TARGET = 70.0
STRENGTH_TARGET_PER_WEEK = 2.0
CONSISTENCY_TARGET = 0.5          # active days as a share of the window


def fitness_drivers(
    activities: Sequence[Mapping[str, Any]],
    wellness: Sequence[Mapping[str, Any]] | None = None,
    zone_rows: Sequence[Mapping[str, Any]] | None = None,
    strength_rows: Sequence[Mapping[str, Any]] | None = None,
    as_of: date | None = None,
    sport: str = "run",
    days: int = 28,
) -> dict[str, Any]:
    """Why the trend is going the way it is, from the numbers rather than a model.

    "Fitness is rising" on its own tells an athlete nothing they can act on. What
    they need is which of their own habits is producing it, and which is working
    against it — and every one of these is already stored: how much of the
    training was easy, whether the hours are growing, how often they showed up,
    what resting heart rate and HRV are doing, whether the load is in the
    productive band, and whether the weather is masking any of it.

    Each driver comes back as helping, holding back, or worth watching, with the
    number behind it. Nothing is weighted or scored into a single index: a made-up
    composite would look authoritative and mean nothing, and the athlete can read
    five plain lines perfectly well.
    """
    as_of = as_of or date.today()
    trend = hr_trend(activities, sport, as_of=as_of, steady_only=False)
    out: dict[str, Any] = {
        "sport": sport,
        "direction": {"improving": "rising", "worsening": "falling",
                      "flat": "flat"}.get(trend["verdict"], "unknown"),
        "change_bpm": trend["normalised_change_bpm"],
        "helping": [], "holding_back": [], "watch": [],
    }

    def add(where: str, name: str, value: str, detail: str) -> None:
        out[where].append({"name": name, "value": value, "detail": detail})

    # 1. How much of it was easy. The single biggest lever in a base block, and
    #    the one most often got wrong in the same direction.
    if zone_rows:
        split = polarisation(zone_rows, since=as_of - timedelta(days=days))
        easy = split.get("easy")
        if easy is not None and split.get("samples", 1):
            if easy >= EASY_SHARE_TARGET:
                add("helping", "Easy training", f"{easy:.0f}% easy",
                    f"At or above the {EASY_SHARE_TARGET:.0f}% a base block wants, "
                    f"so the aerobic work is going in without the fatigue.")
            else:
                add("holding_back", "Too much intensity", f"{easy:.0f}% easy",
                    f"A base block wants {EASY_SHARE_TARGET:.0f}%+. Hard sessions "
                    f"cost recovery you could be spending on volume.")

    # 2. Volume, this window against the one before it.
    now_min = _minutes_between(activities, as_of - timedelta(days=days - 1), as_of)
    before_min = _minutes_between(activities, as_of - timedelta(days=days * 2 - 1),
                                  as_of - timedelta(days=days))
    if now_min or before_min:
        if before_min > 0:
            change = (now_min - before_min) / before_min * 100
            words = f"{now_min / 60:.1f}h vs {before_min / 60:.1f}h"
            if change >= 8:
                add("helping", "Growing volume", words,
                    f"{change:+.0f}% on the previous {days} days. More aerobic "
                    f"hours is the most reliable way this number moves.")
            elif change <= -12:
                add("holding_back", "Volume dropped", words,
                    f"{change:+.0f}% on the previous {days} days. Fitness follows "
                    f"hours, and these have gone down.")
            else:
                add("helping", "Steady volume", words,
                    "Held within a tenth of the previous block, which is what "
                    "lets adaptation accumulate.")
        elif now_min:
            add("helping", "Training started", f"{now_min / 60:.1f}h",
                f"Nothing in the {days} days before this, so there is no "
                f"comparison yet — only a baseline being built.")

    # 3. Showing up. Endurance is mostly an attendance problem.
    all_days = [_as_date(a["start_date"]) for a in activities if a.get("start_date")]
    active = {d for d in all_days
              if d and as_of - timedelta(days=days - 1) <= d <= as_of}
    # Counted against the days there was a record to show up in, not a flat 28.
    # A watch eight days old would otherwise be told it missed twenty days.
    first = min((d for d in all_days if d), default=None)
    window = days
    if first and first > as_of - timedelta(days=days - 1):
        window = max(1, (as_of - first).days + 1)
    share = len(active) / window if window else 0
    if active:
        if share >= CONSISTENCY_TARGET:
            add("helping", "Consistency", f"{len(active)}/{window} days",
                f"{share * 100:.0f}% of days had something logged. Regularity "
                f"beats big weeks with gaps in them.")
        else:
            add("holding_back", "Gaps", f"{len(active)}/{window} days",
                f"Only {share * 100:.0f}% of days had a session. The gaps cost "
                f"more than the sessions gain.")

    # 4 and 5. What the body says about all of it.
    rows = list(wellness or [])
    if rows:
        rhr = baseline_trend(rows, "resting_hr", as_of=as_of, lower_is_better=True)
        if rhr["verdict"] == "improving":
            add("helping", "Resting heart rate falling",
                f"{rhr['recent']:.0f} bpm",
                "Down against your own 28-day baseline, which is the clearest "
                "sign the engine is adapting.")
        elif rhr["verdict"] == "worsening":
            add("watch", "Resting heart rate rising", f"{rhr['recent']:.0f} bpm",
                "Up against your baseline. Usually load, sleep or something "
                "coming on rather than lost fitness.")
        hrv = baseline_trend(rows, "hrv_last_night", as_of=as_of,
                            lower_is_better=False)
        if hrv["verdict"] == "improving":
            add("helping", "HRV rising", f"{hrv['recent']:.0f} ms",
                "Overnight HRV above your baseline: the training is being "
                "absorbed rather than survived.")
        elif hrv["verdict"] == "worsening":
            add("watch", "HRV falling", f"{hrv['recent']:.0f} ms",
                "Below your baseline. Worth an easy day before a hard one.")
        sleep = rolling(rows, "sleep_seconds", 14, as_of)
        if sleep:
            hours = sleep / 3600
            if hours >= 7:
                add("helping", "Sleep", f"{hours:.1f}h",
                    "Averaging seven hours or more over two weeks, which is "
                    "where adaptation actually happens.")
            elif hours < 6.5:
                add("holding_back", "Short sleep", f"{hours:.1f}h",
                    "Under six and a half hours on average. The training is "
                    "going in; the recovery it needs is not.")

    # 6. Load, against the athlete's own four weeks.
    ratio = acwr_from_activities(activities, as_of=as_of)
    if ratio is not None:
        if ACWR_LOW <= ratio <= ACWR_HIGH:
            add("helping", "Load in the productive band", f"{ratio:.2f}",
                f"Between {ACWR_LOW} and {ACWR_HIGH}: enough stimulus to adapt "
                f"to, not more than you are absorbing.")
        elif ratio < ACWR_LOW:
            add("watch", "Load is light", f"{ratio:.2f}",
                "This week is well under your own four-week average. Room to "
                "add, if recovery is good.")
        else:
            add("watch", "Load is running hot", f"{ratio:.2f}",
                f"Above {ACWR_HIGH}. A rise here is where injuries come from, "
                f"not where fitness does.")

    # 7. Legs. Not fitness, but the reason run volume survives.
    legs = _strength_sessions(activities, strength_rows, as_of, days)
    per_week = legs / (days / 7) if days else 0
    if legs:
        if per_week >= STRENGTH_TARGET_PER_WEEK - 0.25:
            add("helping", "Leg work", f"{per_week:.1f}/week",
                "At the twice-a-week the plan asks for. Tendons are what let "
                "run volume keep climbing.")
        else:
            add("holding_back", "Leg work light", f"{per_week:.1f}/week",
                "Under twice a week. It does not raise this number directly; it "
                "protects the running that does.")
    else:
        add("holding_back", "No leg work", "0/week",
            "Nothing logged in this window. Two short sessions a week is what "
            "protects the run volume this all depends on.")

    return out


def add_heat_driver(drivers: dict[str, Any], effect: Mapping[str, Any]) -> None:
    """Fold the weather reading into an existing set of drivers.

    Separate because it needs the weather table, which `fitness_drivers` is
    deliberately not given: everything else here comes from data that exists for
    every athlete, and conditions are only stored for sessions the sync could
    look up. It goes last on the page because it changes how to read every line
    above it — a hot block hides real fitness rather than preventing it.
    """
    hot_share = effect.get("hot_share")
    if not hot_share:
        return
    per_degree = effect.get("bpm_per_deg")
    detail = (f"{hot_share:.0f}% of these sessions were in air too humid to cool "
              f"in")
    if per_degree:
        detail += (f", and each degree of dew point costs about "
                   f"{per_degree:+.1f} bpm at the same pace")
    detail += (". Heart rate at a given pace reads high in this weather, so the "
               "trend above is understating you rather than the other way "
               "round.")
    drivers.setdefault("watch", []).append(
        {"name": "Heat and humidity", "value": f"{hot_share:.0f}% muggy",
         "detail": detail})


def _minutes_between(activities: Sequence[Mapping[str, Any]], start: date,
                     end: date) -> float:
    """Total endurance minutes in a window, inclusive."""
    total = 0.0
    for a in activities:
        if not a.get("start_date") or a.get("sport") not in ENDURANCE_SPORTS:
            continue
        day = _as_date(a["start_date"])
        if day and start <= day <= end:
            total += (a.get("duration_s") or 0) / 60.0
    return round(total, 1)


def _strength_sessions(activities: Sequence[Mapping[str, Any]],
                       strength_rows: Sequence[Mapping[str, Any]] | None,
                       as_of: date, days: int) -> int:
    """Days with leg work, from the watch's own sessions or the app's log."""
    start = as_of - timedelta(days=days - 1)
    seen: set[date] = set()
    for a in activities:
        if a.get("sport") == "strength" and a.get("start_date"):
            day = _as_date(a["start_date"])
            if day and start <= day <= as_of:
                seen.add(day)
    for row in strength_rows or []:
        day = _as_date(row.get("day"))
        if day and start <= day <= as_of:
            seen.add(day)
    return len(seen)


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


# What Garmin's training-status phrases mean, in the athlete's words. The keys
# are the phrase names with their numeric suffix removed: Garmin sends
# PRODUCTIVE_3, PRODUCTIVE_2 and so on, where the number selects which wording
# its own app shows and carries no extra meaning worth keeping.
STATUS_MEANING = {
    "productive": "fitness is rising",
    "maintaining": "holding the fitness you have",
    "recovery": "a light stretch, letting adaptation happen",
    "unproductive": "training is going in, fitness is not coming out",
    "peaking": "race-ready",
    "overreaching": "load is ahead of what your recovery is supporting",
    "strained": "hard recent load against low recovery",
    "high_strain": "hard recent load against low recovery",
    "detraining": "too little training to hold your fitness",
}

# Not statuses at all: placeholders Garmin sends while it works one out.
STATUS_PLACEHOLDERS = ("no_status", "unknown", "none", "not_set", "onboarding",
                       "undefined", "unrecognized")


def status_key(value: object) -> str | None:
    """The status as a lookup key — lowercase, no trailing variant number."""
    text = str(value or "").strip().lower().replace(" ", "_")
    if not text or any(text.startswith(p) for p in STATUS_PLACEHOLDERS):
        return None
    # PRODUCTIVE_3 -> productive. Only a trailing number goes; HIGH_STRAIN keeps
    # the word that makes it a different status.
    parts = text.split("_")
    while len(parts) > 1 and parts[-1].isdigit():
        parts.pop()
    return "_".join(parts) or None


def clean_status(value: object) -> str | None:
    """A Garmin status in plain words, or None when it is a placeholder.

    Two jobs, both learned the hard way. 47 wellness rows were stored with
    "NO_STATUS_2" in them before that was recognised as a sentinel, and the
    dashboard showed it as the athlete's training status. Then a real status
    arrived and did the same thing in a different way: "PRODUCTIVE_3" is an enum
    name, not something to put on a page, and the 3 does not mean a level.
    """
    key = status_key(value)
    if not key:
        return None
    return key.replace("_", " ").title()


def status_meaning(value: object) -> str | None:
    """What that status is telling the athlete, or None if it is not one of the
    ones Garmin documents."""
    key = status_key(value)
    return STATUS_MEANING.get(key or "")


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
    sig.training_status = clean_status(
        (_latest_with(wellness, ("training_status",)) or {}).get(
            "training_status"
        )
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


def zone_bounds(zone_rows: Sequence[dict[str, Any]]) -> dict[int, tuple[int, int | None]]:
    """The athlete's own heart-rate zone boundaries, in bpm.

    Taken from what Garmin reported per activity rather than derived from a
    max-HR formula, so the numbers match what the watch will show mid-session.
    Returns {zone: (low, high)}; the top zone has no upper bound.
    """
    lows: dict[int, float] = {}
    for r in zone_rows:
        z = int(r.get("zone_number") or 0)
        low = r.get("zone_low_bpm")
        if z and low:
            # Garmin can revise the boundaries; the most recent wins.
            lows[z] = float(low)
    if not lows:
        return {}
    out: dict[int, tuple[int, int | None]] = {}
    for z in sorted(lows):
        nxt = lows.get(z + 1)
        out[z] = (int(round(lows[z])), int(round(nxt)) - 1 if nxt else None)
    return out


def aerobic_ceiling_options(
    bounds: dict[int, tuple[int, int | None]], threshold_hr: float | None
) -> dict[str, Any]:
    """What "easy" should mean in bpm for this athlete, and the case for each answer.

    Garmin's zones here are anchored to lactate threshold, not to an estimated
    max HR — Z5 begins exactly at the reported threshold. That makes its Z2
    ceiling 77% of threshold, which is conservative next to the common
    threshold-anchored schemes: Friel puts aerobic endurance at 81-89%, and
    "below the first lactate threshold" is usually taken as roughly 85%.

    So an athlete who feels that Garmin's ceiling is too low is often right. This
    returns the candidates rather than picking one, because the honest answer is
    a talk test — the highest heart rate at which you can still speak in full
    sentences — and that is not in the data.
    """
    garmin = bounds.get(2, (None, None))[1]
    out: dict[str, Any] = {
        "garmin_z2_top": garmin,
        "threshold_hr": int(threshold_hr) if threshold_hr else None,
        "candidates": [],
    }
    if not threshold_hr:
        return out
    t = float(threshold_hr)
    if garmin:
        out["candidates"].append({
            "bpm": int(garmin), "label": "Garmin Z2 ceiling",
            "note": f"{garmin / t * 100:.0f}% of threshold — conservative",
        })
    out["candidates"].extend([
        {"bpm": round(t * 0.83), "label": "Coggan endurance top",
         "note": "83% of threshold"},
        {"bpm": round(t * 0.85), "label": "below LT1 (polarised)",
         "note": "85% of threshold — the common aerobic-base ceiling"},
        {"bpm": round(t * 0.89), "label": "Friel aerobic top",
         "note": "89% of threshold — the upper end of defensible"},
    ])
    return out


def zone_target(
    bounds: dict[int, tuple[int, int | None]],
    zone: str,
    aerobic_ceiling: int | None = None,
) -> str | None:
    """"Z2" -> "112-129 bpm". None when the zone is not a numbered one.

    `aerobic_ceiling` replaces the top of Z2 only. Garmin's Z2 ceiling is
    conservative for many athletes, and the target on the plan should be the
    number the athlete actually intends to train to.
    """
    if not bounds or not zone or not zone.upper().startswith("Z"):
        return None
    try:
        z = int(zone.upper().lstrip("Z"))
    except ValueError:
        return None
    span = bounds.get(z)
    if not span:
        return None
    low, high = span
    if z == 2 and aerobic_ceiling and aerobic_ceiling > low:
        high = int(aerobic_ceiling)
    return f"{low}-{high} bpm" if high else f"{low}+ bpm"


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

    Endurance base wants the large majority easy. A "moderate" bulge — the
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


# A lap has to be long enough for heart rate to have settled before its average
# means anything. Under three minutes is a warm-up segment or a stop.
LAP_MIN_SECONDS = 150.0
# How closely two laps must match on pace before their heart rates can be
# compared. 5% is tight enough that pace is not the explanation.
LAP_PACE_TOLERANCE = 0.05


def lap_drift(laps: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """How much heart rate climbed across a session at unchanged pace.

    This is the same question aerobic decoupling asks, answered from auto-lap
    instead of from the stream. Decoupling needs a session long enough to split
    into two halves that are each worth comparing — in practice an hour — and
    this athlete's longest session is 45 minutes, so that number has never once
    computed. Laps give it at kilometre resolution.

    Only laps within `LAP_PACE_TOLERANCE` of the first usable lap's pace are
    compared, because heart rate rising with pace is not drift, it is effort.
    Returns None-ish when there is nothing honest to say.
    """
    usable = [
        l for l in laps
        if (l.get("duration_s") or 0) >= LAP_MIN_SECONDS
        and l.get("avg_hr") and l.get("avg_speed_mps")
    ]
    if len(usable) < 3:
        return {"drift_bpm": None, "laps_compared": 0,
                "message": "needs three laps of three minutes or more"}

    usable.sort(key=lambda l: l["lap_index"])
    reference = float(usable[0]["avg_speed_mps"])
    matched = [
        l for l in usable
        if abs(float(l["avg_speed_mps"]) - reference) / reference
        <= LAP_PACE_TOLERANCE
    ]
    if len(matched) < 3:
        return {"drift_bpm": None, "laps_compared": len(matched),
                "message": "pace varied too much across laps to separate drift "
                           "from effort"}

    first, last = float(matched[0]["avg_hr"]), float(matched[-1]["avg_hr"])
    drift = last - first
    per_km = None
    covered = sum(float(l.get("distance_m") or 0) for l in matched) / 1000.0
    if covered > 0:
        per_km = drift / covered

    # Plain wording: this is read on the page, so it says what happened and what
    # it means without a training term in it.
    if drift <= 3:
        verdict = "flat"
        message = (f"Your heart rate stayed within {abs(drift):.0f} beats across "
                   f"{len(matched)} kilometres at the same pace. That is what "
                   f"holding it together looks like.")
    elif drift <= 10:
        verdict = "mild"
        message = (f"Your heart rate rose {drift:.0f} beats across "
                   f"{len(matched)} kilometres at the same pace. Normal on a "
                   f"longer run, worth watching if it grows.")
    else:
        verdict = "steep"
        message = (f"Your heart rate rose {drift:.0f} beats across "
                   f"{len(matched)} kilometres without running any faster. The "
                   f"run got harder but not quicker, which usually means it "
                   f"started too fast.")
    return {
        "drift_bpm": round(drift, 1),
        "drift_per_km": round(per_km, 1) if per_km is not None else None,
        "first_hr": round(first), "last_hr": round(last),
        "laps_compared": len(matched), "laps_total": len(laps),
        "verdict": verdict, "message": message,
    }


def lap_pace_spread(laps: Sequence[dict[str, Any]]) -> float | None:
    """Coefficient of variation of lap pace, as a percentage.

    A steady session holds one pace; intervals do not. This is a more direct
    reading of "was this steady" than heart-rate spread, which also moves with
    heat, fatigue and hills.
    """
    speeds = [float(l["avg_speed_mps"]) for l in laps
              if (l.get("duration_s") or 0) >= LAP_MIN_SECONDS
              and l.get("avg_speed_mps")]
    if len(speeds) < 3:
        return None
    mean = fmean(speeds)
    if mean <= 0:
        return None
    return round(pstdev(speeds) / mean * 100, 1)


EARTH_RADIUS_M = 6371008.8
# A split shorter than this fraction of the unit is not shown as its own row.
# The remainder of a 10.04 km ride is 40 metres, and a row saying "0.04 km at
# 3:12" is noise standing next to nine real kilometres.
MIN_PARTIAL_SPLIT = 0.15


def _metres_between(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two fixes, in metres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = (math.sin(dp / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
    return 2 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


def km_splits(
    stream: Sequence[Mapping[str, Any]],
    unit_m: float = 1000.0,
    total_m: float | None = None,
) -> list[dict[str, Any]]:
    """Even splits from the stored samples — the per-kilometre table, not the laps.

    A lap is whatever the athlete pressed. This ride's five were 203 m, 5 km,
    1.9 km, 1.3 km and 1.6 km, so numbering them 1 to 5 read as five kilometres
    of a ten-kilometre ride. These are cut at a fixed distance instead, which is
    what every splits table anyone has seen actually shows.

    Distance comes from GPS where there is any, because it is measured rather
    than inferred; from speed where there is not, which is the case in a pool.
    `total_m` scales the result so the last split ends where the recorded
    distance says it does — GPS over a downsampled stream lands a percent or two
    short, and that error should not become an extra row.
    """
    samples = [r for r in (stream or []) if r.get("t_s") is not None]
    if len(samples) < 4 or unit_m <= 0:
        return []
    samples.sort(key=lambda r: float(r["t_s"]))

    # Step distances first, so the two sources can be compared before choosing.
    gps, speed = [0.0], [0.0]
    for before, after in zip(samples, samples[1:]):
        dt = max(0.0, float(after["t_s"]) - float(before["t_s"]))
        if (before.get("lat") is not None and after.get("lat") is not None
                and before.get("lon") is not None and after.get("lon") is not None):
            gps.append(_metres_between(float(before["lat"]), float(before["lon"]),
                                       float(after["lat"]), float(after["lon"])))
        else:
            gps.append(0.0)
        speed.append(float(after.get("speed_mps") or before.get("speed_mps") or 0) * dt)
    steps = gps if sum(gps) > sum(speed) * 0.5 else speed
    covered = sum(steps)
    if covered <= 0:
        return []
    if total_m and total_m > 0:
        scale = float(total_m) / covered
        steps = [s * scale for s in steps]
        covered = float(total_m)

    out: list[dict[str, Any]] = []
    distance = 0.0            # cumulative, over the whole session
    split_start_t = float(samples[0]["t_s"])
    split_distance = 0.0
    beats: list[tuple[float, float]] = []   # (hr, seconds it stood for)
    climb = 0.0
    boundary = unit_m

    def close(at_time: float, length: float, partial: bool) -> None:
        held = sum(w for _, w in beats)
        out.append({
            "index": len(out) + 1,
            "distance_m": round(length, 1),
            "seconds": round(max(0.0, at_time - split_start_t), 1),
            "avg_hr": (round(sum(hr * w for hr, w in beats) / held, 1)
                       if held else None),
            "elev_gain_m": round(climb, 1),
            "partial": partial,
            "speed_mps": (round(length / (at_time - split_start_t), 3)
                          if at_time > split_start_t else None),
        })

    # Walked by index rather than zipped, because each step's distance was
    # computed by index above and pairing them any other way invites them to
    # drift apart.
    for n in range(1, len(samples)):
        before, after = samples[n - 1], samples[n]
        step = steps[n]
        t_before, t_after = float(before["t_s"]), float(after["t_s"])
        dt = max(0.0, t_after - t_before)
        hr = after.get("hr") if after.get("hr") is not None else before.get("hr")
        if hr is not None and dt > 0:
            beats.append((float(hr), dt))
        if (before.get("altitude_m") is not None
                and after.get("altitude_m") is not None):
            climb += max(0.0, float(after["altitude_m"]) - float(before["altitude_m"]))
        distance += step
        split_distance += step

        # One sample can cross more than one boundary on a downsampled ride.
        while distance >= boundary - 1e-9 and step > 0:
            over = distance - boundary
            share = 1.0 - (over / step)
            at = t_before + dt * max(0.0, min(1.0, share))
            close(at, unit_m, partial=False)
            split_start_t = at
            split_distance = over
            beats = [(float(hr), max(0.0, t_after - at))] if hr is not None else []
            climb = 0.0
            boundary += unit_m

    if split_distance >= unit_m * MIN_PARTIAL_SPLIT:
        close(float(samples[-1]["t_s"]), split_distance, partial=True)
    return out


def zone_bounds_with_ceiling(
    bounds: dict[int, tuple[int, int | None]], ceiling: float | None,
) -> dict[int, tuple[int, int | None]]:
    """Garmin's zones with zone 2's top moved to the athlete's own ceiling.

    Only that one boundary moves. Zone 1's floor, and the thresholds above zone
    3, are physiological and come from the watch; where "easy" stops is the
    judgement the athlete has actually made, and it is the boundary the whole
    dashboard is arguing about.
    """
    if not bounds or not ceiling:
        return dict(bounds)
    top = int(ceiling)
    out = dict(bounds)
    z2 = out.get(2)
    z3 = out.get(3)
    if z2 and top > z2[0]:
        out[2] = (z2[0], top)
        if z3:
            out[3] = (top + 1, z3[1])
    return out


def zone_of(hr: float, bounds: dict[int, tuple[int, int | None]]) -> int | None:
    """Which zone a single heart-rate reading falls in."""
    for number in sorted(bounds):
        low, high = bounds[number]
        if hr >= low and (high is None or hr <= high):
            return number
    return None


def zone_distribution_from_streams(
    streams: dict[str, Sequence[dict[str, Any]]],
    activities: Sequence[dict[str, Any]],
    bounds: dict[int, tuple[int, int | None]],
    sport: str | None = None,
    since: date | None = None,
) -> dict[int, float]:
    """Minutes per zone, from the stored samples and the given boundaries.

    Exists so the zone charts follow the athlete's ceiling rather than Garmin's.
    Garmin's own time-in-zone rows are computed on the watch against its fixed
    bands, so no amount of relabelling makes them reflect a raised zone 2 — the
    minutes have to be recounted.

    Scaled by each activity's duration rather than counted raw: streams are
    downsampled to at most 600 points, so one sample stands for a different
    slice of time in a 20-minute run than in a two-hour ride.
    """
    per_activity = {str(a.get("activity_id")): a for a in activities}
    out: dict[int, float] = {z: 0.0 for z in (bounds or {1: 0, 2: 0, 3: 0, 4: 0, 5: 0})}
    if not bounds:
        return out

    for activity_id, samples in (streams or {}).items():
        activity = per_activity.get(activity_id)
        if activity is None:
            continue
        if sport and (activity.get("sport") or "") != sport:
            continue
        if since is not None:
            day = _as_date(activity.get("start_date")) if activity.get("start_date") else None
            if day is None or day < since:
                continue
        readings = [r["hr"] for r in samples or () if r.get("hr") is not None]
        if not readings:
            continue
        minutes_each = ((activity.get("duration_s") or 0) / 60.0) / len(readings)
        for hr in readings:
            zone = zone_of(float(hr), bounds)
            if zone is not None:
                out[zone] = out.get(zone, 0.0) + minutes_each
    return {z: round(v, 1) for z, v in out.items()}


# Common coaching target for running cadence. Not a law — taller runners sit
# lower, and cadence rises with pace — but below this the stride is usually long
# enough that the foot lands well in front of the body, which brakes and loads
# the knee and shin.
CADENCE_TARGET_SPM = 170.0
CADENCE_LOW_SPM = 160.0


def cadence_stats(
    activities: Sequence[dict[str, Any]], sport: str = "run",
    since: date | None = None,
) -> dict[str, Any]:
    """Cadence and stride length, and what they say together.

    Reported as a pair on purpose. Cadence alone is not a target you can chase:
    it rises naturally with pace, so a faster session shows a higher number
    without anything having improved. Stride length is what closes the loop —
    the same pace at a higher cadence means a shorter stride, which means the
    foot lands closer to underneath the body.
    """
    rows = []
    for a in activities:
        if (a.get("sport") or "") != sport:
            continue
        if since is not None:
            day = _as_date(a.get("start_date")) if a.get("start_date") else None
            if day is None or day < since:
                continue
        cad = a.get("avg_cadence")
        if not cad:
            continue
        # Stride length is derived when Garmin has not sent it. The list endpoint
        # omits the running-dynamics fields that the detail endpoint carries, and
        # the arithmetic is exact rather than an estimate: distance per step is
        # speed divided by steps per second. Checked against a session where
        # Garmin did report it — 87.4 cm derived against 86.74 reported.
        stride = a.get("stride_length_cm")
        if not stride and a.get("avg_speed_mps") and float(cad) > 0:
            stride = float(a["avg_speed_mps"]) / (float(cad) / 60.0) * 100.0
        rows.append({
            "date": _as_date(a["start_date"]),
            "cadence": float(cad),
            "stride_cm": round(float(stride), 1) if stride else None,
            "ground_contact_ms": float(a["ground_contact_ms"])
                                 if a.get("ground_contact_ms") else None,
            "vertical_ratio": float(a["vertical_ratio"])
                              if a.get("vertical_ratio") else None,
            "speed_mps": float(a["avg_speed_mps"]) if a.get("avg_speed_mps") else None,
            "minutes": round((a.get("duration_s") or 0) / 60.0, 1),
        })
    rows.sort(key=lambda r: r["date"])
    if not rows:
        return {"points": [], "avg": None, "verdict": "no_data", "message":
                f"No {sport} sessions with cadence recorded yet."}

    avg = sum(r["cadence"] for r in rows) / len(rows)
    strides = [r["stride_cm"] for r in rows if r["stride_cm"]]
    if avg < CADENCE_LOW_SPM:
        verdict, message = "low", (
            f"Averaging {avg:.0f} steps per minute. Under {CADENCE_LOW_SPM:.0f} "
            f"usually means the stride is long and the foot is landing ahead of "
            f"you, which brakes and loads the knee and shin. Adding 5% is the "
            f"cheapest change available."
        )
    elif avg < CADENCE_TARGET_SPM:
        verdict, message = "fair", (
            f"Averaging {avg:.0f} steps per minute — workable, and a few percent "
            f"quicker would shorten the stride slightly at the same pace."
        )
    else:
        verdict, message = "good", (
            f"Averaging {avg:.0f} steps per minute, which is in the range most "
            f"coaching aims for. Nothing to change."
        )
    return {
        "points": rows,
        "avg": round(avg, 1),
        "avg_stride_cm": round(sum(strides) / len(strides), 1) if strides else None,
        "target": CADENCE_TARGET_SPM,
        "verdict": verdict,
        "message": message,
    }


def polarisation_from_streams(
    streams: dict[str, Sequence[dict[str, Any]]],
    activities: Sequence[dict[str, Any]],
    ceiling: float,
    hard_floor: float | None = None,
    sport: str | None = None,
    since: date | None = None,
) -> dict[str, Any]:
    """Easy / moderate / hard from the heart-rate samples, against `ceiling`.

    Exists because the zone version cannot answer the question the athlete is
    actually asking. `polarisation()` buckets Garmin's own time-in-zone rows, and
    Garmin's Z2 top is fixed — so an athlete who has deliberately set a higher
    aerobic ceiling sees every minute spent between the two counted as
    "moderate", and their easy share never improves no matter how they train.

    Here "easy" means at or below the ceiling they chose. `hard_floor` should be
    the athlete's Z4 lower bound, which is what Garmin itself treats as hard —
    passing anything else makes this number incomparable with the zone version
    and changes two variables at once when only one was configured.

    Computed from the stored samples, so changing the ceiling changes the answer
    without refetching anything from Garmin.

    Sample-counted rather than duration-weighted: streams are downsampled at a
    constant interval per activity, so each sample stands for the same slice of
    time within that activity.
    """
    # Default only as a fallback. The caller should pass the real Z4 boundary.
    floor = float(hard_floor) if hard_floor else float(ceiling) * 1.18
    if floor <= ceiling:
        floor = float(ceiling) + 1.0
    by_sport = {str(a.get("activity_id")): (a.get("sport") or "") for a in activities}
    days = {str(a.get("activity_id")): a.get("start_date") for a in activities}

    easy = moderate = hard = 0
    counted = 0
    for activity_id, samples in (streams or {}).items():
        if sport and by_sport.get(activity_id) != sport:
            continue
        if since is not None:
            day = _as_date(days.get(activity_id)) if days.get(activity_id) else None
            if day is None or day < since:
                continue
        used = False
        for row in samples or ():
            hr = row.get("hr")
            if hr is None:
                continue
            used = True
            if hr <= ceiling:
                easy += 1
            elif hr >= floor:
                hard += 1
            else:
                moderate += 1
        counted += int(used)

    total = easy + moderate + hard
    if total <= 0:
        return {"easy": 0.0, "moderate": 0.0, "hard": 0.0, "samples": 0,
                "activities": 0, "ceiling": int(ceiling),
                "hard_floor": int(floor)}
    return {
        "easy": round(easy / total * 100, 1),
        "moderate": round(moderate / total * 100, 1),
        "hard": round(hard / total * 100, 1),
        "samples": total,
        "activities": counted,
        "ceiling": int(ceiling),
        "hard_floor": int(floor),
    }


# How much data the EF trend actually needs, so the UI can say so plainly
# instead of showing an empty chart.
EF_MIN_FOR_CHART = 2
EF_MIN_FOR_VERDICT = 3
EF_GOOD_SAMPLE = 6


def ef_data_status(
    activities: Sequence[dict[str, Any]], sport: str
) -> dict[str, Any]:
    """How much there is to judge the trend on, and how noisy it will be.

    This used to report how many sessions had been *excluded* for being hard, and
    tell the athlete their four runs did not count. That is a refusal dressed up
    as analysis. Every session counts now; hard ones make the line jumpier, which
    is worth saying once and is not the same as having no data.
    """
    every = ef_points(activities, sport)
    steady = [p for p in every if p.is_steady]
    n, hard = len(every), len(every) - len(steady)

    if n == 0:
        message = f"No {sport} sessions with heart rate yet."
    elif n < EF_MIN_FOR_CHART:
        message = (f"{n} {sport} session{'s' if n != 1 else ''}. Two draws a "
                   f"line, {EF_MIN_FOR_VERDICT} gives it a direction.")
    elif n < EF_MIN_FOR_VERDICT:
        message = (f"{n} {sport} sessions — {EF_MIN_FOR_VERDICT - n} more before "
                   f"this reads as improving or declining.")
    elif n < EF_GOOD_SAMPLE:
        message = (f"{n} {sport} sessions — enough for a direction, "
                   f"{EF_GOOD_SAMPLE - n} more for one you can lean on.")
    else:
        message = f"{n} {sport} sessions — the trend is meaningful."

    if hard and n:
        share = hard / n * 100
        message += (f" {hard} of them ran hard or uneven ({share:.0f}%), which "
                    f"makes the line jump about; easy sessions are what make it "
                    f"smooth.")

    return {
        "sport": sport,
        "sessions": n,
        "steady": len(steady),
        "hard": hard,
        "total": n,
        "needed_for_verdict": max(0, EF_MIN_FOR_VERDICT - n),
        "rejected_reasons": {},
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


MIN_TREND_POINTS = 6
MIN_TREND_SPAN_DAYS = 10


def baseline_trend(
    wellness: Sequence[dict[str, Any]],
    field: str = "resting_hr",
    as_of: date | None = None,
    lower_is_better: bool = True,
    window_days: int = 42,
) -> dict[str, Any]:
    """Direction of travel for a daily metric, as a slope per week.

    Comparing the last 28 days against the previous 28 needs eight weeks of data
    before it says anything at all, which is useless to someone who started three
    weeks ago. Fitting a line to whatever exists answers the same question sooner
    and uses every night rather than two block averages — it needs six readings
    spanning ten days.
    """
    wellness = sorted(wellness, key=lambda r: r["day"])
    as_of = as_of or (_as_date(wellness[-1]["day"]) if wellness else date.today())
    cut = as_of - timedelta(days=window_days)
    pts = [
        (_as_date(r["day"]), float(r[field]))
        for r in wellness
        if r.get(field) is not None and cut <= _as_date(r["day"]) <= as_of
    ]
    out: dict[str, Any] = {
        "field": field, "recent": None, "n": len(pts), "span_days": 0,
        "per_week": None, "change": None, "verdict": "insufficient_data",
    }
    if not pts:
        return out

    values = [v for _, v in pts]
    # The headline number is the last week's average, which is what you feel.
    recent_vals = [v for d, v in pts if d > as_of - timedelta(days=7)] or values[-3:]
    out["recent"] = round(fmean(recent_vals), 1)
    span = (pts[-1][0] - pts[0][0]).days
    out["span_days"] = span
    if len(pts) < MIN_TREND_POINTS or span < MIN_TREND_SPAN_DAYS:
        out["needed"] = (
            f"{max(0, MIN_TREND_POINTS - len(pts))} more readings"
            if len(pts) < MIN_TREND_POINTS
            else f"{MIN_TREND_SPAN_DAYS - span} more days"
        )
        return out

    t0 = pts[0][0]
    slope_per_day = ols_slope([(d - t0).days for d, _ in pts], values)
    if slope_per_day is None:
        return out
    per_week = slope_per_day * 7.0
    out["per_week"] = round(per_week, 2)
    # Total movement across the observed span, which reads more concretely than a
    # rate for someone with three weeks of history.
    out["change"] = round(slope_per_day * span, 1)

    mean = fmean(values) or 1.0
    threshold = max(0.15, abs(mean) * 0.004)   # ~0.4% of the metric per week
    if abs(per_week) < threshold:
        out["verdict"] = "steady"
    else:
        improving = per_week < 0 if lower_is_better else per_week > 0
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


# --------------------------------------------------------------------------
# Load, consistency and environment
#
# Three questions the per-session charts above cannot answer: is the training
# ramping too fast, is it happening at all, and is the weather the reason a
# session felt bad. Each is deterministic and each degrades to an empty list
# rather than an exception when the history is too short.
# --------------------------------------------------------------------------

# The band inside which a rising load is usually productive. Below it fitness
# decays, above it injury risk climbs steeply — the same two numbers the deload
# trigger in the planner uses.
ACWR_LOW = 0.8
ACWR_HIGH = 1.3


def session_load(activity: Mapping[str, Any]) -> float:
    """Garmin's own training load, or minutes as a stand-in.

    Load is missing on a new account and on any activity the watch could not
    score, and a chart that silently drops those sessions understates a ramp.
    Minutes are a poor proxy — they ignore intensity entirely — but they are the
    same proxy `acwr_from_activities` already uses, so the two agree.
    """
    return float(activity.get("training_load")
                 or (activity.get("duration_s") or 0) / 60.0)


def load_ramp(
    activities: Sequence[dict[str, Any]],
    as_of: date | None = None,
    days: int = 90,
    min_history_days: int = MIN_ACWR_HISTORY_DAYS,
) -> list[dict[str, Any]]:
    """Daily load with its rolling 7-day acute and 28-day chronic averages.

    The single number on the dashboard (`acwr`) says where the ramp is today.
    This says how it got there, which is the part that tells you whether to back
    off: 1.25 on the way down is a different week from 1.25 on the way up.

    Both sides are expressed as load per day so they sit on one axis — the acute
    line is the 7-day mean, the chronic line the 28-day mean. The ratio is left
    None until there is enough history for the chronic side to mean anything,
    for the reason spelled out on `acwr_from_activities`.
    """
    as_of = as_of or date.today()
    per_day: dict[date, float] = {}
    for a in activities:
        if not a.get("start_date"):
            continue
        day = _as_date(a["start_date"])
        per_day[day] = per_day.get(day, 0.0) + session_load(a)
    if not per_day:
        return []

    # Start where the data starts, never `days` ago: an account three weeks old
    # should not draw two months of empty axis before its first point.
    first = max(min(per_day), as_of - timedelta(days=days - 1))
    history_start = min(per_day)
    out: list[dict[str, Any]] = []
    day = first
    while day <= as_of:
        acute = sum(per_day.get(day - timedelta(days=i), 0.0) for i in range(7)) / 7.0
        chronic = sum(per_day.get(day - timedelta(days=i), 0.0)
                      for i in range(28)) / 28.0
        ratio = None
        if chronic > 0 and (day - history_start).days >= min_history_days:
            ratio = round(acute / chronic, 2)
        out.append({
            "day": day,
            "load": round(per_day.get(day, 0.0), 1),
            "acute": round(acute, 1),
            "chronic": round(chronic, 1),
            "ratio": ratio,
        })
        day += timedelta(days=1)
    return out


def ramp_verdict(ramp: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Plain reading of where the ramp sits and which way it is moving."""
    rated = [r for r in ramp if r.get("ratio") is not None]
    if not rated:
        return {"ratio": None, "verdict": "insufficient_data",
                "note": f"needs {MIN_ACWR_HISTORY_DAYS} days of history"}
    now = rated[-1]["ratio"]
    week_ago = rated[-8]["ratio"] if len(rated) >= 8 else None
    if now > ACWR_HIGH:
        verdict, note = "high", "ramping faster than the body is absorbing"
    elif now < ACWR_LOW:
        verdict, note = "low", "detraining — the chronic base is falling away"
    else:
        verdict, note = "productive", "inside the productive band"
    if week_ago is not None:
        note += f", {now - week_ago:+.2f} on last week"
    return {"ratio": now, "verdict": verdict, "note": note,
            "week_ago": week_ago}


def weekly_zone_minutes_from_streams(
    streams: Mapping[str, Sequence[dict[str, Any]]],
    activities: Sequence[dict[str, Any]],
    bounds: dict[int, tuple[int, int | None]],
    weeks: int = 8,
    as_of: date | None = None,
    sport: str | None = None,
) -> list[dict[str, Any]]:
    """Minutes per zone per week, against the athlete's own boundaries.

    `zone_distribution_from_streams` answers "where did the last 28 days go".
    This answers "is the mix drifting", which is the question that catches the
    slow slide from base training into accidental tempo work — a single 28-day
    number hides it because the drift and the improvement average out.
    """
    as_of = as_of or date.today()
    this_week = as_of - timedelta(days=as_of.weekday())
    first_week = this_week - timedelta(weeks=weeks - 1)
    per_activity = {str(a.get("activity_id")): a for a in activities}
    buckets: dict[date, dict[int, float]] = {}

    for activity_id, samples in (streams or {}).items():
        activity = per_activity.get(str(activity_id))
        if activity is None or not activity.get("start_date"):
            continue
        if sport and (activity.get("sport") or "") != sport:
            continue
        day = _as_date(activity["start_date"])
        week = day - timedelta(days=day.weekday())
        if week < first_week or week > this_week:
            continue
        readings = [r["hr"] for r in samples or () if r.get("hr") is not None]
        if not readings:
            continue
        # Duration-scaled, for the same reason as the 28-day version: one sample
        # stands for a different slice of time in a 20-minute run than in a ride.
        minutes_each = ((activity.get("duration_s") or 0) / 60.0) / len(readings)
        bucket = buckets.setdefault(week, {})
        for hr in readings:
            zone = zone_of(float(hr), bounds)
            if zone is not None:
                bucket[zone] = bucket.get(zone, 0.0) + minutes_each

    out = []
    for week in sorted(buckets):
        row = {"week_start": week}
        total = 0.0
        for zone in range(1, 6):
            minutes = round(buckets[week].get(zone, 0.0), 1)
            row[f"z{zone}"] = minutes
            total += minutes
        row["total"] = round(total, 1)
        row["easy_pct"] = (round((row["z1"] + row["z2"]) / total * 100, 1)
                           if total > 0 else None)
        out.append(row)
    return out


def weekly_zone_minutes_from_rows(
    rows: Sequence[Mapping[str, Any]],
    weeks: int = 8,
    as_of: date | None = None,
) -> list[dict[str, Any]]:
    """The same weekly table, built from day-and-zone rows already bucketed.

    `weekly_zone_minutes_from_streams` needs every sample in the process to
    produce this. The database can do the bucketing and hand back a few hundred
    rows instead of a few thousand samples, and this turns those rows into the
    same shape — so the chart is unchanged and the memory is not spent. The
    stream version stays as the reference the tests compare against.
    """
    as_of = as_of or date.today()
    this_week = as_of - timedelta(days=as_of.weekday())
    first_week = this_week - timedelta(weeks=weeks - 1)
    buckets: dict[date, dict[int, float]] = {}
    for row in rows or ():
        day = _as_date(row.get("day"))
        if day is None:
            continue
        week = day - timedelta(days=day.weekday())
        if week < first_week or week > this_week:
            continue
        zone = row.get("zone")
        if zone is None:
            continue
        bucket = buckets.setdefault(week, {})
        bucket[int(zone)] = bucket.get(int(zone), 0.0) + float(row.get("minutes") or 0)

    out = []
    for week in sorted(buckets):
        entry: dict[str, Any] = {"week_start": week}
        total = 0.0
        for zone in range(1, 6):
            minutes = round(buckets[week].get(zone, 0.0), 1)
            entry[f"z{zone}"] = minutes
            total += minutes
        entry["total"] = round(total, 1)
        entry["easy_pct"] = (round((entry["z1"] + entry["z2"]) / total * 100, 1)
                             if total > 0 else None)
        out.append(entry)
    return out


def consistency(
    activities: Sequence[dict[str, Any]],
    as_of: date | None = None,
    weeks: int = 16,
    strength_rows: Sequence[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """One row per day, oldest first, for a calendar view of showing up.

    Endurance is mostly an attendance problem, and attendance is the one thing a
    line chart of anything else hides: a week of three good sessions and a week
    of one long one can trend identically.
    """
    as_of = as_of or date.today()
    this_week = as_of - timedelta(days=as_of.weekday())
    start = this_week - timedelta(weeks=weeks - 1)
    per_day: dict[date, dict[str, Any]] = {}

    for a in activities:
        if not a.get("start_date"):
            continue
        day = _as_date(a["start_date"])
        if day < start or day > as_of:
            continue
        cell = per_day.setdefault(day, {"minutes": 0.0, "sports": [],
                                        "load": 0.0, "sessions": 0})
        cell["minutes"] += (a.get("duration_s") or 0) / 60.0
        cell["load"] += session_load(a)
        cell["sessions"] += 1
        sport = a.get("sport") or "other"
        if sport not in cell["sports"]:
            cell["sports"].append(sport)

    for row in strength_rows or []:
        if not row.get("day"):
            continue
        day = _as_date(row["day"])
        if day < start or day > as_of:
            continue
        cell = per_day.setdefault(day, {"minutes": 0.0, "sports": [],
                                        "load": 0.0, "sessions": 0})
        if "strength" not in cell["sports"]:
            cell["sports"].append("strength")

    out = []
    day = start
    while day <= as_of:
        cell = per_day.get(day)
        out.append({
            "day": day,
            "minutes": round(cell["minutes"], 1) if cell else 0.0,
            "load": round(cell["load"], 1) if cell else 0.0,
            "sessions": cell["sessions"] if cell else 0,
            "sports": cell["sports"] if cell else [],
        })
        day += timedelta(days=1)
    return out


def streak(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    """Current and longest run of days with something logged, plus rest days.

    Takes `consistency()` output. Counts back from the most recent day, and
    treats today as not yet broken: a streak should not read as zero at 9am.
    """
    days = list(rows)
    current = 0
    for row in reversed(days[:-1] if days else []):
        if row["sessions"] or row["sports"]:
            current += 1
        else:
            break
    if days and (days[-1]["sessions"] or days[-1]["sports"]):
        current += 1

    longest = run = 0
    for row in days:
        run = run + 1 if (row["sessions"] or row["sports"]) else 0
        longest = max(longest, run)
    active = sum(1 for r in days if r["sessions"] or r["sports"])
    return {"current": current, "longest": longest, "active_days": active,
            "days": len(days)}


# Above this, sweat stops evaporating efficiently and heart rate climbs for the
# same effort. Standard humidity-adjustment territory, not a personal number.
DEW_POINT_EASY_C = 15.0
DEW_POINT_HARD_C = 20.0


def weather_effect(
    activities: Sequence[dict[str, Any]],
    weather: Mapping[str, Mapping[str, Any]],
    sport: str = "run",
) -> dict[str, Any]:
    """Heart rate at a reference pace against dew point, session by session.

    The efficiency charts treat every session as comparable. They are not: at a
    21°C dew point the same pace costs several beats more, and reading that as
    lost fitness is the single easiest way to draw a wrong conclusion from this
    dashboard. This puts the two on one axis so the athlete can see which it is.

    Dew point rather than temperature or humidity, because it is the one number
    that says how much evaporative cooling is actually available.
    """
    points = []
    for p in hr_points(activities, sport):
        row = (weather or {}).get(str(p["activity_id"])) or {}
        dew = _pos(row.get("dew_point_c"))
        if dew is None or p.get("hr_at_reference") is None:
            continue
        points.append({
            "date": p["date"],
            "dew_point_c": round(float(dew), 1),
            "hr_at_reference": p["hr_at_reference"],
            "avg_hr": p["avg_hr"],
            "temp_c": _pos(row.get("temp_c")),
            "condition": row.get("condition") or "",
            "is_steady": p["is_steady"],
        })
    slope = None
    if len(points) >= MIN_TREND_POINTS:
        slope = ols_slope([p["dew_point_c"] for p in points],
                          [p["hr_at_reference"] for p in points])
    hot = [p for p in points if p["dew_point_c"] >= DEW_POINT_HARD_C]
    return {
        "points": points,
        "bpm_per_deg": _r(slope) if slope is not None else None,
        "hot_sessions": len(hot),
        "hot_share": (round(len(hot) / len(points) * 100) if points else None),
    }
