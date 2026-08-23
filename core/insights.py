"""Plain-English summaries of what each dashboard page is saying.

The point is that nobody should have to read four charts to learn one thing. Each
page gets a headline and a few bullets: what the data says, and what it implies
for training.

Two layers, same as the planner:

  * **Deterministic** (`*_insight` functions) — derived from the numbers, always
    available, no API key, no cost. This is the layer that must be good.
  * **AI narration** (`narrate`) — optional prose over the same facts. It is only
    allowed to rephrase what the deterministic layer already established; it is
    never the source of a number or a recommendation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from core.analysis import (
    all_ef_trends,
    baseline_trend,
    ef_data_status,
    polarisation,
    recovery_signals,
    totals,
    week_summaries,
    zone_distribution,
)
from core.schemas import ENDURANCE_SPORTS

log = logging.getLogger("iron_coach.insights")


@dataclass
class PageInsight:
    headline: str
    bullets: list[str] = field(default_factory=list)
    tone: str = "info"  # info | success | warning | error

    def as_facts(self) -> dict[str, Any]:
        return {"headline": self.headline, "points": self.bullets}


def _fmt_hm(minutes: float) -> str:
    h, m = divmod(int(round(minutes or 0)), 60)
    return f"{h}h {m:02d}m" if h else f"{m}m"


# --------------------------------------------------------------------------
# per page
# --------------------------------------------------------------------------


def overview_insight(data: dict, today: date) -> PageInsight:
    acts, wl = data["activities"], data["wellness"]
    tot = totals(acts)
    if not acts:
        return PageInsight("Nothing logged yet.",
                           ["Sync from Garmin to get started."], "info")

    weeks = week_summaries(acts, weeks=4, as_of=today, strength_rows=data["strength"])
    wk = weeks[-1]
    bullets = [
        f"{tot['sessions']} sessions and {_fmt_hm(tot['minutes'])} logged across "
        f"{tot['weeks']} week{'s' if tot['weeks'] != 1 else ''}, "
        f"{tot['km']:.0f} km in total.",
        f"This week so far: {_fmt_hm(wk.total_minutes)} over "
        f"{sum(s.sessions for s in wk.by_sport.values())} sessions.",
    ]
    tone = "info"

    rhr = baseline_trend(wl, "resting_hr", as_of=today, lower_is_better=True)
    if rhr["verdict"] == "improving":
        bullets.append(
            f"Resting heart rate is trending down {abs(rhr['per_week']):.2f} bpm a "
            f"week — the aerobic engine is growing."
        )
        tone = "success"
    elif rhr["verdict"] == "worsening":
        bullets.append(
            f"Resting heart rate is trending up {rhr['per_week']:.2f} bpm a week. "
            f"That usually means accumulated fatigue, illness or poor sleep."
        )
        tone = "warning"
    elif rhr["verdict"] == "insufficient_data":
        bullets.append(
            f"Resting heart rate needs {rhr.get('needed', 'more data')} before a "
            f"trend can be called."
        )

    if data["zones"]:
        pol = polarisation(data["zones"], since=today - timedelta(days=28))
        if pol["hard"] >= 35:
            bullets.append(
                f"Intensity is the headline problem: {pol['hard']:.0f}% of the last "
                f"28 days was in Z4-Z5 and only {pol['easy']:.0f}% was easy. Base "
                f"phase wants that the other way round."
            )
            tone = "error"

    missing = [s for s in ENDURANCE_SPORTS
               if ef_data_status(acts, s)["needed_for_verdict"] > 0]
    if missing:
        bullets.append(
            "Efficiency trends are not measurable yet for "
            + ", ".join(missing)
            + " — they need three steady aerobic sessions each."
        )

    if not data["strength"]:
        bullets.append("No leg strength logged. It is the biggest single protection "
                       "for run volume.")

    headline = {
        "success": "Training is going in the right direction.",
        "warning": "Something in the recovery signals wants attention.",
        "error": "The training is working, but the intensity mix is wrong.",
        "info": "Early days — here is where you stand.",
    }[tone]
    return PageInsight(headline, bullets, tone)


def fitness_insight(data: dict, today: date) -> PageInsight:
    acts = data["activities"]
    trends = all_ef_trends(acts, as_of=today)
    usable = [t for t in trends if t.n_sessions >= 3]
    bullets, tone = [], "info"

    if not usable:
        for sport in ENDURANCE_SPORTS:
            st = ef_data_status(acts, sport)
            if st["total"]:
                bullets.append(f"**{sport.title()}:** {st['message']}")
                if st["rejected_reasons"]:
                    bullets.append(
                        "  Excluded: "
                        + "; ".join(f"{n}× {why}" for why, n in st["rejected_reasons"].items())
                    )
        headline = "Not enough steady sessions to measure efficiency yet."
        if any("Z4-Z5" in b for b in bullets):
            headline = ("Efficiency cannot be measured yet — the sessions are too "
                        "hard, not too few.")
            tone = "warning"
        return PageInsight(headline, bullets or ["No sessions with heart rate yet."], tone)

    for t in usable:
        direction = {"improving": "improving", "declining": "declining",
                     "flat": "flat", "insufficient_data": "unclear"}[t.verdict]
        line = f"**{t.sport.title()}** is {direction}"
        if t.change_pct is not None:
            line += f" ({t.change_pct:+.1f}% against the earlier baseline)"
        line += f", from {t.n_sessions} steady sessions."
        bullets.append(line)
    improving = [t for t in usable if t.verdict == "improving"]
    declining = [t for t in usable if t.verdict == "declining"]
    if improving and not declining:
        headline = ("You are producing more speed or power per heartbeat than before.")
        tone = "success"
    elif declining:
        headline = ("Efficiency is slipping in "
                    + ", ".join(t.sport for t in declining)
                    + " — usually fatigue, heat, or too much intensity.")
        tone = "warning"
    else:
        headline = "Efficiency is holding steady."
    drift = [a["decoupling_pct"] for a in acts if a.get("decoupling_pct") is not None]
    if drift:
        worst = max(drift)
        bullets.append(
            f"Aerobic drift on long sessions peaks at {worst:.1f}%"
            + (" — under 5% is good durability." if worst < 5
               else ", above the 5% mark, so durability is the limiter.")
        )
    return PageInsight(headline, bullets, tone)


def intensity_insight(data: dict, today: date) -> PageInsight:
    zones = data["zones"]
    if not zones:
        return PageInsight("No zone data yet.", ["Sync from Garmin to see this."], "info")
    since = today - timedelta(days=28)
    pol = polarisation(zones, since=since)
    dist = zone_distribution(zones, since=since)
    total = sum(dist.values())
    bullets = [
        f"Over the last 28 days: {_fmt_hm(total)} recorded, "
        f"{pol['easy']:.0f}% easy, {pol['moderate']:.0f}% moderate, "
        f"{pol['hard']:.0f}% hard."
    ]
    per_sport = []
    for sport in ENDURANCE_SPORTS:
        sp = polarisation(zones, sport=sport, since=since)
        if sum(sp.values()) > 0:
            per_sport.append(f"{sport} {sp['easy']:.0f}% easy / {sp['hard']:.0f}% hard")
    if per_sport:
        bullets.append("By sport: " + " · ".join(per_sport) + ".")

    if pol["easy"] >= 70:
        return PageInsight("Your intensity distribution is right for base phase.",
                           bullets, "success")
    if pol["hard"] >= 35:
        bullets.append(
            "Two consequences. Hard sessions cost recovery days without adding much "
            "aerobic base, which shows up as low readiness. And they disqualify "
            "sessions from the efficiency trend, so you lose the ability to measure "
            "whether you are improving."
        )
        bullets.append(
            "The fix is unglamorous: most sessions slow enough to hold a "
            "conversation, keeping the hard work to one session a week."
        )
        return PageInsight(
            f"Too much hard work — {pol['hard']:.0f}% in Z4-Z5 against a base-phase "
            f"target of under 15%.", bullets, "error")
    bullets.append("Push the easy share above 70% and keep hard work to one session "
                   "a week.")
    return PageInsight("The mix is drifting harder than base phase wants.",
                       bullets, "warning")


def recovery_insight(data: dict, today: date) -> PageInsight:
    wl = data["wellness"]
    if not wl:
        return PageInsight("No wellness data yet.", ["Sync from Garmin."], "info")
    sig = recovery_signals(wl, data["activities"], as_of=today)
    bullets, tone = [], "info"

    if sig.training_readiness is not None:
        state = ("low — the rules will force a deload" if sig.training_readiness < 35
                 else "moderate" if sig.training_readiness < 65 else "good")
        bullets.append(f"Training readiness is {sig.training_readiness:.0f} ({state}).")
        if sig.training_readiness < 35:
            tone = "error"
    if sig.rhr_delta is not None:
        bullets.append(
            f"Resting heart rate is {abs(sig.rhr_delta):.1f} bpm "
            f"{'above' if sig.rhr_delta > 0 else 'below'} its 28-day baseline"
            + (" — over 5 bpm above forces a deload." if sig.rhr_delta > 3 else ".")
        )
    if sig.hrv_delta_pct is not None:
        bullets.append(
            f"HRV is {abs(sig.hrv_delta_pct):.0f}% "
            f"{'above' if sig.hrv_delta_pct > 0 else 'below'} baseline."
        )
    if sig.hrv_status is None:
        bullets.append(
            "Garmin has not established an HRV status yet — it needs about three "
            "weeks of consecutive nights wearing the watch."
        )
    if sig.acwr is None:
        bullets.append(
            "Acute-to-chronic load needs roughly three weeks of history before it "
            "means anything, so it is deliberately blank rather than misleading."
        )
    elif sig.acwr > 1.3:
        bullets.append(f"Load ratio {sig.acwr:.2f} — training has ramped faster than "
                       f"your base supports.")
        tone = "error"
    if sig.vo2max_run:
        bullets.append(f"Garmin puts your running VO2max at {sig.vo2max_run:.1f}.")

    headline = {"error": "Recovery is telling you to back off.",
                "warning": "Recovery is borderline.",
                "info": "Recovery signals look workable."}[tone]
    return PageInsight(headline, bullets, tone)


def volume_insight(data: dict, today: date) -> PageInsight:
    weeks = week_summaries(data["activities"], weeks=8, as_of=today,
                           strength_rows=data["strength"])
    loaded = [w for w in weeks if w.total_minutes > 0]
    if not loaded:
        return PageInsight("No volume logged yet.", [], "info")
    latest = loaded[-1]
    bullets = [f"Latest full week: {_fmt_hm(latest.total_minutes)} over "
               f"{sum(s.sessions for s in latest.by_sport.values())} sessions."]
    if len(loaded) >= 2:
        prev = loaded[-2]
        if prev.total_minutes > 0:
            change = (latest.total_minutes - prev.total_minutes) / prev.total_minutes * 100
            bullets.append(
                f"That is {change:+.0f}% on the week before"
                + (" — inside the 10% progression cap." if abs(change) <= 12
                   else ", a bigger jump than the 10% cap allows for planned weeks.")
            )
    by = ", ".join(f"{sp} {_fmt_hm(sw.minutes)}" for sp, sw in
                   sorted(latest.by_sport.items(), key=lambda kv: -kv[1].minutes))
    bullets.append(f"Split: {by}.")
    if latest.rest_days == 0:
        return PageInsight("No rest day last week.",
                           bullets + ["Base phase wants at least one."], "warning")
    return PageInsight("Volume is progressing at a sane rate.", bullets, "info")


def strength_insight(data: dict, today: date) -> PageInsight:
    log_rows = data["strength"]
    if not log_rows:
        return PageInsight(
            "No leg strength logged yet.",
            ["Two sessions a week is the single biggest protection for run volume.",
             "Log it on the watch in strength mode and it imports here automatically."],
            "warning")
    days = sorted({r["day"] for r in log_rows})
    bullets = [f"{len(days)} session(s) logged, most recently {days[-1]}."]
    imported = {r["day"] for r in log_rows if (r.get("notes") or "").startswith("imported")}
    if imported:
        bullets.append(f"{len(imported)} of those came in automatically from the watch.")
    painful = sorted({r["exercise_id"] for r in log_rows if r.get("pain")})
    if painful:
        bullets.append("Pain flagged on: " + ", ".join(painful)
                       + ". Load backs off automatically on those.")
        return PageInsight("Something is hurting — progression has been held back.",
                           bullets, "warning")
    recent = date.fromisoformat(days[-1])
    if (today - recent).days > 10:
        return PageInsight("Strength work has lapsed.",
                           bullets + [f"Last session was {(today - recent).days} days "
                                      f"ago; tendon adaptation fades."], "warning")
    return PageInsight("Strength work is on track.", bullets, "success")


def activities_insight(data: dict, today: date) -> PageInsight:
    acts = data["activities"]
    if not acts:
        return PageInsight("No activities yet.", [], "info")
    tot = totals(acts)
    bullets = [f"{tot['sessions']} sessions stored, {tot['km']:.0f} km, "
               f"{_fmt_hm(tot['minutes'])}."]
    steady = sum(1 for a in acts if a.get("is_steady"))
    bullets.append(f"{steady} of {len(acts)} count as steady aerobic work — those are "
                   f"the ones that feed the efficiency trend.")
    reasons: dict[str, int] = {}
    for a in acts:
        if not a.get("is_steady") and a.get("steady_reason"):
            reasons[a["steady_reason"]] = reasons.get(a["steady_reason"], 0) + 1
    if reasons:
        top = sorted(reasons.items(), key=lambda kv: -kv[1])[:3]
        bullets.append("Most common reason for exclusion: "
                       + "; ".join(f"{why} ({n})" for why, n in top) + ".")
    parents = [a for a in data["all_activities"] if a.get("is_multisport_parent")]
    if parents:
        bullets.append(
            f"{len(parents)} multisport session(s) were split into their individual "
            f"legs, so each sport gets credited properly."
        )
    return PageInsight("Every session, and whether it counts towards the trend.",
                       bullets, "info")


PAGES = {
    "Overview": overview_insight,
    "Activities": activities_insight,
    "Fitness": fitness_insight,
    "Intensity": intensity_insight,
    "Recovery": recovery_insight,
    "Volume": volume_insight,
    "Strength": strength_insight,
}


def for_page(page: str, data: dict, today: date) -> PageInsight | None:
    fn = PAGES.get(page)
    if fn is None:
        return None
    try:
        return fn(data, today)
    except Exception as exc:  # noqa: BLE001 - a summary must never break a page
        log.warning("Insight for %s failed: %s", page, exc)
        return None


# --------------------------------------------------------------------------
# optional AI narration over the deterministic facts
# --------------------------------------------------------------------------

NARRATE_SYSTEM = """\
You explain what a training-dashboard page is telling the athlete looking at it.

You are given facts already computed from their data. Rules:
- Use ONLY those facts. Never add a number, a cause, or a recommendation that is
  not in them. If the facts say data is missing, say what is missing.
- Two or three sentences. Second person, plain language, no bullets, no headings,
  and do not restate the headline you were given.
- Lead with the thing that would change a training decision this week. Where one
  fact explains another, join them: a flat efficiency trend caused by sessions
  being too hard is one sentence, not two.
- End with the single most useful next action, if the facts support one. If they
  do not, stop rather than inventing advice.
- No cheerleading, no hedging, no medical advice. If the news is bad, say it.
Return the paragraph as plain text with no preamble."""

CHART_NOTE_SYSTEM = """\
You write ONE sentence, at most 20 words, saying what a single chart shows.

Rules:
- State the pattern and what it means, not the axes. "Heart rate at the same pace
  is flat across three runs" — never "this chart shows heart rate over time".
- Use ONLY the numbers given. If there are too few points to see a pattern, say
  exactly that instead of guessing at a trend.
- No preamble, no markdown, no quotes. One sentence of plain text."""


def chart_note(title: str, data: Any, backend: Any = None) -> str | None:
    """A one-line reading of a single chart. None when no AI is configured."""
    from core import ai

    try:
        backend = backend or ai.get_backend()
    except ai.AIUnavailable:
        return None
    try:
        text = backend.complete(
            CHART_NOTE_SYSTEM,
            json.dumps({"chart": title, "data": data}, indent=1, default=str)[:2500],
        )
    except Exception as exc:  # noqa: BLE001 - a caption must never break a page
        log.info("Chart note unavailable: %s", exc)
        return None
    return (text or "").strip().strip('"').strip()[:180] or None


def narrate(page: str, insight: PageInsight, backend: Any = None) -> str | None:
    """Prose version of an insight. Returns None when no AI is configured."""
    from core import ai

    try:
        backend = backend or ai.get_backend()
    except ai.AIUnavailable:
        return None
    payload = {"page": page, **insight.as_facts()}
    try:
        text = backend.complete(NARRATE_SYSTEM, json.dumps(payload, indent=2))
    except Exception as exc:  # noqa: BLE001
        log.info("AI narration unavailable: %s", exc)
        return None
    text = (text or "").strip()
    return text[:900] or None
