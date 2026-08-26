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

log = logging.getLogger("aerobic_engine.insights")


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


def _sports(data: dict) -> tuple[str, ...]:
    """Endurance sports in scope for this page.

    The dashboard's sport filter puts its selection in `scoped_to`. Reading it
    here matters because the activity lists are already filtered, so a loop over
    every sport would report "no steady swims yet" on a page the athlete has
    explicitly narrowed to running and riding.
    """
    picked = {str(sp).lower() for sp in (data.get("scoped_to") or ENDURANCE_SPORTS)}
    return tuple(sp for sp in ENDURANCE_SPORTS if sp in picked)


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

    thin = [s for s in _sports(data)
            if ef_data_status(acts, s)["needed_for_verdict"] > 0]
    if thin:
        bullets.append(
            "Not enough sessions yet to trend efficiency in "
            + ", ".join(thin)
            + " — three each is where a direction starts to mean something."
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
        # Every session counts towards this, so a thin verdict here means few
        # sessions and nothing else. It used to list what had been "excluded",
        # which told an athlete with four hard runs that they had no data — when
        # what they had was four runs.
        for sport in _sports(data):
            st = ef_data_status(acts, sport)
            if st["sessions"]:
                bullets.append(f"**{sport.title()}:** {st['message']}")
        headline = "Not enough sessions yet to tell if you are getting fitter."
        return PageInsight(headline, bullets or ["No sessions with heart rate yet."],
                           tone)

    for t in usable:
        direction = {"improving": "improving", "declining": "declining",
                     "flat": "flat", "insufficient_data": "unclear"}[t.verdict]
        line = f"**{t.sport.title()}** is {direction}"
        if t.change_pct is not None:
            line += f" ({t.change_pct:+.1f}% against the earlier baseline)"
        line += f", from {t.n_sessions} session{'s' if t.n_sessions != 1 else ''}."
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
        headline = "You are holding steady — same pace, same heart rate."
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
    for sport in _sports(data):
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
            "Comparing your last week with your last month needs about three "
            "weeks of training on record before it "
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
    bullets.append(f"{steady} of {len(acts)} were steady and easy enough to count — "
                   f"those are "
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


def lifetime_insight(data: dict, today: date) -> PageInsight:
    """The long view: how far they have come, not whether this week is on track.

    Deliberately a different question from every other page here. Weekly framing
    is what makes training feel like treading water; the totals are what show it
    is not.
    """
    from core.analysis import totals

    acts = data["activities"]
    if not acts:
        return PageInsight("Nothing recorded yet.", [], "info")

    tot = totals(acts)
    bullets = [
        f"{tot['sessions']} sessions, {tot['km']:,.0f} km and "
        f"{tot['minutes'] / 60:,.0f} hours on record across "
        f"{tot['weeks']:.0f} week(s) — about "
        f"{tot['sessions'] / max(tot['weeks'], 1):.1f} sessions a week."
    ]
    per = tot["by_sport"]
    ranked = sorted(((sp, r) for sp, r in per.items() if r.get("minutes")),
                    key=lambda kv: -kv[1]["minutes"])
    if ranked:
        bullets.append("Time is split " + ", ".join(
            f"{sp} {r['minutes'] / 60:.0f}h" for sp, r in ranked[:4]) + ".")
    biggest = ranked[0][0] if ranked else None
    if biggest and len(ranked) > 1:
        share = ranked[0][1]["minutes"] / max(tot["minutes"], 1) * 100
        if share > 60:
            bullets.append(
                f"{share:.0f}% of all your training time is {biggest}. For a "
                f"triathlon that is worth knowing: the other two disciplines "
                f"race on the same day."
            )

    records = data.get("records") or []
    if records:
        bullets.append(f"{len(records)} personal records on file from Garmin.")

    weeks = tot["weeks"] or 0
    tone = "info" if weeks < 8 else "success"
    headline = ("Early days — this is the baseline everything later is measured "
                "against." if weeks < 8 else "The long view.")
    return PageInsight(headline, bullets, tone)


def plan_insight(data: dict, today: date) -> PageInsight:
    """What the week ahead asks of you, and why it looks like that.

    Separate from the overview because the question is different: not "how am I
    doing" but "why this week, and what would change it".
    """
    from core.analysis import recovery_signals

    plan = ((data.get("plan") or {}).get("plan")) or {}
    days = [d for d in plan.get("week_plan") or []
            if (d.get("duration_min") or 0) > 0 and d.get("purpose") != "completed"]
    if not days:
        return PageInsight("No plan for this week yet.",
                           ["Build one below, or check in and let it plan around "
                            "how you feel."], "info")

    minutes = sum(d["duration_min"] for d in days)
    by_sport: dict[str, int] = {}
    for d in days:
        by_sport[d["sport"]] = by_sport.get(d["sport"], 0) + 1
    bullets = [
        f"{len(days)} sessions left, {minutes // 60}h {minutes % 60:02d}m: "
        + ", ".join(f"{n}x {sp}" for sp, n in sorted(by_sport.items()))
    ]

    hr = {d.get("target_hr") for d in days if d.get("target_hr")}
    if hr:
        bullets.append("Endurance targets: " + ", ".join(sorted(hr)) + ".")

    flags = plan.get("flags") or []
    for flag in flags[:3]:
        bullets.append(str(flag))

    sig = recovery_signals(data.get("wellness") or [],
                           data.get("activities") or [], as_of=today)
    deload = any("deload" in str(f).lower() for f in flags)
    if deload:
        headline, tone = "This week is deliberately smaller.", "warning"
    elif sig and sig.training_readiness and sig.training_readiness < 50:
        headline, tone = "Planned conservatively — recovery is not strong.", "warning"
    else:
        headline, tone = "The week ahead.", "info"
    if plan.get("source") == "ai_repaired":
        bullets.append("The AI's version broke a limit, so it was repaired in "
                       "code before you saw it.")
    return PageInsight(headline, bullets, tone)


PAGES = {
    "Overview": overview_insight,
    "Plan": plan_insight,
    "Lifetime": lifetime_insight,
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
- Two or three sentences. Second person, no bullets, no headings, and do not
  restate the headline you were given.
- Write the way you would explain it out loud to a friend who runs but has never
  used a training app. Short sentences. Everyday words.
- No jargon. Not "aerobic decoupling", "acute:chronic ratio", "polarised",
  "zone 2 adherence", "envelope", "load management". If a training term is the
  only accurate word, put what it means straight after it in three or four words.
- No stock AI phrasing: no "it's worth noting", "delve", "leverage", "robust",
  "in terms of", "that said", "overall", em-dash asides, or a summary sentence
  that repeats what you just said.
- Say the number, then what to do about it.
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
- Plain words a beginner would understand. No jargon, no stock AI phrasing
  ("it's worth noting", "delve", "leverage", "overall"), no hedging.
- No preamble, no markdown, no quotes. One sentence of plain text."""


CHART_DETAIL_SYSTEM = """\
You write a short paragraph — three or four sentences, at most 75 words — reading
one chart for the athlete whose data it is.

This is the chart their training is judged on, so it earns more than a caption.
Cover, in this order, and only where the numbers support it:
- which way the line is going, and by how much
- whether there is enough data to trust that yet
- anything in the numbers that explains it: the conditions a session was run in,
  a session far from the others, a gap in the record
- what it means for the next week, in one clause

Rules:
- Use ONLY the numbers given. Never invent a cause. If the series is too short to
  read, say so plainly and stop — a confident trend from four points is worse
  than no comment.
- Where air temperature or dew point is given, use it: a hot, humid session costs
  several beats at the same pace, and reading that as lost fitness is the easiest
  wrong conclusion available here.
- Plain words a beginner would understand. No jargon, no stock AI phrasing
  ("it's worth noting", "delve", "leverage", "overall"), no hedging, no headings.
- No preamble, no markdown, no quotes. Plain sentences."""


def chart_note(title: str, data: Any, backend: Any = None,
               detail: bool = False) -> str | None:
    """A reading of a single chart. None when no AI is configured.

    `detail` asks for a paragraph rather than a sentence. Reserved for the chart
    a page exists for — a bug report put it well: the headline chart is the one
    doing the analysis, so a twelve-word caption under it is the wrong ration.
    """
    from core import ai

    try:
        backend = backend or ai.get_backend()
    except ai.AIUnavailable:
        return None
    # AIUnavailable propagates on purpose: the sync-time generator pauses and
    # retries on a rate limit, which it cannot do if this swallows it.
    system = CHART_DETAIL_SYSTEM if detail else CHART_NOTE_SYSTEM
    text = backend.complete(system, _chart_prompt(title, data))
    return (text or "").strip().strip('"').strip()[:700 if detail else 180] or None


# Chart payloads are trimmed rather than sent whole. The cap is generous for a
# one-line caption and it bounds a series that would otherwise grow for years.
CHART_PROMPT_CHARS = 2500


def _chart_prompt(title: str, data: Any) -> str:
    """Serialise a chart's numbers compactly, and trim honestly if still too big.

    Two things this gets right that the obvious version did not. Indented JSON
    spends tokens on whitespace that carries no meaning to a model, so the
    separators are compact. And slicing the string at a character limit used to
    cut mid-structure, handing the model malformed JSON with no indication that
    anything was missing — a caption confidently describing a truncated series is
    worse than no caption. A list is now shortened by dropping whole entries,
    oldest first, and says how many it dropped.
    """
    def dump(payload: Any) -> str:
        return json.dumps(payload, separators=(",", ":"), default=str)

    body = dump({"chart": title, "data": data})
    if len(body) <= CHART_PROMPT_CHARS:
        return body

    if isinstance(data, list) and data:
        kept = list(data)
        while kept and len(dump({"chart": title, "data": kept})) > CHART_PROMPT_CHARS:
            kept = kept[1:]
        if kept:
            return dump({"chart": title, "data": kept,
                         "note": f"showing the most recent {len(kept)} of "
                                 f"{len(data)} points"})
    if isinstance(data, dict) and data:
        # Per-sport series: drop the longest until it fits, naming what went.
        kept = dict(data)
        dropped: list[str] = []
        while (len(kept) > 1
               and len(dump({"chart": title, "data": kept})) > CHART_PROMPT_CHARS):
            widest = max(kept, key=lambda k: len(dump(kept[k])))
            dropped.append(str(widest))
            kept.pop(widest)
        if len(dump({"chart": title, "data": kept})) <= CHART_PROMPT_CHARS:
            payload = {"chart": title, "data": kept}
            if dropped:
                payload["note"] = f"omitted for size: {', '.join(dropped)}"
            return dump(payload)

    # Nothing structural to drop. Say the numbers were cut rather than implying
    # they are complete.
    return (body[:CHART_PROMPT_CHARS]
            + '..."TRUNCATED":"the series was cut for length"}')


def narrate(page: str, insight: PageInsight, backend: Any = None) -> str | None:
    """Prose version of an insight. Returns None when no AI is configured."""
    from core import ai

    try:
        backend = backend or ai.get_backend()
    except ai.AIUnavailable:
        return None
    payload = {"page": page, **insight.as_facts()}
    # AIUnavailable propagates on purpose: the sync-time generator pauses for
    # the interval the provider asked for, which it cannot do if this swallows it.
    text = backend.complete(NARRATE_SYSTEM,
                            json.dumps(payload, indent=2, default=str))
    text = (text or "").strip()
    return text[:900] or None


# --------------------------------------------------------------------------
# Ask the coach
#
# The pages answer the questions I thought of. This answers the one being asked
# right now — "why was Wednesday so hard", "can I move the long ride to Friday" —
# from the same deterministic facts the charts are drawn from, and nothing else.
# --------------------------------------------------------------------------

ASK_SYSTEM = """\
You answer one question from an endurance athlete about their own training data.

You are given the facts: recovery numbers, this week's plan, sessions already
done, recent sessions with heart rate and pace, efficiency trends, how much of
the training was easy or hard, and the limits the plan has to stay inside. Rules:
- Use ONLY those facts. Never invent a number, a session, or a date. If the
  facts do not contain the answer, say which data is missing and stop.
- Never override the rules. If they ask for something the limits forbid — more
  than the weekly cap, training hard through an easy week — say what the rule is
  and what you can offer instead.
- Two to five sentences, second person, no bullets, no headings. Quote the
  numbers that back up what you say.
- Write the way you would explain it out loud to a friend who runs but has never
  used a training app. Short sentences. Everyday words.
- No jargon. Not "aerobic decoupling", "acute:chronic ratio", "polarised",
  "zone 2 adherence", "envelope", "load management". If a training term is the
  only accurate word, put what it means straight after it in three or four words.
- No stock AI phrasing: no "it's worth noting", "delve", "leverage", "robust",
  "in terms of", "that said", "overall", em-dash asides, or a summary sentence
  that repeats what you just said.
- Say the number, then what to do about it.
- No medical advice. Persistent pain gets one line pointing at a physio.
- No cheerleading. If the answer is "you are training too hard", say it.
Return plain text with no preamble."""

# Bounds the question, not the answer: a long paste of someone else's plan is
# not a question about this data, and the facts bundle is the expensive half.
ASK_QUESTION_CHARS = 600


def ask_facts(data: dict, today: date) -> dict[str, Any]:
    """Everything the coach answer is allowed to draw on, and nothing else.

    Deliberately assembled from the same `*_insight` functions the banners use,
    so a spoken answer and the page it sits on cannot disagree. Raw session rows
    are trimmed to the fields that carry meaning — a full activity row is mostly
    identifiers and Garmin bookkeeping.
    """
    acts = data.get("activities") or []
    sig = recovery_signals(data.get("wellness") or [], acts, as_of=today) \
        if data.get("wellness") else None
    weeks = week_summaries(acts, weeks=3, as_of=today,
                           strength_rows=data.get("strength") or [])
    plan = (data.get("plan") or {}).get("plan") or {}

    recent = []
    for a in sorted(acts, key=lambda r: str(r.get("start_date") or ""))[-10:]:
        recent.append({
            "date": str(a.get("start_date")),
            "sport": a.get("sport"),
            "min": round((a.get("duration_s") or 0) / 60.0),
            "km": round((a.get("distance_m") or 0) / 1000.0, 1) or None,
            "avg_hr": a.get("avg_hr"),
            "max_hr": a.get("max_hr"),
            "steady": bool(a.get("is_steady")),
            "why_not_steady": (a.get("steady_reason") or "")[:60] or None,
            "load": a.get("training_load"),
        })

    facts: dict[str, Any] = {
        "today": today.isoformat(),
        "sports_in_view": list(_sports(data)),
        "aerobic_ceiling_bpm": data.get("aerobic_ceiling"),
        "recovery": {
            "resting_hr": sig.rhr_recent if sig else None,
            "resting_hr_vs_baseline": sig.rhr_delta if sig else None,
            "hrv_recent": sig.hrv_recent if sig else None,
            "hrv_pct_vs_baseline": sig.hrv_delta_pct if sig else None,
            "training_readiness": sig.training_readiness if sig else None,
            "training_status": sig.training_status if sig else None,
            "load_ratio_acwr": sig.acwr if sig else None,
            "vo2max_run": sig.vo2max_run if sig else None,
        } if sig else None,
        "weeks": [{"week_start": w.week_start.isoformat(),
                   "minutes": w.total_minutes, "load": w.total_load,
                   "rest_days": w.rest_days,
                   "by_sport": {sp: {"sessions": s.sessions,
                                     "minutes": s.minutes,
                                     "longest_min": s.longest_min}
                                for sp, s in w.by_sport.items()}}
                  for w in weeks],
        "this_week_plan": [{k: d.get(k) for k in
                            ("day", "sport", "duration_min", "target_zone",
                             "target_hr", "purpose", "why")}
                           for d in plan.get("week_plan", [])],
        "plan_flags": plan.get("flags") or [],
        "recent_sessions": recent,
        "checkins": [{"day": str(c.get("day")), "sleep": c.get("sleep"),
                      "soreness": c.get("soreness"),
                      "motivation": c.get("motivation"),
                      "note": (c.get("note") or "")[:140]}
                     for c in (data.get("checkins") or [])[:4]],
    }
    for page in ("Fitness", "Intensity", "Volume"):
        insight = for_page(page, data, today)
        if insight:
            facts.setdefault("page_readings", {})[page] = insight.as_facts()
    return facts


def ask(question: str, data: dict, today: date, backend: Any = None) -> str | None:
    """Answer one question from the facts bundle. None when no AI is configured."""
    from core import ai

    question = (question or "").strip()[:ASK_QUESTION_CHARS]
    if not question:
        return None
    try:
        backend = backend or ai.get_backend()
    except ai.AIUnavailable:
        return None
    payload = {"question": question, "facts": ask_facts(data, today)}
    text = backend.complete(
        ASK_SYSTEM, json.dumps(payload, separators=(",", ":"), default=str))
    return (text or "").strip()[:1200] or None
