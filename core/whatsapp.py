"""Your training, over WhatsApp — both directions.

`core/notify.py` sends. This is the other half: what to say when a message comes
back. It turns a line of text into one of the things the dashboard already does
— today's session, the week, a recovery report, a check-in that re-plans the
rest of the week, a question for the coach — and returns the answer as plain
text a phone can show.

Three deliberate boundaries.

**No new decisions live here.** Every answer comes from `core/planner.py`,
`core/analysis.py` or `core/insights.py`, through the same functions the pages
use. A check-in sent by message re-plans exactly the way a check-in typed into
the app does, guardrails and all — otherwise there would be two coaches, one of
them unaudited.

**The transport is not this module's problem.** `reply()` takes a string and
returns a string, so it is testable without a webhook, a tunnel or a Meta app,
and the same engine serves any transport that can carry text.

**A reply is never worth an exception.** Anything that goes wrong comes back as
a sentence, because the alternative is a phone that silently stops answering.

The commands are forgiving on purpose. Nobody types `/status` at 6am; they type
"how am i doing" or "tired, only 45 min today", and both should work.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import re
from datetime import date, timedelta
from typing import Any

from core import ai, clock, goal as goal_mod, insights, planner, strength
from core.analysis import recovery_signals
from core.schemas import DAYS, Checkin
from core.store import Store, week_start_of

log = logging.getLogger("aerobic_engine.whatsapp")

SPORT_ICON = {"swim": "🏊", "bike": "🚲", "run": "🏃", "strength": "🦵",
              "brick": "🚲🏃", "rest": "😴"}

HELP = (
    "Aerobic Engine, by message:\n"
    "• *today* — today's session and why\n"
    "• *week* — the whole week, one line a day\n"
    "• *report* — recovery, fitness and load\n"
    "• *sleep 7, sore 2, 45 min* — a check-in; the rest of the week is re-planned\n"
    "• *tired, knee feels off* — the same, in your own words\n"
    "• *replan* — build the week again from your last check-in\n"
    "• any question ending in ? — asked of the coach\n"
    "\nThe safety rules do not move for a message. A deload stays a deload."
)

# What a word means on the 1-5 scale the planner uses. "sleep 7" is hours, not a
# score, and someone who slept seven hours does not mean "wrecked" — so anything
# above the scale is read as hours and mapped.
SLEEP_HOURS = ((8.0, 5), (7.0, 4), (6.0, 3), (5.0, 2), (0.0, 1))
FEELINGS = {
    "great": {"motivation": 5, "soreness": 2},
    "good": {"motivation": 4, "soreness": 2},
    "fine": {"motivation": 3},
    "ok": {"motivation": 3},
    "okay": {"motivation": 3},
    "tired": {"motivation": 2, "sleep": 2},
    "exhausted": {"motivation": 1, "sleep": 1, "soreness": 4},
    "wrecked": {"motivation": 1, "soreness": 5},
    "sore": {"soreness": 4},
    "stiff": {"soreness": 4},
    "fresh": {"soreness": 1, "motivation": 4},
    "sick": {"motivation": 1, "soreness": 4},
    "ill": {"motivation": 1, "soreness": 4},
}
CHECKIN_WORDS = ("sleep", "slept", "sore", "soreness", "motivation", "mot",
                 "energy", "min", "mins", "minute", "minutes", "hour", "hours",
                 "time", *FEELINGS)


# --------------------------------------------------------------------------
# What to say
# --------------------------------------------------------------------------


def compose_today(store: Store, today: date | None = None) -> str:
    """The day's session as a short message.

    Composed here rather than by the AI: it is the plan that already exists,
    said in one screen. No model call, no cost, nothing invented at 6am.
    """
    today = today or clock.today()
    facts = planner.build_facts(store, today=today)
    envelope = planner.build_envelope(facts, store)
    verdict = planner.readiness_verdict(facts)
    goal = goal_mod.load(store)
    signals = recovery_signals(store.wellness(), store.activities(), as_of=today)

    stored = (store.latest_plan(week_start_of(today)) or {}).get("plan") or {}
    marked, _ = (planner.refresh_completions(stored, facts, store) if stored
                 else (None, False))
    name = DAYS[today.weekday()]
    todo = [d for d in (marked.week_plan if marked else [])
            if d.day == name and d.purpose != "completed" and d.duration_min > 0]
    done = [d for d in (marked.week_plan if marked else [])
            if d.day == name and d.purpose == "completed"]

    lines = [f"*Aerobic Engine* · {today.strftime('%a %d-%m')}"]
    if goal.set and (goal.weeks_to_go(today) or 0) >= 0:
        lines.append(f"{goal.event or 'Race'} in {goal.weeks_to_go(today)} weeks"
                     f" · {goal.phase(today)} phase")
    lines.append("")

    if todo:
        for entry in todo:
            bits = [f"{entry.duration_min} min"]
            if entry.target_hr:
                bits.append(entry.target_hr)
            elif entry.target_zone not in ("", "n/a", None):
                bits.append(entry.target_zone)
            lines.append(f"{SPORT_ICON.get(entry.sport, '•')} "
                         f"*{entry.sport.upper()}* — {' · '.join(bits)}")
            if entry.sport == "strength" and entry.exercise_ids:
                names = [strength.EXERCISES[e].name for e in entry.exercise_ids
                         if e in strength.EXERCISES]
                if names:
                    lines.append("   " + "; ".join(names))
            if entry.why:
                lines.append(f"   {entry.why}")
    elif done:
        lines.append("✓ Done today: "
                     + ", ".join(f"{d.sport} {d.duration_min} min" for d in done))
    else:
        lines.append("😴 Rest day. That is the plan, not a gap in it.")

    lines.append("")
    reading = []
    if signals.training_readiness:
        reading.append(f"readiness {signals.training_readiness:.0f}")
    if signals.rhr_recent:
        reading.append(f"resting HR {signals.rhr_recent:.0f}")
    if signals.hrv_recent:
        reading.append(f"HRV {signals.hrv_recent:.0f}")
    if reading:
        lines.append(" · ".join(reading))
    lines.append(verdict["headline"])
    if envelope.deload:
        lines.append("Easy week: " + "; ".join(envelope.deload_reasons[:2]))
    left = planner.remaining_budget(facts, envelope)
    lines.append(f"{left / 60:.1f}h left of this week's ceiling")
    return "\n".join(lines).strip()


def compose_week(store: Store, today: date | None = None) -> str:
    """The week, one line a day, with what is already done marked."""
    today = today or clock.today()
    facts = planner.build_facts(store, today=today)
    stored = (store.latest_plan(week_start_of(today)) or {}).get("plan") or {}
    if not stored:
        return ("No plan saved for this week yet. Send *replan* and I will build "
                "one, or open the app and press Plan my week.")
    marked, _ = planner.refresh_completions(stored, facts, store)
    envelope = planner.build_envelope(facts, store)

    by_day: dict[str, list[Any]] = {d: [] for d in DAYS}
    for entry in marked.week_plan:
        if entry.day in by_day:
            by_day[entry.day].append(entry)

    lines = [f"*This week* · {facts.week_start.strftime('%d-%m')} onwards"]
    if envelope.deload:
        lines.append("Easy week — the recovery numbers asked for it.")
    lines.append("")
    name_today = DAYS[today.weekday()]
    for day in DAYS:
        here = "→" if day == name_today else " "
        entries = [e for e in by_day[day] if e.duration_min > 0
                   or e.sport == "rest"]
        if not entries:
            lines.append(f"{here} {day}  rest")
            continue
        parts = []
        for entry in entries:
            done = "✓" if entry.purpose == "completed" else ""
            if entry.sport == "rest" or entry.duration_min <= 0:
                parts.append(f"{SPORT_ICON['rest']} rest")
                continue
            zone = (f" {entry.target_zone}"
                    if entry.target_zone not in ("", "n/a", None) else "")
            parts.append(f"{done}{SPORT_ICON.get(entry.sport, '•')} {entry.sport}"
                         f" {entry.duration_min}′{zone}".strip())
        lines.append(f"{here} {day}  " + " + ".join(parts))
    lines.append("")
    done_minutes = facts.completed_this_week.total_minutes
    lines.append(f"{done_minutes / 60:.1f}h done, "
                 f"{planner.remaining_budget(facts, envelope) / 60:.1f}h left of "
                 f"the ceiling")
    return "\n".join(lines)


def compose_report(store: Store, today: date | None = None) -> str:
    """Recovery, fitness and load — the numbers a coach would ask for first."""
    today = today or clock.today()
    facts = planner.build_facts(store, today=today)
    verdict = planner.readiness_verdict(facts)
    signals = recovery_signals(store.wellness(), store.activities(), as_of=today)
    goal = goal_mod.load(store)

    lines = [f"*How you are* · {today.strftime('%a %d-%m')}", verdict["headline"]]
    if verdict.get("reasons"):
        lines.append("· " + "\n· ".join(verdict["reasons"][:4]))
    lines.append("")

    if signals.rhr_recent:
        drift = ""
        if signals.rhr_baseline:
            gap = signals.rhr_recent - signals.rhr_baseline
            drift = f" ({gap:+.0f} vs baseline)"
        lines.append(f"Resting HR {signals.rhr_recent:.0f}{drift}")
    if signals.hrv_recent:
        drift = ""
        if signals.hrv_baseline:
            pct = (signals.hrv_recent - signals.hrv_baseline) / signals.hrv_baseline
            drift = f" ({pct * 100:+.0f}% vs baseline)"
        lines.append(f"HRV {signals.hrv_recent:.0f}{drift}")
    if signals.training_readiness:
        lines.append(f"Readiness {signals.training_readiness:.0f}")
    if signals.acwr:
        lines.append(f"Load ratio {signals.acwr:.2f} "
                     f"(this week against your own four)")
    if signals.vo2max_run:
        lines.append(f"VO2max {signals.vo2max_run:.0f} (run)")

    lines.append("")
    week = facts.completed_this_week
    sessions = sum(s.sessions for s in week.by_sport.values())
    lines.append(f"This week: {week.total_minutes / 60:.1f}h across "
                 f"{sessions} session{'' if sessions == 1 else 's'}")
    if goal.set:
        lines.append(goal_mod.describe(goal, today))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Check-ins
# --------------------------------------------------------------------------


def _sleep_score(hours: float) -> int:
    for floor, score in SLEEP_HOURS:
        if hours >= floor:
            return score
    return 3


def parse_checkin(text: str) -> dict[str, Any] | None:
    """Read a check-in out of a sentence, or return None if there is not one.

    Handles the two ways people actually write it: named numbers ("sleep 4,
    sore 2, 45 min") and plain words ("tired, knee feels off, only 45 minutes").
    A sleep number above the 1-5 scale is read as hours, because someone who
    types "slept 7" means seven hours and not "off the scale".
    """
    lowered = (text or "").lower()
    if not lowered.strip():
        return None
    found: dict[str, Any] = {}

    for key, pattern in (
        ("sleep", r"\b(?:sleep|slept)\s*[:=]?\s*(\d+(?:\.\d+)?)"),
        ("soreness", r"\b(?:sore|soreness)\s*[:=]?\s*(\d+)"),
        ("motivation", r"\b(?:motivation|mot|energy)\s*[:=]?\s*(\d+)"),
    ):
        match = re.search(pattern, lowered)
        if match:
            value = float(match.group(1))
            if key == "sleep" and value > 5:
                value = _sleep_score(value)
            found[key] = max(1, min(5, int(round(value))))

    minutes = re.search(r"(\d+(?:\.\d+)?)\s*(min|mins|minute|minutes)\b", lowered)
    hours = re.search(r"(\d+(?:\.\d+)?)\s*(h|hr|hrs|hour|hours)\b", lowered)
    if minutes:
        found["time_available_min"] = int(min(480, float(minutes.group(1))))
    elif hours:
        found["time_available_min"] = int(min(480, float(hours.group(1)) * 60))

    for word, effects in FEELINGS.items():
        if re.search(rf"\b{word}\b", lowered):
            for key, value in effects.items():
                found.setdefault(key, value)

    if not found:
        return None
    found["notes"] = (text or "").strip()[:400]
    return found


def apply_checkin(store: Store, text: str, today: date | None = None,
                  use_ai: bool | None = None) -> str:
    """Save a check-in, re-plan the rest of the week, and say what changed.

    The same call the Plan page makes, so the envelope, the deload triggers and
    `enforce()` all apply. A message cannot talk the planner into a week the app
    would refuse.
    """
    today = today or clock.today()
    parsed = parse_checkin(text) or {}
    last = store.latest_checkin() or {}
    checkin = Checkin(
        date=today,
        sleep=parsed.get("sleep", last.get("sleep") or 3),
        soreness=parsed.get("soreness", last.get("soreness") or 3),
        motivation=parsed.get("motivation", last.get("motivation") or 3),
        time_available_min=parsed.get("time_available_min",
                                      last.get("time_available_min") or 90),
        notes=parsed.get("notes", ""),
    )
    store.save_checkin({
        "day": today.isoformat(), "sleep": checkin.sleep,
        "soreness": checkin.soreness, "motivation": checkin.motivation,
        "time_available_min": checkin.time_available_min,
        "notes": checkin.notes, "source": "whatsapp",
    })
    plan = planner.plan_week(
        store, checkin=checkin, today=today,
        use_ai=ai.available() if use_ai is None else use_ai)

    head = (f"Noted: sleep {checkin.sleep}/5, soreness {checkin.soreness}/5, "
            f"motivation {checkin.motivation}/5, "
            f"{checkin.time_available_min} min today.")
    tail = []
    for note in (plan.adjustments_made or [])[:4]:
        tail.append(f"· {note}")
    for flag in (plan.flags or [])[:2]:
        tail.append(f"! {flag}")
    body = compose_week(store, today)
    return "\n".join([head, *tail, "", body]).strip()


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------


def _ask(store: Store, question: str, today: date) -> str:
    """Put the question to the coach, with the same context the app gives it."""
    payload = {
        "activities": store.activities(), "wellness": store.wellness(),
        "zones": store.zones(), "strength": store.strength_log(),
        "checkins": store.checkins(limit=8), "race": store.race_predictions(),
        "records": store.personal_records(), "targets": store.targets(),
        "aerobic_ceiling": store.get_state("aerobic_ceiling_bpm"),
        "plan": store.latest_plan(week_start_of(today)),
    }
    answer = insights.ask(question, payload, today)
    return answer or ("The AI layer did not answer. Send *report* for the "
                      "numbers, which are computed here and always available.")


def reply(store: Store, text: str, today: date | None = None) -> str:
    """One message in, one message out. Never raises."""
    today = today or clock.today()
    body = (text or "").strip()
    lowered = body.lower().strip(" .!?")
    try:
        if not body:
            return HELP
        if lowered in ("help", "?", "hi", "hello", "start", "menu", "commands"):
            return HELP
        if lowered in ("today", "t", "now", "session", "workout"):
            return compose_today(store, today)
        if lowered in ("week", "plan", "this week", "weekly"):
            return compose_week(store, today)
        if lowered in ("report", "status", "stats", "how am i", "how am i doing",
                       "recovery"):
            return compose_report(store, today)
        if lowered in ("tomorrow", "next"):
            return compose_today(store, today + timedelta(days=1))
        if lowered in ("replan", "re-plan", "rebuild", "plan again"):
            return apply_checkin(store, "", today)
        if lowered.startswith("checkin") or lowered.startswith("check in"):
            return apply_checkin(store, body.split(" ", 1)[-1], today)
        if lowered.startswith("ask "):
            return _ask(store, body[4:], today)
        # A check-in reads as one before a question does, because "tired, should
        # I still ride?" is both and the useful answer is the re-planned week.
        if parse_checkin(body) and any(w in lowered for w in CHECKIN_WORDS):
            return apply_checkin(store, body, today)
        if body.endswith("?") or len(body.split()) > 4:
            return _ask(store, body, today)
        return ("I did not follow that.\n\n" + HELP)
    except Exception as exc:  # noqa: BLE001 - see the module docstring
        log.warning("Could not answer %r: %s", body[:60], exc)
        return ("Something broke working that out — it is logged. Try *today* "
                "or *report*, which read straight from your stored sessions.")


# --------------------------------------------------------------------------
# Who is allowed to talk to it
# --------------------------------------------------------------------------


def digits(value: str) -> str:
    return "".join(c for c in str(value or "") if c.isdigit())


def sender_allowed(number: str, allowed: str) -> bool:
    """Is this number on the list?

    Compared on digits, and on the last ten of them, because the same phone
    arrives as 9940173970, 919940173970 and +91 99401 73970 depending on who is
    asking. An empty allowlist allows nobody: a public endpoint that answers
    anyone would hand a stranger this athlete's training and let them re-plan it.
    """
    mine = digits(number)
    if not mine:
        return False
    for entry in str(allowed or "").split(","):
        theirs = digits(entry)
        if theirs and (theirs == mine or theirs[-10:] == mine[-10:]):
            return True
    return False


def verify_signature(body: bytes, header: str, secret: str) -> bool:
    """Meta signs every webhook delivery. Unsigned means not from Meta.

    Without this the endpoint is a URL anyone who finds it can POST to. With no
    secret configured this returns False rather than True: failing open on the
    one control that proves who is calling is not a default worth having.
    """
    if not secret or not header:
        return False
    expected = hmac.new(secret.encode("utf-8"), body,
                        hashlib.sha256).hexdigest()
    sent = header.split("=", 1)[-1].strip()
    return hmac.compare_digest(expected, sent)


def messages_from(payload: dict[str, Any]) -> list[dict[str, str]]:
    """The text messages in a Meta webhook body, as {id, from, text}.

    Meta nests these four deep and sends status callbacks — delivered, read —
    through the same endpoint. Those carry no `messages` key and must not be
    answered, or every reply would trigger its own reply.
    """
    out = []
    for entry in payload.get("entry") or []:
        for change in entry.get("changes") or []:
            value = change.get("value") or {}
            for message in value.get("messages") or []:
                if message.get("type") != "text":
                    continue
                out.append({
                    "id": str(message.get("id") or ""),
                    "from": str(message.get("from") or ""),
                    "text": str((message.get("text") or {}).get("body") or ""),
                })
    return out
