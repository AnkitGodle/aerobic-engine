#!/usr/bin/env python3
"""Send today's session to your phone.

    python scripts/notify.py --dry-run      # print it, send nothing
    python scripts/notify.py                # send it
    NOTIFY_BACKEND=telegram python scripts/notify.py

Meant for a schedule — a cron entry, or the GitHub Action that already runs the
tests. The app only helps on the days you open it; this is the day's plan
arriving whether you do or not.

The message is composed here rather than by the AI: it is the plan that already
exists, said in one screen. No model call, no cost, nothing new invented at 6am.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

from core import clock  # noqa: E402
from core import goal as goal_mod  # noqa: E402
from core import notify, planner, strength  # noqa: E402
from core.analysis import recovery_signals  # noqa: E402
from core.schemas import DAYS  # noqa: E402
from core.store import Store, default_db, week_start_of  # noqa: E402

SPORT_ICON = {"swim": "🏊", "bike": "🚲", "run": "🏃", "strength": "🦵",
              "brick": "🚲🏃", "rest": "😴"}


def compose(store: Store, today: date | None = None) -> str:
    """The day's plan as a short message."""
    # The athlete's date, not the server's: a 6am message composed on a UTC
    # host would otherwise be yesterday's session for half the evening.
    today = today or clock.today()
    facts = planner.build_facts(store, today=today)
    envelope = planner.build_envelope(facts, store)
    verdict = planner.readiness_verdict(facts)
    goal = goal_mod.load(store)
    signals = recovery_signals(store.wellness(), store.activities(), as_of=today)

    stored = (store.latest_plan(week_start_of(today)) or {}).get("plan") or {}
    marked, _ = planner.refresh_completions(stored, facts, store) if stored \
        else (None, False)
    name = DAYS[today.weekday()]
    todo = [d for d in (marked.week_plan if marked else [])
            if d.day == name and d.purpose != "completed" and d.duration_min > 0]
    done = [d for d in (marked.week_plan if marked else [])
            if d.day == name and d.purpose == "completed"]

    lines = [f"Aerobic Engine · {today.strftime('%a %d-%m')}"]
    if goal.set and (goal.weeks_to_go(today) or 0) >= 0:
        lines.append(f"{goal.event or 'Race'} in {goal.weeks_to_go(today)} weeks"
                     f" · {goal.phase(today)} phase")
    lines.append("")

    if todo:
        for d in todo:
            bits = [f"{d.duration_min} min"]
            if d.target_hr:
                bits.append(d.target_hr)
            elif d.target_zone not in ("", "n/a", None):
                bits.append(d.target_zone)
            lines.append(f"{SPORT_ICON.get(d.sport, '•')} {d.sport.upper()}"
                         f" — {' · '.join(bits)}")
            if d.sport == "strength" and d.exercise_ids:
                names = [strength.EXERCISES[e].name for e in d.exercise_ids
                         if e in strength.EXERCISES]
                if names:
                    lines.append("   " + "; ".join(names))
            if d.why:
                lines.append(f"   {d.why}")
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--db", default=None)
    parser.add_argument("--backend", default=None,
                       help="whatsapp | telegram | callmebot (default: "
                            "NOTIFY_BACKEND)")
    parser.add_argument("--dry-run", action="store_true",
                       help="print the message and send nothing")
    args = parser.parse_args()

    load_dotenv()
    with Store(args.db or default_db()) as store:
        message = compose(store)

    print(message)
    if args.dry_run:
        print(f"\n[dry run — would have sent via "
              f"{args.backend or notify.configured()}]")
        return 0

    result = notify.send(message, backend=args.backend)
    print(f"\n[{result.backend}] "
          + ("sent" if result.sent else f"not sent — {result.detail}"))
    return 0 if result.sent else 1


if __name__ == "__main__":
    raise SystemExit(main())
