#!/usr/bin/env python3
"""Send today's session to your phone.

    python scripts/notify.py --dry-run      # print it, send nothing
    python scripts/notify.py                # send it
    python scripts/notify.py --week         # the whole week instead
    python scripts/notify.py --report       # recovery and load instead
    NOTIFY_BACKEND=telegram python scripts/notify.py

Meant for a schedule — a cron entry, or the GitHub Action that already runs the
tests. The app only helps on the days you open it; this is the day's plan
arriving whether you do or not.

The message itself is composed by `core/whatsapp.py`, which is also what answers
when you reply. One implementation, so the 6am push and the reply to "today"
cannot drift apart.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

from core import notify, whatsapp  # noqa: E402
from core.store import Store, default_db  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--db", default=None)
    parser.add_argument("--backend", default=None,
                       help="whatsapp | telegram | callmebot (default: "
                            "NOTIFY_BACKEND)")
    parser.add_argument("--dry-run", action="store_true",
                       help="print the message and send nothing")
    what = parser.add_mutually_exclusive_group()
    what.add_argument("--week", action="store_true", help="the week, not today")
    what.add_argument("--report", action="store_true",
                     help="recovery, fitness and load")
    args = parser.parse_args()

    load_dotenv()
    with Store(args.db or default_db()) as store:
        if args.week:
            message = whatsapp.compose_week(store)
        elif args.report:
            message = whatsapp.compose_report(store)
        else:
            message = whatsapp.compose_today(store)

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
