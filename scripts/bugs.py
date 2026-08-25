#!/usr/bin/env python3
"""Read the bug reports, and mark them done.

The other half of the sidebar's "Report a bug": what is reported from the app is
fixed from here, so a fix session starts with the list rather than with trying to
remember.

    python scripts/bugs.py                    # what is open
    python scripts/bugs.py --all              # everything, with what was done
    python scripts/bugs.py --fix 3 "Ran the timestamps through the local zone"
    python scripts/bugs.py --wontfix 4 "Garmin does not expose that"
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

from core import bugs  # noqa: E402
from core.store import Store, default_db  # noqa: E402


def show(store: Store, status: str | None) -> None:
    rows = bugs.listing(store, status=status, limit=200)
    if not rows:
        print("Nothing reported." if status is None
              else f"No {status} reports.")
        return
    counts = bugs.counts(store)
    print(f"{counts.get(bugs.OPEN, 0)} open · {counts.get(bugs.FIXED, 0)} fixed"
          f" · {counts.get(bugs.WONTFIX, 0)} not fixing\n")
    for row in rows:
        head = f"#{row['id']}  {row['reported_at'][:16].replace('T', ' ')}"
        if row.get("page"):
            head += f"  on {row['page']}"
        if row["status"] != bugs.OPEN:
            head += f"  [{row['status']}]"
        print(head)
        for line in textwrap.wrap(row["text"], 76):
            print(f"    {line}")
        if row.get("resolution"):
            for line in textwrap.wrap(f"→ {row['resolution']}", 76):
                print(f"    {line}")
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--db", default=None, help="database to read")
    parser.add_argument("--all", action="store_true",
                       help="show fixed and won't-fix as well")
    parser.add_argument("--fix", nargs=2, metavar=("ID", "NOTE"),
                       help="mark one report fixed, with what was done")
    parser.add_argument("--wontfix", nargs=2, metavar=("ID", "NOTE"),
                       help="mark one report as not being fixed, and why")
    args = parser.parse_args()

    load_dotenv()
    with Store(args.db or default_db()) as store:
        if args.fix:
            bug_id, note = args.fix
            ok = bugs.resolve(store, int(bug_id), note, status=bugs.FIXED)
            print(f"#{bug_id} marked fixed." if ok else "Could not update it.")
            return 0 if ok else 1
        if args.wontfix:
            bug_id, note = args.wontfix
            ok = bugs.resolve(store, int(bug_id), note, status=bugs.WONTFIX)
            print(f"#{bug_id} left alone." if ok else "Could not update it.")
            return 0 if ok else 1
        show(store, None if args.all else bugs.OPEN)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
