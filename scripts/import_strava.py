#!/usr/bin/env python3
"""Import history from a Strava bulk export.

For the runs that predate the watch. Garmin remains the live source; this fills
in the record behind it so lifetime totals are the whole story rather than the
part that happens to have been recorded on a Forerunner.

    python scripts/import_strava.py --zip export_12345678.zip --dry-run
    python scripts/import_strava.py --zip export_12345678.zip

Get the archive from Strava: Settings → My Account → Download or Delete Your
Account → Request your archive. It arrives by email as a zip.

Safe to run twice: activities are keyed on the Strava id, and anything already
in the database — including everything that synced to Strava from the watch — is
recognised and skipped rather than added again.

Imported activities are marked `source="strava"`, which keeps them out of every
read that feeds the planner or the AI. That is a requirement, not a nicety:
Strava's terms forbid their data being used with a language model.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

from core import strava_import  # noqa: E402
from core.store import Store, default_db  # noqa: E402


def _safe_target(target: str) -> str:
    """A database target fit to print. A Postgres URL carries its password."""
    if "://" not in target:
        return target
    scheme, rest = target.split("://", 1)
    host = rest.split("@", 1)[-1].split("?", 1)[0]
    return f"{scheme}://…@{host}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--zip", required=True,
                       help="path to the Strava export archive")
    parser.add_argument("--db", default=None,
                       help="database to write to (default: DATABASE_URL, "
                            "else the local SQLite file)")
    parser.add_argument(
        "--sports", default=",".join(strava_import.DEFAULT_SPORTS),
        help="sports to import, comma separated. Walks and hikes map to 'walk' "
             "and are left out by default: they are not training here, and "
             "adding them to a lifetime aerobic total makes it mean less")
    parser.add_argument("--tz", default="Asia/Kolkata",
                       help="your local timezone. Strava's export timestamps are "
                            "UTC while its own titles are local, so this decides "
                            "which day an evening session lands on")
    parser.add_argument("--window-min", type=int,
                       default=strava_import.DUPLICATE_WINDOW_MIN,
                       help="how close two starts have to be to count as the "
                            "same session")
    parser.add_argument("--dry-run", action="store_true",
                       help="say what would happen and write nothing")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s")

    target = args.db or default_db()
    sports = [s.strip().lower() for s in args.sports.split(",") if s.strip()]
    print(f"Reading {args.zip}")
    print(f"Writing to {_safe_target(target)}" if not args.dry_run
          else "Dry run — nothing will be written")

    with Store(target) as store:
        try:
            plan = strava_import.import_export(
                store, args.zip, sports=sports, tz=args.tz,
                window_min=args.window_min, dry_run=args.dry_run)
        except FileNotFoundError as exc:
            print(f"\n{exc}")
            return 2

        for label, count in plan.counts.items():
            print(f"  {label:14s} {count}")
        if plan.skipped_sport:
            kinds = sorted(set(plan.skipped_sport))
            print(f"  left out: {', '.join(kinds)} "
                  f"(add with --sports {','.join(sports + kinds)})")
        if plan.to_insert:
            first, last = plan.to_insert[0], plan.to_insert[-1]
            print(f"\n  {first['start_date']} to {last['start_date']}")
            per_sport: dict[str, list[dict]] = {}
            for row in plan.to_insert:
                per_sport.setdefault(row["sport"], []).append(row)
            for sport, rows in sorted(per_sport.items()):
                km = sum((r["distance_m"] or 0) for r in rows) / 1000.0
                hours = sum((r["duration_s"] or 0) for r in rows) / 3600.0
                print(f"  {sport:9s} {len(rows):3d} sessions · {km:7.1f} km · "
                      f"{hours:5.1f} h")
        if plan.duplicates:
            print(f"\n  {len(plan.duplicates)} already in the database, skipped "
                  f"(the weeks that synced from your watch)")
        if plan.to_insert and not args.dry_run:
            print("\nDone. They appear in lifetime totals and the log. They are "
                  "kept out of the planner and every AI prompt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
