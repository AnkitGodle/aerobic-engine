"""Copy a local SQLite database into Postgres, without touching Garmin.

The point of this script is traffic, not convenience. A hosted deployment that
starts on an empty database re-pulls months of history on its first sync —
hundreds of requests from a datacenter IP, which is exactly the pattern that
gets an unofficial-API account flagged. Copying the rows you already have costs
zero Garmin calls.

Safe to re-run: every table upserts on its real key, so a second pass updates
rather than duplicates.

    python scripts/migrate_to_postgres.py --from data/aerobic_engine.db \
        --to "postgresql://…?sslmode=require"
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

from core.store import Store, is_postgres  # noqa: E402

# Parents before children: hr_streams and the per-activity tables carry a
# foreign key to activities, so the order here is a correctness requirement,
# not tidiness.
TABLES: list[tuple[str, tuple[str, ...]]] = [
    ("activities", ("activity_id",)),
    ("activity_metrics", ("activity_id",)),
    ("activity_zones", ("activity_id", "zone_number")),
    ("activity_weather", ("activity_id",)),
    ("exercise_sets", ("id",)),
    ("hr_streams", ("activity_id", "t_s")),
    ("daily_wellness", ("day",)),
    ("race_predictions", ("day",)),
    ("personal_records", ("type_id",)),
    ("strength_log", ("id",)),
    ("checkins", ("id",)),
    ("plans", ("id",)),
    ("weekly_targets", ("sport",)),
    ("ai_notes", ("key",)),
    ("sync_state", ("key",)),
]

# Tables whose id is a BIGSERIAL on the destination. Rows are copied with their
# original ids, so the sequence has to be pushed past them or the next insert
# collides with a row that already exists.
SERIAL_ID = {"exercise_sets", "strength_log", "checkins", "plans"}

BATCH = 1000


def copy(src_path: str, dest_url: str, dry_run: bool = False) -> int:
    if not is_postgres(dest_url):
        raise SystemExit(f"--to must be a Postgres URL, got {dest_url!r}")
    if not Path(src_path).exists():
        raise SystemExit(f"no such SQLite file: {src_path}")

    src = sqlite3.connect(f"file:{src_path}?mode=ro", uri=True)
    src.row_factory = sqlite3.Row
    have = {r["name"] for r in src.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}

    total = 0
    with Store(dest_url) as dest:          # constructing it also runs migrate()
        for table, key in TABLES:
            if table not in have:
                print(f"  {table:18} — not in source, skipped")
                continue
            rows = [dict(r) for r in src.execute(f"SELECT * FROM {table}")]
            if not rows:
                print(f"  {table:18} 0")
                continue

            # Only columns both sides agree on: an older local database may
            # predate a column, and a newer one may have dropped nothing but
            # still differ from what the destination schema built.
            cols = [c for c in rows[0] if c in dest._columns(table)]
            missing = sorted(set(rows[0]) - set(cols))
            updates = ", ".join(
                f"{c}=excluded.{c}" for c in cols if c not in key)
            sql = (
                f"INSERT INTO {table} ({','.join(cols)}) "
                f"VALUES ({','.join('?' * len(cols))}) "
                + (f"ON CONFLICT({','.join(key)}) DO UPDATE SET {updates}"
                   if updates else
                   f"ON CONFLICT({','.join(key)}) DO NOTHING")
            )
            if dry_run:
                print(f"  {table:18} {len(rows)} (dry run)")
                total += len(rows)
                continue
            for i in range(0, len(rows), BATCH):
                chunk = rows[i:i + BATCH]
                with dest.tx():
                    dest.executemany(sql, [[r.get(c) for c in cols]
                                           for r in chunk])
            note = f"  (dropped unknown columns: {', '.join(missing)})" if missing else ""
            print(f"  {table:18} {len(rows)}{note}")
            total += len(rows)

            if table in SERIAL_ID:
                with dest.tx():
                    dest.execute(
                        f"SELECT setval(pg_get_serial_sequence('{table}', 'id'),"
                        f" GREATEST(COALESCE((SELECT MAX(id) FROM {table}), 1), 1))"
                    )

        print("\nDestination now holds:")
        for name, n in sorted(dest.counts().items()):
            print(f"  {name:18} {n}")
    src.close()
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    # .env is read before the defaults are resolved, not after: reading it late
    # is what let the sync write to the wrong database entirely.
    load_dotenv()
    ap.add_argument("--from", dest="src",
                    default=os.getenv("AEROBIC_ENGINE_DB", "data/aerobic_engine.db"))
    ap.add_argument("--to", dest="dest", default=os.getenv("DATABASE_URL"))
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be copied, write nothing")
    a = ap.parse_args()
    if not a.dest:
        raise SystemExit("pass --to, or set DATABASE_URL")
    print(f"copying {a.src} -> Postgres\n")
    n = copy(a.src, a.dest, a.dry_run)
    print(f"\n{n} rows{' (dry run)' if a.dry_run else ''}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
