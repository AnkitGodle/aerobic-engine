#!/usr/bin/env python
"""Incremental Garmin -> database sync (command line).

Thin wrapper over `core.sync`; the dashboard's Refresh button calls the same code.
Run this locally or from a scheduled job.

    python scripts/fetch.py                  # incremental since the newest row
    python scripts/fetch.py --days 90        # re-scan a window
    python scripts/fetch.py --full           # everything Garmin will give us
    python scripts/fetch.py --metrics-only   # recompute EF/zones/decoupling, no network
    python scripts/fetch.py --guard-status   # request budgets and breaker state
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

from core.garmin_guard import GarminBlocked  # noqa: E402
from core.store import default_db  # noqa: E402
from core.sync import guard_status, sync  # noqa: E402

log = logging.getLogger("aerobic_engine.fetch")


def prompt_mfa() -> str:
    """Interactive MFA. Only hit on the first login of a new session."""
    return input("Garmin MFA code: ").strip()


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(description="Sync Garmin Connect into the database")
    # Resolved after load_dotenv() below, never at import time.
    ap.add_argument("--db", default=None)
    ap.add_argument("--days", type=int, help="re-scan this many days back")
    ap.add_argument("--full", action="store_true", help="fetch all available history")
    ap.add_argument("--no-streams", action="store_true", help="skip HR streams")
    ap.add_argument("--no-wellness", action="store_true", help="skip daily wellness")
    ap.add_argument(
        "--stream-limit", type=int, default=100,
        help="max HR streams to fetch per run (one API call each)",
    )
    ap.add_argument(
        "--refresh-wellness", action="store_true",
        help="re-pull wellness days already stored (use after a parsing fix)",
    )
    ap.add_argument("--metrics-only", action="store_true",
                    help="recompute EF/zones/decoupling only, no network")
    ap.add_argument("--guard-status", action="store_true",
                    help="print Garmin request budgets and exit")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S",
    )
    if args.guard_status:
        print(json.dumps(guard_status(args.db), indent=2))
        return 0
    try:
        sync(
            db=args.db, days=args.days, full=args.full, streams=not args.no_streams,
            wellness=not args.no_wellness, metrics_only=args.metrics_only,
            stream_limit=args.stream_limit, refresh_wellness=args.refresh_wellness,
            prompt_mfa=prompt_mfa,
        )
    except GarminBlocked as exc:
        log.error("Refused by the rate guard: %s", exc)
        return 2
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001
        log.error("Sync failed: %s", exc)
        if args.verbose:
            raise
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
