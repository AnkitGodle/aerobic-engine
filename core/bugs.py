"""Bug reports, written where the next person to fix them will be looking.

The alternative was a note on a phone. This lands in the same database as the
training data, so a fix session starts with `python scripts/bugs.py` and a list
of what is actually wrong, in the words it was noticed in — which are usually
better than the words it gets rewritten into a day later.

Deliberately small: an id, when it was reported, the text, a status, and a note
about what was done. No priorities, no labels, no assignee. One person uses this.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

log = logging.getLogger("aerobic_engine.bugs")

OPEN = "open"
FIXED = "fixed"
WONTFIX = "wontfix"
STATUSES = (OPEN, FIXED, WONTFIX)

# Long enough for a paragraph of context, short enough that a paste of a whole
# traceback does not become the report.
MAX_CHARS = 2000


def report(store: Any, text: str, page: str | None = None) -> int | None:
    """Store one report. Returns its id, or None if there was nothing to store.

    Timestamped with an offset: reports can arrive from the hosted app, whose
    clock is UTC, and from a laptop that is not.
    """
    text = (text or "").strip()[:MAX_CHARS]
    if not text:
        return None
    try:
        store.execute(
            "INSERT INTO bug_reports (reported_at, page, text, status)"
            " VALUES (?, ?, ?, ?)",
            [datetime.now(timezone.utc).isoformat(timespec="seconds"),
             (page or "")[:40], text, OPEN],
        )
        store.conn.commit()
        row = store.execute(
            "SELECT MAX(id) AS id FROM bug_reports").fetchone()
        return int(dict(row or {}).get("id") or 0) or None
    except Exception as exc:  # noqa: BLE001 - losing the page would be worse
        log.warning("Could not store the bug report: %s", exc)
        return None


def listing(store: Any, status: str | None = OPEN,
            limit: int = 100) -> list[dict[str, Any]]:
    """Reports, newest first. `status=None` for everything."""
    sql = "SELECT * FROM bug_reports"
    params: list[Any] = []
    if status:
        sql += " WHERE status = ?"
        params.append(status)
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(int(limit))
    try:
        return store.query(sql, params)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not read the bug reports: %s", exc)
        return []


def resolve(store: Any, bug_id: int, note: str = "", status: str = FIXED) -> bool:
    """Mark one report fixed (or won't-fix), with a note on what was done."""
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    try:
        store.execute(
            "UPDATE bug_reports SET status = ?, resolved_at = ?, resolution = ?"
            " WHERE id = ?",
            [status,
             datetime.now(timezone.utc).isoformat(timespec="seconds"),
             (note or "")[:MAX_CHARS], int(bug_id)],
        )
        store.conn.commit()
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not resolve bug %s: %s", bug_id, exc)
        return False


def counts(store: Any) -> dict[str, int]:
    """How many open, fixed and won't-fix."""
    out = {s: 0 for s in STATUSES}
    try:
        for row in store.query(
                "SELECT status, COUNT(*) AS n FROM bug_reports GROUP BY status"):
            out[str(row["status"])] = int(row["n"])
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not count the bug reports: %s", exc)
    return out
