"""Counting visits, without a third party.

Why not an off-the-shelf counter: Streamlit strips `<script>` out of
`st.markdown`, verified in a browser — the tag lands in the DOM and never runs —
so Google Analytics, Plausible and every other JS snippet are inert. The two
things that would work are both worse than counting it here: an `<img>` badge
from a hit-counter service tells that service every time this page is opened, and
a `components.html` iframe does run scripts but only ever measures the iframe.

Sending anything about a page of somebody's health data to an analytics vendor to
learn a number this database can count itself is a bad trade. So:

  * **One row per browser session**, not per rerun. Streamlit re-executes the
    whole script on every click, so counting script runs would report the number
    of times a slider moved.
  * **No raw identifiers.** The user agent (and the forwarding IP, where the host
    provides one) go through a salted hash and are stored only as that, which is
    enough to tell two devices apart and not enough to identify either.
  * **Never breaks the page.** Every function swallows its own failures: a
    counter is the least important thing on screen.
"""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import date, datetime, timedelta
from typing import Any

log = logging.getLogger("aerobic_engine.visits")

# Salts the device hash. Set VISIT_SALT to make the hashes unguessable; without
# it they are still not reversible to an address, just cheaper to brute-force
# against a list of common user agents.
SALT = os.getenv("VISIT_SALT", "aerobic-engine")


def device_hash(user_agent: str | None, address: str | None = None) -> str:
    """A short, salted fingerprint of a device. Not reversible, not identifying."""
    raw = f"{SALT}|{(user_agent or '').strip()}|{(address or '').strip()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def record(store: Any, session_key: str, user_agent: str | None = None,
           address: str | None = None, url: str | None = None) -> None:
    """Count one visit. Safe to call on every rerun of the same session.

    The first call for a session inserts a row; later calls only bump its view
    count and the last-seen time, which is what keeps a visit a visit.
    """
    if not session_key:
        return
    try:
        now = datetime.now().isoformat(timespec="seconds")
        store.execute(
            "INSERT INTO page_visits"
            " (session_key, first_seen, last_seen, views, device_hash, url)"
            " VALUES (?, ?, ?, 1, ?, ?)"
            " ON CONFLICT(session_key) DO UPDATE SET"
            " last_seen = excluded.last_seen, views = page_visits.views + 1",
            [str(session_key)[:64], now, now,
             device_hash(user_agent, address), (url or "")[:200]],
        )
        store.conn.commit()
    except Exception as exc:  # noqa: BLE001 - a counter must never break a page
        log.info("Could not record the visit: %s", exc)


def summary(store: Any, days: int = 7) -> dict[str, int]:
    """Visits, devices and page views. Zeros if the table cannot be read."""
    empty = {"visits": 0, "today": 0, "recent": 0, "devices": 0, "views": 0}
    try:
        today = date.today().isoformat()
        since = (date.today() - timedelta(days=days - 1)).isoformat()
        row = dict(store.execute(
            "SELECT COUNT(*) AS visits,"
            " SUM(CASE WHEN first_seen >= ? THEN 1 ELSE 0 END) AS today,"
            " SUM(CASE WHEN first_seen >= ? THEN 1 ELSE 0 END) AS recent,"
            " COUNT(DISTINCT device_hash) AS devices,"
            " SUM(views) AS views"
            " FROM page_visits", [today, since]).fetchone() or {})
        return {k: int(row.get(k) or 0) for k in empty}
    except Exception as exc:  # noqa: BLE001
        log.info("Could not read the visit counts: %s", exc)
        return empty
