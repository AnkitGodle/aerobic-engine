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


def device_hash(user_agent: str | None) -> str:
    """A short, salted fingerprint of a browser. Not reversible, not identifying.

    The user agent and nothing else. The first version mixed in the forwarding
    address, which on Streamlit Community Cloud changes from one websocket
    connection to the next — so one person opening the page in the morning was
    counted as five devices. A browser string is stable, and being unable to tell
    two identical browsers on different networks apart is the better failure.
    """
    raw = f"{SALT}|{(user_agent or '').strip()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# Not people. A headless browser is a test run, a screenshot or a crawler, and
# counting those turned "devices, all time" into a number in the thousands on a
# dashboard one person reads: every automated page load during development
# arrived with a fresh session and was counted as a new visitor. Matched on the
# user agent, which is the only thing here to match on, and which every one of
# these announces itself in.
NOT_A_VISITOR = ("headlesschrome", "playwright", "puppeteer", "phantomjs",
                 "selenium", "python-requests", "python-urllib", "httpx",
                 "curl/", "wget/", "bot", "crawler", "spider", "monitoring",
                 "uptime", "pingdom", "lighthouse", "chrome-lighthouse")


def counting() -> bool:
    """Is counting switched on at all?

    `VISIT_COUNTING=off` turns it off for a whole server, which is the only
    reliable way to keep a development run out of the numbers: Chrome's newer
    headless mode sends the same user agent as the desktop browser on the same
    machine, so the string cannot tell a screenshot script from the person whose
    dashboard it is. Set it when running the app locally to test.
    """
    return os.getenv("VISIT_COUNTING", "on").strip().lower() not in (
        "off", "0", "false", "no")


def is_automated(user_agent: str | None) -> bool:
    """Is this a test run or a crawler rather than somebody reading the page?

    An empty user agent counts as automated: a real browser always sends one,
    and what does not is something calling the URL directly. This catches the
    honest ones — curl, requests, crawlers, older headless Chrome. It cannot
    catch a headless browser that copies a real user agent, which is what
    `VISIT_COUNTING=off` is for.
    """
    agent = (user_agent or "").strip().lower()
    if not agent:
        return True
    return any(mark in agent for mark in NOT_A_VISITOR)


def record(store: Any, session_key: str, user_agent: str | None = None,
           url: str | None = None) -> None:
    """Count one visit. Safe to call on every rerun of the same session.

    The first call for a session inserts a row; later calls only bump its view
    count and the last-seen time, which is what keeps a visit a visit.
    """
    if not session_key or not counting() or is_automated(user_agent):
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
             device_hash(user_agent), (url or "")[:200]],
        )
        store.conn.commit()
    except Exception as exc:  # noqa: BLE001 - a counter must never break a page
        log.info("Could not record the visit: %s", exc)


def summary(store: Any, days: int = 7) -> dict[str, int]:
    """Visits, devices and page loads. Zeros if the table cannot be read.

    A visit is one device on one day, not one session. Streamlit opens a session
    per websocket connection and reconnects on its own, so the first version
    counted four visits for one person opening the page twice — two of them
    against its internal `/~/+` path. Counting device-days is immune to that and
    is what "visits" means to anyone reading it.
    """
    empty = {"visits": 0, "today": 0, "recent": 0, "devices": 0, "views": 0}
    try:
        today = date.today().isoformat()
        since = (date.today() - timedelta(days=days - 1)).isoformat()
        # SUBSTR rather than a date function: this SQL runs on both SQLite and
        # Postgres, and the stored timestamp is an ISO string in both.
        row = dict(store.execute(
            "SELECT"
            " COUNT(DISTINCT device_hash || SUBSTR(first_seen, 1, 10)) AS visits,"
            " COUNT(DISTINCT CASE WHEN first_seen >= ? THEN device_hash END)"
            "   AS today,"
            " COUNT(DISTINCT CASE WHEN first_seen >= ?"
            "   THEN device_hash || SUBSTR(first_seen, 1, 10) END) AS recent,"
            " COUNT(DISTINCT device_hash) AS devices,"
            " SUM(views) AS views"
            " FROM page_visits", [today, since]).fetchone() or {})
        return {k: int(row.get(k) or 0) for k in empty}
    except Exception as exc:  # noqa: BLE001
        log.info("Could not read the visit counts: %s", exc)
        return empty
