"""What day it is where the athlete is.

`date.today()` is the machine's date, and the machine is Streamlit Community
Cloud, whose clock is UTC. For an athlete in India that is wrong for five and a
half hours of every day: from 18:30 to midnight local, the hosted app still
believed it was yesterday — so the evening rule that rolls the plan to tomorrow
never fired, and "today's session" was the one already done.

Everything user-facing therefore asks here instead. `LOCAL_TZ` sets the zone and
defaults to the athlete's own.
"""

from __future__ import annotations

import os
from datetime import date, datetime
from zoneinfo import ZoneInfo

DEFAULT_TZ = "Asia/Kolkata"


def zone(name: str | None = None) -> ZoneInfo:
    """The athlete's zone: `LOCAL_TZ`, else profile.toml, else the default.

    Read through core/profile.py so the whole app agrees on one answer — a
    dashboard whose dates and whose "what is today" disagreed would be worse
    than either being wrong.
    """
    if name:
        return ZoneInfo(name)
    from core import profile          # noqa: PLC0415 - avoids an import cycle
    try:
        return ZoneInfo(profile.timezone() or DEFAULT_TZ)
    except Exception:  # noqa: BLE001 - a bad zone name must not stop the app
        return ZoneInfo(os.getenv("LOCAL_TZ_FALLBACK", DEFAULT_TZ))


def now(tz: str | None = None) -> datetime:
    """Local wall-clock time, timezone-aware."""
    return datetime.now(zone(tz))


def today(tz: str | None = None) -> date:
    """The athlete's date, not the server's."""
    return now(tz).date()
