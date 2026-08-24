"""Application logging that survives the process, stored in the database.

Everything already logs to stderr, which is the right place when you are sitting
in front of the process. It is the wrong place for the two situations that
actually matter here:

  * A hosted error. Streamlit redacts the message on screen ("The original error
    message is redacted to prevent data leaks") and the container's log sits
    behind another login, so a failure on the phone is unreadable exactly when it
    needs reading.
  * A scheduled sync. The Garmin fetch runs unattended; if it hits a rate limit
    at 6am, nothing keeps that anywhere anyone will look.

So warnings and above go to a table, alongside deliberate milestones written by
`event()`. The Log page reads it back.

Three rules this module holds itself to, because a logging handler that misbehaves
takes down whatever it was supposed to be reporting on:

  * It never raises. A failed write disables the handler and is forgotten.
  * It never logs. A handler that logs its own failure is a loop.
  * It never blocks on the shared connection. Its own short-lived Store instance
    keeps it out of whatever transaction the caller is in the middle of.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import date, datetime, timedelta
from typing import Any

from core.store import Store

# Rows to keep. Generous for a personal app that writes a handful of warnings a
# day, small enough that the table never becomes a cost.
KEEP_ROWS = int(os.getenv("APP_LOG_KEEP", "800"))
# How often to prune: one write in this many. Pruning on every insert doubles the
# cost of logging for no benefit.
PRUNE_EVERY = 40
# Below this nothing is stored. INFO would record every page load's chatter.
DEFAULT_LEVEL = os.getenv("APP_LOG_LEVEL", "WARNING").upper()

_lock = threading.Lock()
_installed: dict[str, DbLogHandler] = {}


class DbLogHandler(logging.Handler):
    """Writes each record to `app_log`. Silently disables itself on failure."""

    def __init__(self, target: str, level: int = logging.WARNING) -> None:
        super().__init__(level)
        self.target = target
        self.writes = 0
        self.disabled_reason = ""

    def emit(self, record: logging.LogRecord) -> None:
        if self.disabled_reason:
            return
        try:
            context: dict[str, Any] = {}
            if record.exc_info:
                context["exception"] = logging.Formatter().formatException(
                    record.exc_info)[-2000:]
            for field in ("module", "funcName", "lineno"):
                value = getattr(record, field, None)
                if value:
                    context[field] = value
            write(self.target, record.levelname, record.name,
                  record.getMessage(), context)
            self.writes += 1
            if self.writes % PRUNE_EVERY == 0:
                prune(self.target)
        except Exception as exc:  # noqa: BLE001 - a logger must never raise
            # No logging call here: that is how a handler becomes a loop.
            self.disabled_reason = f"{type(exc).__name__}: {exc}"


def write(target: str, level: str, logger: str, message: str,
          context: dict[str, Any] | None = None) -> None:
    """One row, on its own connection.

    Its own connection on purpose: the caller may be mid-transaction, and a
    logging write must neither join that transaction nor wait for it.
    """
    with Store(target) as store:
        store.execute(
            "INSERT INTO app_log (at, level, logger, message, context) "
            "VALUES (?, ?, ?, ?, ?)",
            [datetime.now().isoformat(timespec="seconds"), level[:10],
             (logger or "")[:80], str(message)[:2000],
             json.dumps(context, default=str)[:2000] if context else None],
        )
        store.conn.commit()


def prune(target: str, keep: int = KEEP_ROWS) -> int:
    """Drop everything older than the newest `keep` rows. Returns rows removed."""
    with Store(target) as store:
        row = store.execute(
            "SELECT id FROM app_log ORDER BY id DESC LIMIT 1 OFFSET ?",
            [keep],
        ).fetchone()
        if not row:
            return 0
        cutoff = dict(row).get("id") if not isinstance(row, tuple) else row[0]
        cur = store.execute("DELETE FROM app_log WHERE id <= ?", [cutoff])
        store.conn.commit()
        return int(getattr(cur, "rowcount", 0) or 0)


def install(target: str, level: str | int = DEFAULT_LEVEL,
            root: str = "aerobic_engine") -> DbLogHandler | None:
    """Attach the handler once per target. Safe to call on every rerun.

    Returns the handler, or None if the table is unreachable — in which case
    stderr logging carries on untouched, which is the point of failing quietly.
    """
    with _lock:
        existing = _installed.get(target)
        logger = logging.getLogger(root)
        if existing is not None and existing in logger.handlers:
            return existing
        try:
            with Store(target) as store:
                store.execute("SELECT 1 FROM app_log LIMIT 1")
        except Exception:  # noqa: BLE001 - no table, no database logging
            return None
        handler = DbLogHandler(
            target,
            level if isinstance(level, int) else
            getattr(logging, str(level).upper(), logging.WARNING))
        logger.addHandler(handler)
        # The app's loggers are all children of this one, and Streamlit sets the
        # root level; without this an INFO milestone would never reach us.
        if logger.level == logging.NOTSET or logger.level > logging.INFO:
            logger.setLevel(logging.INFO)
        _installed[target] = handler
        return handler


def event(target: str, message: str, **context: Any) -> None:
    """Record a milestone regardless of the handler's level.

    For the things worth knowing happened even when nothing went wrong: a sync
    finished, a plan was built, a workout went to the watch. Written directly
    rather than through the logging module, because those calls are deliberately
    below the stored level.
    """
    try:
        write(target, "EVENT", "aerobic_engine.event", message, context or None)
    except Exception:  # noqa: BLE001 - never worth failing the caller for
        pass


def recent(target: str, limit: int = 200, level: str | None = None,
           since: date | None = None) -> list[dict[str, Any]]:
    """Newest first, for the Log page."""
    sql = "SELECT id, at, level, logger, message, context FROM app_log WHERE 1=1"
    params: list[Any] = []
    if level and level.upper() != "ALL":
        sql += " AND level = ?"
        params.append(level.upper())
    if since:
        sql += " AND at >= ?"
        params.append(since.isoformat())
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(int(limit))
    try:
        with Store(target) as store:
            return store.query(sql, params)
    except Exception:  # noqa: BLE001 - an empty list beats a broken page
        return []


def counts(target: str, days: int = 7) -> dict[str, int]:
    """How many of each level in the last `days`, for a one-line summary."""
    since = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
    try:
        with Store(target) as store:
            rows = store.query(
                "SELECT level, COUNT(*) AS n FROM app_log WHERE at >= ? "
                "GROUP BY level", [since])
        return {str(r["level"]): int(r["n"]) for r in rows}
    except Exception:  # noqa: BLE001
        return {}
