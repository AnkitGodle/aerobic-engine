"""Application logging into the database.

A logging handler that raises takes down whatever it was reporting on, so most of
these are about failure behaviour rather than the happy path.
"""

from __future__ import annotations

import logging

from core import applog
from core.store import Store


def _db(tmp_path) -> str:
    target = str(tmp_path / "log.db")
    Store(target).close()          # creates the schema
    return target


def test_a_warning_is_stored(tmp_path):
    target = _db(tmp_path)
    handler = applog.install(target, level="WARNING", root="test_applog_a")
    assert handler is not None
    logging.getLogger("test_applog_a.thing").warning("Garmin said no: %s", 429)

    rows = applog.recent(target)
    assert len(rows) == 1
    assert rows[0]["level"] == "WARNING"
    assert "Garmin said no: 429" in rows[0]["message"]
    assert "test_applog_a.thing" == rows[0]["logger"]


def test_info_is_not_stored_by_default(tmp_path):
    """Otherwise every page load fills the table with its own chatter."""
    target = _db(tmp_path)
    applog.install(target, level="WARNING", root="test_applog_b")
    logging.getLogger("test_applog_b").info("rendered a page")
    assert applog.recent(target) == []


def test_a_milestone_is_stored_regardless_of_level(tmp_path):
    target = _db(tmp_path)
    applog.event(target, "sync finished", activities=6, streams=3)
    rows = applog.recent(target)
    assert rows[0]["level"] == "EVENT"
    assert "sync finished" in rows[0]["message"]
    assert "activities" in (rows[0]["context"] or "")


def test_an_exception_keeps_its_traceback(tmp_path):
    target = _db(tmp_path)
    applog.install(target, level="WARNING", root="test_applog_c")
    try:
        raise ValueError("no heart rate in that stream")
    except ValueError:
        logging.getLogger("test_applog_c").exception("could not compute metrics")
    row = applog.recent(target)[0]
    assert row["level"] == "ERROR"
    assert "ValueError" in (row["context"] or "")


def test_installing_twice_adds_one_handler(tmp_path):
    """It is called on every Streamlit rerun."""
    target = _db(tmp_path)
    first = applog.install(target, root="test_applog_d")
    second = applog.install(target, root="test_applog_d")
    assert first is second
    assert sum(isinstance(h, applog.DbLogHandler)
               for h in logging.getLogger("test_applog_d").handlers) == 1


def test_a_broken_target_does_not_raise(tmp_path):
    """Failing to log must never be the reason a page dies. A directory where a
    file should be is the cheapest unusable target — Store creates missing
    parent directories, so a non-existent path is not one."""
    broken = str(tmp_path)
    assert applog.install(broken) is None
    applog.event(broken, "still fine")
    assert applog.recent(broken) == []
    assert applog.counts(broken) == {}


def test_a_failing_write_disables_the_handler_quietly(tmp_path):
    target = _db(tmp_path)
    handler = applog.install(target, level="WARNING", root="test_applog_e")
    assert handler is not None
    original = applog.write

    def boom(*_a, **_k):
        raise RuntimeError("database gone")

    applog.write = boom
    try:
        logging.getLogger("test_applog_e").warning("this cannot be stored")
    finally:
        applog.write = original
    assert handler.disabled_reason.startswith("RuntimeError")
    # And it stays quiet rather than retrying on every record.
    logging.getLogger("test_applog_e").warning("nor this")
    assert applog.recent(target) == []


def test_pruning_keeps_the_newest_rows(tmp_path):
    target = _db(tmp_path)
    for n in range(12):
        applog.event(target, f"message {n}")
    removed = applog.prune(target, keep=5)
    assert removed == 7
    rows = applog.recent(target)
    assert len(rows) == 5
    assert rows[0]["message"] == "message 11"


def test_counts_summarise_by_level(tmp_path):
    target = _db(tmp_path)
    applog.install(target, level="WARNING", root="test_applog_f")
    logging.getLogger("test_applog_f").warning("one")
    logging.getLogger("test_applog_f").error("two")
    applog.event(target, "three")
    assert applog.counts(target) == {"WARNING": 1, "ERROR": 1, "EVENT": 1}
