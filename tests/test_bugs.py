"""Bug reports: stored where the fix happens, not on a note somewhere."""

from __future__ import annotations

import pytest

from core import bugs
from core.store import Store


def _store(tmp_path) -> Store:
    return Store(str(tmp_path / "bugs.db"))


def test_a_report_is_stored_and_comes_back(tmp_path):
    store = _store(tmp_path)
    bug_id = bugs.report(store, "Last sync time is five hours out", page="Today")
    assert bug_id
    rows = bugs.listing(store)
    assert len(rows) == 1
    assert rows[0]["text"].startswith("Last sync")
    assert rows[0]["page"] == "Today"
    assert rows[0]["status"] == bugs.OPEN
    # Timezone-aware, because reports arrive from a UTC host and from a laptop.
    assert "+00:00" in rows[0]["reported_at"]
    store.close()


def test_an_empty_report_is_not_stored(tmp_path):
    store = _store(tmp_path)
    assert bugs.report(store, "   ") is None
    assert bugs.listing(store) == []
    store.close()


def test_a_long_paste_is_trimmed_rather_than_refused(tmp_path):
    store = _store(tmp_path)
    bugs.report(store, "x" * 5000)
    assert len(bugs.listing(store)[0]["text"]) == bugs.MAX_CHARS
    store.close()


def test_fixing_one_records_what_was_done(tmp_path):
    store = _store(tmp_path)
    bug_id = bugs.report(store, "Bike shows as dots, not a line")
    assert bugs.resolve(store, bug_id, "Line from two points upwards")
    assert bugs.listing(store, status=bugs.OPEN) == []
    fixed = bugs.listing(store, status=bugs.FIXED)
    assert fixed[0]["resolution"] == "Line from two points upwards"
    assert fixed[0]["resolved_at"]
    store.close()


def test_not_fixing_is_a_separate_answer(tmp_path):
    """"Looked at and decided against" is different from "still open"."""
    store = _store(tmp_path)
    bug_id = bugs.report(store, "Swim workouts should go to the watch")
    bugs.resolve(store, bug_id, "Garmin builds those from pool length",
                 status=bugs.WONTFIX)
    assert bugs.counts(store) == {bugs.OPEN: 0, bugs.FIXED: 0, bugs.WONTFIX: 1}
    store.close()


def test_an_unknown_status_is_refused(tmp_path):
    store = _store(tmp_path)
    bug_id = bugs.report(store, "something")
    with pytest.raises(ValueError):
        bugs.resolve(store, bug_id, "", status="in progress")
    store.close()


def test_listing_everything_includes_the_closed_ones(tmp_path):
    store = _store(tmp_path)
    first = bugs.report(store, "one")
    bugs.report(store, "two")
    bugs.resolve(store, first, "done")
    assert len(bugs.listing(store, status=None)) == 2
    assert len(bugs.listing(store)) == 1
    store.close()


def test_a_broken_store_never_raises(tmp_path):
    class Broken:
        def execute(self, *a, **k):
            raise RuntimeError("gone")

        def query(self, *a, **k):
            raise RuntimeError("gone")

    assert bugs.report(Broken(), "text") is None
    assert bugs.listing(Broken()) == []
    assert bugs.resolve(Broken(), 1, "note") is False
    assert bugs.counts(Broken()) == {bugs.OPEN: 0, bugs.FIXED: 0, bugs.WONTFIX: 0}
