"""Counting visits in our own database, rather than telling an analytics vendor
about a page of someone's health data."""

from __future__ import annotations

from datetime import date, timedelta

from core import visits
from core.store import Store


def _store(tmp_path) -> Store:
    return Store(str(tmp_path / "visits.db"))


def test_a_visit_is_counted_once_however_many_reruns(tmp_path):
    """Streamlit re-executes the whole script on every click. Counting runs would
    report how often a slider moved."""
    store = _store(tmp_path)
    for _ in range(5):
        visits.record(store, "session-a", user_agent="Firefox")
    out = visits.summary(store)
    assert out["visits"] == 1
    assert out["views"] == 5          # the reruns are still visible, separately
    store.close()


def test_two_sessions_are_two_visits(tmp_path):
    store = _store(tmp_path)
    visits.record(store, "session-a", user_agent="Firefox")
    visits.record(store, "session-b", user_agent="Firefox")
    out = visits.summary(store)
    assert out["visits"] == 2
    # Same browser, so one device.
    assert out["devices"] == 1
    store.close()


def test_different_devices_are_counted_apart(tmp_path):
    store = _store(tmp_path)
    visits.record(store, "s1", user_agent="Firefox on a laptop")
    visits.record(store, "s2", user_agent="Safari on a phone")
    assert visits.summary(store)["devices"] == 2
    store.close()


def test_nothing_identifying_is_stored(tmp_path):
    """The point of hashing: enough to tell two devices apart, not enough to say
    whose they are."""
    store = _store(tmp_path)
    visits.record(store, "s1", user_agent="Mozilla/5.0 (iPhone)",
                  address="203.0.113.7", url="https://example.test/app")
    row = store.query("SELECT * FROM page_visits")[0]
    stored = " ".join(str(v) for v in row.values())
    assert "203.0.113.7" not in stored
    assert "iPhone" not in stored
    assert len(row["device_hash"]) == 16
    store.close()


def test_the_hash_is_stable_and_salted(tmp_path):
    first = visits.device_hash("Firefox", "10.0.0.1")
    assert first == visits.device_hash("Firefox", "10.0.0.1")
    assert first != visits.device_hash("Firefox", "10.0.0.2")
    assert first != visits.device_hash("Safari", "10.0.0.1")


def test_today_and_the_last_week_are_separated(tmp_path):
    store = _store(tmp_path)
    visits.record(store, "today-1", user_agent="A")
    old = (date.today() - timedelta(days=20)).isoformat() + "T09:00:00"
    store.execute(
        "INSERT INTO page_visits (session_key, first_seen, last_seen, views,"
        " device_hash, url) VALUES (?, ?, ?, 1, 'x', '')", ["old-1", old, old])
    store.conn.commit()
    out = visits.summary(store)
    assert out["visits"] == 2
    assert out["today"] == 1
    assert out["recent"] == 1
    store.close()


def test_an_empty_session_key_is_ignored(tmp_path):
    store = _store(tmp_path)
    visits.record(store, "", user_agent="A")
    assert visits.summary(store)["visits"] == 0
    store.close()


def test_a_broken_store_never_raises(tmp_path):
    """A counter is the least important thing on the page."""
    class Broken:
        def execute(self, *a, **k):
            raise RuntimeError("database gone")

    visits.record(Broken(), "s1", user_agent="A")      # must not raise
    assert visits.summary(Broken()) == {
        "visits": 0, "today": 0, "recent": 0, "devices": 0, "views": 0}
