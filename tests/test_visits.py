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


def test_two_sessions_from_one_browser_are_one_visit(tmp_path):
    """Deliberate: a visit is a device on a day. Streamlit opens a session per
    websocket connection, so per-session counting turned one person opening the
    page into four visits."""
    store = _store(tmp_path)
    visits.record(store, "session-a", user_agent="Firefox")
    visits.record(store, "session-b", user_agent="Firefox")
    out = visits.summary(store)
    assert out["visits"] == 1
    assert out["devices"] == 1
    store.close()


def test_two_browsers_are_two_visits(tmp_path):
    store = _store(tmp_path)
    visits.record(store, "session-a", user_agent="Firefox")
    visits.record(store, "session-b", user_agent="Safari")
    out = visits.summary(store)
    assert out["visits"] == 2
    assert out["devices"] == 2
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
                  url="https://example.test/app")
    row = store.query("SELECT * FROM page_visits")[0]
    stored = " ".join(str(v) for v in row.values())
    assert "iPhone" not in stored
    assert len(row["device_hash"]) == 16
    store.close()


def test_the_hash_is_stable_across_connections(tmp_path):
    """It must depend on the browser and nothing that moves. Mixing in the
    forwarding address counted one visitor as five, because Community Cloud
    changes it between websocket connections."""
    first = visits.device_hash("Firefox")
    assert first == visits.device_hash("Firefox")
    assert first != visits.device_hash("Safari")
    assert len(first) == 16


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


def test_a_visit_is_one_device_on_one_day(tmp_path):
    """Streamlit opens a session per websocket connection and reconnects on its
    own, so one person opening the page twice produced four sessions — two of
    them against its internal /~/+ path. Device-days are immune to that."""
    store = _store(tmp_path)
    for n in range(4):
        visits.record(store, f"reconnect-{n}", user_agent="Chrome on a laptop")
    out = visits.summary(store)
    assert out["visits"] == 1
    assert out["today"] == 1
    assert out["devices"] == 1
    store.close()


def test_the_same_device_tomorrow_is_a_second_visit(tmp_path):
    store = _store(tmp_path)
    visits.record(store, "s-today", user_agent="Chrome")
    stamp = (date.today() - timedelta(days=1)).isoformat() + "T08:00:00"
    store.execute(
        "INSERT INTO page_visits (session_key, first_seen, last_seen, views,"
        " device_hash, url) VALUES (?, ?, ?, 1, ?, '')",
        ["s-yesterday", stamp, stamp, visits.device_hash("Chrome")])
    store.conn.commit()
    out = visits.summary(store)
    assert out["visits"] == 2
    assert out["today"] == 1
    assert out["devices"] == 1
    store.close()
