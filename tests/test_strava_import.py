"""Importing a Strava bulk export.

The three ways this can quietly corrupt data — the wrong timezone, double
counting what Garmin already has, and imported rows reaching the AI layer — are
what these tests are about.
"""

from __future__ import annotations

import csv
import io
import zipfile
from datetime import datetime

import pytest

from core import strava_import as si
from core.store import Store

HEADER = ["Activity ID", "Activity Date", "Activity Name", "Activity Type",
          "Elapsed Time", "Moving Time", "Distance", "Average Speed",
          "Elevation Gain", "Average Heart Rate", "Calories"]


def _row(activity_id="1", when="Aug 22, 2026, 12:40:43 PM", kind="Run",
         name="Evening Run", moving="2082", distance="4011.6",
         speed="1.927", hr="", elapsed=None):
    return {"Activity ID": activity_id, "Activity Date": when,
            "Activity Name": name, "Activity Type": kind,
            "Elapsed Time": elapsed or moving, "Moving Time": moving,
            "Distance": distance, "Average Speed": speed,
            "Elevation Gain": "10.0", "Average Heart Rate": hr,
            "Calories": "250.0"}


def _archive(tmp_path, rows, inner="activities.csv"):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=HEADER)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    path = tmp_path / "export.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(inner, buffer.getvalue())
    return str(path)


def test_the_exports_clock_is_utc():
    """Strava writes the timestamp in UTC and the title in local time, which is
    how a 12:40 entry is called an evening run. Stored raw, every session after
    18:30 in India lands on the wrong day."""
    assert si.parse_start("Aug 22, 2026, 12:40:43 PM") == \
        datetime(2026, 8, 22, 18, 10, 43)
    # And one that crosses midnight going the other way.
    assert si.parse_start("Aug 22, 2026, 7:00:00 PM").date().isoformat() == \
        "2026-08-23"


def test_a_row_becomes_an_activity():
    parsed = si.parse_row(_row())
    assert parsed["activity_id"] == "strava-1"
    assert parsed["sport"] == "run"
    assert parsed["start_date"] == "2026-08-22"
    assert parsed["duration_s"] == 2082.0
    assert parsed["distance_m"] == 4011.6
    assert parsed["source"] == "strava"
    # No heart rate in the export, and none invented.
    assert parsed["avg_hr"] is None


def test_speed_is_derived_when_the_export_omits_it():
    parsed = si.parse_row(_row(speed="", distance="5000", moving="2500"))
    assert parsed["avg_speed_mps"] == pytest.approx(2.0)


def test_an_unknown_activity_type_is_skipped():
    assert si.parse_row(_row(kind="Kitesurf")) is None


def test_what_garmin_already_has_is_not_imported_again(tmp_path):
    """The recent weeks exist in both services, and counting them twice would
    inflate exactly the totals this is meant to complete."""
    store = Store(str(tmp_path / "db.sqlite"))
    store.upsert_activities([{
        "activity_id": "garmin-1", "sport": "run", "name": "Pune Running",
        "start_time": "2026-08-22T18:10:43", "start_date": "2026-08-22",
        "duration_s": 2088.0, "distance_m": 4011.0,
        "ingested_at": "2026-08-22T19:00:00"}])
    plan = si.import_export(store, _archive(tmp_path, [_row()]))
    assert plan.counts["import"] == 0
    assert len(plan.duplicates) == 1
    assert plan.duplicates[0][1] == "garmin-1"
    store.close()


def test_two_sessions_close_together_are_still_two_sessions(tmp_path):
    """A window wide enough to absorb the difference between two services is
    wide enough to merge two short runs on one evening. Ten minutes, because
    every real overlap matched to the second."""
    store = Store(str(tmp_path / "db.sqlite"))
    plan = si.import_export(store, _archive(tmp_path, [
        _row("1", "May 20, 2026, 2:58:28 PM", distance="1030"),
        _row("2", "May 20, 2026, 3:17:11 PM", distance="870"),
    ]))
    assert plan.counts["import"] == 2
    store.close()


def test_walks_are_left_out_unless_asked_for(tmp_path):
    store = Store(str(tmp_path / "db.sqlite"))
    archive = _archive(tmp_path, [_row("1"), _row("2", kind="Walk")])
    assert si.import_export(store, archive, dry_run=True).counts["import"] == 1
    with_walks = si.import_export(
        store, archive, sports=("run", "walk"), dry_run=True)
    assert with_walks.counts["import"] == 2
    store.close()


def test_imported_activities_are_kept_away_from_the_planner(tmp_path):
    """The boundary that matters: Strava's terms forbid their data being used
    with a language model, and every planning decision here goes through one."""
    store = Store(str(tmp_path / "db.sqlite"))
    si.import_export(store, _archive(tmp_path, [_row()]))

    assert store.activities() == []                    # what the planner reads
    imported = store.activities(include_imported=True)  # what totals read
    assert [a["activity_id"] for a in imported] == ["strava-1"]
    assert imported[0]["source"] == "strava"
    store.close()


def test_running_it_twice_changes_nothing(tmp_path):
    store = Store(str(tmp_path / "db.sqlite"))
    archive = _archive(tmp_path, [_row("1"), _row("2", "Aug 10, 2026, 6:00:00 AM")])
    first = si.import_export(store, archive)
    second = si.import_export(store, archive)
    assert first.counts["import"] == 2
    assert second.counts["import"] == 0
    assert len(store.activities(include_imported=True)) == 2
    store.close()


def test_a_dry_run_writes_nothing(tmp_path):
    store = Store(str(tmp_path / "db.sqlite"))
    plan = si.import_export(store, _archive(tmp_path, [_row()]), dry_run=True)
    assert plan.counts["import"] == 1
    assert store.activities(include_imported=True) == []
    store.close()


def test_the_csv_is_found_in_a_subdirectory(tmp_path):
    """Strava has moved it between the archive root and a folder."""
    rows = si.read_export(_archive(tmp_path, [_row()],
                                  inner="export_123/activities.csv"))
    assert len(rows) == 1


def test_an_archive_without_activities_says_so(tmp_path):
    path = tmp_path / "not-an-export.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("readme.txt", "nothing here")
    with pytest.raises(FileNotFoundError):
        si.read_export(str(path))


def _imported(store, tmp_path):
    si.import_export(store, _archive(tmp_path, [
        _row("1", "Aug 10, 2026, 6:00:00 AM", distance="10000", moving="3600"),
        _row("2", "Aug 11, 2026, 6:00:00 AM", kind="Ride", distance="20000",
             moving="3600"),
    ]))


def test_imported_rows_never_queue_work_against_garmin(tmp_path):
    """The queues are "what still needs fetching from Garmin". An imported row in
    one of them means the next sync spends a request asking Garmin about an
    activity it has never heard of — 48 of them, in the first real import."""
    store = Store(str(tmp_path / "db.sqlite"))
    _imported(store, tmp_path)

    assert store.activities_missing_metrics() == []
    assert store.activities_missing_weather(["run", "bike"]) == []
    assert store.activities_missing_laps(["run", "bike"]) == []
    assert store.activities_missing_streams(["run", "bike"]) == []
    assert store.activities_missing_zones(["run", "bike"]) == []
    assert store.strength_activities_missing_sets() == []
    store.close()


def test_imported_history_does_not_move_the_sync_watermark(tmp_path):
    """`latest_activity_start` decides how far back the next sync fetches, and
    `earliest_activity_date` anchors the four-week deload block. Both are
    statements about Garmin's record."""
    store = Store(str(tmp_path / "db.sqlite"))
    store.upsert_activities([{
        "activity_id": "garmin-1", "sport": "run", "name": "Run",
        "start_time": "2026-08-20T18:00:00", "start_date": "2026-08-20",
        "duration_s": 1800.0, "ingested_at": "2026-08-20T19:00:00"}])
    _imported(store, tmp_path)

    assert store.latest_activity_start() == datetime(2026, 8, 20, 18, 0, 0)
    assert store.earliest_activity_date().isoformat() == "2026-08-20"
    store.close()
