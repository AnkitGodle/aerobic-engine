"""Garmin payload normalisation. No network: these are pure functions over the
shapes a real account actually returned."""

from __future__ import annotations

from core.garmin_client import as_iso_day


def test_epoch_milliseconds_become_a_real_day():
    """Personal records date themselves with `prStartTimeLocal`, which on this
    account is epoch milliseconds. Stored raw it displayed as 16 December 1787,
    because date.fromisoformat reads the first eight digits as a date rather than
    rejecting them."""
    assert as_iso_day("1787121654000") == "2026-08-19"


def test_epoch_seconds_are_also_understood():
    assert as_iso_day("1787121654") == "2026-08-19"


def test_an_iso_timestamp_is_left_alone():
    assert as_iso_day("2026-08-19T06:00:00") == "2026-08-19T06:00:00"


def test_missing_dates_stay_missing():
    assert as_iso_day(None) is None
    assert as_iso_day("") is None


def test_an_unparseable_number_is_returned_rather_than_guessed_at():
    assert as_iso_day("99999999999999") == "99999999999999"
