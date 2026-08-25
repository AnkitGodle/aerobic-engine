"""Whose dashboard this is, and where that comes from.

The bug behind these: `.env` is gitignored, so a hosted deploy — which is a git
pull — had none of it, and the sidebar lost its name and every link. profile.toml
is committed for that reason, and the environment still overrides it.
"""

from __future__ import annotations

import tomllib

import pytest

from core import clock, profile


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """No inherited settings: each test says what it wants set."""
    for key in ("ATHLETE_NAME", "LOCAL_TZ", "WHATSAPP_CONTACT", "REPO_URL",
                "STRAVA_PROFILE_URL", "INSTAGRAM_PROFILE_URL",
                "GITHUB_PROFILE_URL", "WHATSAPP_PROFILE_URL"):
        monkeypatch.delenv(key, raising=False)
    yield


def test_the_committed_file_carries_the_basics():
    """What a hosted deploy has: the file, and no .env."""
    assert profile.name()
    assert profile.timezone()
    assert profile.links()


def test_the_file_is_valid_toml_and_says_what_it_is():
    with profile.PROFILE_FILE.open("rb") as handle:
        parsed = tomllib.load(handle)
    assert set(parsed) >= {"name", "timezone", "links"}
    assert isinstance(parsed["links"], dict)


def test_the_environment_wins(monkeypatch):
    monkeypatch.setenv("ATHLETE_NAME", "Someone Else")
    monkeypatch.setenv("STRAVA_PROFILE_URL", "https://strava.com/athletes/9")
    assert profile.name() == "Someone Else"
    assert ("Strava", "https://strava.com/athletes/9", "#FC5200") in profile.links()


def test_a_blank_environment_variable_does_not_erase_the_file(monkeypatch):
    """An empty value in a host's secrets panel is a common accident, and it
    should mean "not set here" rather than "delete what the file said"."""
    monkeypatch.setenv("ATHLETE_NAME", "")
    assert profile.name() == profile.setting("NOTHING_SET", "name")
    assert profile.name()


def test_the_garmin_name_is_only_a_fallback(monkeypatch):
    monkeypatch.setenv("ATHLETE_NAME", "Configured")
    assert profile.name("From Garmin") == "Configured"
    monkeypatch.setattr(profile, "_FILE", {})
    monkeypatch.setenv("ATHLETE_NAME", "")
    assert profile.name("From Garmin") == "From Garmin"


def test_whatsapp_is_built_from_the_number(monkeypatch):
    monkeypatch.setenv("WHATSAPP_CONTACT", "+91 99401 73970")
    found = dict((label, url) for label, url, _ in profile.links())
    assert found["WhatsApp"] == "https://wa.me/919940173970"


def test_an_explicit_whatsapp_url_wins_over_the_number(monkeypatch):
    monkeypatch.setenv("WHATSAPP_CONTACT", "919940173970")
    monkeypatch.setenv("WHATSAPP_PROFILE_URL", "https://wa.me/1555")
    found = dict((label, url) for label, url, _ in profile.links())
    assert found["WhatsApp"] == "https://wa.me/1555"


def test_nothing_configured_means_no_chips_not_dead_icons(monkeypatch):
    monkeypatch.setattr(profile, "_FILE", {})
    links = profile.links(include_repo=False)
    assert links == ()


def test_the_repo_stands_in_for_a_missing_github_profile(monkeypatch):
    monkeypatch.setattr(profile, "_FILE", {})
    links = dict((label, url) for label, url, _ in profile.links())
    assert links == {"GitHub": profile.REPO_URL}


def test_there_is_never_more_than_one_github_chip():
    """Two identical marks side by side is a puzzle, not a pair of links."""
    labels = [label for label, _, _ in profile.links()]
    assert labels.count("GitHub") == 1


def test_a_missing_file_is_not_an_error(monkeypatch, tmp_path):
    monkeypatch.setattr(profile, "PROFILE_FILE", tmp_path / "nope.toml")
    monkeypatch.setattr(profile, "_FILE", None)
    assert profile.file_values(reload=True) == {}
    assert profile.timezone() == "Asia/Kolkata"


def test_a_broken_file_is_not_an_error(monkeypatch, tmp_path):
    bad = tmp_path / "profile.toml"
    bad.write_text("this is not = = toml", encoding="utf-8")
    monkeypatch.setattr(profile, "PROFILE_FILE", bad)
    monkeypatch.setattr(profile, "_FILE", None)
    assert profile.file_values(reload=True) == {}


def test_the_clock_uses_the_configured_zone(monkeypatch):
    monkeypatch.setenv("LOCAL_TZ", "Europe/London")
    assert str(clock.zone()) == "Europe/London"


def test_the_clock_falls_back_to_the_file(monkeypatch):
    assert str(clock.zone()) == profile.timezone()


def test_a_nonsense_zone_does_not_stop_the_app(monkeypatch):
    monkeypatch.setenv("LOCAL_TZ", "Mars/Olympus_Mons")
    assert str(clock.zone()) == "Asia/Kolkata"
    assert clock.today() is not None
