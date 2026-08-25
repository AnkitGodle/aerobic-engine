"""Whose dashboard this is, and how to reach them.

Everything personal in one place, read from the environment, with no personal
defaults in the code. That is the difference between a repo someone else can run
and a repo with my name baked into it: clone this, copy `.env.example` to `.env`,
fill in four lines, and the app is yours — your name in the sidebar, your links
in the footer, your timezone on every date.

A missing value means the thing is not shown, never a placeholder and never
somebody else's profile. `links()` returns only what is configured, so an
unconfigured instance simply has no chips rather than a row of dead icons.

The one exception is the repository, which is the app's own source and the same
for everyone running it — a link to what the page is, not to who is reading it.
"""

from __future__ import annotations

import logging
import os
import tomllib
from pathlib import Path
from typing import Any

log = logging.getLogger("aerobic_engine.profile")

# A committed file, and the reason it exists: `.env` is gitignored, so a hosted
# deploy is a git pull with none of it. Putting the name and the links in secrets
# works and is a fiddly step to forget, and forgetting it makes the sidebar go
# blank — which is exactly the bug that was reported. None of this is secret: it
# is a name and four public profile URLs. Environment variables still win, so a
# fork can override any of it without editing a tracked file.
PROFILE_FILE = Path(__file__).resolve().parents[1] / "profile.toml"

_FILE: dict[str, Any] | None = None


def file_values(reload: bool = False) -> dict[str, Any]:
    """profile.toml, parsed once. An unreadable or missing file is not an error."""
    global _FILE
    if _FILE is None or reload:
        try:
            with PROFILE_FILE.open("rb") as handle:
                _FILE = tomllib.load(handle)
        except FileNotFoundError:
            _FILE = {}
        except (OSError, tomllib.TOMLDecodeError) as exc:
            log.warning("Could not read %s: %s", PROFILE_FILE.name, exc)
            _FILE = {}
    return _FILE


def setting(env_key: str, *path: str, default: str = "") -> str:
    """One value: the environment first, then profile.toml, then the default."""
    from_env = (os.getenv(env_key) or "").strip()
    if from_env:
        return from_env
    node: Any = file_values()
    for step in path:
        if not isinstance(node, dict):
            return default
        node = node.get(step)
    return str(node).strip() if isinstance(node, (str, int)) else default


REPO_URL = setting("REPO_URL", "repo",
                   default="https://github.com/AnkitGodle/aerobic-engine")

# Label, env var, key in profile.toml's [links], brand colour. The order here is
# the order they appear in the sidebar.
LINKS = (
    ("Strava", "STRAVA_PROFILE_URL", "strava", "#FC5200"),
    ("Instagram", "INSTAGRAM_PROFILE_URL", "instagram", "#E1306C"),
    ("GitHub", "GITHUB_PROFILE_URL", "github", "#8C9AA8"),
    ("WhatsApp", "WHATSAPP_PROFILE_URL", "whatsapp", "#25D366"),
)


def name(fallback: str = "") -> str:
    """The athlete's name. The setting wins; the sync stores one too."""
    return setting("ATHLETE_NAME", "name") or (fallback or "").strip()


def timezone() -> str:
    """The zone every user-facing date is in. See core/clock.py for why."""
    return setting("LOCAL_TZ", "timezone", default="Asia/Kolkata")


def whatsapp_number() -> str:
    """Digits only, country code first, no plus — the form wa.me wants."""
    return "".join(c for c in setting("WHATSAPP_CONTACT", "whatsapp_contact")
                   if c.isdigit())


def links(include_repo: bool = True) -> tuple[tuple[str, str, str], ...]:
    """The configured profile links, as (label, url, colour).

    Two conveniences, both because of what a person actually has to hand.
    WhatsApp is built from a phone number when no URL is set, since wa.me is
    only the shape a number goes in. And the GitHub chip falls back to the
    repository when no personal profile is set — one mark, never two, because
    two identical icons side by side is a puzzle rather than a link.
    """
    out = []
    for label, env_key, file_key, colour in LINKS:
        url = setting(env_key, "links", file_key)
        if not url and label == "WhatsApp" and whatsapp_number():
            url = f"https://wa.me/{whatsapp_number()}"
        if not url and label == "GitHub" and include_repo:
            url = REPO_URL
        if url:
            out.append((label, url, colour))
    return tuple(out)


def configured() -> dict[str, bool]:
    """What is set and what is not, for a setup check that can be printed."""
    out = {"name": bool(name()), "timezone": bool(setting("LOCAL_TZ", "timezone"))}
    for label, env_key, file_key, _ in LINKS:
        out[label.lower()] = bool(setting(env_key, "links", file_key))
    out["whatsapp"] = out["whatsapp"] or bool(whatsapp_number())
    return out
