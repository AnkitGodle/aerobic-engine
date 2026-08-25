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

import os

REPO_URL = os.getenv("REPO_URL", "https://github.com/AnkitGodle/aerobic-engine")

# Label, env var, brand colour. The order is the order they appear.
LINKS = (
    ("Strava", "STRAVA_PROFILE_URL", "#FC5200"),
    ("Instagram", "INSTAGRAM_PROFILE_URL", "#E1306C"),
    ("GitHub", "GITHUB_PROFILE_URL", "#8C9AA8"),
    ("WhatsApp", "WHATSAPP_PROFILE_URL", "#25D366"),
)


def name(fallback: str = "") -> str:
    """The athlete's name. `ATHLETE_NAME` wins; the sync stores one too."""
    return (os.getenv("ATHLETE_NAME") or fallback or "").strip()


def timezone() -> str:
    """The zone every user-facing date is in. See core/clock.py for why."""
    return os.getenv("LOCAL_TZ", "Asia/Kolkata")


def whatsapp_number() -> str:
    """Digits only, country code first, no plus — the form wa.me wants."""
    return "".join(c for c in os.getenv("WHATSAPP_CONTACT", "") if c.isdigit())


def links(include_repo: bool = True) -> tuple[tuple[str, str, str], ...]:
    """The configured profile links, as (label, url, colour).

    Two conveniences, both because of what a person actually has to hand.
    WhatsApp is built from a phone number when no URL is set, since wa.me is
    only the shape a number goes in. And the GitHub chip falls back to the
    repository when no personal profile is set — one mark, never two, because
    two identical icons side by side is a puzzle rather than a link.
    """
    out = []
    for label, key, colour in LINKS:
        url = (os.getenv(key) or "").strip()
        if not url and label == "WhatsApp" and whatsapp_number():
            url = f"https://wa.me/{whatsapp_number()}"
        if not url and label == "GitHub" and include_repo:
            url = REPO_URL
        if url:
            out.append((label, url, colour))
    return tuple(out)


def configured() -> dict[str, bool]:
    """What is set and what is not, for a setup check that can be printed."""
    return {
        "name": bool(name()),
        "timezone": bool(os.getenv("LOCAL_TZ")),
        "strava": bool(os.getenv("STRAVA_PROFILE_URL")),
        "instagram": bool(os.getenv("INSTAGRAM_PROFILE_URL")),
        "github": bool(os.getenv("GITHUB_PROFILE_URL")),
        "whatsapp": bool(whatsapp_number() or os.getenv("WHATSAPP_PROFILE_URL")),
    }
