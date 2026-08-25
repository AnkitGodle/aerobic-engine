#!/usr/bin/env python3
"""Set this up as yours, in one pass.

    python scripts/setup.py            # ask, then write .env
    python scripts/setup.py --check    # say what is configured, change nothing
    python scripts/setup.py --show     # print the .env it would write

Everything personal in this app — the name in the sidebar, the profile links, the
timezone, the Garmin login, the AI key, the phone number a daily message goes to
— is read from `.env`. This asks for those, one question at a time, and writes
the file. Nothing is required: press enter and that feature is simply off.

Existing values are shown as the default and kept if you press enter, so running
it again to add one thing does not wipe the rest. The file is written with a
backup beside it the first time it changes.

What it does not do: log in to Garmin, mint tokens, or create a database. Those
are `scripts/fetch.py` and `DATABASE_URL`, and they come after this — the last
thing printed here is the list of what to run next.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import dotenv_values, load_dotenv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / ".env"

# key, prompt, hint. A blank answer leaves the key out, which is what "off"
# means everywhere in this app.
QUESTIONS: tuple[tuple[str, str, str], ...] = (
    ("ATHLETE_NAME", "Your name (shown in the sidebar)", ""),
    ("LOCAL_TZ", "Your timezone", "e.g. Asia/Kolkata, Europe/London, "
                                  "America/New_York"),
    ("GARMIN_EMAIL", "Garmin Connect email", "only the sync uses it"),
    ("GARMIN_PASSWORD", "Garmin password",
     "needed once to mint a session; can be left blank and typed at the prompt"),
    ("DATABASE_URL", "Postgres URL", "blank = a local SQLite file, which is "
                                     "fine until you host the dashboard"),
    ("AI_BACKEND", "AI provider chain", "gemini,groq — or none to turn the AI off"),
    ("GEMINI_API_KEY", "Gemini API key", "free at aistudio.google.com/apikey"),
    ("GROQ_API_KEY", "Groq API key", "optional second provider, console.groq.com"),
    ("STRAVA_PROFILE_URL", "Your Strava profile URL", "a link out, nothing is read"),
    ("INSTAGRAM_PROFILE_URL", "Your Instagram URL", ""),
    ("GITHUB_PROFILE_URL", "Your GitHub URL", "blank = link to this project"),
    ("WHATSAPP_CONTACT", "Your WhatsApp number",
     "digits only with country code, e.g. 919940173970"),
    ("NOTIFY_BACKEND", "Daily message channel",
     "whatsapp | telegram | callmebot | none"),
)

# Shown but never echoed back as a default, because a terminal history is not
# the place for a key.
SECRET = ("GARMIN_PASSWORD", "GEMINI_API_KEY", "GROQ_API_KEY", "DATABASE_URL",
          "WHATSAPP_TOKEN", "TELEGRAM_TOKEN", "CALLMEBOT_KEY")

NEXT_STEPS = """
Next, in this order:

  1. pip install -r requirements.txt
  2. python scripts/fetch.py --days 400        pull your history from Garmin
  3. streamlit run app/streamlit_app.py        open the dashboard
  4. python scripts/set_pin.py                 a PIN before you host it anywhere
  5. python scripts/notify.py --dry-run        check the daily message reads right

Hosting it: DEPLOY.md. Messages back and forth on WhatsApp:
scripts/whatsapp_webhook.py, whose docstring is the whole set-up.
"""


def ask(key: str, prompt: str, hint: str, current: str) -> str:
    """One question. Enter keeps what is there.

    A secret that is already set is reported as "already set" and never printed,
    not even partly: the first and last few characters of a short password are
    most of it, and this runs in a terminal that keeps history.
    """
    shown = ""
    if current:
        shown = f" [{'already set' if key in SECRET else current}]"
    if hint:
        print(f"  · {hint}")
    answer = input(f"{prompt}{shown}: ").strip()
    return answer or current


def existing() -> dict[str, str]:
    return {k: v for k, v in (dotenv_values(ENV) or {}).items() if v is not None}


def render(values: dict[str, str], previous: str = "") -> str:
    """The new .env: every line we did not ask about, kept as it was."""
    asked = {k for k, _, _ in QUESTIONS}
    kept = []
    for line in (previous or "").splitlines():
        stripped = line.strip()
        key = stripped.split("=", 1)[0].strip() if "=" in stripped else ""
        if key in asked:
            continue                     # replaced below, in a known order
        kept.append(line)
    body = ["# Written by scripts/setup.py. Safe to edit by hand.",
            "# Anything not listed here is off; see .env.example for the rest.",
            ""]
    for key, _, _ in QUESTIONS:
        value = values.get(key, "")
        body.append(f"{key}={value}" if value else f"# {key}=")
    tail = "\n".join(l for l in kept if l.strip()).strip()
    if tail:
        body += ["", "# --- kept from your previous .env " + "-" * 40, tail]
    return "\n".join(body).rstrip() + "\n"


def check() -> int:
    """What is on and what is off, without touching anything."""
    load_dotenv(ENV)
    from core import ai, notify, profile          # noqa: PLC0415 - after dotenv

    rows = [
        ("Name", profile.name() or "not set — the sidebar will say Athlete"),
        ("Timezone", profile.timezone()),
        ("Garmin login", "set" if os.getenv("GARMIN_EMAIL") else "missing — the "
                                                                "sync cannot run"),
        ("Database", "Postgres" if os.getenv("DATABASE_URL")
         else f"local file ({os.getenv('AEROBIC_ENGINE_DB', 'data/aerobic_engine.db')})"),
        ("AI", os.getenv("AI_BACKEND", ai.DEFAULT_CHAIN) if ai.available()
         else "off — the planner will use its rules only"),
        ("Daily message", notify.configured()),
        ("Write PIN", "set" if os.getenv("REFRESH_PIN_HASH") else "not set"),
        ("Read PIN", "set" if os.getenv("READ_PIN_HASH")
         else "not set — anyone with the link can read the dashboard"),
    ]
    width = max(len(label) for label, _ in rows)
    print()
    for label, value in rows:
        print(f"  {label:<{width}}  {value}")
    links = profile.links()
    print(f"  {'Links':<{width}}  "
          + (", ".join(label for label, _, _ in links) if links else "none"))
    print()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true",
                        help="report what is configured and exit")
    parser.add_argument("--show", action="store_true",
                        help="print the file instead of writing it")
    args = parser.parse_args()

    if args.check:
        return check()

    print(__doc__.split("\n\n")[1].strip())
    print("\nPress enter to keep what is shown in brackets, or to skip.\n")

    current = existing()
    values = dict(current)
    try:
        for key, prompt, hint in QUESTIONS:
            values[key] = ask(key, prompt, hint, current.get(key, ""))
            print()
    except (EOFError, KeyboardInterrupt):
        print("\nStopped. Nothing was written.")
        return 1

    text = render(values, ENV.read_text(encoding="utf-8") if ENV.exists() else "")
    if args.show:
        print(text)
        return 0

    if ENV.exists():
        backup = ENV.with_suffix(".env.backup")
        shutil.copy2(ENV, backup)
        print(f"Your previous .env was copied to {backup.name}.")
    ENV.write_text(text, encoding="utf-8")
    print(f"Written to {ENV}.")
    print(NEXT_STEPS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
