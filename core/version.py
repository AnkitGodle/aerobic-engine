"""What is running, and when it last changed.

A dashboard that updates itself several times a week needs to say which version
you are looking at, or a bug report cannot be placed in time — "the chart was
wrong yesterday" means nothing without knowing what yesterday's code was.

Three parts, each chosen to work without a build step:

  * the version, from the `VERSION` file at the repo root — one line, edited by
    hand or by `scripts/release.py`, and the only thing a person has to bump;
  * the commit, read straight out of `.git` rather than by running git, because
    a hosted container may not have the binary even though it has the checkout;
  * when the code last changed, from the newest source file's timestamp — which
    is the honest answer for a deploy that is a git pull rather than a build.

Everything degrades to "" rather than raising. A missing version is a cosmetic
problem and must never be the reason a page fails to render.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from core import clock

ROOT = Path(__file__).resolve().parents[1]
FALLBACK = "0.0.0"
# Only the app's own source counts. A .pyc or a stray file in the working tree
# is not a change to what is deployed.
WATCHED = ("app", "core", "scripts")


def version() -> str:
    """The version from the VERSION file, or the fallback."""
    try:
        text = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    except OSError:
        return FALLBACK
    return text.splitlines()[0].strip() if text else FALLBACK


def commit(short: bool = True) -> str:
    """The checked-out commit, read from `.git` without running git.

    `.git/HEAD` is either a ref to follow or a detached SHA. Packed refs are
    handled too, because a fresh clone on a hosted host usually has them.
    """
    git = ROOT / ".git"
    try:
        head = (git / "HEAD").read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    sha = ""
    if head.startswith("ref:"):
        ref = head.split(" ", 1)[1].strip()
        try:
            sha = (git / ref).read_text(encoding="utf-8").strip()
        except OSError:
            try:
                for line in (git / "packed-refs").read_text(
                        encoding="utf-8").splitlines():
                    if line.endswith(f" {ref}"):
                        sha = line.split(" ", 1)[0].strip()
                        break
            except OSError:
                sha = ""
    elif len(head) >= 7 and all(c in "0123456789abcdef" for c in head[:7]):
        sha = head
    return sha[:7] if short and sha else sha


def changed_at() -> datetime | None:
    """When the newest source file last changed, in the athlete's timezone."""
    newest = 0.0
    for folder in WATCHED:
        base = ROOT / folder
        if not base.is_dir():
            continue
        for path in base.rglob("*.py"):
            try:
                newest = max(newest, path.stat().st_mtime)
            except OSError:
                continue
    if not newest:
        return None
    return datetime.fromtimestamp(newest, tz=clock.zone())


def stamp() -> str:
    """`26-08, 1:10 am` — the same format the rest of the app uses for a time."""
    when = changed_at()
    if when is None:
        return ""
    return when.strftime("%d-%m, %-I:%M %p").lower()


def describe(with_commit: bool = True) -> str:
    """One line for the sidebar: version, commit, and when it last changed."""
    bits = [f"v{version()}"]
    sha = commit() if with_commit else ""
    if sha:
        bits.append(sha)
    when = stamp()
    if when:
        bits.append(f"updated {when}")
    return " · ".join(bits)


def details() -> dict[str, str]:
    """The same facts, separately, for anything that wants to lay them out."""
    return {
        "version": version(),
        "commit": commit(),
        "changed_at": stamp(),
        "environment": ("Streamlit Cloud" if os.getenv("STREAMLIT_RUNTIME_ENV")
                        or os.getenv("STREAMLIT_SERVER_HEADLESS") == "true"
                        else "local"),
    }
