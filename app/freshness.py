"""Drop modules a hosted container kept from a previous deploy.

The failure this exists for, seen three times on Streamlit Community Cloud:

    ImportError
    File "/mount/src/aerobic-engine/app/streamlit_app.py", line 34
        from core.analysis import (

...where the function being imported plainly exists in the checked-out source.
The container had pulled the new revision and re-executed the app script, but
`core.analysis` was still the object it imported from the *previous* revision —
Python only executes a module once per process, and `sys.modules` survived. Any
commit that adds a name to a `core` module and imports it at the top of the app
therefore takes the whole app down until someone reboots it by hand.

So before the first `core` import, this compares each already-imported module
against the file it came from and forgets the ones whose source has changed
since. A cold start has nothing to forget and pays fifteen `stat` calls; a
re-executed script after a deploy re-imports from disk and comes up.

Deliberately stdlib-only and free of app imports: it has to run before anything
it might have to purge.
"""

from __future__ import annotations

import os
import sys
from typing import Callable, Iterable

# The packages whose modules are ours and safe to re-execute. Third-party
# modules are never touched: reloading them is how you get two incompatible
# copies of a class hierarchy in one process.
OWNED = ("core", "app")

# Set on each module the first time it is seen, so a later run can tell whether
# the file has changed underneath it. Lives on the module object rather than in a
# dict here, because this module is itself re-executed on a rerun while the
# module objects in sys.modules are not.
STAMP = "_aerobic_source_mtime"


def _mtime(module: object) -> float | None:
    path = getattr(module, "__file__", None)
    if not path or not path.endswith(".py"):
        return None
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def owned_modules(prefixes: Iterable[str] = OWNED) -> list[str]:
    """Loaded module names belonging to this app, longest first.

    Longest first so `core.analysis` is dropped before `core`, which keeps the
    parent package from being re-imported while a child of it is still stale.
    """
    wanted = tuple(prefixes)
    names = [
        name for name in sys.modules
        if any(name == p or name.startswith(f"{p}.") for p in wanted)
    ]
    return sorted(names, key=len, reverse=True)


def stamp_modules(prefixes: Iterable[str] = OWNED) -> int:
    """Record the source mtime of every loaded app module. Returns how many.

    Called immediately after the app's own imports, which is the only moment the
    stamp is certainly honest: the module and the file agree because the file is
    what was just executed. Left to `purge_stale_modules` alone, a container that
    started cold and then sat idle through a deploy would take its first stamp
    from the *new* file and see nothing to purge — the exact run that needed it.
    """
    stamped = 0
    for name in owned_modules(prefixes):
        module = sys.modules.get(name)
        now = _mtime(module) if module is not None else None
        if now is None:
            continue
        try:
            setattr(module, STAMP, now)
        except (AttributeError, TypeError):  # pragma: no cover - exotic module
            continue
        stamped += 1
    return stamped


def purge_stale_modules(
    prefixes: Iterable[str] = OWNED,
    log: Callable[[str], None] | None = None,
) -> list[str]:
    """Forget every app module whose source file changed after it was imported.

    Returns the names dropped, so the caller can say so in the logs. Nothing is
    reloaded here: the import statements that follow do that, which keeps this
    function free of any opinion about import order.

    All-or-nothing on purpose. Dropping only the changed module would leave
    modules that had imported names *from* it still bound to the old objects,
    and a half-updated `core` is harder to reason about than a fully reloaded
    one.
    """
    stale: list[str] = []
    names = owned_modules(prefixes)
    for name in names:
        module = sys.modules.get(name)
        if module is None:
            continue
        now = _mtime(module)
        if now is None:
            continue
        seen = getattr(module, STAMP, None)
        if seen is None:
            # First sighting. Record it and trust it — the module was imported
            # from the file as it is now.
            try:
                setattr(module, STAMP, now)
            except (AttributeError, TypeError):  # pragma: no cover - exotic module
                pass
        elif now > seen:
            stale.append(name)

    if not stale:
        return []

    for name in names:
        sys.modules.pop(name, None)
    if log:
        log(f"Reloading {len(names)} app modules: "
            f"{', '.join(sorted(stale))} changed on disk since they were imported")
    return sorted(stale)
