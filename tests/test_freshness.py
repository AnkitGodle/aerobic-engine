"""The stale-module guard, which exists because a hosted container survives a
deploy and keeps the previous revision's modules in sys.modules."""

from __future__ import annotations

import sys
import textwrap

from app.freshness import STAMP, owned_modules, purge_stale_modules


def _write(pkg_dir, name: str, body: str) -> None:
    (pkg_dir / f"{name}.py").write_text(textwrap.dedent(body))


def test_a_cold_start_purges_nothing(tmp_path, monkeypatch):
    """First sighting of a module is trusted: it was imported from the file as it
    is now. Purging on every run would drop the caches with it."""
    pkg = tmp_path / "fakecore"
    pkg.mkdir()
    _write(pkg, "__init__", "")
    _write(pkg, "thing", "VALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    import fakecore.thing  # noqa: F401 - importing it is the point

    assert purge_stale_modules(prefixes=("fakecore",)) == []
    # The stamp is recorded, so a later change can be spotted.
    assert getattr(sys.modules["fakecore.thing"], STAMP) > 0
    # Twice in a row is still a no-op — this runs on every rerun.
    assert purge_stale_modules(prefixes=("fakecore",)) == []


def test_a_changed_file_drops_the_whole_package(tmp_path, monkeypatch):
    """All-or-nothing: a module that imported names from the changed one would
    otherwise still be bound to the old objects."""
    pkg = tmp_path / "fakecore2"
    pkg.mkdir()
    _write(pkg, "__init__", "")
    _write(pkg, "thing", "VALUE = 1\n")
    _write(pkg, "other", "VALUE = 9\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    import fakecore2.other  # noqa: F401
    import fakecore2.thing

    purge_stale_modules(prefixes=("fakecore2",))
    # A deploy: the file gains a name the running module does not have.
    stamp = getattr(fakecore2.thing, STAMP)
    _write(pkg, "thing", "VALUE = 1\nNEW_NAME = 2\n")
    import os
    os.utime(pkg / "thing.py", (stamp + 10, stamp + 10))

    dropped = purge_stale_modules(prefixes=("fakecore2",))
    assert dropped == ["fakecore2.thing"]
    assert "fakecore2.other" not in sys.modules
    assert "fakecore2" not in sys.modules

    # The import that follows the purge now finds the new name, which is the
    # whole point: no reboot needed.
    import fakecore2.thing as reloaded
    assert reloaded.NEW_NAME == 2


def test_only_our_own_packages_are_touched(tmp_path, monkeypatch):
    """Reloading a third-party module is how one process ends up with two
    incompatible copies of a class hierarchy."""
    names = owned_modules(prefixes=("core", "app"))
    assert names
    assert all(n.split(".")[0] in ("core", "app") for n in names)
    assert not any(n.startswith(("streamlit", "pandas", "plotly")) for n in names)


def test_children_are_dropped_before_their_parent():
    """Otherwise the parent package is re-imported while a stale child of it is
    still in sys.modules."""
    import core.analysis  # noqa: F401 - has to be loaded for the order to matter

    order = owned_modules(prefixes=("core", "app"))
    assert order.index("core.analysis") < order.index("core")


def test_stamping_at_import_time_closes_the_cold_start_gap(tmp_path, monkeypatch):
    """A container that started cold and sat idle through a deploy would take its
    first stamp from the new file and see nothing to purge — on the one run that
    needed it. Stamping straight after the app's own imports is what prevents
    that, so it is stamped here before the file is touched."""
    import os

    from app.freshness import stamp_modules

    pkg = tmp_path / "fakecore3"
    pkg.mkdir()
    _write(pkg, "__init__", "")
    _write(pkg, "thing", "VALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    import fakecore3.thing

    assert stamp_modules(prefixes=("fakecore3",)) == 2   # package and module
    stamp = getattr(fakecore3.thing, STAMP)

    _write(pkg, "thing", "VALUE = 1\nLATER = 3\n")
    os.utime(pkg / "thing.py", (stamp + 10, stamp + 10))

    assert purge_stale_modules(prefixes=("fakecore3",)) == ["fakecore3.thing"]
    import fakecore3.thing as reloaded
    assert reloaded.LATER == 3
