"""Which build is running.

Not cosmetic: a bug report that says "the chart was wrong yesterday" cannot be
placed in time without it, and this app changes most days. So this checks the
parts degrade rather than raise — a version line is never worth a broken page.
"""

from __future__ import annotations

from core import version


def test_the_version_comes_from_the_file():
    assert version.version() == (version.ROOT / "VERSION").read_text().strip()


def test_the_commit_is_read_without_running_git():
    """A hosted container often has the checkout and not the binary."""
    sha = version.commit()
    assert sha == "" or (len(sha) == 7
                         and all(c in "0123456789abcdef" for c in sha))
    full = version.commit(short=False)
    assert full == "" or len(full) == 40


def test_the_description_reads_like_a_version():
    text = version.describe()
    assert text.startswith("v")
    assert version.version() in text


def test_it_says_when_the_code_last_changed():
    stamp = version.stamp()
    assert stamp and ("am" in stamp or "pm" in stamp)
    assert version.changed_at() is not None


def test_the_timestamp_is_in_the_athletes_zone(monkeypatch):
    monkeypatch.setenv("LOCAL_TZ", "Europe/London")
    london = version.changed_at()
    monkeypatch.setenv("LOCAL_TZ", "Asia/Kolkata")
    kolkata = version.changed_at()
    assert london.timestamp() == kolkata.timestamp()      # same moment
    assert london.utcoffset() != kolkata.utcoffset()      # different wall clock


def test_a_missing_version_file_is_not_an_error(monkeypatch, tmp_path):
    monkeypatch.setattr(version, "ROOT", tmp_path)
    assert version.version() == version.FALLBACK
    assert version.describe().startswith("v0.0.0")


def test_a_missing_checkout_is_not_an_error(monkeypatch, tmp_path):
    monkeypatch.setattr(version, "ROOT", tmp_path)
    assert version.commit() == ""


def test_a_detached_head_still_names_a_commit(monkeypatch, tmp_path):
    git = tmp_path / ".git"
    git.mkdir()
    (git / "HEAD").write_text("a" * 40 + "\n", encoding="utf-8")
    monkeypatch.setattr(version, "ROOT", tmp_path)
    assert version.commit() == "aaaaaaa"


def test_a_packed_ref_is_followed(monkeypatch, tmp_path):
    git = tmp_path / ".git"
    git.mkdir()
    (git / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    header = "# pack-refs with: peeled fully-peeled sorted \n"
    (git / "packed-refs").write_text(
        header + ("b" * 40) + " refs/heads/main\n", encoding="utf-8")
    monkeypatch.setattr(version, "ROOT", tmp_path)
    assert version.commit() == "bbbbbbb"


def test_the_details_say_where_it_is_running():
    detail = version.details()
    assert set(detail) == {"version", "commit", "changed_at", "environment"}
    assert detail["environment"] in ("local", "Streamlit Cloud")
