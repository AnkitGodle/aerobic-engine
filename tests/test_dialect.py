"""SQL is written once in SQLite's flavour and translated for Postgres.

There is no Postgres server in this test environment, so these check the
translation itself: that the generated statements are valid Postgres and that no
SQLite-only syntax survives. The live-server check is a one-liner documented in
DEPLOY.md.
"""

from __future__ import annotations

from core.store import COLUMN_MIGRATIONS, SCHEMA, Store, is_postgres


class FakeStore(Store):
    """Exercises the dialect without opening any connection."""

    def __init__(self, postgres: bool) -> None:  # noqa: D107 - deliberately no super()
        self.postgres = postgres
        self.conn = None
        self.path = None


def test_url_detection():
    assert is_postgres("postgresql://user@host/db")
    assert is_postgres("postgres://user@host/db")
    assert is_postgres("postgresql+psycopg://user@host/db")
    assert not is_postgres("data/aerobic_engine.db")
    assert not is_postgres("/tmp/x.db")


def test_placeholders_are_translated():
    pg = FakeStore(True)
    assert pg.sql("SELECT 1 WHERE a = ? AND b = ?") == "SELECT 1 WHERE a = %s AND b = %s"
    lite = FakeStore(False)
    assert lite.sql("SELECT 1 WHERE a = ?") == "SELECT 1 WHERE a = ?"


def test_autoincrement_becomes_a_postgres_identity():
    pg = FakeStore(True)
    out = pg.sql("CREATE TABLE t (id INTEGER PRIMARY KEY AUTOINCREMENT, x TEXT)")
    assert "BIGSERIAL PRIMARY KEY" in out
    assert "AUTOINCREMENT" not in out


def test_no_sqlite_only_syntax_survives_translation():
    """The whole schema, translated, must be free of SQLite-isms."""
    pg = FakeStore(True)
    banned = ("AUTOINCREMENT", "INSERT OR REPLACE", "PRAGMA", "?")
    for statement in SCHEMA:
        out = pg.sql(statement)
        for token in banned:
            assert token not in out, f"{token!r} survived in: {out[:120]}"


def test_column_migrations_are_portable():
    pg = FakeStore(True)
    for table, column, decl in COLUMN_MIGRATIONS:
        out = pg.sql(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
        assert "AUTOINCREMENT" not in out
        # Postgres accepts INTEGER/REAL/TEXT and DEFAULT clauses as written.
        assert out.startswith("ALTER TABLE ")


def test_upsert_never_emits_insert_or_replace():
    """`INSERT OR REPLACE` has no Postgres equivalent, so it must not be used."""
    joined = "\n".join(SCHEMA)
    assert "INSERT OR REPLACE" not in joined
    import inspect

    src = inspect.getsource(Store._upsert)
    assert "INSERT OR REPLACE" not in src
    assert "ON CONFLICT" in src


def test_sqlite_remains_the_default_and_untouched():
    lite = FakeStore(False)
    for statement in SCHEMA[:6]:
        assert lite.sql(statement) == statement


def test_unsafe_sql_identifiers_are_refused():
    from core.store import _check_identifier

    for good in ("activities", "daily_wellness", "avg_hr", "_x1"):
        assert _check_identifier(good) == good
    for bad in ("activities; DROP TABLE plans", "a b", "1abc", 'x"', "col--", ""):
        try:
            _check_identifier(bad)
            raise AssertionError(f"{bad!r} should have been refused")
        except ValueError:
            pass


def test_upsert_refuses_columns_that_are_not_in_the_table(tmp_path):
    """Column names cannot be bound as parameters, so they are whitelisted."""
    from core.store import Store

    with Store(str(tmp_path / "guard.db")) as s:
        try:
            s.upsert_activities([{"activity_id": "1", "sport": "run",
                                  "start_time": "2026-01-01T00:00:00",
                                  "start_date": "2026-01-01",
                                  "ingested_at": "2026-01-01T00:00:00",
                                  "evil); DROP TABLE plans; --": 1}])
            raise AssertionError("an unknown column should have been refused")
        except ValueError as exc:
            assert "unknown columns" in str(exc)
        # The table is still there.
        assert "plans" in s.counts()
