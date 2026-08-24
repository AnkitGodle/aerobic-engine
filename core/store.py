"""SQLite persistence: schema, migrations, and typed-ish read/write helpers.

Single-file DB, single user. Every write is idempotent (upsert on natural key)
so re-running a sync never duplicates rows. Swapping this for Postgres later
means reimplementing this module only — nothing above it imports sqlite3.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

def default_db() -> str:
    """Where to read and write, resolved on call rather than at import.

    Import-time capture was a real split-brain bug: scripts/fetch.py imports
    this module before it calls load_dotenv(), so DATABASE_URL was still unset
    when the value was computed and the CLI quietly synced into the local SQLite
    file while the dashboard read Neon. Nothing errored; the data just went to
    two different places.

    DATABASE_URL wins over AEROBIC_ENGINE_DB so a hosted deployment cannot be
    sent back to an ephemeral local file by a stray variable.
    """
    return (os.getenv("DATABASE_URL")
            or os.getenv("AEROBIC_ENGINE_DB")
            or "data/aerobic_engine.db")

PG_PREFIXES = ("postgres://", "postgresql://", "postgresql+psycopg://")


_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _check_identifier(name: str) -> str:
    """Guard for table/column names that must be interpolated, not bound."""
    if not _IDENT.match(str(name)):
        raise ValueError(f"unsafe SQL identifier: {name!r}")
    return str(name)


def is_postgres(target: str) -> bool:
    return str(target).startswith(PG_PREFIXES)


_FINGERPRINT_KEY = "schema_fingerprint"
# Which targets this process has already migrated, so repeated short-lived
# connections in one Streamlit run do not each re-check.
_MIGRATED: dict[str, str] = {}

SCHEMA: list[str] = [
    # --- v1 -------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS activities (
        activity_id      TEXT PRIMARY KEY,
        sport            TEXT NOT NULL,          -- swim | bike | run | strength | other
        garmin_type      TEXT,
        name             TEXT,
        start_time       TEXT NOT NULL,          -- ISO local time
        start_date       TEXT NOT NULL,          -- YYYY-MM-DD
        duration_s       REAL,
        moving_s         REAL,
        distance_m       REAL,
        avg_hr           REAL,
        max_hr           REAL,
        avg_speed_mps    REAL,
        avg_power_w      REAL,
        norm_power_w     REAL,
        elevation_gain_m REAL,
        calories         REAL,
        training_load    REAL,
        aerobic_te       REAL,
        anaerobic_te     REAL,
        rpe              REAL,
        pool_length_m    REAL,
        raw_json         TEXT,
        ingested_at      TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_activities_start ON activities(start_time)",
    "CREATE INDEX IF NOT EXISTS ix_activities_sport ON activities(sport, start_time)",
    """
    CREATE TABLE IF NOT EXISTS activity_metrics (
        activity_id    TEXT PRIMARY KEY REFERENCES activities(activity_id),
        ef             REAL,
        ef_metric      TEXT,
        ef_first_half  REAL,
        ef_second_half REAL,
        decoupling_pct REAL,
        is_steady      INTEGER,
        steady_reason  TEXT,
        hr_samples     INTEGER,
        computed_at    TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS hr_streams (
        activity_id TEXT NOT NULL REFERENCES activities(activity_id),
        t_s         REAL NOT NULL,
        hr          REAL,
        speed_mps   REAL,
        power_w     REAL,
        PRIMARY KEY (activity_id, t_s)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS daily_wellness (
        day                    TEXT PRIMARY KEY,   -- YYYY-MM-DD
        resting_hr             REAL,
        hrv_last_night         REAL,
        hrv_7d_avg             REAL,
        hrv_status             TEXT,
        vo2max_run             REAL,
        vo2max_bike            REAL,
        training_readiness     REAL,
        readiness_level        TEXT,
        training_status        TEXT,
        acute_load             REAL,
        chronic_load           REAL,
        load_ratio             REAL,
        sleep_score            REAL,
        body_battery_high      REAL,
        body_battery_low       REAL,
        raw_json               TEXT,
        ingested_at            TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS race_predictions (
        day      TEXT PRIMARY KEY,
        time_5k  REAL,
        time_10k REAL,
        time_half REAL,
        time_marathon REAL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS strength_log (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        day         TEXT NOT NULL,
        activity_id TEXT,
        exercise_id TEXT NOT NULL,
        sets        INTEGER,
        reps        INTEGER,
        load_kg     REAL,
        hold_s      INTEGER,
        rpe         REAL,
        clean       INTEGER DEFAULT 1,   -- completed all prescribed work
        pain        INTEGER DEFAULT 0,
        notes       TEXT,
        created_at  TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_strength_day ON strength_log(day)",
    """
    CREATE TABLE IF NOT EXISTS checkins (
        id                 INTEGER PRIMARY KEY AUTOINCREMENT,
        day                TEXT NOT NULL,
        sleep              INTEGER,
        soreness           INTEGER,
        motivation         INTEGER,
        time_available_min INTEGER,
        notes              TEXT,
        created_at         TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_checkins_day ON checkins(day)",
    """
    CREATE TABLE IF NOT EXISTS plans (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        week_start  TEXT NOT NULL,
        created_at  TEXT NOT NULL,
        source      TEXT NOT NULL,      -- rules | ai | ai_repaired
        payload_json TEXT,
        plan_json   TEXT NOT NULL,
        flags_json  TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_plans_week ON plans(week_start, created_at)",
    """
    CREATE TABLE IF NOT EXISTS activity_zones (
        activity_id   TEXT NOT NULL REFERENCES activities(activity_id),
        zone_number   INTEGER NOT NULL,
        secs_in_zone  REAL,
        zone_low_bpm  REAL,
        PRIMARY KEY (activity_id, zone_number)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS exercise_sets (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        activity_id     TEXT NOT NULL,
        set_index       INTEGER NOT NULL,
        garmin_category TEXT,
        garmin_name     TEXT,
        exercise_id     TEXT,          -- mapped into our library, NULL if unmapped
        reps            REAL,
        duration_s      REAL,
        load_kg         REAL,
        UNIQUE (activity_id, set_index)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS weekly_targets (
        sport        TEXT PRIMARY KEY,
        sessions     INTEGER,
        minutes      REAL,
        distance_km  REAL,
        updated_at   TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ai_notes (
        key          TEXT PRIMARY KEY,   -- page name, or chart identifier
        kind         TEXT,               -- page | chart
        text         TEXT NOT NULL,
        model        TEXT,
        generated_at TEXT NOT NULL
    )
    """,
    # Heat and humidity raise heart rate at a given pace, which is exactly the
    # signal the efficiency chart is built on. Without this, a humid week reads
    # as lost fitness.
    """
    CREATE TABLE IF NOT EXISTS activity_weather (
        activity_id   TEXT PRIMARY KEY REFERENCES activities(activity_id),
        temp_c        REAL,
        apparent_c    REAL,
        dew_point_c   REAL,
        humidity_pct  REAL,
        wind_kph      REAL,
        condition     TEXT,
        fetched_at    TEXT NOT NULL
    )
    """,
    # A mirror of core/strength.py, refreshed on every sync, so the library is
    # queryable and visible in the data. Deliberately a mirror and not the
    # source: the allowlist the AI is validated against stays in code, because a
    # table the app can write is a table that could grow an exercise nobody
    # vetted.
    """
    CREATE TABLE IF NOT EXISTS exercise_library (
        exercise_id TEXT PRIMARY KEY,
        name        TEXT NOT NULL,
        kind        TEXT,
        focus       TEXT,
        target      TEXT,
        sets        INTEGER,
        rep_low     INTEGER,
        rep_high    INTEGER,
        hold_low    INTEGER,
        hold_high   INTEGER,
        unilateral  INTEGER DEFAULT 0,
        tempo       TEXT,
        cue         TEXT,
        setup       TEXT,
        steps       TEXT,
        mistakes    TEXT,
        why         TEXT,
        load_note   TEXT,
        progress_to TEXT,
        synced_at   TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS personal_records (
        type_id     INTEGER PRIMARY KEY,
        sport       TEXT,
        label       TEXT,
        value       REAL,
        achieved_at TEXT,
        activity_id TEXT,
        fetched_at  TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS sync_state (
        key   TEXT PRIMARY KEY,
        value TEXT
    )
    """,
]


# Columns added after the first release. Applied with ALTER TABLE and ignored if
# already present, so an existing database upgrades in place.
COLUMN_MIGRATIONS: list[tuple[str, str, str]] = [
    ("activities", "is_multisport_parent", "INTEGER DEFAULT 0"),
    ("activities", "parent_activity_id", "TEXT"),
    ("daily_wellness", "sleep_seconds", "REAL"),
    ("daily_wellness", "nap_seconds", "REAL"),
    ("daily_wellness", "battery_charged", "REAL"),
    ("daily_wellness", "battery_drained", "REAL"),
    ("weekly_targets", "enabled", "INTEGER DEFAULT 1"),
    # Daily context that was being left on the table. Stress and respiration are
    # recovery signals in their own right; steps and intensity minutes are the
    # load that happens outside a logged session and still has to be recovered
    # from; weight turns bike watts into watts per kilo.
    ("daily_wellness", "steps", "REAL"),
    ("daily_wellness", "stress_avg", "REAL"),
    ("daily_wellness", "stress_max", "REAL"),
    ("daily_wellness", "intensity_moderate_min", "REAL"),
    ("daily_wellness", "intensity_vigorous_min", "REAL"),
    ("daily_wellness", "floors_climbed", "REAL"),
    ("daily_wellness", "active_calories", "REAL"),
    ("daily_wellness", "respiration_avg", "REAL"),
    ("daily_wellness", "respiration_sleep_avg", "REAL"),
    ("daily_wellness", "spo2_avg", "REAL"),
    ("daily_wellness", "spo2_lowest", "REAL"),
    ("daily_wellness", "weight_kg", "REAL"),
    # Speed, elevation and heart rate only mean anything together: a climb
    # explains a heart-rate spike that would otherwise read as lost fitness.
    ("hr_streams", "altitude_m", "REAL"),
    # Running dynamics: the other half of pace, and the part that says how the
    # distance was covered rather than just how fast.
    ("activities", "avg_cadence", "REAL"),
    ("activities", "max_cadence", "REAL"),
    ("activities", "stride_length_cm", "REAL"),
    ("activities", "ground_contact_ms", "REAL"),
    ("activities", "vertical_osc_cm", "REAL"),
    ("activities", "vertical_ratio", "REAL"),
    ("activities", "steps", "REAL"),
    ("hr_streams", "cadence", "REAL"),
    ("hr_streams", "stride_length_cm", "REAL"),
]


# Derived from the DDL rather than maintained by hand, so it cannot drift: any
# change to a table or a column migration changes this, which is what tells an
# existing database it needs migrating again.
SCHEMA_FINGERPRINT = hashlib.sha256(
    "\x00".join([*SCHEMA, *(f"{t}.{c} {d}" for t, c, d in COLUMN_MIGRATIONS)])
    .encode("utf-8")
).hexdigest()[:32]


class Store:
    """Thin SQLite wrapper. Use as a context manager or call close()."""

    def __init__(self, path: str | Path | None = None) -> None:
        target = str(path) if path is not None else default_db()
        self.postgres = is_postgres(target)
        if self.postgres:
            # A hosted deployment needs storage that survives a restart: a free
            # host's disk does not, and losing the database would mean re-pulling
            # months of Garmin history — exactly the traffic that gets an account
            # flagged. Managed Postgres (Neon, Supabase) is the cheap fix.
            import psycopg
            from psycopg.rows import dict_row

            self.url = target.replace("postgresql+psycopg://", "postgresql://")
            self.conn = psycopg.connect(self.url, row_factory=dict_row,
                                        autocommit=False)
            # Nothing here should ever run for a minute, and nothing should wait
            # on a lock for more than a few seconds. Without these a single
            # blocked statement queues every later reader behind it and the page
            # simply hangs, which is far harder to diagnose than an error.
            with self.conn.cursor() as cur:
                cur.execute("SET statement_timeout = '30s'")
                cur.execute("SET lock_timeout = '5s'")
            self.conn.commit()
            self.path = None
        else:
            self.path = Path(target)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(str(self.path), timeout=30)
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA foreign_keys=ON")
        self.migrate()

    # -- dialect -------------------------------------------------------
    def sql(self, statement: str) -> str:
        """Translate the SQLite-flavoured SQL in this module for Postgres.

        All SQL here is written once, with `?` placeholders and SQLite DDL, and
        rewritten on the way out. Keeping one dialect in the source and one
        translation point is far less error-prone than maintaining two copies.
        """
        if not self.postgres:
            return statement
        out = statement.replace(
            "INTEGER PRIMARY KEY AUTOINCREMENT", "BIGSERIAL PRIMARY KEY"
        )
        return out.replace("?", "%s")

    def execute(self, statement: str, params: Sequence[Any] = ()) -> Any:
        cur = self.conn.cursor()
        cur.execute(self.sql(statement), tuple(params))
        return cur

    def executemany(self, statement: str, rows: Sequence[Sequence[Any]]) -> Any:
        cur = self.conn.cursor()
        cur.executemany(self.sql(statement), [tuple(r) for r in rows])
        return cur

    def _columns(self, table: str) -> set[str]:
        if self.postgres:
            return {
                r["column_name"]
                for r in self.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = ?", (table,)
                ).fetchall()
            }
        return {r["name"] for r in self.conn.execute(f"PRAGMA table_info({table})")}

    def insert_returning_id(self, statement: str, params: Sequence[Any]) -> int:
        """INSERT that yields the new row id on both backends."""
        if self.postgres:
            cur = self.execute(statement.rstrip().rstrip(";") + " RETURNING id", params)
            row = cur.fetchone()
            return int(row["id"]) if row else 0
        return int(self.execute(statement, params).lastrowid or 0)

    # -- lifecycle ------------------------------------------------------
    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def tx(self):
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def migrate(self) -> None:
        """Apply schema statements. Idempotent — every statement is IF NOT EXISTS.

        Skipped entirely when the schema already matches. That is not a
        micro-optimisation: the statements are cheap on a local file but each one
        is a network round trip to a managed Postgres, and re-running all 25 on
        every connection cost about 1.8s — paid again on every page interaction,
        because the UI opens a short-lived connection per state read.

        The guard is a fingerprint of the schema itself rather than a hand-kept
        version number, so it cannot fall out of step with the DDL above: change
        a table and the fingerprint changes with it.
        """
        target = self.url if self.postgres else str(self.path)
        if _MIGRATED.get(target) == SCHEMA_FINGERPRINT:
            return
        if self._schema_is_current():
            _MIGRATED[target] = SCHEMA_FINGERPRINT
            return
        try:
            self.execute(
                "CREATE TABLE IF NOT EXISTS schema_meta (version INTEGER NOT NULL)"
            )
            for stmt in SCHEMA:
                self.execute(stmt)
            # One introspection query per table rather than per column: on a
            # remote database the round trips dominate, not the work.
            by_table: dict[str, set[str]] = {}
            for table, column, decl in COLUMN_MIGRATIONS:
                if table not in by_table:
                    by_table[table] = self._columns(table)
                if column not in by_table[table]:
                    self.execute(
                        f"ALTER TABLE {table} ADD COLUMN {column} {decl}"
                    )
                    by_table[table].add(column)
            row = self.execute("SELECT version FROM schema_meta").fetchone()
            if row is None:
                self.execute("INSERT INTO schema_meta(version) VALUES (?)",
                             (len(SCHEMA),))
            else:
                self.execute("UPDATE schema_meta SET version = ?", (len(SCHEMA),))
            self.execute(
                "INSERT INTO sync_state(key, value) VALUES (?,?)"
                " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (_FINGERPRINT_KEY, SCHEMA_FINGERPRINT),
            )
            self.conn.commit()
            _MIGRATED[target] = SCHEMA_FINGERPRINT
        except Exception:
            self.conn.rollback()
            raise

    def _schema_is_current(self) -> bool:
        """One query: has this database already been migrated to this schema?

        A miss is cheap and a false positive is impossible, because the stored
        value is only written after a successful migration of this exact schema.
        """
        try:
            row = self.execute(
                "SELECT value FROM sync_state WHERE key = ?", (_FINGERPRINT_KEY,)
            ).fetchone()
        except Exception:
            # A missing table is the expected miss on a fresh database. Postgres
            # aborts the transaction on any error, so it has to be cleared before
            # the migration itself can run.
            self.conn.rollback()
            return False
        return bool(row) and row["value"] == SCHEMA_FINGERPRINT

    # -- generic helpers ------------------------------------------------
    def query(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        return [dict(r) for r in self.execute(sql, params).fetchall()]

    def _upsert(self, table: str, rows: Iterable[dict[str, Any]], key: str) -> int:
        rows = [r for r in rows if r]
        if not rows:
            return 0
        cols = sorted({k for r in rows for k in r})
        # Column names are interpolated into the statement (they cannot be bound
        # as parameters), so every one is checked against the table's real
        # columns. Today all callers pass fixed keys; this makes a future caller
        # that forwards externally-supplied keys fail loudly instead of building
        # injectable SQL.
        _check_identifier(table)
        known = self._columns(table)
        unknown = [c for c in cols if c not in known or not _IDENT.match(c)]
        if unknown:
            raise ValueError(f"refusing to write unknown columns to {table}: {unknown}")
        placeholders = ",".join("?" for _ in cols)
        updates = ",".join(f"{c}=excluded.{c}" for c in cols if c != key)
        sql = (
            f"INSERT INTO {table} ({','.join(cols)}) VALUES ({placeholders}) "
            f"ON CONFLICT({key}) DO UPDATE SET {updates}"
            if updates
            else f"INSERT INTO {table} ({','.join(cols)}) VALUES ({placeholders}) "
                 f"ON CONFLICT({key}) DO NOTHING"
        )
        with self.tx():
            self.executemany(sql, [[r.get(col) for col in cols] for r in rows])
        return len(rows)

    # -- sync state -----------------------------------------------------
    def get_state(self, key: str, default: str | None = None) -> str | None:
        row = self.execute(
            "SELECT value FROM sync_state WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else default

    def set_state(self, key: str, value: str) -> None:
        with self.tx():
            self.execute(
                "INSERT INTO sync_state(key, value) VALUES (?,?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    # -- activities -----------------------------------------------------
    def upsert_activities(self, rows: Iterable[dict[str, Any]]) -> int:
        return self._upsert("activities", rows, "activity_id")

    def latest_activity_start(self) -> datetime | None:
        row = self.execute("SELECT MAX(start_time) AS m FROM activities").fetchone()
        if not row or not row["m"]:
            return None
        return datetime.fromisoformat(row["m"])

    def earliest_activity_date(self) -> date | None:
        row = self.execute(
            "SELECT MIN(start_date) AS m FROM activities WHERE duration_s > 0"
        ).fetchone()
        if not row or not row["m"]:
            return None
        return datetime.fromisoformat(str(row["m"])[:10]).date()

    def known_activity_ids(self) -> set[str]:
        return {
            r["activity_id"]
            for r in self.execute("SELECT activity_id FROM activities")
        }

    def activities(
        self,
        since: date | None = None,
        sport: str | None = None,
        include_parents: bool = False,
    ) -> list[dict[str, Any]]:
        """Stored activities, newest last.

        Multisport parents are excluded by default: their legs are stored as
        separate activities, so counting both would double the volume and hand
        the analysis a session with no heart rate.
        """
        sql = (
            "SELECT a.*, m.ef, m.ef_metric, m.decoupling_pct, m.is_steady, "
            "m.steady_reason, m.ef_first_half, m.ef_second_half "
            "FROM activities a LEFT JOIN activity_metrics m USING (activity_id) WHERE 1=1"
        )
        params: list[Any] = []
        if not include_parents:
            sql += " AND COALESCE(a.is_multisport_parent, 0) = 0"
        if since:
            sql += " AND a.start_date >= ?"
            params.append(since.isoformat())
        if sport:
            sql += " AND a.sport = ?"
            params.append(sport)
        return self.query(sql + " ORDER BY a.start_time", params)

    def multisport_parents(self, since: date | None = None) -> list[dict[str, Any]]:
        """Parents whose child legs may still need fetching."""
        sql = "SELECT activity_id, start_date, name FROM activities WHERE is_multisport_parent = 1"
        params: list[Any] = []
        if since:
            sql += " AND start_date >= ?"
            params.append(since.isoformat())
        return self.query(sql + " ORDER BY start_date", params)

    def has_children(self, parent_id: str) -> bool:
        row = self.execute(
            "SELECT 1 FROM activities WHERE parent_activity_id = ? LIMIT 1", (parent_id,)
        ).fetchone()
        return row is not None

    def activities_missing_metrics(self) -> list[str]:
        return [
            r["activity_id"]
            for r in self.execute(
                "SELECT activity_id FROM activities "
                "WHERE activity_id NOT IN (SELECT activity_id FROM activity_metrics)"
            )
        ]

    def upsert_metrics(self, rows: Iterable[dict[str, Any]]) -> int:
        return self._upsert("activity_metrics", rows, "activity_id")

    # -- hr streams -----------------------------------------------------
    def replace_stream(self, activity_id: str, samples: Sequence[dict[str, Any]]) -> int:
        with self.tx():
            self.execute("DELETE FROM hr_streams WHERE activity_id = ?", (activity_id,))
            self.executemany(
                # ON CONFLICT rather than SQLite's INSERT OR REPLACE: the
                # dialect translation turns the latter into a bare INSERT, so a
                # repeated t_s in one downsampled stream would raise a unique
                # violation on Postgres but pass silently on SQLite.
                "INSERT INTO hr_streams"
                "(activity_id, t_s, hr, speed_mps, power_w, altitude_m,"
                " cadence, stride_length_cm)"
                " VALUES (?,?,?,?,?,?,?,?) "
                "ON CONFLICT(activity_id, t_s) DO UPDATE SET"
                " hr=excluded.hr, speed_mps=excluded.speed_mps,"
                " power_w=excluded.power_w, altitude_m=excluded.altitude_m,"
                " cadence=excluded.cadence,"
                " stride_length_cm=excluded.stride_length_cm",
                [
                    (
                        activity_id,
                        s.get("t_s"),
                        s.get("hr"),
                        s.get("speed_mps"),
                        s.get("power_w"),
                        s.get("altitude_m"),
                        s.get("cadence"),
                        s.get("stride_length_cm"),
                    )
                    for s in samples
                    if s.get("t_s") is not None
                ],
            )
        return len(samples)

    def stream(self, activity_id: str) -> list[dict[str, Any]]:
        return self.query(
            "SELECT t_s, hr, speed_mps, power_w, altitude_m, cadence,"
            " stride_length_cm FROM hr_streams "
            "WHERE activity_id = ? ORDER BY t_s",
            (activity_id,),
        )

    def activities_missing_streams(self, sports: Sequence[str]) -> list[dict[str, Any]]:
        """Activities that should have an HR stream stored but don't.

        Needed because a sync run with --no-streams would otherwise leave those
        activities without one forever: the fetcher only looks at newly seen ids.
        """
        marks = ",".join("?" for _ in sports)
        return self.query(
            f"SELECT activity_id, sport, start_date FROM activities "
            f"WHERE sport IN ({marks}) AND avg_hr IS NOT NULL AND duration_s > 0 "
            f"AND (activity_id NOT IN (SELECT DISTINCT activity_id FROM hr_streams)"
            # Streams stored before elevation was parsed have no altitude. The
            # condition is deliberately narrow — the activity must report climbing
            # for us to expect a trace — so an indoor session with genuinely no
            # elevation is not re-fetched on every sync forever.
            f"  OR (COALESCE(elevation_gain_m, 0) > 5 AND activity_id NOT IN"
            f"      (SELECT DISTINCT activity_id FROM hr_streams"
            f"       WHERE altitude_m IS NOT NULL))"
            # Same reasoning for cadence: only expect a trace where the activity
            # itself reports one, so a pool swim is not re-fetched forever.
            f"  OR (COALESCE(avg_cadence, 0) > 0 AND activity_id NOT IN"
            f"      (SELECT DISTINCT activity_id FROM hr_streams"
            f"       WHERE cadence IS NOT NULL))) "
            f"ORDER BY start_date DESC",
            list(sports),
        )

    def has_stream(self, activity_id: str) -> bool:
        row = self.execute(
            "SELECT 1 FROM hr_streams WHERE activity_id = ? LIMIT 1", (activity_id,)
        ).fetchone()
        return row is not None

    # -- zones ----------------------------------------------------------
    def upsert_zones(self, rows: Iterable[dict[str, Any]]) -> int:
        rows = [r for r in rows if r]
        if not rows:
            return 0
        with self.tx():
            self.executemany(
                "INSERT INTO activity_zones"
                "(activity_id, zone_number, secs_in_zone, zone_low_bpm) VALUES (?,?,?,?)"
                " ON CONFLICT(activity_id, zone_number) DO UPDATE SET"
                " secs_in_zone=excluded.secs_in_zone, zone_low_bpm=excluded.zone_low_bpm",
                [
                    (r["activity_id"], r["zone_number"], r.get("secs_in_zone"),
                     r.get("zone_low_bpm"))
                    for r in rows
                ],
            )
        return len(rows)

    def zones(self, since: date | None = None) -> list[dict[str, Any]]:
        """Zone rows joined to the sport and date they belong to."""
        sql = (
            "SELECT z.*, a.sport, a.start_date, a.name, a.duration_s "
            "FROM activity_zones z JOIN activities a USING (activity_id) "
            "WHERE COALESCE(a.is_multisport_parent, 0) = 0"
        )
        params: list[Any] = []
        if since:
            sql += " AND a.start_date >= ?"
            params.append(since.isoformat())
        return self.query(sql + " ORDER BY a.start_date, z.zone_number", params)

    def activity_zones(self, activity_id: str) -> list[dict[str, Any]]:
        return self.query(
            "SELECT zone_number, secs_in_zone, zone_low_bpm FROM activity_zones "
            "WHERE activity_id = ? ORDER BY zone_number",
            (activity_id,),
        )

    def activities_missing_zones(self, sports: Sequence[str]) -> list[dict[str, Any]]:
        marks = ",".join("?" for _ in sports)
        return self.query(
            f"SELECT activity_id FROM activities WHERE sport IN ({marks}) "
            f"AND avg_hr IS NOT NULL AND COALESCE(is_multisport_parent, 0) = 0 "
            f"AND activity_id NOT IN (SELECT DISTINCT activity_id FROM activity_zones) "
            f"ORDER BY start_date DESC",
            list(sports),
        )

    # -- exercise sets (watch strength mode) -----------------------------
    def replace_exercise_sets(self, activity_id: str, rows: Sequence[dict[str, Any]]) -> int:
        with self.tx():
            self.execute("DELETE FROM exercise_sets WHERE activity_id = ?", (activity_id,))
            self.executemany(
                "INSERT INTO exercise_sets"
                "(activity_id, set_index, garmin_category, garmin_name, exercise_id,"
                " reps, duration_s, load_kg) VALUES (?,?,?,?,?,?,?,?)",
                [
                    (
                        activity_id, i, r.get("garmin_category"), r.get("garmin_name"),
                        r.get("exercise_id"), r.get("reps"), r.get("duration_s"),
                        r.get("load_kg"),
                    )
                    for i, r in enumerate(rows)
                ],
            )
        return len(rows)

    def exercise_sets(self, activity_id: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM exercise_sets"
        params: list[Any] = []
        if activity_id:
            sql += " WHERE activity_id = ?"
            params.append(activity_id)
        return self.query(sql + " ORDER BY activity_id, set_index", params)

    def strength_activities_missing_sets(self) -> list[dict[str, Any]]:
        return self.query(
            "SELECT activity_id, start_date FROM activities WHERE sport = 'strength' "
            "AND activity_id NOT IN (SELECT DISTINCT activity_id FROM exercise_sets) "
            "ORDER BY start_date DESC"
        )

    def strength_days_logged(self) -> set[str]:
        return {
            r["day"]
            for r in self.execute("SELECT DISTINCT day FROM strength_log")
        }

    # -- wellness -------------------------------------------------------
    def upsert_wellness(self, rows: Iterable[dict[str, Any]]) -> int:
        return self._upsert("daily_wellness", rows, "day")

    def wellness(self, since: date | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM daily_wellness"
        params: list[Any] = []
        if since:
            sql += " WHERE day >= ?"
            params.append(since.isoformat())
        return self.query(sql + " ORDER BY day", params)

    def wellness_days_present(self) -> set[str]:
        return {r["day"] for r in self.execute("SELECT day FROM daily_wellness")}

    def upsert_race_predictions(self, rows: Iterable[dict[str, Any]]) -> int:
        return self._upsert("race_predictions", rows, "day")

    def race_predictions(self, since: date | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM race_predictions"
        params: list[Any] = []
        if since:
            sql += " WHERE day >= ?"
            params.append(since.isoformat())
        return self.query(sql + " ORDER BY day", params)

    # -- strength -------------------------------------------------------
    def log_strength(self, entries: Iterable[dict[str, Any]]) -> int:
        entries = list(entries)
        if not entries:
            return 0
        now = datetime.now().isoformat(timespec="seconds")
        with self.tx():
            self.executemany(
                "INSERT INTO strength_log"
                "(day, activity_id, exercise_id, sets, reps, load_kg, hold_s, rpe,"
                " clean, pain, notes, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        e["day"],
                        e.get("activity_id"),
                        e["exercise_id"],
                        e.get("sets"),
                        e.get("reps"),
                        e.get("load_kg"),
                        e.get("hold_s"),
                        e.get("rpe"),
                        int(e.get("clean", 1)),
                        int(e.get("pain", 0)),
                        e.get("notes", ""),
                        now,
                    )
                    for e in entries
                ],
            )
        return len(entries)

    def strength_log(self, since: date | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM strength_log"
        params: list[Any] = []
        if since:
            sql += " WHERE day >= ?"
            params.append(since.isoformat())
        return self.query(sql + " ORDER BY day, id", params)

    # -- checkins -------------------------------------------------------
    def save_checkin(self, c: dict[str, Any]) -> int:
        with self.tx():
            return self.insert_returning_id(
                "INSERT INTO checkins"
                "(day, sleep, soreness, motivation, time_available_min, notes, created_at)"
                " VALUES (?,?,?,?,?,?,?)",
                (
                    c["day"],
                    c.get("sleep"),
                    c.get("soreness"),
                    c.get("motivation"),
                    c.get("time_available_min"),
                    c.get("notes", ""),
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )

    def checkins(self, since: date | None = None, limit: int = 50) -> list[dict[str, Any]]:
        sql = "SELECT * FROM checkins"
        params: list[Any] = []
        if since:
            sql += " WHERE day >= ?"
            params.append(since.isoformat())
        sql += " ORDER BY day DESC, id DESC LIMIT ?"
        params.append(limit)
        return self.query(sql, params)

    def latest_checkin(self) -> dict[str, Any] | None:
        rows = self.checkins(limit=1)
        return rows[0] if rows else None

    # -- plans ----------------------------------------------------------
    def save_plan(
        self,
        week_start: date,
        plan: dict[str, Any],
        source: str,
        payload: dict[str, Any] | None = None,
    ) -> int:
        with self.tx():
            return self.insert_returning_id(
                "INSERT INTO plans(week_start, created_at, source, payload_json,"
                " plan_json, flags_json) VALUES (?,?,?,?,?,?)",
                (
                    week_start.isoformat(),
                    datetime.now().isoformat(timespec="seconds"),
                    source,
                    json.dumps(payload, default=str) if payload else None,
                    json.dumps(plan, default=str),
                    json.dumps(plan.get("flags", []), default=str),
                ),
            )

    def latest_plan(self, week_start: date | None = None) -> dict[str, Any] | None:
        sql = "SELECT * FROM plans"
        params: list[Any] = []
        if week_start:
            sql += " WHERE week_start = ?"
            params.append(week_start.isoformat())
        rows = self.query(sql + " ORDER BY created_at DESC, id DESC LIMIT 1", params)
        if not rows:
            return None
        row = rows[0]
        row["plan"] = json.loads(row["plan_json"])
        return row

    # -- weekly targets (athlete's own intent) ---------------------------
    def set_targets(self, targets: Iterable[dict[str, Any]]) -> int:
        rows = [t for t in targets if t.get("sport")]
        if not rows:
            return 0
        now = datetime.now().isoformat(timespec="seconds")
        with self.tx():
            self.executemany(
                "INSERT INTO weekly_targets(sport, sessions, minutes, distance_km,"
                " enabled, updated_at) VALUES (?,?,?,?,?,?)"
                " ON CONFLICT(sport) DO UPDATE SET"
                " sessions=excluded.sessions, minutes=excluded.minutes,"
                " distance_km=excluded.distance_km, enabled=excluded.enabled,"
                " updated_at=excluded.updated_at",
                [
                    (t["sport"], t.get("sessions"), t.get("minutes"),
                     t.get("distance_km"), int(t.get("enabled", 1)), now)
                    for t in rows
                ],
            )
        return len(rows)

    def set_sport_enabled(self, enabled: dict[str, bool]) -> int:
        """Flip only the on/off flag, preserving any sessions/minutes already set.

        Separate from set_targets() on purpose: set_targets() writes the whole
        row, so using it here would erase the weekly volume intent behind a sport
        as a side effect of merely toggling that sport off and on again.
        """
        rows = {s: bool(v) for s, v in (enabled or {}).items() if s}
        if not rows:
            return 0
        now = datetime.now().isoformat(timespec="seconds")
        with self.tx():
            self.executemany(
                "INSERT INTO weekly_targets(sport, enabled, updated_at)"
                " VALUES (?,?,?) ON CONFLICT(sport) DO UPDATE SET"
                " enabled=excluded.enabled, updated_at=excluded.updated_at",
                [(sport, int(on), now) for sport, on in rows.items()],
            )
        return len(rows)

    def targets(self) -> dict[str, dict[str, Any]]:
        """Rows the athlete has actually set. A disabled sport is kept, so the
        planner knows to exclude it rather than treating it as unset."""
        return {
            r["sport"]: r
            for r in self.query("SELECT * FROM weekly_targets")
            if (r.get("sessions") or 0) > 0 or (r.get("minutes") or 0) > 0
            or not r.get("enabled", 1)
        }

    def clear_targets(self) -> None:
        with self.tx():
            self.execute("DELETE FROM weekly_targets")

    # -- weather and records --------------------------------------------
    def upsert_weather(self, rows: Iterable[dict[str, Any]]) -> int:
        return self._upsert("activity_weather", rows, "activity_id")

    def weather(self) -> dict[str, dict[str, Any]]:
        """Keyed by activity_id, because it is only ever read alongside one."""
        return {r["activity_id"]: r
                for r in self.query("SELECT * FROM activity_weather")}

    def activities_missing_weather(self, sports: Sequence[str]) -> list[dict[str, Any]]:
        """Outdoor sessions with no weather row yet.

        Pool swims are excluded by the caller: there is no weather indoors, and
        asking would spend a request to learn nothing.
        """
        if not sports:
            return []
        marks = ",".join("?" * len(sports))
        return self.query(
            f"SELECT activity_id, sport, start_date FROM activities"
            f" WHERE sport IN ({marks})"
            f" AND COALESCE(is_multisport_parent, 0) = 0"
            f" AND activity_id NOT IN (SELECT activity_id FROM activity_weather)"
            f" ORDER BY start_time DESC", list(sports))

    def sync_exercise_library(self, rows: Iterable[dict[str, Any]]) -> int:
        return self._upsert("exercise_library", rows, "exercise_id")

    def exercise_library(self) -> list[dict[str, Any]]:
        return self.query("SELECT * FROM exercise_library ORDER BY focus, name")

    def set_personal_records(self, rows: Iterable[dict[str, Any]]) -> int:
        return self._upsert("personal_records", rows, "type_id")

    def personal_records(self) -> list[dict[str, Any]]:
        # A zero is Garmin saying "no record of this kind", not a record of zero.
        return self.query(
            "SELECT * FROM personal_records WHERE value IS NOT NULL AND value > 0"
            " ORDER BY sport, type_id")

    # -- AI notes, written at sync time and read on render ---------------
    def set_notes(self, notes: Iterable[dict[str, Any]]) -> int:
        rows = [n for n in notes if n.get("key") and n.get("text")]
        if not rows:
            return 0
        now = datetime.now().isoformat(timespec="seconds")
        with self.tx():
            self.executemany(
                "INSERT INTO ai_notes(key, kind, text, model, generated_at)"
                " VALUES (?,?,?,?,?) ON CONFLICT(key) DO UPDATE SET"
                " kind=excluded.kind, text=excluded.text, model=excluded.model,"
                " generated_at=excluded.generated_at",
                [(n["key"], n.get("kind", "page"), n["text"], n.get("model"), now)
                 for n in rows],
            )
        return len(rows)

    def notes(self) -> dict[str, dict[str, Any]]:
        return {r["key"]: r for r in self.query("SELECT * FROM ai_notes")}

    # -- misc -----------------------------------------------------------
    COUNTED_TABLES = (
        "activities", "activity_metrics", "hr_streams", "activity_zones",
        "exercise_sets", "daily_wellness", "strength_log", "checkins",
        "plans", "weekly_targets", "ai_notes",
    )

    def counts(self) -> dict[str, int]:
        """Row counts for every table, in a single round trip.

        Eleven separate COUNT(*) queries cost nothing against a local file and
        about a second against a managed Postgres, and this runs on every page
        load. One SELECT with eleven scalar subqueries is the same work for the
        database and a tenth of the latency.
        """
        cols = ", ".join(
            f"(SELECT COUNT(*) FROM {_check_identifier(t)}) AS {_check_identifier(t)}"
            for t in self.COUNTED_TABLES
        )
        # dict(), because sqlite3.Row supports indexing but not .get().
        row = dict(self.execute(f"SELECT {cols}").fetchone() or {})
        return {t: int(row.get(t) or 0) for t in self.COUNTED_TABLES}

    def get_states(self, keys: Sequence[str]) -> dict[str, str | None]:
        """Several state values in one query, for the same reason as counts()."""
        keys = list(keys)
        if not keys:
            return {}
        marks = ",".join("?" * len(keys))
        rows = self.query(
            f"SELECT key, value FROM sync_state WHERE key IN ({marks})", keys)
        found = {r["key"]: r["value"] for r in rows}
        return {k: found.get(k) for k in keys}


def week_start_of(d: date) -> date:
    """Monday of the week containing d."""
    return d - timedelta(days=d.weekday())
