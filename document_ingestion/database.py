"""Stage 4: persist events into SQLite with dates as first-class citizens.

Schema honours the user's two-table design:

    dates(date, event_ids, source_documents)
    events(event_id, date, event_detail, source_documents)

`dates` is intentionally redundant — it caches the per-day rollup so a
timeline query is one cheap ORDER-BY scan and never needs a GROUP BY.
The two tables are kept consistent by `EventStore.add_events`.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

from .event_extractor import Event

logger = logging.getLogger(__name__)


SCHEMA = """
CREATE TABLE IF NOT EXISTS dates (
    date              TEXT PRIMARY KEY,            -- ISO yyyy-mm-dd
    event_ids         TEXT NOT NULL DEFAULT '',    -- comma-separated
    source_documents  TEXT NOT NULL DEFAULT ''     -- comma-separated, deduped
);

CREATE TABLE IF NOT EXISTS events (
    event_id          TEXT PRIMARY KEY,
    date              TEXT NOT NULL,            -- ISO yyyy-mm-dd or 'undetermined'
    action            TEXT NOT NULL,            -- event / action
    party             TEXT NOT NULL DEFAULT 'undetermined',
    court             TEXT NOT NULL DEFAULT 'undetermined',
    source_documents  TEXT NOT NULL,
    page_reference    TEXT NOT NULL DEFAULT 'undetermined',
    excerpt           TEXT NOT NULL DEFAULT '',
    other_actors      TEXT NOT NULL DEFAULT '',     -- other named persons, '; '-sep
    -- role-validation gate (entity_resolver):
    actor_id          TEXT NOT NULL DEFAULT '',
    actor_canonical   TEXT NOT NULL DEFAULT '',
    forum_id          TEXT NOT NULL DEFAULT '',
    forum_role        TEXT NOT NULL DEFAULT '',
    role_status       TEXT NOT NULL DEFAULT '',
    canonical_party   TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (date) REFERENCES dates(date)
);

CREATE INDEX IF NOT EXISTS idx_events_date ON events(date);

CREATE TABLE IF NOT EXISTS case_actors (
    actor_key         TEXT PRIMARY KEY,            -- A0xx canonical or X<hash> provisional
    name              TEXT NOT NULL,
    role              TEXT NOT NULL DEFAULT 'undetermined',
    canonical         INTEGER NOT NULL DEFAULT 0,  -- 1 if a curated A0xx actor
    mention           TEXT NOT NULL DEFAULT '',    -- verbatim grounding quote
    source_documents  TEXT NOT NULL DEFAULT '',    -- comma-separated
    mentions          INTEGER NOT NULL DEFAULT 0   -- times seen
);

-- The full role-validated cast of each event: one row per (event, participant).
-- The role gate runs on every participant, not just the primary party.
CREATE TABLE IF NOT EXISTS event_actors (
    event_id          TEXT NOT NULL,
    actor_key         TEXT NOT NULL DEFAULT '',    -- A0xx / X<hash> / '' if unresolved
    name              TEXT NOT NULL,               -- as named in this event
    canonical_name    TEXT NOT NULL DEFAULT '',
    forum_id          TEXT NOT NULL DEFAULT '',
    forum_role        TEXT NOT NULL DEFAULT '',    -- petitioner/respondent in forum
    stated_role       TEXT NOT NULL DEFAULT '',
    role_status       TEXT NOT NULL DEFAULT '',    -- ok/conflict/court/na/unresolved
    is_primary        INTEGER NOT NULL DEFAULT 0,  -- 1 = the event's party field
    PRIMARY KEY (event_id, name)
);

CREATE INDEX IF NOT EXISTS idx_event_actors_event ON event_actors(event_id);
CREATE INDEX IF NOT EXISTS idx_event_actors_actor ON event_actors(actor_key);
"""


class EventStore:
    _GATE_COLUMNS = (
        "actor_id", "actor_canonical", "forum_id",
        "forum_role", "role_status", "canonical_party",
    )

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            self._migrate(conn)

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Add newer columns to a pre-existing events table if missing."""
        have = {r["name"] for r in conn.execute("PRAGMA table_info(events)")}
        for col in ("other_actors", *self._GATE_COLUMNS):
            if col not in have:
                conn.execute(
                    f"ALTER TABLE events ADD COLUMN {col} TEXT NOT NULL DEFAULT ''"
                )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Writes

    def add_events(self, events: Iterable[Event]) -> int:
        events = list(events)
        if not events:
            return 0
        with self._connect() as conn:
            for ev in events:
                self._upsert_date(conn, ev.date, ev.event_id, ev.source_document)
                g = _gate_fields(ev.party, ev.source_document, ev.court)
                conn.execute(
                    "INSERT OR REPLACE INTO events "
                    "(event_id, date, action, party, court, "
                    "source_documents, page_reference, excerpt, other_actors, "
                    "actor_id, actor_canonical, forum_id, forum_role, "
                    "role_status, canonical_party) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        ev.event_id, ev.date, ev.action, ev.party, ev.court,
                        ev.source_document, ev.page_reference, ev.excerpt,
                        getattr(ev, "other_actors", ""),
                        g["actor_id"], g["actor_canonical"], g["forum_id"],
                        g["forum_role"], g["role_status"], g["canonical_party"],
                    ),
                )
                # register the primary party + any other named persons as actors
                self._register_from_event(conn, ev)
                # role-validate the full cast (primary + other_actors)
                self._insert_cast(conn, ev.event_id, ev.party,
                                  getattr(ev, "other_actors", ""),
                                  ev.source_document, ev.court)
        logger.info("Inserted %d events", len(events))
        return len(events)

    def _insert_cast(self, conn, event_id, party, other_actors, source, court) -> None:
        from .entity_resolver import resolve_cast

        try:
            cast = resolve_cast(party or "", other_actors or "", source or "", court or "")
        except Exception as exc:
            logger.debug("cast resolution unavailable: %s", exc)
            return
        for c in cast:
            conn.execute(
                "INSERT OR REPLACE INTO event_actors "
                "(event_id, actor_key, name, canonical_name, forum_id, "
                "forum_role, stated_role, role_status, is_primary) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (event_id, c.get("actor_id") or "", c["name"],
                 c.get("actor_canonical") or "", c.get("forum_id") or "",
                 c.get("forum_role") or "", c.get("stated_role") or "",
                 c.get("role_status") or "", c["is_primary"]),
            )

    def backfill_event_actors(self) -> dict:
        """(Re)build the event_actors cast for every existing event.

        Lets a database populated before the cast gate existed gain the full
        role-validated participant list without re-extracting from the model.
        """
        from collections import Counter

        tally: Counter[str] = Counter()
        with self._connect() as conn:
            conn.execute("DELETE FROM event_actors")
            rows = list(conn.execute(
                "SELECT event_id, party, other_actors, source_documents, court "
                "FROM events"
            ))
            for r in rows:
                self._insert_cast(conn, r["event_id"], r["party"],
                                  r["other_actors"], r["source_documents"], r["court"])
            for cr in conn.execute("SELECT role_status, is_primary FROM event_actors"):
                tally[("primary" if cr["is_primary"] else "secondary") + ":" + cr["role_status"]] += 1
        logger.info("Backfilled event_actors: %s", dict(tally))
        return dict(tally)

    def event_cast(self, event_id: str) -> list[sqlite3.Row]:
        with self._connect() as conn:
            return list(conn.execute(
                "SELECT * FROM event_actors WHERE event_id = ? "
                "ORDER BY is_primary DESC, name", (event_id,)
            ))

    def actor_participations(self, actor_key: str) -> list[sqlite3.Row]:
        """Every event a given actor participated in, in ANY role."""
        with self._connect() as conn:
            return list(conn.execute(
                "SELECT ea.*, e.date, e.action FROM event_actors ea "
                "JOIN events e ON e.event_id = ea.event_id "
                "WHERE ea.actor_key = ? ORDER BY e.date", (actor_key,)
            ))

    def cast_role_conflicts(self) -> list[sqlite3.Row]:
        """Role conflicts across the full cast (primary AND secondary actors)."""
        with self._connect() as conn:
            return list(conn.execute(
                "SELECT ea.*, e.date FROM event_actors ea "
                "JOIN events e ON e.event_id = ea.event_id "
                "WHERE ea.role_status = 'conflict' ORDER BY e.date"
            ))

    # ------------------------------------------------------------------
    # Actor registry  ("every name in the record is an actor")

    def _register_from_event(self, conn: sqlite3.Connection, ev) -> None:
        names = []
        if ev.party and ev.party != "undetermined":
            names.append((ev.party, ev.excerpt))
        for nm in (getattr(ev, "other_actors", "") or "").split(";"):
            nm = nm.strip()
            if nm:
                names.append((nm, ev.excerpt))
        for name, mention in names:
            self._upsert_actor(conn, name, "undetermined", mention,
                               ev.source_document)

    def register_roster(self, roster: list[dict]) -> int:
        """Register a document's full person-roster (each grounded by a mention)."""
        with self._connect() as conn:
            for r in roster:
                self._upsert_actor(conn, r["name"], r.get("role", "undetermined"),
                                   r.get("mention", ""), r.get("source_document", ""))
        return len(roster)

    def _upsert_actor(self, conn, name, role, mention, source) -> None:
        from .entity_resolver import actor_key, is_person_name

        if not is_person_name(name):     # skip roles/institutions/generic phrases
            return
        key, display, canonical = actor_key(name)
        if not key:
            return
        row = conn.execute(
            "SELECT role, mention, source_documents, mentions FROM case_actors "
            "WHERE actor_key = ?", (key,)
        ).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO case_actors(actor_key, name, role, canonical, "
                "mention, source_documents, mentions) VALUES(?,?,?,?,?,?,1)",
                (key, display, role or "undetermined", 1 if canonical else 0,
                 mention or "", source or ""),
            )
            return
        new_role = row["role"] if row["role"] != "undetermined" else (role or "undetermined")
        new_mention = row["mention"] or (mention or "")
        srcs = _csv_add(row["source_documents"], source) if source else row["source_documents"]
        conn.execute(
            "UPDATE case_actors SET role=?, mention=?, source_documents=?, "
            "mentions=mentions+1 WHERE actor_key=?",
            (new_role, new_mention, srcs, key),
        )

    def all_actors(self) -> list[sqlite3.Row]:
        with self._connect() as conn:
            return list(conn.execute(
                "SELECT * FROM case_actors ORDER BY canonical DESC, mentions DESC"
            ))

    def backfill_entities(self) -> dict:
        """(Re)compute role-gate fields for every existing event.

        Returns a tally of role_status outcomes. Used to enrich a database
        that was populated before the gate existed.
        """
        from collections import Counter

        tally: Counter[str] = Counter()
        with self._connect() as conn:
            rows = list(conn.execute(
                "SELECT event_id, party, source_documents, court FROM events"
            ))
            for r in rows:
                g = _gate_fields(r["party"], r["source_documents"], r["court"])
                tally[g["role_status"]] += 1
                conn.execute(
                    "UPDATE events SET actor_id=?, actor_canonical=?, forum_id=?, "
                    "forum_role=?, role_status=?, canonical_party=? WHERE event_id=?",
                    (g["actor_id"], g["actor_canonical"], g["forum_id"],
                     g["forum_role"], g["role_status"], g["canonical_party"],
                     r["event_id"]),
                )
        logger.info("Backfilled %d events: %s", len(rows), dict(tally))
        return dict(tally)

    def role_conflicts(self) -> list[sqlite3.Row]:
        with self._connect() as conn:
            return list(conn.execute(
                "SELECT * FROM events WHERE role_status='conflict' "
                "ORDER BY date, event_id"
            ))

    def _upsert_date(
        self, conn: sqlite3.Connection, date: str, event_id: str, source: str
    ) -> None:
        row = conn.execute(
            "SELECT event_ids, source_documents FROM dates WHERE date = ?",
            (date,),
        ).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO dates(date, event_ids, source_documents) VALUES(?,?,?)",
                (date, event_id, source),
            )
            return
        ids = _csv_add(row["event_ids"], event_id)
        srcs = _csv_add(row["source_documents"], source)
        conn.execute(
            "UPDATE dates SET event_ids = ?, source_documents = ? WHERE date = ?",
            (ids, srcs, date),
        )

    # ------------------------------------------------------------------
    # Reads

    def all_dates(self) -> list[sqlite3.Row]:
        with self._connect() as conn:
            return list(conn.execute("SELECT * FROM dates ORDER BY date ASC"))

    def events_on(self, date: str) -> list[sqlite3.Row]:
        with self._connect() as conn:
            return list(conn.execute(
                "SELECT * FROM events WHERE date = ? ORDER BY event_id",
                (date,),
            ))

    def get_event(self, event_id: str) -> sqlite3.Row | None:
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM events WHERE event_id = ?", (event_id,)
            ).fetchone()

    def iter_events_chronologically(self) -> Iterator[sqlite3.Row]:
        with self._connect() as conn:
            yield from conn.execute(
                "SELECT * FROM events ORDER BY date ASC, event_id ASC"
            )

    # ------------------------------------------------------------------
    # Evidence ledger export

    def export_ledger(self) -> dict:
        """Introspect every user table and return a JSON-ready evidence ledger.

        The ledger lists each table with its column schema, row count, and full
        contents. `events` is the primary evidence record (its `event_id` is the
        evidence ID); `dates` is the per-day rollup index into it. Indexes are
        not part of the evidence and are excluded.
        """
        from datetime import datetime, timezone

        with self._connect() as conn:
            table_names = [
                r["name"]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name NOT LIKE 'sqlite_%' ORDER BY name"
                )
            ]
            tables: dict[str, dict] = {}
            for name in table_names:
                cols = [
                    {"name": c["name"], "type": c["type"],
                     "primary_key": bool(c["pk"]), "not_null": bool(c["notnull"])}
                    for c in conn.execute(f'PRAGMA table_info("{name}")')
                ]
                rows = [dict(r) for r in conn.execute(f'SELECT * FROM "{name}"')]
                tables[name] = {
                    "columns": cols,
                    "row_count": len(rows),
                    "rows": rows,
                }

        return {
            "ledger_type": "evidence_and_event_tracking",
            "database": str(self.db_path),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "table_count": len(tables),
            "tables": tables,
        }


def _gate_fields(party: str, source_document: str, court: str) -> dict:
    """Run the role-validation gate; degrade gracefully if reference absent."""
    try:
        from .entity_resolver import validate
        g = validate(party or "", source_document or "", court or "")
        return {k: ("" if v is None else v) for k, v in g.items()}
    except Exception as exc:  # missing reference file, etc.
        logger.debug("entity gate unavailable: %s", exc)
        return {
            "actor_id": "", "actor_canonical": "", "forum_id": "",
            "forum_role": "", "role_status": "", "canonical_party": party or "",
        }


def _csv_add(existing: str, value: str) -> str:
    items = [s for s in existing.split(",") if s] if existing else []
    if value not in items:
        items.append(value)
    return ",".join(items)
