"""The card box.

A plain SQLite file in Home Assistant's config directory holding two things:

* ``examples``  -- the cards: sentence on the front, action to run on the back.
* ``decisions`` -- a log of every request, so we can measure later how well
  kassistant actually decides.

We deliberately do not use a vector database. With a few thousand cards, a plain
dot product across the whole matrix takes about a millisecond -- faster than any
index structure would pay off, and without an extra dependency.

Every method here blocks. Callers must run them through
``hass.async_add_executor_job`` to keep the event loop free.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np

_LOGGER = logging.getLogger(__name__)

SOURCE_LEARNED = "learned"
SOURCE_SEED = "seed"

# How often a sentence has to produce the same action before that card is
# allowed to answer on its own.
#
# This is what stands in for guessing whether a follow-up was a correction. If
# the fallback agent gets something wrong once, the card it leaves behind has a
# weight of one and is never used -- it would take the same mistake twice for
# the same sentence. Anything the user actually says regularly crosses the line
# on the second time of asking. Seeded cards are exempt: they come from Home
# Assistant's own curated templates, not from a guess.
MIN_LEARNED_WEIGHT = 2.0

# Starting size of the index buffers; they double from here.
_INITIAL_CAPACITY = 256

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS examples (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    utterance   TEXT    NOT NULL,
    norm        TEXT    NOT NULL,
    language    TEXT    NOT NULL,
    action      TEXT    NOT NULL,
    source      TEXT    NOT NULL,
    weight      REAL    NOT NULL DEFAULT 1.0,
    hits        INTEGER NOT NULL DEFAULT 0,
    embedding   BLOB,
    created_at  TEXT    NOT NULL,
    last_hit_at TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS examples_unique
    ON examples (norm, language, action);

CREATE TABLE IF NOT EXISTS decisions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT    NOT NULL,
    utterance   TEXT    NOT NULL,
    language    TEXT,
    mode        TEXT    NOT NULL,
    tier        TEXT    NOT NULL,
    score       REAL,
    example_id  INTEGER,
    proposed    TEXT,
    executed    INTEGER NOT NULL DEFAULT 0,
    observed    TEXT,
    latency_ms  INTEGER
);

CREATE INDEX IF NOT EXISTS decisions_ts ON decisions (ts);
"""


def is_eligible(source: str, weight: float) -> bool:
    """May this card answer on its own?

    Seeded cards always may. A learned one has to have been confirmed by the
    same sentence producing the same action again.
    """
    return source == SOURCE_SEED or weight >= MIN_LEARNED_WEIGHT


def canonical_action(action: Any) -> str:
    """Render an action as a stable string so duplicate cards are recognised."""
    return json.dumps(action, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(slots=True, frozen=True)
class Match:
    """A hit in the card box."""

    example_id: int
    score: float
    utterance: str
    action: dict[str, Any]
    source: str


class Store:
    """Wraps the database and the in-memory search index."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        # Search index: parallel arrays of row ids and their vectors, held in
        # buffers that grow by doubling. Appending row by row into an exactly
        # sized array copies the whole thing every time -- measured at six
        # seconds for four thousand cards, against a tenth of a second for the
        # database writes themselves.
        self._ids: np.ndarray = np.zeros((0,), dtype=np.int64)
        self._matrix: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        # Whether each indexed row may answer yet. Kept beside the vectors so a
        # search masks them out instead of finding them and rejecting them.
        self._eligible: np.ndarray = np.zeros((0,), dtype=bool)
        self._count = 0
        # example id -> row in the index, so a card that gains weight can be
        # promoted without rebuilding everything.
        self._position: dict[int, int] = {}

    # -- Lifecycle ------------------------------------------------------------

    def setup(self) -> None:
        """Open the database, create the schema, load the index."""
        with self._lock:
            self._conn = sqlite3.connect(self._path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
            self._reload_index()
        _LOGGER.debug("Card box opened: %s (%d cards)", self._path, len(self._ids))

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    # -- Writing --------------------------------------------------------------

    def add_example(
        self,
        *,
        utterance: str,
        norm: str,
        language: str,
        action: dict[str, Any],
        source: str,
        embedding: np.ndarray | None = None,
    ) -> int | None:
        """Add a card.

        If it already exists its weight is raised instead -- a sentence you say
        often should count for more than one you said once. Returns the id of
        the card, or ``None`` if nothing was stored.
        """
        conn = self._require_conn()
        action_json = canonical_action(action)
        now = datetime.now(UTC).isoformat()
        blob = embedding.astype(np.float32).tobytes() if embedding is not None else None

        with self._lock:
            existing = conn.execute(
                "SELECT id FROM examples WHERE norm = ? AND language = ? AND action = ?",
                (norm, language, action_json),
            ).fetchone()

            if existing is not None:
                conn.execute(
                    "UPDATE examples SET weight = weight + 1.0 WHERE id = ?",
                    (existing["id"],),
                )
                conn.commit()
                example_id = int(existing["id"])

                # The repeat may be what confirms this card. Promote it in place
                # rather than waiting for the next restart to rebuild the index.
                row = conn.execute(
                    "SELECT source, weight FROM examples WHERE id = ?", (example_id,)
                ).fetchone()
                position = self._position.get(example_id)
                if position is not None and is_eligible(row["source"], row["weight"]):
                    self._eligible[position] = True

                return example_id

            cursor = conn.execute(
                """
                INSERT INTO examples
                    (utterance, norm, language, action, source, embedding, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (utterance, norm, language, action_json, source, blob, now),
            )
            conn.commit()
            example_id = int(cursor.lastrowid)

            if blob is not None and embedding is not None:
                self._append_to_index(
                    example_id, embedding, eligible=is_eligible(source, 1.0)
                )

        return example_id

    def log_decision(
        self,
        *,
        utterance: str,
        language: str | None,
        mode: str,
        tier: str,
        score: float | None = None,
        example_id: int | None = None,
        proposed: Any = None,
        executed: bool = False,
        observed: Any = None,
        latency_ms: int | None = None,
    ) -> int:
        """Append an entry to the decision log."""
        conn = self._require_conn()
        with self._lock:
            cursor = conn.execute(
                """
                INSERT INTO decisions
                    (ts, utterance, language, mode, tier, score, example_id,
                     proposed, executed, observed, latency_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(UTC).isoformat(),
                    utterance,
                    language,
                    mode,
                    tier,
                    score,
                    example_id,
                    json.dumps(proposed, ensure_ascii=False)
                    if proposed is not None
                    else None,
                    int(executed),
                    json.dumps(observed, ensure_ascii=False)
                    if observed is not None
                    else None,
                    latency_ms,
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def attach_observation(self, decision_id: int, observed: Any) -> None:
        """Record after the fact what actually happened."""
        conn = self._require_conn()
        with self._lock:
            conn.execute(
                "UPDATE decisions SET observed = ? WHERE id = ?",
                (json.dumps(observed, ensure_ascii=False), decision_id),
            )
            conn.commit()

    def mark_hit(self, example_id: int) -> None:
        conn = self._require_conn()
        with self._lock:
            conn.execute(
                "UPDATE examples SET hits = hits + 1, last_hit_at = ? WHERE id = ?",
                (datetime.now(UTC).isoformat(), example_id),
            )
            conn.commit()

    # -- Reading --------------------------------------------------------------

    def search(self, vector: np.ndarray, language: str | None = None) -> Match | None:
        """Find the most similar card. ``None`` when the box is empty.

        Simplification: the language is checked *after* the search. If the best
        card is in another language we return nothing rather than falling back
        to the runner-up. That costs nothing in a single-language household and
        errs on the safe side -- the request then goes to the fallback agent.
        Doing it properly means carrying the language in the index.
        """
        with self._lock:
            if self._count == 0:
                return None
            if self._matrix.shape[1] != vector.shape[0]:
                _LOGGER.warning(
                    "Vector length mismatch (%d stored, %d queried). "
                    "Was the embedding model changed?",
                    self._matrix.shape[1],
                    vector.shape[0],
                )
                return None

            scores = self._matrix[: self._count] @ vector.astype(np.float32)
            # Cards awaiting confirmation are masked out rather than found and
            # then rejected: an unconfirmed near-match must not hide a confirmed
            # one that sits slightly further away.
            scores = np.where(self._eligible[: self._count], scores, -np.inf)
            best = int(np.argmax(scores))
            if not np.isfinite(scores[best]):
                return None
            example_id = int(self._ids[best])
            score = float(scores[best])

        row = self.get_example(example_id)
        if row is None:
            return None

        if language is not None and row["language"] != language:
            return None

        return Match(
            example_id=example_id,
            score=score,
            utterance=row["utterance"],
            action=json.loads(row["action"]),
            source=row["source"],
        )

    def get_example(self, example_id: int) -> sqlite3.Row | None:
        conn = self._require_conn()
        with self._lock:
            return conn.execute(
                "SELECT * FROM examples WHERE id = ?", (example_id,)
            ).fetchone()

    def stats(self) -> dict[str, int]:
        """Numbers for the diagnostics view."""
        conn = self._require_conn()
        with self._lock:
            examples = conn.execute("SELECT COUNT(*) AS n FROM examples").fetchone()[
                "n"
            ]
            decisions = conn.execute("SELECT COUNT(*) AS n FROM decisions").fetchone()[
                "n"
            ]
            indexed = self._count
        return {"examples": examples, "decisions": decisions, "indexed": indexed}

    def known_keys(self, language: str) -> set[tuple[str, str]]:
        """Every ``(norm, action)`` pair already stored for a language.

        Seeding uses this to skip sentences it has stored before. Without it a
        repeated run would send thousands of sentences to the embedding service
        only to throw the answers away at the duplicate check -- which is the
        difference between reseeding being cheap enough to do automatically and
        being something the user has to be asked about.
        """
        conn = self._require_conn()
        with self._lock:
            rows = conn.execute(
                "SELECT norm, action FROM examples WHERE language = ?", (language,)
            ).fetchall()
        return {(row["norm"], row["action"]) for row in rows}

    def decision_stats(self, window: int = 200) -> dict[str, Any]:
        """How kassistant has been deciding lately.

        This is the instrument for the one judgement call the user has to make:
        whether the confidence threshold is set right for their own speech. It
        counts what kassistant *decided*, not what it executed, so the numbers
        mean the same thing in observe, shadow and active mode -- which is the
        whole point of being able to watch before arming it.
        """
        conn = self._require_conn()
        with self._lock:
            rows = conn.execute(
                "SELECT tier, score, latency_ms FROM decisions ORDER BY id DESC LIMIT ?",
                (window,),
            ).fetchall()

        tiers: dict[str, int] = {}
        for row in rows:
            tiers[row["tier"]] = tiers.get(row["tier"], 0) + 1

        # In observe mode the router is never asked, so those requests say
        # nothing about how well it recognises. Counting them would report a
        # steady zero percent -- which reads as "recognised nothing" when the
        # truth is "did not look", and makes the number useless for the one
        # thing it exists for: deciding when to switch the mode.
        measured = [row for row in rows if row["tier"] != "observe"]
        handled = tiers.get("fastpath", 0)
        scores = [row["score"] for row in measured if row["score"] is not None]
        latencies = [
            row["latency_ms"] for row in measured if row["latency_ms"] is not None
        ]

        return {
            "sampled": len(measured),
            "not_measured": len(rows) - len(measured),
            "handled": handled,
            "handled_pct": round(100 * handled / len(measured), 1)
            if measured
            else None,
            "avg_score": round(sum(scores) / len(scores), 3) if scores else None,
            "avg_latency_ms": round(sum(latencies) / len(latencies))
            if latencies
            else None,
            "tiers": tiers,
        }

    def card_stats(self) -> dict[str, Any]:
        """Where the cards came from."""
        conn = self._require_conn()
        with self._lock:
            rows = conn.execute(
                "SELECT source, COUNT(*) AS n FROM examples GROUP BY source"
            ).fetchall()
            indexed = self._count
            awaiting = int(np.count_nonzero(~self._eligible[: self._count]))
        by_source = {row["source"]: row["n"] for row in rows}
        return {
            "total": sum(by_source.values()),
            "searchable": indexed,
            "awaiting_confirmation": awaiting,
            "by_source": by_source,
        }

    def examples_without_embedding(self, limit: int = 256) -> list[sqlite3.Row]:
        """Cards still missing a vector -- e.g. because Ollama was down."""
        conn = self._require_conn()
        with self._lock:
            return conn.execute(
                "SELECT id, norm FROM examples WHERE embedding IS NULL LIMIT ?",
                (limit,),
            ).fetchall()

    def set_embedding(self, example_id: int, embedding: np.ndarray) -> None:
        conn = self._require_conn()
        with self._lock:
            conn.execute(
                "UPDATE examples SET embedding = ? WHERE id = ?",
                (embedding.astype(np.float32).tobytes(), example_id),
            )
            conn.commit()
            row = conn.execute(
                "SELECT source, weight FROM examples WHERE id = ?", (example_id,)
            ).fetchone()
            self._append_to_index(
                example_id,
                embedding,
                eligible=is_eligible(row["source"], row["weight"]),
            )

    # -- Index ----------------------------------------------------------------

    def _reload_index(self) -> None:
        """Read every stored vector into memory.

        The caller already holds ``self._lock``.
        """
        conn = self._require_conn()
        rows = conn.execute(
            "SELECT id, embedding, source, weight FROM examples "
            "WHERE embedding IS NOT NULL ORDER BY id"
        ).fetchall()

        self._position = {}

        if not rows:
            self._ids = np.zeros((0,), dtype=np.int64)
            self._matrix = np.zeros((0, 0), dtype=np.float32)
            self._eligible = np.zeros((0,), dtype=bool)
            self._count = 0
            return

        vectors = [np.frombuffer(row["embedding"], dtype=np.float32) for row in rows]
        dimension = vectors[0].shape[0]
        keep = [
            (row, vec)
            for row, vec in zip(rows, vectors, strict=True)
            if vec.shape[0] == dimension
        ]

        if len(keep) != len(rows):
            _LOGGER.warning(
                "Skipped %d cards with a mismatching vector length",
                len(rows) - len(keep),
            )

        self._ids = np.asarray([row["id"] for row, _ in keep], dtype=np.int64)
        self._matrix = np.vstack([vec for _, vec in keep]).astype(np.float32)
        self._eligible = np.asarray(
            [is_eligible(row["source"], row["weight"]) for row, _ in keep], dtype=bool
        )
        self._count = len(keep)
        self._position = {int(row["id"]): i for i, (row, _) in enumerate(keep)}

    def _append_to_index(
        self, example_id: int, embedding: np.ndarray, *, eligible: bool
    ) -> None:
        """Append one vector to the index. The caller holds ``self._lock``."""
        vector = embedding.astype(np.float32).reshape(-1)

        # A card that is already indexed is updated in place. Appending again
        # would leave the earlier row orphaned but still searchable, so a stale
        # vector could keep winning matches for a card that has moved on.
        if (position := self._position.get(example_id)) is not None:
            if self._matrix.shape[1] == vector.shape[0]:
                self._matrix[position] = vector
                self._eligible[position] = eligible
            return

        if self._count == 0:
            self._matrix = np.zeros((_INITIAL_CAPACITY, vector.shape[0]), np.float32)
            self._ids = np.zeros((_INITIAL_CAPACITY,), dtype=np.int64)
            self._eligible = np.zeros((_INITIAL_CAPACITY,), dtype=bool)
        elif self._matrix.shape[1] != vector.shape[0]:
            _LOGGER.warning("Vector length does not match the index, card not indexed")
            return
        elif self._count == self._matrix.shape[0]:
            self._grow()

        self._matrix[self._count] = vector
        self._ids[self._count] = example_id
        self._eligible[self._count] = eligible
        self._position[example_id] = self._count
        self._count += 1

    def _grow(self) -> None:
        """Double the buffers. The caller holds ``self._lock``."""
        capacity = self._matrix.shape[0] * 2
        matrix = np.zeros((capacity, self._matrix.shape[1]), dtype=np.float32)
        matrix[: self._count] = self._matrix[: self._count]
        ids = np.zeros((capacity,), dtype=np.int64)
        ids[: self._count] = self._ids[: self._count]
        eligible = np.zeros((capacity,), dtype=bool)
        eligible[: self._count] = self._eligible[: self._count]
        self._matrix = matrix
        self._ids = ids
        self._eligible = eligible

    # -- Helpers --------------------------------------------------------------

    def _require_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Store.setup() was not called")
        return self._conn
