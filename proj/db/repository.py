"""
repository.py

The single point of database access for the whole project (per the
approved architecture: "db/repository.py — all DB read/write functions,
single access point"). Every other module — ingestion pipeline, and later
NLP/AERF modules — talks to SQLite only through this class, never by
opening sqlite3 connections themselves. This keeps every other module
unit-testable with a temporary/mock database.

Phase 1 implements: initialize_database, paper CRUD (insert/get/exists),
and chunk CRUD (bulk insert/get). Functions for entities/claims/etc. will
be added here in later phases without changing this file's existing
functions.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from config.config_loader import get_settings, PROJECT_ROOT
from db.models import Chunk, Paper
from utils.exceptions import DatabaseError
from utils.logging_config import get_logger

logger = get_logger(__name__)

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def _default_db_path() -> Path:
    settings = get_settings()
    db_setting = settings.get("paths", {}).get("database_file", "data/aerf.db")
    db_path = Path(db_setting)
    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


class Repository:
    """
    Thin wrapper around a SQLite database file. One Repository instance
    per database path; each method opens and closes its own connection
    (safe for the small, low-concurrency workload of this project, and
    simplest to reason about / unit test).
    """

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path is not None else _default_db_path()

    # ---------------------------------------------------------------
    # Connection handling
    # ---------------------------------------------------------------

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            yield conn
            conn.commit()
        except sqlite3.Error as exc:
            if conn is not None:
                conn.rollback()
            raise DatabaseError(f"Database operation failed: {exc}") from exc
        finally:
            if conn is not None:
                conn.close()

    # ---------------------------------------------------------------
    # Schema initialization
    # ---------------------------------------------------------------

    def initialize_database(self) -> None:
        """Create all tables (idempotent — safe to call every run)."""
        if not SCHEMA_PATH.exists():
            raise DatabaseError(f"schema.sql not found at {SCHEMA_PATH}")

        schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
        try:
            with self._connect() as conn:
                conn.executescript(schema_sql)
        except DatabaseError:
            raise
        logger.info("Database initialized at %s", self.db_path)

    # ---------------------------------------------------------------
    # Papers
    # ---------------------------------------------------------------

    def paper_exists(self, source_filename: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM papers WHERE source_filename = ? LIMIT 1",
                (source_filename,),
            ).fetchone()
        return row is not None

    def insert_paper(self, paper: Paper) -> Paper:
        """
        Insert a new paper row. Raises DatabaseError if a paper with the
        same source_filename already exists (UNIQUE constraint) — callers
        should check paper_exists() first if they want to skip re-ingestion
        gracefully instead of hitting this error.
        """
        try:
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO papers
                        (title, authors, publication_year, venue,
                         source_filename, source_path, page_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        paper.title,
                        paper.authors,
                        paper.publication_year,
                        paper.venue,
                        paper.source_filename,
                        paper.source_path,
                        paper.page_count,
                    ),
                )
                paper.paper_id = cursor.lastrowid
        except DatabaseError as exc:
            if "UNIQUE constraint failed" in str(exc):
                raise DatabaseError(
                    f"Paper with source_filename='{paper.source_filename}' "
                    "already exists. Use paper_exists() to check before "
                    "inserting, or get_paper_by_filename() to fetch it."
                ) from exc
            raise
        logger.info("Inserted paper id=%s (%s)", paper.paper_id, paper.source_filename)
        return paper

    def get_paper_by_filename(self, source_filename: str) -> Paper | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM papers WHERE source_filename = ?",
                (source_filename,),
            ).fetchone()
        return self._row_to_paper(row) if row else None

    def get_paper_by_id(self, paper_id: int) -> Paper | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM papers WHERE paper_id = ?", (paper_id,)
            ).fetchone()
        return self._row_to_paper(row) if row else None

    def count_papers(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM papers").fetchone()
        return row["n"]

    def update_paper_metadata(
        self,
        paper_id: int,
        title: str | None = None,
        authors: str | None = None,
        publication_year: int | None = None,
        venue: str | None = None,
    ) -> Paper | None:
        """
        Set title/authors/publication_year/venue on an already-inserted
        paper. Only fields actually passed (non-None) are updated — the
        others are left exactly as they already are in the row. This is
        the correction path for papers that were ingested before their
        metadata was known/threaded through (see pipeline/orchestrator.py's
        `ingest_paper(..., metadata=...)` and
        scripts/backfill_paper_metadata.py), so it must never overwrite an
        existing real value with None just because a caller didn't happen
        to supply that one field.

        Returns the updated Paper (via get_paper_by_id), or None if no
        paper with that id exists. Raises DatabaseError on a SQL failure,
        same as the rest of this class — no silent failure.
        """
        fields, values = [], []
        for column, value in (
            ("title", title),
            ("authors", authors),
            ("publication_year", publication_year),
            ("venue", venue),
        ):
            if value is not None:
                fields.append(f"{column} = ?")
                values.append(value)
        if not fields:
            return self.get_paper_by_id(paper_id)

        values.append(paper_id)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE papers SET {', '.join(fields)} WHERE paper_id = ?",
                values,
            )
        updated = self.get_paper_by_id(paper_id)
        if updated is not None:
            logger.info("Updated metadata for paper_id=%s (%s)", paper_id, updated.source_filename)
        return updated

    @staticmethod
    def _row_to_paper(row: sqlite3.Row) -> Paper:
        return Paper(
            paper_id=row["paper_id"],
            title=row["title"],
            authors=row["authors"],
            publication_year=row["publication_year"],
            venue=row["venue"],
            source_filename=row["source_filename"],
            source_path=row["source_path"],
            page_count=row["page_count"],
            ingested_at=row["ingested_at"],
        )

    # ---------------------------------------------------------------
    # Chunks
    # ---------------------------------------------------------------

    def insert_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Bulk-insert chunks for a paper, returning them with chunk_id set."""
        if not chunks:
            return []

        try:
            with self._connect() as conn:
                for chunk in chunks:
                    cursor = conn.execute(
                        """
                        INSERT INTO chunks
                            (paper_id, page_number, char_start, char_end, text)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            chunk.paper_id,
                            chunk.page_number,
                            chunk.char_start,
                            chunk.char_end,
                            chunk.text,
                        ),
                    )
                    chunk.chunk_id = cursor.lastrowid
        except DatabaseError:
            raise
        logger.info("Inserted %d chunk(s) for paper_id=%s", len(chunks), chunks[0].paper_id)
        return chunks

    def get_chunks_by_paper(self, paper_id: int) -> list[Chunk]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM chunks
                WHERE paper_id = ?
                ORDER BY page_number, char_start
                """,
                (paper_id,),
            ).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def count_chunks(self, paper_id: int | None = None) -> int:
        with self._connect() as conn:
            if paper_id is None:
                row = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) AS n FROM chunks WHERE paper_id = ?", (paper_id,)
                ).fetchone()
        return row["n"]


    # ---------------------------------------------------------------
    # Phase 2 scientific NLP persistence
    # ---------------------------------------------------------------

    def ensure_phase2_schema(self) -> None:
        """Ensure the Phase 2 tables/indexes exist; idempotent."""
        statements = [
            """CREATE TABLE IF NOT EXISTS entities (
                entity_id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id INTEGER NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
                paper_id INTEGER NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
                text TEXT NOT NULL, label TEXT NOT NULL,
                span_start INTEGER NOT NULL, span_end INTEGER NOT NULL,
                UNIQUE(chunk_id, span_start, span_end, label, text)
            )""",
            """CREATE TABLE IF NOT EXISTS claims (
                claim_id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id INTEGER NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
                chunk_id INTEGER NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
                text TEXT NOT NULL, span_start INTEGER NOT NULL, span_end INTEGER NOT NULL,
                extracted_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(paper_id, chunk_id, span_start, span_end)
            )""",
            """CREATE TABLE IF NOT EXISTS claim_entities (
                claim_id INTEGER NOT NULL REFERENCES claims(claim_id) ON DELETE CASCADE,
                entity_id INTEGER NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
                PRIMARY KEY(claim_id, entity_id)
            )""",
            """CREATE TABLE IF NOT EXISTS relations (
                relation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                claim_id INTEGER NOT NULL REFERENCES claims(claim_id) ON DELETE CASCADE,
                subject_entity_id INTEGER NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
                predicate TEXT NOT NULL,
                object_entity_id INTEGER NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
                UNIQUE(claim_id, subject_entity_id, predicate, object_entity_id)
            )""",
            """CREATE TABLE IF NOT EXISTS claim_embeddings (
                claim_id INTEGER PRIMARY KEY REFERENCES claims(claim_id) ON DELETE CASCADE,
                model_name TEXT NOT NULL, vector BLOB NOT NULL, dimensions INTEGER NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS processed_chunks (
                paper_id INTEGER NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
                chunk_id INTEGER NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
                processed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (paper_id, chunk_id)
            )""",
        ]
        with self._connect() as conn:
            for sql in statements:
                conn.execute(sql)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_paper ON entities(paper_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_claims_paper ON claims(paper_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_relations_claim ON relations(claim_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_proc_chunks_paper ON processed_chunks(paper_id)")

    def insert_entity_p2(self, paper_id, chunk_id, entity):
        with self._connect() as conn:
            conn.execute("""INSERT OR IGNORE INTO entities
                (paper_id,chunk_id,text,label,span_start,span_end) VALUES (?,?,?,?,?,?)""",
                (paper_id, chunk_id, entity.text, entity.label, entity.start, entity.end))
            row=conn.execute("""SELECT entity_id FROM entities WHERE chunk_id=? AND span_start=?
                AND span_end=? AND label=? AND text=?""",
                (chunk_id, entity.start, entity.end, entity.label, entity.text)).fetchone()
            return row["entity_id"]

    def insert_claim_p2(self, paper_id, chunk_id, claim):
        with self._connect() as conn:
            conn.execute("""INSERT OR IGNORE INTO claims
                (paper_id,chunk_id,text,span_start,span_end) VALUES (?,?,?,?,?)""",
                (paper_id, chunk_id, claim.text, claim.start, claim.end))
            row=conn.execute("""SELECT claim_id FROM claims WHERE paper_id=? AND chunk_id=?
                AND span_start=? AND span_end=?""",
                (paper_id, chunk_id, claim.start, claim.end)).fetchone()
            return row["claim_id"]

    def link_claim_entity_p2(self, claim_id, entity_id):
        with self._connect() as conn:
            conn.execute("INSERT OR IGNORE INTO claim_entities(claim_id,entity_id) VALUES(?,?)",
                         (claim_id, entity_id))

    def insert_relation_p2(self, claim_id, subject_entity_id, predicate, object_entity_id):
        with self._connect() as conn:
            conn.execute("""INSERT OR IGNORE INTO relations
                (claim_id,subject_entity_id,predicate,object_entity_id) VALUES(?,?,?,?)""",
                (claim_id, subject_entity_id, predicate, object_entity_id))
            row=conn.execute("""SELECT relation_id FROM relations WHERE claim_id=?
                AND subject_entity_id=? AND predicate=? AND object_entity_id=?""",
                (claim_id, subject_entity_id, predicate, object_entity_id)).fetchone()
            return row["relation_id"]

    def upsert_embedding_p2(self, claim_id, model_name, vector):
        raw = vector.tobytes() if hasattr(vector, "tobytes") else bytes(vector)
        dims = len(vector) if hasattr(vector, "__len__") else 0
        with self._connect() as conn:
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(claim_embeddings)").fetchall()}
            if "dimensions" in cols:
                conn.execute("""INSERT INTO claim_embeddings(claim_id,model_name,vector,dimensions)
                    VALUES(?,?,?,?) ON CONFLICT(claim_id) DO UPDATE SET
                    model_name=excluded.model_name, vector=excluded.vector, dimensions=excluded.dimensions""",
                    (claim_id, model_name, raw, dims))
            else:
                conn.execute("""INSERT INTO claim_embeddings(claim_id,model_name,vector)
                    VALUES(?,?,?) ON CONFLICT(claim_id) DO UPDATE SET
                    model_name=excluded.model_name, vector=excluded.vector""",
                    (claim_id, model_name, raw))

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> Chunk:
        return Chunk(
            chunk_id=row["chunk_id"],
            paper_id=row["paper_id"],
            page_number=row["page_number"],
            char_start=row["char_start"],
            char_end=row["char_end"],
            text=row["text"],
            created_at=row["created_at"],
        )

    # ---------------------------------------------------------------
    # Phase 3 scientific reasoning persistence
    # ---------------------------------------------------------------
    def ensure_phase3_schema(self) -> None:
        statements = [
            "ALTER TABLE claim_pairs ADD COLUMN nli_entailment REAL",
            "ALTER TABLE claim_pairs ADD COLUMN nli_contradiction REAL",
            "ALTER TABLE claim_pairs ADD COLUMN nli_neutral REAL",
            "ALTER TABLE claim_pairs ADD COLUMN relation_alignment_score REAL",
            "ALTER TABLE claim_pairs ADD COLUMN matching_method TEXT",
            "ALTER TABLE claim_pairs ADD COLUMN entity_alignment_json TEXT",
            "ALTER TABLE claim_pairs ADD COLUMN nli_model_name TEXT",
            "ALTER TABLE aerf_scores ADD COLUMN score_source TEXT",
            "ALTER TABLE evidence_graph_edges ADD COLUMN evidence_id INTEGER",
            "ALTER TABLE knowledge_evolution_snapshots ADD COLUMN evidence_json TEXT",
            "ALTER TABLE research_gaps ADD COLUMN evidence_json TEXT",
            "CREATE TABLE IF NOT EXISTS evidence_graph_edges (edge_id INTEGER PRIMARY KEY AUTOINCREMENT, source_node_type TEXT NOT NULL, source_node_id INTEGER NOT NULL, target_node_type TEXT NOT NULL, target_node_id INTEGER NOT NULL, edge_type TEXT NOT NULL, weight REAL, evidence_id INTEGER)",
            "CREATE TABLE IF NOT EXISTS knowledge_evolution_snapshots (snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT, topic_key TEXT NOT NULL, as_of_year INTEGER NOT NULL, dominant_claim_id INTEGER, dominant_aerf_score REAL, supporting_claim_count INTEGER, contradicting_claim_count INTEGER, evidence_json TEXT, FOREIGN KEY(dominant_claim_id) REFERENCES claims(claim_id))",
            "CREATE TABLE IF NOT EXISTS research_gaps (gap_id INTEGER PRIMARY KEY AUTOINCREMENT, topic_key TEXT NOT NULL, gap_type TEXT NOT NULL, description TEXT, related_claim_ids TEXT, evidence_json TEXT, detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
            "CREATE TABLE IF NOT EXISTS explanations (explanation_id INTEGER PRIMARY KEY AUTOINCREMENT, claim_id INTEGER NOT NULL, explanation_type TEXT NOT NULL, content_json TEXT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY(claim_id) REFERENCES claims(claim_id) ON DELETE CASCADE)",
            "CREATE INDEX IF NOT EXISTS idx_claim_pairs_a ON claim_pairs(claim_a_id)",
            "CREATE INDEX IF NOT EXISTS idx_claim_pairs_b ON claim_pairs(claim_b_id)",
            "CREATE INDEX IF NOT EXISTS idx_aerf_current ON aerf_scores(claim_id,is_current)",
        ]
        with self._connect() as conn:
            for sql in statements:
                if sql.startswith("ALTER TABLE"):
                    try: conn.execute(sql)
                    except sqlite3.OperationalError as e:
                        msg=str(e).lower()
                        if "duplicate column name" not in msg and "no such table" not in msg: raise
                else: conn.execute(sql)

    def list_papers(self):
        with self._connect() as conn:
            rows=conn.execute("SELECT * FROM papers ORDER BY paper_id").fetchall()
        return [self._row_to_paper(r) for r in rows]

    def get_claim(self, claim_id):
        with self._connect() as conn:
            r=conn.execute("SELECT * FROM claims WHERE claim_id=?",(claim_id,)).fetchone()
        return dict(r) if r else None

    def get_claims(self, exclude_paper_id=None):
        with self._connect() as conn:
            if exclude_paper_id is None: rows=conn.execute("SELECT * FROM claims ORDER BY claim_id").fetchall()
            else: rows=conn.execute("SELECT * FROM claims WHERE paper_id<>? ORDER BY claim_id",(exclude_paper_id,)).fetchall()
        return [dict(r) for r in rows]

    def get_claims_for_paper(self,paper_id):
        with self._connect() as conn:
            rows=conn.execute("SELECT * FROM claims WHERE paper_id=? ORDER BY claim_id",(paper_id,)).fetchall()
        return [dict(r) for r in rows]

    def get_claim_embedding(self, claim_id):
        import numpy as np
        with self._connect() as conn:
            r=conn.execute("SELECT * FROM claim_embeddings WHERE claim_id=?",(claim_id,)).fetchone()
        if not r: return None
        return np.frombuffer(r["vector"],dtype=np.float32)

    def get_claim_entities(self, claim_id):
        with self._connect() as conn:
            rows=conn.execute("""SELECT e.* FROM entities e JOIN claim_entities ce ON ce.entity_id=e.entity_id WHERE ce.claim_id=? ORDER BY e.entity_id""",(claim_id,)).fetchall()
        return [dict(r) for r in rows]

    def get_claim_relations(self, claim_id):
        with self._connect() as conn:
            rows=conn.execute("""SELECT r.*, se.text subject_text, oe.text object_text FROM relations r JOIN entities se ON se.entity_id=r.subject_entity_id JOIN entities oe ON oe.entity_id=r.object_entity_id WHERE r.claim_id=? ORDER BY r.relation_id""",(claim_id,)).fetchall()
        return [dict(r) for r in rows]

    def upsert_claim_pair(self,d):
        import json
        with self._connect() as conn:
            # Logical claim-pair relationships are unordered: (A,B) and (B,A)
            # are the same cross-document relationship. Deleting only the
            # exact (claim_a_id, claim_b_id) orientation let a single-paper
            # Phase3Pipeline.run(paper_id=X) followed by run(paper_id=Y) for
            # the same pair's other paper insert a second row in the
            # opposite orientation instead of replacing the first -- there
            # is no unordered uniqueness constraint on this table (see
            # schema.sql), so this DELETE is what enforces it. Deleting
            # both orientations before inserting keeps exactly one row per
            # logical pair regardless of which side happened to be `a` in
            # whichever call computed it most recently.
            conn.execute(
                """DELETE FROM claim_pairs WHERE (claim_a_id=? AND claim_b_id=?) OR (claim_a_id=? AND claim_b_id=?)""",
                (d["claim_a_id"],d["claim_b_id"],d["claim_b_id"],d["claim_a_id"]),
            )
            conn.execute("""INSERT INTO claim_pairs(claim_a_id,claim_b_id,cosine_similarity,nli_label,nli_confidence,entity_overlap_score,validated_as,nli_entailment,nli_contradiction,nli_neutral,relation_alignment_score,matching_method,entity_alignment_json,nli_model_name) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (d["claim_a_id"],d["claim_b_id"],d["cosine_similarity"],d["nli_label"],d["nli_confidence"],d["entity_overlap_score"],d["validated_as"],d.get("nli_entailment"),d.get("nli_contradiction"),d.get("nli_neutral"),d.get("relation_alignment_score"),d.get("matching_method"),json.dumps(d.get("entity_alignment",[])),d.get("nli_model_name")))

    def get_all_claim_pair_keys(self):
        """Unordered (min_id, max_id) of every stored claim pair."""
        with self._connect() as conn:
            rows=conn.execute("SELECT claim_a_id,claim_b_id FROM claim_pairs").fetchall()
        return {(min(r["claim_a_id"],r["claim_b_id"]),max(r["claim_a_id"],r["claim_b_id"])) for r in rows}

    def paper_has_claims(self, paper_id):
        with self._connect() as conn:
            r=conn.execute("SELECT 1 FROM claims WHERE paper_id=? LIMIT 1",(paper_id,)).fetchone()
        return r is not None

    def get_processed_chunk_ids(self, paper_id: int) -> set[int]:
        """Return the set of chunk IDs already processed by Phase 2 for this paper."""
        with self._connect() as conn:
            # Sync any completed chunks from entities into processed_chunks if missing
            conn.execute(
                """INSERT OR IGNORE INTO processed_chunks (paper_id, chunk_id)
                   SELECT DISTINCT paper_id, chunk_id FROM entities WHERE paper_id=?""",
                (paper_id,),
            )
            rows = conn.execute(
                "SELECT chunk_id FROM processed_chunks WHERE paper_id=?", (paper_id,)
            ).fetchall()
        return {r["chunk_id"] for r in rows}

    def mark_chunk_processed(self, paper_id: int, chunk_id: int) -> None:
        """Mark a chunk as fully processed by Phase 2."""
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO processed_chunks (paper_id, chunk_id) VALUES (?, ?)",
                (paper_id, chunk_id),
            )

    def mark_chunks_processed(self, paper_id: int, chunk_ids: list[int]) -> None:
        """Batch-mark chunks as fully processed by Phase 2."""
        if not chunk_ids:
            return
        with self._connect() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO processed_chunks (paper_id, chunk_id) VALUES (?, ?)",
                [(paper_id, cid) for cid in chunk_ids],
            )

    def paper_is_phase2_complete(self, paper_id: int) -> bool:
        """Check whether all chunks for a paper have completed Phase 2."""
        with self._connect() as conn:
            total = conn.execute(
                "SELECT COUNT(*) as cnt FROM chunks WHERE paper_id=?", (paper_id,)
            ).fetchone()["cnt"]
            if total == 0:
                return False
            done = conn.execute(
                "SELECT COUNT(*) as cnt FROM processed_chunks WHERE paper_id=?", (paper_id,)
            ).fetchone()["cnt"]
            return done >= total

    def get_claims_count_for_paper(self, paper_id: int) -> int:
        """Return the number of claims extracted for a paper."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM claims WHERE paper_id=?", (paper_id,)
            ).fetchone()
            return row["cnt"] if row else 0

    def delete_claim_pairs_by_matching_method(self, matching_method: str) -> int:
        """
        Removes claim_pairs rows whose matching_method column matches
        exactly (case-sensitive). Returns the number of rows deleted.

        Exists for correcting data written by evaluation/diagnostic tooling
        into a database it should never have touched (see
        scripts/backfill_paper_metadata.py's sibling cleanup use in
        scripts/clean_contaminated_claim_pairs.py) -- scoped to an exact
        matching_method string rather than any broader/heuristic filter,
        so it can only ever remove rows that self-identify as having come
        from a specific, named source, never rows produced by the real
        Phase3Pipeline (whose matching_method is always
        'embedding→entity-alignment→bidirectional-NLI', see
        matching/claim_matcher.py).
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM claim_pairs WHERE matching_method = ?", (matching_method,)
            )
            deleted = cursor.rowcount
        logger.info("Deleted %d claim_pairs row(s) with matching_method=%r", deleted, matching_method)
        return deleted

    def get_validated_neighbors(self,claim_id,relation):
        with self._connect() as conn:
            rows=conn.execute("""SELECT cp.*, CASE WHEN cp.claim_a_id=? THEN cp.claim_b_id ELSE cp.claim_a_id END neighbor_id, c.text neighbor_text, c.paper_id neighbor_paper_id FROM claim_pairs cp JOIN claims c ON c.claim_id=CASE WHEN cp.claim_a_id=? THEN cp.claim_b_id ELSE cp.claim_a_id END WHERE (cp.claim_a_id=? OR cp.claim_b_id=?) AND cp.validated_as=?""",(claim_id,claim_id,claim_id,claim_id,relation)).fetchall()
            out=[]
            for r in rows:
                score=self.get_current_aerf(r["neighbor_id"])
                out.append({**dict(r),"neighbor_score": None if score is None else score["score"]})
        return out

    def get_direct_neighbor_ids(self,claim_id):
        with self._connect() as conn:
            rows=conn.execute("SELECT claim_a_id,claim_b_id FROM claim_pairs WHERE claim_a_id=? OR claim_b_id=?",(claim_id,claim_id)).fetchall()
        return {r["claim_b_id"] if r["claim_a_id"]==claim_id else r["claim_a_id"] for r in rows}

    def get_claim_pair_confidences(self,claim_id):
        with self._connect() as conn:
            rows=conn.execute("SELECT nli_confidence FROM claim_pairs WHERE claim_a_id=? OR claim_b_id=?",(claim_id,claim_id)).fetchall()
        return [r["nli_confidence"] for r in rows if r["nli_confidence"] is not None]

    def provenance_completeness(self,claim_id):
        with self._connect() as conn:
            r=conn.execute("""SELECT c.paper_id,c.chunk_id,c.span_start,c.span_end,ch.page_number,ch.char_start,ch.char_end FROM claims c JOIN chunks ch ON ch.chunk_id=c.chunk_id WHERE c.claim_id=?""",(claim_id,)).fetchone()
        if not r:return 0.0
        fields=[r[x] for x in ("paper_id","chunk_id","page_number","span_start","span_end","char_start","char_end")]
        return sum(x is not None for x in fields)/len(fields)

    def get_current_aerf(self,claim_id):
        import json
        with self._connect() as conn:
            r=conn.execute("SELECT * FROM aerf_scores WHERE claim_id=? AND is_current=1 ORDER BY score_id DESC LIMIT 1",(claim_id,)).fetchone()
        if not r:return None
        d=dict(r); d["factor_breakdown"]=json.loads(d["factor_breakdown"]); return d

    def store_aerf_score(self,claim_id,score,factors,version):
        import json
        with self._connect() as conn:
            conn.execute("UPDATE aerf_scores SET is_current=0 WHERE claim_id=?",(claim_id,))
            conn.execute("INSERT INTO aerf_scores(claim_id,score,factor_breakdown,weights_version,is_current,score_source) VALUES(?,?,?,?,1,?)",(claim_id,float(score),json.dumps(factors,sort_keys=True),version,"evidence-derived"))

    def store_aerf_history(self,claim_id,old_score,new_score,paper_id,claim_trigger,deltas,reason):
        import json
        with self._connect() as conn:
            conn.execute("INSERT INTO aerf_score_history(claim_id,old_score,new_score,triggered_by_paper_id,triggered_by_claim_id,factor_deltas,reason) VALUES(?,?,?,?,?,?,?)",(claim_id,old_score,new_score,paper_id,claim_trigger,json.dumps(deltas,sort_keys=True),reason))

    def upsert_graph_edge(self,sn,si,tn,ti,etype,weight,evidence_id=None):
        with self._connect() as conn:
            conn.execute("INSERT OR IGNORE INTO evidence_graph_edges(source_node_type,source_node_id,target_node_type,target_node_id,edge_type,weight,evidence_id) VALUES(?,?,?,?,?,?,?)",(sn,si,tn,ti,etype,weight,evidence_id))

    def get_claim_pair(self,a,b):
        with self._connect() as conn:
            r=conn.execute("SELECT * FROM claim_pairs WHERE claim_a_id=? AND claim_b_id=?",(min(a,b),max(a,b))).fetchone()
        return dict(r) if r else None

    def store_evolution(self,topic,year,dominant,score,support,contradict,evidence):
        import json
        with self._connect() as conn:
            conn.execute("INSERT INTO knowledge_evolution_snapshots(topic_key,as_of_year,dominant_claim_id,dominant_aerf_score,supporting_claim_count,contradicting_claim_count,evidence_json) VALUES(?,?,?,?,?,?,?)",(topic,year,dominant,score,support,contradict,json.dumps(evidence,sort_keys=True)))

    def store_gap(self,topic,gap_type,description,claim_ids,evidence):
        import json
        with self._connect() as conn:
            conn.execute("INSERT INTO research_gaps(topic_key,gap_type,description,related_claim_ids,evidence_json) VALUES(?,?,?,?,?)",(topic,gap_type,description,json.dumps(claim_ids),json.dumps(evidence,sort_keys=True)))

    def store_explanation(self,claim_id,etype,content):
        import json
        with self._connect() as conn:
            conn.execute("INSERT INTO explanations(claim_id,explanation_type,content_json) VALUES(?,?,?)",(claim_id,etype,json.dumps(content,sort_keys=True)))

    def get_all_claims(self):
        with self._connect() as conn:
            rows=conn.execute("SELECT * FROM claims ORDER BY claim_id").fetchall()
        return [dict(r) for r in rows]

    def get_chunk(self,chunk_id):
        with self._connect() as conn:
            r=conn.execute("SELECT * FROM chunks WHERE chunk_id=?",(chunk_id,)).fetchone()
        return dict(r) if r else None

    def get_claim_page(self,claim_id):
        with self._connect() as conn:
            r=conn.execute("SELECT ch.page_number FROM claims c JOIN chunks ch ON ch.chunk_id=c.chunk_id WHERE c.claim_id=?",(claim_id,)).fetchone()
        return None if r is None else r['page_number']

    def get_evolution_snapshots(self):
        with self._connect() as conn:
            return [dict(r) for r in conn.execute("SELECT * FROM knowledge_evolution_snapshots ORDER BY topic_key,as_of_year,snapshot_id").fetchall()]

    def ensure_session_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS analysis_sessions (
                    session_id          TEXT PRIMARY KEY,
                    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    name                TEXT,
                    status              TEXT DEFAULT 'completed',
                    paper_count         INTEGER DEFAULT 0,
                    error_message       TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_papers (
                    session_id          TEXT NOT NULL,
                    paper_id            INTEGER NOT NULL,
                    source_filename     TEXT,
                    PRIMARY KEY (session_id, paper_id)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_session_papers_session ON session_papers(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_session_papers_paper ON session_papers(paper_id)")

    def create_session(self, session_id: str, name: str | None = None, status: str = 'processing', paper_count: int = 0) -> dict:
        self.ensure_session_schema()
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO analysis_sessions(session_id, created_at, name, status, paper_count) VALUES(?, CURRENT_TIMESTAMP, ?, ?, ?)",
                (session_id, name, status, paper_count)
            )
        return self.get_session(session_id)

    def update_session(self, session_id: str, status: str | None = None, paper_count: int | None = None, error_message: str | None = None) -> None:
        self.ensure_session_schema()
        updates = []
        params = []
        if status is not None:
            updates.append("status = ?")
            params.append(status)
            if status == "completed":
                updates.append("error_message = NULL")
        if paper_count is not None:
            updates.append("paper_count = ?")
            params.append(paper_count)
        if error_message is not None:
            updates.append("error_message = ?")
            params.append(error_message)
        if updates:
            params.append(session_id)
            sql = f"UPDATE analysis_sessions SET {', '.join(updates)} WHERE session_id = ?"
            with self._connect() as conn:
                conn.execute(sql, tuple(params))

    def link_session_paper(self, session_id: str, paper_id: int, source_filename: str | None = None) -> None:
        self.ensure_session_schema()
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO session_papers(session_id, paper_id, source_filename) VALUES(?, ?, ?)",
                (session_id, paper_id, source_filename)
            )

    def get_session(self, session_id: str) -> dict | None:
        self.ensure_session_schema()
        with self._connect() as conn:
            r = conn.execute("SELECT * FROM analysis_sessions WHERE session_id = ?", (session_id,)).fetchone()
            if not r:
                return None
            res = dict(r)
            papers = conn.execute(
                "SELECT p.paper_id, p.source_filename, p.title, p.publication_year FROM session_papers sp JOIN papers p ON sp.paper_id = p.paper_id WHERE sp.session_id = ? ORDER BY p.paper_id",
                (session_id,)
            ).fetchall()
            res['papers'] = [dict(p) for p in papers]
            res['paper_ids'] = [p['paper_id'] for p in res['papers']]
            return res

    def list_sessions(self) -> list[dict]:
        self.ensure_session_schema()
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM analysis_sessions ORDER BY created_at DESC").fetchall()
            res = []
            for r in rows:
                d = dict(r)
                papers = conn.execute(
                    "SELECT p.paper_id, p.source_filename, p.title FROM session_papers sp JOIN papers p ON sp.paper_id = p.paper_id WHERE sp.session_id = ? ORDER BY p.paper_id",
                    (d['session_id'],)
                ).fetchall()
                d['papers'] = [dict(p) for p in papers]
                d['paper_ids'] = [p['paper_id'] for p in d['papers']]
                res.append(d)
            return res

    def get_session_paper_ids(self, session_id: str) -> list[int]:
        self.ensure_session_schema()
        with self._connect() as conn:
            rows = conn.execute("SELECT paper_id FROM session_papers WHERE session_id = ? ORDER BY paper_id", (session_id,)).fetchall()
        return [r['paper_id'] for r in rows]

    def get_session_papers(self, session_id: str) -> list[Paper]:
        pids = self.get_session_paper_ids(session_id)
        return [self.get_paper_by_id(pid) for pid in pids if self.get_paper_by_id(pid) is not None]

    def get_claims_by_paper_ids(self, paper_ids: list[int] | set[int]) -> list[dict]:
        pids = list(paper_ids)
        if not pids:
            return []
        placeholders = ','.join('?' for _ in pids)
        sql = f"""
            SELECT c.claim_id, c.paper_id, c.chunk_id, c.text, c.span_start, c.span_end, c.extracted_at,
                   p.source_filename, p.title, p.publication_year,
                   ch.page_number
            FROM claims c
            JOIN papers p ON c.paper_id = p.paper_id
            JOIN chunks ch ON c.chunk_id = ch.chunk_id
            WHERE c.paper_id IN ({placeholders})
            ORDER BY c.paper_id, c.claim_id
        """
        with self._connect() as conn:
            rows = conn.execute(sql, tuple(pids)).fetchall()
        return [dict(r) for r in rows]

    def get_claim_pairs_by_paper_ids(self, paper_ids: list[int] | set[int]) -> list[dict]:
        pids = list(paper_ids)
        if not pids:
            return []
        placeholders = ','.join('?' for _ in pids)
        sql = f"""
            SELECT cp.*,
                   ca.paper_id AS paper_a_id, cb.paper_id AS paper_b_id,
                   pa.source_filename AS paper_a_filename, pb.source_filename AS paper_b_filename,
                   ca.text AS claim_a_text, cb.text AS claim_b_text
            FROM claim_pairs cp
            JOIN claims ca ON cp.claim_a_id = ca.claim_id
            JOIN claims cb ON cp.claim_b_id = cb.claim_id
            JOIN papers pa ON ca.paper_id = pa.paper_id
            JOIN papers pb ON cb.paper_id = pb.paper_id
            WHERE ca.paper_id IN ({placeholders}) AND cb.paper_id IN ({placeholders})
            ORDER BY cp.pair_id
        """
        with self._connect() as conn:
            rows = conn.execute(sql, tuple(pids) + tuple(pids)).fetchall()
        return [dict(r) for r in rows]

    def get_aerf_scores_by_paper_ids(self, paper_ids: list[int] | set[int]) -> list[dict]:
        import json
        pids = list(paper_ids)
        if not pids:
            return []
        placeholders = ','.join('?' for _ in pids)
        sql = f"""
            SELECT s.score_id, s.claim_id, s.score, s.factor_breakdown, s.weights_version, s.computed_at,
                   c.paper_id, c.text AS claim_text, p.source_filename
            FROM aerf_scores s
            JOIN claims c ON s.claim_id = c.claim_id
            JOIN papers p ON c.paper_id = p.paper_id
            WHERE c.paper_id IN ({placeholders}) AND s.is_current = 1
            ORDER BY s.claim_id
        """
        with self._connect() as conn:
            rows = conn.execute(sql, tuple(pids)).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            if isinstance(d.get('factor_breakdown'), str):
                try:
                    d['factor_breakdown'] = json.loads(d['factor_breakdown'])
                except Exception:
                    pass
            result.append(d)
        return result

    def get_graph_edges_by_paper_ids(self, paper_ids: list[int] | set[int]) -> list[dict]:
        pids = list(paper_ids)
        if not pids:
            return []
        placeholders = ','.join('?' for _ in pids)
        sql = f"""
            SELECT e.*
            FROM evidence_graph_edges e
            WHERE (e.source_node_type = 'claim' AND e.source_node_id IN (SELECT claim_id FROM claims WHERE paper_id IN ({placeholders})))
               OR (e.target_node_type = 'claim' AND e.target_node_id IN (SELECT claim_id FROM claims WHERE paper_id IN ({placeholders})))
            ORDER BY e.edge_id
        """
        with self._connect() as conn:
            rows = conn.execute(sql, tuple(pids) + tuple(pids)).fetchall()
        return [dict(r) for r in rows]

    def get_gaps_by_paper_ids(self, paper_ids: list[int] | set[int]) -> list[dict]:
        import json
        pids = list(paper_ids)
        if not pids:
            return []
        claims = self.get_claims_by_paper_ids(pids)
        claim_id_set = {str(c['claim_id']) for c in claims}
        int_claim_id_set = {c['claim_id'] for c in claims}
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM research_gaps ORDER BY gap_id").fetchall()
        matching_gaps = []
        for r in rows:
            d = dict(r)
            raw_ids = d.get('related_claim_ids', '')
            matched = False
            if raw_ids:
                try:
                    parsed = json.loads(raw_ids)
                    if isinstance(parsed, list):
                        if any(x in int_claim_id_set or str(x) in claim_id_set for x in parsed):
                            matched = True
                except Exception:
                    if any(cid in str(raw_ids) for cid in claim_id_set):
                        matched = True
            if matched:
                matching_gaps.append(d)
        return matching_gaps

