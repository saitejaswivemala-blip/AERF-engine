
import sqlite3
from pathlib import Path

# Keep this pointed at the TEST database.
DB = Path("experiments/runs/experiment_FINAL_2026-09-02.db")


def main():
    if not DB.exists():
        raise FileNotFoundError(f"Database not found: {DB}")

    con = sqlite3.connect(DB)
    con.execute("PRAGMA foreign_keys = ON")

    try:
        con.execute("BEGIN IMMEDIATE")

        duplicates = con.execute("""
            SELECT e.entity_id,
                   (
                       SELECT MIN(x.entity_id)
                       FROM entities x
                       WHERE x.paper_id = e.paper_id
                         AND x.chunk_id = e.chunk_id
                         AND x.text = e.text
                         AND x.label = e.label
                         AND x.span_start = e.span_start
                         AND x.span_end = e.span_end
                   )
            FROM entities e
            WHERE EXISTS (
                SELECT 1
                FROM entities x
                WHERE x.paper_id = e.paper_id
                  AND x.chunk_id = e.chunk_id
                  AND x.text = e.text
                  AND x.label = e.label
                  AND x.span_start = e.span_start
                  AND x.span_end = e.span_end
                  AND x.entity_id < e.entity_id
            )
            ORDER BY e.entity_id
        """).fetchall()

        print("Duplicate entity IDs to merge:", len(duplicates))

        for old_id, keep_id in duplicates:
            # Preserve claim-to-entity links.
            claim_ids = con.execute(
                "SELECT claim_id FROM claim_entities WHERE entity_id = ?",
                (old_id,),
            ).fetchall()

            for (claim_id,) in claim_ids:
                con.execute("""
                    INSERT OR IGNORE INTO claim_entities
                    (claim_id, entity_id)
                    VALUES (?, ?)
                """, (claim_id, keep_id))

            # Preserve relations after remapping entity IDs.
            relations = con.execute("""
                SELECT claim_id, subject_entity_id,
                       predicate, object_entity_id
                FROM relations
                WHERE subject_entity_id = ?
                   OR object_entity_id = ?
            """, (old_id, old_id)).fetchall()

            for claim_id, subject_id, predicate, object_id in relations:
                if subject_id == old_id:
                    subject_id = keep_id

                if object_id == old_id:
                    object_id = keep_id

                if subject_id != object_id:
                    con.execute("""
                        INSERT OR IGNORE INTO relations
                        (claim_id, subject_entity_id,
                         predicate, object_entity_id)
                        VALUES (?, ?, ?, ?)
                    """, (claim_id, subject_id, predicate, object_id))

            # Remap source entity references in the evidence graph.
            con.execute("""
                UPDATE evidence_graph_edges
                SET source_node_id = ?
                WHERE source_node_type = 'entity'
                  AND source_node_id = ?
            """, (keep_id, old_id))

            # Remap target entity references in the evidence graph.
            con.execute("""
                UPDATE evidence_graph_edges
                SET target_node_id = ?
                WHERE target_node_type = 'entity'
                  AND target_node_id = ?
            """, (keep_id, old_id))

            # Remove old references and duplicate entity.
            con.execute(
                "DELETE FROM claim_entities WHERE entity_id = ?",
                (old_id,),
            )

            con.execute("""
                DELETE FROM relations
                WHERE subject_entity_id = ?
                   OR object_entity_id = ?
            """, (old_id, old_id))

            con.execute(
                "DELETE FROM entities WHERE entity_id = ?",
                (old_id,),
            )

        # Prevent future duplicate entity insertions.
        con.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS uq_entities_identity
            ON entities(chunk_id, span_start, span_end, label, text)
        """)

        # Validate foreign keys.
        fk_errors = con.execute(
            "PRAGMA foreign_key_check"
        ).fetchall()

        if fk_errors:
            raise RuntimeError(
                f"Foreign-key errors found: {fk_errors[:10]}"
            )

        # Validate that duplicate groups are gone.
        remaining = con.execute("""
            SELECT COUNT(*)
            FROM (
                SELECT chunk_id, span_start, span_end, label, text
                FROM entities
                GROUP BY chunk_id, span_start, span_end, label, text
                HAVING COUNT(*) > 1
            )
        """).fetchone()[0]

        if remaining:
            raise RuntimeError(
                f"Duplicate groups still remain: {remaining}"
            )

        con.commit()

        print("\nSUCCESS: Cleanup completed.")
        print("Remaining duplicate groups:", remaining)
        print("Foreign-key check passed.\n")

        tables = [
            "papers",
            "chunks",
            "entities",
            "claims",
            "claim_entities",
            "relations",
            "claim_embeddings",
            "evidence_graph_edges",
            "research_gaps",
            "knowledge_evolution_snapshots",
        ]

        for table in tables:
            exists = con.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type='table' AND name=?",
                (table,),
            ).fetchone()

            if exists:
                count = con.execute(
                    f"SELECT COUNT(*) FROM {table}"
                ).fetchone()[0]
                print(f"{table}: {count}")

    except Exception:
        con.rollback()
        raise

    finally:
        con.close()


if __name__ == "__main__":
    main()

