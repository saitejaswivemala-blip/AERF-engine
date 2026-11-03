"""
scripts/diagnose_entity_alignment_v2.py

READ-ONLY diagnostic. Does not write to any database. Does not modify
reviewed_claim_mapping.json. Does not change any threshold.

Compares v1 (exact-string) vs v2 (normalized + alias-aware) entity overlap
for the 4 cross-paper, fully-mapped reviewed benchmark pairs, using the
real entities already stored in the target database. This is the source
of the numbers documented in matching/entity_alignment.py's module
docstring and in the invention-disclosure/audit documents.

Usage:
    python scripts/diagnose_entity_alignment_v2.py --db experiments/runs/experiment.db
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from matching.entity_alignment import align_entities

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_entities(conn, claim_id):
    rows = conn.execute(
        "SELECT e.text FROM claim_entities ce JOIN entities e ON ce.entity_id = e.entity_id "
        "WHERE ce.claim_id = ?",
        (claim_id,),
    ).fetchall()
    return [{"text": r[0]} for r in rows]


def v1_overlap(entities_a, entities_b):
    ea = {e["text"].strip().lower() for e in entities_a}
    eb = {e["text"].strip().lower() for e in entities_b}
    u = ea | eb
    return len(ea & eb) / len(u) if u else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--threshold", type=float, default=0.20)
    parser.add_argument(
        "--mapping",
        default=str(PROJECT_ROOT / "experiments" / "annotations" / "reviewed_claim_mapping.json"),
    )
    parser.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "experiments" / "results" / "entity_alignment_v2_diagnostic.json"),
    )
    args = parser.parse_args()

    mapping = json.loads(Path(args.mapping).read_text())["mappings"]
    conn = sqlite3.connect(args.db)  # opened read-only in effect: no INSERT/UPDATE/DELETE ever issued below

    results = {}
    for pair_id, entry in mapping.items():
        a = entry.get("claim_a", {})
        b = entry.get("claim_b", {})
        ca, cb = a.get("db_claim_id"), b.get("db_claim_id")
        if ca is None or cb is None:
            results[pair_id] = {"status": "UNRESOLVED"}
            continue
        pa = conn.execute("SELECT paper_id FROM claims WHERE claim_id=?", (ca,)).fetchone()
        pb = conn.execute("SELECT paper_id FROM claims WHERE claim_id=?", (cb,)).fetchone()
        if pa is None or pb is None:
            results[pair_id] = {"status": "CLAIM_NOT_FOUND"}
            continue
        if pa[0] == pb[0]:
            results[pair_id] = {"status": "OUT_OF_SCOPE_SAME_PAPER"}
            continue

        entities_a = get_entities(conn, ca)
        entities_b = get_entities(conn, cb)
        v1 = v1_overlap(entities_a, entities_b)
        v2_result = align_entities(entities_a, entities_b, threshold=args.threshold)

        results[pair_id] = {
            "status": "EVALUATED_CROSS_PAPER",
            "claim_a_id": ca,
            "claim_b_id": cb,
            "v1_overlap": round(v1, 4),
            "v1_passed": v1 >= args.threshold,
            "v2_overlap": round(v2_result.overlap_score, 4),
            "v2_passed": v2_result.passed,
            "v2_matches": [
                {
                    "source": m.source_entity,
                    "target": m.target_entity,
                    "category": m.category.value,
                    "score": m.score,
                    "reason": m.reason,
                }
                for m in v2_result.matches
            ],
            "v2_rejection_reason": v2_result.rejection_reason,
        }

    output = {
        "threshold": args.threshold,
        "source_database": args.db,
        "note": "Diagnostic only. Read-only: no database or mapping file was modified. "
                "n=4 evaluable cross-paper pairs -- not a statistically valid benchmark.",
        "results": results,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))
    print(f"\nWritten to: {out_path}")


if __name__ == "__main__":
    main()
