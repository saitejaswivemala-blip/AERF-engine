#!/usr/bin/env python3
"""
scripts/inspect_claim_mapping.py

READ-ONLY diagnostic. Produces a table showing, for each of the 8 pairs in
experiments/annotations/ground_truth_claim_pairs.json, what the human
annotation's claim_a/claim_b text actually resolves to among the real
claims already in experiments/runs/experiment.db -- so a human can see
exactly what the resolver matched (or didn't) before trusting any
evaluation built on top of it.

Does NOT modify: the database, matching/, nlp/, pipeline/, aerf/,
experiments/annotations/ground_truth_claim_pairs.json, or
scripts/evaluate_real_results.py. Imports `resolve_claim` from that file
rather than reimplementing it, so the mapping shown here is guaranteed to
be identical to what the evaluator actually uses -- no separate, possibly-
diverging matching logic. Does not lower any threshold and does not force
a mapping: an annotation claim that doesn't clear resolve_claim()'s
existing thresholds is reported as UNRESOLVED, exactly as the evaluator
would treat it, with its best (rejected) candidate shown for transparency.

This script does not call the NLI model, does not run matching, and does
not compute AERF scores -- it only resolves text and, where both sides of
a pair resolve, reports the embedding cosine similarity between the two
resolved claims (using the real, unmodified ClaimMatcher.cosine on the
real, unmodified embeddings already in the database) so you can see
whether a resolved pair would even clear the similarity gate.

The "semantically the same claim?" column is NOT computed by any model or
heuristic here -- automating that judgment would mean either inventing a
verdict or quietly reusing NLI for a task framed as manual inspection.
That column is left for YOU (or whoever reviews this table) to fill in by
reading the human text against the matched DB text side by side, which
this script prints in full specifically to make that easy. Where a pair is
UNRESOLVED, the column is filled with "N/A (unresolved)" rather than left
ambiguous.

Usage:
    python scripts/inspect_claim_mapping.py
    python scripts/inspect_claim_mapping.py --db experiments/runs/experiment.db
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.repository import Repository  # noqa: E402
from matching.claim_matcher import ClaimMatcher  # noqa: E402
from scripts.evaluate_real_results import resolve_claim  # noqa: E402

MANIFEST_PATH = Path("experiments/datasets/paper_manifest.json")
ANNOTATIONS_PATH = Path("experiments/annotations/ground_truth_claim_pairs.json")
REVIEWED_MAPPING_PATH = (
    Path(__file__).resolve().parent.parent
    / "experiments"
    / "annotations"
    / "reviewed_claim_mapping.json"
)


def _fmt_status(res: dict) -> str:
    if res["status"] == "resolved":
        return f"resolved ({res['method']}, score={res['score']})"
    return f"UNRESOLVED ({res.get('reason', 'no reason given')})"


def load_reviewed_mapping(path: Path = REVIEWED_MAPPING_PATH) -> dict | None:
    """Load the optional explicit mapping artifact without changing thresholds."""
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    mappings = data.get("mappings")
    if not isinstance(mappings, dict):
        raise ValueError(f"Reviewed mapping file has no mappings object: {path}")
    return mappings


def resolve_reviewed_claim(pair_id: str, side: str, mappings: dict, db_claims: list[dict]) -> dict:
    """Resolve only an explicitly reviewed ID, validating its stored text."""
    entry = mappings.get(pair_id, {}).get(side)
    if not isinstance(entry, dict):
        return {"status": "unresolved", "reason": "no reviewed mapping entry"}

    claim_id = entry.get("db_claim_id")
    rationale = entry.get("rationale")
    if claim_id is None:
        return {"status": "unresolved", "reason": rationale or "reviewed as unresolved"}

    matches = [claim for claim in db_claims if claim["claim_id"] == claim_id]
    if len(matches) != 1:
        return {"status": "unresolved", "reason": f"reviewed claim_id={claim_id} is not present in this paper"}

    claim = matches[0]
    if claim["text"] != entry.get("db_claim_text"):
        return {"status": "unresolved", "reason": f"reviewed text does not match claim_id={claim_id}"}

    return {
        "status": "resolved",
        "method": "reviewed_mapping",
        "score": 1.0,
        "claim_id": claim_id,
        "matched_text": claim["text"],
        "rationale": rationale,
    }


def resolve_benchmark_claim(pair_id: str, side: str, annotation_text: str,
                            db_claims: list[dict], mappings: dict | None) -> dict:
    """Prefer the complete reviewed artifact; otherwise retain legacy resolution."""
    if mappings is not None:
        return resolve_reviewed_claim(pair_id, side, mappings, db_claims)
    return resolve_claim(annotation_text, db_claims)


def build_table(db_path: Path) -> list[dict]:
    manifest = json.loads(MANIFEST_PATH.read_text())
    annotations = json.loads(ANNOTATIONS_PATH.read_text())
    reviewed_mapping = load_reviewed_mapping()
    repo = Repository(db_path)  # opened read-only in effect: this script issues no writes

    manifest_to_db_paper = {}
    for p in manifest["papers"]:
        paper = repo.get_paper_by_filename(Path(p["pdf_path"]).name)
        if paper is not None:
            manifest_to_db_paper[p["paper_id"]] = paper.paper_id

    claims_by_db_paper = {
        db_pid: repo.get_claims_for_paper(db_pid) for db_pid in set(manifest_to_db_paper.values())
    }

    rows = []
    for rec in annotations:
        pa_db = manifest_to_db_paper.get(rec["paper_a"])
        pb_db = manifest_to_db_paper.get(rec["paper_b"])

        row = {
            "pair_id": rec["pair_id"],
            "paper_a": rec["paper_a"], "paper_b": rec["paper_b"],
            "same_paper": rec["paper_a"] == rec["paper_b"],
            "human_label": rec["human_label"],
            "human_claim_a": rec["claim_a"], "human_claim_b": rec["claim_b"],
        }

        if pa_db is None or pb_db is None:
            row["error"] = f"paper not found in database: " \
                            f"{'paper_a=' + rec['paper_a'] if pa_db is None else ''} " \
                            f"{'paper_b=' + rec['paper_b'] if pb_db is None else ''}".strip()
            rows.append(row)
            continue

        res_a = resolve_benchmark_claim(rec["pair_id"], "claim_a", rec["claim_a"],
                        claims_by_db_paper.get(pa_db, []), reviewed_mapping)
        res_b = resolve_benchmark_claim(rec["pair_id"], "claim_b", rec["claim_b"],
                        claims_by_db_paper.get(pb_db, []), reviewed_mapping)

        row["resolution_a_status"] = _fmt_status(res_a)
        row["resolution_b_status"] = _fmt_status(res_b)
        row["db_claim_a_id"] = res_a.get("claim_id")
        row["db_claim_a_text"] = res_a.get("matched_text")
        row["db_claim_b_id"] = res_b.get("claim_id")
        row["db_claim_b_text"] = res_b.get("matched_text")
        row["db_claim_a_best_rejected_candidate"] = res_a.get("best_candidate") or res_a.get("candidate_claim_ids")
        row["db_claim_b_best_rejected_candidate"] = res_b.get("best_candidate") or res_b.get("candidate_claim_ids")
        row["mapping_a_rationale"] = res_a.get("rationale")
        row["mapping_b_rationale"] = res_b.get("rationale")

        both_resolved = res_a["status"] == "resolved" and res_b["status"] == "resolved"
        if both_resolved:
            va = repo.get_claim_embedding(res_a["claim_id"])
            vb = repo.get_claim_embedding(res_b["claim_id"])
            row["embedding_cosine_similarity"] = round(ClaimMatcher.cosine(va, vb), 4) if va is not None and vb is not None else None
            row["semantically_same_claim"] = "REVIEW MANUALLY (see human_claim_* vs db_claim_*_text above)"
        else:
            row["embedding_cosine_similarity"] = None
            row["semantically_same_claim"] = "N/A (unresolved)"

        rows.append(row)
    return rows


def print_table(rows: list[dict]) -> None:
    for r in rows:
        print("=" * 100)
        print(f"{r['pair_id']}  ({r['paper_a']} vs {r['paper_b']}"
              f"{'  [SAME PAPER]' if r.get('same_paper') else ''})  human_label={r['human_label']}")
        if "error" in r:
            print(f"  ERROR: {r['error']}")
            continue
        print(f"  human claim A : {r['human_claim_a']}")
        print(f"  db match A    : {r.get('db_claim_a_text')!r}  "
              f"(claim_id={r.get('db_claim_a_id')})  [{r['resolution_a_status']}]")
        if r.get("mapping_a_rationale"):
            print(f"  mapping rationale A: {r['mapping_a_rationale']}")
        if r.get("db_claim_a_id") is None:
            print(f"                   best rejected candidate: {r.get('db_claim_a_best_rejected_candidate')}")
        print(f"  human claim B : {r['human_claim_b']}")
        print(f"  db match B    : {r.get('db_claim_b_text')!r}  "
              f"(claim_id={r.get('db_claim_b_id')})  [{r['resolution_b_status']}]")
        if r.get("mapping_b_rationale"):
            print(f"  mapping rationale B: {r['mapping_b_rationale']}")
        if r.get("db_claim_b_id") is None:
            print(f"                   best rejected candidate: {r.get('db_claim_b_best_rejected_candidate')}")
        print(f"  embedding cosine similarity (A vs B, if both resolved): {r.get('embedding_cosine_similarity')}")
        print(f"  semantically the same claim?: {r['semantically_same_claim']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default="experiments/runs/experiment.db")
    parser.add_argument("--json-out", default="experiments/results/claim_mapping_inspection.json",
                         help="Where to also save the table as JSON. Pass '' to skip saving.")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise FileNotFoundError(
            f"{db_path} does not exist. This script only reads an already-populated "
            f"database; it does not create one."
        )

    rows = build_table(db_path)
    print_table(rows)

    n_resolved_both = sum(1 for r in rows if "error" not in r
                          and r.get("db_claim_a_id") is not None and r.get("db_claim_b_id") is not None)
    print("=" * 100)
    print(f"\n{n_resolved_both}/{len(rows)} pairs have BOTH sides resolved to a real DB claim.")
    print("No thresholds were changed and no mapping was forced -- unresolved sides are reported as such.")

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(rows, indent=2, default=str))
        print(f"\nTable also saved to {out_path}")

    return rows


if __name__ == "__main__":
    main()
