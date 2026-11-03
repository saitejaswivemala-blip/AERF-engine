#!/usr/bin/env python3
"""
scripts/inspect_top_matches.py

READ-ONLY diagnostic, in two parts.

PART 1: prints every claim currently in experiments/runs/experiment.db,
grouped by paper (P1..P5 per experiments/datasets/paper_manifest.json),
with claim_id and exact text. No filtering, no ranking -- a plain listing.

PART 2: for each *unique* claim text used across the 8 ground-truth pairs
in experiments/annotations/ground_truth_claim_pairs.json, ranks that
claim's own paper's DB claims by TEXT similarity and prints the top 2-3
candidates with their scores -- not just the single best match
scripts/evaluate_real_results.py's resolve_claim() would accept or reject,
but the full short-list, so a human can see what else was close.

IMPORTANT SCOPE NOTE: "similarity" here is LEXICAL/TEXT similarity (token
Jaccard overlap + a character-sequence ratio), not embedding-based semantic
similarity. Ranking the human-written annotation text against real claims
by true semantic (embedding) similarity would require running the
annotation text through the same real sentence-embedding model used
elsewhere in this project -- that's a model invocation, which this task
was scoped as read-only inspection and explicitly not authorized to do.
Lexical similarity is a reasonable proxy (and is exactly what
resolve_claim() itself already uses for its non-exact/non-substring tier)
but is not the same thing as semantic similarity; two claims can be
lexically distant but semantically equivalent, or vice versa. Read the
printed text yourself for the actual semantic judgment -- that's why full
text is printed for every candidate, not just a score.

Does NOT modify: the database, matching/, nlp/, pipeline/, aerf/,
experiments/annotations/ground_truth_claim_pairs.json,
scripts/evaluate_real_results.py, or scripts/inspect_claim_mapping.py.
No thresholds are changed and no mapping is forced or written anywhere --
this only prints and optionally saves a report of what it found.

Usage:
    python scripts/inspect_top_matches.py
    python scripts/inspect_top_matches.py --db experiments/runs/experiment.db --top-k 3
"""
from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.repository import Repository  # noqa: E402
from scripts.evaluate_real_results import _tokens, resolve_claim  # noqa: E402

MANIFEST_PATH = Path("experiments/datasets/paper_manifest.json")
ANNOTATIONS_PATH = Path("experiments/annotations/ground_truth_claim_pairs.json")


def resolve_manifest_papers(repo: Repository) -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text())
    mapping = {}
    for p in manifest["papers"]:
        paper = repo.get_paper_by_filename(Path(p["pdf_path"]).name)
        if paper is not None:
            mapping[p["paper_id"]] = paper.paper_id
        else:
            mapping[p["paper_id"]] = None
    return mapping


def print_all_claims(repo: Repository, manifest_to_db: dict) -> dict:
    """PART 1. Returns {manifest_paper_id: [claim dict, ...]} for reuse in part 2."""
    claims_by_manifest_id = {}
    print("=" * 100)
    print("PART 1 -- all claims currently in the database, grouped by paper")
    print("=" * 100)
    total = 0
    for mpid in sorted(manifest_to_db):
        db_pid = manifest_to_db[mpid]
        if db_pid is None:
            print(f"\n{mpid}: NOT FOUND in database (paper not ingested under the expected filename)")
            claims_by_manifest_id[mpid] = []
            continue
        claims = repo.get_claims_for_paper(db_pid)
        claims_by_manifest_id[mpid] = claims
        total += len(claims)
        print(f"\n{mpid} (db paper_id={db_pid}): {len(claims)} claim(s)")
        for c in claims:
            print(f"  claim_id={c['claim_id']:<5} {c['text']}")
    print(f"\nTOTAL claims across all resolved papers: {total}")
    return claims_by_manifest_id


def rank_candidates(target_text: str, db_claims: list[dict], top_k: int) -> list[dict]:
    target_tokens = _tokens(target_text)
    target_norm = " ".join(target_text.strip().lower().split())
    scored = []
    for c in db_claims:
        db_tokens = _tokens(c["text"])
        union = target_tokens | db_tokens
        jaccard = len(target_tokens & db_tokens) / len(union) if union else 0.0
        db_norm = " ".join(c["text"].strip().lower().split())
        seq_ratio = difflib.SequenceMatcher(None, target_norm, db_norm).ratio()
        scored.append({"claim_id": c["claim_id"], "text": c["text"],
                        "jaccard_token_overlap": round(jaccard, 3),
                        "sequence_ratio": round(seq_ratio, 3)})
    scored.sort(key=lambda d: (d["jaccard_token_overlap"], d["sequence_ratio"]), reverse=True)
    return scored[:top_k]


def collect_unique_ground_truth_claims() -> dict:
    """Returns {(manifest_paper_id, claim_text): [ (pair_id, 'A'|'B'), ... ]}."""
    annotations = json.loads(ANNOTATIONS_PATH.read_text())
    unique = {}
    for rec in annotations:
        key_a = (rec["paper_a"], rec["claim_a"])
        key_b = (rec["paper_b"], rec["claim_b"])
        unique.setdefault(key_a, []).append((rec["pair_id"], "A"))
        unique.setdefault(key_b, []).append((rec["pair_id"], "B"))
    return unique


def print_top_matches(repo: Repository, claims_by_manifest_id: dict, top_k: int) -> list[dict]:
    print("\n" + "=" * 100)
    print(f"PART 2 -- top {top_k} closest DB claims (by lexical/text similarity) for each "
          f"unique ground-truth claim")
    print("=" * 100)
    unique_claims = collect_unique_ground_truth_claims()
    report_rows = []
    for (mpid, text), used_in in unique_claims.items():
        used_str = ", ".join(f"{pid}.{side}" for pid, side in used_in)
        print(f"\n--- used in: {used_str}  (paper {mpid}) ---")
        print(f"human text: {text}")
        db_claims = claims_by_manifest_id.get(mpid, [])
        if not db_claims:
            print(f"  (no claims found for {mpid} in the database)")
            report_rows.append({"paper": mpid, "human_text": text, "used_in": used_str, "candidates": []})
            continue
        top = rank_candidates(text, db_claims, top_k)
        resolver_outcome = resolve_claim(text, db_claims)
        print(f"  [resolve_claim() would give: "
              f"{'resolved -> claim_id=' + str(resolver_outcome['claim_id']) + ' (' + resolver_outcome['method'] + ')' if resolver_outcome['status']=='resolved' else 'UNRESOLVED (' + resolver_outcome.get('reason','') + ')'}]")
        for i, cand in enumerate(top, 1):
            marker = " <-- resolve_claim() pick" if (resolver_outcome["status"] == "resolved"
                                                       and cand["claim_id"] == resolver_outcome.get("claim_id")) else ""
            print(f"  #{i} claim_id={cand['claim_id']:<5} "
                  f"jaccard={cand['jaccard_token_overlap']:.3f} seq_ratio={cand['sequence_ratio']:.3f}{marker}")
            print(f"       {cand['text']}")
        report_rows.append({"paper": mpid, "human_text": text, "used_in": used_str,
                             "resolver_outcome": resolver_outcome, "candidates": top})
    return report_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default="experiments/runs/experiment.db")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--json-out", default="experiments/results/top_matches_inspection.json",
                         help="Pass '' to skip saving.")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise FileNotFoundError(f"{db_path} does not exist. This script only reads an "
                                 f"already-populated database; it does not create one.")

    repo = Repository(db_path)
    manifest_to_db = resolve_manifest_papers(repo)
    claims_by_manifest_id = print_all_claims(repo, manifest_to_db)
    report_rows = print_top_matches(repo, claims_by_manifest_id, args.top_k)

    print("\n" + "=" * 100)
    print("No thresholds were changed, no files were modified, no mapping was written anywhere.")
    print("Text similarity above is lexical (token overlap + sequence ratio), not embedding-based "
          "semantic similarity -- read the printed candidate text yourself for the real judgment.")

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "all_claims_by_paper": {mpid: [dict(c) for c in cs] for mpid, cs in claims_by_manifest_id.items()},
            "ground_truth_top_matches": report_rows,
        }, indent=2, default=str))
        print(f"\nFull report also saved to {out_path}")


if __name__ == "__main__":
    main()
