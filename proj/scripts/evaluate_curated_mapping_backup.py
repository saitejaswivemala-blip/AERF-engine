#!/usr/bin/env python3
"""
scripts/evaluate_curated_mapping.py

Runs the real, unmodified run_matching() + metrics.evaluate() on exactly
the 4 hand-curated pairs (gt_01, gt_02, gt_03, gt_05) in
experiments/annotations/curated_mapping_gt01_gt02_gt03_gt05.json, using
claim_ids a human has manually confirmed against the real database (see
experiments/reports/curated_mapping_rationale.md).

Does NOT touch gt_04/gt_06 (excluded -- no corresponding P5 claim) or
gt_07/gt_08 (same-paper, out of scope for this file by design). Does NOT
modify experiments/annotations/ground_truth_claim_pairs.json, the curated
mapping file, matching/, nlp/, pipeline/, aerf/, or the database schema --
read-only against the DB (get_claim, get_claim_embedding, get_claim_entities
only), and the mapping file is only read, never written, by this script.

REFUSES TO RUN if any claim_a_id/claim_b_id in the mapping is still null,
or if _status isn't "FILLED_IN" -- this is a hard guard against silently
evaluating an incomplete/template mapping, not just a documentation note.

Also refuses to run if any mapped claim_id does not actually exist in the
database, or exists under a different paper than the mapping's paper_a/
paper_b says -- a manually-entered claim_id could be a typo, and this
script would rather fail loudly than silently evaluate the wrong claim.

The result is explicitly labeled n=4, diagnostic, not a benchmark -- see
experiments/reports/curated_mapping_rationale.md for why, and this
script's own validity_note field in its output for the same statement in
machine-readable form.

Usage:
    python scripts/evaluate_curated_mapping.py
    python scripts/evaluate_curated_mapping.py --db experiments/runs/experiment.db
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.repository import Repository  # noqa: E402
from nlp.nli.nli_classifier import NLIClassifier  # noqa: E402
from experiments.lib.experiment_config import FULL_AERF  # noqa: E402
from experiments.lib.pipeline_runner import run_matching  # noqa: E402
from experiments.lib.metrics import evaluate, evaluation_to_dict  # noqa: E402

MANIFEST_PATH = Path("experiments/datasets/paper_manifest.json")
MAPPING_PATH = Path("experiments/annotations/curated_mapping_gt01_gt02_gt03_gt05.json")
RESULTS_PATH = Path("experiments/results/curated_evaluation_gt01_02_03_05.json")

EXPECTED_PAIR_IDS = {"gt_01", "gt_02", "gt_03", "gt_05"}


def load_and_validate_mapping() -> dict:
    if not MAPPING_PATH.exists():
        raise FileNotFoundError(f"{MAPPING_PATH} not found.")
    mapping = json.loads(MAPPING_PATH.read_text())

    if mapping.get("_status") != "FILLED_IN":
        raise ValueError(
            f"{MAPPING_PATH}'s _status is {mapping.get('_status')!r}, not 'FILLED_IN'. "
            f"This mapping has not been completed with real claim IDs yet -- see "
            f"experiments/reports/curated_mapping_rationale.md for how to fill it in. "
            f"Refusing to run rather than evaluate a template."
        )

    pairs_by_id = {p["pair_id"]: p for p in mapping["pairs"]}
    if set(pairs_by_id) != EXPECTED_PAIR_IDS:
        raise ValueError(f"Expected exactly {EXPECTED_PAIR_IDS}, found {set(pairs_by_id)}.")

    missing = []
    for pid, p in pairs_by_id.items():
        if p.get("claim_a_id") is None:
            missing.append(f"{pid}.claim_a_id")
        if p.get("claim_b_id") is None:
            missing.append(f"{pid}.claim_b_id")
        if not p.get("mapping_rationale") or p["mapping_rationale"].startswith("FILL IN"):
            missing.append(f"{pid}.mapping_rationale")
    if missing:
        raise ValueError(
            f"{MAPPING_PATH} is missing required fields, refusing to guess: {missing}"
        )
    return mapping


def validate_claim_against_paper(repo: Repository, manifest_to_db: dict,
                                  claim_id: int, expected_manifest_paper_id: str, side: str, pair_id: str) -> dict:
    claim = repo.get_claim(claim_id)
    if claim is None:
        raise ValueError(f"{pair_id}.{side}: claim_id={claim_id} does not exist in the database.")
    expected_db_paper_id = manifest_to_db.get(expected_manifest_paper_id)
    if expected_db_paper_id is None:
        raise ValueError(f"{pair_id}.{side}: manifest paper {expected_manifest_paper_id} "
                          f"not resolved in the database at all.")
    if claim["paper_id"] != expected_db_paper_id:
        raise ValueError(
            f"{pair_id}.{side}: claim_id={claim_id} belongs to db paper_id={claim['paper_id']}, "
            f"but the mapping says it should be {expected_manifest_paper_id} "
            f"(db paper_id={expected_db_paper_id}). This looks like a typo in the curated "
            f"mapping -- refusing to run rather than silently evaluate the wrong claim."
        )
    return claim


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default="experiments/runs/experiment.db")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise FileNotFoundError(f"{db_path} does not exist.")

    mapping = load_and_validate_mapping()
    manifest = json.loads(MANIFEST_PATH.read_text())
    repo = Repository(db_path)

    manifest_to_db = {}
    for p in manifest["papers"]:
        paper = repo.get_paper_by_filename(Path(p["pdf_path"]).name)
        if paper is not None:
            manifest_to_db[p["paper_id"]] = paper.paper_id

    pairs_by_id = {p["pair_id"]: p for p in mapping["pairs"]}
    claim_dicts, entries = {}, []
    for pid in sorted(EXPECTED_PAIR_IDS):
        p = pairs_by_id[pid]
        claim_a = validate_claim_against_paper(repo, manifest_to_db, p["claim_a_id"], p["paper_a"], "claim_a", pid)
        claim_b = validate_claim_against_paper(repo, manifest_to_db, p["claim_b_id"], p["paper_b"], "claim_b", pid)
        claim_dicts[p["claim_a_id"]] = claim_a
        claim_dicts[p["claim_b_id"]] = claim_b
        entries.append(p)
        print(f"{pid}: claim_a_id={p['claim_a_id']} (db paper_id={claim_a['paper_id']}), "
              f"claim_b_id={p['claim_b_id']} (db paper_id={claim_b['paper_id']}) -- validated OK")
        print(f"  rationale: {p['mapping_rationale']}")

    print("\nLoading real NLIClassifier() (downloads/loads the real model if not cached)...")
    real_nli = NLIClassifier()

    pairs = [(claim_dicts[p["claim_a_id"]], claim_dicts[p["claim_b_id"]]) for p in entries]
    match_records = run_matching(repo, FULL_AERF, real_nli, pairs)

    per_pair = []
    for p, m in zip(entries, match_records):
        correct = m.predicted_label == p["human_label"]
        per_pair.append({
            "pair_id": p["pair_id"], "paper_a": p["paper_a"], "paper_b": p["paper_b"],
            "claim_a_id": p["claim_a_id"], "claim_b_id": p["claim_b_id"],
            "human_label": p["human_label"], "system_prediction": m.predicted_label,
            "cosine_similarity": m.cosine_similarity, "entity_overlap_score": m.entity_overlap_score,
            "nli_label": m.nli_label, "nli_confidence": m.nli_confidence, "correct": correct,
            "mapping_rationale": p["mapping_rationale"],
        })

    print("\n--- CURATED CROSS-DOCUMENT EVALUATION (n=4, DIAGNOSTIC, NOT A BENCHMARK) ---")
    for p in per_pair:
        mark = "correct" if p["correct"] else "WRONG"
        print(f"  {p['pair_id']}: predicted={p['system_prediction']:12s} human={p['human_label']:12s} "
              f"[{mark}] (sim={p['cosine_similarity']:.3f}, nli={p['nli_label']}/{p['nli_confidence']})")

    metric_records = [{"system_prediction": p["system_prediction"], "human_label": p["human_label"]} for p in per_pair]
    result = evaluate(metric_records, min_n_for_stats=10)
    print(f"\naccuracy={result.accuracy:.3f} macro_f1={result.macro_f1:.3f}")
    for w in result.warnings:
        print(f"WARNING: {w}")

    validity_note = (
        "n=4. This is a curated, non-random subset chosen because these were the pairs with "
        "confirmed real-claim extraction on both sides (see "
        "experiments/reports/curated_mapping_rationale.md). It is NOT a scientifically valid "
        "benchmark of AERF Engine's accuracy: the sample is far below any statistically "
        "meaningful size, was not randomly selected, and a single pair's outcome moves the "
        "headline accuracy by 25 percentage points. Treat this strictly as a diagnostic "
        "spot-check of real NLI + real matching on 4 specific, manually-verified claim pairs -- "
        "report the per-pair predictions above alongside any aggregate number, never the "
        "aggregate number alone."
    )
    print(f"\n{validity_note}")

    output = {
        "db_path": str(db_path), "n": len(per_pair), "diagnostic_only": True,
        "excluded_pairs": mapping["excluded_pairs"],
        "same_paper_diagnostics_only": mapping["same_paper_diagnostics_only"],
        "predictions": per_pair,
        "evaluation": evaluation_to_dict(result),
        "validity_note": validity_note,
        "nli_model_name": real_nli.model_name,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nSaved to {RESULTS_PATH}")
    return output


if __name__ == "__main__":
    main()
