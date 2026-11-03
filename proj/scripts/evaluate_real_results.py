#!/usr/bin/env python3
"""
scripts/evaluate_real_results.py

Standalone evaluation of the 8 human-annotated ground-truth claim pairs
(experiments/annotations/ground_truth_claim_pairs.json) against REAL system
predictions, using claims/embeddings ALREADY in experiments/runs/experiment.db
(real Phase 1 + real Phase 2 + real Phase 3 having already been run there --
see FINAL_RUN_GUIDE.md). This script does not ingest, does not run Phase 2,
does not run Phase 3's candidate generation, and does not create or modify
the database schema. It only reads.

Pipeline:
  1. Open experiments/runs/experiment.db (must already exist -- refuses to
     run otherwise; never created or recreated here).
  2. Load the 8 ground-truth pairs.
  3. For each pair, resolve paper_a/paper_b ("P1".."P5") to a DB paper_id via
     experiments/datasets/paper_manifest.json + Repository.get_paper_by_filename(),
     then resolve claim_a/claim_b (annotation text) to an actual row in
     Repository.get_claims_for_paper(paper_id) by text matching -- exact
     match first, then substring, then token-overlap similarity above a
     conservative threshold. No claim_id or label is ever invented: a pair
     that can't be confidently resolved is reported as UNRESOLVED, not
     guessed.
  4. Same-paper pairs (paper_a == paper_b, i.e. gt_07/gt_08 here) are
     resolved but evaluated SEPARATELY, never mixed into the main metrics.
     Reason: Phase3Pipeline.run() (pipeline/phase3.py)'s candidate-generation
     loop explicitly skips same-paper claim pairs
     (`if a['paper_id']==b['paper_id']: continue`), so same-paper claims are
     outside what the production cross-document matcher is designed to
     evaluate. experiments/lib/pipeline_runner.run_matching() itself has no
     such guard (it processes whatever explicit pairs it's given), so it CAN
     technically be run on them -- this script does so, for diagnostic
     visibility only, and keeps the result in a clearly separate section of
     the output, excluded from the headline accuracy/F1 numbers.
  5. Resolved, cross-document pairs are run through the REAL, unmodified
     experiments.lib.pipeline_runner.run_matching(), using
     experiments.lib.experiment_config.FULL_AERF (the complete baseline
     matching pipeline: similarity gate -> entity-alignment gate ->
     bidirectional NLI) and a REAL nlp.nli.nli_classifier.NLIClassifier() --
     never a fake/test double.
  6. Predictions are scored against human_label with
     experiments.lib.metrics.evaluate() / evaluation_to_dict().
  7. Full results (including every unresolved/excluded pair, clearly
     labeled) are saved to experiments/results/real_ground_truth_evaluation.json.

Does not modify matching/, nlp/, pipeline/, aerf/, or the database schema.
Does not delete or recreate experiment.db.

Usage:
    python scripts/evaluate_real_results.py
    python scripts/evaluate_real_results.py --db experiments/runs/experiment.db
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.repository import Repository  # noqa: E402
from nlp.nli.nli_classifier import NLIClassifier  # noqa: E402
from experiments.lib.experiment_config import FULL_AERF  # noqa: E402
from experiments.lib.pipeline_runner import run_matching  # noqa: E402
from experiments.lib.metrics import evaluate, evaluation_to_dict  # noqa: E402

MANIFEST_PATH = Path("experiments/datasets/paper_manifest.json")
ANNOTATIONS_PATH = Path("experiments/annotations/ground_truth_claim_pairs.json")
RESULTS_PATH = Path("experiments/results/real_ground_truth_evaluation.json")

_WORD_RE = re.compile(r"[a-z0-9]+")


def _normalise(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def resolve_claim(annotation_text: str, db_claims: list[dict]) -> dict:
    """
    Try to resolve annotation_text to one row in db_claims (each a dict
    with 'claim_id' and 'text', from Repository.get_claims_for_paper()).
    Returns a dict describing the outcome -- never guesses past the stated
    thresholds, and always reports the method/score used so the match is
    auditable rather than opaque.
    """
    norm_target = _normalise(annotation_text)
    target_tokens = _tokens(annotation_text)

    exact = [c for c in db_claims if _normalise(c["text"]) == norm_target]
    if len(exact) == 1:
        return {"status": "resolved", "method": "exact", "score": 1.0,
                "claim_id": exact[0]["claim_id"], "matched_text": exact[0]["text"]}
    if len(exact) > 1:
        return {"status": "unresolved", "reason": "multiple exact text matches (ambiguous)",
                "candidate_claim_ids": [c["claim_id"] for c in exact]}

    substring = [c for c in db_claims
                 if norm_target in _normalise(c["text"]) or _normalise(c["text"]) in norm_target]
    if len(substring) == 1:
        return {"status": "resolved", "method": "substring", "score": 1.0,
                "claim_id": substring[0]["claim_id"], "matched_text": substring[0]["text"]}
    if len(substring) > 1:
        return {"status": "unresolved", "reason": "multiple substring matches (ambiguous)",
                "candidate_claim_ids": [c["claim_id"] for c in substring]}

    scored = []
    for c in db_claims:
        db_tokens = _tokens(c["text"])
        union = target_tokens | db_tokens
        jaccard = len(target_tokens & db_tokens) / len(union) if union else 0.0
        scored.append((jaccard, c))
    scored.sort(key=lambda t: t[0], reverse=True)

    JACCARD_THRESHOLD = 0.5
    if scored and scored[0][0] >= JACCARD_THRESHOLD:
        best_score, best_claim = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else 0.0
        if best_score - second_score < 0.05 and second_score >= JACCARD_THRESHOLD:
            return {"status": "unresolved",
                    "reason": f"top two token-overlap matches too close to call "
                               f"({best_score:.2f} vs {second_score:.2f})",
                    "candidate_claim_ids": [scored[0][1]["claim_id"], scored[1][1]["claim_id"]]}
        return {"status": "resolved", "method": "token_overlap", "score": round(best_score, 3),
                "claim_id": best_claim["claim_id"], "matched_text": best_claim["text"]}

    return {"status": "unresolved",
            "reason": f"no candidate reached the token-overlap threshold "
                      f"({JACCARD_THRESHOLD}); best was "
                      f"{scored[0][0]:.3f} against claim_id={scored[0][1]['claim_id']}"
                      if scored else "paper has no claims in the database",
            "best_candidate": (scored[0][1]["claim_id"] if scored else None)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default="experiments/runs/experiment.db",
                         help="Path to the EXISTING database with real Phase 1/2/3 output. "
                              "Never created or recreated by this script.")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise FileNotFoundError(
            f"{db_path} does not exist. This script only reads an already-populated "
            f"database (real Phase 1 ingestion + real Phase 2 already run) -- it does not "
            f"ingest, run Phase 2, or create the database itself. See FINAL_RUN_GUIDE.md."
        )
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"{MANIFEST_PATH} not found.")
    if not ANNOTATIONS_PATH.exists():
        raise FileNotFoundError(f"{ANNOTATIONS_PATH} not found.")

    manifest = json.loads(MANIFEST_PATH.read_text())
    annotations = json.loads(ANNOTATIONS_PATH.read_text())
    repo = Repository(db_path)

    # Step 1: resolve each manifest paper_id ("P1".."P5") to a DB paper_id.
    manifest_to_db_paper = {}
    for p in manifest["papers"]:
        filename = Path(p["pdf_path"]).name
        paper = repo.get_paper_by_filename(filename)
        if paper is not None:
            manifest_to_db_paper[p["paper_id"]] = paper.paper_id
    print(f"Resolved {len(manifest_to_db_paper)}/{len(manifest['papers'])} manifest papers "
          f"to DB paper_ids: {manifest_to_db_paper}")

    # Cache each resolved paper's claims once.
    claims_by_db_paper = {
        db_pid: repo.get_claims_for_paper(db_pid) for db_pid in set(manifest_to_db_paper.values())
    }
    for mpid, db_pid in manifest_to_db_paper.items():
        print(f"  {mpid} -> db paper_id={db_pid}: {len(claims_by_db_paper[db_pid])} claim(s) in DB")

    # Step 2: resolve every annotation's claim_a / claim_b.
    resolved_pairs, unresolved_pairs = [], []
    for rec in annotations:
        pa_db = manifest_to_db_paper.get(rec["paper_a"])
        pb_db = manifest_to_db_paper.get(rec["paper_b"])
        entry = {"pair_id": rec["pair_id"], "paper_a": rec["paper_a"], "paper_b": rec["paper_b"],
                 "human_label": rec["human_label"], "same_paper": rec["paper_a"] == rec["paper_b"]}

        if pa_db is None or pb_db is None:
            entry["status"] = "unresolved"
            entry["reason"] = "paper_a or paper_b not found in database"
            unresolved_pairs.append(entry)
            continue

        res_a = resolve_claim(rec["claim_a"], claims_by_db_paper.get(pa_db, []))
        res_b = resolve_claim(rec["claim_b"], claims_by_db_paper.get(pb_db, []))
        entry["resolution_a"] = res_a
        entry["resolution_b"] = res_b

        if res_a["status"] != "resolved" or res_b["status"] != "resolved":
            entry["status"] = "unresolved"
            entry["reason"] = "claim_a and/or claim_b could not be confidently resolved to a DB claim"
            unresolved_pairs.append(entry)
            continue

        entry["status"] = "resolved"
        entry["claim_a_id"] = res_a["claim_id"]
        entry["claim_b_id"] = res_b["claim_id"]
        resolved_pairs.append(entry)

    print(f"\n{len(resolved_pairs)}/{len(annotations)} ground-truth pairs resolved to real DB claims.")
    if unresolved_pairs:
        print(f"{len(unresolved_pairs)} pair(s) UNRESOLVED (excluded from evaluation, not guessed):")
        for e in unresolved_pairs:
            print(f"  {e['pair_id']}: {e.get('reason')}")

    cross_document = [e for e in resolved_pairs if not e["same_paper"]]
    same_paper = [e for e in resolved_pairs if e["same_paper"]]
    if same_paper:
        print(f"\n{len(same_paper)} resolved pair(s) are SAME-PAPER "
              f"({[e['pair_id'] for e in same_paper]}) -- excluded from the headline "
              f"cross-document metrics per Phase3Pipeline.run()'s same-paper skip; "
              f"evaluated separately below for diagnostic visibility only.")

    # Step 3: real NLI, real matching -- cross-document pairs only.
    print("\nLoading real NLIClassifier() (downloads/loads the real model if not cached)...")
    real_nli = NLIClassifier()

    def run_group(entries, label):
        if not entries:
            return None, []
        claims = {c["claim_id"]: c for pid in set(manifest_to_db_paper.values())
                  for c in claims_by_db_paper[pid]}
        pairs = [(claims[e["claim_a_id"]], claims[e["claim_b_id"]]) for e in entries]
        match_records = run_matching(repo, FULL_AERF, real_nli, pairs)
        per_pair = []
        for e, m in zip(entries, match_records):
            per_pair.append({
                "pair_id": e["pair_id"], "paper_a": e["paper_a"], "paper_b": e["paper_b"],
                "claim_a_id": e["claim_a_id"], "claim_b_id": e["claim_b_id"],
                "human_label": e["human_label"], "system_prediction": m.predicted_label,
                "cosine_similarity": m.cosine_similarity, "entity_overlap_score": m.entity_overlap_score,
                "nli_label": m.nli_label, "nli_confidence": m.nli_confidence,
                "correct": m.predicted_label == e["human_label"],
            })
        metric_records = [{"system_prediction": p["system_prediction"], "human_label": p["human_label"]}
                           for p in per_pair]
        eval_result = evaluate(metric_records, min_n_for_stats=10) if metric_records else None
        print(f"\n--- {label} ({len(per_pair)} pair(s)) ---")
        for p in per_pair:
            mark = "correct" if p["correct"] else "WRONG"
            print(f"  {p['pair_id']}: predicted={p['system_prediction']:12s} "
                  f"human={p['human_label']:12s} [{mark}] "
                  f"(sim={p['cosine_similarity']:.3f}, nli={p['nli_label']}/{p['nli_confidence']})")
        if eval_result:
            print(f"  accuracy={eval_result.accuracy:.3f} macro_f1={eval_result.macro_f1:.3f}")
            for w in eval_result.warnings:
                print(f"  WARNING: {w}")
        return eval_result, per_pair

    cross_doc_eval, cross_doc_predictions = run_group(cross_document, "CROSS-DOCUMENT (headline metrics)")
    same_paper_eval, same_paper_predictions = run_group(same_paper, "SAME-PAPER (diagnostic only, NOT headline)")

    n_evaluable = len(cross_document)
    n_total_cross_doc_annotations = len([r for r in annotations if r["paper_a"] != r["paper_b"]])
    validity_note = (
        f"{n_evaluable}/{n_total_cross_doc_annotations} cross-document ground-truth pairs were "
        f"actually resolved and evaluated."
    )
    if n_evaluable < n_total_cross_doc_annotations:
        validity_note += (
            " This is NOT a scientifically valid accuracy/F1 measurement of AERF Engine: "
            "some ground-truth pairs could not be matched to real extracted claims and were "
            "excluded rather than guessed (see 'unresolved_pairs' below for exactly which "
            "ones and why). Any accuracy/F1 figure above reflects only the subset that was "
            "resolvable, on an already-tiny (n<10) ground-truth set -- treat it as a "
            "diagnostic spot-check, not a benchmark result, regardless of its value."
        )
    else:
        validity_note += (
            " All cross-document ground-truth pairs were resolved and evaluated. The sample "
            "is still small (n<10); metrics.evaluate()'s own min_n_for_stats warning (see "
            "'warnings' in cross_document_evaluation below) applies regardless of whether "
            "every pair resolved."
        )
    print(f"\n{validity_note}")

    output = {
        "db_path": str(db_path),
        "manifest_to_db_paper": manifest_to_db_paper,
        "n_ground_truth_pairs": len(annotations),
        "n_resolved": len(resolved_pairs),
        "n_unresolved": len(unresolved_pairs),
        "unresolved_pairs": unresolved_pairs,
        "cross_document_evaluation": evaluation_to_dict(cross_doc_eval) if cross_doc_eval else None,
        "cross_document_predictions": cross_doc_predictions,
        "same_paper_evaluation_DIAGNOSTIC_ONLY": evaluation_to_dict(same_paper_eval) if same_paper_eval else None,
        "same_paper_predictions_DIAGNOSTIC_ONLY": same_paper_predictions,
        "validity_note": validity_note,
        "nli_model_name": real_nli.model_name,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nFull results saved to {RESULTS_PATH}")
    return output


if __name__ == "__main__":
    main()
