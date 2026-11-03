#!/usr/bin/env python3
"""Evaluate reviewed benchmark pairs through the existing ClaimMatcher.

This is a small diagnostic evaluation, not a statistically valid benchmark.
It reads exact claims and persisted Phase 2 inputs, invokes ClaimMatcher for
cross-paper pairs, and never writes claim_pairs or any other database table.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.repository import Repository  # noqa: E402
from matching.claim_matcher import ClaimMatcher  # noqa: E402
from nlp.nli.nli_classifier import NLIClassifier  # noqa: E402

ANNOTATIONS_PATH = Path("experiments/annotations/ground_truth_claim_pairs.json")
MAPPING_PATH = Path("experiments/annotations/reviewed_claim_mapping.json")
DEFAULT_DB_PATH = Path("experiments/runs/experiment.db")


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _entity_overlap(entities_a: list[dict], entities_b: list[dict]) -> tuple[float, set[str]]:
    entity_a = {entity["text"].strip().lower() for entity in entities_a}
    entity_b = {entity["text"].strip().lower() for entity in entities_b}
    union = entity_a | entity_b
    overlap = len(entity_a & entity_b) / len(union) if union else 0.0
    return overlap, union


def evaluate_mapping(repo, annotations: list[dict], mapping: dict, matcher=None) -> list[dict]:
    """Return one stage-aware diagnostic row per reviewed benchmark pair."""
    if matcher is None:
        matcher = ClaimMatcher(NLIClassifier(), entity_alignment_version="v2")
    rows = []
    for annotation in annotations:
        pair_id = annotation["pair_id"]
        pair_mapping = mapping.get(pair_id, {})
        mapped_a = pair_mapping.get("claim_a", {})
        mapped_b = pair_mapping.get("claim_b", {})
        claim_a_id = mapped_a.get("db_claim_id")
        claim_b_id = mapped_b.get("db_claim_id")
        row = {
            "pair_id": pair_id,
            "human_label": annotation["human_label"],
            "claim_a_id": claim_a_id,
            "claim_b_id": claim_b_id,
            "predicted_relationship": None,
            "status": "UNRESOLVED",
            "stage": None,
            "correct": None,
        }

        if claim_a_id is None or claim_b_id is None:
            row["stage"] = "unresolved_mapping"
            rows.append(row)
            continue

        claim_a = repo.get_claim(claim_a_id)
        claim_b = repo.get_claim(claim_b_id)
        if claim_a is None or claim_b is None:
            row["stage"] = "unresolved_mapping"
            row["reason"] = "reviewed claim ID was not found in the database"
            rows.append(row)
            continue

        row["paper_a_id"] = claim_a["paper_id"]
        row["paper_b_id"] = claim_b["paper_id"]
        if claim_a["paper_id"] == claim_b["paper_id"]:
            row["status"] = "OUT_OF_SCOPE_SAME_PAPER"
            row["stage"] = "out_of_scope_same_paper"
            rows.append(row)
            continue

        embedding_a = repo.get_claim_embedding(claim_a_id)
        embedding_b = repo.get_claim_embedding(claim_b_id)
        entities_a = repo.get_claim_entities(claim_a_id)
        entities_b = repo.get_claim_entities(claim_b_id)
        relations_a = repo.get_claim_relations(claim_a_id)
        relations_b = repo.get_claim_relations(claim_b_id)
        row["status"] = "EVALUATED"

        if embedding_a is None or embedding_b is None:
            row["stage"] = "missing_embedding"
            rows.append(row)
            continue

        cosine = ClaimMatcher.cosine(embedding_a, embedding_b)
        row["cosine_similarity"] = cosine
        if cosine < matcher.similarity_threshold:
            row["stage"] = "cosine_rejected"
            rows.append(row)
            continue

        entity_overlap, entity_union = _entity_overlap(entities_a, entities_b)
        row["entity_overlap_score"] = entity_overlap
        if entity_union and entity_overlap < matcher.entity_threshold:
            row["stage"] = "entity_alignment_rejected"
            rows.append(row)
            continue

        decision = matcher.match(
            claim_a, claim_b, entities_a, entities_b, relations_a, relations_b,
            embedding_a, embedding_b,
        )
        if decision is None:
            # Defensive guard if matcher gate behavior changes independently.
            row["stage"] = "matcher_rejected"
            rows.append(row)
            continue

        row["stage"] = "nli/classified"
        row["predicted_relationship"] = decision.validated_as
        row["nli_label"] = decision.nli_label
        row["nli_confidence"] = decision.nli_confidence
        row["nli_entailment"] = decision.nli_entailment
        row["nli_contradiction"] = decision.nli_contradiction
        row["nli_neutral"] = decision.nli_neutral
        row["correct"] = decision.validated_as == annotation["human_label"]
        rows.append(row)

    return rows


def summary(rows: list[dict]) -> dict:
    fully_mapped = [row for row in rows if row["claim_a_id"] is not None and row["claim_b_id"] is not None]
    evaluated_cross_paper = [row for row in fully_mapped if row["status"] == "EVALUATED"]
    classified = [row for row in evaluated_cross_paper if row["stage"] == "nli/classified"]
    unresolved = [row for row in rows if row["status"] == "UNRESOLVED"]
    same_paper = [row for row in rows if row["status"] == "OUT_OF_SCOPE_SAME_PAPER"]
    correct = sum(row["correct"] is True for row in classified)
    incorrect = sum(row["correct"] is False for row in classified)
    return {
        "total_pairs": len(rows),
        "fully_mapped": len(fully_mapped),
        "evaluated_cross_paper": len(evaluated_cross_paper),
        "same_paper_out_of_scope": len(same_paper),
        "unresolved": len(unresolved),
        "classified": len(classified),
        "correct": correct,
        "incorrect": incorrect,
        "accuracy_over_evaluated": correct / len(classified) if classified else None,
    }


def print_report(rows: list[dict]) -> None:
    for row in rows:
        if row["status"] == "UNRESOLVED":
            correctness = "UNRESOLVED"
        elif row["status"] == "OUT_OF_SCOPE_SAME_PAPER":
            correctness = "OUT_OF_SCOPE_SAME_PAPER"
        elif row["correct"] is None:
            correctness = "NOT_CLASSIFIED"
        else:
            correctness = "CORRECT" if row["correct"] else "INCORRECT"
        print(
            f"{row['pair_id']}: human={row['human_label']} "
            f"claim_ids=({row['claim_a_id']}, {row['claim_b_id']}) "
            f"stage={row['stage']} predicted={row['predicted_relationship'] or 'N/A'} {correctness}"
        )
        if row.get("nli_label") is not None:
            print(f"  NLI: label={row['nli_label']} confidence={row['nli_confidence']:.4f} "
                  f"entailment={row['nli_entailment']:.4f} "
                  f"contradiction={row['nli_contradiction']:.4f} "
                  f"neutral={row['nli_neutral']:.4f}")

    result = summary(rows)
    print("\nSummary")
    for key in ("total_pairs", "fully_mapped", "evaluated_cross_paper",
                "same_paper_out_of_scope", "unresolved", "classified",
                "correct", "incorrect"):
        print(f"{key}: {result[key]}")
    accuracy = result["accuracy_over_evaluated"]
    print(f"diagnostic_accuracy_over_evaluated: {accuracy:.4f}" if accuracy is not None
          else "diagnostic_accuracy_over_evaluated: N/A")
    print("This is a small diagnostic evaluation, not a statistically valid benchmark.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--mapping", type=Path, default=MAPPING_PATH)
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS_PATH)
    args = parser.parse_args()

    if not args.db.exists():
        raise FileNotFoundError(f"Database does not exist: {args.db}")
    mapping_data = load_json(args.mapping)
    mapping = mapping_data["mappings"]
    annotations = load_json(args.annotations)
    rows = evaluate_mapping(Repository(args.db), annotations, mapping)
    print_report(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
