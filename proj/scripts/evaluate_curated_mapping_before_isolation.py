#!/usr/bin/env python3
"""
Real-model curated comparison for AERF Engine.

Runs the same 4 manually verified cross-document pairs through:
  - Baseline A: similarity only
  - Baseline B: similarity + NLI
  - Baseline C: similarity + entity alignment + NLI
  - Full AERF
  - AERF without entity alignment
  - AERF without adaptive scoring
  - AERF without provenance weighting

This is a diagnostic experiment, NOT a statistically valid benchmark,
because the curated dataset contains only 4 cross-document pairs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.repository import Repository
from nlp.nli.nli_classifier import NLIClassifier

from experiments.lib.experiment_config import (
    BASELINE_A_SIMILARITY_ONLY,
    BASELINE_B_SIMILARITY_NLI,
    BASELINE_C_SIMILARITY_ENTITY_NLI,
    FULL_AERF,
    ABLATION_WITHOUT_ENTITY_ALIGNMENT,
    ABLATION_WITHOUT_ADAPTIVE_SCORING,
    ABLATION_WITHOUT_PROVENANCE_WEIGHTING,
)

from experiments.lib.pipeline_runner import (
    run_matching,
    score_claims_with_config,
)

from experiments.lib.metrics import evaluate


MANIFEST_PATH = Path(
    "experiments/datasets/paper_manifest.json"
)

MAPPING_PATH = Path(
    "experiments/annotations/curated_mapping_gt01_gt02_gt03_gt05.json"
)

RESULTS_PATH = Path(
    "experiments/results/curated_comparison_real_models.json"
)


EXPECTED_PAIR_IDS = {
    "gt_01",
    "gt_02",
    "gt_03",
    "gt_05",
}


CONFIGS = [
    BASELINE_A_SIMILARITY_ONLY,
    BASELINE_B_SIMILARITY_NLI,
    BASELINE_C_SIMILARITY_ENTITY_NLI,
    FULL_AERF,
    ABLATION_WITHOUT_ENTITY_ALIGNMENT,
    ABLATION_WITHOUT_ADAPTIVE_SCORING,
    ABLATION_WITHOUT_PROVENANCE_WEIGHTING,
]


def load_mapping():

    if not MAPPING_PATH.exists():
        raise FileNotFoundError(
            f"{MAPPING_PATH} not found."
        )

    mapping = json.loads(
        MAPPING_PATH.read_text()
    )

    if mapping.get("_status") != "FILLED_IN":
        raise ValueError(
            f"{MAPPING_PATH} is not marked FILLED_IN. "
            "Refusing to evaluate incomplete ground truth."
        )

    pairs_by_id = {
        p["pair_id"]: p
        for p in mapping["pairs"]
    }

    if set(pairs_by_id) != EXPECTED_PAIR_IDS:
        raise ValueError(
            f"Expected {EXPECTED_PAIR_IDS}, "
            f"found {set(pairs_by_id)}"
        )

    for pid, p in pairs_by_id.items():

        if (
            p.get("claim_a_id") is None
            or p.get("claim_b_id") is None
        ):
            raise ValueError(
                f"{pid} has missing claim IDs."
            )

        if (
            not p.get("mapping_rationale")
            or p["mapping_rationale"].startswith("FILL IN")
        ):
            raise ValueError(
                f"{pid} has no completed mapping rationale."
            )

    return mapping


def validate_claim(
    repo,
    manifest_to_db,
    claim_id,
    expected_paper,
    side,
    pair_id,
):

    claim = repo.get_claim(claim_id)

    if claim is None:
        raise ValueError(
            f"{pair_id}.{side}: "
            f"claim_id={claim_id} does not exist."
        )

    expected_db_id = manifest_to_db.get(
        expected_paper
    )

    if expected_db_id is None:
        raise ValueError(
            f"{pair_id}.{side}: paper "
            f"{expected_paper} does not resolve "
            "in database."
        )

    if claim["paper_id"] != expected_db_id:
        raise ValueError(
            f"{pair_id}.{side}: claim {claim_id} "
            f"belongs to paper {claim['paper_id']}, "
            f"expected {expected_db_id}."
        )

    return claim


def build_pairs(repo, mapping):

    manifest = json.loads(
        MANIFEST_PATH.read_text()
    )

    manifest_to_db = {}

    for p in manifest["papers"]:

        paper = repo.get_paper_by_filename(
            Path(p["pdf_path"]).name
        )

        if paper is not None:
            manifest_to_db[
                p["paper_id"]
            ] = paper.paper_id

    pairs_by_id = {
        p["pair_id"]: p
        for p in mapping["pairs"]
    }

    claim_dicts = {}
    entries = []

    for pid in sorted(EXPECTED_PAIR_IDS):

        p = pairs_by_id[pid]

        claim_a = validate_claim(
            repo,
            manifest_to_db,
            p["claim_a_id"],
            p["paper_a"],
            "claim_a",
            pid,
        )

        claim_b = validate_claim(
            repo,
            manifest_to_db,
            p["claim_b_id"],
            p["paper_b"],
            "claim_b",
            pid,
        )

        claim_dicts[
            p["claim_a_id"]
        ] = claim_a

        claim_dicts[
            p["claim_b_id"]
        ] = claim_b

        entries.append(p)

        print(
            f"{pid}: "
            f"claim_a_id={p['claim_a_id']} "
            f"(db paper_id={claim_a['paper_id']}), "
            f"claim_b_id={p['claim_b_id']} "
            f"(db paper_id={claim_b['paper_id']}) "
            f"-- validated OK"
        )

        print(
            f"  rationale: "
            f"{p['mapping_rationale']}"
        )

    pairs = [
        (
            claim_dicts[p["claim_a_id"]],
            claim_dicts[p["claim_b_id"]],
        )
        for p in entries
    ]

    return entries, pairs


def evaluate_configuration(
    repo,
    config,
    nli,
    entries,
    pairs,
):

    # ------------------------------------------------------------
    # Step 1: Run matching and real NLI
    # ------------------------------------------------------------

    records = run_matching(
        repo,
        config,
        nli,
        pairs,
    )

    if len(records) != len(entries):
        raise RuntimeError(
            f"{config.name}: expected "
            f"{len(entries)} records, "
            f"got {len(records)}."
        )

    # IMPORTANT:
    # predictions must be initialized OUTSIDE the loop.
    predictions = []

    # ------------------------------------------------------------
    # Step 2: Persist real NLI evidence
    # ------------------------------------------------------------

    for entry, record in zip(
        entries,
        records,
    ):

        # Store model-generated NLI evidence in claim_pairs.
        #
        # IMPORTANT:
        # validated_as remains None.
        #
        # This means:
        #   - the NLI prediction is stored as evidence
        #   - it is NOT treated as human-validated truth
        #   - the human labels remain untouched
        #
        # This allows the AERF semantic factor to consume the
        # real NLI confidence without contaminating the ground truth.

        if record.nli_label is not None:

            repo.upsert_claim_pair(
                {
                    "claim_a_id": record.claim_a_id,
                    "claim_b_id": record.claim_b_id,
                    "cosine_similarity": record.cosine_similarity,
                    "nli_label": record.nli_label,
                    "nli_confidence": record.nli_confidence,
                    "entity_overlap_score": record.entity_overlap_score,

                    # DO NOT treat model prediction as validation.
                    "validated_as": None,

                    # The current NLI interface exposes label,
                    # probabilities and confidence. The runner's
                    # MatchRecord only stores label/confidence, so
                    # these optional probability fields remain None.
                    "nli_entailment": None,
                    "nli_contradiction": None,
                    "nli_neutral": None,

                    "relation_alignment_score":
                        record.entity_overlap_score,

                    "matching_method":
                        "curated_real_model_evaluation",

                    "entity_alignment": [],

                    "nli_model_name":
                        getattr(
                            nli,
                            "model_name",
                            None,
                        ),
                }
            )

        # --------------------------------------------------------
        # Step 3: Build evaluation prediction record
        # --------------------------------------------------------

        predictions.append(
            {
                "pair_id": entry["pair_id"],
                "paper_a": entry["paper_a"],
                "paper_b": entry["paper_b"],
                "claim_a_id": entry["claim_a_id"],
                "claim_b_id": entry["claim_b_id"],
                "human_label": entry["human_label"],
                "system_prediction":
                    record.predicted_label,
                "cosine_similarity":
                    record.cosine_similarity,
                "entity_overlap_score":
                    record.entity_overlap_score,
                "nli_label":
                    record.nli_label,
                "nli_confidence":
                    record.nli_confidence,
                "correct": (
                    record.predicted_label
                    == entry["human_label"]
                ),
            }
        )

    # ------------------------------------------------------------
    # Step 4: Calculate diagnostic metrics
    # ------------------------------------------------------------

    metric_records = [
        {
            "system_prediction":
                x["system_prediction"],
            "human_label":
                x["human_label"],
        }
        for x in predictions
    ]

    metrics = evaluate(
        metric_records,
        min_n_for_stats=10,
    )

    # ------------------------------------------------------------
    # Step 5: Run AERF scoring AFTER NLI evidence is persisted
    # ------------------------------------------------------------

    scores = {}

    if config.use_aerf_scoring:

        claim_ids = sorted(
            {
                entry["claim_a_id"]
                for entry in entries
            }
            |
            {
                entry["claim_b_id"]
                for entry in entries
            }
        )

        scores = score_claims_with_config(
            repo,
            config,
            claim_ids,
        )

    # ------------------------------------------------------------
    # Step 6: Return complete result
    # ------------------------------------------------------------

    return {
        "configuration": config.name,
        "description": config.description,

        "settings": {
            "similarity_threshold":
                config.similarity_threshold,

            "entity_threshold":
                config.entity_threshold,

            "nli_threshold":
                config.nli_threshold,

            "use_entity_alignment":
                config.use_entity_alignment,

            "use_nli":
                config.use_nli,

            "use_aerf_scoring":
                config.use_aerf_scoring,

            "use_adaptive_updates":
                config.use_adaptive_updates,

            "zero_provenance_weight":
                config.zero_provenance_weight,
        },

        "n": len(predictions),

        "accuracy":
            metrics.accuracy,

        "macro_f1":
            metrics.macro_f1,

        "warnings":
            metrics.warnings,

        "predictions":
            predictions,

        "aerf_scores":
            scores,
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--db",
        default="experiments/runs/experiment.db",
    )

    args = parser.parse_args()

    db_path = Path(args.db)

    if not db_path.exists():
        raise FileNotFoundError(
            f"{db_path} does not exist."
        )

    print("=" * 78)

    print(
        "AERF ENGINE — REAL MODEL "
        "CONFIGURATION COMPARISON"
    )

    print("=" * 78)

    print(
        f"Database: {db_path}"
    )

    print(
        "Dataset: 4 manually verified "
        "cross-document pairs"
    )

    print(
        "Status: DIAGNOSTIC ONLY — "
        "NOT A BENCHMARK"
    )

    print()

    # ------------------------------------------------------------
    # Load mapping and database
    # ------------------------------------------------------------

    mapping = load_mapping()

    repo = Repository(db_path)

    entries, pairs = build_pairs(
        repo,
        mapping,
    )

    print()

    print(
        "Loading real NLI model..."
    )

    # ------------------------------------------------------------
    # Real NLI model
    # ------------------------------------------------------------

    real_nli = NLIClassifier()

    all_results = []

    # ------------------------------------------------------------
    # Evaluate all configurations
    # ------------------------------------------------------------

    for index, config in enumerate(
        CONFIGS,
        start=1,
    ):

        print()

        print("-" * 78)

        print(
            f"[{index}/{len(CONFIGS)}] "
            f"{config.name}"
        )

        print(
            config.description
        )

        result = evaluate_configuration(
            repo,
            config,
            real_nli,
            entries,
            pairs,
        )

        all_results.append(result)

        print(
            f"accuracy={result['accuracy']:.3f} "
            f"macro_f1={result['macro_f1']:.3f}"
        )

        for p in result["predictions"]:

            print(
                f"  {p['pair_id']}: "
                f"{p['system_prediction']:12s} "
                f"(human={p['human_label']:10s}, "
                f"sim={p['cosine_similarity']:.3f}, "
                f"nli={p['nli_label']}/"
                f"{p['nli_confidence']})"
            )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    print()

    print("=" * 78)

    print("SUMMARY")

    print("=" * 78)

    print(
        f"{'CONFIGURATION':42s} "
        f"{'ACC':>8s} "
        f"{'MACRO-F1':>10s}"
    )

    print("-" * 64)

    for result in all_results:

        print(
            f"{result['configuration']:42s} "
            f"{result['accuracy']:8.3f} "
            f"{result['macro_f1']:10.3f}"
        )

    validity_note = (
        "This is a diagnostic comparison on n=4 "
        "manually curated cross-document pairs. "
        "It is NOT a statistically valid benchmark. "
        "The sample is non-random and too small for "
        "scientifically meaningful precision/recall/F1 "
        "claims. Results must be reported together "
        "with the individual pair predictions and "
        "configuration details."
    )

    output = {
        "database":
            str(db_path),

        "n":
            len(entries),

        "diagnostic_only":
            True,

        "nli_source":
            "real",

        "nli_model_name":
            real_nli.model_name,

        "configs":
            all_results,

        "excluded_pairs":
            mapping.get(
                "excluded_pairs",
                [],
            ),

        "same_paper_diagnostics_only":
            mapping.get(
                "same_paper_diagnostics_only",
                [],
            ),

        "validity_note":
            validity_note,
    }

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_PATH.write_text(
        json.dumps(
            output,
            indent=2,
            default=str,
        )
    )

    print()

    print(validity_note)

    print()

    print(
        f"Saved: {RESULTS_PATH}"
    )


if __name__ == "__main__":
    main()