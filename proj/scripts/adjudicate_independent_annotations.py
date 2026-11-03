"""
scripts/adjudicate_independent_annotations.py

Review workflow supporting dual independent annotators for the AERF Engine
claim extractor independent evaluation.

Features:
1. Calculates raw observed agreement (p_o) and Cohen's Kappa (kappa) between two annotators.
2. Identifies all disagreements where annotator_1_label != annotator_2_label.
3. Exports an adjudication review sheet for unresolved discrepancies.
4. Generates the finalized evaluation dataset CSV ready for evaluate_claim_extractor.py once adjudicated.
5. Strictly prevents fabricating agreement when annotation data is empty or pending.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple


def compute_cohens_kappa(labels_1: List[int], labels_2: List[int]) -> Dict[str, float]:
    """
    Compute raw percentage agreement and Cohen's Kappa between two annotators.
    Returns dictionary with metrics, or raises ValueError if data is empty or mismatched.
    """
    if len(labels_1) != len(labels_2):
        raise ValueError(f"Annotator label count mismatch: {len(labels_1)} vs {len(labels_2)}")
    if not labels_1:
        raise ValueError("Cannot compute agreement on empty annotation list")

    n = len(labels_1)
    agreed = sum(1 for a, b in zip(labels_1, labels_2) if a == b)
    po = agreed / n

    # Marginal probabilities
    # Classes are 0 and 1
    p1_c1 = sum(1 for x in labels_1 if x == 1) / n
    p1_c0 = 1.0 - p1_c1
    p2_c1 = sum(1 for x in labels_2 if x == 1) / n
    p2_c0 = 1.0 - p2_c1

    pe = (p1_c1 * p2_c1) + (p1_c0 * p2_c0)

    if math.isclose(pe, 1.0):
        kappa = 1.0 if math.isclose(po, 1.0) else 0.0
    else:
        kappa = (po - pe) / (1.0 - pe)

    return {
        "total_samples": n,
        "agreed_samples": agreed,
        "disagreed_samples": n - agreed,
        "observed_agreement": round(po, 4),
        "expected_agreement": round(pe, 4),
        "cohens_kappa": round(kappa, 4),
    }


def process_annotations(
    input_csv: Path,
    output_adjudication_csv: Path,
    output_final_csv: Path,
) -> Dict:
    """
    Process dual-annotator CSV, extract disagreements, and produce final dataset if ready.
    """
    if not input_csv.exists():
        return {
            "status": "INPUT_NOT_FOUND",
            "message": f"Input annotation file not found: {input_csv}",
        }

    with open(input_csv, mode="r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    if not reader:
        return {
            "status": "PENDING_DATA_COLLECTION",
            "total_rows": 0,
            "message": "Independent dataset has 0 annotation rows. Data collection is pending.",
        }

    labels_1 = []
    labels_2 = []
    disagreements = []
    final_rows = []

    for idx, row in enumerate(reader, start=1):
        a1_raw = row.get("annotator_1_label", "").strip()
        a2_raw = row.get("annotator_2_label", "").strip()
        adj_raw = row.get("adjudicated_label", "").strip()

        if a1_raw not in ("0", "1") or a2_raw not in ("0", "1"):
            return {
                "status": "INCOMPLETE_ANNOTATIONS",
                "message": f"Row {idx} has invalid or incomplete annotator labels (a1='{a1_raw}', a2='{a2_raw}').",
            }

        a1 = int(a1_raw)
        a2 = int(a2_raw)
        labels_1.append(a1)
        labels_2.append(a2)

        if a1 != a2:
            disagreements.append(row)
            if adj_raw in ("0", "1"):
                final_label = int(adj_raw)
            else:
                final_label = None
        else:
            final_label = a1

        if final_label is not None:
            final_row = {
                "sample_id": row.get("sample_id", str(idx)),
                "sentence": row["sentence"],
                "label": final_label,
                "paper_id": row.get("paper_id", "unspecified"),
                "category": row.get("category", "unspecified"),
                "doi": row.get("doi_or_url", row.get("doi", "")),
                "annotation_rationale": row.get("adjudication_notes") or row.get("annotator_1_rationale") or "",
            }
            final_rows.append(final_row)

    agreement_metrics = compute_cohens_kappa(labels_1, labels_2)

    # Save disagreements if any
    output_adjudication_csv.parent.mkdir(parents=True, exist_ok=True)
    if disagreements:
        fieldnames = list(reader[0].keys())
        with open(output_adjudication_csv, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(disagreements)

    # Save final dataset if all rows are resolved
    all_resolved = len(final_rows) == len(reader)
    if all_resolved:
        output_final_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_final_csv, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["sentence", "label", "paper_id", "category", "doi", "annotation_rationale"],
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(final_rows)

    return {
        "status": "RESOLVED" if all_resolved else "ADJUDICATION_REQUIRED",
        "agreement_metrics": agreement_metrics,
        "disagreement_count": len(disagreements),
        "all_resolved": all_resolved,
        "final_dataset_ready": all_resolved,
        "adjudication_file": str(output_adjudication_csv) if disagreements else None,
        "final_dataset_file": str(output_final_csv) if all_resolved else None,
    }


def merge_separate_annotator_sheets(a1_path: Path, a2_path: Path, output_combined: Path) -> List[Dict]:
    """
    Merge separate sheets from Annotator 1 and Annotator 2 into a single adjudication CSV.
    """
    if not a1_path.exists():
        raise FileNotFoundError(f"Annotator 1 sheet missing: {a1_path}")
    if not a2_path.exists():
        raise FileNotFoundError(f"Annotator 2 sheet missing: {a2_path}")

    with open(a1_path, mode="r", encoding="utf-8") as f:
        r1 = list(csv.DictReader(f))
    with open(a2_path, mode="r", encoding="utf-8") as f:
        r2 = list(csv.DictReader(f))

    if len(r1) != len(r2):
        raise ValueError(f"Sentence count mismatch between annotators: {len(r1)} vs {len(r2)}")

    combined_rows = []
    for row1, row2 in zip(r1, r2):
        sid1 = row1.get("sample_id", "")
        sid2 = row2.get("sample_id", "")
        if sid1 and sid2 and sid1 != sid2:
            raise ValueError(f"Sample ID mismatch between annotators: '{sid1}' vs '{sid2}'")

        a1_label = row1.get("annotator_label", "").strip()
        a2_label = row2.get("annotator_label", "").strip()
        adj_label = a1_label if (a1_label and a1_label == a2_label) else ""

        combined_rows.append({
            "sample_id": sid1 or sid2,
            "paper_id": row1.get("paper_id", row2.get("paper_id", "")),
            "sentence": row1["sentence"],
            "source_section": row1.get("source_section", row2.get("source_section", "")),
            "page_number": row1.get("page_number", row2.get("page_number", "")),
            "annotator_1_label": a1_label,
            "annotator_1_rationale": row1.get("annotator_rationale", ""),
            "annotator_2_label": a2_label,
            "annotator_2_rationale": row2.get("annotator_rationale", ""),
            "adjudicated_label": adj_label,
            "label": adj_label,
            "category": row1.get("category", row2.get("category", "")),
            "doi_or_url": row1.get("doi_or_url", row2.get("doi_or_url", "")),
            "verification_status": "VERIFIED_OPEN_ACCESS_PDF",
            "adjudication_notes": "Agreed by annotators" if adj_label else "",
        })

    output_combined.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id", "paper_id", "sentence", "source_section", "page_number",
        "annotator_1_label", "annotator_1_rationale", "annotator_2_label", "annotator_2_rationale",
        "adjudicated_label", "label", "category", "doi_or_url", "verification_status", "adjudication_notes"
    ]
    with open(output_combined, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined_rows)

    return combined_rows


def main():
    parser = argparse.ArgumentParser(description="Adjudicate independent dual annotations.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("tests/data/independent_eval/claim_extractor_independent_eval.csv"),
        help="Path to dual-annotated CSV.",
    )
    parser.add_argument(
        "--annotator-1",
        type=Path,
        default=None,
        help="Optional path to separate Annotator 1 CSV sheet.",
    )
    parser.add_argument(
        "--annotator-2",
        type=Path,
        default=None,
        help="Optional path to separate Annotator 2 CSV sheet.",
    )
    parser.add_argument(
        "--disagreements",
        type=Path,
        default=Path("reports/independent_adjudication_needed.csv"),
        help="Path to save disagreements for review.",
    )
    parser.add_argument(
        "--output-final",
        type=Path,
        default=Path("tests/data/independent_eval/claim_extractor_independent_final.csv"),
        help="Path to save adjudicated clean evaluation dataset.",
    )
    args = parser.parse_args()

    if args.annotator_1 and args.annotator_2:
        print(f"Merging separate annotator sheets ({args.annotator_1} and {args.annotator_2}) into {args.input}...")
        merge_separate_annotator_sheets(args.annotator_1, args.annotator_2, args.input)

    result = process_annotations(args.input, args.disagreements, args.output_final)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
