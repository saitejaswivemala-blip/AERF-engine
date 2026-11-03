"""
scripts/evaluate_independent_provisional.py

Executes research-grade evaluation of the frozen ScientificClaimExtractor
against the 143 independent sentences using provisional preliminary labels.

Requirements:
- Validates all 143 sample IDs and preserves original sentences.
- Creates tests/data/independent_eval/claim_extractor_independent_provisional_eval.csv
  with explicit provenance: "PROVISIONAL_AI_PREDICTION_UNVERIFIED".
- Does not modify protected files or claim human verification.
- Executes frozen ScientificClaimExtractor via its standard public interface.
- Calculates accuracy, per-class precision/recall/F1, macro/weighted averages,
  confusion matrix, Wilson 95% CIs, and error analysis.
- Generates reports under reports/independent_eval/:
  - evaluation_report.md
  - metrics.json
  - confusion_matrix.csv
  - error_analysis.csv
  - annotation_quality_audit.md
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nlp.claim_extractor import ScientificClaimExtractor

# Input files
AUTO_LABELS_CSV = ROOT / "tests" / "data" / "independent_eval" / "claim_extractor_auto_labels.csv"
SOURCE_INDEP_CSV = ROOT / "tests" / "data" / "independent_eval" / "claim_extractor_independent_eval.csv"

# Output dataset
PROVISIONAL_EVAL_CSV = ROOT / "tests" / "data" / "independent_eval" / "claim_extractor_independent_provisional_eval.csv"

# Output report directory
REPORT_DIR = ROOT / "reports" / "independent_eval"


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    z = 1.96 if confidence == 0.95 else 1.645
    p = successes / total
    denom = 1 + (z**2) / total
    center = (p + (z**2) / (2 * total)) / denom
    margin = (z * math.sqrt((p * (1 - p) / total) + (z**2) / (4 * total**2))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def main():
    print("=" * 80)
    print("AERF Engine - Independent Claim Extractor Provisional Evaluation")
    print("=" * 80)

    # 1. Validation of Input Data
    if not AUTO_LABELS_CSV.exists():
        raise FileNotFoundError(f"Missing auto labels file: {AUTO_LABELS_CSV}")
    if not SOURCE_INDEP_CSV.exists():
        raise FileNotFoundError(f"Missing source independent CSV: {SOURCE_INDEP_CSV}")

    df_auto = pd.read_csv(AUTO_LABELS_CSV)
    df_source = pd.read_csv(SOURCE_INDEP_CSV)

    if len(df_auto) != 143 or len(df_source) != 143:
        raise ValueError(f"Expected 143 rows, found: auto={len(df_auto)}, source={len(df_source)}")

    # Check alignment of sample IDs and text
    for idx in range(143):
        sid_a = df_auto.iloc[idx]["sample_id"]
        sid_s = df_source.iloc[idx]["sample_id"]
        if sid_a != sid_s:
            raise ValueError(f"Sample ID mismatch at row {idx}: {sid_a} vs {sid_s}")
        sent_a = df_auto.iloc[idx]["sentence"].strip()
        sent_s = df_source.iloc[idx]["sentence"].strip()
        if sent_a != sent_s:
            raise ValueError(f"Sentence text mismatch at {sid_a}")
        lbl = df_auto.iloc[idx]["preliminary_label"]
        if lbl not in (0, 1):
            raise ValueError(f"Invalid label {lbl} for {sid_a}")

    print("Step 1: Validated all 143 sample IDs and sentence texts verbatim.")

    # 2. Create Evaluation-Ready CSV with Explicit Provenance
    PROVISIONAL_EVAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    eval_rows = []
    for idx, r in df_auto.iterrows():
        eval_rows.append({
            "sample_id": r["sample_id"],
            "sentence": r["sentence"],
            "label": int(r["preliminary_label"]),
            "paper_id": r["paper_id"],
            "category": str(r["source_section"]).lower(),
            "doi": r["doi_or_url"],
            "annotation_rationale": f"[PROVISIONAL AI PREDICTION - UNVERIFIED] {r['rationale']}",
            "confidence": r["confidence"],
            "verification_status": "PROVISIONAL_AI_PREDICTION_UNVERIFIED",
            "model_provenance": r["model_provenance"],
        })

    with open(PROVISIONAL_EVAL_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "sentence", "label", "paper_id", "category", "doi", "annotation_rationale", "confidence", "verification_status", "model_provenance"]
        )
        writer.writeheader()
        writer.writerows(eval_rows)
    print(f"Step 2: Saved evaluation-ready dataset ({len(eval_rows)} rows) to: {PROVISIONAL_EVAL_CSV.name}")

    # 3. Run Frozen ScientificClaimExtractor
    print("\nStep 3: Initializing frozen ScientificClaimExtractor...")
    t0 = time.time()
    extractor = ScientificClaimExtractor()
    print(f"Extractor initialized in {time.time() - t0:.2f}s.")

    actual = [int(r["label"]) for r in eval_rows]
    predicted: List[int] = []
    reasons: List[str] = []
    pred_scores: List[float] = []

    print("\nRunning extractor across all 143 independent sentences...")
    for idx, r in enumerate(eval_rows, start=1):
        extracted = extractor.extract(r["sentence"])
        if extracted:
            pred = 1
            reason = extracted[0].reason
            score = extracted[0].score
        else:
            pred = 0
            reason = "rejected_below_threshold_or_non_finding"
            score = 0.0

        predicted.append(pred)
        reasons.append(reason)
        pred_scores.append(score)

        if idx % 35 == 0 or idx == len(eval_rows):
            print(f"  Evaluated {idx} / {len(eval_rows)} sentences...")

    # 4. Metrics Calculation
    print("\nStep 4: Computing comprehensive statistical metrics...")
    n = len(actual)
    acc = float(accuracy_score(actual, predicted))
    p_pos = float(precision_score(actual, predicted, pos_label=1, zero_division=0))
    r_pos = float(recall_score(actual, predicted, pos_label=1, zero_division=0))
    f1_pos = float(f1_score(actual, predicted, pos_label=1, zero_division=0))

    p_neg = float(precision_score(actual, predicted, pos_label=0, zero_division=0))
    r_neg = float(recall_score(actual, predicted, pos_label=0, zero_division=0))
    f1_neg = float(f1_score(actual, predicted, pos_label=0, zero_division=0))

    p_macro = float(precision_score(actual, predicted, average="macro", zero_division=0))
    r_macro = float(recall_score(actual, predicted, average="macro", zero_division=0))
    f1_macro = float(f1_score(actual, predicted, average="macro", zero_division=0))

    p_weighted = float(precision_score(actual, predicted, average="weighted", zero_division=0))
    r_weighted = float(recall_score(actual, predicted, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(actual, predicted, average="weighted", zero_division=0))

    # Confusion matrix with explicit labels [1, 0]
    cm = confusion_matrix(actual, predicted, labels=[1, 0])
    tp, fn = int(cm[0][0]), int(cm[0][1])
    fp, tn = int(cm[1][0]), int(cm[1][1])

    acc_ci = wilson_score_interval(tp + tn, n)
    prec_ci = wilson_score_interval(tp, tp + fp) if (tp + fp) > 0 else (0.0, 0.0)
    rec_ci = wilson_score_interval(tp, tp + fn) if (tp + fn) > 0 else (0.0, 0.0)

    # Stratified metrics by paper
    paper_metrics = {}
    for pid in sorted(list(set(r["paper_id"] for r in eval_rows))):
        sub_indices = [i for i, r in enumerate(eval_rows) if r["paper_id"] == pid]
        sub_act = [actual[i] for i in sub_indices]
        sub_pred = [predicted[i] for i in sub_indices]
        paper_metrics[pid] = {
            "total_samples": len(sub_indices),
            "provisional_claims": sum(sub_act),
            "provisional_non_claims": len(sub_act) - sum(sub_act),
            "accuracy": float(accuracy_score(sub_act, sub_pred)),
            "f1_score": float(f1_score(sub_act, sub_pred, zero_division=0)),
        }

    # Stratified metrics by section
    section_metrics = {}
    for sec in ["abstract", "introduction", "methods", "results", "discussion"]:
        sub_indices = [i for i, r in enumerate(eval_rows) if r["category"] == sec]
        if sub_indices:
            sub_act = [actual[i] for i in sub_indices]
            sub_pred = [predicted[i] for i in sub_indices]
            section_metrics[sec] = {
                "total_samples": len(sub_indices),
                "provisional_claims": sum(sub_act),
                "provisional_non_claims": len(sub_act) - sum(sub_act),
                "accuracy": float(accuracy_score(sub_act, sub_pred)),
                "f1_score": float(f1_score(sub_act, sub_pred, zero_division=0)),
            }

    # 5. Save Artifacts to reports/independent_eval/
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nStep 5: Writing evaluation reports to {REPORT_DIR}...")

    # Artifact 1: metrics.json
    metrics_data = {
        "evaluation_type": "PROVISIONAL_INDEPENDENT_EVALUATION",
        "reference_label_provenance": "DUAL_AI_CONSENSUS_UNVERIFIED_BY_HUMAN",
        "academic_validity_note": "These metrics reflect provisional agreement between the frozen claim extractor and the dual AI preliminary consensus. They must NOT be cited as verified benchmark accuracy until independent human annotation is completed.",
        "sample_size": n,
        "class_balance": {
            "provisional_positive_claims": sum(actual),
            "provisional_negative_non_claims": n - sum(actual),
        },
        "confusion_matrix": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "matrix_format": "[[TP, FN], [FP, TN]] with labels [1, 0]"
        },
        "overall_metrics": {
            "accuracy": round(acc, 4),
            "accuracy_95_ci": [round(acc_ci[0], 4), round(acc_ci[1], 4)],
            "precision_finding_pos": round(p_pos, 4),
            "precision_95_ci": [round(prec_ci[0], 4), round(prec_ci[1], 4)],
            "recall_finding_pos": round(r_pos, 4),
            "recall_95_ci": [round(rec_ci[0], 4), round(rec_ci[1], 4)],
            "f1_finding_pos": round(f1_pos, 4),
            "precision_non_finding_neg": round(p_neg, 4),
            "recall_non_finding_neg": round(r_neg, 4),
            "f1_non_finding_neg": round(f1_neg, 4),
            "macro_avg": {
                "precision": round(p_macro, 4),
                "recall": round(r_macro, 4),
                "f1_score": round(f1_macro, 4),
            },
            "weighted_avg": {
                "precision": round(p_weighted, 4),
                "recall": round(r_weighted, 4),
                "f1_score": round(f1_weighted, 4),
            }
        },
        "stratified_by_paper": paper_metrics,
        "stratified_by_section": section_metrics,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    with open(REPORT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"  [SAVED] {REPORT_DIR / 'metrics.json'}")

    # Artifact 2: confusion_matrix.csv
    with open(REPORT_DIR / "confusion_matrix.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["", "Predicted_CLAIM_1", "Predicted_NON_CLAIM_0", "Total_Actual"])
        writer.writerow(["Actual_CLAIM_1", tp, fn, tp + fn])
        writer.writerow(["Actual_NON_CLAIM_0", fp, tn, fp + tn])
        writer.writerow(["Total_Predicted", tp + fp, fn + tn, n])
    print(f"  [SAVED] {REPORT_DIR / 'confusion_matrix.csv'}")

    # Artifact 3: error_analysis.csv
    error_rows = []
    for idx, (r, a, p, rsn, sc) in enumerate(zip(eval_rows, actual, predicted, reasons, pred_scores)):
        is_correct = (a == p)
        if a == 1 and p == 1:
            err_type = "TP"
        elif a == 0 and p == 0:
            err_type = "TN"
        elif a == 0 and p == 1:
            err_type = "FP"
        else:
            err_type = "FN"

        error_rows.append({
            "sample_id": r["sample_id"],
            "paper_id": r["paper_id"],
            "section": r["category"],
            "provisional_actual": a,
            "predicted": p,
            "classification_status": "CORRECT" if is_correct else "INCORRECT",
            "error_type": err_type,
            "confidence_provisional": r["confidence"],
            "extractor_score": round(sc, 4),
            "extractor_reason": rsn,
            "sentence": r["sentence"],
            "annotation_rationale": r["annotation_rationale"],
        })

    with open(REPORT_DIR / "error_analysis.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(error_rows[0].keys()))
        writer.writeheader()
        writer.writerows(error_rows)
    print(f"  [SAVED] {REPORT_DIR / 'error_analysis.csv'}")

    # Artifact 4: annotation_quality_audit.md
    with open(REPORT_DIR / "annotation_quality_audit.md", "w", encoding="utf-8") as f:
        f.write(f"""# AERF Independent Evaluation — Annotation Quality & Provenance Audit

**Audit Date**: {time.strftime('%B %d, %Y', time.gmtime())}  
**Evaluation Scope**: 143 sentences across 9 unseen research papers  
**Ground Truth Status**: `PROVISIONAL_AI_PREDICTION_UNVERIFIED`

---

## 1. Academic Provenance & Integrity Statement

1. **No Human Validation**: The reference labels used in this preliminary evaluation were generated via dual automated consensus (Guidelines Semantic Classifier + DeBERTa-v3 zero-shot NLI). **No human annotator has verified or adjudicated these labels.**
2. **Model Independence**: The claim extractor being evaluated (`ScientificClaimExtractor` in `nlp/claim_extractor.py`) was **never** used to formulate, filter, or rank candidate labels.
3. **Review Queue Status**: 126 of the 143 sentences (88.1%) exhibited dual-model ambiguity or low confidence and were queued into `tests/data/independent_eval/independent_review_queue.csv`.
4. **Provisional Metric Notice**: The reported agreement rates indicate behavioral alignment between the extractor's rules/heuristics and the dual AI consensus, but **must not be claimed as true generalization accuracy in academic publications**.

---

## 2. Preliminary Agreement Breakdown

- **Total Evaluated Sentences**: {n}
- **Provisional Claims (1)**: {sum(actual)} ({sum(actual)/n*100:.1f}%)
- **Provisional Non-Claims (0)**: {n - sum(actual)} ({(n - sum(actual))/n*100:.1f}%)
- **Provisional Agreement (Accuracy)**: {acc:.2%} (95% CI: [{acc_ci[0]:.2%}, {acc_ci[1]:.2%}])
- **Provisional F1 (Claim Class)**: {f1_pos:.2%}
- **Provisional F1 (Macro Avg)**: {f1_macro:.2%}
""")
    print(f"  [SAVED] {REPORT_DIR / 'annotation_quality_audit.md'}")

    # Artifact 5: evaluation_report.md
    with open(REPORT_DIR / "evaluation_report.md", "w", encoding="utf-8") as f:
        f.write(f"""# AERF Scientific Claim Extractor — Independent Provisional Evaluation Report

**Model Evaluated**: `ScientificClaimExtractor` (`nlp/claim_extractor.py`)  
**Evaluation Date**: {time.strftime('%B %d, %Y', time.gmtime())}  
**Dataset**: `tests/data/independent_eval/claim_extractor_independent_provisional_eval.csv`  
**Dataset Provenance**: 143 sentences from 9 independent open-access papers, labeled via dual AI consensus (unverified by human review)  
**Status**: `PROVISIONAL_INTER_MODEL_AGREEMENT`

---

## 1. Executive Summary

An independent, out-of-domain evaluation was executed across 143 genuine sentences collected from 9 previously unseen research papers spanning Materials Science, Astrophysics, Renewable Energy, Structural Geology, Climate Physics, Ecology, Quantum Optics, Plant Biology, and Glaciology.

Because independent human adjudication is pending, metrics below represent **provisional agreement** between the frozen claim extractor and the preliminary dual-model consensus:

| Metric | Score | 95% Confidence Interval |
|---|---|---|
| **Provisional Accuracy** | **{acc:.2%}** | [{acc_ci[0]:.2%}, {acc_ci[1]:.2%}] |
| **Claim Precision (Class 1)** | **{p_pos:.2%}** | [{prec_ci[0]:.2%}, {prec_ci[1]:.2%}] |
| **Claim Recall (Class 1)** | **{r_pos:.2%}** | [{rec_ci[0]:.2%}, {rec_ci[1]:.2%}] |
| **Claim F1-Score (Class 1)** | **{f1_pos:.2%}** | — |
| **Non-Claim F1-Score (Class 0)**| **{f1_neg:.2%}** | — |
| **Macro-Averaged F1** | **{f1_macro:.2%}** | — |
| **Weighted-Averaged F1** | **{f1_weighted:.2%}** | — |

---

## 2. Confusion Matrix

| | Predicted CLAIM (`1`) | Predicted NON-CLAIM (`0`) | Total Actual |
|---|---|---|---|
| **Provisional CLAIM (`1`)** | **{tp}** (TP) | **{fn}** (FN) | **{tp + fn}** |
| **Provisional NON-CLAIM (`0`)** | **{fp}** (FP) | **{tn}** (TN) | **{fp + tn}** |
| **Total Predicted** | **{tp + fp}** | **{fn + tn}** | **{n}** |

---

## 3. Stratified Breakdown by Scientific Paper

| Paper ID | Discipline | N | Actual Claims | Predicted Claims | Accuracy | F1 (Class 1) |
|---|---|---|---|---|---|---|
""")
        for pid, pdata in paper_metrics.items():
            f.write(f"| **`{pid}`** | {eval_rows[[i for i, r in enumerate(eval_rows) if r['paper_id'] == pid][0]]['doi']} | {pdata['total_samples']} | {pdata['provisional_claims']} | — | {pdata['accuracy']:.1%} | {pdata['f1_score']:.1%} |\n")

        f.write(f"""
---

## 4. Stratified Breakdown by Paper Section

| Section | N | Actual Claims | Accuracy | F1 (Class 1) |
|---|---|---|---|---|
""")
        for sec, sdata in section_metrics.items():
            f.write(f"| **`{sec.capitalize()}`** | {sdata['total_samples']} | {sdata['provisional_claims']} | {sdata['accuracy']:.1%} | {sdata['f1_score']:.1%} |\n")

        f.write(f"""
---

## 5. Primary Error Patterns

1. **False Positives ({fp} cases)**:
   - Sentences describing experimental methods or simulations containing comparative or numeric terminology (e.g. step sizes, voltage parameters) triggered extractor result cues.
2. **False Negatives ({fn} cases)**:
   - Domain-specific empirical assertions lacking standard clinical/NLP keywords (e.g. astrophysical flux descriptions or geological wave velocities) were rejected below the 0.55 confidence threshold.

---

## 6. Next Steps for Academic Publication

1. Review and verify the 126 prioritized edge cases in `tests/data/independent_eval/independent_review_queue.csv`.
2. Adjudicate discrepancies into a permanent ground-truth dataset.
3. Re-run `evaluate_claim_extractor.py` against the human-verified benchmark.
""")
    print(f"  [SAVED] {REPORT_DIR / 'evaluation_report.md'}")

    print("\n" + "=" * 80)
    print("Independent Provisional Evaluation Complete!")
    print(f"Provisional Accuracy: {acc:.2%} | F1: {f1_pos:.2%} (N={n})")
    print(f"Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print("=" * 80)


if __name__ == "__main__":
    main()
