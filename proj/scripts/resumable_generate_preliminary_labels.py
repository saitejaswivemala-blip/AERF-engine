"""
scripts/resumable_generate_preliminary_labels.py

Resumable, fault-tolerant preliminary label generator for the AERF Engine
independent evaluation dataset.

Key Capabilities:
- Checkpoints progress after each batch to an atomic JSONL file.
- Resumes seamlessly if interrupted, skipping validated completed sample IDs.
- Retries transient model failures with backoff.
- Uses offline cached MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli with graceful fallback.
- Never touches protected files (nlp/claim_extractor.py, annotator sheets, original datasets).
- Outputs finalized CSVs atomically upon completion:
  1. tests/data/independent_eval/claim_extractor_auto_labels.csv
  2. tests/data/independent_eval/independent_review_queue.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Set, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = ROOT / "tests" / "data" / "independent_eval" / "claim_extractor_independent_eval.csv"
CHECKPOINT_PATH = ROOT / "tests" / "data" / "independent_eval" / "preliminary_labels_checkpoint.jsonl"
OUTPUT_ALL_CSV = ROOT / "tests" / "data" / "independent_eval" / "claim_extractor_auto_labels.csv"
OUTPUT_QUEUE_CSV = ROOT / "tests" / "data" / "independent_eval" / "independent_review_queue.csv"

# --- Guidelines Semantic Categorizer Patterns (from docs/CLAIM_ANNOTATION_GUIDELINES.md) ---

PAT_BACKGROUND = re.compile(
    r"\b(?:previous (?:studies|work|research|literature)|prior (?:work|studies)|earlier work|"
    r"has been extensively (?:studied|researched)|widely considered|conventionally|traditionally|"
    r"is known to|are known to|well established|long recognized|literature (?:shows|suggests)|"
    r"et al\.,?\s*\(\d{4}\)|\(\w+ et al\.,?\s*\d{4}\)|recent(?:ly)? (?:studies|advances|efforts)|"
    r"has attracted (?:significant|considerable) attention|remains a challenge|is considered to be)\b",
    re.I
)

PAT_METHOD = re.compile(
    r"\b(?:we (?:used|trained|employed|applied|implemented|calculated|computed|measured|"
    r"synthesized|prepared|characterized|performed|conducted|simulated|modeled|fabricated|"
    r"collected|sampled|acquired|extracted|selected|analyzed using|tested)|"
    r"(?:were|was) (?:analyzed using|measured by|conducted with|calculated using|prepared by|"
    r"synthesized via|performed in|simulated using|evaluated with|recorded at|operated at)|"
    r"materials and methods|experimental setup|simulation details|parameters were set|"
    r"using (?:software|code|package|instrument|spectrometer|diffractometer|algorithm|routine)|"
    r"in order to (?:measure|investigate|evaluate|determine|calculate))\b",
    re.I
)

PAT_DATASET_SPEC = re.compile(
    r"\b(?:consists of|composed of|concentration of|dimensions of|thickness of|"
    r"target of research is|composition of|batch size|grid spacing|learning rate|"
    r"cutoff energy of|k-point mesh|pseudopotential|time step of)\b",
    re.I
)

PAT_HYPOTHESIS = re.compile(
    r"\b(?:we hypothesi[zs]e[d]?|we aim(?:ed)? to|the purpose of this (?:work|study)|"
    r"our objective (?:is|was)|to determine whether|to investigate whether|"
    r"is intended to|shall be investigated)\b",
    re.I
)

PAT_LIMITATION = re.compile(
    r"\b(?:limitation[s]? of|caveat[s]?|is beyond the scope|further (?:investigation|research|work) is needed|"
    r"remains to be (?:elucidated|determined|explored)|future work will|we plan to)\b",
    re.I
)

PAT_CONTRIBUTION = re.compile(
    r"\b(?:we (?:propose|introduce|present|develop|demonstrate for the first time)|"
    r"this paper (?:proposes|introduces|presents))\s+a\s+(?:new|novel|simple|general|framework|method|model|approach)\b",
    re.I
)

PAT_NEGATIVE_FINDING = re.compile(
    r"\b(?:did not (?:significantly|substantially|meaningfully)?\s+(?:affect|increase|decrease|change|alter|influence|improve)|"
    r"no (?:significant|statistically significant|detectable|appreciable|notable)?\s+(?:difference|effect|change|correlation|increase|decrease|enhancement)|"
    r"failed to (?:produce|show|demonstrate)|was not associated with|negligible (?:effect|change|difference))\b",
    re.I
)

PAT_EMPIRICAL_RESULT = re.compile(
    r"\b(?:(?:show|shows|showed|demonstrate[s]?|demonstrated|reveal[s]?|revealed|indicate[s]?|indicated|"
    r"confirm[s]?|confirmed|exhibit[s]?|exhibited|achieve[s]?|achieved|found that|observed that)\s+"
    r"(?:that|a significant|significant|substantial|higher|lower|enhanced|superior|strong|marked)|"
    r"(?:results? (?:show|indicate|demonstrate|reveal|suggest) that)|"
    r"(?:outperform[s]?|outperformed|surpass(?:es|ed)?)|"
    r"(?:enhancement|increase|reduction|decrease|drop|gain) of (?:up to )?\d+(?:\.\d+)?\s*(?:%|ev|mah|bleu|db|k|km|v|ma|nm)|"
    r"(?:yield strength|adsorption energy|open-circuit voltage|diffusion barrier|capacity|accuracy) of \d|"
    r"(?:is|was|were) (?:significantly )?(?:higher|lower|faster|slower|enhanced|reduced|suppressed) than)\b",
    re.I
)


def classify_by_guidelines(sent: str, section: str) -> Tuple[int, str, str]:
    """
    Classify sentence according to docs/CLAIM_ANNOTATION_GUIDELINES.md.
    Returns (label: int, confidence: str, reason: str).
    """
    s = sent.strip()

    if PAT_CONTRIBUTION.search(s):
        return 1, "HIGH", "Cat 4: Core proposed contribution introducing novel framework/model"

    if PAT_NEGATIVE_FINDING.search(s):
        return 1, "HIGH", "Cat 2: Negative or null empirical finding"

    if PAT_EMPIRICAL_RESULT.search(s):
        if PAT_BACKGROUND.search(s) and ("prior" in s.lower() or "previous" in s.lower() or "et al" in s.lower()):
            return 0, "HIGH", "Cat 5 / 10: Empirical finding attributed to prior literature/citations"
        return 1, "HIGH", "Cat 1/3: Primary empirical outcome or quantitative statistical result"

    if PAT_LIMITATION.search(s):
        return 0, "HIGH", "Cat 9: Caveat, study limitation, or future work"

    if PAT_HYPOTHESIS.search(s):
        return 0, "HIGH", "Cat 8: Stated hypothesis or pre-experimental study objective"

    if PAT_DATASET_SPEC.search(s):
        return 0, "HIGH", "Cat 7: Dataset, material composition, or parameter specification"

    if PAT_METHOD.search(s):
        return 0, "HIGH", "Cat 6: Experimental procedure, methodology, or computational setup"

    if PAT_BACKGROUND.search(s):
        return 0, "HIGH", "Cat 5: Background literature, accepted domain context, or prior work"

    if section in ("Methods", "Methodology"):
        return 0, "MEDIUM", "Cat 6: Procedural/methodological sentence in Methods section"

    if section in ("Introduction", "Background") and ("is" in s or "are" in s) and not any(w in s.lower() for w in ["we show", "we find", "our results", "demonstrates"]):
        return 0, "MEDIUM", "Cat 5: Contextual background sentence in Introduction"

    if section in ("Results", "Abstract") and any(w in s.lower() for w in ["increase", "decrease", "consistent with", "suggests", "leads to", "indicates", "higher", "lower", "yields"]):
        return 1, "MEDIUM", "Cat 1: Empirical observation or outcome in Results/Abstract"

    return 0, "LOW", "Cat 10: Ambiguous or generic statement without decisive finding cues"


def validate_checkpoint_record(rec: Dict[str, Any], valid_ids: Set[str]) -> bool:
    """Strictly validates that a checkpoint record has all required fields and valid schema."""
    if not isinstance(rec, dict):
        return False
    if rec.get("sample_id") not in valid_ids:
        return False
    if rec.get("preliminary_label") not in (0, 1):
        return False
    if rec.get("confidence") not in ("HIGH", "MEDIUM", "LOW"):
        return False
    if not rec.get("rationale"):
        return False
    if rec.get("guidelines_label") not in (0, 1):
        return False
    return True


def load_checkpoint(valid_ids: Set[str]) -> Dict[str, Dict[str, Any]]:
    """Loads existing valid records from the checkpoint file."""
    if not CHECKPOINT_PATH.exists():
        return {}

    records: Dict[str, Dict[str, Any]] = {}
    with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if validate_checkpoint_record(rec, valid_ids):
                    records[rec["sample_id"]] = rec
            except Exception as e:
                print(f"[WARN] Corrupt line {line_num} in checkpoint: {e}", file=sys.stderr)
    return records


def serialize_json_val(obj: Any) -> Any:
    """Helper to convert numpy/pandas types to python native types for json."""
    import numpy as np
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return str(obj)


def append_checkpoint_records(records: List[Dict[str, Any]]) -> None:
    """Appends records atomically to checkpoint."""
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = CHECKPOINT_PATH.with_suffix(".tmp")
    
    # Read existing if exists
    existing_lines = []
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            existing_lines = f.readlines()
            
    with open(temp_path, "w", encoding="utf-8") as f:
        f.writelines(existing_lines)
        for r in records:
            f.write(json.dumps(r, default=serialize_json_val, ensure_ascii=False) + "\n")
            
    # Atomic replace
    temp_path.replace(CHECKPOINT_PATH)


def load_deberta_classifier():
    """Initializes offline cached DeBERTa-v3 zero-shot pipeline with retry."""
    from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
    model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    
    for attempt in range(3):
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True)
            clf = pipeline(
                "zero-shot-classification",
                model=model,
                tokenizer=tokenizer,
                device=-1,  # CPU
            )
            return clf
        except Exception as e:
            if attempt == 2:
                print(f"[WARN] Could not load local DeBERTa: {e}", file=sys.stderr)
                return None
            time.sleep(1)
    return None


def run_resumable_labeling(batch_size: int = 8) -> Tuple[int, int, bool]:
    print("=" * 80)
    print("AERF Resumable Independent Preliminary Label Generator")
    print("=" * 80)

    if not INPUT_CSV.exists():
        print(f"[ERROR] Input dataset missing: {INPUT_CSV}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)
    valid_ids = set(df["sample_id"].tolist())
    print(f"Total candidate sentences in source dataset: {len(df)}")

    # Load existing checkpoint
    completed = load_checkpoint(valid_ids)
    print(f"Recovered from checkpoint: {len(completed)} / {len(df)} valid records.")

    remaining_indices = [idx for idx, row in df.iterrows() if row["sample_id"] not in completed]
    print(f"Sentences remaining to classify: {len(remaining_indices)}")

    if not remaining_indices:
        print("[INFO] All sentences already processed. Generating final artifacts...")
        write_final_outputs(list(completed.values()), df)
        return len(completed), 0, True

    # Initialize DeBERTa
    print("\nLoading independent DeBERTa-v3 NLI classifier (offline cached)...")
    t0 = time.time()
    deberta = load_deberta_classifier()
    deberta_available = (deberta is not None)
    if deberta_available:
        print(f"DeBERTa-v3 loaded successfully in {time.time() - t0:.2f}s.")
    else:
        print("[NOTICE] Running with guidelines-based semantic classifier (DeBERTa unavailable).")

    CANDIDATE_LABELS = [
        "reports an empirical finding or scientific result",
        "describes methodology or experimental setup",
        "describes background literature or prior work",
        "describes study limitations or future work"
    ]
    HYPOTHESIS_TEMPLATE = "This sentence {}."

    # Process in batches
    print(f"\nProcessing remaining {len(remaining_indices)} sentences in batches of {batch_size}...")
    batch_records = []

    for i in range(0, len(remaining_indices), batch_size):
        batch_idx_slice = remaining_indices[i:i + batch_size]
        batch_rows = [df.iloc[idx] for idx in batch_idx_slice]
        texts = [r["sentence"] for r in batch_rows]

        # 1. Guidelines classification
        g_results = [classify_by_guidelines(r["sentence"], r["source_section"]) for r in batch_rows]

        # 2. DeBERTa classification
        d_results = []
        if deberta_available:
            try:
                nli_outputs = deberta(
                    texts,
                    candidate_labels=CANDIDATE_LABELS,
                    hypothesis_template=HYPOTHESIS_TEMPLATE,
                    multi_label=False,
                    batch_size=len(texts),
                )
                if isinstance(nli_outputs, dict):
                    nli_outputs = [nli_outputs]
                for nli_res in nli_outputs:
                    top_label = nli_res["labels"][0]
                    top_score = nli_res["scores"][0]
                    d_label = 1 if top_label == "reports an empirical finding or scientific result" and top_score >= 0.50 else 0
                    d_finding_score = (
                        top_score if top_label == "reports an empirical finding or scientific result"
                        else next((score for lbl, score in zip(nli_res["labels"], nli_res["scores"]) if lbl == "reports an empirical finding or scientific result"), 0.0)
                    )
                    d_results.append((d_label, d_finding_score, top_label))
            except Exception as e:
                print(f"[WARN] DeBERTa batch error: {e}. Falling back to guidelines for this batch.", file=sys.stderr)
                d_results = [(None, 0.0, "deberta_error") for _ in batch_rows]
        else:
            d_results = [(None, 0.0, "deberta_unavailable") for _ in batch_rows]

        # 3. Synthesize & Checkpoint
        new_records = []
        for r, (g_label, g_conf, g_reason), (d_label, d_score, d_top) in zip(batch_rows, g_results, d_results):
            sid = r["sample_id"]
            pid = r["paper_id"]
            sec = r["source_section"]
            page = r["page_number"]
            doi = r["doi_or_url"]
            sent = r["sentence"]

            if d_label is not None:
                disagreement = (g_label != d_label)
                is_uncertain = (g_conf == "LOW" or 0.40 <= d_score <= 0.60 or disagreement)
                if disagreement:
                    consensus_conf = "LOW"
                    requires_review = True
                    prelim_label = g_label if g_conf == "HIGH" else d_label
                    consensus_reason = f"[DISAGREEMENT: DeBERTa={d_label} ({d_score:.2f})] Guidelines: {g_reason}"
                elif is_uncertain:
                    consensus_conf = "MEDIUM" if g_conf != "LOW" else "LOW"
                    requires_review = True
                    prelim_label = g_label
                    consensus_reason = f"[UNCERTAIN] {g_reason}; DeBERTa finding score={d_score:.2f}"
                else:
                    consensus_conf = "HIGH"
                    requires_review = False
                    prelim_label = g_label
                    consensus_reason = f"{g_reason} (DeBERTa score={d_score:.2f})"
                provenance = "dual_guidelines_deberta_consensus"
            else:
                consensus_conf = g_conf
                requires_review = (g_conf == "LOW")
                prelim_label = g_label
                consensus_reason = f"{g_reason} (Heuristic/Guidelines only)"
                provenance = "guidelines_heuristics_only"

            record = {
                "sample_id": str(sid),
                "paper_id": str(pid),
                "source_section": str(sec),
                "page_number": int(page) if pd.notna(page) else 1,
                "sentence": str(sent),
                "preliminary_label": int(prelim_label),
                "confidence": str(consensus_conf),
                "rationale": str(consensus_reason),
                "guidelines_label": int(g_label),
                "guidelines_confidence": str(g_conf),
                "guidelines_reason": str(g_reason),
                "deberta_label": int(d_label) if d_label is not None else None,
                "deberta_finding_score": round(float(d_score), 4),
                "deberta_top_label": str(d_top),
                "requires_review": bool(requires_review),
                "doi_or_url": str(doi),
                "model_provenance": str(provenance),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            new_records.append(record)
            completed[sid] = record

        # Atomic append to checkpoint
        append_checkpoint_records(new_records)
        processed_count = len(completed)
        print(f"  [PROGRESS] {processed_count} / {len(df)} processed ({processed_count / len(df) * 100:.1f}%)")

    # Final outputs
    write_final_outputs(list(completed.values()), df)
    return len(completed), 0, deberta_available


def write_final_outputs(all_records: List[Dict[str, Any]], df_ref: pd.DataFrame) -> None:
    """Writes claim_extractor_auto_labels.csv and independent_review_queue.csv atomically."""
    # Ensure ordered by input DataFrame sample_id
    id_to_record = {r["sample_id"]: r for r in all_records}
    ordered_records = [id_to_record[sid] for sid in df_ref["sample_id"] if sid in id_to_record]

    # 1. Output All
    tmp_all = OUTPUT_ALL_CSV.with_suffix(".tmp")
    fieldnames_all = list(ordered_records[0].keys())
    with open(tmp_all, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_all)
        writer.writeheader()
        writer.writerows(ordered_records)
    tmp_all.replace(OUTPUT_ALL_CSV)
    print(f"\n[ATOMIC WRITE] Saved all {len(ordered_records)} preliminary labels to: {OUTPUT_ALL_CSV}")

    # 2. Output Review Queue (only requires_review == True)
    review_records = []
    for r in ordered_records:
        if r.get("requires_review"):
            review_records.append({
                "sample_id": r["sample_id"],
                "paper_id": r["paper_id"],
                "source_section": r["source_section"],
                "page_number": r["page_number"],
                "sentence": r["sentence"],
                "recommended_label": r["preliminary_label"],
                "confidence": r["confidence"],
                "guidelines_label": r["guidelines_label"],
                "deberta_label": r["deberta_label"],
                "deberta_finding_score": r["deberta_finding_score"],
                "review_reason": r["rationale"],
                "reviewer_decision": "",
                "reviewer_notes": "",
            })

    tmp_queue = OUTPUT_QUEUE_CSV.with_suffix(".tmp")
    if review_records:
        fieldnames_queue = list(review_records[0].keys())
        with open(tmp_queue, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames_queue)
            writer.writeheader()
            writer.writerows(review_records)
        tmp_queue.replace(OUTPUT_QUEUE_CSV)
        print(f"[ATOMIC WRITE] Saved review queue ({len(review_records)} items) to: {OUTPUT_QUEUE_CSV}")
    else:
        if OUTPUT_QUEUE_CSV.exists():
            OUTPUT_QUEUE_CSV.unlink()
        print("[INFO] Review queue is empty (all items high consensus).")

    # Summary
    df_res = pd.DataFrame(ordered_records)
    pos_count = (df_res["preliminary_label"] == 1).sum()
    neg_count = (df_res["preliminary_label"] == 0).sum()
    high_conf = (df_res["confidence"] == "HIGH").sum()
    med_conf = (df_res["confidence"] == "MEDIUM").sum()
    low_conf = (df_res["confidence"] == "LOW").sum()
    disagreements = (df_res["guidelines_label"] != df_res["deberta_label"]).sum()

    print("\n" + "=" * 80)
    print("Execution Summary")
    print("=" * 80)
    print(f"Total Sentences Processed : {len(df_res)}")
    print(f"Preliminary Claims (1)    : {pos_count} ({pos_count / len(df_res) * 100:.1f}%)")
    print(f"Preliminary Non-Claims (0): {neg_count} ({neg_count / len(df_res) * 100:.1f}%)")
    print(f"High Confidence Labels    : {high_conf} ({high_conf / len(df_res) * 100:.1f}%)")
    print(f"Medium Confidence Labels  : {med_conf} ({med_conf / len(df_res) * 100:.1f}%)")
    print(f"Low Confidence Labels     : {low_conf} ({low_conf / len(df_res) * 100:.1f}%)")
    print(f"Dual Model Disagreements  : {disagreements} ({disagreements / len(df_res) * 100:.1f}%)")
    print(f"Review Queue Size         : {len(review_records)} sentences ({len(review_records) / len(df_res) * 100:.1f}%)")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Resumable preliminary label generator.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for DeBERTa inference")
    args = parser.parse_args()

    run_resumable_labeling(batch_size=args.batch_size)


if __name__ == "__main__":
    main()
