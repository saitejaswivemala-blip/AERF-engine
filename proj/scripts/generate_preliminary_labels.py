"""
scripts/generate_preliminary_labels.py

Generates preliminary evaluation labels for all 143 sentences in
tests/data/independent_eval/claim_extractor_independent_eval.csv without
running the model being evaluated (ScientificClaimExtractor in nlp/claim_extractor.py).

Dual Independent Pipeline:
1. Guidelines-Based Semantic Parser (docs/CLAIM_ANNOTATION_GUIDELINES.md):
   Strictly maps sentences to the 10 documented guideline categories.
2. Independent Local DeBERTa-v3 NLI Zero-Shot Classifier:
   (MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli).

Outputs:
- tests/data/independent_eval/claim_extractor_auto_labels.csv (all 143 rows)
- tests/data/independent_eval/independent_review_queue.csv (flagged uncertain/disagreements only)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import re
import sys
import time
from typing import Dict, List, Tuple

import pandas as pd
from transformers import pipeline

ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = ROOT / "tests" / "data" / "independent_eval" / "claim_extractor_independent_eval.csv"
OUTPUT_ALL = ROOT / "tests" / "data" / "independent_eval" / "claim_extractor_auto_labels.csv"
OUTPUT_QUEUE = ROOT / "tests" / "data" / "independent_eval" / "independent_review_queue.csv"

# --- Guidelines Semantic Categorizer Patterns ---

# Cat 5: Background / Prior Literature
PAT_BACKGROUND = re.compile(
    r"\b(?:previous (?:studies|work|research|literature)|prior (?:work|studies)|earlier work|"
    r"has been extensively (?:studied|researched)|widely considered|conventionally|traditionally|"
    r"is known to|are known to|well established|long recognized|literature (?:shows|suggests)|"
    r"et al\.,?\s*\(\d{4}\)|\(\w+ et al\.,?\s*\d{4}\)|recent(?:ly)? (?:studies|advances|efforts)|"
    r"has attracted (?:significant|considerable) attention|remains a challenge|is considered to be)\b",
    re.I
)

# Cat 6: Methodology / Experimental Procedure
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

# Cat 7: Dataset / Material Description / Hyperparameters
PAT_DATASET_SPEC = re.compile(
    r"\b(?:consists of|composed of|concentration of|dimensions of|thickness of|"
    r"target of research is|composition of|batch size|grid spacing|learning rate|"
    r"cutoff energy of|k-point mesh|pseudopotential|time step of)\b",
    re.I
)

# Cat 8: Hypotheses / Research Goals
PAT_HYPOTHESIS = re.compile(
    r"\b(?:we hypothesi[zs]e[d]?|we aim(?:ed)? to|the purpose of this (?:work|study)|"
    r"our objective (?:is|was)|to determine whether|to investigate whether|"
    r"is intended to|shall be investigated)\b",
    re.I
)

# Cat 9: Limitations / Future Work
PAT_LIMITATION = re.compile(
    r"\b(?:limitation[s]? of|caveat[s]?|is beyond the scope|further (?:investigation|research|work) is needed|"
    r"remains to be (?:elucidated|determined|explored)|future work will|we plan to)\b",
    re.I
)

# Cat 4: Core Proposed Contribution
PAT_CONTRIBUTION = re.compile(
    r"\b(?:we (?:propose|introduce|present|develop|demonstrate for the first time)|"
    r"this paper (?:proposes|introduces|presents))\s+a\s+(?:new|novel|simple|general|framework|method|model|approach)\b",
    re.I
)

# Cat 2: Negative / Null Findings
PAT_NEGATIVE_FINDING = re.compile(
    r"\b(?:did not (?:significantly|substantially|meaningfully)?\s+(?:affect|increase|decrease|change|alter|influence|improve)|"
    r"no (?:significant|statistically significant|detectable|appreciable|notable)?\s+(?:difference|effect|change|correlation|increase|decrease|enhancement)|"
    r"failed to (?:produce|show|demonstrate)|was not associated with|negligible (?:effect|change|difference))\b",
    re.I
)

# Cat 1 & 3: Quantitative & Primary Empirical Findings
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

    # Cat 4: Core Proposed Contribution (Label 1)
    if PAT_CONTRIBUTION.search(s):
        return 1, "HIGH", "Cat 4: Core proposed contribution introducing novel framework/model"

    # Cat 2: Negative/Null Finding (Label 1)
    if PAT_NEGATIVE_FINDING.search(s):
        return 1, "HIGH", "Cat 2: Negative or null empirical finding"

    # Cat 1 & 3: Primary Empirical / Quantitative Result (Label 1)
    if PAT_EMPIRICAL_RESULT.search(s):
        # Verify it is not attributing finding to past literature
        if PAT_BACKGROUND.search(s) and ("prior" in s.lower() or "previous" in s.lower() or "et al" in s.lower()):
            return 0, "HIGH", "Cat 5 / 10: Empirical finding attributed to prior literature/citations"
        return 1, "HIGH", "Cat 1/3: Primary empirical outcome or quantitative statistical result"

    # Cat 9: Limitations & Future Work (Label 0)
    if PAT_LIMITATION.search(s):
        return 0, "HIGH", "Cat 9: Caveat, study limitation, or future work"

    # Cat 8: Hypotheses & Research Questions (Label 0)
    if PAT_HYPOTHESIS.search(s):
        return 0, "HIGH", "Cat 8: Stated hypothesis or pre-experimental study objective"

    # Cat 7: Dataset & Material Specifications (Label 0)
    if PAT_DATASET_SPEC.search(s):
        return 0, "HIGH", "Cat 7: Dataset, material composition, or parameter specification"

    # Cat 6: Methodology & Experimental Actions (Label 0)
    if PAT_METHOD.search(s):
        return 0, "HIGH", "Cat 6: Experimental procedure, methodology, or computational setup"

    # Cat 5: Background & Prior Work (Label 0)
    if PAT_BACKGROUND.search(s):
        return 0, "HIGH", "Cat 5: Background literature, accepted domain context, or prior work"

    # Section-specific context fallbacks
    if section in ("Methods", "Methodology"):
        return 0, "MEDIUM", "Cat 6: Procedural/methodological sentence in Methods section"

    if section in ("Introduction", "Background") and ("is" in s or "are" in s) and not any(w in s.lower() for w in ["we show", "we find", "our results", "demonstrates"]):
        return 0, "MEDIUM", "Cat 5: Contextual background sentence in Introduction"

    if section in ("Results", "Abstract") and any(w in s.lower() for w in ["increase", "decrease", "consistent with", "suggests", "leads to", "indicates", "higher", "lower", "yields"]):
        return 1, "MEDIUM", "Cat 1: Empirical observation or outcome in Results/Abstract"

    # Uncertain edge case
    return 0, "LOW", "Cat 10: Ambiguous or generic statement without decisive finding cues"


def main():
    print("=" * 80)
    print("AERF Independent Claim Extractor: Dual Preliminary Label Generation")
    print("=" * 80)

    if not INPUT_CSV.exists():
        print(f"[ERROR] Input dataset missing: {INPUT_CSV}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} candidate sentences from: {INPUT_CSV.name}")

    # Load independent NLI model
    print("\nInitializing independent DeBERTa-v3 NLI zero-shot classifier...")
    t0 = time.time()
    deberta = pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        device=-1,  # CPU
    )
    print(f"Loaded DeBERTa-v3 classifier in {time.time() - t0:.2f}s.")

    CANDIDATE_LABELS = [
        "reports an empirical finding or scientific result",
        "describes methodology or experimental setup",
        "describes background literature or prior work",
        "describes study limitations or future work"
    ]
    HYPOTHESIS_TEMPLATE = "This sentence {}."

    auto_rows = []
    review_queue = []

    print("\nRunning dual annotation across all 143 sentences...")
    for idx, row in df.iterrows():
        sid = row["sample_id"]
        pid = row["paper_id"]
        sent = row["sentence"]
        sec = row["source_section"]
        page = row["page_number"]
        doi = row["doi_or_url"]

        # 1. Guidelines Semantic Classification
        g_label, g_conf, g_reason = classify_by_guidelines(sent, sec)

        # 2. DeBERTa Zero-Shot Classification
        nli_res = deberta(
            sent,
            candidate_labels=CANDIDATE_LABELS,
            hypothesis_template=HYPOTHESIS_TEMPLATE,
            multi_label=False
        )
        top_label = nli_res["labels"][0]
        top_score = nli_res["scores"][0]
        
        # DeBERTa binary decision: finding vs non-finding
        d_label = 1 if top_label == "reports an empirical finding or scientific result" and top_score >= 0.50 else 0
        d_finding_score = (
            top_score if top_label == "reports an empirical finding or scientific result"
            else next((score for lbl, score in zip(nli_res["labels"], nli_res["scores"]) if lbl == "reports an empirical finding or scientific result"), 0.0)
        )

        # 3. Consensus & Uncertainty Synthesis
        disagreement = (g_label != d_label)
        is_uncertain = (g_conf == "LOW" or 0.40 <= d_finding_score <= 0.60 or disagreement)

        if disagreement:
            consensus_conf = "LOW"
            requires_review = True
            # Prefer guidelines logic when specific guideline pattern triggered with HIGH confidence
            if g_conf == "HIGH":
                prelim_label = g_label
                consensus_reason = f"[DISAGREEMENT: DeBERTa={d_label} ({d_finding_score:.2f})] Prioritized Guidelines: {g_reason}"
            else:
                prelim_label = d_label
                consensus_reason = f"[DISAGREEMENT: Guidelines={g_label}] Prioritized DeBERTa: {top_label} ({top_score:.2f})"
        elif is_uncertain:
            consensus_conf = "MEDIUM" if g_conf != "LOW" else "LOW"
            requires_review = True
            prelim_label = g_label
            consensus_reason = f"[UNCERTAIN] {g_reason}; DeBERTa finding score={d_finding_score:.2f}"
        else:
            consensus_conf = "HIGH"
            requires_review = False
            prelim_label = g_label
            consensus_reason = f"{g_reason} (DeBERTa consensus score={d_finding_score:.2f})"

        record = {
            "sample_id": sid,
            "paper_id": pid,
            "source_section": sec,
            "page_number": page,
            "sentence": sent,
            "preliminary_label": prelim_label,
            "confidence": consensus_conf,
            "rationale": consensus_reason,
            "guidelines_label": g_label,
            "guidelines_confidence": g_conf,
            "guidelines_reason": g_reason,
            "deberta_label": d_label,
            "deberta_finding_score": round(d_finding_score, 4),
            "deberta_top_label": top_label,
            "requires_review": requires_review,
            "doi_or_url": doi,
        }
        auto_rows.append(record)

        if requires_review:
            review_queue.append({
                "sample_id": sid,
                "paper_id": pid,
                "source_section": sec,
                "page_number": page,
                "sentence": sent,
                "recommended_label": prelim_label,
                "guidelines_label": g_label,
                "deberta_label": d_label,
                "deberta_finding_score": round(d_finding_score, 4),
                "review_reason": consensus_reason,
                "reviewer_decision": "",
                "reviewer_notes": "",
            })

        if (idx + 1) % 25 == 0 or (idx + 1) == len(df):
            print(f"  Processed {idx + 1}/{len(df)} sentences... ({len(review_queue)} flagged for review)")

    # Save to claim_extractor_auto_labels.csv
    OUTPUT_ALL.parent.mkdir(parents=True, exist_ok=True)
    fieldnames_all = list(auto_rows[0].keys())
    with open(OUTPUT_ALL, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_all)
        writer.writeheader()
        writer.writerows(auto_rows)
    print(f"\n[SAVED] All 143 preliminary labels saved to: {OUTPUT_ALL}")

    # Save to independent_review_queue.csv
    if review_queue:
        fieldnames_queue = list(review_queue[0].keys())
        with open(OUTPUT_QUEUE, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames_queue)
            writer.writeheader()
            writer.writerows(review_queue)
        print(f"[SAVED] Review queue ({len(review_queue)} items) saved to: {OUTPUT_QUEUE}")
    else:
        print("[INFO] Zero rows flagged for review! High consensus across all items.")

    # Summary statistics
    df_res = pd.DataFrame(auto_rows)
    pos_count = (df_res["preliminary_label"] == 1).sum()
    neg_count = (df_res["preliminary_label"] == 0).sum()
    high_conf = (df_res["confidence"] == "HIGH").sum()
    med_conf = (df_res["confidence"] == "MEDIUM").sum()
    low_conf = (df_res["confidence"] == "LOW").sum()
    disagreements = (df_res["guidelines_label"] != df_res["deberta_label"]).sum()

    print("\n" + "=" * 80)
    print("Preliminary Auto-Labeling Summary")
    print("=" * 80)
    print(f"Total Sentences Processed : {len(df_res)}")
    print(f"Preliminary Claims (1)    : {pos_count} ({pos_count / len(df_res) * 100:.1f}%)")
    print(f"Preliminary Non-Claims (0): {neg_count} ({neg_count / len(df_res) * 100:.1f}%)")
    print(f"High Confidence Labels    : {high_conf} ({high_conf / len(df_res) * 100:.1f}%)")
    print(f"Medium Confidence Labels  : {med_conf} ({med_conf / len(df_res) * 100:.1f}%)")
    print(f"Low Confidence Labels     : {low_conf} ({low_conf / len(df_res) * 100:.1f}%)")
    print(f"Dual Model Disagreements  : {disagreements} ({disagreements / len(df_res) * 100:.1f}%)")
    print(f"Review Queue Size         : {len(review_queue)} sentences ({len(review_queue) / len(df_res) * 100:.1f}%)")
    print("=" * 80)


if __name__ == "__main__":
    main()
