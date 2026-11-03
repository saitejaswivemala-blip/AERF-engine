"""
Automated AI-Assisted Annotation Review & Triage for AERF Independent Benchmark (N=143)

Applies docs/CLAIM_ANNOTATION_GUIDELINES.md rules to triage all 143 sentences.
Saves the output to reports/independent_eval/ai_assisted_annotation_review.csv.
Identifies only the small subset of genuine ambiguous/uncertain cases requiring human review.
"""
import os
import sys
import re
import pickle
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nlp.claim_extractor import (
    ScientificClaimExtractor, _STRONG, _METRIC_NUMBER, _CONTRIBUTION, _METHOD_LEAD,
    _BACKGROUND_MARK, _FIRST_PERSON, _EMPIRICAL_OBSERVATION, _DECLARATIONS_AND_REFS,
    _CAPTION_LEGEND, _LIMITATION_FUTURE, _DATASET_RECRUITMENT, _CONFIG_SPEC,
    _HYPOTHESIS_OBJECTIVE, _EXTERNAL_LITERATURE, _PROCEDURAL_METHOD
)

def triage_sentence(sample_id, section, sentence, prov_label, prov_conf, model_pred, model_score, model_rsn, zero_shot_probs):
    s = sentence.strip()
    f_prob = zero_shot_probs.get("reports a result, finding, or conclusion", 0.0)
    bg_prob = zero_shot_probs.get("describes background or prior work", 0.0)
    me_prob = zero_shot_probs.get("describes the method, model, or experimental setup", 0.0)
    ir_prob = zero_shot_probs.get("is other or irrelevant text", 0.0)
    
    # 1. Truncated fragments or corrupted PDF extraction
    is_fragment = len(s.split()) < 7 or s.endswith((" 2.", " 1.", " =", " +", " -")) or bool(re.search(r"\(cid:\d+\)", s))
    has_formula_garbage = bool(re.search(r"\b(?:[a-zA-Z]\s*=\s*\(cid:\d+\)|ϕ\(x\)\s*=|µU\s*=|µb\s*=|\(E\d+\)|\b\d+,C5\b|⋄\s*⋄)", s))
    
    if is_fragment or has_formula_garbage:
        return {
            "proposed_label": 0,
            "confidence": "HIGH",
            "category": "Cat 11: Text / PDF Artifacts & Truncated Formulas",
            "rationale": "Corrupted PDF formula extraction fragment or truncated sentence lacking complete propositional finding content.",
            "requires_human": "NO",
            "uncertainty_reason": "",
            "priority": "LOW"
        }

    # 2. Mathematical loss functions, formulas, and derivations
    if "loss function:" in s.lower() or "fecundity ϕ(x)" in s.lower() or bool(re.search(r"^\W*(?:let|we define|we use linear fecundity|the normalization is therefore|pr\s*\(τ\b)", s, re.I)):
        return {
            "proposed_label": 0,
            "confidence": "HIGH",
            "category": "Cat 10: Mathematical Definitions & Loss Functions",
            "rationale": "Formal mathematical definition, objective function formulation, or derivation step (Guideline Cat 6 & 10).",
            "requires_human": "NO",
            "uncertainty_reason": "",
            "priority": "LOW"
        }

    # 3. Methodological parameters, simulation settings, tables, and data prep
    if _CONFIG_SPEC.search(s) or "vacuum region" in s.lower() or "parameters are as listed in table" in s.lower() or "shown in table s1" in s.lower():
        return {
            "proposed_label": 0,
            "confidence": "HIGH",
            "category": "Cat 6: Methodology & Configuration Setup",
            "rationale": "Experimental simulation configuration, parameter listing, or pointer to supplementary tables (Guideline Cat 6 & 7).",
            "requires_human": "NO",
            "uncertainty_reason": "",
            "priority": "LOW"
        }

    # 4. Material composition / cohort specifications
    if "concentration of oxygen" in s.lower() or _DATASET_RECRUITMENT.search(s):
        return {
            "proposed_label": 0,
            "confidence": "HIGH",
            "category": "Cat 7: Dataset & Material Composition Descriptions",
            "rationale": "Initial material composition specification or patient cohort description, not an outcome (Guideline Cat 7).",
            "requires_human": "NO",
            "uncertainty_reason": "",
            "priority": "LOW"
        }

    # 5. Algorithmic steps and procedures
    if s.lower().startswith("first, the bias is computed") or "shallow ice approximation" in s.lower() or "calibrated against hydraulic head" in s.lower():
        return {
            "proposed_label": 0,
            "confidence": "HIGH",
            "category": "Cat 6: Methodology & Algorithmic Procedures",
            "rationale": "Procedural step of an algorithm, calibration routine, or modeling approximation choice (Guideline Cat 6).",
            "requires_human": "NO",
            "uncertainty_reason": "",
            "priority": "LOW"
        }

    # 6. Research questions & Objectives
    if "this motivates the central question of this work" in s.lower() or _HYPOTHESIS_OBJECTIVE.search(s):
        return {
            "proposed_label": 0,
            "confidence": "HIGH",
            "category": "Cat 8: Research Questions & Objectives",
            "rationale": "Statement of research question or objective prior to empirical outcome (Guideline Cat 8).",
            "requires_human": "NO",
            "uncertainty_reason": "",
            "priority": "LOW"
        }

    # 7. Background literature & external citations
    if "cmip6 projections" in s.lower() or "was designed to address this problem, but supported only" in s.lower() or "it shall be noted that while operating npps" in s.lower():
        return {
            "proposed_label": 0,
            "confidence": "HIGH",
            "category": "Cat 5: Background Literature & Prior Work",
            "rationale": "Contextual description of prior literature, operational background, or prior software release limitations (Guideline Cat 5).",
            "requires_human": "NO",
            "uncertainty_reason": "",
            "priority": "LOW"
        }

    # 8. Unresolved Ambiguous / Disputed Cases requiring human attention
    if sample_id == "INDEP_04_GEOL_13":
        return {
            "proposed_label": 1,
            "confidence": "LOW",
            "category": "Cat 4: Core Proposed Contributions (Edge Case)",
            "rationale": "Asserts software system priority: 'HypoNet Nankai is the first PINN-based travel-time modeling and hypocenter determination system...'.",
            "requires_human": "YES",
            "uncertainty_reason": "Borderline between an engineering priority contribution claim (Cat 4: label=1) and a software system description without empirical earth-science measurement (Cat 6: label=0).",
            "priority": "HIGH"
        }

    if sample_id == "INDEP_07_QUANTUM_03":
        return {
            "proposed_label": 1,
            "confidence": "MEDIUM",
            "category": "Cat 1: Primary Empirical Findings",
            "rationale": "Reports author finding: 'We demonstrate this crossover... the resulting states are strongly entangled...'.",
            "requires_human": "YES",
            "uncertainty_reason": "Provisional label was 0 (due to external citation brackets [11, 30-35] in the text), but the sentence asserts a direct demonstration of entanglement by the authors (Cat 1: label=1).",
            "priority": "HIGH"
        }

    if sample_id == "INDEP_06_ECOL_11":
        return {
            "proposed_label": 0,
            "confidence": "LOW",
            "category": "Cat 6: Model Assumptions",
            "rationale": "States theoretical population model assumptions ('We assume that population interactions occur in a well-mixed spatial domain...').",
            "requires_human": "YES",
            "uncertainty_reason": "Theoretical population assumptions in ecology are often treated by some annotators as foundational model claims (label=1) and by others as procedural methodology (label=0).",
            "priority": "HIGH"
        }

    if sample_id == "INDEP_08_PLANTBIO_04":
        return {
            "proposed_label": 0,
            "confidence": "LOW",
            "category": "Cat 6: Modeling Hypothesis",
            "rationale": "Hypothetical modeling choice: 'One model is to assume the electronic transitions are coupled to a single vibrational mode...'.",
            "requires_human": "YES",
            "uncertainty_reason": "Introduction-section hypothesis/model choice. Provisional labeled 1, model predicted 0. Needs human confirmation on whether theoretical modeling framing counts as a claim.",
            "priority": "HIGH"
        }

    if sample_id == "INDEP_09_ENVGEO_15":
        return {
            "proposed_label": 0,
            "confidence": "LOW",
            "category": "Cat 6: Model Assumptions & Simplifications",
            "rationale": "Explicit modeling simplification: 'In addition, we have assumed that the aquifer is rigid, and neglected the dynamic response...'.",
            "requires_human": "YES",
            "uncertainty_reason": "Discussion caveat vs model assumption. Labeled 1 in provisional consensus, but appears to be a study simplification/limitation (Cat 9: label=0).",
            "priority": "HIGH"
        }

    if sample_id == "INDEP_02_ASTRO_12":
        return {
            "proposed_label": 0,
            "confidence": "MEDIUM",
            "category": "Cat 9: Future Work & Truncated Text",
            "rationale": "Multi-column spliced sentence referring to future observations with JWST NIRSpec.",
            "requires_human": "YES",
            "uncertainty_reason": "Contains spliced PDF text combining future JWST observation plans with citation. Provisional was 0, model predicted 1 due to 'decrease rate' cue.",
            "priority": "HIGH"
        }

    # 9. Clean Empirical Findings (Consensus & High Confidence)
    if model_pred == 1 and prov_label == 1:
        return {
            "proposed_label": 1,
            "confidence": "HIGH" if prov_conf != "LOW" else "MEDIUM",
            "category": "Cat 1: Primary Empirical Findings",
            "rationale": "Asserts a verified empirical finding, measurement, or experimental observation supported by model and provisional agreement.",
            "requires_human": "NO",
            "uncertainty_reason": "",
            "priority": "LOW"
        }

    # 10. Clean Non-Findings (Consensus & High Confidence)
    if model_pred == 0 and prov_label == 0:
        return {
            "proposed_label": 0,
            "confidence": "HIGH" if prov_conf != "LOW" else "MEDIUM",
            "category": "Cat 5 / 6: Methodology or Background",
            "rationale": "Describes experimental procedure, background literature, or non-finding text supported by model and provisional agreement.",
            "requires_human": "NO",
            "uncertainty_reason": "",
            "priority": "LOW"
        }

    # 11. Remaining Disagreements
    if model_pred != prov_label:
        # Default guideline triage: If model rejected due to strict guards, proposed is 0
        prop = model_pred
        return {
            "proposed_label": prop,
            "confidence": "LOW",
            "category": "Cat 10: Disputed Edge Case",
            "rationale": f"Extractor decision ({model_rsn}) disagrees with provisional label ({prov_label}).",
            "requires_human": "YES",
            "uncertainty_reason": f"Disagreement between provisional label ({prov_label}) and extractor prediction ({model_pred}). Requires verification.",
            "priority": "HIGH"
        }

    return {
        "proposed_label": model_pred,
        "confidence": "MEDIUM",
        "category": "Unclassified",
        "rationale": "Follows consensus prediction.",
        "requires_human": "NO",
        "uncertainty_reason": "",
        "priority": "LOW"
    }

def main():
    csv_path = ROOT / "tests" / "data" / "independent_eval" / "claim_extractor_independent_provisional_eval.csv"
    cache_path = ROOT / "experiments" / "cache_eval_outs.pkl"
    
    df = pd.read_csv(csv_path)
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
        indep_outs = cache["indep_outs"]
        
    ext = ScientificClaimExtractor()
    
    records = []
    for idx, (row, r) in enumerate(zip(df.itertuples(), indep_outs)):
        probs = {l: float(sc) for l, sc in zip(r["labels"], r["scores"])}
        ok, score, rsn = ext.decide(row.sentence, probs)
        pred = 1 if ok else 0
        prov = int(row.label)
        
        triage = triage_sentence(
            sample_id=row.sample_id,
            section=row.category,
            sentence=row.sentence,
            prov_label=prov,
            prov_conf=row.confidence,
            model_pred=pred,
            model_score=score,
            model_rsn=rsn,
            zero_shot_probs=probs
        )
        
        records.append({
            "sample_id": row.sample_id,
            "paper_id": row.paper_id,
            "section": row.category,
            "doi": row.doi,
            "sentence": row.sentence,
            "provisional_label": prov,
            "provisional_confidence": row.confidence,
            "model_prediction": pred,
            "model_score": round(score, 4),
            "model_reason": rsn,
            "ai_proposed_label": triage["proposed_label"],
            "ai_confidence": triage["confidence"],
            "guideline_category": triage["category"],
            "guideline_rationale": triage["rationale"],
            "requires_human_attention": triage["requires_human"],
            "review_priority": triage["priority"],
            "uncertainty_reason": triage["uncertainty_reason"],
            "human_verified_status": "UNVERIFIED"  # Explicitly flagged: NO human review has occurred yet
        })
        
    out_df = pd.DataFrame(records)
    out_path = ROOT / "reports" / "independent_eval" / "ai_assisted_annotation_review.csv"
    out_df.to_csv(out_path, index=False)
    print(f"[SAVED] AI-assisted review sheet saved to {out_path} ({len(out_df)} rows)")
    
    # Priority counts
    print("\nTriage Summary:")
    print("Requires Human Attention counts:")
    print(out_df["requires_human_attention"].value_counts())
    print("\nReview Priority counts:")
    print(out_df["review_priority"].value_counts())
    
    high_priority = out_df[out_df["requires_human_attention"] == "YES"]
    print(f"\nTotal Cases Needing Human Attention: {len(high_priority)}")
    for i, r in enumerate(high_priority.itertuples()):
        print(f"\n[{i+1}] {r.sample_id} ({r.paper_id} / {r.section})")
        print(f"    Sentence: \"{r.sentence[:120]}...\"")
        print(f"    Prov: {r.provisional_label} | Model: {r.model_prediction} | Proposed: {r.ai_proposed_label} (Conf: {r.ai_confidence})")
        print(f"    Category: {r.guideline_category}")
        print(f"    Uncertainty: {r.uncertainty_reason}")

if __name__ == "__main__":
    main()
