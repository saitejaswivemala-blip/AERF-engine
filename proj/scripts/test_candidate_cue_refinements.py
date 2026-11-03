import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import re
import csv
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from nlp.claim_extractor import (
    ScientificClaimExtractor, _STRONG, _METRIC_NUMBER, _CONTRIBUTION, _METHOD_LEAD,
    _BACKGROUND_MARK, _FIRST_PERSON, _EMPIRICAL_OBSERVATION, _DECLARATIONS_AND_REFS,
    _CAPTION_LEGEND, _LIMITATION_FUTURE, _DATASET_RECRUITMENT, _CONFIG_SPEC,
    _HYPOTHESIS_OBJECTIVE, _EXTERNAL_LITERATURE, _PROCEDURAL_METHOD
)

# Let's inspect potential pattern enhancements
# Pattern A: Expanding comparative physical adjectives in _STRONG:
# Currently: r"\b(?:significantly|substantially|consistently)\s+(?:higher|lower|better|worse|improv\w*|reduc\w*|increas\w*|outperform\w*|faster)"
# Enhanced A: r"\b(?:significantly|substantially|consistently)\s+(?:higher|lower|better|worse|steeper|shallower|greater|smaller|improv\w*|reduc\w*|increas\w*|outperform\w*|faster)"

# Pattern B: Contribution cue for system in subject position:
# Currently: r"\bwe (?:propose|introduce|present|develop|design)\b[^.]{0,40}\b(?:new|novel|simple|efficient|first|general)\b"
# Enhanced B: r"\b(?:is|are)\s+the\s+first\s+[^.]{0,60}\s+(?:system|model|framework|approach|architecture|pipeline|method)\b"

def run_simulation(enable_pattern_a=False, enable_pattern_b=False):
    with open(ROOT / "experiments" / "cache_eval_outs.pkl", "rb") as f:
        cache = pickle.load(f)
        dev_outs = cache["dev_outs"]
        indep_outs = cache["indep_outs"]
        
    with open(ROOT / "tests/data/claim_extractor_eval.csv", encoding="utf-8") as f:
        dev_rows = list(csv.DictReader(f))
    dev_sents = [r["sentence"] for r in dev_rows]
    dev_actual = [int(r["label"]) for r in dev_rows]
    
    df_indep = pd.read_csv(ROOT / "tests/data/independent_eval/claim_extractor_independent_provisional_eval.csv")
    indep_sents = df_indep["sentence"].tolist()
    indep_actual = df_indep["label"].astype(int).tolist()
    
    # Instantiate extractor
    ext = ScientificClaimExtractor()
    
    # Custom cue_strength for simulation
    def custom_cue_strength(s: str) -> str:
        method_lead = bool(_METHOD_LEAD.search(s))
        
        # Check standard strong
        strong_matches = any(p.search(s) for p in _STRONG)
        if enable_pattern_a:
            p_comp = re.compile(r"\b(?:significantly|substantially|consistently)\s+(?:higher|lower|better|worse|steeper|shallower|greater|smaller|improv\w*|reduc\w*|increas\w*|outperform\w*|faster)\b", re.I)
            if p_comp.search(s):
                strong_matches = True
                
        if strong_matches:
            return "strong"
        if _EMPIRICAL_OBSERVATION.search(s):
            return "strong"
        if _METRIC_NUMBER.search(s) and not method_lead:
            return "strong"
            
        contrib_matches = bool(_CONTRIBUTION.search(s))
        if enable_pattern_b:
            p_sys_contrib = re.compile(r"\b(?:is|are)\s+the\s+first\s+[^.]{0,60}\s+(?:system|model|framework|approach|architecture|pipeline|method)\b", re.I)
            if p_sys_contrib.search(s):
                contrib_matches = True
                
        if contrib_matches:
            return "contribution"
        return "none"
        
    def custom_decide(s: str, probs: dict):
        f = probs.get(ext.LABELS[0], 0.0)
        bg = probs.get(ext.LABELS[1], 0.0)
        me = probs.get(ext.LABELS[2], 0.0)
        ir = probs.get(ext.LABELS[3], 0.0)
        top = max(probs, key=probs.get)
        method_lead = bool(_METHOD_LEAD.search(s))
        cue = custom_cue_strength(s)
        background_style = bool(_BACKGROUND_MARK.search(s)) and not _FIRST_PERSON.search(s)

        if _DECLARATIONS_AND_REFS.search(s):
            return False, f, "declaration_ref"
        if _CAPTION_LEGEND.search(s) and not any(k in s.lower() for k in ["show that", "demonstrate", "reveals", "proves"]):
            return False, f, "caption_legend"
        if _LIMITATION_FUTURE.search(s) and cue == "none":
            return False, f, "limitation_future"
        if _DATASET_RECRUITMENT.search(s) and cue == "none" and f < 0.75:
            return False, f, "dataset_recruitment"
        if _CONFIG_SPEC.search(s) and cue == "none":
            return False, f, "config_spec"
        if _HYPOTHESIS_OBJECTIVE.search(s) and cue == "none":
            return False, f, "hypothesis_or_objective"
        if _EXTERNAL_LITERATURE.search(s) and not any(k in s.lower() for k in ["our results", "here we", "we show", "we find", "dispute earlier claims"]):
            return False, f, "external_literature"
        if _PROCEDURAL_METHOD.search(s) and cue == "none" and f < 0.70:
            return False, f, "procedural_method"

        if top == ext.LABELS[0] and f >= ext.threshold:
            if method_lead and cue == "none" and f < 0.75:
                return False, f, "method_lead"
            return True, f, "zero_shot"

        if ir >= 0.40 or top == ext.LABELS[3]:
            return False, f, "irrelevant"

        if cue == "strong" and not background_style and not method_lead and f >= 0.15 and me < 0.85 and bg < 0.85:
            return True, f, "result_cue"
        if cue == "strong" and method_lead and f >= 0.30 and me < 0.85:
            return True, f, "result_cue"
        if cue == "contribution" and not background_style and f >= 0.10 and bg < 0.85 and me < 0.85:
            return True, f, "contribution_cue"
        return False, f, "rejected"

    # Evaluate Dev
    dev_preds = []
    for s, r in zip(dev_sents, dev_outs):
        probs = {l: float(sc) for l, sc in zip(r["labels"], r["scores"])}
        ok, score, rsn = custom_decide(s, probs)
        dev_preds.append(1 if ok else 0)
        
    # Evaluate Indep
    indep_preds = []
    for s, r in zip(indep_sents, indep_outs):
        probs = {l: float(sc) for l, sc in zip(r["labels"], r["scores"])}
        ok, score, rsn = custom_decide(s, probs)
        indep_preds.append(1 if ok else 0)
        
    def calc_metrics(actual, preds):
        acc = accuracy_score(actual, preds)
        p1 = precision_score(actual, preds, pos_label=1, zero_division=0)
        r1 = recall_score(actual, preds, pos_label=1, zero_division=0)
        f1_1 = f1_score(actual, preds, pos_label=1, zero_division=0)
        mf1 = f1_score(actual, preds, average="macro", zero_division=0)
        cm = confusion_matrix(actual, preds, labels=[1, 0])
        return acc, p1, r1, f1_1, mf1, int(cm[0][0]), int(cm[1][0]), int(cm[0][1]), int(cm[1][1])

    dev_res = calc_metrics(dev_actual, dev_preds)
    indep_res = calc_metrics(indep_actual, indep_preds)
    
    # Check specific samples
    row_13 = df_indep[df_indep["sample_id"] == "INDEP_04_GEOL_13"].iloc[0]
    row_14 = df_indep[df_indep["sample_id"] == "INDEP_04_GEOL_14"].iloc[0]
    
    pred_13 = indep_preds[df_indep.index[df_indep["sample_id"] == "INDEP_04_GEOL_13"][0]]
    pred_14 = indep_preds[df_indep.index[df_indep["sample_id"] == "INDEP_04_GEOL_14"][0]]
    
    return dev_res, indep_res, pred_13, pred_14

print("Baseline Simulation (Current Candidates):")
dev_b, indep_b, p13_b, p14_b = run_simulation(False, False)
print(f"Dev:   Acc={dev_b[0]:.4f}, P={dev_b[1]:.4f}, R={dev_b[2]:.4f}, F1={dev_b[3]:.4f}, Macro={dev_b[4]:.4f} (TP={dev_b[5]}, FP={dev_b[6]}, FN={dev_b[7]}, TN={dev_b[8]})")
print(f"Indep: Acc={indep_b[0]:.4f}, P={indep_b[1]:.4f}, R={indep_b[2]:.4f}, F1={indep_b[3]:.4f}, Macro={indep_b[4]:.4f} (TP={indep_b[5]}, FP={indep_b[6]}, FN={indep_b[7]}, TN={indep_b[8]})")
print(f"Sample INDEP_04_GEOL_13 pred: {p13_b}")
print(f"Sample INDEP_04_GEOL_14 pred: {p14_b}")
print("-" * 70)

print("\nSimulation A (Enhance comparative physical adjectives with steeper/shallower/greater/smaller):")
dev_a, indep_a, p13_a, p14_a = run_simulation(True, False)
print(f"Dev:   Acc={dev_a[0]:.4f}, P={dev_a[1]:.4f}, R={dev_a[2]:.4f}, F1={dev_a[3]:.4f}, Macro={dev_a[4]:.4f} (TP={dev_a[5]}, FP={dev_a[6]}, FN={dev_a[7]}, TN={dev_a[8]})")
print(f"Indep: Acc={indep_a[0]:.4f}, P={indep_a[1]:.4f}, R={indep_a[2]:.4f}, F1={indep_a[3]:.4f}, Macro={indep_a[4]:.4f} (TP={indep_a[5]}, FP={indep_a[6]}, FN={indep_a[7]}, TN={indep_a[8]})")
print(f"Sample INDEP_04_GEOL_13 pred: {p13_a}")
print(f"Sample INDEP_04_GEOL_14 pred: {p14_a}")
print("-" * 70)

print("\nSimulation B (System Contribution in subject position):")
dev_b2, indep_b2, p13_b2, p14_b2 = run_simulation(False, True)
print(f"Dev:   Acc={dev_b2[0]:.4f}, P={dev_b2[1]:.4f}, R={dev_b2[2]:.4f}, F1={dev_b2[3]:.4f}, Macro={dev_b2[4]:.4f} (TP={dev_b2[5]}, FP={dev_b2[6]}, FN={dev_b2[7]}, TN={dev_b2[8]})")
print(f"Indep: Acc={indep_b2[0]:.4f}, P={indep_b2[1]:.4f}, R={indep_b2[2]:.4f}, F1={indep_b2[3]:.4f}, Macro={indep_b2[4]:.4f} (TP={indep_b2[5]}, FP={indep_b2[6]}, FN={indep_b2[7]}, TN={indep_b2[8]})")
print(f"Sample INDEP_04_GEOL_13 pred: {p13_b2}")
print(f"Sample INDEP_04_GEOL_14 pred: {p14_b2}")
print("-" * 70)

print("\nSimulation A + B:")
dev_ab, indep_ab, p13_ab, p14_ab = run_simulation(True, True)
print(f"Dev:   Acc={dev_ab[0]:.4f}, P={dev_ab[1]:.4f}, R={dev_ab[2]:.4f}, F1={dev_ab[3]:.4f}, Macro={dev_ab[4]:.4f} (TP={dev_ab[5]}, FP={dev_ab[6]}, FN={dev_ab[7]}, TN={dev_ab[8]})")
print(f"Indep: Acc={indep_ab[0]:.4f}, P={indep_ab[1]:.4f}, R={indep_ab[2]:.4f}, F1={indep_ab[3]:.4f}, Macro={indep_ab[4]:.4f} (TP={indep_ab[5]}, FP={indep_ab[6]}, FN={indep_ab[7]}, TN={indep_ab[8]})")
print(f"Sample INDEP_04_GEOL_13 pred: {p13_ab}")
print(f"Sample INDEP_04_GEOL_14 pred: {p14_ab}")
