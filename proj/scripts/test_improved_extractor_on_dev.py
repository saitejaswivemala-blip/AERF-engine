"""
scripts/test_improved_extractor_on_dev.py

Ablation & validation comparison on the 80-sentence development dataset
(tests/data/claim_extractor_eval.csv).
Compares:
1. Baseline Extractor (frozen)
2. Improved Discourse-Aware Hybrid Extractor
"""

import copy
import csv
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from nlp.claim_extractor import ScientificClaimExtractor, ClaimResult, _STRONG, _METHOD_LEAD, _BACKGROUND_MARK, _FIRST_PERSON, _CONTRIBUTION, _METRIC_NUMBER

# Additional discourse-aware patterns
_HYPOTHESIS_OBJECTIVE = re.compile(
    r"\b(?:we (?:hypothesi[zs]e[d]?|aim(?:ed)? to|sought to|set out to)|"
    r"the (?:aim|objective|goal|purpose) of this (?:study|work|paper) (?:is|was))\b",
    re.I
)

_EXTERNAL_LITERATURE = re.compile(
    r"\b(?:prior (?:work|studies|research)|previous (?:studies|work|trials|literature)|"
    r"earlier (?:work|studies)|recent (?:studies|trials|literature|meta-analysis)|"
    r"a \d{4} meta-analysis|contrasts with (?:recent )?(?:findings|results)|"
    r"in \[\d+\]|has been (?:shown|reported|suggested) by|\b[A-Z][a-z]+ et al\.,?\s*\(\d{4}\))\b",
    re.I
)

_LIMITATION_FUTURE = re.compile(
    r"\b(?:limitation[s]? of (?:this|our|the)|caveat[s]? of|future (?:work|research) (?:will|should|could)|"
    r"we plan to investigate|remains to be (?:determined|explored))\b",
    re.I
)

_PROCEDURAL_METHOD = re.compile(
    r"\b(?:were (?:encoded|analyzed|measured|computed|simulated|synthesized|prepared|tested|evaluated) (?:using|with|by|via)|"
    r"we (?:used|trained|employed|applied|computed|simulated|synthesized|prepared|measured|conducted|performed)|"
    r"for the (?:base )?model,? we (?:used|trained)|sensitivity analyses excluding)\b",
    re.I
)

# Expanded Scientific Result cues
_EXTRA_RESULT_CUES = [
    re.compile(r"\b(?:dispute[s]?|disputed|contradict[s]?|contradicted|refute[s]?|refuted|rule[s]? out)\b", re.I),
    re.compile(r"\b(?:did not|does not|failed to)\s+(?:significantly|substantially)?\s*(?:affect|influence|reduce|increase|alter|change|differ|show|produce)\b", re.I),
    re.compile(r"\bno (?:statistically )?significant (?:effect|difference|association|impact|reduction|increase|benefit|correlation)\b", re.I),
    re.compile(r"\b(?:was|were|is|are) (?:found|observed|shown|demonstrated|identified|determined) to (?:be|have|increase|decrease|exhibit|promote|suppress|enhance|undergo)\b", re.I),
    re.compile(r"\b(?:our |these )?(?:results|findings|data|evidence|observations)\s+(?:indicate[s]?|suggest[s]?|demonstrate[s]?|reveal[s]?|confirm[s]?|support[s]?)\s+(?:that|the conclusion that)\b", re.I),
    re.compile(r"\b(?:enhancement|increase|reduction|decrease|drop|gain) of (?:up to )?\d+(?:\.\d+)?\s*(?:%|ev|mah|bleu|db|k|km|v|ma|nm|g|mpa)\b", re.I),
]

COMBINED_STRONG = _STRONG + _EXTRA_RESULT_CUES


class ImprovedScientificClaimExtractor(ScientificClaimExtractor):
    """
    Improved discourse-aware hybrid claim extractor:
    - Adds negative discourse guards (hypotheses, literature citations, limitations, procedural setup)
    - Expands scientific result cues to cover negative/null findings, passive empirical assertions, and quantitative units
    """
    def cue_strength(self, s: str) -> str:
        method_lead = bool(_METHOD_LEAD.search(s))
        if any(p.search(s) for p in COMBINED_STRONG):
            return "strong"
        if _METRIC_NUMBER.search(s) and not method_lead:
            return "strong"
        if _CONTRIBUTION.search(s):
            return "contribution"
        return "none"

    def decide(self, s: str, probs: dict):
        f = probs.get(self.LABELS[0], 0.0)
        bg = probs.get(self.LABELS[1], 0.0)
        me = probs.get(self.LABELS[2], 0.0)
        ir = probs.get(self.LABELS[3], 0.0)
        top = max(probs, key=probs.get)
        method_lead = bool(_METHOD_LEAD.search(s))
        cue = self.cue_strength(s)
        background_style = bool(_BACKGROUND_MARK.search(s)) and not _FIRST_PERSON.search(s)

        # 1. Negative discourse guards (Precision guards)
        # Hypothesis / Objective guard
        if _HYPOTHESIS_OBJECTIVE.search(s) and cue == "none":
            return False, f, "hypothesis_or_objective"

        # External prior literature citation guard
        if _EXTERNAL_LITERATURE.search(s) and not any(k in s.lower() for k in ["our results", "here we", "we show", "we find", "dispute earlier claims"]):
            return False, f, "external_literature"

        # Study limitation or future work guard
        if _LIMITATION_FUTURE.search(s) and cue == "none":
            return False, f, "limitation_future"

        # Procedural method guard
        if _PROCEDURAL_METHOD.search(s) and cue == "none" and f < 0.75:
            return False, f, "procedural_method"

        # 2. Decision Logic
        if top == self.LABELS[0] and f >= self.threshold:
            if method_lead and cue == "none" and f < 0.80:
                return False, f, "method_lead"
            return True, f, "zero_shot"

        if ir >= 0.40 or top == self.LABELS[3]:
            return False, f, "irrelevant"

        # 3. Cue Rescue Logic
        if cue == "strong" and not background_style and not method_lead and f >= 0.15 and me < 0.85 and bg < 0.85:
            return True, f, "result_cue"

        if cue == "strong" and method_lead and f >= 0.30 and me < 0.85:
            return True, f, "result_cue"

        if cue == "contribution" and not background_style and f >= 0.10 and bg < 0.85 and me < 0.85:
            return True, f, "contribution_cue"

        return False, f, "rejected"


def evaluate_extractor(ext, rows):
    actual = [int(r["label"]) for r in rows]
    predicted = []
    reasons = []
    for r in rows:
        res = ext.extract(r["sentence"])
        if res:
            predicted.append(1)
            reasons.append(res[0].reason)
        else:
            predicted.append(0)
            reasons.append("rejected")

    acc = accuracy_score(actual, predicted)
    p = precision_score(actual, predicted, pos_label=1, zero_division=0)
    r = recall_score(actual, predicted, pos_label=1, zero_division=0)
    f1 = f1_score(actual, predicted, pos_label=1, zero_division=0)
    cm = confusion_matrix(actual, predicted, labels=[1, 0])
    return {
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f1,
        "tp": int(cm[0][0]),
        "fn": int(cm[0][1]),
        "fp": int(cm[1][0]),
        "tn": int(cm[1][1]),
    }


def main():
    dev_csv = Path("tests/data/claim_extractor_eval.csv")
    with open(dev_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Loaded {len(rows)} development samples.")
    base_ext = ScientificClaimExtractor()
    imp_ext = ImprovedScientificClaimExtractor(nlp=base_ext._nlp, classifier=base_ext._classifier_instance())

    base_res = evaluate_extractor(base_ext, rows)
    imp_res = evaluate_extractor(imp_ext, rows)

    print("\n" + "=" * 70)
    print("DEVELOPMENT DATASET COMPARISON (80 samples)")
    print("=" * 70)
    print(f"{'Metric':<18} | {'Baseline':<12} | {'Improved':<12} | {'Delta':<10}")
    print("-" * 70)
    for m in ["accuracy", "precision", "recall", "f1"]:
        b = base_res[m]
        i = imp_res[m]
        delta = i - b
        sign = "+" if delta >= 0 else ""
        print(f"{m:<18} | {b:<12.2%} | {i:<12.2%} | {sign}{delta:<10.2%}")
    print("-" * 70)
    print(f"Confusion Matrix Baseline: TP={base_res['tp']}, FP={base_res['fp']}, FN={base_res['fn']}, TN={base_res['tn']}")
    print(f"Confusion Matrix Improved: TP={imp_res['tp']}, FP={imp_res['fp']}, FN={imp_res['fn']}, TN={imp_res['tn']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
