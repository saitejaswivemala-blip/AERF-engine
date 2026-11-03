import csv
import json
import math
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

ROOT = Path(r"C:\Users\tejas\Downloads\AERF-engine-FIXED\proj")
DATASET_PATH = ROOT / "tests" / "data" / "claim_extractor_eval.csv"
PREDICTIONS_PATH = ROOT / "reports" / "claim_extractor_predictions.csv"
OUTPUT_JSON = ROOT / "reports" / "claim_extractor_paper_split.json"

with PREDICTIONS_PATH.open("r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

# Papers with evidence of prior use in rules, diagnostic tests, or Phase 2/3 mappings
DEV_PAPERS = {
    "Vaswani et al. (2017)",
    "Bolland et al. (2022)",
    "Jolliffe et al. (2024)",
    "Jolliffe et al. (2022) CORONAVIT",
    "Zhang et al. (2022)"
}

# Papers not in diagnostic tests or prior reviewed claim mappings
HELD_OUT_PAPERS = {
    "Devlin et al. (2018)",
    "Martineau et al. (2017)"
}

dev_rows = [r for r in rows if r["paper_id"] in DEV_PAPERS]
held_out_rows = [r for r in rows if r["paper_id"] in HELD_OUT_PAPERS]

def calc_split_stats(split_rows):
    act = [int(r["actual_label"]) for r in split_rows]
    pred = [int(r["predicted_label"]) for r in split_rows]
    acc = accuracy_score(act, pred) if act else 0.0
    prec = precision_score(act, pred, zero_division=0) if act else 0.0
    rec = recall_score(act, pred, zero_division=0) if act else 0.0
    f1 = f1_score(act, pred, zero_division=0) if act else 0.0
    cm = confusion_matrix(act, pred, labels=[1, 0]) if act else [[0, 0], [0, 0]]
    tp, fn = map(int, cm[0])
    fp, tn = map(int, cm[1])
    
    # 95% Wilson score interval for accuracy
    n = len(act)
    z = 1.96
    p_hat = acc
    denom = 1 + (z**2) / n
    center = (p_hat + (z**2) / (2 * n)) / denom
    margin = (z * math.sqrt((p_hat * (1 - p_hat) / n) + (z**2) / (4 * n**2))) / denom
    ci_lower = max(0.0, center - margin)
    ci_upper = min(1.0, center + margin)

    return {
        "sample_count": n,
        "positive_claims": sum(act),
        "negative_non_claims": n - sum(act),
        "accuracy": float(acc),
        "accuracy_95ci": [round(ci_lower, 4), round(ci_upper, 4)],
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "confusion_matrix": cm.tolist()
    }

split_data = {
    "split_rationale": (
        "Partitioned at the paper level to prevent sentence leakage between sets. "
        "The development partition contains papers directly referenced in rule comments, "
        "unit test fixtures, or earlier experiment claim-mapping annotations. "
        "The held-out partition contains papers whose sentences were not used in unit test fixtures."
    ),
    "methodology_disclaimer": (
        "CAUTION FOR PUBLICATION: The held-out partition represents an internal diagnostic partition, "
        "NOT an independent out-of-domain test benchmark. Both Devlin et al. (NLP) and Martineau et al. (Vitamin D) "
        "share domain context with the development papers. Fully independent testing requires collecting papers "
        "from unseen scientific disciplines (e.g. physics, biochemistry, social science)."
    ),
    "development_tuning_partition": {
        "papers": sorted(list(DEV_PAPERS)),
        "metrics": calc_split_stats(dev_rows)
    },
    "diagnostic_held_out_partition": {
        "papers": sorted(list(HELD_OUT_PAPERS)),
        "metrics": calc_split_stats(held_out_rows)
    }
}

OUTPUT_JSON.write_text(json.dumps(split_data, indent=2), encoding="utf-8")
print(f"Generated paper split report at {OUTPUT_JSON}")
print("Dev split:", split_data["development_tuning_partition"]["metrics"])
print("Held-out split:", split_data["diagnostic_held_out_partition"]["metrics"])
