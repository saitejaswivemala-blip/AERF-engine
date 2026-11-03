import csv
import json
import re
from pathlib import Path
import pdfplumber

ROOT = Path(r"C:\Users\tejas\Downloads\AERF-engine-FIXED\proj")
DATASET_PATH = ROOT / "tests" / "data" / "claim_extractor_eval.csv"
SYNTHETIC_PATH = ROOT / "tests" / "data" / "claim_extractor_synthetic_eval.csv"
REPORTS_DIR = ROOT / "reports"

with DATASET_PATH.open("r", encoding="utf-8-sig") as f:
    eval_rows = list(csv.DictReader(f))

with SYNTHETIC_PATH.open("r", encoding="utf-8-sig") as f:
    synth_rows = list(csv.DictReader(f))

# ---------------------------------------------------------
# Part A: Leakage & Duplicate Audit
# ---------------------------------------------------------

# Read diagnostic test file and mapping files
diag_file = ROOT / "tests" / "unit" / "test_claim_extractor_diagnostic.py"
diag_text = diag_file.read_text(encoding="utf-8") if diag_file.exists() else ""

reviewed_mapping_file = ROOT / "experiments" / "annotations" / "reviewed_claim_mapping.json"
reviewed_claims = []
if reviewed_mapping_file.exists():
    rev_data = json.loads(reviewed_mapping_file.read_text(encoding="utf-8"))
    for m in rev_data.get("mappings", {}).values():
        if m.get("claim_a", {}).get("db_claim_text"):
            reviewed_claims.append((m["claim_a"]["db_claim_text"], "reviewed_claim_mapping:claim_a"))
        if m.get("claim_b", {}).get("db_claim_text"):
            reviewed_claims.append((m["claim_b"]["db_claim_text"], "reviewed_claim_mapping:claim_b"))

extractor_code_file = ROOT / "nlp" / "claim_extractor.py"
extractor_code = extractor_code_file.read_text(encoding="utf-8") if extractor_code_file.exists() else ""

def token_jaccard(s1, s2):
    t1 = set(re.findall(r"\w+", s1.lower()))
    t2 = set(re.findall(r"\w+", s2.lower()))
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)

leakage_records = []

# 1. Check exact duplicates within eval dataset
seen_exact = {}
for i, r in enumerate(eval_rows, 1):
    norm = re.sub(r"\s+", " ", r["sentence"].strip().lower())
    if norm in seen_exact:
        leakage_records.append({
            "audit_id": f"DUP_{i}",
            "sample_id": i,
            "sentence": r["sentence"],
            "target_artifact": "tests/data/claim_extractor_eval.csv",
            "leakage_or_duplicate_type": "EXACT_INTERNAL_DUPLICATE",
            "similarity_metric": "Exact String Match (1.00)",
            "similarity_score": 1.0,
            "evidence": f"Identical to Sample {seen_exact[norm]}",
            "severity": "HIGH",
            "recommended_action": "Deduplicate evaluation dataset"
        })
    else:
        seen_exact[norm] = i

# 2. Check near-duplicates within eval dataset
for i in range(len(eval_rows)):
    for j in range(i + 1, len(eval_rows)):
        s1 = eval_rows[i]["sentence"]
        s2 = eval_rows[j]["sentence"]
        jacc = token_jaccard(s1, s2)
        if jacc >= 0.70 and s1.strip().lower() != s2.strip().lower():
            leakage_records.append({
                "audit_id": f"NEAR_DUP_{i+1}_{j+1}",
                "sample_id": i + 1,
                "sentence": s1,
                "target_artifact": f"Sample {j+1}",
                "leakage_or_duplicate_type": "NEAR_DUPLICATE_INTERNAL",
                "similarity_metric": "Token Jaccard Similarity (threshold >= 0.70)",
                "similarity_score": round(jacc, 3),
                "evidence": f"Near duplicate with Sample {j+1}: '{s2}'",
                "severity": "MEDIUM",
                "recommended_action": "Review whether both sentences provide unique evaluation signal"
            })

# 3. Check overlap with synthetic dataset
for i, r in enumerate(eval_rows, 1):
    s_real = r["sentence"]
    for s_syn in synth_rows:
        jacc = token_jaccard(s_real, s_syn["sentence"])
        if jacc >= 0.60:
            leakage_records.append({
                "audit_id": f"SYN_OVERLAP_{i}",
                "sample_id": i,
                "sentence": s_real,
                "target_artifact": "tests/data/claim_extractor_synthetic_eval.csv",
                "leakage_or_duplicate_type": "REAL_SYNTHETIC_OVERLAP",
                "similarity_metric": "Token Jaccard",
                "similarity_score": round(jacc, 3),
                "evidence": f"Overlap with synthetic sample: {s_syn['sentence']}",
                "severity": "HIGH",
                "recommended_action": "Isolate synthetic data completely"
            })

# 4. Check overlap with diagnostic tests (test_claim_extractor_diagnostic.py)
for i, r in enumerate(eval_rows, 1):
    s = r["sentence"]
    norm_s = re.sub(r"\s+", " ", s.strip().lower())
    # Check if sentence or significant subclause is in diagnostic test
    if norm_s in diag_text.lower():
        leakage_records.append({
            "audit_id": f"DIAG_LEAK_{i}",
            "sample_id": i,
            "sentence": s,
            "target_artifact": "tests/unit/test_claim_extractor_diagnostic.py",
            "leakage_or_duplicate_type": "DEV_TEST_LEAKAGE",
            "similarity_metric": "Verbatim Substring in Diagnostic Fixtures",
            "similarity_score": 1.0,
            "evidence": "Exact string found in FINDINGS or NOT_FINDINGS of unit test suite",
            "severity": "CRITICAL",
            "recommended_action": "Must not treat as held-out test data; designate as development/tuning set"
        })
    else:
        # Check token Jaccard against diagnostic sentences
        diag_sents = re.findall(r'"([^"\n]{25,200})"', diag_text)
        for ds in diag_sents:
            jacc = token_jaccard(s, ds)
            if jacc >= 0.70:
                leakage_records.append({
                    "audit_id": f"DIAG_NEAR_LEAK_{i}",
                    "sample_id": i,
                    "sentence": s,
                    "target_artifact": "tests/unit/test_claim_extractor_diagnostic.py",
                    "leakage_or_duplicate_type": "DEV_TEST_PARAPHRASE_LEAKAGE",
                    "similarity_metric": "Token Jaccard (threshold >= 0.70)",
                    "similarity_score": round(jacc, 3),
                    "evidence": f"Near-match with diagnostic fixture: '{ds}'",
                    "severity": "HIGH",
                    "recommended_action": "Designate source paper as development paper, not unseen test"
                })
                break

# 5. Check overlap with reviewed claim mapping in experiments/annotations
for i, r in enumerate(eval_rows, 1):
    s = r["sentence"]
    for c_text, c_src in reviewed_claims:
        jacc = token_jaccard(s, c_text)
        if jacc >= 0.75:
            leakage_records.append({
                "audit_id": f"REV_MAP_LEAK_{i}",
                "sample_id": i,
                "sentence": s,
                "target_artifact": f"experiments/annotations/{c_src}",
                "leakage_or_duplicate_type": "ANNOTATION_CORPUS_REUSE",
                "similarity_metric": "Token Jaccard (threshold >= 0.75)",
                "similarity_score": round(jacc, 3),
                "evidence": f"Overlap with stored claim in Phase 2/3 experiment corpus: '{c_text[:70]}...'",
                "severity": "HIGH",
                "recommended_action": "Document that evaluation corpus shares text with Phase 2 experimental database"
            })
            break

# 6. Check overlap with nlp/claim_extractor.py regexes/cues
for i, r in enumerate(eval_rows, 1):
    s = r["sentence"]
    # Check if specific phrases in s were hardcoded in claim_extractor.py
    hardcoded_phrases = [
        "28.4 BLEU", "Transformer", "We propose a new simple network architecture",
        "WMT 2014 English-German", "superior in quality while being more parallelizable"
    ]
    for hp in hardcoded_phrases:
        if hp.lower() in s.lower() and hp.lower() in extractor_code.lower():
            leakage_records.append({
                "audit_id": f"CODE_RULE_LEAK_{i}",
                "sample_id": i,
                "sentence": s,
                "target_artifact": "nlp/claim_extractor.py",
                "leakage_or_duplicate_type": "RULE_DEVELOPMENT_OVERFIT_RISK",
                "similarity_metric": "Literal Match of Heuristic Regex Key",
                "similarity_score": 1.0,
                "evidence": f"Phrase '{hp}' appears in extractor docstrings, heuristic cues, or regexes",
                "severity": "HIGH",
                "recommended_action": "Vaswani et al. (2017) must be categorized as a Development/Rule-Tuning paper"
            })
            break

# Write leakage audit CSV
leakage_csv = REPORTS_DIR / "claim_extractor_leakage_audit.csv"
fieldnames_leak = [
    "audit_id", "sample_id", "sentence", "target_artifact",
    "leakage_or_duplicate_type", "similarity_metric", "similarity_score",
    "evidence", "severity", "recommended_action"
]
with leakage_csv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames_leak)
    writer.writeheader()
    for row in leakage_records:
        writer.writerow(row)

print(f"Generated leakage audit CSV with {len(leakage_records)} flags.")
