import pandas as pd
from pathlib import Path
import hashlib
import sys

ROOT = Path(__file__).resolve().parent.parent

f1_path = ROOT / "tests/data/independent_eval/claim_extractor_independent_provisional_eval.csv"
f2_path = ROOT / "reports/independent_eval/ai_assisted_annotation_review.csv"
out_path = ROOT / "reports/independent_eval/priority_review.csv"

def get_hash(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

hash1_before = get_hash(f1_path)
hash2_before = get_hash(f2_path)

df1 = pd.read_csv(f1_path)
df2 = pd.read_csv(f2_path)

# Merge records on sample_id
merged = pd.merge(
    df1[["sample_id", "sentence", "label", "annotation_rationale"]],
    df2[[
        "sample_id", "ai_proposed_label", "model_prediction", "ai_confidence",
        "guideline_category", "guideline_rationale", "requires_human_attention",
        "uncertainty_reason"
    ]],
    on="sample_id",
    how="inner"
)

# Rename to required column names
merged["original_label"] = merged["label"]
merged["extractor_prediction"] = merged["model_prediction"]
merged["original_annotation_rationale"] = merged["annotation_rationale"]

# Condition: original_label differs from ai_proposed_label OR requires_human_attention == 'YES'
cond_diff = merged["original_label"] != merged["ai_proposed_label"]
cond_human = merged["requires_human_attention"] == "YES"
condition = cond_diff | cond_human

filtered = merged[condition].copy()

# Construct reason_for_review
reasons = []
for idx, r in filtered.iterrows():
    diff = r["original_label"] != r["ai_proposed_label"]
    human = r["requires_human_attention"] == "YES"
    unc = str(r["uncertainty_reason"]).strip() if pd.notna(r["uncertainty_reason"]) else ""
    
    if diff and human:
        rsn = f"Original provisional label ({r['original_label']}) differs from AI-proposed label ({r['ai_proposed_label']}). {unc}".strip()
    elif diff and not human:
        rsn = f"Original provisional label ({r['original_label']}) differs from AI-proposed label ({r['ai_proposed_label']}) based on {r['guideline_category']}."
    else: # not diff and human
        rsn = f"Flagged for human verification: {unc}".strip()
    reasons.append(rsn)

filtered["reason_for_review"] = reasons

# Desired columns
target_columns = [
    "sample_id",
    "sentence",
    "original_label",
    "ai_proposed_label",
    "extractor_prediction",
    "ai_confidence",
    "original_annotation_rationale",
    "guideline_category",
    "guideline_rationale",
    "reason_for_review"
]

out_df = filtered[target_columns].copy()
out_df.to_csv(out_path, index=False)

hash1_after = get_hash(f1_path)
hash2_after = get_hash(f2_path)

print(f"Total rows in priority review: {len(out_df)}")
print(f"Unique sample_id count: {out_df['sample_id'].nunique()}")
print(f"\nSource File Integrity:")
print(f"File 1 ({f1_path.name}): {'UNCHANGED' if hash1_before == hash1_after else 'CHANGED!'}")
print(f"  SHA-256: {hash1_after}")
print(f"File 2 ({f2_path.name}): {'UNCHANGED' if hash2_before == hash2_after else 'CHANGED!'}")
print(f"  SHA-256: {hash2_after}")
