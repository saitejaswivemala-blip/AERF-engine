import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "reports/independent_eval/ai_assisted_annotation_review.csv")
high = df[df["requires_human_attention"] == "YES"]

print(f"Total High Priority Cases: {len(high)}\n")
for i, r in enumerate(high.itertuples()):
    print(f"[{i+1}] {r.sample_id} ({r.paper_id} / {r.section})")
    print(f"    Sentence: \"{r.sentence}\"")
    print(f"    Prov: {r.provisional_label} | Model: {r.model_prediction} | Proposed: {r.ai_proposed_label} (Conf: {r.ai_confidence})")
    print(f"    Category: {r.guideline_category}")
    print(f"    Rationale: {r.guideline_rationale}")
    print(f"    Uncertainty: {r.uncertainty_reason}")
    print("-" * 70)
