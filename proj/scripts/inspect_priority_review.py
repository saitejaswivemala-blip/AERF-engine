import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "reports/independent_eval/priority_review.csv")

print(f"Total rows: {len(df)}")
for idx, r in df.iterrows():
    print(f"[{idx+1}] {r['sample_id']} | Orig={r['original_label']} | AI_Prop={r['ai_proposed_label']} | Model={r['extractor_prediction']} | Conf={r['ai_confidence']}")
    print(f"    Text: {r['sentence']}")
    print(f"    Orig Rationale: {r['original_annotation_rationale']}")
    print(f"    Guideline Cat: {r['guideline_category']}")
    print(f"    Guideline Rationale: {r['guideline_rationale']}")
    print(f"    Reason for Review: {r['reason_for_review']}")
    print("=" * 70)
