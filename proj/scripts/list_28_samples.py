import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "reports/independent_eval/priority_review.csv")
for idx, r in df.iterrows():
    print(f"{idx+1}. {r['sample_id']}")
    print(f"   Sentence: {r['sentence']}")
    print(f"   Orig: {r['original_label']} | Prop: {r['ai_proposed_label']} | Model: {r['extractor_prediction']}")
    print(f"   Cat: {r['guideline_category']}")
    print("-" * 50)
