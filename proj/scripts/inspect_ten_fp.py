import pandas as pd
import pickle
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nlp.claim_extractor import ScientificClaimExtractor

df = pd.read_csv(ROOT / 'tests/data/independent_eval/claim_extractor_independent_provisional_eval.csv')
with open(ROOT / 'experiments/cache_eval_outs.pkl', 'rb') as f:
    cache = pickle.load(f)

ext = ScientificClaimExtractor()
fps = []
for idx, (row, r) in enumerate(zip(df.itertuples(), cache['indep_outs'])):
    probs = {l: float(sc) for l, sc in zip(r['labels'], r['scores'])}
    ok, score, rsn = ext.decide(row.sentence, probs)
    pred = 1 if ok else 0
    prov = int(row.label)
    if prov == 0 and pred == 1:
        fps.append((row.sample_id, row.category, rsn, score, row.sentence))

print(f'Total Prov 0 -> Pred 1: {len(fps)}')
for i, fp in enumerate(fps):
    print(f'[{i+1}] {fp[0]} ({fp[1]}) score={fp[3]:.4f} reason={fp[2]}:')
    print(f'    "{fp[4]}"\n')
