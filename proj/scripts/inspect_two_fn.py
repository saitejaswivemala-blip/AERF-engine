import pandas as pd
import pickle
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nlp.claim_extractor import ScientificClaimExtractor, _STRONG, _METRIC_NUMBER, _EMPIRICAL_OBSERVATION, _CONTRIBUTION

df = pd.read_csv(ROOT / 'tests/data/independent_eval/claim_extractor_independent_provisional_eval.csv')
with open(ROOT / 'experiments/cache_eval_outs.pkl', 'rb') as f:
    cache = pickle.load(f)

extractor = ScientificClaimExtractor()

for sid in ['INDEP_04_GEOL_13', 'INDEP_04_GEOL_14']:
    row = df[df['sample_id'] == sid].iloc[0]
    s = row['sentence']
    print('='*70)
    print(f'SAMPLE ID: {sid}')
    print(f'Paper: {row["paper_id"]} | Section: {row["category"]} | DOI: {row["doi"]}')
    print(f'Provisional Label: {row["label"]} | Confidence: {row["confidence"]}')
    print(f'Sentence:\n"{s}"\n')
    
    # Check zero shot output
    r = None
    for s_text, out in zip(df['sentence'], cache['indep_outs']):
        if s_text == s:
            r = out
            break
    if r:
        print('Zero-Shot Probs:')
        for l, sc in zip(r['labels'], r['scores']):
            print(f'  {l}: {sc:.4f}')
        probs = {l: float(sc) for l, sc in zip(r['labels'], r['scores'])}
        ok, score, reason = extractor.decide(s, probs)
        print(f'\nExtractor Decision: is_claim={ok}, score={score:.4f}, reason={reason}')
        print(f'Cue strength: {extractor.cue_strength(s)}')
        print(f'Matches _STRONG: {any(bool(p.search(s)) for p in _STRONG)}')
        print(f'Matches _CONTRIBUTION: {bool(_CONTRIBUTION.search(s))}')
        print(f'Matches _EMPIRICAL_OBSERVATION: {bool(_EMPIRICAL_OBSERVATION.search(s))}')
        print(f'Matches _METRIC_NUMBER: {bool(_METRIC_NUMBER.search(s))}')
