import sys, os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import pandas as pd
from nlp.claim_extractor import (
    ScientificClaimExtractor, _STRONG, _METRIC_NUMBER, _CONTRIBUTION,
    _METHOD_LEAD, _BACKGROUND_MARK, _HYPOTHESIS_OBJECTIVE, _EXTERNAL_LITERATURE,
    _LIMITATION_FUTURE, _PROCEDURAL_METHOD
)

ext = ScientificClaimExtractor()
clf = ext._classifier_instance()

csv_path = ROOT / "tests" / "data" / "independent_eval" / "claim_extractor_independent_provisional_eval.csv"
df = pd.read_csv(csv_path)
claims = df[df['label'] == 1].copy()

# Run classifier on all 109 provisional claims
texts = claims['sentence'].tolist()
outs = ext._classify(clf, [t[:700] for t in texts])

fn_records = []
for idx, (row, r) in enumerate(zip(claims.itertuples(), outs)):
    s = row.sentence
    probs = {l: float(sc) for l, sc in zip(r['labels'], r['scores'])}
    ok, score, reason = ext.decide(s, probs)
    if not ok:
        top_lbl = max(probs, key=probs.get)
        f = probs[ext.LABELS[0]]
        bg = probs[ext.LABELS[1]]
        me = probs[ext.LABELS[2]]
        ir = probs[ext.LABELS[3]]
        cue = ext.cue_strength(s)
        
        hyp = bool(_HYPOTHESIS_OBJECTIVE.search(s))
        ext_lit = bool(_EXTERNAL_LITERATURE.search(s))
        lim = bool(_LIMITATION_FUTURE.search(s))
        proc = bool(_PROCEDURAL_METHOD.search(s))
        meth_lead = bool(_METHOD_LEAD.search(s))
        bg_style = bool(_BACKGROUND_MARK.search(s))
        
        fn_records.append({
            'sample_id': row.sample_id,
            'top_label': top_lbl[:15],
            'f_score': round(f, 3),
            'me_score': round(me, 3),
            'bg_score': round(bg, 3),
            'ir_score': round(ir, 3),
            'cue': cue,
            'reason': reason,
            'guards': f'hyp={hyp},lit={ext_lit},lim={lim},proc={proc},lead={meth_lead},bg={bg_style}',
            'text': s[:90]
        })

df_fn = pd.DataFrame(fn_records)
print(f'Total FNs: {len(df_fn)}')
print('\nFN Reason counts:')
print(df_fn['reason'].value_counts())
print('\nFN Top label counts:')
print(df_fn['top_label'].value_counts())
print('\nFN Cue counts:')
print(df_fn['cue'].value_counts())
print('\nFN Score distribution for finding (f_score):')
print(df_fn['f_score'].describe())
print('\nTop 15 FNs where top_label is reports a result:')
print(df_fn[df_fn['top_label'].str.startswith('reports')][['sample_id', 'f_score', 'reason', 'text']].head(15))
print('\nTop 15 FNs where top_label is describes the met:')
print(df_fn[df_fn['top_label'].str.startswith('describes the met')][['sample_id', 'f_score', 'me_score', 'cue', 'text']].head(15))
