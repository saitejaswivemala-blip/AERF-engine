import sys, importlib.util, pickle
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

with open(ROOT / 'experiments' / 'cache_eval_outs.pkl', 'rb') as f:
    cache = pickle.load(f)
dev_outs = cache['dev_outs']
indep_outs = cache['indep_outs']

df_dev = pd.read_csv(ROOT / 'tests' / 'data' / 'claim_extractor_eval.csv')
dev_sents = df_dev['sentence'].tolist()
dev_labels = df_dev['label'].tolist()

df_indep = pd.read_csv(ROOT / 'tests' / 'data' / 'independent_eval' / 'claim_extractor_independent_provisional_eval.csv')
indep_sents = df_indep['sentence'].tolist()
indep_labels = df_indep['label'].tolist()

def load_ext(rel_path, mod_name):
    full_path = (ROOT / rel_path).resolve()
    code = full_path.read_text(encoding="utf-8")
    import types
    mod = types.ModuleType(mod_name)
    sys.modules[mod_name] = mod
    exec(code, mod.__dict__)
    return mod.ScientificClaimExtractor()

# Baseline 1: The earliest legacy baseline stored in reports/backups_pre_improvement/claim_extractor_baseline.py
ext_legacy = load_ext('reports/backups_pre_improvement/claim_extractor_baseline.py', 'ext_legacy')

# Baseline 2: The pre-improvement frozen baseline stored in nlp/claim_extractor.py.bak
ext_bak = load_ext('nlp/claim_extractor.py.bak', 'ext_bak')

# Current Best: nlp/claim_extractor.py
import nlp.claim_extractor as ce_curr
ext_curr = ce_curr.ScientificClaimExtractor()

def eval_m(ext, sents, labels, outs):
    preds = []
    for s, r in zip(sents, outs):
        probs = {l: float(sc) for l, sc in zip(r['labels'], r['scores'])}
        ok, score, rsn = ext.decide(s, probs)
        preds.append(1 if ok else 0)
    acc = accuracy_score(labels, preds)
    p = precision_score(labels, preds, zero_division=0)
    r = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(labels, preds, average='weighted', zero_division=0)
    cm = confusion_matrix(labels, preds)
    return acc, p, r, f1, macro_f1, weighted_f1, cm

print("="*80)
print("REPRODUCING ALL MODEL CONFIGURATIONS EXACTLY")
print("="*80)

for name, ext in [
    ("1. Earliest Legacy Baseline (backups_pre_improvement)", ext_legacy),
    ("2. Pre-Improvement Frozen Baseline (claim_extractor.py.bak)", ext_bak),
    ("3. Current Improved Extractor (nlp/claim_extractor.py)", ext_curr)
]:
    print(f"\n--- {name} ---")
    acc_d, p_d, r_d, f1_d, mac_d, wgt_d, cm_d = eval_m(ext, dev_sents, dev_labels, dev_outs)
    print(f"Dev Set (N=80, Expert Human Ground Truth):")
    print(f"  Acc: {acc_d:.2%}, Prec: {p_d:.2%}, Rec: {r_d:.2%}, F1: {f1_d:.2%}, Macro: {mac_d:.2%}, Wgt: {wgt_d:.2%}")
    print(f"  CM: TN={cm_d[0][0]}, FP={cm_d[0][1]}, FN={cm_d[1][0]}, TP={cm_d[1][1]}")

    acc_i, p_i, r_i, f1_i, mac_i, wgt_i, cm_i = eval_m(ext, indep_sents, indep_labels, indep_outs)
    print(f"Independent Set (N=143, Provisional AI Consensus):")
    print(f"  Acc: {acc_i:.2%}, Prec: {p_i:.2%}, Rec: {r_i:.2%}, F1: {f1_i:.2%}, Macro: {mac_i:.2%}, Wgt: {wgt_i:.2%}")
    print(f"  CM: TN={cm_i[0][0]}, FP={cm_i[0][1]}, FN={cm_i[1][0]}, TP={cm_i[1][1]}")
