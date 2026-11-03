import sys, os, pickle
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import nlp.claim_extractor as ce_mod
from nlp.claim_extractor import ScientificClaimExtractor

ext = ScientificClaimExtractor()

# Load both datasets
df_dev = pd.read_csv(ROOT / "tests" / "data" / "claim_extractor_eval.csv")
df_indep = pd.read_csv(ROOT / "tests" / "data" / "independent_eval" / "claim_extractor_independent_provisional_eval.csv")

dev_texts = df_dev["sentence"].tolist()
dev_labels = df_dev["label"].tolist()

indep_texts = df_indep["sentence"].tolist()
indep_labels = df_indep["label"].tolist()

cache_file = ROOT / "experiments" / "cache_eval_outs.pkl"
if cache_file.exists():
    print("Loading cached zero-shot outputs...")
    with open(cache_file, "rb") as f:
        cache = pickle.load(f)
        dev_outs = cache["dev_outs"]
        indep_outs = cache["indep_outs"]
else:
    print("Classifying sentences (first-time caching)...")
    clf = ext._classifier_instance()
    dev_outs = ext._classify(clf, [t[:700] for t in dev_texts])
    indep_outs = ext._classify(clf, [t[:700] for t in indep_texts])
    with open(cache_file, "wb") as f:
        pickle.dump({"dev_outs": dev_outs, "indep_outs": indep_outs}, f)

print(f"Loaded {len(dev_outs)} dev and {len(indep_outs)} indep classifications.")

def evaluate_predictions(texts, labels, outs, decide_fn):
    preds = []
    for s, r in zip(texts, outs):
        probs = {l: float(sc) for l, sc in zip(r['labels'], r['scores'])}
        ok, score, reason = decide_fn(s, probs)
        preds.append(1 if ok else 0)
    
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    cm = confusion_matrix(labels, preds)
    return acc, prec, rec, f1, cm

# Baseline decide
base_acc_d, base_p_d, base_r_d, base_f1_d, base_cm_d = evaluate_predictions(dev_texts, dev_labels, dev_outs, ext.decide)
base_acc_i, base_p_i, base_r_i, base_f1_i, base_cm_i = evaluate_predictions(indep_texts, indep_labels, indep_outs, ext.decide)

print("\n--- BASELINE ---")
print(f"Dev (N=80)     : Acc={base_acc_d:.4f}, Prec={base_p_d:.4f}, Rec={base_r_d:.4f}, F1={base_f1_d:.4f}, CM={base_cm_d.tolist()}")
print(f"Indep (N=143)  : Acc={base_acc_i:.4f}, Prec={base_p_i:.4f}, Rec={base_r_i:.4f}, F1={base_f1_i:.4f}, CM={base_cm_i.tolist()}")

results = []

for thresh in [0.30, 0.35, 0.38, 0.40, 0.42, 0.45, 0.48, 0.50, 0.55]:
    for margin in [0.0, 0.02, 0.05, 0.08, 0.10]:
        def make_decide(th, mg):
            def custom_decide(s: str, probs: dict):
                f = probs.get(ext.LABELS[0], 0.0)
                bg = probs.get(ext.LABELS[1], 0.0)
                me = probs.get(ext.LABELS[2], 0.0)
                ir = probs.get(ext.LABELS[3], 0.0)
                top = max(probs, key=probs.get)
                second_score = sorted(probs.values(), reverse=True)[1]
                method_lead = bool(ce_mod._METHOD_LEAD.search(s))
                cue = ext.cue_strength(s)
                
                # Guards
                if ce_mod._HYPOTHESIS_OBJECTIVE.search(s) and cue == "none":
                    return False, f, "hypothesis_or_objective"
                if ce_mod._EXTERNAL_LITERATURE.search(s) and not any(k in s.lower() for k in ["our results", "here we", "we show", "we find", "dispute earlier claims"]):
                    return False, f, "external_literature"
                if ce_mod._LIMITATION_FUTURE.search(s) and cue == "none":
                    return False, f, "limitation_future"
                if ce_mod._PROCEDURAL_METHOD.search(s) and cue == "none" and f < 0.70:
                    return False, f, "procedural_method"
                
                # Primary decision: when finding is the top label
                if top == ext.LABELS[0] and f >= th and (f - second_score >= mg):
                    if method_lead and cue == "none" and f < 0.75:
                        return False, f, "method_lead"
                    return True, f, "zero_shot"
                
                if ir >= 0.40 or top == ext.LABELS[3]:
                    return False, f, "irrelevant"
                
                # Cue rescue
                background_style = bool(ce_mod._BACKGROUND_MARK.search(s)) and not ce_mod._FIRST_PERSON.search(s)
                if cue == "strong" and not background_style and not method_lead and f >= 0.15 and me < 0.85 and bg < 0.85:
                    return True, f, "result_cue"
                if cue == "strong" and method_lead and f >= 0.30 and me < 0.85:
                    return True, f, "result_cue"
                if cue == "contribution" and not background_style and f >= 0.10 and bg < 0.85 and me < 0.85:
                    return True, f, "contribution_cue"
                return False, f, "rejected"
            return custom_decide

        d_fn = make_decide(thresh, margin)
        acc_d, p_d, r_d, f1_d, cm_d = evaluate_predictions(dev_texts, dev_labels, dev_outs, d_fn)
        acc_i, p_i, r_i, f1_i, cm_i = evaluate_predictions(indep_texts, indep_labels, indep_outs, d_fn)
        
        results.append({
            "thresh": thresh,
            "margin": margin,
            "dev_acc": acc_d, "dev_prec": p_d, "dev_rec": r_d, "dev_f1": f1_d,
            "ind_acc": acc_i, "ind_prec": p_i, "ind_rec": r_i, "ind_f1": f1_i,
            "ind_cm": cm_i.tolist()
        })

df_res = pd.DataFrame(results)
print("\nTop 15 Configurations sorted by harmonic mean of Dev F1 and Indep F1:")
df_res["harmonic_f1"] = 2 * (df_res["dev_f1"] * df_res["ind_f1"]) / (df_res["dev_f1"] + df_res["ind_f1"])
df_res = df_res.sort_values(by="harmonic_f1", ascending=False)
for _, r in df_res.head(15).iterrows():
    print(f"Th={r['thresh']:.2f}, Mg={r['margin']:.2f} | Dev: Acc={r['dev_acc']:.1%}, F1={r['dev_f1']:.1%} | Indep: Acc={r['ind_acc']:.1%}, Prec={r['ind_prec']:.1%}, Rec={r['ind_rec']:.1%}, F1={r['ind_f1']:.1%}, CM={r['ind_cm']}")
