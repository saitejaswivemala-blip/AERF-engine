"""
Script to recalculate metrics based on human-adjudicated labels from:
reports/independent_eval/independent_human_review_adjudication_sheet.csv

If human adjudication is in progress or completed, it computes rigorous metrics
on the adjudicated subset. If 0 rows have been reviewed, it reports the current
audit status (0.0% reviewed) and provides guidelines for the reviewer.
"""
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def main():
    sheet_path = ROOT / "reports" / "independent_eval" / "independent_human_review_adjudication_sheet.csv"
    if not sheet_path.exists():
        print(f"Error: Adjudication sheet not found at {sheet_path}")
        sys.exit(1)
        
    df = pd.read_csv(sheet_path)
    total_samples = len(df)
    
    # Filter for valid adjudicated labels (0 or 1)
    df_adj = df[df["human_adjudicated_label"].astype(str).str.strip().isin(["0", "1", "0.0", "1.0"])].copy()
    num_adjudicated = len(df_adj)
    pct_reviewed = (num_adjudicated / total_samples) * 100.0
    
    print("=" * 70)
    print("AERF INDEPENDENT BENCHMARK ADJUDICATION STATUS AUDIT")
    print("=" * 70)
    print(f"Total benchmark samples:       {total_samples}")
    print(f"Human-adjudicated samples:     {num_adjudicated} ({pct_reviewed:.1f}%)")
    print(f"Pending human review:          {total_samples - num_adjudicated} ({100.0 - pct_reviewed:.1f}%)")
    
    # High priority review items
    high_priority = df[df["review_priority"] == "HIGH"]
    print(f"Disputed rows (Model != Prov): {len(high_priority)}")
    
    if num_adjudicated == 0:
        print("\n[NOTE] 0 rows currently contain human adjudicated labels.")
        print("To adjudicate samples:")
        print("  1. Open reports/independent_eval/independent_human_review_adjudication_sheet.csv")
        print("  2. Fill 'human_adjudicated_label' with 1 (Finding/Claim) or 0 (Non-claim)")
        print("  3. Fill 'human_adjudication_rationale' and 'human_reviewer_id'")
        print("  4. Re-run this script to compute human-validated benchmark metrics.")
        print("\nCurrent provisional metrics (evaluated against unverified dual consensus):")
        
        # Compute metrics against provisional labels
        actual = df["provisional_label"].astype(int).tolist()
        pred = df["model_prediction"].astype(int).tolist()
        
        acc = accuracy_score(actual, pred)
        p1 = precision_score(actual, pred, pos_label=1, zero_division=0)
        r1 = recall_score(actual, pred, pos_label=1, zero_division=0)
        f1_1 = f1_score(actual, pred, pos_label=1, zero_division=0)
        macro_f1 = f1_score(actual, pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(actual, pred, average="weighted", zero_division=0)
        cm = confusion_matrix(actual, pred, labels=[1, 0])
        tp, fn, fp, tn = int(cm[0][0]), int(cm[0][1]), int(cm[1][0]), int(cm[1][1])
        
        print(f"  Accuracy:     {acc:.4f} ({acc*100:.2f}%)")
        print(f"  Claim Prec:   {p1:.4f} ({p1*100:.2f}%)")
        print(f"  Claim Rec:    {r1:.4f} ({r1*100:.2f}%)")
        print(f"  Claim F1:     {f1_1:.4f} ({f1_1*100:.2f}%)")
        print(f"  Macro F1:     {macro_f1:.4f} ({macro_f1*100:.2f}%)")
        print(f"  Weighted F1:  {weighted_f1:.4f} ({weighted_f1*100:.2f}%)")
        print(f"  Confusion:    TP={tp}, FP={fp}, FN={fn}, TN={tn}")
        print("=" * 70)
        return
        
    # If human labels exist
    df_adj["human_label"] = df_adj["human_adjudicated_label"].astype(float).astype(int)
    actual = df_adj["human_label"].tolist()
    pred = df_adj["model_prediction"].astype(int).tolist()
    
    acc = accuracy_score(actual, pred)
    p1 = precision_score(actual, pred, pos_label=1, zero_division=0)
    r1 = recall_score(actual, pred, pos_label=1, zero_division=0)
    f1_1 = f1_score(actual, pred, pos_label=1, zero_division=0)
    macro_f1 = f1_score(actual, pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(actual, pred, average="weighted", zero_division=0)
    cm = confusion_matrix(actual, pred, labels=[1, 0])
    tp, fn, fp, tn = int(cm[0][0]), int(cm[0][1]), int(cm[1][0]), int(cm[1][1])
    
    print("\nMETRICS ON HUMAN-ADJUDICATED SUBSET:")
    print(f"  Adjudicated N: {num_adjudicated}")
    print(f"  Accuracy:      {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Claim Prec:    {p1:.4f} ({p1*100:.2f}%)")
    print(f"  Claim Rec:     {r1:.4f} ({r1*100:.2f}%)")
    print(f"  Claim F1:      {f1_1:.4f} ({f1_1*100:.2f}%)")
    print(f"  Macro F1:      {macro_f1:.4f} ({macro_f1*100:.2f}%)")
    print(f"  Weighted F1:   {weighted_f1:.4f} ({weighted_f1*100:.2f}%)")
    print(f"  Confusion:     TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print("=" * 70)

if __name__ == "__main__":
    main()
