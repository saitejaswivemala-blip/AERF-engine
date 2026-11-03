import json
import csv
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent

with open(ROOT / "reports/final_evaluation/final_metrics.json", encoding="utf-8") as f:
    metrics_json = json.load(f)

dev_imp = metrics_json["development_benchmark"]["improved_aerf_2_extractor"]
indep_imp = metrics_json["independent_provisional_benchmark"]["improved_aerf_2_extractor"]

print("=== FINAL_METRICS.JSON VERIFICATION ===")
print("Dev Improved Extractor:")
print(f"  Acc: {dev_imp['accuracy']:.4f}, Prec: {dev_imp['precision_claim']:.4f}, Rec: {dev_imp['recall_claim']:.4f}, F1: {dev_imp['f1_claim']:.4f}, Macro: {dev_imp['macro_f1']:.4f}, Wgt: {dev_imp['weighted_f1']:.4f}")
print(f"  TP={dev_imp['tp']}, FP={dev_imp['fp']}, FN={dev_imp['fn']}, TN={dev_imp['tn']}")

print("Indep Improved Extractor:")
print(f"  Acc: {indep_imp['accuracy']:.4f}, Prec: {indep_imp['precision_claim']:.4f}, Rec: {indep_imp['recall_claim']:.4f}, F1: {indep_imp['f1_claim']:.4f}, Macro: {indep_imp['macro_f1']:.4f}, Wgt: {indep_imp['weighted_f1']:.4f}")
print(f"  TP={indep_imp['tp']}, FP={indep_imp['fp']}, FN={indep_imp['fn']}, TN={indep_imp['tn']}")

print("\n=== BASELINE_VS_IMPROVED.CSV VERIFICATION ===")
with open(ROOT / "reports/final_evaluation/baseline_vs_improved.csv", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        if "AERF 2.0" in r["Model_Configuration"]:
            print(f"[{r['Dataset'][:15]}] Acc={r['Accuracy']}, Prec={r['Claim_Precision']}, Rec={r['Claim_Recall']}, F1={r['Claim_F1']}, Macro={r['Macro_F1']}, Wgt={r['Weighted_F1']}, CM={r['Confusion_Matrix']}")

print("\n=== FINAL_REPORT.MD VERIFICATION ===")
with open(ROOT / "reports/final_evaluation/final_report.md", encoding="utf-8") as f:
    text = f.read()

for line in text.splitlines():
    if "AERF 2.0 Improved" in line:
        print(line)
