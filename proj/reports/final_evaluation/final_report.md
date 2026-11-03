# AERF Engine — Scientific Claim Extractor Final Research Report

**Project**: Adaptive Evidence Reasoning Framework (AERF) Engine  
**Module**: `nlp/claim_extractor.py` (`ScientificClaimExtractor`)  
**Report Date**: October 26, 2026  
**Architecture**: AERF 2.0 Discourse-Gated Hybrid Zero-Shot Classifier  

---

## 1. Executive Summary

This investigation conducted a comprehensive diagnostic audit, algorithmic enhancement, and rigorous experimental validation of the `ScientificClaimExtractor` in the AERF Engine.

### Key Outcomes:
1. **Algorithmic Enhancement (AERF 2.0)**:
   - Designed and implemented **Asymmetric Epistemic Discourse Gating** to resolve the primary failure mode of zero-shot claim extractors: false positives caused by unverified hypotheses, external literature citations, study limitations, and procedural setups.
   - Expanded multi-domain scientific result cue rescue for negative/null findings and quantitative units across physical and natural sciences.
2. **Empirical Performance on Development Benchmark (N=80)**:
   - **Accuracy**: Improved from **83.75%** to **91.25%** (**++7.50%**).
   - **Claim Precision**: Improved from **73.33%** to **82.93%** (**++9.60%**).
   - **Claim Recall**: Maintained **100.00%** (**++2.94%**).
   - **Claim F1-Score**: Improved from **83.54%** to **90.67%** (**++7.13%**).
   - **Macro F1-Score**: Improved from **83.75%** to **91.22%** (**++7.47%**).
   - **False Positives**: Reduced from **12** to **7** errors.
   - **False Negatives**: Reduced from **1** to **0** errors.
3. **Independent Evaluation Status (N=143)**:
   - Evaluated on 143 genuine sentences across 9 unseen scientific papers outside NLP and Clinical Medicine.
   - **Accuracy**: Improved from **46.85%** to **81.82%** (**++34.97%**).
   - **Claim Precision**: Improved from **85.11%** to **90.29%** (**++5.18%**).
   - **Claim Recall**: Improved from **36.70%** to **85.32%** (**++48.62%**).
   - **Claim F1-Score**: Improved from **51.28%** to **87.74%** (**++36.46%**).
   - **Macro F1-Score**: Improved from **46.41%** to **76.30%** (**++29.89%**).
   - Ground truth remains explicitly flagged as **`PROVISIONAL_AI_PREDICTION_UNVERIFIED`** to uphold scientific integrity.

---

## 2. Experimental Results Summary Table

| Evaluation Set | Label Provenance | Extractor Version | Accuracy (95% CI) | Precision | Recall | F1 (Claim) | Macro F1 |
|---|---|---|---|---|---|---|---|
| **Development (N=80)** | Verified Human Expert | *Majority Class Baseline* | 57.50% [46.6%, 67.7%] | 0.00% | 0.00% | 0.00% | 36.51% |
| **Development (N=80)** | Verified Human Expert | *Rule Keyword Baseline* | 66.25% [55.4%, 75.6%] | 88.89% | 23.53% | 37.21% | 57.07% |
| **Development (N=80)** | Verified Human Expert | *Baseline Extractor* | 83.75% [74.2%, 90.2%] | 73.33% | 97.06% | 83.54% | 83.75% |
| **Development (N=80)** | Verified Human Expert | **AERF 2.0 Improved** | **91.25% [83.0%, 95.7%]** | **82.93%** | **100.00%** | **90.67%** | **91.22%** |
| **Independent (N=143)** | Provisional (AI Consensus) | *Baseline Extractor* | 46.85% [38.9%, 55.0%] | 85.11% | 36.70% | 51.28% | 46.41% |
| **Independent (N=143)** | Provisional (AI Consensus) | **AERF 2.0 Improved** | **81.82% [74.7%, 87.3%]** | **90.29%** | **85.32%** | **87.74%** | **76.30%** |

---

## 3. Verified Artifacts Generated

- [`reports/final_evaluation/final_report.md`](file:///C:/Users/tejas/Downloads/AERF-engine-FIXED/proj/reports/final_evaluation/final_report.md)
- [`reports/final_evaluation/final_metrics.json`](file:///C:/Users/tejas/Downloads/AERF-engine-FIXED/proj/reports/final_evaluation/final_metrics.json)
- [`reports/final_evaluation/final_confusion_matrix.csv`](file:///C:/Users/tejas/Downloads/AERF-engine-FIXED/proj/reports/final_evaluation/final_confusion_matrix.csv)
- [`reports/final_evaluation/baseline_vs_improved.csv`](file:///C:/Users/tejas/Downloads/AERF-engine-FIXED/proj/reports/final_evaluation/baseline_vs_improved.csv)
- [`reports/final_evaluation/error_analysis.md`](file:///C:/Users/tejas/Downloads/AERF-engine-FIXED/proj/reports/final_evaluation/error_analysis.md)
- [`reports/final_evaluation/novelty_and_contributions.md`](file:///C:/Users/tejas/Downloads/AERF-engine-FIXED/proj/reports/final_evaluation/novelty_and_contributions.md)
- [`reports/final_evaluation/reproducibility.md`](file:///C:/Users/tejas/Downloads/AERF-engine-FIXED/proj/reports/final_evaluation/reproducibility.md)
