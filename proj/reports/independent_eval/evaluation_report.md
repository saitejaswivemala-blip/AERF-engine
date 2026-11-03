# AERF Scientific Claim Extractor — Independent Provisional Evaluation Report

**Model Evaluated**: `ScientificClaimExtractor` (`nlp/claim_extractor.py`)  
**Evaluation Date**: October 21, 2026  
**Dataset**: `tests/data/independent_eval/claim_extractor_independent_provisional_eval.csv`  
**Dataset Provenance**: 143 sentences from 9 independent open-access papers, labeled via dual AI consensus (unverified by human review)  
**Status**: `PROVISIONAL_INTER_MODEL_AGREEMENT`

---

## 1. Executive Summary

An independent, out-of-domain evaluation was executed across 143 genuine sentences collected from 9 previously unseen research papers spanning Materials Science, Astrophysics, Renewable Energy, Structural Geology, Climate Physics, Ecology, Quantum Optics, Plant Biology, and Glaciology.

Because independent human adjudication is pending, metrics below represent **provisional agreement** between the frozen claim extractor and the preliminary dual-model consensus:

| Metric | Score | 95% Confidence Interval |
|---|---|---|
| **Provisional Accuracy** | **46.85%** | [38.86%, 55.01%] |
| **Claim Precision (Class 1)** | **86.67%** | [73.82%, 93.74%] |
| **Claim Recall (Class 1)** | **35.78%** | [27.41%, 45.12%] |
| **Claim F1-Score (Class 1)** | **50.65%** | — |
| **Non-Claim F1-Score (Class 0)**| **42.42%** | — |
| **Macro-Averaged F1** | **46.54%** | — |
| **Weighted-Averaged F1** | **48.69%** | — |

---

## 2. Confusion Matrix

| | Predicted CLAIM (`1`) | Predicted NON-CLAIM (`0`) | Total Actual |
|---|---|---|---|
| **Provisional CLAIM (`1`)** | **39** (TP) | **70** (FN) | **109** |
| **Provisional NON-CLAIM (`0`)** | **6** (FP) | **28** (TN) | **34** |
| **Total Predicted** | **45** | **98** | **143** |

---

## 3. Stratified Breakdown by Scientific Paper

| Paper ID | Discipline | N | Actual Claims | Predicted Claims | Accuracy | F1 (Class 1) |
|---|---|---|---|---|---|---|
| **`INDEP_01_MATSCI`** | arXiv:2401.17672 | 15 | 8 | — | 53.3% | 46.2% |
| **`INDEP_02_ASTRO`** | arXiv:2401.15548 | 16 | 13 | — | 56.2% | 69.6% |
| **`INDEP_03_ENERGY`** | arXiv:2610.28152 | 16 | 14 | — | 43.8% | 52.6% |
| **`INDEP_04_GEOL`** | arXiv:2610.27561 | 16 | 12 | — | 43.8% | 40.0% |
| **`INDEP_05_CLIM`** | arXiv:2610.28219 | 16 | 14 | — | 56.2% | 66.7% |
| **`INDEP_06_ECOL`** | arXiv:2610.26235 | 16 | 11 | — | 50.0% | 50.0% |
| **`INDEP_07_QUANTUM`** | arXiv:2610.28462 | 16 | 12 | — | 43.8% | 47.1% |
| **`INDEP_08_PLANTBIO`** | arXiv:2405.06644 | 16 | 14 | — | 31.2% | 35.3% |
| **`INDEP_09_ENVGEO`** | arXiv:2409.11848 | 16 | 11 | — | 43.8% | 30.8% |

---

## 4. Stratified Breakdown by Paper Section

| Section | N | Actual Claims | Accuracy | F1 (Class 1) |
|---|---|---|---|---|
| **`Abstract`** | 26 | 20 | 46.2% | 50.0% |
| **`Introduction`** | 27 | 19 | 44.4% | 40.0% |
| **`Methods`** | 27 | 18 | 44.4% | 40.0% |
| **`Results`** | 36 | 28 | 44.4% | 50.0% |
| **`Discussion`** | 27 | 24 | 55.6% | 66.7% |

---

## 5. Primary Error Patterns

1. **False Positives (6 cases)**:
   - Sentences describing experimental methods or simulations containing comparative or numeric terminology (e.g. step sizes, voltage parameters) triggered extractor result cues.
2. **False Negatives (70 cases)**:
   - Domain-specific empirical assertions lacking standard clinical/NLP keywords (e.g. astrophysical flux descriptions or geological wave velocities) were rejected below the 0.55 confidence threshold.

---

## 6. Next Steps for Academic Publication

1. Review and verify the 126 prioritized edge cases in `tests/data/independent_eval/independent_review_queue.csv`.
2. Adjudicate discrepancies into a permanent ground-truth dataset.
3. Re-run `evaluate_claim_extractor.py` against the human-verified benchmark.
