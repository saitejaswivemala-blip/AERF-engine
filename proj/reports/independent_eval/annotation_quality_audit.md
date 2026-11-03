# AERF Independent Evaluation — Annotation Quality & Provenance Audit

**Audit Date**: October 21, 2026  
**Evaluation Scope**: 143 sentences across 9 unseen research papers  
**Ground Truth Status**: `PROVISIONAL_AI_PREDICTION_UNVERIFIED`

---

## 1. Academic Provenance & Integrity Statement

1. **No Human Validation**: The reference labels used in this preliminary evaluation were generated via dual automated consensus (Guidelines Semantic Classifier + DeBERTa-v3 zero-shot NLI). **No human annotator has verified or adjudicated these labels.**
2. **Model Independence**: The claim extractor being evaluated (`ScientificClaimExtractor` in `nlp/claim_extractor.py`) was **never** used to formulate, filter, or rank candidate labels.
3. **Review Queue Status**: 126 of the 143 sentences (88.1%) exhibited dual-model ambiguity or low confidence and were queued into `tests/data/independent_eval/independent_review_queue.csv`.
4. **Provisional Metric Notice**: The reported agreement rates indicate behavioral alignment between the extractor's rules/heuristics and the dual AI consensus, but **must not be claimed as true generalization accuracy in academic publications**.

---

## 2. Preliminary Agreement Breakdown

- **Total Evaluated Sentences**: 143
- **Provisional Claims (1)**: 109 (76.2%)
- **Provisional Non-Claims (0)**: 34 (23.8%)
- **Provisional Agreement (Accuracy)**: 46.85% (95% CI: [38.86%, 55.01%])
- **Provisional F1 (Claim Class)**: 50.65%
- **Provisional F1 (Macro Avg)**: 46.54%
