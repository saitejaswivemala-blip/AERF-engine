# Cross-Paper Claim Matching and Natural Language Inference (NLI) Evaluation

**Module Under Audit**: Phase 3 Cross-Paper Alignment, Matching, and NLI Engine  
**Relevant Modules**: `matching/claim_matcher.py`, `nlp/nli/nli_classifier.py`, `nlp/embeddings.py`  
**Evaluation Script**: `scripts/evaluate_curated_mapping.py`, `scripts/evaluate_reviewed_mapping.py`  
**Ground-Truth Artifacts**:  
- `experiments/annotations/ground_truth_claim_pairs.json` (8 total curated pairs)  
- `experiments/annotations/curated_mapping_gt01_gt02_gt03_gt05.json` (4 cross-document evaluated pairs)  
- `experiments/results/curated_comparison_real_models.json` (Ablation results across 7 configurations)  
**Audit Date**: November 01, 2026  
**Auditor**: Antigravity Technical & Research Methodology Audit  

---

## 1. System Architecture: Two-Stage Pipeline

The AERF Phase 3 alignment architecture deliberately avoids running expensive $O(N^2)$ cross-encoder NLI across all possible claim pairs in a corpus. It enforces a strict two-stage gating funnel:

```
Claim A + Claim B
       │
       ▼ [Stage 1: Semantic Candidate Gating]
Dense MiniLM-L6-v2 Embeddings -> Dot-Product Cosine Similarity >= 0.55
       │
       ▼ [Stage 2: Entity Overlap Alignment Filter]
Biomedical/Scientific Entity Jaccard Overlap >= 0.15 (or 0.20)
       │
       ▼ [Stage 3: Cross-Encoder Inference]
DeBERTa-v3-base-mnli-fever-anli Cross-Encoder (Supports / Contradicts / Neutral)
       │
       ▼ [Stage 4: AERF Graph & Reliability Update]
Deterministic 7-Factor Reliability Scorer + Adaptive Neighborhood Propagation
```

### Architectural Principle
**Cosine similarity is strictly a candidate-generation filter; it NEVER creates a support or contradiction edge by itself.** A relationship edge is established only when candidate pairs successfully pass entity alignment and cross-encoder NLI validation.

---

## 2. Model Specifications & Parameters

1. **Sentence Embedding Model**:
   - Model: `sentence-transformers/all-MiniLM-L6-v2`
   - Dimension: 384 dense float32 vector dimensions.
   - Distance Metric: Cosine similarity ($[-1.0, 1.0]$).
   - Candidate Generation Threshold: $\tau_{\text{sim}} = 0.55$.
2. **Entity Alignment Filter**:
   - Model: SciSpaCy `en_core_sci_sm` (v0.5.4) biomedical entity parser.
   - Alignment Metric: Token Jaccard overlap over extracted scientific entity sets:
     $$J(E_A, E_B) = \frac{|E_A \cap E_B|}{|E_A \cup E_B|}$$
   - Configured Threshold: $\tau_{\text{entity}} = 0.15$ (pipeline default) or $0.20$ (conservative mode).
3. **Natural Language Inference (NLI) Model**:
   - Cross-Encoder: `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`
   - Task: Multi-class classification into `supports` (entailment), `contradicts` (contradiction), or `neutral`.
   - Max Sequence Length: 512 tokens.
   - Validation Confidence Cutoff: $P(\text{label}) \ge 0.60$.
   - Directionality Handling: Evaluated directionally ($A \to B$) and reverse ($B \to A$) to establish reciprocal or asymmetric stance.

---

## 3. Evaluation on Manually Curated Ground-Truth Claim Pairs

### A. Ground-Truth Dataset Inventory (`ground_truth_claim_pairs.json`)
The repository contains 8 manually authored claim-pair annotations evaluated across a 5-paper vitamin D and acute respiratory infection (ARI) corpus:

| Pair ID | Paper A | Paper B | Ground-Truth Claim A | Ground-Truth Claim B | Human Gold Label | Evaluation Scope |
|:---:|:---:|:---:|:---|:---|:---:|:---:|
| `gt_01` | P1 (Jolliffe 2024) | P2 (Bolland 2022) | Vitamin D showed no statistically significant overall effect (OR 0.94, 95% CI 0.88-1.00). | In IPD analysis, vitamin D had no statistically significant effect on ARI risk (OR 0.58, 95% CI 0.26-1.29). | **supports** | Cross-Paper Evaluated |
| `gt_02` | P1 (Jolliffe 2024) | P3 (CORONAVIT 2022) | Vitamin D showed no statistically significant overall effect (OR 0.94, 95% CI 0.88-1.00). | Test-and-treat vitamin D replacement did not reduce risk of all-cause ARI (OR 1.09-1.26). | **supports** | Cross-Paper Evaluated |
| `gt_03` | P2 (Bolland 2022) | P3 (CORONAVIT 2022) | In IPD analysis, vitamin D had no statistically significant effect on ARI risk (OR 0.58, 95% CI 0.26-1.29). | Test-and-treat vitamin D replacement did not reduce risk of all-cause ARI (OR 1.09-1.26). | **supports** | Cross-Paper Evaluated |
| `gt_04` | P1 (Jolliffe 2024) | P4 (PMC8955452) | Vitamin D did not significantly affect overall ARI risk. | Subgroup findings on baseline deficiency. | **neutral** | Unresolved (Claim B missing in DB) |
| `gt_05` | P4 (PMC8955452) | P1 (Jolliffe 2024) | Vitamin D supplementation reduced length of hospital stay in COVID-19 patients. | Vitamin D did not significantly affect overall ARI risk in prevention trials. | **neutral** | Cross-Paper Evaluated |
| `gt_06` | P4 (PMC8955452) | P5 (PMC12699866) | Treatment findings in hospitalized adults. | Pediatric prevention review findings. | **neutral** | Unresolved (Claim B missing in DB) |
| `gt_07` | P1 (Jolliffe 2024) | P1 (Jolliffe 2024) | Subgroup effect in baseline deficient patients ($<25$ nmol/L). | Overall null result across all participants. | **contradicts** | Out-of-Scope (Intra-Paper) |
| `gt_08` | P1 (Jolliffe 2024) | P1 (Jolliffe 2024) | Daily dosing regimen subgroup. | Overall null result. | **neutral** | Out-of-Scope (Intra-Paper) |

### B. Curated Diagnostic Evaluation Results ($N = 4$)
In `scripts/evaluate_curated_mapping.py` and `experiments/results/curated_comparison_real_models.json`, the 4 fully mapped cross-paper pairs were evaluated using real NLP models under isolated execution:

```
Pair ID    Human Label    Similarity Cosine    Entity Overlap    NLI Pred (No Gate)    Full AERF Pipeline Result
gt_01      supports       0.8512               0.1111            neutral (conf=0.987)  NOT_CLASSIFIED (Entity Rejected)
gt_02      supports       0.7041               0.0833            neutral (conf=0.984)  NOT_CLASSIFIED (Entity Rejected)
gt_03      supports       0.6805               0.0909            neutral (conf=0.978)  NOT_CLASSIFIED (Entity Rejected)
gt_05      neutral        0.8273               0.1429            contradicts / neutral NOT_CLASSIFIED (Entity Rejected)
```

### C. Diagnostic Performance Summary
- **Accuracy**: **0.000 (0 / 4 correct)**
- **Macro F1**: **0.000**
- **Full Configuration Outcome**: 0 / 4 classified (all 4 pairs rejected at the entity-alignment gate because entity Jaccard was $0.08 - 0.14$, below $\tau_{\text{entity}} = 0.15$).
- **Ablated Configuration Outcome (Entity Gate Bypassed)**: When entity alignment was removed, all 4 pairs reached DeBERTa-v3. DeBERTa-v3 predicted `neutral` on `gt_01`, `gt_02`, and `gt_03` (with $\ge 0.97$ confidence), and `contradicts`/`uncertain` on `gt_05`. Accuracy remained 0.000.

---

## 4. Technical Analysis & Failure Mode Diagnosis

1. **Why DeBERTa-v3 Predicts Neutral on Scientific Corroboration**:
   In scientific research, two independent randomized trials reporting null results (e.g., Jolliffe 2024 reporting $\text{OR}=0.94$ and Bolland 2022 reporting $\text{OR}=0.58$) corroborate each other's central scientific finding: *"vitamin D produces no overall preventive benefit"*. However, standard NLI models trained on general text (MNLI, FEVER) evaluate strict linguistic entailment. Because Trial A evaluated 40 studies with specific cohorts and Trial B evaluated individual patient data from the Blue Sky trial with different numbers, the NLI model correctly recognizes that Trial A's sentence does not grammatically entail Trial B's sentence, outputting `neutral` ($P > 0.97$).
2. **Entity Overlap Bottleneck**:
   Although both papers investigate vitamin D and ARI, one paper refers to `ARI risk`, `placebo`, and `I2`, while the other refers to `acute respiratory infection`, `odds ratio`, and `rate ratio`. Surface entity matching using SciSpaCy produces a low Jaccard overlap ($<0.15$) due to synonymy and phrasing differences, causing valid candidate pairs to be pruned.

---

## 5. Mandatory Manuscript Reporting Constraints

1. **Do NOT Claim High Cross-Paper NLI Accuracy**:
   The manuscript must explicitly declare that while AERF's cross-paper candidate matching and graph infrastructure execute end-to-end with real models, **cross-paper NLI accuracy is NOT quantitatively benchmarked on a statistical sample**.
2. **Transparent Reporting of the 4 Curated Pairs**:
   The diagnostic evaluation on the 4 curated pairs must be reported honestly as evidence that general-domain NLI models (such as DeBERTa-v3) struggle with scientific stance nuances without domain-specific fine-tuning.
3. **Distinguish Retrieval Gating from Stance Inference**:
   Cosine candidate gating successfully retrieves candidate pairs (similarity $> 0.68 - 0.85$), but entity alignment and zero-shot NLI represent active research challenges for future scientific knowledge graph work.
