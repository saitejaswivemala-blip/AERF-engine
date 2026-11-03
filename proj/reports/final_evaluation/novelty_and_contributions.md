# Technical Novelty & Academic Contributions of the AERF Scientific Claim Extractor

## 1. Prior Art & Baseline Limitations

Existing scientific claim extraction architectures in the literature generally fall into two extremes:
1. **Heavy Supervised Neural Models** (e.g., SciBERT, RoBERTa fine-tuned on PubMed-RCT or Evidence-Inference):
   - *Limitation*: Heavy computational footprint, high latency, domain-rigid transfer (models trained on clinical abstracts perform poorly on physical sciences), and vulnerability to false positives on sentences describing prior work.
2. **Standard Zero-Shot NLI Classification** (e.g., standard MNLI models with fixed prompt templates):
   - *Limitation*: Substantial recall bias. NLI models fail to distinguish *epistemic status* (whether an assertion is an unverified hypothesis, external background literature, or the author's own finding).

---

## 2. Technical Contribution in AERF Engine

AERF Engine implements a **Lightweight Discourse-Gated Hybrid Extractor** that bridges small zero-shot models and deterministic scientific discourse grammar.

### Primary Contribution:
- **Asymmetric Epistemic Discourse Gating**:
  Rather than relying solely on scalar zero-shot softmax probabilities, AERF couples zero-shot inference with an explicit epistemic discourse router. The router enforces strict non-finding guards against:
  - *Unverified Hypotheses & Study Aims* (`_HYPOTHESIS_OBJECTIVE`)
  - *External Literature & Citational Attribution* (`_EXTERNAL_LITERATURE`)
  - *Caveats, Scoping Constraints & Future Work* (`_LIMITATION_FUTURE`)
  - *Procedural Instrumentation Actions* (`_PROCEDURAL_METHOD`)

### Supporting Contributions:
1. **Multi-Domain Scientific Result Cue Routing**:
   A deterministic rescue mechanism specifically targeting non-biomedical empirical units (eV, mAh, MPa, BLEU, hazard ratios) and negative/refutational findings (`dispute`, `failed to show`, `no statistically significant effect`), recovering claims when zero-shot confidence falls below threshold.
2. **Zero-Overhead Exact Span Extraction**:
   Integrates robust multi-column glued-text segmentation and abbreviation-boundary merging directly into the tokenization pipeline, preserving exact byte offsets for document evidence synthesis without requiring neural boundary token classification.

---

## 3. Controlled Ablation Results (Development Benchmark, N=80)

| Configuration | Accuracy | Precision | Recall | F1 (Claim) | Macro F1 |
|---|---|---|---|---|---|
| **Majority Class Baseline** | 57.50% | 0.00% | 0.00% | 0.00% | 36.51% |
| **Keyword-Only Rule Baseline** | 78.75% | 76.67% | 67.65% | 71.88% | 77.83% |
| **Baseline Zero-Shot Extractor** | 83.75% | 73.33% | 97.06% | 83.54% | 83.25% |
| **AERF 2.0 (Full Discourse Hybrid)** | **92.50%** | **85.00%** | **100.00%** | **91.89%** | **92.20%** |

- **Impact of Epistemic Discourse Gating**: +11.67% Precision, +8.75% Accuracy, cutting False Positives from 12 to 6.
- **Impact of Expanded Scientific Cues**: +2.94% Recall (achieving 100.0% recall on development benchmark findings).

---

## 4. Threats to Validity & Limitations

1. **Development Dataset Size**: The verified development set contains 80 sentences across 7 papers. While statistically diverse across NLP and Clinical Medicine, larger multi-annotator corpora are needed for wider benchmarking.
2. **Provisional Independent Corpus**: The 143-sentence independent corpus across 9 physical and environmental domains currently relies on automated dual-model consensus labels. Until human dual annotation is finalized, out-of-domain metrics reflect inter-model agreement rather than certified accuracy.
