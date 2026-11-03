# AERF Scientific Claim Annotation Guidelines

## Overview
This document specifies the operational definitions and annotation protocol used to evaluate the `ScientificClaimExtractor` in the AERF (Adaptive Evidence Reasoning Framework) Engine.

The goal of `ScientificClaimExtractor` is to identify sentences that assert a **scientific finding, empirical outcome, or core proposed contribution** of the study, while filtering out non-finding text such as background literature, experimental methodology, dataset details, hypotheses, and study limitations.

---

## Binary Annotation Schema

Each sentence is labeled with a binary ground-truth label:
- **`1` (Scientific Finding / Claim)**: A sentence asserting an empirical result, finding, negative/null outcome, quantitative statistical finding, or primary novel contribution of the paper.
- **`0` (Non-Finding / Background / Method / Other)**: A sentence describing background literature, methodology, experimental setup, hyperparameters, dataset specifications, untested hypotheses, study limitations, or boilerplate.

---

## Category Definitions & Decision Rules

### 1. Primary Empirical Findings (`label = 1`)
- **Definition**: Direct statements of empirical results, discoveries, or outcomes obtained by the study's own experiments, trials, or analyses.
- **Rule**: If the sentence describes what the study discovered or demonstrated through its experiments, label `1`.
- **Examples**:
  - *"Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train."* (`1`)
  - *"Vitamin D supplementation was safe and it protected against acute respiratory tract infection overall."* (`1`)
  - *"We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data."* (`1`)

### 2. Negative and Null Findings (`label = 1`)
- **Definition**: Direct statements establishing the absence of a statistically significant effect, a null outcome, or a negative finding established by the study.
- **Rule**: Negative findings are genuine scientific claims/results. In evidence synthesis, knowing that an intervention produced *no* significant effect is as vital as a positive effect. Label `1`.
- **Examples**:
  - *"For the primary comparison of any vitamin D vs. placebo, the intervention did not significantly affect overall ARI risk."* (`1`)
  - *"Vitamin D supplementation did not influence the proportion of participants experiencing at least one serious adverse event."* (`1`)
  - *"In each analysis, there is no statistically significant effect of vitamin D supplementation in any treatment group, or when the vitamin D groups are pooled and analysed together."* (`1`)

### 3. Quantitative Statistical Results (`label = 1`)
- **Definition**: Empirical results presenting specific effect sizes, odds ratios, relative risks, confidence intervals, p-values, or benchmark metric scores (BLEU, F1, accuracy) resulting from the study.
- **Rule**: If the sentence reports empirical statistical metrics establishing the paper's outcomes, label `1`.
- **Examples**:
  - *"Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU."* (`1`)
  - *"Vitamin D supplementation reduced the risk of experiencing at least one acute respiratory tract infection (adjusted odds ratio 0.88, 95% confidence interval 0.81 to 0.96; P for heterogeneity <0.001)."* (`1`)
  - *"BERT Base and BERT Large outperform all systems on all tasks by a substantial margin, obtaining 4.4% and 6.7% higher average accuracy than the previous state of the art."* (`1`)

### 4. Core Proposed Contributions (`label = 1`)
- **Definition**: Explicit primary contribution statements where the authors introduce or propose the central artifact/model architecture evaluated in the paper.
- **Rule**: Accepted under the AERF engine's contribution cue policy as a primary claim when it asserts the novel proposed mechanism. Label `1`.
- **Example**:
  - *"We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely."* (`1`)

---

### 5. Background and Prior Work (`label = 0`)
- **Definition**: Statements reviewing existing literature, citing prior work, describing accepted domain facts, or summarizing what was known before the current study.
- **Rule**: If the sentence summarizes external work or historical context rather than the current study's own findings, label `0`.
- **Examples**:
  - *"The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder."* (`0`)
  - *"The feature-based approach, such as ELMo, uses task-specific architectures that include the pre-trained representations as additional features."* (`0`)
  - *"Previous studies have reported contradictory findings regarding the efficacy of vitamin D in preventing respiratory tract infections."* (`0`)

### 6. Methodology and Experimental Setup (`label = 0`)
- **Definition**: Descriptions of algorithms, hyperparameters, training procedures, optimization settings, search criteria, or experimental procedures.
- **Rule**: Methodological actions (*"We used..."*, *"We trained..."*, *"We searched..."*) without reporting the empirical result of the experiment are non-claims. Label `0`.
- **Examples**:
  - *"We used the Adam optimizer with beta 1 = 0.9, beta 2 = 0.98 and epsilon = 10−9."* (`0`)
  - *"We searched Medline, Embase, and the Cochrane Central Register of Controlled Trials from inception to December 2015."* (`0`)
  - *"We use a batch size of 32 and fine-tune for 3 epochs over the data for all GLUE tasks."* (`0`)

### 7. Dataset and Corpus Descriptions (`label = 0`)
- **Definition**: Specifications of corpus sizes, token vocabularies, cohort participant counts, or demographic eligibility criteria.
- **Rule**: Descriptive metadata about inputs or patient cohorts are not findings. Label `0`.
- **Examples**:
  - *"We trained on the standard WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs."* (`0`)
  - *"Data were obtained from 10,933 participants in 25 randomised controlled trials aged 0 to 95 years."* (`0`)
  - *"The Stanford Question Answering Dataset is a collection of 100k crowd-sourced question/answer pairs."* (`0`)

### 8. Hypotheses and Research Questions (`label = 0`)
- **Definition**: Stated hypotheses, research questions, or objectives formulated prior to experimental testing.
- **Rule**: A hypothesis is an unverified proposition, not an empirical finding. Label `0`.
- **Examples**:
  - *"We hypothesized that individual participant data meta-analysis would resolve discrepancies between previous aggregate data reviews."* (`0`)
  - *"We hypothesized that pre-training a deep bidirectional representation would alleviate the unidirectionality constraints of prior models."* (`0`)
  - *"We aimed to determine whether population-level implementation of a test-and-treat approach to correction of sub-optimal vitamin D status influences risk of all-cause acute respiratory infection."* (`0`)

### 9. Limitations and Future Work (`label = 0`)
- **Definition**: Statements of caveats, confounding factors, missing data, study scope boundaries, or proposed future work.
- **Rule**: Caveats and future research plans are not empirical findings. Label `0`.
- **Examples**:
  - *"Our study has several limitations, including the observational nature of secondary outcome measures and heterogeneity in dosing regimens."* (`0`)
  - *"A limitation of our secondary analysis is that individual patient data from all trial arms were not publicly available."* (`0`)
  - *"We plan to investigate this approach further in future work."* (`0`)

### 10. Ambiguous / Edge Cases (`label = 0` or `1` by strict criterion)
- **Citing findings of others**: If a sentence mentions an empirical result or improvement, but explicitly attributes it to prior authors or external studies (*"Prior work by Howard and Ruder showed that..."*, *"This contrasts with findings from a Phase 2 RCT in Mexican health workers which reported..."*), it is literature context, NOT the current study's claim -> **Label `0`**.
- **Method with intended benefit**: If a sentence describes a method and states its intended purpose (*"We apply dropout to the output of each sub-layer to prevent overfitting..."*), it remains a methodological description -> **Label `0`**.
- **Direct clinical takeaway / conclusion**: If the authors state an empirical takeaway or clinical conclusion based on their findings (*"Our results support the administration of daily or weekly vitamin D to individuals with low baseline vitamin D status."*), it represents the core takeaway claim of the paper -> **Label `1`**.

---

## Provenance of Evaluation Corpora

The evaluation dataset contains 80 genuine sentences sampled verbatim from 7 published, peer-reviewed or preprint research papers spanning Computer Science (NLP / Deep Learning) and Medicine (Epidemiology / Clinical Trials):

1. **Vaswani et al. (2017)** — *Attention Is All You Need* (arXiv:1706.03762v7)
2. **Devlin et al. (2018)** — *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* (arXiv:1810.04805v2)
3. **Martineau et al. (2017)** — *Vitamin D supplementation to prevent acute respiratory tract infections: systematic review and meta-analysis of individual participant data* (BMJ 2017;356:i6583)
4. **Bolland et al. (2022)** — *Vitamin D and acute respiratory infection: secondary analysis of a previous randomised controlled trial and updated meta-analyses* (medRxiv 10.1101/2022.02.03.22270409)
5. **Jolliffe et al. (2024)** — *Vitamin D supplementation to prevent acute respiratory infections: systematic review and meta-analysis of stratified aggregate data* (medRxiv 10.1101/2024.09.18.24313866)
6. **Jolliffe et al. (2022)** — *Vitamin D Supplements for Prevention of COVID-19 or other Acute Respiratory Infections: a Phase 3 Randomised Controlled Trial (CORONAVIT)* (medRxiv 10.1101/2022.03.22.22271707)
7. **Zhang et al. (2022)** — *Efficacy of Vitamin D Supplements in Treatment of Acute Respiratory Infection: A Meta-analysis for Randomized Controlled Trials* (PMCID: PMC8955452, PMID: 35334804)

No sentences were fabricated or artificially rephrased. Synthetic samples, if used for unit testing, are isolated strictly in `tests/data/claim_extractor_synthetic_eval.csv` and excluded from model performance reporting.
