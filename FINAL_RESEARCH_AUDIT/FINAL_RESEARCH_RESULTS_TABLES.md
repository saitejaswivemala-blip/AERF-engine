# Publication-Ready Tables of Verified Research Results

**Document Purpose**: Camera-ready tables in Markdown and LaTeX format for immediate insertion into the AERF research manuscript.  
**Auditor**: Antigravity Technical & Research Methodology Audit  
**Date**: November 01, 2026  
**Artifact Directory**: `FINAL_RESEARCH_AUDIT/`  

---

## Table 1: Claim Extraction Benchmark Performance

### Markdown Format

| Evaluation Set | Label Provenance & Annotation Status | Extractor Model / Configuration | Decision Threshold ($\tau$) | Accuracy (95% CI) | Claim Precision | Claim Recall | Claim F1 | Macro F1 | Weighted F1 | Confusion Matrix |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Development** ($N=80$) | Human Expert Curated (Single Annotator) | Majority Class Baseline | N/A | 57.50% [46.6%, 67.7%] | 0.00% | 0.00% | 0.00% | 36.51% | 41.98% | $\text{TP}=0, \text{FP}=0, \text{FN}=34, \text{TN}=46$ |
| **Development** ($N=80$) | Human Expert Curated (Single Annotator) | Rule-Based Keyword Baseline | Keyword regex | 66.25% [55.4%, 75.6%] | 88.89% | 23.53% | 37.21% | 57.07% | 60.04% | $\text{TP}=8, \text{FP}=1, \text{FN}=26, \text{TN}=45$ |
| **Development** ($N=80$) | Human Expert Curated (Single Annotator) | Baseline Extractor (Pre-Improvement) | 0.55 | 83.75% [74.2%, 90.3%] | 73.33% | 97.06% | 83.54% | 83.75% | 83.78% | $\text{TP}=33, \text{FP}=12, \text{FN}=1, \text{TN}=34$ |
| **Development** ($N=80$) | Human Expert Curated (Single Annotator) | **AERF 2.0 (Active Frozen Extractor)** | **0.35** | **91.25% [83.0%, 95.7%]** | **82.93%** | **100.00%** | **90.67%** | **91.22%** | **91.30%** | $\mathbf{\text{TP}=34, \text{FP}=7, \text{FN}=0, \text{TN}=39}$ |
| **Cross-Discipline Stress Test** ($N=143$) | Provisional AI Consensus (0/143 Human Verified) | Baseline Extractor (Pre-Improvement) | 0.55 | 46.85% [38.9%, 55.0%] | 85.11% | 36.70% | 51.28% | 46.41% | 48.97% | $\text{TP}=40, \text{FP}=7, \text{FN}=69, \text{TN}=27$ |
| **Cross-Discipline Stress Test** ($N=143$) | Provisional AI Consensus (0/143 Human Verified) | **AERF 2.0 (Active Frozen Extractor)** | **0.35** | **81.82% [74.7%, 87.3%]** | **90.29%** | **85.32%** | **87.74%** | **76.30%** | **82.30%** | $\mathbf{\text{TP}=93, \text{FP}=10, \text{FN}=16, \text{TN}=24}$ |

### LaTeX Format

```latex
\begin{table*}[t]
\centering
\small
\caption{Performance comparison of scientific claim extraction configurations across the Development Benchmark ($N=80$, expert-curated ground truth) and the Cross-Discipline Stress Test ($N=143$, 9 physical and natural science disciplines against provisional consensus reference). 95\% confidence intervals computed via Wilson score method.}
\label{tab:claim_extraction_results}
\begin{tabular}{llcccccccc}
\toprule
\textbf{Benchmark} & \textbf{Model Configuration} & \textbf{Threshold} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} & \textbf{Claim F1} & \textbf{Macro F1} & \textbf{TP/FP/FN/TN} \\
\midrule
\multirow{4}{*}{\shortstack[l]{Development\\($N=80$)}}
& Majority Class Baseline & -- & 57.50\% & 0.00\% & 0.00\% & 0.00\% & 36.51\% & 0 / 0 / 34 / 46 \\
& Rule Keyword Baseline & -- & 66.25\% & 88.89\% & 23.53\% & 37.21\% & 57.07\% & 8 / 1 / 26 / 45 \\
& Baseline Extractor & 0.55 & 83.75\% & 73.33\% & 97.06\% & 83.54\% & 83.75\% & 33 / 12 / 1 / 34 \\
& \textbf{AERF 2.0 (Ours)} & \textbf{0.35} & \textbf{91.25\%} & \textbf{82.93\%} & \textbf{100.00\%} & \textbf{90.67\%} & \textbf{91.22\%} & \textbf{34 / 7 / 0 / 39} \\
\midrule
\multirow{2}{*}{\shortstack[l]{Cross-Discipline\\Stress Test ($N=143$)$^\dagger$}}
& Baseline Extractor & 0.55 & 46.85\% & 85.11\% & 36.70\% & 51.28\% & 46.41\% & 40 / 7 / 69 / 27 \\
& \textbf{AERF 2.0 (Ours)} & \textbf{0.35} & \textbf{81.82\%} & \textbf{90.29\%} & \textbf{85.32\%} & \textbf{87.74\%} & \textbf{76.30\%} & \textbf{93 / 10 / 16 / 24} \\
\bottomrule
\multicolumn{9}{l}{\footnotesize $^\dagger$Reference labels for the $N=143$ stress test reflect dual-model AI consensus without human adjudication; metrics measure inter-model agreement.}
\end{tabular}
\end{table*}
```

---

## Table 2: Ablation Study across Gating and Scoring Components

### Markdown Format

| Configuration | Cosine Similarity Gate | Entity Overlap Gate | DeBERTa-v3 NLI Model | AERF Reliability Scoring | Adaptive Score Propagation | Provenance Factor Weight | Evaluated Pairs ($N$) | Diagnostic Accuracy | gt_01 Result (supports) | gt_02 Result (supports) | gt_03 Result (supports) | gt_05 Result (neutral) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline A** (Similarity Only) | $\ge 0.55$ | None | None | None | None | None | 4 | 25.00% | supports | neutral | neutral | supports |
| **Baseline B** (Similarity + NLI) | $\ge 0.55$ | None | Active | None | None | None | 4 | 0.00% | neutral | neutral | neutral | uncertain |
| **Baseline C** (Similarity + Entity + NLI)| $\ge 0.55$ | $\ge 0.15$ | Active | None | None | None | 4 | 0.00% | neutral | neutral | uncertain | uncertain |
| **Full AERF Pipeline** | $\ge 0.55$ | $\ge 0.15$ | Active | Active | Active | 0.10 | 4 | 0.00% | neutral | neutral | uncertain | uncertain |
| **w/o Entity Alignment** | $\ge 0.55$ | None | Active | Active | Active | 0.10 | 4 | 0.00% | neutral | neutral | neutral | uncertain |
| **w/o Adaptive Scoring** | $\ge 0.55$ | $\ge 0.15$ | Active | Active | Static | 0.10 | 4 | 0.00% | neutral | neutral | uncertain | uncertain |
| **w/o Provenance Weighting**| $\ge 0.55$ | $\ge 0.15$ | Active | Active | Active | 0.00 | 4 | 0.00% | neutral | neutral | uncertain | uncertain |

### LaTeX Format

```latex
\begin{table*}[t]
\centering
\small
\caption{Diagnostic ablation analysis across Phase 3 alignment, stance validation, and reliability scoring modules on manually curated cross-document claim pairs ($N=4$). All configurations evaluated in isolated SQLite databases with real model weights.}
\label{tab:ablation_results}
\begin{tabular}{lcccccc}
\toprule
\textbf{Configuration} & \textbf{Sim. Gate} & \textbf{Entity Gate} & \textbf{NLI Stance} & \textbf{AERF Scorer} & \textbf{Adaptive Update} & \textbf{Diagnostic Accuracy} \\
\midrule
Baseline A (Similarity Only) & $\ge 0.55$ & None & None & None & None & 25.00\% (1/4)$^*$ \\
Baseline B (Similarity + NLI) & $\ge 0.55$ & None & Active & None & None & 0.00\% (0/4) \\
Baseline C (Sim. + Entity + NLI) & $\ge 0.55$ & $\ge 0.15$ & Active & None & None & 0.00\% (0/4) \\
\textbf{Full AERF Engine} & $\ge 0.55$ & $\ge 0.15$ & Active & Active & Active & 0.00\% (0/4) \\
\quad -- w/o Entity Alignment & $\ge 0.55$ & None & Active & Active & Active & 0.00\% (0/4) \\
\quad -- w/o Adaptive Scoring & $\ge 0.55$ & $\ge 0.15$ & Active & Active & Static & 0.00\% (0/4) \\
\quad -- w/o Provenance Factor & $\ge 0.55$ & $\ge 0.15$ & Active & Active ($w_p=0$) & Active & 0.00\% (0/4) \\
\bottomrule
\multicolumn{7}{l}{\footnotesize $^*$Baseline A accuracy represents coincidental alignment; it classifies all candidate pairs as `supports` without stance inference.}
\end{tabular}
\end{table*}
```

---

## Table 3: System Pipeline Complexity and Database Inventory

### Markdown Format

| System Database Corpus | Number of Ingested Papers | Chunks Extracted | Extracted Claims | Cross-Paper Claim Pairs | Evidence Graph Edges | Research Gaps Identified | Database Size on Disk | Primary Research Function |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| `experiment_FINAL_2026-09-02.db` | 11 | 185 | 816 | 671 | 12,286 | 2,412 | 8.24 MB | Global Persistent System Benchmark Corpus |
| `session_20261028_173527_bb5398.db`| 2 | 52 | 400 | 815 | 4,542 | 333 | 3.37 MB | Multi-Paper Isolated Analysis Session |
| `eval_dbs/full_aerf.db` | 5 | 74 | 43 | 41 | 1,422 | 90 | 0.74 MB | Real-Model Ablation & Curated Evaluation Seed |
