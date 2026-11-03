# AERF Engine — Multi-Paper Validation Experiment Report

## 1. Experimental status

This report records the current verified state of the AERF Engine experiment after execution with real NLP model weights. The experiment is a **working-system validation and diagnostic evaluation**, not a statistically valid benchmark.

The Phase 1–3 pipeline executed end-to-end on a five-paper biomedical corpus. Real SciSpaCy NER, real sentence embeddings, and real transformer NLI were available during the successful run. The current run therefore establishes that the implemented pipeline can execute its intended real-model path; it does **not** establish that the model predictions are scientifically accurate.

## 2. Baseline and experiment isolation

The original Phase 3 implementation was preserved at git tag `baseline-phase3-46tests`. Experiment infrastructure and later validation fixes were kept separate from the baseline architecture wherever possible. The current package also contains regression tests for bugs discovered during validation.

## 3. Corpus

Five papers concerning vitamin D supplementation and acute respiratory infection (ARI) were used:

| Paper | Description | Phase-2 chunks |
|---|---|---:|
| P1 | Jolliffe et al. 2024 updated ARI prevention meta-analysis | 19 |
| P2 | Bolland et al. 2022 secondary analysis / updated meta-analysis | 22 |
| P3 | CORONAVIT 2022 Phase 3 RCT | 12 |
| P4 | PMC8955452 treatment meta-analysis | 11 |
| P5 | PMC12699866 pediatric systematic review/meta-analysis | 10 |

P4 and P5 in the current corpus remain **abstract/indexed-text inputs**, not independently retrieved full-paper experimental inputs. They must not be described as full-text validation until replaced with verified full text.

## 4. Real model execution

The successful real-model run reported:

- SciSpaCy scientific NER: **OK**
- Transformer NLI: **OK**
- Sentence-transformer embeddings: **OK**
- NLI source: `real`
- Direct NLI smoke inference produced a high-confidence entailment result (confidence approximately 0.9868), confirming that the configured NLI checkpoint could perform inference.

### Phase 2 results

| Paper | Chunks | Entities | Claims | Relations | Embeddings |
|---|---:|---:|---:|---:|---:|
| P1 | 19 | 156 | 9 | 0 | 9 |
| P2 | 22 | 169 | 13 | 0 | 13 |
| P3 | 12 | 114 | 8 | 0 | 8 |
| P4 | 11 | 82 | 8 | 0 | 8 |
| P5 | 10 | 80 | 5 | 0 | 5 |
| **Total** | **74** | **601** | **43** | **0** | **43** |

The zero relation count exposed a genuine implementation defect: the relation extractor compared SpaCy **lemmas** against inflected predicate strings (`reduces`, `improves`, etc.), while SpaCy returns base lemmas such as `reduce` and `improve`. This has now been corrected by matching the approved base lemmas. Regression tests were added. The real five-paper pipeline should be rerun once with the corrected extractor before treating relation counts as final.

### Phase 3 results from the successful pre-fix real run

The successful real run produced:

```text
claims_considered = 43
candidate_relationships_persisted = 37
supports = 4
contradicts = 3
neutral = 16
uncertain = 14
adaptive_score_changes = 7
graph_edges_synced = 467
evolution_snapshots = 237
research_gaps = 30
```

These numbers are retained as evidence of the real-model execution. Because the relation-extraction lemma defect was discovered afterward, a **fresh post-fix Phase 3 run is required** before using these numbers as the final relation-aware experiment result.

## 5. Database integrity checks

The successful fresh run produced 37 persisted claim-pair records. Their validation labels were:

- supports: 4
- contradicts: 3
- neutral: 16
- uncertain: 14

All persisted AERF score records in the inspected run were `evidence-derived`. Seven AERF history records were created, showing that adaptive score recomputation was actually exercised.

The claim-pair schema includes explicit NLI label/confidence fields, validation status, entity-overlap score, relation-alignment score, matching method, entity-alignment JSON, and NLI model name. The implementation was also corrected so validated NLI predictions are persisted in `validated_as`; human ground-truth labels remain separate and are never overwritten.

## 6. Curated real-model diagnostic evaluation

Four manually verified cross-document claim mappings were selected for a focused diagnostic check:

| Pair | Human label | Real-system prediction |
|---|---|---|
| GT-01 (P1 claim 4 ↔ P2 claim 13) | supports | neutral |
| GT-02 (P1 claim 4 ↔ P3 claim 26) | supports | neutral |
| GT-03 (P2 claim 13 ↔ P3 claim 26) | supports | neutral |
| GT-05 (P4 claim 38 ↔ P1 claim 4) | neutral | contradicts |

The diagnostic accuracy and macro-F1 were both 0.000. **This must not be presented as a benchmark**: the sample contains only four deliberately selected pairs and is not statistically representative.

The important finding is qualitative: real NLI inference executed, but the current configuration did not reproduce the human labels on these four pairs. In particular, the real NLI model returned neutral for the three human-labelled support pairs and contradiction for the neutral pair. This is a model/pipeline-quality limitation, not evidence that the pipeline failed to execute.

The entity-overlap values for the curated pairs were below the configured 0.2 threshold, which explains why the full entity-alignment configuration can produce `uncertain` even when real NLI inference is available. The threshold has **not** been lowered merely to improve the diagnostic score.

## 7. Adaptive AERF mechanism

A separate real execution exercised the AERF scorer and adaptive updater using a sequence of evidence arrival:

| Evidence state | AERF score |
|---|---:|
| Initial P1 overall-null claim | 0.3700 |
| + supporting P2 evidence | 0.4612 |
| + hypothetical contradictory evidence | 0.4246 |
| + higher-quality P3 supporting evidence | 0.4681 |

The scoring mechanism is deterministic, factorized, bounded, and history-producing. The trajectory demonstrates the intended adaptive direction: support increases the score, contradictory evidence decreases it, and stronger corroborating evidence increases it again. The contradictory step is explicitly hypothetical and must not be interpreted as a finding from the five-paper corpus.

## 8. AERF formulation

The implementation uses the project's seven-factor formulation:

```text
AERF(c) = wS*S + wM*M + wCr*Cr + wT*T + wSem*Sem + wPr*Pr - wCp*Cp
```

The score is clipped to `[0,1]`, with non-negative weights summing to one. Factors represent source characteristics, methodology, corroboration, contradiction pressure, temporal relevance, semantic confidence, and provenance completeness. Weights are externalized and versioned rather than embedded in application code.

The central architectural rule remains: **cosine similarity is a candidate-generation gate only; it must never by itself create a support or contradiction decision.** The intended decision path is similarity → entity alignment → NLI → validated relationship → AERF/adaptive propagation → explanation.

## 9. Automated tests

After the relation-extractor fix and its regression tests, the complete test suite passes:

```text
113 passed in 1.21s
```

This is a software-correctness result, not an NLP accuracy benchmark. The tests include unit/integration/experiment infrastructure checks and specifically cover the corrected relation predicate handling.

## 10. Important implementation fixes made during validation

1. **Paper metadata backfill** — manifest metadata is now passed into ingestion and missing database metadata can be backfilled without overwriting existing values.
2. **NLI validation persistence** — real NLI predictions are persisted as `validated_as`; human labels remain independent ground truth.
3. **Phase 3 directional coverage** — corpus-wide matching no longer skips claims merely because their IDs were inserted earlier/later.
4. **Claim-pair orientation deduplication** — both `(A,B)` and `(B,A)` orientations are removed before inserting the logical pair.
5. **Relation predicate lemma bug** — approved predicates now match SpaCy base lemmas (`treat`, `cause`, `prevent`, `improve`, `increase`, `reduce`). Regression tests cover dependency extraction and the entity-anchored fallback.

## 11. Limitations

- The curated ground truth is too small for meaningful accuracy/F1 claims.
- P4/P5 are still abstract/indexed-text inputs and should be replaced with verified full text.
- The current real NLI model demonstrably disagrees with the four manually verified mappings; therefore no claim of high scientific classification accuracy is justified.
- The pre-fix real Phase 3 relation count was zero because of the discovered lemma bug; a post-fix rerun is required for final relation-aware results.
- The current relation extractor is deliberately conservative. Its fallback requires an entity on both sides of an approved predicate and should be evaluated on real scientific sentences after rerun.
- The evidence graph is currently a graph-backed evidence store, not a graph-algorithm reasoning engine.
- No dashboard/UI result is claimed in this experiment report.

## 12. Patent/evidence interpretation

The strongest currently demonstrated technical mechanism is the **gate → adjudicate → score → adapt → explain** pipeline, especially the rule preventing similarity alone from creating contradiction/support labels and the adaptive AERF history mechanism. These are implementation observations, not a legal patentability opinion.

The current experiments do **not** establish a measured accuracy advantage over simpler baselines. A patent filing should therefore avoid presenting the four-pair diagnostic as proof of superiority. The defensible evidence is that the architecture is implemented, executable with real models, traceable through SQLite provenance/evidence records, and capable of adaptive score recomputation with history.

## 13. Reproducibility

Primary run command:

```powershell
PYTHONPATH=. python scripts/run_experiment.py --stage all
```

For the final post-fix run, rerun the same command from a clean copy/database after confirming the configured SciSpaCy, embedding, and NLI models are available. Preserve:

- `experiments/runs/experiment.db`
- `experiments/results/real_pipeline_result.json`
- `experiments/results/curated_evaluation_gt01_02_03_05.json`
- test output
- the exact source archive used for the run

Do not overwrite the backed-up successful real run until the post-fix rerun has completed and been inspected.

## 14. Immediate next step

**One step only:** rerun the full experiment once with the corrected relation extractor. Do not tune thresholds, change the NLI model, or build the dashboard yet. First obtain a clean post-fix run and inspect whether scientifically meaningful relations are now persisted. After that, the next decision can be made from evidence rather than assumptions.
