# AERF Engine — Final Real-Model Run Guide

## Purpose

The experiment infrastructure is complete, but the published baseline/ablation metrics must **not** be treated as scientific results until the real NLP models have been executed on the five-paper corpus.

Current verified state:

- Phase 1–3 baseline preserved at git tag `baseline-phase3-46tests`.
- Experiment infrastructure added in a separate commit/tag.
- Full test suite in this package: **81 passed** in the verification environment.
- Five-paper corpus is present under `experiments/datasets/pdfs/`.
- Real Phase 1 ingestion and real AERF adaptive-score execution have already been performed.
- Real SciSpaCy/NLI/embedding inference was **not** completed in the previous sandbox because model-host access was blocked.

## 1. Recommended Windows environment

Use Python 3.12.x, matching the environment in which the project was developed and tested. Do not create a fresh environment with unconstrained `torch`, `numpy`, `scipy`, and `scikit-learn` versions; the project intentionally pins compatible versions below.

From PowerShell in the project root:

```powershell
python --version
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks `Activate.ps1`, you do **not** need activation. Use the venv executable explicitly:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 2. Install the SciSpaCy model

The application expects:

```text
en_core_sci_sm
```

Install the model package compatible with the installed SciSpaCy version. For the current Phase 2/3 environment, the previously verified package is `en_core_sci_sm==0.5.4`.

```powershell
.\venv\Scripts\python.exe -m pip install "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz" --timeout 300
```

Then verify:

```powershell
.\venv\Scripts\python.exe -c "import spacy; nlp=spacy.load('en_core_sci_sm'); print('SCISPACY OK')"
```

A `FutureWarning` from spaCy during model loading is acceptable if the final line says `SCISPACY OK`.

## 3. Verify the core environment before downloading large models

```powershell
.\venv\Scripts\python.exe -c "import numpy, scipy, sklearn, torch, spacy, scispacy, transformers, sentence_transformers, pytest; print('ENVIRONMENT OK')"
```

Also verify the PyTorch version:

```powershell
.\venv\Scripts\python.exe -c "import torch; print(torch.__version__); print('TORCH OK')"
```

The intended CPU installation is PyTorch 2.6.x.

## 4. Verify model downloads

The project uses these three model assets:

### Scientific NER

```text
en_core_sci_sm
```

### Claim classification / NLI

```text
valhalla/distilbart-mnli-12-3
```

### Claim embeddings

```text
sentence-transformers/all-MiniLM-L6-v2
```

Run the project's model-access check first:

```powershell
.\venv\Scripts\python.exe scripts\run_experiment.py --stage real_models
```

Expected successful output is approximately:

```text
SciSpaCy model: OK
NLI model: OK
Embedding model: OK
```

If any model fails, **stop there**. Do not run or report scientific benchmark results until the missing model is available.

## 5. Run the full existing test suite

```powershell
.\venv\Scripts\python.exe -m pytest -q
```

Expected baseline for this package:

```text
81 passed
```

Do not proceed if the original Phase 1–3 functionality has been broken.

## 6. Run real corpus ingestion

The five-paper PDFs are already included in:

```text
experiments/datasets/pdfs/
```

Run:

```powershell
.\venv\Scripts\python.exe scripts\run_experiment.py --stage ingest
```

This creates:

```text
experiments/runs/experiment.db
```

and:

```text
experiments/results/ingest_result.json
```

## 7. Run the real Phase 2 pipeline on every paper

The experiment database is separate from the normal application database. After ingestion, inspect `experiments/results/ingest_result.json` for the database paper IDs.

For each paper ID, run Phase 2 against the experiment database. The current CLI defaults to `data/aerf.db`, so **do not assume that `main.py --phase2-paper-id` operates on `experiments/runs/experiment.db`**.

Instead, first verify whether the repository/CLI supports selecting a database path. If it does not, use the project's experiment runner or make the smallest explicit database-path change necessary. Do not copy the experiment database over the baseline database.

The required real Phase 2 operations are:

```text
PDF chunks
→ SciSpaCy NER
→ semantic claim extraction
→ relation extraction
→ Sentence-Transformer embeddings
→ SQLite persistence
```

For every paper, record:

- chunks
- entities
- claims
- relations
- embeddings
- execution errors

Do not manually enter claim counts.

## 8. Run real Phase 3 cross-document reasoning

After **all five papers** have Phase 2 claims and embeddings in the same experiment database, run Phase 3 against the corpus.

The important requirement is that the database contains claims from multiple papers before Phase 3 runs.

Phase 3 should then perform:

```text
cross-paper candidate generation
→ cosine similarity gate
→ entity alignment
→ bidirectional NLI
→ validated relationship
→ AERF scoring
→ adaptive update
→ evidence graph
→ evolution
→ research-gap detection
→ explanations
```

A single-paper Phase 3 run is not sufficient to validate the cross-document mechanism.

## 9. Ground-truth evaluation

The existing file:

```text
experiments/annotations/ground_truth_claim_pairs.json
```

contains human-annotated relationships.

Important: the annotations are ground truth only because they were explicitly human-labelled. The system must populate `system_prediction`; it must never overwrite `human_label`.

Before calculating metrics, verify that each evaluated pair has:

```text
human_label != null
system_prediction != null
```

Then run the experiment evaluation harness against the real model outputs.

**Do not use `infra_smoke_test` numbers as scientific results.** That stage intentionally uses a fake NLI model.

## 10. Baselines and ablations

The final real experiment must compare:

1. Similarity only
2. Similarity + NLI
3. Similarity + entity alignment + NLI
4. Full AERF
5. Without entity alignment
6. Without adaptive scoring
7. Without provenance weighting
8. Similarity + NLI only, where the configuration is not otherwise duplicated

For every configuration report:

- number of evaluated pairs
- supports
- contradictions
- neutral
- uncertain
- precision
- recall
- F1
- confusion matrix
- false-support rate
- false-contradiction rate

Do not interpret a metric if the evaluated sample is too small; expand the annotation set first.

## 11. Adaptive AERF experiment

Run:

```powershell
.\venv\Scripts\python.exe scripts\run_experiment.py --stage adaptive_evolution
```

This is already the verified real-AERF mechanism test. It is useful as an implementation check, but its documented human-label/hypothetical-evidence substitutions mean it is **not** evidence that the full end-to-end ML system is accurate.

The real end-to-end experiment should additionally demonstrate adaptive updates using relationships produced by the real NLI pipeline.

## 12. Adversarial evaluation

Run the adversarial set only with the real NLI/embedding/entity pipeline for scientific conclusions.

The existing infrastructure contains ten cases covering:

- high similarity + contradiction
- high similarity + neutral
- negation
- population mismatch
- condition mismatch
- temporal mismatch
- conflicting numerical findings
- evidence-strength differences
- duplicates
- incomplete claims

The fake-NLI smoke-test output is only a software test.

## 13. P4/P5 corpus warning

The current experiment report correctly records that the previous environment could not independently retrieve the full text of P4/P5 from PMC because of an automated access challenge.

Before treating P4/P5 as full-paper experimental inputs, verify that the PDFs/text in `experiments/datasets/pdfs/` contain the complete paper rather than only an abstract-derived reconstruction.

If either file is abstract-only, replace it with a legitimately downloadable full-text paper or obtain the complete open-access PDF through a normal browser and update the manifest.

Do not claim full-paper processing for an abstract-only file.

## 14. What counts as a successful final run

A successful final run requires ALL of the following:

- all three real models load locally;
- all five corpus papers are fully ingested;
- Phase 2 runs with real NER/claim/embedding models;
- multiple papers coexist in one experiment database;
- Phase 3 produces cross-document candidate relationships;
- real NLI validates relationship types;
- human ground truth is matched to system predictions;
- baseline and AERF configurations are evaluated on the same cases;
- ablations are evaluated on the same cases;
- adaptive AERF updates are recorded historically;
- adversarial cases are evaluated with real models;
- no fake-model output is included as scientific evidence;
- results are saved as reproducible JSON/report artifacts;
- all tests pass.

## 15. What must NOT be claimed yet

Until the real-model run is complete, do **not** claim:

- AERF has higher accuracy than similarity;
- AERF beats NLI;
- AERF reduces false contradictions;
- AERF improves research-gap detection;
- the system is patentable;
- the mechanism is novel over all prior art.

The correct current statement is that the implementation and evaluation infrastructure exist and are ready for real-model validation.

## 16. Final command sequence

After the environment and models are verified, the intended sequence is:

```powershell
cd C:\path\to\aerf-engine

.\venv\Scripts\python.exe -m pytest -q

.\venv\Scripts\python.exe scripts\run_experiment.py --stage real_models

.\venv\Scripts\python.exe scripts\run_experiment.py --stage ingest

.\venv\Scripts\python.exe scripts\run_experiment.py --stage adaptive_evolution
```

Then run the real Phase 2 and Phase 3 multi-paper workflow using the experiment database, followed by the real baseline/ablation evaluation.

## 17. Evidence to preserve for the patent workflow

Keep these artifacts together and never overwrite them:

```text
experiments/datasets/paper_manifest.json
experiments/annotations/ground_truth_claim_pairs.json
experiments/results/*.json
experiments/reports/*.md
FINAL_RUN_GUIDE.md
git history / tags
```

For each final experiment, preserve the exact code commit, model names/versions, configuration, paper manifest, annotations, execution command, and raw results.
