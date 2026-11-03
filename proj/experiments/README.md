# AERF Engine — Multi-Paper Validation Experiment

Read `reports/experiment_report.md` first — it's the actual deliverable.
`reports/candidate_paper_selection_report.md` documents how the 5-paper
corpus was chosen.

## Layout

- `datasets/` — paper manifest, corpus-building script, generated PDFs
- `annotations/` — human-labeled ground-truth claim pairs (real, from the
  corpus) — see `annotation_store.py`'s docstring for the "system prediction
  never becomes ground truth" rule
- `lib/` — the experiment infrastructure (metrics, config, matching harness,
  adversarial cases, adaptive-evolution runner)
- `results/` — JSON output from every run; never overwritten, each stage
  writes its own file
- `runs/` — SQLite databases produced by each stage
- `reports/` — the two markdown reports

## Quickstart

```bash
cd aerf-engine
PYTHONPATH=. python -m pytest -q
PYTHONPATH=. python scripts/run_experiment.py --stage all
```

## Status (see experiment_report.md for the full picture)

Real Phase 1 ingestion: done. Real AERF adaptive-scoring trajectory: done.
Real NER/claim-extraction/NLI/embedding inference: **not possible in this
sandbox** (no route to `huggingface.co` or the SciSpaCy model host — confirmed,
not assumed). Baseline/ablation/adversarial numbers in this directory are an
infrastructure smoke test using a fake model, clearly labeled as such
everywhere they appear. Get real model access and rerun `--stage real_models`
before trusting any accuracy number here.

## Real-model handoff

For the actual Windows run with internet access, follow the repository-root
`FINAL_RUN_GUIDE.md`. In particular, do not treat `--stage infra_smoke_test`
as a scientific benchmark: it intentionally uses a fake NLI model. The real
models must load successfully before any baseline/ablation accuracy is
reported.
