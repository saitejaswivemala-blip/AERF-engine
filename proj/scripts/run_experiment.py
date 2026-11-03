#!/usr/bin/env python3
"""
Reproducible entrypoint for the multi-paper AERF validation experiment.

    python scripts/run_experiment.py --stage ingest
    python scripts/run_experiment.py --stage adaptive_evolution
    python scripts/run_experiment.py --stage infra_smoke_test
    python scripts/run_experiment.py --stage all

Stages:
  ingest              Real Phase 1 ingestion of experiments/datasets/pdfs/*
                       into experiments/runs/experiment.db. No ML models
                       required; runs identically in any environment.
  adaptive_evolution   Runs experiments/lib/adaptive_evolution_experiment.py
                       (real AERFScorer/AdaptiveAERFUpdater on real corpus
                       claim text; see that module's docstring for the one
                       documented substitution it makes).
  infra_smoke_test     Runs the baseline/ablation harness and metrics
                       module against the adversarial cases using a fake,
                       deterministic NLI stand-in (same pattern as the
                       shipped unit tests) purely to prove the
                       infrastructure computes correctly. THIS IS NOT A
                       SCIENTIFIC RESULT about AERF Engine's real-world
                       accuracy -- see the printed banner and the main
                       experiment report.
  real_models          Attempts to load the real SciSpaCy/NLI/embedding
                       models and run the full pipeline on the real
                       ground-truth claim pairs. Expected to fail in any
                       sandbox without outbound access to huggingface.co
                       and the SciSpaCy model host; prints the exact error
                       so a user with network access can diagnose from
                       there. Never falls back to a fake model silently.
  all                  Runs ingest, adaptive_evolution, infra_smoke_test,
                       then attempts real_models.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS_DIR = Path("experiments/results")


def stage_ingest():
    from db.repository import Repository
    from pipeline.orchestrator import ingest_paper
    from main import load_metadata_by_filename

    manifest_path = Path("experiments/datasets/paper_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    # Real title/authors/year/venue, keyed by PDF filename, from the same
    # manifest -- reuses main.py's own field mapping (manifest "year" ->
    # Paper/DB "publication_year") rather than a second, possibly-diverging
    # copy of it. Without this, re-running this stage (which deletes and
    # recreates experiments/runs/experiment.db below) would silently erase
    # any previously-applied metadata fix on every re-run.
    metadata_by_filename = load_metadata_by_filename(manifest_path)
    db_path = Path("experiments/runs/experiment.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    repo = Repository(db_path)
    repo.initialize_database()
    out = []
    for p in manifest["papers"]:
        pdf_path = Path(p["pdf_path"])
        paper = ingest_paper(pdf_path, repository=repo,
                             metadata=metadata_by_filename.get(pdf_path.name))
        out.append({"paper_id": p["paper_id"], "db_paper_id": paper.paper_id,
                     "source_filename": paper.source_filename})
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "ingest_result.json").write_text(json.dumps(out, indent=2))
    print(f"Ingested {len(out)} papers into {db_path}. Result: experiments/results/ingest_result.json")
    return out


def stage_adaptive_evolution():
    from experiments.lib.adaptive_evolution_experiment import run
    out = run("experiments/runs/adaptive_evolution.db")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "adaptive_evolution_result.json").write_text(json.dumps(out, indent=2, default=str))
    for t in out["transitions"]:
        print(f"{t['step']:28s} score={t['score']:.4f}  evidence={t.get('evidence_added')}")
    return out


def stage_infra_smoke_test():
    print("=" * 78)
    print("INFRASTRUCTURE SMOKE TEST -- NOT A SCIENTIFIC RESULT")
    print("Uses a deterministic fake NLI (same pattern as the shipped unit tests),")
    print("not the real distilbart-mnli model, because this environment cannot")
    print("download model weights. This proves the baseline/ablation/metrics")
    print("harness computes correctly; it says nothing about AERF Engine's")
    print("real-world accuracy. See experiments/reports/experiment_report.md.")
    print("=" * 78)
    import numpy as np
    from db.repository import Repository
    from db.models import Paper, Chunk
    from nlp.nli.nli_classifier import NLIResult
    from experiments.lib.experiment_config import ALL_BASELINES, ALL_ABLATIONS
    from experiments.lib.pipeline_runner import run_matching
    from experiments.lib.adversarial_cases import CASES
    from experiments.lib.metrics import evaluate, evaluation_to_dict

    class KeywordFakeNLI:
        """Deterministic stand-in: labels by simple negation/keyword heuristics
        over the adversarial cases' own text, NOT a trained model. Used only
        to give the harness something non-trivial to compute over without
        network access. See module docstring."""
        def predict_bidirectional(self, a, b):
            a_l, b_l = a.lower(), b.lower()
            neg_a = " not " in a_l or "did not" in a_l or "no effect" in a_l
            neg_b = " not " in b_l or "did not" in b_l or "no effect" in b_l
            if neg_a != neg_b:
                return NLIResult("contradiction", 0.1, 0.8, 0.1, 0.8, "keyword-fake")
            return NLIResult("neutral", 0.2, 0.2, 0.6, 0.6, "keyword-fake")

    db_path = Path("experiments/runs/infra_smoke.db")
    if db_path.exists():
        db_path.unlink()
    repo = Repository(db_path)
    repo.initialize_database(); repo.ensure_phase2_schema(); repo.ensure_phase3_schema()

    class C:
        pass

    records_by_config = {}
    for config in {c.name: c for c in (ALL_BASELINES + ALL_ABLATIONS)}.values():
        pairs = []
        for case in CASES:
            tag = f"{config.name}_{case.case_id}"
            pa = repo.insert_paper(Paper(f"{tag}_a.pdf", f"{tag}_a.pdf", 1))
            pb = repo.insert_paper(Paper(f"{tag}_b.pdf", f"{tag}_b.pdf", 1))
            repo.insert_chunks([Chunk(pa.paper_id, 1, 0, len(case.claim_a), case.claim_a)])
            repo.insert_chunks([Chunk(pb.paper_id, 1, 0, len(case.claim_b), case.claim_b)])
            ca_obj = C(); ca_obj.text = case.claim_a; ca_obj.start = 0; ca_obj.end = len(case.claim_a)
            cb_obj = C(); cb_obj.text = case.claim_b; cb_obj.start = 0; cb_obj.end = len(case.claim_b)
            a = repo.insert_claim_p2(pa.paper_id, 1, ca_obj)
            b = repo.insert_claim_p2(pb.paper_id, 2, cb_obj)
            vec = np.array([1, 0, 0], dtype=np.float32)  # placeholder, no real embedding model
            repo.upsert_embedding_p2(a, "placeholder", vec)
            repo.upsert_embedding_p2(b, "placeholder", vec)
            claims = {c["claim_id"]: c for c in repo.get_all_claims()}
            pairs.append((claims[a], claims[b], case.expected_label))

        recs = run_matching(repo, config, KeywordFakeNLI(), [(a, b) for a, b, _ in pairs])
        eval_records = [{"system_prediction": r.predicted_label, "human_label": exp}
                         for r, (_, _, exp) in zip(recs, pairs)]
        result = evaluate(eval_records, min_n_for_stats=10)
        records_by_config[config.name] = evaluation_to_dict(result)
        print(f"{config.name:38s} macro_f1={result.macro_f1:.3f} accuracy={result.accuracy:.3f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "infra_smoke_test_result.json").write_text(json.dumps(records_by_config, indent=2))
    return records_by_config


def stage_real_models():
    print("Attempting to load real models (SciSpaCy NER, zero-shot claim classifier, "
          "NLI, sentence-transformer embeddings)...")
    errors = {}
    try:
        import spacy
        spacy.load("en_core_sci_sm")
        print("  SciSpaCy model: OK")
    except Exception as e:
        errors["scispacy"] = f"{type(e).__name__}: {e}"
        print(f"  SciSpaCy model: FAILED -- {type(e).__name__}: {e}")
    try:
        from transformers import pipeline
        pipeline("text-classification", model="valhalla/distilbart-mnli-12-3", top_k=None)
        print("  NLI model: OK")
    except Exception as e:
        errors["nli"] = f"{type(e).__name__}: {e}"
        print(f"  NLI model: FAILED -- {type(e).__name__}: {e}")
    try:
        from sentence_transformers import SentenceTransformer
        SentenceTransformer("all-MiniLM-L6-v2")
        print("  Embedding model: OK")
    except Exception as e:
        errors["embeddings"] = f"{type(e).__name__}: {e}"
        print(f"  Embedding model: FAILED -- {type(e).__name__}: {e}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "real_models_attempt.json").write_text(json.dumps(errors, indent=2))
    if errors:
        print("\nReal-model execution could not complete in this environment. "
              "See experiments/results/real_models_attempt.json for exact errors, and "
              "experiments/reports/experiment_report.md 'Reproducibility' section for what "
              "a user with unrestricted network access needs to do differently.")
    return errors


def stage_real_pipeline():
    """
    Run the actual Phase 2 + Phase 3 pipeline on the already-ingested
    multi-paper experiment database using the real ML models.

    Database:
        experiments/runs/experiment.db

    Phase 2:
        SciSpaCy NER
        zero-shot claim extraction
        relation extraction
        sentence embeddings

    Phase 3:
        cross-document matching
        entity alignment
        bidirectional real NLI
        AERF scoring
        adaptive updates
        evidence graph
        evolution
        research gaps
        explanations

    This stage does NOT use fake models.
    """
    from db.repository import Repository
    from pipeline.phase2 import Phase2Pipeline
    from pipeline.phase3 import Phase3Pipeline

    db_path = Path("experiments/runs/experiment.db")

    if not db_path.exists():
        raise FileNotFoundError(
            f"Experiment database not found: {db_path}. "
            "Run --stage ingest first."
        )

    repo = Repository(db_path)
    repo.initialize_database()
    repo.ensure_phase2_schema()
    repo.ensure_phase3_schema()

    papers = repo.list_papers()

    print("=" * 78)
    print("REAL MULTI-PAPER PIPELINE")
    print("Phase 2 + Phase 3 using real ML models")
    print(f"Database: {db_path}")
    print(f"Papers: {len(papers)}")
    print("=" * 78)

    # -------------------------
    # REAL PHASE 2
    # -------------------------
    print("\n--- REAL PHASE 2 ---")

    phase2 = Phase2Pipeline(repo)
    phase2_results = []

    for paper in papers:
        print(f"\nProcessing paper_id={paper.paper_id}: {paper.source_filename}")
        result = phase2.run(paper.paper_id)
        phase2_results.append(result)

        print(
            f"  chunks={result['chunks']} "
            f"entities={result['entities']} "
            f"claims={result['claims']} "
            f"relations={result['relations']} "
            f"embeddings={result['embeddings']}"
        )

    # -------------------------
    # REAL PHASE 3
    # -------------------------
    print("\n--- REAL PHASE 3 ---")
    print("Running corpus-wide cross-document reasoning...")

    phase3 = Phase3Pipeline(repo)
    phase3_result = phase3.run(paper_id=None)

    print("\nPhase 3 result:")
    print(json.dumps(phase3_result, indent=2, default=str))

    # -------------------------
    # SAVE RESULTS
    # -------------------------
    result = {
        "database": str(db_path),
        "papers": len(papers),
        "phase2": phase2_results,
        "phase3": phase3_result,
        "nli_source": "real",
        "scientific_result_status": (
            "REAL MODEL EXECUTION. "
            "Ground-truth benchmark metrics are reported separately."
        ),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = RESULTS_DIR / "real_pipeline_result.json"
    output_path.write_text(json.dumps(result, indent=2, default=str))

    print("\n" + "=" * 78)
    print("REAL PIPELINE COMPLETE")
    print(f"Result saved: {output_path}")
    print("=" * 78)

    return result


STAGES = {
    "ingest": stage_ingest,
    "adaptive_evolution": stage_adaptive_evolution,
    "infra_smoke_test": stage_infra_smoke_test,
    "real_models": stage_real_models,
    "real_pipeline": stage_real_pipeline,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=list(STAGES) + ["all"], default="all")
    args = parser.parse_args()
    if args.stage == "all":
        for name, fn in STAGES.items():
            print(f"\n--- stage: {name} ---")
            fn()
    else:
        STAGES[args.stage]()


if __name__ == "__main__":
    main()
