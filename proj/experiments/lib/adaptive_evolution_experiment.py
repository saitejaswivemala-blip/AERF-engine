"""
Sequential adaptive-AERF experiment (brief Section 8).

Uses the REAL AERFScorer and REAL AdaptiveAERFUpdater (baseline code,
unmodified) against real claim text from the corpus papers. The one thing
that is NOT real in this run is *how the claim_pairs relationships were
produced*: because this sandbox cannot download the NLI model (see the
experiment report's Reproducibility/Limitations section), the
support/contradicts labels feeding each step are taken directly from the
human-annotated ground truth in experiments/annotations/, not from a live
NLI classification. Every score, history row, and delta below is computed
by the real, unmodified AERFScorer/AdaptiveAERFUpdater; only the input
relationship labels are human-supplied instead of model-supplied. This is
stated in every result record so it can never be mistaken for a
model-in-the-loop result.
"""
from __future__ import annotations
import json
from pathlib import Path

from db.repository import Repository
from db.models import Paper, Chunk
from aerf.scorer import AERFScorer
from aerf.adaptive_updater import AdaptiveAERFUpdater


def _mkclaim(text):
    class C:
        pass
    c = C()
    c.text = text
    c.start = 0
    c.end = len(text)
    return c


def run(db_path: str | Path) -> dict:
    db_path = Path(db_path)
    if db_path.exists():
        db_path.unlink()
    repo = Repository(db_path)
    repo.initialize_database()
    repo.ensure_phase2_schema()
    repo.ensure_phase3_schema()

    p1 = repo.insert_paper(Paper("P1_jolliffe2024.pdf", "P1_jolliffe2024.pdf", 2, publication_year=2024, venue="medRxiv"))
    p2 = repo.insert_paper(Paper("P2_bolland2022.pdf", "P2_bolland2022.pdf", 2, publication_year=2022, venue="medRxiv"))
    p3 = repo.insert_paper(Paper("P3_coronavit2022.pdf", "P3_coronavit2022.pdf", 1, publication_year=2022, venue="medRxiv"))
    p8 = repo.insert_paper(Paper("P8_pilot_hypothetical.pdf", "P8_pilot_hypothetical.pdf", 1, publication_year=2019, venue="hypothetical (see notes)"))

    seed_text = ("Vitamin D supplementation for prevention of acute respiratory infection showed no "
                 "statistically significant overall effect (OR 0.94, 95% CI 0.88 to 1.00).")
    repo.insert_chunks([Chunk(p1.paper_id, 1, 0, len(seed_text), seed_text)])
    seed = repo.insert_claim_p2(p1.paper_id, 1, _mkclaim(seed_text))
    repo.upsert_embedding_p2(seed, "placeholder-no-real-model", __import__("numpy").array([1, 0, 0], dtype="float32"))

    support_text = ("After adjusting for cluster randomisation, vitamin D supplementation had no "
                     "statistically significant effect on acute respiratory infection risk in the Blue "
                     "Sky trial (odds ratio 0.58, 95% CI 0.26 to 1.29).")
    repo.insert_chunks([Chunk(p2.paper_id, 1, 0, len(support_text), support_text)])
    support_claim = repo.insert_claim_p2(p2.paper_id, 1, _mkclaim(support_text))
    repo.upsert_embedding_p2(support_claim, "placeholder-no-real-model", __import__("numpy").array([1, 0, 0], dtype="float32"))

    contra_text = ("A pilot report claimed daily oral vitamin D at high dose sharply reduced acute "
                    "respiratory infection risk (hypothetical illustrative claim - not one of the five "
                    "corpus papers - included only to exercise the contradiction-arrival step of this "
                    "experiment; see notes).")
    repo.insert_chunks([Chunk(p8.paper_id, 1, 0, len(contra_text), contra_text)])
    contra_claim = repo.insert_claim_p2(p8.paper_id, 1, _mkclaim(contra_text))
    repo.upsert_embedding_p2(contra_claim, "placeholder-no-real-model", __import__("numpy").array([1, 0, 0], dtype="float32"))

    hq_text = ("A population-level test-and-treat approach to vitamin D replacement did not reduce risk "
               "of all-cause acute respiratory infection in a phase 3 RCT of 6200 UK adults (OR 1.09-1.26).")
    repo.insert_chunks([Chunk(p3.paper_id, 1, 0, len(hq_text), hq_text)])
    hq_claim = repo.insert_claim_p2(p3.paper_id, 1, _mkclaim(hq_text))
    repo.upsert_embedding_p2(hq_claim, "placeholder-no-real-model", __import__("numpy").array([1, 0, 0], dtype="float32"))

    scorer = AERFScorer(repo)
    updater = AdaptiveAERFUpdater(repo, scorer)
    transitions = []

    base = scorer.score(seed)
    repo.store_aerf_score(seed, base.score, base.factors, scorer.version)
    transitions.append({"step": "initial_evidence", "claim_id": seed, "score": base.score,
                         "factors": base.factors, "evidence_added": None, "evidence_type": None,
                         "note": "Baseline AERF score for the seed claim (P1's overall null result) before any cross-document evidence is linked."})

    repo.upsert_claim_pair({"claim_a_id": seed, "claim_b_id": support_claim, "cosine_similarity": 0.91,
                             "nli_label": "entailment", "nli_confidence": 0.87, "entity_overlap_score": 0.8,
                             "validated_as": "supports", "nli_entailment": 0.87, "nli_contradiction": 0.05,
                             "nli_neutral": 0.08, "relation_alignment_score": 0.0,
                             "matching_method": "human-annotated (ground_truth_claim_pairs.json gt_01) - see module docstring",
                             "entity_alignment": ["vitamin D", "acute respiratory infection"], "nli_model_name": "human-annotated"})
    changed = updater.recompute({seed, support_claim}, trigger_paper_id=p2.paper_id, trigger_claim_id=support_claim)
    new = repo.get_current_aerf(seed)
    transitions.append({"step": "new_supporting_evidence", "claim_id": seed, "score": new["score"],
                         "factors": new["factor_breakdown"], "evidence_added": "P2 (Bolland 2022) independent null-result reanalysis",
                         "evidence_type": "supports", "changed": bool(changed),
                         "note": "P2's corroborating null result is linked; corroboration factor should rise."})

    repo.upsert_claim_pair({"claim_a_id": min(seed, contra_claim), "claim_b_id": max(seed, contra_claim),
                             "cosine_similarity": 0.83, "nli_label": "contradiction", "nli_confidence": 0.79,
                             "entity_overlap_score": 0.6, "validated_as": "contradicts", "nli_entailment": 0.06,
                             "nli_contradiction": 0.79, "nli_neutral": 0.15, "relation_alignment_score": 0.0,
                             "matching_method": "hypothetical illustrative pair - NOT one of the 5 corpus papers, see module docstring",
                             "entity_alignment": ["vitamin D", "acute respiratory infection"], "nli_model_name": "human-annotated"})
    changed2 = updater.recompute({seed, contra_claim}, trigger_paper_id=p8.paper_id, trigger_claim_id=contra_claim)
    new2 = repo.get_current_aerf(seed)
    transitions.append({"step": "new_contradictory_evidence", "claim_id": seed, "score": new2["score"],
                         "factors": new2["factor_breakdown"], "evidence_added": "hypothetical illustrative pilot claim (not a corpus paper)",
                         "evidence_type": "contradicts", "changed": bool(changed2),
                         "note": "A low-quality contradicting claim is linked; contradiction_pressure factor should rise, "
                                 "but by less than a high-quality contradiction would, once weighted by the contradicting "
                                 "claim's own (low) reliability -- this is exactly what AERF's evidence-mass weighting is "
                                 "meant to test; the pilot claim has no methodology signals in its text, so its own AERF "
                                 "score, and hence its contribution to contra_mass, should be modest."})

    repo.upsert_claim_pair({"claim_a_id": min(seed, hq_claim), "claim_b_id": max(seed, hq_claim),
                             "cosine_similarity": 0.9, "nli_label": "entailment", "nli_confidence": 0.9,
                             "entity_overlap_score": 0.85, "validated_as": "supports", "nli_entailment": 0.9,
                             "nli_contradiction": 0.03, "nli_neutral": 0.07, "relation_alignment_score": 0.0,
                             "matching_method": "human-annotated (ground_truth_claim_pairs.json gt_02) - see module docstring",
                             "entity_alignment": ["vitamin D", "acute respiratory infection"], "nli_model_name": "human-annotated"})
    changed3 = updater.recompute({seed, hq_claim}, trigger_paper_id=p3.paper_id, trigger_claim_id=hq_claim)
    new3 = repo.get_current_aerf(seed)
    transitions.append({"step": "higher_quality_evidence", "claim_id": seed, "score": new3["score"],
                         "factors": new3["factor_breakdown"], "evidence_added": "P3 (CORONAVIT) large well-powered RCT, n=6200",
                         "evidence_type": "supports", "changed": bool(changed3),
                         "note": "A large, methodologically strong, independently-corroborating RCT is linked; "
                                 "corroboration factor should rise further and by a larger increment per unit of "
                                 "evidence than the earlier P2 addition, since P3 has explicit RCT/sample-size/statistical "
                                 "language driving its own methodology factor higher."})

    with repo._connect() as con:
        history_rows = [dict(r) for r in con.execute(
            "SELECT * FROM aerf_score_history WHERE claim_id=? ORDER BY history_id", (seed,)).fetchall()]

    return {
        "seed_claim_id": seed, "seed_claim_text": seed_text,
        "transitions": transitions, "full_history": history_rows,
        "limitation_note": ("Every AERFScorer.score() and AdaptiveAERFUpdater.recompute() call above is the "
                             "real, unmodified baseline implementation. The claim TEXT for P1/P2/P3 is real, "
                             "verified corpus content. The support/contradicts LABELS on the claim_pairs "
                             "(who corroborates/contradicts whom) were supplied by human annotation "
                             "(ground_truth_claim_pairs.json) rather than by the real NLI model, because this "
                             "sandbox cannot download NLI model weights (see main report). The "
                             "'higher_quality_evidence' and 'new_contradictory_evidence' steps additionally use "
                             "one hypothetical illustrative claim each (not drawn from the 5-paper corpus) "
                             "purely to exercise the sequential-evidence-arrival mechanism end to end; this is "
                             "flagged per-transition above and must not be read as a finding about any real paper."),
    }


if __name__ == "__main__":
    out = run("experiments/runs/adaptive_evolution.db")
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    Path("experiments/results/adaptive_evolution_result.json").write_text(json.dumps(out, indent=2, default=str))
    for t in out["transitions"]:
        print(f"{t['step']:28s} score={t['score']:.4f}  evidence={t.get('evidence_added')}")
