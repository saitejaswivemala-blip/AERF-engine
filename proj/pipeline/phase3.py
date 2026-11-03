from __future__ import annotations
import os
import time
import logging
from typing import Callable
import torch

from db.repository import Repository
from nlp.nli.nli_classifier import NLIClassifier
from matching.claim_matcher import ClaimMatcher
from aerf.scorer import AERFScorer
from aerf.adaptive_updater import AdaptiveAERFUpdater
from graph.evidence_graph import sync_evidence_graph
from evolution.timeline_builder import build_evolution
from gaps.gap_detector import detect_gaps
from explainability.explanation_builder import build_claim_explanation

logger = logging.getLogger(__name__)

# Configure CPU threads to prevent thermal throttling and laptop freeze/shutdown
_MAX_TORCH_THREADS = max(1, min(4, (os.cpu_count() or 4) // 2))
if torch.get_num_threads() > _MAX_TORCH_THREADS:
    torch.set_num_threads(_MAX_TORCH_THREADS)

class Phase3Pipeline:
    def __init__(self, repository: Repository, nli_model='MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli', similarity=.55, entity=.15, nli_conf=.60, weights=None, entity_alignment_version='v2'):
        self.repo=repository
        self.nli=NLIClassifier(nli_model)
        self.matcher=ClaimMatcher(self.nli,similarity,entity,nli_conf,entity_alignment_version=entity_alignment_version)
        self.scorer=AERFScorer(repository,weights=weights or None,version='aerf-v1')
        self.updater=AdaptiveAERFUpdater(repository,self.scorer)

    def run(
        self,
        paper_id=None,
        paper_ids=None,
        progress_callback: Callable[[dict], None] | None = None,
        cancel_checker: Callable[[], bool] | None = None,
    ):
        self.repo.ensure_phase3_schema()
        if paper_ids is not None:
            scoped_paper_ids = set(paper_ids)
            all_claims = [c for c in self.repo.get_all_claims() if c['paper_id'] in scoped_paper_ids]
            claims = all_claims
            candidate_pairs = [(a,b) for i,a in enumerate(all_claims) for b in all_claims[i+1:] if a['paper_id']!=b['paper_id']]
            scope_pids = scoped_paper_ids
        elif paper_id is not None:
            claims = self.repo.get_claims_for_paper(paper_id)
            all_claims = self.repo.get_all_claims()
            target_claims = [c for c in all_claims if c['paper_id']==paper_id]
            other_claims = [c for c in all_claims if c['paper_id']!=paper_id]
            candidate_pairs = [(a,b) for a in target_claims for b in other_claims]
            scope_pids = None
        else:
            claims = self.repo.get_all_claims()
            all_claims = claims
            candidate_pairs = [(a,b) for i,a in enumerate(all_claims) for b in all_claims[i+1:] if a['paper_id']!=b['paper_id']]
            scope_pids = None

        matched=0; relations={'supports':0,'contradicts':0,'neutral':0,'uncertain':0}
        affected=set()

        import numpy as np
        emb={}; ents={}; rels={}
        for c in all_claims:
            cid=c['claim_id']; v=self.repo.get_claim_embedding(cid)
            if v is None: continue
            n=np.linalg.norm(v); emb[cid]=(v/n) if n else v
            ents[cid]=self.repo.get_claim_entities(cid); rels[cid]=self.repo.get_claim_relations(cid)

        done=set()
        if hasattr(self.repo,'get_all_claim_pair_keys'):
            done=self.repo.get_all_claim_pair_keys()

        sim_gate=float(self.matcher.similarity_threshold)

        filtered_candidates = []
        for a, b in candidate_pairs:
            ia, ib = a['claim_id'], b['claim_id']
            if ia not in emb or ib not in emb: continue
            if (min(ia, ib), max(ia, ib)) in done: continue
            if float(np.dot(emb[ia], emb[ib])) < sim_gate: continue
            filtered_candidates.append((a, b))

        total_filtered = len(filtered_candidates)
        start_time = time.perf_counter()
        evaluated_pairs = 0

        for idx, (a, b) in enumerate(filtered_candidates, start=1):
            if cancel_checker and cancel_checker():
                logger.info("Cancellation requested during Phase 3 evidence matching.")
                raise InterruptedError("Analysis cancelled by user.")

            ia, ib = a['claim_id'], b['claim_id']
            d = self.matcher.match(a, b, ents[ia], ents[ib], rels[ia], rels[ib], emb[ia], emb[ib])
            evaluated_pairs += 1

            if d is not None:
                self.repo.upsert_claim_pair({
                    'claim_a_id': d.claim_a_id,
                    'claim_b_id': d.claim_b_id,
                    'cosine_similarity': d.cosine_similarity,
                    'nli_label': d.nli_label,
                    'nli_confidence': d.nli_confidence,
                    'entity_overlap_score': d.entity_overlap_score,
                    'validated_as': d.validated_as,
                    'nli_entailment': d.nli_entailment,
                    'nli_contradiction': d.nli_contradiction,
                    'nli_neutral': d.nli_neutral,
                    'relation_alignment_score': d.relation_alignment_score,
                    'matching_method': d.method,
                    'entity_alignment': sorted(set(x['text'] for x in ents[ia]) & set(x['text'] for x in ents[ib])),
                    'nli_model_name': self.nli.model_name
                })
                matched += 1
                relations[d.validated_as] = relations.get(d.validated_as, 0) + 1
                affected.update((d.claim_a_id, d.claim_b_id))

            if progress_callback:
                elapsed = time.perf_counter() - start_time
                avg_time = elapsed / evaluated_pairs if evaluated_pairs > 0 else 0.0
                eta = avg_time * (total_filtered - evaluated_pairs)
                progress_callback({
                    "stage": "phase3",
                    "current_pair": evaluated_pairs,
                    "total_pairs": total_filtered,
                    "matched": matched,
                    "supports": relations.get("supports", 0),
                    "contradicts": relations.get("contradicts", 0),
                    "neutral": relations.get("neutral", 0),
                    "elapsed_s": elapsed,
                    "eta_s": eta,
                })

        # Establish a baseline for every claim, then adaptively update evidence-connected claims.
        for c in all_claims:
            if self.repo.get_current_aerf(c['claim_id']) is None:
                result=self.scorer.score(c['claim_id'])
                self.repo.store_aerf_score(c['claim_id'],result.score,result.factors,result.weights_version)
        changed=self.updater.recompute(affected,trigger_paper_id=paper_id,trigger_claim_id=None) if affected else []
        edges=sync_evidence_graph(self.repo, paper_ids=scope_pids)
        evolution=build_evolution(self.repo, paper_ids=scope_pids)
        gaps=detect_gaps(self.repo, paper_ids=scope_pids)

        scoped_cids = {c['claim_id'] for c in all_claims} if scope_pids else None
        for cid in (affected or {c['claim_id'] for c in claims}):
            build_claim_explanation(self.repo, cid, paper_ids=scope_pids, scoped_claim_ids=scoped_cids)

        return {
            'paper_id': paper_id,
            'paper_ids': list(scoped_paper_ids) if paper_ids is not None else None,
            'claims_considered': len(claims),
            'candidate_relationships_persisted': matched,
            'validated_relationships': relations,
            'adaptive_score_changes': len(changed),
            'graph_edges_synced': edges,
            'evolution_snapshots': len(evolution),
            'research_gaps': len(gaps),
        }
