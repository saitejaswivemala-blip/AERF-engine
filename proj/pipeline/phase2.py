from __future__ import annotations
import os
import time
import logging
from typing import Callable
import torch

from db.repository import Repository
from nlp.ner import ScientificNER
from nlp.claim_extractor import ScientificClaimExtractor
from nlp.relation_extractor import ScientificRelationExtractor
from nlp.embeddings import ClaimEmbedder

logger = logging.getLogger(__name__)

# Configure CPU threads to prevent thermal throttling and laptop freeze/shutdown
_MAX_TORCH_THREADS = max(1, min(4, (os.cpu_count() or 4) // 2))
if torch.get_num_threads() > _MAX_TORCH_THREADS:
    torch.set_num_threads(_MAX_TORCH_THREADS)


class Phase2Pipeline:
    """Run NER -> claims -> relations -> claim embeddings on Phase-1 chunks."""
    def __init__(self, repository: Repository, ner_model="en_core_sci_sm",
                 claim_model="valhalla/distilbart-mnli-12-3", embedding_model="all-MiniLM-L6-v2",
                 claim_threshold=0.55):
        self.repo=repository
        self.ner=ScientificNER(ner_model)
        nlp=self.ner.nlp
        self.claims=ScientificClaimExtractor(claim_model, claim_threshold, nlp=nlp)
        self.relations=ScientificRelationExtractor(nlp=nlp)
        self.embedder=ClaimEmbedder(embedding_model)

    def run(
        self,
        paper_id: int,
        force: bool = False,
        progress_callback: Callable[[dict], None] | None = None,
        cancel_checker: Callable[[], bool] | None = None,
    ):
        if self.repo.get_paper_by_id(paper_id) is None:
            raise ValueError(f"Unknown paper_id: {paper_id}")
        self.repo.ensure_phase2_schema()
        chunks = self.repo.get_chunks_by_paper(paper_id)
        total_chunks = len(chunks)

        done_chunks = self.repo.get_processed_chunk_ids(paper_id) if not force else set()
        existing_claims = self.repo.get_claims_count_for_paper(paper_id)

        if total_chunks > 0 and len(done_chunks) >= total_chunks and not force:
            logger.info("Paper %d already fully processed (%d chunks, %d claims). Skipping.", paper_id, total_chunks, existing_claims)
            return {
                "paper_id": paper_id,
                "chunks": total_chunks,
                "processed_chunks": total_chunks,
                "entities": 0,
                "claims": existing_claims,
                "relations": 0,
                "embeddings": 0,
                "skipped": True,
            }

        counts = {
            "paper_id": paper_id,
            "chunks": total_chunks,
            "processed_chunks": len(done_chunks),
            "entities": 0,
            "claims": existing_claims,
            "relations": 0,
            "embeddings": 0,
            "timing_ner_s": 0.0,
            "timing_claims_s": 0.0,
            "timing_relations_s": 0.0,
            "timing_embeddings_s": 0.0,
        }

        start_time = time.perf_counter()
        processed_this_run = 0

        for ch in chunks:
            if cancel_checker and cancel_checker():
                logger.info("Cancellation requested during Phase 2 for paper %d.", paper_id)
                raise InterruptedError("Analysis cancelled by user.")

            if ch.chunk_id in done_chunks and not force:
                continue

            # Stage 1: NER
            t0 = time.perf_counter()
            entities = self.ner.extract(ch.text)
            counts["timing_ner_s"] += time.perf_counter() - t0
            entity_rows = [(e, self.repo.insert_entity_p2(paper_id, ch.chunk_id, e)) for e in entities]
            counts["entities"] += len(entity_rows)

            # Stage 2: Claims
            t0 = time.perf_counter()
            claims = self.claims.extract(ch.text)
            counts["timing_claims_s"] += time.perf_counter() - t0

            # Stage 3: Embeddings
            t0 = time.perf_counter()
            vecs = self.embedder.encode([c.text for c in claims]) if claims else []
            counts["timing_embeddings_s"] += time.perf_counter() - t0

            # Stage 4: Relations & Linking
            t0 = time.perf_counter()
            for claim, vec in zip(claims, vecs):
                cid = self.repo.insert_claim_p2(paper_id, ch.chunk_id, claim)
                counts["claims"] += 1
                local_entities = []
                for e, eid in entity_rows:
                    if claim.start <= e.start and e.end <= claim.end:
                        local_entities.append((e, eid))
                        self.repo.link_claim_entity_p2(cid, eid)
                # Relation extractor expects offsets relative to the claim text
                rel_ents = [
                    type("Ent", (), {"start_char": e.start - claim.start, "end_char": e.end - claim.start, "text": e.text})
                    for e, _ in local_entities
                ]
                for rel in self.relations.extract(claim.text, entities=rel_ents):
                    se = next(
                        (eid for e, eid in local_entities if e.start - claim.start == rel.subject_start and e.end - claim.start == rel.subject_end),
                        None,
                    )
                    oe = next(
                        (eid for e, eid in local_entities if e.start - claim.start == rel.object_start and e.end - claim.start == rel.object_end),
                        None,
                    )
                    if se is not None and oe is not None:
                        self.repo.insert_relation_p2(cid, se, rel.predicate, oe)
                        counts["relations"] += 1
                self.repo.upsert_embedding_p2(cid, self.embedder.model_name, vec)
                counts["embeddings"] += 1
            counts["timing_relations_s"] += time.perf_counter() - t0

            # Checkpoint chunk completion
            self.repo.mark_chunk_processed(paper_id, ch.chunk_id)
            done_chunks.add(ch.chunk_id)
            counts["processed_chunks"] = len(done_chunks)
            processed_this_run += 1

            if progress_callback:
                elapsed = time.perf_counter() - start_time
                avg_per_chunk = elapsed / processed_this_run if processed_this_run > 0 else 0.0
                remaining = total_chunks - len(done_chunks)
                eta = avg_per_chunk * remaining
                progress_callback({
                    "stage": "phase2",
                    "paper_id": paper_id,
                    "processed_chunks": len(done_chunks),
                    "total_chunks": total_chunks,
                    "claims_found": counts["claims"],
                    "entities_found": counts["entities"],
                    "elapsed_s": elapsed,
                    "eta_s": eta,
                })

        return counts
