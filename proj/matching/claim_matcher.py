from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from matching.entity_alignment import align_entities as _align_entities_v2

@dataclass(frozen=True)
class MatchDecision:
    claim_a_id: int
    claim_b_id: int
    cosine_similarity: float
    entity_overlap_score: float
    relation_alignment_score: float
    nli_label: str
    nli_confidence: float
    validated_as: str
    nli_entailment: float
    nli_contradiction: float
    nli_neutral: float
    method: str
    entity_alignment_detail: object = None  # AlignmentResult when entity_alignment_version="v2", else None

class ClaimMatcher:
    """
    Similarity is only a candidate gate; NLI + entity alignment decide the
    relationship.

    entity_alignment_version:
      "v1" (default) -- ORIGINAL, UNCHANGED behavior: exact-string overlap
        after strip().lower() only. This is what every existing test,
        experiment result, and documented number (README.md, experiment
        reports) was produced with. Do not change this default.
      "v2" (opt-in) -- routes entity comparison through
        matching/entity_alignment.py: Unicode/case/punctuation
        normalization, conservative plural handling, and a small explicit
        alias table (e.g. OR <-> odds ratio), while explicitly refusing to
        merge related-but-distinct measures (e.g. rate ratio vs relative
        risk) or coincidental character substrings. See that module's
        docstring and tests/unit/test_entity_alignment.py for the real
        diagnostic evidence behind this option.
    """
    def __init__(self, nli, similarity_threshold=.55, entity_threshold=.2, nli_threshold=.6,
                 entity_alignment_version="v1"):
        self.nli=nli; self.similarity_threshold=float(similarity_threshold)
        self.entity_threshold=float(entity_threshold); self.nli_threshold=float(nli_threshold)
        if entity_alignment_version not in ("v1", "v2"):
            raise ValueError("entity_alignment_version must be 'v1' or 'v2'")
        self.entity_alignment_version = entity_alignment_version
    @staticmethod
    def cosine(a,b):
        a=np.asarray(a,dtype=float); b=np.asarray(b,dtype=float)
        d=np.linalg.norm(a)*np.linalg.norm(b)
        return float(np.dot(a,b)/d) if d else 0.0
    @staticmethod
    def _relation_alignment(a,b):
        if not a and not b: return 0.0
        sa={(r.get('predicate'),r.get('subject_text','').lower(),r.get('object_text','').lower()) for r in a}
        sb={(r.get('predicate'),r.get('subject_text','').lower(),r.get('object_text','').lower()) for r in b}
        u=sa|sb
        return len(sa&sb)/len(u) if u else 0.0
    def match(self, claim_a, claim_b, entities_a, entities_b, relations_a, relations_b, emb_a, emb_b):
        sim=self.cosine(emb_a,emb_b)
        if sim < self.similarity_threshold: return None
        alignment_detail = None
        if self.entity_alignment_version == "v2":
            alignment_detail = _align_entities_v2(entities_a, entities_b, threshold=self.entity_threshold)
            overlap = alignment_detail.overlap_score
            u = bool(entities_a or entities_b)
        else:
            ea={e['text'].strip().lower() for e in entities_a}; eb={e['text'].strip().lower() for e in entities_b}
            u=ea|eb; overlap=len(ea&eb)/len(u) if u else 0.0
        if u and overlap < self.entity_threshold: return None
        ra=self._relation_alignment(relations_a,relations_b)
        n=self.nli.predict_bidirectional(claim_a['text'],claim_b['text'])
        aligned=(overlap>=self.entity_threshold) if u else sim>=max(.75,self.similarity_threshold)
        validated='uncertain'
        if aligned and n.confidence>=self.nli_threshold:
            validated={'entailment':'supports','contradiction':'contradicts','neutral':'neutral'}.get(n.label,'uncertain')
        return MatchDecision(claim_a['claim_id'],claim_b['claim_id'],sim,overlap,ra,n.label,n.confidence,validated,n.entailment,n.contradiction,n.neutral,'embedding→entity-alignment→bidirectional-NLI',alignment_detail)
