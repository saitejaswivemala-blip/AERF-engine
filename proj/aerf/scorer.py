from __future__ import annotations
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class AERFResult:
    score:float
    factors:dict
    weighted_components:dict
    weights_version:str

class AERFScorer:
    """Deterministic, evidence-derived AERF scorer. Weights are configuration, not learned here."""
    def __init__(self, repo, weights=None, version="aerf-v1", smoothing=2.0, decay_window=10.0):
        self.repo=repo
        self.weights=weights or {"source":.10,"methodology":.15,"corroboration":.20,"contradiction":.20,"temporal":.10,"semantic":.15,"provenance":.10}
        self.version=version; self.k=float(smoothing); self.decay_window=float(decay_window)
    @staticmethod
    def _clip(x): return max(0.0,min(1.0,float(x)))
    def factors(self, claim_id):
        c=self.repo.get_claim(claim_id); paper=self.repo.get_paper_by_id(c["paper_id"])
        source=sum(x is not None and str(x).strip()!="" for x in (paper.title,paper.authors,paper.publication_year,paper.venue))/4
        text=c["text"].lower()
        design=any(x in text for x in ("randomized","randomised","controlled","cohort","case-control","meta-analysis","systematic review"))
        sample=any(ch.isdigit() for ch in text) and any(x in text for x in ("n=","n =","sample","patients","participants"))
        stats=any(x in text for x in ("p <","p=","p =","confidence interval","95% ci","odds ratio","hazard ratio"))
        methodology=self._clip(.4*design+.3*sample+.3*stats)
        support=self.repo.get_validated_neighbors(claim_id,"supports")
        contra=self.repo.get_validated_neighbors(claim_id,"contradicts")
        support_mass=sum(float(x["neighbor_score"] or .5) * float(x["nli_confidence"] or 0) for x in support)
        contra_mass=sum(float(x["neighbor_score"] or .5) * float(x["nli_confidence"] or 0) for x in contra)
        corroboration=self._clip(support_mass/(self.k+support_mass))
        contradiction=self._clip(contra_mass/(self.k+contra_mass))
        years=[p.publication_year for p in self.repo.list_papers() if p.publication_year is not None]
        py=paper.publication_year
        temporal=.5 if py is None or not years else self._clip(1-(max(years)-py)/self.decay_window)
        nlis=self.repo.get_claim_pair_confidences(claim_id)
        semantic=max([float(x) for x in nlis], default=.5)
        provenance=self.repo.provenance_completeness(claim_id)
        return {"source":self._clip(source),"methodology":methodology,"corroboration":corroboration,"contradiction":contradiction,"temporal":temporal,"semantic":self._clip(semantic),"provenance":self._clip(provenance)}
    def score(self, claim_id):
        f=self.factors(claim_id); w=self.weights
        components={k:w[k]*f[k] for k in w}
        raw=sum(components[k] for k in ("source","methodology","corroboration","temporal","semantic","provenance"))-components["contradiction"]
        return AERFResult(self._clip(raw),f,components,self.version)
