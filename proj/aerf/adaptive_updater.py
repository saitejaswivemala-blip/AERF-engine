from __future__ import annotations
class AdaptiveAERFUpdater:
    def __init__(self, repo, scorer, epsilon=.001): self.repo=repo; self.scorer=scorer; self.epsilon=epsilon
    def recompute(self, claim_ids, trigger_paper_id=None, trigger_claim_id=None):
        affected=set(claim_ids)
        for cid in list(affected): affected.update(self.repo.get_direct_neighbor_ids(cid))
        changed=[]
        for cid in sorted(affected):
            old=self.repo.get_current_aerf(cid)
            result=self.scorer.score(cid)
            self.repo.store_aerf_score(cid,result.score,result.factors,result.weights_version)
            if old is not None:
                old_score=float(old['score'])
                if abs(old_score-result.score)>self.epsilon:
                    deltas={k:result.factors[k]-float(old['factor_breakdown'].get(k,result.factors[k])) for k in result.factors}
                    direction='increased' if result.score>old_score else 'decreased'
                    reason=(f"Reliability {direction} by {abs(result.score-old_score):.4f}; "
                            f"corroboration={result.factors['corroboration']:.4f}; "
                            f"contradiction_pressure={result.factors['contradiction']:.4f}; "
                            f"semantic={result.factors['semantic']:.4f}; temporal={result.factors['temporal']:.4f}.")
                    self.repo.store_aerf_history(cid,old_score,result.score,trigger_paper_id,trigger_claim_id,deltas,reason)
                    changed.append((cid,old_score,result.score))
        return changed
