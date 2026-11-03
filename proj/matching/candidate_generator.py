from __future__ import annotations
from .claim_matcher import ClaimMatcher

def generate_candidates(repo, similarity_threshold=.55):
    claims=repo.get_all_claims()
    out=[]
    for i,a in enumerate(claims):
        va=repo.get_claim_embedding(a['claim_id'])
        if va is None: continue
        for b in claims[i+1:]:
            if a['paper_id']==b['paper_id']: continue
            vb=repo.get_claim_embedding(b['claim_id'])
            if vb is None: continue
            sim=ClaimMatcher.cosine(va,vb)
            if sim>=similarity_threshold: out.append((a,b,sim))
    return out
