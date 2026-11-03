from __future__ import annotations

def detect_gaps(repo, low_corroboration=.35, unresolved_margin=.15, paper_ids=None):
    claims=repo.get_all_claims()
    if paper_ids is not None:
        pids = set(paper_ids)
        claims = [c for c in claims if c['paper_id'] in pids]
        claim_ids = {c['claim_id'] for c in claims}
    else:
        claim_ids = None
    gaps=[]
    for c in claims:
        score=repo.get_current_aerf(c['claim_id'])
        if not score: continue
        f=score['factor_breakdown']
        support=repo.get_validated_neighbors(c['claim_id'],'supports')
        contra=repo.get_validated_neighbors(c['claim_id'],'contradicts')
        if claim_ids is not None:
            support = [x for x in support if x['neighbor_id'] in claim_ids]
            contra = [x for x in contra if x['neighbor_id'] in claim_ids]
        ents=repo.get_claim_entities(c['claim_id']); topic=ents[0]['text'].strip().lower() if ents else f'claim:{c["claim_id"]}'
        if f['corroboration'] < low_corroboration and not contra and not support:
            repo.store_gap(topic,'low_corroboration','Claim has no validated cross-document corroboration or contradiction.',[c['claim_id']],{'claim_id':c['claim_id'],'corroboration':f['corroboration']})
            gaps.append((topic,'low_corroboration',c['claim_id']))
        elif support and contra:
            sm=sum(float(x['neighbor_score'] or .5)*float(x['nli_confidence'] or 0) for x in support)
            cm=sum(float(x['neighbor_score'] or .5)*float(x['nli_confidence'] or 0) for x in contra)
            if abs(sm-cm) <= unresolved_margin:
                repo.store_gap(topic,'unresolved_contradiction','Supporting and contradicting evidence have comparable evidence mass; the disagreement remains unresolved.',[c['claim_id']]+[x['neighbor_id'] for x in support+contra],{'support_mass':sm,'contradiction_mass':cm})
                gaps.append((topic,'unresolved_contradiction',c['claim_id']))
    return gaps
