from __future__ import annotations
from collections import defaultdict

def build_evolution(repo, paper_ids=None):
    claims=repo.get_all_claims()
    if paper_ids is not None:
        pids = set(paper_ids)
        claims = [c for c in claims if c['paper_id'] in pids]
        claim_ids = {c['claim_id'] for c in claims}
    else:
        claim_ids = None
    by_topic=defaultdict(list)
    for c in claims:
        ents=repo.get_claim_entities(c['claim_id'])
        topics={e['text'].strip().lower() for e in ents}
        for topic in topics: by_topic[topic].append(c)
    snapshots=[]
    for topic, cs in by_topic.items():
        by_year=defaultdict(list)
        for c in cs:
            p=repo.get_paper_by_id(c['paper_id'])
            if p and p.publication_year is not None: by_year[p.publication_year].append(c)
        for year, year_claims in sorted(by_year.items()):
            scored=[]
            for c in year_claims:
                s=repo.get_current_aerf(c['claim_id'])
                scored.append((0 if s is None else s['score'],c['claim_id']))
            dom=max(scored)[1] if scored else None
            domscore=max(scored)[0] if scored else None
            support=0; contra=0; evidence=[]
            for c in year_claims:
                s_neigh = repo.get_validated_neighbors(c['claim_id'],'supports')
                c_neigh = repo.get_validated_neighbors(c['claim_id'],'contradicts')
                if claim_ids is not None:
                    s_neigh = [x for x in s_neigh if x['neighbor_id'] in claim_ids]
                    c_neigh = [x for x in c_neigh if x['neighbor_id'] in claim_ids]
                support += len(s_neigh)
                contra += len(c_neigh)
            evidence=[c['claim_id'] for c in year_claims]
            repo.store_evolution(topic,year,dom,domscore,support,contra,evidence)
            snapshots.append((topic,year,dom,domscore,support,contra))
    return snapshots
