def build_consensus(repo):
    # Timeline snapshots already contain the dominant AERF-weighted claim per topic/year.
    return repo.get_evolution_snapshots()
