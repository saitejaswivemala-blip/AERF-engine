# Phase 3 — Cross-document Scientific Reasoning + AERF

Phase 3 extends the existing Phase 1/2 SQLite architecture. It does not add a
second database and does not implement the Streamlit dashboard.

## Pipeline

Claims + embeddings → similarity candidate blocking → entity alignment →
bidirectional NLI → validated support/contradiction/neutral/uncertain edges →
AERF → adaptive score history → evidence graph → temporal evolution →
evidence-based gaps → explainability.

### Model

The default NLI model is `valhalla/distilbart-mnli-12-3`, which is already used
by Phase 2 for semantic claim classification. Reusing it avoids another model
family while providing the required three-way MNLI labels.

### Contradiction safety rule

Embedding similarity is only a candidate-generation gate. A relationship is
never labelled contradictory from similarity alone. Candidate pairs require
entity alignment (when entities exist) and a sufficiently confident
bidirectional NLI result.

## AERF formulation

For claim c, seven normalized factors are used:

`S` source-metadata completeness; `M` reported methodological-rigor signals;
`Cr` corroboration mass; `Cp` contradiction pressure; `T` temporal relevance;
`Sem` semantic/NLI confidence; `Pr` provenance completeness.

`AERF(c) = wS*S + wM*M + wCr*Cr + wT*T + wSem*Sem + wPr*Pr - wCp*Cp`

The result is clipped to [0,1]. Weights are versioned in
`config/aerf_weights.yaml`. They are initial heuristic parameters, not learned
or claimed to be optimal.

Corroboration uses the current reliability of independent supporting claims:

`Cr = support_mass / (k + support_mass)`

Contradiction pressure uses current reliability and NLI confidence:

`Cp = contradiction_mass / (k + contradiction_mass)`

Temporal relevance is a recency signal, not a truth signal:

`T = max(0, 1 - (max_year - publication_year)/decay_window)`

Missing metadata is not invented; factors use neutral/partial representations
where the underlying information is unavailable.

## Adaptive updates

When new cross-document evidence is validated, the endpoints and direct
neighbors are recomputed. Every changed current score becomes a new version;
the previous score remains in `aerf_scores` and the transition is recorded in
`aerf_score_history` with factor deltas, trigger identifiers, and a
number-derived reason.

Duplicate evidence is constrained by the claim-pair uniqueness operation and
therefore is not allowed to multiply corroboration indefinitely.

## Run

From the project root after Phase 2 data exists:

```powershell
python main.py --phase3-paper-id 2
```

Use an already-ingested paper ID. Phase 3 considers that paper's claims against
claims from other papers in the same SQLite database. For the first meaningful
cross-document run, ingest and run Phase 2 on at least two papers.

Run the full test suite:

```powershell
pytest -v
```

## Limitations

- Methodological strength is a transparent text-signal approximation, not a
  verified peer-review or statistical-quality assessment.
- Venue/source factor is metadata completeness, not journal prestige or impact
  factor; no unsupported external quality ranking is fabricated.
- Claim matching is O(n²) over claims after blocking and is intended for the
  project's small research-prototype scale.
- Research-gap detection is deliberately evidence-pattern based and should be
  treated as candidate gaps for human review.
- NLI outputs depend on the selected pretrained model and have not been
  calibrated here on a project-specific scientific contradiction dataset.
- No patentability or novelty conclusion is made by the implementation.
- Streamlit/dashboard work is intentionally excluded from Phase 3.
