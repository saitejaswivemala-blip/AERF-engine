# Candidate Paper Selection Report

## 1. Papers considered from REVIEW1

REVIEW1.docx's literature survey (20 papers) was read in full before any new
search was done, per the task instructions. All 20 concern claim-verification,
NLI, contradiction-detection, or explainability *methods* (SciFact, SciFact-Open,
SCITAB, Clinical Contradiction Detection, ContraCrow/PaperQA2, Sentence-BERT,
BioNLI, RAV, etc.) — they are papers *about* systems like AERF Engine, not papers
that themselves report primary scientific findings.

**All 20 were rejected for this corpus.** None contain claims of the shape AERF
Engine is built to compare ("X reduces/increases/treats/prevents Y") in a form
that would let two papers genuinely support, contradict, or stay neutral on the
same real-world question. Extracting "claims" from them would mean sentences
like *"SciFact contains 1.4K expert-written claims"* — true, but not the kind of
cross-document scientific disagreement the experiment needs to exercise. They
remain correctly used elsewhere (the technical audit's prior-art section) as
prior art on the *methods*, which is a different role from experimental content.

## 2. Additional papers selected

A new, coherent topic was chosen instead: **vitamin D supplementation and acute
respiratory infection (ARI) risk** — a real, actively debated question with
large open-access RCTs and meta-analyses, including a documented public
methodological dispute (see P2 below). Search and retrieval used `web_search`
and `web_fetch` only; no paywalled, pirated, or Sci-Hub sources were used or
considered.

| ID | Title (short) | Year | Venue | Access | Full text retrieved? |
|---|---|---|---|---|---|
| P1 | Vitamin D supplementation to prevent ARI: updated meta-analysis (Jolliffe/Martineau et al.) | 2024 | medRxiv preprint | Open, freely downloadable | **Yes** — full text fetched directly |
| P2 | Vitamin D and ARI: secondary analysis of the Blue Sky trial (Bolland, Avenell, Grey, Gamble) | 2022 | medRxiv preprint, CC-BY-NC 4.0 | Open, freely downloadable | **Yes** — full text fetched directly |
| P3 | CORONAVIT: Phase 3 RCT of vitamin D for COVID-19/ARI prevention (Jolliffe/Martineau et al.) | 2022 | medRxiv preprint, CC-BY-NC-ND 4.0 | Open, freely downloadable | **Yes** — full text fetched directly |
| P4 | Efficacy of vitamin D in TREATMENT of ARI: meta-analysis of 18 RCTs | 2022 | PMC (PMCID PMC8955452) | Open access (PMC) | **No** — PMC page blocked automated fetch (reCAPTCHA interstitial); indexed abstract retrieved via search only |
| P5 | Vitamin D in prevention/treatment of ARI in pediatric populations: systematic review | 2025 | PMC (PMCID PMC12699866) | Open access (PMC) | **No** — same PMC block; indexed abstract only |

Full manifest with DOIs, URLs, license, and per-paper selection rationale is in
`experiments/datasets/paper_manifest.json`.

## 3. Why each paper is useful

- **P1** is the largest, most current meta-analysis (46 RCTs, 64,086
  participants) and itself contains internal heterogeneity: a null overall
  result alongside statistically significant protective subgroups (children
  1.00–15.99y, daily dosing, 400–1000 IU/day). This alone stresses claim/entity
  alignment (is "the daily-dosing subgroup" the same claim as "overall"?).
- **P2** is a genuine, published methodological rebuttal of the trial data that
  drove earlier positive meta-analyses (including a 2021 predecessor of P1).
  This is real scientific disagreement, not a constructed example.
- **P3** is a single large, well-powered RCT (not a meta-analysis) in a
  different population (general UK adults) and a different intervention design
  (test-and-treat), giving a distinct evidence-strength/design profile.
- **P4** is the only treatment-focused (vs. prevention-focused) paper found,
  giving a genuinely different claim type to test whether the matcher can tell
  "treats an existing infection" apart from "prevents a future one" rather than
  conflating them as agreement or disagreement.
- **P5** targets the same population (children) as P1's positive subgroup but
  reaches a null conclusion — a real same-population, differing-condition pair
  (P1's subgroup is prevention-only/daily-dosing-only; P5 pools prevention and
  treatment across dosing regimens), useful for testing whether the system
  correctly treats this as genuinely ambiguous rather than a clean
  contradiction.

## 4. Expected cross-document relationships

Documented, with the exact claim wording used, in
`experiments/annotations/ground_truth_claim_pairs.json` (8 human-annotated
pairs: gt_01–gt_08). Summary:

- 3 pairs are genuine **supports** (independent null results corroborating each
  other: P1↔P2, P1↔P3, P2↔P3).
- 1 pair is a genuine **contradicts**, drawn from within P1 itself (its own
  2021 predecessor result vs. its 2024 update — a real "evidence changed the
  conclusion" case, valuable for the adaptive-AERF experiment).
- 4 pairs are **neutral** by design (subgroup-vs-overall, or genuinely
  different clinical questions/populations that should not be scored as either
  agreement or disagreement).

No pair was fabricated to force a target label; several were deliberately
chosen because the honest answer is "not comparable" or "genuinely ambiguous"
(gt_04, gt_05, gt_06), since a system that only ever gets asked easy questions
teaches you nothing.

## 5. Verification of free accessibility

- P1, P2, P3: fetched in full directly from `medrxiv.org` in this session — no
  access barrier encountered. P2 and P3 carry explicit CC-BY-NC / CC-BY-NC-ND
  licenses; P1 carries medRxiv's standard preprint license (author/funder
  retains copyright, medRxiv granted a perpetual display license). None
  requires a subscription or purchase.
- P4, P5: PMC confirms both are indexed in PubMed Central, which only carries
  open-access or public-access content; the article pages are free to any
  reader in a normal browser. This session's `web_fetch` tool was blocked by
  an automated bot-check specific to `pmc.ncbi.nlm.nih.gov`, not by any access
  restriction — see the Reproducibility section of the main experiment report
  for exactly what a user with normal browser access needs to do to complete
  these two records.

## 6. Final selected corpus

P1, P2, P3, P4, P5 — 5 papers, one coherent topic, genuine cases of support,
contradiction, neutral/differing-condition, and differing evidence strength
(large multi-trial meta-analyses vs. a single RCT vs. abstract-only records).
