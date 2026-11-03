# Curated Evaluation Mapping — gt_01, gt_02, gt_03, gt_05

## What this is

A hand-curated, explicitly-scoped subset of the 8 ground-truth pairs in
`experiments/annotations/ground_truth_claim_pairs.json`, mapped to specific
real DB claim IDs, for a real-NLI cross-document evaluation run.
`ground_truth_claim_pairs.json` itself is **not modified** — the curated
mapping lives in a separate file,
`experiments/annotations/curated_mapping_gt01_gt02_gt03_gt05.json`, which
copies the original `human_label` and claim text read-only for reference.

## Which pairs are in scope, and why

| Pair | Status | Reason |
|---|---|---|
| gt_01 | **Included** | P1 vs P2, both sides have a corresponding extracted claim |
| gt_02 | **Included** | P1 vs P3, both sides have a corresponding extracted claim |
| gt_03 | **Included** | P2 vs P3, both sides have a corresponding extracted claim |
| gt_05 | **Included** | P4 vs P1, both sides have a corresponding extracted claim |
| gt_04 | **Excluded** | Per your read-only inspection output: P5 has no extracted claim corresponding to the daily-dosing-subgroup comparison this pair needs. Not independently re-verified here (I don't have your database) — recorded as your finding, not mine. |
| gt_06 | **Excluded** | Same reason as gt_04 — shares gt_04's P5 side. |
| gt_07 | **Out of scope for this mapping** | Same-paper (P1 vs P1). Stays a separate diagnostic, per the existing rule that `Phase3Pipeline.run()`'s candidate generator explicitly skips same-paper pairs (`pipeline/phase3.py`). This file does not touch it. |
| gt_08 | **Out of scope for this mapping** | Same reason as gt_07. |

## Why claim IDs are filled in by hand, not auto-resolved

`scripts/evaluate_real_results.py`'s `resolve_claim()` already does automatic
text resolution (exact → substring → token-overlap ≥0.5) for all 8 pairs.
This curated mapping exists as a **second, independent check** for exactly
the 4 pairs where that matters most (the ones intended to carry the
headline cross-document result) — a human (you, using
`scripts/inspect_claim_mapping.py` / `scripts/inspect_top_matches.py`'s
real output) confirms the specific `claim_id` for each side, rather than
trusting the automatic resolver silently. This is stricter than, not a
replacement for, the automatic path — if a manually-confirmed `claim_id`
happens to differ from what `resolve_claim()` would have picked, that's
worth noticing, not smoothing over.

**No claim_id in this mapping is invented.** The template ships with every
`claim_a_id`/`claim_b_id` set to `null`, and
`scripts/evaluate_curated_mapping.py` refuses to run while any required
field is still `null` — see that script for the exact check.

## Validity of the result this produces

Once run, the result is explicitly **n=4, diagnostic, not a benchmark**:

- 4 cross-document pairs is far below `metrics.evaluate()`'s own
  `min_n_for_stats` (10) — its warning will fire and should not be
  suppressed or explained away.
- These 4 pairs were not randomly sampled; they were chosen because they
  were the ones with the cleanest available claim extraction, which is
  itself a form of selection — do not generalize this result to "AERF
  Engine's accuracy" beyond these 4 specific pairs.
- A correct or incorrect prediction on any single one of the 4 changes the
  headline accuracy by 25 percentage points. Report the per-pair predictions
  (which `scripts/evaluate_curated_mapping.py` prints and saves), not just
  the aggregate number.
- gt_07/gt_08 (same-paper) are never merged into this n=4 figure, by design
  — see above.

## How to complete and run this

1. Run `scripts/inspect_claim_mapping.py` and/or
   `scripts/inspect_top_matches.py` against your real
   `experiments/runs/experiment.db` (already documented in
   `FINAL_RUN_GUIDE.md`).
2. For each of gt_01, gt_02, gt_03, gt_05, read off the real `claim_id` for
   claim_a and claim_b, and fill in
   `experiments/annotations/curated_mapping_gt01_gt02_gt03_gt05.json`'s
   `claim_a_id`/`claim_b_id` fields — plus a one-line `mapping_rationale`
   per pair (which resolution method matched it, or why you chose it
   manually if the automatic resolver disagreed).
3. Set `_status` to `"FILLED_IN"`.
4. Run:
   ```
   python scripts/evaluate_curated_mapping.py
   ```
