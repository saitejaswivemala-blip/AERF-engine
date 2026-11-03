from experiments.lib.experiment_config import (
    BASELINE_A_SIMILARITY_ONLY, BASELINE_B_SIMILARITY_NLI,
    BASELINE_C_SIMILARITY_ENTITY_NLI, FULL_AERF,
    ABLATION_WITHOUT_ENTITY_ALIGNMENT, ABLATION_WITHOUT_ADAPTIVE_SCORING,
    ABLATION_WITHOUT_PROVENANCE_WEIGHTING, ALL_BASELINES, ALL_ABLATIONS,
)


def test_baseline_a_disables_entity_alignment_and_nli():
    assert BASELINE_A_SIMILARITY_ONLY.use_entity_alignment is False
    assert BASELINE_A_SIMILARITY_ONLY.use_nli is False
    assert BASELINE_A_SIMILARITY_ONLY.label_from_similarity_only is True


def test_baseline_b_enables_nli_disables_entity_alignment():
    assert BASELINE_B_SIMILARITY_NLI.use_nli is True
    assert BASELINE_B_SIMILARITY_NLI.use_entity_alignment is False


def test_baseline_c_matches_shipped_claimmatcher_gates():
    assert BASELINE_C_SIMILARITY_ENTITY_NLI.use_entity_alignment is True
    assert BASELINE_C_SIMILARITY_ENTITY_NLI.use_nli is True
    assert BASELINE_C_SIMILARITY_ENTITY_NLI.use_aerf_scoring is False


def test_full_aerf_enables_everything():
    assert FULL_AERF.use_entity_alignment is True
    assert FULL_AERF.use_nli is True
    assert FULL_AERF.use_aerf_scoring is True
    assert FULL_AERF.use_adaptive_updates is True


def test_ablation_without_entity_alignment_only_differs_there():
    assert ABLATION_WITHOUT_ENTITY_ALIGNMENT.use_entity_alignment is False
    assert ABLATION_WITHOUT_ENTITY_ALIGNMENT.use_nli is True
    assert ABLATION_WITHOUT_ENTITY_ALIGNMENT.use_aerf_scoring is True


def test_ablation_without_adaptive_disables_only_adaptive_updates():
    assert ABLATION_WITHOUT_ADAPTIVE_SCORING.use_adaptive_updates is False
    assert ABLATION_WITHOUT_ADAPTIVE_SCORING.use_aerf_scoring is True


def test_ablation_without_provenance_zeroes_that_weight_only():
    assert ABLATION_WITHOUT_PROVENANCE_WEIGHTING.zero_provenance_weight is True
    assert ABLATION_WITHOUT_PROVENANCE_WEIGHTING.use_entity_alignment is True


def test_all_baseline_and_ablation_lists_are_nonempty_and_named_uniquely():
    assert len(ALL_BASELINES) == 4
    assert len(ALL_ABLATIONS) == 5
    assert len({c.name for c in ALL_BASELINES}) == 4
    assert len({c.name for c in ALL_ABLATIONS}) == 5
