import pytest
from experiments.lib.metrics import evaluate, evaluation_to_dict


def test_perfect_predictions_give_f1_of_one():
    records = [{"system_prediction": "supports", "human_label": "supports"}] * 5
    result = evaluate(records, min_n_for_stats=1)
    assert result.per_class["supports"].f1 == 1.0
    assert result.accuracy == 1.0


def test_confusion_matrix_counts_are_correct():
    records = [
        {"system_prediction": "supports", "human_label": "supports"},
        {"system_prediction": "contradicts", "human_label": "supports"},
        {"system_prediction": "contradicts", "human_label": "contradicts"},
    ]
    result = evaluate(records, min_n_for_stats=1)
    assert result.confusion_matrix["supports"]["supports"] == 1
    assert result.confusion_matrix["supports"]["contradicts"] == 1
    assert result.confusion_matrix["contradicts"]["contradicts"] == 1


def test_false_contradiction_rate():
    # 2 true supports, 1 wrongly predicted contradicts
    records = [
        {"system_prediction": "contradicts", "human_label": "supports"},
        {"system_prediction": "supports", "human_label": "supports"},
    ]
    result = evaluate(records, min_n_for_stats=1)
    assert result.false_contradiction_rate == pytest.approx(0.5)


def test_false_support_rate():
    records = [
        {"system_prediction": "supports", "human_label": "contradicts"},
        {"system_prediction": "contradicts", "human_label": "contradicts"},
    ]
    result = evaluate(records, min_n_for_stats=1)
    assert result.false_support_rate == pytest.approx(0.5)


def test_small_n_triggers_warning():
    records = [{"system_prediction": "supports", "human_label": "supports"}] * 3
    result = evaluate(records, min_n_for_stats=10)
    assert any("below the minimum" in w for w in result.warnings)


def test_missing_label_class_triggers_warning_not_crash():
    records = [{"system_prediction": "supports", "human_label": "supports"}] * 3
    result = evaluate(records, min_n_for_stats=1)
    assert result.per_class["contradicts"].support == 0
    assert result.per_class["contradicts"].f1 == 0.0
    assert any("contradicts" in w for w in result.warnings)


def test_invalid_label_raises():
    with pytest.raises(ValueError):
        evaluate([{"system_prediction": "supports", "human_label": "not_a_label"}])


def test_evaluation_to_dict_is_json_serialisable():
    import json
    records = [{"system_prediction": "neutral", "human_label": "neutral"}]
    result = evaluate(records, min_n_for_stats=1)
    json.dumps(evaluation_to_dict(result))  # raises if not serialisable
