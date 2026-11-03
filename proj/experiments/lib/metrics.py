"""
Quantitative evaluation for the multi-paper experiment.

Pure, model-free, deterministic metric computation over a list of
(system_prediction, human_label) pairs. Deliberately has no dependency on
the rest of AERF Engine so it can be unit-tested in isolation and trusted
regardless of whether real NLP models are available in a given
environment.

Labels are one of: 'supports', 'contradicts', 'neutral', 'uncertain'.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict

LABELS = ("supports", "contradicts", "neutral", "uncertain")


@dataclass(frozen=True)
class ClassMetrics:
    label: str
    precision: float
    recall: float
    f1: float
    support: int  # number of ground-truth instances of this label


@dataclass(frozen=True)
class EvaluationResult:
    n: int
    confusion_matrix: dict  # {true_label: {pred_label: count}}
    per_class: dict  # {label: ClassMetrics}
    macro_precision: float
    macro_recall: float
    macro_f1: float
    accuracy: float
    false_support_rate: float  # predicted 'supports' when true label was 'contradicts'
    false_contradiction_rate: float  # predicted 'contradicts' when true label was 'supports'
    warnings: list = field(default_factory=list)


def _safe_div(n, d):
    return n / d if d else 0.0


def evaluate(records: list[dict], min_n_for_stats: int = 10) -> EvaluationResult:
    """
    records: list of {'system_prediction': str, 'human_label': str}.
    Both fields must be one of LABELS. Records with a human_label outside
    LABELS are rejected (raises ValueError) rather than silently skipped,
    since a bad label is a data-quality bug that should surface immediately
    for a scoring function whose whole purpose is trustworthy numbers.
    """
    n = len(records)
    warnings = []
    confusion = {t: {p: 0 for p in LABELS} for t in LABELS}
    for r in records:
        true, pred = r["human_label"], r["system_prediction"]
        if true not in LABELS or pred not in LABELS:
            raise ValueError(f"Label outside {LABELS}: true={true!r} pred={pred!r}")
        confusion[true][pred] += 1

    per_class = {}
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[t][label] for t in LABELS if t != label)
        fn = sum(confusion[label][p] for p in LABELS if p != label)
        support = sum(confusion[label].values())
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        per_class[label] = ClassMetrics(label, precision, recall, f1, support)

    present_labels = [l for l in LABELS if per_class[l].support > 0]
    macro_p = _safe_div(sum(per_class[l].precision for l in present_labels), len(present_labels))
    macro_r = _safe_div(sum(per_class[l].recall for l in present_labels), len(present_labels))
    macro_f1 = _safe_div(sum(per_class[l].f1 for l in present_labels), len(present_labels))
    accuracy = _safe_div(sum(confusion[l][l] for l in LABELS), n)

    false_support = confusion["contradicts"]["supports"]
    false_contradiction = confusion["supports"]["contradicts"]
    n_contradicts = per_class["contradicts"].support
    n_supports = per_class["supports"].support
    false_support_rate = _safe_div(false_support, n_contradicts)
    false_contradiction_rate = _safe_div(false_contradiction, n_supports)

    if n < min_n_for_stats:
        warnings.append(
            f"n={n} is below the minimum ({min_n_for_stats}) for statistically meaningful "
            "precision/recall/F1. Treat these numbers as a smoke test of the metric "
            "computation, not as an evaluation of AERF Engine's real-world accuracy."
        )
    for label in LABELS:
        if per_class[label].support == 0:
            warnings.append(f"No ground-truth instances of label '{label}' in this dataset; "
                             f"its precision/recall/F1 are reported as 0.0 by convention, not measured.")

    return EvaluationResult(
        n=n, confusion_matrix=confusion, per_class=per_class,
        macro_precision=macro_p, macro_recall=macro_r, macro_f1=macro_f1,
        accuracy=accuracy, false_support_rate=false_support_rate,
        false_contradiction_rate=false_contradiction_rate, warnings=warnings,
    )


def evaluation_to_dict(result: EvaluationResult) -> dict:
    return {
        "n": result.n,
        "confusion_matrix": result.confusion_matrix,
        "per_class": {k: vars(v) for k, v in result.per_class.items()},
        "macro_precision": result.macro_precision,
        "macro_recall": result.macro_recall,
        "macro_f1": result.macro_f1,
        "accuracy": result.accuracy,
        "false_support_rate": result.false_support_rate,
        "false_contradiction_rate": result.false_contradiction_rate,
        "warnings": result.warnings,
    }
