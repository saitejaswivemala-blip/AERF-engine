
"""
Diagnostic tests for nlp/claim_extractor.py.

A) Code-logic tests use fake model scores.
B) Real-model evaluation uses the installed spaCy and HF models.

Run all tests:
    python -m pytest tests/unit/test_claim_extractor_diagnostic.py -v -s

Run only the real-model evaluation:
    python -m pytest tests/unit/test_claim_extractor_diagnostic.py -k real_model -v -s
"""

import re
import pytest
from nlp.claim_extractor import ScientificClaimExtractor as E

L = E.LABELS  # [finding, background, method, irrelevant]


def P(f, bg, me, ir):
    return {L[0]: f, L[1]: bg, L[2]: me, L[3]: ir}


# ---------------------------------------------------------------- examples

FINDINGS = [
    "Experiments show that our proposed model significantly improves accuracy over existing methods.",
    "Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable.",
    "On the WMT 2014 English-to-German translation task, the big transformer model outperforms the best previously reported models by more than 2.0 BLEU, establishing a new state-of-the-art BLEU score of 28.4.",
    "The intervention significantly reduced mortality compared with placebo (hazard ratio 0.72, 95% CI 0.60-0.86).",
]

NOT_FINDINGS = {
    "background": "The best performing models also connect the encoder and decoder through an attention mechanism.",
    "background_cited": "Prior work by Smith et al. showed that attention improves translation accuracy over recurrent models.",
    "method": "We trained the model on the WMT 2014 English-German dataset for 100,000 steps using the Adam optimizer.",
    "method2": "We use dropout to improve generalization of the network during training.",
    "irrelevant": "Permission to make digital or hard copies of all or part of this work is granted without fee.",
}

# Fake scores for logic tests only. These are NOT real-model predictions.
FAKE = {
    FINDINGS[0]: P(.386, .125, .463, .026),
    FINDINGS[1]: P(.20, .05, .708, .02),
    FINDINGS[2]: P(.30, .05, .50, .02),
    FINDINGS[3]: P(.85, .05, .08, .02),
    NOT_FINDINGS["background"]: P(.15, .612, .20, .03),
    NOT_FINDINGS["background_cited"]: P(.20, .70, .08, .02),
    NOT_FINDINGS["method"]: P(.10, .05, .80, .05),
    NOT_FINDINGS["method2"]: P(.10, .05, .80, .05),
    NOT_FINDINGS["irrelevant"]: P(.03, .05, .12, .80),
}


# ------------------------------------------------------------ fake spaCy

class _Tok:
    def __init__(self, pos):
        self.pos_ = pos


class _Sent:
    def __init__(self, text, s, e, verb=True):
        self.text = text
        self.start_char = s
        self.end_char = e
        self._t = [_Tok("VERB" if verb else "NOUN")]

    def __iter__(self):
        return iter(self._t)


class _Doc:
    def __init__(self, sents):
        self.sents = sents


class FakeNLP:
    """Simple test splitter for sentence-boundary logic."""

    def __init__(self, verb=True):
        self.verb = verb

    def __call__(self, text):
        out, pos = [], 0
        for m in re.finditer(r"(?<=[.!?])\s+", text):
            out.append(_Sent(text[pos:m.start()], pos, m.start(), self.verb))
            pos = m.end()
        if pos < len(text):
            out.append(_Sent(text[pos:], pos, len(text), self.verb))
        return _Doc(out)


class FakeClassifier:
    def __call__(self, x, labels, multi_label=False, hypothesis_template=None):
        items = x if isinstance(x, list) else [x]
        res = []
        for t in items:
            p = FAKE.get(t.strip(), P(.1, .1, .1, .7))
            order = sorted(p, key=p.get, reverse=True)
            res.append({
                "labels": order,
                "scores": [p[k] for k in order],
            })
        return res if isinstance(x, list) else res[0]


def ex(verb=True):
    return E(nlp=FakeNLP(verb), classifier=FakeClassifier(), threshold=.55)


# ================================================================= A) logic tests

@pytest.mark.parametrize("s", FINDINGS)
def test_findings_are_accepted(s):
    r = ex().extract(s)
    assert len(r) == 1 and r[0].text == s


@pytest.mark.parametrize("name,s", NOT_FINDINGS.items())
def test_method_background_irrelevant_are_rejected(name, s):
    assert ex().extract(s) == []


def test_contribution_sentence_accepted_but_not_plain_method():
    s = "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms."
    FAKE[s] = P(.20, .584, .15, .02)
    assert [c.text for c in ex().extract(s)] == [s]


def test_threshold_not_lowered_noise_stays_out():
    s = "The encoder maps an input sequence of symbol representations to continuous representations."
    FAKE[s] = P(.40, .30, .25, .05)
    assert ex().extract(s) == []


def test_offsets_exact_with_mixed_text_glued_heading_and_abbrev():
    a, b, c = FINDINGS[0], NOT_FINDINGS["method"], FINDINGS[3]
    text = f"Abstract{a} {b} See Smith et al. for details. {c}"
    FAKE[a] = P(.386, .125, .463, .026)
    res = ex().extract(text)
    assert [r.text for r in res] == [a, c]
    for r in res:
        assert text[r.start:r.end] == r.text


def test_glued_sentences_are_split():
    a, c = FINDINGS[0], FINDINGS[3]
    text = a + c
    assert [r.text for r in ex().extract(text)] == [a, c]


def test_eligibility_lenient_without_verb_pos_but_rejects_junk():
    assert ex(verb=False).extract(FINDINGS[0])
    for junk in (
        "Table 2: Results on the test set for all models.",
        "[12] Vaswani et al. arXiv preprint arXiv:1706.03762, 2017.",
        "Page 7 of 12 -- 1234567 89012",
    ):
        assert not E._eligible_text(junk, True)


def test_claim_result_interface_unchanged():
    r = ex().extract(FINDINGS[3])[0]
    assert (
        r.text, r.start, r.end, r.label
    ) == (
        FINDINGS[3], 0, len(FINDINGS[3]), "scientific_finding"
    )
    assert isinstance(r.score, float)


# ============================================================ B) real-model evaluation

@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("transformers")
    or not __import__("importlib").util.find_spec("spacy"),
    reason="spaCy/transformers not installed",
)
def test_real_model_accuracy_report():
    try:
        import spacy
        spacy.load("en_core_sci_sm")
    except Exception:
        pytest.skip("en_core_sci_sm not installed")

    # Use actual model and actual extractor thresholds.
    real = E()

    cases = (
        [(s, True) for s in FINDINGS]
        + [(s, False) for s in NOT_FINDINGS.values()]
    )

    tp = fp = fn = tn = 0

    print("\n")
    print("=" * 65)
    print("AERF ENGINE - REAL MODEL PERFORMANCE REPORT")
    print("=" * 65)

    for sentence, expected in cases:
        predicted = bool(real.extract(sentence))

        if predicted and expected:
            tp += 1
        elif predicted and not expected:
            fp += 1
        elif not predicted and expected:
            fn += 1
        else:
            tn += 1

        status = "CORRECT" if predicted == expected else "INCORRECT"
        print(
            f"{status:9} | Expected: {str(expected):5} "
            f"| Predicted: {str(predicted):5} | {sentence[:65]}"
        )

    total = tp + fp + fn + tn

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) else 0.0
    )

    print("\n" + "-" * 65)
    print(f"Total test samples       : {total}")
    print(f"Correct predictions     : {tp + tn}")
    print(f"Incorrect predictions   : {fp + fn}")
    print(f"True Positives (TP)      : {tp}")
    print(f"False Positives (FP)     : {fp}")
    print(f"False Negatives (FN)     : {fn}")
    print(f"True Negatives (TN)      : {tn}")
    print("-" * 65)
    print(f"Accuracy                 : {accuracy:.2%}")
    print(f"Precision                : {precision:.2%}")
    print(f"Recall                   : {recall:.2%}")
    print(f"F1-score                 : {f1:.2%}")

    print("\nConfusion Matrix")
    print("                       Predicted")
    print("                    Positive  Negative")
    print(f"Actual Positive     {tp:8d}  {fn:8d}")
    print(f"Actual Negative     {fp:8d}  {tn:8d}")
    print("=" * 65)

    # Keep the original minimum-quality checks.
    assert recall >= 0.75, "Too many real findings missed"
    assert fp == 0 or precision >= 0.75, (
        "Too many non-findings accepted"
    )
