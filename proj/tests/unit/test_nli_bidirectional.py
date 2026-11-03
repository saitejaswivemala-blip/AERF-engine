from nlp.nli.nli_classifier import NLIClassifier, NLIResult


def test_bidirectional_entailment_keeps_directional_support():
    model = NLIClassifier(classifier=object())
    calls = []

    def fake_predict(a, b):
        calls.append((a, b))
        if len(calls) == 1:
            return NLIResult("entailment", .82, .05, .13, .82, "fake")
        return NLIResult("neutral", .12, .04, .84, .84, "fake")

    model.predict = fake_predict
    result = model.predict_bidirectional("A", "B")
    assert result.label == "entailment"
    assert result.entailment == .82


def test_bidirectional_contradiction_is_averaged():
    model = NLIClassifier(classifier=object())
    calls = []

    def fake_predict(a, b):
        calls.append((a, b))
        if len(calls) == 1:
            return NLIResult("contradiction", .05, .80, .15, .80, "fake")
        return NLIResult("contradiction", .10, .70, .20, .70, "fake")

    model.predict = fake_predict
    result = model.predict_bidirectional("A", "B")
    assert result.label == "contradiction"
    assert abs(result.contradiction - .75) < 1e-9
