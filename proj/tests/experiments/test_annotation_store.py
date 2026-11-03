import pytest
from experiments.lib.annotation_store import AnnotationStore, AnnotationRecord


def make_record(pair_id="p1"):
    return AnnotationRecord(pair_id=pair_id, paper_a="A", paper_b="B",
                             claim_a="X improves Y", claim_b="X does not improve Y")


def test_system_prediction_never_becomes_ground_truth_automatically():
    r = make_record()
    r.system_prediction = "supports"
    assert r.human_label is None
    assert not r.is_annotated()


def test_annotate_requires_explicit_call():
    r = make_record()
    r.annotate("contradicts", annotator="alice", notes="clear negation")
    assert r.human_label == "contradicts"
    assert r.annotator == "alice"
    assert r.is_annotated()


def test_annotate_rejects_invalid_label():
    r = make_record()
    with pytest.raises(ValueError):
        r.annotate("definitely_true", annotator="alice")


def test_store_round_trips_through_disk(tmp_path):
    path = tmp_path / "annotations.json"
    store = AnnotationStore(path)
    r = make_record()
    r.annotate("contradicts", annotator="alice")
    store.add(r)
    store.save()

    reloaded = AnnotationStore(path)
    assert len(reloaded.all()) == 1
    assert reloaded.get("p1").human_label == "contradicts"


def test_duplicate_pair_id_rejected_without_overwrite(tmp_path):
    store = AnnotationStore(tmp_path / "a.json")
    store.add(make_record("dup"))
    with pytest.raises(ValueError):
        store.add(make_record("dup"))
    store.add(make_record("dup"), overwrite=True)  # should not raise


def test_to_metric_records_requires_system_prediction(tmp_path):
    store = AnnotationStore(tmp_path / "a.json")
    r = make_record()
    r.annotate("supports", annotator="alice")
    store.add(r)
    with pytest.raises(ValueError):
        store.to_metric_records()

    r.system_prediction = "supports"
    assert store.to_metric_records() == [{"system_prediction": "supports", "human_label": "supports"}]


def test_unannotated_records_excluded_from_metric_records(tmp_path):
    store = AnnotationStore(tmp_path / "a.json")
    annotated = make_record("done")
    annotated.annotate("neutral", annotator="alice")
    annotated.system_prediction = "neutral"
    store.add(annotated)
    store.add(make_record("pending"))
    assert len(store.annotated()) == 1
    assert len(store.unannotated()) == 1
    assert len(store.to_metric_records()) == 1
