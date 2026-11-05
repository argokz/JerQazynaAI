import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

pd = pytest.importorskip("pandas")

from jerqazyna.data import review
from jerqazyna.features.cache import Digests


@pytest.fixture
def registry(tmp_path, monkeypatch):
    source = tmp_path / "source.csv"
    source.write_text("name\nobject\n", encoding="utf-8")
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "run.json").write_text("{}", encoding="utf-8")
    records = [{"source": "source.csv", "layer": None, "row": "0"}]
    labels = pd.DataFrame([{"name": "object", "status": "unknown", "country": None,
                            "source_records": json.dumps(records), "partition": "train"}])
    monkeypatch.setattr(review, "parent_labels", lambda *args: (labels.copy(deep=True), {"dataset_id": "parent"}, {"run_id": "run"}))
    sha = Digests().file(source)
    decision = {"decision_id": "country", "record": records[0] | {"sha256": sha},
                "state": "confirmed", "before": {"country": None}, "changes": {"country": "Kazakhstan"},
                "reason": "Derived from geometry; does not verify coordinates", "evidence": [
                    {"uri": "local:source.csv", "locator": "data row 0", "finding": "fixture", "sha256": sha}]}
    return tmp_path, baseline, labels, decision


def execute(registry, decisions):
    root, baseline, _, _ = registry
    config = review.ReviewConfig(review_id="check", decisions=decisions)
    return review.run_review(SimpleNamespace(task_id="toy"), config, root, baseline)


def test_review_immutable_parent_pending_and_idempotence(registry):
    root, _, labels, decision = registry
    before = Digests().file(root / "source.csv")
    pending = deepcopy(decision)
    pending.update(decision_id="status", state="pending", before={"status": "unknown"}, changes={"status": "deposit"})
    rejected = deepcopy(pending)
    rejected.update(decision_id="rejected", state="rejected")
    path = execute(registry, [decision, pending, rejected])
    table = pd.read_csv(path / "train_labels_curated.csv")
    assert table.at[0, "country"] == "Kazakhstan"
    assert table.at[0, "status"] == labels.at[0, "status"] == "unknown"
    assert len(pd.read_csv(path / "diff.csv")) == 1
    assert Digests().file(root / "source.csv") == before
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["spec"]["parent_dataset_id"] == "parent"
    assert not manifest["training_pipeline_changed"]
    assert execute(registry, [decision, pending, rejected]) == path
    (path / "diff.csv").write_text("corrupt")
    with pytest.raises(ValueError, match="Повреждено"):
        execute(registry, [decision, pending, rejected])


def test_review_stale_source_hash(registry):
    root, _, _, decision = registry
    (root / "source.csv").write_text("changed")
    with pytest.raises(ValueError, match="SHA-256"):
        execute(registry, [decision])


def test_review_ambiguous_match(registry):
    _, _, labels, decision = registry
    labels.loc[1] = labels.loc[0]
    with pytest.raises(ValueError, match="Неоднозначное"):
        execute(registry, [decision])


def test_review_duplicate_edit(registry):
    decision = registry[3]
    second = deepcopy(decision)
    second["decision_id"] = "second"
    with pytest.raises(ValueError, match="Повторная"):
        execute(registry, [decision, second])


def test_review_wrong_before_and_evidence(registry):
    decision = deepcopy(registry[3])
    decision["before"]["country"] = "Uzbekistan"
    with pytest.raises(ValueError, match="before"):
        execute(registry, [decision])
    decision = deepcopy(registry[3])
    decision["evidence"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="доказательства"):
        execute(registry, [decision])


def test_review_rejects_unpaired_coordinate(registry):
    decision = deepcopy(registry[3])
    decision.update(before={"lon": 70}, changes={"lon": 71})
    with pytest.raises(ValueError, match="парой"):
        review.Decision.model_validate(decision)


def test_blank_pending_status_preserved_in_derived_table(registry):
    decision = deepcopy(registry[3])
    registry[2].loc[0, "status"] = None
    decision.update(state="pending", before={"status": "unknown"}, changes={"status": "deposit"})
    result = execute(registry, [decision])
    assert pd.isna(pd.read_csv(result / "train_labels_curated.csv").at[0, "status"])
    assert pd.read_csv(result / "diff.csv").empty


def test_coordinate_move_into_closed_cell_rejected(registry, monkeypatch):
    import numpy as np

    from jerqazyna.features.cache import write_json
    from jerqazyna.grid.template import GridSpec, write_layer

    root, _, labels, decision = registry
    labels["lon"], labels["lat"] = 66.2, 47.8
    grid = GridSpec("EPSG:4326", 1, 66, 48, 2, 2)
    split = root / "data/features/toy/splits/0123456789abcdef"
    split.mkdir(parents=True)
    write_layer(split / "train.tif", grid, np.array([[1, 0], [0, 0]], np.uint8))
    write_json(split / "split.json", {"files": {"train.tif": Digests().file(split / "train.tif")}})
    dataset = {"dataset_id": "parent", "spec": {"grid": grid.__dict__, "split_id": "0123456789abcdef",
                "split_manifest_sha256": Digests().file(split / "split.json")}}
    monkeypatch.setattr(review, "parent_labels", lambda *args: (labels.copy(deep=True), dataset, {"run_id": "run"}))
    decision = deepcopy(decision)
    decision.update(before={"lon": 66.2, "lat": 47.8}, changes={"lon": 67.2, "lat": 47.8})
    with pytest.raises(ValueError, match="закрытую"):
        execute(registry, [decision])
