import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from jerqazyna.features.cache import Digests
from jerqazyna.validation.protocol import BlindProtocol, load_protocol, verify_frozen_inputs

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/protocols/blind_balkhash_2026-11-05.yaml"


def raw_protocol():
    return yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))


def test_registered_protocol_is_valid_and_hashed():
    protocol, digest = load_protocol(PROTOCOL)
    assert protocol.objects.n_positive == 17
    assert [h.id for h in protocol.hypotheses if h.role == "primary"] == ["H1_random"]
    assert len(digest) == 64


@pytest.mark.parametrize("change", [
    lambda d: d["hypotheses"][1].update(role="primary"),
    lambda d: d["hypotheses"][2].update(test="exact_binomial_one_sided", alpha=0.05),
    lambda d: d["hypotheses"][0].pop("null_capture"),
    lambda d: d["evaluation"].update(descriptive_area_fractions=[0.05, 0.20]),
    lambda d: d.update(unexpected=True),
])
def test_ambiguous_protocols_are_rejected(change):
    data = raw_protocol()
    change(data)
    with pytest.raises(ValidationError):
        BlindProtocol.model_validate(data)


def frozen_layout(tmp_path, protocol, *, holdout=False, statuses=("deposit",) * 8 + ("prospect",) * 9):
    split = tmp_path / "split"
    split.mkdir()
    (split / "split.json").write_text(json.dumps(
        {"split_id": protocol.split_id, "provisional": protocol.region_provisional}), encoding="utf-8")
    rows = ["name,status,partition"] + [f"t{i},{s},test" for i, s in enumerate(statuses)]
    rows += ["u,,test", "a,deposit,train"]
    (split / "labels.csv").write_text("\n".join(rows), encoding="utf-8")
    run = tmp_path / protocol.frozen_model.run_dir
    run.mkdir(parents=True)
    (run / "model.joblib").write_bytes(b"frozen model")
    digest = Digests().file(run / "model.joblib")
    (run / "run.json").write_text(json.dumps({"run_id": protocol.frozen_model.run_id,
                                              "holdout_evaluated": holdout,
                                              "files": {"model.joblib": digest}}), encoding="utf-8")
    (run / "cv.json").write_text(json.dumps({"selected_classifier": protocol.frozen_model.classifier}),
                                 encoding="utf-8")
    return split, digest


def test_frozen_inputs_match_metadata_only(tmp_path):
    protocol, _ = load_protocol(PROTOCOL)
    split, digest = frozen_layout(tmp_path, protocol)
    protocol = protocol.model_copy(update={
        "frozen_model": protocol.frozen_model.model_copy(update={"model_sha256": digest})})
    report = verify_frozen_inputs(protocol, tmp_path, split)
    assert report["n_positive"] == 17
    assert report["n_other_status"] == 1


@pytest.mark.parametrize("case, message", [
    ("model", "Хеш модели"),
    ("holdout", "уже оценивалась"),
    ("count", "положительных объектов"),
])
def test_frozen_inputs_reject_drift(tmp_path, case, message):
    protocol, _ = load_protocol(PROTOCOL)
    statuses = ("deposit",) * 8 + ("prospect",) * (8 if case == "count" else 9)
    split, digest = frozen_layout(tmp_path, protocol, holdout=case == "holdout", statuses=statuses)
    if case != "model":
        protocol = protocol.model_copy(update={
            "frozen_model": protocol.frozen_model.model_copy(update={"model_sha256": digest})})
    with pytest.raises(ValueError, match=message):
        verify_frozen_inputs(protocol, tmp_path, split)
