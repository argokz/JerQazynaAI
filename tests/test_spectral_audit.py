import json
from types import SimpleNamespace

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("rasterio")

from jerqazyna.features.cache import Digests
from jerqazyna.grid.template import GridSpec, write_layer
from jerqazyna.validation import spectral_audit as audit


def test_zero_is_observed_and_missing_and_mine_are_separate():
    values = np.array([[0., np.nan, 2., np.nan, 4.]])
    train = np.array([[True, True, True, True, False]])
    states = audit.quality_states(np.isfinite(values), np.array([[0, 0, 1, 1, 0]]), train)
    np.testing.assert_equal(states, [[4., 1., 2., 3., np.nan]])
    result = audit.counts(states, train)
    assert result["cells"] == 4
    assert result["observed_fraction"] == .5
    assert result["usable_fraction"] == .25
    assert sum(result[n + "_cells"] for n in audit.REASONS.values()) == 4


@pytest.mark.parametrize("value", [np.nan, .5, -1, 2])
def test_missing_or_nonbinary_mine_mask_rejected(value):
    with pytest.raises(ValueError, match="mine mask"):
        audit.quality_states(np.array([[True]]), np.array([[value]]), np.array([[True]]))


def test_bias_quartiles_ignore_closed_cells_and_missing():
    train = np.array([[True, True, True, True, False, True]])
    density = np.array([[0., 1., 2., 3., 1e9, np.nan]])
    strata, cuts = audit.bias_strata(density, train)
    assert cuts == [.75, 1.5, 2.25]
    np.testing.assert_equal(strata, [[0, 1, 2, 3, -1, -1]])
    density[0, 4] = -1e9
    np.testing.assert_equal(audit.bias_strata(density, train)[0], strata)


def test_tied_density_values_stay_together_and_empty_bins_remain_empty():
    density = np.array([[0., 0., 0., 1.]])
    train = np.ones(density.shape, bool)
    strata, cuts = audit.bias_strata(density, train)
    assert cuts == [0., 0., .25]
    np.testing.assert_equal(strata, [[2, 2, 2, 3]])
    empty = audit.counts(np.full(density.shape, 4.), strata == 0)
    assert empty["cells"] == 0 and empty["usable_fraction"] is None


def test_geometry_check_and_nodata_conversion(tmp_path):
    grid = GridSpec("EPSG:4326", 1., 0., 2., 2, 2)
    path = tmp_path / "input.tif"
    write_layer(path, grid, np.array([[0., 1.], [np.nan, 2.]], dtype=np.float32))
    np.testing.assert_equal(audit.aligned(path, grid), [[0., 1.], [np.nan, 2.]])
    with pytest.raises(ValueError, match="geometry"):
        audit.aligned(path, GridSpec("EPSG:4326", 1., .01, 2., 2, 2))


def test_legacy_inventory_does_not_claim_derivation_and_detects_registered_tamper(tmp_path):
    (tmp_path / "layer__band.tif").write_bytes(b"original")
    d = Digests()
    item = audit.input_inventory(tmp_path, ["layer__band.tif"], {}, d)["layer__band.tif"]
    assert item["provenance"] == "legacy_unregistered"
    assert not item["derivation_verified"]
    cache = {"layer": {"files": {"layer__band.tif": item["sha256"]}}}
    (tmp_path / "layer__band.tif").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="Changed registered"):
        audit.input_inventory(tmp_path, ["layer__band.tif"], cache, Digests())


def test_unknown_labels_preserved_without_becoming_positives():
    labels = pd.DataFrame({"fold": [0, 0], "country_iso_geometry": ["KAZ"] * 2,
                           "usgs_tracts_diagnostic": ["tract"] * 2,
                           "status": ["prospect", "unknown"], "baseline_positive": [True, False],
                           "state": [4, 4]})
    result = audit.label_groups(labels, "state")
    unknown = next(v for v in result["status"] if v["group"] == "unknown")
    assert unknown["usable_labels"] == 1
    assert unknown["usable_positive"] == 0
    assert labels.status.tolist() == ["prospect", "unknown"]


def test_real_raster_audit_is_isolated_idempotent_and_keeps_inputs(tmp_path, monkeypatch):
    grid = GridSpec("EPSG:4326", 1., 0., 2., 5, 2)
    base = tmp_path / "data/features/test"
    split = tmp_path / "split"
    countries = tmp_path / "countries"
    baseline = tmp_path / "baseline"
    for directory in (base, split, countries, baseline):
        directory.mkdir(parents=True)
    (base / "cache.json").write_text('{"layers":{}}')
    (base / "layers.json").write_text('[]')
    (countries / "summary.json").write_text('{"strata":{"1":{"name":"KAZ"}}}')
    (baseline / "run.json").write_text('{"spec":{"experiment":{"n_folds":5}}}')
    train = np.array([[1.] * 5, [0.] * 5], dtype=np.float32)
    write_layer(split / "train.tif", grid, train)
    write_layer(split / "test.tif", grid, 1 - train)
    write_layer(split / "buffer.tif", grid, train * 0)
    write_layer(countries / "train_countries.tif", grid, np.where(train > 0, 1., np.nan))
    write_layer(baseline / "cv_fold.tif", grid, np.array([range(5), range(5)], np.float32))
    for layer, bands in audit.CHANNELS.items():
        for band in (*bands, "coverage"):
            write_layer(base / f"{layer}__{band}.tif", grid, np.ones(grid.shape, np.float32))
    write_layer(base / "mine_mask__mask.tif", grid, np.array([[0, 1, 0, 0, 0], [0] * 5], np.float32))
    write_layer(base / "exploration_density__per_1000km2.tif", grid, np.arange(10, dtype=np.float32).reshape(grid.shape))
    labels = pd.DataFrame({"label_id": ["a", "b"], "name": ["positive", "unresolved"],
                           "status": ["prospect", "unknown"], "row": [0, 0], "col": [0, 1],
                           "fold": [0, 1], "block": [1, -1], "baseline_positive": [True, False],
                           "country_iso_geometry": ["KAZ"] * 2, "country_issue": ["consistent"] * 2,
                           "usgs_tracts_diagnostic": ["tract"] * 2})
    pilot = SimpleNamespace(task="task.yaml", split="split", baseline="baseline")
    monkeypatch.setattr(audit, "train_context", lambda *a: (pilot, labels.copy(), grid, {}))
    monkeypatch.setattr(audit, "load_task", lambda *a: SimpleNamespace(task_id="test"))
    monkeypatch.setattr(audit, "render", lambda *a: None)
    config = audit.SpectralAuditConfig(assessment_id="test", pilot="pilot", country_audit="countries")
    # This snapshot covers all fixture sources, before any diagnostic output exists.
    before = {p: Digests().file(p) for p in tmp_path.rglob("*") if p.is_file()}
    output = audit.audit(config, tmp_path)
    summary = json.loads((output / "summary.json").read_text())
    assert not summary["holdout_evaluated"] and not summary["models_trained"]
    assert not summary["ready_for_training"]
    assert summary["layers"]["joint"]["usable_cells"] == 4
    for layer in ("sentinel2", "aster_alteration", "joint"):
        states = audit.aligned(output / f"{layer}__state.tif", grid)
        assert np.isnan(states[1]).all()
        assert states[0, 1] == 2
    assert all(Digests().file(p) == sha for p, sha in before.items())
    manifest_before = (output / "manifest.json").read_bytes()
    assert audit.audit(config, tmp_path) == output
    assert manifest_before == (output / "manifest.json").read_bytes()
    (output / "labels_coverage.csv").write_text("corrupted")
    with pytest.raises(ValueError, match="Changed assessment input"):
        audit.audit(config, tmp_path)
