"""Экспорт COG на синтетических растрах: формат, точность, манифест, пропуск без изменений."""

import json

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

from jerqazyna_web.export import ExportError, export
from jerqazyna_web.registry import load_registry

from jerqazyna.data.download import sha256_of
from jerqazyna.grid.template import write_layer

BLOCK = 128


def _export(project, catalog, **kwargs):
    registry = load_registry(project["registry"], catalog)
    return export(registry, project["root"], project["out"], blocksize=BLOCK,
                  log=lambda *_: None, **kwargs)


def test_export_writes_lossless_cogs_and_manifest(project, catalog):
    manifest = _export(project, catalog)
    out = project["out"]
    assert set(manifest.layers) == {
        "dem_elevation", "magnetics_anomaly", "faults_active_distance", "research_score"}
    dem = manifest.layers["dem_elevation"]
    path = out / dem.file
    assert dem.file == "cogs/dem_elevation.tif"
    with rasterio.open(path) as cog:
        assert cog.tags(ns="IMAGE_STRUCTURE")["LAYOUT"] == "COG"
        assert cog.overviews(1)
        assert cog.shape == project["grid"].shape
        np.testing.assert_array_equal(cog.read(1), project["arrays"]["dem__elevation_m.tif"])
    assert dem.cog_sha256 == sha256_of(path)
    assert dem.source_sha256 == sha256_of(
        project["root"] / "data/features/test_task/dem__elevation_m.tif")
    assert dem.provenance.recorded_sha256 == dem.source_sha256
    assert dem.provenance.details == {"layer": "dem", "recipe": "r-dem"}
    # Одна NaN-ячейка внутри области: покрытие чуть меньше 1.
    assert 0.99 < dem.coverage < 1.0
    assert manifest.layers["research_score"].provenance.details == {"run_id": "run-1"}
    assert manifest.layers["faults_active_distance"].provenance is None
    # Маска области лежит вне cogs/: TiTiler её не видит.
    assert manifest.area.file == "area_mask.tif"
    saved = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert saved["grid"] == project["grid_dict"]
    assert saved["layers"]["dem_elevation"]["cog_sha256"] == dem.cog_sha256


def test_rerun_skips_unchanged_and_rebuilds_changed(project, catalog):
    first = _export(project, catalog)
    cog = project["out"] / first.layers["magnetics_anomaly"].file
    mtime = cog.stat().st_mtime_ns
    second = _export(project, catalog)
    assert cog.stat().st_mtime_ns == mtime
    assert second.layers["magnetics_anomaly"] == first.layers["magnetics_anomaly"]

    # Слой без манифеста происхождения: новое содержимое -> новый COG.
    faults = project["root"] / "data/features/test_task/faults_active__distance_km.tif"
    write_layer(faults, project["grid"], np.zeros(project["grid"].shape, np.float32))
    third = _export(project, catalog, only=["faults_active_distance"])
    assert third.layers["faults_active_distance"].source_sha256 == sha256_of(faults)
    assert (third.layers["faults_active_distance"].cog_sha256
            != first.layers["faults_active_distance"].cog_sha256)
    # Прочие слои в манифесте сохраняются при частичном экспорте.
    assert third.layers["dem_elevation"] == first.layers["dem_elevation"]


def test_damaged_cog_is_rewritten(project, catalog):
    first = _export(project, catalog, only=["dem_elevation"])
    cog = project["out"] / first.layers["dem_elevation"].file
    cog.write_bytes(b"broken")
    second = _export(project, catalog, only=["dem_elevation"])
    assert sha256_of(cog) == second.layers["dem_elevation"].cog_sha256


def test_source_changed_after_preparation_is_rejected(project, catalog):
    dem = project["root"] / "data/features/test_task/dem__elevation_m.tif"
    write_layer(dem, project["grid"], np.zeros(project["grid"].shape, np.float32))
    with pytest.raises(ExportError, match="не совпадает с хешем"):
        _export(project, catalog, only=["dem_elevation"])


def test_raster_off_grid_is_rejected(project, catalog):
    from jerqazyna.grid.template import GridSpec

    grid = project["grid"]
    other = GridSpec(grid.crs, grid.cell_size, grid.x_min + 1000, grid.y_max,
                     grid.width, grid.height)
    faults = project["root"] / "data/features/test_task/faults_active__distance_km.tif"
    write_layer(faults, other, np.zeros(other.shape, np.float32))
    with pytest.raises(ExportError, match="не на сетке"):
        _export(project, catalog, only=["faults_active_distance"])


def test_unknown_layer_and_foreign_manifest(project, catalog):
    with pytest.raises(ExportError, match="Нет включённых слоёв"):
        _export(project, catalog, only=["nope"])
    _export(project, catalog, only=["dem_elevation"])
    path = project["out"] / "manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["grid"]["width"] += 1
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ExportError, match="другой задачи или сетки"):
        _export(project, catalog, only=["dem_elevation"])
