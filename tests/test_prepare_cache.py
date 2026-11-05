import json
import os

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
gpd = pytest.importorskip("geopandas")

from rasterio.transform import from_origin
from shapely.geometry import box

from jerqazyna.cli import main
from jerqazyna.config import CENTRAL_ASIA_ALBERS, LayerConfig, TaskConfig
from jerqazyna.features.layers import PROCESSORS, SourceMissing
from jerqazyna.features.prepare import prepare
from jerqazyna.grid.template import read_layer

CRS = CENTRAL_ASIA_ALBERS


@pytest.fixture
def project(tmp_path):
    boundaries = tmp_path / "data/raw/geoboundaries"
    boundaries.mkdir(parents=True)
    country = gpd.GeoDataFrame(geometry=[box(0, 0, 4000, 4000)], crs=CRS).to_crs(4326)
    country.to_file(boundaries / "geoBoundaries-AAA-ADM0.geojson", driver="GeoJSON")
    country.to_file(tmp_path / "blind.geojson", driver="GeoJSON")
    labels = tmp_path / "labels.gpkg"
    gpd.GeoDataFrame(
        {"Name": ["A"], "SiteStatus": ["deposit"]},
        geometry=gpd.points_from_xy([1500], [1500]), crs=CRS,
    ).to_file(labels, driver="GPKG")
    source = tmp_path / "raw"
    source.mkdir()
    with rasterio.open(
        source / "value.tif", "w", driver="GTiff", width=4, height=4, count=1,
        dtype="float32", crs=CRS, transform=from_origin(0, 4000, 1000, 1000),
    ) as dst:
        dst.write(np.ones((4, 4), np.float32), 1)
    task = TaskConfig.model_validate({
        "task_id": "toy", "deposit_type": "toy", "training_countries": ["AAA"],
        "labels": {"source": "labels.gpkg"},
        "layers": [{"name": "value", "kind": "raster", "source": "raw", "group": "terrain"}],
        "validation": {"blind_region": {"name": "x", "geometry": "blind.geojson"}},
    })
    return tmp_path, task, tmp_path / "data/features/toy"


def run(project, **kwargs):
    root, task, _ = project
    return prepare(task, root, log=lambda *_: None, **kwargs)


@pytest.mark.parametrize("force", [False, True])
def test_grid_change_never_reuses_or_overwrites_old_grid(project, force):
    _, task, out = project
    run(project)
    original = (out / "grid.json").read_bytes()
    task.grid.cell_size_m = 2000
    with pytest.raises(ValueError, match="новый task_id"):
        run(project, force=force)
    assert (out / "grid.json").read_bytes() == original


def test_input_content_changed_with_same_size_and_mtime_recomputes(project):
    root, _, out = project
    run(project)
    assert run(project)[0]["status"] == "готово ранее"
    path = root / "raw/value.tif"
    stat = path.stat()
    with rasterio.open(path, "r+") as src:
        src.write(np.full((4, 4), 2, np.float32), 1)
    assert path.stat().st_size == stat.st_size
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert run(project)[0]["status"] == "готово"
    assert np.nanmax(read_layer(out / "value__value.tif")) == 2


def test_layer_config_and_output_integrity_are_checked(project):
    _, task, out = project
    run(project)
    task.layers[0].role = "benchmark"
    # Проверяем изменение метаданных уже существующего канала.
    assert run(project)[0]["status"] == "готово"
    inventory = json.loads((out / "layers.json").read_text())
    assert inventory[0]["role"] == "benchmark"
    (out / "value__value.tif").write_bytes(b"truncated")
    assert run(project)[0]["status"] == "готово"
    assert run(project)[0]["status"] == "готово ранее"


def test_missing_band_and_partial_file_are_not_complete(project, monkeypatch):
    monkeypatch.setitem(PROCESSORS, "raster", lambda layer, ctx: {
        "a": np.ones(ctx.grid.shape, np.float32), "b": np.zeros(ctx.grid.shape, np.float32),
    })
    _, _, out = project
    run(project)
    (out / "value__b.tif").unlink()
    (out / "value__b.part.tif").write_bytes(b"partial")
    assert run(project)[0]["status"] == "готово"
    assert (out / "value__b.tif").exists()
    assert not (out / "value__b.part.tif").exists()


def test_partial_selection_revokes_changed_unselected_source(project):
    root, task, out = project
    (root / "other").mkdir()
    (root / "other/value.tif").write_bytes((root / "raw/value.tif").read_bytes())
    task.layers.append(LayerConfig(name="other", kind="raster", source="other", group="terrain"))
    run(project)
    with rasterio.open(root / "other/value.tif", "r+") as src:
        src.write(np.full((4, 4), 3, np.float32), 1)
    assert run(project, only=["value"])[0]["status"] == "готово ранее"
    inventory = json.loads((out / "layers.json").read_text())
    assert {r["layer"] for r in inventory} == {"value"}


def test_missing_primary_labels_revokes_old_labels(project):
    root, _, out = project
    run(project)
    (root / "labels.gpkg").unlink()
    with pytest.raises(SourceMissing, match="Обязательные метки"):
        run(project)
    assert not (out / "labels.csv").exists()


def test_unknown_layer_selection_fails_before_writing(project):
    _, _, out = project
    with pytest.raises(ValueError, match="Неизвестные слои"):
        run(project, only=["typo"])
    assert not out.exists()


def test_boundary_change_updates_mask_and_invalidates_layers(project):
    root, _, out = project
    run(project)
    old = read_layer(out / "area_mask.tif").sum()
    # Сохраняем bounds, меняем форму: сетка та же, область меньше.
    shape = box(0, 0, 4000, 4000).difference(box(1000, 1000, 3000, 3000))
    gpd.GeoDataFrame(geometry=[shape], crs=CRS).to_crs(4326).to_file(
        root / "data/raw/geoboundaries/geoBoundaries-AAA-ADM0.geojson", driver="GeoJSON",
    )
    assert run(project)[0]["status"] == "готово"
    assert read_layer(out / "area_mask.tif").sum() < old


def test_only_gravity_validates_dem_first_and_dem_change_revokes_gravity(project, monkeypatch):
    _, task, out = project
    task.layers = [
        LayerConfig(name="gravity", kind="ggmplus_bouguer", source="raw", group="geophysics"),
        LayerConfig(name="dem", kind="dem", source="raw", group="terrain"),
    ]
    monkeypatch.setitem(PROCESSORS, "dem", lambda layer, ctx: {
        "elevation_m": np.ones(ctx.grid.shape, np.float32),
    })
    monkeypatch.setitem(PROCESSORS, "ggmplus_bouguer", lambda layer, ctx: {
        "value": np.ones(ctx.grid.shape, np.float32),
    })
    report = run(project, only=["gravity"])
    assert [r["layer"] for r in report] == ["dem", "gravity"]
    assert all(r["status"] == "готово ранее" for r in run(project, only=["gravity"]))
    run(project, only=["dem"], force=True)
    inventory = json.loads((out / "layers.json").read_text())
    assert {r["layer"] for r in inventory} == {"dem"}
    assert run(project, only=["gravity"])[-1]["status"] == "готово"


def test_failed_processor_cannot_leave_ready_marker(project, monkeypatch):
    _, _, out = project
    run(project)

    def fail(layer, ctx):
        raise RuntimeError("simulated crash")

    monkeypatch.setitem(PROCESSORS, "raster", fail)
    with pytest.raises(RuntimeError, match="simulated crash"):
        run(project, force=True)
    assert json.loads((out / "cache.json").read_text())["layers"] == {}
    assert json.loads((out / "layers.json").read_text()) == []


@pytest.mark.parametrize("required,expected", [(True, 1), (False, 0)])
def test_cli_missing_layer_exit_code(project, required, expected):
    root, task, _ = project
    task.layers.append(LayerConfig(name="absent", kind="raster", source="missing",
                                   group="terrain", required=required))
    config = root / "task.json"
    config.write_text(task.model_dump_json(), encoding="utf-8")  # JSON — подмножество YAML
    assert main(["prepare", str(config), "--root", str(root)]) == expected


def test_check_inputs_detects_missing_blind_geometry(project):
    root, task, _ = project
    config = root / "task.json"
    config.write_text(task.model_dump_json(), encoding="utf-8")
    assert main(["validate-config", str(config), "--root", str(root), "--check-inputs"]) == 0
    (root / "blind.geojson").unlink()
    assert main(["validate-config", str(config), "--root", str(root), "--check-inputs"]) == 1


def test_split_generation_is_repeatable_and_checks_artifacts(project):
    from jerqazyna.validation.holdout import prepare_split

    root, task, _ = project
    task.validation.blind_region.buffer_m = 100
    task.validation.blind_region.provisional = True
    gpd.GeoDataFrame(geometry=[box(2000, 0, 4000, 4000)], crs=CRS).to_crs(4326).to_file(
        root / "blind.geojson", driver="GeoJSON",
    )
    dest = prepare_split(task, root, log=lambda *_: None)
    assert prepare_split(task, root, log=lambda *_: None) == dest
    metadata = json.loads((dest / "split.json").read_text())
    assert metadata["provisional"]
    train, test = read_layer(dest / "train.tif"), read_layer(dest / "test.tif")
    assert not (train.astype(bool) & test.astype(bool)).any()
    (dest / "test.tif").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Повреждённое разбиение"):
        prepare_split(task, root, log=lambda *_: None)
