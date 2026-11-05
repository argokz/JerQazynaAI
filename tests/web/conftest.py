"""Синтетический проект для тестов веб-карты: маленькая сетка, растры и манифесты в tmp_path.

Реальные данные не нужны; пакет web/api/jerqazyna_web подключается через sys.path.
"""

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import yaml

import jerqazyna  # noqa: F401  (PROJ/GDAL до rasterio)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "web" / "api"))

from web_helpers import (
    AREA_COLS,
    CRS,
    FEATURES,
    HEIGHT,
    ROOT,
    WIDTH,
    X_MIN,
    Y_MAX,
    registry_dict,
)


@pytest.fixture
def project(tmp_path):
    """Корень проекта с растрами на сетке и манифестами cache.json, area.json, run.json."""
    pytest.importorskip("rasterio")
    from jerqazyna.data.download import sha256_of
    from jerqazyna.grid.template import GridSpec, write_layer

    root = tmp_path / "project"
    features = root / FEATURES
    grid = GridSpec(CRS, 1000.0, X_MIN, Y_MAX, WIDTH, HEIGHT)
    features.mkdir(parents=True)
    grid.save(features / "grid.json")
    rows, cols = np.mgrid[0:HEIGHT, 0:WIDTH]
    area = np.ones(grid.shape, np.uint8)
    area[:, :AREA_COLS] = 0
    dem = (rows * 10 + cols).astype(np.float32)
    dem[5, 100] = np.nan
    arrays = {
        "dem__elevation_m.tif": dem,
        "magnetics__anomaly_nt.tif": (cols - rows).astype(np.float32),
        "faults_active__distance_km.tif": (rows / 10).astype(np.float32),
    }
    write_layer(features / "area_mask.tif", grid, area)
    for name, array in arrays.items():
        write_layer(features / name, grid, array)
    cache = {"version": 1, "layers": {
        "dem": {"recipe": "r-dem", "files": {
            "dem__elevation_m.tif": sha256_of(features / "dem__elevation_m.tif")}},
        "magnetics": {"recipe": "r-mag", "files": {
            "magnetics__anomaly_nt.tif": sha256_of(features / "magnetics__anomaly_nt.tif")}},
    }}
    (features / "cache.json").write_text(json.dumps(cache), encoding="utf-8")
    (features / "area.json").write_text(
        json.dumps({"mask_sha256": sha256_of(features / "area_mask.tif")}), encoding="utf-8")
    run = root / "outputs" / "run"
    score = np.full(grid.shape, np.nan, np.float32)
    score[100:, 50:] = 0.25
    write_layer(run / "train_score.tif", grid, score)
    (run / "run.json").write_text(json.dumps({
        "run_id": "run-1", "files": {"train_score.tif": sha256_of(run / "train_score.tif")},
    }), encoding="utf-8")
    registry = root / "web" / "layers.yaml"
    registry.parent.mkdir()
    registry.write_text(
        yaml.safe_dump(registry_dict(FEATURES), allow_unicode=True), encoding="utf-8")
    return {"root": root, "registry": registry, "grid": grid, "arrays": arrays, "area": area,
            "score": score, "out": root / "data" / "web", "grid_dict": asdict(grid)}


@pytest.fixture
def catalog():
    from jerqazyna.data.catalog import load_catalog

    return load_catalog(ROOT / "configs" / "sources.yaml")
