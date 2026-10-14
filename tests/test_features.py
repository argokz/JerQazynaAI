import json

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
gpd = pytest.importorskip("geopandas")

from rasterio.transform import from_origin
from shapely.geometry import LineString, box

from jerqazyna.config import CENTRAL_ASIA_ALBERS, TaskConfig
from jerqazyna.features.prepare import prepare
from jerqazyna.features.raster_ops import (
    Accumulator,
    TileArray,
    distance_km,
    point_density,
    polygon_fraction,
)
from jerqazyna.grid.template import GridSpec, read_layer

CRS = CENTRAL_ASIA_ALBERS


@pytest.fixture
def grid():
    return GridSpec.from_bounds(CRS, 1000, (0, 0, 10_000, 10_000))


def test_grid_snaps_to_cell_multiples():
    g = GridSpec.from_bounds(CRS, 1000, (150, -2_300, 9_700, 4_100))
    assert (g.x_min, g.y_max, g.width, g.height) == (0, 5_000, 10, 8)
    assert g.window((1_500, 0, 3_500, 2_000)) == (slice(3, 5), slice(1, 4))
    assert g.window((50_000, 0, 60_000, 1)) is None


def test_accumulator_exact_mean_across_tile_seam(grid):
    # Два тайла по 100 м со стыком посередине ячейки (x = 4500 м): там среднее должно быть 2.
    left = TileArray(np.ones((100, 45), np.float32), from_origin(0, 10_000, 100, 100), CRS)
    right = TileArray(np.full((100, 55), 3, np.float32), from_origin(4_500, 10_000, 100, 100), CRS)
    acc = Accumulator(grid, ("mean", "min", "max"))
    assert acc.add(left) and acc.add(right)
    mean = acc.mean()
    assert mean[5, 0] == pytest.approx(1)
    assert mean[5, 4] == pytest.approx(2)
    assert mean[5, 9] == pytest.approx(3)
    assert acc.vmax[5, 4] == 3 and acc.vmin[5, 4] == 1


def test_accumulator_ignores_nan(grid):
    data = np.full((100, 100), 5, np.float32)
    data[:, :50] = np.nan
    acc = Accumulator(grid)
    acc.add(TileArray(data, from_origin(0, 10_000, 100, 100), CRS))
    mean = acc.mean()
    assert np.isnan(mean[0, 0])
    assert mean[0, 7] == pytest.approx(5)


def test_tile_bounds_order_and_geographic_tile():
    # Узкий тайл далеко от начала координат ловит перепутанный порядок границ.
    tile = TileArray(np.ones((10, 20), np.float32), from_origin(75, 47, 0.05, 0.1), "EPSG:4326")
    assert tile.bounds == pytest.approx((75, 46, 76, 47))
    g = GridSpec.from_bounds(CRS, 1000, (300_000, 0, 500_000, 250_000))
    acc = Accumulator(g)
    assert acc.add(tile)
    assert np.nanmean(acc.mean()) == pytest.approx(1)


def test_polygon_fraction_half_cell(grid):
    frac = polygon_fraction([box(0, 9_000, 500, 10_000)], grid, factor=10, block=3)
    assert frac[0, 0] == pytest.approx(0.5)
    assert frac.sum() == pytest.approx(0.5)


def test_distance_to_line(grid):
    d = distance_km([LineString([(500, 0), (500, 10_000)])], grid)
    assert d[5, 0] == 0
    assert d[5, 5] == pytest.approx(5)


def test_point_density_preserves_count(grid):
    big = GridSpec.from_bounds(CRS, 1000, (0, 0, 200_000, 200_000))
    x = np.array([100_500.0, 100_500.0, 100_500.0])
    y = np.array([100_500.0, 100_500.0, 100_500.0])
    dens = point_density(x, y, big, sigma_km=5)
    assert dens.sum() / 1000 == pytest.approx(3, rel=1e-3)


def _write_raster(path, data, transform, crs="EPSG:4326"):
    with rasterio.open(
        path, "w", driver="GTiff", width=data.shape[1], height=data.shape[0], count=1,
        dtype="float32", crs=crs, transform=transform, nodata=-9999,
    ) as dst:
        dst.write(data.astype("float32"), 1)


def test_prepare_end_to_end(tmp_path):
    root = tmp_path
    # Страна-квадрат 1°×1° у 46° с. ш., 70° в. д.
    gb = root / "data" / "raw" / "geoboundaries"
    gb.mkdir(parents=True)
    gpd.GeoDataFrame(geometry=[box(70, 46, 71, 47)], crs="EPSG:4326").to_file(
        gb / "geoBoundaries-AAA-ADM0.geojson", driver="GeoJSON"
    )
    # Растр: значение = долгота, шаг 0,01°
    rast = root / "data" / "raw" / "toy"
    rast.mkdir(parents=True)
    lon = np.tile(np.linspace(69.505, 71.495, 200), (200, 1))
    _write_raster(rast / "toy_value.tif", lon, from_origin(69.5, 47.5, 0.01, 0.01))
    # Метки
    labels = root / "data" / "raw" / "labels"
    labels.mkdir(parents=True)
    gpd.GeoDataFrame(
        {"Name": ["A", "B"], "SiteStatus": ["deposit", "prospect"]},
        geometry=gpd.points_from_xy([70.5, 75.0], [46.5, 46.5]), crs="EPSG:4326",
    ).to_file(labels / "labels.gpkg", driver="GPKG")

    task = TaskConfig.model_validate({
        "task_id": "toy", "deposit_type": "toy", "training_countries": ["AAA"],
        "grid": {"cell_size_m": 5000},
        "labels": {"source": "data/raw/labels/labels.gpkg"},
        "layers": [
            {"name": "toy", "group": "geophysics", "source": "data/raw/toy", "kind": "raster",
             "params": {"pattern": "*.tif"}},
            {"name": "missing", "group": "geology", "source": "data/raw/nothing", "kind": "raster"},
        ],
        "validation": {"blind_region": {"name": "x", "geometry": "x.geojson"}},
    })
    report = prepare(task, root, log=lambda *_: None)
    out = root / "data" / "features" / "toy"
    assert {r["layer"]: r["status"] for r in report} == {"toy": "готово", "missing": "нет данных"}

    grid = GridSpec.load(out / "grid.json")
    mask = read_layer(out / "area_mask.tif").astype(bool)
    value = read_layer(out / "toy__value.tif")
    assert 0.8 < mask.mean() <= 1
    assert np.nanmin(value[mask]) > 69.9 and np.nanmax(value[mask]) < 71.1

    table = __import__("pandas").read_csv(out / "labels.csv")
    assert table.set_index("name").in_area.to_dict() == {"A": True, "B": False}
    row = table.set_index("name").loc["A"]
    assert 0 <= row.row < grid.height and 0 <= row.col < grid.width

    inventory = json.loads((out / "layers.json").read_text(encoding="utf-8"))
    assert inventory[0]["file"] == "toy__value.tif" and inventory[0]["role"] == "feature"

    # Повторный запуск не пересчитывает готовое
    again = prepare(task, root, log=lambda *_: None)
    assert again[0]["status"] == "готово ранее"


def test_ggmplus_tile_orientation(tmp_path):
    from jerqazyna.features.layers import GGMPLUS_SIZE, read_ggmplus_tile

    n = GGMPLUS_SIZE
    # В файле: по меридианам с запада на восток, внутри меридиана — с юга на север.
    lat_idx = np.tile(np.arange(n, dtype=">i2"), n)   # номер точки от юга
    path = tmp_path / "N45E075.dg"
    lat_idx.tofile(path)
    tile = read_ggmplus_tile(path)
    assert tile.data.shape == (n, n)
    assert tile.data[0, 0] == pytest.approx((n - 1) / 10)   # север
    assert tile.data[-1, 0] == 0                            # юг
    assert tile.bounds == pytest.approx((75, 45, 80, 50))
