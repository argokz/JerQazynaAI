import numpy as np
import pytest

gpd = pytest.importorskip("geopandas")
pytest.importorskip("rasterio")

from shapely.geometry import box

from jerqazyna.config import CENTRAL_ASIA_ALBERS
from jerqazyna.grid.template import GridSpec
from jerqazyna.validation.holdout import region_masks


def test_polygon_buffer_is_measured_from_boundary(tmp_path):
    grid = GridSpec.from_bounds(CENTRAL_ASIA_ALBERS, 1000, (0, 0, 10_000, 10_000))
    path = tmp_path / "region.gpkg"
    # Граница x=5100: центр 4500 находится всего в 600 м от полигона,
    # но в 1000 м от ближайшего тестового центра 5500.
    gpd.GeoDataFrame(geometry=[box(5100, 0, 10_000, 10_000)], crs=grid.crs).to_file(path)
    area = np.ones(grid.shape, bool)
    train, test, buffer = region_masks(grid, area, path, 750)
    assert not train[:, 4].any()
    assert buffer[:, 4].all()
    assert test[:, 5:].all()
    assert train[:, :4].all()
    assert np.array_equal(train.astype(int) + test + buffer, area.astype(int))


def test_holdout_respects_area_and_requires_both_partitions(tmp_path):
    grid = GridSpec.from_bounds(CENTRAL_ASIA_ALBERS, 1000, (0, 0, 10_000, 10_000))
    path = tmp_path / "region.gpkg"
    gpd.GeoDataFrame(geometry=[box(6000, 0, 10_000, 10_000)], crs=grid.crs).to_file(path)
    area = np.ones(grid.shape, bool)
    area[0] = False
    for mask in region_masks(grid, area, path, 1000):
        assert not mask[0].any()
    area[:, :6] = False
    with pytest.raises(ValueError, match="непустые"):
        region_masks(grid, area, path, 1000)
