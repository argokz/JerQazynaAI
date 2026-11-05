import json

import numpy as np
import pytest

h5netcdf = pytest.importorskip("h5netcdf")
pytest.importorskip("h5py")
rasterio = pytest.importorskip("rasterio")

from jerqazyna.data import emit, emit_prepare
from jerqazyna.features.cache import Digests, write_json
from jerqazyna.grid.template import GridSpec, read_layer, write_layer


@pytest.fixture
def triplet(tmp_path):
    paths = {}
    for kind in ("MIN", "MINUNCERT", "MASK"):
        level = "L2A" if kind == "MASK" else "L2B"
        path = tmp_path / f"EMIT_{level}_{kind}_001_20240601T000000_2222205_001.nc"
        paths[kind] = path
        with h5netcdf.File(path, "w") as file:
            file.dimensions = {"downtrack": 4, "crosstrack": 4, "ortho_y": 4, "ortho_x": 4}
            file.attrs["geotransform"] = [66, .005, 0, 48, 0, -.005]
            file.attrs["spatial_ref"] = "EPSG:4326"
            location = file.create_group("location")
            yy, xx = np.indices((4, 4))
            for name, data, dims in [
                ("glt_x", xx + 1, ("ortho_y", "ortho_x")),
                ("glt_y", yy + 1, ("ortho_y", "ortho_x")),
                ("lat", 48 - (yy + .5) * .005, ("downtrack", "crosstrack")),
                ("lon", 66 + (xx + .5) * .005, ("downtrack", "crosstrack"))]:
                var = location.create_variable(name, dims, dtype="f8", fillvalue=-9999)
                var[:] = data
            if kind != "MASK":
                fields = ("mineral_id", "band_depth") if kind == "MIN" else ("fit", "band_depth_uncertainty")
                for group in (1, 2):
                    for field in fields:
                        source_field = "band_depth_unc" if field == "band_depth_uncertainty" else field
                        var = file.create_variable(f"group_{group}_{source_field}", ("downtrack", "crosstrack"), dtype="f4", fillvalue=-9999)
                        var[:] = np.ones((4, 4)) * (1 if field == "mineral_id" else .2)
                        if field == "mineral_id":
                            var[0, 0] = 0
                            var[0, 1] = -9999
                        else:
                            var[0, 0] = -9999  # no-match can have absent depth/uncertainty
            if kind == "MIN":
                metadata = file.create_group("mineral_metadata")
                metadata.dimensions["minerals"] = 1
                var = metadata.create_variable("index", ("minerals",), dtype="i4")
                var[:] = [1]
            if kind == "MASK":
                file.dimensions["bands"] = 8
                var = file.create_variable("mask", ("downtrack", "crosstrack", "bands"), dtype="f4", fillvalue=-9999)
                var[:] = np.zeros((4, 4, 8))
                var[0, 2, 7] = 1
                parameters = file.create_group("sensor_band_parameters")
                var = parameters.create_variable("mask_bands", ("bands",), dtype="S32")
                var[:] = np.array(["cloud", "cirrus", "water", "buffer", "aerosol", "water_vapor", "aot", "Aggregate Flag"], dtype="S32")
    return paths


def test_glt_one_based_invalid_indices_and_no_match():
    data = np.array([[0, 7], [8, 9]], float)
    gx = np.array([[1, 2, 0, -9999, 3, 1.5]])
    gy = np.ones(gx.shape)
    result, valid = emit_prepare.apply_glt(data, gx, gy)
    assert result[0, :2].tolist() == [0, 7]
    assert np.isnan(result[0, 2:]).all()
    assert valid.sum() == 2


def test_triplet_qa_fill_and_no_match_are_separate(triplet):
    arrays, reason, transform, metadata = emit_prepare.read_triplet(triplet)
    assert arrays["group_1_mineral_id"][0, 0] == 0
    assert reason[0].tolist() == [4, 1, 2, 4]
    assert transform.c == 66 and transform.f == 48
    assert metadata["no_match_id"] == 0
    assert metadata["aggregate_band_index_zero_based"] == 7
    assert metadata["source_variable_names"]["group_1_band_depth_uncertainty"] == "group_1_band_depth_unc"
    emit.validate_scientific(triplet["MIN"])


@pytest.mark.parametrize("field", ["glt_x", "lat"])
def test_mismatched_geometry_rejected(triplet, field):
    with h5netcdf.File(triplet["MASK"], "a") as file:
        file.groups["location"].variables[field][1, 1] += .001
    with pytest.raises(ValueError, match="mismatch"):
        emit_prepare.read_triplet(triplet)


def test_mismatched_scene_rejected(triplet):
    path = triplet["MASK"]
    other = path.with_name(path.name.replace("_001.nc", "_002.nc"))
    path.rename(other)
    triplet["MASK"] = other
    with pytest.raises(ValueError, match="version/time/orbit/scene"):
        emit_prepare.read_triplet(triplet)


def test_nonbinary_aggregate_flag_rejected(triplet):
    with h5netcdf.File(triplet["MASK"], "a") as file:
        file.variables["mask"][0, 0, 7] = 2
    with pytest.raises(ValueError, match="QA"):
        emit_prepare.read_triplet(triplet)


def test_fraction_denominators_qa_mine_absence_and_no_match():
    reason = np.array([[4, 4], [2, 3]])
    ids = np.array([[0, 2], [0, 0]])
    fractions = emit_prepare.quality_fractions(reason, ids, ids, factor=2)
    assert fractions["observed"].item() == 1
    assert fractions["usable"].item() == .5
    assert fractions["aggregate_QA"].item() == .25
    assert fractions["mine_mask"].item() == .25
    assert fractions["group_1_no_match_fraction"].item() == .5
    missing = emit_prepare.quality_fractions(np.zeros((2, 2)), ids, ids, factor=2)
    assert missing["usable"].item() == 0
    assert np.isnan(missing["group_1_no_match_fraction"]).all()


def test_warp_preserves_exact_mineral_id_zero_and_nan():
    from rasterio.transform import from_origin

    grid = GridSpec("EPSG:4326", 1, 66, 48, 2, 2)
    values = np.array([[0, 2], [3, np.nan]], np.float32)
    output = emit_prepare.warp(values, from_origin(66, 48, 1, 1), 4326, grid)
    assert output[0, 0] == 0
    np.testing.assert_array_equal(output, values)


def test_fill_and_declared_scaling_decoded_once(tmp_path):
    path = tmp_path / "array.nc"
    with h5netcdf.File(path, "w") as file:
        file.dimensions["x"] = 3
        var = file.create_variable("test", ("x",), dtype="i2", fillvalue=-9999)
        var[:] = [0, 2, -9999]
        var.attrs["scale_factor"] = .5
        var.attrs["add_offset"] = 1
        result = emit_prepare.decoded(var)
        assert result[:2].tolist() == [1, 2]
        assert np.isnan(result[2])


def test_corrupt_download_is_not_scientific(tmp_path):
    path = tmp_path / "login.nc"
    path.write_text("<html>Login</html>")
    with pytest.raises(ValueError, match="scientific"):
        emit.validate_scientific(path)


def test_prepare_end_to_end_isolation_mine_date_union_and_reproducibility(triplet, tmp_path, monkeypatch):
    import geopandas as gpd
    from shapely.geometry import box

    root = tmp_path
    grid = GridSpec.from_bounds("EPSG:6933", 1000, (6368000, 5450000, 6373000, 5455000))
    # Derive a small 1 km grid around the actual synthetic GLT footprint.
    from rasterio.warp import transform_bounds

    bounds = transform_bounds(4326, "EPSG:6933", 66, 47.98, 66.02, 48)
    grid = GridSpec.from_bounds("EPSG:6933", 1000, bounds)
    (root / "split").mkdir()
    train = np.ones(grid.shape, np.uint8)
    train[-1, -1] = 0  # stand-in for a closed/buffer cell
    write_layer(root / "split/train.tif", grid, train)
    mine_dir = root / "mines"
    mine_dir.mkdir()
    gpd.GeoDataFrame(geometry=[box(66.015, 47.98, 66.02, 48)], crs=4326).to_file(mine_dir / "mine.gpkg")
    c = emit.PilotConfig(pilot_id="toy", task="task", split="split", baseline="baseline",
                         areas=[emit.PilotArea(name="area", bbox=(66, 47.98, 66.02, 48))])
    raw = root / "raw"
    raw.mkdir()
    for path in triplet.values():
        path.rename(raw / path.name)
    state = {"state": "ready", "files": {p.name: {"sha256": Digests().file(p), "bytes": p.stat().st_size} for p in raw.iterdir()}}
    write_json(root / "fetch.json", state)
    assets = [{"kind": k, "name": p.name} for k, p in triplet.items()]
    scene = {"key": ["001", "20240601T000000", "2222205", "001"], "date": "2024-06-01", "area": "area", "assets": assets}
    anchor = {"grid": grid.__dict__, "mine_source": {"source": "mines", "params": {"pattern": "*.gpkg"}}, "mine_buffer_m": 0}
    saved = {"search_id": "fixture", "spec": {"scenes": [scene, scene], "anchors": anchor}}
    monkeypatch.setattr(emit_prepare, "frozen_search", lambda *args: (root, saved))
    output = emit_prepare.prepare(c, root)
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(manifest["summary"]["dates"]) == 1
    assert len(manifest["summary"]["scenes"]) == 2
    fraction = read_layer(output / "area/combined/usable.tif")
    dates = read_layer(output / "area/combined/usable_date_count.tif")
    assert np.nanmax(fraction) <= 1 and np.nanmax(dates) <= 1
    assert np.isnan(fraction[-1, -1]) and np.isnan(dates[-1, -1])
    assert np.nansum(read_layer(output / "area/combined/mine_mask.tif")) > 0
    assert emit_prepare.prepare(c, root) == output
    # Native IDs survive QA; QA reason is exported separately instead of replacing ID 0.
    native = output / "area/2024-06-01/001_20240601T000000_2222205_001"
    assert (native / "group_2_mineral_id.tif").exists()
    assert json.loads((native / "metadata.json").read_text())["no_match_id"] == 0
    from jerqazyna.data.emit_validate import check_native, source_mines

    checked = check_native(scene, native, raw, grid, source_mines(root, anchor, grid))
    assert checked["source_value_checks"] > 0
    assert checked["sampled_no_match_group_1"] > 0
