"""Independent checks of raw->native GLT samples and published 1 km area accounting."""

from __future__ import annotations

import argparse
import contextlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np

from jerqazyna.data.emit import frozen_search, load_pilot, ready_files, write_json
from jerqazyna.features.cache import Digests, fingerprint
from jerqazyna.grid.template import GridSpec, read_layer

FACT = 20
REASONS = {0: "no_observation_or_invalid_GLT", 1: "invalid_geolocation_or_fill",
           2: "aggregate_QA", 3: "mine_mask", 4: "usable"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def source_mines(root, anchor, grid):
    import geopandas as gpd
    from shapely.geometry import box

    layer = anchor["mine_source"]
    expanded = box(*grid.bounds).buffer(anchor["mine_buffer_m"])
    polygons = []
    for path in sorted((root / layer["source"]).rglob(layer["params"].get("pattern", "*.gpkg"))):
        crs = gpd.read_file(path, rows=0).crs
        bounds = gpd.GeoSeries([expanded], crs=grid.crs).to_crs(crs).total_bounds
        frame = gpd.read_file(path, bbox=tuple(bounds)).to_crs(grid.crs)
        polygons.extend(frame.geometry.buffer(anchor["mine_buffer_m"]).tolist())
    return polygons


def contains_any(polygons, x, y):
    import shapely

    result = np.zeros(np.shape(x), bool)
    points = shapely.points(x, y)
    for polygon in polygons:
        result |= shapely.covers(polygon, points)
    return result


def decode_samples(variable, rows, cols):
    values = np.asarray(variable[:])[rows, cols].astype(float)
    missing = ~np.isfinite(values) | (values == -9999)
    if "_FillValue" in variable.attrs:
        missing |= values == variable.attrs["_FillValue"]
    values = values * variable.attrs.get("scale_factor", 1) + variable.attrs.get("add_offset", 0)
    values[missing] = np.nan
    return values


def check_native(scene, native, raw, grid, mines):
    import h5netcdf
    import rasterio
    from pyproj import Geod, Transformer
    from shapely.ops import transform

    metadata = read_json(native / "metadata.json")
    with rasterio.open(native / "quality_reason.tif") as raster:
        reasons, affine = raster.read(1), raster.transform
    with rasterio.open(native / "group_1_mineral_id.tif") as raster:
        first_ids = raster.read(1)
    choices = []
    masks = [reasons == value for value in REASONS]
    masks += [(reasons == 4) & (first_ids == 0), (reasons == 4) & (first_ids > 0)]
    for mask in masks:
        candidates = np.flatnonzero(mask)
        if candidates.size:
            choices.extend(candidates[np.linspace(0, candidates.size - 1, min(8, candidates.size), dtype=int)].tolist())
    if not choices:
        raise ValueError("No native pixels available for scientific verification")
    rr, cc = np.unravel_index(np.unique(choices), reasons.shape)
    lons = affine.c + (cc + .5) * affine.a
    lats = affine.f + (rr + .5) * affine.e
    paths = {a["kind"]: raw / a["name"] for a in scene["assets"]}
    checked_fields = 0
    with contextlib.ExitStack() as stack:
        files = {kind: stack.enter_context(h5netcdf.File(path, "r")) for kind, path in paths.items()}
        loc = files["MIN"].groups["location"]
        gt = np.asarray(files["MIN"].attrs["geotransform"], float)
        column_offset = round((affine.c - gt[0]) / gt[1])
        row_offset = round((affine.f - gt[3]) / gt[5])
        gx = np.asarray(loc.variables["glt_x"][:])[rr + row_offset, cc + column_offset]
        gy = np.asarray(loc.variables["glt_y"][:])[rr + row_offset, cc + column_offset]
        shape = files["MIN"].variables["group_1_mineral_id"].shape
        glt_valid = (gx >= 1) & (gx <= shape[1]) & (gy >= 1) & (gy <= shape[0])
        rx, ry = np.clip(gx.astype(int) - 1, 0, shape[1] - 1), np.clip(gy.astype(int) - 1, 0, shape[0] - 1)
        values = {}
        for canonical, original in metadata["source_variable_names"].items():
            kind = "MIN" if original.endswith(("mineral_id", "band_depth")) else "MINUNCERT"
            expected = decode_samples(files[kind].variables[original], ry, rx)
            expected[~glt_valid] = np.nan
            values[canonical] = expected
            with rasterio.open(native / f"{canonical}.tif") as raster:
                actual = raster.read(1)[rr, cc]
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7, equal_nan=True,
                                       err_msg=f"Raw/GLT/native mismatch: {scene['key']}/{canonical}")
            checked_fields += len(rr)
        raw_lon = decode_samples(loc.variables["lon"], ry, rx)
        raw_lat = decode_samples(loc.variables["lat"], ry, rx)
        valid = glt_valid & np.isfinite(raw_lon) & np.isfinite(raw_lat) & (np.abs(raw_lon) <= 180) & (np.abs(raw_lat) <= 90)
        for group in (1, 2):
            ids = values[f"group_{group}_mineral_id"]
            valid &= np.isfinite(ids) & (ids >= 0) & (ids == np.floor(ids))
            for field in ("fit", "band_depth", "band_depth_uncertainty"):
                valid &= (ids == 0) | np.isfinite(values[f"group_{group}_{field}"])
        bands = files["MASK"].groups["sensor_band_parameters"].variables["mask_bands"][:]
        names = [b.decode() if isinstance(b, bytes) else str(b) for b in bands]
        band = next(i for i, name in enumerate(names) if "aggregate" in name.lower())
        quality = np.asarray(files["MASK"].variables["mask"][:])[ry, rx, band]
        expected_reason = np.where(glt_valid, 1, 0)
        expected_reason[valid] = 2
        expected_reason[valid & (quality == 0)] = 4
        geographic = [transform(Transformer.from_crs(grid.crs, 4326, always_xy=True).transform, g) for g in mines]
        expected_reason[(expected_reason == 4) & contains_any(geographic, lons, lats)] = 3
        np.testing.assert_array_equal(reasons[rr, cc], expected_reason, err_msg="Raw QA/mine reason mismatch")
        finite = glt_valid & np.isfinite(raw_lon) & np.isfinite(raw_lat)
        _, _, distances = Geod(ellps="WGS84").inv(lons[finite], lats[finite], raw_lon[finite], raw_lat[finite])
    return {"area": scene["area"], "date": scene["date"], "key": scene["key"],
            "native_pixels": len(rr), "source_value_checks": checked_fields,
            "sampled_no_match_group_1": int((values["group_1_mineral_id"] == 0).sum()),
            "sampled_reason_codes": np.unique(expected_reason).tolist(),
            "sampled_center_distance_m_p50_p95_max": np.quantile(distances, [.5, .95, 1]).tolist() if len(distances) else None}


def warp_native(path, grid):
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    with rasterio.open(path) as source:
        output = np.full(grid.shape, np.nan, np.float32)
        reproject(source.read(1), output, src_transform=source.transform, src_crs=source.crs,
                  dst_transform=grid.transform, dst_crs=grid.crs, src_nodata=np.nan,
                  dst_nodata=np.nan, resampling=Resampling.nearest)
    return output


def metrics(reason, ids):
    result = {name: np.count_nonzero(reason == code) / reason.size for code, name in REASONS.items()}
    result["observed"] = np.count_nonzero(reason >= 2) / reason.size
    usable = np.count_nonzero(reason == 4)
    for group in (1, 2):
        result[f"group_{group}_no_match_fraction"] = (
            np.count_nonzero((reason == 4) & (ids[group - 1] == 0)) / usable if usable else None)
    return result


def check_metrics(directory, cells, reasons, ids):
    maps = {name: read_layer(directory / f"{name}.tif") for name in metrics(reasons[0], [v[0] for v in ids])}
    checks = []
    for i, (row, col) in enumerate(cells):
        expected = metrics(reasons[i], [v[i] for v in ids])
        observed = {name: float(values[row, col]) if np.isfinite(values[row, col]) else None for name, values in maps.items()}
        errors = []
        for name in expected:
            if expected[name] is None or observed[name] is None:
                if expected[name] != observed[name]:
                    raise ValueError(f"Missing/zero semantic mismatch: {directory}/{row}/{col}/{name}")
            else:
                errors.append(abs(expected[name] - observed[name]))
        if errors and max(errors) > 1e-6:
            raise ValueError(f"1 km reproduction mismatch: {directory}/{row}/{col}: {max(errors)}")
        checks.append({"row": int(row), "col": int(col), "expected": expected, "actual": observed,
                       "max_absolute_error": max(errors, default=0)})
    return checks


def plots(prepared, output, areas):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fields = [("observed", "Наблюдаемая площадь"), ("usable", "Пригодная площадь"),
              ("aggregate_QA", "Отбраковано QA"), ("mine_mask", "Маска выработок"),
              ("group_1_no_match_fraction", "No-match, группа 1"),
              ("group_2_no_match_fraction", "No-match, группа 2"),
              ("invalid_geolocation_or_fill", "Fill / геолокация"), ("usable_date_count", "Пригодных дат")]
    for area in areas:
        grid = GridSpec.load(prepared / area / "grid.json")
        bounds = [grid.bounds[i] / 1000 for i in (0, 2, 1, 3)]
        fig, axes = plt.subplots(2, 4, figsize=(16, 9), constrained_layout=True)
        for axis, (name, title) in zip(axes.flat, fields):
            values = read_layer(prepared / area / "combined" / f"{name}.tif")
            count = name == "usable_date_count"
            palette = plt.get_cmap("viridis").copy()
            palette.set_bad("white")
            image = axis.imshow(values, extent=bounds, origin="upper", vmin=0, vmax=2 if count else 1,
                                cmap=palette, interpolation="nearest")
            axis.set(title=title, xlabel="Восток, км", ylabel="Север, км")
            bar = fig.colorbar(image, ax=axis, shrink=.7, label="дат" if count else "доля")
            if count:
                bar.set_ticks([0, 1, 2])
        fig.suptitle(f"EMIT V001 — {area}\n1 км; площадь по сетке 50 м. Белый: нет значения / вне AOI∩train", fontsize=16)
        fig.savefig(output / f"quality_{area}.png", dpi=160)
        plt.close(fig)


def validate(config, root):

    root = Path(root).resolve()
    dest, saved = frozen_search(root, config)
    ready_files(dest, saved)
    pointer = read_json(dest / "latest_prepare.json")
    prepared = dest / pointer["directory"]
    digests = Digests()
    if digests.file(prepared / "manifest.json") != pointer["sha256"]:
        raise ValueError("Prepared manifest changed")
    manifest = read_json(prepared / "manifest.json")
    for name, sha in manifest["files"].items():
        if digests.file(prepared / name) != sha:
            raise ValueError(f"Prepared file changed: {name}")
    spec = {"prepared_manifest_sha256": pointer["sha256"], "implementation_sha256": digests.file(Path(__file__)),
            "dependencies": {p: version(p) for p in ("rasterio", "h5netcdf", "shapely", "matplotlib")},
            "method": "Raw GLT/QA samples plus independent recomputation from native TIFFs; 400 area samples per cell"}
    output = dest / ("validation_" + fingerprint(spec)[:16])
    if (output / "manifest.json").exists():
        old = read_json(output / "manifest.json")
        if old["spec"] != spec or any(digests.file(output / n) != h for n, h in old["files"].items()):
            raise ValueError("Validation generation changed")
        return output
    output.mkdir(parents=True, exist_ok=True)
    result = {"native_checks": [], "cell_checks": [], "date_count_checks": 0, "holdout_evaluated": False}
    for area in config.areas:
        grid = GridSpec.load(prepared / area.name / "grid.json")
        fine = grid.refined(FACT)
        mines = source_mines(root, saved["spec"]["anchors"], grid)
        scenes = [s for s in saved["spec"]["scenes"] if s["area"] == area.name]
        dates = sorted({s["date"] for s in scenes})
        cells = set()
        for date in dates:
            path = prepared / area.name / date / "grid_1km"
            interior = read_layer(path / "aoi_train_area_fraction.tif") == 1
            for field in ("observed", "usable", "aggregate_QA", "mine_mask", "group_1_no_match_fraction"):
                values = read_layer(path / f"{field}.tif")
                candidates = np.flatnonzero(interior & np.isfinite(values) & (values > 0))
                if candidates.size:
                    for idx in (candidates[0], candidates[np.argmax(values.ravel()[candidates])]):
                        cells.add(tuple(np.unravel_index(idx, grid.shape)))
        cells = sorted(cells)
        if not cells:
            raise ValueError("No interior observed cells for numerical checks")
        yy, xx = np.indices((FACT, FACT))
        xs = np.stack([grid.x_min + (col + (xx + .5) / FACT) * grid.cell_size for _, col in cells])
        ys = np.stack([grid.y_max - (row + (yy + .5) / FACT) * grid.cell_size for row, _ in cells])
        mine_samples = contains_any(mines, xs, ys)
        date_counts = np.zeros(len(cells), int)
        all_reasons = np.zeros(xs.shape, np.uint8)
        all_ids = [np.full(xs.shape, np.nan, np.float32) for _ in (1, 2)]
        for date in dates:
            reasons = np.zeros(xs.shape, np.uint8)
            ids = [np.full(xs.shape, np.nan, np.float32) for _ in (1, 2)]
            for scene in sorted([s for s in scenes if s["date"] == date], key=lambda s: s["key"]):
                native = prepared / area.name / date / "_".join(scene["key"])
                result["native_checks"].append(check_native(scene, native, dest / "raw", grid, mines))
                samples = []
                for name in ("quality_reason", "group_1_mineral_id", "group_2_mineral_id"):
                    warped = warp_native(native / f"{name}.tif", fine)
                    samples.append(np.stack([warped[row * FACT:(row + 1) * FACT, col * FACT:(col + 1) * FACT]
                                             for row, col in cells]))
                source_reason = np.nan_to_num(samples[0], nan=0).astype(np.uint8)
                source_reason[(source_reason == 4) & mine_samples] = 3
                replace = source_reason > reasons
                reasons[replace] = source_reason[replace]
                for group in range(2):
                    ids[group][replace] = samples[group + 1][replace]
            checked = check_metrics(prepared / area.name / date / "grid_1km", cells, reasons, ids)
            result["cell_checks"].extend({"area": area.name, "date": date, **row} for row in checked)
            date_counts += np.any(reasons == 4, axis=(1, 2))
            replace = reasons > all_reasons
            all_reasons[replace] = reasons[replace]
            for group in range(2):
                all_ids[group][replace] = ids[group][replace]
        checked = check_metrics(prepared / area.name / "combined", cells, all_reasons, all_ids)
        result["cell_checks"].extend({"area": area.name, "date": "combined", **row} for row in checked)
        counts = read_layer(prepared / area.name / "combined/usable_date_count.tif")
        np.testing.assert_array_equal([counts[r, c] for r, c in cells], date_counts)
        result["date_count_checks"] += len(cells)
        # Check every 1 km export against the original saved train mask.
        original = GridSpec(**saved["spec"]["anchors"]["grid"])
        rs = round((original.y_max - grid.y_max) / original.cell_size)
        cs = round((grid.x_min - original.x_min) / original.cell_size)
        allowed = read_layer(root / config.split / "train.tif")[rs:rs + grid.height, cs:cs + grid.width] > 0
        for directory in [prepared / area.name / "combined", *[prepared / area.name / d / "grid_1km" for d in dates]]:
            for path in directory.glob("*.tif"):
                if not np.isnan(read_layer(path)[~allowed]).all():
                    raise ValueError("Validation found a result outside saved train")
    plots(prepared, output, [area.name for area in config.areas])
    result["native_pixel_checks"] = sum(row["native_pixels"] for row in result["native_checks"])
    result["source_value_checks"] = sum(row["source_value_checks"] for row in result["native_checks"])
    result["max_cell_absolute_error"] = max(row["max_absolute_error"] for row in result["cell_checks"])
    write_json(output / "validation.json", result)
    write_json(output / "manifest.json", {"spec": spec, "passed": True,
               "files": {p.name: digests.file(p) for p in sorted(output.iterdir()) if p.is_file() and ".part" not in p.name}})
    write_json(dest / "latest_validation.json", {"directory": output.name, "sha256": digests.file(output / "manifest.json")})
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pilot")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    print(validate(load_pilot(args.pilot), Path(args.root)))


if __name__ == "__main__":
    main()
