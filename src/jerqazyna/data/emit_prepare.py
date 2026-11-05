"""GLT mineral products and quality accounting; no model or label-dependent thresholds."""

from __future__ import annotations

import contextlib
import json
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np

from jerqazyna.data.emit import area_geometry, frozen_search, local_path, ready_files
from jerqazyna.features.cache import Digests, fingerprint, write_json
from jerqazyna.grid.template import GridSpec, read_layer, write_layer

FACTOR = 20  # 50 m equal-area quadrature, 400 equal-area samples per 1 km cell.
REASONS = {0: "no_observation_or_invalid_GLT", 1: "invalid_geolocation_or_fill",
           2: "aggregate_QA", 3: "mine_mask", 4: "usable"}


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def decoded(variable):
    raw = np.asarray(variable[:])
    array = raw.astype(np.float64)
    valid = np.isfinite(array)
    for key in ("_FillValue", "missing_value"):
        if key in variable.attrs:
            for fill in np.atleast_1d(variable.attrs[key]):
                valid &= raw != fill
    valid &= raw != -9999
    array = array * variable.attrs.get("scale_factor", 1) + variable.attrs.get("add_offset", 0)
    array[~valid] = np.nan
    return array


def apply_glt(array, glt_x, glt_y):
    """GLT indexes are 1-based; reject fractional and out-of-range indexes, never wrap -1."""
    if glt_x.shape != glt_y.shape or glt_x.ndim != 2 or array.ndim != 2:
        raise ValueError("Invalid GLT dimensions")
    valid = np.isfinite(glt_x) & np.isfinite(glt_y)
    valid &= (glt_x >= 1) & (glt_x <= array.shape[1]) & (glt_y >= 1) & (glt_y <= array.shape[0])
    valid &= (glt_x == np.floor(glt_x)) & (glt_y == np.floor(glt_y))
    out = np.full(glt_x.shape, np.nan, dtype=np.float32)
    out[valid] = array[glt_y[valid].astype(int) - 1, glt_x[valid].astype(int) - 1]
    return out, valid


def read_triplet(paths):
    import h5netcdf
    from affine import Affine

    from jerqazyna.data.emit import ASSET

    keys = []
    for kind, path in paths.items():
        match = ASSET.fullmatch(path.name)
        if not match or match[1] != kind:
            raise ValueError("Invalid scientific file name")
        keys.append(match.groups()[1:])
    if set(paths) != {"MIN", "MINUNCERT", "MASK"} or len(set(keys)) != 1:
        raise ValueError("Files differ in version/time/orbit/scene")
    with contextlib.ExitStack() as stack:
        files = {k: stack.enter_context(h5netcdf.File(v, "r")) for k, v in paths.items()}
        reference = files["MIN"]
        location = reference.groups["location"]
        gx, gy = decoded(location.variables["glt_x"]), decoded(location.variables["glt_y"])
        gt = np.asarray(reference.attrs["geotransform"], float)
        if gt.shape != (6,) or not np.isfinite(gt).all() or gt[1] <= 0 or gt[5] >= 0 or gt[2] or gt[4]:
            raise ValueError("Unsupported geographic geotransform")
        transform = Affine.from_gdal(*gt)
        # EMIT V001 GLTs are geographic EPSG:4326; validate the declared CRS.
        from rasterio.crs import CRS

        if CRS.from_user_input(reference.attrs["spatial_ref"]) != CRS.from_epsg(4326):
            raise ValueError("Unexpected GLT CRS")
        arrays, attrs, source_variables = {}, {}, {}
        for kind in ("MIN", "MINUNCERT"):
            for group in (1, 2):
                fields = ("mineral_id", "band_depth") if kind == "MIN" else ("fit", "band_depth_uncertainty")
                for field in fields:
                    name = f"group_{group}_{field}"
                    candidates = [name]
                    if field == "band_depth_uncertainty":
                        candidates.append(f"group_{group}_band_depth_unc")
                    found = [candidate for candidate in candidates if candidate in files[kind].variables]
                    if len(found) != 1:
                        raise ValueError(f"Missing or ambiguous EMIT variable: {name}")
                    var = files[kind].variables[found[0]]
                    if tuple(var.dimensions) != ("downtrack", "crosstrack"):
                        raise ValueError(f"Unexpected dimensions: {name}")
                    arrays[name] = decoded(var)
                    attrs[name] = json_value(dict(var.attrs))
                    source_variables[name] = found[0]
        shape = arrays["group_1_mineral_id"].shape
        for name, array in arrays.items():
            if array.shape != shape:
                raise ValueError(f"Raw geometry mismatch: {name}")
        lat, lon = decoded(location.variables["lat"]), decoded(location.variables["lon"])
        if lat.shape != shape or lon.shape != shape:
            raise ValueError("Geolocation shape mismatch")
        for kind, file in files.items():
            loc = file.groups["location"]
            if not np.array_equal(np.asarray(file.attrs["geotransform"]), gt):
                raise ValueError(f"Geotransform mismatch: {kind}")
            for name, original in (("glt_x", gx), ("glt_y", gy), ("lat", lat), ("lon", lon)):
                if not np.array_equal(decoded(loc.variables[name]), original, equal_nan=True):
                    raise ValueError(f"Geolocation/GLT mismatch: {kind}/{name}")
        bands = files["MASK"].groups["sensor_band_parameters"].variables["mask_bands"][:]
        names = [str(json_value(b)).strip() for b in bands]
        indexes = [i for i, name in enumerate(names) if "aggregate" in name.casefold()]
        if len(indexes) != 1:
            raise ValueError("Aggregate quality flag not uniquely identified by metadata")
        mask_var = files["MASK"].variables["mask"]
        if tuple(mask_var.dimensions[:2]) != ("downtrack", "crosstrack"):
            raise ValueError("MASK dimension order mismatch")
        qa = decoded(mask_var)[..., indexes[0]]
        if qa.shape != shape or np.any(np.isfinite(qa) & ~np.isin(qa, [0, 1])):
            raise ValueError("Invalid aggregate QA geometry/values")
        geovalid = np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) <= 90) & (np.abs(lon) <= 180)
        id_valid = np.ones(shape, bool)
        for group in (1, 2):
            ids = arrays[f"group_{group}_mineral_id"]
            id_valid &= np.isfinite(ids) & (ids >= 0) & (ids == np.floor(ids))
            # No-match (0) can have absent depth/uncertainty; this is not fill observation.
            for field in ("band_depth", "band_depth_uncertainty", "fit"):
                id_valid &= (ids == 0) | np.isfinite(arrays[f"group_{group}_{field}"])
        raw_reason = np.ones(shape, np.float32)
        observed = geovalid & id_valid
        raw_reason[observed] = 2
        raw_reason[observed & (qa == 0)] = 4
        reason, glt_valid = apply_glt(raw_reason, gx, gy)
        reason[~glt_valid] = 0
        orthos = {k: apply_glt(v, gx, gy)[0] for k, v in arrays.items()}
        metadata = {"variables": attrs, "source_variable_names": source_variables,
                    "mask_bands": names, "aggregate_band_index_zero_based": indexes[0],
                    "root_attributes": {kind: json_value(dict(file.attrs)) for kind, file in files.items()},
                    "mineral_metadata": {name: {"values": json_value(var[:]), "attributes": json_value(dict(var.attrs))}
                                         for name, var in reference.groups["mineral_metadata"].variables.items()},
                    "no_match_id": 0, "reason_codes": REASONS,
                    "fit_interpretation": "As distributed, with declared scale/offset decoded once. NASA tutorial notes V001 fits can be half the correlation; no extra factor or filtering applied.",
                    "glt_valid_pixels": int(glt_valid.sum()), "glt_invalid_pixels": int((~glt_valid).sum())}
        # Independent geometry diagnostic: GLT-assigned centers vs original raw lat/lon.
        from pyproj import Geod

        ortho_lat = apply_glt(lat, gx, gy)[0]
        ortho_lon = apply_glt(lon, gx, gy)[0]
        yy, xx = np.indices(gx.shape)
        center_lon = transform.c + (xx + .5) * transform.a
        center_lat = transform.f + (yy + .5) * transform.e
        valid = glt_valid & np.isfinite(ortho_lat) & np.isfinite(ortho_lon)
        _, _, distances = Geod(ellps="WGS84").inv(center_lon[valid], center_lat[valid],
                                                 ortho_lon[valid], ortho_lat[valid])
        distances = distances[np.isfinite(distances)]
        metadata["glt_raw_center_distance_m"] = {"n": int(distances.size), "p50_p95_max":
                                                np.quantile(distances, [.5, .95, 1]).tolist() if distances.size else None}
    return orthos, reason, transform, metadata


def warp(array, transform, src_crs, grid, resampling=None):
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    # Zero is a valid mineral ID (no match). Some GDAL versions nudge valid zeros
    # away from a destination nodata=0 to the smallest positive float32 value.
    output = np.full(grid.shape, np.nan, np.float32)
    reproject(array, output, src_transform=transform, src_crs=src_crs,
              dst_transform=grid.transform, dst_crs=grid.crs, src_nodata=np.nan, dst_nodata=np.nan,
              resampling=resampling or Resampling.nearest)
    return output


def quadrature(array, factor=FACTOR):
    h, w = array.shape
    if h % factor or w % factor:
        raise ValueError("Quadrature must align with target grid")
    return array.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))


def quality_fractions(reason, group1, group2, factor=FACTOR):
    usable = reason == 4
    fractions = {name: quadrature(reason == code, factor) for code, name in REASONS.items()}
    fractions["observed"] = quadrature(reason >= 2, factor)
    for group, ids in [(1, group1), (2, group2)]:
        numerator = quadrature(usable & (ids == 0), factor)
        denominator = fractions["usable"]
        fractions[f"group_{group}_no_match_fraction"] = np.divide(
            numerator, denominator, out=np.full(denominator.shape, np.nan), where=denominator > 0)
    return fractions


def mine_geometries(root, anchor, crs, bbox):
    import geopandas as gpd

    source = anchor["mine_source"]
    files = sorted(local_path(root, source["source"]).rglob(source["params"].get("pattern", "*.gpkg")))
    if not files:
        raise ValueError("No source mine polygons")
    result = []
    for path in files:
        # Spatial filter in file CRS before reading, expanded for the mine buffer.
        from shapely.geometry import box

        expanded = box(*bbox).buffer(anchor["mine_buffer_m"])
        file_crs = gpd.read_file(path, rows=0).crs
        if file_crs is None:
            raise ValueError("Mine polygons have no CRS")
        native_bbox = gpd.GeoSeries([expanded], crs=crs).to_crs(file_crs).total_bounds
        frame = gpd.read_file(path, bbox=tuple(native_bbox)).to_crs(crs)
        if not frame.geometry.is_valid.all():
            raise ValueError("Invalid mine polygons")
        result.extend(frame.geometry.buffer(anchor["mine_buffer_m"]).tolist())
    return result


def write_native(path, array, transform):
    import rasterio

    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".part.tif")
    with rasterio.open(temp, "w", driver="GTiff", height=array.shape[0], width=array.shape[1],
                       count=1, dtype="float32", crs="EPSG:4326", transform=transform,
                       nodata=np.nan, tiled=True, compress="deflate", predictor=3) as dataset:
        dataset.write(array.astype(np.float32), 1)
    temp.replace(path)


def uncertainty_summary(orthos, usable):
    summary = {}
    for name, data in orthos.items():
        if not ("uncertainty" in name or "fit" in name):
            continue
        group = name.split("_")[1]
        sample = data[usable & (orthos[f"group_{group}_mineral_id"] > 0) & np.isfinite(data)]
        summary[name] = {"n": int(sample.size), "quantiles_p05_p50_p95":
                         np.quantile(sample, [.05, .5, .95]).tolist() if sample.size else None}
    return summary


def prepare(config, root):
    import rasterio
    from pyproj import Transformer
    from rasterio.features import rasterize
    from rasterio.windows import Window
    from rasterio.windows import transform as window_transform
    from shapely.ops import transform as shapely_transform

    root = Path(root).resolve()
    dest, saved = frozen_search(root, config)
    state = ready_files(dest, saved)
    d = Digests()
    spec = {"search_id": saved["search_id"], "raw_files": state["files"],
            "implementation": {p: d.file(Path(__file__).with_name(p)) for p in ("emit.py", "emit_prepare.py")},
            "dependencies": {p: version(p) for p in ("numpy", "rasterio", "h5netcdf", "h5py", "pyproj", "geopandas")},
            "area_quadrature_m": 1000 / FACTOR, "no_label_thresholds": True}
    output = dest / ("prepared_" + fingerprint(spec)[:16])
    manifest = output / "manifest.json"
    if manifest.exists():
        old = json.loads(manifest.read_text(encoding="utf-8"))
        if old["spec"] != spec or any(d.file(output / p) != sha for p, sha in old["files"].items()):
            raise ValueError("Corrupt prepared pilot")
        write_json(dest / "latest_prepare.json", {"directory": output.name, "sha256": d.file(manifest)})
        return output
    output.mkdir(parents=True, exist_ok=True)
    anchor = saved["spec"]["anchors"]
    grid = GridSpec(**anchor["grid"])
    train = read_layer(local_path(root, config.split) / "train.tif") > 0
    summary = {"scenes": [], "dates": [], "areas": [],
               "aggregation": "50 m equal-area quadrature (400 samples per 1 km cell); area approximation, not exact polygon intersection. Date overlaps resolved before aggregation; fractions refer to full 1 km cell.",
               "date_mosaic": "Prefer usable over mine-excluded over QA over invalid; equal quality ties prefer first sorted scene. No uncertainty threshold."}
    for area in config.areas:
        geometry = area_geometry(area, grid.crs)
        slices = grid.window(geometry.bounds)
        if slices is None:
            raise ValueError("Pilot outside saved grid")
        rs, cs = slices
        coarse = GridSpec(grid.crs, grid.cell_size, grid.x_min + cs.start * grid.cell_size,
                          grid.y_max - rs.start * grid.cell_size, cs.stop - cs.start, rs.stop - rs.start)
        fine = coarse.refined(FACTOR)
        allowed = rasterize([(geometry, 1)], out_shape=fine.shape, transform=fine.transform).astype(bool)
        allowed &= np.repeat(np.repeat(train[rs, cs], FACTOR, axis=0), FACTOR, axis=1)
        coarse_allowed = train[rs, cs] & (quadrature(allowed) > 0)
        mines = mine_geometries(root, anchor, grid.crs, coarse.bounds)
        mine = (rasterize([(g, 1) for g in mines], out_shape=fine.shape, transform=fine.transform).astype(bool)
                if mines else np.zeros(fine.shape, bool))
        all_reason = np.zeros(fine.shape, np.uint8)
        all_ids = [np.full(fine.shape, np.nan, np.float32) for _ in (1, 2)]
        dates_count = np.zeros(coarse.shape, np.float32)
        area_scenes = [s for s in saved["spec"]["scenes"] if s["area"] == area.name]

        def save_maps(subdir, reason, ids, allowed=allowed, coarse_allowed=coarse_allowed, coarse=coarse):
            subdir.mkdir(parents=True, exist_ok=True)
            fractions = quality_fractions(reason, *ids)
            fractions["aoi_train_area_fraction"] = quadrature(allowed)
            for name, values in fractions.items():
                values = values.astype(np.float32)
                values[~coarse_allowed] = np.nan
                write_layer(subdir / f"{name}.tif", coarse, values, name)
            n = int(coarse_allowed.sum())
            totals = {name: float(np.nansum(values[coarse_allowed])) for name, values in fractions.items()
                      if not name.endswith("no_match_fraction")}
            for group in (1, 2):
                values = fractions[f"group_{group}_no_match_fraction"] * fractions["usable"]
                numerator = float(np.nansum(values[coarse_allowed]))
                totals[f"group_{group}_no_match_fraction"] = numerator / totals["usable"] if totals["usable"] else None
            for field in ("observed", "usable", "aggregate_QA", "mine_mask", "invalid_geolocation_or_fill"):
                totals[f"{field}_fraction_of_aoi_train"] = (totals[field] / totals["aoi_train_area_fraction"]
                                                          if totals["aoi_train_area_fraction"] else None)
            return totals | {"train_aoi_cells": n}

        for date in sorted({s["date"] for s in area_scenes}):
            date_reason = np.zeros(fine.shape, np.uint8)
            date_ids = [np.full(fine.shape, np.nan, np.float32) for _ in (1, 2)]
            for scene in sorted([s for s in area_scenes if s["date"] == date], key=lambda s: s["key"]):
                paths = {a["kind"]: dest / "raw" / a["name"] for a in scene["assets"]}
                orthos, reason, transform, metadata = read_triplet(paths)
                # Crop native GLT grid to AOI, then restrict each native pixel to saved train.
                from rasterio.windows import from_bounds

                window = from_bounds(*area.bbox, transform=transform)
                c0, r0 = max(0, int(np.floor(window.col_off))), max(0, int(np.floor(window.row_off)))
                c1 = min(reason.shape[1], int(np.ceil(window.col_off + window.width)))
                r1 = min(reason.shape[0], int(np.ceil(window.row_off + window.height)))
                if c1 <= c0 or r1 <= r0:
                    raise ValueError("Selected scene has no native AOI intersection")
                selection = (slice(r0, r1), slice(c0, c1))
                native_transform = window_transform(Window(c0, r0, c1 - c0, r1 - r0), transform)
                native_reason = reason[selection].copy()
                native_orthos = {k: v[selection].copy() for k, v in orthos.items()}
                yy, xx = np.indices(native_reason.shape)
                lons = native_transform.c + (xx + .5) * native_transform.a
                lats = native_transform.f + (yy + .5) * native_transform.e
                x, y = Transformer.from_crs(4326, grid.crs, always_xy=True).transform(lons, lats)
                row, col = grid.rowcol(x, y)
                native_allowed = (row >= 0) & (row < grid.height) & (col >= 0) & (col < grid.width)
                native_allowed[native_allowed] &= train[row[native_allowed], col[native_allowed]]
                w, s, e, n = area.bbox
                native_allowed &= (lons >= w) & (lons <= e) & (lats >= s) & (lats <= n)
                if mines:
                    geographic = [shapely_transform(Transformer.from_crs(grid.crs, 4326, always_xy=True).transform, g) for g in mines]
                    native_mine = rasterize([(g, 1) for g in geographic], out_shape=native_reason.shape,
                                            transform=native_transform).astype(bool)
                    native_reason[(native_reason == 4) & native_mine] = 3
                native_reason[~native_allowed] = np.nan
                scene_id = "_".join(scene["key"])
                scene_out = output / area.name / date / scene_id
                write_native(scene_out / "quality_reason.tif", native_reason, native_transform)
                for name, values in native_orthos.items():
                    values[~native_allowed] = np.nan
                    write_native(scene_out / f"{name}.tif", values, native_transform)
                metadata["uncertainty_summary"] = uncertainty_summary(native_orthos, native_reason == 4)
                write_json(scene_out / "metadata.json", metadata)
                scene_reason = warp(native_reason, native_transform, 4326, fine)
                scene_reason[~np.isfinite(scene_reason) | ~allowed] = 0
                scene_reason[(scene_reason == 4) & mine] = 3
                ids = [warp(native_orthos[f"group_{g}_mineral_id"], native_transform, 4326, fine) for g in (1, 2)]
                replace = scene_reason > date_reason
                date_reason[replace] = scene_reason[replace]
                for target, values in zip(date_ids, ids):
                    target[replace] = values[replace]
                stats = save_maps(scene_out / "grid_1km", scene_reason, ids)
                summary["scenes"].append({"area": area.name, "date": date, "key": scene["key"],
                                          "cell_area_equivalents": stats, "uncertainty": metadata["uncertainty_summary"],
                                          "glt_raw_center_distance_m": metadata["glt_raw_center_distance_m"],
                                          "native_pixel_counts": {str(code): int((native_reason == code).sum()) for code in REASONS}})
            stats = save_maps(output / area.name / date / "grid_1km", date_reason, date_ids)
            dates_count += quadrature(date_reason == 4) > 0
            summary["dates"].append({"area": area.name, "date": date, "cell_area_equivalents": stats})
            replace = date_reason > all_reason
            all_reason[replace] = date_reason[replace]
            for target, values in zip(all_ids, date_ids):
                target[replace] = values[replace]
        stats = save_maps(output / area.name / "combined", all_reason, all_ids)
        dates_count[~coarse_allowed] = np.nan
        write_layer(output / area.name / "combined/usable_date_count.tif", coarse, dates_count)
        coarse.save(output / area.name / "grid.json")
        summary["areas"].append({"area": area.name, "grid": asdict(coarse), "cell_area_equivalents": stats})
        # Verify every exported 1 km map against the saved train mask, including zero-valued counts.
        for path in (output / area.name).rglob("*.tif"):
            if "grid_1km" not in path.parts and "combined" not in path.parts:
                continue
            with rasterio.open(path) as dataset:
                values = dataset.read(1)
                if not np.isnan(values[~train[rs, cs]]).all():
                    raise ValueError("Final result escaped saved train mask")
    write_json(manifest, {"spec": spec, "summary": summary, "holdout_evaluated": False,
                          "files": {p.relative_to(output).as_posix(): d.file(p)
                                    for p in sorted(output.rglob("*")) if p.is_file() and p != manifest and ".part" not in p.name}})
    write_json(dest / "latest_prepare.json", {"directory": output.name, "sha256": d.file(manifest)})
    return output
