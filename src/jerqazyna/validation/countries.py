"""Страны только для диагностики и фоновых квот; исходные метки и X не меняются."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from jerqazyna.features.cache import Digests, fingerprint, write_json
from jerqazyna.grid.template import write_layer


def country_audit(task, dataset, root: Path, names: dict[str, str]):
    import geopandas as gpd
    import shapely
    from pyproj import Transformer
    from rasterio.features import rasterize

    isos = sorted(task.training_countries + task.extra_countries)
    if set(names) != set(isos) or len({s.casefold() for s in names.values()}) != len(names):
        raise ValueError("Нужны уникальные имена всех стран задачи")
    digests = Digests()
    sources = {iso: root / "data/raw/geoboundaries" / f"geoBoundaries-{iso}-ADM0.geojson" for iso in isos}
    boundaries = {iso: digests.file(p) for iso, p in sources.items()}
    area = json.loads((root / "data/features" / task.task_id / "area.json").read_text(encoding="utf-8"))
    if boundaries != area["spec"]["boundaries"]:
        raise ValueError("Границы стран отличаются от использованных для train-маски")
    spec = {"dataset_id": dataset.metadata["dataset_id"], "boundaries": boundaries, "names": names,
            "implementation_sha256": digests.file(Path(__file__)),
            "policy": "cell_center; unresolved_and_overlap_explicit; train_only"}
    audit_id = fingerprint(spec)
    dest = root / "data/diagnostics" / task.task_id / "countries" / audit_id[:16]
    marker = dest / "audit.json"
    if marker.exists():
        saved = json.loads(marker.read_text(encoding="utf-8"))
        if saved["spec"] != spec or any(digests.file(dest / p) != h for p, h in saved["files"].items()):
            raise ValueError("Повреждён аудит стран")
        return dest
    geometries = []
    for iso in isos:
        frame = gpd.read_file(sources[iso])
        if frame.empty or frame.crs is None or not frame.geometry.is_valid.all():
            raise ValueError(f"Некорректная геометрия страны: {iso}")
        geometries.append(frame.to_crs(dataset.grid.crs).geometry.union_all())
    grid = dataset.grid
    code = np.zeros(grid.shape, np.uint8)
    count = np.zeros(grid.shape, np.uint8)
    for i, geometry in enumerate(geometries, 1):
        inside = rasterize([(geometry, 1)], out_shape=grid.shape, transform=grid.transform,
                           dtype="uint8", all_touched=False).astype(bool)
        code[inside] = i
        count += inside
    overlap = len(isos) + 1
    code[count > 1] = overlap
    strata = code.ravel()[dataset.cells]
    labels = dataset.labels.copy()
    x, y = Transformer.from_crs(4326, grid.crs, always_xy=True).transform(
        labels.lon.to_numpy(), labels.lat.to_numpy(),
    )
    points = shapely.points(x, y)
    membership = np.stack([shapely.covers(g, points) for g in geometries], axis=1)
    labels["country_candidates"] = ["|".join(np.array(isos)[row]) for row in membership]
    labels["country_iso_geometry"] = [isos[int(np.flatnonzero(row)[0])] if row.sum() == 1 else ""
                                      for row in membership]
    declared = labels.get("country", labels.name * 0).fillna("").astype(str).str.strip()
    labels["country_declared"] = declared
    lookup = {value.casefold(): iso for iso, value in names.items()} | {iso.casefold(): iso for iso in isos}
    labels["country_iso_declared"] = declared.str.casefold().map(lookup).fillna("")
    labels["country_issue"] = [
        "unresolved_geometry" if not geo else "missing_declared" if not raw else
        "unrecognized_declared" if not iso else "consistent" if geo == iso else "mismatch"
        for raw, iso, geo in zip(declared, labels.country_iso_declared, labels.country_iso_geometry)
    ]
    labels["coordinate_projection_error_m"] = np.hypot(x - labels.x, y - labels.y)
    rows, cols = grid.rowcol(x, y)
    labels["coordinate_cell_consistent"] = (rows == labels.row) & (cols == labels.col)
    labels["status_record_conflict"] = [
        len({str(r.get("status_raw", "")).strip().casefold() for r in json.loads(records)
             if str(r.get("status_raw", "")).strip().casefold() in {"deposit", "prospect"}}) > 1
        for records in labels.source_records
    ]
    labels["country_cell_stratum"] = [
        str(int(strata[int(row)])) if row >= 0 else "not_in_table" for row in labels.matrix_row
    ]
    code_names = {0: "UNASSIGNED", **dict(enumerate(isos, 1)), overlap: "OVERLAP"}
    counts = np.bincount(strata, minlength=overlap + 1)
    summary = {
        "n_train_cells": len(strata), "strata": {str(k): {"name": v, "n_cells": int(counts[k])}
                                                  for k, v in code_names.items()},
        "country_issues": labels.country_issue.value_counts().to_dict(),
        "geometry_country_counts": labels.country_iso_geometry.value_counts().to_dict(),
        "geometry_positive_counts": labels.loc[labels.status.isin(["deposit", "prospect"]),
                                                "country_iso_geometry"].value_counts().to_dict(),
        "status_counts": labels.status.value_counts().to_dict(),
        "status_sources": labels.groupby(["status", "status_source"], dropna=False).size().reset_index(
            name="n_objects").fillna("").to_dict("records"),
        "n_status_record_conflicts": int(labels.status_record_conflict.sum()),
        "max_coordinate_projection_error_m": float(labels.coordinate_projection_error_m.max()),
        "n_coordinate_cell_mismatches": int((~labels.coordinate_cell_consistent).sum()),
        "holdout_evaluated": False, "original_labels_changed": False,
    }
    dest.mkdir(parents=True, exist_ok=True)
    np.save(dest / "strata.npy", strata, allow_pickle=False)
    labels.to_csv(dest / "labels_audit.csv", index=False, encoding="utf-8")
    train_code = np.full(grid.width * grid.height, np.nan, np.float32)
    train_code[dataset.cells] = strata
    write_layer(dest / "train_countries.tif", grid, train_code.reshape(grid.shape), "country stratum; diagnostic; train only")
    write_json(dest / "summary.json", summary)
    write_json(marker, {"audit_id": audit_id, "spec": spec,
                        "files": {p.name: digests.file(p) for p in sorted(dest.iterdir()) if p != marker}})
    return dest


def eligible_rows(dataset, config, folds, fold):
    """Полный допустимый фон для расчёта квот, без использования validation-статусов."""
    from scipy.spatial import cKDTree

    xy = dataset.labels[["x", "y"]].to_numpy(dtype=float)
    xy = xy[folds.allowed_xy(xy, fold, config.cv_buffer_m)]
    tree = cKDTree(xy) if len(xy) else None
    known = dataset.labels.matrix_row.to_numpy(dtype=np.int64)
    selected = []
    for start in range(0, len(dataset.cells), 65536):
        rows = np.arange(start, min(start + 65536, len(dataset.cells)))
        rows = folds.training_rows(rows, fold, config.cv_buffer_m)
        rows = rows[~np.isin(rows, known)]
        if tree is not None:
            rows = rows[tree.query(folds.xy(dataset.cells[rows]))[0] > config.background_exclusion_m]
        selected.append(rows)
    return np.concatenate(selected)


def equal_quotas(capacities, n):
    capacities = np.asarray(capacities, dtype=np.int64)
    if n < 0 or n > capacities.sum() or (capacities < 0).any():
        raise ValueError("Недостаточная ёмкость стран для фоновой выборки")
    quotas = np.zeros_like(capacities)
    remaining = n
    while remaining:
        active = np.flatnonzero(quotas < capacities)
        share = max(remaining // len(active), 1)
        for i in active:
            take = min(share, int(capacities[i] - quotas[i]), remaining)
            quotas[i] += take
            remaining -= take
    return quotas


def country_background(eligible, strata, n_countries, size, seed, fold):
    """Равные квоты доступных стран; нерешённые ячейки сохраняют долю площади."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, fold + 1, 1]))
    codes = strata[eligible]
    unresolved = (codes == 0) | (codes == n_countries + 1)
    if len(eligible) < size:
        raise ValueError("Недостаточно допустимого фона")
    n_unresolved = int(np.rint(size * unresolved.mean()))
    capacities = np.bincount(codes, minlength=n_countries + 2)[1:n_countries + 1]
    quotas = equal_quotas(capacities, size - n_unresolved)
    samples = [rng.choice(eligible[unresolved], n_unresolved, replace=False)] if n_unresolved else []
    for code, quota in enumerate(quotas, 1):
        if quota:
            samples.append(rng.choice(eligible[codes == code], int(quota), replace=False))
    result = np.concatenate(samples)
    rng.shuffle(result)
    return result
