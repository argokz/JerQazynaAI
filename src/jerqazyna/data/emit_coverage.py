"""Train-only EMIT support audit. Never reads model scores or changes training labels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml
from pydantic import Field

from jerqazyna.config import ConfigModel, load_task
from jerqazyna.data.emit import anchors, frozen_search, load_pilot, write_json
from jerqazyna.data.review import local_path, parent_labels, source_sha
from jerqazyna.features.cache import Digests, fingerprint
from jerqazyna.grid.template import GridSpec, read_layer


class CoverageConfig(ConfigModel):
    assessment_id: str = Field(pattern=r"^[a-z][a-z0-9_-]+$")
    pilot: str
    country_audit: str
    tile_degrees: float = Field(1., gt=0, le=2)
    boundary_subdivisions: int = Field(3, ge=0, le=4)
    minimum_positive_per_fold: int = Field(5, ge=2)
    minimum_blocks_per_fold: int = Field(2, ge=2)
    minimum_usable_fraction: float = Field(.5, gt=0, le=1)
    minimum_usable_dates: int = Field(2, ge=1, le=2)
    max_tiles: int = Field(16, ge=1, le=40)
    transfer_reserve_fraction: float = Field(.25, ge=.1, le=.75)
    transfer_fixed_reserve_bytes: int = Field(268435456, ge=0)


def load_coverage(path):
    return CoverageConfig.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def checked_manifest(directory, filename, digests):
    meta = read_json(directory / filename)
    for name, sha in meta["files"].items():
        if digests.file(directory / name) != sha:
            raise ValueError(f"Changed assessment input: {directory / name}")
    return meta


def train_context(config, root):
    import geopandas as gpd
    import pandas as pd
    import shapely

    root = Path(root).resolve()
    pilot = load_pilot(local_path(root, config.pilot))
    anchor = anchors(root, pilot)
    task = load_task(local_path(root, pilot.task))
    labels, dataset, run = parent_labels(task, root, local_path(root, pilot.baseline))
    labels = labels.fillna({"status": "unknown"}).copy()
    d = Digests()
    baseline = local_path(root, pilot.baseline)
    if d.file(baseline / "cv_fold.tif") != run["files"]["cv_fold.tif"]:
        raise ValueError("Baseline fold map changed")
    grid = GridSpec(**anchor["grid"])
    fold_map = read_layer(baseline / "cv_fold.tif")
    labels["fold"] = [int(fold_map[r, c]) if np.isfinite(fold_map[r, c]) else -1
                      for r, c in zip(labels.row, labels.col)]
    # Read only identifier columns, never any saved OOF predictions.
    if d.file(baseline / "oof_objects.csv") != run["files"]["oof_objects.csv"]:
        raise ValueError("Baseline object identifiers changed")
    identifiers = pd.read_csv(baseline / "oof_objects.csv", usecols=["row", "col", "fold", "block"])
    if identifiers.duplicated(["row", "col"]).any():
        raise ValueError("Ambiguous baseline object cell")
    block_lookup = {(int(r.row), int(r.col)): (int(r.fold), int(r.block)) for r in identifiers.itertuples()}
    labels["block"] = [block_lookup.get((r, c), (-1, -1))[1] for r, c in zip(labels.row, labels.col)]
    labels["baseline_positive"] = labels.status.isin(run["spec"]["experiment"]["positive_statuses"])
    if int(labels.baseline_positive.sum()) != len(identifiers):
        raise ValueError("Baseline positive selection changed")
    for row in labels[labels.baseline_positive].itertuples():
        if block_lookup[(row.row, row.col)][0] != row.fold:
            raise ValueError("Baseline object and raster folds disagree")
    country_dir = local_path(root, config.country_audit)
    audit = checked_manifest(country_dir, "audit.json", d)
    if audit["spec"]["dataset_id"] != dataset["dataset_id"]:
        raise ValueError("Country audit refers to another dataset")
    countries = pd.read_csv(country_dir / "labels_audit.csv", keep_default_na=False,
                            usecols=["row", "col", "country_iso_geometry", "country_issue"])
    labels = labels.merge(countries, on=["row", "col"], how="left", validate="one_to_one")
    if labels.country_iso_geometry.isna().any():
        raise ValueError("Incomplete country audit")
    # These tracts incorporate known deposits; they are diagnostic strata, not independent belts.
    tract_layer = next(layer for layer in task.layers if layer.name == "usgs_2010_tracts")
    tracts = gpd.read_file(local_path(root, tract_layer.source), layer=tract_layer.params["layer"],
                           columns=["Tract_ID", "Tract_name"])
    if tracts.crs is None or not tracts.geometry.is_valid.all():
        raise ValueError("Invalid diagnostic tract geometry")
    tracts = tracts.to_crs(grid.crs)
    points = shapely.points(labels.x, labels.y)
    membership = [shapely.covers(geom, points) for geom in tracts.geometry]
    labels["usgs_tracts_diagnostic"] = ["|".join(sorted(str(tracts.iloc[j].Tract_ID)
                                                   for j, mask in enumerate(membership) if mask[i])) or "outside_tracts"
                                          for i in range(len(labels))]
    labels["label_id"] = [fingerprint({"source_records": s, "row": int(r), "col": int(c)})[:16]
                          for s, r, c in zip(labels.source_records, labels.row, labels.col)]
    refs = {"anchors": anchor, "parent_labels_sha256": dataset["files"]["labels.csv"],
            "country_audit_sha256": d.file(country_dir / "audit.json"),
            "tract_source_sha256": source_sha(root, tract_layer.source, d),
            "fold_sha256": d.file(baseline / "cv_fold.tif"),
            "object_identifiers_sha256": d.file(baseline / "oof_objects.csv")}
    return pilot, labels, grid, refs


def sample_cell(path, x, y):
    import rasterio

    with rasterio.open(path) as source:
        row, col = source.index(x, y)
        if not (0 <= row < source.height and 0 <= col < source.width):
            return None
        value = float(next(source.sample([(x, y)]))[0])
    return value if np.isfinite(value) else None


def support_summary(labels, eligible, n_folds, min_positive, min_blocks):
    eligible = np.asarray(eligible, bool) & labels.baseline_positive.to_numpy(bool)
    folds = []
    for fold in range(n_folds):
        rows = labels[eligible & labels.fold.eq(fold).to_numpy()]
        blocks = set(rows.block) - {-1}
        folds.append({"fold": fold, "positive_labels": len(rows), "positive_blocks": len(blocks),
                      "passes_minimum": len(rows) >= min_positive and len(blocks) >= min_blocks})
    return {"folds": folds, "passes_all_folds": all(r["passes_minimum"] for r in folds),
            "eligible_positive_labels": int(eligible.sum()),
            "note": "Operational feasibility floor, not statistical power or evidence of predictive value"}


def known_mine_constraints(labels, root, anchor, grid):
    """Area ceiling from existing mine polygons, before spending bytes on EMIT."""
    import shapely
    from rasterio.features import rasterize

    from jerqazyna.data.emit_prepare import mine_geometries

    result = []
    for label in labels.itertuples():
        left = grid.x_min + label.col * grid.cell_size
        top = grid.y_max - label.row * grid.cell_size
        cell = GridSpec(grid.crs, grid.cell_size, left, top, 1, 1)
        polygons = mine_geometries(root, anchor, grid.crs, cell.bounds)
        if polygons:
            shape = shapely.box(*cell.bounds)
            exact = shapely.intersection(shapely.union_all(polygons), shape).area / grid.cell_size**2
            fine = cell.refined(20)
            sampled = rasterize([(g, 1) for g in polygons], out_shape=fine.shape,
                                 transform=fine.transform).mean()
        else:
            exact = sampled = 0.
        result.append({"label_id": label.label_id, "mine_cell_fraction_exact": float(exact),
                       "mine_cell_fraction_50m": float(sampled),
                       "maximum_usable_cell_fraction_50m": 1 - float(sampled)})
    return result


def pilot_label_coverage(labels, pilot, prepared):
    fields = {"observed": "observed", "usable": "usable", "qa": "aggregate_QA", "mine": "mine_mask",
              "fill": "invalid_geolocation_or_fill", "no_match_g1": "group_1_no_match_fraction",
              "no_match_g2": "group_2_no_match_fraction"}
    rows = []
    for label in labels.itertuples():
        row = {"label_id": label.label_id, "area": "outside_pilot", "usable_date_count": 0,
               "native_quality_reason": None, **{f"{name}_fraction": None for name in fields}}
        for area in pilot.areas:
            w, s, e, n = area.bbox
            if not (w <= label.lon < e and s <= label.lat < n):
                continue
            row["area"] = area.name
            base = prepared / area.name / "combined"
            part = sample_cell(base / "aoi_train_area_fraction.tif", label.x, label.y)
            if part is None or part <= 0:
                raise ValueError("Train label inside AOI has no valid area cell")
            for name, filename in fields.items():
                value = sample_cell(base / f"{filename}.tif", label.x, label.y)
                row[f"{name}_fraction"] = (value / part if name not in {"no_match_g1", "no_match_g2"}
                                           else value) if value is not None else None
            row["no_observation_fraction"] = max(0., 1 - row["observed_fraction"] - row["fill_fraction"])
            row["usable_date_count"] = int(sample_cell(base / "usable_date_count.tif", label.x, label.y) or 0)
            reasons = [sample_cell(path, label.lon, label.lat)
                       for path in sorted((prepared / area.name).glob("*/*/quality_reason.tif"))]
            row["native_quality_reason"] = int(max([v for v in reasons if v is not None], default=0))
        rows.append(row)
    return rows


def grouped_support(labels):
    result = {}
    for field in ("fold", "country_iso_geometry", "usgs_tracts_diagnostic", "status"):
        result[field] = []
        for group, frame in labels.groupby(field, dropna=False):
            result[field].append({"group": str(group), "train_labels": len(frame),
                                  "baseline_positive": int(frame.baseline_positive.sum()),
                                  "inside_pilot": int(frame.area.ne("outside_pilot").sum()),
                                  "any_usable_area": int(frame.usable_fraction.fillna(0).gt(0).sum()),
                                  "primary_eligible_positive": int((frame.primary_eligible & frame.baseline_positive).sum())})
    return result


def assess(config, root):
    import pandas as pd

    root = Path(root).resolve()
    pilot, labels, grid, refs = train_context(config, root)
    dest, _ = frozen_search(root, pilot)
    pointer = read_json(dest / "latest_prepare.json")
    d = Digests()
    prepared = dest / pointer["directory"]
    if d.file(prepared / "manifest.json") != pointer["sha256"]:
        raise ValueError("Prepared pointer changed")
    checked_manifest(prepared, "manifest.json", d)
    spec = {"config": config.model_dump(), "inputs": refs, "prepared": pointer,
            "implementation_sha256": d.file(Path(__file__)),
            "mine_implementation_sha256": d.file(Path(__file__).with_name("emit_prepare.py"))}
    output = root / "data/diagnostics" / "porphyry_cu_balkhash" / config.assessment_id / fingerprint(spec)[:16]
    if (output / "manifest.json").exists():
        checked_manifest(output, "manifest.json", d)
        return output
    coverage = pd.DataFrame(pilot_label_coverage(labels, pilot, prepared))
    labels = labels.merge(coverage, on="label_id", validate="one_to_one")
    labels = labels.merge(pd.DataFrame(known_mine_constraints(labels, root, refs["anchors"], grid)),
                           on="label_id", validate="one_to_one")
    labels["preflight_mine_allows_primary"] = labels.maximum_usable_cell_fraction_50m.ge(config.minimum_usable_fraction)
    labels["planning_target"] = labels.preflight_mine_allows_primary & labels.country_issue.ne("mismatch")
    labels["primary_eligible"] = (labels.usable_fraction.fillna(0).ge(config.minimum_usable_fraction)
                                  & labels.usable_date_count.ge(config.minimum_usable_dates))
    n_folds = read_json(local_path(root, pilot.baseline) / "run.json")["spec"]["experiment"]["n_folds"]
    summary = {"train_labels": len(labels), "baseline_positive_labels": int(labels.baseline_positive.sum()),
               "inside_pilot": int(labels.area.ne("outside_pilot").sum()),
               "any_usable_labels": int(labels.usable_fraction.fillna(0).gt(0).sum()),
               "primary_support": support_summary(labels, labels.primary_eligible, n_folds,
                                                   config.minimum_positive_per_fold, config.minimum_blocks_per_fold),
               "groups": grouped_support(labels), "holdout_evaluated": False,
               "independent_geological_belts_available": False,
               "known_mask_support_ceiling": support_summary(labels, labels.planning_target, n_folds,
                                                              config.minimum_positive_per_fold, config.minimum_blocks_per_fold),
               "limitations": ["USGS tracts use known deposits and are diagnostic only",
                               "Cell area fractions differ from quality at the exact label coordinate",
                               "Thresholds fixed in the protocol; all scenarios retained without selection",
                               "Missing outside-pilot coverage is not an observed QA rejection",
                               "Country geometry is derived, not an independent coordinate validation",
                               "Coordinate/country mismatch excluded from spending target without changing its source record"],
               "scenarios": []}
    for fraction in (0., .25, .5, .75):
        for dates in (1, 2):
            eligible = labels.usable_fraction.fillna(0).gt(0) & labels.usable_fraction.fillna(0).ge(fraction)
            eligible &= labels.usable_date_count.ge(dates)
            summary["scenarios"].append({"minimum_fraction": fraction, "minimum_dates": dates,
                                         **support_summary(labels, eligible, n_folds, config.minimum_positive_per_fold,
                                                           config.minimum_blocks_per_fold)})
    output.mkdir(parents=True, exist_ok=True)
    # Do not publish source records or model predictions in the support table.
    columns = ["label_id", "name", "status", "lon", "lat", "row", "col", "fold", "block", "baseline_positive",
               "country_iso_geometry", "country_issue", "usgs_tracts_diagnostic", *coverage.columns.drop("label_id"),
               "mine_cell_fraction_exact", "mine_cell_fraction_50m", "maximum_usable_cell_fraction_50m",
               "preflight_mine_allows_primary", "planning_target",
               "primary_eligible"]
    labels[columns].to_csv(output / "labels_coverage.csv", index=False, encoding="utf-8")
    write_json(output / "summary.json", summary)
    lines = ["# EMIT: поддержка меток на train", "", f"Train: {len(labels)}, положительных baseline: {summary['baseline_positive_labels']}.",
             f"В AOI: {summary['inside_pilot']}; с ненулевой пригодной площадью: {summary['any_usable_labels']}.",
             "", "| Fold | Пригодных положительных | Пространственных блоков | Минимум пройден |",
             "|---|---:|---:|---|"]
    for row in summary["primary_support"]["folds"]:
        lines.append(f"| {row['fold']} | {row['positive_labels']} | {row['positive_blocks']} | {row['passes_minimum']} |")
    lines += ["", (f"Основной критерий: доля ≥{config.minimum_usable_fraction}, дат ≥{config.minimum_usable_dates}; "
              f"не менее {config.minimum_positive_per_fold} объектов и {config.minimum_blocks_per_fold} блоков в каждом fold."),
              "Это порог выполнимости, не расчёт статистической мощности. Обучений нет; закрытый регион не оценивался.", ""]
    lines.extend("- " + item for item in summary["limitations"])
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(output / "manifest.json", {"spec": spec, "files": {p.name: d.file(p) for p in output.iterdir() if p.is_file()}})
    home = output.parent
    write_json(home / "latest_assessment.json", {"directory": output.name, "sha256": d.file(output / "manifest.json")})
    return output
