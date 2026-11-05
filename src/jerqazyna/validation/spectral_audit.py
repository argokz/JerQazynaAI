"""Read-only audit of saved spectral rasters; no cache repair or model training."""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

import numpy as np
import yaml
from pydantic import Field

from jerqazyna.config import ConfigModel, load_task
from jerqazyna.data.emit_coverage import (
    checked_manifest,
    read_json,
    support_summary,
    train_context,
)
from jerqazyna.data.review import local_path
from jerqazyna.features.cache import Digests, fingerprint, write_json
from jerqazyna.grid.template import write_layer

CHANNELS = {
    "sentinel2": ("albedo_swir22", "clay_ratio", "ferrous_ratio", "iron_oxide_ratio", "ndvi"),
    "aster_alteration": ("argillic_fraction", "phyllic_fraction", "silicic_fraction"),
}
REASONS = {1: "missing_input", 2: "mine_mask", 3: "missing_input_and_mine_mask", 4: "usable"}


class SpectralAuditConfig(ConfigModel):
    assessment_id: str = Field(pattern=r"^[a-z][a-z0-9_-]+$")
    pilot: str
    country_audit: str
    minimum_positive_per_fold: int = Field(5, ge=2)
    minimum_blocks_per_fold: int = Field(2, ge=2)


def load_audit(path):
    return SpectralAuditConfig.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def aligned(path, grid):
    import rasterio

    with rasterio.open(path) as src:
        if (src.count != 1 or (src.height, src.width) != grid.shape
                or src.crs != rasterio.crs.CRS.from_user_input(grid.crs)
                or not src.transform.almost_equals(grid.transform)):
            raise ValueError(f"Raster geometry differs from baseline: {path}")
        return src.read(1, masked=True).astype(np.float32).filled(np.nan)


def quality_states(observed, mine, train):
    """NaN outside train; zero feature values remain observations, never missing."""
    if observed.shape != train.shape or mine.shape != train.shape:
        raise ValueError("Mask shapes disagree")
    if not np.isin(mine[train], [0, 1]).all():
        raise ValueError("Missing or non-binary mine mask on train")
    out = np.full(train.shape, np.nan, dtype=np.float32)
    out[train] = np.where(observed[train], np.where(mine[train] > 0, 2, 4),
                          np.where(mine[train] > 0, 3, 1))
    return out


def bias_strata(density, train):
    """Fixed quartiles of all finite train cells, independent of labels and coverage."""
    values = density[train & np.isfinite(density)]
    if not len(values):
        raise ValueError("No finite exploration density on train")
    if np.any(values < 0):
        raise ValueError("Negative exploration density")
    cuts = np.quantile(values, [.25, .5, .75])
    strata = np.full(train.shape, -1, dtype=np.int8)
    valid = train & np.isfinite(density)
    strata[valid] = np.searchsorted(cuts, density[valid], side="right")
    return strata, cuts.tolist()


def counts(states, subset):
    total = int(subset.sum())
    result = {"cells": total}
    for code, name in REASONS.items():
        result[name + "_cells"] = int(((states == code) & subset).sum())
    result["observed_cells"] = result["mine_mask_cells"] + result["usable_cells"]
    result["observed_fraction"] = result["observed_cells"] / total if total else None
    result["usable_fraction"] = result["usable_cells"] / total if total else None
    return result


def input_inventory(base, names, cache, digests):
    result = {}
    for name in names:
        layer = name.split("__")[0]
        sha = digests.file(base / name)
        expected = cache.get(layer, {}).get("files", {}).get(name)
        if expected is not None and expected != sha:
            raise ValueError(f"Changed registered raster: {name}")
        result[name] = {"sha256": sha, "provenance": (
            "registered_file_hash_only" if expected else "legacy_unregistered"),
            "derivation_verified": False}
    return result


def label_groups(labels, column):
    result = {}
    for field in ("fold", "country_iso_geometry", "usgs_tracts_diagnostic", "status"):
        result[field] = [{"group": str(group), "labels": len(frame),
                          "baseline_positive": int(frame.baseline_positive.sum()),
                          "usable_labels": int(frame[column].eq(4).sum()),
                          "usable_positive": int((frame[column].eq(4) & frame.baseline_positive).sum())}
                         for group, frame in labels.groupby(field, dropna=False)]
    return result


def audit(config, root):
    root = Path(root).resolve()
    pilot, labels, grid, refs = train_context(config, root)
    task = load_task(local_path(root, pilot.task))
    base = root / "data/features" / task.task_id
    d = Digests()
    names = [f"{layer}__{band}.tif" for layer, bands in CHANNELS.items()
             for band in (*bands, "coverage")]
    names += ["mine_mask__mask.tif", "exploration_density__per_1000km2.tif"]
    inventory = input_inventory(base, names, read_json(base / "cache.json")["layers"], d)
    code = {name: d.file(Path(__file__).parents[1] / name) for name in (
        "validation/spectral_audit.py", "data/emit_coverage.py", "data/emit.py",
        "data/review.py", "features/cache.py", "features/layers.py", "grid/template.py")}
    spec = {"config": config.model_dump(), "context": refs, "rasters": inventory,
            "cache_manifest_sha256": d.file(base / "cache.json"),
            "inventory_sha256": d.file(base / "layers.json"), "code": code,
            "packages": {name: version(name) for name in ("numpy", "pandas", "rasterio", "shapely")}}
    output = root / "data/diagnostics" / task.task_id / config.assessment_id / fingerprint(spec)[:16]
    if (output / "manifest.json").exists():
        checked_manifest(output, "manifest.json", d)
        return output
    split = local_path(root, pilot.split)
    train = aligned(split / "train.tif", grid) > 0
    closed = (aligned(split / "test.tif", grid) > 0) | (aligned(split / "buffer.tif", grid) > 0)
    if np.any(train & closed) or not train.any():
        raise ValueError("Invalid train isolation")
    row, col = labels.row.to_numpy(int), labels.col.to_numpy(int)
    if not train[row, col].all():
        raise ValueError("Label outside saved train")
    mine = aligned(base / "mine_mask__mask.tif", grid)
    density = aligned(base / "exploration_density__per_1000km2.tif", grid)
    strata, cuts = bias_strata(density, train)
    country_dir = local_path(root, config.country_audit)
    countries = aligned(country_dir / "train_countries.tif", grid)
    country_names = read_json(country_dir / "summary.json")["strata"]
    if not np.isin(countries[train], [int(k) for k in country_names]).all():
        raise ValueError("Incomplete train country strata")
    fold_map = aligned(local_path(root, pilot.baseline) / "cv_fold.tif", grid)
    n_folds = read_json(local_path(root, pilot.baseline) / "run.json")["spec"]["experiment"]["n_folds"]
    if not np.isin(fold_map[train], range(n_folds)).all():
        raise ValueError("Incomplete baseline fold assignments")
    labels["mine_mask_legacy"] = mine[row, col]
    labels["exploration_density_legacy"] = density[row, col]
    labels["exploration_quartile"] = strata[row, col]
    summary = {"train_cells": int(train.sum()), "train_labels": len(labels),
               "baseline_positive_labels": int(labels.baseline_positive.sum()),
               "minimum_positive_per_fold": config.minimum_positive_per_fold,
               "minimum_blocks_per_fold": config.minimum_blocks_per_fold,
               "bias_quartile_cuts": cuts, "reason_codes": REASONS, "layers": {},
               "holdout_evaluated": False, "models_trained": False,
               "ready_for_training": False, "scientific_download_bytes": 0,
               "limitations": [
                   "Legacy raster hashes freeze current bytes; historical source/recipe linkage is unverified.",
                   "Reproducible preparation and cache registration required before any training.",
                   "Mine exclusion uses the existing binary 1 km mask (>0), not the EMIT 50 m area policy.",
                   "Finite ASTER zero means no mapped alteration within coverage, not missing input or proof of absence.",
                   "Sentinel-2 ratios are broad spectral proxies, not mineral IDs or concentrations.",
                   "Current S2 code uses SCL 4/5/6/7 and medians; saved per-pixel date counts/QA reasons absent.",
                   "ASTER is a USGS interpreted alteration map; coverage is not a pixel-level QA guarantee.",
                   "MRDS density and USGS tracts reflect study history; associations are descriptive, not causal.",
                   "Equal MRDS values stay together: repeated quartile cuts produce empty bins, never zero coverage.",
                   "Country comes from geometry; coordinate ambiguity and 20 unknown statuses remain.",
                   "The fold minimum is an operational floor, not statistical power or predictive value."]}
    output.mkdir(parents=True, exist_ok=True)
    joint = train.copy()
    maps = []
    for layer, bands in [*CHANNELS.items(), ("joint", ())]:
        observed = train.copy() if layer != "joint" else joint.copy()
        channel_stats = {}
        for band in bands:
            values = aligned(base / f"{layer}__{band}.tif", grid)
            valid = np.isfinite(values) & train
            observed &= valid
            channel_stats[band] = {"finite_train_cells": int(valid.sum()),
                                   "zero_train_cells": int((valid & (values == 0)).sum())}
            labels[f"{layer}__{band}"] = values[row, col]
        if layer != "joint":
            joint &= observed
            coverage = aligned(base / f"{layer}__coverage.tif", grid)
            labels[f"{layer}__coverage"] = coverage[row, col]
        states = quality_states(observed, mine, train)
        labels[f"{layer}__state"] = states[row, col].astype(int)
        support = support_summary(labels, labels[f"{layer}__state"].eq(4), n_folds,
                                  config.minimum_positive_per_fold, config.minimum_blocks_per_fold)
        grouped = {}
        for field, raster, groups in (
            ("country", countries, {int(k): v["name"] for k, v in country_names.items()}),
            ("fold", fold_map, {k: str(k) for k in range(n_folds)}),
            ("exploration_quartile", strata, {-1: "missing", **{k: str(k + 1) for k in range(4)}}),
        ):
            grouped[field] = [{"group": name, **counts(states, train & (raster == k))}
                              for k, name in groups.items()]
        summary["layers"][layer] = {**counts(states, train), "channels": channel_stats,
                                    "support": support, "cells_by": grouped,
                                    "labels_by": label_groups(labels, f"{layer}__state")}
        write_layer(output / f"{layer}__state.tif", grid, states, str(REASONS))
        if np.any(np.isfinite(states[~train])):
            raise ValueError("Finite spectral audit result outside train")
        maps.append((layer, states))
    columns = ["label_id", "name", "status", "row", "col", "fold", "block", "baseline_positive",
               "country_iso_geometry", "country_issue", "usgs_tracts_diagnostic", "mine_mask_legacy",
               "exploration_density_legacy", "exploration_quartile"]
    columns += [name for name in labels if "__" in name]
    labels[columns].to_csv(output / "labels_coverage.csv", index=False, encoding="utf-8")
    write_json(output / "summary.json", summary)
    render(output, maps, summary, labels)
    lines = ["# Sentinel-2 / ASTER: аудит сохранённых train-растров", "",
             "Хеши фиксируют существующие файлы; происхождение старого расчёта не подтверждено.",
             "Подключение к обучению отложено до воспроизводимой подготовки.", "",
             "| Слой | Наблюдаемые train | Пригодные train | Пригодные положительные по fold |",
             "|---|---:|---:|---|"]
    for layer, info in summary["layers"].items():
        folds = "/".join(str(f["positive_labels"]) for f in info["support"]["folds"])
        lines.append(f"| {layer} | {info['observed_fraction']:.2%} | {info['usable_fraction']:.2%} | {folds} |")
    lines += ["", "Состояния: 1 — нет входа; 2 — маска выработок; 3 — обе причины; 4 — пригодно.",
              "Доли относятся к числу ячеек train на равноплощадной сетке, а не доле чистых пикселей.",
              "Пригодность: все каналы слоя конечны и бинарная маска равна нулю; coverage не заменяет каналы.",
              "Страны, fold и квартили MRDS приведены в summary.json; все 72 метки — в labels_coverage.csv.",
              (f"Минимум: {config.minimum_positive_per_fold} положительных и "
               f"{config.minimum_blocks_per_fold} блока на fold; это не оценка мощности. Обучений и загрузок нет."),
              "", *["- " + item for item in summary["limitations"]]]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(output / "manifest.json", {"spec": spec, "files": {
        p.name: d.file(p) for p in sorted(output.iterdir()) if p.is_file()}})
    write_json(output.parent / "latest_assessment.json", {
        "directory": output.name, "sha256": d.file(output / "manifest.json")})
    return output


def render(output, maps, summary, labels):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    cmap = ListedColormap(["#d9d9d9", "#d55e00", "#882255", "#228833"])
    norm = BoundaryNorm([.5, 1.5, 2.5, 3.5, 4.5], 4)
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), constrained_layout=True)
    for ax, (name, states) in zip(axes, maps):
        image = ax.imshow(states, cmap=cmap, norm=norm, interpolation="nearest")
        ax.scatter(labels.col, labels.row, s=9, facecolors="none", edgecolors="black", linewidths=.4)
        ax.set_title(name)
        ax.set_axis_off()
    fig.colorbar(image, ax=axes, shrink=.65, ticks=[1, 2, 3, 4],
                 label="1 missing; 2 mine; 3 missing+mine; 4 usable; white outside train")
    fig.savefig(output / "train_coverage.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    n_folds = len(summary["layers"]["joint"]["support"]["folds"])
    for i, name in enumerate(summary["layers"]):
        info = summary["layers"][name]
        axes[0].bar(np.arange(n_folds) + (i - 1) * .24,
                    [f["positive_labels"] for f in info["support"]["folds"]], width=.24, label=name)
        groups = info["cells_by"]["exploration_quartile"][1:]
        axes[1].plot(range(1, 5), [100 * v["usable_fraction"] if v["cells"] else np.nan
                                  for v in groups], marker="o", label=name)
    axes[0].axhline(summary["minimum_positive_per_fold"], color="black", linestyle="--", linewidth=.8)
    axes[0].set(xlabel="Baseline spatial fold", ylabel="Usable positive labels", xticks=range(n_folds))
    axes[1].set(xlabel="MRDS density quartile (all train)", ylabel="Usable train cells (%)", xticks=range(1, 5))
    axes[1].set_title("Equal density values kept together; empty bins omitted", fontsize=10)
    axes[0].legend()
    axes[1].legend()
    fig.savefig(output / "support_and_study_bias.png", dpi=160)
    plt.close(fig)
