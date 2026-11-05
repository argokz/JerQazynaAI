"""Public CMR survey and budgeted expansion plan; no scientific downloads or training."""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from jerqazyna.data import emit
from jerqazyna.data.emit_coverage import checked_manifest, read_json, support_summary, train_context
from jerqazyna.features.cache import Digests, fingerprint


def safe_bbox(bbox, grid, train, closed):
    from rasterio.features import rasterize

    area = emit.PilotArea(name="candidate", bbox=bbox)
    geometry = emit.area_geometry(area, grid.crs)
    window = grid.window(geometry.bounds)
    if window is None:
        return False
    rows, cols = window
    mask = rasterize([(geometry, 1)], out_shape=train[window].shape,
                     transform=grid.window_transform(rows, cols), all_touched=True).astype(bool)
    return bool(np.any(mask & train[window]) and not np.any(mask & closed[window]))


def candidate_tiles(labels, step, subdivisions, allowed):
    """Fixed grid, halved only to avoid closed geometry; no model-dependent locations."""
    tiles, excluded = {}, []
    for label in labels.itertuples():
        for depth in range(subdivisions + 1):
            size = step / 2**depth
            west, south = math.floor(label.lon / size) * size, math.floor(label.lat / size) * size
            bbox = (west, south, west + size, south + size)
            if allowed(bbox):
                tiles[bbox] = emit.PilotArea(name="tile_" + fingerprint(bbox)[:12], bbox=bbox)
                break
        else:
            excluded.append(label.label_id)
    return sorted(tiles.values(), key=lambda a: a.bbox), excluded


def metadata_query(cache, params, umm=False):
    import requests

    endpoint = emit.CMR.replace(".json", ".umm_json") if umm else emit.CMR
    headers = {"Accept": "application/vnd.nasa.cmr.umm_results+json; version=1.6.6"} if umm else {}
    spec = {"endpoint": endpoint, "params": params, "headers": headers}
    path = cache / (fingerprint(spec)[:24] + ".json")
    if path.exists():
        saved = read_json(path)
        if saved["spec"] != spec or saved["payload_sha256"] != fingerprint(saved["payload"]):
            raise ValueError("Public CMR cache changed")
        return saved["payload"]
    response = requests.get(endpoint, params=params, headers=headers, timeout=(30, 120))
    response.raise_for_status()
    payload = response.json()
    emit.write_json(path, {"spec": spec, "payload": payload, "payload_sha256": fingerprint(payload)})
    return payload


def discover_area(cache, area, pilot):
    catalog = {}
    for kind, short_name in (("MIN", "EMITL2BMIN"), ("RFL", "EMITL2ARFL")):
        params = {"short_name": short_name, "version": "001", "page_size": 2000,
                  "bounding_box": ",".join(map(str, area.bbox)),
                  "temporal": f"{pilot.start}T00:00:00Z,{pilot.end}T23:59:59Z"}
        entries, page = [], 1
        while True:
            payload = metadata_query(cache, {**params, "page_num": page})
            batch = payload["feed"]["entry"]
            entries.extend(batch)
            if len(batch) < 2000:
                break
            page += 1
        if len({item["id"] for item in entries}) != len(entries):
            raise ValueError("Duplicate CMR pagination")
        catalog[kind] = entries
    return catalog


def archive_sizes(items):
    """Use per-file UMM-G bytes, never the total RFL granule (which contains full cubes)."""
    result = {}
    for item in items:
        for file in item["umm"].get("DataGranule", {}).get("ArchiveAndDistributionInformation", []):
            name = file.get("Name", "")
            if not emit.ASSET.fullmatch(name):
                continue
            size = file.get("SizeInBytes")
            if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
                raise ValueError(f"Missing exact per-file size: {name}")
            record = {"bytes": size, "checksum": file.get("Checksum"),
                      "cmr_id": item["meta"]["concept-id"],
                      "revision_id": item["meta"].get("revision-id"),
                      "metadata_version": item["umm"].get("MetadataSpecification", {}).get("Version")}
            if name in result and result[name] != record:
                raise ValueError("Conflicting per-file CMR metadata")
            result[name] = record
    return result


def available_catalog_dates(catalog, area, pilot, labels):
    """Diagnostic over every complete date, independent of the top-two-date selection."""
    import datetime as dt

    import shapely

    mask_keys = {key for entry in catalog["RFL"] for key, found in [emit.assets(entry)] if key and "MASK" in found}
    # Test original WGS84 edges directly. Projecting only polygon vertices bends
    # the represented edge and can misclassify points close to a scene boundary.
    points = shapely.points(labels.lon, labels.lat)
    region = shapely.box(*area.bbox)
    by_label = {label_id: set() for label_id in labels.label_id}
    for entry in catalog["MIN"]:
        date = entry["time_start"][:10]
        key, found = emit.assets(entry)
        if (key not in mask_keys or set(found) != {"MIN", "MINUNCERT"}
                or int(date[5:7]) not in pilot.months or not pilot.start <= dt.date.fromisoformat(date) <= pilot.end):
            continue
        polygons = []
        for group in entry.get("polygons", []):
            for coordinates in group:
                values = list(map(float, coordinates.split()))
                polygon = shapely.Polygon(list(zip(values[1::2], values[0::2])))
                if not polygon.is_valid:
                    raise ValueError("Invalid geographic CMR footprint")
                polygons.append(polygon)
        if not polygons:
            raise ValueError("Missing geographic CMR footprint")
        geometry = shapely.union_all(polygons).intersection(region)
        for label_id in labels.loc[shapely.covers(geometry, points), "label_id"]:
            by_label[label_id].add(date)
    return by_label


def guarded_cost(total_bytes, config):
    return math.ceil(total_bytes * (1 + config.transfer_reserve_fraction)) + config.transfer_fixed_reserve_bytes


def choose_tiles(candidates, labels, sizes, config, remaining):
    """Greedy coverage of fold deficits, then countries/unknowns, per charged byte."""
    chosen, covered, charged = [], set(), 0
    frame = labels.set_index("label_id")
    for _ in range(config.max_tiles):
        best = None
        for candidate in candidates:
            if candidate in chosen or not candidate["scenes"]:
                continue
            names = {a["name"] for s in candidate["scenes"] for a in s["assets"]}
            # Count a duplicate in another pilot again: reuse is not assumed by the budget.
            cost = sum(sizes[name]["bytes"] for name in names)
            if guarded_cost(charged + cost, config) > remaining:
                continue
            new = set(candidate["potential_label_ids"]) - covered
            if "planning_target" in frame:
                new = {key for key in new if bool(frame.loc[key, "planning_target"])}
            old = frame.loc[sorted(covered)] if covered else frame.iloc[:0]
            fresh = frame.loc[sorted(new)] if new else frame.iloc[:0]
            gain = 0.
            for fold in sorted(labels.fold.unique()):
                if fold < 0:
                    continue
                a = old[old.baseline_positive & old.fold.eq(fold)]
                b = fresh[fresh.baseline_positive & fresh.fold.eq(fold)]
                gain += min(config.minimum_positive_per_fold, len(a) + len(b)) - min(config.minimum_positive_per_fold, len(a))
                blocks = set(a.block) - {-1}
                more = set(b.block) - {-1}
                gain += .5 * (min(config.minimum_blocks_per_fold, len(blocks | more))
                               - min(config.minimum_blocks_per_fold, len(blocks)))
            gain += .25 * len(set(fresh.country_iso_geometry) - set(old.country_iso_geometry))
            gain += .1 * int((~fresh.baseline_positive).sum())
            if gain <= 0:
                continue
            rank = (gain / max(cost, 1), len(new), candidate["area"]["name"])
            if best is None or rank > best[0]:
                best = rank, candidate, cost
        if best is None:
            break
        _, selected, cost = best
        chosen.append(selected)
        targets = set(selected["potential_label_ids"])
        if "planning_target" in frame:
            targets = {key for key in targets if bool(frame.loc[key, "planning_target"])}
        covered |= targets
        charged += cost
    return chosen, covered, charged


def freeze_pilot(root, pilot, area, catalog, anchor):
    """Freeze the already-surveyed catalog through the same original selection protocol."""
    scenes, coverage, incomplete = emit.select_scenes(catalog["MIN"], catalog["RFL"], area, pilot, anchor["grid"]["crs"])
    data = {area.name: catalog}
    counts = emit.verify_areas(root, pilot, anchor)
    spec = {"protocol": pilot.recipe(), "anchors": anchor, "catalog_sha256": fingerprint(data),
            "selection_implementation_sha256": Digests().file(Path(emit.__file__)),
            "train_cells_all_touched": counts, "collections": ["EMITL2BMIN.001", "EMITL2ARFL.001"],
            "scenes": scenes, "coverage": coverage, "incomplete_cmr_ids": incomplete}
    sid = fingerprint(spec)
    directory = emit.pilot_home(root, pilot) / sid[:16]
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "search.json").exists():
        old = read_json(directory / "search.json")
        if old["spec"] != spec or Digests().file(directory / "catalog.json") != old["catalog_sha256"]:
            raise ValueError("Expanded frozen search changed")
    else:
        emit.write_json(directory / "catalog.json", data)
        emit.write_json(directory / "search.json", {"search_id": sid, "spec": spec,
                        "catalog_sha256": Digests().file(directory / "catalog.json")})
    emit.write_json(emit.pilot_home(root, pilot) / "latest_search.json",
                    {"path": directory.relative_to(root).as_posix(), "sha256": Digests().file(directory / "search.json")})
    return directory


def plan(config, root):
    import pandas as pd
    import yaml

    from jerqazyna.data.emit_coverage import assess
    from jerqazyna.grid.template import read_layer

    root = Path(root).resolve()
    assessment = assess(config, root)
    pilot, labels, grid, refs = train_context(config, root)
    constraints = pd.read_csv(assessment / "labels_coverage.csv", usecols=["label_id", "planning_target", "preflight_mine_allows_primary"])
    labels = labels.merge(constraints, on="label_id", validate="one_to_one")
    split = emit.local_path(root, pilot.split)
    train = read_layer(split / "train.tif") > 0
    closed = (read_layer(split / "test.tif") > 0) | (read_layer(split / "buffer.tif") > 0)
    areas, excluded = candidate_tiles(labels, config.tile_degrees, config.boundary_subdivisions,
                                      lambda bbox: safe_bbox(bbox, grid, train, closed))
    home = root / "data/pilots" / config.assessment_id
    cache = home / "public_cmr_cache"
    cache.mkdir(parents=True, exist_ok=True)
    catalogs = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        pending = {executor.submit(discover_area, cache, area, pilot): area.name for area in areas}
        for future in as_completed(pending):
            name = pending[future]
            catalogs[name] = future.result()
            print(f"CMR metadata: {len(catalogs)}/{len(areas)} areas", flush=True)
    candidates = []
    all_catalog_dates = {key: set() for key in labels.label_id}
    for area in areas:
        entries = catalogs[area.name]
        scenes, coverage, missing = emit.select_scenes(entries["MIN"], entries["RFL"], area, pilot, grid.crs)
        selected_ids = {scene["min_cmr_id"] for scene in scenes}
        selected_catalog = {"MIN": [entry for entry in entries["MIN"] if entry["id"] in selected_ids],
                            "RFL": entries["RFL"]}
        selected_dates = available_catalog_dates(selected_catalog, area, pilot, labels)
        potential = {key for key, values in selected_dates.items() if len(values) >= config.minimum_usable_dates}
        dates = available_catalog_dates(entries, area, pilot, labels)
        for label_id, values in dates.items():
            all_catalog_dates[label_id] |= values
        candidates.append({"area": area.model_dump(mode="json"), "scenes": scenes, "coverage": coverage,
                           "incomplete_cmr_ids": missing, "potential_label_ids": sorted(potential),
                           "target_label_ids": sorted(potential & set(labels.loc[labels.planning_target, "label_id"]))})
    ids = sorted({a["cmr_id"] for candidate in candidates for s in candidate["scenes"] for a in s["assets"]})
    items = []
    for start in range(0, len(ids), 40):
        batch = ids[start:start + 40]
        payload = metadata_query(cache, {"concept_id[]": batch, "page_size": 2000}, umm=True)
        current = payload["items"]
        if {item["meta"]["concept-id"] for item in current} != set(batch):
            raise ValueError("Incomplete UMM-G metadata batch")
        items.extend(current)
        print(f"UMM-G per-file sizes: {min(start + 40, len(ids))}/{len(ids)} granules", flush=True)
    sizes = archive_sizes(items)
    names = {a["name"] for c in candidates for s in c["scenes"] for a in s["assets"]}
    if names - sizes.keys():
        raise ValueError("Some scientific assets lack exact size metadata")
    ledger = read_json(root / "data/pilots/emit_transfer_ledger.json")
    remaining = pilot.budget_bytes - ledger["charged_bytes"]
    chosen, covered, charged = choose_tiles(candidates, labels, sizes, config, remaining)
    n_folds = read_json(emit.local_path(root, pilot.baseline) / "run.json")["spec"]["experiment"]["n_folds"]
    possible = set().union(*(set(c["potential_label_ids"]) for c in candidates))
    possible &= set(labels.loc[labels.planning_target, "label_id"])
    spec = {"config": config.model_dump(), "inputs": refs, "catalog_sha256": fingerprint(catalogs),
            "umm_sha256": fingerprint(items), "ledger_sha256": Digests().file(root / "data/pilots/emit_transfer_ledger.json"),
            "implementation_sha256": Digests().file(Path(__file__)),
            "coverage_implementation_sha256": Digests().file(Path(__file__).with_name("emit_coverage.py")),
            "assessment_manifest_sha256": Digests().file(assessment / "manifest.json")}
    output = home / ("plan_" + fingerprint(spec)[:16])
    if (output / "manifest.json").exists():
        checked_manifest(output, "manifest.json", Digests())
        return output
    output.mkdir(parents=True, exist_ok=True)
    emit.write_json(output / "catalogs.json", catalogs)
    emit.write_json(output / "umm_sizes.json", {name: sizes[name] for name in sorted(names)})
    emit.write_json(output / "candidates.json", candidates)
    emitted = []
    for rank, selected in enumerate(chosen, 1):
        area = emit.PilotArea.model_validate(selected["area"])
        extended = pilot.model_copy(update={"pilot_id": f"{config.assessment_id}_expand_{fingerprint(spec)[:8]}_{rank:02}",
                                            "areas": [area]})
        path = output / f"pilot_{rank:02}.yaml"
        path.write_text(yaml.safe_dump(extended.recipe(), sort_keys=False), encoding="utf-8")
        frozen = freeze_pilot(root, extended, area, catalogs[area.name], refs["anchors"])
        emitted.append({"config": path.relative_to(root).as_posix(), "search": frozen.relative_to(root).as_posix(),
                        "search_manifest_sha256": Digests().file(frozen / "search.json"),
                        "area": selected["area"], "potential_label_ids": selected["potential_label_ids"]})
    labels["potential_two_dates_any_candidate"] = labels.label_id.isin(possible)
    labels["potential_two_dates_selected"] = labels.label_id.isin(covered)
    labels["excluded_by_closed_geometry"] = labels.label_id.isin(excluded)
    labels["all_catalog_complete_dates"] = ["|".join(sorted(all_catalog_dates[key])) for key in labels.label_id]
    labels["all_catalog_complete_date_count"] = [len(all_catalog_dates[key]) for key in labels.label_id]
    columns = ["label_id", "name", "status", "lon", "lat", "fold", "block", "baseline_positive", "country_iso_geometry",
               "potential_two_dates_any_candidate", "potential_two_dates_selected", "excluded_by_closed_geometry",
               "all_catalog_complete_dates", "all_catalog_complete_date_count", "planning_target", "preflight_mine_allows_primary"]
    labels[columns].to_csv(output / "labels_geometric_support.csv", index=False, encoding="utf-8")
    summary = {"candidate_tiles": len(candidates), "selected_tiles": len(chosen), "pilot_configs": emitted,
               "candidate_positive_support": support_summary(labels, labels.label_id.isin(possible), n_folds,
                                                              config.minimum_positive_per_fold, config.minimum_blocks_per_fold),
               "selected_positive_support": support_summary(labels, labels.label_id.isin(covered), n_folds,
                                                             config.minimum_positive_per_fold, config.minimum_blocks_per_fold),
               "all_catalog_date_support_before_mine_mask": support_summary(labels,
                    labels.all_catalog_complete_date_count.ge(config.minimum_usable_dates), n_folds,
                    config.minimum_positive_per_fold, config.minimum_blocks_per_fold),
               "all_catalog_date_support_after_known_constraints": support_summary(labels,
                    labels.all_catalog_complete_date_count.ge(config.minimum_usable_dates) & labels.planning_target, n_folds,
                    config.minimum_positive_per_fold, config.minimum_blocks_per_fold),
               "scientific_downloads_this_stage": 0, "scientific_bytes_this_stage": 0,
               "estimated_file_bytes_with_cross_pilot_duplicates": charged,
               "estimated_bytes_with_transfer_reserve": guarded_cost(charged, config),
               "remaining_budget_before_plan": remaining, "shared_charged_bytes": ledger["charged_bytes"],
               "holdout_evaluated": False,
               "readiness": "geometric_candidate_only" if chosen else "no_candidate",
               "limitations": ["CMR geometry does not establish usable QA coverage or minimum cell fraction",
                               "Selected areas are label-enriched; not representative of territory-wide prevalence",
                               "Two distinct dates required at label coordinate; known mine ceiling applied, QA still unknown",
                               "USGS tract strata are not independent geological belts",
                               "No new training; labels and original baseline unchanged"],
               "selection_policy": "Greedy fold count deficit + 0.5 block deficit + 0.25 new country + 0.1 unknown, per file byte; fixed 1-degree tiles, boundary-only subdivision; tie by tile name"}
    if not summary["candidate_positive_support"]["passes_all_folds"]:
        summary["readiness"] = "infeasible_under_frozen_top_two_date_policy"
        summary["recommended_action"] = "defer_scientific_downloads_and_model_comparison"
    emit.write_json(output / "summary.json", summary)
    lines = ["# План расширения EMIT", "", f"Кандидатов: {len(candidates)}; выбрано: {len(chosen)}.",
             f"Файлы с повторами между пилотами: {charged} байт; с резервом: {guarded_cost(charged, config)} байт.",
             f"Остаток общего бюджета: {remaining} байт. Научные файлы на этапе планирования не загружались.",
             "", "| Fold | Потенциальных положительных на двух датах | Блоков | Минимум |", "|---|---:|---:|---|"]
    for row in summary["selected_positive_support"]["folds"]:
        lines.append(f"| {row['fold']} | {row['positive_labels']} | {row['positive_blocks']} | {row['passes_minimum']} |")
    lines += ["", "Геометрический охват — верхняя оценка доступности, не готовность к сравнению моделей.", ""]
    lines.extend("- " + item for item in summary["limitations"])
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    emit.write_json(output / "manifest.json", {"spec": spec,
                    "files": {p.name: Digests().file(p) for p in sorted(output.iterdir()) if p.is_file()}})
    emit.write_json(home / "latest_plan.json", {"directory": output.name, "sha256": Digests().file(output / "manifest.json")})
    return output
