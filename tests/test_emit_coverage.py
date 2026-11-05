import json
from types import SimpleNamespace

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("shapely")
pytest.importorskip("rasterio")

from jerqazyna.data.emit_coverage import (
    CoverageConfig,
    known_mine_constraints,
    pilot_label_coverage,
    support_summary,
)
from jerqazyna.data.emit_sampling import (
    archive_sizes,
    available_catalog_dates,
    candidate_tiles,
    choose_tiles,
    metadata_query,
    safe_bbox,
)
from jerqazyna.grid.template import GridSpec, write_layer


def config(**changes):
    return CoverageConfig(assessment_id="test", pilot="pilot.yaml", country_audit="audit", **changes)


def labels():
    return pd.DataFrame({"label_id": ["a", "b", "c", "d"], "fold": [0, 0, 1, 1],
                         "block": [1, 1, 2, 3], "baseline_positive": [True, True, True, False],
                         "country_iso_geometry": ["KAZ", "KAZ", "KGZ", "KGZ"],
                         "lon": [66.2, 66.6, 67.2, 67.6], "lat": [47.2] * 4,
                         "x": [.2, .6, 1.2, 1.6], "y": [.2] * 4})


def test_support_requires_distinct_blocks_and_does_not_count_unknown():
    summary = support_summary(labels(), [True] * 4, 2, 2, 2)
    assert not summary["passes_all_folds"]
    assert summary["folds"][0]["positive_labels"] == 2
    assert summary["folds"][0]["positive_blocks"] == 1
    assert summary["folds"][1]["positive_labels"] == 1
    assert summary["eligible_positive_labels"] == 3


def test_fixed_tiles_subdivide_only_when_closed_and_preserve_exclusions():
    frame = labels()
    areas, excluded = candidate_tiles(frame, 1, 1, lambda bbox: bbox[2] <= 66.5)
    assert [area.bbox for area in areas] == [(66., 47., 66.5, 47.5)]
    assert excluded == ["b", "c", "d"]


def test_safe_bbox_checks_touched_closed_cells_even_if_center_outside():
    grid = GridSpec("EPSG:4326", 1., 0., 2., 2, 2)
    train = np.ones(grid.shape, bool)
    closed = np.zeros(grid.shape, bool)
    closed[0, 1] = True
    assert safe_bbox((.1, .1, .8, .8), grid, train, closed)
    assert not safe_bbox((.8, 1.1, 1.1, 1.8), grid, train, closed)
    assert not safe_bbox((5., 5., 6., 6.), grid, train, closed)


def test_two_dates_are_distinct_not_two_overlapping_scenes():
    from jerqazyna.data.emit import PilotArea, PilotConfig

    def entry(day, kind):
        prefix = "L2A_MASK" if kind == "RFL" else "L2B_MIN"
        key = f"001_2025060{day}T000000_2222205_001"
        names = [f"EMIT_{prefix}_{key}.nc"]
        if kind == "MIN":
            names.append(f"EMIT_L2B_MINUNCERT_{key}.nc")
        return {"time_start": f"2025-06-0{day}T00:00:00Z", "polygons": [["47 66 48 66 48 67 47 67 47 66"]],
                "links": [{"href": "https://data.lpdaac.earthdatacloud.nasa.gov/test/" + name} for name in names],
                "id": kind + key}

    area = PilotArea(name="test", bbox=(66, 47, 67, 48))
    pilot = PilotConfig(pilot_id="test", task="task", baseline="run", split="split", areas=[area])
    catalog = {kind: [entry(1, kind), entry(1, kind)] for kind in ("MIN", "RFL")}
    assert available_catalog_dates(catalog, area, pilot, labels())["a"] == {"2025-06-01"}
    for kind, entries in catalog.items():
        entries.append(entry(2, kind))
    result = available_catalog_dates(catalog, area, pilot, labels())
    assert result["a"] == result["b"] == {"2025-06-01", "2025-06-02"}
    assert result["c"] == result["d"] == set()


def test_date_count_preserves_original_wgs84_scene_edges():
    from jerqazyna.data.emit import PilotArea, PilotConfig

    key = "001_20230604T000000_2222205_001"
    coords = "46.3699036 84.5759125 45.9154434 83.3911133 45.189888 83.6694183 45.6443481 84.8542175 46.3699036 84.5759125"
    def entry(kind, names):
        return {"id": kind, "time_start": "2023-06-04T00:00:00Z", "polygons": [[coords]],
                "links": [{"href": f"https://data.lpdaac.earthdatacloud.nasa.gov/test/EMIT_{name}_{key}.nc"} for name in names]}
    area = PilotArea(name="test", bbox=(84, 45, 85, 46))
    pilot = PilotConfig(pilot_id="test", task="task", baseline="run", split="split", areas=[area])
    catalog = {"MIN": [entry("min", ["L2B_MIN", "L2B_MINUNCERT"])], "RFL": [entry("rfl", ["L2A_MASK"])]}
    frame = pd.DataFrame({"label_id": ["one"], "lon": [84.4781], "lat": [45.5011]})
    assert available_catalog_dates(catalog, area, pilot, frame)["one"] == {"2023-06-04"}


def test_archive_sizes_excludes_rfl_cube_and_requires_exact_positive_bytes():
    name = "EMIT_L2A_MASK_001_20250601T000000_2222205_001.nc"
    item = {"meta": {"concept-id": "G1", "revision-id": 1},
            "umm": {"DataGranule": {"ArchiveAndDistributionInformation": [
                {"Name": name, "SizeInBytes": 123},
                {"Name": name.replace("MASK", "RFL"), "SizeInBytes": 2_000_000_000}]}}}
    assert archive_sizes([item])[name]["bytes"] == 123
    assert len(archive_sizes([item])) == 1
    item["umm"]["DataGranule"]["ArchiveAndDistributionInformation"][0]["SizeInBytes"] = None
    with pytest.raises(ValueError, match="exact per-file"):
        archive_sizes([item])


def test_public_metadata_cache_rejects_modified_payload(tmp_path, monkeypatch):
    import requests

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"feed": {"entry": []}}

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: Response())
    assert metadata_query(tmp_path, {"page_num": 1}) == {"feed": {"entry": []}}
    path = next(tmp_path.glob("*.json"))
    saved = json.loads(path.read_text())
    saved["payload"]["feed"]["entry"] = ["changed"]
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="cache changed"):
        metadata_query(tmp_path, {"page_num": 1})


def test_budget_reserves_retries_and_counts_duplicate_files_in_separate_pilots():
    c = config(minimum_positive_per_fold=2, minimum_blocks_per_fold=2, max_tiles=2,
               transfer_fixed_reserve_bytes=10, transfer_reserve_fraction=.25)
    candidates = [{"area": {"name": "first"}, "scenes": [{"assets": [{"name": "shared"}]}],
                   "potential_label_ids": ["a", "b"]},
                  {"area": {"name": "second"}, "scenes": [{"assets": [{"name": "shared"}]}],
                   "potential_label_ids": ["c"]}]
    sizes = {"shared": {"bytes": 100}}
    chosen, covered, cost = choose_tiles(candidates, labels(), sizes, c, remaining=200)
    assert len(chosen) == 1 and cost == 100 and covered
    chosen, covered, cost = choose_tiles(candidates, labels(), sizes, c, remaining=260)
    assert len(chosen) == 2 and cost == 200 and covered == {"a", "b", "c"}
    assert choose_tiles(candidates, labels(), sizes, c, remaining=134)[0] == []


def test_label_coverage_distinguishes_outside_pilot_from_observed_mine_mask(tmp_path):
    grid = GridSpec("EPSG:4326", 1., 0., 2., 2, 2)
    area = SimpleNamespace(name="area", bbox=(0., 0., 2., 2.))
    pilot = SimpleNamespace(areas=[area])
    base = tmp_path / "area/combined"
    for name in ("aoi_train_area_fraction", "observed", "mine_mask", "usable", "aggregate_QA",
                 "invalid_geolocation_or_fill", "usable_date_count", "group_1_no_match_fraction", "group_2_no_match_fraction"):
        value = 1. if name in {"aoi_train_area_fraction", "observed", "mine_mask"} else 0.
        if "no_match" in name:
            value = np.nan
        write_layer(base / f"{name}.tif", grid, np.full(grid.shape, value, np.float32))
    native = tmp_path / "area/2025-06-01/scene/quality_reason.tif"
    write_layer(native, grid, np.full(grid.shape, 3, np.float32))
    frame = pd.DataFrame({"label_id": ["inside", "outside"], "lon": [.5, 3.], "lat": [.5, 3.],
                          "x": [.5, 3.], "y": [.5, 3.]})
    result = pilot_label_coverage(frame, pilot, tmp_path)
    assert result[0]["mine_fraction"] == 1 and result[0]["usable_fraction"] == 0
    assert result[0]["native_quality_reason"] == 3
    assert result[0]["no_match_g1_fraction"] is None
    assert result[1]["area"] == "outside_pilot" and result[1]["observed_fraction"] is None


def test_known_mine_constraint_uses_same_50m_grid_and_retains_exact_area(monkeypatch):
    from shapely.geometry import box

    from jerqazyna.data import emit_prepare

    monkeypatch.setattr(emit_prepare, "mine_geometries", lambda *args: [box(0, 0, 725, 1000)])
    grid = GridSpec("EPSG:6933", 1000., 0., 1000., 1, 1)
    frame = pd.DataFrame({"label_id": ["one"], "row": [0], "col": [0]})
    result = known_mine_constraints(frame, None, {}, grid)[0]
    assert result["mine_cell_fraction_exact"] == pytest.approx(.725)
    assert result["mine_cell_fraction_50m"] in {.70, .75}
    assert result["maximum_usable_cell_fraction_50m"] == pytest.approx(1 - result["mine_cell_fraction_50m"])


def test_plan_does_not_count_or_spend_for_excluded_coordinate_or_mine_target():
    frame = labels()
    frame["planning_target"] = [False, False, True, True]
    candidates = [{"area": {"name": "excluded"}, "scenes": [{"assets": [{"name": "one"}]}],
                   "potential_label_ids": ["a", "b"]},
                  {"area": {"name": "mixed"}, "scenes": [{"assets": [{"name": "one"}]}],
                   "potential_label_ids": ["a", "c"]}]
    chosen, covered, cost = choose_tiles(candidates, frame, {"one": {"bytes": 100}},
                                         config(transfer_fixed_reserve_bytes=0), remaining=1000)
    assert len(chosen) == 1 and chosen[0]["area"]["name"] == "mixed"
    assert covered == {"c"} and cost == 100
