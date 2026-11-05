import json

import numpy as np
import pytest

gpd = pytest.importorskip("geopandas")
rasterio = pytest.importorskip("rasterio")
pytest.importorskip("sklearn")
import joblib
from rasterio.transform import from_origin
from shapely.geometry import box

from jerqazyna.cli import main
from jerqazyna.config import CENTRAL_ASIA_ALBERS, LayerConfig, TaskConfig
from jerqazyna.features.layers import PROCESSORS
from jerqazyna.features.prepare import prepare
from jerqazyna.grid.template import read_layer
from jerqazyna.models.config import ExperimentConfig
from jerqazyna.models.dataset import Dataset, build_dataset, map_cells
from jerqazyna.models.pipeline import predict_run, run_experiment
from jerqazyna.models.sampling import SpatialFolds, sample_background
from jerqazyna.validation.holdout import prepare_split


@pytest.fixture
def project(tmp_path):
    root = tmp_path
    boundaries = root / "data/raw/geoboundaries"
    boundaries.mkdir(parents=True)
    gpd.GeoDataFrame(geometry=[box(0, 0, 20000, 20000)], crs=CENTRAL_ASIA_ALBERS).to_crs(4326).to_file(
        boundaries / "geoBoundaries-AAA-ADM0.geojson", driver="GeoJSON",
    )
    gpd.GeoDataFrame(geometry=[box(16000, 0, 20000, 20000)], crs=CENTRAL_ASIA_ALBERS).to_crs(4326).to_file(
        root / "blind.geojson", driver="GeoJSON",
    )
    xy = [(x, y) for x in (1500, 5500, 9500, 13500) for y in (1500, 5500, 9500, 13500, 17500)]
    xy.extend([(18500, 8500), (2500, 2500)])
    gpd.GeoDataFrame(
        {"Name": [f"train-{i}" for i in range(20)] + ["holdout-secret", "unknown-status"],
         "SiteStatus": ["deposit"] * 20 + ["deposit", ""]},
        geometry=gpd.points_from_xy(*np.array(xy).T), crs=CENTRAL_ASIA_ALBERS,
    ).to_file(root / "labels.gpkg", driver="GPKG")
    source = root / "raw"
    source.mkdir()
    xx, yy = np.meshgrid((np.arange(20) + .5) * 1000, (19.5 - np.arange(20)) * 1000)
    data = (np.abs(xx - 7000) / 1000 + yy / 20000).astype(np.float32)
    data[2:4, :2] = np.nan
    data[:, 16:] = 1e9  # Наблюдения из test не должны попасть в fit или заполнение NaN.
    with rasterio.open(source / "distance.tif", "w", driver="GTiff", width=20, height=20,
                       count=1, crs=CENTRAL_ASIA_ALBERS, transform=from_origin(0, 20000, 1000, 1000),
                       dtype="float32", nodata=np.nan) as dst:
        dst.write(data, 1)
    task = TaskConfig.model_validate({
        "task_id": "toy", "deposit_type": "toy", "training_countries": ["AAA"],
        "labels": {"source": "labels.gpkg"},
        "layers": [{"name": "distance", "kind": "raster", "source": "raw", "group": "geology"}],
        "validation": {"blind_region": {"name": "test", "geometry": "blind.geojson",
                                         "buffer_m": 1000, "provisional": True},
                       "spatial_block_size_m": 4000},
    })
    experiment = ExperimentConfig(
        experiment_id="toy", layers=["distance"], heuristic_band="distance__value",
        n_folds=3, cv_buffer_m=200, background_cells=30, background_exclusion_m=100,
        background_seeds=[11, 29], forest={"n_estimators": 5, "max_depth": 3, "min_samples_leaf": 1},
        workers=1, batch_size=32, grouped_bootstrap_iterations=100,
    )
    prepare(task, root, log=lambda *_: None)
    split = prepare_split(task, root, log=lambda *_: None)
    return root, task, experiment, split


def test_dataset_excludes_holdout_and_preserves_coverage_population(project):
    root, task, config, split = project
    data = build_dataset(task, config, root, split, log=lambda *_: None)
    train = read_layer(split / "train.tif").ravel().astype(bool)
    assert train[data.cells].all()
    assert not data.labels.name.str.contains("holdout").any()
    assert data.X.max() < 1e9
    assert data.metadata["n_excluded_coverage"] > 0
    assert (data.labels.matrix_row < 0).any()
    assert build_dataset(task, config, root, split, log=lambda *_: None).path == data.path
    assert map_cells(np.array([2, 4, 7]), np.array([0, 2, 3, 7, 10])).tolist() == [-1, 0, -1, 2, -1]


def test_untrusted_roles_and_stale_sources_are_rejected(project):
    root, task, config, split = project
    task.layers.append(LayerConfig(name="leak", kind="raster", source="raw", group="geology", role="benchmark"))
    config.layers = ["leak"]
    with pytest.raises(ValueError, match="запрещена"):
        build_dataset(task, config, root, split, log=lambda *_: None)
    config.layers = ["distance"]
    with rasterio.open(root / "raw/distance.tif", "r+") as dst:
        values = dst.read(1)
        values[0, 0] = 1234
        dst.write(values, 1)
    with pytest.raises(ValueError, match="актуального слоя"):
        build_dataset(task, config, root, split, log=lambda *_: None)


def test_background_and_folds_respect_exclusions_and_are_reproducible(project):
    from scipy.spatial import cKDTree

    root, task, config, split = project
    config.background_exclusion_m = 1200
    data = build_dataset(task, config, root, split, log=lambda *_: None)
    folds = SpatialFolds(data, task.validation.spatial_block_size_m, config.n_folds, task.seed)
    for block in np.unique(folds.blocks):
        assert np.unique(folds.fold[folds.blocks == block]).size == 1
    for fold in range(config.n_folds):
        sampled = sample_background(data, config, folds, 42, fold)
        assert np.array_equal(sampled, sample_background(data, config, folds, 42, fold))
        assert len(np.unique(sampled)) == config.background_cells
        assert not np.isin(sampled, data.labels.matrix_row).any()
        assert (folds.fold[sampled] != fold).all()
        assert folds.allowed_xy(folds.xy(data.cells[sampled]), fold, config.cv_buffer_m).all()
        known = data.labels[["x", "y"]].to_numpy()
        known = known[folds.allowed_xy(known, fold, config.cv_buffer_m)]
        distances, _ = cKDTree(known).query(folds.xy(data.cells[sampled]))
        assert (distances > config.background_exclusion_m).all()


def test_complete_experiment_fit_preprocessing_only_on_fold_training_data(project):
    root, task, config, split = project
    run = run_experiment(task, config, root, split, log=lambda *_: None)
    metadata = json.loads((run / "run.json").read_text(encoding="utf-8"))
    report = json.loads((run / "cv.json").read_text(encoding="utf-8"))
    assert metadata["holdout_evaluated"] is False
    assert report["n_evaluated_objects"] < report["n_training_objects"]
    data = Dataset(root / metadata["dataset_path"])
    folds = SpatialFolds(data, task.validation.spatial_block_size_m, config.n_folds, task.seed)
    for sample_file in (run / "checkpoints").glob("logistic_*fold*.npz"):
        with np.load(sample_file) as sample:
            fit_rows = np.r_[sample["positive_rows"], sample["background_rows"]]
        fold = int(sample_file.stem.rsplit("fold", 1)[1])
        assert folds.allowed_xy(folds.xy(data.cells[fit_rows]), fold, config.cv_buffer_m).all()
        model = joblib.load(sample_file.with_suffix(".joblib"))
        np.testing.assert_allclose(model.named_steps["simpleimputer"].statistics_,
                                   np.nanmedian(data.X[fit_rows], axis=0))
    original = read_layer(run / "train_score.tif")
    test_mask = read_layer(split / "test.tif").astype(bool)
    assert np.isnan(original[test_mask]).all()
    repeat = root / "repeat.tif"
    predict_run(run, root, repeat, log=lambda *_: None)
    np.testing.assert_allclose(read_layer(repeat), original, equal_nan=True)
    assert run_experiment(task, config, root, split, log=lambda *_: None) == run
    assert main(["evaluate", str(run)]) == 0
    (run / "cv.json").write_text("{}", encoding="utf-8")
    assert main(["evaluate", str(run)]) == 1


def test_corrupt_table_is_not_reused(project):
    root, task, config, split = project
    data = build_dataset(task, config, root, split, log=lambda *_: None)
    (data.path / "labels.csv").write_text("corrupt", encoding="utf-8")
    with pytest.raises(ValueError, match="Повреждена"):
        Dataset(data.path)


def test_spectral_features_require_and_apply_mine_mask(project, monkeypatch):
    root, task, config, split = project
    task.layers[0].group = "spectral"
    task.layers.append(LayerConfig(name="mine", kind="mine_mask", source="raw", group="mask", role="mask"))

    def mask(layer, ctx):
        values = np.zeros(ctx.grid.shape, np.uint8)
        xs = ctx.grid.x_min + (np.arange(ctx.grid.width) + .5) * ctx.grid.cell_size
        values[:, xs > 8000] = 1
        return {"mask": values}

    monkeypatch.setitem(PROCESSORS, "mine_mask", mask)
    prepare(task, root, log=lambda *_: None)
    data = build_dataset(task, config, root, split, log=lambda *_: None)
    xs = data.grid.x_min + (data.cells % data.grid.width + .5) * data.grid.cell_size
    assert (xs < 8000).all()
    assert data.columns == ["distance__value"]
    assert data.metadata["coverage"]["distance__value"] < .8


def test_sensitivity_uses_fixed_population_and_paired_samples(project, monkeypatch):
    from jerqazyna.validation import sensitivity

    root, task, config, split = project
    task.layers.append(LayerConfig(name="aux", kind="raster", source="raw", group="geology"))
    config.layers.append("aux")
    config.models = ["random_forest"]
    # Обе группы меток должны присутствовать, чтобы проверить общую оценочную популяцию.
    import geopandas as gpd

    labels = gpd.read_file(root / "labels.gpkg")
    labels.loc[:7, "SiteStatus"] = "prospect"
    labels.to_file(root / "labels.gpkg", driver="GPKG")
    prepare(task, root, log=lambda *_: None)
    split = prepare_split(task, root, log=lambda *_: None)
    baseline = run_experiment(task, config, root, split, log=lambda *_: None)
    study = sensitivity.SensitivityConfig.model_validate({
        "study_id": "sensitivity", "variants": [
            {"name": "reference", "block_size_m": 4000, "layers": ["distance", "aux"],
             "positive_statuses": ["deposit", "prospect"]},
            {"name": "ablation", "block_size_m": 4000, "layers": ["aux"],
             "positive_statuses": ["deposit", "prospect"]},
            {"name": "deposit", "block_size_m": 4000, "layers": ["aux"],
             "positive_statuses": ["deposit"]},
        ],
    })
    dest = sensitivity.run_sensitivity(task, study, root, baseline, split, log=lambda *_: None)
    summary = json.loads((dest / "summary.json").read_text())
    a, b, c = summary["results"]
    assert summary["holdout_evaluated"] is False and summary["final_model_changed"] is False
    assert a["reused_baseline_oof"] and not b["reused_baseline_oof"]
    assert b["columns"] == ["aux__value"]
    assert len({v["n_cells"] for v in (a, b, c)}) == 1
    assert len({v["common_population"]["metrics"][-1]["n_deposits"] for v in (a, b, c)}) == 1
    assert c["n_fit_positive_cells"] < a["n_fit_positive_cells"]
    assert c["deposit_subgroup"]["metrics"][-1]["n_deposits"] == a["deposit_subgroup"]["metrics"][-1]["n_deposits"]
    for path in (dest / "ablation/checkpoints").glob("*.npz"):
        with np.load(path) as actual, np.load(baseline / "checkpoints" / path.name) as expected:
            np.testing.assert_array_equal(actual["positive_rows"], expected["positive_rows"])
            np.testing.assert_array_equal(actual["background_rows"], expected["background_rows"])
        model = joblib.load(path.with_suffix(".joblib"))
        assert model.named_steps["simpleimputer"].n_features_in_ == 1
    test = read_layer(split / "test.tif").astype(bool)
    assert np.isnan(read_layer(dest / "deposit/oof_score.tif")[test]).all()
    # Завершённая серия только проверяет данные и не вызывает новый fit.
    monkeypatch.setattr(sensitivity, "_fit", lambda *_: pytest.fail("повторный fit"))
    assert sensitivity.run_sensitivity(task, study, root, baseline, split, log=lambda *_: None) == dest
    (dest / "ablation/oof_score.tif").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Повреждены"):
        sensitivity.run_sensitivity(task, study, root, baseline, split, log=lambda *_: None)


def test_sensitivity_rejects_new_layers_before_training(project):
    from jerqazyna.validation.sensitivity import SensitivityConfig, run_sensitivity

    root, task, config, split = project
    baseline = run_experiment(task, config, root, split, log=lambda *_: None)
    study = SensitivityConfig.model_validate({"study_id": "invalid", "variants": [
        {"name": "leak", "block_size_m": 4000, "layers": ["benchmark"],
         "positive_statuses": ["deposit"]},
    ]})
    with pytest.raises(ValueError, match="исключать"):
        run_sensitivity(task, study, root, baseline, split, log=lambda *_: None)


def test_paired_interval_preserves_joint_model_and_heuristic_hits():
    from types import SimpleNamespace

    from jerqazyna.validation.sensitivity import paired_comparison

    folds = SimpleNamespace(n_folds=2, fold=np.repeat([0, 1], 4), blocks=np.repeat([10, 20], 4))
    task = SimpleNamespace(seed=42, validation=SimpleNamespace(area_fractions=[.5], confidence=.95))
    config = SimpleNamespace(grouped_bootstrap_iterations=1000)
    # Оба метода попадают в одни объекты: неопределённость разности должна быть нулевой,
    # хотя индивидуальные интервалы захвата широкие.
    result = paired_comparison(np.array([4., 3., 2., 1., 4., 3., 2., 1.]),
                               np.array([3., 4., 1., 2., 3., 4., 1., 2.]),
                               np.array([0, 2, 4, 6]), folds, task, config)[0]
    assert result["capture_difference"] == 0
    assert result["grouped_interval"]["lower"] == result["grouped_interval"]["upper"] == 0


def test_country_quota_capacities_and_unresolved_area_share():
    from jerqazyna.validation.countries import country_background, equal_quotas

    assert equal_quotas([2, 100, 100], 30).tolist() == [2, 14, 14]
    assert equal_quotas([0, 10], 5).tolist() == [0, 5]
    with pytest.raises(ValueError):
        equal_quotas([1, 2], 5)
    codes = np.repeat([0, 1, 2, 3], [10, 100, 100, 10]).astype(np.uint8)
    eligible = np.arange(len(codes))
    a = country_background(eligible, codes, 2, 55, 42, 0)
    np.testing.assert_array_equal(a, country_background(eligible, codes, 2, 55, 42, 0))
    assert len(np.unique(a)) == 55
    counts = np.bincount(codes[a], minlength=4)
    assert counts[1:3].tolist() == [25, 25]
    assert counts[0] + counts[3] == 5


def test_country_audit_preserves_labels_and_marks_overlaps_and_holdout(project):
    from jerqazyna.validation.countries import country_audit

    root, task, config, split = project
    gpd.GeoDataFrame(geometry=[box(6000, 0, 12000, 20000)], crs=CENTRAL_ASIA_ALBERS).to_crs(4326).to_file(
        root / "data/raw/geoboundaries/geoBoundaries-BBB-ADM0.geojson", driver="GeoJSON",
    )
    task.extra_countries = ["BBB"]
    prepare(task, root, log=lambda *_: None)
    split = prepare_split(task, root, log=lambda *_: None)
    data = build_dataset(task, config, root, split, log=lambda *_: None)
    before = (data.path / "labels.csv").read_bytes()
    dest = country_audit(task, data, root, {"AAA": "Country A", "BBB": "Country B"})
    summary = json.loads((dest / "summary.json").read_text())
    strata = np.load(dest / "strata.npy")
    assert summary["strata"]["3"]["n_cells"] > 0  # Явный overlap вместо приоритета файла.
    assert summary["strata"]["2"]["n_cells"] == 0
    assert set(np.unique(strata)) == {1, 3}
    assert summary["original_labels_changed"] is False
    assert (data.path / "labels.csv").read_bytes() == before
    mask = read_layer(split / "test.tif").astype(bool)
    assert np.isnan(read_layer(dest / "train_countries.tif")[mask]).all()
    assert country_audit(task, data, root, {"AAA": "Country A", "BBB": "Country B"}) == dest
    boundary = root / "data/raw/geoboundaries/geoBoundaries-BBB-ADM0.geojson"
    boundary.write_text(boundary.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="отличаются"):
        country_audit(task, data, root, {"AAA": "Country A", "BBB": "Country B"})


def test_country_pu_series_uses_fixed_tree_budget_and_fold_local_background(project, monkeypatch):
    from jerqazyna.validation import country_pu
    from jerqazyna.validation.countries import eligible_rows

    root, task, config, split = project
    config.models = ["random_forest"]
    baseline = run_experiment(task, config, root, split, log=lambda *_: None)
    study = country_pu.CountryPUConfig.model_validate({
        "study_id": "country_pu", "country_names": {"AAA": "Testland"}, "variants": [
            {"name": "reference", "sampling": "area", "seeds": [11, 29], "trees_per_member": 5},
            {"name": "balanced", "sampling": "country_equal", "seeds": [11, 29], "trees_per_member": 5},
            {"name": "bag_area", "sampling": "area", "seeds": [11, 29, 31, 37, 41], "trees_per_member": 2},
            {"name": "bag_country", "sampling": "country_equal", "seeds": [11, 29, 31, 37, 41], "trees_per_member": 2},
        ],
    })
    dest = country_pu.run_country_pu(task, study, root, baseline, split, log=lambda *_: None)
    report = json.loads((dest / "summary.json").read_text())
    assert report["holdout_evaluated"] is False and report["final_model_changed"] is False
    assert report["results"][0]["reused_baseline_oof"]
    data = Dataset(root / json.loads((baseline / "run.json").read_text())["dataset_path"])
    folds = SpatialFolds(data, task.validation.spatial_block_size_m, config.n_folds, task.seed)
    pools = {k: eligible_rows(data, config, folds, k) for k in range(config.n_folds)}
    n_objects = report["results"][0]["common_population"]["metrics"][-1]["n_deposits"]
    for result in report["results"]:
        assert result["common_population"]["metrics"][-1]["n_deposits"] == n_objects
        assert result["country_diagnostics"][-1]["countries"]["AAA"]["n_objects"] == n_objects
        assert result["variant"]["trees_per_member"] * len(result["variant"]["seeds"]) == 10
        test = read_layer(split / "test.tif").astype(bool)
        assert np.isnan(read_layer(dest / result["variant"]["name"] / "oof_score.tif")[test]).all()
    for path in (dest / "bag_country/checkpoints").glob("*.npz"):
        fold = int(path.stem.rsplit("fold", 1)[1])
        with np.load(path) as sample:
            assert np.isin(sample["background_rows"], pools[fold]).all()
            assert len(np.unique(sample["background_rows"])) == config.background_cells
        assert joblib.load(path.with_suffix(".joblib")).named_steps["randomforestclassifier"].n_estimators == 2
    monkeypatch.setattr(country_pu, "_fit", lambda *_: pytest.fail("повторный fit"))
    assert country_pu.run_country_pu(task, study, root, baseline, split, log=lambda *_: None) == dest
    (dest / "balanced/oof_objects.csv").write_text("corrupt")
    with pytest.raises(ValueError, match="Повреждены"):
        country_pu.run_country_pu(task, study, root, baseline, split, log=lambda *_: None)


def test_country_pu_rejects_unequal_ensemble_tree_budgets():
    from jerqazyna.validation.country_pu import CountryPUConfig

    with pytest.raises(ValueError, match="число деревьев"):
        CountryPUConfig.model_validate({"study_id": "bad", "country_names": {"AAA": "A"}, "variants": [
            {"name": "a", "sampling": "area", "seeds": [1], "trees_per_member": 500},
            {"name": "b", "sampling": "area", "seeds": [1, 2], "trees_per_member": 500},
        ]})
