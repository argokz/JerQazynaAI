from pathlib import Path

import pytest
from pydantic import ValidationError

from jerqazyna.cli import main
from jerqazyna.config import GridConfig, TaskConfig, ValidationConfig, load_task

TASK = Path(__file__).parents[1] / "configs" / "tasks" / "porphyry_cu_balkhash.yaml"


def test_task_config_loads():
    task = load_task(TASK)
    assert task.task_id == "porphyry_cu_balkhash"
    assert "usgs_2010_tracts" in [layer.name for layer in task.benchmark_layers]
    assert "usgs_2010_tracts" not in [layer.name for layer in task.feature_layers]


@pytest.mark.parametrize("crs", [
    "EPSG:4326", "+proj=utm +zone=43 +datum=WGS84", "EPSG:3857", "EPSG:32643",
    "+proj=lcc +lat_1=40 +lat_2=52 +lon_0=70 +datum=WGS84 +units=m", "not-a-crs",
])
def test_non_equal_area_crs_rejected(crs):
    with pytest.raises(ValidationError):
        GridConfig(crs=crs)


def test_cli_validate_config(capsys):
    assert main(["validate-config", str(TASK)]) == 0
    assert "porphyry_cu_balkhash" in capsys.readouterr().out


@pytest.mark.parametrize("crs", ["EPSG:6933", "EPSG:3035"])
def test_equal_area_epsg_accepted(crs):
    assert GridConfig(crs=crs).crs == crs


@pytest.mark.parametrize("fields", [
    {"area_fractions": [-1, .2]}, {"area_fractions": [1.1]}, {"area_fractions": []},
    {"success_area_fraction": 2}, {"success_area_fraction": float("nan")},
    {"confidence": float("inf")},
])
def test_invalid_validation_parameters_rejected(fields):
    with pytest.raises(ValidationError):
        ValidationConfig(blind_region={"name": "x", "geometry": "x"}, **fields)


@pytest.mark.parametrize("change", [
    {"task_id": "../escape"}, {"sead": 1},
    {"layers": [{"name": "x", "kind": "typo", "source": "x", "group": "terrain"}]},
    {"grid": {"cell_size_m": 1000, "cellsize_m": 2000}},
])
def test_unknown_fields_and_unsafe_names_rejected(change):
    data = load_task(TASK).model_dump()
    data.update(change)
    with pytest.raises(ValidationError):
        TaskConfig.model_validate(data)


@pytest.mark.parametrize("layer", [
    {"name": "x", "group": "terrain", "source": "x", "kind": "dem",
     "params": {"fine_factor": 0}},
    {"name": "x", "group": "terrain", "source": "x", "kind": "dem",
     "params": {"fine_fator": 10}},
    {"name": "x", "group": "geophysics", "source": "x", "kind": "magnetics"},
])
def test_invalid_processor_params_rejected(layer):
    data = load_task(TASK).model_dump()
    data["layers"] = [layer]
    with pytest.raises(ValidationError):
        TaskConfig.model_validate(data)
