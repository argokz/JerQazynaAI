from pathlib import Path

import pytest
from pydantic import ValidationError

from jerqazyna.cli import main
from jerqazyna.config import GridConfig, load_task

TASK = Path(__file__).parents[1] / "configs" / "tasks" / "porphyry_cu_balkhash.yaml"


def test_task_config_loads():
    task = load_task(TASK)
    assert task.task_id == "porphyry_cu_balkhash"
    assert "usgs_2010_tracts" in [layer.name for layer in task.benchmark_layers]
    assert "usgs_2010_tracts" not in [layer.name for layer in task.feature_layers]


@pytest.mark.parametrize("crs", ["EPSG:4326", "+proj=utm +zone=43 +datum=WGS84", "EPSG:3857"])
def test_non_equal_area_crs_rejected(crs):
    with pytest.raises(ValidationError):
        GridConfig(crs=crs)


def test_cli_validate_config(capsys):
    assert main(["validate-config", str(TASK)]) == 0
    assert "porphyry_cu_balkhash" in capsys.readouterr().out
