"""Фиксируемый протокол первого эксперимента, отдельно от предметной задачи."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from jerqazyna.config import ConfigModel


class ForestSettings(ConfigModel):
    n_estimators: int = Field(500, ge=1)
    max_depth: int = Field(12, ge=1)
    min_samples_leaf: int = Field(5, ge=1)


class ExperimentConfig(ConfigModel):
    experiment_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    layers: list[str] = Field(min_length=1)
    positive_statuses: list[Literal["deposit", "prospect", "unknown"]] = Field(
        default=["deposit", "prospect"], min_length=1,
    )
    min_valid_fraction: float = Field(.5, gt=0, le=1)
    n_folds: int = Field(5, ge=2, le=20)
    cv_buffer_m: float = Field(10000, ge=0)
    background_cells: int = Field(50000, ge=10)
    background_seeds: list[int] = Field(default=[42, 137], min_length=1)
    background_exclusion_m: float = Field(5000, ge=0)
    models: list[Literal["logistic", "random_forest"]] = Field(
        default=["logistic", "random_forest"], min_length=1,
    )
    forest: ForestSettings = ForestSettings()
    logistic_c: float = Field(1.0, gt=0)
    workers: int = Field(2, ge=1, le=8)
    batch_size: int = Field(65536, ge=1)
    heuristic_band: str = "lithology__distance_km"
    grouped_bootstrap_iterations: int = Field(2000, ge=100)

    @model_validator(mode="after")
    def check_protocol(self):
        for values in (self.layers, self.models, self.background_seeds, self.positive_statuses):
            if len(values) != len(set(values)):
                raise ValueError("Списки протокола не должны содержать повторов")
        if any(seed < 0 for seed in self.background_seeds):
            raise ValueError("Seeds должны быть неотрицательными")
        return self


def load_experiment(path: str | Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
