"""Предрегистрированный протокол слепого теста и проверка зафиксированных входов.

Проверка читает только метаданные: идентификаторы разбиения и прогона, хеш модели
и число объектов теста по статусам. Признаки и прогнозы закрытого региона не читаются.
"""

import csv
import json
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import Field, model_validator

from jerqazyna.config import ConfigModel
from jerqazyna.features.cache import Digests

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class FrozenModel(ConfigModel):
    run_dir: str
    run_id: Sha256
    model_sha256: Sha256
    classifier: Literal["logistic", "random_forest"]


class ProtocolObjects(ConfigModel):
    positive_statuses: list[Literal["deposit", "prospect"]] = Field(min_length=1)
    n_positive: int = Field(ge=1)
    uncovered_policy: Literal["miss"]


class Heuristic(ConfigModel):
    band: str
    nan_fill: Literal["train_median"]


class Benchmark(ConfigModel):
    layer: str


class Evaluation(ConfigModel):
    area_mask: Literal["test_cells"]
    min_valid_fraction: float = Field(gt=0, le=1)
    primary_area_fraction: float = Field(gt=0, lt=1)
    descriptive_area_fractions: list[float] = Field(default_factory=list)
    capture_targets: list[float] = Field(min_length=1)
    confidence: float = Field(gt=0, lt=1)
    heuristic: Heuristic
    benchmark: Benchmark

    @model_validator(mode="after")
    def check_fractions(self):
        values = self.descriptive_area_fractions + self.capture_targets
        if any(not 0 < value < 1 for value in values):
            raise ValueError("Доли площади и цели захвата должны лежать в (0, 1)")
        if self.primary_area_fraction in self.descriptive_area_fractions:
            raise ValueError("Основная доля площади не дублируется в описательных")
        return self


class Hypothesis(ConfigModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_]+$")
    role: Literal["primary", "secondary", "reference", "descriptive"]
    statement: str
    test: Literal["exact_binomial_one_sided", "exact_mcnemar_one_sided"] | None = None
    null_capture: float | None = Field(None, gt=0, lt=1)
    alpha: float | None = Field(None, gt=0, lt=1)
    threshold: float | None = Field(None, gt=0, lt=1)
    low_power: bool = False

    @model_validator(mode="after")
    def check_test(self):
        tested = self.role in ("primary", "secondary")
        if tested != (self.test is not None and self.alpha is not None):
            raise ValueError(f"{self.id}: тест и alpha задаются только для primary/secondary")
        if (self.test == "exact_binomial_one_sided") != (self.null_capture is not None):
            raise ValueError(f"{self.id}: null_capture нужен только биномиальному тесту")
        if (self.role == "reference") != (self.threshold is not None):
            raise ValueError(f"{self.id}: threshold задаётся только справочному критерию")
        return self


class Ties(ConfigModel):
    model: Literal["expected_capture"]
    hypothesis_tests: Literal["conservative"]


class BlindProtocol(ConfigModel):
    protocol_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    task_config: str
    split_id: Sha256
    region_provisional: bool
    frozen_model: FrozenModel
    objects: ProtocolObjects
    evaluation: Evaluation
    hypotheses: list[Hypothesis] = Field(min_length=1)
    ties: Ties

    @model_validator(mode="after")
    def check_hypotheses(self):
        ids = [h.id for h in self.hypotheses]
        if len(ids) != len(set(ids)):
            raise ValueError("Идентификаторы гипотез повторяются")
        if sum(h.role == "primary" for h in self.hypotheses) != 1:
            raise ValueError("Нужна ровно одна основная гипотеза")
        return self


def load_protocol(path: str | Path) -> tuple[BlindProtocol, str]:
    """Возвращает протокол и SHA-256 его файла: хеш записывается в отчёт теста."""
    path = Path(path)
    protocol = BlindProtocol.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    return protocol, Digests().file(path)


def verify_frozen_inputs(protocol: BlindProtocol, root: Path, split: Path) -> dict:
    """Сверяет протокол с сохранёнными разбиением и моделью до открытия региона."""
    root, split = Path(root), Path(split)
    saved_split = json.loads((split / "split.json").read_text(encoding="utf-8"))
    if saved_split["split_id"] != protocol.split_id:
        raise ValueError("Разбиение не совпадает с протоколом")
    if saved_split["provisional"] != protocol.region_provisional:
        raise ValueError("Статус границы региона не совпадает с протоколом")
    run_dir = root / protocol.frozen_model.run_dir
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if run["run_id"] != protocol.frozen_model.run_id:
        raise ValueError("Прогон модели не совпадает с протоколом")
    if run["holdout_evaluated"]:
        raise ValueError("Модель уже оценивалась на закрытом регионе")
    digests = Digests()
    if (run["files"]["model.joblib"] != protocol.frozen_model.model_sha256
            or digests.file(run_dir / "model.joblib") != protocol.frozen_model.model_sha256):
        raise ValueError("Хеш модели не совпадает с протоколом")
    cv = json.loads((run_dir / "cv.json").read_text(encoding="utf-8"))
    if cv["selected_classifier"] != protocol.frozen_model.classifier:
        raise ValueError("Классификатор прогона не совпадает с протоколом")
    with open(split / "labels.csv", encoding="utf-8", newline="") as stream:
        test = [row["status"] for row in csv.DictReader(stream) if row["partition"] == "test"]
    n_positive = sum(status in protocol.objects.positive_statuses for status in test)
    if n_positive != protocol.objects.n_positive:
        raise ValueError(f"В тесте {n_positive} положительных объектов, протокол ожидает "
                         f"{protocol.objects.n_positive}")
    return {"split_id": protocol.split_id, "run_id": protocol.frozen_model.run_id,
            "model_sha256": protocol.frozen_model.model_sha256, "n_positive": n_positive,
            "n_other_status": int(len(test) - n_positive)}
