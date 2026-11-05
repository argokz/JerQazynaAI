from types import SimpleNamespace

import numpy as np
import pytest

from jerqazyna.models.evaluation import evaluate_oof, grouped_interval


def test_cv_area_is_ranked_within_each_fold_despite_different_score_scales():
    folds = SimpleNamespace(n_folds=2, fold=np.array([0, 0, 1, 1]), blocks=np.array([0, 0, 1, 1]))
    # Оба объекта ниже верхних 50% своего fold. Глобальная сортировка ошибочно
    # поставила бы объект со score=99 выше всех ячеек второго fold.
    result = evaluate_oof(np.array([100., 99., 2., 1.]), np.array([1, 3]), folds,
                          [.5], .95, 1000, 42)
    assert result["metrics"][0]["capture"] == 0
    assert result["metrics"][0]["expected_hits"] == 0
    assert all(fold["capture"] == 0 for fold in result["folds"])


def test_group_bootstrap_resamples_joint_hits_of_objects_in_same_block():
    result = grouped_interval(np.array([1., 1., 0., 0.]), np.array([10, 10, 20, 20]),
                              iterations=5000, confidence=.95, seed=42)
    assert result["n_positive_blocks"] == 2
    assert result["lower"] == 0 and result["upper"] == 1


def test_nonfinite_oof_cannot_be_silently_dropped():
    folds = SimpleNamespace(n_folds=2, fold=np.array([0, 0, 1, 1]), blocks=np.array([0, 0, 1, 1]))
    with pytest.raises(ValueError, match="Неполные"):
        evaluate_oof(np.array([1., 0., np.nan, 0.]), np.array([0, 3]), folds, [.5], .95, 100, 42)
