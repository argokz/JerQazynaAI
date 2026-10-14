import numpy as np
import pytest

from jerqazyna.validation import (
    bootstrap_capture,
    capture_at,
    capture_at_matched_area,
    detection_curve,
)


def test_perfect_model_captures_everything_in_top_area():
    scores = np.zeros(100)
    deposits = np.array([3, 17, 42])
    scores[deposits] = 1.0
    assert capture_at(scores, deposits, 0.05) == 1.0


def test_worst_model_captures_nothing():
    scores = np.ones(100)
    deposits = np.array([3, 17, 42])
    scores[deposits] = 0.0
    assert capture_at(scores, deposits, 0.20) == 0.0


def test_detection_curve_is_monotonic():
    rng = np.random.default_rng(0)
    scores = rng.random(1000)
    deposits = rng.choice(1000, 20, replace=False)
    area, captured = detection_curve(scores, deposits)
    assert np.all(np.diff(area) >= 0)
    assert captured[-1] == 1.0


def test_random_scores_capture_close_to_area_fraction():
    rng = np.random.default_rng(1)
    scores = rng.random(200_000)
    deposits = rng.choice(200_000, 5000, replace=False)
    assert capture_at(scores, deposits, 0.2) == pytest.approx(0.2, abs=0.02)


def test_bootstrap_interval_is_wide_for_few_deposits():
    scores = np.linspace(1, 0, 1000)
    # 15 объектов, 10 из них в верхних 20% площади
    deposits = np.r_[np.arange(0, 200, 20), np.arange(500, 1000, 100)]
    est = bootstrap_capture(scores, deposits, 0.2, iterations=4000, seed=0)
    assert est.n_deposits == 15
    assert est.capture == pytest.approx(10 / 15)
    assert est.lower < est.capture < est.upper
    assert est.upper - est.lower > 0.3
    assert not est.passes(0.70)


def test_matched_area_comparison():
    scores = np.zeros(100)
    deposits = np.array([1, 2, 3, 4])
    scores[[1, 2]] = 1.0
    benchmark = np.zeros(100, dtype=bool)
    benchmark[:50] = True  # широкие зоны: 50% площади, все объекты внутри
    area, bench_cap, model_cap = capture_at_matched_area(scores, deposits, benchmark)
    assert area == 0.5
    assert bench_cap == 1.0
    assert model_cap == 1.0


def test_empty_deposits_raise():
    with pytest.raises(ValueError):
        capture_at(np.zeros(10), np.array([], dtype=int), 0.2)
