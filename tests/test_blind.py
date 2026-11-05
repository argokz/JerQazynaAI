from pathlib import Path

import numpy as np
import pytest

from jerqazyna.validation.blind import area_for_capture, evaluate_blind
from jerqazyna.validation.protocol import load_protocol

PROTOCOL = Path(__file__).resolve().parents[1] / "configs/protocols/blind_balkhash_2026-11-05.yaml"
N_CELLS = 1000


@pytest.fixture
def protocol():
    return load_protocol(PROTOCOL)[0]


def objects_on_top(n=17):
    """Score растёт с индексом ячейки; объекты — в самых высоких ячейках."""
    scores = np.arange(N_CELLS, dtype=float)
    rows = np.arange(N_CELLS - n, N_CELLS)
    return scores, rows


def test_perfect_ranking_passes_all_hypotheses(protocol):
    scores, rows = objects_on_top()
    heuristic = -scores  # обратное ранжирование: эвристика промахивается
    tracts = np.zeros(N_CELLS, np.uint8)
    tracts[-100:] = 1
    result = evaluate_blind(scores, heuristic, tracts, rows, protocol)
    h = result["hypotheses"]
    assert h["H1_random"]["k"] == 17 and h["H1_random"]["passed"]
    assert h["H2_heuristic"]["model_only"] == 17 and h["H2_heuristic"]["passed"]
    assert h["R70_reference"]["passed"]
    assert h["D_usgs"]["tract_area_fraction"] == pytest.approx(.1)
    assert h["D_usgs"]["tract_capture"] == 1 and h["D_usgs"]["model_capture"] == 1
    assert result["verdict"] == "beats_random_and_heuristic"


def test_uncovered_objects_are_misses(protocol):
    scores, rows = objects_on_top()
    rows[:11] = -1  # остаётся 6 из 17: ниже порога H1 (нужно 7)
    result = evaluate_blind(scores, scores.copy(), np.zeros(N_CELLS), rows, protocol)
    assert result["n_uncovered"] == 11
    assert result["hypotheses"]["H1_random"]["k"] == 6
    assert not result["hypotheses"]["H1_random"]["passed"]
    assert result["verdict"] == "not_transferred"
    primary = next(m for m in result["model"]["capture"] if m["area_fraction"] == .2)
    assert primary["capture"] == pytest.approx(6 / 17)


def test_tied_heuristic_is_credited_conservatively(protocol):
    scores, rows = objects_on_top()
    heuristic = np.zeros(N_CELLS)  # все равны: каждому объекту дробное попадание 0,2
    result = evaluate_blind(scores, heuristic, np.zeros(N_CELLS), rows, protocol)
    h2 = result["hypotheses"]["H2_heuristic"]
    assert h2["model_only"] == 0 and h2["heuristic_only"] == 0 and not h2["passed"]
    assert result["verdict"] == "beats_random_not_heuristic"
    heuristic_primary = next(m for m in result["heuristic"]["capture"] if m["area_fraction"] == .2)
    assert heuristic_primary["capture"] == pytest.approx(.2)


def test_area_for_capture_interpolates_and_reports_unreachable():
    scores = np.arange(10, dtype=float)
    rows = np.array([9, 8, 0, -1])
    assert area_for_capture(scores, rows, .5) == pytest.approx(.2)
    assert area_for_capture(scores, rows, .8) is None  # 1 из 4 без покрытия
    assert area_for_capture(scores, np.array([-1, -1]), .5) is None


def test_object_count_must_match_protocol(protocol):
    scores, rows = objects_on_top(16)
    with pytest.raises(ValueError, match="протокол ожидает 17"):
        evaluate_blind(scores, scores, np.zeros(N_CELLS), rows, protocol)
