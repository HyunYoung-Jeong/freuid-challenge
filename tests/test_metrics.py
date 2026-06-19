import numpy as np
import pytest

from src.metrics import compute_apcer_at_bpcer, compute_audet


def test_perfect_predictions():
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    scores = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    audet = compute_audet(labels, scores)
    assert audet < 0.01

    apcer = compute_apcer_at_bpcer(labels, scores, bpcer_target=0.01)
    assert apcer < 0.01


def test_worst_predictions():
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    scores = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    audet = compute_audet(labels, scores)
    assert audet > 0.9


def test_random_predictions():
    np.random.seed(42)
    labels = np.concatenate([np.zeros(500), np.ones(500)])
    scores = np.random.rand(1000)
    audet = compute_audet(labels, scores)
    assert 0.1 < audet < 0.9


def test_audet_lower_is_better():
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    good_scores = np.array([0.1, 0.1, 0.2, 0.1, 0.1, 0.9, 0.8, 0.9, 0.85, 0.9])
    bad_scores = np.array([0.4, 0.5, 0.6, 0.5, 0.4, 0.5, 0.6, 0.4, 0.5, 0.6])
    assert compute_audet(labels, good_scores) < compute_audet(labels, bad_scores)


def test_apcer_at_bpcer_returns_float():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.2, 0.3, 0.7, 0.8])
    result = compute_apcer_at_bpcer(labels, scores, bpcer_target=0.01)
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0
