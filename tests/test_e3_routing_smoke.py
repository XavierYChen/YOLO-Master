"""Unit tests for the dependency-light E3 smoke helpers."""

import pytest

from scripts import e3_routing_smoke as MODULE


@pytest.mark.parametrize(
    ("values", "entropy", "gini"),
    [
        ([0.25, 0.25, 0.25, 0.25], 1.0, 0.0),
        ([1.0, 0.0, 0.0, 0.0], 0.0, 1.0),
    ],
)
def test_balance_metrics_have_expected_limits(values, entropy, gini):
    assert MODULE.normalized_entropy(values) == pytest.approx(entropy)
    assert MODULE.normalized_gini(values) == pytest.approx(gini)


def test_family_filter_is_explicit():
    assert MODULE.family_matches("moe", "A2C2fMoE")
    assert MODULE.family_matches("mot", "C2fMoT")
    assert MODULE.family_matches("latent", "LatentMixture")
    assert not MODULE.family_matches("latent", "A2C2fMoE")
    assert not MODULE.family_matches("mot", "A2C2fMoE")


def test_nearest_rank_percentile():
    assert MODULE.percentile([4.0, 1.0, 3.0, 2.0], 0.5) == 2.0
    assert MODULE.percentile([4.0, 1.0, 3.0, 2.0], 0.95) == 4.0
