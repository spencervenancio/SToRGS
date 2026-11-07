import json

import numpy as np

from genomics.paths import ROOT
from genomics.threshold_transfer import FEATURE_NAMES, observable_features, select_with_rule
from tests.fixtures.thesis_observable_features import (
    observable_features as thesis_observable_features,
)


def test_summaries_match_original_function_exactly():
    scores = np.random.default_rng(2).uniform(0, 5, (40, 10))
    scores[0] = 0
    a, b = observable_features(scores), thesis_observable_features(scores)
    for actual, expected in zip(a, b):
        np.testing.assert_array_equal(actual, expected)


def test_frozen_formula_matches_standardization_and_is_scale_invariant():
    path = ROOT / 'external/threshold_functions.json'
    rule = json.loads(path.read_text())['all_family']
    scores = np.random.default_rng(7).uniform(0, 10, (30, 10))
    selected, thresholds, _, features, _, _ = select_with_rule(scores, rule)
    coefficients = np.zeros(8)
    for name, value in zip(rule['feature_names'], rule['coefficients']):
        coefficients[FEATURE_NAMES.index(name)] = value
    mean, scale = np.array(rule['standardizer_mean']), np.array(rule['standardizer_scale'])
    standardized_intercept = rule['intercept'] + coefficients @ mean
    via_standardized = standardized_intercept + ((features - mean) / scale) @ rule['standardized_coefficients']
    np.testing.assert_allclose(thresholds, np.clip(via_standardized, 0, 1), atol=1e-14)
    again = select_with_rule(scores * 100, rule)
    np.testing.assert_array_equal(selected, again[0])
    np.testing.assert_allclose(thresholds, again[1], atol=1e-14)
    assert not select_with_rule(np.zeros((1, 10)), rule)[0].any()
