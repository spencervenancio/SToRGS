import numpy as np
import torch

from genomics.baselines import (
    dinucleotide_counts,
    gradient_tile_scores,
    pooled_threshold,
    relative_selection,
    sequence_metrics,
    shuffle_dinucleotides,
)


def test_shuffle_preserves_edges_endpoints_and_is_reproducible():
    for tokens in [np.zeros(200, dtype=np.uint8), np.arange(200, dtype=np.uint8) % 4,
                   np.random.default_rng(7).integers(0, 4, 200, dtype=np.uint8)]:
        a = shuffle_dinucleotides(tokens, rng=np.random.default_rng(8))
        b = shuffle_dinucleotides(tokens, rng=np.random.default_rng(8))
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(dinucleotide_counts(a), dinucleotide_counts(tokens))
        assert a[0] == tokens[0] and a[-1] == tokens[-1]
        np.testing.assert_array_equal(np.bincount(a, minlength=4), np.bincount(tokens, minlength=4))


def test_gradient_projection_and_tile_sum():
    class LinearScore(torch.nn.Module):
        def forward(self, x):
            return (x * torch.arange(1, 5).reshape(1, 4, 1)).sum((1, 2))

    tokens = np.tile(np.arange(200) % 4, (3, 1))
    scores, tracks = gradient_tile_scores(LinearScore(), tokens, batch_size=2)
    np.testing.assert_array_equal(scores, np.full((3, 10), 50))
    np.testing.assert_array_equal(tracks[0], tokens[0] + 1)


def test_relative_ties_zero_and_pooled_no_solution():
    actual = relative_selection(np.array([[0, 0], [100, 5], [100, 4]]), .05)
    np.testing.assert_array_equal(actual, [[False, False], [True, True], [True, False]])
    threshold, _ = pooled_threshold(np.array([1, 2, 3]), np.array([1, 2, 3]), .1)
    assert np.isinf(threshold)
    threshold, curve = pooled_threshold(np.array([1, 1, 2, 2]), np.zeros(100), .03)
    assert threshold == 1
    assert curve['observed_count'][0] == 4


def test_fdp_and_power_empty_and_all_null():
    values = sequence_metrics(np.array([[False, False], [True, False], [True, True]]),
                              np.array([[False, False], [False, False], [True, False]]))
    np.testing.assert_array_equal(values['fdp'], [0, 1, .5])
    assert np.isnan(values['power'][:2]).all()
    assert values['power'][2] == 1
