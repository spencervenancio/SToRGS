from types import SimpleNamespace

import numpy as np

from genomics.published_seqlets import seqlets_to_tiles


def test_mapping_uses_native_core_and_preserves_boundary_overlap():
    coord = SimpleNamespace(example_idx=0, start=9, end=50)
    mask = seqlets_to_tiles([coord], 2, flank=10)
    # Native [9,50), central [19,40): overlaps tiles0 and1, never tile2.
    np.testing.assert_array_equal(mask[0], [True, True] + [False] * 8)
    assert not mask[1].any()
