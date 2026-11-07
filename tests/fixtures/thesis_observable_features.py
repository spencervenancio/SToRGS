"""Verbatim copy of `observable_features` from the senior-thesis repo.

Source: experiments/run_saliency_threshold_all_designs.py, thesis HEAD be45906
plus uncommitted working-tree edits; file sha256 3534ba9d2b0be0db754244fe271207fe418c79bfae05a9a957a4b2e827eca8ed.
It is the helper that produced external/threshold_functions.json. Kept as a frozen
reference so genomics.threshold_transfer can be checked against it. Do not edit.
"""
import numpy as np  # noqa: F401  (used by the copied function)


def observable_features(A):
    """Build observable summaries from normalized saliency maps."""
    U = A / np.maximum(A.max(axis=1, keepdims=True), 1e-12)
    rows = []
    for u in U:
        ordered = np.sort(u)[::-1]
        gaps = ordered[:-1] - ordered[1:]
        largest_gaps = np.sort(gaps)[::-1]
        mass = u.sum()
        rows.append([
            np.quantile(u, 0.25), np.quantile(u, 0.50), np.quantile(u, 0.75),
            u.std(), largest_gaps[0], largest_gaps[1] if len(gaps) > 1 else 0.0,
            u.max() / mass if mass > 0 else 0.0,
            ordered[:3].sum() / mass if mass > 0 else 0.0,
        ])
    return np.asarray(rows), U
