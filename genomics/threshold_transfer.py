"""Transfer a frozen mathematical-function saliency threshold, without refitting."""
import numpy as np

FEATURE_NAMES = ['q25', 'q50', 'q75', 'std', 'largest_gap', 'second_gap', 'top1_mass', 'top3_mass']


def observable_features(scores):
    """Match the original run_saliency_threshold_all_designs helper exactly."""
    normalized = scores / np.maximum(scores.max(axis=1, keepdims=True), 1e-12)
    rows = []
    for u in normalized:
        ordered = np.sort(u)[::-1]
        gaps = np.sort(ordered[:-1] - ordered[1:])[::-1]
        mass = u.sum()
        rows.append([np.quantile(u, .25), np.quantile(u, .50), np.quantile(u, .75),
                     u.std(), gaps[0], gaps[1] if len(gaps) > 1 else 0.,
                     u.max() / mass if mass > 0 else 0.,
                     ordered[:3].sum() / mass if mass > 0 else 0.])
    return np.asarray(rows), normalized


def frozen_thresholds(features, rule):
    """Exported coefficients already include standardization; do not scale twice."""
    indices = [FEATURE_NAMES.index(name) for name in rule['feature_names']]
    raw = rule['intercept'] + features[:, indices] @ np.asarray(rule['coefficients'])
    return np.clip(raw, rule['clip_lower'], rule['clip_upper']), raw


def select_with_rule(scores, rule):
    features, normalized = observable_features(np.asarray(scores, dtype=float))
    thresholds, raw = frozen_thresholds(features, rule)
    degenerate = scores.max(axis=1) <= 0
    selected = (normalized >= thresholds[:, None]) & ~degenerate[:, None]
    return selected, thresholds, raw, features, normalized, degenerate
