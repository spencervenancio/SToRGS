"""Matched attribution baselines; calibration estimates are not guarantees."""
import numpy as np
import torch


def dinucleotide_counts(tokens):
    return np.bincount(4 * tokens[:-1].astype(int) + tokens[1:], minlength=16)


def shuffle_dinucleotides(tokens, *, rng):
    """Shuffle successor order with each final edge pinned (DeepLIFT-style).

    Preserves dinucleotides/endpoints; not asserted uniform over all valid strings.
    """
    tokens = np.asarray(tokens, dtype=np.uint8)
    successors = []
    for base in range(4):
        following = np.flatnonzero(tokens[:-1] == base) + 1
        if len(following) > 1:
            following = np.concatenate([rng.permutation(following[:-1]), following[-1:]])
        successors.append(following)
    cursors = np.zeros(4, dtype=int)
    position = 0
    result = np.empty_like(tokens)
    result[0] = tokens[0]
    for j in range(1, len(tokens)):
        base = tokens[position]
        position = successors[base][cursors[base]]
        cursors[base] += 1
        result[j] = tokens[position]
    return result


def gradient_tile_scores(model, tokens, *, tile_width=20, batch_size=256):
    """Absolute observed-base gradients of scalar logit, summed within tiles."""
    model.eval()
    eye = np.eye(4, dtype=np.float32)
    scores, tracks = [], []
    for start in range(0, len(tokens), batch_size):
        x = torch.from_numpy(eye[tokens[start:start + batch_size]].transpose(0, 2, 1).copy())
        x.requires_grad_(True)
        logits = model(x)
        gradient, = torch.autograd.grad(logits.sum(), x)
        observed = (gradient * x).sum(1).abs().detach().numpy()
        tracks.append(observed)
        scores.append(observed.reshape(len(observed), -1, tile_width).sum(2))
    return np.concatenate(scores), np.concatenate(tracks)


def relative_selection(scores, fraction):
    maximum = scores.max(axis=1, keepdims=True)
    return (maximum > 0) & (scores >= fraction * maximum)


def pooled_threshold(observed, controls, target):
    """Lowest positive observed cutoff satisfying a scaled control-tail ratio."""
    observed, controls = np.sort(observed.ravel()), np.sort(controls.ravel())
    candidates = np.unique(observed[observed > 0])
    real_counts = len(observed) - np.searchsorted(observed, candidates, side='left')
    null_counts = len(controls) - np.searchsorted(controls, candidates, side='left')
    estimate = np.minimum(1., (null_counts + 1) * len(observed) / len(controls) / real_counts)
    passing = np.flatnonzero(estimate <= target)
    threshold = float(candidates[passing[0]]) if len(passing) else float('inf')
    return threshold, dict(threshold=candidates, observed_count=real_counts,
                           control_count=null_counts, estimated_false_fraction=estimate)


def sequence_metrics(selected, needed):
    r = selected.sum(1)
    tp = (selected & needed).sum(1)
    v = r - tp
    n_true = needed.sum(1)
    power = np.full(len(r), np.nan)
    np.divide(tp, n_true, out=power, where=n_true > 0)
    return dict(selected_count=r, true_positive_count=tp, false_positive_count=v,
                needed_count=n_true, fdp=v / np.maximum(r, 1), power=power)
