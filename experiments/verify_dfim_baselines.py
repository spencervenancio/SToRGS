"""Independent numerical checks of saved baseline scores and calibration."""
import importlib.util
import json

import numpy as np
import torch

from genomics.baselines import gradient_tile_scores
from genomics.paths import DATA_DIR, RESULTS_DIR


def main():
    import gzip

    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled = False
    out = RESULTS_DIR / 'genomics_dfim_baselines'
    model_dir = RESULTS_DIR / 'genomics_dfim_cnn'
    spec = importlib.util.spec_from_file_location('frozen', model_dir / 'cnn.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.SequenceCNN()
    model.load_state_dict(torch.load(model_dir / 'model.pt', weights_only=True))
    model.eval()
    saved = np.load(out / 'scores.npz')
    with gzip.open(DATA_DIR / 'dfim/sequences.fa.gz', 'rt') as f:
        seqs = f.read().splitlines()[1::2]
    lookup = {b: i for i, b in enumerate('ACGT')}
    tokens = np.array([[lookup[b] for b in seqs[i]] for i in saved['test_indices'][:256]])
    scores, _ = gradient_tile_scores(model, tokens)
    np.testing.assert_array_equal(scores, saved['test_scores'][:256])
    x = torch.from_numpy(np.eye(4, dtype=np.float32)[tokens[:8]].transpose(0, 2, 1).copy())
    x.requires_grad_(True)
    custom = model(x)
    native = model.network(x).squeeze(-1)
    g_custom, = torch.autograd.grad(custom.sum(), x)
    g_native, = torch.autograd.grad(native.sum(), x)
    torch.testing.assert_close(custom, native, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(g_custom, g_native, rtol=1e-4, atol=1e-5)
    curve = np.load(out / 'calibration_curve.npz')
    index = int(curve['estimated_false_fraction'].argmin())
    t = curve['threshold'][index]
    # Brute-force counts independently of sorted-search calibration.
    real_count = int((saved['validation_scores'] >= t).sum())
    control_count = int((saved['control_scores'] >= t).sum())
    estimate = min(1., (control_count + 1) * saved['validation_scores'].size
                   / saved['control_scores'].size / real_count)
    assert np.isclose(estimate, curve['estimated_false_fraction'][index])
    result = dict(recomputed_first_256_scores_bitwise_equal=True,
                  native_vs_custom_forward_max_difference=float((custom - native).abs().max().detach()),
                  native_vs_custom_gradient_max_difference=float((g_custom - g_native).abs().max()),
                  minimum_control_ratio_recomputed=estimate, calibration_count_check=True,
                  observed_tail_count=real_count, control_tail_count=control_count,
                  best_candidate_threshold=float(t))
    (out / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
