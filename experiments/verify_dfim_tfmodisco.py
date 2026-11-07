"""Repeat upstream selection and attribution without changing model/settings."""
import gzip
import hashlib
import importlib.util
import json
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from genomics.paths import ROOT
from genomics.published_seqlets import load_upstream


def main():
    out = ROOT / 'results/genomics_dfim_tfmodisco'
    cfg = yaml.safe_load((out / 'config.yaml').read_text())
    sys.path.insert(0, str(ROOT / cfg['attribution_package_directory']))
    from tangermeme.deep_lift_shap import deep_lift_shap

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.mkldnn.enabled = False
    warnings.filterwarnings('ignore', message='max_unpooling2d_forward_out does not')
    coord_module, patches = load_upstream(ROOT / cfg['upstream_directory'], out / 'verification_figures')
    with h5py.File(out / 'native_seqlets.h5') as f:
        original = coord_module.CoordProducerResults.from_hdf5(f['results'])
    saved = np.load(out / 'attributions.npz')
    native_sums = coord_module.get_simple_window_sum_function(cfg['sliding'])(arrs=saved['signed_tracks'])
    for coord in original.coords:
        coord.score = native_sums[coord.example_idx][coord.start + cfg['flank']]
    settings = {key: cfg[key] for key in ['sliding', 'flank', 'suppress', 'target_fdr',
                'min_passing_windows_frac', 'max_passing_windows_frac', 'separate_pos_neg_thresholds', 'max_seqlets_total']}
    # Refit the original algorithm independently, not just reusing its thresholds.
    repeat = coord_module.FixedWindowAroundChunks(**settings, verbose=False)(
        score_track=saved['signed_tracks'], null_track=saved['null_signed_tracks'])
    def key(c):
        return (c.example_idx, c.start, c.end, c.score)
    assert [key(c) for c in original.coords] == [key(c) for c in repeat.coords]
    assert original.tnt_results.pos_threshold == repeat.tnt_results.pos_threshold
    assert original.tnt_results.neg_threshold == repeat.tnt_results.neg_threshold
    for name, (before, after) in patches.items():
        expected = before.replace('from sklearn.neighbors.kde import KernelDensity',
                                  'from sklearn.neighbors import KernelDensity')
        expected = expected.replace('os.makedirs("figures")', f'os.makedirs({str(out / "verification_figures")!r})')
        expected = expected.replace('("figures/scoredist_" +', f'({str(out / "verification_figures/scoredist_")!r} +')
        assert expected == after, name
    model_dir = ROOT / cfg['model_directory']
    spec = importlib.util.spec_from_file_location('frozen', model_dir / 'cnn.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.SequenceCNN()
    model.load_state_dict(torch.load(model_dir / 'model.pt', weights_only=True))
    model.network[4] = torch.nn.MaxPool1d(200)
    model.eval()

    class Output(torch.nn.Module):
        def __init__(self, predictor):
            super().__init__()
            self.predictor = predictor

        def forward(self, x):
            return self.predictor(x).unsqueeze(-1)

    with gzip.open(ROOT / cfg['data_directory'] / 'sequences.fa.gz', 'rt') as f:
        sequences = f.read().splitlines()[1::2]
    lookup = {b: i for i, b in enumerate('ACGT')}
    tokens = np.array([[lookup[b] for b in sequences[i]] for i in saved['test_indices'][:128]])
    x = torch.from_numpy(np.eye(4, dtype=np.float32)[tokens].transpose(0, 2, 1).copy())
    options = dict(references=torch.zeros(128, 1, 4, 200), n_shuffles=1,
                   batch_size=128, target=0, hypothetical=False, warning_threshold=1e-4, device='cpu')
    first = deep_lift_shap(Output(model), x, **options).sum(1).numpy()
    second = deep_lift_shap(Output(model), x, **options).sum(1).numpy()
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first, saved['signed_tracks'][:128])
    result = dict(native_refit_thresholds_equal=True, all_6371_native_coordinates_and_scores_equal=True,
                  runtime_changes_only_deprecated_import_and_plot_paths=True,
                  repeated_128_attributions_bitwise_equal=True,
                  repeated_attributions_match_saved=True,
                  verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
