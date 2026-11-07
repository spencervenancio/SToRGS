"""Pinned BPNet-era TF-MoDISco seqlet selection on the existing DFIM CNN."""
import argparse
import csv
import difflib
import gzip
import hashlib
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import sklearn
import torch
import yaml

from experiments.run_dfim_baselines import aggregate
from genomics.baselines import dinucleotide_counts, sequence_metrics, shuffle_dinucleotides
from genomics.paths import ROOT
from genomics.published_seqlets import load_upstream, seqlets_to_tiles
from genomics.seed import spawn

matplotlib.use('Agg')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', nargs='?', default='experiments/configs/genomics_dfim_tfmodisco.yaml')
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text())
    sys.path.insert(0, str(ROOT / cfg['attribution_package_directory']))
    import tangermeme
    from tangermeme.deep_lift_shap import deep_lift_shap

    out = ROOT / cfg['output_directory']
    if out.exists():
        raise FileExistsError(out)
    warnings.filterwarnings('ignore', message='max_unpooling2d_forward_out does not')
    start_time = time.time()
    torch.set_num_threads(cfg['threads'])
    torch.set_num_interop_threads(1)
    torch.backends.mkldnn.enabled = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    rng = np.random.default_rng(cfg['seed'])
    model_dir, tile_dir = ROOT / cfg['model_directory'], ROOT / cfg['tile_directory']
    data, upstream = ROOT / cfg['data_directory'], ROOT / cfg['upstream_directory']
    metadata = json.loads((model_dir / 'metadata.json').read_text())
    hashes = {name: sha(data / name) for name in metadata['input_sha256']}
    assert hashes == metadata['input_sha256']
    with gzip.open(data / 'sequences.fa.gz', 'rt') as f:
        lines = f.read().splitlines()
    names = np.array([n[1:] for n in lines[::2]])
    lookup = {b: i for i, b in enumerate('ACGT')}
    tokens = np.array([[lookup[b] for b in s] for s in lines[1::2]], dtype=np.uint8)
    test = np.load(model_dir / 'splits.npz')['test']
    positions = np.sort(rng.choice(len(test), cfg['null_sequences'], replace=False))
    children = spawn(rng, len(positions))
    control = np.stack([shuffle_dinucleotides(tokens[test[j]], rng=child)
                        for j, child in zip(positions, children)])
    for j, shuffled in zip(positions, control):
        assert np.array_equal(dinucleotide_counts(tokens[test[j]]), dinucleotide_counts(shuffled))
        assert tokens[test[j]][0] == shuffled[0] and tokens[test[j]][-1] == shuffled[-1]
    spec = importlib.util.spec_from_file_location('frozen_dfim', model_dir / 'cnn.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.SequenceCNN()
    model.load_state_dict(torch.load(model_dir / 'model.pt', weights_only=True))
    model.eval()
    eye = np.eye(4, dtype=np.float32)
    probe = torch.from_numpy(eye[tokens[test[:32]]].transpose(0, 2, 1).copy()).requires_grad_(True)
    original = model(probe)
    original_gradient, = torch.autograd.grad(original.sum(), probe)
    model.network[4] = torch.nn.MaxPool1d(200)
    replacement = model(probe)
    replacement_gradient, = torch.autograd.grad(replacement.sum(), probe)
    torch.testing.assert_close(original, replacement, rtol=0, atol=0)
    torch.testing.assert_close(original_gradient, replacement_gradient, rtol=0, atol=0)
    with torch.no_grad():
        reference_output = float(model(torch.zeros(1, 4, 200)).item())
    class Output(torch.nn.Module):
        def __init__(self, predictor):
            super().__init__()
            self.predictor = predictor

        def forward(self, x):
            return self.predictor(x).unsqueeze(-1)

    explainer_model = Output(model)

    def attribute(values, name):
        tracks, residuals, logits = [], [], []
        for offset in range(0, len(values), cfg['batch_size']):
            x = torch.from_numpy(eye[values[offset:offset + cfg['batch_size']]].transpose(0, 2, 1).copy())
            x.requires_grad_(True)
            attribution = deep_lift_shap(
                explainer_model, x, references=torch.zeros(len(x), 1, 4, 200),
                n_shuffles=1, batch_size=cfg['batch_size'], target=0,
                hypothetical=False, warning_threshold=1e-4, device='cpu', verbose=False)
            with torch.no_grad():
                residual = attribution.sum((1, 2)) - (model(x) - reference_output)
            tracks.append(attribution.sum(1).detach().numpy())
            residuals.append(residual.detach().numpy())
            with torch.no_grad():
                logits.append(model(x).numpy())
        tracks, residuals, logits = np.concatenate(tracks), np.concatenate(residuals), np.concatenate(logits)
        relative = np.abs(residuals) / np.maximum(1, np.abs(logits - reference_output))
        print(f'{name}: max absolute residual {np.abs(residuals).max()}, max relative {relative.max()}', flush=True)
        return tracks, residuals, logits, relative

    tracks, residuals, logits, relative = attribute(tokens[test], 'test')
    null_tracks, null_residuals, null_logits, null_relative = attribute(control, 'shuffled')
    # Save all computed diagnostics even if the attribution acceptance gate fails.
    out.mkdir(parents=True)
    (out / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=True))
    np.savez_compressed(out / 'attributions.npz', signed_tracks=tracks, null_signed_tracks=null_tracks,
                        residuals=residuals, null_residuals=null_residuals, logits=logits,
                        null_logits=null_logits, relative_residuals=relative, null_relative_residuals=null_relative,
                        test_indices=test, null_test_positions=positions, control_tokens=control)
    acceptance = bool(np.all(np.abs(residuals) <= .01 * np.maximum(1, abs(logits - reference_output)) + 1e-4)
                      and np.all(np.abs(null_residuals) <= .01 * np.maximum(1, abs(null_logits - reference_output)) + 1e-4))
    diagnostic = dict(pooling_forward_bitwise_equal=True, pooling_gradients_bitwise_equal=True,
                      reference_logit=reference_output, completeness_gate_passed=acceptance,
                      max_relative_residual=float(max(relative.max(), null_relative.max())))
    (out / 'attribution_diagnostics.json').write_text(json.dumps(diagnostic, indent=2))
    (out / 'README.md').write_text('# TF-MoDISco published seqlet-stage baseline\n\n'
        'Reproduce: `MPLCONFIGDIR=/tmp/thesis_mpl_cache python -m experiments.run_dfim_tfmodisco`\n'
        'Use config.yaml and pinned upstream source. See protocol and report for transfer differences.\n'
        'Attributions and completeness diagnostics are saved even if the pre-run gate fails.\n')
    sources = [Path(__file__), ROOT / 'genomics/published_seqlets.py',
               ROOT / 'genomics/baselines.py', ROOT / 'genomics/seed.py', model_dir / 'cnn.py']
    for source in sources:
        shutil.copy2(source, out / source.name)
    with (out / 'metadata.json').open('w') as f:
        json.dump(dict(input_sha256=hashes, model_sha256=sha(model_dir / 'model.pt'),
                       upstream_commit=cfg['upstream_commit'],
                       source_sha256={str(p.relative_to(ROOT)): sha(p) for p in sources},
                       python=platform.python_version(), torch=torch.__version__, tangermeme=tangermeme.__version__,
                       numpy=np.__version__, sklearn=sklearn.__version__,
                       git_sha=subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip(),
                       git_dirty=True), f, indent=2)
    if not acceptance:
        raise RuntimeError('DeepLift completeness gate failed; diagnostics saved; no field baseline claim')
    coord_module, patches = load_upstream(upstream, out / 'upstream_figures')
    source_dir = out / 'upstream_source'
    source_dir.mkdir()
    for name, (original_code, runtime_code) in patches.items():
        (source_dir / (name + '.original.py')).write_text(original_code)
        (source_dir / (name + '.runtime.py')).write_text(runtime_code)
        (source_dir / (name + '.patch')).write_text(''.join(difflib.unified_diff(
            original_code.splitlines(True), runtime_code.splitlines(True), fromfile=name + '.original.py', tofile=name + '.runtime.py')))
    settings = {key: cfg[key] for key in ['sliding', 'flank', 'suppress', 'target_fdr',
                'min_passing_windows_frac', 'max_passing_windows_frac', 'separate_pos_neg_thresholds', 'max_seqlets_total']}
    caller = coord_module.FixedWindowAroundChunks(**settings, verbose=True)
    results = caller(score_track=tracks, null_track=null_tracks)
    with h5py.File(out / 'native_seqlets.h5', 'w') as f:
        results.save_hdf5(f.create_group('results'))
    selections = seqlets_to_tiles(results.coords, len(test), flank=cfg['flank'])
    with np.load(tile_dir / 'tiles.npz') as annotations:
        assert np.array_equal(test, annotations['source_indices'])
        roles, needed, targets = annotations['role'], annotations['has_needed_motif'], annotations['targets']
    values = sequence_metrics(selections, needed)
    summary = {key: aggregate(values, mask) for key, mask in
               [('all', np.ones(len(test), dtype=bool)), ('positive', targets == 1), ('negative', targets == 0)]}
    summary['by_primary_role'] = {role: dict(tiles=int((roles == role).sum()),
                                           selected=int((selections & (roles == role)).sum()),
                                           selection_fraction=float(selections[roles == role].mean()))
                                  for role in sorted(set(roles.ravel()))}
    summary['native_seqlets'] = len(results.coords)
    summary['native_positive_seqlets'] = int(sum(c.score > 0 for c in results.coords))
    summary['native_negative_seqlets'] = int(sum(c.score < 0 for c in results.coords))
    summary['seconds'] = time.time() - start_time
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    thresholds = dict(positive=results.tnt_results.pos_threshold, negative=results.tnt_results.neg_threshold,
                      transformed_positive=results.tnt_results.transformed_pos_threshold,
                      transformed_negative=results.tnt_results.transformed_neg_threshold)
    (out / 'thresholds.json').write_text(json.dumps(thresholds, indent=2) + '\n')
    np.savez_compressed(out / 'tile_selections.npz', selections=selections, source_indices=test, **values)
    with (out / 'native_seqlets.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['test_position', 'source_index', 'sequence_name', 'start', 'end', 'core_start', 'core_end', 'signed_core_score'])
        for c in results.coords:
            writer.writerow([c.example_idx, int(test[c.example_idx]), names[test[c.example_idx]],
                             c.start, c.end, c.start + cfg['flank'], c.end - cfg['flank'], float(c.score)])
    print(json.dumps(dict(thresholds=thresholds, summary=summary), indent=2), flush=True)


if __name__ == '__main__':
    main()
