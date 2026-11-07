"""Run frozen fixed-relative and pooled-shuffle calibration on DFIM tiles."""
import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import platform
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from genomics.baselines import (
    dinucleotide_counts,
    gradient_tile_scores,
    pooled_threshold,
    relative_selection,
    sequence_metrics,
    shuffle_dinucleotides,
)
from genomics.paths import ROOT
from genomics.seed import spawn


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate(values, mask):
    selected = values['selected_count'][mask]
    false = values['false_positive_count'][mask]
    power = values['power'][mask]
    return dict(sequences=int(mask.sum()), selected_tiles=int(selected.sum()),
                false_selected_tiles=int(false.sum()),
                mean_sequence_fdp=float(values['fdp'][mask].mean()),
                pooled_fdp=float(false.sum() / max(selected.sum(), 1)),
                fraction_sequences_with_any_selection=float((selected > 0).mean()),
                fraction_sequences_fdp_above_point1=float((values['fdp'][mask] > .1).mean()),
                mean_power=float(np.nanmean(power)) if np.isfinite(power).any() else None,
                fdp_quantiles=np.quantile(values['fdp'][mask], [0, .25, .5, .75, 1]).tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', nargs='?', default='experiments/configs/genomics_dfim_baselines.yaml')
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text())
    assert cfg['score'] == 'absolute_observed_base_logit_gradient_sum'
    assert cfg['shuffle'] == 'pinned_final_successor_dinucleotide'
    assert cfg['calibration_split'] == 'validation' and cfg['evaluation_split'] == 'test'
    out = ROOT / cfg['output_directory']
    if out.exists():
        raise FileExistsError(out)
    start = time.time()
    torch.manual_seed(cfg['seed'])
    torch.set_num_threads(cfg['threads'])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False
    model_dir = ROOT / cfg['model_directory']
    tile_dir = ROOT / cfg['tile_directory']
    data = ROOT / cfg['data_directory']
    metadata = json.loads((model_dir / 'metadata.json').read_text())
    input_hashes = {name: sha(data / name) for name in metadata['input_sha256']}
    assert input_hashes == metadata['input_sha256']
    with gzip.open(data / 'sequences.fa.gz', 'rt') as f:
        lines = f.read().splitlines()
    names = np.array([name[1:] for name in lines[::2]])
    lookup = {b: i for i, b in enumerate('ACGT')}
    tokens = np.array([[lookup[b] for b in s] for s in lines[1::2]], dtype=np.uint8)
    with np.load(model_dir / 'splits.npz') as splits:
        validation = splits['validation']
        test = splits['test']
        train = splits['train']
    assert not np.intersect1d(validation, test).size
    assert not np.intersect1d(train, test).size
    spec = importlib.util.spec_from_file_location('frozen_dfim', model_dir / 'cnn.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.SequenceCNN()
    model.load_state_dict(torch.load(model_dir / 'model.pt', weights_only=True))
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(np.eye(4, dtype=np.float32)[tokens[test[:256]]]
                                       .transpose(0, 2, 1).copy())).numpy()
    previous = np.load(model_dir / 'predictions.npz')['test_logits'][:256]
    assert np.array_equal(logits, previous), 'Checkpoint inference changed'
    options = dict(tile_width=cfg['tile_width'], batch_size=cfg['batch_size'])
    validation_scores, validation_tracks = gradient_tile_scores(model, tokens[validation], **options)
    test_scores, test_tracks = gradient_tile_scores(model, tokens[test], **options)
    print('Observed validation/test scores computed', flush=True)
    controls, shuffled_sequences = [], []
    control_children = spawn(np.random.default_rng(cfg['seed']), cfg['control_replicates'])
    for replicate, rng in enumerate(control_children):
        shuffled = np.stack([shuffle_dinucleotides(row, rng=rng) for row in tokens[validation]])
        # Check each control, not merely a sample of the shuffles.
        for original, control in zip(tokens[validation], shuffled):
            assert np.array_equal(dinucleotide_counts(original), dinucleotide_counts(control))
            assert original[0] == control[0] and original[-1] == control[-1]
        scores, _ = gradient_tile_scores(model, shuffled, **options)
        controls.append(scores)
        shuffled_sequences.append(shuffled)
        print(f'Control replicate {replicate + 1}/{cfg["control_replicates"]} complete', flush=True)
    controls = np.stack(controls)
    threshold, curve = pooled_threshold(validation_scores, controls, cfg['pooled_target'])
    selections = dict(fixed_relative=relative_selection(test_scores, cfg['relative_fraction']),
                      pooled_shuffle=(test_scores >= threshold))
    with np.load(tile_dir / 'tiles.npz') as annotations:
        assert np.array_equal(annotations['source_indices'], test)
        assert np.array_equal(annotations['sequence_names'], names[test])
        roles = annotations['role']
        targets = annotations['targets']
        needed = annotations['has_needed_motif']
        decoy = annotations['has_decoy']
        mixed = annotations['mixed_role']
        assert np.all((roles == 'needed_motif') == needed)
    assert test_scores.shape == roles.shape == (9000, 10)
    assert np.isfinite(test_scores).all() and np.isfinite(controls).all()
    summary, metrics = {}, {}
    for method, selected in selections.items():
        values = sequence_metrics(selected, needed)
        metrics[method] = values
        summary[method] = {
            'all': aggregate(values, np.ones(len(test), dtype=bool)),
            'positive': aggregate(values, targets == 1),
            'negative': aggregate(values, targets == 0),
            'by_primary_role': {role: dict(tiles=int((roles == role).sum()),
                                          selected=int((selected & (roles == role)).sum()),
                                          selection_fraction=float(selected[roles == role].mean()))
                                for role in sorted(set(roles.ravel()))},
            'selected_tiles_with_decoy_overlap': int((selected & decoy).sum()),
            'selected_mixed_tiles': int((selected & mixed).sum()),
        }
    threshold_info = dict(relative_fraction=cfg['relative_fraction'],
                          pooled_threshold=threshold if np.isfinite(threshold) else None,
                          pooled_selects_none=not np.isfinite(threshold), pooled_target=cfg['pooled_target'],
                          calibration_sequences=len(validation), calibration_tiles=validation_scores.size,
                          control_sequences=len(validation) * len(controls), control_tiles=controls.size)
    if np.isfinite(threshold):
        index = np.flatnonzero(curve['threshold'] == threshold)[0]
        threshold_info['estimated_pooled_false_fraction_at_threshold'] = float(curve['estimated_false_fraction'][index])
    # Outputs only after executed computation; no results directory used for planning.
    out.mkdir(parents=True)
    (out / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=True))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (out / 'thresholds.json').write_text(json.dumps(threshold_info, indent=2) + '\n')
    np.savez_compressed(out / 'scores.npz', validation_scores=validation_scores, test_scores=test_scores,
                        validation_base_tracks=validation_tracks, test_base_tracks=test_tracks,
                        validation_indices=validation, test_indices=test, control_scores=controls,
                        control_tokens=np.stack(shuffled_sequences), roles=roles, targets=targets)
    np.savez_compressed(out / 'calibration_curve.npz', **curve)
    np.savez_compressed(out / 'selections.npz', **selections)
    with (out / 'per_sequence.csv').open('w', newline='') as f:
        fields = ['method', 'source_index', 'sequence_name', 'target', *metrics['fixed_relative']]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for method, values in metrics.items():
            for j, idx in enumerate(test):
                row = dict(method=method, source_index=int(idx), sequence_name=names[idx], target=int(targets[j]))
                row.update({key: array[j].item() for key, array in values.items()})
                if not np.isfinite(row['power']):
                    row['power'] = ''
                writer.writerow(row)
    with (out / 'per_tile.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['source_index', 'sequence_name', 'target', 'tile_index', 'start', 'end',
                         'role', 'score', 'fixed_relative_selected', 'pooled_shuffle_selected'])
        for j, idx in enumerate(test):
            for tile in range(10):
                writer.writerow([int(idx), names[idx], int(targets[j]), tile, tile * 20, (tile + 1) * 20,
                                 roles[j, tile], float(test_scores[j, tile]),
                                 bool(selections['fixed_relative'][j, tile]), bool(selections['pooled_shuffle'][j, tile])])
    sources = [Path(__file__), ROOT / 'genomics/baselines.py', ROOT / 'genomics/seed.py', model_dir / 'cnn.py']
    for source in sources:
        shutil.copy2(source, out / source.name)

    def git(*cmd):
        return subprocess.run(['git', '-C', str(ROOT), *cmd], capture_output=True, text=True).stdout.strip()

    provenance = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                      seconds=time.time() - start, git_sha=git('rev-parse', 'HEAD'),
                      git_dirty=bool(git('status', '--porcelain')), input_sha256=input_hashes,
                      model_sha256=sha(model_dir / 'model.pt'), split_sha256=sha(model_dir / 'splits.npz'),
                      tile_annotations_sha256=sha(tile_dir / 'tiles.npz'),
                      source_sha256={str(p.relative_to(ROOT)): sha(p) for p in sources},
                      python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                      platform=platform.platform())
    (out / 'metadata.json').write_text(json.dumps(provenance, indent=2) + '\n')
    (out / 'README.md').write_text('''# DFIM tile selection baselines

Reproduce from repo root:
`python -m experiments.run_dfim_baselines experiments/configs/genomics_dfim_baselines.yaml`
The configured output directory must not exist. Change only output_directory in
a copied config to rerun. Exact inputs/hashes/source snapshots are in metadata.json.
Requires the saved CNN and test-tile annotations; no refitting occurs.

Score = sum within 20 bp of abs(gradient of task3 logit at the observed base),
i.e. absolute input×gradient. Fixed selects >=5% of the sequence maximum, with
zero-score sequences selecting nothing. Pooled uses 10 dinucleotide shuffles of
each validation sequence. At every positive observed validation score t, compute
min(1, (control_tail_count+1)*observed_tile_count/control_tile_count/observed_tail_count).
Choose the lowest threshold with estimate <=.1. No solution means no selections.
Apply that one frozen threshold to test. No test scores/labels enter calibration.
This is an adapted pooled-control baseline, NOT a full published seqlet pipeline.
No isotonic fitting, selection-fraction caps or motif clustering are included.

Shuffles permute successor order except the final outgoing edge, independently
per replicate via genomics.seed.spawn; endpoints/dinucleotide counts are checked for
every control. No uniform-shuffle claim. Exact control tokens and scores saved.

Truth is the previously fixed planted-motif convention. Needed tiles are true;
background, decoy and unused are false. Mixed signal/decoy tiles are true. This
is not truth about the CNN or a sampler-relative null. Per-sequence FDP is
V/max(R,1); mean FDP across test sequences is the empirical FDR estimate over
this test distribution, NOT an individual-sequence guarantee. Pooled FDP is
sum(V)/max(sum(R),1), a different quantity. Power is TP/number_needed, and undefined
(blank in CSV) for negatives. Category selection fractions for null categories
are false-positive rates, not category-specific FDR. Nominal pooled .1 is an
estimated calibration target, not guaranteed true FDR. Validation also served
checkpoint selection; test remained held out.

Artifacts: scores.npz includes all observed base/tile scores, 90,000 control
sequences and their tile scores; calibration_curve.npz contains the full tail
curve; selections.npz and per_tile.csv contain calls; per_sequence.csv contains
counts, FDP and power. summary.json separates positives/negatives and categories.
See experiment_design/genomics_dfim_baselines/protocol.md for the frozen design.
Reference: https://pmc.ncbi.nlm.nih.gov/articles/PMC8812996/ (pooled shuffle-null
calibration); https://raw.githubusercontent.com/kundajelab/deeplift/master/deeplift/dinuc_shuffle.py
(successor shuffling). The ratio rule and fixed tiles here are an adaptation.
''')
    print(json.dumps(dict(thresholds=threshold_info, summary=summary), indent=2), flush=True)


if __name__ == '__main__':
    main()
