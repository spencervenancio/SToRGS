"""Apply the saved all-family mathematical threshold unchanged to DNA tiles."""
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import yaml

from experiments.run_dfim_baselines import aggregate
from genomics.baselines import sequence_metrics
from genomics.paths import ROOT
from genomics.threshold_transfer import FEATURE_NAMES, select_with_rule


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', nargs='?', default='experiments/configs/genomics_dfim_threshold_transfer.yaml')
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text())
    assert cfg['feature_order'] == FEATURE_NAMES and cfg['normalization_floor'] == 1e-12
    out = ROOT / cfg['output_directory']
    if out.exists():
        raise FileExistsError(out)
    rule_file = ROOT / cfg['source_rule_file']
    rule = json.loads(rule_file.read_text())[cfg['source_rule_key']]
    with np.load(ROOT / cfg['score_file']) as data:
        scores, indices = data['test_scores'].astype(float), data['test_indices']
    with np.load(ROOT / cfg['annotation_file']) as data:
        assert np.array_equal(indices, data['source_indices'])
        roles, needed, targets, names = (data[key] for key in ['role', 'has_needed_motif', 'targets', 'sequence_names'])
    selected, thresholds, raw, features, normalized, degenerate = select_with_rule(scores, rule)
    values = sequence_metrics(selected, needed)
    summary = {key: aggregate(values, mask) for key, mask in
               [('all', np.ones(len(scores), dtype=bool)), ('positive', targets == 1), ('negative', targets == 0)]}
    summary['by_primary_role'] = {role: dict(tiles=int((roles == role).sum()),
                                           selected=int((selected & (roles == role)).sum()),
                                           selection_fraction=float(selected[roles == role].mean()))
                                  for role in sorted(set(roles.ravel()))}
    summary['threshold_quantiles_0_25_50_75_100'] = np.quantile(thresholds, [0, .25, .5, .75, 1]).tolist()
    summary['clipped_to_zero'] = int((raw <= 0).sum())
    summary['clipped_to_one'] = int((raw >= 1).sum())
    summary['degenerate_maps'] = int(degenerate.sum())
    out.mkdir(parents=True)
    (out / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=True))
    (out / 'frozen_rule.json').write_text(json.dumps(rule, indent=2) + '\n')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    np.savez_compressed(out / 'predictions.npz', source_indices=indices, scores=scores,
                        normalized_scores=normalized, features=features, thresholds=thresholds,
                        unclipped_thresholds=raw, selected=selected, degenerate=degenerate, **values)
    with (out / 'per_sequence.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['source_index', 'sequence_name', 'target', 'threshold', 'unclipped_threshold', *values])
        for i, idx in enumerate(indices):
            writer.writerow([int(idx), names[i], int(targets[i]), thresholds[i], raw[i],
                             *[a[i].item() if np.isfinite(a[i]) else '' for a in values.values()]])
    with (out / 'per_tile.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['source_index', 'sequence_name', 'tile_index', 'role', 'normalized_score', 'threshold', 'selected'])
        for i, idx in enumerate(indices):
            for tile in range(10):
                writer.writerow([int(idx), names[i], tile, roles[i, tile], normalized[i, tile], thresholds[i], bool(selected[i, tile])])
    fixed = json.loads((ROOT / cfg['fixed_summary']).read_text())['fixed_relative']
    field = json.loads((ROOT / cfg['field_summary']).read_text())
    report = '''# Transfer of Spencer's mathematical-function threshold to DFIM

The frozen all-family Lasso rule matches the formula in the meeting notes.
It was trained on24 mathematical functions with10 features; no DNA data were
used to refit, select, or recalibrate its coefficients. We use the exact exported
coefficients, not the rounded equation below:

`tau = clip(.14320056 + .07017495*q25 + .39596138*q75 - .19217223*top1_mass - .16158832*top3_mass, 0, 1)`

For each sequence: sum absolute observed-base logit gradients within each20-bp
tile; divide the10 scores by their maximum; compute the original eight summaries;
predict tau; select every normalized tile score>=tau. Exported coefficients
already incorporate the fitted standardizer, so no second standardization occurs.
This transfers the original scalar-feature rule to groups of DNA bases. Both
representations have10 features, but that does not establish distributional similarity.

| Held-out tile measure | Frozen mathematical rule | Fixed5% | TF-MoDISco |
|---|---:|---:|---:|
'''
    for label, group, key in [('Selected tiles', 'all', 'selected_tiles'),
                             ('Mean sequence FDP, all', 'all', 'mean_sequence_fdp'),
                             ('Mean sequence FDP, positives', 'positive', 'mean_sequence_fdp'),
                             ('Mean power, positives', 'positive', 'mean_power'),
                             ('Negatives with any selection', 'negative', 'fraction_sequences_with_any_selection')]:
        a = [s[group][key] for s in [summary, fixed, field]]
        report += '| ' + label + ' | ' + ' | '.join(f'{x:,}' if key == 'selected_tiles' else f'{x:.2%}' for x in a) + ' |\n'
    report += '\n| Tile role | Learned selection rate |\n|---|---:|\n'
    for role, row in summary['by_primary_role'].items():
        report += f'| {role} | {row["selection_fraction"]:.2%} |\n'
    report += f'\nMedian predicted relative threshold: {np.median(thresholds):.4f}; range {thresholds.min():.4f}–{thresholds.max():.4f}.\n'
    report += '''
The matched comparison with the fixed rule changes only the threshold, not scores
or truth. The TF-MoDISco practical-method comparison also changes attribution and
selection unit; its seqlets were mapped to tiles. Needed tiles follow the frozen
planted-mask convention. Negatives have no needed tiles, so every selection there
is false. Mean sequence FDP is an empirical FDR estimate across the test
population, not a within-sequence repeated-sampling guarantee. The mathematical
rule was trained toward minimum-important saliency, not FDR control.

Reproduce from repo root: `python -m experiments.run_dfim_threshold_transfer`.
Use the pinned config, rule and score hashes in metadata.json. A rerun needs a
fresh output_directory in a copied config. Saved source/rule snapshots, raw
features, unclipped/clipped thresholds and every selection are provided here.
No new CNN inference, fitting, random sampling, or threshold search is performed.
'''
    (out / 'TRANSFER_REPORT.md').write_text(report)
    (out / 'README.md').write_text('# Frozen mathematical-rule transfer\n\nSee TRANSFER_REPORT.md for results, definitions and reproduction.\n')
    sources = [Path(__file__), ROOT / 'genomics/threshold_transfer.py',
               ROOT / 'genomics/baselines.py', ROOT / 'experiments/run_dfim_baselines.py']
    for p in sources:
        shutil.copy2(p, out / p.name)
    provenance = dict(source_rule_sha256=sha(rule_file), source_rule_key=cfg['source_rule_key'],
                      input_sha256={key: sha(ROOT / cfg[key]) for key in ['score_file', 'annotation_file', 'fixed_summary', 'field_summary']},
                      source_sha256={str(p.relative_to(ROOT)): sha(p) for p in sources},
                      git_sha=subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip(),
                      git_dirty=True, numpy=np.__version__, randomness='none; no fitting')
    (out / 'metadata.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
