"""Finalize and report the saved upstream seqlet results; no recalibration."""
import csv
import hashlib
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import yaml

from experiments.run_dfim_baselines import aggregate
from genomics.baselines import sequence_metrics
from genomics.paths import ROOT
from genomics.published_seqlets import load_upstream, seqlets_to_tiles
from genomics.tiles import parse_embeddings


def main():
    import gzip

    out = ROOT / 'results/genomics_dfim_tfmodisco'
    cfg = yaml.safe_load((out / 'config.yaml').read_text())
    coord_module, _ = load_upstream(ROOT / cfg['upstream_directory'], out / 'upstream_figures')
    with h5py.File(out / 'native_seqlets.h5') as f:
        results = coord_module.CoordProducerResults.from_hdf5(f['results'])
    attribution = np.load(out / 'attributions.npz')
    native_sums = coord_module.get_simple_window_sum_function(cfg['sliding'])(arrs=attribution['signed_tracks'])
    for coord in results.coords:
        coord.score = native_sums[coord.example_idx][coord.start + cfg['flank']]
    test = attribution['test_indices']
    with np.load(ROOT / cfg['tile_directory'] / 'tiles.npz') as annotations:
        assert np.array_equal(test, annotations['source_indices'])
        roles, needed, targets, names = (annotations[k] for k in ['role', 'has_needed_motif', 'targets', 'sequence_names'])
    selections = seqlets_to_tiles(results.coords, len(test), flank=cfg['flank'])
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
    with gzip.open(ROOT / cfg['data_directory'] / 'simulation_metadata.txt.gz', 'rt') as f:
        planted = list(csv.DictReader(f, delimiter='\t'))
    native_r, native_true = np.zeros(len(test), dtype=int), np.zeros(len(test), dtype=int)
    for c in results.coords:
        j = c.example_idx
        core_start, core_end = c.start + cfg['flank'], c.end - cfg['flank']
        m = planted[test[j]]
        instances = parse_embeddings(m['embeddings'], m['sequence'])
        true = bool(targets[j] and any(i['motif'] in {'ELF1', 'SIX5'} and
                                      i['start'] < core_end and i['end'] > core_start for i in instances))
        native_r[j] += 1
        native_true[j] += true
    native_fdp = (native_r - native_true) / np.maximum(native_r, 1)
    summary['native_core_overlap_evaluation'] = dict(
        mean_sequence_fdp=float(native_fdp.mean()),
        pooled_fdp=float((native_r - native_true).sum() / max(native_r.sum(), 1)),
        mean_positive_sequence_fdp=float(native_fdp[targets == 1].mean()),
        definition='Seqlet core is true iff any base overlaps planted ELF1/SIX5 in a positive sequence')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    tnt = results.tnt_results
    thresholds = {key: float(getattr(tnt, attribute)) for key, attribute in
                  [('positive', 'pos_threshold'), ('negative', 'neg_threshold'),
                   ('transformed_positive', 'transformed_pos_threshold'),
                   ('transformed_negative', 'transformed_neg_threshold')]}
    # Determine passing-window fraction on the saved maps; native logs retain initial/final thresholds.
    windows = np.lib.stride_tricks.sliding_window_view(attribution['signed_tracks'], 21, axis=1).sum(2)
    thresholds['fraction_windows_passing_final'] = float(((windows > thresholds['positive']) |
                                                       (windows < thresholds['negative'])).mean())
    thresholds['cap_activated'] = False  # Verified from original upstream log: raw and final thresholds agree.
    (out / 'thresholds.json').write_text(json.dumps(thresholds, indent=2) + '\n')
    np.savez_compressed(out / 'tile_selections.npz', selections=selections, source_indices=test,
                        native_seqlet_counts=native_r, native_fdp=native_fdp, **values)
    with (out / 'native_seqlets.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['test_position', 'source_index', 'sequence_name', 'start', 'end', 'core_start', 'core_end', 'signed_core_score'])
        for c in results.coords:
            writer.writerow([c.example_idx, int(test[c.example_idx]), names[c.example_idx],
                             c.start, c.end, c.start + cfg['flank'], c.end - cfg['flank'], float(c.score)])
    with (out / 'per_sequence.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['test_position', 'source_index', 'sequence_name', 'target', *values])
        for j, idx in enumerate(test):
            writer.writerow([j, int(idx), names[j], int(targets[j]),
                             *[a[j].item() if np.isfinite(a[j]) else '' for a in values.values()]])
    report = '''# Published field reference: TF-MoDISco seqlet calling

**Method:** original TF-MoDISco v0.5.1.1, commit
`a12b47b72418666854219791bc3ba8ec9234d71d`, using its native
`FixedWindowAroundChunks` implementation. This is the selection stage used by
[BPNet (Avsec et al., 2021)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8812996/),
not the earlier hand-written pooled-count ratio. No custom threshold estimator.
This run stops at seqlet calling; it does not evaluate motif clustering or CWM hit scanning.

## What is held faithful, and what transfers to our model
| Component | Choice and provenance |
|---|---|
| Attribution | Signed DeepLIFT, zero reference, as in BPNet; tangermeme 1.6.0 implementation on our PyTorch CNN instead of legacy Keras |
| Target | Our task3 scalar logit; BPNet's profile-output target does not exist in this model |
| Windows / nominal level | Published 21 bp / target_seqlet_fdr 0.01 |
| Null | Published count of 4,800 randomly selected inputs, one dinucleotide shuffle each |
| Threshold fitting | Upstream isotonic fit and both positive/negative tails, unchanged |
| Extraction | Upstream greedy suppression; release-default flank10 and suppress20 |
| Passing fraction | Upstream release defaults 3–20%, retained; no override activated in this run |
| Analysis corpus | All 9,000 synthetic test sequences, pooled without labels; BPNet analyzed TF-bound genomic peaks |
| Evaluation mapping | Central21-bp cores overlap fixed20-bp tiles; native41-bp seqlets retained separately |

The numeric selector source is unchanged. Compatibility changes only replace a
deprecated sklearn import and redirect diagnostic plots. Exact original/runtime
source and diffs are in upstream_source/. Model weights stay frozen. The global
pool is expressed as equivalent MaxPool1d(200) for attribution hooks; both ordinary
outputs and gradients were checked to match bitwise. PyTorch's unpool warning
requires deterministic_algorithms(warn_only=True); repeatability is checked separately.
Captum DeepLift failed an attribution completeness check and was excluded before
running the selector; its failed diagnostics are retained separately. Tangermeme's
full-run maximum normalized completeness residual was below 0.000007.

## Measured results
'''
    report += f'Native seqlets: **{summary["native_seqlets"]:,}** '
    report += f'({summary["native_positive_seqlets"]:,} positive-score, {summary["native_negative_seqlets"]:,} negative-score).\n\n'
    report += '| Tile-level test measure | TF-MoDISco | Fixed 5% comparator |\n|---|---:|---:|\n'
    fixed = json.loads((ROOT / 'results/genomics_dfim_baselines/summary.json').read_text())['fixed_relative']
    for label, group, key in [('Selected tiles', 'all', 'selected_tiles'),
                             ('Mean sequence FDP, all', 'all', 'mean_sequence_fdp'),
                             ('Mean sequence FDP, positives', 'positive', 'mean_sequence_fdp'),
                             ('Mean power, positives', 'positive', 'mean_power')]:
        a, b = summary[group][key], fixed[group][key]
        report += f'| {label} | {a:,} | {b:,} |\n' if key == 'selected_tiles' else f'| {label} | {a:.2%} | {b:.2%} |\n'
    report += f'\nNative core-overlap mean FDP: {summary["native_core_overlap_evaluation"]["mean_sequence_fdp"]:.2%}; '
    report += f'on positives only: {summary["native_core_overlap_evaluation"]["mean_positive_sequence_fdp"]:.2%}.\n'
    report += '''
The fixed comparator uses absolute gradient×input, whereas this field reference
uses signed DeepLIFT; this is a practical-method comparison, not an isolated
thresholding ablation. Native seqlet truth is any core overlap with a needed
planted motif. Tile truth and scoring follow the previously frozen convention.
Both metrics are empirical against that annotation convention, not estimates of
error against the CNN's true functional dependence. Negative-score seqlets are
part of the published caller; retain them rather than discarding them after seeing
outcomes. Negative examples have no needed tiles under our convention, which
differs from a null about the model's score. A nominal1% setting is not a proven
per-sequence FDR bound, and these results do not establish the success of method(c).

## Reproduce and inspect
Config: config.yaml. Entry point: experiments/run_dfim_tfmodisco.py. Set
MPLCONFIGDIR=/tmp/thesis_mpl_cache and NUMBA_CACHE_DIR=/tmp/thesis_numba.
The configured output directory must not exist. Upstream source and the tangermeme
wheel are cached under data/third_party/; exact hashes are in provenance files.
The full pre-run protocol is experiment_design/genomics_dfim_tfmodisco/protocol.md.
Attributions/control sequences, completeness residuals, native seqlets (HDF5/CSV),
thresholds, mapped selections, and individual sequence metrics are saved here.
The reporter can regenerate tables from native_seqlets.h5 without recalibration.
'''
    (out / 'FIELD_BASELINE_REPORT.md').write_text(report)
    (out / 'README.md').write_text('# TF-MoDISco field-reference run\n\nSee FIELD_BASELINE_REPORT.md for method, differences, results and reproduction.\n')
    provenance = dict(reporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      upstream_file_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                            for p in (out / 'upstream_source').glob('*.original.py')},
                      tangermeme_wheel_sha256=hashlib.sha256(next((ROOT / cfg['attribution_package_directory']).glob('*.whl')).read_bytes()).hexdigest(),
                      export_recovery='Initial numeric run saved native HDF5; int64 JSON serialization failed. This reporter exports the saved results without rerunning selection.')
    (out / 'report_provenance.json').write_text(json.dumps(provenance, indent=2))
    shutil.copy2(__file__, out / Path(__file__).name)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
