"""Write a concise report and plot from saved baseline outputs."""
import csv
import json

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from genomics.paths import RESULTS_DIR


def main():
    out = RESULTS_DIR / 'genomics_dfim_baselines'
    summary = json.loads((out / 'summary.json').read_text())
    thresholds = json.loads((out / 'thresholds.json').read_text())
    with (out / 'per_sequence.csv').open() as f:
        rows = list(csv.DictReader(f))
    names = ['fixed_relative', 'pooled_shuffle']
    labels = ['Fixed 5%', 'Pooled shuffle']
    colors = ['#4878a8', '#d08038']
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), layout='constrained')
    intervals = {}
    for method in names:
        values = np.array([float(r['fdp']) for r in rows if r['method'] == method])
        intervals[method] = 1.96 * values.std(ddof=1) / np.sqrt(len(values))
    means = [summary[n]['all']['mean_sequence_fdp'] for n in names]
    axes[0].bar(labels, means, color=colors, yerr=[intervals[n] for n in names], capsize=4)
    axes[0].axhline(.1, color='gray', ls='--', lw=1, label='Pooled calibration target')
    axes[0].set(title='Mean per-sequence FDP', ylim=(0, 1), ylabel='Fraction')
    axes[0].legend(fontsize=8)
    axes[1].bar(labels, [summary[n]['positive']['mean_power'] for n in names], color=colors)
    axes[1].set(title='Power on positive sequences', ylim=(0, 1), ylabel='Fraction of needed tiles selected')
    roles = ['background', 'decoy', 'present_but_unused_motif', 'needed_motif']
    x = np.arange(len(roles))
    for i, (name, label, color) in enumerate(zip(names, labels, colors)):
        axes[2].bar(x + (i - .5) * .35,
                    [summary[name]['by_primary_role'][r]['selection_fraction'] for r in roles],
                    width=.35, label=label, color=color)
    axes[2].set(xticks=x, xticklabels=['Background', 'Decoy', 'Unused', 'Needed'],
                title='Selection rate by tile role', ylim=(0, 1), ylabel='Fraction selected')
    axes[2].tick_params(axis='x', labelrotation=25)
    axes[2].legend(fontsize=8)
    fig.suptitle('DFIM held-out tile baselines — planted-mask evaluation', fontsize=12)
    fig.savefig(out / 'baseline_comparison.png', dpi=180)
    plt.close(fig)
    report = '''# DFIM exploratory tile-selection baseline results

The pooled count-ratio method here is an exploratory adaptation, not the published field reference.

Frozen CNN, 9,000 held-out sequences, ten 20-bp tiles each. Both methods use
absolute observed-base logit gradients summed within tiles. The fixed baseline
uses 5% of each sequence maximum. Pooled calibration uses ten dinucleotide
shuffles per validation sequence and a control-tail false-fraction estimate
at target 10%; its single threshold is then frozen for test evaluation.
This pooled method is a tile-based adaptation, not an exact published pipeline.

| Test measure | Fixed 5% | Pooled shuffle |
|---|---:|---:|
'''
    for title, group, key in [
        ('Mean sequence FDP (all sequences)', 'all', 'mean_sequence_fdp'),
        ('Pooled FDP (all selected tiles)', 'all', 'pooled_fdp'),
        ('Mean sequence FDP (positives)', 'positive', 'mean_sequence_fdp'),
        ('Mean sequence FDP (negatives)', 'negative', 'mean_sequence_fdp'),
        ('Mean power (positives)', 'positive', 'mean_power'),
        ('Negatives with any selection', 'negative', 'fraction_sequences_with_any_selection'),
    ]:
        report += f'| {title} | ' + ' | '.join(f'{summary[n][group][key]:.2%}' for n in names) + ' |\n'
    report += '\n| Tile role | Fixed selection rate | Pooled selection rate |\n|---|---:|---:|\n'
    for role in roles:
        report += f'| {role} | ' + ' | '.join(f'{summary[n]["by_primary_role"][role]["selection_fraction"]:.2%}' for n in names) + ' |\n'
    curve = np.load(out / 'calibration_curve.npz')
    saved_scores = np.load(out / 'scores.npz')
    diagnostics = dict(
        minimum_calibration_false_fraction=float(curve['estimated_false_fraction'].min()),
        validation_score_quantiles=np.quantile(saved_scores['validation_scores'], [.5, .9, .99]).tolist(),
        shuffled_score_quantiles=np.quantile(saved_scores['control_scores'], [.5, .9, .99]).tolist(),
        test_score_quantiles_by_role={r: np.quantile(saved_scores['test_scores'][saved_scores['roles'] == r],
                                                  [.5, .9, .99]).tolist() for r in roles})
    (out / 'score_diagnostics.json').write_text(json.dumps(diagnostics, indent=2) + '\n')
    if thresholds['pooled_selects_none']:
        report += ('\nNo pooled threshold met the 10% calibration target; its minimum estimated '
                   f'false fraction was {diagnostics["minimum_calibration_false_fraction"]:.2%}. '
                   'The frozen rule therefore selected nothing: zero false discoveries AND zero power.\n')
    else:
        report += f'\nFrozen pooled threshold: {thresholds["pooled_threshold"]}.\n' 
    report += '''
FDP is false selections divided by max(number selected,1). Its average over
held-out sequences estimates FDR over this test distribution; it is not a
per-sequence repeated-randomization guarantee. The figure's mean-FDP bars show
approximate 95% sampling intervals (1.96 × sequence standard error), conditional
on this frozen model/calibration. Pooled FDP is a different, selection-weighted
quantity. Power is undefined on negatives; they have no needed tiles under the
chosen convention, so any selection there is false. Mixed needed/decoy tiles
are counted as needed, matching the fixed annotation policy.

![Baseline comparison](baseline_comparison.png)

High error on unused motifs could reflect a mismatch between a sensitivity
score and the planned positive-AND planted mask; it need not be model failure.
Attribution spillover, off-simplex gradients and chance motifs can also disagree
with planted annotations. The experiment does not isolate those mechanisms.
An observed pooled target is not a formal bound on true FDR. These results do
not establish that per-sequence randomization will solve the problem: that
requires an explicit sampler-relative null and justified multiplicity handling.

All baselines remain available unchanged for comparison with a later method.
See README.md for reproduction and raw-output definitions, thresholds.json for
calibration details, and per_sequence.csv/per_tile.csv for every individual call.
'''
    (out / 'BASELINE_REPORT.md').write_text(report)
    (out / 'sampling_intervals.json').write_text(json.dumps(intervals, indent=2) + '\n')
    print(report)


if __name__ == '__main__':
    main()
