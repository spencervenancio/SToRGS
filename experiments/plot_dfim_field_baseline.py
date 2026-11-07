"""Show practical tile comparison and native-versus-tile evaluation units."""
import json

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from genomics.paths import RESULTS_DIR


def main():
    out = RESULTS_DIR / 'genomics_dfim_tfmodisco'
    field = json.loads((out / 'summary.json').read_text())
    fixed = json.loads((RESULTS_DIR / 'genomics_dfim_baselines/summary.json').read_text())['fixed_relative']
    native = field['native_core_overlap_evaluation']
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout='constrained')
    axes[0].bar(['Fixed 5%', 'TF-MoDISco'], [fixed['positive']['mean_sequence_fdp'], field['positive']['mean_sequence_fdp']], color=['#4878a8', '#459d80'])
    axes[0].set(title='Tile FDP on positive sequences', ylabel='Mean false-discovery proportion', ylim=(0, 1))
    axes[1].bar(['Fixed 5%', 'TF-MoDISco'], [fixed['positive']['mean_power'], field['positive']['mean_power']], color=['#4878a8', '#459d80'])
    axes[1].set(title='Tile power on positive sequences', ylabel='Mean fraction of needed tiles selected', ylim=(0, 1))
    axes[2].bar(['Native seqlets', 'Mapped tiles'], [native['mean_positive_sequence_fdp'], field['positive']['mean_sequence_fdp']], color=['#459d80', '#d08038'])
    axes[2].set(title='Evaluation unit changes the result', ylabel='Mean FDP on positive sequences', ylim=(0, .15))
    for a in axes:
        a.spines[['top', 'right']].set_visible(False)
        for bar in a.patches:
            a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + .007, f'{bar.get_height():.1%}', ha='center', fontsize=9)
    fig.suptitle('TF-MoDISco v0.5.1.1 seqlet stage on the frozen DFIM CNN')
    fig.savefig(out / 'field_comparison.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
