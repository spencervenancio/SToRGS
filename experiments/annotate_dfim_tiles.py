"""Label 20-bp tiles of the frozen DFIM test split from planted annotations."""
import argparse
import csv
import gzip
import hashlib
import json
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from genomics.paths import ROOT
from genomics.tiles import ROLES, annotate_tiles, parse_embeddings


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', nargs='?', default='experiments/configs/genomics_dfim_tiles.yaml')
    parser.add_argument('--output')
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text())
    if args.output:
        cfg['output_directory'] = args.output
    assert cfg['overlap_rule'] == 'any_positive_base_overlap'
    assert cfg['primary_role_priority'] == list(reversed(ROLES))
    assert cfg['target'] == 'task3'
    out = ROOT / cfg['output_directory']
    if out.exists():
        raise FileExistsError(out)
    data = ROOT / cfg['data_directory']
    split_path = ROOT / cfg['split_file']
    indices = np.load(split_path)[cfg['split']]
    with gzip.open(data / 'sequences.fa.gz', 'rt') as f:
        lines = f.read().splitlines()
    names, sequences = [s[1:] for s in lines[::2]], lines[1::2]
    with gzip.open(data / 'labels.txt.gz', 'rt') as f:
        labels = list(csv.DictReader(f, delimiter='\t'))
    with gzip.open(data / 'simulation_metadata.txt.gz', 'rt') as f:
        annotations = list(csv.DictReader(f, delimiter='\t'))
    assert len(names) == len(labels) == len(annotations)
    assert len(indices) == 9000 and len(set(indices)) == 9000
    hashes = {name: digest(data / name) for name in
              ['sequences.fa.gz', 'labels.txt.gz', 'simulation_metadata.txt.gz']}
    model_metadata = json.loads((split_path.parent / 'metadata.json').read_text())
    assert hashes == model_metadata['input_sha256'], 'Data changed since model training'
    output_rows = []
    for position, idx in enumerate(indices):
        idx = int(idx)
        name, sequence, label, annotation = names[idx], sequences[idx], labels[idx], annotations[idx]
        assert name == label['name'] == annotation['seqName']
        assert sequence == annotation['sequence'] and len(sequence) == cfg['sequence_length']
        instances = parse_embeddings(annotation['embeddings'], sequence)
        assert bool(int(label['task1'])) == any(m['motif'] == 'ELF1' for m in instances)
        assert bool(int(label['task2'])) == any(m['motif'] == 'SIX5' for m in instances)
        tiles = annotate_tiles(sequence, instances, bool(int(label[cfg['target']])), cfg['tile_width'])
        assert ''.join(t['tile_sequence'] for t in tiles) == sequence
        for tile in tiles:
            output_rows.append(dict(test_position=position, source_index=idx, sequence_name=name,
                                    target=int(label[cfg['target']]), **tile))
    assert len(output_rows) == 90000
    summary = dict(sequences=len(indices), tiles=len(output_rows),
                   primary_roles=dict(Counter(r['role'] for r in output_rows)),
                   primary_roles_by_target={str(y): dict(Counter(r['role'] for r in output_rows
                                                                if r['target'] == y)) for y in [0, 1]},
                   mixed_role_tiles=sum(r['mixed_role'] for r in output_rows),
                   annotation_tile_deletion_changes_label=sum(
                       r['annotation_tile_deletion_changes_label'] for r in output_rows),
                   motif_instances_crossing_tile_boundaries=sum(
                       1 for i in indices for m in parse_embeddings(annotations[i]['embeddings'], sequences[i])
                       if m['start'] // cfg['tile_width'] != (m['end'] - 1) // cfg['tile_width']))
    # Create result directory only once the annotation computation has succeeded.
    out.mkdir(parents=True)
    (out / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=True))
    with (out / 'tiles.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0]))
        writer.writeheader()
        for row in output_rows:
            writer.writerow(dict(row, motif_instances=json.dumps(row['motif_instances'], separators=(',', ':'))))
    shape = (len(indices), cfg['sequence_length'] // cfg['tile_width'])
    arrays = {key: np.array([r[key] for r in output_rows]).reshape(shape)
              for key in ['tile_sequence', 'role', 'has_needed_motif', 'has_unused_motif', 'has_decoy',
                          'is_background', 'mixed_role', 'annotation_tile_deletion_changes_label',
                          'signal_overlap_bp', 'decoy_overlap_bp']}
    np.savez_compressed(out / 'tiles.npz', source_indices=indices,
                        sequence_names=np.array([names[i] for i in indices]),
                        targets=np.array([int(labels[i]['task3']) for i in indices]), **arrays)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    def git(*cmd):
        return subprocess.run(['git', '-C', str(ROOT), *cmd], capture_output=True, text=True).stdout.strip()
    sources = [Path(__file__), ROOT / 'genomics/tiles.py']
    provenance = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                      git_sha=git('rev-parse', 'HEAD'), git_dirty=bool(git('status', '--porcelain')),
                      input_sha256=hashes, split_sha256=digest(split_path),
                      source_sha256={str(p.relative_to(ROOT)): digest(p) for p in sources},
                      numpy=np.__version__, pyyaml=yaml.__version__, randomness='none; frozen split reused')
    for p in sources:
        shutil.copy2(p, out / p.name)
    (out / 'metadata.json').write_text(json.dumps(provenance, indent=2) + '\n')
    (out / 'README.md').write_text('''# DFIM test-tile annotations

Reproduce from the repository root:
`python -m experiments.annotate_dfim_tiles experiments/configs/genomics_dfim_tiles.yaml --output results/genomics_dfim_tiles_rerun`
Use the input hashes in metadata.json and the existing frozen model split.
No model inference, retraining, selection or randomization was performed.

Coordinates are zero-based half-open. Each sequence has ten 20-bp tiles.
Any positive base overlap counts, so a boundary-spanning motif marks both tiles.
CSV stores tile DNA, original row/name, target, copy counts and overlapping motif
instances (JSON with original coordinates and overlap lengths). NPZ arrays are
(9000,10) in the exact saved test-index order; source indices/names/targets are
one-dimensional. CSV is sorted by test position, then tile index.

Four primary roles: needed_motif = planted ELF1/SIX5 in a positive sequence;
present_but_unused_motif = planted ELF1/SIX5 in a negative sequence;
decoy = AP1/TAL1 without overlapping signal motif; background = no planted overlap.
For mixed-role tiles, needed > unused > decoy > background; independent flags and
motif records retain all roles. Unannotated chance matches remain background.

Needed is the planned union-of-planted-motifs convention, NOT individual-copy
necessity, a model-relative null, or a proved randomization-test null. Repeated
copies remain needed under this convention. annotation_tile_deletion_changes_label
separately marks positive tiles intersecting every copy of at least one required
motif. It assumes overlap disrupts the instance and ignores new motif creation;
it is not a replacement-based effect calculation. Negative-sequence unused
labels apply to the intended positive-AND convention, not all possible edits.
''')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
