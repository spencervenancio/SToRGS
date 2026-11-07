"""Audit saved DFIM predictions and exact-string task difficulty."""
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from genomics.paths import DATA_DIR, RESULTS_DIR


def main():
    out = RESULTS_DIR / 'genomics_dfim_cnn'
    with gzip.open(DATA_DIR / 'dfim/sequences.fa.gz', 'rt') as f:
        sequences = f.read().splitlines()[1::2]
    with gzip.open(DATA_DIR / 'dfim/labels.txt.gz', 'rt') as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    target = np.array([int(r['task3']) for r in rows])
    splits = np.load(out / 'splits.npz')
    predictions = np.load(out / 'predictions.npz')
    # These fixed strings are read from the planted-annotation vocabulary, not fitted.
    rule = np.array(['AACCCGGAAGTG' in s and 'AATAGGGTATCATATAT' in s for s in sequences])
    vocabulary = {}
    with gzip.open(DATA_DIR / 'dfim/simulation_metadata.txt.gz', 'rt') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            for embedding in row['embeddings'].split(','):
                motif, sequence = embedding.split('_', 1)[1].rsplit('-', 1)
                vocabulary.setdefault(motif, set()).add(sequence)
    result = {'planted_strings_by_motif': {k: sorted(v) for k, v in vocabulary.items()}}
    for name in splits.files:
        idx = splits[name]
        logits = predictions[name + '_logits']
        positives = logits[target[idx] == 1]
        negatives = logits[target[idx] == 0]
        result[name] = {
            'cnn_accuracy_recomputed': accuracy_score(target[idx], logits >= 0),
            'cnn_auroc_recomputed': roc_auc_score(target[idx], logits),
            'fixed_string_rule_accuracy': accuracy_score(target[idx], rule[idx]),
            'minimum_positive_logit': float(positives.min()),
            'maximum_negative_logit': float(negatives.max()),
        }
    result['split_intersection_sizes'] = {
        a + '_' + b: len(np.intersect1d(splits[a], splits[b]))
        for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]
    }
    result['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out / 'performance_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
