"""Standalone deterministic training: python -m experiments.train_dfim_cnn CONFIG."""
import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import platform
import random
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import sklearn
import torch
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, roc_auc_score


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--root', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--model-source', type=Path)
    args = parser.parse_args()
    root = args.root or Path(__file__).resolve().parents[1]
    cfg = json.loads(Path(args.config).read_text())
    out = args.output or root / cfg['output_directory']
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    model_source = args.model_source or root / 'genomics/cnn.py'
    spec = importlib.util.spec_from_file_location('dfim_model', model_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    seed = cfg['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(cfg['threads'])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False
    data = root / cfg['data_directory']
    with gzip.open(data / 'labels.txt.gz', 'rt') as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    with gzip.open(data / 'sequences.fa.gz', 'rt') as f:
        lines = f.read().splitlines()
    names, seqs = lines[::2], lines[1::2]
    assert [n[1:] for n in names] == [r['name'] for r in rows]
    assert len(seqs) == 60000 and all(len(s) == 200 for s in seqs)
    assert len(set(seqs)) == len(seqs), 'Duplicate sequences need grouped splitting'
    lookup = {b: i for i, b in enumerate('ACGT')}
    encoded = np.array([[lookup[b] for b in s] for s in seqs])
    x = torch.from_numpy(np.eye(4, dtype=np.float32)[encoded].transpose(0, 2, 1).copy())
    y = torch.tensor([float(r[cfg['target']]) for r in rows])
    groups = np.array([r['task1'] + r['task2'] for r in rows])
    rng = np.random.default_rng(seed)
    split = {k: [] for k in ['train', 'validation', 'test']}
    for group in sorted(set(groups)):
        idx = rng.permutation(np.flatnonzero(groups == group))
        a = int(len(idx) * cfg['split_fractions'][0])
        b = a + int(len(idx) * cfg['split_fractions'][1])
        for key, part in zip(split, [idx[:a], idx[a:b], idx[b:]]):
            split[key].extend(part.tolist())
    split = {k: np.array(sorted(v), dtype=np.int64) for k, v in split.items()}
    model = module.SequenceCNN(cfg['channels'], cfg['kernels'], cfg['hidden'])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'],
                                 weight_decay=cfg['weight_decay'], foreach=False)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    batch = cfg['batch_size']

    def predict(idx):
        model.eval()
        with torch.no_grad():
            return torch.cat([model(x[v]) for v in np.array_split(idx, max(1, int(np.ceil(len(idx) / batch))))]).numpy()

    history, best, best_state, best_epoch = [], float('inf'), None, None
    start = time.time()
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        order = split['train'][torch.randperm(len(split['train'])).numpy()]
        total = 0.
        for offset in range(0, len(order), batch):
            idx = order[offset:offset + batch]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x[idx]), y[idx])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        logits = predict(split['validation'])
        val = torch.nn.functional.binary_cross_entropy_with_logits(torch.from_numpy(logits), y[split['validation']]).item()
        row = dict(epoch=epoch, train_bce=total / len(order), validation_bce=val,
                   validation_auroc=roc_auc_score(y[split['validation']], logits))
        history.append(row)
        if val < best:
            best, best_epoch = val, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(json.dumps(row), flush=True)
    model.load_state_dict(best_state)
    # Create results only after training has produced computed outputs.
    out.mkdir(parents=True)
    (out / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=True))
    (out / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    torch.save(best_state, out / 'model.pt')
    np.savez(out / 'splits.npz', **split)
    metrics = {}
    predictions = {}
    for key, idx in split.items():
        logits = predict(idx)
        probs = torch.sigmoid(torch.from_numpy(logits)).numpy()
        truth = y[idx].numpy()
        predictions[key + '_logits'] = logits
        metrics[key] = dict(n=len(idx), positives=int(truth.sum()),
                            bce=log_loss(truth, probs, labels=[0, 1]),
                            auroc=roc_auc_score(truth, logits),
                            accuracy=accuracy_score(truth, logits >= 0),
                            confusion_matrix=confusion_matrix(truth, logits >= 0).tolist(),
                            accuracy_by_pattern={g: accuracy_score(truth[groups[idx] == g], logits[groups[idx] == g] >= 0) for g in sorted(set(groups))})
    prevalence = float(y[split['train']].mean())
    metrics['constant_baseline_test'] = dict(accuracy=float(np.mean(y[split['test']].numpy() == (prevalence >= .5))),
                                           auroc=.5, bce=log_loss(y[split['test']], np.full(len(split['test']), prevalence), labels=[0, 1]))
    np.savez(out / 'predictions.npz', **predictions)
    (out / 'history.json').write_text(json.dumps(history, indent=2))
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    tensor_hash = hashlib.sha256()
    for key, value in best_state.items():
        tensor_hash.update(key.encode())
        tensor_hash.update(value.numpy().tobytes())
    def git(*cmd):
        return subprocess.run(['git', '-C', str(root), *cmd], capture_output=True, text=True).stdout.strip()
    meta = dict(seed=seed, best_epoch=best_epoch, seconds=time.time() - start,
                parameter_count=sum(p.numel() for p in model.parameters()),
                python=platform.python_version(), platform=platform.platform(), machine=platform.machine(),
                torch=torch.__version__, numpy=np.__version__, sklearn=sklearn.__version__,
                git_sha=git('rev-parse', 'HEAD'), git_dirty=bool(git('status', '--porcelain')),
                tensor_sha256=tensor_hash.hexdigest(),
                input_sha256={n: sha(data / n) for n in ['sequences.fa.gz', 'labels.txt.gz', 'simulation_metadata.txt.gz']},
                state_shapes={k: list(v.shape) for k, v in best_state.items()})
    (out / 'metadata.json').write_text(json.dumps(meta, indent=2))
    shutil.copy2(__file__, out / 'train_dfim_cnn.py')
    shutil.copy2(model_source, out / 'cnn.py')
    (out / 'requirements-freeze.txt').write_text(subprocess.run(['python', '-m', 'pip', 'freeze'], capture_output=True, text=True).stdout)
    (out / 'torch-build.txt').write_text(torch.__config__.show())
    (out / 'README.md').write_text('# DFIM task3 trained CNN\n\n'
        'Standalone reproduction (from this directory):\n\n'
        '```sh\npython train_dfim_cnn.py config.json --root /path/to/data-root --model-source cnn.py --output /path/to/new-output\n```\n\n'
        'The data root must contain data/dfim with the three original gzip files. '
        'Match metadata.json, torch-build.txt and requirements-freeze.txt for retraining. '
        'Load model.pt with SequenceCNN from cnn.py and load_state_dict(torch.load(..., weights_only=True)) '
        'to recover the exact saved weights directly. Input is float32 N×4×200, channels A,C,G,T; '
        'output is one logit; sigmoid gives task3 probability. No reverse-complement augmentation. '
        'No normalization, dropout, batch normalization, scheduler, clipping or class weighting. '
        'PyTorch default layer initialization and Adam defaults (betas .9/.999, eps 1e-8, amsgrad false). '
        'Each epoch uses torch.randperm after initialization; final short batch is retained. '
        'Select strictly smallest validation BCE (earliest tie), not final epoch. '
        'Convolutions use stride 1, dilation 1, groups 1, bias true and zero padding. '
        'CPU one thread, float32, deterministic algorithms, MKLDNN disabled. '
        'Split generator is NumPy default_rng; sorted strata 01,10,11, sorted indices within splits. '
        'The script is the authoritative executable specification. '
        'Cross-platform bitwise retraining is not promised; the checkpoint is exact.\n\n'
        'See metrics.json, history.json, metadata.json and splits.npz. '
        'No feature selection or error-control claim is made.\n')
    print(json.dumps(dict(metadata=meta, metrics=metrics)), flush=True)


if __name__ == '__main__':
    main()
