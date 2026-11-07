#!/usr/bin/env bash
# EXPLORATORY install/feasibility checks, run 2026-10-25 on Apple M3 / macOS 14.4.1 / conda-forge Python 3.10.13 / torch 2.11.0.
# Design-deciding only; NOT a thesis result. Nothing is installed into the active env (dry runs + a throwaway venv).
# Usage: SP=<scratch dir> bash install_checks.sh   (run from this directory so the probe scripts are found)
set -u
SP=${SP:-$(mktemp -d)}
# 1. Wheels-only resolution against the existing env (installs nothing)
python3 -m pip install --dry-run --only-binary=:all: tangermeme modisco-lite memelite logomaker pyjaspar   # resolves (wheels only)
python3 -m pip install --dry-run --only-binary=:all: finemo              # ResolutionImpossible: no pybigwig distribution
python3 -m pip install --dry-run --only-binary=:all: bpnet-lite==1.1.0   # No matching distribution found for macs3
python3 -m pip install --dry-run --only-binary=:all: bpnet-lite          # silently backtracks to bpnet-lite 0.8.1
python3 -m pip install --dry-run --only-binary=:all: inmotifin           # resolves
python3 -I pypi_meta.py tangermeme bpnet-lite modisco-lite memelite captum logomaker simdna pyjaspar finemo grelu chrombpnet kipoi pyfaidx pybigwig numba shap modisco multimolecule inmotifin
# 2. Throwaway venv that reuses the existing torch/numpy/captum
python3 -m venv --system-site-packages "$SP/venv_tm"
"$SP/venv_tm/bin/pip" install --only-binary=:all: tangermeme
"$SP/venv_tm/bin/pip" install --no-deps bpnet-lite==1.1.0 simdna==0.4.3.2
"$SP/venv_tm/bin/pip" install --no-deps finemo==0.41 && "$SP/venv_tm/bin/pip" install --only-binary=:all: polars jaxtyping hdf5plugin jinja2
"$SP/venv_tm/bin/python" -I -c "import finemo"   # fails: No module named 'pyBigWig' (finemo/data_io.py line 26)
"$SP/venv_tm/bin/pip" install --only-binary=:all: inmotifin==4.1.1 && "$SP/venv_tm/bin/python" -I -m inmotifin --help
# 3. Probes (sibling scripts)
"$SP/venv_tm/bin/python" -I bench.py
"$SP/venv_tm/bin/python" -I recheck.py
python3 -I apis.py
mkdir -p "$SP/dl_koo" "$SP/dl_hf" "$SP/dl_jaspar"
"$SP/venv_tm/bin/python" -I probe2.py "$SP"
"$SP/venv_tm/bin/python" -I probe3.py
"$SP/venv_tm/bin/python" -I train_probe.py "$SP/dl_koo/synthetic_code_dataset.h5"
# 4. Ground-truth marking by class in the Koo datasets (exact snippet that was run; the second file errors because
#    every test sequence in the 12-task synthetic_dataset.h5 has >=1 positive label, so the "negative" subset is empty)
python3 -I -c "
import h5py, numpy as np
for fn in ['synthetic_code_dataset.h5','synthetic_dataset.h5']:
    with h5py.File('$SP/dl_koo/'+fn,'r') as f: M=f['model_test'][:]; y=f['Y_test'][:]; X=f['X_test'][:]
    ic=2+np.sum(np.where(M>0, M*np.log2(np.clip(M,1e-12,1)),0),axis=1)
    pos=ic>0.1; anyy=y.max(1)>0
    print(fn, '| X one-hot exact:', bool(np.all(X.sum(1)==1)), '| frac positions IC>0.1: label+ %.3f, label- %.3f' % (pos[anyy].mean(), pos[~anyy].mean()), '| seqs w/ no marked pos: + %d, - %d' % ((pos[anyy].sum(1)==0).sum(), (pos[~anyy].sum(1)==0).sum()), '| marked pos per seq median + %d / - %d' % (np.median(pos[anyy].sum(1)), np.median(pos[~anyy].sum(1))))
    agree=(X.argmax(1)==M.argmax(1))[pos].mean(); print('   at marked positions, X base == argmax(model) in %.2f of cases' % agree)
"
