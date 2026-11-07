# Pinned inputs from other projects

| file | source | used by |
|---|---|---|
| `threshold_functions.json` | senior-thesis repo, `results/saliency_threshold_function_family_benchmark/threshold_functions.json` (copied 2026-11-07, sha256 `4d6435a1…873ef`) | `experiments/run_dfim_threshold_transfer.py`, `tests/test_threshold_transfer.py` |

The file is the frozen `all_family` threshold rule fitted on the thesis's
simulated saliency benchmark. It is an input to this project and is never refit
here. `tests/fixtures/thesis_observable_features.py` is a verbatim copy of the
thesis helper that computed its features.

`dfim_download_manifest.json` records the source commit, sizes and SHA-256s of
the DFIM data in `data/dfim/`. The data itself is not redistributed.
