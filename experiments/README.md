# Experiments (DFIM planted-motif benchmark)

Standalone runners, run from the repo root. Reusable code is in `genomics/`,
protocols in `experiment_design/<name>/`, outputs in `results/<name>/`.

## Pipeline

Each stage reads the previous stage's `results/` directory through its config.

| # | stage | entry point (`python -m experiments.…`) | config | writes |
|---|---|---|---|---|
| 1 | train CNN | `train_dfim_cnn configs/genomics_dfim_cnn.json`, then `audit_dfim_cnn` | `genomics_dfim_cnn.json` | `results/genomics_dfim_cnn/` |
| 2 | tile labels | `annotate_dfim_tiles` | `genomics_dfim_tiles.yaml` | `results/genomics_dfim_tiles/` |
| 3 | fixed / pooled baselines | `run_dfim_baselines`, then `verify_…`, `report_dfim_baselines` | `genomics_dfim_baselines.yaml` | `results/genomics_dfim_baselines/` |
| 4 | TF-MoDISco field baseline | `run_dfim_tfmodisco`, then `verify_…`, `report_…`, `plot_dfim_field_baseline` | `genomics_dfim_tfmodisco.yaml` | `results/genomics_dfim_tfmodisco/` |
| 5 | threshold transfer | `run_dfim_threshold_transfer` | `genomics_dfim_threshold_transfer.yaml` | `results/genomics_dfim_threshold_transfer/` |

`train_dfim_cnn` needs the config path spelled out
(`experiments/configs/genomics_dfim_cnn.json`). The other runners
default to their config in `configs/`. Stage 4 also needs `h5py` and the
vendored packages under `data/third_party/` named in its config.

## Dependence on the senior-thesis saliency work

Stage 5 applies a rule fitted in the thesis repo, pinned as
`external/threshold_functions.json`. `tests/test_threshold_transfer.py` checks
the feature summaries against a verbatim copy of the thesis helper in
`tests/fixtures/`.

## Paths before 2026-11-07

Results written before the split from the thesis repo cite the old commands,
for example `python -m experiments.run_dfim_baselines` run from the thesis
root, with configs in `experiments/configs/` there and `lfs.genomics` as the
package. The module and config names are the same here. Only the package
(`lfs.genomics` became `genomics`) and the repo root changed. Every results
directory also contains copies of the exact scripts that produced it.
