# SToRGS – Saliency Thresholding of Regulatory Genomic Sequences

Sequence-attribution tools already make per-sequence calls (seqlets, motif
hits) but attach no error rate to them. This project asks whether the important
regions of a *single* sequence can be selected with a stated false-discovery
rate, using a planted-motif benchmark where the true regions are known.

It grew out of the senior-thesis work on local feature selection
([spencervenancio/senior-thesis](https://github.com/spencervenancio/senior-thesis))
and is now a separate project. Nothing here imports the thesis code. The one
thesis artifact it uses is pinned in `external/`.

## Benchmark

DFIM synthetic data (Greenside et al. 2018): 60,000 random 200-bp sequences
with ELF1 and/or SIX5 planted. The task-3 label is ELF1 AND SIX5. AP1 and TAL1
decoys are planted independently of the label. Each test sequence is cut into
ten 20-bp tiles, and each tile is labelled from the planted positions
(background, decoy, required motif, or motif that is present but not used).

## Layout

```
genomics/            package: sequence CNN, tiles, baselines, seqlet mapping, threshold transfer
experiments/         standalone runners; configs/ holds one config per experiment (see its README)
experiment_design/   frozen pre-run protocols, one folder per experiment
tests/
agent_notes/         agent-written summaries, kept apart from human notes
  lab_log.md         dated log of what was run and why
  reports/           research reports (literature and design surveys)
  research_notes/    cited notes behind the reports, plus exploratory tooling probes
notes/               Spencer's own notes (human-written; agents do not write here)
external/            pinned inputs from other projects, with provenance
data/                downloaded data and vendored upstream code (gitignored)
results/             outputs of executed runs (gitignored)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # add ,tfmodisco for the TF-MoDISco baseline
pytest
```

Data: download `sequences.fa.gz`, `labels.txt.gz` and
`simulation_metadata.txt.gz` from
[kundajelab/dfim](https://github.com/kundajelab/dfim/tree/c3e22d7074f40ca29671488ce8a016d2c27a1541/data/embedded_motif_interaction_data)
into `data/dfim/`, and check them against `external/dfim_download_manifest.json`.
The TF-MoDISco baseline also needs tangermeme 1.6.0 and tfmodisco v0.5.1.1
under `data/third_party/`, as named in `experiments/configs/genomics_dfim_tfmodisco.yaml`.

## Running

Runners are executed from the repo root, in pipeline order. See
`experiments/README.md`.

```bash
python -m experiments.train_dfim_cnn experiments/configs/genomics_dfim_cnn.json
python -m experiments.annotate_dfim_tiles
python -m experiments.run_dfim_baselines
```

Every runner refuses to overwrite an existing results directory, and it copies
its resolved config, its source files and their SHA-256s into the directory it
writes.
