# CLAUDE.md

Research project: per-sequence (local) feature selection with error control for
regulatory DNA models, benchmarked on DFIM planted motifs. Split out of the
senior-thesis repo (`../thesis`, public at spencervenancio/senior-thesis) on
2026-11-07. The goal is papers, not a thesis.

`README.md` orients. `agent_notes/lab_log.md` records what was run. The research
record of what Spencer actually adopted is the thesis repo's
`notes/meeting_notes.tex`, which Spencer writes. Never add prose there.

## Commands

```bash
pip install -e ".[dev]"     # ,tfmodisco for the field baseline
pytest
ruff check genomics experiments tests
```

## Conventions

- **Agent notes and human notes are separate.** Every Markdown note an agent
  writes (summaries, reports, research notes, the lab log) goes in
  `agent_notes/` and is committed. `notes/` is for Spencer's own notes only.
  Agents never write there.

- **Never report an experiment from an unsaved inline command.** Every reported
  result needs a runnable entry point in `experiments/`, a config in
  `experiments/configs/`, and a self-contained `results/<name>/` directory with
  the resolved config and a README. Exploratory work must be labelled exploratory.
- **Designs never go in `results/`.** Protocols, hypotheses and acceptance
  criteria go in `experiment_design/<name>/`. Create `results/<name>/` only once
  the run has produced artifacts. Frozen protocols are not edited after the run.
- **Never move or edit an executed results directory.** Its config, README and
  source hashes record the paths in effect when it ran.
- **The lab log is append-only.** Old entries keep the paths they were written with.
- **Pass `rng=`, and give each (sequence, tile) its own stream via
  `genomics.seed.spawn`.** A shared Generator silently correlates draws.
- **Paths come from `genomics.paths`.** No absolute paths, no `os.chdir`.
- **Plot functions return `(fig, ax)`** and never call `plt.show()`.
- `external/` inputs are frozen and are never refit here. Record a new source and
  sha256 in `external/README.md` before replacing one.

## Never commit

This repo is public.

- `data/`: DFIM has no license and must not be redistributed. The vendored
  tangermeme and tfmodisco code is there too.
- `results/`: commit the config, not the artifact.
- `lit/**/*.pdf`.
- Local absolute paths (`/Users/...`) in any committed file. Check `git status`
  before any broad `git add`.

## Open problems

Do not paper over these.

- No selection rule here has per-sequence error control yet. The fixed relative
  cutoff, the pooled shuffle null and the transferred thesis rule are baselines,
  not answers.
- Tiles where a motif is present but unused (lone ELF1 or SIX5, redundant copies)
  are the expected failure case, because the label depends on the pair.
