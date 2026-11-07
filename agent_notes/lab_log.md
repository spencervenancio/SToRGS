# Genomics predictor training log

## 2026-10-25 — DFIM task3 CNN
Goal: freeze a predictive model for subsequent local selection experiments, with
an independently reproducible weight artifact. Protocol:
experiment_design/genomics_dfim_cnn/protocol.md.

Action: implemented lfs/genomics/cnn.py and experiments/train_dfim_cnn.py, using
experiments/configs/genomics_dfim_cnn.json. Checked known missing lfs/data files;
used an independent runner rather than experiments/run.py. Existing user edits
were retained, no staging or commits performed.

A native Conv1d CPU attempt was interrupted because batches took ~0.6s. Switched
to equivalent unfold+linear convolution (~0.05s exploratory timing) before the
completed run. Mathematical forward and input-gradient equivalence checked in
tests/genomics/test_dfim_cnn.py; `python -m pytest -q` on that file passes.
Bare `pytest` could not find lfs in this environment; module invocation resolves it.
Ruff passes on the new runner, model package and test.

Completed 20 epochs in 143.57s. Chosen epoch 20 by validation BCE. Validation
AUROC 1, test AUROC 1, test accuracy 8999/9000; constant baseline 6000/9000.
Artifacts: results/genomics_dfim_cnn/, especially MODEL_REPORT.md, model.pt,
config.yaml, metadata.json, splits.npz and history.json. The repo was dirty;
standalone exact source snapshots accompany the checkpoint.

Inference (high confidence within this synthetic split): predictive adequacy gate
passes. This is not evidence of correct attribution or FDR control. Strongest
alternative to intended motif reliance is reliance on unintended sequence cues;
future motif/decoy perturbations can falsify intended reliance. Decision: retain
this predictor for later selection work; do not infer selection validity.

Independent full rerun from source snapshots writes to
results/genomics_dfim_cnn_reproduction/. Verification status is recorded in
results/genomics_dfim_cnn/reproduction_verification.json when complete.

Verification completed: independent 20-epoch retraining matched all weight
tensors, predictions, splits and history bit for bit. Handoff archive includes
original data, standalone source, config, environment and checkpoint.

## 2026-10-25 — test-tile annotation
Executed `python -m experiments.annotate_dfim_tiles` with
experiments/configs/genomics_dfim_tiles.yaml; protocol in
experiment_design/genomics_dfim_tiles/protocol.md. Used frozen test indices,
validated names, sequences, annotation strings, labels and input hashes against
training metadata. Produced 90,000 tiles: background 53,451; decoy 12,391;
present-but-unused 14,555; needed 9,603. 1,668 mixed signal/decoy tiles retain
both flags; primary role prioritizes signal. Boundary overlaps mark both tiles.
Unit tests: 3 passed; Ruff passes. Independent deterministic rerun in
results/genomics_dfim_tiles_verification produced identical CSV and NPZ arrays.
Artifacts and reproduction instructions: results/genomics_dfim_tiles/.
Needed/unused are the planned planted-mask convention, not fitted-model or
replacement-test nulls; copy counts and annotation-only deletion flags retained.
Decision: tiles are ready for scoring; no selection or error-control inference.

## 2026-10-25 — matched attribution baselines
Protocol frozen in experiment_design/genomics_dfim_baselines/protocol.md.
Executed `python -m experiments.run_dfim_baselines` using
experiments/configs/genomics_dfim_baselines.yaml. Shared score is absolute
observed-base logit gradient summed over each 20-bp tile (absolute input×gradient).
Fixed fraction .05 inherited from existing select_threshold default. Pooled rule
uses validation-only control-tail ratio target .1, 10 independent dinucleotide
shuffles per validation sequence, with 900,000 control tile scores. This is a
transparent tile adaptation, not an exact TF-MoDISco or BPNet implementation.
Current remotely fetched tangermeme main lacks the X_control option claimed in
previous notes, so that purported interface was not used.

Artifacts: results/genomics_dfim_baselines/, including config, input/model/split
hashes, source snapshots, control tokens, base/tile scores, complete calibration
curve, selections, individual sequence/tile tables, report and figure.
Fixed: 62,430 selected; mean sequence FDP .821813 overall, .465439 on positives;
positive-sequence mean power .966722. Every negative has a selection, hence FDP1.
Pooled: no candidate met .1 (minimum estimated ratio .141829), so selects none;
FDP0 and power0, not a useful controlled-discovery result. No q adjustment made.

Verification: four unit tests pass; Ruff passes. Recomputed first256 tile scores
match bitwise. Native vs custom CNN forward outputs match exactly on eight test
examples; gradients differ by at most 7.16e-7. Independent brute-force tail counts
reproduce the minimum calibration ratio (3,225 observed vs 4,573 control tails,
scaled for ten controls). All90,000 controls preserve dinucleotide counts and
endpoints. Commands: python -m experiments.verify_dfim_baselines;
python -m experiments.report_dfim_baselines.

Inference (medium): under the chosen planted-mask convention, the fixed cutoff
is permissive and the pooled control adaptation provides no useful discoveries.
This does not establish failure of published pipelines or success of a prospective
per-sequence method. A score/control mismatch remains possible; counterfactual
model dependence and sampler-relative ground truth remain distinct. Retain both
frozen baselines. No(c) implementation or guarantee claim has been made.

## 2026-10-25 — correction: measured published seqlet-stage reference
User rejected the ad hoc pooled-tail ratio as a field baseline. Marked that run
explicitly exploratory/non-field in its report and README; its zero-selection
result is not evidence against published methods.

Named replacement: original TF-MoDISco v0.5.1.1, commit
 a12b47b72418666854219791bc3ba8ec9234d71d, seqlet-calling stage used by BPNet2021.
Protocol experiment_design/genomics_dfim_tfmodisco/protocol.md records published
choices, release defaults and benchmark transfers separately. Upstream numeric
method is unchanged; only deprecated sklearn import and figure paths patched.
Use signed DeepLIFT zero-reference attribution,21bp sliding windows,.01 nominal
seqlet FDR,4,800 dinucleotide-shuffled inputs, release-default flank10/suppress20
and .03-.20 passing-window caps. No motif clustering/CWM-scanning claim.

Captum's DeepLift failed pre-selection completeness (max normalized residual
.176986); retained results/genomics_dfim_tfmodisco_captum_failed. A numeric probe
of tangermeme1.6.0 reduced maximum absolute residual to9.54e-6. The same DeepLIFT
method/reference and unchanged selector were then run with this released engine.
Full accepted run max residual3.15e-5, max normalized6.82e-6. PyTorch unpool is
flagged conservatively, so warn_only=True; repeated128 maps match bitwise.
The mathematically equivalent pooling wrapper has bitwise-equal ordinary outputs
and gradients. This is a transfer to our frozen CNN, not a BPNet model reproduction.

Executed experiments/run_dfim_tfmodisco.py with matching YAML. Both thresholds
and all6,371 native coordinates/scores reproduced under independent refitting.
No passing-window cap activated. Native thresholds -31.8404/+17.5926. Initial
export hit a NumPy-int JSON serialization error after saving native HDF5. Reporter
recovered native signed scores with upstream's own window-sum function (HDF5
coordinates omit score), then exported results without recalibration. Corrected
exporter and exact executed source both retained. No selection changes made.

Measured field reference:6,371 native seqlets;12,606 mapped tiles; positive-sequence
tile power .695861, positive tile meanFDP .102289, all-sequence meanFDP .314985.
Native core meanFDP .002722 on positives, .281796 overall. Strong evaluation-unit
effect: correct native cores can project onto neighboring background tiles.
Inference(medium): the native caller provides useful selections and a much
stronger practical baseline than the ad hoc empty-selection run. The all-sequence
corpus includes negative synthetic examples unlike BPNet's bound-peak corpus;
the planted-mask convention is not a fitted-model null. These findings do not
establish per-sequence FDR control or validate a future method(c). Retain original
fixed cutoff comparator and this named reference; do not call the ad hoc rule
"the field's approach". Outputs:results/genomics_dfim_tfmodisco/, especially
FIELD_BASELINE_REPORT.md, native_seqlets.csv/h5, thresholds, attributions,
per_sequence.csv, tile_selections.npz, original/runtime upstream sources/diffs,
verification.json and field_comparison.png.

Final validation: all9 genomics tests pass; Ruff passes on field-runner, reporter, verifier, plotter, upstream adapter and mapping test.

## 2026-10-25 — frozen mathematical-rule transfer to DNA tiles
User requested their trained mathematical-function threshold setter. Located
all_family in saliency_threshold_function_family_benchmark/threshold_functions.json;
it exactly matches the rounded .1432/.0702/.3960/-.1922/-.1616 meeting-note formula.
Used full precision exported raw-scale coefficients and original max-normalized
8-summary preprocessing. No fitting, DNA calibration, or rule sweep. Source rule
trained on24 functions,10 scalar features; transferred to10 existing20bp scores.
This grouping/score-domain change is explicit in the design and result report.

Protocol:experiment_design/genomics_dfim_threshold_transfer/protocol.md.
Command:python -m experiments.run_dfim_threshold_transfer. Config in matching YAML.
Saved outputs:results/genomics_dfim_threshold_transfer/, including frozen_rule,
input/source hashes, raw features/thresholds/calls and per-sequence/tile tables.
Tests: feature summaries agree exactly with AST-isolated original helper;
standardized and raw-coefficient predictions agree; scale invariance/zero maps
checked. Two tests passed; Ruff passes on new helper, runner and test.

Result:65,108 tiles selected. Mean sequence FDP .833816 overall, .501447 on
positives; positive power .984861. Matched fixed5%: .821813/.465439/.966722.
Threshold median .0621095, range0-.361233;2,176(24.18%) clip to zero and select all
10 tiles; no degenerate inputs. Retain this frozen result without post-hoc repair.
Inference(medium): modest recall increase costs precision; this rule's original
minimum-important objective is not false-discovery calibration and does not
provide a strong off-the-shelf transfer result here. This conclusion applies to
this rule and tile representation only. Existing fixed and TF-MoDISco baselines
remain valid comparators; the TF-MoDISco comparison additionally changes attribution.

## 2026-11-07 — Repository reorganization (no runs)
Moved the genomics thread into a genomics/ subfolder of each layer, separate
from the early saliency/MinShap work. Nothing was rerun, and results/ was not
touched.
- experiments/*dfim*.py -> experiments/genomics/; module paths are now
  experiments.genomics.<script>.
- experiments/configs/genomics_dfim_* -> experiments/genomics/configs/.
- experiment_design/genomics_dfim_* -> experiment_design/genomics/.
- reports/ -> notes/reports/genomics/; research_notes/ (this log included) ->
  notes/research_notes/genomics/.
Entries above cite the old paths; they are left as written. Script edits
changed only paths, imports, and train_dfim_cnn's repo-root depth
(parents[1] -> parents[2]), so source SHA-256s recorded in earlier results no
longer match the working tree. The copies stored in each results directory are
the authoritative versions.

## 2026-11-07 — Split into its own repository (no runs)
The genomics thread now lives in its own repo (this one) instead of inside the
senior-thesis repo, because the goal is papers rather than a thesis. Nothing was
rerun. results/, data/dfim and data/third_party were moved without edits.
- lfs/genomics/ -> genomics/; lfs.paths and lfs.seed -> genomics/paths.py and
  genomics/seed.py (seed.py copied verbatim).
- experiments/*dfim*.py and their configs -> experiments/ and experiments/configs/.
- experiment_design/genomics_dfim_* -> experiment_design/.
- Thesis reports/ and research_notes/ -> notes/reports/ and notes/research_notes/.
  This log was research_notes/genomics_training_log.md.
- Threshold-transfer input pinned as external/threshold_functions.json (sha256
  4d6435a1a07676c9406386ac69a1dce00a9be47ec96877e5ec30e234486873ef). The thesis
  observable_features helper was copied verbatim into tests/fixtures/.
Entries above cite thesis-repo paths and are left as written. The script edits
changed only imports and paths, so source SHA-256s recorded in earlier results
no longer match the working tree. The copies inside each results directory are
authoritative.

## 2026-11-07 — notes/ renamed to agent_notes/ (no runs)
Agent-written Markdown (this log, reports/, research_notes/) now lives in
agent_notes/ and is version-controlled. notes/ is reserved for Spencer's own
notes. Local absolute links in local_repo_state.md were rewritten as plain-text
senior-thesis paths. The repo directory is now ~/Downloads/SToRGS.
