# DFIM task3 predictor training

Decision: obtain a frozen predictor suitable for later selection experiments.
Hypotheses: a small sequence CNN learns the planted AND label, or predictive
performance is inadequate. Nearest baseline: constant training prevalence.
One architecture, seed 20261025, stratified 70/15/15 split by the three label
patterns; 20 epochs, select minimum validation BCE, evaluate test only afterward.
Readouts: BCE, AUROC, accuracy, confusion matrix and accuracy by label pattern.
Acceptance: validation AUROC >= .95; test results are descriptive, not tuning input.
If the gate fails, keep this as a failed baseline and design another run explicitly.
No selection or FDR claim is tested. Planted motifs are withheld during training.
Deterministic CPU float32, one thread; rerun from scratch and compare every tensor.
Full source snapshots, split indices, source hashes, environment, resolved config,
training history and checkpoint accompany outputs. Dirty repository state is recorded;
source snapshots, rather than git SHA alone, define this run.

Implementation timing adjustment before the completed run: native Conv1d backward
cost ~0.6s/batch on this CPU. An initial run was interrupted after early epochs
(no reported final model). Use unfold + linear convolution, same mathematical
architecture and initialization, ~0.05s/batch. Floating-point operation order
changes, so the final source snapshot defines the reproduction implementation.
