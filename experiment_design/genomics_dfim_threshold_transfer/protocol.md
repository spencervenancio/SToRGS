# Frozen mathematical-function threshold transfer
Apply all_family rule in results/saliency_threshold_function_family_benchmark/
threshold_functions.json, matching the formula in Spencer's meeting notes.
No retraining, recalibration, coefficient rounding, or DNA-dependent rule choice.
Score input is the existing (9000,10) absolute observed-base logit-gradient tile
sums from results/genomics_dfim_baselines/scores.npz. This is a declared transfer
from scalar mathematical features to20-bp groups; both domains have10 features.
Reuse original max normalization (floor1e-12), eight summaries, population std,
NumPy linear-interpolated quantiles, raw-scale coefficients (already account for
the fitted StandardScaler), clip[0,1], inclusive u>=threshold. All-zero maps are
flagged degenerate and select nothing, not assigned an inferred meaningful set.
Primary comparison fixed5% on identical scores; secondary practical-method
comparison TF-MoDISco uses different attributions/native selection units.
Readouts: selected count, per-sequence FDP (including empty sets as0), power on
positives only, positives/negatives and tile types, threshold range/zero clipping.
Hypotheses: shape-dependent thresholds improve precision without excessive power
loss, or mathematical-function calibration transfers poorly. No metric-dependent
follow-up sweep; retain baseline regardless. These are empirical planted-mask
results, not guarantees or evidence of general genomics transfer. The original
oracle-min training objective prioritizes retaining important features, not FDR.
Save rule/source/config/input hashes, summaries, thresholds, calls and reproduction
instructions. Verify feature extraction against the original helper by AST loading
its function without the missing lfs.data imports; check scale invariance and raw
vs standardized formula equivalence before reporting.
