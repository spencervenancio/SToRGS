# DFIM tile selection baselines: frozen protocol

Decision: establish held-out performance of fixed relative and pooled shuffle
thresholding before implementing per-sequence randomization. This run does not
implement (c), tune thresholds using test annotations, or assert FDR control.

One frozen CNN/checkpoint, original validation/test splits and test tile labels.
Score: sum over a tile of absolute observed-base logit gradient,
sum_position abs(sum_channel onehot * d(logit_task3)/d(onehot)). This is absolute
input×gradient for one-hot input. Sum of per-base magnitudes, not absolute signed
sum. No gradient correction, normalization, or sigmoid. Same score for both rules.

(a) select s_ij >= .05*max_j(s_ij), unless all ten scores are zero (select none).
This inherits lfs.selection.saliency.select_threshold's default relative fraction,
adapted to tile sums. No tuning or claimed error level.

(b) calibrate on all 9,000 validation sequences, with 10 independent
whole-sequence dinucleotide shuffles each (90,000 shuffled sequences). Preserve
first/last base and exact counts of all dinucleotides. Use the shuffled-successor
algorithm with final outgoing successor pinned, as in DeepLIFT; not claimed to
sample uniformly over every possible dinucleotide-preserving permutation.
Score all ten tiles of each shuffled sequence using the same frozen CNN.
At each distinct positive observed validation score t, estimate pooled false
fraction as min(1, [(1 + #control_scores>=t) * n_observed_tiles/n_control_tiles]
/ #observed_scores>=t). Select the lowest t with estimate <= .10; if none exists,
select no tiles. No passing-fraction cap, isotonic fit, seqlet extraction or motif
clustering. Apply the resulting single frozen absolute threshold to all test
sequences. This is a transparent pooled-control baseline inspired by genomics
shuffle-null calibration, NOT exact TF-MoDISco, BPNet, or tangermeme reproduction.
References: https://pmc.ncbi.nlm.nih.gov/articles/PMC8812996/ (pooled shuffle null);
https://raw.githubusercontent.com/kundajelab/deeplift/master/deeplift/dinuc_shuffle.py
(successor shuffle). Current remote tangermeme main does not expose the optional
X_control interface claimed in earlier research notes; do not rely on that claim.

RNG: seed 20261026, independent child stream per replicate via lfs.seed.spawn;
single-threaded CPU float32 deterministic attribution. Freeze all choices above.
Evaluate all 9,000 test sequences: per-sequence FDP=V/max(R,1), power=TP/number
needed tiles only when number needed>0; retain NA otherwise. Report mean FDP
(empirical FDR over sampled sequences), pooled V/sum(R), positive/negative strata,
and selection rate by each primary tile role. Decoy and unused selections count
as false under the planted-mask convention. Report mixed flags additionally.
Per-sequence FDP is observed, not a per-sequence FDR guarantee; no claim about
individual-sequence repeated-randomization validity. Threshold q=.1 is a pooled
calibration target, not an established true FDR bound. Validation was already used
for checkpoint selection; test remains held out. Background chance motifs and
CNN receptive-field spillover may differ from planted-mask roles.

Hypotheses: pooled controls improve precision over the permissive fixed cutoff;
or whole-sequence controls fail to represent null tiles beside signal. Falsifier:
held-out FDP and role-specific errors contradict the first explanation. One run,
no rescue sweep. Retain both baseline outputs regardless of outcome for future
matched comparison with (c). Acceptance: integrity checks, analytical gradient
check, exact dinucleotide preservation and threshold/tie tests pass; save all
raw scores, controls, masks, exact config, provenance, source and README before
reporting. No performance-based stopping or adjustment.
