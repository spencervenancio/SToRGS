# Published field baseline: TF-MoDISco v0.5.1.1 seqlet calling

Decision: replace the ad hoc pooled-count ratio as the field reference with a
named published selection stage. Source: Avsec et al., Nature Genetics 2021,
BPNet Methods, https://pmc.ncbi.nlm.nih.gov/articles/PMC8812996/.
TF-MoDISco v0.5.1.1 tag resolves to a12b47b72418666854219791bc3ba8ec9234d71d.
Run its FixedWindowAroundChunks implementation, not a custom threshold surrogate.

Published settings: signed DeepLIFT scores with zero reference; 21-bp windows;
nominal target_seqlet_fdr=.01; null from 4,800 dinucleotide-reshuffled inputs.
Release workflow defaults: flank10, suppress20, passing-window fraction .03-.20,
separate_pos_neg_thresholds=False, no total seqlet cap. Preserve signed windows,
isotonic calibration, both sign tails, fallback/caps and greedy extraction.
No custom threshold repair, test-label tuning or nominal-q sweep.
This measures the seqlet-selection stage, before motif clustering/CWM scanning.
Do not claim reproduction of BPNet itself or its full motif-instance pipeline.

Necessary transfers to this benchmark, declared before running:
- Freeze our task3 CNN instead of BPNet's profile predictor; explain its scalar
  task3 logit (there is no profile output). Use Captum DeepLift Rescale with zero
  reference instead of BPNet's legacy Keras implementation. Check completeness.
- Replace AdaptiveMaxPool1d(1) with mathematically equivalent MaxPool1d(200) only
  to expose the pooling rule to Captum; verify predictions/gradients agree.
- Pool the 9,000 test maps without labels as the field procedure pools maps in
  its analysis corpus. Unlike BPNet's TF-bound peaks, this includes negative
  synthetic examples. Report positive and negative strata separately; this is
  an explicit difference in input domain, not evidence of biological failure.
- Choose 4,800 distinct test indices uniformly without labels, one dinucleotide
  shuffle each, seed20261027 with independent streams via lfs.seed.spawn.
  Shuffling uses the already tested DeepLIFT-style pinned-successor algorithm.
- 200-bp sequences replace 1kb peaks. Keep published21-bp windows and native
  flanks/suppression, NOT force the caller to20-bp nonoverlapping tiles.
- Save native41-bp seqlets. Map their central21-bp cores to our fixed20-bp tiles
  by any positive overlap for tile-level comparisons, matching annotation overlap
  convention. This mapping is evaluation only, not a change in seqlet extraction.

Administrative compatibility only: move deprecated sklearn.neighbors.kde import
to sklearn.neighbors; redirect upstream diagnostic plot paths to run outputs.
Load upstream util/core/value_provider/coordproducers without package __init__
(which imports unrelated legacy clustering/TensorFlow); preserve method bodies.
Archive exact original source, patches and runtime hashes.

Nearest controls: retain fixed5% result and ad hoc pooled-control result, but
mark the latter exploratory/non-field. Native seqlet selections are the field
reference; tile overlap is only a comparison endpoint. Existing planted-mask
labels stay frozen. No result-dependent parameter changes.

Hypotheses: native selection produces useful nonempty regions, or its field
assumptions/settings transfer poorly. Readouts: raw/final thresholds, whether caps
activated, signed core sums, native seqlet counts, mapped tile counts, per-sequence
FDP and power, positive/negative strata and primary tile categories. These are
empirical planted-mask results, not guaranteed model-relative or biological FDR.
Stop if signed attribution completeness or numerical equivalence fails; fix
implementation defects before reporting. Empty selections remain valid outputs,
not a reason to tune. A result contrary to expected utility falsifies only this
named method/config transfer; the unchanged fixed cutoff remains a comparator.

Pre-run numerical probe: Captum DeepLift's completeness residual on sixteen
random backgrounds was up to .0644 logit units (~.21% of output-reference
change), unchanged by float64/native convolution/single-example execution. Treat
this as the implemented attribution rule's residual, not rounding or something
to silently repair. Before full-run acceptance, require residual <=1% of
max(1,abs(output-reference)) +1e-4 for every analyzed input. Save full residuals.
If it fails, report and diagnose; do not claim the reference baseline is complete.
PyTorch marks Captum's max_unpool backward as lacking a deterministic kernel.
Use deterministic_algorithms(warn_only=True) for attribution, CPU single thread,
and require repeated attribution on a subset to agree. Our global pool has only
one output per channel, so overlapping unpool writes cannot occur within it;
record this compatibility setting rather than asserting the strict flag passed.

Attribution-engine correction, before any selector outcome: Captum failed the
completeness gate (max normalized residual17.7%); saved as
results/genomics_dfim_tfmodisco_captum_failed. Retire that attribution engine for
this run, not the seqlet method. Tangermeme1.6.0's genomics DeepLIFT implementation
on the same16 probe inputs achieved max absolute residual9.54e-6, versus
Captum .0644, while keeping the same model and zero reference. Use that released
implementation with one explicitly supplied zero reference, hypothetical=False,
target0, no attribution averaging, no controls-as-reference. Run strict deterministic
algorithms=True again (tangermeme includes an appropriate unpool implementation).
Same completeness gate and all seqlet settings remain frozen. Preserve both
engine source/provenance; distinguish the modern PyTorch engine from BPNet's
legacy Keras engine. This changes an implementation defect before measuring the
field selector, not a threshold selected from evaluation results.
Tangermeme's fast nonoverlapping-unpool path also triggers PyTorch's conservative
strict-determinism flag. Retain warn_only=True and verify repeated outputs; do not
claim strict deterministic-kernel coverage. No numeric rule is patched.
