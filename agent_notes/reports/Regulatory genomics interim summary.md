# Regulatory genomics local feature selection: interim summary

*Compiled the evening of 2026-10-25 from the summaries the five research strands handed back, as insurance while the full report is still being written. The full report, `reports/Regulatory genomics local feature selection.md`, supersedes this once it exists. The detailed, cited notes are already saved in `research_notes/Regulatory genomics local feature selection/`: `landscape_and_gap.md`, `ground_truth.md`, `statistical_approaches.md`, `tooling_feasibility.md` and `local_repo_state.md`. Nothing here has been run as an experiment.*

## Bottom line

- **Concern 1 is half right.** Genomics attribution is already local, and per-sequence *sets* already exist: seqlets, motif hits, CREME tile calls. No existing tool gives those sets a per-sequence error rate against a stated null.
- **Concern 2: ground truth exists, in three kinds.**
  - Planted motifs give exact truth about the generating function.
  - Wet-lab mutagenesis gives noisy, per-sequence biological truth.
  - Exhaustive perturbation of the fitted model gives exact truth about f̂ only.
  - Each supports a different claim.
- **Tangible next step:** a planted-motif simulation with a small CNN. Compare per-sequence selection rules on per-sequence FDR and power, broken out by motif role (isolated, redundant pair, repressor, gate on or off). The CNN trains in under a minute on the M3.

## Concern 1: what is already local, and what the gap is

**Already done:**
- **Attribution maps** (DeepLIFT/DeepSHAP, ISM, saliency) are per-sequence scores.
- **Seqlet calling** selects spans.
  - tfmodisco-lite's default `target_seqlet_fdr=0.2` is pooled over all sequences. It comes from isotonic regression against a Laplace fit to the observed scores, with no shuffled null in that code path, and caps on the fraction of windows passing override it.
  - BPNet (2021) used a dinucleotide-shuffled null at FDR 0.01.
  - tangermeme's `recursive_seqlets` docstring says outright: "These p-values rank spans; they are not error rates… a threshold of 0.01 does not mean that 1% of the calls are false." Its optional `X_control` mode gives a *pooled* FDR against dinucleotide-shuffled controls.
- **Motif-hit calling** produces the per-sequence sets practitioners actually use, and it has no p-values or FDR.
  - BPNet's CWM scanning uses percentile cutoffs.
  - Fi-NeMo is a non-negative lasso with a default λ of 0.7, and its README says parameters "should be selected holistically".
- **Other local methods:**
  - CREME (Toneyan & Koo 2024) works one locus at a time but uses fixed cutoffs and no p-values.
  - SQUID gives coefficient maps, not sets.
  - ChromBPNet variant scores have calibrated local p-values, against 10⁶ dinucleotide-shuffled contexts.
- **Every one of these nulls is about the fitted model f̂**, not the true sequence→activity function.

**Is the 0/1 question useful?** Yes. Motif counts, spacing analyses, cooperativity claims and the choice of what to mutate in the lab all start from called instances.

**The gap.** Wanted: for one sequence x, select motif instances or windows (groups of positions, not single bases), with:
- a per-sequence FDR or FWER against a stated local null;
- robustness to redundancy and to a non-monotone value function (CREME reports "necessity does not imply sufficiency");
- an explicit statement of whether the claim is about f̂ or about biology.

General machine learning has local tests with error control. Burns, Thomason & Tansey (2020) apply BH or BY across the subsets of a single input; see also SHAP-XRT (2023) and Teneggi & Sulam (2024). No application to genomic sequence models was found.

**Novelty status**, from the statistical strand. The search was not exhaustive for 2025–2026 preprints, so check before claiming any of these.

| Idea | Status |
|---|---|
| Per-sequence BH on shuffled nulls | Partially done (DFIM 2018, TF-MoDISco, tangermeme, AlphaGenome sham-deletion null 2026), but none states its null precisely or validates per-sequence FDR against planted truth |
| Hierarchical or window-level testing for genomic DNNs | Apparently open; templates exist (KnockoffZoom, multilayer knockoff filter, TreeBH) |
| Comparing selection rules under redundancy | Apparently open |
| Exact CRT with a dinucleotide/Markov conditional sampler | Apparently open. If the simulated background comes from the same sampler and test sequences are held out, FDR control is a theorem |
| Attribution sum as the CRT statistic (a per-query calibrated saliency threshold) | No genomics precedent found |

The exact sampler is the researcher's own derivation, not found in the literature: uShuffle with k=2 and fixed endpoints, applied to the window extended by one base on each side, or Markov-bridge sampling.

## Concern 2: ground truth, and which claim each kind supports

1. **Synthetic planted motifs: truth about the generating function.**
   - Existing benchmarks score "all planted positions," not "positions the label depends on at x."
   - None has a min/limiting-factor output, saturation, a repressor, or a mask that changes with context. That is an inference worth noting: your function-family designs already cover these.
   - **DFIM (2018).** 60k × 200 bp; tasks are ELF1 present, SIX5 present, and both (an AND gate). AP1 and TAL1 are decoys independent of the label. Planted positions are in `simulation_metadata.txt.gz` (4.2 MB), and a trained model is included. No license is listed.
   - **Koo `synthetic_code_dataset.h5`** (17 MB, `github.com/p-koo/exponential_activations/data`).
     - 20k × 200 bp, about 50% positives.
     - The `model_*` arrays hold each sequence's generating probabilities, so the mask is information content above 0.1.
     - Planted motifs appear in both classes, so in negatives they are decoys, and not clean nulls.
     - Koo's Task 1/3 data is also on Zenodo 4301062 (MIT license).
2. **Wet-lab perturbations: per-sequence biological truth, noisy, one assay and one cell type.**
   - **Kircher 2019 saturation-mutagenesis MPRA (CAGI5).** 21 elements and 31,243 single-edit effects, with per-element agreement of about 0.55–0.99. Single edits can't reveal redundancy. Open portal: kircherlab.bihealth.org/satMutMPRA; raw data at GEO GSE126550.
   - **DeepSTARR oligo screens** are the only data found that measure redundancy directly, by knocking out one motif instance or all of them. The model and data are on Zenodo 5502060 (CC BY 4.0). Some of these numbers come from a preprint summary and are unverified.
3. **Model-relative: exhaustive ISM or ablation on f̂. Exact, noiseless, and silent about biology on its own.**
   - It is the field's faithfulness reference; gradient correction is mean-centered ISM.
   - CREME calls itself "strictly a model interpretability tool."

This lines up with the 09-21 notes. Layer 3 is the "null about f̂" case; layers 1–2 are needed for any claim about f = E[Y|X].

## Proposed experiment (not yet run)

The statistical and ground-truth strands converged on this.

- **Data:** planted JASPAR motifs on a Markov background. A probe of a hand-written generator using memelite made 20k × 200 bp with masks in 0.19 s. Alternatively, use DFIM or the Koo file.
- **Model:** a small CNN. A probe on the Koo data trained 30 epochs in 28.6 s on M3 MPS, with test AUROC 0.980.
- **Arms:**
  1. exact CRT with change in prediction as the statistic, using the generator's own background model as the conditional sampler;
  2. exact CRT with the window's attribution sum as the statistic;
  3. shuffle-null + BH (what the field does);
  4. a seqlet-style threshold (tfmodisco-lite or tangermeme).
- **A possible fifth arm** (inferred from the repo survey's notes on what can be reused): your two-stage decoy rule, with decoy *windows* in place of N(0,1) decoy columns. Those columns are not exchangeable with one-hot channels, but the decoy, mirror and BH scoring code works unchanged on (queries × windows) arrays.
- **Strata:** isolated motif, redundant pair, repressor, and gate-on vs. gate-off. These map onto your never-used / gate / inactive-branch error breakdown.
- **Metrics:** per-sequence FDR and power within each stratum.
- **Decision rule:**
  - If shuffle-null BH controls FDR as well as the exact CRT in every stratum, the contribution is comparing selection rules under redundancy.
  - Otherwise, the exact CRT (plus the attribution-sum statistic) is the contribution.
- **Budget:** BH resolution needs K ≥ m/α − 1 conditional draws per window, so 50 windows at α = 0.1 needs K ≥ 499.
- **Expectations:**
  - Resampling one copy of a redundant pair misses both copies. That costs power, not FDR control.
  - Repressors need two-sided statistics.
- **Simpler fallback for tonight:** use the DFIM data.
  - Compute the exact effect on f̂ of removing every combination of the k planted instances (2^k patterns).
  - Score selections against the causal instances, and use the decoys to estimate the false-discovery proportion.
  - Report gate-on and gate-off sequences separately.

## Blockers before running (verified this evening)

- **`experiments/run.py` fails to import on `main`.**
  - `lfs/data/__init__.py` and `lfs/data/patches.py` are missing. Both exist on `design/complexity-vs-selection` and on `origin/maintenance-pass-2026-08-21`.
  - The cause is `.gitignore:42`: `data/` also matches `lfs/data/`.
  - The same rule ignores `lfs/data/function_family_benchmark.py`.
- **`lit/AI Summaries/` is untracked and not ignored**, so a broad `git add` would publish it. `research_notes/` and `reports/` are in the same state.
- **No genomics libraries are installed.**
  - Probe-checked install line: `pip install tangermeme==1.6.0 memelite==0.5.0 logomaker==0.8.7`. It is all wheels and leaves torch and numpy unchanged in a dry run.
  - Don't install gReLU (it downgrades captum), chrombpnet (it needs TensorFlow 2.8) or shap 0.54.1 (it needs Python ≥3.12).
  - bpnet-lite needs `--no-deps`.
- **MPS pitfalls:**
  - Cast the Koo file's float64 arrays to float32.
  - Gradient correction subtracts the mean over the channel axis: axis 1 for (N, 4, L).
- **Repo conventions:** the design goes in `experiment_design/<name>/` and the config in `experiments/configs/`. Create `results/<name>/` only after the run, with `config.yaml` and `README.md`.

## Exploratory probe timings (not results)

These are single runs in a throwaway venv on the M3. The scripts are in `research_notes/Regulatory genomics local feature selection/tooling_probes/`.

| Model | Method | Time |
|---|---|---|
| Koo-sized CNN, 4,000 sequences, MPS | gradient-corrected saliency | ~2 s |
| | tangermeme DeepLIFT/SHAP, 20 references | ~10 s |
| | captum IG | ~10 s |
| | ISM | ~50 s |
| BPNet-sized model, 2114 bp, MPS | ISM | 5.6 s per sequence (~1.6 h per 1,000) |

At BPNet size, send ISM to the cluster.

## Repo context worth carrying into the meeting

- **The two-stage decoy rule passed all five pre-registered confirmation claims:** F1 0.814 vs. 0.780 for the Lasso, FDR 0.133 vs. 0.205. Gate and inactive-branch features are still selected at 0.34–0.58, so the open problem is now explicitly the local part.
- **The Null Importance paper is `lit/2609.19511v1.pdf`** (Raskutti, Sankaran & Ye, arXiv Sep 2026). It is probably the paper sent before 09-21; that is inferred, since the notes don't name it.
- **Every saliency result was produced from a dirty git tree** (`git_dirty = True`).
