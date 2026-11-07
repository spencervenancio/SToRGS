# Regulatory-genomics local interpretation pipeline: where binary selection happens, how cutoffs are set, and what error control exists

Provenance: source-code facts come from the GitHub `main` branches as fetched on 2026-10-25 (defaults can change between releases). Paper facts come from full texts (PMC/Europe PMC XML, arXiv, CSHL repository PDFs) unless marked "search summary". Where I say what a null hypothesis *is*, I mark it **[stated]** if the source says so and **[inferred]** if I am reading it off the construction.

---

## 1. Which attribution methods are standard (BPNet, ChromBPNet, DeepSTARR, Enformer/Borzoi), what baselines do they use, and what are "hypothetical" vs "actual" contributions?

### Takeaway
Attribution maps are **already local**: each method gives one score map per input sequence and per output head, and the output is always a **score, never a set**. Kundaje-lab models (BPNet, ChromBPNet) use DeepLIFT/DeepSHAP. DeepSTARR uses DeepSHAP with 100 dinucleotide-shuffled references. Enformer and Borzoi use gradient×input, plus in-silico mutagenesis (ISM) or shuffle perturbations. The reference ("baseline") differs across projects: all-zeros for BPNet (2021), 20 dinucleotide shuffles averaged for ChromBPNet, 100 shuffles for DeepSTARR. So "make it local" is not new at this stage. The open question is what happens after the scores exist.

### Cited Findings
**Glossary for a statistics reader**
- **Transcription factor (TF) and motif:** "Combinations of Transcription Factors (TFs) bind combinations of motifs in DNA sequence at non-coding regulatory elements to control gene expression." — [TF-MoDISco technical note v0.5.6.5, arXiv:1811.00416v5 (30 Apr 2020)](https://arxiv.org/abs/1811.00416)
- **PWM:** a first-layer CNN filter "learns a sequence pattern that is analogous to a position weight matrix (PWM). The filter is scanned (convolved) with the sequence to produce a score for the strength of the match at each position." — [TF-MoDISco note](https://arxiv.org/abs/1811.00416)
- **PFM/CWM:** for each discovered motif, the position frequency matrix (PFM) and the contribution weight matrix (CWM) are computed by averaging base frequencies and contribution scores, respectively, over aligned seqlets. — [BPNet, Avsec et al., Nat Genet, Feb 2021 (PMC8812996)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8812996/)
- **Seqlet:** "segments of the input, termed 'seqlets', that have substantial contribution to one or more of the output tasks." — [TF-MoDISco note](https://arxiv.org/abs/1811.00416)
- **ChIP-nexus:** BPNet "uses DNA sequence to predict base-resolution ChIP-nexus binding profiles of pluripotency TFs." — [BPNet](https://pmc.ncbi.nlm.nih.gov/articles/PMC8812996/)
- **Footprint:** "some bound proteins protect the DNA from enzymatic cleavage… characteristic 'footprint' patterns of reduced enzymatic cleavage that align with TF motifs." — [ChromBPNet preprint, Pampari et al., bioRxiv 2024.12.25.630221 (PMC11741299)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11741299/)
- **Hit calling:** Fi-NeMo is "a GPU-accelerated motif instance calling tool for identifying transcription factor binding sites from neural network contribution scores." — [Fi-NeMo README](https://github.com/kundajelab/Fi-NeMo)

**Hypothetical vs actual (projected) contributions**
- "A hypothetical importance score answers the question 'if I were to mutate the sequence at a particular position to a different letter, what would the importance on the newly-introduced letter look like?'" For gradient×input the hypothetical score is just the gradient. For DeepLIFT it is the multipliers at the current input times the difference-from-reference of the hypothetical base. The note says hypothetical scores "behave like an autocomplete of the sequence". — [TF-MoDISco note](https://arxiv.org/abs/1811.00416)
- tfmodisco-lite takes hypothetical scores as input and forms the actual scores as `contrib_scores = np.multiply(one_hot, hypothetical_contribs)`. It sums these over bases (`contrib_scores.sum(axis=2)`) before seqlet calling. Seqlets therefore use the actual, per-position scalar scores, while clustering also uses the hypothetical ones. — [tfmodisco-lite `tfmodisco.py`](https://github.com/jmschrei/tfmodisco-lite/blob/main/modiscolite/tfmodisco.py)
- tangermeme calls the actual scores "projected" attributions, "i.e., the attribution of the observed characters." — [tangermeme `seqlet.py`](https://github.com/jmschrei/tangermeme/blob/main/tangermeme/seqlet.py)

**Model by model**
- **BPNet (2021):** DeepLIFT is adapted to the profile head by explaining a softmax-weighted sum of the profile logits (`wn = tf.reduce_mean(tf.reduce_sum(tf.stop_gradient(tf.nn.softmax(f, dim=−2)) * f, axis=−2), axis=−1)`). "For the reference input r, all zeroes were used since it showed the highest correlation with in-silico mutagenesis contribution scores." — [BPNet Methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC8812996/)
- **ChromBPNet (preprint Dec 2024; PMC Jan 2025):** DeepLIFT/DeepSHAP is applied to the count head and the profile head separately. "We create 20 such [dinucleotide-]shuffled sequences for each input, serving as our reference set… The final contribution scores are obtained by averaging across these multiple references." Bias-corrected scores come from the residual TF sub-model with the Tn5/DNase bias sub-model disconnected. Uncorrected profile contributions "highlighted several spurious bases" that resemble Tn5 bias. Models are trained in a 5-fold cross-validation setup. — [ChromBPNet](https://pmc.ncbi.nlm.nih.gov/articles/PMC11741299/)
- **DeepSTARR (Nat Genet, May 2022):** the contribution script sets `dinuc_shuffle_n=100` ("number of dinucleotide shuffled sequences per sequence as background"), uses `shap.DeepExplainer`, and projects difference-from-reference onto each of the four one-hot bases to get hypothetical scores. — [DeepSTARR contribution-score code](https://github.com/bernardo-de-almeida/DeepSTARR/blob/main/DeepSTARR/DeepSTARR_nucl_contr_scores.py); paper: [de Almeida et al. 2022](https://doi.org/10.1038/s41588-022-01048-5)
- **Enformer (Nat Methods, 2021):** "we computed two different gene expression contribution scores — input gradients (gradient × input) and attention weights". "In silico mutagenesis and gradient × input are tissue- or cell-type-specific, since they are computed with respect to a particular output CAGE sample." ISM around fine-mapped variant rs11644125 "revealed an affected SP1 transcription factor motif." — [Enformer, Avsec et al. (PMC8490152)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8490152/)
- **Borzoi (Nat Genet, 2025):** uses gradient saliencies ("Grad x Inp") of the coverage tracks or the attention matrix. Tissue-specific residual saliency is computed by subtracting the average gradient of all other tissues, over 1,000 tissue-specific genes, and is then clustered with TF-MoDISco v0.5.14.1 and Tomtom/HOCOMOCO. For the perturbation version, enhancers are "dinucleotide-shuffled with a 2 kb window… each shuffle was repeated 16 times for both forward and reverse orientations and for all four model replicates (128 times total)"; Enformer gets 64 shuffles per orientation. — [Borzoi, Linder et al. (PMC11985352)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11985352/)
- **The gradient family as used in genomics:** "saliency maps, integrated gradients, SmoothGrad, and expected gradients". The proposed correction subtracts the per-position mean gradient over nucleotides, μ_l = (1/A) Σ_a G_{l,a}, a fix that is "general for all data with categorical inputs". — [Majdandzic, Rajesh & Koo, Genome Biol 2023 (PMC10169356)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10169356/)
- SQUID's authors say standard attribution methods, including DeepSHAP, "assume that nucleotide effects on DNN predictions are locally additive." — [SQUID, Seitz et al., Nat Mach Intell, Jun 2024](https://doi.org/10.1038/s42256-024-00851-5)

### Inferences
- In the student's language, DeepSHAP with a set of dinucleotide-shuffled references is an interventional, Shapley-style local attribution against a **reference distribution**. That distribution is "dinucleotide shuffles of x". This is exactly where a local value function v_x(S) would be defined in genomics.
- Every attribution here is a property of the **fitted model f̂**. Fold and replicate ensembles (5 folds for ChromBPNet, 4 replicates for Borzoi) are averaged into one map. Fi-NeMo, for example, averages several contribution files ([README](https://github.com/kundajelab/Fi-NeMo)). None of them uses the spread across fits as uncertainty.
- Choosing the reference is choosing the null of "feature absent". The field switched from zeros (BPNet 2021) to shuffles (ChromBPNet, DeepSTARR) without a statistical rationale. The stated reasons were agreement with ISM (BPNet) and keeping dinucleotide composition (ChromBPNet).

### Gaps
- I did not verify which reference the current ENCODE BPNet pipeline (bpnet-refactor) uses. BPNet 2021 used zeros and ChromBPNet uses 20 shuffles.
- I did not check Basenji's (2018) attribution choices, or whether IG or SmoothGrad is used in production by any of these pipelines (Majdandzic only lists them as the gradient family).

---

## 2. Seqlet calling: how do TF-MoDISco (technical note), tfmodisco-lite, and tangermeme identify seqlets, with what null, which FDR parameter, and are thresholds per sequence or pooled?

### Takeaway
Seqlet calling is the **first binary (selection) step**. In every implementation the cutoff is set **once for the whole dataset**: window scores are pooled across all sequences. The calls themselves are made **per sequence** by greedy, non-overlapping extraction.

TF-MoDISco's "FDR" is an isotonic-regression, two-groups estimate. It is computed against one of two nulls: attributions on dinucleotide-shuffled sequences (optional in v0.5, and what BPNet used), or a Laplace curve fitted to the observed scores (the only option in tfmodisco-lite). Hard caps on the fraction of windows allowed to pass (3%–20%) can override it.

tangermeme's `recursive_seqlets` documentation states outright that its default p-values "are not error rates". Its optional FDR mode uses dinucleotide-shuffle control attributions, again pooled over all sequences. No implementation gives a per-sequence error statement.

### Cited Findings
**TF-MoDISco technical note (v0.5.6.5; arXiv v5 dated 30 Apr 2020)**
- Windows: "the importance scores are summed in sliding windows of core size sliding_window_size with a stride size of 1." — [note §3.1](https://arxiv.org/abs/1811.00416)
- Null, option A: "If a null distribution of scores is provided (e.g. by computing importance scores on dinucleotide-shuffled sequences), then the sliding window importance scores are also computed on the null." — [note §3.1](https://arxiv.org/abs/1811.00416)
- Null, option B: "a Laplace distribution is used the characterize the noise. The Laplace distribution is fit in a way that assumes the low-magnitude scores are mostly noise." It is fitted to "each of the 5th, 10th, 15th…95th percentiles of positive window scores", and the curve with "the fastest decrease (i.e. the highest value of lambda)" is kept. Negatives are handled separately. — [note §3.1](https://arxiv.org/abs/1811.00416)
- FDR: "a FDR is calculated at each window score threshold… by estimating the proportion of windows originating from the null distribution at each threshold using isotonic regression." Null windows are reweighted to equal total weight, which "corresponds to a prior of 50% from the null distribution, though the specific choice of prior is less important given that the FDR cutoffs are adjustable by the user." — [note §3.1](https://arxiv.org/abs/1811.00416)
- Caps: if the fraction passing is below `min_passing_windows_frac` ("set to 0.03 by default") or above `max_passing_windows_frac` ("0.2 by default"), thresholds are reset on absolute values to the nearest bound. — [note §3.1](https://arxiv.org/abs/1811.00416)
- Extraction "is then repeated for each sequence": take the highest-magnitude window, expand by `flank_size`, drop windows overlapping it by more than 50%, repeat. Seqlets from different tasks are then unioned. — [note §3.1](https://arxiv.org/abs/1811.00416)
- The note says v0.5.6.5 changed "how the null distribution is handled in Section 3.1" relative to v0.4.2.2. — [note](https://arxiv.org/abs/1811.00416)

**tfmodisco-lite (current reference implementation)**
- `TFMoDISco` defaults: `sliding_window_size=21`, `flank_size=10`, `target_seqlet_fdr=0.2`, `min_passing_windows_frac=0.03`, `max_passing_windows_frac=0.2`, `weak_threshold_for_counting_sign=0.8`, `min_metacluster_size=100`, `max_seqlets_per_metacluster=20000`. Suppression radius is `int(0.5*sliding_window_size)+flank_size` = 20. — [tfmodisco-lite `tfmodisco.py`](https://github.com/jmschrei/tfmodisco-lite/blob/main/modiscolite/tfmodisco.py)
- Null: `_laplacian_null` fits Laplace tails to the **observed** smoothed tracks. The center is the histogram mode; percentiles 5–95; the most aggressive λ per side. It draws 10,000 samples with seed 1234. This code path does not use a shuffled-sequence null. — [tfmodisco-lite `extract_seqlets.py`](https://github.com/jmschrei/tfmodisco-lite/blob/main/modiscolite/extract_seqlets.py)
- `_isotonic_thresholds` fits isotonic regression of observed (label 1) against null (label 0). It clips the implied null fraction to [0.95, 1.0] (`min_frac_neg=0.95`) and takes the first value whose estimated precision is at least 1 − target_fdr. — [extract_seqlets.py](https://github.com/jmschrei/tfmodisco-lite/blob/main/modiscolite/extract_seqlets.py)
- Pooling: window values from **all** sequences are concatenated (and subsampled to 1,000,000 if larger) before thresholds are set. Extraction then runs row by row, i.e. per sequence. — [extract_seqlets.py](https://github.com/jmschrei/tfmodisco-lite/blob/main/modiscolite/extract_seqlets.py)
- Seqlets whose core sum exceeds `threshold` go to the positive metacluster, those below `−threshold` to the negative one, and the rest are discarded. — [tfmodisco.py](https://github.com/jmschrei/tfmodisco-lite/blob/main/modiscolite/tfmodisco.py)
- CLI: the README example uses `-n 2000` seqlets ("this is low, but a good starting point"). The default window is 400 bp around each region's center (`-w`). `-n` "takes top seqlets in order that they are identified, where identification occurs per region (in order that they are inputted)… For unbiased sampling, shuffle the input regions beforehand." — [tfmodisco-lite README](https://github.com/jmschrei/tfmodisco-lite)

**How the field sets these in practice**
- BPNet: "Significant seqlets were selected by computing contribution scores over a width of 21 bp and using the FDR threshold of 0.01 (target_seqlet_fdr). The null distribution was estimated from 4,800 randomly selected peaks with contribution scores computed on reshuffled sequences while preserving dinucleotide counts." This gave 145,748 seqlets. "Due to memory constraints (250 GB), 50,000 seqlets were used for each TF" in clustering. — [BPNet Methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC8812996/)
- ChromBPNet: "we run modisco-lite on all peaks with 1,000,000 seqlets, employing the command 'modisco motifs -n 1,000,000 -w 500'" (v2.2.0). Motifs with fewer than 100 seqlets are filtered out, and the rest are merged with `gimme cluster -t 0.99`. ChromBPNet calls seqlets "high-confidence predictive instances". — [ChromBPNet](https://pmc.ncbi.nlm.nih.gov/articles/PMC11741299/)

**tangermeme** (code on `main`; paper: [Schreiber, Nat Methods, first published 2026-10-08](https://doi.org/10.1038/s41592-026-03254-z); [preprint Aug 2025](https://doi.org/10.1101/2025.08.08.669296))
- `tfmodisco_seqlets` reimplements the TF-MoDISco procedure with the same defaults (window 21, flank 10, target_fdr 0.2, passing fraction 0.03–0.2). — [tangermeme `seqlet.py`](https://github.com/jmschrei/tangermeme/blob/main/tangermeme/seqlet.py)
- `recursive_seqlets` defaults: `threshold=0.01`, `min_seqlet_len=4`, `max_seqlet_len=25`, `additional_flanks=0`, `n_bins=1000`, `X_control=None`, `target_fdr=0.1`. It calls positive seqlets only. — [seqlet.py](https://github.com/jmschrei/tangermeme/blob/main/tangermeme/seqlet.py)
- Null construction: "each attribution is discretized into one of `n_bins` integer bins, and the distribution of summed bins is created for all potential seqlet lengths by treating positions as independent draws from the histogram of all bins." — [seqlet.py docstring](https://github.com/jmschrei/tangermeme/blob/main/tangermeme/seqlet.py)
- Recursion rule: "all spans from `min_seqlet_len` to `max_seqlet_len`, starting at any position in the seqlet, and fully contained by the borders, must have a p-value below the threshold." — [seqlet.py](https://github.com/jmschrei/tangermeme/blob/main/tangermeme/seqlet.py)
- **Explicit disclaimer [stated]:** "These p-values rank spans; they are not error rates. Attributions are usually correlated along the sequence, so neighboring positions tend to be high together and spans reach large sums more often than independent draws would… The null is built from `X`, so it also contains the signal being tested, and a span's p-value depends on the other examples in `X`. As a result, a threshold of 0.01 does not mean that 1% of the calls are false, and the same threshold can mean very different fractions of false calls in different models and datasets." — [seqlet.py](https://github.com/jmschrei/tangermeme/blob/main/tangermeme/seqlet.py)
- **FDR mode [stated]:** with `X_control`, seqlets are called on the controls at 117 thresholds from 0.1 to 1e-30. "Every control seqlet is counted as false, so the estimated false discovery rate at a threshold is the number of control seqlets plus one, scaled to the number of positions in `X`, divided by the number of seqlets called on `X`." The suggested control is "the attributions of several dinucleotide shuffles of each sequence in `X`". The implied null: "A false call here is a seqlet that is no more frequent than in sequences with the same dinucleotide composition. Shuffling can create real motif matches that the model responds to, so the estimate may level off above zero." — [seqlet.py](https://github.com/jmschrei/tangermeme/blob/main/tangermeme/seqlet.py)

### Inferences
- **Local vs global:** the threshold is global, because it is pooled over every window of every sequence. Each sequence's seqlet set is a local output, but any error rate attached to it is a **dataset average**. A sequence with weak signal can contribute only false seqlets while the dataset-level estimate still looks fine.
- **What the null is about:** the shuffled-sequence nulls (BPNet's 4,800 shuffled peaks; tangermeme `X_control`) are nulls about **f̂'s attribution behaviour** on composition-matched random DNA **[inferred for TF-MoDISco; stated by tangermeme]**. The Laplace option is not a hypothesis test at all. It is an empirical-Bayes noise model fitted to the same data, assuming small scores are noise **[stated]**. Neither null conditions on the rest of the sequence x₋S, and neither concerns the true sequence→activity function.
- The tfmodisco-lite "FDR" can be overridden in two ways. The 3%/20% caps fix the selected fraction directly, and `-n` caps the count in input order. The target defaults to 0.2, BPNet used 0.01, and no calibration has been shown.
- The windowing (21 bp core + 2×10 bp flanks; 4–25 bp in tangermeme) is the field's de facto answer to correlated neighbouring positions. Tangermeme states the correlation explicitly. The analogue in images is choosing patches instead of pixels.

### Gaps
- I found no published study comparing TF-MoDISco's nominal seqlet FDR to the realized FDR, for example on synthetic sequences with known motif positions.
- I could not get commit dates for the tangermeme docstring (the GitHub API call failed), so I cannot say when the "not error rates" language was added.

---

## 3. Motif instance ("hit") calling: how do Fi-NeMo, BPNet's CWM scanning, TF-MoDISco, and PWM scanners decide what is a hit? Which parameters are heuristic, and is there any calibration?

### Takeaway
Hit calling is the **per-sequence selection step practitioners actually use**: "region r contains an instance of motif m at position p". Both attribution-based hit callers in current use rely only on heuristic cutoffs:
- **BPNet CWM scanning:** a percentile of the seqlet distribution.
- **Fi-NeMo:** a non-negative lasso with λ = 0.7, plus user-chosen importance cutoffs.

**Neither reports p-values, an FDR, or calibration.** Sequence-only scanners (MOODS, and Tomtom-based seqlet annotation) do report p-values, but those test DNA-sequence similarity to a motif. They do not test whether f̂ (or biology) depends on the instance.

### Cited Findings
- **BPNet CWM scanning:** the match score is a continuous Jaccard similarity between the CWM and the L1-normalized local contributions (best strand). The contrib score is the L1 norm of contributions in the window. Quote: "three criteria were defined based on thresholds identified from the TF-MoDISco's corresponding seqlets: (i) The 'match' score >20th percentile of those of the seqlets. This stringent threshold more effectively discriminates between similar motifs. (ii) The 'contrib' score is higher than the seqlets lowest 'contrib' score. (iii) The log odds score with respect to the PWM derived from the PFM is larger than 0." The Methods give no p-value or FDR for instances. The only FDR anywhere is the 0.01 seqlet target. — [BPNet Methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC8812996/)
- **Fi-NeMo objective [stated in code]:** "minimize_c: ||contribs − conv_transpose(c * importance_scale, cwms) * sequences||²₂ + λ||c||₁ subject to: c ≥ 0". This is a non-negative lasso solved by proximal gradient (`c_next = F.relu(c_next)  # Ensure non-negativity constraint`), with per-motif λ. Contributions are normalized per region by their RMS (`global_scale`). — [Fi-NeMo `hitcaller.py`](https://github.com/kundajelab/Fi-NeMo/blob/main/src/finemo/hitcaller.py)
- **Fi-NeMo parameters:**
  - `-l/--global-lambda` "L1 regularization weight determining the sparsity of hits. Default is 0.7."
  - 0.8–0.9 gives "fewer, higher-confidence hits"; 0.5–0.6 is "more sensitive, may include weaker hits".
  - "Default of 0.7 works well for chromatin accessibility data"; "ChIP-seq data may benefit from lower values (0.6)".
  - `-t/--cwm-trim-threshold` default 0.3; `-w/--region-width` default 1000.
  — [Fi-NeMo README](https://github.com/kundajelab/Fi-NeMo)
- **Fi-NeMo caveats [stated]:**
  - "hits are not identified monotonically: i.e., hits identified from a higher, more confident lambda (e.g., 0.9) will NOT necessarily be identified from a lower, more sensitive lambda (e.g., 0.7). Parameters should be selected holistically."
  - "Scale Invariance: Hit calling depends on motif and contribution score shapes, not absolute magnitudes. Use `hit_coefficient_global` or `hit_importance` for importance-based thresholding." No value is suggested for that cutoff.
  - Outputs: `hit_coefficient`, `hit_coefficient_global` ("This is the primary hit score"), `hit_similarity` (cosine), `hit_importance`, plus per-peak `nll` and `dual_gap`, which are optimization diagnostics.
  - The README, read in full, says nothing about significance, FDR, or calibration.
  - The method paper is still "manuscript in preparation" (Wang, Ramalingam, Tseng, Schreiber, et al.).
  — [Fi-NeMo README](https://github.com/kundajelab/Fi-NeMo)
- **ChromBPNet's use of hit callers:** FiNeMo identified instances of full-depth motifs in models trained on subsampled reads, and full-depth instances served as "ground truth" for recall. For the ChIP-seq benchmark, instances were called by "sequence similarity to the TF-MODISCO PWMs (p ≤ 0.05, MOODS PWM scanner)". Another analysis used "MOODS (p-value = 0.001)" plus a 20 bp window around the ChIP summit. — [ChromBPNet](https://pmc.ncbi.nlm.nih.gov/articles/PMC11741299/)
- **TF-MoDISco itself:** the v0.5.6.5 note has no section on calling motif instances genome-wide. MEME PWMs are used only to *initialize clusters* among seqlets, using "the Bayes optimal threshold specified by MEME for the PWM". — [TF-MoDISco note §6](https://arxiv.org/abs/1811.00416). The tfmodisco-lite README points to Fi-NeMo for "Motif instance detection using TF-MoDISco patterns". — [tfmodisco-lite README](https://github.com/jmschrei/tfmodisco-lite)
- **tangermeme `annotate_seqlets`:** runs TOMTOM on each seqlet's one-hot sequence against a PWM database and returns the most significant match and its p-value. "If no motifs have a statistical significance below the provided threshold, an index of -1 is returned." "The computation is independent for each seqlet." — [tangermeme `annotate.py`](https://github.com/jmschrei/tangermeme/blob/main/tangermeme/annotate.py)

### Inferences
- **Where selection happens:** hits are local sets, one per region. Their cutoffs, though, are tuned globally: BPNet uses percentiles of the pooled seqlet distribution, and Fi-NeMo uses one λ for all regions after per-region rescaling.
- **Fi-NeMo is lasso selection without post-selection inference.** λ is a sparsity penalty, not an error rate. Lasso supports are known not to control FDR in general (general statistical background, not from the sources above). The non-monotonicity in λ is the usual lasso path behaviour under correlated, competing columns (overlapping or similar CWMs).
- **PWM/Tomtom p-values [inferred]** measure how unusual a sequence match is against a random-DNA background model, or how similar a seqlet is to a database motif. A significant PWM match is not evidence that f̂'s prediction at x depends on that instance. ChromBPNet's use of MOODS p ≤ 0.05 to define instances for benchmarking shows that sequence-significance and model-dependence get conflated in practice.
- Because of Fi-NeMo's scale invariance, a region with weak overall activity can still produce large normalized coefficients. Deciding whether a hit matters for the prediction is left to an un-calibrated importance cutoff.

### Gaps
- No public Fi-NeMo benchmark of hit precision/recall or calibration; the manuscript is not out as of this check.
- I did not fetch FIMO/MOODS documentation, so the description of their background-model null is standard background, not a sourced quote.
- I did not check whether the legacy TF-MoDISco v0.5.x code base had a separate hit-scoring module that the technical note omits.

---

## 4. Is the 0/1 (selection) question useful to practitioners? What downstream claims rest on instance calls, and how are they validated?

### Takeaway
Yes. Most of the downstream biology uses discrete instance calls as input: counts per region, pairwise spacing and syntax (helical periodicity, composite elements), cooperativity, variant interpretation, and the choice of which instances to mutate in experiments. Validation comes in two kinds:
- **Tests of f̂:** in-silico injection, marginalization, and ablation.
- **Tests of the true function on selected cases:** CRISPR point mutations (BPNet), more than 40,000 mutant enhancers (DeepSTARR), and QTL/MPRA concordance (ChromBPNet).

Neither kind is used to **calibrate the error rate of the instance calls**.

### Cited Findings
- **BPNet downstream claims:**
  - Spacing: "the pairwise distance between our mapped Nanog motif instances showed a strong helical spacing preference for multiples of ~10.5 bp."
  - Co-occurrence: odds against a region-index permutation null, "p-value <10−5 using Pearson's Chi-squared test".
  - "In-silico motif interaction analysis": 128 random 1 kb backgrounds, motif B placed up to 160 bp away, corrected fold-change (h_AB − (h_B − h_Ø))/h_A; a genomic version replaces motifs with random bases.
  - CRISPR validation: "TFs often cooperate in a directional manner, which we validate using CRISPR-induced point mutations" (Nanog TGA→GGC and Sox2 TTG→AGG in mESCs, read out by ChIP-nexus).
  — [BPNet](https://pmc.ncbi.nlm.nih.gov/articles/PMC8812996/)
- **ChromBPNet downstream claims:**
  - Marginal footprints: "inserting TF/bias motifs of interest within the center of background sequences that are GC-content matched with the peaks", averaging over forward and reverse-complement predictions.
  - The FOS–TEAD composite "shows strong cooperative effects only at the fixed 6 bp spacing."
  - Instance-level contributions correlate with ChIP-based BPNet models (Pearson r > 0.89 for CTCF).
  - Variant scores (logFC, JSD, AAQ, IPS) are benchmarked against caQTL/dsQTL data.
  - A transgenic mouse reporter assay at E11.5 tested one de novo variant.
  — [ChromBPNet](https://pmc.ncbi.nlm.nih.gov/articles/PMC11741299/)
- **DeepSTARR:** the authors mutated all instances of each motif type and measured log2 fold-change in enhancer activity, testing ">40,000 wildtype and mutant Drosophila and human enhancers". They report that "functionally nonequivalent instances of the same TF motif are determined by motif-flanking sequence and intermotif distances" (abstract, via search summary). Public tracks include "mutated motif instances and respective log2 fold-changes in enhancer activity." — [DeepSTARR paper](https://doi.org/10.1038/s41588-022-01048-5); [DeepSTARR README](https://github.com/bernardo-de-almeida/DeepSTARR)
- **The tooling assumes discrete calls:**
  - Fi-NeMo's report writes `motif_occurrences.tsv`, "The number of hits of each motif in each input region". — [Fi-NeMo README](https://github.com/kundajelab/Fi-NeMo)
  - tangermeme provides `count_annotations`, `pairwise_annotations`, and `pairwise_annotations_spacing` over annotated seqlets. — [tangermeme `annotate.py`](https://github.com/jmschrei/tangermeme/blob/main/tangermeme/annotate.py)

### Inferences
- False-positive instances leak directly into spacing histograms, co-occurrence tests, motif counts, and the choice of mutagenesis targets. So the 0/1 question is not academic: it is the input to the field's headline syntax claims.
- The two validation routes correspond to the two nulls. In-silico tests address "does f̂ depend on this instance?". CRISPR, STARR-seq, and MPRA address "does biology depend on it?". Neither route is set up as an error-rate statement over the selected set.
- DeepSTARR's mutant-enhancer data (instance-level experimental log2FC) is a plausible semi-real benchmark for any calibrated local selection method.

### Gaps
- I found no source that reports an instance-level false-discovery proportion, i.e. the fraction of called instances whose mutation has no measurable effect, for any hit caller.

---

## 5. Which existing works already frame position/motif importance as a hypothesis test (GIA, CREME, in-silico marginalization, SQUID), and are those tests local or global?

### Takeaway
- **GIA:** global. Population effect sizes "through the lens of the DNN", with no p-values for single sequences.
- **BPNet/ChromBPNet injection and marginal footprinting:** global. Averages over random or GC-matched background libraries.
- **CREME:** **local**. It perturbs one Enformer locus at a time, but turns averaged effect sizes into calls with **fixed cutoffs** (0.3 / 0.5 / 0.95 / 0.2) and **no p-values or error control**.
- **SQUID:** local surrogate modelling around one sequence. Its output is coefficient maps, not selections.
- **Closest to a calibrated local test:** ChromBPNet's per-variant empirical p-values. Their null, though, is f̂'s behaviour in dinucleotide-shuffled contexts.
- **General ML** already has local, error-controlled tests about a fitted model: Burns et al. 2019/2020, SHAP-XRT 2023, and Miwa et al. 2023. I found no application of these to genomic sequence-to-function models.

### Cited Findings
- **GIA (Koo et al., PLoS Comput Biol 2021):**
  - It "quantifies the population-level effect size that putative patterns have on model predictions". Eq. 1 is I = E_{x^φ_i∼D}[y|x] − E_{x∼D}[y|x], which "quantifies the global importance of pattern ϕ across a population of sequences while marginalizing out contributions from other positions."
  - "GIA quantifies the causal effect size of the interventional patterns through the lens of the DNN and is thus subject to the quality of the learned sequence-function relationship… at its core, a model interpretability tool."
  - Background models include a position-specific profile model, random shuffles, dinucleotide shuffles, and quartile-sampled observed sequences. The authors frame GIA as "an avenue to quantitatively test hypotheses".
  — [GIA (PMC8118286)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8118286/)
- **CREME (Toneyan & Koo; preprint v1 3 Jul 2023; Nat Genet 2024):**
  - Tests: TSS Context Dependence, Context Swap, CRE Necessity, CRE Sufficiency, TSS–CRE Distance, Higher-order CRE Interaction, Multiplicity. All are applied to Enformer, one locus at a time, on 5 kb tiles.
  - Necessity: shuffle the tile, "Repeat 10 times and calculate the mean", effect = (WT − SHUFFLE)/WT.
  - Sufficiency: add the CRE and TSS tiles to a dinucleotide-shuffled sequence, effect = (CRE-TSS-MUTANT − TSS-CONTROL)/WT.
  - Cutoffs: enhancing context ">0.95 (N=152)", neutral "absolute effect size… less than 0.2 (N=40)", silencing "<−0.5 (N=11)". Enhancers are 0.3–0.5 (weak) or ≥ 0.5 (strong), silencers symmetric. Loci with predicted TSS activity below 30 are excluded.
  - Higher-order CRE sets are found by "an iterative greedy search".
  — [CREME preprint](https://www.biorxiv.org/content/10.1101/2023.07.03.547592); published version: [CSHL record of Nat Genet 2024 paper](https://repository.cshl.edu/id/eprint/41673/)
- **CREME's evidence of a non-additive, non-monotone local value function:** "Evidently, necessity does not imply sufficiency". "TSS activity increased when taken out of the original (neutral or silencing) context, suggesting competition among CREs". Enhancer sets are stratified "according to interaction type, redundancy… cooperativity". — [CREME preprint](https://www.biorxiv.org/content/10.1101/2023.07.03.547592)
- **SQUID (Seitz, McCandlish, Kinney & Koo, Nat Mach Intell, Jun 2024):**
  - "A local library is generated by partially mutagenizing a specific sequence of interest", at "a partial mutagenesis rate of 10% per nucleotide", with 100,000 variants.
  - Surrogates: additive or pairwise genotype–phenotype maps, a global-epistasis (GE) nonlinearity, and heteroscedastic skewed-t noise, fitted with MAVE-NN. "SQUID can be viewed as a generalization" of LIME.
  - Its significance tests compare *methods* (attribution variation across loci), not individual positions.
  — [SQUID (CSHL PDF of NMI article)](https://repository.cshl.edu/41591/1/10.1038.s42256-024-00851-5.pdf)
- **ChromBPNet variant p-values:** "we begin by establishing a null distribution of IPS scores through the generation of 1,000,000 synthetic examples. For each example, we perform dinucleotide shuffling on the input reference sequence intended for SNP insertion, followed by the reintroduction of variant alleles into these shuffled sequences… to calculate empirical p-values." — [ChromBPNet](https://pmc.ncbi.nlm.nih.gov/articles/PMC11741299/)
- **General-ML local testing (not genomics):**
  - Burns, Thomason & Tansey "reframe black box model interpretability as a multiple hypothesis testing problem… testing whether the model prediction is significantly different from what would be expected if the features were replaced with uninformative counterfactuals". One of their procedures "provably controls the false discovery rate but… is not yet feasible for large-scale applications". — [arXiv:1904.00045](https://arxiv.org/abs/1904.00045)
  - SHAP-XRT is a local conditional-randomization test whose Shapley marginal contributions bound the expected p-values. — [Teneggi et al., TMLR 2023, arXiv:2207.07038](https://arxiv.org/abs/2207.07038)
  - Miwa, Duy & Takeuchi give selective-inference p-values for a salient region that "can provably control the probability of false positive detections of salient regions". — [arXiv:2301.02437 (Jan 2023)](https://arxiv.org/abs/2301.02437)
- **Knockoff selection in sequence analysis is global:** KIMI uses model-X knockoffs to select k-mers that separate groups of sequences at a target FDR (search summary; not read in full). — [KIMI listing](https://midasnetwork.us/papers/kimi-knockoff-inference-for-motif-identification-from-molecular-sequences-with-controlled-false-discovery-rate/)

### Inferences
- **Classification [inferred]:**

  | Method | Scope | Null is about | Error control |
  |---|---|---|---|
  | GIA | global | f̂ | none |
  | BPNet injection / ChromBPNet marginal footprints | global | f̂ | none |
  | CREME | local (one locus) | f̂ | none: effect-size cutoffs only |
  | SQUID | local | f̂ | none: estimation, not selection |
  | ChromBPNet IPS | local (one variant) | f̂ | empirical p-value against a pooled shuffled-context null |

  The implied IPS null is "this variant's predicted effect is no larger than when the same alleles are placed in dinucleotide-shuffled sequence". That null removes the very context in which the variant acts.
- CREME's ten shuffle replicates per tile are Monte-Carlo draws from a per-locus null. They could support a per-tile randomization p-value, but are only averaged. This is the nearest existing scaffold to a local test with error control.
- CREME's findings that "necessity ≠ sufficiency", that CREs compete, and that redundancy exists are genomic evidence for the thesis's open problem 1. The local value function v_x(S) is **not monotone**. Redundant instances also defeat one-at-a-time (necessity) tests, which is the genomic form of correlated features.

### Gaps
- I read only the CREME preprint (v1). I did not verify whether the Nat Genet 2024 version added statistical tests or a motif-level fine-mapping test with thresholds.
- I found no genomics paper applying CRT- or knockoff-style per-sequence tests to sequence-to-function models.

---

## 6. Published commentary on threshold arbitrariness, attribution noise, false-positive instances, method disagreement, or missing error control

### Takeaway
The field openly acknowledges noisy attributions that vary between models (Koo lab), off-simplex gradient noise, and assay-bias motifs posing as TF motifs (ChromBPNet). It also acknowledges that some thresholds are not error rates: tangermeme's documentation says so, and Fi-NeMo warns that hits are non-monotone in λ. I found **no published work that measures false-positive rates of motif-instance calls** or proposes per-sequence error control.

### Cited Findings
- **Not error rates:** "a threshold of 0.01 does not mean that 1% of the calls are false, and the same threshold can mean very different fractions of false calls in different models and datasets." — [tangermeme `seqlet.py`](https://github.com/jmschrei/tangermeme/blob/main/tangermeme/seqlet.py)
- **Model-to-model inconsistency:** "attribution maps typically harbor spurious importance scores to an extent that varies from model to model, even for DNNs whose predictions generalize well. Thus, the standard approach for model selection, which relies on performance of a held-out validation set, does not guarantee that a high-performing DNN will provide reliable explanations." — [Majdandzic et al., PMLR 200 (MLCB), Nov 2022](https://repository.cshl.edu/id/eprint/40956/)
- **Off-simplex noise:** "random off-simplex function behavior can introduce a random gradient component orthogonal to the simplex, which manifest as spurious noise in the input gradients". In practice attribution maps show "spurious importance scores for seemingly arbitrary nucleotides". — [Majdandzic, Rajesh & Koo, Genome Biol 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10169356/)
- **Rough fitted functions:** "Non-biological noise in attribution maps can arise from the inherent roughness in the DNN function, a phenomenon that is often associated with benign overfitting." — [SQUID](https://repository.cshl.edu/41591/1/10.1038.s42256-024-00851-5.pdf)
- **Instability:** "methods which rely on importance scores can be unstable and unreliable." — [arXiv:2410.06211, "A mechanistically interpretable neural network for regulatory genomics" (Oct 2024)](https://arxiv.org/abs/2410.06211). A search summary also reports that this paper shows IG, DeepLIFT-SHAP, and ISM disagreeing, sometimes in sign, at true motifs in a near-perfect synthetic model. I did not verify this in the full text.
- **Assay bias posing as motifs:** uncorrected profile contributions show "spurious bases" from Tn5 bias. "We advise performing careful verification of bias correction via model interpretation". — [ChromBPNet](https://pmc.ncbi.nlm.nih.gov/articles/PMC11741299/)
- **Threshold choices described as heuristics by their own authors:**
  - The 50% prior "is less important given that the FDR cutoffs are adjustable by the user". — [TF-MoDISco note](https://arxiv.org/abs/1811.00416)
  - Fi-NeMo: "Parameters should be selected holistically". — [Fi-NeMo README](https://github.com/kundajelab/Fi-NeMo)
  - BPNet's 20th-percentile cutoff is justified as "stringent" for telling similar motifs apart, not by any error rate. — [BPNet](https://pmc.ncbi.nlm.nih.gov/articles/PMC8812996/)
  - The seqlet cap depends on input order. — [tfmodisco-lite README](https://github.com/jmschrei/tfmodisco-lite)
- **Localization vs logic:** an ICLR 2026 submission reports that strong motif localization can coexist with poor faithfulness to the rules combining motifs (search summary only). — [ICLR 2026 listing, "Beyond Motif Localization"](https://iclr.cc/virtual/2026/10017715)
- **Review:** Novakovsky et al. survey xAI approaches in genomics and their "underlying assumptions and limitations" (abstract only). — [Nat Rev Genet 2023 (online Oct 2022)](https://doi.org/10.1038/s41576-022-00532-2)

### Inferences
- The critiques focus on **score quality**: noise, robustness, and model selection. The fixes offered are better scores (gradient correction, surrogates, consistency-based model selection, bias factorization). Nobody has posed the inferential question "how many of the hits called in this sequence are false, and false with respect to what?". That is the opening for a statistics thesis.
- The field's cutoffs (λ = 0.7, the 20th percentile, the 3–20% caps, the 0.3/0.5 effect cutoffs) are the same kind of admitted heuristic as the thesis's own saliency `select_threshold`.

### Gaps
- Novakovsky et al. is paywalled. I read only the abstract and cannot quote its specific discussion of thresholds.

---

## 7. Bottom line: what exactly is done, and what is missing?

### Takeaway
**Done:**
- Local scores (attribution maps).
- Local sets (seqlets and hits), but with **global, uncalibrated cutoffs** or plain sparsity penalties.
- Dataset-level FDR-like estimates for seqlets, against nulls about f̂ (shuffled-DNA attributions or a Laplace fit to the same data).
- Local interventional tests without error control (CREME).
- Global interventional effect sizes (GIA, marginalization).
- Local surrogates without selection (SQUID).

**Missing:** a procedure that, for **one sequence x and fitted model f̂**, returns a set of **motif instances or contiguous position groups** with a **stated per-sequence FDR or FWER**, against an **explicitly stated local null**. It must also cope with correlated and redundant groups and a non-monotone local value function, and it must say whether the claim is about f̂ or about the true sequence→activity function.

### Cited Findings
**Stage map** (each row's facts are sourced in sections 1–5):

| Stage | Local or global | Score or set | How the cutoff is set | Guarantee claimed / null |
|---|---|---|---|---|
| Attribution (DeepLIFT/DeepSHAP, grad×input, ISM) — [BPNet](https://pmc.ncbi.nlm.nih.gov/articles/PMC8812996/), [ChromBPNet](https://pmc.ncbi.nlm.nih.gov/articles/PMC11741299/), [Enformer](https://pmc.ncbi.nlm.nih.gov/articles/PMC8490152/) | Local (per sequence × output) | Score | None | None. Relative to a reference (zeros, or 20/100 dinucleotide shuffles); about f̂ [inferred] |
| Seqlets, TF-MoDISco / tfmodisco-lite — [note](https://arxiv.org/abs/1811.00416), [code](https://github.com/jmschrei/tfmodisco-lite/blob/main/modiscolite/extract_seqlets.py) | Threshold pooled over all sequences; extraction per sequence | Set (21 bp core + 10 bp flanks) | Isotonic "FDR" against a Laplace fit to observed scores (lite) or shuffled-sequence attributions (v0.5 option). Target 0.2 (BPNet 0.01). Overridden by 3–20% passing caps and an `-n` cap | Dataset-level estimate, uncalibrated. Null = noise in f̂'s attributions [stated for Laplace; inferred for shuffle] |
| Seqlets, tangermeme `recursive_seqlets` — [code](https://github.com/jmschrei/tangermeme/blob/main/tangermeme/seqlet.py) | Null pooled; calls per sequence | Set (4–25 bp) | p < 0.01 under an independent-position convolution null built from the same data; or FDR from dinucleotide-shuffle controls (target 0.1) | Default: "not error rates" [stated]. With controls: pooled FDR estimate; null = "no more frequent than in sequences with the same dinucleotide composition" [stated]; about f̂ |
| Pattern discovery (TF-MoDISco clustering) — [code](https://github.com/jmschrei/tfmodisco-lite/blob/main/modiscolite/tfmodisco.py) | Global | Set of motifs (CWM/PFM) | Leiden (50 runs), similarity thresholds, ≥ 20 seqlets, information content ≥ 0.6 over 6 bp | None |
| Hits, BPNet CWM scan — [BPNet](https://pmc.ncbi.nlm.nih.gov/articles/PMC8812996/) | Per region; cutoffs from pooled seqlets | Set | Match > 20th percentile of seqlets; contrib > seqlet minimum; PWM log-odds > 0 | None |
| Hits, Fi-NeMo — [README](https://github.com/kundajelab/Fi-NeMo), [code](https://github.com/kundajelab/Fi-NeMo/blob/main/src/finemo/hitcaller.py) | Per region (RMS-normalized); one global λ | Set (non-negative lasso support) | λ = 0.7 (0.6 for ChIP); trim 0.3; user-chosen coefficient/importance cutoffs | None; non-monotone in λ [stated] |
| PWM scan / Tomtom annotation — [ChromBPNet](https://pmc.ncbi.nlm.nih.gov/articles/PMC11741299/), [tangermeme](https://github.com/jmschrei/tangermeme/blob/main/tangermeme/annotate.py) | Per region or per seqlet | Set with p-values | p ≤ 0.05 or 0.001 (MOODS); Tomtom threshold | p-values about sequence or motif similarity, not about f̂'s dependence [inferred] |
| GIA / injection / marginal footprints — [GIA](https://pmc.ncbi.nlm.nih.gov/articles/PMC8118286/) | Global | Effect size | None | None; "through the lens of the DNN" [stated] |
| CREME — [preprint](https://www.biorxiv.org/content/10.1101/2023.07.03.547592) | Local (one locus) | Effect size, then category | 0.3 / 0.5 / 0.95 / 0.2 cutoffs; 10 shuffles averaged | None; about f̂ |
| SQUID — [SQUID](https://doi.org/10.1038/s42256-024-00851-5) | Local | Score (surrogate parameters) | None | None |
| ChromBPNet variant IPS — [ChromBPNet](https://pmc.ncbi.nlm.nih.gov/articles/PMC11741299/) | Local (one variant) | Score + empirical p-value | Not stated | Pooled null of 10⁶ shuffled contexts; about f̂ |
| Wet-lab validation (CRISPR, STARR-seq/MPRA, QTL) — [BPNet](https://pmc.ncbi.nlm.nih.gov/articles/PMC8812996/), [DeepSTARR](https://doi.org/10.1038/s41588-022-01048-5) | Selected instances | Measured effect | Not applicable | About the true function, but confirmatory: no calibration of the selection |

- Correlation is acknowledged in the code: "Attributions are usually correlated along the sequence". — [tangermeme](https://github.com/jmschrei/tangermeme/blob/main/tangermeme/seqlet.py)
- Redundancy and competition among regulatory elements show up in model perturbations. — [CREME](https://www.biorxiv.org/content/10.1101/2023.07.03.547592)

### Inferences
**What is already done**
- Per-sequence attribution exists, so a local version of a global *attribution* method is not a contribution.
- Per-sequence *selection* exists as well (seqlets, hits, CREME categories), so "producing a set" is also not new.

**What remains: the precise gap**
- **(a) Error control at the level of the single sequence.** For a given x, a guarantee such as E[V(x)/max(R(x),1)] ≤ q, where V(x) is the number of false selections and R(x) the number of selections, or P(V(x) ≥ 1) ≤ α. Today's FDRs are averaged over the dataset, and today's hit callers have no error statement at all.
- **(b) A stated local null.** Today's nulls are either "f̂'s attributions on shuffled DNA" or "a noise curve fitted to the same scores". Neither conditions on the rest of x. A natural local null about f̂ is H₀,g(x): replacing group g with draws from a reference distribution, given the flanks x₋g, leaves the prediction (or the per-sample loss, under the thesis's "increase in loss" convention) unchanged in expectation. This is a CRT/SHAP-XRT-style local null; Burns et al. and SHAP-XRT exist in general ML but have not been brought to genomics.
- **(c) Groups as the unit of selection.** A motif spans about 6–20 bp, and attributions are spatially correlated, so the right unit is the motif instance or window (like an image patch), not a single base. The structure is hierarchical: bases ⊂ instances ⊂ CREs (5 kb tiles), which suggests closed or hierarchical testing.
- **(d) Redundancy and non-monotonicity.** CREME documents redundancy and competition. A necessity-only test misses redundant instances, and sufficiency is not monotone. This is the thesis's open problem 1, appearing naturally in genomics data.
- **(e) Being explicit about which function the claim concerns.** Every existing guarantee-like quantity concerns **f̂**. A claim about the true function would have to absorb refit variability: Majdandzic et al. show attributions change across models of similar accuracy, and the fold/replicate ensembles already exist but are only averaged. Synthetic designs with known local support, and DeepSTARR-style mutagenesis data, are possible ground truth for checking calibration.

**Where the thesis fits**
- A defensible contribution is the **calibration layer between attribution and hit calling**: take candidate groups (Fi-NeMo hits, seqlets, or windows) and attach per-sequence p-values or e-values with FDR control under a stated local null. The ingredients are already in the field's toolbox: dinucleotide-shuffle reference draws, CREME-style perturbation replicates, and fold ensembles. What has not been done is turning them into an error-controlled selection rule.

### Gaps
- I did not search exhaustively for 2025–2026 preprints that might already propose per-sequence FDR for genomic attributions. The searches I did run (knockoffs and DNA-sequence deep learning; local hypothesis tests for attributions) turned up only global knockoff methods (KIMI, DeepLINK) and general-ML local tests. A targeted bioRxiv search for terms like "conformal attribution genomics" or "selective inference motif" would close this.
- No source quantifies how often called instances fail experimental validation, so the size of the false-positive problem in real hit sets is unmeasured.
