# Feasibility: per-sequence feature selection with error control for DNA sequence models

Scope: test each 20-bp window of a held-out sequence by replacing it with background-model draws, compute permutation p-values from a trained CNN, apply BH within the sequence, and validate on synthetic sequences with planted motifs. Budget: 15 tool calls, so coverage is shallow. Full texts were not read except a keyword extraction from Burns et al.

## Closest precedents (instance-wise tests on images, text, sequences; any on genomic models 2020–2026?)

### Takeaway
The general method exists and has been shown on images and text (Burns et al.; SHAP-XRT; x-SKIT). I found no paper that applies an instance-wise CRT/IRT with BH to a genomic sequence model. The genomics side has close ingredients, such as motif embedding into background sequences (GIA) and knockoff FDR for k-mers (KIMI), but they are population-level, not per-sequence.

### Cited Findings
- Burns, Thomason & Tansey frame interpretation as multiple hypothesis testing: a feature is important if the prediction differs significantly from what it would be with the feature "replaced with uninformative counterfactuals". They give two methods. The IRT "provably controls the false discovery rate but... is not yet feasible for large-scale applications", and the OSFT is a one-sample approximation. They control FDR via Benjamini–Hochberg and point to Benjamini–Yekutieli under arbitrary dependence. Case studies are an ImageNet classifier and a movie-review text classifier, with counterfactuals from deep generative models (inpainting). There is no biology case study. Published at FODS 2020. — [arXiv 1904.00045](https://arxiv.org/pdf/1904.00045); [MSK record](https://synapse.mskcc.org/synapse/works/174747)
- SHAP-XRT (Teneggi, Bharti, Romano, Sulam, TMLR 2023) defines a CRT for conditional independence *at the level of individual samples*. For binary classification it shows that Shapley marginal contributions bound the expected p-values of these local tests. — [arXiv 2207.07038](https://arxiv.org/abs/2207.07038v4); [TMLR](https://mlanthology.org/tmlr/2023/teneggi2023tmlr-shapxrt)
- Teneggi & Sulam (NeurIPS 2024) define a *local conditional* semantic-independence null for a fixed input. Their x-SKIT tests it by sequential kernelized betting, shown on synthetic data and vision-language image classifiers. A secondary summary says it includes an FDR strategy; I did not verify this. — [arXiv 2405.19146](https://arxiv.org/pdf/2405.19146); [NeurIPS](https://mlanthology.org/neurips/2024/teneggi2024neurips-testing)
- An additional-mutual-information (AMI) paper reportedly extends CRT-style estimators to find important features "in a particular instance". The snippet did not mention FDR. — [PapersWithCode](https://cs.paperswithcode.com/paper/model-agnostic-feature-selection-with)
- Global Importance Analysis (Koo et al., PLoS Comp Bio 2021) measures a putative pattern's effect on a genomic DNN's predictions *across a population* of designed synthetic sequences. It found that ResidualBind uses motif number, spacing, and context such as GC bias and RNA structure. — [CSHL repository](https://repository.cshl.edu/id/eprint/41349/); [bioRxiv](https://www.biorxiv.org/content/10.1101/2020.09.08.288068v1.full.pdf)
- KIMI applies model-X knockoffs to motif/k-mer discovery with FDR control (Bioinformatics 2020). It works at population level over sequence groups and uses no CNN. — [MIDAS listing](https://midasnetwork.us/papers/kimi-knockoff-inference-for-motif-identification-from-molecular-sequences-with-controlled-false-discovery-rate/)
- BaGGLS (2025) applies Bayesian shrinkage to CNN attribution scores, benchmarked on synthetic sequences with planted motifs. It is a statistical post hoc layer but not an FDR procedure. — [arXiv 2511.15330](https://arxiv.org/pdf/2511.15330)
- SQUID (Seitz, McCandlish, Kinney, Koo, 2024) fits interpretable surrogate models to genomic DNNs over mutagenesis libraries in a user-specified region of sequence space. It notes that ISM assumes locally additive nucleotide effects. — [SQUID preprint](https://repository.cshl.edu/41358/1/2023_Seitz_Interpreting_cis_regulatory_mechanisms_preprint.pdf)

### Inferences
- The specific combination (local IRT + background-model counterfactuals + within-sequence BH + genomic CNN) appears unoccupied, but only as a straightforward *application*. It is not a new method. Novelty would come from the DNA-specific counterfactual and from a careful empirical study, not from the test itself.
- The DNA setting is easier than images in one important way. A background model (i.i.d., Markov-k, dinucleotide-shuffle) gives an explicit, cheap counterfactual sampler, where Burns needed an inpainting network. That makes the exact IRT computationally feasible here.

### Gaps
- Could not pull Google Scholar citation lists for 1904.00045 or 2207.07038. A genomic follow-up may exist in the citations and would only show up there.
- Did not check protein language model work for instance-wise tests. One search returned nothing relevant.

## Strongest objections

### Takeaway
Every listed objection has a source behind it. The sharpest are: (a) the null is about the model, and with deterministic replacement it is false almost by construction; (b) background draws are off-manifold relative to the trained distribution; (c) a CNN is non-additive, so window-level nulls interact. None of these makes the project uninformative, but each limits what can be claimed.

### Cited Findings
- **Null trivially false / about the model.** Burns et al. argue that plain conditional-independence nulls are unsuitable for interpreting a fixed model, because "the null hypothesis will always be false because any change to X_j will numerically alter Ŷ". They replace them with a null that the prediction is not *surprising* relative to the counterfactual distribution. — [arXiv 1904.00045](https://arxiv.org/pdf/1904.00045)
- **Out-of-distribution counterfactuals.** Burns et al. require that counterfactuals lie in the support of P(X), because model "behavior on out-of-distribution inputs can be counterintuitive". — [arXiv 1904.00045](https://arxiv.org/pdf/1904.00045). Hooker, Mentch & Zhou show that permute-and-predict breaks feature dependence and "forces extrapolation", which yields misleading importance. Conditional permutation avoids this but can underestimate importance. — [arXiv 1905.03151](https://arxiv.org/pdf/1905.03151); [ALE paper](https://arxiv.org/pdf/2512.21124)
- **Genomic attributions are fragile.** Attribution maps carry spurious importance scores that vary from model to model even when generalization is good. A noise source arises from how DNNs treat one-hot DNA off the simplex. — [Majdandzic et al.](https://repository.cshl.edu/id/eprint/40956/1/Selecting%20deep%20neural%20networks%20that%20yield%20consistent%20attribution-based%20interpretations%20for%20genomics.pdf); [Genome Biology 2023](https://GenomeBiology.biomedcentral.com/counter/pdf/10.1186/s13059-023-02956-3.pdf). Attributions are also unstable across random initializations. — [Tseng et al., bioRxiv](https://www.biorxiv.org/content/10.1101/2020.06.11.147272v1.full.pdf)
- **Background models can be too permissive.** In motif discovery, background-model nulls inflate apparent significance, and held-out data is a stronger check. — [PMC3492406](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3492406/)
- **Dependence between tests.** Burns et al. use BH but name Benjamini–Yekutieli for arbitrary dependence. They note that OSFT draws are valid knockoffs only under conditions, with a looser FDR bound otherwise. — [arXiv 1904.00045](https://arxiv.org/pdf/1904.00045)

### Inferences
- **Model-vs-biology.** The guarantee is "the CNN's output at this sequence depends on window w beyond what background draws produce". It is not "window w contains a functional site". This is acceptable if stated. On synthetic data with known planted motifs the two can be compared, which is the main scientific content of the validation.
- **Trivial nulls in simulation.** If the CNN learned the planted motif, motif windows reject and pure-background windows are (approximately) exchangeable with their replacements, so the null really holds there. The risk is circular: the generator, the background sampler, and the CNN's training distribution are all the same model, so FDR control is nearly guaranteed. A non-trivial test needs a mismatch, for example training on Markov-2 background while testing replacements from i.i.d. or shuffled backgrounds, or real genomic flanks.
- **Power with few windows.** A 200-bp sequence gives about 10 non-overlapping windows. BH with m≈10 and a permutation p-value floor of 1/(B+1) needs B ≳ 10/α ≈ 200 draws per window to reject at all. That is cheap for a CNN. Overlapping windows create strong positive dependence, which motivates BY or a cruder Bonferroni comparison.
- **Partial motif overlap.** Motifs straddling window boundaries split their signal across windows. Combined with CNN non-additivity (SQUID's point about ISM's additivity assumption), this means window nulls are not separable. This mirrors the non-monotonicity problem in the thesis's local value function.
- No source I found argues that instance-wise tests are uninformative in general. Burns' critique targets conditional-independence nulls, not their own surprise-based null.

### Gaps
- I found no paper that empirically tests whether instance-wise FDR claims transfer from synthetic to real genomic models.

## Comparable thesis or workshop-scope work

### Takeaway
I found no student thesis or MLCB/ICML comp-bio workshop paper doing this exact project. The nearest workshop-scale analogue is attribution-evaluation work on genomic models.

### Cited Findings
- Ozbulak et al. (NeurIPS 2022 workshop) evaluate the fidelity of attribution methods on genomic models using point mutations. This is a workshop-sized empirical study of genomic interpretability. — [mlanthology](https://mlanthology.org/neuripsw/2022/ozbulak2022neuripsw-utilizing); [arXiv 2212.06151](https://arxiv.org/pdf/2212.06151)
- WTKO-CNN (2026 preprint) is a small-scale pipeline of CNN, saliency, k-mer clustering, and validation against MEME/TOMTOM/HOMER, with no FDR step. — [arXiv 2605.24034](https://arxiv.org/pdf/2605.24034)
- Searches for MLCB papers combining CRTs with sequence models returned nothing. — [search returned CRT methods only, e.g. dCRT](https://academicnewsletter.sufe.edu.cn/info/448186)

### Inferences
- A workshop paper of this scope would plausibly look like this: one synthetic generator with planted motifs; one or two CNNs; empirical FDR and power curves versus α; a baseline of ISM or saliency thresholded heuristically; and one real-data illustration (e.g. a held-out sequence from a public TF-binding or STARR-seq model) labeled qualitative.

### Gaps
- Could not search MLCB proceedings or the PMLR accepted-paper lists directly.

## Verdict: what is realistic vs. overreaching

### Takeaway
Realistic, provided it is framed as a careful empirical study of an existing test (local IRT) in a domain where the counterfactual sampler is explicit. It is not a dead end. It overreaches if it claims biological discovery control, new theory, or a real-genome guarantee.

### Inferences (no direct sources; based on the findings above)
- **Realistic (few months):**
  1. Implement exact local IRT with background-model replacement and BH/BY within a sequence, reusing the thesis's `lfs` p-value machinery.
  2. Synthetic validation with planted motifs, reporting empirical per-sequence FDR and power against α, window size, and motif strength.
  3. A deliberate *mismatch* axis (background used for replacement ≠ background used for generation/training) to show when FDR breaks. This answers the OOD objection empirically and keeps the null from being trivially true.
  4. A comparison against thresholded ISM/saliency, which links directly to the thesis's open threshold-calibration problem.
  5. At most a qualitative real-sequence demo.
- **Overreaching:**
  - Claiming FDR control over *biologically functional* windows on real genomes, since the guarantee is about the model.
  - Learned/generative counterfactuals (diffusion or language-model infilling) with validity proofs.
  - Theory for dependent overlapping windows beyond citing BY.
  - Variable-length or motif-aligned (non-grid) hypotheses.
  - Claims of novelty for the test itself (Burns 2020, SHAP-XRT 2023, and x-SKIT 2024 already define local nulls).
- **Main risk:** the result reads as "BH works when the null is true by construction". The mismatch experiment and the model-vs-ground-truth comparison are what make it a finding.
