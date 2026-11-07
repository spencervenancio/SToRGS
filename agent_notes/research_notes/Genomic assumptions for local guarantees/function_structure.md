# Structural assumptions on the sequence-to-activity function f, and what guarantees they buy

Scope: grammar and sparsity models of f, discreteness and exact enumeration, statistical inference from MPRA and perturbation data, and recovery theory. TF-MoDISco, tangermeme, Fi-NeMo, CREME, GIA and gradient correction are covered elsewhere and left out here. Budget: 13 tool calls, mostly search snippets plus two arXiv fetches. Anything not fetched in full is marked as coming from a snippet.

## Q1. Grammar and sparsity models (biophysical, billboard/enhanceosome, additive/pairwise, ExplaiNN, SQUID): do any give CIs or tests for per-sequence motif effects?

### Takeaway
Genomics offers several restricted forms of f: additive or pairwise models, global epistasis, a sum of single-motif units, and thermodynamic models. None of the work found turns these into per-sequence confidence intervals or tests for which motif instances matter. The one source of uncertainty quantification found is Gaussian-process posteriors on sequence-function maps. Thermodynamic models are documented to be *non-identifiable* in practice.

### Cited Findings
- SQUID (Seitz, McCandlish, Kinney, Koo; Nature Machine Intelligence 2024) works in three steps. It generates an in-silico MAVE library of variants around one sequence, scores them with the DNN, fits an additive or pairwise surrogate with MAVE-NN, and reads the surrogate's parameters as attributions. — [bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2023.11.14.567120.full.pdf); [NMI version, CSHL repository](https://repository.cshl.edu/41591/1/10.1038.s42256-024-00851-5.pdf)
- SQUID's stated motivation is that standard attribution assumes local additivity, so it misses epistasis and global nonlinearity. Pairwise surrogates were more accurate than additive ones for pairs of AP-1 sites, and SQUID "quantif[ied] epistatic interactions that were otherwise obscured by global nonlinearities." — same sources (via search snippet)
- MAVE-NN factors f into a deterministic sequence-to-latent-phenotype map, then a nonlinear "global epistasis" link, then an explicit noise model. This is meant to "deconvolve mutational effects from … confounding experimental nonlinearities and noise." — [MAVE-NN bioRxiv](https://www.biorxiv.org/content/10.1101/2020.07.14.201475.full.pdf); [Genome Biology 2022 figure page](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-022-02661-7/figures/2)
- ExplaiNN (Novakovsky et al., Genome Biology 2023) builds f as a linear combination of independent CNN units, each with *one* convolutional filter. It is inspired by neural additive models and claims both global and "local (individual sequence level)" interpretation, with performance comparable to state-of-the-art models. — [PMC10303849](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10303849/)
- Thermodynamic sequence-to-expression models "have shown parameter non-identifiability": many distinct parameter sets fit the same data about equally well. — [PMC8686167](https://pmc.ncbi.nlm.nih.gov/articles/PMC8686167)
- Fitting thermodynamic models to a few closely related enhancers gives an *ensemble* of models that predict different effects for the same mutation. — [bioRxiv 2020.02.07.939264](https://www.biorxiv.org/content/10.1101/2020.02.07.939264.full.pdf). Samee et al. (Cell Systems 2015) search the parameter space systematically, keep every model consistent with wild-type data, prune them with perturbation data, and validate predictions by site mutagenesis in embryos. — [PMC5094195](https://pmc.ncbi.nlm.nih.gov/articles/PMC5094195)
- Enhanceosome grammar means a precise arrangement of sites is required. Billboard grammar means order, spacing and orientation can change with little effect. Melnikov et al. (2012) tested more than 27,000 variants of a billboard-like synthetic cAMP enhancer and the enhanceosome-like IFN-β enhancer, and fit quantitative sequence-activity models. — [Nature Biotechnology](https://www.nature.com/articles/nbt.2137); [PMC3297981](https://pmc.ncbi.nlm.nih.gov/articles/PMC3297981)
- Zhou, Martí-Gómez, Petti and McCandlish (bioRxiv 2025, eLife reviewed preprint) give Gaussian-process priors on sequence-function maps with site-, allele- and mutation-specific epistasis parameters. GP regression "gives a full description of prediction uncertainty through the posterior covariance," which the authors present as an advantage over neural networks that "lack uncertainty quantification." — [bioRxiv 2025.08.15.670613](https://www.biorxiv.org/content/10.1101/2025.08.15.670613.full.pdf); [eLife RP 108964](https://elifesciences.org/reviewed-preprints/108964v1)
- McCandlish's earlier PNAS 2022 work estimates the variance contributed by each order of interaction from how phenotypic correlations decay with mutational distance, then runs empirical-Bayes GP regression under that prior. — [CSHL repository](https://repository.cshl.edu/id/eprint/40723/1/2022-McCandlish-Higher-order-epistasis-and-phenotypic-prediction.pdf)

### Inferences
- SQUID's design is the closest existing analogue to "local selection" in the thesis sense. It fits a restricted f around one sequence x by sampling a neighbourhood. But it uses the surrogate's fitted parameters as point estimates, with no reported standard errors, tests or selection rule.
- A low-order surrogate fitted to DNN queries is an ordinary regression on simulated data. Coefficient CIs from it would be cheap, but they would measure uncertainty from the *sampling of the surrogate fit* (which shrinks with more queries), not error in the claim about f.
- The ExplaiNN structure f(x) = Σ_k g_k(conv_k(x)) makes "which unit matters at x" well defined, because the units are additive. Under that structure the local value function is additive across units, so the thesis's non-monotonicity problem would vanish *between* units. It does not vanish within a unit if g_k is non-monotone.
- Non-identifiability of thermodynamic parameters does not by itself imply that the per-sequence *support* is non-identifiable. Different parameter sets could agree on which sites matter. No source found checks this.

### Gaps
- I found no paper reporting per-sequence CIs or tests for motif effects from SQUID, MAVE-NN, ExplaiNN or thermodynamic models.
- I did not verify whether the GP paper reports posterior intervals for individual mutational effects, as opposed to predictions. The search snippet covers only prediction uncertainty.

## Q2. Does discreteness help? Exact enumeration, exhaustive ISM, exact Shapley over motif instances

### Takeaway
Discreteness has been used for *exact* Shapley computation through sparse q-ary Fourier and Möbius transforms (SHAP zero, NeurIPS 2025). Exactness there depends on assuming f is Fourier-sparse with low interaction order and on uniform marginalization over nucleotides. There are no CIs or tests. I found no work that enumerates subsets of a handful of motif instances exactly *and* attaches inference.

### Cited Findings
- SHAP zero (Tsui et al., NeurIPS 2025) assumes "sequence models tend to have a sparse Fourier transform." Interactions are capped at order ℓ, "typically a small constant," with ℓ ≤ 5 cited. Absent positions are marginalized *uniformly over the q nucleotides*, not over the data distribution. — [arXiv 2410.19236v4 (HTML)](https://arxiv.org/html/2410.19236v4)
- The map from Fourier to q-ary Möbius to Shapley values and interactions is stated as exact equalities (Props 3.2–3.3, Thms 3.4–3.5). The authors also say SHAP zero and KernelSHAP "approximate slightly different quantities due to their marginalization assumptions," and that the recovered coefficient count varied, "likely due to the model not being exactly sparse." — same source
- Sample complexity is O(s·n²) and runtime O(s·n³) for the top-s Fourier coefficients. Recovery guarantees are deferred to earlier q-ary sparse Fourier transform work. No end-to-end error bound and no CIs or tests are given; results are reported as correlations and runtimes. — same source
- Exact Shapley interactions need exponentially many model calls in the input dimension. Other interaction-SHAP methods (e.g. SHAP-IQ) rely on subsampling. — [arXiv 2410.19236](https://export.arxiv.org/pdf/2410.19236)
- Ding et al. give an exact transformation from a CNN first-layer kernel to a PWM: the PWM log-likelihood equals a constant plus the kernel convolution. — [bioRxiv 163220](https://www.biorxiv.org/content/10.1101/163220.full.pdf)

### Inferences
- With the thesis's own design, exact enumeration is affordable when the "players" are m candidate motif instances or windows rather than single nucleotides: 2^m model calls, so m ≈ 10–15 is feasible. Exact enumeration removes Monte-Carlo error in φ, but not uncertainty about f or about the replacement distribution used for "absent" windows. The guarantee then hinges on which distribution is chosen (uniform, shuffled, or background), which is the same marginalization caveat SHAP zero states.
- Fourier sparsity with order ℓ ≤ 5 is a usable structural assumption for f. In principle it would support a theorem of the form "if f is s-sparse of degree ℓ, the local support is recovered with probability ≥ 1−δ." I found no paper that states this for per-sequence *support*.

### Gaps
- I did not read the last third of the SHAP zero paper (beyond character 100k), so any noisy-recovery results in the appendix are unchecked.
- I found no published exhaustive-ISM or exact motif-subset Shapley study for regulatory DNA that includes tests.

## Q3. Statistical inference on motif effects from MPRA or perturbation data: CIs and FDR on per-element calls

### Takeaway
Calibrated inference exists at two levels. Element-level differential activity (mpralm) controls type I error, and motif-*occurrence* scanning (FIMO) gives q-values. Data-driven knockoffs for k-mer motifs (KIMI) give finite-sample FDR. None of these targets "does the model's or the biology's output at this specific sequence depend on this instance."

### Cited Findings
- mpralm fits linear models (not count models) for differential MPRA activity and reports good type I error control and power. — [PMC6417258](https://pmc.ncbi.nlm.nih.gov/articles/PMC6417258); [bioRxiv 196394](https://www.biorxiv.org/content/10.1101/196394v2.full)
- MPRA characterization studies usually regress activity on several sequence features at once. — [PMC6417258](https://pmc.ncbi.nlm.nih.gov/articles/PMC6417258) (search snippet)
- Kheradpour et al. (Genome Research 2013) tested about 2,000 predicted enhancers with motif disruptions. Disrupting activator motifs abolished function, and the strength of the motif match, conservation, and nucleosome exclusion predicted activity. — [PMC3638136](https://pmc.ncbi.nlm.nih.gov/articles/PMC3638136)
- FIMO computes a log-likelihood ratio score and an exact DP p-value for each position, and converts p-values to q-values. Current docs say this uses Benjamini–Hochberg; the 2011 paper describes Storey's bootstrap. So the estimator depends on the version. — [MEME-suite docs](https://meme-suite.org/meme/doc/fimo.html); [PMC3065696](https://pmc.ncbi.nlm.nih.gov/articles/PMC3065696)
- KIMI applies model-X knockoffs to k-mer motif discovery, with FDR control at any target level and more power than BH or q-values in simulation. — [MIDAS listing](https://midasnetwork.us/papers/kimi-knockoff-inference-for-motif-identification-from-molecular-sequences-with-controlled-false-discovery-rate/)
- DeepPINK-style knockoff and DNN methods (DeepLINK, DeepPIG, DiffKnock) control FDR for *global* feature selection. I found none applied to per-sequence DNA motif instances. — [DeepLINK, PMC8433583](https://pmc.ncbi.nlm.nih.gov/articles/PMC8433583); [DeepPIG, PMC11227546](https://pmc.ncbi.nlm.nih.gov/articles/PMC11227546); [DiffKnock, arXiv 2510.01418](https://www.arxiv.org/abs/2510.01418)

### Inferences
- FIMO's FDR is about *motif presence*, which is a different hypothesis from *functional relevance at x*. This is a useful contrast for the thesis: the null "this window looks like a PWM match by chance" is calibrated, while "the prediction does not depend on this window" is not.
- The knockoff results are global and dataset-level. A per-sequence analogue would need knockoffs of windows conditional on the rest of x, for example dinucleotide-shuffled windows. That links to the thesis's replacement-distribution choice.

### Gaps
- I did not confirm whether any MPRA paper reports per-motif regression CIs or FDR across motif-disruption elements; snippets mention regression but not interval reporting.
- KIMI's details (venue, k-mer model, whether it is per-sequence) were not fetched.

## Q4. Has anyone proven that, under a motif-grammar assumption, a selection procedure recovers the true per-sequence support?

### Takeaway
I found no such result. The nearest theory is (a) sample-complexity guarantees for sparse q-ary Fourier recovery, which SHAP zero inherits, (b) a counting-argument "motif identifiability theorem" for sparse latent models (SPARLING), not tied to regulatory DNA, and (c) empirical identifiability studies.

### Cited Findings
- SPARLING (arXiv 2302.01976) proves a "Motif Identifiability Theorem" by a counting argument for extremely sparse latent activations. — [arXiv 2302.01976](https://arxiv.org/pdf/2302.01976) (search snippet)
- A University of Birmingham PhD thesis studies whether interpreting a trained network recovers "true" motifs when only predictive performance is observed. It is empirical, not a proof. — [etheses.bham.ac.uk/12755](https://etheses.bham.ac.uk/id/eprint/12755)
- SHAP zero refers to earlier q-ary sparse Fourier transform work for its theory and gives no end-to-end per-sequence error bound. — [arXiv 2410.19236v4](https://arxiv.org/html/2410.19236v4)

### Inferences
- This is a real gap the thesis can occupy. A natural model is f(x) = Σ_{instances i in x} β_{motif(i)} + Σ_{pairs within distance d} γ, a billboard model with bounded-range pairwise terms. Under it, the local support S(x) is the set of instances with nonzero contribution at x. Bounded order plus sparsity gives the structure needed for a recovery or FDR statement on S(x).
- The billboard and enhanceosome distinction maps onto the thesis's open problems. Billboard is roughly additive, so the local V is roughly monotone and the MinShap guarantee is plausible. Enhanceosome is strongly interacting, so non-monotonicity is expected and support is only defined jointly.

### Gaps
- My search covered only "provable recovery / identifiability" phrasings. Work on sparse-additive or hierarchical-interaction selection (e.g., group lasso or hierarchical lasso on motif features) applied to MPRA, which might have consistency results, was not searched.
