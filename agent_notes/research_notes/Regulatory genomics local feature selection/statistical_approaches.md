# Statistical approaches to error-controlled local (per-instance) feature selection for black-box models, and their adaptation to genomic sequence models

*Notation used throughout:*
- f̂ is the trained model, fixed after training. f = E[Y | X] is the population regression function.
- x is one input sequence. W is a window, meaning a contiguous set of positions.
- Q(· | x_{−W}) is a user-chosen reference (conditional) sampler.
- "Local" means a claim about one input x.

*Coverage:* literature current to 25 Oct 2026. Dates are first-posting or venue dates.

*Format:* every item in "Cited Findings" carries a source. "Inferences" are my reasoning from those findings. Unverified recollections are listed under "Gaps".

---

## 1. General local testing frameworks: which methods give per-instance selection or importance tests with guarantees, and what is known about inference on ∂f/∂x_j at a point?

### Takeaway
Per-instance tests with finite-sample validity exist, but almost all of them test a null about the fixed fitted model relative to a user-chosen reference sampler. The null is that "f̂(x) is a typical value of f̂(x̃) when x_S is redrawn from Q(·|x_{−S})" (Burns et al. 2020; SHAP-XRT 2023; Teneggi & Sulam 2024).

Guarantees about the data (about f or Y) come in only three forms:
- **Global:** CRT, model-X knockoffs, HRT, SPVIM.
- **Regional:** Gimenez & Zou 2019.
- **Marginal over test points:** local LOCO 2018; Watson et al. 2023.

Pointwise inference on ∂f/∂x_j exists for classical smoothers: local polynomials, B-spline and GP priors, KRR, and debiased local functionals. These all run at nonparametric rates. DNN results are early, and nothing covers trained genomic CNNs on one-hot inputs.

### Cited Findings

#### Instance-wise conditional randomization tests: Burns, Thomason & Tansey (arXiv Mar 2019; FODS 2020)
- **Framing.** Interpretation is treated as multiple hypothesis testing. A feature is important if the prediction differs significantly from what would be expected if the feature were replaced by an uninformative counterfactual. The paper gives one provably FDR-controlling procedure that is "not yet practical at large scale" and one approximate test — [arXiv 1904.00045](https://arxiv.org/pdf/1904.00045). The earlier title was "Interpreting Black Box Models with Statistical Guarantees" — [DeepAI listing](https://deepai.org/publication/interpreting-black-box-models-with-statistical-guarantees)
- **What the null is about.** The null concerns the fitted model's output, not Y. Definition 0 states the general form: "the null hypothesis, H0, is that T_P(f(X)) is stochastically less than T_{Q|x−S}(f(X))". The testable *point null* is f(x) ~ f(X̃) with X̃_S ~ Q(X_S | X_{−S} = x_{−S}). The authors use it "as a proxy for the familywise null" — [ar5iv full text](https://ar5iv.arxiv.org/html/1904.00045)
- **Why not test against Ŷ.** The authors reject conditional-independence testing against Ŷ because for dense models "the null hypothesis will always be false" — [ar5iv](https://ar5iv.arxiv.org/html/1904.00045)
- **IRT (the provably valid procedure).**
  - Takes one input x and a set of subsets S_1…S_N.
  - Draws K conditional samples per subset to form randomization p-values.
  - Applies Benjamini–Hochberg across those N tests. FDR is controlled under independence or positive dependence; Benjamini–Yekutieli covers arbitrary dependence.
  - It is impractical because "Running these models thousands of times per feature is intractable" — [ar5iv](https://ar5iv.arxiv.org/html/1904.00045)
- **OSFT (the approximate procedure).**
  - Uses one conditional draw per subset and the knockoff selection rule on z(i) = t − t̃(i).
  - FDR control (Theorem 2) requires independent difference statistics. The draws "are valid knockoffs only when all features are independent."
  - For M independent samples with at most N subsets each, the bound relaxes to Nα. The experiments pool selections across many images or reviews — [ar5iv](https://ar5iv.arxiv.org/html/1904.00045)
- **Samplers.**
  - Images: a pretrained generative inpainting model fills masked patches for Inception v3.
  - Text: a BERT model fine-tuned for infilling. A masked LM needs no separate conditional model.
  - Q is user-supplied and need not equal the true P(X_j | X_{−j}). It should keep counterfactuals in-support, because "model behavior on out-of-distribution inputs can be counterintuitive" — [ar5iv](https://ar5iv.arxiv.org/html/1904.00045)
- **Groups.** Tested units are arbitrary subsets S. Image patches must be "non-overlapping". Word-level tests "can miss features that involve composition", and spans are left to future work — [ar5iv](https://ar5iv.arxiv.org/html/1904.00045)
- **Empirical weak spot.** "The one notable exception was when interpreting the discontinuous model with correlated inputs". There, OSFT's two-sided true-positive rate was very low — [ar5iv](https://ar5iv.arxiv.org/html/1904.00045)
- **Cost.** The methods are "less efficient than most popular interpretability methods", which need about one forward and backward pass. The authors argue "a higher computational budget is the cost of FDR control" — [ar5iv](https://ar5iv.arxiv.org/html/1904.00045)

#### SHAP-XRT: Teneggi, Bharti, Romano & Sulam (arXiv Jul 2022; TMLR 2023)
- **Local null.** For sample x, feature j, and coalition C not containing j: H0_{j,C}: f(X̃_{C∪{j}}) =d f(X̃_C). Here X̃_C equals x on C and draws the complement from X_{−C} | X_C = x_C — [arXiv 2207.07038v5](https://arxiv.org/html/2207.07038v5). Venue — [TMLR 2023 listing](https://mlanthology.org/tmlr/2023/teneggi2023tmlr-shapxrt)
- **What the null is about.** It "evaluates conditional independence of the response of a deterministic model f on an individual sample". Randomness comes only from the reference draws — [arXiv 2207.07038v5](https://arxiv.org/html/2207.07038v5)
- **p-value.** p̂ = (1 + Σ_k 1[t̃^(k) ≥ t])/(K+1). It is valid for any statistic T under the null — [arXiv 2207.07038v5](https://arxiv.org/html/2207.07038v5)
- **Theorem 2 (binary Y, f ∈ [0,1]).** Let γ be the expected marginal contribution. Then E[p] ≤ 1 − (K/(K+1))γ. If γ < 0, then (1+Kγ²)/(K+1) ≤ E[p] ≤ 1 — [arXiv 2207.07038v5](https://arxiv.org/html/2207.07038v5)
- **Composite null.** H0_j = ∩_C H0_{j,C}. The p-value p̂_j = 2 Σ_C w_C p̂_{j,C}, with Shapley weights and Vovk–Wang averaging, is valid. Its expectation satisfies E[p̂_j] ≤ 2(1 − (K/(K+1))ϕ_j) — [arXiv 2207.07038v5](https://arxiv.org/html/2207.07038v5)
- **Limitations.**
  - The test is one-sided; use 1 − f for the other direction.
  - Large negative contributions give no evidence against the null.
  - There is no FDR procedure. The paper notes that comparing every p-value to α inflates type I error and mentions Bonferroni-type correction.
  - Exact Shapley values need exponentially many subsets.
  - Experiments use image quadrants via h-Shap — [arXiv 2207.07038v5](https://arxiv.org/html/2207.07038v5)

#### Testing semantic importance by betting: Teneggi & Sulam (arXiv May 2024; NeurIPS 2024)
- **Nulls.** Ŷ is the fixed model's prediction and Z_j is a concept score.
  - Global: Ŷ ⟂ Z_j.
  - Global conditional: Ŷ ⟂ Z_j | Z_{−j}.
  - Local conditional, for a fixed z and S ⊆ [m]∖{j}: Ŷ_{S∪{j}} =d Ŷ_S, where Ŷ_C = g(H̃_C) and H̃_C ~ P_{H | Z_C = z_C} — [arXiv 2405.19146v2](https://arxiv.org/html/2405.19146v2)
- **What the nulls are about.** The tests concern the fixed model's predictions, not labels. The paper contrasts this with PCBM, which tests dependence on Y — [arXiv 2405.19146v2](https://arxiv.org/html/2405.19146v2)
- **Testing by betting.**
  - The wealth process is K_t = K_{t−1}(1 + v_t κ_t), and the test rejects at τ = min{t : K_t ≥ 1/α}.
  - Type I control is anytime-valid: P_H0(∃t : K_t ≥ 1/α) ≤ α.
  - FDR across m concepts uses a greedy e-value post-processor that rejects when K ≥ m/(α|S̃|). Any self-consistent e-testing procedure controls FDR at α.
  - Concepts that reject sooner are ranked as more important, but guarantees on rejection times are open — [arXiv 2405.19146v2](https://arxiv.org/html/2405.19146v2)
- **Sampler and cost.** x-SKIT needs samples from P_{H | Z_C = z_C}; the paper uses nonparametric samplers. Cost is O(T_n τ²) with T_n = n² — [arXiv 2405.19146v2](https://arxiv.org/html/2405.19146v2)
- **Venue.** NeurIPS 2024, Advances in NeurIPS 37, pp. 76450–76499 — [proceedings listing](https://www.proceedings.com/079017-2435.html)

#### Region-local knockoffs: Gimenez & Zou (ICML 2019)
- **Authorship correction.** "Discovering Conditionally Salient Features with Statistical Guarantees" (ICML 2019, PMLR 97:2290–2298) is by Jaime Roquero Gimenez and James Zou only — [PMLR](https://proceedings.mlr.press/v97/gimenez19a.html). Ghorbani co-authored a separate AISTATS 2019 paper, "Knockoffs for the Mass" — [ar5iv 1807.06214](https://ar5iv.labs.arxiv.org/html/1807.06214)
- **Definition 2.1.** Feature j is a local null at x if X_j ⟂ Y | X_{−j} = x_{−j}. It is an r-local null if this holds at every point within sup-norm distance r of x. Global nulls are contained in every r-local null set — [ar5iv 1905.12177](https://ar5iv.labs.arxiv.org/html/1905.12177)
- **Error rate.** FDR^r(x) is the expected number of selected r-local nulls divided by max(1, |Ŝ(x)|). Theorem 3.1 bounds sup_l E[FDP^r(z_l)] ≤ q at L points that are pairwise at least 2r apart. The focus is "point-wise FDR^r at a finite number of distinct points far apart" — [ar5iv](https://ar5iv.labs.arxiv.org/html/1905.12177)
- **Assumptions.** P^X is known, knockoffs are generated from the whole dataset, the density is positive, and importance scores are r-local. The paper asserts the guarantee holds "regardless of the method used to partition the space", but the proofs treat the points as given — [ar5iv](https://ar5iv.labs.arxiv.org/html/1905.12177)
- **Scope and evidence.** The guarantee is region-level, not per observation, and power is limited by how much data lies near the point. Experiments use simulated HMM genotypes (d = 200, q = 0.2) with the known optimal partition — [ar5iv](https://ar5iv.labs.arxiv.org/html/1905.12177)

#### Global data-level tools these build on, and robustness to an approximate sampler
- **Model-X knockoffs.** FDR control holds in finite samples when Y | X is arbitrary and completely unknown, provided the covariates are i.i.d. with a known distribution. The authors report preliminary evidence of robustness to estimated distributions — [Candès, Fan, Janson & Lv, JRSSB 2018, arXiv 1610.02351](https://arxiv.org/abs/1610.02351v4)
- **Holdout Randomization Test (HRT).**
  - A specialized CRT that uses data splitting to keep computation feasible.
  - Model-agnostic, with a valid p-value per feature, so FDR or type I error can be controlled.
  - More power than a knockoff competitor in simulations — [Tansey, Veitch, Zhang, Rabadan & Blei, JCGS 2022 (31:151–162), arXiv 1811.00645](https://arxiv.org/pdf/1811.00645). Code — [tansey/hrt](https://github.com/tansey/hrt/blob/master/README.md)
- **Robust knockoffs.** FDR inflation is proportional to the errors in estimating each feature's conditional distribution given the others, with no modeling assumptions — [Barber, Candès & Samworth, AoS 2020, arXiv 1801.03896](https://arxiv.org/abs/1801.03896)
- **Conditional permutation test.** Type I error bounds are stated in terms of the error in approximating X | Z. For the worst-case statistic, CPT inflation is no larger than CRT inflation — [Berrett, Wang, Barber & Samworth, JRSSB 2020](https://ideas.repec.org/a/bla/jorssb/v82y2020i1p175-197.html); [arXiv 1807.05405](https://ar5iv.labs.arxiv.org/html/1807.05405)

#### Local LOCO and conformal intervals around explanations
- **Local LOCO (Lei, G'Sell, Rinaldo, Tibshirani & Wasserman, JASA 2018).**
  - Δ_j(X_{n+1}, Y_{n+1}) = |Y_{n+1} − μ̂_{(−j)}(X_{n+1})| − |Y_{n+1} − μ̂(X_{n+1})|, where μ̂_{(−j)} is refit without covariate j.
  - With C a conformal prediction set, W_j(x) = {|y − μ̂_{(−j)}(x)| − |y − μ̂(x)| : y ∈ C(x)} satisfies P(Δ_j ∈ W_j(X_{n+1}) for all j) ≥ 1 − α.
  - This holds in finite samples, uniformly over j, and "there is no need to do any multiplicity adjustment".
  - But coverage "is marginal over X_{n+1}, and in general, does not hold conditionally at X_{n+1} = x". Plotting W_j(X_i) supports only "loosely" local statements — [paper PDF, §6.1](https://www.stat.cmu.edu/~ryantibs/papers/conformal.pdf); [arXiv 1604.04173](https://arxiv.org/abs/1604.04173v2)
- **Information-theoretic Shapley values (Watson, O'Hara, Tax, Mudd & Guy, NeurIPS 2023).**
  - Shapley values are computed for conditional entropy or loss.
  - Local "context-specific" independence, Y ⟂ X_j | x_S, implies a zero marginal payoff. The converse fails only on a Lebesgue-null set of distributions.
  - Inference is a split-conformal interval for ϕ(j, X_{n+1}) with P(ϕ ∈ [q_lo, q_hi]) ≥ 1 − α. The paper states this is "not a conditional coverage claim", and there is no multiple-testing treatment — [arXiv 2306.05724v2](https://arxiv.org/html/2306.05724v2)
- **Recent conformal–Shapley work.** These papers attribute the *width* of conformal intervals to features; they are not feature tests — [arXiv 2505.13118 (2025)](https://arxiv.org/pdf/2505.13118); [ICML 2026 "A Tale of Two Uncertainties"](https://icml.cc/virtual/2026/84068)

#### Shapley confidence intervals and Monte Carlo uncertainty for a fixed model
- **Williamson & Feng (ICML 2020).**
  - Inference on the *population* Shapley variable importance measure (SPVIM, global). The estimator samples Θ(n) subsets, and its asymptotic distribution gives CIs and tests.
  - Motivation: earlier nonparametric measures give near-zero importance to highly correlated features.
  - In-hospital mortality estimates were similar across ML algorithms — [PMLR](https://proceedings.mlr.press/v119/williamson20a.html); [arXiv 2006.09481](https://arxiv.org/pdf/2006.09481)
- **BayesLIME and BayesSHAP (Slack, Hilgard, Singh & Lakkaraju, NeurIPS 2021).** Credible intervals cover the perturbation-sampling uncertainty of LIME and KernelSHAP — [arXiv 2008.05030](https://arxiv.org/pdf/2008.05030). The authors' slides report about 95–97% empirical coverage for BayesLIME's 95% intervals and about 88–90% for BayesSHAP — [slides](https://xaiseminars.doc.ic.ac.uk/materials/2022-06-22_Slack.pdf)
- **Goldwasser & Hooker (arXiv Jan 2024; v5 Jul 2025).** Hypothesis tests that make the top-K of local or global SHAP/LIME rankings stable. Sampling algorithms identify the K most important features with probability above 1 − α. Instability is attributed to random sampling — [arXiv 2401.15800](https://arxiv.org/abs/2401.15800)
- **AICO (Giesecke, Horel & Jirachotkulthorn; arXiv Jun 2025, v7 Jul 2026).** A masking-based method claiming "exact, finite-sample feature p-values and confidence intervals" and requiring "no retraining, surrogate modeling, or distributional assumptions" — [arXiv 2506.23396](https://arxiv.org/abs/2506.23396). A search summary describes its local measure as a per-sample feature effect and its global measure as the median of that effect — [arXiv HTML](https://arxiv.org/html/2506.23396)

#### Selective inference for thresholded saliency maps: Miwa, Duy & Takeuchi (ICLR 2023)
- **Data model.** The input is X = s + ε with ε ~ N(0, σ²I) and σ known. There is a reference input X^ref = s^ref + ε^ref.
- **Selection.** The salient region is ℳ_X = {i : A_i(X) ≥ τ}.
- **Nulls.** Either the mean of s over ℳ_X equals the mean of s^ref over ℳ_X, or s_i = s_i^ref for all i in ℳ_X.
- **Inference.** The p-value conditions on the selection event (the observed ℳ_X) and uses a truncated normal or truncated χ distribution.
- **Requirements.** Piecewise-linear networks (ReLU, max-pooling, convolution). Experiments use CAM saliency — [ar5iv 2301.02437](https://ar5iv.labs.arxiv.org/html/2301.02437); [ICLR 2023](https://www.iclr.cc/virtual/2023/poster/11275)

#### Inference on derivatives and gradients (the advisor's flagged area)
- **Horel & Giesecke (JMLR 2020).** A gradient-based, global significance test for single-hidden-layer neural network regression, treated as a sieve estimator. The asymptotic null distribution is a mixture of χ² — [JMLR](https://jmlr.org/papers/v21/19-264.html); [arXiv 1902.06021](https://arxiv.org/pdf/1902.06021v2)
- **Liu & Li (JMLR 2023).**
  - Plug-in KRR estimators for arbitrary mixed partial derivatives, with L∞ error bounds.
  - Minimax-optimal up to log factors for Hölder and Sobolev classes, when kernel eigenvalues decay polynomially.
  - One tuning parameter works for every derivative order — [JMLR v24 21-1110](https://www.jmlr.org/papers/v24/21-1110.html)
- **Liu & Li (arXiv Oct 2022, rev. Dec 2025).** Differentiating a GP posterior attains minimax rates for all derivative orders simultaneously, with one hyperparameter choice. Empirical Bayes tuning is analyzed — [arXiv 2210.11626](https://arxiv.org/abs/2210.11626v3)
- **Liu, Li & Li (arXiv Aug 2023, rev. Feb 2026).**
  - Plug-in KRR derivatives in smoothing-spline ANOVA, with L∞ rates and a minimax lower bound.
  - A bootstrap test of "whether a derivative is zero", with correct size and power against local alternatives — [arXiv 2308.13905](https://arxiv.org/abs/2308.13905)
- **Yoo & Ghosal (AoS 2016).**
  - Tensor-product B-spline priors.
  - Posterior contraction at minimax rates for the regression function and its mixed partial derivatives, in pointwise, L2, and L∞ norms.
  - Pointwise, L2, and L∞ credible sets have guaranteed frequentist coverage with optimal size — [arXiv 1411.6716](https://arxiv.org/pdf/1411.6716)
- **Local polynomials with robust bias correction.** Calonico, Cattaneo & Farrell (JASA 2018; Bernoulli 2022 coverage-error results that cover derivative estimation). Implemented in `nprobust::lprobust` — [CRAN nprobust](https://cran.r-project.org/web/packages/nprobust/index.html); [lprobust manual](https://rdrr.io/cran/nprobust/man/lprobust.html)
- **Debiased ML for local functionals.**
  - Chernozhukov, Newey & Singh (Econometrics Journal 2022) give honest confidence bands for regular and non-regular functionals. These include derivatives "conditional on a covariate subvector fixed at a point" — [arXiv 1802.08667](https://arxiv.org/abs/1802.08667); [MIT DSpace](https://dspace.mit.edu/handle/1721.1/144461)
  - RieszNet and ForestRiesz automate debiasing using only black-box oracle access to the functional — [arXiv 2110.03031](https://arxiv.org/abs/2110.03031v2)
- **Deep neural networks.**
  - Seminar abstracts from 2024 describe a test that a partial derivative of a high-dimensional regression function is zero. The estimator convolves a DNN fit with the derivative of a smoothing kernel. The abstracts say smoothing "becomes beneficial and even necessary". The statistic is asymptotically normal under the null, assuming a latent-factor structure — [KCL seminar](https://mth.kcl.ac.uk/statistics/Spring2024/2024-02-29/); [Erasmus seminar](https://eur.nl/en/ese/events/inference-derivatives-high-dimensional-regression-function-deep-neural-network-2024-04-04)
  - Separately, sparse-DNN estimates of input partial derivatives converge at O(n^{−1/4}), versus O(n^{−1/2}) for the model itself — [Wu & Li, arXiv 2406.18137](https://arxiv.org/abs/2406.18137)
- **Functional decompositions.** Hiabu, Meyer & Wright (AISTATS 2023) give a functional ANOVA (fANOVA) identification under which interventional SHAP and partial dependence plots are read off main and interaction components. Exact computation is feasible only for low-dimensional structures — [PMLR](https://proceedings.mlr.press/v206/hiabu23a.html)

### Inferences
- **Two different probability spaces.**
  - (A) *Sampler-relative nulls about f̂ at one x* (Burns IRT, SHAP-XRT, x-SKIT). The only randomness is the reference draw, so validity is exact (up to Monte Carlo) for any statistic. The guarantee reads: *if x_S had been drawn from Q(·|x_{−S}), the test rejects with probability ≤ α.*
  - (B) *Data-level nulls* (CRT, knockoffs, HRT, Gimenez & Zou, Watson). These need many i.i.d. (X, Y) pairs and a known or estimated P_X. They become global, regional, or marginal over test points.
  - Neither family yields a per-instance claim about f.
- **Family (A) fits planted-motif simulations exactly.** Suppose the background is generated from Q and f̂ is independent of the test sequence (held out from training). Then every window that does not overlap a planted motif satisfies the null exactly. FDR against planted ground truth becomes a theorem rather than an empirical hope: BH under PRDS, and BY or e-BH under arbitrary dependence.
- **Gimenez & Zou's sup-norm balls degenerate on one-hot DNA.** Distances between one-hot sequences are 0 or 1, so any r < 1 gives a singleton ball with no data. A genomic version needs a coarser neighborhood, such as "sequences containing partner motif B". That matches their "switch feature" simulation design.
- **Derivative inference does not transfer cleanly to DNA.**
  - A pointwise ∂f/∂x_j(x) is a non-regular target: it needs smoothing, converges slower than √n, and requires bias handling. Average derivatives get √n debiased inference but are global.
  - For one-hot DNA, f = E[Y|X] is defined only on the simplex vertices, so ∂f/∂x_j is not identified by data. Only finite differences between sequences (ISM contrasts) are identified.
  - That turns "derivative inference" into inference on function values at neighboring lattice points. Majdandzic & Koo's off-simplex correction (Q2) is the practical face of this.
- **Any statistic can drive the test.** Because family (A) is valid for any statistic T, a saliency, IG, or ISM window sum can serve as T. Its (1−α) quantile under resampling then gives a per-query calibrated threshold τ(x, W). This connects to the meeting-notes idea τ(x) = f(T(x)), with a precise (sampler-relative) error guarantee.

### Gaps
- The retrieved text of Teneggi & Sulam does not discuss what happens when the conditional sampler is misspecified.
- **DNN derivative test.** Seminar listings name Weining Wang or Yue Zhao as speaker. The arXiv ID and author list are not confirmed.
- **Liu, Li & Li.** It is not established whether the derivative-zero test is pointwise or global.
- **AICO.** The exact null, the masking scheme, and whether any per-sample test exists were not verified from the full text.
- **CRT robustness.** The exact metric in the CRT/CPT robustness bound (for example, total variation) was not verified. The source says only "error in the approximation of X|Z".
- **Not verified this session, so excluded from findings:**
  - Covert & Lee's KernelSHAP confidence intervals.
  - Verdinelli & Wasserman's critiques of LOCO and Shapley under correlation.
  - Farrell, Liang & Misra on DNN inference.
  - RKHS "learning gradients" (Mukherjee & Zhou).
- **"Functional interpretability".** The advisor's meaning is not pinned down. fANOVA decompositions and functionals of f (gradients, IG) are two plausible readings.

---

## 2. Genomics-specific inferential work: knockoffs for sequences, KnockoffZoom-style groups, DeepPINK-style uses, conditional randomization with sequence samplers, shuffled-sequence nulls, gradient correction, attribution priors, ensembles and distillation, EvoAug, and motif-hit significance

### Takeaway
Genomics has three relevant strands, and none provides validated per-sequence error control:
1. **Global knockoff methods** on k-mers or tabular features (KIMI, DeepPINK, DeepLINK), plus GWAS machinery for contiguous groups at multiple resolutions (HMM knockoffs, KnockoffZoom, the multilayer knockoff filter, TreeBH). The GWAS machinery is a direct template for windows.
2. **Shuffle- or perturbation-based nulls on fitted genomic DNNs** (DFIM, Global Importance Analysis, CREME, AlphaGenome sham deletions, seqlet-calling thresholds). These are either heuristic, or they report p-values without multiplicity control or planted-truth validation.
3. **Attribution-reliability methods** (off-simplex gradient correction, Fourier attribution prior, EvoAug, DEGU ensembles, consistency-based model selection). These reduce noise but give no error guarantees.

I found no per-sequence CRT or knockoff procedure for motif instances in a genomic DNN with validated FDR.

### Cited Findings

#### Knockoffs in sequence and genomic settings (all global, across samples)
- **KIMI (Bai, Ren, Fan & Sun, Bioinformatics 2020).** Model-X knockoffs select k-mers that distinguish groups of sequences, with "guaranteed false discovery rate (FDR) control". It beat BH and q-value methods in simulations and was applied to viral versus bacterial contigs — [MIDAS summary](https://midasnetwork.us/papers/kimi-knockoff-inference-for-motif-identification-from-molecular-sequences-with-controlled-false-discovery-rate/)
- **DeepPINK (Lu, Fan, Lv & Noble, NeurIPS 2018).** An MLP with a pairwise-connected layer linking each feature to its knockoff, giving FDR-controlled global selection — [arXiv 1809.01185](https://arxiv.org/pdf/1809.01185); [NeurIPS](https://papers.nips.cc/paper/2018/hash/29daf9442f3c0b60642b14c081b4a556-Abstract.html)
- **DeepLINK (Zhu, Fan, Kong, Lv & Sun, PNAS 2021).** Autoencoder-generated knockoffs plus MLP selection, applied to genomics and microbiome data — [PNAS](https://www.pnas.org/doi/full/10.1073/pnas.2104683118); [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8433583)
- **Recent knockoff generators.**
  - DiffKnock (arXiv Oct 2025) uses diffusion-based knockoffs with a transformer denoiser and "approximates exchangeability". I found no DNA-motif application — [arXiv 2510.01418](https://arxiv.org/html/2510.01418)
  - Liao, Shi & Xie (arXiv Jun 2026) do knockoff FDR screening of DNN inputs, with no instance-wise or genomic focus — [arXiv 2606.04404](https://arxiv.org/abs/2606.04404)
- **HMM knockoffs (Sesia, Sabatti & Candès, Biometrika 2019).** An exact, efficient knockoff sampler for covariates that follow a hidden Markov model, with GWAS FDR control. Implemented in the R package SNPknock — [arXiv 1706.04677](https://arxiv.org/pdf/1706.04677); [RePEc](https://ideas.repec.org/a/oup/biomet/v106y2019i1p1-18..html). Before Metropolized knockoffs, concrete implementations of the generic SCIP sampler existed only for Gaussians and discrete Markov chains (the latter by Sesia et al.) — [Bates et al., arXiv 1903.00434](https://arxiv.org/pdf/1903.00434)

#### Group and multi-resolution templates for contiguous windows
- **KnockoffZoom (Sesia, Katsevich, Bates, Candès & Sabatti, Nat Commun 2020).**
  - Groups: contiguous LD blocks from adjacency-constrained hierarchical clustering on PLINK r², fixed before the phenotype is examined. Block widths run from 0.226 Mb down to 0.001 Mb, plus single SNPs.
  - Group null: the trait is independent of the group's genotypes given all other variants. A group is null only if all its subgroups are null.
  - FDR is controlled "at each resolution separately", not jointly.
  - Group knockoffs come from a fastPHASE HMM, and the algorithms scale linearly — [PMC7046731](https://pmc.ncbi.nlm.nih.gov/articles/PMC7046731/)
- **Multilayer knockoff filter (Katsevich & Sabatti, AoAS 2019).** Simultaneous FDR control at the individual and group levels, with small power loss — [arXiv 1706.09375](https://arxiv.org/pdf/1706.09375). A 2024 follow-up proposes e-filter variants for stability — [arXiv 2409.17039](https://www.arxiv.org/pdf/2409.17039v1)
- **TreeBH (Bogomolov, Peterson, Benjamini & Sabatti, Biometrika 2021).** Error control at multiple levels of a tree of hypotheses under stated dependence assumptions, plus a modified version valid under any dependence — [arXiv 1705.07529](https://arxiv.org/pdf/1705.07529)

#### Shuffle- and perturbation-based nulls already used on fitted genomic DNNs
- **DFIM (Greenside, Shimko, Fordyce & Kundaje, Bioinformatics 2018).**
  - The null for feature-interaction scores is built from dinucleotide-shuffled inputs, with a Gaussian fitted to the null scores and a p < 0.05 cutoff.
  - Benjamini–Hochberg is named, but the family of tests is not specified.
  - Planted ELF1–SIX5 interactions were recovered with AP1/TAL1 decoys as negatives. No explicit FDR was reported — [PMC6129272](https://pmc.ncbi.nlm.nih.gov/articles/PMC6129272/)
- **Global Importance Analysis (GIA; Koo et al., PLoS Comput Biol 2021).**
  - Defined as the average, over sampled backgrounds, of the prediction difference with versus without an embedded pattern.
  - Backgrounds come from a profile model, random shuffles, dinucleotide shuffles, or quartile subsets.
  - "GIA is, at its core, a model interpretability tool". No significance test is applied to GIA values.
  - Attribution scores are "meaningful only within the context of an individual sequence" — [PLoS CB](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008925)
- **CREME (Toneyan & Koo, Nat Genet 2024).**
  - Tiles: 38 non-overlapping 5-kb tiles over the central 200 kb of the Enformer or Borzoi input.
  - Necessity test: dinucleotide-shuffle one tile.
  - Sufficiency test: shuffled background plus the TSS tile plus the tile of interest.
  - Context test: shuffle everything except the TSS tile.
  - Shuffle count: the Methods say 10 shuffles; the Results say 100.
  - Calls use heuristic cutoffs on normalized effects. For example, Enformer context is "enhancing" above 0.95, neutral below 0.05 in absolute value, and silencing below −0.3; sufficiency enhancers exceed 0.3.
  - No multiple-testing correction is applied to CRE calls — [PMC12065635](https://pmc.ncbi.nlm.nih.gov/articles/PMC12065635/); [docs](https://creme-nn.readthedocs.io/)
- **AlphaGenome in-silico ablation (Liang et al., Briefings in Bioinformatics, Aug 2026).**
  - FIMO nominates candidates at p ≤ 1e−3. Each motif is deleted within a 1-Mb context.
  - The null is K = 3000 random, length-matched "sham" deletions in the same context. A motif is called if its log fold change falls outside the 2.5–97.5 percentile of that null.
  - Tests are per motif, with no BH or FDR. There is no planted-truth or MPRA benchmark.
  - Redundancy is discussed only qualitatively, as a limitation — [Brief Bioinform 27(4) bbag422](https://academic.oup.com/bib/article/27/4/bbag422/8753614)
- **SQUID (Seitz, McCandlish, Kinney & Koo, Nat Mach Intell 2024).** Surrogate models are fit to the DNN within a user-specified region of sequence space. They account for global nonlinearity and heteroskedastic noise, give more consistent motifs, and include epistasis surrogates. There are no error-control claims — [CSHL](https://repository.cshl.edu/id/eprint/41591/); [bioRxiv](https://www.biorxiv.org/content/10.1101/2023.11.14.567120.full.pdf)
- **Evaluation baselines (Han & Kim, arXiv Jul 2026).**
  - Raw motif-recovery rates need their chance level. For contiguous overlap the uniform chance level is L/(N−L+1), which ranges from 0.0118 to 0.0427 across 268 UniBind TFs.
  - Perturbation curves can be non-monotone, and a fully masked input can still score above the decision boundary — [arXiv 2609.16271](https://arxiv.org/abs/2609.16271)

#### Connections to the genomics tools' built-in thresholds (details belong to the companion notes)
- **tangermeme.**
  - `tfmodisco_seqlets`: `target_fdr` (default 0.2) is "A FDR value to set on attribution score sums over windows when separating called seqlets from background". The API page does not document how the FDR is estimated.
  - `recursive_seqlets`: p-values come from distributions of attribution sums for each seqlet length, with a p-value threshold of 0.01. A span is called only if all of its internal spans pass — [tangermeme seqlet API](https://tangermeme.readthedocs.io/en/latest/api/seqlet.html)
  - CREsted exposes an `fdr` parameter (default 0.05) for seqlet finding — [CREsted](https://crested.readthedocs.io/en/latest/api/_autosummary/crested.tl.modisco.tfmodisco.html)
- **FIMO (Grant, Bailey & Noble, Bioinformatics 2011).** PWM log-likelihood-ratio scores are converted to p-values by dynamic programming under a zero-order background (default threshold p < 1e−4), then to q-values via BH. It scans 3.5 Mb/s on one CPU — [PMC3065696](https://pmc.ncbi.nlm.nih.gov/articles/PMC3065696); [Noble lab](https://noble.gs.washington.edu/papers/grant2011fimo.html)

#### Attribution reliability: gradient correction, priors, augmentation, ensembles
- **Off-simplex gradient correction (Majdandzic, Rajesh & Koo, Genome Biology 2023).**
  - DNNs learn functions off the one-hot simplex, which adds a random gradient component orthogonal to the simplex. That component shows up as attribution noise.
  - Removing it improves saliency, integrated gradients, SmoothGrad, and expected gradients.
  - The scale of random initialization drives the off-simplex gradient, and the correction can also act as a training regularizer — [PMC10169356](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10169356/); [bioRxiv](https://www.biorxiv.org/content/10.1101/2022.04.29.490102.full.pdf)
- **Fourier attribution prior (Tseng, Shrikumar & Kundaje, NeurIPS 2020).** Genomic models are "sensitive to random initializations, often resulting in noisy and irreproducible attributions". The prior penalizes high-frequency components of attribution scores during training. DeepSHAP is computed against dinucleotide-shuffled references — [bioRxiv](https://www.biorxiv.org/content/10.1101/2020.06.11.147272v1); [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2020/file/1487987e862c44b91a0296cf3866387e-Paper.pdf)
- **Consistency-based model selection (Majdandzic et al., PMLR vol. 200, 2022).** Spurious importance scores vary across models with similar accuracy. The paper proposes consistency metrics, needing no ground truth, for choosing models — [PMLR v200](https://proceedings.mlr.press/v200/majdandzic22a/majdandzic22a.pdf); [CSHL](https://repository.cshl.edu/id/eprint/40956/)
- **DEGU (Zhou et al., Koo lab; bioRxiv Nov 2024; ICLR 2025 MLGenX; npj Artificial Intelligence Jan 2026).**
  - A student model distilled from an ensemble predicts the ensemble mean and spread; the spread is used as epistemic uncertainty.
  - An optional auxiliary task estimates aleatoric uncertainty from experimental replicates.
  - Reported benefits: more consistent attributions, and conformal prediction for coverage — [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.11.13.623485.full.pdf); [CSHL npj record](https://repository.cshl.edu/id/eprint/42066/)
- **EvoAug (Lee et al., Genome Biology 2023).** Evolution-inspired augmentations (mutations, indels, inversions, translocations) followed by fine-tuning on the original data. Reported to improve attribution interpretability — [Genome Biology](https://link.springer.com/article/10.1186/s13059-023-02941-w)

#### Conditional samplers for DNA
- **uShuffle (Jiang et al., BMC Bioinformatics 2008).** Uniform random permutations that preserve exact k-let counts, using the Euler algorithm with Wilson's arborescence step. Altschul & Erickson gave the first truly uniform doublet-preserving shuffle — [PMC2375906](https://pmc.ncbi.nlm.nih.gov/articles/PMC2375906). The Euler method keeps the first and last letters fixed — [universalmotif vignette](https://bioc.r-universe.dev/universalmotif/doc/SequenceSearches.Rmd)
- **DNA language models (gLMs) as samplers.**
  - Tang, Somia, Yu & Koo (Genome Biology 2025): frozen gLM embeddings (Nucleotide Transformer, DNABERT2, HyenaDNA, GPN) give no clear advantage over one-hot models. Attributions from masked-nucleotide entropy "did not reflect any known motifs" and may reflect "diffuse, low-level sequence statistics" — [PMC12261763](https://pmc.ncbi.nlm.nih.gov/articles/PMC12261763/)
  - Cheng & Zhang (arXiv Apr 2026): compositional effects outweigh positional effects 46-fold. Evo2-1B and Caduceus score misplaced elements higher than correctly placed ones, and all models tested were insensitive to strand — [arXiv 2604.06549](https://arxiv.org/abs/2604.06549)
- **Sampling with masked-LM conditionals.**
  - Gibbs sampling with these conditionals gives low-quality samples, so they are better used as Metropolis–Hastings proposals — [arXiv 2106.02736](https://arxiv.org/pdf/2106.02736)
  - Inconsistent conditionals have no joint distribution, and Gibbs sampling may amplify errors — [arXiv 2407.21046](https://arxiv.org/html/2407.21046v1)

### Inferences
- **Exact conditional samplers for DNA are cheap under explicit null models.**
  - Under a k-th-order Markov background, X_W | X_{−W} depends only on the k flanking bases on each side. It can be sampled exactly by forward–backward (bridge) sampling.
  - Under the "uniform given dinucleotide counts" model (Altschul–Erickson), the window conditional is also exact. Uniformly shuffling the segment [a−1, b+1] with k = 2 and its endpoints fixed gives the conditional distribution of the window given the rest. This works because the doublets outside the segment and the boundary letters are fixed.
  - This is my derivation. I found no paper that states it or uses it as a CRT sampler.
- **The shuffle-based p-values in current use test a marginal null.** DFIM, the seqlet thresholds, and the sham deletions all compare a statistic at x with its distribution on other sequences, shuffled sequences, or random deletions. None tests the conditional null at x. Windows next to strong motifs (attribution spillover) or with unusual composition can violate the conditional null, so FDR relative to planted truth is not guaranteed. This is checkable.
- **gLM conditionals are probably no better than Markov or dinucleotide models for regulatory DNA** (Tang et al. 2025; Cheng & Zhang 2026), and they add sampling pathologies. At most, use them as a comparison arm.
- **Hierarchies map onto sequences.** KnockoffZoom, the multilayer filter, and TreeBH map naturally onto positions, then windows, then regions within one sequence. Knockoff guarantees are across i.i.d. samples. A per-sequence hierarchy would apply the same logic to p-values from one input; TreeBH accepts any p-values.

### Gaps
- KIMI's knockoff generator for k-mer features was not verified; only an abstract-level summary was read.
- The tangermeme API page does not document TF-MoDISco's internal null construction (shuffled sequences versus a Laplace fit). This is deferred to the companion researcher's notes.
- DFIM does not specify whether its BH family is within-sequence or pooled.
- No study found quantifies motif-instance-level agreement of selections across seeds or ensembles with error control.
- **Not verified this session:**
  - ChromBPNet and BPNet fold-averaging practices.
  - Fi-NeMo's hit-calling statistics (companion notes).
  - The input lengths of BPNet, DeepSTARR, and Enformer.

---

## 3. Method by method: what exactly is H0, is it local, is error controlled within one sequence or across many, and does it handle contiguous groups?

### Takeaway
- **Per-instance methods test f̂.** Every genuinely per-instance method tests a sampler-relative null about f̂: Burns IRT, SHAP-XRT, x-SKIT, and heuristically CREME, GIA, and sham deletions.
- **Data nulls are not per-instance.** Every method with a null about the data distribution is global or regional (CRT, knockoffs, HRT, KnockoffZoom, Gimenez–Zou), or gives coverage that is marginal over test points (local LOCO, Watson).
- **Within-one-instance FDR has only two formal routes:** Burns IRT (BH or BY over subsets of one x) and x-SKIT (e-value FDR over concepts).
- **Contiguous groups:** handling is mature only in the GWAS knockoff literature.

### Cited Findings

#### (a) Per-instance tests and heuristics about a fixed f̂
(The symbol "∣" means "given" inside the tables.)

| Method | H0 at input x | Error control and its scope | Contiguous groups | Requirements and cost |
|---|---|---|---|---|
| Burns et al. IRT — [FODS 2020](https://ar5iv.arxiv.org/html/1904.00045) | Point null f̂(x) ~ f̂(X̃), with X̃_S ~ Q(· ∣ x_{−S}) | Randomization p-values; BH across the N subsets of one x (independence or PRDS); BY otherwise | Arbitrary subsets; non-overlapping patches | User-chosen in-support Q; K draws × N subsets forward passes |
| Burns et al. OSFT — [same](https://ar5iv.arxiv.org/html/1904.00045) | Same, with a single draw and the knockoff rule | FDR only if statistics and features are independent; pooled bound Nα over M samples | Same | 1 draw per subset |
| SHAP-XRT — [TMLR 2023](https://arxiv.org/html/2207.07038v5) | f̂(X̃_{C∪{j}}) =d f̂(X̃_C), complement drawn from the data conditional given x_C | Type I per (j, C); composite over C via 2 × Shapley-weighted average p; no FDR | Features can be patches or quadrants | Conditional sampler for every C; exponential in coalitions |
| x-SKIT — [NeurIPS 2024](https://arxiv.org/html/2405.19146v2) | Ŷ_{S∪{j}} =d Ŷ_S, with H̃_C ~ P(H ∣ Z_C = z_C) | Anytime-valid type I; FDR across concepts via e-values | Concepts are semantic groups | Sampler for P(H ∣ Z_C); O(n²τ²) |
| Miwa, Duy & Takeuchi — [ICLR 2023](https://ar5iv.labs.arxiv.org/html/2301.02437) | Mean of signal over the selected salient region equals the reference mean | Selective type I, conditional on the selected region | Region = thresholded saliency set | Gaussian input noise with known σ; piecewise-linear networks |
| CREME — [Nat Genet 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC12065635/) | No formal null; normalized effect of dinucleotide-shuffling a tile | None (heuristic cutoffs) | Fixed 5-kb tiles | 10 (or 100) shuffles per tile |
| AlphaGenome sham ablation — [Brief Bioinform 2026](https://academic.oup.com/bib/article/27/4/bbag422/8753614) | Motif-deletion LFC exchangeable with 3000 random length-matched deletions | Per-motif 95% interval; no multiplicity correction | Motif instance | 3000 forward passes per motif at 1 Mb |
| GIA — [PLoS CB 2021](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008925) | No formal null; mean change in prediction from an embedded pattern over backgrounds | None | Embedded patterns | Background sampler |
| DFIM — [Bioinformatics 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6129272/) | Interaction score versus a Gaussian fit on dinucleotide-shuffled inputs | p < 0.05 plus BH (family unspecified) | Position pairs | Shuffles per input |

#### (b) Data-level nulls (about Y, P, or f)

| Method | H0 | Error control and its scope | Groups | Requirements |
|---|---|---|---|---|
| Model-X CRT and knockoffs — [JRSSB 2018](https://arxiv.org/abs/1610.02351v4) | Y ⟂ X_j ∣ X_{−j} | Global type I (CRT); FDR (knockoffs) | Group knockoffs exist | Known P_X |
| HRT — [JCGS 2022](https://arxiv.org/pdf/1811.00645) | Same; the model fitted on the training split is the test statistic | Global per-feature p-values, then FDR | — | Conditional sampler; data split |
| Gimenez & Zou — [ICML 2019](https://ar5iv.labs.arxiv.org/html/1905.12177) | X_j ⟂ Y ∣ X_{−j} = z_{−j} for all z in B(x, r) | FDR^r at L well-separated points | — | Known P_X; data in the region |
| KnockoffZoom — [Nat Commun 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7046731/) | Y ⟂ X_G ∣ X_{−G}, with G contiguous | FDR per resolution (separately) | Adjacency-constrained clusters | HMM model of X |
| Multilayer knockoff filter — [AoAS 2019](https://arxiv.org/pdf/1706.09375) | Layered group nulls | Simultaneous FDR across layers | Groups | Knockoffs |
| Local LOCO — [JASA 2018](https://www.stat.cmu.edu/~ryantibs/papers/conformal.pdf) | Interval for Δ_j(X, Y), defined by the fitted μ̂ and μ̂_{(−j)} | Simultaneous coverage over j; marginal over the test point | — | Refit without j |
| Watson et al. — [NeurIPS 2023](https://arxiv.org/html/2306.05724v2) | Y ⟂ X_j ∣ x_S implies zero payoff; interval for ϕ(j, X_{n+1}) | Marginal coverage per feature; no FDR | — | Sample split; i.i.d. data |
| Horel & Giesecke — [JMLR 2020](https://jmlr.org/papers/v21/19-264.html) | Gradient-based statistic equals 0 (global) | Asymptotic χ²-mixture test | — | Single-layer sieve NN |
| Pointwise derivative CIs — [Yoo & Ghosal](https://arxiv.org/pdf/1411.6716); [local polynomial with robust bias correction](https://rdrr.io/cran/nprobust/man/lprobust.html); [Chernozhukov–Newey–Singh](https://arxiv.org/abs/1802.08667) | Value of ∂f/∂x_j at a point (or a localized functional) | Pointwise CI coverage (asymptotic or frequentist) | — | Smoothing; low to moderate dimension or sparsity |

### Inferences
- **Within-one-sequence FDR off the shelf.** There are only two ready-made routes:
  - BH or BY over the windows of one sequence, using CRT p-values (Burns IRT).
  - e-value FDR in the style of x-SKIT, which is robust to arbitrary dependence among windows of the same sequence.
- **No contiguous-group multi-resolution control exists per instance.** The obvious construction is per-window CRT p-values at several widths, combined with TreeBH or its arbitrary-dependence variant. I found no prior use of this in sequence models.

### Gaps
- No source found evaluates Burns-style IRT on genomic models, or measures how dependent the p-values of adjacent windows in one sequence are.

---

## 4. Null about the fitted f̂ vs null about f = E[Y|X]: which approaches address each, and how do retraining over seeds, bootstrap, ensembles, or sample splitting enter?

### Takeaway
- **Per-instance machinery addresses f̂-nulls only.** Burns, SHAP-XRT, and x-SKIT say so explicitly, and GIA calls itself "a model interpretability tool".
- **For f̂-nulls, the leftover randomness is computational:** Monte Carlo or reference sampling. Goldwasser–Hooker and BayesSHAP quantify it.
- **f-level claims exist in only two forms:** global tests (CRT, HRT, knockoffs, SPVIM), or nonparametric derivative inference in classical smoothers.
- **DNN training randomness is handled only heuristically** (ensembles, DEGU, consistency-based selection).
- **The Rashomon effect limits even perfect seed-averaging:** it targets "the training procedure", not f.

### Cited Findings
- **f̂-nulls stated explicitly.**
  - Burns et al.: the null concerns f(X) under a user-chosen Q that need not equal P — [ar5iv](https://ar5iv.arxiv.org/html/1904.00045)
  - SHAP-XRT: the null concerns "the response of a deterministic model f on an individual sample" — [arXiv](https://arxiv.org/html/2207.07038v5)
  - Teneggi & Sulam: the tests concern the fixed model's predictions, unlike PCBM, which tests Y — [arXiv](https://arxiv.org/html/2405.19146v2)
- **GIA.** "GIA is, at its core, a model interpretability tool". Its estimates are "subject to the quality of the learned sequence-function relationship". Computing the estimand directly from data would need large randomized datasets — [PLoS CB](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008925)
- **Computational uncertainty only.** BayesLIME and BayesSHAP credible intervals cover perturbation-sampling uncertainty — [arXiv 2008.05030](https://arxiv.org/pdf/2008.05030). Goldwasser & Hooker attribute rank instability to random sampling — [arXiv 2401.15800](https://arxiv.org/abs/2401.15800)
- **Sample splitting.**
  - HRT fits once and tests on held-out data — [arXiv 1811.00645](https://arxiv.org/pdf/1811.00645)
  - Local LOCO defines its target through the fitted μ̂ and μ̂_{(−j)}, with coverage marginal over a new (X, Y) — [paper](https://www.stat.cmu.edu/~ryantibs/papers/conformal.pdf)
  - Watson et al. fit on one half and compute intervals on the other — [arXiv](https://arxiv.org/html/2306.05724v2)
- **Data-level global inference that does not depend on the learner.** SPVIM estimates were similar across ML algorithms in a mortality application — [arXiv 2006.09481](https://arxiv.org/pdf/2006.09481). Model-X and CRT validity requires only a known P_X, not a correct model — [arXiv 1610.02351](https://arxiv.org/abs/1610.02351v4)
- **Rashomon effect and underspecification.** Different well-performing models can rely on different variables. Model class reliance reports the range of reliance across all good models — [Fisher, Rudin & Dominici, JMLR 2019](https://jmlr2020.csail.mit.edu/papers/volume20/18-760/18-760.pdf)
- **Evidence from genomics.**
  - Seed sensitivity yields "noisy and irreproducible attributions" — [Tseng et al.](https://www.biorxiv.org/content/10.1101/2020.06.11.147272v1)
  - Importance scores vary across equally accurate models — [Majdandzic et al. 2022](https://repository.cshl.edu/id/eprint/40956/)
  - DEGU uses ensemble spread as epistemic uncertainty, with conformal coverage — [DEGU](https://www.biorxiv.org/content/10.1101/2024.11.13.623485.full.pdf)
- **"True to the model or true to the data".** Interventional and observational Shapley value functions differ, and neither is best in general — [Chen, Janizek, Lundberg & Lee 2020](https://arxiv.org/pdf/2006.16234)
- **f-level pointwise derivative inference** exists only for smoothers and priors, at nonparametric rates — [Yoo & Ghosal](https://arxiv.org/pdf/1411.6716); [Liu & Li](https://www.jmlr.org/papers/v24/21-1110.html); [Chernozhukov–Newey–Singh](https://arxiv.org/abs/1802.08667)

### Inferences
- **f̂-nulls.** Testing reduces to computing a well-defined, sampler-relative tail probability. That is still a meaningful calibrated threshold. But it says nothing about biology unless f̂ ≈ f on the relevant counterfactuals, and off-distribution counterfactuals are exactly where that fails.
- **A per-sequence f-null is essentially untestable without a model.** Each genomic sequence is unique and has one Y, so E[Y|X=x] at a single x has no replicates. Practical routes:
  - **(a) Treat training as random.** Rerun the f̂-level test across seeds, bootstrap resamples, or CV folds. Combine the p-values with a dependence-robust rule: twice the average p-value (the Vovk–Wang rule SHAP-XRT's Lemma 1 invokes) or an average of e-values. The resulting claim is about "the training procedure on data like D" (an algorithmic-stability target), not about f.
  - **(b) Restrict to ISM contrasts.** Use f(x′) − f(x) for x′ one mutation away, with ensemble or conformal intervals for f at x and x′. This is function-value inference and needs no derivative.
  - **(c) Report at two levels.** Use global data-level tests (HRT or CRT on held-out chromosomes) for the importance of a motif class, and per-sequence f̂-tests to localize instances.
- **The meeting-notes threshold definition is a seed/bootstrap statement.** The notes define a level-α threshold by Pr(|∂f̂/∂x_j(x)| > τ) ≤ α over training randomness. Retraining can estimate this, but no paper I found guarantees it for DNNs. As the notes already say, measuring transfer is not the same as controlling it.

### Gaps
- No theory found for per-instance tests about f with deep models.
- No genomics study combines per-sequence error control with seed or bootstrap retraining.
- Stability selection (Meinshausen & Bühlmann; Shah & Samworth) gives resampling-based error bounds for global selection. I did not verify whether a local, per-instance analogue exists.

---

## 5. How do these frameworks behave under redundancy (two copies of a motif) and non-monotone local value functions?

### Takeaway
- **Necessity-type tests miss redundant copies.** These tests resample one copy and keep the rest. Under a saturating, OR-like model, each copy's single-window null is approximately true for f̂, so neither redundant copy is rejected. This is lost power, not an FDR violation.
- **Other tests catch them.** Sufficiency-type tests (C = ∅), Shapley-combined composite nulls (SHAP-XRT's ∩_C), and group or hierarchical tests do detect the copies.
- **Non-monotonicity.** It does not break equality-in-distribution tests. It does break one-sided statistics, and it makes Shapley-weighted p-value combinations mix signs across coalitions.

### Cited Findings
- **Saturation.** Perturbation and gradient methods "fail to model saturation". Zeroing either of two saturated inputs leaves the output unchanged, so both look unimportant — [Shrikumar, Greenside & Kundaje, DeepLIFT, ICML 2017](https://arxiv.org/pdf/1704.02685)
- **Sufficient input subsets (Carter, Mueller, Jain & Gifford, AISTATS 2019).** Minimal feature sets that alone produce the same decision, found by backward selection. Evaluated on text, image, and genomic models — [PMLR](https://proceedings.mlr.press/v89/carter19a.html)
- **SHAP-XRT's composite null.** H0_j = ∩_C H0_{j,C} is false if any coalition's test is false. Large negative contributions give no evidence against the one-sided null; use 1 − f for the other direction — [arXiv 2207.07038v5](https://arxiv.org/html/2207.07038v5)
- **Necessity versus sufficiency in genomics.** CREME separates necessity (shuffle the element in its native context) from sufficiency (place the element in a shuffled context) — [PMC12065635](https://pmc.ncbi.nlm.nih.gov/articles/PMC12065635/). The AlphaGenome ablation paper lists redundancy as an unaddressed limitation — [Brief Bioinform 2026](https://academic.oup.com/bib/article/27/4/bbag422/8753614)
- **Nested group nulls.** A group is null only if all its subgroups are null — [KnockoffZoom](https://pmc.ncbi.nlm.nih.gov/articles/PMC7046731/)
- **Correlated features.**
  - Nonparametric, LOCO-type importance can be near zero for highly correlated features, which motivated SPVIM — [Williamson & Feng](https://arxiv.org/pdf/2006.09481)
  - Burns et al.'s OSFT had very low two-sided true-positive rate for a discontinuous model with correlated inputs, and word-level tests miss composition — [ar5iv](https://ar5iv.arxiv.org/html/1904.00045)
- **Conditional saliency.** Gimenez & Zou's "switch features" make globally relevant features locally null in some regions — [ar5iv](https://ar5iv.labs.arxiv.org/html/1905.12177)
- **Non-monotone curves in genomics.** Perturbation curves in genomic attribution evaluation can be non-monotone, and a fully masked input can score above the decision boundary — [Han & Kim 2026](https://arxiv.org/abs/2609.16271)

### Inferences
- **Mapping onto the thesis's open problem #1 (non-monotone V).**
  - A CRT null is equality in distribution, so it does not need V to be monotone.
  - Direction enters only through the sidedness of the statistic. Use two-sided statistics, or test f and 1 − f separately, so that repressor motifs (negative local effects) can be selected.
  - Do not clamp: a negative Δ is a legitimate direction of rejection.
- **Predicted behavior under OR-redundancy, with copies A and B:**
  - A single-window necessity test of A (B present) has an approximately true f̂-null, and so does the test of B (A present). Neither copy is selected.
  - A test of the union window A∪B rejects.
  - Sufficiency tests (each copy alone in background) reject both copies.
  - Hierarchical testing (TreeBH over nested windows) reports the region and declines to localize below it, which is the statistically honest answer.
- **Predicted behavior under AND-synergy (both copies required):** necessity tests reject both copies, and sufficiency tests reject neither.
- **These predictions can be checked directly.** Compare five selection rules on OR, AND, and repressor grammars: necessity-CRT, sufficiency-CRT, Shapley-combined p-value, hierarchical group test, and the thesis's MinShap-type rule. That answers "which rule recovers which planted truth" with error control.
- **Planted truth is not the same as model truth.** In redundancy designs a planted copy can be a true null for f̂. Report FDR against planted-null windows (guaranteed under a matched sampler) separately from power against planted motifs, and record which "truth" each rule targets.

### Gaps
- No source found systematically compares error-controlled selection rules on redundant or synergistic motif grammars in genomic DNNs.
- The retrieved text does not discuss how x-SKIT's e-value FDR behaves under redundancy.

---

## 6. Which approaches leave a tractable open problem testable empirically in hours to days? Novelty assessment of candidates (i)–(iv), plus others

### Takeaway

| Candidate | Status |
|---|---|
| (i) Per-sequence FDR from shuffled-sequence nulls + BH | **Partially done.** Shuffle-null p-values or FDR-like thresholds exist (DFIM, seqlet calling, sham deletions). None is validated as per-sequence FDR control against planted truth, and the formal null those p-values test is never stated. |
| (ii) Hierarchical / group testing over windows | **Apparently open** for genomic DNNs. |
| (iii) Error-controlled selection rules compared under redundancy | **Apparently open.** |
| (iv) Exact-conditional CRT with a Markov or dinucleotide sampler | **Apparently open.** |

"Apparently open" means I found no paper doing it for genomic DNNs.

The cleanest thesis-scale design uses (iv) as the engine, with (ii) and (iii) as the experimental axes, on small models with held-out planted-motif data. Under a matched sampler the FDR guarantee is a theorem, so the experiment's job is to test power, and calibration on real genomic backgrounds.

### Cited Findings
- **(i) What already exists.**
  - DFIM: dinucleotide-shuffled null, Gaussian fit, and BH, applied to interactions — [PMC6129272](https://pmc.ncbi.nlm.nih.gov/articles/PMC6129272/)
  - TF-MoDISco seqlet calling with `target_fdr` 0.2 on window sums, and tangermeme recursive-seqlet p-values with a 0.01 threshold — [tangermeme](https://tangermeme.readthedocs.io/en/latest/api/seqlet.html)
  - Per-motif sham-deletion nulls, with no multiplicity control or planted validation — [Brief Bioinform 2026](https://academic.oup.com/bib/article/27/4/bbag422/8753614)
  - FIMO's BH q-values, computed on PWM scores rather than model attributions — [PMC3065696](https://pmc.ncbi.nlm.nih.gov/articles/PMC3065696)
  - Chance-corrected recovery as an evaluation baseline — [arXiv 2609.16271](https://arxiv.org/abs/2609.16271)
- **(ii) Templates.**
  - Contiguous adjacency-constrained groups with per-resolution FDR — [KnockoffZoom](https://pmc.ncbi.nlm.nih.gov/articles/PMC7046731/)
  - Simultaneous multilayer FDR — [multilayer knockoff filter](https://arxiv.org/pdf/1706.09375)
  - Tree-structured FDR — [TreeBH](https://arxiv.org/pdf/1705.07529)
  - Per-instance tests on image quadrants via h-Shap — [SHAP-XRT](https://arxiv.org/html/2207.07038v5)
  - Fixed 5-kb tiles without any testing — [CREME](https://pmc.ncbi.nlm.nih.gov/articles/PMC12065635/)
- **(iii) Pieces.**
  - The saturation example — [DeepLIFT](https://arxiv.org/pdf/1704.02685)
  - Necessity versus sufficiency — [CREME](https://pmc.ncbi.nlm.nih.gov/articles/PMC12065635/)
  - Sufficient subsets, including on genomic data — [SIS](https://proceedings.mlr.press/v89/carter19a.html)
  - Redundancy flagged as a limitation — [Brief Bioinform 2026](https://academic.oup.com/bib/article/27/4/bbag422/8753614)
- **(iv) Pieces.**
  - Per-instance CRT with BH for images and text, using inpainting and BERT samplers — [Burns et al.](https://ar5iv.arxiv.org/html/1904.00045)
  - Dinucleotide shuffles used as backgrounds, not as CRT conditionals — [GIA](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008925); [CREME](https://pmc.ncbi.nlm.nih.gov/articles/PMC12065635/)
  - Exact uniform k-let shuffles — [uShuffle](https://pmc.ncbi.nlm.nih.gov/articles/PMC2375906)
  - Markov-chain and HMM knockoff samplers — [Sesia et al.](https://arxiv.org/pdf/1706.04677)
  - Weak gLM conditionals in regulatory DNA — [Tang et al. 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12261763/); [Cheng & Zhang 2026](https://arxiv.org/abs/2604.06549)
- **Robustness when the sampler is only approximate** (relevant to real, non-Markov genomes) — [Berrett et al.](https://ar5iv.labs.arxiv.org/html/1807.05405); [Barber et al.](https://arxiv.org/abs/1801.03896)

### Inferences

#### (i) Shuffled-null + BH per sequence — partially done; validation open
- **What is new.** Not the procedure itself, but stating its null and testing it.
- **Prediction.** Marginal shuffle nulls will inflate FDR on planted-null windows next to motifs (attribution spillover), on windows with atypical composition, and on GC-rich backgrounds. They may be conservative elsewhere.
- **Falsifier.** Empirical FDR ≤ α in all of those strata.
- **Cost.** Hours. Attributions on a few hundred shuffles per sequence are cheap for short-input CNNs.

#### (ii) Hierarchical windows — apparently open for genomic DNNs
- **Design.** A dyadic tree of windows, with CRT p-values at each node, combined with TreeBH (its arbitrary-dependence variant) or with e-values.
- **Deliverable.** Region-level claims where single windows are redundant.
- **Falsifier.** At matched FDR, hierarchical selection gives no power gain over flat BH on redundant designs.

#### (iii) Selection rules under redundancy — apparently open; targets thesis open problem #1
- **Design.** OR, AND, and repressor grammars, comparing the five rules listed in Q5.
- **Report.** Which planted truth each rule recovers.

#### (iv) Exact-conditional CRT with a Markov or dinucleotide sampler — apparently open
- **Samplers.** Exact ones exist: Markov bridge sampling, and my uShuffle-with-fixed-endpoints derivation (Q2).
- **Validity condition.** The test sequence must be held out from training, because f̂ must not depend on x_W.
- **Simulation design.** Generate the background by Markov bridge sampling around the planted motifs. Overwriting positions after generation breaks exactness for windows adjacent to motifs.
- **Compute.** m windows × K draws forward passes per sequence.
- **Resolution floor.** A K-draw p-value can be no smaller than 1/(K+1), and BH needs a smallest p ≤ α/m. So K ≥ m/α − 1; for m = 50 and α = 0.1, K ≥ 499.
- **Falsifier for the implementation.** Empirical FDR above α on planted nulls under a matched sampler, beyond Monte Carlo error, means a bug.

#### Further candidates I propose
- **(v) Attribution-statistic CRT.**
  - Use a saliency, IG, or ISM window sum (with the Majdandzic correction) as the statistic T. Its (1−α) resampling quantile is then a per-query calibrated threshold.
  - Validity holds for any T (SHAP-XRT; Burns), so the open question is purely about power: T = change in prediction versus attribution sums.
  - I found no genomics precedent. This addresses thesis open problem #2 (threshold calibration) in a precise sense.
- **(vi) Robustness to a misspecified sampler.**
  - Generate backgrounds from real genomic segments or from high-order Markov chains.
  - Test with order-0, order-1, or dinucleotide samplers, and optionally a gLM.
  - Measure type I inflation on planted nulls against the approximation-error bounds of Berrett et al. and Barber et al.
- **(vii) Seed and ensemble replication.** Run the per-model CRT on 5–10 seeds and report selection stability plus combined p-values. This measures the f̂-versus-f gap empirically; it does not control it.

#### Recommended minimal decisive experiment
- **Simulator.** A Markov background with planted PWM instances, labels from OR, AND, and repressor rules, and a small CNN. Evaluate on a held-out test set.
- **Arms (4).**
  1. Exact CRT with change in prediction as the statistic.
  2. Exact CRT with an attribution sum as the statistic.
  3. Shuffle-null BH.
  4. A seqlet-style threshold.
- **Strata (3).** Isolated motif, redundant pair, repressor.
- **Primary metrics.** Per-sequence FDR against planted-null windows, and power.
- **Decision rule.** If shuffle-null BH controls FDR as well as the exact CRT in every stratum, the conditional machinery adds nothing, and the thesis should pivot to (iii). Otherwise the contribution is (iv) plus (v).

### Gaps
- The novelty assessments rest on searches of arXiv, bioRxiv, and journals through Oct 2026, and could miss preprints. Before claiming that (i) is open in its validation sense, cross-check the companion notes on the internals of TF-MoDISco, tangermeme, and Fi-NeMo.
- I found no paper from the Tansey or Koo labs that applies a per-sequence CRT or HRT to genomic DNNs. Absence of evidence is not conclusive.
- I did not source runtimes for BPNet, ChromBPNet, DeepSTARR, or Enformer forward passes. The compute estimates above are stated only as forward-pass counts.
