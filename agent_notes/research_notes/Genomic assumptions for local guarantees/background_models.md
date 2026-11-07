# Background DNA models as a source of statistical guarantees

Scope: the precedent for "background DNA follows a known iid or Markov model, so a test is exact", and whether anyone has turned that into per-sequence guarantees for selections made by deep sequence models. The research budget was about 15 tool calls. Items marked **[recalled]** come from background knowledge and were not re-fetched this session; verify them before citing.

## 1. Motif scanning: exact p-values under iid or Markov backgrounds

### Takeaway
Exact motif-scan p-values are an old, standard technique. A dynamic program computes the exact null distribution of a PWM log-odds score at one position under a 0-order background, and some tools support order-k. The guarantee is narrow: the score at one fixed window is exact *if* the background model is true. Genome-wide FDR comes from Benjamini–Hochberg (BH) layered on top, with no check that the model is true.

### Cited findings
- FIMO converts log-odds scores to p-values with a dynamic programming algorithm that assumes a **zero-order background**. The default threshold is p < 1e-4, and p-values are converted to q-values by Benjamini–Hochberg. — [FIMO docs (MIT mirror)](https://web.mit.edu/meme/current/share/doc/fimo.html); [FIMO 5.0.5 docs](https://mitra.stanford.edu/kundaje/marinovg/oak/various/programs/meme-5.0.5/doc/fimo.html)
- The MEME suite's background is "normally a 0-order Markov model", but higher-order Markov backgrounds can be supplied with `-bfile` (this is stated for MEME, not confirmed for FIMO's p-value DP). — [MEME doc pages, via search summary](https://web.mit.edu/meme_v4.11.4/share/doc/fimo.html)
- MOODS (Korhonen et al., Bioinformatics 2009) implements the "well-known dynamic programming algorithm for converting P-values into score thresholds", with the background either user-specified or estimated from the sequence. A 2016 follow-up supports high-order PWMs, SNPs and indels. — [MOODS paper, PMC2778336](https://pmc.ncbi.nlm.nih.gov/articles/PMC2778336); [MOODS wiki](https://github.com/jhkorhonen/MOODS/wiki/Getting-Started)
- TFM-pvalue, wrapped in Python as pytfmpval, computes exact score↔p-value conversion for a PWM. — [Biostars pytfmpval](https://biostars.org/p/391895). **[recalled]** Touzet & Varré 2007 (Algorithms Mol Biol) use lazy, granularity-refined DP under an iid (Bernoulli) background. I did not confirm that it supports Markov backgrounds.
- motifcounter computes motif-hit count statistics under an **order-d Markov background**. It recommends low orders (d = 1–2) because very high-order backgrounds "might capture relevant TFBS signals" and mask real enrichment. — [motifcounter vignette](https://bioconductor.posit.co/packages/3.19/bioc/vignettes/motifcounter/inst/doc/motifcounter.html)
- The exact local-score distribution for Markovian sequences comes from a transition-matrix power (Hassenforder & Mercier 2007). This shows exact Markov-null computation reaches beyond single-window PWM scores. — [AISM 59(4):741](https://www.ism.ac.jp/editsec/aism/pdf/059_4_0741.pdf)
- Exact motif-count distributions under Markov assumptions have "high computational complexity", and overlapping occurrences are the main difficulty, so compound-Poisson approximations are used instead. — [BMC Bioinformatics 2021, s12859-021-04355-6](https://link.springer.com/article/10.1186/s12859-021-04355-6)
- Critique: "the choice of the background model is quite important"; Markov backgrounds capture CpG depletion. — [Frontiers in Genetics 2016](https://www.frontiersin.org/journals/genetics/articles/10.3389/fgene.2016.00024/full). Kiesel et al., quoted second-hand: "small enrichment factors can occur frequently in practice simply due to an imperfect background model." — [PubMed 21595908](https://pubmed.ncbi.nlm.nih.gov/21595908/)
- Practitioners often calibrate empirically rather than trusting the model. For example, BaMM estimated false positives by scanning the **reversed human genome**, and benchmarked against negatives simulated from a 2nd-order background. — [Siebert & Söding, NAR 2016](https://academic.oup.com/nar/article/44/13/6055/2457621); [BaMM NAR Genom Bioinf 2021](https://academic.oup.com/nargab/article/3/2/lqab026/6240708)

### Inferences
- The motif-scan "exact p-value" applies to a **fixed, pre-specified score function on a fixed window**. That makes it structurally the closest precedent to a per-window CRT: the PWM plays the role of the test statistic and the background is the null sampler. The CNN differs only in that its score is not decomposable, so the DP is unavailable and Monte Carlo resampling is required. Validity carries over; tractability does not.
- What the motif literature guarantees is *conditional on the background being true*. It is widely treated as a convention, not a verified property. The field's habit of empirical calibration (reversed or shuffled genome) shows practitioners do not trust nominal p-values in absolute terms.

### Gaps
- I did not verify whether FIMO's p-value DP itself uses higher-order backgrounds, or whether TFM-pvalue supports Markov backgrounds.

## 2. Genetic knockoffs: HMM and Markov-chain knockoffs

### Takeaway
Model-X knockoffs give exact finite-sample FDR when the joint law of X is known. Sesia–Sabatti–Candès made this practical for X following a discrete Markov chain or HMM, using a sequential sampler. The use case is **genotypes across individuals** (SNPs along a haplotype), with dataset-level selection. The one sequence-motif knockoff paper, KIMI, explicitly tried Markov-chain knockoffs on DNA bases and abandoned them.

### Cited findings
- "Gene Hunting with Knockoffs for Hidden Markov Models" constructs exact knockoff copies of discrete Markov chains and HMMs by a sequential procedure. — [arXiv 1706.04677](https://arxiv.org/pdf/1706.04677). **[recalled]** It was published in Biometrika 2019 and implemented in the SNPknock package. The HMM is fastPHASE-style, fit to genotypes, and the FDR guarantee is exact only if the fitted HMM is the true law, with robustness shown empirically.
- "Relaxing the Assumptions of Knockoffs by Conditioning" builds conditional knockoffs, including for Markov chains, that need only a model *family* rather than known parameters. — [arXiv 1903.02806](https://arxiv.org/pdf/1903.02806). **[recalled]** The authors are Huang & Janson, Ann. Statist. 2020. This is relevant if the thesis wants to estimate Markov transition probabilities without losing exactness.
- **[recalled]** KnockoffZoom (Sesia, Katsevich, Bates, Candès, Sabatti, PNAS 2020) runs multi-resolution HMM knockoffs on UK Biobank genotypes. Selection is of genomic *regions* across a population, not per individual.
- KIMI (Bioinformatics 2021) is model-X knockoffs for motif identification from k-mer frequencies, with an L1-logistic selector. The authors tried Markov knockoff contigs (Sesia et al.) but found the resulting k-mer frequencies "are not exchangeable in distribution" with the originals. They instead assume the k-mer frequency matrix is multivariate normal, conceding the "normality assumption may be mis-specified and the constructed knockoff variables are not ideal." KIMI requires p < n, is dataset-level, and uses no deep model. A Markov order-3 model is used only for simulation. — [KIMI, Bioinformatics 37(6):759](https://academic.oup.com/bioinformatics/article/37/6/759/5942973)

### Inferences
- The knockoff case that works rests on a Markov law **across positions within one individual's genome**, with exchangeability **across n individuals**. That is the same factorization a thesis would invoke for bases within a sequence. But FDR there is an expectation over a dataset of n i.i.d. samples. A per-sequence claim has n = 1, so knockoff FDR does not transfer directly. A per-window CRT p-value (type-I per window), followed by BH across windows of one sequence, is the natural local analogue. BH validity would then need PRDS or BY, because windows share the sequence.
- KIMI's failure is instructive. Markov knockoffs at the base level are valid for bases. They broke only because KIMI's *features* were derived k-mer counts. A CNN taking one-hot bases as input avoids that problem.

### Gaps
- I found no paper that applies HMM or Markov knockoffs to nucleotide identities as features of a sequence model.

## 3. CRT, knockoffs or permutation tests with a Markov or shuffle sampler on trained DNA deep models

### Takeaway
I found **no** paper that runs a CRT, knockoffs or a permutation test with a Markov or dinucleotide-shuffle sampler on a trained DNA CNN, per sequence, with a stated type-I or FDR guarantee. Deep-knockoff work in genomics (DeepPINK, DeepLINK) is tabular and dataset-level. Shuffles are used routinely in DNA interpretability, but as references or baselines, not as calibrated nulls.

### Cited findings
- DeepPINK (Lu, Fan, Lv, Noble, NeurIPS 2018) is a knockoff-paired-input DNN for FDR-controlled selection that assumes Gaussian features. — [arXiv 1809.01185](https://arxiv.org/pdf/1809.01185)
- DeepLINK (Zhu, Fan, Kong, Lv, Sun, PNAS 2021) uses an autoencoder knockoff generator with an MLP selector under latent-factor models. Its applications are microbiome and single-cell data, not DNA sequence. — [PMC8433583](https://pmc.ncbi.nlm.nih.gov/articles/PMC8433583); [NSF PAR](https://par.nsf.gov/biblio/10307954)
- Dinucleotide-preserving shuffles are an established null in RNA and ncRNA work. Their quality matters: dinucleotide shuffles of alignments "greatly inflated" false positives compared with mononucleotide shuffling in one ncRNA-detection setting. — [BMC Bioinf 8:33](https://link.springer.com/article/10.1186/1471-2105-8-33); [SISSIz, BMC Bioinf 9:248](https://link.springer.com/article/10.1186/1471-2105-9-248)
- Han & Kim (arXiv 2609.16271, 2026) show attribution "recovery rates" across 268 UniBind TFs have chance levels from 0.0118 to 0.0427, set analytically by motif and window length. Perturbation curves can be non-monotone, so their AUC is not a valid faithfulness measure. — [arXiv abs](https://arxiv.org/abs/2609.16271). Conflict: a search snippet said the paper uses dinucleotide-preserving shuffled controls and treats its score as an effect size, but the abstract page describes only an analytic chance level. In either case, no test guarantee is claimed.
- Reyero-Lobo et al. (arXiv 2610.11388, Oct 2026) give sequential model-X CI tests with e-values and bound type-I inflation when the model-X kernel is *estimated*: P(sup E ≥ 1/α) − α ≤ Σδ_n + α[exp(λΣa_n b_n) − 1]. The test is global, and applied to HIV mutations, not DNA sequence models. — [arXiv 2610.11388](https://arxiv.org/html/2610.11388)
- Other DNA-attribution "significance" work uses shuffled-versus-real Wilcoxon comparisons or Poisson tests against background reads. These are aggregate validation checks, not per-sequence error control. — [miRNA attribution alignment, PMC10045089](https://pmc.ncbi.nlm.nih.gov/articles/PMC10045089/); [differential DL, PMC7049686](https://pmc.ncbi.nlm.nih.gov/articles/PMC7049686/)

### Inferences
- The gap appears real, though the search was limited. A per-window CRT for a trained CNN uses a fixed fitted model, so no refits are needed. It resamples window W from P(X_W | X_{−W}), which under an order-k Markov law depends only on the k flanking bases on each side and can be sampled exactly by forward-backward. It is valid for H0: Y ⊥ X_W | X_{−W} *under the assumed law*. This is a straightforward combination of known parts (Sesia-style Markov conditionals plus the CRT), which suits a thesis. The novelty is the local, per-sequence framing and the honest calibration study.
- Caveat on the estimand: with a fixed model and a single sequence, "Y ⊥ X_W" must be redefined. Options are the model output's deviation from its conditional null distribution, or a loss at the observed label. Otherwise the test asks whether the CNN's prediction is sensitive to W relative to Markov-plausible replacements, a statement about the model, not about biology.

### Gaps
- DFIM (**[recalled]** Greenside et al., Bioinformatics 2018) scores interactions via DeepLIFT under mutation, with no null or guarantee. Not re-verified here.
- I did not check 2024–2026 bioRxiv thoroughly for "CRT DNA" or "knockoff genomic language model". One extended search found nothing.

## 4. How far real genomic background departs from low-order Markov

### Takeaway
There is qualitative evidence that 0- and 1-order Markov models miss real structure: repeats, CpG and Alu, long-range correlations, and mixtures of competing contexts. I found **no** study quantifying the resulting miscalibration of motif or CRT p-values. Indirect evidence: the field calibrates empirically, and choosing a different shuffle null changed false-positive rates substantially in ncRNA detection.

### Cited findings
- "Zeroth- and first-order Markovian models of simple repeats fail to fully account for 'linguistic' features of non-coding DNA." — [arXiv q-bio/0502009](https://arxiv.org/pdf/q-bio/0502009)
- Complete genomes are better represented by *multiple competing* finite-context models than by one Markov model, which implies non-stationarity. — [PLoS ONE 2011](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0021588)
- Long-range correlations and the effect of tandem repeats on scaling are documented in DNA. — [arXiv 1411.2761](https://arxiv.org/pdf/1411.2761); [arXiv q-bio/0510042](https://arxiv.org/pdf/q-bio/0510042)
- Alu elements (about 280 bp, GC- and CpG-rich) distort CpG-based statistics. — [BMC Genomics 12(S2):S10](https://link.springer.com/article/10.1186/1471-2164-12-S2-S10)
- Higher-order backgrounds can absorb real motif signal (motifcounter), and high-order models overfit, since parameters grow as 4^k (BaMM). — [motifcounter](https://bioconductor.posit.co/packages/3.19/bioc/vignettes/motifcounter/inst/doc/motifcounter.html); [BaMM 2021](https://academic.oup.com/nargab/article/3/2/lqab026/6240708)

### Inferences
- Tension for the thesis: an exact CRT is exact for the *assumed* law. Raising k makes the law more realistic, but the conditional P(X_W | X_{−W}) then also encodes motif-like structure, which reduces power and changes the null being tested. A low-order law is a convention, not a fact about DNA. The defensible framing: "exact under a stated reference law, with sensitivity to k reported", plus an empirical calibration check on real non-functional sequence (e.g., genomic background windows with no signal) to quantify type-I inflation. That check would itself be a contribution, since none was found.
- Under misspecification, the robustness bounds for estimated model-X kernels (Reyero-Lobo et al. 2026; also Berrett et al. 2020 **[recalled]**) give the formal language for "approximately valid" claims.

### Gaps
- I found no quantitative study of FIMO-style p-value calibration on real genomic background, nor of how much type-I error rises when a CRT uses an order-k Markov null on real DNA.
