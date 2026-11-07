# Ground truth for validating local (per-sequence) feature/motif selection in regulatory-genomics sequence-to-function models

Researcher notes, compiled 25 Oct 2026. Scope: the kinds of "truth" a per-sequence selection method (for one input sequence x, choose the positions or motif instances the prediction depends on, with error control) can be validated against. Out of scope here because a companion researcher covered them: CREME, GIA, SQUID, tangermeme, TF-MoDISco/Fi-NeMo thresholds. They are cited below only where needed.

Primary sources were read in full text (Europe PMC / PMC / PMLR / arXiv PDF), and benchmark code and data files were inspected directly where noted ("verified in file"). Numbers taken from a WebFetch summary of a page, not from my own reading of the full text, are marked "(page summary)".

---

## 0. The student's concern: do we have notions of ground truth to build an interpretability method against?

### Takeaway
Yes, but there are three different notions of truth, and each answers a different question. (1) **Synthetic planted motifs** give exact, per-sequence truth about the *data-generating* function, but only for simplified regulatory logic. (2) **Wet-lab perturbation data** give noisy, signed, per-sequence effects of *specific edits* in a *specific assay and cell type*. Saturation-mutagenesis MPRAs are the most local of these. (3) **Model-relative truth** is the exact perturbation effect on the fitted model f̂ (exhaustive in-silico mutagenesis or motif ablation). It is noise-free and per-sequence, and it says nothing about biology by itself. The field already uses all three, but treats them as separate validation layers. No source I found claims that one notion settles "what the sequence really depends on."

### Cited Findings
**Prior framing of the problem in the field**
- Prakash, Shrikumar & Kundaje (PMLR 165, 2022) name two benchmark strategies. The first is fully simulated data, which "has the benefit of being in control of the exact ground truth," but "these simulated sequences may lack the complexity of real biological data." The second is real data checked with heuristics: known motifs, footprints, SNP enrichment, empirical validation. There, "it is difficult to gain insight into which patterns the interpretation method may have missed, as the ground truth is not definitively known." — [Prakash et al. 2022, PMLR](https://proceedings.mlr.press/v165/prakash22a/prakash22a.pdf)
- Koo & Ploenzke (Nat Mach Intell, Feb 2021) found that "CNNs with better test performance do not necessarily imply more interpretable representations with attribution methods." They describe learning good first-layer motifs as "a necessary (but not sufficient) quality control" for trusting attributions (page summary). — [Koo & Ploenzke 2021 (PMC8315445)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8315445)
- A 2023 review (Novakovsky, Dexter, Libbrecht, Wasserman, Mostafavi; Nat Rev Genet) says "the underlying basis on which predictive models make such predictions is often unknown." It surveys interpretation approaches with "their underlying assumptions and limitations in the context of typical high-throughput biological datasets." — [Novakovsky et al. 2023](https://doi.org/10.1038/s41576-022-00532-2)

**Catalog of ground-truth types (details and sources in sections 1–3)**

| Type | What exactly is "true" | Local or population | How prior work scored attributions against it | Main caveat | Example data |
|---|---|---|---|---|---|
| Synthetic planted motifs | Positions, and the generating PWM probabilities, of the embedded motif instances in each sequence, plus the label logic | Local by construction | Per-sequence position-level AUROC/AUPR ([Koo & Ploenzke 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8315445); [Majdandzic et al. 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10169356/)); per-sequence auPRC ([Prakash 2022](https://proceedings.mlr.press/v165/prakash22a/prakash22a.pdf)); total importance per motif match ([DeepLIFT](https://arxiv.org/abs/1704.02685)); interaction-score distributions ([DFIM](https://pmc.ncbi.nlm.nih.gov/articles/PMC6129272/)) | Simplified logic. "Planted" is not the same as "relevant at x" | Koo Task 1/3 HDF5 ([Zenodo 4301062](https://doi.org/10.5281/zenodo.4301062)); [DFIM data](https://github.com/kundajelab/dfim) |
| Saturation-mutagenesis MPRA | Signed log2 effect of each SNV / 1-bp deletion within one element, in one cell line, on a plasmid | Local, per position (single edits) | Pearson/Spearman correlation and AUROC against measured effects ([Kircher 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6687891/); [CAGI5](https://pmc.ncbi.nlm.nih.gov/articles/PMC6879779/)) | Plasmid context. Replicate r ranges 0.55–0.99 by element | [satMutMPRA portal](https://kircherlab.bihealth.org/satMutMPRA/) (TSV/CSV) |
| Motif-instance mutation libraries | Activity change when one instance, or all instances, of a motif are scrambled | Local, per motif instance | PCC of predicted vs. measured instance importance ([DeepSTARR bioRxiv](https://www.biorxiv.org/content/10.1101/2021.10.05.463203v1.full)) | Episomal assay; specific cell lines | GEO GSE183939 ([DeepSTARR GitHub](https://github.com/bernardo-de-almeida/DeepSTARR)) |
| Tiling MPRA (Sharpr) | Inferred activating or repressive nucleotides | Local, but inferred by a model | — | Deconvolution model in the loop | [Ernst 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC5125825/) |
| CRISPR in-situ edits | Effect of edits in native chromatin | Local; few loci | Compared against model-predicted profiles ([BPNet](https://pmc.ncbi.nlm.nih.gov/articles/PMC8812996/)) | Very low throughput | BPNet GEO GSE137193 |
| Footprints, ChIP-exo/nexus | Where a protein is bound (occupancy), not the effect on a readout | Local, per position | Overlap/enrichment sanity checks ([Prakash 2022](https://proceedings.mlr.press/v165/prakash22a/prakash22a.pdf)) | TFs with short residence time leave no footprint ([Sung 2014](https://pubmed.ncbi.nlm.nih.gov/25242143/)) | [Vierstra footprints](https://pmc.ncbi.nlm.nih.gov/articles/PMC7410829/) |
| QTL / allele-specific binding | Effect of one natural allele at one site | Per variant; discovered through population association or allelic imbalance | Variant-scoring AUROC/Pearson ([DART-Eval](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12340898/); [GkmExplain](https://pmc.ncbi.nlm.nih.gov/articles/PMC6612808/)) | Uncertainty about which variant is causal; sparse | [Degner 2012](https://pubmed.ncbi.nlm.nih.gov/22307276/); [ADASTRA](https://doi.org/10.1038/s41467-021-23007-0) |
| Model-relative | Exact f̂(x′) − f̂(x) for any edit set x → x′ | Local, exact | Correlation of gradients with ISM ([TISM](https://pmc.ncbi.nlm.nih.gov/articles/PMC11404212/)) | Silent about biology; depends on the chosen replacement/"removal" distribution | Any trained model, e.g. DeepSTARR ([Zenodo 5502060](https://doi.org/10.5281/zenodo.5502060)) |

**Jargon for a statistics reader**
- **MPRA (massively parallel reporter assay):** "a set of high-throughput technologies that measure the functional effects of thousands of sequences/variants on gene regulatory activity." — [MPRAbase, Genome Res 2025](https://genome.cshlp.org/content/early/2025/04/22/gr.280387.124.abstract). Mechanically, activity is "measured by counting associated 3′ UTR tags in amplicons derived from RNA … normalized to its relative abundance in plasmid DNA." It is a barcode count ratio, RNA over DNA. — [Kircher et al. 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6687891/)
- **STARR-seq:** a reporter assay that "tests candidates downstream of a core-promoter as comprehensive libraries with hundreds of millions of fragments." The candidate sequence transcribes itself, so its abundance in RNA is the readout. — [Muerdter et al. 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC5793997/)
- **Saturation mutagenesis:** "a library of mutations covering all bases" of an element, with each variant's effect measured. — [Shigaki et al. 2019 (CAGI5)](https://pubmed.ncbi.nlm.nih.gov/31106481/)
- **Footprint:** "In vivo binding of regulatory factors shields bound DNA elements from nuclease attack, giving rise to protected single-nucleotide-resolution DNA 'footprints'." — [Vierstra et al. 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7410829/)
- **QTL (quantitative trait locus):** a genetic variant associated with a quantitative molecular trait across individuals. A dsQTL is a site where "DNase-seq read depth correlated significantly with genotype at a nearby" SNP or indel ([Degner 2012](https://pubmed.ncbi.nlm.nih.gov/22307276/)). caQTLs and dsQTLs are "genetic variants with statistically significant effects on chromatin accessibility" from ATAC-seq and DNase-seq respectively ([DART-Eval](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12340898/)). bQTLs are QTLs for TF binding ([Tehranchi 2016](https://pubmed.ncbi.nlm.nih.gov/27087447/)).
- **Allele-specific binding (ASB):** "Differential transcription factor binding in heterozygous genomic loci." — [Abramov et al. 2021](https://doi.org/10.1038/s41467-021-23007-0)
- **ChIP-exo / ChIP-nexus:** ChIP-seq whose "resolution … is improved by adding an exonuclease digestion step." — [Avsec et al. 2021 (BPNet)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8812996/)
- **ISM (in-silico mutagenesis):** "measuring the impact of exhaustive perturbations of every nucleotide in an input sequence on the output." — [GkmExplain 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6612808/)

### Inferences
- These are three estimands, not three noisy views of one truth:
  - **S\*_gen(x):** the support of the generating function at x. Only knowable in simulation.
  - **Δ_assay(x, e):** the measured effect of edit e on x, in an assay and cell type.
  - **Δ_f̂(x, e):** the exact effect of e on the fitted model.
- A selection procedure with a statistical guarantee is most naturally stated for an f̂-defined estimand. That is consistent with how the thesis already defines value functions on fitted models. Synthetic data are the only place where Δ_f̂ can be checked against S\*_gen, which tests whether the model learned the generating logic. Experimental data then test whether Δ_f̂ is biologically meaningful.
- So the student's worry is partly a framing problem. "Ground truth" has to be indexed by estimand. Once it is, each layer exists and is usable, with known failure modes.

### Gaps
- I found no source that formally reconciles the three notions or proposes a single estimand spanning them.
- The Novakovsky 2023 review was read at abstract level only.

---

## 1. Synthetic planted-motif benchmarks: mask definitions, decoys and negatives, and designs with sequence-dependent ("local") truth

### Takeaway
Every major synthetic benchmark records exactly where motifs were planted, and several are downloadable with that record. The scoring mask is almost always "all planted positions of label-associated motifs," thresholded by PWM information content. It is not "the positions the label actually depends on at x." Several designs do have sequence-dependent relevance, but it is scored loosely, if at all:
- AND logic: DeepLIFT's TAL1+GATA1 task, GkmExplain, and DFIM's ELF1+SIX5 task.
- Redundant copies: 0–3 or 1–5 instances per sequence.
- Label-independent decoys: DFIM's AP1/TAL1.

None of the sources I checked has a quantitative min (limiting-factor) or saturating output, a repressor (negative-sign) motif, or a position-level mask that changes with context.

### Cited Findings
**DeepLIFT simulation (Shrikumar, Greenside, Kundaje; ICML 2017)**
- **Design.** 200-bp backgrounds with ACGT probabilities 0.3/0.2/0.2/0.3. "0-3 instances of a given motif were inserted at random non-overlapping positions." Motifs are GATA1 and TAL1 from Kheradpour & Kellis 2014. — [DeepLIFT arXiv:1704.02685](https://arxiv.org/abs/1704.02685)
- **Labels.** Three binary tasks: "both GATA1 and TAL1 present" (an AND gate), "GATA1 present," and "TAL1 present." A quarter of sequences each are labeled 111, 010, 001 and 000. The paper indexes the tasks 1–3 in Methods but 0–2 in Results. — [DeepLIFT](https://arxiv.org/abs/1704.02685)
- **Scoring, no binary mask.** "We found the top 5 matches (as ranked by their log-odds score) to each motif for each sequence from the test set, as well as the total importance allocated to the match … for each task." Chance matches in background, and the sequences with no motifs (plotted as black dots), are included in the evaluation. The desiderata are: high scores for TAL1 on the TAL1 task and low on the GATA1 task; scores increasing with match strength; and, for the AND task, "higher scores on sequences containing both kinds of motifs vs. sequences containing only one kind (revealing cooperativity)." — [DeepLIFT](https://arxiv.org/abs/1704.02685)
- **Known false positives.** Gradient×input and guided backprop×input "can assign unduly high importance to weak motif matches." — [DeepLIFT](https://arxiv.org/abs/1704.02685)
- **Download (verified in file).** The DeepLIFT repo's genomics example loads `sequences.simdata.gz`: 0.63 MB, 8,000 rows, columns `seqName, sequence, embeddings, task1, task2, task3`. Sequences with no motifs are named `empty_*` and have empty `embeddings`. A pre-trained Keras model (JSON + weights) is hosted alongside. — [notebook](https://github.com/kundajelab/deeplift/blob/master/examples/genomics/genomics_simulation.ipynb); [data file](https://raw.githubusercontent.com/AvantiShri/model_storage/db919b12f750e5844402153233249bb3d24e9e9a/deeplift/genomics/sequences.simdata.gz)

**GkmExplain (Shrikumar, Prakash, Kundaje; Bioinformatics/ISMB 2019)**
- **Design.** 8,000 sequences of 200 bp with the same background. "0-3 instances of TAL1_known1 … and GATA1_disc1" are planted. Labels are "25% of sequences contained both … and were labeled +1," and sequences with one or neither motif are −1. This is a pure AND gate. — [GkmExplain (PMC6612808)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6612808/)
- **Redundancy example.** On a sequence with "two instances of TATA," "ISM does not clearly highlight them." — [GkmExplain](https://pmc.ncbi.nlm.nih.gov/articles/PMC6612808/)

**DFIM, Deep Feature Interaction Maps (Greenside, Shimko, Fordyce, Kundaje; Bioinformatics/ECCB 2018). The closest existing design to the thesis's gate**
- **Design.** "We simulated 60 K random DNA sequences (0.46 G/C frequency) of length 200 bp." Set 1 has 1–2 ELF1 instances, Set 2 has 1–2 SIX5, and Set 3 has 1–2 of both. "We further independently embedded 0 or 1 instances of the AP1 and TAL1 motifs in a random subset of sequences across all 3 sets." Ground-truth interactions are ELF1–SIX5 pairs only. — [DFIM (PMC6129272)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6129272/)
- **Downloadable ground truth (verified in file).** `data/embedded_motif_interaction_data/` in the repo contains:
  - `simulation_metadata.txt.gz` (4.2 MB), with columns `seqName, sequence, embeddings`. Embeddings look like `pos-122_ELF1_known2-AACCCGGAAGTG,pos-45_TAL1_known1-CTGAACAGATGGTCGG`, i.e. exact start position plus planted string.
  - `labels.txt.gz`, with columns `task1, task2, task3`.
  - A trained model: weights 0.75 MB plus architecture JSON.
  - Class labels decoded: 20,000 sequences per class. cls1 (ELF1) is labeled (1,0,0), cls2 (SIX5) is (0,1,0), and cls3 (both) is (1,1,1). So task3 is the AND gate.
  - The decoys are balanced across classes. For cls1, the counts are ELF1 only 5,048; +AP1 5,043; +AP1+TAL1 4,971; +TAL1 4,938, and cls2/cls3 are similar. AP1 and TAL1 are therefore genuinely label-independent.
  - The GitHub API lists no license for the repo.
  - — [kundajelab/dfim](https://github.com/kundajelab/dfim)
- **Scoring.** The feature interaction score (FIS) is "the change in the importance score of the target feature when the source feature is perturbed." It was computed "for all pairs of embedded motif instances … for all sequences in the positive class." "Only pairs of SIX5 and ELF1 motifs (positive control) showed strong FIS … compared to all other pairs of motifs (negative controls)." — [DFIM](https://pmc.ncbi.nlm.nih.gov/articles/PMC6129272/)

**Koo & Eddy 2019 (PLoS Comp Bio) and Koo & Ploenzke 2021 (Nat Mach Intell): "Task 1" and "Task 3"**
- **Task 1.** 200-nt sequences "implanted with 1 to 5 known TF motifs, randomly selected with replacement from a pool of 12." The design assumes "the only important pattern for a given TF binding event is the presence of a PWM-like motif." — [Koo & Eddy 2019 (PMC6941814)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6941814/). The pool is Arid3, CEBPB, FOSL1, Gabpa, MAFK, MAX, MEF2A, NFYB, SP1, SRF, STAT1, YY1, with 30,000 sequences and multitask presence labels (page summary). — [Koo & Ploenzke 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8315445)
- **Task 3.** A binary "billboard" design. Positives contain "3 to 5 'core motifs,' randomly selected with replacement from a pool of 10 position frequency matrices," i.e. forward and reverse-complement CEBPB, Gabpa, MAX, SP1, YY1. Negatives follow "the same steps with the exception that the pool of motifs include 100 non-overlapping 'background motifs' from the JASPAR database." There are 20,000 sequences with a 0.7/0.1/0.2 split. — [Majdandzic et al. 2023 (PMC10169356)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10169356/)
- **Files (verified).** [Zenodo 4301062](https://doi.org/10.5281/zenodo.4301062) holds a 37.7 MB zip, "p-koo/exponential_activations: beta," dated 2020-12-02. Zenodo lists the license as "other-open," and the zip contains an MIT LICENSE.md. Inside:
  - `data/synthetic_code_dataset.h5` (17.1 MB, Task 3): keys `X_{train,valid,test}` with shapes (14000|2000|4000, 4, 200); `Y_*` (N,1); and `model_{train,valid,test}` (N, 4, 200). `model_*` is the per-sequence generating probability matrix, and the mask is derived from it.
  - `data/synthetic_dataset.h5` (17.8 MB, Task 1): 21000/3000/6000 × 4 × 200, with Y having 12 columns.
  - In Task 3's test set, 2,015 of 4,000 sequences are positive. Both positives and negatives average about 40 positions with information above 0.1 bits (my computation from the file). Negatives therefore carry planted (background) motifs with recorded masks too.
- **Mask and metric, Koo & Ploenzke code (verified in file).**
  - `helper.interpretability_performance` computes per-position information from `model_test`, sets `label[gt_info > 0.01] = 1` with all other positions negative, and computes AUROC/AUPR per sequence.
  - The script selects `true_index = np.where(y_test[:,0] == 1)[0]` and uses `X_model = test_model[true_index][:500]`. Only the first 500 positive-class test sequences are scored.
  - The per-position score is the sum over nucleotides of attribution × input (page summary).
  - — [helper.py](https://github.com/p-koo/exponential_activations/blob/master/code/helper.py); [task3_step2_attribution_scores.py](https://github.com/p-koo/exponential_activations/blob/master/code/task3_step2_attribution_scores.py)
- **Methods compared.** Saliency, ISM (L2 norm across nucleotides), integrated gradients, and DeepSHAP. ISM "consistently yields the most reliable attribution maps" (page summary). — [Koo & Ploenzke 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8315445)

**Majdandzic, Rajesh & Koo 2023 (Genome Biology 24:109, May 2023): gradient-correction benchmark on Task 3**
- **Mask.** Positive positions satisfy "I_i > 0.1 (i.e., 5% of maximum information content for DNA), while positions with an information content of zero are given a negative label." AUROC and AUPR are "calculated for each sequence." — [Majdandzic 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10169356/)
- **Code (verified).** `interpretability_performance(..., threshold=0.1)` scores only positions with `(gt_info > threshold) | (gt_info == min(gt_info))`. Low-information motif positions (0 < I ≤ 0.1) are dropped from the evaluation rather than counted as negatives. The evaluation again uses the first 500 positive test sequences. A third metric is cosine similarity against the ground-truth probability matrix minus 0.25, which "zero[es] out" non-informative positions. — [helper.py](https://github.com/p-koo/GradientCorrection/blob/main/gradient_correction/helper.py); [2_evaluate_models_synthetic.py](https://github.com/p-koo/GradientCorrection/blob/main/analysis/2_evaluate_models_synthetic.py); [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10169356/)
- **Threshold difference within one lab.** Koo & Ploenzke 2021 uses I > 0.01 with no exclusion zone. Majdandzic 2023 uses I > 0.1 and excludes intermediate positions. (Code above.)
- **Robustness and code.** 50 random initializations per model. Code is MIT-licensed at [p-koo/GradientCorrection](https://github.com/p-koo/GradientCorrection). — [Majdandzic 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10169356/)

**Prakash, Shrikumar & Kundaje 2022 (PMLR 165, MLCB 2021 meeting): "more realistic" semi-synthetic data**
- **Pipeline.**
  1. HOMER motif discovery on real positives versus negatives.
  2. FIMO motif calling on both sets.
  3. Dinucleotide-shuffle each positive and "implant the motif instances … at their known original locations (note that the original motif match is implanted; we do not sample from the motif PWM)."
  4. Build negatives the same way from shuffled positives, implanting the negatives' own FIMO hits. Negatives therefore contain real motif instances.
  5. Re-run FIMO "to identify both the implanted motif instances as well as motif instances that may have spontaneously appeared."
  - The sequences are 400 bp around ENCODE IDR peak summits in A549, GM12878, H1ESC, HepG2 and K562 (ENCSR149XIL, ENCSR000ELW, ENCSR794OFW, ENCSR000EOT, ENCSR000EMT).
  - — [Prakash 2022](https://proceedings.mlr.press/v165/prakash22a/prakash22a.pdf)
- **Mask and metric.** "Positive positions [are] sequence positions that fall within motif regions and negative positions … within nonmotif regions." The per-sequence auPRC ranks positions "in descending order of the absolute value of importance." The evaluation is "limited … to the top 10K training-set sequences … most strongly predicted as positive," and reports mean auPRC ± SEM. — [Prakash 2022](https://proceedings.mlr.press/v165/prakash22a/prakash22a.pdf)
- **Findings.** "DeepLIFT and ISM tend to perform best and second-best." GradCAM does poorly. IG sometimes trails gradient×input. DeepLIFT with dinucleotide-shuffled references beats all-zero or base-frequency references, improving up to about 20 references. — [Prakash 2022](https://proceedings.mlr.press/v165/prakash22a/prakash22a.pdf)
- **Repo (verified via GitHub API/tree).** [kundajelab/interpret-benchmark](https://github.com/kundajelab/interpret-benchmark) contains `data_simulation/` scripts and `measure-motif-importance.ipynb` notebooks. I found no pre-built dataset files in the tree, so the data must be regenerated from ENCODE. Last push was Nov 2021, and no license is listed.

**simdna (Kundaje lab), a generator with recorded ground truth**
- Ships ENCODE and HOCOMOCO motif sets and can load HOMER/JASPAR motifs. Embedders include `XOREmbedder`, `PairEmbeddable` (fixed-spacing grammars), `RepeatedEmbedder` and `RandomSubsetOfEmbedders`. Quantity generators include Poisson and zero-inflated. Backgrounds can be zero-order, first-order, or dinucleotide shuffles. The `.simdata` output "encodes all the sequences, as well as all of the embedded motifs in each sequence." MIT license (page summary). — [kundajelab/simdna](https://github.com/kundajelab/simdna)
- Last push was 2023-02-22 (verified via GitHub API). — [kundajelab/simdna](https://github.com/kundajelab/simdna)

**Newer benchmark suites (2024–2026)**
- **DART-Eval** (Kundaje lab; NeurIPS 2024 Datasets & Benchmarks). Its motif task uses "1,443 consensus TF motif sequences" from HOCOMOCO v12, each placed at the center of a neutral background against a shuffled-motif control, with 200 pairs per motif and 577,400 sequences. This tests *model sensitivity* paired by construction; it does not score attribution maps. Data are on Synapse (syn59522070). — [DART-Eval (NeurIPS)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/71998bfc3217ffe1cca1ee084dfadadd-Abstract.html)
- **Sun et al. (bioRxiv, Jan 2025).** A benchmark of sequence-to-function models "and interpretability methods" on 369 ATAC-seq datasets. From the abstract, it reports that "Negative samples derived from naturally inactive regions outperform synthetic sequences." The abstract shows no planted-motif ground truth; I did not read the full text. — [Sun et al. 2025](https://www.biorxiv.org/content/10.1101/2025.01.06.631405v1)
- **Reynolds & Pan (PLoS Comp Bio, 2025).** Genotype data rather than DNA sequence, but the evaluation design carries over. "Attribution recall was assessed using synthetic spike-in variants with known additive, dominant, recessive, and epistatic effects." "Attribution precision estimated specificity using an equal number of null decoy variants." Stability was measured across an ensemble. — [Reynolds & Pan 2025](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013784)
- **XAI-Units (arXiv, June 2025), generic ML, search snippet only.** Pairs each synthetic dataset with a handcrafted, perfectly accurate model, because synthetic "ground truth" attributions may not match an imperfect trained model. — [XAI-Units](https://arxiv.org/abs/2506.01059)

### Inferences
- **Mask ≠ local support.** In every position-level benchmark above, the mask is "planted positions of label-associated motifs." Under the presence ("OR") labels of Task 1/3 or DeepLIFT's single-motif tasks, a sequence with three copies has all three in the mask. Yet the local value function at x may give each copy near-zero marginal effect (section 4). So these masks reward a "union of all relevant instances" notion. They penalize a "minimal sufficient set" notion. The thesis must declare which one its error guarantee targets.
- **Decoys differ in quality.**
  - DFIM's AP1/TAL1 are balanced across classes in the file, so they are clean label-independent decoys (true nulls).
  - Koo Task 3's background motifs appear only in negatives, per the paper's description. They may then be legitimate *evidence for the negative class*, not irrelevant decoys. A model that uses them is not wrong. They are unsafe as "null" features for FDR estimation unless one first verifies that core and background pools are disjoint and that f̂ ignores the background motifs.
  - Prakash's negatives contain real motif instances, so "negative class" does not mean "no motif."
- **Negatives are mostly skipped.** The Koo-lab code scores only positive-class sequences. Prakash scores only top-predicted positives (from the training set). A local selector must also say something about x with a "low" prediction, where the support of an AND gate is defined by an *absence*, which cannot be selected as a set of positions. This is the same definitional issue as the thesis's gate design y = x1·x2·1{x3>0} when the gate is off.
- **Direct mappings to the thesis's simulated designs, using existing downloadable data:**
  - Gate/AND: DFIM task3, DeepLIFT's "both" task, GkmExplain.
  - Redundancy (max/OR): DeepLIFT's single-motif tasks with 0–3 copies; Task 1/3 with 1–5 or 3–5 motifs.
  - Decoys: DFIM AP1/TAL1.
  - Spacing grammar: simdna `PairEmbeddable` (no existing benchmark found).
  - XOR: simdna `XOREmbedder` (no benchmark found).
  - Limiting-factor (min), saturation, and repressor designs would be new.
- **Scoring options.** Since the generating matrices (`model_*`) and embedding strings are stored, these files support more than position-level AUROC. They also support instance-level scoring: each planted instance is one candidate group, which is the natural unit for correlated positions.

### Gaps
- I could not confirm whether Task 3's negative-class pool excludes the five core motifs. The paper wording ("the pool of motifs include 100 … background motifs") is ambiguous.
- Not checked: the Tseng et al. 2020 Fourier-prior simulations, TF-MoDISco paper simulations, and the BaGGLS arXiv preprint (Nov 2025), which a search snippet described as using synthetic DNA with known motifs.
- I found no published synthetic genomics benchmark that scores *context-dependent masks*, where the same instance is relevant in one sequence and not another and the mask reflects this, or that targets FDR-controlled selection. Absence in the sources checked is not proof that none exists.
- I did not verify that the DeepLIFT-repo `sequences.simdata.gz` is the exact dataset in the paper. Its size (8,000) matches GkmExplain's description of the same design.

---

## 2. Experimental per-sequence ground truth: saturation-mutagenesis MPRA first, then other assays

### Takeaway
The only experimental data that give **per-position, signed effects for a single sequence** at scale are saturation-mutagenesis MPRAs: Kircher et al. 2019 and the CAGI5 challenge built on it, 21 elements and 31,243 variant effects. They are public, as TSV/CSV files with log2 effect, p-value and tag counts. Their limits:
- They measure *single* edits in a WT background, so they cannot see redundancy.
- They come from plasmids in specific cell lines.
- Replicate concordance ranges from about 0.55 to 0.99 Pearson across elements.
- The best CAGI5 predictors reached only about 0.3–0.45 correlation.

Motif-instance mutation libraries such as DeepSTARR's oligo screens add exactly what saturation mutagenesis lacks: single-instance versus all-instance knockouts, which make redundancy measurable. Footprints, ChIP-nexus and QTLs are local but measure occupancy or single natural variants, not the effect of position sets.

### Cited Findings
**Kircher et al. 2019 (Nat Commun 10:3583; accepted 15 Jul 2019), the core per-position dataset**
- **Elements.** "Ten promoters (of TERT, LDLR, HBB, HBG, HNF4A, MSMB, PKLR, F9, FOXE1, and GP1BB) and ten enhancers (of SORT1, ZRS, BCL11A, IRF4, IRF6, MYC (2×), RET, TCF7L2, and ZFAND3)," plus the ultraconserved UC88. They span "28–73% in GC content and 187–601 base pairs." Length was capped at 600 bp for subassembly. — [Kircher 2019 (PMC6687891)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6687891/)
- **Assay.** Error-prone PCR at "less than 1 change per 100 bp." Plasmid MPRA in pGL4.11 (promoters) or pGL4.23 (enhancers, minimal promoter) backbones with a luciferase reporter and random tags in the 3′ UTR. Libraries of 50k–2M are "transfected as plasmids," with "three independent cultures (replicates)" per library. — [Kircher 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6687891/)
- **Estimation.** Variant effects are regression coefficients: "we fit a multiple linear regression model of log2(RNAj) ~ log2(DNAj) + N + offset … and report the coefficients of N as effects for each variant." Because each molecule carries a haplotype with a few mutations, the reported effect is an additive single-variant effect in an otherwise near-WT background. — [Kircher 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6687891/)
- **Scale.** "31,243 potential mutations (min. ten tags) at 9834 unique positions." — [Kircher 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6687891/)
- **Replicate concordance.**
  - Between fully independent libraries: "Pearson correlation of 0.93 (LDLR) and 0.94 (SORT1)." Requiring ≥10 tags raises this to "0.97 (LDLR) and 0.96 (SORT1)." At ≥50 tags it reaches 0.98, but SNV coverage falls to 86.3% and 1-bp-deletion coverage to 15.6%.
  - "Lowest replicate correlation for BCL11A, FOXE1, and one of the MYC elements (rs1198622)." Conversely, "HBG1, IRF4, LDLR, SORT1, and TERT exhibited high reproducibility … (Pearson correlation > 0.9)."
  - — [Kircher 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6687891/)
- **Signal density.** "3–52% of variants using a lenient p-value threshold of <0.01 … average 22%." This proportion "is strongly correlated with the performance of transfection replicates (Pearson correlation of 0.78)," which the authors call "a circular relation." — [Kircher 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6687891/)
- **Predictability and context.** "Average Spearman correlations across our 21 regions … DeepSEA (0.22) performed best." Context matters: in the TERT promoter, E2F-site mutational effects appear in glioblastoma cells but "none of these effects were observed in HEK293T cells." — [Kircher 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6687891/)
- **Access (verified).**
  - The portal is live as of 25 Oct 2026; both URLs returned HTTP 200: [kircherlab.bihealth.org/satMutMPRA](https://kircherlab.bihealth.org/satMutMPRA/) and [mpra.gs.washington.edu/satMutMPRA](https://mpra.gs.washington.edu/satMutMPRA/).
  - Files are TSV or CSV for GRCh37 and GRCh38, "one variant per row," with columns `Chromosome, Position, Ref, Alt, Tags, DNA, RNA, Value, P-Value`. `Value` is the "Log2 variant expression effect derived from the fit of the linear model." Deletions are shown as `-`.
  - Terms: the University of Washington terms apply; the data are shared under the Fort Lauderdale principle, and the producers reserve first global analysis (page summary).
  - Raw data: GEO [GSE126550](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE126550). The article is CC BY 4.0.
  - — [satMutMPRA portal](https://kircherlab.bihealth.org/satMutMPRA/); [Kircher 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6687891/)
- **Database index.** MPRAbase (Genome Res 2025) indexes "130 experiments, encompassing 17,718,677 elements tested across 35 cell types and 4 organisms," including saturation-mutagenesis MPRA. Search results report a dedicated saturation-mutagenesis tab and the address mprabase.com. — [MPRAbase](https://genome.cshlp.org/content/early/2025/04/22/gr.280387.124.abstract)

**CAGI5 "regulation saturation" challenge (closed 3 May 2018; analysis Shigaki et al., Hum Mutat 2019)**
- **Scope.** Predict "17,500+" SNVs in 9 promoters (F9, GP1BB, HBB, HBG, HNF4A, LDLR, MSMB, PKLR, TERT) and 5 enhancers (IRF4, IRF6, MYC rs6983267, SORT1, ZFAND3), with matched cell lines.
- **Submissions.** For each variant: Direction (−1 repressive, 0 none, +1 activating), P_Direction, Confidence (Kircher's rescaled p-value), and SD.
- **Data.** About 25% of alleles were released as training data, in tab-delimited files with columns `Chrom` (hg19), `Pos`, `Ref`, `Alt`, `Direction`, `Confidence`.
- **Access.** Registered users only, under the CAGI Data Use Agreement.
- **Noise ceilings.** Replicate Pearson correlations per locus range from LDLR 0.99 to MYC 0.55.
- **Caveats.** Flanking positions 20 bp outside each construct "should have been removed." "Many predictions of repressive effects may be small." "Expression effects are evaluated in a plasmid system and not the full regulatory environment of the cell." (Page summary.)
- — [CAGI5 challenge page](https://genomeinterpretation.ccs.neu.edu/cagi5-regulation-saturation.html)
- **Results.**
  - "Seven groups submitted." The primary metric was Pearson correlation of discretized predictions with measured effects. The best groups reached about 0.31–0.32 (discretized) and about 0.45 (continuous scores), against CADD at 0.11.
  - deltaSVM from one DHS dataset reached 0.30/0.38. Combining many ENCODE deltaSVMs reached 0.40, and 0.56–0.58 under a 50/50 within-element split.
  - DNN-derived features had the best group-average performance (0.29). MSMB, HNF4A and MYC were hardest.
  - The page itself is inconsistent on some numbers (e.g., 0.308 vs. 0.318 for the top group).
  - (Page summary.) — [Shigaki et al. 2019 (PMC6879779)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6879779/)

**Motif-instance mutation libraries (local at the instance level; redundancy measurable)**
- **DeepSTARR (de Almeida et al., Nat Genet 2022; bioRxiv Oct 2021).** UMI-STARR-seq of oligo libraries in *Drosophila* S2 cells (page summary of the bioRxiv full text):
  - 3,415 motif mutations in 804 developmental and 872 housekeeping enhancers. The Methods give 855/905, an internal inconsistency.
  - Motifs tested: GATA, AP-1, twist, Trl, Dref, Ohler1, Ohler6, plus random controls.
  - "Mutated either in all instances at once or one instance at a time."
  - Oligo-screen replicate PCC 0.93–0.98. Genome-wide screens: 0.73 (developmental) and 0.76 (housekeeping).
  - Predicted vs. measured importance of individual GATA instances: PCC 0.53 (n = 1,013).
  - In 57% of enhancers with multiple instances, one instance was ≥2-fold more important than another. Swapping 2-bp flanks switched instance contributions.
  - About 1,083 human enhancers were also tested in HCT116 cells.
  - — [DeepSTARR bioRxiv](https://www.biorxiv.org/content/10.1101/2021.10.05.463203v1.full); [Nat Genet DOI](https://doi.org/10.1038/s41588-022-01048-5)
- **DeepSTARR data.**
  - Raw oligo-library data: GEO GSE183939, per the README, which also links GSE183936 for the genome-wide screen, inconsistently.
  - Processed figure data at data.starklab.org.
  - The model on Kipoi, `kipoi.org/models/DeepSTARR/`.
  - Training data and model on [Zenodo 5502060](https://doi.org/10.5281/zenodo.5502060) (CC BY 4.0; verified): `Sequences_Train.fa` 116.9 MB, `Sequences_activity_Train.txt` 43.7 MB, val/test about 12 MB FASTA and about 4.5 MB activity each, `DeepSTARR.model.h5` 2.6 MB.
  - Input is 249 bp; the environment is Python 3.7, TF 1.14 and Keras 2.2.4.
  - — [DeepSTARR GitHub](https://github.com/bernardo-de-almeida/DeepSTARR)
- **Kheradpour et al. (Genome Res 2013).** "2104 wild-type sequences and 3314 engineered enhancer variants containing targeted motif disruptions, each using 10 barcode tags and two replicates," in K562 and HepG2. The library covers five activators (HNF1, HNF4, FOXA, GATA, NFE2L2) and two repressors (GFI1, ZFP161). "Scrambling repressor motifs leads to aberrant reporter expression in cell lines where the enhancers are usually inactive." — [Kheradpour 2013 (PMC3638136)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3638136/)
- **Sharpr-MPRA (Ernst et al., Nat Biotech 2016).** "4.6 million nucleotides spanning 15,000 putative regulatory regions tiled at 5-nucleotide resolution," using "overlapping 145-nucleotide constructs" over 295-bp regions in HepG2 and K562. A probabilistic graphical model then infers "activating and repressive nucleotides," so the per-nucleotide truth is model-inferred. — [Ernst 2016 (PMC5125825)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5125825/)

**Assay context caveats (episomal vs. native chromatin)**
- **Episomal vs. integrated.** Comparing episomal and lentiviral-integrated versions of 2,236 liver enhancers, "the activities of chromosomally integrated sequences are substantially different from the activities of the identical sequences assayed on episomes." Integrated assays are "more reproducible and more strongly predictable." — [Inoue et al. 2017 (PMC5204343)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5204343/)
- **MPRA design.** Nine designs tested on 2,440 enhancers show "subtle but significant differences." Testing 192-, 354- and 678-mers shows "sizable differences," i.e. extended sequence context matters. — [Klein et al. 2020](https://pubmed.ncbi.nlm.nih.gov/33046894/)
- **Plasmid artifacts.** In human cells the plasmid origin of replication acts as a competing core promoter, and transfection triggers a type-I interferon response. "These problems cause strongly confounding false-positives and -negatives in luciferase assays and STARR-seq screens." Fly cells "do not seem to be affected." — [Muerdter et al. 2018 (PMC5793997)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5793997/)
- **lentiMPRA at scale.** Agarwal et al. (Nature 2025) tested >680,000 sequences in HepG2, K562 and WTC11, of which 41.7% were active. "As lentiviral integrations are random, lentiMPRA measures the functional effect of cCREs averaged across different genomic locations."
  - Data in the ENCODE portal: ENCSR463IRX, ENCSR460LZI, ENCSR022GQD, ENCSR382BVV, ENCSR244FWB, ENCSR405QCT, ENCSR203UFY, ENCSR336MKI.
  - Models on Zenodo 10558183 and 13908857.
  - — [Agarwal 2025 (PMC11903340)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11903340/)

**Native-chromatin perturbations (local but low-throughput)**
- **CRISPR tiling.** Pooled CRISPR-Cas9 guide libraries performed "in situ saturating mutagenesis" of the human and mouse BCL11A erythroid enhancers. This "reveals critical minimal features and discrete vulnerabilities." — [Canver et al. 2015](https://pubmed.ncbi.nlm.nih.gov/26375006/)
- **BPNet CRISPR validation (Avsec et al., Nat Genet 2021).** "Two-base substitutions in either the Sox2 motif (TTG to AGG) or the Nanog motif (TGA to GGC)" at a single mouse-ESC locus (Nanog motif at chr10:85,539,756-85,539,765, mm10), with ChIP-nexus readout. "Nanog binding was almost completely lost near the Sox2 mutation." When Nanog's motif was mutated, "the surrounding periodic Nanog binding was also reduced as predicted."
  - Data: GEO GSE137193 and GSE134680; Zenodo 3371163, 3371215 and 4294813; the model on Kipoi [BPNet-OSKN](http://kipoi.org/models/BPNet-OSKN/).
  - — [BPNet (PMC8812996)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8812996/)

**Occupancy and natural-variant data**
- **Footprint atlas.** DNase I maps across "243 human cell and tissue types and states" delineate "about 4.5 million compact genomic elements." Available at vierstra.org/resources/dgf, Zenodo 10.5281/zenodo.3603548, and the ENCODE portal. — [Vierstra 2020 (PMC7410829)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7410829/)
- **Footprint caveat.** "TFs with short DNA residence times have no footprints at bound motif elements," and the cleavage profile "originates from the DNA sequence … rather than from the protein." — [Sung et al. 2014](https://pubmed.ncbi.nlm.nih.gov/25242143/)
- **dsQTLs.** In 70 Yoruba lymphoblastoid cell lines, "8,902 locations" were found at FDR 10%, "strongly enriched within inferred transcription factor binding sites." The same paper notes "it remains difficult to identify the causal variants underlying eQTLs." — [Degner et al. 2012](https://pubmed.ncbi.nlm.nih.gov/22307276/)
- **bQTLs.** Pooled ChIP-seq bQTL mapping for five TFs found "thousands of cis-acting QTLs." — [Tehranchi et al. 2016](https://pubmed.ncbi.nlm.nih.gov/27087447/)
- **ASB.** ADASTRA reports ">half a million entries at nearly 270 thousand single-nucleotide polymorphisms" from >7,000 ChIP-seq experiments. — [Abramov et al. 2021](https://doi.org/10.1038/s41467-021-23007-0)
- **QTL benchmarks.** DART-Eval restricts caQTLs and dsQTLs to those in peaks "to enrich for likely causal" variants. A ChromBPNet-type supervised model reached Pearson 0.671 and AUROC 0.772 on the African-ancestry caQTL set. — [DART-Eval (PMC12340898)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12340898/). ChromBPNet itself is a bioRxiv preprint (Dec 2024). — [ChromBPNet](https://doi.org/10.1101/2024.12.25.630221)

### Inferences
- **Genuinely local, per-position experimental truth:**
  - Saturation-mutagenesis MPRA: single edits, signed effects, p-values.
  - Sharpr: inferred.
  - CRISPR tiling: coarse, but in native chromatin.
- **Local at the motif-instance level:** DeepSTARR and Kheradpour libraries.
- **Local but occupancy, not effect:** footprints and ChIP-nexus.
- **Single-variant, population-ascertained:** QTL/ASB.
- **What Kircher measures.** Kircher's `Value` is essentially a wet-lab, regression-estimated, first-order ISM in a WT background. It is the right truth for a selection method whose estimand is a single-position marginal effect, and it is structurally blind to redundancy. If two sites are redundant, both show near-zero single-mutation effects.
- **What DeepSTARR adds.** Its "one instance at a time" versus "all instances at once" design is the only experimental resource I found that directly identifies redundancy (subadditivity) per enhancer. It is the most aligned with the thesis's open problems.
- **Noise ceiling.** The noise ceiling must be used per element. With replicate r of 0.55 for MYC versus 0.99 for LDLR, an agreement score of 0.4 means very different things. CAGI5 rescaled scores by replicate correlations for this reason.
- **Cell-type matching.** A model used as f̂ for validation against Kircher must be trained on a matching cell type: the TERT HEK293T versus GBM differences, and CAGI5's finding that enhancer performance was high only "for a few matched cell types."

### Gaps
- Not retrieved: quantitative resolution and noise for CRISPR tiling (guide density, indel heterogeneity), and the 2023–2026 base- and prime-editing tiling screens.
- Not verified: whether Kircher's data are also in MaveDB; DeepSTARR's processed motif-mutant tables (format and columns); lentiMPRA replicate numbers.
- The DeepSTARR numbers come from the bioRxiv v1 page summary. The published Nat Genet 2022 version may differ.
- A search snippet mentioned a CAGI5 follow-up reporting "information leakage" when training on segments adjacent to test positions (code at bitbucket.org/autosomeru_cagi2018/cagi2018_regsat). I did not verify it, and it would matter for any within-element train/test split.

---

## 3. Model-relative ground truth (exact perturbation of f̂) vs. truth about biology

### Takeaway
When the estimand is defined on the fitted model, exhaustive ISM or motif ablation is exact. The literature treats ISM as the reference that cheaper gradient methods approximate. Taylor-ISM recovers ISM at about r = 0.7, and gradient correction is equivalent to mean-centered ISM. But papers are explicit that model-relative truth is not biological truth. Better accuracy does not guarantee better attributions. ISM can trail DeepLIFT against *synthetic* truth, and Prakash et al. raise the possibility that the "truth" mask itself is the problem. Model explanations are framed as hypotheses that need wet-lab tests, as in BPNet's CRISPR follow-up and DeepSTARR's mutant libraries.

### Cited Findings
- **ISM as the faithful reference.**
  - Forward-only methods need no off-manifold correction. "Attribution methods that are based only on forward propagation, such as in silico mutagenesis, do not require this corrections as the DNN's behavior off the simplex does not affect predictions; all data, including test data, lives on the simplex."
  - Gradients, by contrast, carry "a random gradient component orthogonal to the simplex." The fix is "subtracting the original gradient components by the mean gradients across components for each position."
  - — [Majdandzic 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10169356/)
- **Gradient ≈ ISM, quantified (Sasse, Chikina & Mostafavi; iScience 2024).**
  - ISM(s0, l, b1) ≈ ∂f/∂(l,b1) − ∂f/∂(l,b0), i.e. "TISM."
  - "We observed an average correlation value of 0.7 between TISM and ISM values," and "87% of TISM profiles … had a correlation value of at least 0.6."
  - "TISM's similarity to ISM is within the range of ISM profiles from different models."
  - They show the Majdandzic correction "represents an approximation of attribution maps that are derived from ISM values."
  - — [TISM (PMC11404212)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11404212/)
- **ISM vs. synthetic truth.** On Koo's synthetic tasks, ISM "consistently yields the most reliable attribution maps" (page summary; [Koo & Ploenzke 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8315445)). On Prakash's semi-synthetic data, "ISM … performs slightly worse than DeepLIFT." Their two explanations:
  - Saturation: "mutating a single base may not be enough to substantially disrupt some motif instances."
  - Ground-truth definition: "perhaps some motifs identified in the positive set are not in fact bound due to missing contextual features, and ISM correctly reflects this."
  - — [Prakash 2022](https://proceedings.mlr.press/v165/prakash22a/prakash22a.pdf)
- **The perturbation's reference distribution changes the "truth."** DeepLIFT with dinucleotide-shuffled references outperforms all-zero or base-frequency references ([Prakash 2022](https://proceedings.mlr.press/v165/prakash22a/prakash22a.pdf)). Linear-path IG interpolation produces fractional one-hot inputs "which the model has never seen during training," and logits "can move further away from the actual logits during interpolation" ([Prakash 2022](https://proceedings.mlr.press/v165/prakash22a/prakash22a.pdf)).
- **The model as an "in-silico oracle," then wet-lab tests.** BPNet "used the trained BPNet model as an in-silico oracle to measure how the distance between motif pairs affects TF cooperativity." Motif A was embedded in random sequences with motif B added at a range of distances, averaged over 128 random sequences in the supplementary videos. A genomic-sequence version gave smaller effects, "likely because the genomic motif instances were often of lower affinity." The authors then argue that "precise oracle predictions … allow less scalable in vivo experiments such as the CRISPR editing experiments to be performed on the most interesting and promising observations." — [BPNet](https://pmc.ncbi.nlm.nih.gov/articles/PMC8812996/)
- **Explicit statement of the distinction** (covered in depth by the companion notes): CREME's authors state it "is strictly a model interpretability tool," that results "would reflect artifacts learned by the model," and that its insights "should be treated as a hypothesis and not a replacement for laboratory-based experiments" (page summary). — [Toneyan & Koo 2024 (PMC12065635)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12065635/)
- **Bridging the layers.** DeepSTARR compared model-derived motif-instance importance with experimentally measured instance knockouts: PCC 0.53 for 1,013 GATA instances. The model also "outperforms PWM scores for instance contribution" (page summary). — [DeepSTARR bioRxiv](https://www.biorxiv.org/content/10.1101/2021.10.05.463203v1.full)
- **Generic-ML parallel (search snippet only).** XAI-Units argues that synthetic-data ground truth can mismatch an imperfect trained model, and pairs each dataset with a handcrafted perfect model. — [XAI-Units](https://arxiv.org/abs/2506.01059)

### Inferences
- **Two senses of "faithful."** Exhaustive ISM or ablation of f̂ is noise-free, local and exact. A selection method with an error guarantee relative to f̂ can therefore be verified exactly, by computing the full ablation lattice for small instance counts. Any residual error then belongs to the method, not to label noise.
- **What f̂ adds to biology.** Whether Δ_f̂ tracks biology is an empirical property of f̂, not of the selection method. A method can be perfectly faithful to f̂ and wrong about biology. This should be stated separately from the thesis's guarantee.
- **The "removal" operation must be part of the estimand.** Sequence features have no natural "absent" value. Single-base substitution (ISM), dinucleotide shuffles of a window, and replacement by background-model samples all define different local value functions V_x(S). The thesis's `x_S` conditioning maps onto this choice. Prakash's reference results show the choice changes rankings.
- **Testing whether f̂ is good enough.** A defensible protocol uses synthetic data, where S\*_gen is known, to test whether selections on f̂ recover S\*_gen when f̂ is accurate, and then quantifies degradation as f̂ gets worse. Koo & Ploenzke's "better accuracy ≠ better interpretability" warns that test AUROC alone will not certify f̂.

### Gaps
- I did not find a genomics paper that formally defines an attribution "estimand" with respect to a replacement distribution and proves error control. The closest are surrogate and in-silico-perturbation frameworks covered by the companion notes.
- The SQUID full text was not retrieved (out of scope here).

---

## 4. How ground-truth definitions handle correlated positions, redundancy and context (thesis open problems)

### Takeaway
- **Correlated, jointly important positions.** Benchmarks either score every motif position separately with an information-content threshold, or move to the motif-instance level. The instance-level scoring is DeepLIFT's total importance per match and DeepSTARR's instance knockouts.
- **Redundancy.** Repeatedly identified as a failure of perturbation methods (the saturation problem), but planted masks still count every redundant copy as "true."
- **Context.** Instance importance shifts with flanks and spacing, in both experiment and model, and the experimental truth itself depends on cell type and assay.
- **Non-monotonicity.** It has clear biological analogs (repressor motifs, negative-effect variants), but synthetic benchmarks are almost all activator-only.

### Cited Findings
- **Correlated positions.** Masks are thresholded by information content: I > 0.01 (Koo & Ploenzke code) or I > 0.1 with intermediate positions excluded (Majdandzic code). Low-information positions inside a planted motif are thus either negatives or ignored. — [exponential_activations helper.py](https://github.com/p-koo/exponential_activations/blob/master/code/helper.py); [GradientCorrection helper.py](https://github.com/p-koo/GradientCorrection/blob/main/gradient_correction/helper.py)
- **Instance-level scoring.** DeepLIFT scores the "total importance allocated to the match," i.e. summed over the motif window. — [DeepLIFT](https://arxiv.org/abs/1704.02685)
- **Redundancy as saturation.** "At the point where i1 = 1 and i2 = 1, perturbing either i1 or i2 to 0 will not produce a change in the output. Note that the gradient of the output w.r.t the inputs is also zero when i1 + i2 > 1." — [DeepLIFT, Fig. 1](https://arxiv.org/abs/1704.02685)
- **Redundancy in genomics.** "ISM also fails to highlight important features masked by saturation due to buffering interactions with other features (e.g. multiple motif instances in a sequence that buffer each other)." — [DFIM](https://pmc.ncbi.nlm.nih.gov/articles/PMC6129272/). ISM "can fail to reveal the presence of motifs when nonlinear saturation effects are present." — [GkmExplain](https://pmc.ncbi.nlm.nih.gov/articles/PMC6612808/)
- **Redundancy at enhancer scale** (companion-notes topic). CREME attributes "subadditivity" to "other redundant enhancers," and finds "Enformer's predictions of gene expression saturate as the dosage of enhancers (or silencers) increases" (page summary). — [Toneyan & Koo 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC12065635/)
- **Context in experiments.** "Instances of the same TF motif" are often not equivalent. Flank swaps of 2 bp switch contributions, and AP-1 instances closer to GATA are more important (page summary). — [DeepSTARR bioRxiv](https://www.biorxiv.org/content/10.1101/2021.10.05.463203v1.full)
- **Context in models.** BPNet finds "soft motif syntax": "interactions between two motifs occur in a flexible but distance-dependent fashion that is specific for each motif pair," including a ~10.5-bp Nanog periodicity. — [BPNet](https://pmc.ncbi.nlm.nih.gov/articles/PMC8812996/)
- **Context in the definition of truth.**
  - Prakash: some planted motifs may be unbound "due to missing contextual features." — [Prakash 2022](https://proceedings.mlr.press/v165/prakash22a/prakash22a.pdf)
  - Cell-type dependence of TERT variant effects. — [Kircher 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6687891/)
  - Episomal vs. integrated differences. — [Inoue 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5204343/)
- **Signed truth (non-monotonicity analogs).**
  - CAGI5 labels each variant −1/0/+1. — [CAGI5 page](https://genomeinterpretation.ccs.neu.edu/cagi5-regulation-saturation.html)
  - Sharpr distinguishes "activating and repressive nucleotides." — [Ernst 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC5125825/)
  - Kheradpour's repressor-motif scrambles raise expression. — [Kheradpour 2013](https://pmc.ncbi.nlm.nih.gov/articles/PMC3638136/)
  - Prakash ranks positions by *absolute* importance, discarding sign in the evaluation. — [Prakash 2022](https://proceedings.mlr.press/v165/prakash22a/prakash22a.pdf)

### Inferences
- **Correlated features → group selection.** The evidence favors defining the selectable unit as a motif instance or window, not a nucleotide. Within-motif positions are jointly necessary, and their per-position importance mostly reflects PWM information content. Position-level AUROC partly measures how well attributions reproduce PWM shape.
- **Redundancy → state the estimand explicitly.** In redundant designs a planted-mask benchmark is not a neutral referee:
  - A selector targeting marginal or necessary features ("which single instance, if removed, changes f̂(x)?") will legitimately return ∅, or one copy.
  - A selector targeting a minimal sufficient set will return one copy.
  - Only a "union of all instances that participate in some sufficient set" estimand returns all copies.
  - DeepSTARR's single- vs. all-instance data can adjudicate experimentally. On f̂, the full 2^k ablation lattice over k planted instances can adjudicate exactly.
- **Non-monotone V is real biology, not an artifact.** Adding a repressor instance, or a variant with a −1 effect, lowers activity. A synthetic design with planted repressor motifs, which I did not find in existing benchmarks, would test the thesis's open problem 1 directly. Kircher, CAGI5 and Sharpr supply signed experimental analogs.
- **Context → local truth must be conditional.** Because instance importance depends on flanks, spacing, cell type and assay, ground truth is a function of (x, assay, cell type). This argues for model-relative estimands with assay-matched f̂ rather than "the" biological support of x.

### Gaps
- Not found: a benchmark that scores selection at the motif-instance level with FDR control, or that evaluates false-discovery proportion using planted label-independent decoys. Reynolds & Pan 2025 uses null decoys for "precision," but on genotypes, not sequences.
- I have no quantitative data on how often redundancy occurs in real enhancers, beyond DeepSTARR's instance non-equivalence statistics and CREME's subadditivity (companion notes).

---

## 5. What is tractable on a laptop in a few hours, and what would convince a thesis committee later

### Takeaway
- **A few hours on a laptop:** use downloadable synthetic sets whose ground truth is stored per sequence. Best is DFIM's AND-gate set with label-independent decoys, a 0.75 MB trained model, and exact planted positions. Next is Koo Task 3 (17 MB HDF5 with generating matrices). Compute **exact model-relative ablations** of f̂ over the planted instances alongside the planted-motif truth.
- **Most convincing later:** validate selections from an assay-matched model against **experimental local perturbations**. Kircher saturation mutagenesis covers per-position, signed, single edits and is public as TSV. DeepSTARR's single- vs. all-instance motif knockouts cover redundancy at the instance level. Both should be reported against per-element noise ceilings.

### Cited Findings
- **DFIM simulated set.** `sequences.fa.gz` 3.6 MB, `labels.txt.gz` 0.15 MB, `simulation_metadata.txt.gz` 4.2 MB with embedding positions, plus `model_weights.h5` 0.75 MB and architecture JSON. 60,000 sequences of 200 bp. (Verified via GitHub tree and file.) — [kundajelab/dfim](https://github.com/kundajelab/dfim)
- **Koo Task 1/3.** HDF5 files of 17.1–17.8 MB inside a 37.7 MB Zenodo zip with an MIT license file, storing `model_*` ground-truth probability matrices. Scoring code is public. (Verified.) — [Zenodo 4301062](https://doi.org/10.5281/zenodo.4301062); [GradientCorrection](https://github.com/p-koo/GradientCorrection)
- **DeepLIFT TAL1/GATA1 AND set.** A 0.63 MB `.simdata` file with embeddings, plus a pre-trained Keras model. (Verified.) — [deeplift genomics notebook](https://github.com/kundajelab/deeplift/blob/master/examples/genomics/genomics_simulation.ipynb)
- **simdna for new designs.** MIT; XOR, pair/spacing and density embedders; output records embeddings. — [simdna](https://github.com/kundajelab/simdna)
- **DeepSTARR as an f̂ with experimental counterpart.** Model 2.6 MB on [Zenodo 5502060](https://doi.org/10.5281/zenodo.5502060) (CC BY 4.0) and on Kipoi. Input 249 bp. Legacy TF 1.14 / Keras 2.2.4 environment. — [DeepSTARR GitHub](https://github.com/bernardo-de-almeida/DeepSTARR)
- **Kircher data.** One TSV/CSV per element with log2 effect and p-value; 21 elements, ≤600 bp. Data use is under Fort Lauderdale terms. — [satMutMPRA portal](https://kircherlab.bihealth.org/satMutMPRA/)
- **CAGI5 version.** The curated version with direction labels requires registration and a DUA. — [CAGI5 page](https://genomeinterpretation.ccs.neu.edu/cagi5-regulation-saturation.html)
- **Assay-matched models for human cell lines.** lentiMPRA-trained models for HepG2, K562 and WTC11 (Zenodo 10558183 and 13908857; GitHub autosome-ru/human_legnet). — [Agarwal 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11903340/)

### Inferences
- **Laptop plan, a few hours (recommended first).**
  1. Load the DFIM set, which contains the thesis's gate design (task3 = ELF1 AND SIX5) with true decoys (AP1/TAL1).
  2. Train or load a small CNN.
  3. For each test sequence, enumerate the planted instances (k is usually 2–6 including decoys) and compute f̂ on all 2^k instance-ablation patterns, using a fixed replacement rule such as a dinucleotide shuffle of the instance window. This gives the exact local value function V_x(S) on f̂: monotone or not, redundancy or not.
  4. Score the selector on three things, per sequence, separately for "on" and "off" gate states: recovery of S\*_gen (the planted causal instances); false selections among decoys, which gives an empirical false-discovery proportion; and agreement with the exact f̂ lattice.
  - Koo Task 3 adds a billboard OR/redundancy design with 3–5 motifs, but its negatives' background motifs are not guaranteed nulls (section 1). The run sizes, with ISM costing 200 × 3 forward passes per sequence, look CPU-feasible, but I did not benchmark them.
- **New synthetic designs to add with simdna.** These cover the cases the literature lacks:
  - min/limiting factor: activity = min over two motif classes;
  - saturating dose: activity = 1 − exp(−number of copies);
  - a repressor motif with negative effect, for non-monotone V;
  - spacing-dependent AND, via `PairEmbeddable`.
  Each gives a known S\*(x) that changes from sequence to sequence.
- **Convincing validation (later; days, not hours).**
  - (a) **DeepSTARR f̂ vs. DeepSTARR oligo knockouts.** This is the best single match to the thesis. Experimental single- vs. all-instance knockouts give a local, instance-level, redundancy-aware truth, with replicate PCC 0.93–0.98, alongside the exact f̂ ablations of the same model. Expect friction from the legacy TF/Keras stack, Kipoi, or a reimplementation, and the GEO accession inconsistency.
  - (b) **Kircher saturation mutagenesis vs. a cell-type-matched f̂.** This tests per-position signed calls. Restrict to high-reproducibility elements (LDLR, SORT1, TERT, IRF4, HBG1, all > 0.9) and report agreement relative to each element's replicate ceiling. Remember it cannot test redundancy.
  - (c) **QTL/ASB and footprints as secondary sanity checks only**, because of their sparsity and occupancy-vs-effect mismatch.

### Gaps
- I did not run any of these pipelines, so the time estimates are inferences.
- I did not check whether the DFIM Keras model loads under current TF/Keras versions, or whether DeepSTARR's Kipoi model runs without the legacy environment.
- I did not inspect the processed DeepSTARR oligo-library tables, so their exact columns (per-instance mutant IDs, activity log2FC) are unconfirmed.
