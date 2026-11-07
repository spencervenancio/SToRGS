# Tooling and logistics feasibility: a local-feature-selection experiment on regulatory-genomics sequence models, tonight, on a macOS laptop

Scope: logistics only (what can be downloaded, installed and run tonight). Conceptual ground-truth types are covered by another researcher. Researched 2026-10-25.

Target machine, verified locally: Apple M3 (8 cores; torch defaults to 4 threads), 16 GB RAM, macOS 14.4.1. The default `python3` is conda-forge Python 3.10.13 at `/opt/anaconda3`, with torch 2.11.0, numpy 1.26.4, scipy 1.11.4, scikit-learn 1.5.1, skorch 1.3.1, captum 0.9.0, numba 0.60.0, h5py 3.11.0 and pandas 2.3.1. MPS is available. No genomics libraries are installed.

**Provenance of local numbers.** Every "measured on this M3" number below comes from an EXPLORATORY, design-deciding probe: a single run, mostly random-initialized weights (fine for timing), and not a thesis result. The scripts are saved in `tooling_probes/`, and `install_checks.sh` there lists the exact commands. Nothing was installed into the user's environment. Installs went into a throwaway `python3 -m venv --system-site-packages` venv in the session scratchpad.

## 1. Fastest credible route tonight: the shopping list

### Takeaway
The fastest credible route needs only one new library, tangermeme, and even that is optional because captum is already installed. Download the Koo-lab `synthetic_code_dataset.h5`: 17 MB, 20,000 × 200 bp, binary labels, plus a per-sequence ground-truth PWM array (`model_*`) that converts directly to a position mask. Train a roughly 50k-parameter CNN on MPS (about 30 s for 30 epochs; test AUROC 0.980 in a probe). Saliency, DeepLIFT/SHAP and IG on all 4,000 test sequences take about 2–16 s each on MPS, and ISM about 50 s. A multi-method, multi-seed run fits in well under an hour on the laptop, so the Slurm cluster is only a fallback, mainly for ISM on ChromBPNet-sized real models.

### Cited Findings
- `synthetic_code_dataset.h5` (17.1 MB) and `synthetic_dataset.h5` (17.8 MB) are plain files in the GitHub repo `p-koo/exponential_activations/data`. They downloaded in 0.8 s and 7.8 s — [GitHub data folder](https://github.com/p-koo/exponential_activations/tree/master/data); [probe: probe2.py](<tooling_probes/probe2.py>)
- The binary file holds `X_train` (14000, 4, 200), `X_valid` (2000, 4, 200), `X_test` (4000, 4, 200), `Y_*` (n, 1) at about 50% positives, and `model_train/valid/test` of the same shape as X. These are ground-truth probability matrices whose 4 channels sum to 1 at every position. All arrays are float64 — [probe: probe2.py](<tooling_probes/probe2.py>)
- The paper defines ground truth as follows. Information content I_i = log2(4) − Σ_j M_ij log2 M_ij. Positions with I_i > 0.1 are positive (motif) and zero-IC positions are negative. Attribution AUROC is computed per sequence — [Majdandzic et al. 2023, Genome Biology](https://pmc.ncbi.nlm.nih.gov/articles/PMC10169356/)
- A Majdandzic-style CNN-deep (52,633 params) trained on this file on MPS ran 30 epochs in 28.6 s. Best validation AUROC was 0.980 (epoch 28) and test AUROC 0.980 — [probe: train_probe.py](<tooling_probes/train_probe.py>)
- On MPS with the same architecture and 256 sequences of L = 200: captum IntegratedGradients (50 steps) 0.63 s, captum DeepLiftShap (20 shared baselines) 1.01 s, tangermeme `deep_lift_shap` (20 per-sequence dinucleotide shuffles) 0.63 s, tangermeme gradient-corrected saliency 0.11 s, tangermeme ISM 3.19 s. Linear extrapolation per 4,000 sequences gives 10 / 16 / 10 / 2 / 50 s — [probe: probe3.py](<tooling_probes/probe3.py>)
- `tangermeme memelite modisco-lite logomaker pyjaspar` resolve wheels-only on this Mac. pip would add only biopython 1.88, figwig 0.1.0, hdf5plugin 7.1.0, igraph 1.0.0, leidenalg 0.11.0, logomaker 0.8.7, memelite 0.5.0, modisco-lite 3.0.0, numba-progress 1.2.0, pyfaidx 0.9.2, pyjaspar 4.0.0, tangermeme 1.6.0 and texttable 1.7.2, with no change to torch or numpy — [probe: install_checks.sh](<tooling_probes/install_checks.sh>)

### Inferences
Shopping list, in order. Expected wall-clock times are for this M3: measured where stated, otherwise extrapolated.
1. **Data (under 1 min).** `curl -L -o synthetic_code_dataset.h5 https://raw.githubusercontent.com/p-koo/exponential_activations/master/data/synthetic_code_dataset.h5`. The URL pattern is from the GitHub API `download_url`; master is the branch the raw notebook fetches worked against. Optionally fetch the 12-task companion `.../data/synthetic_dataset.h5`. Ground-truth mask: `ic = 2 + (M*log2 M).sum(axis=1)` over the channel axis of the (N, 4, L) `model_*` array, then `mask = ic > 0.1`. Cast to float32 before `.to("mps")`, because MPS rejects float64 (§7).
2. **Install (about 1–2 min, all wheels).** `pip install tangermeme==1.6.0 memelite==0.5.0 logomaker==0.8.7`. Preferably do this in `python3 -m venv --system-site-packages ~/venvs/genomics` so the thesis env is untouched. That pattern was tested: it reuses torch 2.11 and captum 0.9.
3. **Model (about 30 s per seed on MPS, measured).** Use Majdandzic CNN-deep: conv24×19 → conv32×7 → pool4 → conv48×7 → pool4 → conv64×3 → pool3 → FC96 → 1, with BatchNorm and dropout. Fifty seeds, matching the paper's 50 random initializations, would take about 25 min on MPS (extrapolated). For L = 200, CPU is as fast as MPS, so seeds can be split across both.
4. **Attributions on all 4,000 test sequences (MPS, extrapolated).** Saliency with gradient correction about 2 s; DeepLIFT/SHAP with 20 dinucleotide shuffles about 10 s; IG about 10 s; ISM about 50 s.
5. **Score selection rules per sequence against the mask.** Start with label-1 sequences, where every marked position is a planted *core* motif and every unmarked position is iid uniform background. In label-0 sequences the marked positions are planted motifs from a background pool that includes the core motifs, so "marked" means "planted", not "label-relevant" (§2).
6. **If the design needs controlled decoys, interactions or point-dependent supports.** Generate a dataset from the JASPAR 2026 MEME file with `memelite.io.read_meme` plus numpy: 20k × 200 bp with masks took 0.19 s (measured). Alternatively use inMOTIFin 4.1.1 or simdna 0.4.3.2 (§2).
7. **Real-model extension (secondary).** The ENCODE ChromBPNet `chrombpnet_nobias` fold-0 h5 (25.6 MB) loads with `bpnetlite.bpnet.BPNet.from_chrombpnet`, and DeepLIFT/SHAP then costs 2.5 s/sequence on MPS (measured). ISM at that size goes to Slurm (§4, §6).

### Gaps
- The copy of `synthetic_code_dataset.h5` in `p-koo/GradientCorrection/data` has the same byte size (17,121,924). The two copies were not hashed against each other.
- The AUROC of 0.980 is a single exploratory run, not a reported result. The thesis's own experiments/configs machinery would be needed before citing it.

## 2. Synthetic data with ground-truth motif positions: download or generate?

### Takeaway
A ground-truth dataset can be downloaded now. Both Koo-lab synthetic datasets ship per-sequence ground-truth PWM arrays (`model_*`) that convert to position masks, are 17–18 MB each on GitHub, and load with h5py. simdna 0.4.3.2 installs and generates on Python 3.10 and records motif start positions, but its own dinucleotide shuffle is broken (Python 2 `xrange`). A planted-motif generator from JASPAR PWMs is about 20 lines of numpy plus `memelite.io.read_meme`. inMOTIFin (2025; authors include Anthony Mathelier) is the most complete modern simulator I found and installs from wheels, but whether it writes per-instance coordinates was not verified. I found no 2024–2026 benchmark with downloadable position masks.

### Cited Findings
**Koo-lab datasets (download).**
- The `p-koo/exponential_activations/data` folder contains `synthetic_code_dataset.h5` (16 MiB), `synthetic_dataset.h5` (17 MiB), `IRF1_400_h3k27ac.h5` (2 MiB), `ZBED2_400_h3k27ac.h5` (2 MiB), `JASPAR_CORE_2016_vertebrates.meme` (344 KiB) and `pfm_vertebrates.txt` (133 KiB). Generation notebooks are in `code/generate_data/` (task1, task2, task3) — [GitHub API listing via probe: apis.py](<tooling_probes/apis.py>); [repo](https://github.com/p-koo/exponential_activations)
- The repo's code targets "Tensorflow r1.15 and r2.0 (tested)". The data files themselves are framework-neutral HDF5 — [exponential_activations README](https://github.com/p-koo/exponential_activations)
- `synthetic_code_dataset.h5` shapes are listed in §1. In it, `X` is exact one-hot (every position sums to 1) and is a *sample* from the PWM in `model_*`. At marked positions the observed base equals the PWM argmax in 79% of cases. About 20% of positions are marked (IC > 0.1) in both classes (0.201 in label-1, 0.197 in label-0), the median is 37 marked positions per sequence in both classes, and no sequence has zero marked positions — [probe: install_checks.sh step 4](<tooling_probes/install_checks.sh>)
- `synthetic_dataset.h5` (12-task) holds `X_train` (21000, 4, 200), `X_valid` (3000, …), `X_test` (6000, …), multi-label `Y_*` (n, 12) at about 13–15% positives per task, and `model_*` (n, 4, 200). Every test sequence has at least one positive label — [probe: probe2.py](<tooling_probes/probe2.py>); [probe: install_checks.sh step 4](<tooling_probes/install_checks.sh>)
- Binary-dataset generator code (task3), as I read it:
  - Core motifs are SP1, Gabpa, CEBPB, MAX and YY1 from JASPAR, plus their reverse complements: 10 PWMs in all.
  - 10,000 positives are sampled from "grammar" PWMs that concatenate core motifs, separated by spacers that are exactly uniform (`np.ones((4, sep))/4`).
  - 10,000 negatives are sampled the same way from a `background_motifs` pool made of the 10 core PWMs plus 50 random JASPAR motifs and their reverse complements.
  - The number of motifs per sequence comes from `cum_dist = cumsum([0, 0, 0.5, 0.25, 0.17, .05, 0.3])`, which by my reading gives 3–7 motifs (50% 3, 25% 4, 17% 5, 5% 6, 3% 7).
  - The seed is 22 and the train/valid/test split is 0.7/0.1/0.2.
  - [task3 notebook](https://github.com/p-koo/exponential_activations/blob/master/code/generate_data/task3_generate_synthetic_code_dataset.ipynb)
- 12-task generator (task1): the core TFs are Arid3a, CEBPB, FOSL1, Gabpa, MAFK, MAX, MEF2A, NFYB, SP1, SRF, STAT1 and YY1, plus reverse complements, with seed 33 — [task1 notebook](https://github.com/p-koo/exponential_activations/blob/master/code/generate_data/task1_generate_synthetic_dataset.ipynb)
- The paper describes the binary set as follows:
  - 20,000 sequences × 200 nt, split 0.7/0.1/0.2.
  - Positives embed 3 to 5 core motifs drawn from 10 PFMs.
  - Negatives draw from a pool of 100 non-overlapping JASPAR background motifs.
  - Ground-truth importance = probability matrix − 0.25.
  - Metrics are attribution AUROC, AUPR and cosine similarity.
  - [Majdandzic et al. 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10169356/)
  - This partly contradicts the notebook code: the code allows up to 7 motifs, and its background pool contains the 10 core PWMs plus 100 random-motif PWMs (50 motifs × 2 strands) — [task3 notebook](https://github.com/p-koo/exponential_activations/blob/master/code/generate_data/task3_generate_synthetic_code_dataset.ipynb)
- The GradientCorrection repo's `data/` holds `synthetic_code_dataset.h5` (17,121,924 bytes) and `download.txt`. The README points to Zenodo for model weights and processed data — [GitHub API listing via probe: probe2.py](<tooling_probes/probe2.py>); [GradientCorrection](https://github.com/p-koo/GradientCorrection)
- Zenodo 10.5281/zenodo.7011631 is a single file, `gradient_correction.zip` (about 790 MiB, published 2022-04-29), described as "Model weights and processed data, including DeepSTARR, Basset and ChIP-seq analysis" — [Zenodo 7011631](https://zenodo.org/records/7011631); [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10169356/)

**simdna (generate).**
- simdna's last PyPI release is 0.4.3.2 (2019-06-27), sdist only, with `install_requires` of numpy>=1.9, matplotlib and scipy. It bundles `encode_motifs.txt.gz` and the HOCOMOCO v10 human mono motifs — [PyPI simdna](https://pypi.org/project/simdna/); sdist `setup.py` read without executing it, as recorded in [probe: install_checks.sh](<tooling_probes/install_checks.sh>)
- On Python 3.10:
  - `LoadedEncodeMotifs` loads 2,067 ENCODE motifs.
  - 1,000 × 200 bp sequences with 2 PWM-sampled motifs each generated in 0.20 s.
  - Each sequence's `.embeddings` records the motif and sampled string plus `startPos`, e.g. `('CTCF_known1-TGTCCACCAGGTGGCGATG', 2)` and `('GATA_disc1-ACAGATAAGG', 87)`.
  - `simdna.dinuc_shuffle.dinuc_shuffle` raises `NameError: name 'xrange' is not defined` (the call is at `simdna/dinuc_shuffle.py` line 9).
  - [probe: bench.py](<tooling_probes/bench.py>)

**JASPAR plus a hand-written generator.**
- JASPAR CORE vertebrates non-redundant MEME files are online. [2026 version](https://jaspar.elixir.no/download/data/2026/CORE/JASPAR2026_CORE_vertebrates_non-redundant_pfms_meme.txt): HTTP 200, 552,150 bytes. [2024 version](https://jaspar.elixir.no/download/data/2024/CORE/JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt): HTTP 200, 469,100 bytes — checked in [probe: probe2.py](<tooling_probes/probe2.py>)
- `memelite.io.read_meme` returns a dict of motif name → numpy array of shape (4, w). On the JASPAR 2016 MEME file it found 518 motifs, e.g. `'MA0002.2 RUNX1'` with shape (4, 11). A numpy generator built on it (20,000 × 200 bp, 1–3 PWM samples planted in positives, boolean per-position mask) ran in 0.19 s. `tangermeme.ersatz.substitute(X, "CACGTG", start=20)` inserted the string at position 20 as expected — [probe: probe2.py](<tooling_probes/probe2.py>)
- tangermeme 1.6.0 signatures:
  - `ersatz.substitute(X, motif, start=None, alphabet=['A','C','G','T'], ignore=['N'], return_packed=None)`
  - `ersatz.insert(X, motif, start=None, …)`
  - `ersatz.dinucleotide_shuffle(X, start=0, end=-1, n=20, random_state=None, verbose=False, allow_N=False, return_packed=None, n_jobs=1)`
  - `utils.random_one_hot(shape, probs=None, dtype='int8', random_state=None, return_packed=False)`
  - [probe: bench.py](<tooling_probes/bench.py>)
- pyjaspar 4.0.0 (2025-10-24) is a 55,831 KiB (about 57 MB) wheel, needs Python ≥3.8 and biopython ≥1.80 — [PyPI pyjaspar](https://pypi.org/project/pyjaspar/)

**inMOTIFin (2025).**
- inMOTIFin is by Ferenc, Martini, Rauluseviciute, Sandve and Mathelier, submitted 2025-06-25. It offers "insertion of motif instances with precise control over their positions, co-occurrences, and spacing" and "direct modification of real sequences". It is distributed on PyPI and Docker Hub, and its evaluation code is at bitbucket.org/CBGR/inmotifin_evaluation — [arXiv 2506.20769](https://arxiv.org/abs/2506.20769)
- PyPI version 4.1.1 (2025-10-30), Python ≥3.9, pure-Python wheel, depending on click, numpy, pandas, dagsim, pyjaspar, biopython and hmmlearn. It resolves wheels-only on this Mac (dagsim 1.0.10, graphviz 0.21, hmmlearn 0.3.3, …) — [PyPI inMOTIFin](https://pypi.org/project/inMOTIFin/); [probe: install_checks.sh](<tooling_probes/install_checks.sh>)
- After installation, `python -m inmotifin` exposes `motif-in-seq`, `motifs`, `multimers` and `random-sequences`.
  - `motif-in-seq` options include `--motif_files`, `--jaspar_db_version`, `--background_type`, `--num_shuffle`, `--markov_order`, `--position_type` (with Gaussian `--position_means`/`--position_variances`), and motif-group co-occurrence controls.
  - The writer emits TSV, JSON, FASTA and MEME files, and the controller has a `mask_motif_in_seq` method.
  - Measured locally; the commands are in [probe: install_checks.sh](<tooling_probes/install_checks.sh>)
  - Docs: [inmotifin.readthedocs.io](https://inmotifin.readthedocs.io/en/latest/)

**2024–2026 benchmarks (none with confirmed downloadable masks).**
- "Beyond Motif Localization: Probing Rule-Level Signals in Synthetic Genomic Grammars" (Ramu, Peres Da Silva, Nagarajan; ICLR 2026 LMRL workshop poster):
  - It uses "synthetic DNA datasets with known ground-truth grammars", evaluates five attribution methods, and introduces a Grammar Satisfiability Score.
  - Findings: "strong motif localization coexists with poor logical faithfulness" and "saliency structure persists under progressive parameter randomization".
  - No data release is stated.
  - [ICLR page](https://iclr.cc/virtual/2026/10017715); [OpenReview](https://openreview.net/forum?id=S2l5Ar2C5t)
- Tseng, Eraslan, Biancalani and Scalia, "A mechanistically interpretable neural network for regulatory genomics", arXiv, 2024-10-08 — [arXiv 2410.06211](https://arxiv.org/abs/2410.06211)
  - Per a search-result snippet of the PDF, it has a simulated REST task. Positives contain left and right half-motifs at set spacings; negatives contain one half or random background.
  - The same snippet says background scanning was needed because a 6-bp GATA motif appears by chance in more than 12% of random backgrounds — [PDF](https://arxiv.org/pdf/2410.06211)
  - The abstract page links no code or data.
- BaGGLS (arXiv, Nov 2025) applies its method to attribution scores from models trained on synthetic sequences with known ground truth, following "Tseng et al. (2024)" with a REST motif made of two submotifs. This comes from a search-result summary only — [arXiv 2511.15330](https://arxiv.org/pdf/2511.15330)
- "Towards More Realistic Simulated Datasets for Benchmarking Deep Learning Models in Regulatory Genomics" builds synthetic datasets from chromatin-accessibility data across five cell types. It is a search snippet, and its data availability was not checked — [bioRxiv 2021.12.26.474224](https://www.biorxiv.org/content/10.1101/2021.12.26.474224.full.pdf)

### Inferences
- For local selection, label-1 sequences of the binary Koo set give a clean true support: marked positions are core motifs and unmarked positions are exactly uniform noise. Label-0 sequences carry planted motifs as well, including core-motif PWMs, so they work as a decoy or null test rather than a recovery test.
- The Koo ground truth is *generative* (what was planted), not *model-dependent* (what the trained CNN actually uses). A model may ignore weak low-IC planted positions, and since X is a PWM sample, 21% of marked positions are off-consensus. A model may also respond to chance motif matches in the background. Scanning backgrounds with FIMO (`memelite`) would quantify that contamination; this is suggested by the Tseng snippet about chance GATA matches.
- The 12-task file's `model_*` arrays do not record which PWM sits where. Recovering per-task supports would mean matching model columns to the 12 PWMs. This is an inference from the h5 keys; a direct check was not run.
- A hand-written generator is the fastest way to get exactly the design the thesis needs, such as decoy motifs, AND/OR rules, or a point-dependent support like the repo's `conditional_interaction` design. It is cheap: under a second for 20k sequences. inMOTIFin adds Markov and shuffled backgrounds and real-sequence insertion, at the cost of learning its config.
- If simdna is used, take dinucleotide shuffles from `tangermeme.ersatz.dinucleotide_shuffle`.

### Gaps
- Whether inMOTIFin's TSV lists per-instance sequence id, start, end and strand was not verified; one small test run is needed.
- No downloadable data or masks were found for the ICLR 2026 grammar benchmark or for Tseng et al. 2024.
- The 3–5 versus 3–7 motifs-per-sequence discrepancy comes from reading the code; the notebook was not re-run.

## 3. Training a small CNN on a Mac: architectures, time, AUROC

### Takeaway
Small CNNs train in seconds to minutes on the M3. Majdandzic's CNN-deep reached test AUROC 0.980 on `synthetic_code_dataset` in 29 s (30 epochs, MPS; exploratory). MPS gives no speed-up for tiny L = 200 models but about 7× at L = 1000. A BPNet-sized model (64 filters × 8 dilated layers, 2114-bp input) trains at about 1 min/epoch per 20k sequences on MPS versus about 6.6 min on CPU.

### Cited Findings
- Majdandzic architectures, all with batch normalization:
  - **CNN-shallow:** conv 24 filters × 19, max-pool 50; conv 48 × 3, max-pool 2; FC 96.
  - **CNN-deep:** conv 24 × 19; conv 32 × 7, max-pool 4; conv 48 × 7, max-pool 4; conv 64 × 3, max-pool 3; FC 96.
  - Either may use ReLU or exponential activation in the first layer.
  - Dropout is 0.1–0.5 and L2 is 1e-6.
  - Training: BCE loss, batch 100, 100 epochs, Adam at lr 1e-3 decayed ×0.2 after 3 epochs without validation-AUC improvement, 50 random inits.
  - They trained on a single P100, with each epoch taking under 2 s (paraphrased).
  - [Majdandzic et al. 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10169356/)
- The same paper's DeepSTARR setup used 402,296 training sequences × 249 bp. Test Pearson r was 0.68 (Dev) and 0.75 (HK) — [Majdandzic et al. 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10169356/)
- Probe on MPS, CNN-deep-like (52,633 params, BN, dropout 0.1–0.5, Adam 1e-3, batch 100, early-stop patience 5): 30 epochs in 28.6 s, best validation AUROC 0.980 at epoch 28, test AUROC 0.980 — [probe: train_probe.py](<tooling_probes/train_probe.py>)
- DeepBind/Koo-like SmallCNN (conv32×19 → pool4 → conv64×7 → pool4 → FC128), one epoch over 20k sequences at batch 128:
  - L = 200 (115,425 params): CPU 3.05 s, MPS 3.05 s.
  - L = 1000 (525,025 params): CPU 12.24 s, MPS 1.66 s.
  - [probe: bench.py](<tooling_probes/bench.py>)
- bpnet-lite `BPNet.__init__` has the signature `(n_filters=64, n_layers=8, n_outputs=2, n_control_tracks=2, count_loss_weight=1, profile_output_bias=True, count_output_bias=True, name=None, trimming=None, verbose=True)`. With 64 filters, 8 layers and a single output, the model has 109,122 params. The ChromBPNet-sized configuration (512 × 8) has 6,377,986 params, and its forward pass costs 72.9 ms/sequence on CPU and 17.0 ms/sequence on MPS at 2114 bp — [probe: bench.py](<tooling_probes/bench.py>)
- BPNet-64×8 training on 2114-bp inputs, batch 64, count plus profile loss: MPS did 1,024 sequences in 3.0 s, about 1.0 min/epoch at 20k (extrapolated). CPU did 256 sequences in 5.0 s, about 6.6 min/epoch at 20k (extrapolated) — [probe: probe3.py](<tooling_probes/probe3.py>)
- EvoAug2's `model_zoo` defines a PyTorch `DeepSTARR` class (the architecture only) — [EvoAug2 docs](https://evoaug2.readthedocs.io/en/latest/_modules/evoaug_utils/model_zoo.html)

### Inferences
- Reproducing Majdandzic's 50-initialization design is feasible tonight: about 25 min on MPS at 30 epochs each (extrapolated).
- For L = 1000 and 20–50k sequences, a SmallCNN needs about 1.7–4 s/epoch on MPS (extrapolated), so 50 epochs take about 1.5–3.5 min. A small BPNet at 20k × 2114 bp takes about 20 min for 20 epochs on MPS.
- A test AUROC of about 0.98 means the binary Koo task is learnable on the laptop with no special tricks. It is not saturated, which leaves room for attribution quality to vary across seeds.

### Gaps
- Typical published classification AUROCs for these Koo datasets were not retrieved; the extracted sections of the paper give no numbers.
- Exponential-activation variants were not trained. The probe is a single seed.

## 4. Attribution and ISM cost on the laptop (tangermeme, captum) versus a GPU

### Takeaway
On the M3 with MPS, gradient methods are nearly free. DeepLIFT/SHAP with 20 shuffled references costs about 2.5 ms/sequence for a Koo-size CNN, about 0.11 s/sequence for BPNet-size, and about 2.5 s/sequence for a real ChromBPNet. ISM is the expensive method: about 12 ms/sequence at L = 200, but about 5.6 s/sequence for BPNet at 2114 bp, which is about 1.6 h per 1,000 sequences. The M3 is about 13–90× slower than the H200 numbers in the tangermeme README. ISM on large real models is the job to send to Slurm.

### Cited Findings
- **BPNet-64×8 count head, L = 2114, measured.**
  - `deep_lift_shap`, 16 sequences × 20 dinucleotide-shuffle references: CPU 14.91 s (0.932 s/sequence), MPS 1.81 s (0.113 s/sequence).
  - Saliency, 16 sequences: CPU 0.167 s, MPS 0.147 s.
  - ISM (default `fast=True`), 2 sequences (about 6,342 variants each): CPU 111.75 s (55.9 s/sequence), MPS 11.24 s (5.62 s/sequence).
  - [probe: bench.py](<tooling_probes/bench.py>)
- **Real ENCODE ChromBPNet (`chrombpnet_nobias`, ENCSR839IAD fold 0) loaded into PyTorch, count head, measured.** `deep_lift_shap` with 20 references: MPS 4 sequences in 10.12 s (2.53 s/sequence); CPU 2 sequences in 15.69 s (7.84 s/sequence) — [probe: probe2.py](<tooling_probes/probe2.py>)
- **Koo-size CNN-deep, L = 200, 256 sequences, measured.**
  - CPU: captum IG (50 steps) 4.14 s, captum DeepLiftShap 4.35 s, tangermeme DeepLIFT/SHAP 3.70 s, saliency 0.07 s, ISM 10.56 s.
  - MPS: 0.63 / 1.01 / 0.63 / 0.11 / 3.19 s in the same order.
  - [probe: probe3.py](<tooling_probes/probe3.py>)
- **SmallCNN, L = 200, 64 sequences, measured.** DeepLIFT/SHAP: CPU 0.51 s, MPS 0.61 s. ISM: CPU 5.23 s, MPS 0.97 s — [probe: bench.py](<tooling_probes/bench.py>)
- **tangermeme README benchmark (source-measured).** Hardware: NVIDIA H200; tangermeme 1.5.1, Python 3.11, PyTorch 2.14.1.
  - DeepLIFT/SHAP with 20 dinucleotide shuffles: 1,000 BPNet sequences in 8.6 s (Captum 8.3 s); 100 ChromBPNet sequences in 3.5 s (Captum 3.4 s).
  - ISM, 320 BPNet (GATA2) sequences: 19.6 s with `fast=True`, 77.9 s with `fast=False`, 104.5 s with Captum `Occlusion`, 105.3 s with a hand-written loop.
  - ISM, 32 ChromBPNet sequences: 14.2 / 54.3 / 74.5 / 57.6 s in the same order.
  - On the ChromBPNet profile head, the median DeepLIFT convergence delta was 5.8e-7 for tangermeme versus 0.18 for Captum. The README attributes this to Captum treating a softmax as linear.
  - [tangermeme README](https://github.com/jmschrei/tangermeme)
- captum 0.9.0 `IntegratedGradients` and `DeepLiftShap` both run on MPS — [probe: recheck.py](<tooling_probes/recheck.py>)
- ChromBPNet-sized (512 × 8) forward pass: 17.0 ms/sequence on MPS and 72.9 ms/sequence on CPU at 2114 bp — [probe: bench.py](<tooling_probes/bench.py>)

### Inferences
Extrapolated cost per 1,000 sequences, linear in N, from the measured per-sequence costs above:

| Model / input | Saliency | DeepLIFT/SHAP (20 refs) | ISM (3L variants) | Where |
|---|---|---|---|---|
| Koo CNN, 200 bp | under 1 s | 2.5 s MPS / 14 s CPU | 12 s MPS / 41 s CPU | laptop |
| BPNet-64×8, 2114 bp | about 10 s | 1.9 min MPS / 15.5 min CPU | 1.6 h MPS / 15.5 h CPU | laptop (MPS) or Slurm for ISM |
| Real ChromBPNet (6.4M params), 2114 bp | seconds | 42 min MPS / 2.2 h CPU | est. 11–30 h MPS | Slurm for ISM |
| H200, per README | — | BPNet 8.6 s; ChromBPNet 35 s | BPNet 61 s; ChromBPNet 7.4 min | — |

- The ChromBPNet ISM figure on MPS is an estimate. The high end is forward cost × 6,342 variants (about 108 s/sequence). The low end applies README's GATA2-BPNet:ChromBPNet fast-ISM ratio (7.2×) to the measured MPS BPNet ISM time (about 40 s/sequence).
- Speed ratio of M3 MPS to H200: DeepLIFT/SHAP about 13× slower for BPNet and about 72× for ChromBPNet; ISM about 92× slower for BPNet. This assumes the README models match the probe sizes, which was not confirmed.
- Off MPS, the probes show that CPU is competitive only for the tiny L = 200 models.

### Gaps
- The README does not state input lengths or model hyperparameters for its "BPNet (GATA2)" and "ChromBPNet" benchmarks.
- Local timings mostly used random weights. DeepLIFT/SHAP and gradient cost do not depend on weight values; the fast-ISM path also should not, but this was not verified.
- ISM on the real ChromBPNet and memory footprints were not measured locally.

## 5. Libraries: current versions, exact install lines, Apple Silicon status

### Takeaway
Everything core is on PyPI with wheels that install on macOS arm64 under Python 3.10, and it leaves torch 2.11 and numpy 1.26 alone: tangermeme 1.6.0, memelite 0.5.0, modisco-lite 3.0.0, logomaker 0.8.7, pyjaspar 4.0.0 and inMOTIFin 4.1.1, verified by a wheels-only pip dry run. Three things break a naive install on this Mac:
- bpnet-lite 1.1.0 hard-depends on macs3, which has no Mac wheel, and a wheels-only resolve silently falls back to bpnet-lite 0.8.1.
- finemo imports pyBigWig, which has no Mac wheel.
- gReLU pins `captum==0.5.0`.

Workarounds are `--no-deps`, bioconda, or a separate venv.

### Cited Findings
Versions and release dates are from the PyPI JSON API on 2026-10-25 ([probe: pypi_meta.py](<tooling_probes/pypi_meta.py>)), unless noted.
- **tangermeme 1.6.0** (2026-10-20). Python ≥3.10. Depends on torch ≥2.0, numba ≥0.58, numpy ≥1.23, pandas ≥2.0, scipy ≥1.10, scikit-learn ≥1.3, memelite ≥0.2, figwig ≥0.1.0, pyfaidx ≥0.7.2.1, joblib, matplotlib and tqdm. First release 0.0.1 (2024-02-16); 26 releases, recently 1.3.0 (2026-06-23), 1.4.0, 1.4.1, 1.5.0 (2026-09-25) and 1.6.0 — [PyPI](https://pypi.org/project/tangermeme/)
- The tangermeme README says to install with `pip install tangermeme` or `uv add tangermeme`, requires Python 3.10+, and ships a Claude Code skill installable with `tangermeme-install-skills` — [tangermeme README](https://github.com/jmschrei/tangermeme)
- **bpnet-lite 1.1.0** (2026-10-19). No `requires_python`. Depends on torch ≥1.9, h5py ≥3.7, seaborn, modisco ≥2.0.0, tangermeme ≥0.2.3, macs3 and bam2bw — [PyPI](https://pypi.org/project/bpnet-lite/)
  - The README warns: "we are removing support for _training_ ChromBPNet models" — [bpnet-lite README](https://github.com/jmschrei/bpnet-lite)
- **modisco-lite 3.0.0** (2026-10-17), Python ≥3.9. It pins leidenalg==0.11.0 and depends on igraph, numba, logomaker, h5py, hdf5plugin, memelite and numba-progress — [PyPI](https://pypi.org/project/modisco-lite/)
  - The tfmodisco-lite README says it "has been merged into the official TF-MoDISco repository" and gives `pip install modisco`. Usage: `modisco motifs -s ohe.npz -a shap.npz -n 2000 -o modisco_results.h5`, on (N, 4, L) arrays saved as npz or npy, with a default `-w` window of 400. Reporting: `modisco report -i … -o report/ -s report/ -m motifs.txt`, which uses TOMTOM — [tfmodisco-lite README](https://github.com/jmschrei/tfmodisco-lite)
  - The official **modisco 2.5.2** was released 2025-10-12 — [PyPI](https://pypi.org/project/modisco/)
- **memelite 0.5.0** (2026-09-29), Python ≥3.9; depends on numba ≥0.55.1, numpy, pandas, pyfaidx and tqdm. First release 2025-04-30 — [PyPI](https://pypi.org/project/memelite/). The tangermeme README describes the memesuite-lite project as "implementations of FIMO and Tomtom that do not require PyTorch" — [tangermeme README](https://github.com/jmschrei/tangermeme)
- **captum 0.9.0** (2026-04-17), Python ≥3.10, torch ≥2.3. The previous release was 0.8.0 (2025-03-27) — [PyPI](https://pypi.org/project/captum/)
- **logomaker 0.8.7** (2025-03-25) — [PyPI](https://pypi.org/project/logomaker/)
- **pyjaspar 4.0.0** (2025-10-24) — [PyPI](https://pypi.org/project/pyjaspar/)
- **simdna 0.4.3.2** (2019-06-27), sdist only — [PyPI](https://pypi.org/project/simdna/)
- **finemo 0.41** (2025-10-03). Depends on numpy, scipy, torch, numba, polars ≥1.0, matplotlib, h5py, hdf5plugin, tqdm, pyBigWig, pyfaidx, jinja2 and jaxtyping — [PyPI](https://pypi.org/project/finemo/)
  - The Fi-NeMo README:
    - Install: `pip install finemo`.
    - Its core optimization is "GPU-accelerated" and in PyTorch; preprocessing does not require a GPU.
    - Inputs: contribution scores as bigWig, ChromBPNet H5, BPNet H5 or tfmodisco-lite `.npy/.npz`, plus motif CWMs in tfmodisco-lite H5 format.
    - Commands: `finemo extract-regions-*`, `call-hits`, `report`, `collapse-hits` and `intersect-hits`.
    - Output: `hits.tsv` with zero-indexed `start` and `end` (end exclusive), `strand`, `motif_name`, `hit_coefficient_global` and other columns.
    - It does not mention CPU-only use, MPS or macOS.
    - [Fi-NeMo README](https://github.com/kundajelab/Fi-NeMo)
- **grelu 1.1.0** (2026-03-24) pins `captum==0.5.0`, and depends on wandb ≥0.14, huggingface_hub, pytorch-lightning ≥2.0, enformer-pytorch, tangermeme ≥0.5.0, modisco-lite ≥2.4.0, pyjaspar ≥3.0 and others — [PyPI](https://pypi.org/project/gReLU/)
- **chrombpnet 1.0.1** (2025-07-11) pins tensorflow==2.8.0, numpy==1.23.4 and protobuf==3.20 — [PyPI](https://pypi.org/project/chrombpnet/). tensorflow 2.8.0 has no macOS arm64 wheels — [PyPI tensorflow 2.8.0](https://pypi.org/project/tensorflow/2.8.0/)
- **kipoi 0.8.6** was last released 2022-07-29 and pins attrs ≤21.4.0 — [PyPI](https://pypi.org/project/kipoi/)
- **shap 0.54.1** (2026-10-16) requires Python ≥3.12 and numpy ≥2, so it cannot be installed on this Python 3.10 — [PyPI](https://pypi.org/project/shap/)
- **Other current versions:**
  - numba 0.68.0 (2026-09-30) requires numpy <2.6,≥1.22 and Python ≥3.10 — [PyPI](https://pypi.org/project/numba/)
  - torch latest is 2.14.1 (2026-09-30), with cp310 macOS arm64 wheels — [PyPI](https://pypi.org/project/torch/)
  - multimolecule 0.2.2 (2026-10-15) requires Python ≥3.10 and depends on transformers, datasets, danling[torch], chanfig and torchmetrics — [PyPI](https://pypi.org/project/multimolecule/)
  - squid-nn 0.4.9 (2026-08-28) — [PyPI](https://pypi.org/project/squid-nn/)
  - evoaug 1.0.6 (2023-02-26) — [PyPI](https://pypi.org/project/evoaug/)
- **Mac wheel status on PyPI:**
  - pybigwig 0.3.26 ships only manylinux wheels, no macOS — [PyPI](https://pypi.org/project/pyBigWig/)
  - MACS3 3.1.0 (2026-10-21) has no macOS arm64 wheel — [PyPI](https://pypi.org/project/MACS3/)
  - bam2bw 0.5.1 and figwig 0.1.0 are pure-Python wheels — [PyPI bam2bw](https://pypi.org/project/bam2bw/); [PyPI figwig](https://pypi.org/project/figwig/)
- **Bioconda has osx-arm64 builds** of pybigwig (0.3.24–0.3.26) and macs3 (3.0.4, 3.0.5, 3.1.0) — [anaconda.org bioconda/pybigwig](https://anaconda.org/bioconda/pybigwig); [anaconda.org bioconda/macs3](https://anaconda.org/bioconda/macs3)
- **Install tests on this Mac:**
  - A wheels-only dry run of `finemo` fails with "no matching distributions available … pybigwig".
  - `bpnet-lite==1.1.0` fails with "No matching distribution found for macs3".
  - Unpinned `bpnet-lite` with `--only-binary=:all:` resolves to **bpnet-lite 0.8.1**.
  - `pip install --no-deps bpnet-lite==1.1.0` works, and `bpnetlite` imports without macs3 (BPNet, CountWrapper, ChromBPNet, `from_chrombpnet` and `from_chrombpnet_lite` all present).
  - `finemo` installed with `--no-deps` fails at import (`finemo/data_io.py` line 26: `import pyBigWig`).
  - [probe: install_checks.sh](<tooling_probes/install_checks.sh>); [probe: bench.py](<tooling_probes/bench.py>)
- **tangermeme 1.6.0 API, from `inspect` on the installed package:**
  - `deep_lift_shap(model, X, args=None, target=0, batch_size=32, references=dinucleotide_shuffle, n_shuffles=20, return_references=False, hypothetical=False, warning_threshold=0.001, additional_nonlinear_ops=None, print_convergence_deltas=False, raw_outputs=False, only_warn=False, dtype=None, device=None, random_state=None, verbose=False)`
  - `saliency(model, X, args=None, target=0, batch_size=32, hypothetical=False, gradient_correction=True, func=None, dtype=None, device=None, verbose=False)`
  - `saturation_mutagenesis(model, X, args=None, start=0, end=-1, batch_size=32, target=None, hypothetical=False, raw_outputs=False, dtype=None, device=None, verbose=False, func=None, fast=True)`, in module `tangermeme.saturation_mutagenesis`
  - [probe: recheck.py](<tooling_probes/recheck.py>); [probe: bench.py](<tooling_probes/bench.py>)
- tangermeme README module list: `ablate`/`ablate_annotations`, `marginalize`, `ersatz` (`insert`, `substitute`, `dinucleotide_shuffle`, `local_dinucleotide_shuffle`), `seqlet.recursive_seqlets`/`tfmodisco_seqlets`, `annotate.annotate_seqlets` and others, and `plot.plot_logo`/`plot_pwm`/`interactive_logo` — [tangermeme README](https://github.com/jmschrei/tangermeme)

### Inferences
Recommended install lines, in a separate venv that reuses the existing torch and captum:
```bash
python3 -m venv --system-site-packages ~/venvs/genomics && source ~/venvs/genomics/bin/activate
# core: all wheels on macOS arm64 / py3.10; leaves torch 2.11.0, numpy 1.26.4, captum 0.9.0 untouched (dry-run verified)
pip install "tangermeme==1.6.0" "memelite==0.5.0" "logomaker==0.8.7"
# motif discovery / JASPAR (optional)
pip install "modisco-lite==3.0.0" "pyjaspar==4.0.0"        # pyjaspar is a ~57 MB wheel; pulls biopython 1.88
# BPNet/ChromBPNet model classes and TF-h5 loaders without macs3/bam2bw/modisco (verified import)
pip install --no-deps "bpnet-lite==1.1.0"
# simulators (optional)
pip install --no-deps "simdna==0.4.3.2"                     # deps already present; do not use simdna.dinuc_shuffle
pip install "inmotifin==4.1.1"
# Fi-NeMo needs pyBigWig, which has no Mac wheel on PyPI; bioconda has osx-arm64 builds (untested here):
#   conda install -c conda-forge -c bioconda pybigwig  &&  pip install "finemo==0.41"
# or run Fi-NeMo on the Linux Slurm cluster, where PyPI pyBigWig wheels exist.
# Do NOT `pip install grelu` into the thesis env (pins captum==0.5.0); do NOT pip install chrombpnet (TF 2.8 pin).
```
- Keep the thesis env's captum at 0.9.0. The repo's `[torch]` extra already uses captum, and gReLU would silently downgrade it.

### Gaps
- Whether `modisco-lite` 3.0.0 and the official `modisco` 2.5.2 conflict when both are installed (they may both provide a `modisco` command; not checked) was not tested.
- Building pyBigWig or macs3 from source with pip on this Mac was not tried.
- Fi-NeMo on CPU or MPS was not tested.

## 6. Pretrained real models: where the weights live and whether PyTorch can load them

### Takeaway
- **ChromBPNet:** works tonight. Models for 1,512 ENCODE DNase/ATAC datasets are on Hugging Face, at 25.6 MB per fold. One loaded into PyTorch with bpnet-lite (verified) and runs DeepLIFT/SHAP at 2.5 s/sequence on MPS.
- **ENCODE BPNet atlas:** 2,339 TF ChIP-seq models at 0.55 MB per fold, but Keras-only, and both bpnet-lite loaders fail on them.
- **BPNet-OSKN (Avsec 2021):** Keras weights (1.8 MB) and `motif-instances.bed` (21.7 MB) are on Zenodo 4294904. A PyTorch port is on Hugging Face at `multimolecule/bpnet` (untested).
- **DeepSTARR:** a 2 MB Keras h5 on Zenodo 5502060, with no official PyTorch port.
- **gReLU zoo:** 12 repos under Genentech on Hugging Face, none of them DeepSTARR.
- **Kipoi:** stale (last release 2022), and its BPNet-OSKN entry specifies a Python 3.7 conda env.

### Cited Findings
- **ChromBPNet ENCODE atlas.**
  - Models were trained on 1,512 ENCODE DNase-seq and ATAC-seq datasets across 408 biosamples, in Hugging Face repos named `kundajelab/encode-chrombpnet-ATAC-<exp>-<annot>`.
  - There are 5 folds. Each fold holds `chrombpnet`, `chrombpnet_nobias` (recommended for discovery) and the bias model, as `.h5` or SavedModel `.tar`. All are hg38.
  - [HF ENCSR839IAD](https://huggingface.co/kundajelab/encode-chrombpnet-ATAC-ENCSR839IAD-ENCSR608VHS)
- Files in fold 0 of that repo: `model.chrombpnet_nobias.fold_0.ENCSR839IAD.h5` 25.6 MB, `model.chrombpnet.fold_0…h5` 26.4 MB, `model.bias_scaled.fold_0…h5` 2.7 MB, plus tars. The whole repo is 108 files, about 632 MiB. The nobias h5 downloaded in 1.6 s. `bpnetlite.bpnet.BPNet.from_chrombpnet(path)` loaded it into PyTorch (6,377,986 params, n_filters 512, n_layers 8) — [probe: probe2.py](<tooling_probes/probe2.py>); [probe: apis.py](<tooling_probes/apis.py>)
- The bpnet-lite README says `ChromBPNet.from_chrombpnet(bias, accessibility)`, run on an ENCODE `ENCFF142IOR.tar.gz` example, "produces identical predictions to the original TensorFlow model". Its examples use (N, 4, 2114) inputs — [bpnet-lite README](https://github.com/jmschrei/bpnet-lite)
- **ENCODE BPNet atlas.**
  - Covers "2,339 ENCODE transcription factor ChIP-seq experiments spanning 788 targets across 175 biosamples", with 5 folds and an MIT license.
  - Each fold has `fold_X/model.h5` (Keras) and `saved_model/`.
  - Input (N, 2114, 4); outputs are profile logits (N, 1000, 2) and logcounts (N, 1). The control inputs "can be passed as zeros".
  - Loading is shown only through TF/Keras with `bpnet.model.custommodel`.
  - [HF REST/Ishikawa example](https://huggingface.co/kundajelab/encode-bpnet-REST-ChIP-seq-Ishikawa-ENCSR000BUU-ENCSR174BCS)
- That REST example's `fold_0/model.h5` is 549 KiB; the repo is 23 files, about 13 MiB. Its layer names are `main_conv_0`, `main_add_1…8`, `logcounts_predictions` and so on. Both `BPNet.from_chrombpnet` and `from_chrombpnet_lite` raise `KeyError: 'Unable to open object (component not found)'` on it — [probe: probe2.py](<tooling_probes/probe2.py>)
- **BPNet-OSKN (Avsec et al.).**
  - Kipoi's `model.yaml` uses `model.BPNetOldSeqModel`, with weights at `https://zenodo.org/record/4294904/files/bpnet.model.h5` (md5 bbe883baef261877bfad07d05feb627d).
  - It uses a 1000-bp input (`auto_resize_len: 1000`) and a conda env with `python=3.7`. License MIT.
  - [Kipoi model.yaml](https://github.com/kipoi/models/blob/master/BPNet-OSKN/model.yaml)
- Zenodo 4294904, "BPNet manuscript data 1", contains:
  - `bpnet.model.h5` 1.8 MB
  - `motif-instances.bed` 21.7 MB
  - `regions.bed` 4.3 MB
  - `PWM,CWM,ChIP-nexus-profiles.tar.gz` 0.1 MB
  - per-TF `*.importance.{profile,counts}.bw` at about 513–520 MB each
  - `*.preds.{pos,neg}.bw` at about 421–506 MB
  - `dfabf.Oct4-Sox2-subset.parq` 3,039.6 MB
  - `output.tar.gz` 11,098.9 MB
  - [probe: probe2.py](<tooling_probes/probe2.py>); [Zenodo 4294904](https://zenodo.org/records/4294904)
- The older record, Zenodo 3371164, is a single `output.tar.gz` of about 10 GiB, published 2019-08-19 — [Zenodo 3371164](https://zenodo.org/records/3371164)
- **PyTorch port of OSKN at `multimolecule/bpnet`.**
  - Files: `model.safetensors` 521 KiB and `pytorch_model.bin` 531 KiB.
  - The model card says "The MultiMolecule team has confirmed that the provided model and checkpoints are producing the same intermediate representations as the original implementation."
  - Input is a 1000-bp window. Output shape is (batch, profile_length, 8): Oct4/Sox2/Nanog/Klf4 × forward/reverse strand.
  - Install with `pip install multimolecule`. The implementation is licensed AGPL-3.0.
  - [HF multimolecule/bpnet](https://huggingface.co/multimolecule/bpnet); [probe: apis.py](<tooling_probes/apis.py>)
- **DeepSTARR.** Zenodo 5502060, "DeepSTARR manuscript data" (2021-09-20), contains `DeepSTARR.model.h5` (2 MiB), `DeepSTARR.model.json` (11 KiB), `Sequences_Train.fa` (112 MiB), `Sequences_Val.fa` and `Sequences_Test.fa` (11 MiB each), and activity text files (42 / 4 / 4 MiB). Total about 187 MiB — [Zenodo 5502060](https://zenodo.org/records/5502060); [Stark lab index](https://data.starklab.org/almeida/DeepSTARR/Model)
  - Kipoi Singularity containers: `kipoi-docker_deepstarr-slim.sif` (421 MiB) — [Zenodo 6639247](https://zenodo.org/records/6639247); and `kipoi-docker_deepstarr.sif` (2 GiB) — [Zenodo 5874507](https://zenodo.org/records/5874507)
  - No official PyTorch port was found. EvoAug2 defines the architecture in PyTorch — [EvoAug2](https://evoaug2.readthedocs.io/en/latest/_modules/evoaug_utils/model_zoo.html)
  - `xinHesysu/bioseq-models` has DeepSTARR checkpoints, but warns (per a search summary) that most checkpoints are project-trained reproductions rather than author-released weights — [HF bioseq-models](https://huggingface.co/xinHesysu/bioseq-models)
  - Majdandzic's Zenodo zip includes their own DeepSTARR weights — [Zenodo 7011631](https://zenodo.org/records/7011631)
- **gReLU zoo.**
  - The Hugging Face `Genentech` org lists 12 models: retinal-models, assayllm-handoff-grpo, assayllm-sft, assayformer, test-model, human-mpra-gosai-2024-model, GM12878_dnase-model, borzoi-model, enformer-model, human-chromhmm-fullstack-model, human-atac-catlas-model and decima-model. None is DeepSTARR — [HF API via probe: probe2.py](<tooling_probes/probe2.py>); [HF Genentech](https://huggingface.co/Genentech)
  - The Borzoi and Enformer repos are loaded via `hf_hub_download` plus `LightningModel.load_from_checkpoint` — [HF borzoi-model](https://huggingface.co/Genentech/borzoi-model/commit/9d9e6915f21aba8bfeccca4af2c72deef291af8d)
  - The gReLU paper describes a zoo of large pre-trained models that can be downloaded, used and fine-tuned (paraphrased from a search summary) — [gReLU bioRxiv](https://www.biorxiv.org/content/10.1101/2024.09.18.613778v1)

### Inferences
- None of these real models comes with position-level ground truth. Three practical stand-ins:
  - Insert a model's known motif into dinucleotide-shuffled backgrounds at known positions with `tangermeme.ersatz.substitute`, which gives semi-synthetic ground truth.
  - Use OSKN's `motif-instances.bed` (21.7 MB) as a silver standard.
  - Use Fi-NeMo hit calls, which need pyBigWig.
- Real-sequence work needs a genome FASTA (hg38 for ENCODE, mm10 for OSKN) and pyfaidx. That download was not sized here.
- Loading ENCODE BPNet-atlas h5 files into PyTorch would need a hand-written h5py weight map from the visible layer names. That is feasible but not tonight's critical path.

### Gaps
- Not tested: loading `multimolecule/bpnet` or running it on MPS; converting DeepSTARR Keras → PyTorch; mapping BPNet-atlas weights.
- The ENCODE portal's own ChromBPNet file accessions, and how they map to the Hugging Face mirrors, were not checked.

## 7. Pitfalls that will bite tonight

### Takeaway
- **MPS rejects float64**, and the Koo h5 arrays are float64.
- **Apply the off-simplex correction on the channel axis.** For an (N, 4, L) tensor that is axis 1. The GradientCorrection README snippet has a `/=` typo and assumes (N, L, 4).
- **Know which variant is being scored.** tangermeme defaults return *actual* (not hypothetical) contributions, gradient-corrected saliency, and ISM projected onto the observed base.
- **Baselines differ.** captum IG with a zero baseline and captum DeepLiftShap with one shared baseline set behave differently from tangermeme's per-sequence dinucleotide shuffles.
- **Dependency traps:** macs3, pyBigWig, gReLU's captum pin, chrombpnet's TF 2.8 pin, and shap's Python 3.12 floor.
- **Koo masks mark planted motifs in both classes**, not only label-relevant ones.

### Cited Findings
- `torch.zeros(2, dtype=torch.float64, device="mps")` raises "TypeError: Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32" — [probe: probe3.py](<tooling_probes/probe3.py>). The Koo `X_*`, `Y_*` and `model_*` arrays are float64 — [probe: probe2.py](<tooling_probes/probe2.py>)
- **Gradient correction.**
  - The paper's correction is G_corrected(l, a) = G(l, a) − (1/A) Σ_a G(l, a), equivalently removing the component along the simplex normal. In NumPy: `attr_map = attr_map - np.mean(attr_map, axis=2, keepdims=True)` for (N, L, A).
  - The paper reports about 5–15% of positions with a gradient angle above 60°, 10–20% above 45°, and 20–40% above 30°.
  - It says the correction "should always be applied" but "is not a 'magic bullet'".
  - [Majdandzic et al. 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10169356/)
  - The GradientCorrection README instead shows `attr_map /= np.mean(attr_map, axis=2, keepdims=True)`, a division that contradicts the paper's subtraction — [GradientCorrection README](https://github.com/p-koo/GradientCorrection)
- **tangermeme defaults.**
  - `saliency` has `gradient_correction=True` and `hypothetical=False`. `deep_lift_shap` has `hypothetical=False`, `references=dinucleotide_shuffle`, `n_shuffles=20`, `warning_threshold=0.001`, `only_warn=False` and `additional_nonlinear_ops=None`. `saturation_mutagenesis` has `fast=True` and `raw_outputs=False`. Every function has `device=None` — [probe: recheck.py](<tooling_probes/recheck.py>); [probe: bench.py](<tooling_probes/bench.py>)
  - Per the README (paraphrased), saliency subtracts the mean gradient across the alphabet by default, following Majdandzic et al. (`gradient_correction=False` turns this off), and ISM by default reports mean-subtracted changes multiplied by the observed one-hot sequence (`raw_outputs=True` returns raw predictions). It also warns that gradients saturate, so near-zero values can hide essential bases — [tangermeme README](https://github.com/jmschrei/tangermeme)
- **Captum and the softmax.** On the ChromBPNet profile head, Captum's DeepLIFT convergence delta was 0.18 versus 5.8e-7 for tangermeme, attributed to Captum treating a softmax as linear — [tangermeme README](https://github.com/jmschrei/tangermeme)
- **Version traps:**
  - gReLU pins `captum==0.5.0` — [PyPI](https://pypi.org/project/gReLU/)
  - chrombpnet pins TF 2.8.0 / numpy 1.23.4 — [PyPI](https://pypi.org/project/chrombpnet/)
  - shap needs Python ≥3.12 / numpy ≥2 — [PyPI](https://pypi.org/project/shap/)
  - bpnet-lite → macs3 and finemo → pyBigWig have no Mac wheels — [probe: install_checks.sh](<tooling_probes/install_checks.sh>)
  - simdna's dinucleotide shuffle is Python 2 only — [probe: bench.py](<tooling_probes/bench.py>)
  - Checkpoints saved by gReLU 1.0.4 reportedly do not load in 1.0.6+. This is a community note, seen only in a search result — [HF retinal-models (community)](https://huggingface.co/dedonnoc/retinal-models-mouse)
- **Ground-truth semantics.** Koo `model_*` marks about 20% of positions in both classes, and the negatives' background pool includes the core PWMs — [probe: install_checks.sh](<tooling_probes/install_checks.sh>); [task3 notebook](https://github.com/p-koo/exponential_activations/blob/master/code/generate_data/task3_generate_synthetic_code_dataset.ipynb). Chance motif matches in random backgrounds are common: per a search snippet, a 6-bp GATA motif appears by chance in more than 12% of backgrounds — [Tseng et al. 2024 PDF](https://arxiv.org/pdf/2410.06211)
- **Reverse complements.** Both Koo generators plant reverse-complement PWMs as separate motifs (`reverse = pwm[:,::-1]; reverse[::-1,:]`) — [task3 notebook](https://github.com/p-koo/exponential_activations/blob/master/code/generate_data/task3_generate_synthetic_code_dataset.ipynb)

### Inferences
- **Layout.** PyTorch and tangermeme use (N, 4, L), and tfmodisco expects length-last (N, 4, L) too. Do the gradient correction as `g - g.mean(dim=1, keepdim=True)`. Copying the README's `axis=2` onto an (N, 4, L) tensor would average over positions, which is wrong.
- **Baselines and the simplex.** A zero baseline (captum IG) lies off the one-hot simplex, and captum's DeepLiftShap takes one baseline distribution shared across the batch. Per-sequence dinucleotide-shuffled references therefore need tangermeme or a per-sequence loop. Baseline choice is a design variable, not a default to inherit.
- **Actual vs hypothetical.** For per-position selection, *actual* contributions (projected onto the observed base) give one score per position. *Hypothetical* scores (4 per position) are what motif-discovery tools typically consume, but the modisco README did not state which it expects, so check before running modisco.
- **Normalization.** Attribution scales vary across sequences, so any selection threshold pooled across sequences needs per-sequence normalization or a rank-based rule. This is directly relevant to the thesis's open threshold-calibration problem.
- **Device and threads.** Pass `device="mps"` (or `"cpu"`) explicitly rather than trusting `device=None`. torch reported 4 threads on this 8-core M3, so `torch.set_num_threads` may matter for CPU runs (untested).
- **Custom nonlinearities.** An exponential first-layer activation, as in Koo & Ploenzke, is a custom nonlinearity. tangermeme's `additional_nonlinear_ops` exists for registering such ops; how captum's DeepLift handles an exp module was not checked.

### Gaps
- Reverse-complement averaging conventions for BPNet/ChromBPNet attributions were not researched.
- Whether tangermeme's DeepLIFT handles `torch.exp` without registration was not tested.
- Whether TF-MoDISco requires hypothetical rather than actual scores was not confirmed from a primary source here.
