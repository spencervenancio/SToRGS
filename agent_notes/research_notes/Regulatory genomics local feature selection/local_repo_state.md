# Local thesis repository: what has been built and run, and what can be reused for a regulatory-genomics local feature selection experiment

_Repo root: the senior-thesis repo, branch `main`, HEAD `be45906`. Inspected read-only on 2026-10-25. Nothing was executed except read-only git commands and one bytecode-free import check (`python -B -c "from lfs.data import get_patches"`). Links point to absolute local paths. "L" means line numbers in the working-tree copy. "Untracked" means the file exists on disk but is not in git._

## 1. The `lfs` package: selection methods, signatures, enforced conventions, simulated designs, and how `local_support(x)` is represented

### Takeaway
`lfs` has three wrapper selectors that refit the model (`minshap`, `max_p`, `loco`). All three work on arbitrary feature groups ("patches", meaning lists of index arrays). It also has gradient-attribution helpers, including the uncalibrated `select_threshold` and the untested `path_gradients`/`reduce_path` min-IG scaffold, plus a newer untracked set of "null-reference" threshold rules: decoy p-values, a two-stage decoy screen, mirror statistics, BH and a conformal quantile. The code itself enforces the five conventions (`loss=` only, importance = increase in loss, unfitted estimators, `rng=` → `spawn`, `lfs.paths`). Ground truth is carried by `SimulatedDataset.local_support(x)`, which returns an integer index array, alongside a separate `local_effect_support(x)`. The newer benchmark module instead returns boolean `(n, 10)` masks. **On the `main` working tree, `lfs/data/__init__.py` and `lfs/data/patches.py` (which define `get_patches` and `single_features`) are missing**, so anything importing them fails.

### Cited Findings
**Selection methods: signatures and outputs**
- `minshap(model, patches, X_train, y_train, loss=None, K=100, alpha=0.05, early_stopping_patience=5, n_jobs=-1, local=False, x_S=None, k=50, rng=None, metric=None, higher_is_better=None)` returns a `SelectionResult` (a dict subclass with `.selected` and `.n_selected`). Its keys are `phi_min` (the minimum over the K permutations), `t_j = sqrt(-2·log(alpha)·mean_k sigma)`, `rejected = phi_min >= t_j`, and the raw `phi` and `sigma` arrays of shape (K, n_patches) — lfs/selection/minshap.py L16–110 (senior-thesis: `lfs/selection/minshap.py`)
- `metric=` or `higher_is_better=` raises `TypeError` and tells the caller to pass `loss=` — lfs/selection/minshap.py L82–88 (senior-thesis: `lfs/selection/minshap.py`). `max_p` does the same for `metric=` — lfs/selection/maxp.py L31–35 (senior-thesis: `lfs/selection/maxp.py`)
- `max_p(...)` takes the same parameters plus `two_sided=True`. It computes `z = phi/sqrt(sigma)`, sets z = 0 (so p = 1) when sigma = 0 as the conservative choice, takes `p_max` across the K permutations, and sets `rejected = p_max < alpha`. It also reports `n_degenerate` — lfs/selection/maxp.py L15–66 (senior-thesis: `lfs/selection/maxp.py`)
- `loco(model, patches, X_train, X_test, y_train, y_test, loss=None, early_stopping_patience=5, n_jobs=-1, local=False, x_S=None, k=50)` needs a **fitted** model for its baseline. It zeroes the patch in both train and test, refits a deepcopy, and returns mean test loss minus the full-model loss ("increase in loss, larger = more important") — lfs/selection/loco.py L9–10, L22–95 (senior-thesis: `lfs/selection/loco.py`)
- The shared engine is `run_permutations` / `permutation_pass`. `phi[j]` is the reduction in loss from adding patch j at its position in a random order. Masking sets every feature outside the active patches to 0 (`_mask_to`). Each step refits a deepcopy on the masked **training** X and evaluates per-sample loss on the same training rows, or on the k-NN subset in local mode. `sigma = var(delta, ddof=1)/n_eval`. The null model is fit once on all-zero X. The K permutations get child streams from `spawn`. joblib uses threads for skorch and processes otherwise — lfs/selection/_permutation.py L7–12, L51–56, L59–102, L105–171 (senior-thesis: `lfs/selection/_permutation.py`)
- Local mode uses `Neighborhood(X, x_S, k)`: a Euclidean k-NN over **all** features, built once and held fixed across refits — lfs/metrics/local.py L12–37 (senior-thesis: `lfs/metrics/local.py`). Local mode raises an error if fewer than 2 neighbours are found — lfs/selection/_permutation.py L133–138 (senior-thesis: `lfs/selection/_permutation.py`)

**Saliency selection and the min-IG scaffold**
- `attribute(model, x, target=None, method in {'saliency','input_x_gradient','integrated_gradients'}, n_steps=50, abs_value=True)` uses Captum and reshapes the input to `(1, -1)`. `attribute_batch` loops over rows — lfs/selection/saliency.py L12, L20–79 (senior-thesis: `lfs/selection/saliency.py`)
- `binary_margin_saliency(model, X)` returns gradients of the identifiable margin f1 − f0 for 2-logit classifiers. `regression_saliency(model, X)` returns gradients of a scalar output. Both reshape to `(n, -1)` and are **uncommitted** additions — lfs/selection/saliency.py L82–122 (senior-thesis: `lfs/selection/saliency.py`); `git diff lfs/selection/saliency.py`
- `select_threshold(attribution, frac=0.05)` selects `attribution >= frac·max`, and its docstring says it "has no error control". `select_top_k(attribution, k)` is the fixed-budget rule — lfs/selection/saliency.py L125–142 (senior-thesis: `lfs/selection/saliency.py`)
- `path_gradients(model, x, target=None, baseline=None, n_steps=50)` returns the per-step gradients along the straight path, shape (n_steps × p), plus `delta = x − baseline`. The default baseline is all zeros. `reduce_path(grads, delta, reduction in {'mean','min','min_abs','max_abs'})` gives standard IG for `'mean'`. `min_integrated_gradients` wraps both. `dropout_curve` sweeps support size against threshold — lfs/selection/saliency.py L145–255 (senior-thesis: `lfs/selection/saliency.py`)
- No experiment, config or test on `main` calls `path_gradients`, `reduce_path` or `min_integrated_gradients` (grep over `lfs/`, `experiments/`, `tests/`). This matches CLAUDE.md's statement that the min-IG question is "scaffolding, not a result — no claim has been tested" — CLAUDE.md, "Open problems" (senior-thesis: `CLAUDE.md`)

**Null-reference rules (untracked module)**
- `mirror_statistic(g1, g2) = sign(g1·g2)(|g1|+|g2|)`. `mirror_threshold(M, q, offset=1)` uses the knockoff+ form, or the plug-in form with offset=0. `mirror_select(M, q, pooled=False)` sets a cutoff per query or, if pooled, per batch. `decoy_pvalues(real, decoy)` gives `p_j = (1 + #{k : decoy_k >= real_j})/(m+1)`. `decoy_select(real, decoy, alpha)`. `two_stage_decoy_select(real_local, decoy_local, real_global, decoy_global, alpha_global, alpha_local)` screens on mean training-input saliency and then selects locally. `benjamini_hochberg(pvalues, q)` works row-wise. `conformal_quantile(scores, level)` — lfs/selection/null_reference.py L29–121 (senior-thesis: `lfs/selection/null_reference.py`)
- The module states its assumptions directly. Decoy validity holds "When the real inputs are independent… under any training procedure that treats input columns symmetrically". Neither mirrors nor decoys say anything about "a gate, an inactive branch" — lfs/selection/null_reference.py L8–24 (senior-thesis: `lfs/selection/null_reference.py`)
- Null types (untracked module): `absent` (f never uses x_j), `gate` (f uses x_j but j is never in S*(x)) and `inactive` (j is in S*(x') for some other x' but not at x). `formula_dependence` detects dependence by permuting each column. `error_rates(selected, truth, codes)` returns per-query `fdr`, `fwer`, `recall`, `n_selected`, `fp_<type>`, `pfer_<type>` and `n_null_<type>` — lfs/metrics/local_nulls.py L1–89 (senior-thesis: `lfs/metrics/local_nulls.py`)
- Recovery scoring uses `recovery_scores(selected, truth, n_features)` (precision, recall, F1, exact match, counts), `false_discovery_rate` and `power`. These accept masks or index arrays — lfs/metrics/recovery.py L9–76 (senior-thesis: `lfs/metrics/recovery.py`)

**Enforced conventions in code**
- The losses are `squared_error`, `zero_one` and `cross_entropy` (which needs probabilities), registered in `REGISTRY`. With `loss=None`, `resolve()` infers `zero_one` for low-cardinality integer targets and `squared_error` otherwise. The docstring explains that V and sigma must come from the same per-sample loss — lfs/metrics/pointwise.py L5–17, L22–77 (senior-thesis: `lfs/metrics/pointwise.py`)
- `set_seed(seed, deterministic_torch=False)` seeds python, numpy and torch and returns a Generator. `spawn(rng, n)` splits it through `SeedSequence.spawn` so that "Sharing one Generator across joblib workers silently duplicates draws" is avoided — lfs/seed.py L14–56 (senior-thesis: `lfs/seed.py`)
- `lfs.paths` defines `ROOT`, `DATA_DIR`, `RESULTS_DIR` and `LIT_DIR` relative to the repo — lfs/paths.py L8–13 (senior-thesis: `lfs/paths.py`)
- Models: `MLP` (one hidden ReLU layer), `DeepMLP`, `SoftplusMLP` (uncommitted) and `InteractionMLP`. `AdaptiveNeuralNetClassifier`/`Regressor` rebuild the module whenever the input or output shape changes and coerce X to 2-D float32. `neural_net(..., module=MLP)` and `neural_net_regressor(..., module=DeepMLP)` return **unfitted** estimators when called without data — lfs/models/nn.py L15–198 (senior-thesis: `lfs/models/nn.py`). The `module=` passthrough and module-kwargs forwarding are uncommitted (`git diff lfs/models/nn.py`). Linear baselines are `logistic_regression` and `linear_regression` — lfs/models/logit.py L5–27 (senior-thesis: `lfs/models/logit.py`)
- Every plotting function returns `(fig, ax)`. The image plots default to `img_size=28` (MNIST) — lfs/viz/__init__.py L1 (senior-thesis: `lfs/viz/__init__.py`), lfs/viz/importance.py (senior-thesis: `lfs/viz/importance.py`), lfs/viz/saliency.py (senior-thesis: `lfs/viz/saliency.py`)

**Simulated designs and `local_support(x)`**
- `SimulatedDataset(name, df, support, _local_support, _local_effect_support, feature_names)`:
  - `local_support(x)` returns an integer index array and falls back to the global `support`.
  - `local_effect_support(x)` is kept "deliberately separate", because a feature can be part of the function's structure while being locally constant at x. It falls back to `local_support`.
  - `support_mask(local_x)` returns a boolean mask.
  - lfs/data/simulated.py L29–82 (senior-thesis: `lfs/data/simulated.py`)
- The registered designs are `xor` (x1, x2 ~ Unif(−1,1) plus noise), `linear_additive`, `nonlinear_additive`, `conditional_interaction`, `logistic` (a continuous response in (0,1)) and `logistic_bernoulli`. Every design except `xor` draws standard-normal X with p = 10 (`_draw`). Names are fixed to x1…x10 — lfs/data/simulated.py L26, L85–229 (senior-thesis: `lfs/data/simulated.py`); README table README.md L142–158 (senior-thesis: `README.md`)
- `conditional_interaction` is the only design whose support varies by point. Its `local_support(x)` returns the gates {x3, x8} plus the active pair for each gate. Its `local_effect_support(x)` drops the gates, "Away from a boundary, the hard gates are locally constant" — lfs/data/simulated.py L148–187 (senior-thesis: `lfs/data/simulated.py`)
- `lfs/data/function_family_benchmark.py` is **untracked and also gitignored**:
  - `FunctionSpec(name, family, task, signal_np, signal_torch, support | support_fn, covariance_rho, target_prevalence, deterministic_classification)` defines 24 functions in 8 families.
  - `truth_masks(X)` returns boolean (n, 10) masks built with `np.zeros(10)`, and `true_gradient(X)` uses torch autograd.
  - `draw_inputs` produces standard-normal or AR(1) (ρ^|i−j|) inputs in 10 dimensions. `prepare_registry` and `make_dataset` are the entry points.
  - lfs/data/function_family_benchmark.py L20–301 (senior-thesis: `lfs/data/function_family_benchmark.py`); `git check-ignore -v` reports `.gitignore:42:data/`
- The 8 families and their local supports are tabulated in the meeting notes — notes/meeting_notes.tex L501–565 (senior-thesis: `notes/meeting_notes.tex`)

**Missing modules on `main`**
- Only `lfs/data/mnist.py` and `lfs/data/simulated.py` are tracked under `lfs/data/` (`git ls-files lfs`). On disk, `lfs/data/` has no `__init__.py` and no `patches.py`, and only stale `.pyc` files for both remain in `lfs/data/__pycache__/` (`ls -la lfs/data/__pycache__`). `from lfs.data import get_patches` raises `ImportError: cannot import name 'get_patches' from 'lfs.data' (unknown location)`, and `lfs.data` resolves as a namespace package (import check run 2026-10-25).
- The two files exist only in commit `7c4a06d` "fix: commit package sources excluded by over-broad gitignore rules" on branch `design/complexity-vs-selection`, and in `6d8870b` on `origin/maintenance-pass-2026-08-21` (`git log --all -- lfs/data/__init__.py lfs/data/patches.py`; `git show --stat 7c4a06d`).

### Inferences
- The patches abstraction (a list of index arrays) is the natural carrier for genomic group structure. A one-hot sequence flattened to a 4·L vector maps to one patch per position (4 channel indices) or one per contiguous window (4·w indices), exactly as MNIST pixels map to patches. `minshap`, `max_p` and `loco` would accept such patches unchanged. The patch-building helpers are missing on `main`, however, so they must be restored, for example from `7c4a06d`, or rewritten.
- Zero-masking plus refitting deletes a window's information from every training row. With one-hot DNA a masked position becomes an all-zero, "N"-like column. That is coherent for refit methods, but the cost is K × n_groups refits of a sequence model.
- MinShap scores each step on the training rows (in-sample). With high-capacity sequence CNNs, overfitting could inflate phi, so a held-out evaluation set would be a design choice to state.
- Local MinShap's Euclidean k-NN over one-hot sequences reduces to a Hamming-type distance. Unless the data has strong shared structure, a random-background sample will contain few genuinely near neighbours. Local-mode MinShap is therefore a poor fit for DNA as written.

### Gaps
- I did not run the test suite, which the constraints forbid, so I have not verified method correctness on this checkout.
- I did not diff the branch copies of `lfs/data/__init__.py` and `patches.py` against whatever the README and runner expect.

## 2. Experiment infrastructure: `experiments/run.py`, configs, `experiment_design/` protocols, and the saliency-threshold line of work

### Takeaway
There are two runner styles:
- **`experiments/run.py`** is the original YAML runner. It writes timestamped directories, handles only MNIST and the simulated designs with MinShap, Max-p and LOCO, and **cannot be imported on `main`** because of the missing `lfs.data` names.
- **Bespoke `experiments/run_saliency_*.py` runners**, mostly untracked, run the whole saliency-threshold line. They take `--output results/<name>` and refuse to overwrite it, write to a temporary directory and then move it into place, stamp `config_sha256`, `git_sha` and `git_dirty`, and generate `REPORT.md` and `ADVISOR_BRIEF.md` in code.

Every planned experiment has a pre-registered design in `experiment_design/<name>/`, with hypotheses, numeric decision rules and locked fresh test seeds. The line ran in order: learned threshold rule → function-family benchmark → gradient-fidelity rescue → ensemble recalibration → retrain-only rerun (Step 0) → null-reference thresholds (Step 1) → boundary analysis and two-stage decoy rule (Step 2) → registered confirmation.

### Cited Findings
**`experiments/run.py`**
- The module docstring says each run writes "a self-contained directory under `results/` holding the resolved config, a metadata stamp (git SHA, seed, timings, library versions), the raw arrays, and any figures". A `sweep:` block expands to the cartesian product of its values — experiments/run.py L1–12, L121–138 (senior-thesis: `experiments/run.py`)
- Directory names follow `results/<YYYY-MM-DD_HHMMSS>_<name>[_<sweeplabel>]_s<seed>/`. Each holds `arrays.npz`, `config.yaml` (with `_`-prefixed keys stripped), `meta.json` (name, seed, method, n_patches, elapsed, finished_at, git sha/branch/dirty, versions, sweep_label), `metrics.json` (recovery scores and FDR against `design.support`, or against `local_support(X_train[query_index])` in local mode) and PNG figures — experiments/run.py L141–237 (senior-thesis: `experiments/run.py`); README.md L81–93 (senior-thesis: `README.md`)
- Supported data sources are `mnist` and `simulated` (which uses `single_features`). Methods are `loco`, `minshap` and `max_p`. Models are `logistic_regression`, `linear_regression`, `neural_net` and `neural_net_regressor` — experiments/run.py L45–84, L177–191 (senior-thesis: `experiments/run.py`)
- Line 33 imports `get_patches`, `load_mnist`, `simulated` and `single_features` from `lfs.data`, which fails on `main` (see §1) — experiments/run.py L33 (senior-thesis: `experiments/run.py`)
- The configs that use `run.py` are `mnist_local_minshap.yaml` (K=50, loss zero_one, n_jobs=1, local k=50, query 42), `mnist_patch_sweep.yaml` (patch size {2, 4, 7, 14} × {minshap, max_p}), `mnist_loco_baseline.yaml`, `simulated_recovery.yaml` (3 designs × 5 seeds) and `conditional_local.yaml` (n=20000, local k=200, queries 0–4) — experiments/configs/ (senior-thesis: `experiments/configs`)

**Saliency runners (the pattern now in use)**
- The CLI is `python -m experiments.run_<name> <config> --output results/<name>` (required), with `--dry-run` and `--preflight-only`. The runner raises `FileExistsError` "refusing to overwrite existing output". It writes everything to a temporary directory and `shutil.move`s it into place. `meta.json` records `started_at`, `finished_at`, `elapsed_seconds`, python, platform, `config_source`, `config_sha256`, `git_sha`, `git_dirty` and versions — experiments/run_saliency_threshold_function_family_benchmark.py L1485–1610 (senior-thesis: `experiments/run_saliency_threshold_function_family_benchmark.py`)
- The model recipe is driven by config. `_training_kwargs` reads `optimizer` (sgd or adam) and `early_stopping` (patience, load_best on `valid_loss`). Classification must use raw-logit `CrossEntropyLoss`, and regression must use `neural_net_regressor` — same file, L138–177 (senior-thesis: `experiments/run_saliency_threshold_function_family_benchmark.py`)
- Every fit is wrapped as `_fit()` → `set_seed(seed, deterministic_torch=True)` and then `bench._model(...)`. Mirror halves, decoy models and decoy draws each get a seed unique to (function, world): base + 1000·function_index + data_seed — experiments/run_saliency_threshold_null_reference.py L71–131 (senior-thesis: `experiments/run_saliency_threshold_null_reference.py`); experiments/configs/saliency_threshold_null_reference.yaml (senior-thesis: `experiments/configs/saliency_threshold_null_reference.yaml`)
- Decoy construction appends m = 19 **i.i.d. N(0,1) columns** to X, fits one model and reads off both local saliency and the training-mean ("global") saliency for the real and decoy columns — experiments/run_saliency_threshold_two_stage.py L79–112 (senior-thesis: `experiments/run_saliency_threshold_two_stage.py`)
- Seed hygiene is checked in code: `load_config` raises "test seeds must be fresh" if a test seed is in the calibration seeds, the registered test seeds or `previously_used_test_seeds` — experiments/run_saliency_threshold_two_stage.py L65–76 (senior-thesis: `experiments/run_saliency_threshold_two_stage.py`)
- Summaries are function-balanced means with a hierarchical bootstrap that resamples families, then functions, then worlds (2,000 replicates) — experiments/run_saliency_threshold_null_reference.py L230–256 (senior-thesis: `experiments/run_saliency_threshold_null_reference.py`)
- The active benchmark data settings are p = 10, n = 1800, train 1200, 100 query points per world, regression noise sd 0.1, calibration seeds 600–603 and registered test seeds 700–704. The base models are 60 epochs, lr 0.003, batch 256, hidden 64. The threshold model is a standardized linear Lasso with alphas {0.003, 0.01, 0.03, 0.1, 0.3}. The 7-clause claim gate is spelled out — experiments/configs/saliency_threshold_function_family_benchmark.yaml (senior-thesis: `experiments/configs/saliency_threshold_function_family_benchmark.yaml`)
- The active predictor recipe is Adam, ≤500 epochs, early stopping with patience 30 and `load_best`. The latest test seeds are 730–734. Previously used test seeds are 710–714 and 720–724. Two-stage uses α_global 0.05, levels {0.05, 0.1, 0.2} and 19 decoys — experiments/configs/saliency_threshold_two_stage_confirmation.yaml L1–18 (senior-thesis: `experiments/configs/saliency_threshold_two_stage_confirmation.yaml`)
- Smoke configs exist for the benchmark, null-reference, trained-predictor and two-stage runs (`*_smoke.yaml`) — experiments/configs/ (senior-thesis: `experiments/configs`)

**Experiment-design documents: protocol and decision rules per experiment**
- **Learned query-level threshold (method).**
  - Definitions: u_j = a_j / max_k a_k with a_j = |∂f_c/∂x_j|, where f_c is the f1 − f0 margin for classification. The eight summaries are z = (q.25, q.50, q.75, sd, g1, g2, m1, m3). The prediction is t̂ = clip[0,1](β0 + βᵀz), selecting Ŝ(x) = {j : u_j ≥ t̂}. The target is t*(x) = min_{j∈S*(x)} u_j, which is "recall-preserving".
  - Scope, as stated: "a computationally calibrated method, not a theorem providing error control".
  - Sources: experiment_design/saliency_threshold_function_family_benchmark/METHOD.md §1–2, §5 (senior-thesis: `experiment_design/saliency_threshold_function_family_benchmark/METHOD.md`). The same three algorithms (Summaries, Calibrate with leave-one-seed-out grouped CV, Select, with degenerate maps reported) appear in notes/meeting_notes.tex L571–630 (senior-thesis: `notes/meeting_notes.tex`)
- **Single function.**
  - Hypotheses: H1 a shared rule, H2 a mechanism-specific rule, H3 a shared but nonlinear rule.
  - Success requires all of: gain over the pooled fixed rule ≥ 0.01; no more than 0.02 below the per-design reference; no design loses more than 0.05; at most 5 nonzero terms.
  - If it succeeds, the next step is leave-one-mechanism-out.
  - Source: experiment_design/saliency_threshold_single_function/EXPERIMENT_DESIGN.md §2, §8 (senior-thesis: `experiment_design/saliency_threshold_single_function/EXPERIMENT_DESIGN.md`)
- **Corrected classifier and AUC gate.**
  - Stage A is a classifier-adequacy gate. Stage B (a pooled threshold on locked seeds 500–504) runs only if Stage A passes.
  - The amendment replaced the accuracy-vs-majority gate with four conditions: finite losses, ≥99% nonzero maps, log loss below the prevalence predictor, and a lower AUROC bootstrap bound above 0.5 on development seeds 402–403.
  - Sources: experiment_design/saliency_threshold_corrected_classifier/EXPERIMENT_DESIGN.md (senior-thesis: `experiment_design/saliency_threshold_corrected_classifier/EXPERIMENT_DESIGN.md`); GATE_AMENDMENT.md (senior-thesis: `experiment_design/saliency_threshold_corrected_classifier/GATE_AMENDMENT.md`)
- **Function-family benchmark.**
  - Claim under test: one threshold rule learned from observable summaries "transfers to previously unseen function families and improves local-support recovery over one fixed threshold". The test is leave-one-family-out (LOFO).
  - Every failure must be assigned to one of five layers: predictor accuracy, gradient fidelity, saliency–estimand compatibility, target/representation/transfer, or stability.
  - The 7-clause claim gate requires: ≥22/24 adequate functions; gain over fixed ≥ 0.01; ≤ 0.02 below the all-family rule; ≤ 0.03 below per-function; ≥6/8 family contrasts nonnegative; no family more than 0.08 below its per-function reference; median refit Jaccard ≥ 0.75.
  - It also specifies a follow-up for each failure layer.
  - Source: EXPERIMENT_DESIGN.md §1, §7, §8 (senior-thesis: `experiment_design/saliency_threshold_function_family_benchmark/EXPERIMENT_DESIGN.md`)
- **Gradient-fidelity rescue.**
  - Hypotheses: H1 gradients are variance-limited (fix with a 5-model ReLU ensemble), H2 ReLU geometry is the problem (fix with Softplus), H3 the attribution object itself is the limit.
  - "Promising" requires a best-F1 gain ≥ 0.05 on the 4 failure functions, no function worse by more than 0.03, and median Jaccard ≥ 0.75.
  - Source: experiment_design/saliency_gradient_fidelity_rescue/EXPERIMENT_DESIGN.md (senior-thesis: `experiment_design/saliency_gradient_fidelity_rescue/EXPERIMENT_DESIGN.md`)
- **Ensemble recalibration.**
  - Hypotheses: H1 a pooled calibration mismatch (mean gain ≥ 0.03, worst change ≥ −0.03), H2 a function-specific mismatch, H3 a summary or target limitation.
  - Source: experiment_design/saliency_ensemble_threshold_recalibration/EXPERIMENT_DESIGN.md (senior-thesis: `experiment_design/saliency_ensemble_threshold_recalibration/EXPERIMENT_DESIGN.md`)
- **Trained predictor (Step 0).**
  - Hypotheses: H_artifact (a paired oracle best-F1 gain ≥ 0.05 means undertraining), H_intrinsic (a gain < 0.02 means MLP gradients are intrinsically poor), H_mismatch (threshold rules are specific to the saliency representation).
  - Rules: "Run once. No recipe sweep." If H_artifact holds, Adam becomes the active saliency source.
  - Source: experiment_design/saliency_threshold_trained_predictor/EXPERIMENT_DESIGN.md §3, §7–9 (senior-thesis: `experiment_design/saliency_threshold_trained_predictor/EXPERIMENT_DESIGN.md`)
- **Null reference (Step 1).**
  - Arms: `fixed`, `lasso`, `mirror_batch@q` (knockoff+ cutoff pooled over a world's 100 queries), `decoy@α` (19 N(0,1) decoys), and `sandwich_all` / `sandwich_lofo` (conformalized quantile regressions of the null ceiling and the active floor).
  - Predictions: P1 noise nulls are controllable; P2 bias-type nulls are not; P3 correlation breaks decoy exchangeability; P4 a measured null beats the incumbent; P5 learned guarantees do not transfer.
  - The design records that "Per-query FDR is therefore not attainable by mirror procedures at p = 10".
  - Deliberately not run: BH on decoy p-values ("needs ≥ 99 decoys at p = 10") and model-X knockoffs.
  - Source: experiment_design/saliency_threshold_null_reference/EXPERIMENT_DESIGN.md §3–10 (senior-thesis: `experiment_design/saliency_threshold_null_reference/EXPERIMENT_DESIGN.md`)
- **Null-reference boundary (Step 2, Part A).**
  - A descriptive binning of gate and inactive false positives by distance to the switching surface, with bins [0, 0.1), [0.1, 0.25), [0.25, 0.5), [0.5, 1) and [1, ∞).
  - H_band: errors are confined near the surface. H_spread: the rate stays above α + 0.02 in the two outer bins.
  - Source: experiment_design/saliency_threshold_two_stage_boundary/EXPERIMENT_DESIGN.md §2 (senior-thesis: `experiment_design/saliency_threshold_two_stage_boundary/EXPERIMENT_DESIGN.md`)
- **Two-stage / paired (Step 2, Part B).**
  - Every arm reads one decoy model per world. Arms: `fixed`, `lasso`, `decoy@α`, `two_stage@α`, `volcano@α` (decoy AND u ≥ c) and `two_stage_volcano@α`.
  - Predictions: R1 replication; R2 the screen buys power; R3 the relative cut removes at least half of the leakage; R4 a guaranteed rule comes within 0.02 F1 of the Lasso.
  - The arms were amended after Step 1 and Part A, and are evaluated only on fresh worlds 720–724.
  - Source: same file, §3a–3 (senior-thesis: `experiment_design/saliency_threshold_two_stage_boundary/EXPERIMENT_DESIGN.md`)
- **Two-stage confirmation.**
  - Registered claims C1–C5, all of which must hold: C1 no F1 loss vs. Lasso (≥ −0.02); C2 lower FDR (upper 95% bound < 0); C3 two-stage unused-feature rate ≤ 0.06; C4 decoy@0.05 replication ≤ 0.06; C5 the screen helps (lower bound > 0). Fresh seeds 730–734.
  - Source: experiment_design/saliency_threshold_two_stage_confirmation/EXPERIMENT_DESIGN.md §4 (senior-thesis: `experiment_design/saliency_threshold_two_stage_confirmation/EXPERIMENT_DESIGN.md`)

### Inferences
- A new genomics experiment fits the saliency-runner pattern (fixed `--output`, temp directory then move, config hash, refusal to overwrite, code-generated REPORT, locked fresh seeds) better than `run.py`. `run.py` is MNIST/simulated-only and currently cannot be imported.
- Several pieces could be reused almost verbatim if a genomics runner emits per-query arrays of shape (n_queries, n_groups): `hierarchical_mean`, `score_selection`, `error_rates`, the decoy, mirror and BH rules, and the code-evaluated registered-claims pattern (`analyze_saliency_two_stage_paired.py`).
- Two pieces need rewriting: the data layer (`make_dataset`, `truth_masks`, `true_gradient`, all hard-coded to p = 10 Gaussian inputs) and the decoy generator (N(0,1) columns).

### Gaps
- I read docstrings, function maps and key functions, not every line of the 13 saliency runners. `run_saliency_threshold_single_function.py` and the corrected-classifier runner were read only through their reports and designs.
- I could not confirm whether the sparse and variance runners reuse `observable_features` from `run_saliency_threshold_all_designs.py`. That matters because of the uncommitted `second_gap` fix described in §7.

## 3. Results: what ran, headline numbers, stated conclusions, decisions taken, and what has not run

### Takeaway
`results/` exists locally (gitignored) and holds 18 saliency-threshold result directories from `main`, plus 16 older result directories (run groups and their `_report`/analysis folders) produced on branch `design/complexity-vs-selection`. In order, the record shows:
- Learned shape-based thresholds give only a small gain over a fixed cutoff, and they do not transfer across function families.
- The largest loss came from an undertrained predictor, fixed by Adam with early stopping.
- A **measured null** (decoy columns) delivers its nominal per-feature false-positive rate exactly, but only for features the function never uses.
- The **two-stage decoy rule** (global screen at α 0.05, then local selection at α 0.2, 19 decoys) passed a pre-registered confirmation and is the recommended deployable rule.
- Gate and inactive-branch errors, the genuinely local part, remain uncontrolled.

No genomics experiment has been designed or run. Several named follow-ups, and all `run.py` configs, have no results.

### Cited Findings
**Early pilots (same mechanism, no held-out family)**
- On `linear_additive`, the polynomial min-target rule has F1 0.954 against 0.946 for the fixed min-fraction rule and 0.987 for the direct oracle. The README states that "The earlier inline pilot is not treated as a result because it was not saved with an exact runnable configuration" — results/saliency_threshold_min_polynomial/README.md (senior-thesis: `results/saliency_threshold_min_polynomial/README.md`)
- Across the six registered designs, deployable `poly_min` F1 is 0.852 (xor), 0.954 (linear), 0.782 (nonlinear), 0.716 (conditional_interaction, structural truth), 0.721 (logistic) and 0.723 (logistic_bernoulli). FDR is about 0.39–0.40 on the conditional and logistic designs — results/saliency_threshold_all_designs/REPORT.md (senior-thesis: `results/saliency_threshold_all_designs/REPORT.md`)
- With local-effect truth (gates excluded), `conditional_interaction` `poly_min` F1 drops to 0.536 and FDR rises to 0.595. The lab note calls this "a target-definition comparison, not a model improvement comparison" — results/saliency_threshold_all_designs_local_effect/REPORT.md, LAB_NOTE.md (senior-thesis: `results/saliency_threshold_all_designs_local_effect/LAB_NOTE.md`)
- A sparse Lasso with 1–8 terms matches degree-2 Ridge with 33–44 terms; for example, linear_additive scores 0.955 (1 term) against 0.954 — results/saliency_threshold_sparse/REPORT.md (senior-thesis: `results/saliency_threshold_sparse/REPORT.md`)
- Local-variance features did not help: xor `poly_min` fell from 0.852 to 0.800 and the other designs moved by ±0.003 — results/saliency_threshold_variance/REPORT.md (senior-thesis: `results/saliency_threshold_variance/REPORT.md`)

**Single function, corrected classifier and AUC gate**
- The single pooled function reached macro F1 0.7413 [0.737, 0.745]: +0.0687 against fixed and −0.0003 against per-design. It used 6 terms where at most 5 were allowed, so the outcome is `did_not_meet_predeclared_success_rule`. The "strongest confound is predictive-model adequacy": most epochs in the classification fits had non-finite training loss — results/saliency_threshold_single_function/REPORT.md (senior-thesis: `results/saliency_threshold_single_function/REPORT.md`)
- In the corrected classifier's Stage A, `logistic_bernoulli` accuracy of 0.7875 exactly tied the naive baseline, so the outcome was `fail_stop_before_stage_b` — results/saliency_threshold_corrected_classifier/REPORT.md (senior-thesis: `results/saliency_threshold_corrected_classifier/REPORT.md`)
- Under the AUC-gate amendment, Stage A passed. Stage B gave candidate macro F1 0.7420 [0.7347, 0.7472]: +0.0656 against fixed and −0.0008 against per-design. The outcome was `successful_for_corrected_known_mechanism_family`, with the caveat "an interpolation result, not evidence for transfer to an unseen mechanism" — results/saliency_threshold_corrected_classifier_auc_gate/REPORT.md (senior-thesis: `results/saliency_threshold_corrected_classifier_auc_gate/REPORT.md`)

**Function-family benchmark (24 functions, LOFO)**
- 22 of 24 functions were predictively adequate. LOFO candidate macro F1 was **0.599** [0.571, 0.627]: +0.009 [0.004, 0.014] against LOFO fixed, −0.003 against all-family and −0.043 against per-function. Five clauses failed, so the result is `broad_transfer_not_supported` — results/saliency_threshold_function_family_benchmark/REPORT.md (senior-thesis: `results/saliency_threshold_function_family_benchmark/REPORT.md`)
- The loss decomposition runs: true gradient with per-query oracle 1.000 → fitted gradient with oracle 0.758 (**−0.242**, "predictor gradients corrupt the ranking") → direct-minimum 0.726 → per-function 0.642 → pooled 0.602 → LOFO 0.599 → LOFO fixed 0.590. Median refit Jaccard was 0.556, against a 0.75 gate. `parity_three` reaches only 0.462 even with the true gradient — results/saliency_threshold_function_family_benchmark/ADVISOR_BRIEF.md (senior-thesis: `results/saliency_threshold_function_family_benchmark/ADVISOR_BRIEF.md`)
- The per-family table of this run is reproduced in the meeting notes: linear_additive 0.728, correlated 0.697, higher_order 0.643, pairwise 0.632, smooth 0.595, probabilistic 0.580, nonsmooth 0.502 and conditional 0.445 — notes/meeting_notes.tex L479–499 (senior-thesis: `notes/meeting_notes.tex`)
- The generated REPORT.md begins with the stray characters `;la` before its first heading, which suggests the generated file was hand-edited — results/saliency_threshold_function_family_benchmark/REPORT.md L1 (senior-thesis: `results/saliency_threshold_function_family_benchmark/REPORT.md`)

**Gradient-fidelity rescue and ensemble recalibration**
- The rescue decision was `mixed_gradient_improvement_requires_followup`. The 5-model ReLU ensemble raised Jaccard from 0.500–0.625 to 0.875–0.889, but the mean best-F1 gain was +0.033 (below the 0.05 gate) and the frozen-rule gain was +0.006. Softplus cut grouped-max best-F1 by 0.098 — results/saliency_gradient_fidelity_rescue/REPORT.md, ADVISOR_BRIEF.md (senior-thesis: `results/saliency_gradient_fidelity_rescue/ADVISOR_BRIEF.md`)
- The recalibration decision was `function_specific_calibration_supported`. Mean F1 on the 4 difficult functions rose from 0.485 to 0.568 with a pooled rule (+0.083, but it missed the no-harm guard by 0.005) and to 0.602 with per-function rules (+0.117). For comparison, the direct-minimum oracle scores 0.706 and the best threshold 0.730. Stated next step: run the "ensemble-calibrated rule through the full registered 24-function, leave-one-family-out protocol" — results/saliency_ensemble_threshold_recalibration/ADVISOR_BRIEF.md (senior-thesis: `results/saliency_ensemble_threshold_recalibration/ADVISOR_BRIEF.md`)

**Threshold-source line (Steps 0–2), the current frontier**
- **Step 0, trained predictor.** The outcome was **H_artifact**. The paired fitted-gradient best-F1 oracle improved by **+0.153** [+0.119, +0.183]. LOFO pooled Lasso rose from 0.599 to 0.795 and per-function from 0.642 to 0.817. The claim gate still reads `broad_transfer_not_supported` — results/saliency_threshold_trained_predictor/REPORT.md (senior-thesis: `results/saliency_threshold_trained_predictor/REPORT.md`)
  - The lab note adds that on the new saliency the learned Lasso no longer beats a fixed threshold (+0.013 [−0.015, +0.039]). After proper training, "72–100% of best-cutoff false positives in the five local-support functions are gate or inactive-branch features". Gate features keep normalized saliency of 0.29–0.42, against 0.07–0.17 for unused features.
  - Decision: "Adam + early stopping is the active predictor" — experiment_design/saliency_threshold_trained_predictor/LAB_NOTE.md (senior-thesis: `experiment_design/saliency_threshold_trained_predictor/LAB_NOTE.md`)
- **Step 1, null reference** (fresh seeds 710–714, 23 adequate functions):
  - F1: fixed 0.775, lasso 0.791, `decoy@0.05` 0.737, `mirror_batch@0.1` 0.760, `sandwich_all@0.05` 0.538.
  - Decoy unused-feature false-positive rate on independent inputs was **0.049 / 0.098 / 0.198** at α = 0.05 / 0.1 / 0.2. The decoy rule selected gate nulls at 0.337–0.500 and inactive nulls at 0.398–0.578.
  - Mirror batch FDR was 0.037 / 0.078 / 0.160 on constant-support functions and 0.310–0.460 on local-support ones.
  - `sandwich_lofo` worst-family miscoverage was 0.373 / 0.536 / 0.770.
  - P1, P2, P3 and P5 held; **P4 failed**.
  - results/saliency_threshold_null_reference/REPORT.md (senior-thesis: `results/saliency_threshold_null_reference/REPORT.md`)
  - The lab note's summary: "A measured null delivers its guarantee exactly, on the nulls it models"; "Neither measured null controls bias-type nulls"; "Learned guarantees do not transfer"; correlated inputs push the unused-feature rate to 1.2–1.45× its independent matches, and "Model-X knockoffs would be the repair" — experiment_design/saliency_threshold_null_reference/LAB_NOTE.md (senior-thesis: `experiment_design/saliency_threshold_null_reference/LAB_NOTE.md`)
- **Step 2, Part A (boundary).** The verdict was H_band False and H_spread True at every level; `decoy@0.05` selected 0.341 in the worst outer bin against 0.467 innermost — results/saliency_null_reference_boundary/REPORT.md (senior-thesis: `results/saliency_null_reference_boundary/REPORT.md`). The lab note describes two regimes. In well-fit functions a "used-somewhere feature's leaked gradient is small relative to the actives but far above a never-used feature's noise". In poorly fit ones, "nothing separates at any distance" — experiment_design/saliency_threshold_two_stage_boundary/LAB_NOTE.md (senior-thesis: `experiment_design/saliency_threshold_two_stage_boundary/LAB_NOTE.md`)
- **Step 2, Part B (720–724).**
  - `two_stage@0.2` had the highest F1 at 0.781, against 0.766 for lasso and 0.743 for fixed. R1 and R2 held; R3 and R4 failed.
  - On local-support functions, fixed (0.640) beats every decoy arm.
  - results/saliency_threshold_two_stage/REPORT.md (senior-thesis: `results/saliency_threshold_two_stage/REPORT.md`)
  - Post-hoc pairing gave `two_stage@0.2 − lasso` F1 +0.014 [−0.014, +0.045] and FDR −0.087 [−0.133, −0.042] — results/saliency_two_stage_paired/REPORT.md (senior-thesis: `results/saliency_two_stage_paired/REPORT.md`)
  - The 19 decoy columns degrade fit on harder functions: the `conditional_gated_pairs` MSE ratio goes from 0.229 for the full model to 0.704 for the decoy model — experiment_design/saliency_threshold_two_stage_boundary/LAB_NOTE.md (senior-thesis: `experiment_design/saliency_threshold_two_stage_boundary/LAB_NOTE.md`)
- **Confirmation (730–734, 22 adequate functions).**
  - `two_stage@0.2` F1 was **0.814** [0.769, 0.858], against 0.780 for lasso and 0.761 for fixed. Its per-query FDR was 0.133 against 0.205 for lasso, and its unused-feature rate was 0.018.
  - On constant-support functions: F1 0.883 and FDR 0.030.
  - On local-support functions: F1 0.577 and FDR 0.482, while fixed (0.644) is best.
  - results/saliency_threshold_two_stage_confirmation/REPORT.md (senior-thesis: `results/saliency_threshold_two_stage_confirmation/REPORT.md`)
  - Claims **C1–C5 all True**: C1 +0.034, C2 upper bound −0.032, C3 0.018, C4 0.050, C5 lower bound +0.050 — results/saliency_two_stage_confirmation_claims/REPORT.md (senior-thesis: `results/saliency_two_stage_confirmation_claims/REPORT.md`)
- **Current decision.** "The two-stage decoy rule (α_global = 0.05, α_local = 0.2, 19 decoys) is the recommended deployable local selection rule, stated with its scope. The open problem is now explicitly the local part." The guarantee "covers features the function never uses, under independent inputs… gate and inactive-branch errors… remain uncontrolled". Untested: "other n, p, correlation structures, and real data" — experiment_design/saliency_threshold_two_stage_confirmation/LAB_NOTE.md (senior-thesis: `experiment_design/saliency_threshold_two_stage_confirmation/LAB_NOTE.md`)

**Older results from branch `design/complexity-vs-selection` (the only MinShap evidence)**
- `results/INDEX.md` was generated by `python -m experiments.index`, which exists only on that branch. It lists groups at that branch's SHAs (`6b7e12b`, `a7501a5`, `5f0593f`, `7d89a8c`) — results/INDEX.md (senior-thesis: `results/INDEX.md`). The branch-only files include `experiments/index.py`, `analyze_separability.py` and `run_indicator_gate_baselines.py` (`git ls-tree` comparison).
- In the gate controls (5 seeds, width 256, K = 10), MinShap selected the main effect x5 in 5/5 runs but the **pure-synergy pair x1/x2 in 0/5** and the gate x3 in 0/5. The mean aggregator, on the same contributions, selected the synergy pair 5/5. Max-p behaved like MinShap — results/gate_controls_analysis/REPORT.md (senior-thesis: `results/gate_controls_analysis/REPORT.md`)
- In the indicator-gate baselines, "LOCO is the clean positive baseline" (gated-minus-control x3 score +0.209). "MinShap, Max-p, and GCM select x3 in 0/5 gated runs". The gradient methods rank x3 last — results/indicator_gate_method_baselines/REPORT.md (senior-thesis: `results/indicator_gate_method_baselines/REPORT.md`)
- In the threshold-free separability runs, the oracle `auc_active` is 1 everywhere. Fitted `auc_active` is 0.93–0.999, but the fraction of queries where some cutoff recovers the support exactly is only 0.018–0.129 for the gated designs — results/separability_by_mechanism_report/REPORT.md (senior-thesis: `results/separability_by_mechanism_report/REPORT.md`). On `single_gated_interaction`, `auc_silent` rises from 0.499 to 0.826 as n goes from 1k to 30k, which the report reads as "the model class smoothing a step, not detection" — results/separability_sample_size_report/REPORT.md (senior-thesis: `results/separability_sample_size_report/REPORT.md`)

**Provenance of every saliency result**
- All 18 `results/saliency*/meta.json` files record **`git_dirty = True`**, at `git_sha` `be459069` or `04762e30` (python extraction of the meta files). Most of the runners that produced them are untracked (`git status`), so the SHA alone cannot reproduce them. The newer runs also record a `config_sha256`.

**Not yet run**
- Every `experiment_design/<name>/` directory has a matching executed results directory. `two_stage_boundary` maps to `saliency_null_reference_boundary`, `saliency_threshold_two_stage` and `saliency_two_stage_paired`.
- The `run.py` configs (`mnist_local_minshap`, `mnist_patch_sweep`, `mnist_loco_baseline`, `simulated_recovery`, `conditional_local`) have no result directories in `results/`. No timestamped top-level directories exist (`ls results`).
- The smoke configs have no result directories.
- Named follow-ups with no design or run on `main`:
  - an ensemble-calibrated 24-function LOFO run (ensemble ADVISOR_BRIEF (senior-thesis: `results/saliency_ensemble_threshold_recalibration/ADVISOR_BRIEF.md`));
  - model-X knockoffs for correlated inputs (null_reference design §10 (senior-thesis: `experiment_design/saliency_threshold_null_reference/EXPERIMENT_DESIGN.md`));
  - a sigmoid-gate experiment (all_designs_local_effect LAB_NOTE (senior-thesis: `results/saliency_threshold_all_designs_local_effect/LAB_NOTE.md`));
  - any test of the min-IG scaffold (grep, §1);
  - GCM, which README lists as "not started" (README.md L18–26 (senior-thesis: `README.md`)), although a GCM baseline was computed on the branch;
  - any genomics experiment (§4).

### Inferences
- The tested and replicated component is the noise-referenced decoy rule with its per-feature false-positive guarantee. In genomics this is the closest analogue to a calibrated "is this window more than noise?" call. The record also says it does not answer "is this a material share of the local effect?" Redundant and cooperative motifs are exactly the gate/inactive-type nulls that remain uncontrolled.
- MinShap zeroed out pure synergy, which is an AND-type interaction. Cooperative motif pairs (AND logic) and redundant motif copies (OR logic) are both cases where a minimum over orderings is expected to drive phi_min toward 0. That makes motif redundancy a natural design axis that exposes open problem 1 rather than hiding it.
- Threshold rules are specific to the saliency representation (trained_predictor LAB_NOTE (senior-thesis: `experiment_design/saliency_threshold_trained_predictor/LAB_NOTE.md`): an SGD-calibrated rule loses 0.038 against recalibration). No learned Lasso threshold calibrated on the p = 10 Gaussian benchmark should therefore be transferred to DNA saliency.

### Gaps
- I did not open `metrics.json` or `arrays.npz` files. Every number above is copied from the generated REPORT, ADVISOR_BRIEF and LAB_NOTE files.
- I did not verify which `observable_features` version (before or after the uncommitted `second_gap` fix, §7) produced `saliency_threshold_all_designs`, `all_designs_local_effect`, `sparse` and `variance`.

## 4. Which components assume continuous inputs, which would work unchanged on one-hot DNA with group structure, and what genomics material already exists

### Takeaway
No genomics code, data loader, config, notebook, design or result exists anywhere in the repo, on any branch or in the stash. Genomics appears only in Spencer's meeting notes (the 10-12-2026 pivot section, which is uncommitted and ends with an empty "Existing Methods" heading) and in untracked AI-generated material: research notes and a report dated 2026-10-24, and a PDF "field guide". The generic array-level tools (recovery and null-type scoring, the decoy, mirror, BH and conformal rules, losses, seeding, bootstrap) work unchanged on (queries × positions or windows) arrays. The data generators, decoy construction, perturbation neighbourhoods, k-NN localisation, default IG baseline and all models assume continuous, flat, low-dimensional inputs. There is no CNN or sequence model, and no genomics library is installed.

### Cited Findings
**Genomics material already present**
- An exact word-bounded grep (genome, DNA, motif, BPNet, ChromBPNet, one-hot, nucleotide, tangermeme, modisco, ChIP, DeepLIFT, seqlet, JASPAR, PWM, MPRA) over code, configs, designs and results finds hits only in `notes/meeting_notes.tex`. The notebooks have 0 word-bounded hits. Branches `design/complexity-vs-selection`, `experiment/complexity-saliency`, `origin/maintenance-pass-2026-08-21` and `stash@{0}` have hits only in `notes/meeting_notes.tex` (`git grep` per ref).
- **Meeting notes, 10-12-2026.**
  - The advisor's three steps: "step 1 is to find a domain where saliency/IG/other functional methods are used / step 2 is the testing (0/1 question) useful / step 3 coming up with an empirical method in this domain or context" — notes/meeting_notes.tex L893–905 (senior-thesis: `notes/meeting_notes.tex`)
  - Problem statement: "$\mathcal{X} \in \{0,1\}^{4\times n}$, a one-hot-encoding of the 4 different neucliatide bases… In the case of *BPNet* it is the base-by-base TF binding profile which is readout from the ChIP-nexus assay".
  - Motivation: "Regulatory genomics researchers often have to make their own thresholding decisions based on heuristics… We would like to understand for some particular sequence, which positions (which motif instances) did the prediction depend on?"
  - The "Existing Methods" paragraph is empty after its lead sentence.
  - notes/meeting_notes.tex L907–925 (senior-thesis: `notes/meeting_notes.tex`)
- **Earlier genomics mentions.** The 04-23 notes use a genomics example in which A is globally associated but independent of Y given B — notes/meeting_notes.tex L115 (senior-thesis: `notes/meeting_notes.tex`). The 04-30 notes say local selection is "more about analysis… eg. genomics phenotype examples" — L154 (senior-thesis: `notes/meeting_notes.tex`). The 06-04 families list includes pharmacogenomic gates (CYP2D6/codeine and others) and paralog redundancy — L315–358 (senior-thesis: `notes/meeting_notes.tex`)
- **Untracked AI research notes (dated 2026-10-24).** They rank regulatory DNA first, "the best fit for all three of Raskutti's steps", with climate as a "controlled arm". They name TF-MoDISco, tangermeme and Fi-NeMo as the competitors, and planted motifs (Koo-lab benchmark), ISM and MPRA as three ground truths. They call off-simplex gradient correction a "mandatory baseline" and the window "the natural testing unit" — notes/reports/Saliency testing application domains.md L3, L47–55, L79–83 (senior-thesis: `notes/reports/Saliency%20testing%20application%20domains.md`)
- The genomics note proposes a minimal study (paraphrased): planted motifs with count, co-occurrence and redundancy varied; small CNNs across seeds; saliency, IG and DeepSHAP with gradient correction; a window-level calibrated threshold; per-sequence FDR and power; and comparison against TF-MoDISco, tangermeme and Fi-NeMo — notes/research_notes/Saliency testing application domains/genomics.md L151–166 (senior-thesis: `notes/research_notes/Saliency%20testing%20application%20domains/genomics.md`). These are second-hand web claims made by an earlier AI session, and the note itself marks some as "not re-verified this session" (L3).
- An untracked PDF, "Interpretability Methods for Regulatory DNA Deep Learning Models: A Field Guide to Gradient-Based Attribution", sits in `lit/AI Summaries/` (21 pages, printed from Chrome on 2026-10-25) (`pdfinfo`). It is not listed in `lit/README.md` or `notes/ref.bib`.
- **Installed packages.** The venv has captum 0.9.0, skorch 1.4.0, torch 2.14.0 and torchvision 0.29.0. It has no tangermeme, bpnet-lite, modisco, pyfaidx, pyBigWig, logomaker, Fi-NeMo, simdna or shap (`ls .venv/lib/python3.11/site-packages`). `pyproject.toml` declares only the numpy, scipy, pandas, sklearn, matplotlib, joblib, tqdm and pyyaml core plus the torch, torchvision, skorch and captum extras — pyproject.toml (senior-thesis: `pyproject.toml`)

**Components that assume continuous or Gaussian, flat, p = 10 inputs**
- `simulated.py`: standard-normal `_draw` (`xor` is uniform), names fixed to x1…x10, and `_frame` uses `FEATURE_NAMES[:p]` — lfs/data/simulated.py L26, L85–100 (senior-thesis: `lfs/data/simulated.py`)
- `function_family_benchmark.py`: `draw_inputs` is standard-normal or AR(1) in 10 dimensions, and `truth_masks` uses `np.zeros(10)` — lfs/data/function_family_benchmark.py L59–67, L240–245 (senior-thesis: `lfs/data/function_family_benchmark.py`)
- Decoys are i.i.d. N(0,1) columns, and validity is argued for formula-absent features under independent inputs with column-symmetric training — experiments/run_saliency_threshold_two_stage.py L86–90 (senior-thesis: `experiments/run_saliency_threshold_two_stage.py`); lfs/selection/null_reference.py L13–17 (senior-thesis: `lfs/selection/null_reference.py`)
- The variance experiment's neighbourhood uses "eight independent Gaussian perturbations per query with standard deviation `0.1`" — results/saliency_threshold_variance/README.md (senior-thesis: `results/saliency_threshold_variance/README.md`)
- `formula_dependence` permutes single columns — lfs/metrics/local_nulls.py L25–40 (senior-thesis: `lfs/metrics/local_nulls.py`)
- `path_gradients` defaults to a zero baseline on a straight path — lfs/selection/saliency.py L167–168 (senior-thesis: `lfs/selection/saliency.py`)
- `Neighborhood` uses Euclidean k-NN — lfs/metrics/local.py L20–26 (senior-thesis: `lfs/metrics/local.py`)
- The models are fully connected only (`MLP`, `DeepMLP`, `SoftplusMLP`, `InteractionMLP`), `_coerce` casts X to float32 and `n_input = X.shape[1]` — lfs/models/nn.py L15–117, L138–141 (senior-thesis: `lfs/models/nn.py`)
- The saliency helpers flatten inputs (`reshape(1, -1)` / `reshape(len, -1)`) and require (n, 2) or (n, 1) outputs — lfs/selection/saliency.py L52, L95–100, L114–119 (senior-thesis: `lfs/selection/saliency.py`)
- The image plots default to 28×28 — lfs/viz/importance.py (senior-thesis: `lfs/viz/importance.py`)

**Components that are generic array code (no input-distribution assumption)**
- Recovery scoring (masks or index arrays) — lfs/metrics/recovery.py (senior-thesis: `lfs/metrics/recovery.py`); `error_rates` and `null_type_codes` on (n_queries, p) boolean arrays — lfs/metrics/local_nulls.py L43–86 (senior-thesis: `lfs/metrics/local_nulls.py`)
- `mirror_*`, `decoy_pvalues`, `decoy_select`, `two_stage_decoy_select`, `benjamini_hochberg` and `conformal_quantile` — lfs/selection/null_reference.py (senior-thesis: `lfs/selection/null_reference.py`)
- `select_top_k`, `select_threshold`, `dropout_curve` and `reduce_path` — lfs/selection/saliency.py (senior-thesis: `lfs/selection/saliency.py`)
- The pointwise losses, `seed.spawn` and `lfs.paths` — lfs/metrics/pointwise.py (senior-thesis: `lfs/metrics/pointwise.py`), lfs/seed.py (senior-thesis: `lfs/seed.py`)
- `minshap`, `max_p` and `loco` with patches as index groups — lfs/selection/_permutation.py (senior-thesis: `lfs/selection/_permutation.py`)
- `observable_features`, `hierarchical_mean` and `score_selection` in the runners — experiments/run_saliency_threshold_null_reference.py L168–256 (senior-thesis: `experiments/run_saliency_threshold_null_reference.py`)

### Inferences
- **Reuse readiness for 4×L one-hot inputs** (my synthesis from the code facts above):

| component | status for one-hot DNA | what would have to change |
|---|---|---|
| `recovery_scores`, `error_rates`, `null_type_codes`, `hierarchical_mean`, registered-claims analysis | unchanged | aggregate the 4 channels to positions or windows first, e.g. sum or max of \|grad\| over channels |
| `decoy_*`, `two_stage_decoy_select`, `mirror_*`, BH, conformal | math unchanged | the null construction must change (below) |
| `minshap`/`max_p`/`loco` with patches = positions or windows | mechanically unchanged | patch builder missing on `main`; cost is K × n_windows refits; MinShap evaluates in-sample |
| saliency helpers | unchanged **if** the model accepts a flat 4L vector and reshapes internally | a CNN module that takes a flat input, or rewritten helpers |
| `Adaptive*` skorch wrappers | unchanged for a flat-input CNN module | a new `module_cls` (no CNN exists) |
| `path_gradients` / IG | runs, but the zero baseline and straight path are off-simplex | a reference choice, e.g. uniform 0.25 or dinucleotide-shuffled |
| data layer (`simulated`, `function_family_benchmark`) | not reusable | a planted-motif generator with a per-sequence `local_effect_support(x)` at position or window level, keeping the `FunctionSpec` pattern (numpy and torch signal, `support_fn`, autograd `true_gradient`) |
| N(0,1) decoy columns | not valid | decoys must match the input type, e.g. appended random one-hot positions or windows; the exchangeability argument assumes column-symmetric training, which a CNN with shared weights, receptive fields and edge effects does not provide |
| Gaussian-perturbation variance features, k-NN `Neighborhood` | not meaningful | Hamming or ISM-style neighbourhoods |

- The p = 10 limits recorded in the designs may relax at genomic scale. Those limits were "per-query FDR not attainable by mirror procedures at p = 10" and "BH needs ≥ 99 decoys at p = 10". With L = 200 positions or tens of windows per sequence, per-sequence FDR via knockoff+ could become reachable, but BH on decoy p-values needs on the order of n_tests/q decoys to resolve small p-values. Both points are untested inferences.
- Off-simplex correction (per-position mean subtraction across the 4 channels, from the AI genomics notes) is not implemented anywhere in `lfs`. A genomics experiment would need it as a baseline arm.

### Gaps
- I found no statement by Spencer choosing a dataset (synthetic planted motifs versus a BPNet/ChIP-nexus dataset), a model, or a scalar target. BPNet outputs profile and counts heads, while the saliency helpers need one scalar output.
- The AI genomics notes' web citations (Koo-lab Zenodo data, TF-MoDISco thresholds, Fi-NeMo) are outside my local-files scope and were not re-verified here.

## 5. Literature: `lit/README.md` and `notes/ref.bib` entries, the "Null Importance paper", and the framework it offers

### Takeaway
`notes/ref.bib` has 4 entries and `lit/README.md` maps 4 PDFs. Neither lists the newer local PDFs. The "Null Importance paper" is almost certainly the local `lit/2609.19511v1.pdf`: **"Null importance: Disentangling relevance for interpretable machine learning"** by Garvesh Raskutti, Kris Sankaran and Jiaxin Ye (arXiv:2609.19511v1 [stat.ML], 16 Sep 2026, "Submitted to Statistical Science"). It is very likely also "the paper that Dr. Raskutti sent" in the 09-21-2026 notes. The meeting notes never name it explicitly, so this identification is an inference. According to the repo's notes, the paper gives a taxonomy of nulls in which saliency targets the *pointwise functional null*, ∂f(x)/∂x_j = 0. Spencer's notes use it to restate selection as a level-α claim at x for every j in N^func-pointwise(x).

### Cited Findings
- **`notes/ref.bib` entries:**
  - `zheng_minshap`, "MinShap: A Modified Shapley Value Approach for Feature Selection" (Zheng & Raskutti), marked an unpublished manuscript;
  - `gimenez2019discoveringconditionallysalientfeatures` (Roquero Gimenez & Zou, arXiv 1905.12177);
  - `li2019graphbased`, the GTV paper (Li, Mark, Raskutti, Willett, Song, Neiman, arXiv 1803.07658);
  - `simonyan2014deepinside` (Simonyan, Vedaldi & Zisserman, arXiv 1312.6034).
  - notes/ref.bib L4–45 (senior-thesis: `notes/ref.bib`)
- `lit/README.md` maps those four PDFs. It lists Integrated Gradients (Sundararajan et al., arXiv 1703.01365) and GCM (Shah & Peters) as "Gaps worth filling", and notes "A misfiled personal document was removed" — lit/README.md L5–22 (senior-thesis: `lit/README.md`). `lit/Shap_Feature_Selection.pdf` is the unpublished MinShap manuscript; I did not read it, per the constraints.
- **Local PDFs not in `lit/README.md` or `ref.bib`:**
  - `lit/2609.19511v1.pdf`, saved 2026-09-27, 29 pages. Its title page reads "Null importance: Disentangling relevance for interpretable machine learning… arXiv:2609.19511v1 [stat.ML] 16 Sep 2026… Garvesh Raskutti, Kris Sankaran and Jiaxin Ye… Submitted to Statistical Science" (`pdfinfo`, first-page `pdftotext`).
  - `lit/Go:No-GO.pdf`, "Beyond Go/No-Go Decisions: A Regional Selection Framework for Uncertainty-Aware Molecule Screening" (Weihua Shi, … D. Kolaczyk; page dated 28 May 2026; 66 pages). It is not referenced anywhere in the notes (grep).
  - The AI-summary PDF from §4.
- **The 09-21-2026 section:**
  - It opens "Read the paper that Dr. Raskutti sent" and contains the subsection "Null Importance Paper": "The Null Importance paper allows us to re-frame our method in terms of a pointwise-functional null. If we stick to this convention, we truly avoid the issue with indicator variables. The paper provides us with a framework to fit within."
  - Under "Statistical Question" it separates a null about the fitted f̂ ("there is no estimation error, and saliency is the target") from a null about f = E[Y|X]. For the latter it defines a level-α threshold at x as Pr(|∂f̂/∂x_j(x)| > τ) ≤ α for every j ∈ N^func-pointwise(x). It adds: "Simulation can measure how well the rule transfers… but measuring it isn't the same as controlling it."
  - notes/meeting_notes.tex L567–654 (senior-thesis: `notes/meeting_notes.tex`)
- **The framework as recorded in the repo's AI research notes:**
  - five null families: marginal statistical, conditional statistical, risk-based, causal, and functional invariance;
  - functional invariance at three levels: global (f(x) = g(x_{−j}) for all x), local (the same on an open neighbourhood U(x)) and pointwise (∂f(x)/∂x_j = 0), with "Under smoothness, global ⇒ local ⇒ pointwise";
  - saliency maps target the pointwise functional null; LOCO maps to the risk null, knockoffs and GCM to the conditional statistical null, and MinShap to a conditional risk-based null that holds only under monotonicity;
  - coincidence lemmas, for example Lemma 1 (global functional ⇔ conditional statistical, under deterministic Y = f(X), rich conditional support and non-redundancy), and counterexamples such as XOR;
  - experiments including local attributions on MNIST, TCGA multiomics and genomic perturbation modelling.
  - notes/research_notes/Saliency testing application domains/statistical_methods.md L10–28, L62–63 (senior-thesis: `notes/research_notes/Saliency%20testing%20application%20domains/statistical_methods.md`); notes/reports/Saliency testing application domains.md L7 (senior-thesis: `notes/reports/Saliency%20testing%20application%20domains.md`)
- The same notes call it "The best candidate for the 'Null Importance paper'". They add that it "does not, however, appear to give a calibrated per-point test", and that "the research notes could not read Sections 5.2–5.3, so it is unconfirmed whether the preprint proposes a *test* for the pointwise null" — statistical_methods.md L10 (senior-thesis: `notes/research_notes/Saliency%20testing%20application%20domains/statistical_methods.md`); report L7 (senior-thesis: `notes/reports/Saliency%20testing%20application%20domains.md`)
- The repo's code already encodes the local/pointwise distinction. `local_support(x)` counts hard gates as relevant, while `local_effect_support(x)` does not — lfs/data/simulated.py L166–178 (senior-thesis: `lfs/data/simulated.py`). The `absent`/`gate`/`inactive` null taxonomy is reported separately — lfs/metrics/local_nulls.py L1–18 (senior-thesis: `lfs/metrics/local_nulls.py`)
- The 09-28 notes add the advisor's criteria: "it's ok to just get a method which works in practice"; a method "has any bones" if it either "beats the state of the art" or has "some nice theory"; and τ(x) = f(T(x)). They also record that $\hat f \approx f \nRightarrow \hat f' \approx f'$ — notes/meeting_notes.tex L656–666, L886–891 (senior-thesis: `notes/meeting_notes.tex`)

### Inferences
- The case for "Dr. Raskutti sent" = Null Importance paper: the subsection sits under the 09-21 header that says he sent a paper; Raskutti is first author; the arXiv date (16 Sep) falls just before the meeting; and the PDF was saved on 27 Sep. Taken together, this is high-confidence but not explicit.
- For one-hot DNA the pointwise derivative null is awkward, because no realizable perturbation corresponds to an infinitesimal change of a one-hot entry. A finite-change (ISM-type, "local functional") null at window level may be the more natural estimand. That choice should be made explicitly against the paper's taxonomy.

### Gaps
- Per the constraints, I did not read the Null Importance PDF beyond its title page. Statements about its content rest on AI notes that admit gaps (Sections 5.2–5.3 unread). Whether the paper proposes a pointwise test, and what its genomic-perturbation experiment does, remains unverified.
- I did not examine why the Go/No-Go paper was added on 2026-09-29. Its regional-selection framing may be relevant, but nothing in the repo discusses it.

## 6. Conventions a new experiment must follow

### Takeaway
A genomics experiment must:
- start as a pre-run design in `experiment_design/<name>/` (decision, hypotheses, arms, locked fresh seeds, numeric decision rules, "not run" list);
- be implemented as a runnable `experiments/run_<name>.py` plus `experiments/configs/<name>.yaml` (and a smoke config);
- write only executed outputs to `results/<name>/`, including `config.yaml`, `README.md` and `meta.json` with git and config hashes.

It must also respect the code-level invariants (`loss=`, importance as loss increase, unfitted estimators, `rng=`/`spawn`, `(fig, ax)` plots, `n_jobs=1` for skorch, `lfs.paths`), and keep PDFs, data and results out of the public repo.

### Cited Findings
- **The artifact boundary.** Put "pre-run protocols, hypotheses, proposed matrices, planned configs, and acceptance criteria in `experiment_design/<name>/`". Put "runnable experiment code and configs in `experiments/`". Put "only artifacts produced by an executed run in `results/<name>/`… A results directory must contain computed output plus the exact resolved config and a reproduction README before any result is reported" — AGENTS.md L5–13 (senior-thesis: `AGENTS.md`); experiment_design/README.md (senior-thesis: `experiment_design/README.md`)
- **CLAUDE.md "Conventions that are easy to get wrong":**
  - `loss=`, never `metric=`; importance is always "increase in loss"; estimators are passed unfitted (only `loco` takes a fitted model); pass `rng=`.
  - Plot functions return `(fig, ax)` and never call `plt.show()`; `n_jobs=1` for skorch models "until a run is known-good"; paths come from `lfs.paths`, with no `os.chdir` and no absolute paths.
  - "Never report an experiment from an unsaved inline command… Exploratory notebook runs must be labeled exploratory"; "Experiment designs never go in `results/`… After execution, copy the resolved config into the results directory, but keep the planning documents in `experiment_design/`".
  - CLAUDE.md (senior-thesis: `CLAUDE.md`)
- **CLAUDE.md "Never commit"** (the repo is public): `lit/*.pdf` (copyrighted, plus the unpublished Zheng & Raskutti manuscript), `data/`, `results/` and `.claude/settings.local.json`. "Check `git status` before staging broadly" — CLAUDE.md (senior-thesis: `CLAUDE.md`); .gitignore (senior-thesis: `.gitignore`)
- **CLAUDE.md "Research work, not just code":**
  - Planned experiments and their decision rules go in `experiment_design/`.
  - "Anything you might want to cite needs a seed and a git SHA behind it".
  - `notebooks/` is exploration, and `mnist_exp.ipynb` and `simulated_datasets.ipynb` are "*expected* to be stale".
  - Recovery is scored automatically from the designs' true support, including `local_support(x)`.
  - CLAUDE.md (senior-thesis: `CLAUDE.md`); notebooks/README.md L8–16 (senior-thesis: `notebooks/README.md`)
- **CLAUDE.md "Open problems — do not paper over":** do not clamp a negative `phi_min`, which comes from non-monotonicity of the local value function. Threshold calibration is "the central open statistical question" and `select_threshold` "is an admitted heuristic" — CLAUDE.md (senior-thesis: `CLAUDE.md`); README.md L28–41 (senior-thesis: `README.md`)
- **Meeting notes are Spencer's prose.** "add only requested LaTeX structure (e.g. algorithm blocks), never prose" (uncommitted CLAUDE.md edit) — CLAUDE.md (senior-thesis: `CLAUDE.md`); `git diff CLAUDE.md`
- **Required skills.** Use `$experiment-design` before proposing a run ("Ask whether each cell changes a decision"), `$sensemaking` before trusting a result ("Name a falsifier") and `$skeptical-labmate` before claiming a method works. `$lab-notebook`, `$handoff` and `$slurm` apply once runs outgrow the laptop — CLAUDE.md, "Skills" (senior-thesis: `CLAUDE.md`)
- **Practices visible in the executed line**, beyond the written rules:
  - configs carry `name`, `status: executable` and `design_source:` pointing at the design file — experiments/configs/saliency_threshold_two_stage_confirmation.yaml L1–3 (senior-thesis: `experiments/configs/saliency_threshold_two_stage_confirmation.yaml`);
  - test seeds must be fresh, and the runner checks this against `previously_used_test_seeds` — run_saliency_threshold_two_stage.py L65–76 (senior-thesis: `experiments/run_saliency_threshold_two_stage.py`);
  - "No arm's levels or hyperparameters may be changed after the test worlds are opened; any follow-up uses new test seeds" — null_reference design §9 (senior-thesis: `experiment_design/saliency_threshold_null_reference/EXPERIMENT_DESIGN.md`);
  - pre-run amendments from smoke runs are recorded in the design before test worlds are opened — null_reference design §4 (senior-thesis: `experiment_design/saliency_threshold_null_reference/EXPERIMENT_DESIGN.md`);
  - post-hoc findings are re-registered and confirmed on fresh worlds — two_stage_confirmation design §2 (senior-thesis: `experiment_design/saliency_threshold_two_stage_confirmation/EXPERIMENT_DESIGN.md`);
  - the runner refuses to overwrite `--output` and stamps `config_sha256`, `git_sha` and `git_dirty` — run_saliency_threshold_function_family_benchmark.py L1510–1610 (senior-thesis: `experiments/run_saliency_threshold_function_family_benchmark.py`);
  - each design directory carries a LAB_NOTE with question, action, evidence, result, falsifier, confidence and decision impact — e.g. two_stage_confirmation LAB_NOTE (senior-thesis: `experiment_design/saliency_threshold_two_stage_confirmation/LAB_NOTE.md`).

### Inferences
- A genomics design would follow the local precedent most closely if it named its estimand and its null types up front. That means distinguishing the f̂-null from the E[Y|X]-null, and separating motif-absent, redundant/inactive and gate-like windows, as `local_nulls.py` does. Each guaranteed arm would then be judged only on the nulls its assumption covers, with registered claims evaluated by code.

### Gaps
- `results/INDEX.md` describes a "closed group" rule: `experiments.run` refuses to add to a group quiet for more than 7 days without `--extend`. That rule lives only on branch `design/complexity-vs-selection`. Its status on `main` is unclear.

## 7. Current git state (read-only commands)

### Takeaway
`main` is one commit ahead of `origin/main`. Most of the saliency-threshold line is uncommitted or untracked: the null-reference library, the 10 newer runners, 21 configs, `experiment_design/`, the 09-14 through 10-12 meeting-notes sections and the AI research notes. There are also three hygiene hazards:
- `lfs/data/function_family_benchmark.py` is gitignored and would silently stay out of any commit;
- `lfs/data/__init__.py` and `patches.py` are missing on `main`;
- the untracked `lit/AI Summaries/*.pdf` is not covered by the `lit/*.pdf` ignore rule, so a broad `git add -A` would publish it.

### Cited Findings
- **Branch and commits.** On branch `main`, "Your branch is ahead of 'origin/main' by 1 commit". HEAD is `be45906` "Add reproducible saliency threshold experiments", which added `run_saliency_threshold.py`, `run_saliency_threshold_all_designs.py`, three configs and two plot scripts, and touched `lfs/data/simulated.py`, `CLAUDE.md` and `notebooks/thresholding.ipynb`. `origin/main` is `04762e3` (`git status`; `git show --stat be45906`).
- **Modified, unstaged (8 files, +2380/−40).**
  - `CLAUDE.md` (18 lines changed) and `README.md` (8 lines changed) add the experiment-design boundary and the no-prose rule.
  - `experiments/run_saliency_threshold_all_designs.py` (+17): `second_gap` changes from `gaps[1]` (the second adjacent gap) to the second-largest gap, and a `train_loss_finite_fraction` diagnostic is added.
  - `lfs/models/__init__.py` (3 lines changed) and `lfs/models/nn.py` (30 lines changed) add `SoftplusMLP`, `module=` and module-kwargs passthrough.
  - `lfs/selection/saliency.py` (+45) adds `binary_margin_saliency` and `regression_saliency`.
  - `notebooks/thresholding.ipynb` (+1653).
  - `notes/meeting_notes.tex` (+646). HEAD holds only the sections through 06-04-2026; the 09-14, 09-21, 09-28 and 10-12 sections exist only in the working tree.
  - Sources: `git diff --stat`; `git diff`; `git show HEAD:notes/meeting_notes.tex | grep section`
- **Untracked.** `AGENTS.md`, `experiment_design/` (9 designs), `experiments/analyze_saliency_null_reference_boundary.py`, `experiments/analyze_saliency_two_stage_paired.py`, 19 `experiments/configs/saliency_*.yaml` (including 4 smoke configs), 10 `experiments/run_saliency_*.py`, `experiments/saliency_switch_distance.py`, `lfs/metrics/local_nulls.py`, `lfs/selection/null_reference.py`, `lit/AI Summaries/`, `notes/reports/`, `notes/research_notes/` and 7 test files (`git status`).
- **Ignored but required.** `lfs/data/function_family_benchmark.py` matches `.gitignore:42:data/` (`git check-ignore -v`). It is imported by at least 8 runners and analysis scripts (grep of `from lfs.data.function_family_benchmark`).
- **Missing on `main` but present on other refs.** `lfs/data/__init__.py` and `lfs/data/patches.py` exist on `design/complexity-vs-selection` (`7c4a06d`) and `origin/maintenance-pass-2026-08-21` (`6d8870b`). `experiments/run.py` cannot import. Three test files, with 45 test functions between them (`test_patches.py`, `test_runner.py`, `test_selection.py`), import the missing modules. The suite has 110 `def test_` functions in total, while README and CLAUDE.md say "103 tests" (grep count). The tests were not run.
- **Other refs.**
  - Local `design/complexity-vs-selection` (`2ab95d2`, 11 commits ahead of its origin) holds the gate, separability and complexity experiments plus `experiments/index.py` and `_convention.py`.
  - `experiment/complexity-saliency` (`73c38dd`) and `origin/maintenance-pass-2026-08-21` (`46f4149`).
  - Tags `pre-restructure` and `pre-restructure-backup` at `e9ed93b`.
  - `stash@{0}` "preserve pre-existing user work before thresholding cleanup", whose untracked files are `.codex-log/codex-tui.log`, `conditional_families.tex`, `conditional_response_designs.md` and `recommendation_episodes_memo.md`.
  - Sources: `git branch -a -vv`; `git log --all --graph`; `git stash list`; `git show --stat 'stash@{0}^3'`
- **Remote.** `origin` is `https://github.com/spencervenancio/senior-thesis.git` (`git remote -v`).
- **Ignore coverage of the PDFs.** The `.gitignore` rule `lit/*.pdf` covers `lit/2609.19511v1.pdf` and `lit/Go:No-GO.pdf`, which do not appear as untracked. `lit/AI Summaries/` does appear as untracked, so its PDF is not ignored (`git status`; .gitignore (senior-thesis: `.gitignore`)).

### Inferences
- Before any genomics experiment can claim "a seed and a git SHA behind it", someone would need to commit the saliency line, force-add or un-ignore `function_family_benchmark.py`, and restore `lfs/data/__init__.py` and `patches.py`. As things stand, every existing saliency result was produced with `git_dirty = True` by mostly untracked code. These commits are for Spencer to decide; I made none.
- Results produced before the `second_gap` fix may have used the second adjacent gap rather than the second-largest gap as summary g2. That affects at least the run produced with `run_saliency_threshold_all_designs.py`. Its practical effect on reported numbers is unknown.

### Gaps
- I could not determine whether the uncommitted `second_gap` change predates or postdates the `saliency_threshold_all_designs` runs. Their `meta.json` has a SHA and a dirty flag but no code hash.
- I did not inspect `.claude/settings.local.json` or the stash's `.codex-log` contents, which are outside scope.
