# DFIM CNN: model and training summary

## Task and data
Predict whether **both ELF1 and SIX5 motifs are present** in a synthetic DNA
sequence (DFIM `task3`). Input: 200 bases, encoded as a 4 × 200 one-hot array
with channels A, C, G, T. Motif positions are withheld from the model and
reserved for later feature-selection evaluation.

All 60,000 sequences were used: 42,000 training, 9,000 validation, and 9,000
test. Each split contains equal numbers of ELF1-only, SIX5-only, and both-motif
examples, making one-third positive. Splitting was random within these groups
(seed 20261025). No exact duplicate sequences were found.

## Architecture
**18,465 trainable parameters.**

| Layer | Settings | Output shape, excluding batch |
|---|---|---|
| Input | One-hot DNA | 4 × 200 |
| Convolution + ReLU | 32 filters, width 15, padding 7 | 32 × 200 |
| Convolution + ReLU | 64 filters, width 7, padding 3 | 64 × 200 |
| Global max pooling | Maximum over positions, per channel | 64 |
| Dense + ReLU | 64 → 32 | 32 |
| Dense | 32 → 1 | One logit |

Convolutions have stride 1, dilation 1, and biases; dense layers also have
biases. Sigmoid converts the logit to a probability. No dropout or batch normalization.

The convolutions learn short sequence patterns. Pooling records whether each
learned pattern appears strongly anywhere in the sequence; dense layers combine
these signals to predict the AND label. This is the architecture's intent, not
proof of which motifs the trained filters recognize.

## Training
- Binary cross-entropy on logits; no class weighting.
- Adam: learning rate 0.001, betas (0.9, 0.999), epsilon 1e-8; no weight decay.
- Batch size 256, shuffled each epoch; 20 epochs.
- PyTorch Kaiming-uniform initialization; seed 20261025.
- Checkpoint selected by lowest validation loss: **epoch 20**.
- No augmentation, learning-rate schedule, early stopping, or hyperparameter search.
- CPU float32 training took approximately **144 seconds**.

## Held-out performance
| Measure | Test result |
|---|---:|
| Accuracy | 99.989% (8,999 / 9,000 correct) |
| AUROC | 1.000 |
| Binary cross-entropy | 0.0002503 |
| False positives / false negatives | 1 / 0 |

The majority-class baseline achieves 66.7% accuracy. Test data were evaluated
after checkpoint selection. This establishes strong prediction on this synthetic
benchmark; it does **not** establish correct attribution or FDR control.

## Reproducibility and further detail
A second complete run reproduced every weight and prediction bit for bit on the
same environment: Python 3.10.13, PyTorch 2.11.0, NumPy 1.26.4, macOS ARM64.
Training used deterministic CPU operations and one thread. For speed, convolution
used an equivalent window-and-matrix implementation, checked against native
Conv1d outputs and input gradients. That arithmetic is recorded because it can
affect exact retraining.

The checkpoint recovers the exact trained weights. The full specification covers
initialization order, splits, arithmetic, optimizer settings, and input hashes.
Independent reconstruction from prose alone has not been tested.

- [Full specification](../../results/genomics_dfim_cnn/MODEL_REPORT.md)
- [Handoff ZIP: weights, code, config, environment, splits, and original data](../../results/genomics_dfim_cnn/dfim_cnn_handoff.zip)

The run used a dirty working tree; exact source snapshots accompany the checkpoint.

## Why prediction is nearly perfect
A follow-up audit found that ELF1 and SIX5 are each planted as one fixed DNA
string throughout this dataset. Checking whether both strings occur classifies
all 60,000 sequences perfectly, including every test example. This is an easy
synthetic pattern-recognition task, not representative of real genomics prediction.
The saved logits reproduce the reported metrics, and split indices do not overlap.
Perfect AUROC with one classification error is consistent: all positive logits
exceed all negative logits, but one negative logit is above the fixed zero cutoff.
Audit artifacts: results/genomics_dfim_cnn/performance_audit.json and
experiments/audit_dfim_cnn.py. This audit does not establish motif-specific model reliance.
