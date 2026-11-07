# Read this first: where things stand and what to do tonight

## 1. What was actually done today

- **Only reading and searching.** No experiment has been run, and nothing in your repo was changed.
- The outputs are five sets of research notes (literature, ground truth, statistics, tooling, repo survey) and one long report.
- A few tools were timed in a scratch environment to see what is feasible on your laptop. Those timings are **not results**.

## 2. Your two concerns, answered

**"Existing genomics methods are already local, so what's left?"**
- You're right that they're local. They already give a score per DNA position, and they already pick out per-sequence sets: "seqlets" (high-scoring stretches of the map) and "motif hits".
- What they don't do is say how many of those picks are wrong. Their thresholds are heuristics or dataset-wide averages. One tool's own documentation says its p-values "are not error rates".
- **The gap:** pick the important regions of *one* sequence with a stated false-discovery rate.

**"Is there ground truth?"**
- **Yes:** synthetic DNA with motifs planted at known positions. You know exactly which regions matter in each sequence, the same way your simulated functions know `local_support(x)`.
- Lab mutation data exists for later, but it is noisy and not needed tonight.

## 3. The next step: one experiment

**Question:** for a single sequence, can we select the important 20-bp regions while controlling the false-discovery rate, and does the field's usual threshold already do that?

**Data:** the DFIM synthetic dataset (Greenside et al. 2018, github.com/kundajelab/dfim, about 8 MB).
- 60,000 random DNA sequences, each 200 bp long.
- Motif ELF1, motif SIX5, or both are planted in each. The label "both present" is an AND gate, like your gated designs.
- Two decoy motifs are planted at random and never affect the label.
- Every planted position is recorded.

**Steps:**
1. **Train a small CNN** on that data. This takes minutes on your laptop.
2. **Cut each test sequence into 10 tiles of 20 bp.** From the planted positions, label each tile as background, decoy, a motif the label needs, or a motif that is present but doesn't matter here (for example ELF1 without SIX5).
3. **Run three selection rules on each sequence:**
   - **(a) Your fixed relative cutoff** on saliency.
   - **(b) The field's approach:** a single threshold set across all sequences using shuffled DNA.
   - **(c) A per-sequence permutation test:** replace one tile with random DNA 999 times, check how unusual the real tile's effect on the prediction is, and apply BH within that sequence.
4. **Measure, per sequence:** the false-discovery rate and how many true motifs each rule finds, separately for each tile type.

**What the outcomes mean:**
- If (b) breaks down per sequence and (c) holds, then per-sequence calibration is your contribution.
- If (b) is fine too, the contribution shifts to the hard cases: motifs that are present but not used, and redundant copies.
- Either way, you expect "present but not used" tiles to be the failure case. That is your existing open problem, now with biological names: cooperativity and redundancy.

Compute is roughly an hour. Writing the code is most of the evening.

## 4. Fix before coding

- **Put new code in a new `lfs/genomics/` folder.** `.gitignore` line 42 (`data/`) silently ignores anything new in `lfs/data/`.
- **`lit/AI Summaries/` is not gitignored.** Don't run `git add -A`.

## 5. One sentence for the meeting

"Genomics tools already make per-sequence selections but give no error rate for them. I'm testing, on a planted-motif benchmark, whether a per-sequence permutation test controls that error where the field's pooled threshold doesn't."

*(Caveat: this hasn't been checked against 2025–26 preprints yet, so call it "not found in the literature so far", not "novel".)*

The long report (`Regulatory genomics local feature selection.md`) is reference material only. You don't need to read it.
