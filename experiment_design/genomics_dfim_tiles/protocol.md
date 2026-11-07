# Annotate existing DFIM test split
Use the frozen model's exact 9,000 test indices; no resplitting or model access.
Ten nonoverlapping 20-bp tiles, zero-based half-open coordinates [0,20),...,[180,200).
Any positive base overlap marks a motif as overlapping a tile. Verify planted
strings match FASTA and metadata sequences and labels match motif presence.
Primary role priority: needed_motif > present_but_unused_motif > decoy > background.
Preserve independent flags, overlap base counts and full motif-instance records.
Needed means ELF1/SIX5 in a task3-positive sequence, a union-of-planted-motifs
convention, not single-copy necessity or a fitted-model null. Negative-sequence
ELF1/SIX5 are unused under the planned positive-AND convention, not proof of no
model effect under arbitrary substitutions. Preserve copy counts and a separate
annotation-level tile-deletion flag (all copies of at least one required motif
intersect this tile). It assumes any overlapping deletion disrupts a motif and
ignores newly created motifs; it is not a permutation-replacement ground truth.
Checks: 90,000 rows, exactly ten contiguous tiles per sequence, correct boundary
and mixed-type handling; deterministic rerun, input hashes, exact config, source
snapshots and reproduction README. No selection or FDR claim.
