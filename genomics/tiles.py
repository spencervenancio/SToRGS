"""Planted annotation roles, distinct from model or randomization-test nulls."""
import re

EMBEDDING = re.compile(r'pos-(\d+)_(ELF1_known2|SIX5_known1|AP1_disc3|TAL1_known1)-([ACGT]+)')
SIGNAL = {'ELF1', 'SIX5'}
ROLES = ('background', 'decoy', 'present_but_unused_motif', 'needed_motif')


def parse_embeddings(text, sequence):
    instances = []
    for token in text.split(',') if text else []:
        match = EMBEDDING.fullmatch(token)
        if not match:
            raise ValueError(f'Unrecognized embedding: {token}')
        start, name, bases = match.groups()
        start = int(start)
        end = start + len(bases)
        if sequence[start:end] != bases:
            raise ValueError(f'Planted string mismatch at {start}')
        instances.append(dict(motif=name.split('_')[0], start=start, end=end, sequence=bases))
    return instances


def annotate_tiles(sequence, instances, positive, width=20):
    if len(sequence) % width:
        raise ValueError('Tile width must divide sequence length')
    counts = {name: sum(m['motif'] == name for m in instances) for name in SIGNAL}
    if bool(positive) != all(counts.values()):
        raise ValueError('Task3 label does not match planted AND rule')
    tiles = []
    for start in range(0, len(sequence), width):
        end = start + width
        overlapping = [dict(m, overlap_bp=min(end, m['end']) - max(start, m['start']))
                       for m in instances if m['start'] < end and m['end'] > start]
        signal_bp = sum(m['overlap_bp'] for m in overlapping if m['motif'] in SIGNAL)
        decoy_bp = sum(m['overlap_bp'] for m in overlapping if m['motif'] not in SIGNAL)
        needed, unused = bool(positive and signal_bp), bool(not positive and signal_bp)
        role = ROLES[3] if needed else ROLES[2] if unused else ROLES[1] if decoy_bp else ROLES[0]
        essential = bool(positive and any(
            all(m['start'] < end and m['end'] > start for m in instances if m['motif'] == name)
            for name in SIGNAL))
        tiles.append(dict(tile_index=start // width, start=start, end=end,
                          tile_sequence=sequence[start:end], role=role,
                          has_needed_motif=needed, has_unused_motif=unused,
                          has_decoy=bool(decoy_bp), is_background=not overlapping,
                          mixed_role=bool(signal_bp and decoy_bp),
                          signal_overlap_bp=signal_bp, decoy_overlap_bp=decoy_bp,
                          annotation_tile_deletion_changes_label=essential,
                          elf1_copies=counts['ELF1'], six5_copies=counts['SIX5'],
                          motif_instances=overlapping))
    return tiles
