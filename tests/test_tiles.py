import pytest

from genomics.tiles import annotate_tiles, parse_embeddings


def motif(name, start, end):
    return dict(motif=name, start=start, end=end, sequence='A' * (end - start))


def test_boundaries_mixed_and_background():
    annotations = [motif('ELF1', 18, 23), motif('SIX5', 40, 45), motif('AP1', 21, 25)]
    tiles = annotate_tiles('A' * 200, annotations, True)
    assert len(tiles) == 10
    assert tiles[0]['signal_overlap_bp'] == 2
    assert tiles[1]['signal_overlap_bp'] == 3
    assert tiles[1]['mixed_role'] and tiles[1]['has_decoy']
    assert tiles[1]['role'] == 'needed_motif'
    assert tiles[1]['annotation_tile_deletion_changes_label']
    assert not tiles[1]['is_background']
    assert tiles[2]['signal_overlap_bp'] == 5
    assert tiles[3]['is_background']


def test_unused_decoy_and_redundancy():
    t = annotate_tiles('A' * 200, [motif('ELF1', 0, 5), motif('AP1', 40, 45)], False)
    assert t[0]['role'] == 'present_but_unused_motif'
    assert t[2]['role'] == 'decoy'
    assert not any(r['annotation_tile_deletion_changes_label'] for r in t)
    t = annotate_tiles('A' * 200, [motif('ELF1', 0, 5), motif('ELF1', 40, 45),
                                motif('SIX5', 80, 85)], True)
    assert t[0]['role'] == 'needed_motif'
    assert not t[0]['annotation_tile_deletion_changes_label']
    assert t[4]['annotation_tile_deletion_changes_label']


def test_parse_and_invalid_annotations():
    assert parse_embeddings('pos-0_ELF1_known2-AAA', 'AAAA')[0]['end'] == 3
    with pytest.raises(ValueError):
        parse_embeddings('pos-0_ELF1_known2-AAA', 'CCCC')
    with pytest.raises(ValueError):
        annotate_tiles('A' * 200, [motif('ELF1', 0, 5)], True)
