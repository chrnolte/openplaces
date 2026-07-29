"""Tests for the parcel-id standardization engine in openplaces.geo.ids.

Covers the conversion operations that appear in the auto-selected best solutions,
and the hardened duplicate guard that keeps `parcel_id_local` from collapsing
distinct `parcel_id_assessor` values.
"""

import pandas as pd

from openplaces.geo.ids import (
    _adds_duplicates,
    compute_parcel_id_local,
    convert_parcel_id,
    dominant_parcel_id_pattern,
    simplest_parcel_id_pattern,
)


def test_simple_keeps_alphanumeric_uppercase():
    s = pd.Series(['r02520-011-008-000', 'R03600-005-440 -000', None])
    out = convert_parcel_id(s, conv_code='simple')
    assert out.iloc[0] == 'R02520011008000'
    assert out.iloc[1] == 'R03600005440000'  # internal space removed
    assert pd.isna(out.iloc[2])


def test_string_lengths_skip_empty():
    out = convert_parcel_id(
        pd.Series(['00123456789']), None, 'string_lengths: 3 3 5 & skip_empty: 1'
    )
    assert out.iloc[0] == '1|234|56789'  # leading zeros stripped per group


def test_split_groups_drop_cols():
    out = convert_parcel_id(
        pd.Series(['12-345']), '^([0-9]+)-([0-9]+)$', 'drop_cols: 0 & skip_empty: 1'
    )
    assert out.iloc[0] == '345'


def test_keep_length_merge_after():
    out = convert_parcel_id(
        pd.Series(['001-002-003']),
        '^([0-9]+)-([0-9]+)-([0-9]+)$',
        'keep_length: 0 & merge_after: 0 & skip_empty: 1',
    )
    # group 0 '001' kept (not zero-stripped) + stripped group 1 '2' -> '0012'; '3'
    assert out.iloc[0] == '0012|3'


def test_no_conv_uppercases_only():
    out = convert_parcel_id(pd.Series(['abc-123']), None, 'no_conv')
    assert out.iloc[0] == 'ABC-123'


def test_adds_duplicates_detects_collapse():
    raw = pd.Series(['A1', 'A2', 'B1'])
    assert _adds_duplicates(raw, pd.Series(['A', 'A', 'B']), 0.0)
    assert not _adds_duplicates(raw, pd.Series(['A1', 'A2', 'B1']), 0.0)


def test_simplest_pattern_picks_lowest_complexity_that_clears_both_gates():
    # A 2-segment key ('A-1') would collide between the first two rows;
    # the simplest pattern that avoids that is the 3-segment one.
    s = pd.Series(['A-1-2', 'A-1-3', 'A-2-2'])
    assert simplest_parcel_id_pattern(s) == 'Sx-Sx-Sx'


def test_simplest_pattern_rejects_colliding_low_complexity_candidate():
    s = pd.Series(['A-1-2', 'A-1-3'])
    pattern = simplest_parcel_id_pattern(s)
    # Whatever pattern wins, it must not collapse the two distinct raw ids.
    candidate = convert_parcel_id(
        s.astype('string').str.strip().str.upper(), pattern, 'skip_empty: 1'
    )
    assert candidate.nunique() == 2


def test_simplest_pattern_falls_back_to_dominant_when_ratio_gate_unclearable():
    # An unreachable min_match_ratio means every candidate is skipped in the
    # ladder walk -- must fall back to dominant_parcel_id_pattern's result
    # for the same series/ratio rather than raising or returning nothing.
    s = pd.Series(['A-1-2', 'A-1-3', 'A-2-2'])
    assert simplest_parcel_id_pattern(
        s, min_match_ratio=1.1
    ) == dominant_parcel_id_pattern(s, min_match_ratio=1.1)


def test_compute_falls_back_when_conversion_collapses():
    raw = pd.Series(['R001', 'R002', 'R003'])
    # This instruction would extract only the leading 'R', collapsing all rows.
    instruction = {'X': {'pattern': '^(R)[0-9]+$', 'conv': 'skip_empty: 1'}}
    out = compute_parcel_id_local(
        raw, 'X', instruction=instruction, kind='parcel', tolerance=0.0
    )
    # Guard rejects the collapsing conversion and falls back -> distinct keys.
    assert out.nunique() == 3


def test_instruction_tolerance_overrides_default_and_avoids_fallback():
    # Two legitimately-the-same-parcel raw ids (leading zero on the first
    # segment differs), plus one genuinely distinct id -- collapsing the
    # first pair is the intended standardization (mirrors real Vilas County
    # WI data), not a bug, but it exceeds the default 0.5% tolerance for a
    # 3-row sample.
    raw = pd.Series(['010-1044', '10-1044', '10-1057'])
    pattern = 'Sx-[S.]x(-Sx)(-Sx)(-Sx)'
    conv = 'skip_empty: 1'

    tight = {'X': {'pattern': pattern, 'conv': conv}}
    fallback = compute_parcel_id_local(raw, 'X', instruction=tight, kind='parcel')
    # Guard rejects the collapsing conversion at the default tolerance and
    # falls back to raw ids -- all three stay distinct.
    assert fallback.nunique() == 3

    widened = {'X': {'pattern': pattern, 'conv': conv, 'tolerance': 0.5}}
    out = compute_parcel_id_local(raw, 'X', instruction=widened, kind='parcel')
    assert out.nunique() == 2
    assert out.iloc[0] == out.iloc[1] == '10|1044'
