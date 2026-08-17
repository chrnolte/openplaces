"""Tests for the parcel-id standardization engine in openplaces.geo.ids.

Covers the conversion operations that appear in the auto-selected best solutions,
and the hardened duplicate guard that keeps `parcel_id_local` from collapsing
distinct `parcel_id_assessor` values.
"""

import pandas as pd
import pytest

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


def test_pipe_replaces_separators_instead_of_deleting():
    s = pd.Series(['00015 002 000', '00015-002-000', None])
    out = convert_parcel_id(s, conv_code='pipe')
    assert out.iloc[0] == '00015|002|000'
    assert out.iloc[1] == '00015|002|000'  # space and dash both normalize to '|'
    assert pd.isna(out.iloc[2])


def test_pipe_keeps_ids_distinct_that_simple_would_collapse():
    # '1-23' and '12-3' both strip to '123' under 'simple', colliding two
    # otherwise-distinct raw ids; 'pipe' keeps the segment boundary.
    s = pd.Series(['1-23', '12-3'])
    simple_out = convert_parcel_id(s, conv_code='simple')
    pipe_out = convert_parcel_id(s, conv_code='pipe')
    assert simple_out.iloc[0] == simple_out.iloc[1] == '123'
    assert pipe_out.iloc[0] == '1|23'
    assert pipe_out.iloc[1] == '12|3'
    assert pipe_out.iloc[0] != pipe_out.iloc[1]


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


def test_compute_falls_back_when_conversion_matches_nothing():
    """A pattern that fits no row must not yield an all-null key.

    This is the failure the duplicate guard cannot see: it compares only
    rows where both raw and candidate are non-null, so an all-null
    candidate leaves it an empty mask and a clean verdict. Downstream the
    null key silently drops the whole table wherever it is joined by
    parcel_id_local -- observed on four NC counties whose bundled pattern
    expected dashes their PIN does not have.
    """
    raw = pd.Series(['6945835847', '5977190758', '5977298749'])
    # Written for a dashed id; this source has no separators at all.
    instruction = {'X': {'pattern': 'Sx-Sx-Sx', 'conv': 'skip_empty: 1'}}

    with pytest.warns(UserWarning, match='falling back to pipe'):
        out = compute_parcel_id_local(raw, 'X', instruction=instruction)

    # No separators to keep, so pipe and simple agree here.
    assert out.notna().all()
    assert out.tolist() == ['6945835847', '5977190758', '5977298749']


def test_compute_keeps_a_conversion_that_only_drops_a_few_rows():
    """A deliberate partial conversion is left alone.

    MassGIS-style instructions legitimately fail on a minority of rows; at
    that scale the instruction still fits the source and must not be
    replaced.
    """
    raw = pd.Series([f'{100 + i}-{200 + i}' for i in range(19)] + ['NOMATCHHERE'])
    instruction = {'X': {'pattern': r'^(\d+)-(\d+)$', 'conv': 'skip_empty: 1'}}

    out = compute_parcel_id_local(raw, 'X', instruction=instruction)

    # One row of twenty (5%) fails to convert -- under the fallback bar.
    assert out.isna().sum() == 1
    assert out.iloc[0] == '100|200'


def test_compute_warns_without_falling_back_on_moderate_loss():
    """Between the warn floor and the fallback bar, warn but keep the key."""
    raw = pd.Series(
        [f'{100 + i}-{200 + i}' for i in range(15)] + [f'NOMATCH{i}' for i in range(5)]
    )
    instruction = {'X': {'pattern': r'^(\d+)-(\d+)$', 'conv': 'skip_empty: 1'}}

    with pytest.warns(UserWarning, match='will not join'):
        out = compute_parcel_id_local(raw, 'X', instruction=instruction)

    # 25% lost: warned about, but the conversion is still the one used.
    assert out.isna().sum() == 5
    assert out.iloc[0] == '100|200'


def test_max_loss_is_tunable():
    """The fallback bar can be tightened by the caller.

    The fallback lands on ``pipe`` rather than ``simple``: concatenating
    segments can fuse two otherwise-distinct raw ids, so the separator is
    kept whenever pipe clears the duplicate guard (see the fallback order
    in ``geo/ids.py``). ``simple`` remains the last resort behind it.
    """
    raw = pd.Series(
        [f'{100 + i}-{200 + i}' for i in range(15)] + [f'NOMATCH{i}' for i in range(5)]
    )
    instruction = {'X': {'pattern': r'^(\d+)-(\d+)$', 'conv': 'skip_empty: 1'}}

    with pytest.warns(UserWarning, match='falling back to pipe'):
        out = compute_parcel_id_local(raw, 'X', instruction=instruction, max_loss=0.1)

    assert out.notna().all()
    assert out.iloc[0] == '100|200'


def test_degenerate_source_column_is_reported():
    """A populated-but-useless key is worse than a null one, so say so.

    Falling back cannot invent precision the source never had: NC OneMap's
    ALTPARNO in Pamlico County is a block code, 140 distinct values across
    17,109 parcels. Both pipe and simple reproduce that faithfully and the
    duplicate guard passes, because the duplicates come from the source.
    """
    raw = pd.Series([f'{i % 3}' for i in range(60)])
    instruction = {'X': {'pattern': 'Sx-Sx-Sx', 'conv': 'skip_empty: 1'}}

    with pytest.warns(UserWarning, match='the source column is not a parcel-level id'):
        out = compute_parcel_id_local(raw, 'X', instruction=instruction)

    assert out.notna().all()
    assert out.nunique() == 3
