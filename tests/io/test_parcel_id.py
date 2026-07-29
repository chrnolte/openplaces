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


def test_compute_falls_back_when_conversion_collapses():
    raw = pd.Series(['R001', 'R002', 'R003'])
    # This instruction would extract only the leading 'R', collapsing all rows.
    instruction = {'X': {'pattern': '^(R)[0-9]+$', 'conv': 'skip_empty: 1'}}
    out = compute_parcel_id_local(
        raw, 'X', instruction=instruction, kind='parcel', tolerance=0.0
    )
    # Guard rejects the collapsing conversion and falls back -> distinct keys.
    assert out.nunique() == 3
