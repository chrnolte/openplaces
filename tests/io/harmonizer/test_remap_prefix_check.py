"""Crosswalk keys: longest prefix decides, and a merging cut is reported."""

import warnings

import pandas as pd
import pytest

from openplaces.io.harmonizer.links.discovery import (
    _match_longest_prefix,
    _report_merges,
)


def test_a_longer_key_splits_what_a_shorter_one_would_merge():
    codes = pd.Series(['F1', 'F2', 'F9', 'A1', 'Z'], dtype='string')
    keys = pd.Index(['F', 'F1', 'F2', 'A'])
    assert _match_longest_prefix(codes, keys).tolist() == ['F1', 'F2', 'F', 'A', pd.NA]


def test_a_refining_suffix_still_falls_under_its_base_code():
    codes = pd.Series(['101', '1010', '101R', '999'], dtype='string')
    out = _match_longest_prefix(codes, pd.Index(['101', '102']))
    assert out.tolist() == ['101', '101', '101', pd.NA]


def test_cutting_codes_the_source_always_writes_in_full_warns():
    codes = pd.Series(['F1'] * 5 + ['F2'] * 3 + ['A1'] * 4, dtype='string')
    matched = _match_longest_prefix(codes, pd.Index(['F', 'A']))
    with pytest.warns(UserWarning, match=r'F <- F1 \(5\), F2 \(3\)'):
        _report_merges(codes, matched, 'state-remap.csv')


def test_an_optional_suffix_beside_bare_codes_does_not_warn():
    # Half the codes are the bare base code: the key is the source's own.
    codes = pd.Series(['101'] * 5 + ['1010'] * 3 + ['101R'] * 2, dtype='string')
    matched = _match_longest_prefix(codes, pd.Index(['101']))
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        _report_merges(codes, matched, 'use-remap.csv')


def test_cutting_without_merging_distinct_codes_does_not_warn():
    codes = pd.Series(['F1'] * 4 + ['A1'] * 4, dtype='string')
    matched = _match_longest_prefix(codes, pd.Index(['F', 'A']))
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        _report_merges(codes, matched, 'state-remap.csv')
