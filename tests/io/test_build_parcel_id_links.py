"""Tests for the offline parcel_id_local link generator in
openplaces.geo.build_parcel_id_links.

Covers the cross-dataset pattern/conversion search (`find_best_parcel_id_link`)
and its separator-preservation tie-break, plus the ported conversion-option
candidate library.
"""

import pandas as pd

from openplaces.geo.build_parcel_id_links import (
    _BARE_CANDIDATES,
    _conv_option_candidates,
    find_best_parcel_id_link,
    propose_parcel_id_overrides,
)
from openplaces.geo.ids import convert_parcel_id


def test_conv_option_candidates_always_include_bare_pipe_and_simple():
    # Even for a pattern with no curated options of its own, both bare
    # whole-string ops must be tried.
    candidates = _conv_option_candidates('Empty')
    assert candidates == _BARE_CANDIDATES


def test_pipe_beats_simple_when_separator_removal_collapses_ids():
    # 'A#1-2' and 'A#12' are genuinely distinct raw ids, but stripping all
    # separators (as 'simple' does) collapses both to 'A12' -- exactly the
    # collision the pipe-preference principle exists to avoid.
    pc = pd.Series(['A#1-2', 'A#12'])
    za = pd.Series(['A#1-2', 'A#999'])

    simple_pc = convert_parcel_id(pc, None, 'simple')
    pipe_pc = convert_parcel_id(pc, None, 'pipe')
    assert simple_pc.nunique() == 1  # both collapse to 'A12'
    assert pipe_pc.nunique() == 2  # 'A|1|2' and 'A|12' stay distinct

    result = find_best_parcel_id_link(pc, za)
    # 'simple' can only ever match 0/2 here (both pc rows are duplicates of
    # each other post-conversion); the winner must do strictly better.
    assert result['success'] == 0.5
    assert result['conv_pc'] != 'simple'
    assert result['conv_za'] != 'simple'


def test_recovers_real_wi_pin_standardization():
    # 'SC-612-10' -> 'SC|612|10' is the real, production-validated
    # standardization for WI-ON PINs (see
    # tests/io/ingester/test_parcel_id_overrides.py::
    # test_tax_and_parcel_side_conversions_agree_for_oneida_and_vilas).
    # The generator should independently rediscover a conversion that
    # reproduces it from raw data alone, with no pattern/conv given.
    pc = pd.Series(['SC-612-10', 'SC-613-05', 'TN-100-01', 'TN-100-22'])
    za = pd.Series(['SC-612-10', 'SC-613-05', 'ZZ-999-99'])

    result = find_best_parcel_id_link(pc, za)
    assert result['success'] == 0.5
    assert result['conv_pc'] != 'simple'

    converted = convert_parcel_id(pc, result['pattern_pc'], result['conv_pc'])
    assert converted.iloc[0] == 'SC|612|10'


def test_propose_parcel_id_overrides_returns_empty_for_uncovered_admin_id():
    # No recipe covers this made-up admin unit -- must return an empty
    # frame with the documented schema, not raise.
    out = propose_parcel_id_overrides(['ZZ-ZZ-ZZ'])
    assert out.empty
    assert list(out.columns) == [
        'admin_id',
        'kind',
        'pattern',
        'conv',
        'success',
        'cross_validated',
        'source_entity_type',
        'source_id',
    ]
