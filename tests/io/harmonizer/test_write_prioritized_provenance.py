"""Tests for `_write_prioritized`'s optional per-cell provenance tracking
(`provenance_token`), which stamps a `{name}_source` sidecar for exactly the
cells a call writes: gap-filled cells, or on a majority overwrite every cell
the incoming source holds a value for. Opt-in, so most callers never create
the sidecar at all. Values never depend on whether provenance is tracked.
"""

import pandas as pd

import openplaces.io.harmonizer.links as links
from openplaces.io.harmonizer import _ensure_object_source_column
from openplaces.recipe import get_recipe_by_id


def _source_list(series):
    """Normalize a source sidecar's NA cells (pd.NA or NaN) to plain None."""
    return [None if pd.isna(v) else v for v in series]


def test_new_column_with_token_stamps_every_written_cell():
    spine = pd.DataFrame(index=['a', 'b'])
    new_vals = pd.Series([1.0, 2.0], index=['a', 'b'])
    links._write_prioritized(spine, 'value', new_vals, provenance_token='county')
    assert spine['value'].tolist() == [1.0, 2.0]
    assert _source_list(spine['value_source']) == ['county', 'county']


def test_gap_fill_stamps_only_the_filled_cell():
    spine = pd.DataFrame({'value': [10.0, 20.0, None]}, index=['a', 'b', 'c'])
    # Sparse (1/3, below the default 0.5 majority threshold): only fills c's
    # gap, taking the gap-fill (not overwrite) branch.
    new_vals = pd.Series([None, None, 30.0], index=['a', 'b', 'c'])
    links._write_prioritized(spine, 'value', new_vals, provenance_token='county')
    assert spine['value'].tolist() == [10.0, 20.0, 30.0]
    assert _source_list(spine['value_source']) == [None, None, 'county']


def test_majority_overwrite_stamps_every_written_cell_including_equal_ones():
    spine = pd.DataFrame({'value': [10.0, 20.0, 30.0]}, index=['a', 'b', 'c'])
    # Majority coverage: overwrites outright. 'b' gets the value it
    # already had, and is still the incoming source's value.
    new_vals = pd.Series([11.0, 20.0, 31.0], index=['a', 'b', 'c'])
    links._write_prioritized(spine, 'value', new_vals, provenance_token='state')
    assert spine['value'].tolist() == [11.0, 20.0, 31.0]
    assert _source_list(spine['value_source']) == ['state', 'state', 'state']


def _statewide_then_roll():
    """A spine whose values a statewide layer supplied, and a county
    roll covering three of four parcels: one value differs, one equals
    the statewide value, and one parcel the roll leaves empty."""
    spine = pd.DataFrame(
        {
            'land_value': [100.0, 200.0, 300.0, 400.0],
            'land_value_source': pd.Series(['statewide'] * 4, dtype=object).values,
        },
        index=['a', 'b', 'c', 'd'],
    )
    roll = pd.Series([150.0, 200.0, 350.0, None], index=['a', 'b', 'c', 'd'])
    return spine, roll


def test_roll_overwrite_labels_every_cell_it_wrote_and_keeps_the_rest():
    spine, roll = _statewide_then_roll()
    links._write_prioritized(spine, 'land_value', roll, provenance_token='county')
    # 'b' equals the statewide value and is labeled the roll's; 'd',
    # which the roll left empty, keeps its value and its old label.
    assert _source_list(spine['land_value_source']) == [
        'county',
        'county',
        'county',
        'statewide',
    ]
    assert spine.loc['d', 'land_value'] == 400.0


def test_tracking_provenance_does_not_change_any_value():
    tracked, roll = _statewide_then_roll()
    untracked, _ = _statewide_then_roll()
    links._write_prioritized(tracked, 'land_value', roll, provenance_token='county')
    links._write_prioritized(untracked, 'land_value', roll)
    pd.testing.assert_series_equal(tracked['land_value'], untracked['land_value'])
    assert tracked['land_value'].tolist() == [150.0, 200.0, 350.0, 400.0]


def test_gap_fill_does_not_relabel_a_cell_whose_value_it_did_not_write():
    idx = ['a', 'b', 'c', 'd', 'e']
    spine = pd.DataFrame(
        {
            'value': [10.0, 20.0, None, None, None],
            'value_source': pd.Series(
                ['state', 'state', None, None, None], dtype=object
            ).values,
        },
        index=idx,
    )
    # Below majority (2 of 5): 'a' holds an equal value, but a gap
    # fill never writes over an existing cell, so its label stays.
    new_vals = pd.Series([10.0, None, 30.0, None, None], index=idx)
    links._write_prioritized(spine, 'value', new_vals, provenance_token='county')
    assert spine['value'].tolist()[:3] == [10.0, 20.0, 30.0]
    assert _source_list(spine['value_source']) == [
        'state',
        'state',
        'county',
        None,
        None,
    ]


def test_no_token_creates_no_source_column():
    spine = pd.DataFrame({'value': [10.0]}, index=['a'])
    new_vals = pd.Series([11.0], index=['a'])
    links._write_prioritized(spine, 'value', new_vals)
    assert 'value_source' not in spine.columns


def test_repeated_calls_only_move_the_sidecar_for_actually_changed_cells():
    spine = pd.DataFrame({'value': [None, None]}, index=['a', 'b'])
    links._write_prioritized(
        spine,
        'value',
        pd.Series([1.0, None], index=['a', 'b']),
        provenance_token='first',
    )
    # A second source only supplies 'b' -- 'a' must keep its original value
    # and provenance untouched.
    links._write_prioritized(
        spine,
        'value',
        pd.Series([None, 2.0], index=['a', 'b']),
        provenance_token='second',
    )
    assert spine['value'].tolist() == [1.0, 2.0]
    assert _source_list(spine['value_source']) == ['first', 'second']


def test_parcel_geospine_roll_link_tracks_value_provenance_and_tax_year():
    """The roll link overwrites the geometry source's values (US-TX-HAR),
    so it must label them and carry the roll's tax year with them."""
    recipe = get_recipe_by_id('US_parcel-geospine-2026')
    steps = [
        s
        for s in recipe['pipeline']
        if s.get('step') == 'link_by_id'
        and s.get('auto_discover')
        and s.get('entity_type') == 'property'
        and s.get('mode') == 'aggregate'
    ]
    assert len(steps) == 1
    step = steps[0]
    assert {'land_value', 'improvement_value', 'total_value'} <= set(
        step['track_provenance']
    )
    assert 'tax_year' in step['columns']


def test_ensure_object_source_column_casts_existing_numeric_column():
    # A prior join can leave an all-NaN float64 column behind (e.g. pandas
    # infers a numeric dtype for an all-missing column) without ever
    # populating it -- that dtype can't hold an arbitrary source-id string,
    # so it must be cast to object rather than crash on the next assignment.
    df = pd.DataFrame({'value_source': pd.Series([float('nan'), float('nan')])})
    assert pd.api.types.is_numeric_dtype(df['value_source'])
    _ensure_object_source_column(df, 'value_source')
    assert df['value_source'].dtype == object
    df.loc[0, 'value_source'] = 'county'
    assert df['value_source'].tolist()[0] == 'county'


def test_ensure_object_source_column_casts_existing_categorical_column():
    df = pd.DataFrame({'value_source': pd.Categorical(['a', 'a'])})
    _ensure_object_source_column(df, 'value_source')
    assert df['value_source'].dtype == object
    df.loc[0, 'value_source'] = 'county'
    assert df['value_source'].tolist() == ['county', 'a']
