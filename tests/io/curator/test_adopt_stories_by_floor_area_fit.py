"""Per-unit adoption of roll story counts. Fabricated values only."""

import pandas as pd

from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState
from openplaces.io.curator.reconcilers import adopt_stories_by_floor_area_fit

SQFT = 1 / 0.09290304


def _state(df):
    return CurateState(
        recipe={},
        entity_recipe={},
        admin_id=AdminId('US', 'MA', 'XX'),
        verbose=False,
        timer=None,
        curated=df,
    )


def _frame(n, roll, nsi, floors_true, extra=None):
    # Footprints of 100 m2 each on their own parcel; the roll's floor
    # area matches floors_true.
    rows = {
        'n_stories': [nsi] * n,
        'n_stories_parcel': [roll] * n,
        'living_area_sqft_parcel': [100 * floors_true * SQFT] * n,
        'area_m2': [100.0] * n,
        'parcel_id': [f'p{i}' for i in range(n)],
        'priority_on_parcel': ['primary'] * n,
    }
    df = pd.DataFrame(rows)
    if extra is not None:
        df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
    return df


def test_present_counts_are_kept_by_default():
    df = _frame(120, roll=3, nsi=1, floors_true=3)
    out = adopt_stories_by_floor_area_fit(_state(df), min_rows=100).curated
    assert (out['n_stories'] == 1).all()


def test_missing_counts_are_filled_in_an_adopting_unit():
    df = _frame(120, roll=3, nsi=1, floors_true=3)
    df.loc[:9, 'n_stories'] = None
    out = adopt_stories_by_floor_area_fit(_state(df), min_rows=100).curated
    assert (out.loc[:9, 'n_stories'] == 3).all()
    assert (out.loc[:9, 'n_stories_source'] == 'parcel').all()
    assert (out.loc[10:, 'n_stories'] == 1).all()


def test_missing_counts_are_not_filled_in_a_non_adopting_unit():
    df = _frame(120, roll=3, nsi=1, floors_true=1)
    df.loc[:9, 'n_stories'] = None
    out = adopt_stories_by_floor_area_fit(_state(df), min_rows=100).curated
    assert out.loc[:9, 'n_stories'].isna().all()


def test_override_present_replaces_the_count_when_asked():
    df = _frame(120, roll=3, nsi=1, floors_true=3)
    out = adopt_stories_by_floor_area_fit(
        _state(df), min_rows=100, override_present=True
    ).curated
    assert (out['n_stories'] == 3).all()
    assert (out['n_stories_source'] == 'parcel').all()


def test_unit_where_nsi_fits_better_is_unchanged():
    df = _frame(120, roll=3, nsi=1, floors_true=1)
    out = adopt_stories_by_floor_area_fit(
        _state(df), min_rows=100, override_present=True
    ).curated
    assert (out['n_stories'] == 1).all()


def test_too_few_disagreeing_rows_is_unchanged():
    df = _frame(20, roll=3, nsi=1, floors_true=3)
    out = adopt_stories_by_floor_area_fit(
        _state(df), min_rows=100, override_present=True
    ).curated
    assert (out['n_stories'] == 1).all()


def test_parcel_with_two_primaries_is_never_changed():
    # Both rows of the shared parcel have no count; only the sole
    # primaries may be filled.
    extra = {
        'n_stories': [None, None],
        'n_stories_parcel': [3, 3],
        'living_area_sqft_parcel': [300 * SQFT, 300 * SQFT],
        'area_m2': [100.0, 100.0],
        'parcel_id': ['shared', 'shared'],
        'priority_on_parcel': ['primary', 'primary'],
    }
    df = _frame(120, roll=3, nsi=1, floors_true=3, extra=extra)
    df.loc[:9, 'n_stories'] = None
    out = adopt_stories_by_floor_area_fit(_state(df), min_rows=100).curated
    assert (out['n_stories'].iloc[:10] == 3).all()
    assert out['n_stories'].iloc[120:].isna().all()


def test_zero_and_implausible_roll_counts_are_ignored():
    # Rows 0-2 have no count to fill; the rest carry NSI's count, which
    # is what the unit's gate is measured on.
    df = _frame(120, roll=3, nsi=1, floors_true=3)
    df.loc[:2, 'n_stories'] = None
    df.loc[0, 'n_stories_parcel'] = 0
    df.loc[1, 'n_stories_parcel'] = 400
    out = adopt_stories_by_floor_area_fit(_state(df), min_rows=100).curated
    assert out['n_stories'].isna().iloc[0]
    assert out['n_stories'].isna().iloc[1]
    assert out['n_stories'].iloc[2] == 3


def test_a_unit_with_no_comparable_rows_adopts_nothing():
    # Every count missing: the gate has nothing to measure, so no fill.
    df = _frame(120, roll=3, nsi=1, floors_true=3)
    df['n_stories'] = None
    out = adopt_stories_by_floor_area_fit(_state(df), min_rows=100).curated
    assert out['n_stories'].isna().all()


def test_half_stories_round_up():
    # A roll's 2.5 is a three-count, not banker's 2.
    df = _frame(120, roll=2.5, nsi=1, floors_true=3)
    df.loc[:4, 'n_stories'] = None
    out = adopt_stories_by_floor_area_fit(_state(df), min_rows=100).curated
    assert (out.loc[:4, 'n_stories'] == 3).all()


def test_a_fractional_incumbent_that_rounds_to_the_roll_is_no_disagreement():
    # NSI's mean of 1.33 rounds to the roll's 1: nothing to test, so the
    # unit has no evidence and adopts nothing.
    df = _frame(120, roll=1, nsi=1.33, floors_true=1)
    df.loc[:4, 'n_stories'] = None
    out = adopt_stories_by_floor_area_fit(_state(df), min_rows=100).curated
    assert out.loc[:4, 'n_stories'].isna().all()


def test_missing_columns_is_a_no_op():
    df = _frame(5, roll=3, nsi=1, floors_true=3).drop(columns='n_stories_parcel')
    out = adopt_stories_by_floor_area_fit(_state(df)).curated
    assert (out['n_stories'] == 1).all()
