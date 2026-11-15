"""The one step deciding a building's number of stories. Fabricated
values only."""

import numpy as np
import pandas as pd

from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState
from openplaces.io.curator.estimators.stories import decide_stories
from openplaces.io.steps import resolve_step_columns

SQFT = 1 / 0.09290304
EVIDENCE = [
    'n_stories_building_nsi',
    'n_stories_footprint_fema',
    'n_stories_building_cheer',
]


def _state(df):
    return CurateState(
        recipe={},
        entity_recipe={},
        admin_id=AdminId('US', 'MA', 'XX'),
        verbose=False,
        timer=None,
        curated=df,
    )


def _decide(df, **params):
    return decide_stories(_state(df), evidence=EVIDENCE, **params).curated


def _lanes(nsi, height=np.nan, cheer=np.nan):
    """One row per entry, each building alone on its own lot."""
    n = len(nsi)
    return pd.DataFrame(
        {
            'n_stories_building_nsi': nsi,
            'height_footprint_fema': [height] * n if np.isscalar(height) else height,
            'n_stories_building_cheer': [cheer] * n if np.isscalar(cheer) else cheer,
        }
    )


def _roll_frame(n, roll, nsi, floors_true, extra=None):
    # Buildings of 100 m2 each on their own lot; the roll's floor area
    # matches floors_true.
    rows = {
        'n_stories_building_nsi': [nsi] * n,
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


def test_first_available_lane_wins_in_order():
    df = _lanes(nsi=[2.0, np.nan, np.nan, np.nan], height=[14.0, 14.0, np.nan, np.nan])
    df['n_stories_building_cheer'] = [1.0, 1.0, 1.0, np.nan]
    out = _decide(df)
    assert out['n_stories'].tolist()[:3] == [2, 4, 1]
    assert pd.isna(out['n_stories'].iloc[3])
    assert out['n_stories_source'].tolist()[:3] == ['nsi', 'fema', 'cheer']
    assert pd.isna(out['n_stories_source'].iloc[3])


def test_imagery_lane_is_off_until_listed_as_evidence():
    # The street-imagery count is gathered but not a lane by default. A
    # test or research run lists it in evidence, and its cells then say
    # so in the source column.
    df = _lanes(nsi=[2.0, np.nan])
    df['n_stories_brails'] = [3.0, 3.0]
    off = _decide(df.copy())
    assert off['n_stories'].iloc[0] == 2
    assert pd.isna(off['n_stories'].iloc[1])
    on = decide_stories(
        _state(df.copy()), evidence=['n_stories_brails', *EVIDENCE]
    ).curated
    assert on['n_stories'].tolist() == [3, 3]
    assert (on['n_stories_source'] == 'brails').all()


def test_out_of_range_evidence_falls_through():
    df = _lanes(nsi=[219.0, 0.4, 111.0], height=7.0)
    out = _decide(df)
    assert out['n_stories'].tolist() == [2, 2, 2]
    assert (out['n_stories_source'] == 'fema').all()


def test_range_bounds_are_inclusive():
    df = _lanes(nsi=[1.0, 110.0])
    out = _decide(df)
    assert out['n_stories'].tolist() == [1, 110]


def test_half_values_round_up_once():
    # An NSI mean of 2.5 is 3 and of 3.5 is 4 (half to even gives 2, 4).
    df = _lanes(nsi=[1.5, 2.5, 3.5, 1.33])
    out = _decide(df)
    assert out['n_stories'].tolist() == [2, 3, 4, 1]
    assert str(out['n_stories'].dtype) == 'Int64'


def test_height_converts_at_three_and_a_half_meters_half_up():
    df = _lanes(nsi=[np.nan] * 3, height=[8.75, 3.0, -1.0])
    out = _decide(df)
    # 8.75 / 3.5 = 2.5 -> 3; 3.0 / 3.5 -> at least 1; negative -> none.
    assert out['n_stories_footprint_fema'].tolist()[:2] == [3.0, 1.0]
    assert pd.isna(out['n_stories_footprint_fema'].iloc[2])
    assert out['n_stories'].tolist()[:2] == [3, 1]
    assert pd.isna(out['n_stories'].iloc[2])


def test_floor_height_is_a_parameter():
    df = _lanes(nsi=[np.nan], height=10.0)
    out = _decide(df, floor_height_m=5.0)
    assert out['n_stories'].iloc[0] == 2


def test_absent_evidence_columns_are_skipped():
    df = pd.DataFrame({'n_stories_building_cheer': [2.0]})
    out = _decide(df)
    assert out['n_stories'].iloc[0] == 2
    assert out['n_stories_source'].iloc[0] == 'cheer'


def test_a_stale_source_on_an_empty_row_is_cleared():
    df = _lanes(nsi=[np.nan])
    df['n_stories_source'] = ['nsi']
    out = _decide(df)
    assert pd.isna(out['n_stories'].iloc[0])
    assert pd.isna(out['n_stories_source'].iloc[0])


def test_roll_fills_missing_counts_in_an_adopting_unit():
    df = _roll_frame(120, roll=3, nsi=1, floors_true=3)
    df.loc[:9, 'n_stories_building_nsi'] = None
    out = _decide(df)
    assert (out.loc[:9, 'n_stories'] == 3).all()
    assert (out.loc[:9, 'n_stories_source'] == 'parcel').all()
    # Present counts are never overwritten.
    assert (out.loc[10:, 'n_stories'] == 1).all()
    assert (out.loc[10:, 'n_stories_source'] == 'nsi').all()


def test_roll_does_not_fill_in_a_unit_where_it_fits_worse():
    df = _roll_frame(120, roll=3, nsi=1, floors_true=1)
    df.loc[:9, 'n_stories_building_nsi'] = None
    out = _decide(df)
    assert out.loc[:9, 'n_stories'].isna().all()


def test_roll_needs_min_rows_and_min_share():
    df = _roll_frame(20, roll=3, nsi=1, floors_true=3)
    df.loc[:4, 'n_stories_building_nsi'] = None
    assert _decide(df).loc[:4, 'n_stories'].isna().all()
    assert (_decide(df, min_rows=10).loc[:4, 'n_stories'] == 3).all()
    assert _decide(df, min_rows=10, min_share=1.01).loc[:4, 'n_stories'].isna().all()


def test_roll_fills_only_the_sole_building_of_a_lot():
    extra = {
        'n_stories_building_nsi': [None, None],
        'n_stories_parcel': [3, 3],
        'living_area_sqft_parcel': [300 * SQFT, 300 * SQFT],
        'area_m2': [100.0, 100.0],
        'parcel_id': ['shared', 'shared'],
        'priority_on_parcel': ['primary', 'primary'],
    }
    df = _roll_frame(120, roll=3, nsi=1, floors_true=3, extra=extra)
    df.loc[:9, 'n_stories_building_nsi'] = None
    out = _decide(df)
    assert (out['n_stories'].iloc[:10] == 3).all()
    assert out['n_stories'].iloc[120:].isna().all()


def test_roll_zero_and_implausible_counts_are_ignored():
    df = _roll_frame(120, roll=3, nsi=1, floors_true=3)
    df.loc[:2, 'n_stories_building_nsi'] = None
    df.loc[0, 'n_stories_parcel'] = 0
    df.loc[1, 'n_stories_parcel'] = 400
    out = _decide(df)
    assert pd.isna(out['n_stories'].iloc[0])
    assert pd.isna(out['n_stories'].iloc[1])
    assert out['n_stories'].iloc[2] == 3


def test_roll_half_stories_round_up():
    df = _roll_frame(120, roll=2.5, nsi=1, floors_true=3)
    df.loc[:4, 'n_stories_building_nsi'] = None
    out = _decide(df)
    assert (out.loc[:4, 'n_stories'] == 3).all()


def test_a_fractional_incumbent_that_rounds_to_the_roll_is_no_disagreement():
    # NSI's mean of 1.33 rounds to the roll's 1: nothing to test, so the
    # unit has no evidence and adopts nothing.
    df = _roll_frame(120, roll=1, nsi=1.33, floors_true=1)
    df.loc[:4, 'n_stories_building_nsi'] = None
    out = _decide(df)
    assert out.loc[:4, 'n_stories'].isna().all()
    assert (out.loc[5:, 'n_stories'] == 1).all()


def test_missing_roll_columns_skip_the_gap_fill():
    df = _roll_frame(120, roll=3, nsi=1, floors_true=3)
    df.loc[:9, 'n_stories_building_nsi'] = None
    out = _decide(df.drop(columns='living_area_sqft_parcel'))
    assert out.loc[:9, 'n_stories'].isna().all()
    assert _decide(df, roll_column=None).loc[:9, 'n_stories'].isna().all()


def test_declaration_resolves_from_the_recipe_parameters():
    resolved = resolve_step_columns(decide_stories, {'evidence': EVIDENCE})
    assert resolved.decides == frozenset({'n_stories'})
    assert {'n_stories', 'n_stories_source', 'n_stories_footprint_fema'} == set(
        resolved.writes
    )
    assert set(EVIDENCE) | {
        'height_footprint_fema',
        'n_stories_parcel',
        'living_area_sqft_parcel',
    } <= set(resolved.reads)
