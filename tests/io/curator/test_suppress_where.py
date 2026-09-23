"""suppress_where: the equality condition and the indicator condition.

Fabricated values only.
"""

import pandas as pd
import pytest

from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState
from openplaces.io.curator.reconcilers import suppress_where
from openplaces.recipe import get_recipe_by_id


def _state(df):
    return CurateState(
        recipe={},
        entity_recipe={},
        admin_id=AdminId('US'),
        verbose=False,
        timer=None,
        curated=df,
    )


def _footprints():
    return pd.DataFrame(
        {
            'geometry_source': [
                'condo_cluster.parcel.somecounty',
                'condo_cluster.parcel.somecounty',
                'obm',
                'obm',
            ],
            'area_m2': [4.0, 900.0, 4.0, 150.0],
            'structure_value': [800_000.0, 9_000_000.0, 5_000.0, 250_000.0],
            'structure_value_per_area': [200_000.0, 10_000.0, 1_250.0, 1_667.0],
            'structure_value_source': ['parcel', 'parcel', 'parcel', 'nsi'],
        }
    )


SLIVER = [
    {'type': 'keyword', 'column': 'geometry_source', 'pattern': r'^condo_cluster\.'},
    {'type': 'numeric_below', 'column': 'area_m2', 'max': 20},
]


def test_sliver_cluster_loses_its_value_and_its_source():
    out = suppress_where(
        _state(_footprints()),
        column=['structure_value', 'structure_value_per_area'],
        indicators=SLIVER,
    ).curated
    assert pd.isna(out.loc[0, 'structure_value'])
    assert pd.isna(out.loc[0, 'structure_value_per_area'])
    assert pd.isna(out.loc[0, 'structure_value_source'])


def test_every_indicator_has_to_hold():
    out = suppress_where(
        _state(_footprints()), column='structure_value', indicators=SLIVER
    ).curated
    # A large cluster and a small ordinary footprint both keep theirs.
    assert out.loc[1, 'structure_value'] == 9_000_000.0
    assert out.loc[2, 'structure_value'] == 5_000.0
    assert out.loc[3, 'structure_value_source'] == 'nsi'


def test_equality_condition_still_works():
    df = pd.DataFrame({'n_dwellings': [2.0, 3.0], 'land_use': ['Vacant', 'Retail']})
    out = suppress_where(
        _state(df),
        column='n_dwellings',
        condition_column='land_use',
        condition_value='Vacant',
    ).curated
    assert pd.isna(out.loc[0, 'n_dwellings'])
    assert out.loc[1, 'n_dwellings'] == 3.0


def test_both_conditions_combine():
    df = _footprints().assign(flag=[True, True, True, False])
    out = suppress_where(
        _state(df),
        column='structure_value',
        condition_column='flag',
        indicators=[{'type': 'numeric_below', 'column': 'area_m2', 'max': 20}],
    ).curated
    assert out['structure_value'].isna().tolist() == [True, False, True, False]


def test_no_condition_is_an_error():
    with pytest.raises(ValueError, match='condition_column or indicators'):
        suppress_where(_state(_footprints()), column='structure_value')


def test_absent_column_is_skipped():
    out = suppress_where(
        _state(_footprints()),
        column=['not_there', 'structure_value'],
        indicators=SLIVER,
    ).curated
    assert pd.isna(out.loc[0, 'structure_value'])


def test_the_footprint_recipe_guards_sliver_clusters_after_the_ratio_exists():
    steps = get_recipe_by_id('US_footprint-openplaces-2026')['pipeline']
    names = [s['step'] for s in steps]
    guard = next(
        i
        for i, s in enumerate(steps)
        if s['step'] == 'suppress_where' and 'structure_value' in str(s.get('column'))
    )
    assert names.index('derive_metrics') < guard
    assert names.index('select_value_source_by_admin_unit') < guard
    assert 'structure_value_per_area' in steps[guard]['column']
