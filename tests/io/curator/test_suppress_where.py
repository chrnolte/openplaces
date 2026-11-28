"""empty_selected_cells: the equality condition and the indicator condition.

Fabricated values only.
"""

import pandas as pd
import pytest

from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState
from openplaces.io.curator.reconcilers import empty_selected_cells
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
    out = empty_selected_cells(
        _state(_footprints()),
        column=['structure_value', 'structure_value_per_area'],
        indicators=SLIVER,
    ).curated
    assert pd.isna(out.loc[0, 'structure_value'])
    assert pd.isna(out.loc[0, 'structure_value_per_area'])
    assert pd.isna(out.loc[0, 'structure_value_source'])


def test_every_indicator_has_to_hold():
    out = empty_selected_cells(
        _state(_footprints()), column='structure_value', indicators=SLIVER
    ).curated
    # A large cluster and a small ordinary footprint both keep theirs.
    assert out.loc[1, 'structure_value'] == 9_000_000.0
    assert out.loc[2, 'structure_value'] == 5_000.0
    assert out.loc[3, 'structure_value_source'] == 'nsi'


def test_equality_condition_still_works():
    df = pd.DataFrame({'n_dwellings': [2.0, 3.0], 'land_use': ['Vacant', 'Retail']})
    out = empty_selected_cells(
        _state(df),
        column='n_dwellings',
        condition_column='land_use',
        condition_value='Vacant',
    ).curated
    assert pd.isna(out.loc[0, 'n_dwellings'])
    assert out.loc[1, 'n_dwellings'] == 3.0


def test_both_conditions_combine():
    df = _footprints().assign(flag=[True, True, True, False])
    out = empty_selected_cells(
        _state(df),
        column='structure_value',
        condition_column='flag',
        indicators=[{'type': 'numeric_below', 'column': 'area_m2', 'max': 20}],
    ).curated
    assert out['structure_value'].isna().tolist() == [True, False, True, False]


def test_no_condition_is_an_error():
    with pytest.raises(ValueError, match='condition_column or indicators'):
        empty_selected_cells(_state(_footprints()), column='structure_value')


def test_absent_column_is_skipped():
    out = empty_selected_cells(
        _state(_footprints()),
        column=['not_there', 'structure_value'],
        indicators=SLIVER,
    ).curated
    assert pd.isna(out.loc[0, 'structure_value'])


def test_the_footprint_recipe_guards_sliver_clusters_after_the_ratio_exists():
    steps = get_recipe_by_id('US_building-openplaces-2026')['pipeline']
    names = [s['step'] for s in steps]
    guard = next(
        i
        for i, s in enumerate(steps)
        if s['step'] == 'empty_selected_cells'
        and 'structure_value' in str(s.get('column'))
    )
    assert names.index('derive_metrics') < guard
    assert names.index('split_curated_values') < guard
    assert 'structure_value_per_area' in steps[guard]['column']


def test_structure_value_is_the_roll_figure_only():
    """No step of the building curate writes NSI into structure_value.

    The maintainer's ruling of 2026-11-14 (plans/architecture-review-
    diagnosis-and-first-principles.md, section 8): a town whose roll has
    no value keeps structure_value missing for the imputation to fill;
    NSI's estimates stay in their own columns.
    """
    steps = get_recipe_by_id('US_building-openplaces-2026')['pipeline']
    for step in steps:
        if step['step'] == 'pick_first_value':
            assert 'structure_value' not in step.get('priority', {})
    writers = [
        s
        for s in steps
        if s['step'] == 'split_curated_values'
        and 'structure_value' in s['columns'].values()
    ]
    assert len(writers) == 1
    assert writers[0]['columns']['improvement_value_imputed'] == 'structure_value'
