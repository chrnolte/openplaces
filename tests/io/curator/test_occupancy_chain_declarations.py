"""The steps of the occupancy and dwelling chain declare their columns.

plans/occupancy-and-dwellings-one-decider.md, O1. A step decides a
column only where it rebuilds every row of it; a step that keeps rows
it does not claim writes. Every id, class and count below is
fabricated.
"""

from __future__ import annotations

import pandas as pd

from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState
from openplaces.io.curator.estimators.manufactured_homes import (
    flag_manufactured_home_communities,
)
from openplaces.io.curator.imputers import (
    impute_from_group_statistic,
    impute_n_dwellings,
)
from openplaces.io.curator.indicators import indicator_columns
from openplaces.io.curator.reconcilers import reconcile_land_use, resolve_by_vote
from openplaces.io.steps import resolve_step_columns, undeclared_new_columns

DECISIONS = [
    {
        'class': 'Multi-Family',
        'indicators': [
            {'type': 'numeric_at_least', 'column': 'count_a', 'min': 2},
            {
                'type': 'any_of',
                'indicators': [
                    {'type': 'equals', 'column': 'kind_b', 'value': 'not_a_column'},
                    {
                        'type': 'value_share_below',
                        'value': 'part_c',
                        'total': ['part_c', 'part_d'],
                        'max_ratio': 0.5,
                    },
                ],
            },
        ],
        'require': [{'type': 'not_null', 'column': 'gate_e'}],
        'insufficient_alone': {
            'labels': ['shape'],
            'where': {'type': 'in_set', 'column': 'context_f', 'values': ['x']},
        },
    },
    {'class_column': 'earlier_g', 'indicators': [{'type': 'not_null', 'column': 'h'}]},
]


def _state(curated):
    return CurateState(
        recipe={},
        entity_recipe={},
        admin_id=AdminId('US', 'XX', 'YY'),
        verbose=False,
        timer=None,
        curated=curated,
    )


def test_indicator_columns_walks_nested_indicators_and_decisions():
    assert indicator_columns(DECISIONS) == [
        'count_a',
        'kind_b',
        'part_c',
        'part_d',
        'gate_e',
        'context_f',
        'earlier_g',
        'h',
    ]


def test_an_equals_value_is_a_literal_not_a_column():
    found = indicator_columns([{'type': 'equals', 'column': 'a', 'value': 'b'}])
    assert found == ['a']


def test_a_vote_that_rebuilds_its_target_decides_it_and_its_flag():
    params = {
        'target': 'class_z',
        'decisions': DECISIONS,
        'preserve_base': False,
        'flag_column': 'flag_y',
        'flag_class': 'Multi-Family',
        'review_column': 'review_x',
    }
    resolved = resolve_step_columns(resolve_by_vote, params)
    assert resolved.decides == {'class_z', 'flag_y'}
    assert {'class_z', 'class_z_source', 'flag_y', 'review_x'} <= resolved.writes
    assert {'class_z', 'count_a', 'gate_e', 'earlier_g'} <= resolved.reads


def test_a_vote_that_keeps_unclaimed_rows_only_writes():
    params = {'target': 'class_z', 'decisions': DECISIONS, 'base_output': 'before_w'}
    resolved = resolve_step_columns(resolve_by_vote, params)
    assert resolved.decides == frozenset()
    assert {'class_z', 'class_z_source', 'before_w'} <= resolved.writes


def test_a_vote_adds_no_column_it_does_not_declare():
    curated = pd.DataFrame({'count_a': [3, 1, None], 'gate_e': [1, 1, 1]})
    params = {
        'target': 'class_z',
        'decisions': DECISIONS[:1],
        'preserve_base': False,
        'base_output': 'before_w',
        'flag_column': 'flag_y',
        'flag_class': 'Multi-Family',
        'review_column': 'review_x',
    }
    before = list(curated.columns)
    out = resolve_by_vote(_state(curated.copy()), **params).curated
    assert out['class_z'].astype(object).tolist()[0] == 'Multi-Family'
    assert undeclared_new_columns(resolve_by_vote, params, before, out.columns) == []


def test_the_community_flag_decides_both_columns_it_writes():
    curated = pd.DataFrame(
        {
            'parcel_id': ['lot1'] * 5 + ['lot2'],
            'occupancy_type': ['Manufactured Home'] * 4
            + ['Secondary', 'Single-Family'],
        }
    )
    resolved = resolve_step_columns(flag_manufactured_home_communities, {})
    assert resolved.decides == {
        'manufactured_home_community',
        'n_manufactured_homes_per_parcel',
    }
    before = list(curated.columns)
    out = flag_manufactured_home_communities(_state(curated.copy())).curated
    assert out['manufactured_home_community'].tolist() == [True] * 5 + [False]
    assert (
        undeclared_new_columns(flag_manufactured_home_communities, {}, before, out)
        == []
    )


def test_fills_write_and_a_cohort_statistic_decides():
    fill = resolve_step_columns(impute_n_dwellings, {})
    assert fill.decides == frozenset()
    assert {'n_dwellings', 'n_dwellings_source'} == fill.writes
    assert 'occupancy_type_building_nsi' in fill.reads

    land_use = resolve_step_columns(
        reconcile_land_use, {'columns': [{'column': 'group_a', 'label': 'a'}]}
    )
    assert land_use.decides == frozenset()
    assert {'group_a', 'group_parcel', 'land_use_class'} <= land_use.reads

    cohort = resolve_step_columns(
        impute_from_group_statistic,
        {'group_column': 'use_a', 'value_column': 'group_b', 'output': 'group_c'},
    )
    assert cohort.decides == {'group_c'}
    assert cohort.reads == {'use_a', 'group_b'}
