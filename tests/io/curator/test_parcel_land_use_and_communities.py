"""Tests for the parcel-curation lane and the footprint community sidecar.

Covers:
- classify_parcel_land_use: manufactured-home park vs RV park vs other from the
  assessor keyword + linked-NSI group + footprint-morphology mix.
- link_curated_entity: joining another curated entity's attributes onto the
  current one by a shared parcel id.
- flag_manufactured_home_communities: per-parcel manufactured-home count + flag.
- _habitable_threshold: the size cutoff that keeps sheds Secondary.
"""

from __future__ import annotations

import pandas as pd

import openplaces.io.curator.evidence as ev
from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState
from openplaces.io.curator.inferers import (
    _habitable_threshold,
    classify_parcel_land_use,
    flag_manufactured_home_communities,
)


def _state(df, recipe=None):
    return CurateState(
        recipe=recipe or {'entity': 'e', 'admin_id': 'US'},
        entity_recipe={},
        admin_id=AdminId('US'),
        verbose=False,
        timer=None,
        curated=df,
    )


_LAND_USE_RULES = [
    {
        'class': 'Manufactured Home Park',
        'min_score': 2,
        'indicators': [
            {
                'type': 'keyword',
                'column': 'use_group_combined',
                'pattern': 'MOBILE|MANUFACTURED',
            },
            {
                'type': 'in_set',
                'column': 'group_parcel',
                'values': ['Manufactured Home'],
            },
            {
                'type': 'count_at_least',
                'column': 'n_small_elongated_footprints_per_parcel',
                'min': 4,
            },
        ],
    },
    {
        'class': 'RV Park',
        'min_score': 1,
        'indicators': [
            {
                'type': 'keyword',
                'column': 'use_group_combined',
                'pattern': 'RV|CAMPGROUND|RECREATION',
            },
            {'type': 'in_set', 'column': 'group_parcel', 'values': ['Recreation']},
        ],
    },
]


def test_classify_parcel_land_use_separates_mh_park_from_rv_park():
    df = pd.DataFrame(
        {
            'use_group_combined': [
                'MOBILE HOME PARK',
                'RV CAMPGROUND',
                'SINGLE FAMILY',
            ],
            'group_parcel': ['Manufactured Home', 'Recreation', 'Single-Family'],
            'n_small_elongated_footprints_per_parcel': [8, 1, 0],
        }
    )
    out = classify_parcel_land_use(
        _state(df),
        rules=_LAND_USE_RULES,
        output='land_use_class',
        flag_column='manufactured_home_community',
        flag_class='Manufactured Home Park',
    ).curated

    classes = out['land_use_class'].astype(object).tolist()
    assert classes[0] == 'Manufactured Home Park'
    assert classes[1] == 'RV Park'
    assert pd.isna(classes[2])
    assert out['manufactured_home_community'].tolist() == [True, False, False]


def test_link_curated_entity_joins_by_parcel_id(monkeypatch):
    footprints = pd.DataFrame(
        {
            'parcel_id': ['a', 'b', 'a'],
            'occupancy_type': ['x', 'y', 'z'],
        }
    )
    parcels = pd.DataFrame(
        {
            'parcel_id': ['a', 'b'],
            'use_group_combined': ['MOBILE', 'SFR'],
            'manufactured_home_community': [True, False],
        }
    )
    monkeypatch.setattr(ev, 'get_recipe_by_id', lambda rid: {'stage': 'curate'})
    monkeypatch.setattr(ev, 'get_output_path', lambda recipe, admin_id: 'parcels.pq')
    monkeypatch.setattr(ev, 'read_parquet', lambda path: parcels)

    out = ev.link_curated_entity(
        _state(footprints),
        recipe_id='US_parcel-openplaces-2026',
        columns={
            'use_group_combined': 'use_group_combined_parcel',
            'manufactured_home_community': 'manufactured_home_community',
        },
    ).curated

    assert out['use_group_combined_parcel'].tolist() == ['MOBILE', 'SFR', 'MOBILE']
    assert out['manufactured_home_community'].tolist() == [True, False, True]


def test_flag_manufactured_home_communities_counts_per_parcel():
    occ = {'rules': {'manufactured_home_geometry': {'class': 'Manufactured Home'}}}
    df = pd.DataFrame(
        {
            'parcel_id_local': ['p'] * 5 + ['q'] * 2,
            'occupancy_type': ['Manufactured Home'] * 7,
        }
    )
    out = flag_manufactured_home_communities(
        _state(df, recipe={'entity': 'e', 'admin_id': 'US', 'occupancy': occ}),
        min_homes=3,
    ).curated

    p = out['parcel_id_local'] == 'p'
    q = out['parcel_id_local'] == 'q'
    assert out.loc[p, 'n_manufactured_homes_per_parcel'].eq(5).all()
    assert out.loc[p, 'manufactured_home_community'].all()
    # 2 homes is not > 3, so q is not a community.
    assert not out.loc[q, 'manufactured_home_community'].any()


def test_flag_manufactured_home_communities_prefers_parcel_id_over_local():
    # Two distinct real parcels ('p1', 'p2') share one ambiguous
    # parcel_id_local ('DUP'). p1 has 5 MH footprints, p2 has 0. Grouping by
    # the globally-unique parcel_id (preferred when present) keeps them
    # separate; grouping by the colliding parcel_id_local would pool both
    # into one bogus 5-home "community" spanning both parcels.
    occ = {'rules': {'manufactured_home_geometry': {'class': 'Manufactured Home'}}}
    df = pd.DataFrame(
        {
            'parcel_id': ['p1'] * 5 + ['p2'] * 2,
            'parcel_id_local': ['DUP'] * 7,
            'occupancy_type': ['Manufactured Home'] * 5 + ['Single-Family'] * 2,
        }
    )
    out = flag_manufactured_home_communities(
        _state(df, recipe={'entity': 'e', 'admin_id': 'US', 'occupancy': occ}),
        min_homes=3,
    ).curated

    p1 = out['parcel_id'] == 'p1'
    p2 = out['parcel_id'] == 'p2'
    assert out.loc[p1, 'n_manufactured_homes_per_parcel'].eq(5).all()
    assert out.loc[p1, 'manufactured_home_community'].all()
    assert out.loc[p2, 'n_manufactured_homes_per_parcel'].eq(0).all()
    assert not out.loc[p2, 'manufactured_home_community'].any()


def test_habitable_threshold_floor_and_average():
    config = {'habitable_fraction': 0.5, 'habitable_floor_m2': 25.0}
    # Three MH-classed footprints averaging 100 m2 -> threshold 50 m2.
    curated = pd.DataFrame(
        {'occupancy_type': ['Manufactured Home'] * 3, 'area_m2': [80.0, 100.0, 120.0]}
    )
    result = curated['occupancy_type'].astype(object)
    assert _habitable_threshold(curated, result, 'Manufactured Home', config) == 50.0

    # Too few samples -> falls back to manufactured_home_avg_m2 (90) * 0.5 = 45,
    # but the floor (25) does not bind here.
    cfg2 = {**config, 'manufactured_home_avg_m2': 90.0}
    small = pd.DataFrame({'occupancy_type': ['Manufactured Home'], 'area_m2': [100.0]})
    res2 = small['occupancy_type'].astype(object)
    assert _habitable_threshold(small, res2, 'Manufactured Home', cfg2) == 45.0
