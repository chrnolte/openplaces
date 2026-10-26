"""A footprint's dwelling count comes from its lot's roll first.

The roll's dwelling count is split over a lot's footprints by the
weights its improvement value uses, and it is the first lane of
`n_dwellings` in the footprint recipe's `reconcile_values`, ahead of
Overture's address points and NSI. One Overture point on a building
whose roll states several units counted one dwelling, so the building's
value per dwelling was its lot's whole improvement. All rows fabricated.
"""

import pandas as pd
import pytest

from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState, reconcilers
from openplaces.io.harmonizer.apportion import apportion_reference_values
from openplaces.recipe import get_recipe_by_id

RECIPE = 'US_building-openplaces-2026'


def _pairs(rows):
    return pd.DataFrame(rows, columns=['fid', 'parcel_id', 'area_intersection_m2'])


def test_a_lot_with_one_valued_building_gives_it_every_dwelling():
    # One primary dwelling building and three secondaries, the shape of
    # a condominium lot whose second building stayed secondary.
    pairs = _pairs(
        [
            ('F1', 'P1', 300.0),
            ('F2', 'P1', 230.0),
            ('F3', 'P1', 120.0),
            ('F4', 'P1', 40.0),
        ]
    )
    ref = pd.DataFrame(
        {'improvement_value': [7_000_000.0], 'n_dwellings': [5.0]},
        index=pd.Index(['P1']),
    )
    priority = pd.Series(
        ['primary', 'secondary', 'secondary', 'secondary'],
        index=['F1', 'F2', 'F3', 'F4'],
    )
    out = apportion_reference_values(
        pairs, ref, spine_id_col='fid', priority=priority, dwelling_linked_ids={'F1'}
    )
    assert out.loc['F1', 'n_dwellings'] == 5.0
    assert out.loc['F1', 'improvement_value'] == 7_000_000.0
    # Secondaries hold no dwelling; their zero is emptied in the recipe.
    assert (out.loc[['F2', 'F3', 'F4'], 'n_dwellings'] == 0.0).all()
    # Value per dwelling is the lot's improvement per unit.
    per_home = out.loc['F1', 'improvement_value'] / out.loc['F1', 'n_dwellings']
    assert per_home == 1_400_000.0


def test_dwellings_follow_the_improvement_value_weights():
    # Two primary buildings, floors measured: the split is by floor
    # area for both columns, so each building's value per dwelling is
    # the same, the lot's improvement per unit.
    pairs = _pairs([('F1', 'P1', 200.0), ('F2', 'P1', 100.0)])
    ref = pd.DataFrame(
        {'improvement_value': [3_000_000.0], 'n_dwellings': [12.0]},
        index=pd.Index(['P1']),
    )
    priority = pd.Series(['primary', 'primary'], index=['F1', 'F2'])
    floors = pd.Series([1.0, 4.0], index=['F1', 'F2'])
    out = apportion_reference_values(
        pairs, ref, spine_id_col='fid', priority=priority, volume_weight=floors
    )
    # Floor area 200 and 400: a third and two thirds.
    assert out.loc['F1', 'n_dwellings'] == 4.0
    assert out.loc['F2', 'n_dwellings'] == 8.0
    per_home = out['improvement_value'] / out['n_dwellings']
    assert per_home.round(2).nunique() == 1
    assert per_home.iloc[0] == pytest.approx(250_000.0)


def _step(name, occurrence=1, **match):
    steps = [
        s
        for s in get_recipe_by_id(RECIPE)['pipeline']
        if s.get('step') == name and all(s.get(k) == v for k, v in match.items())
    ]
    return steps[occurrence - 1]


def test_the_recipe_apportions_the_roll_count():
    step = _step('apportion_curated_values')
    assert step['columns']['n_dwellings'] == 'n_dwellings_parcel'


def test_the_roll_count_is_the_first_lane():
    step = next(
        s
        for s in get_recipe_by_id(RECIPE)['pipeline']
        if s.get('step') == 'reconcile_values'
        and 'n_dwellings' in (s.get('priority') or {})
    )
    assert step['priority']['n_dwellings'] == [
        'n_dwellings_parcel',
        'n_dwellings_overture',
        'n_dwellings_nsi',
    ]


def test_a_zero_from_the_roll_falls_through_to_overture_and_nsi():
    curated = pd.DataFrame(
        {
            # Apportioned roll count: a share, a secondary's or a
            # commercial lot's zero, a lot whose roll states nothing.
            # commercial lot's zero, a lot whose roll states nothing, a
            # half share that would round to no dwelling.
            'n_dwellings_parcel': [2.5, 0.0, None, 0.0, 0.5],
            'n_dwellings_overture': [1.0, 3.0, None, None, 1.0],
            'n_dwellings_nsi': [1.0, 1.0, 4.0, None, None],
        },
        index=pd.Index(['F1', 'F2', 'F3', 'F4', 'F5'], name='footprint_id'),
    )
    state = CurateState(
        recipe={},
        entity_recipe={},
        admin_id=AdminId('US', 'XX', 'YY'),
        verbose=False,
        timer=None,
        curated=curated,
    )
    gate = next(
        s
        for s in get_recipe_by_id(RECIPE)['pipeline']
        if s.get('step') == 'suppress_where' and s.get('column') == 'n_dwellings_parcel'
    )
    params = {k: v for k, v in gate.items() if k != 'step'}
    state = reconcilers.suppress_where(state, **params)
    state = reconcilers.reconcile_values(
        state,
        priority={
            'n_dwellings': [
                'n_dwellings_parcel',
                'n_dwellings_overture',
                'n_dwellings_nsi',
            ]
        },
    )
    out = state.curated
    assert out['n_dwellings'].tolist()[:3] == [2.5, 3.0, 4.0]
    assert pd.isna(out.loc['F4', 'n_dwellings'])
    assert out.loc['F1', 'n_dwellings_source'] == 'parcel'
    assert out.loc['F2', 'n_dwellings_source'] == 'overture'
    assert out.loc['F5', 'n_dwellings'] == 1.0
    assert out.loc['F5', 'n_dwellings_source'] == 'overture'
