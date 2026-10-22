"""US-MA parcel patch: no home class on protected open space without a roll.

Fabricated rows: a reservation with no roll record loses its class; a
house with a roll record on restricted land, and a lot with no roll
record but little open space, keep theirs.
"""

import pandas as pd

from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState
from openplaces.io.curator.reconcilers import suppress_where
from openplaces.recipe import apply_recipe_patches, get_recipe_by_id

PARCEL = 'US_parcel-openplaces-2026'


def _steps(admin):
    recipe, _ = apply_recipe_patches(get_recipe_by_id(PARCEL), AdminId(admin))
    return recipe['pipeline']


def _open_space_rule(steps):
    return next(
        s
        for s in steps
        if s['step'] == 'suppress_where'
        and any(
            i.get('column') == 'protected_open_space_share'
            for i in s.get('indicators', [])
        )
    )


def test_rule_runs_after_the_vote_and_the_open_space_merge_in_ma_only():
    steps = _steps('US-MA-STM')
    names = [s['step'] for s in steps]
    rule = steps.index(_open_space_rule(steps))
    assert rule > names.index('reconcile_land_use')
    assert rule > names.index('merge_enrichments')
    other = _steps('US-NC-NHA')
    assert not any(
        i.get('column') == 'protected_open_space_share'
        for s in other
        for i in s.get('indicators', []) or []
    )


def test_open_space_without_a_roll_record_loses_its_home_class():
    rule = _open_space_rule(_steps('US-MA-STM'))
    curated = pd.DataFrame(
        {
            'land_use_class': ['Multi-Family', 'Single-Family', 'Single-Family'],
            'land_use_class_source': ['nsi', 'parcel', 'fema'],
            'property_source': ['none', 'massgis', 'none'],
            'protected_open_space_share': [0.84, 0.95, 0.10],
        },
        index=pd.Index(['reservation', 'restricted_house', 'small_share'], name='id'),
    )
    state = CurateState(
        recipe={},
        entity_recipe=None,
        admin_id=AdminId('US-MA-STM'),
        verbose=False,
        timer=None,
        curated=curated,
    )
    params = {
        k: v for k, v in rule.items() if k not in ('step', 'enabled', 'phase_override')
    }
    out = suppress_where(state, **params).curated['land_use_class']
    assert pd.isna(out['reservation'])
    assert out['restricted_house'] == 'Single-Family'
    assert out['small_share'] == 'Single-Family'
