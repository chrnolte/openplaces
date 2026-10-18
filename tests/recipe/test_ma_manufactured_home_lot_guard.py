"""The US-MA guard: Manufactured Home only on a lot with a residential group."""

import pandas as pd

from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState
from openplaces.io.curator.reconcilers import resolve_by_vote
from openplaces.recipe import apply_recipe_patches, get_recipe_by_id

RID = 'US_footprint-openplaces-2026'


def _guard(admin):
    recipe, _ = apply_recipe_patches(get_recipe_by_id(RID), AdminId(admin))
    steps = recipe['pipeline']
    votes = [i for i, s in enumerate(steps) if s['step'] == 'resolve_by_vote']
    guard = [
        i
        for i in votes
        if any(
            d.get('source') == 'manufactured_home_lot_guard'
            for d in steps[i]['decisions']
        )
    ]
    return steps, votes, guard


def test_guard_follows_the_occupancy_vote_in_massachusetts_only():
    steps, votes, guard = _guard('US-MA-SOM')
    assert guard == [votes[3]]  # right after the third vote
    assert steps[votes[2]]['base_output'] == 'occupancy_type_prevote'
    _, _, none = _guard('US-NC-NHA')
    assert none == []


def test_guard_on_fabricated_outlines():
    steps, _, guard = _guard('US-MA-SOM')
    params = {k: v for k, v in steps[guard[0]].items() if k != 'step'}
    mh = 'Manufactured Home'
    frame = pd.DataFrame(
        {
            'occupancy_type': [mh, mh, mh, mh, 'Single-Family'],
            'group_parcel': [None, 'Government Services', mh, 'Single Family', None],
            'priority_on_parcel': [
                'secondary',
                'primary',
                'primary',
                'primary',
                'primary',
            ],
            'dwelling_multiplicity': ['single'] * 5,
            'occupancy_type_prevote': [None] * 5,
        },
        index=['SHED', 'KIOSK', 'MH_LOT', 'HOUSE_LOT', 'HOUSE'],
    )
    state = CurateState(
        recipe={},
        entity_recipe={},
        admin_id=None,
        verbose=False,
        timer=None,
        curated=frame,
    )
    out = resolve_by_vote(state, **params).curated['occupancy_type'].astype(str)
    assert out['SHED'] == 'Secondary'  # empty lot group
    assert out['KIOSK'] == 'Single-Family'  # non-residential lot group
    assert out['MH_LOT'] == mh  # code 103 lot keeps the class
    assert out['HOUSE_LOT'] == mh  # residential lot keeps the class
    assert out['HOUSE'] == 'Single-Family'
