"""The US-MA guard: Manufactured Home only where a source states it."""

import pandas as pd

from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState
from openplaces.io.curator.reconcilers import resolve_by_vote
from openplaces.recipe import apply_recipe_patches, get_recipe_by_id

RID = 'US_footprint-openplaces-2026'
GUARD = 'manufactured_home_stated_guard'


def _guard(admin):
    recipe, _ = apply_recipe_patches(get_recipe_by_id(RID), AdminId(admin))
    steps = recipe['pipeline']
    votes = [i for i, s in enumerate(steps) if s['step'] == 'resolve_by_vote']
    guard = [
        i for i in votes if any(d.get('source') == GUARD for d in steps[i]['decisions'])
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
            'occupancy_type': [mh, mh, mh, mh, mh, 'Single-Family', mh],
            'occupancy_type_source': [
                'morphology',
                'morphology+keyword_probability',
                'morphology',
                'morphology+keyword_probability+keyword',
                'nsi+no_improvement_value',
                'dwellings',
                'morphology',
            ],
            'group_parcel': [
                None,
                'Single Family',
                mh,
                'Single Family',
                None,
                None,
                'College/University',
            ],
            'priority_on_parcel': ['secondary'] + ['primary'] * 6,
            'dwelling_multiplicity': ['single'] * 7,
            'occupancy_type_prevote': [None] * 6 + ['College/University'],
        },
        index=['SHED', 'ROW_UNIT', 'MH_LOT', 'KEYWORD', 'NSI', 'HOUSE', 'COLLEGE'],
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
    assert out['SHED'] == 'Secondary'  # shape alone, secondary
    assert out['ROW_UNIT'] == 'Single-Family'  # shape and score alone
    assert out['MH_LOT'] == mh  # code 103 lot keeps the class
    assert out['KEYWORD'] == mh  # the assessor's keyword states it
    assert out['NSI'] == mh  # NSI states it
    assert out['HOUSE'] == 'Single-Family'
    assert out['COLLEGE'] == 'College/University'  # class before the vote
