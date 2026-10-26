"""The parcel vote's RV Park decision needs RV evidence.

A recreation group and four footprints also describe a club house, a
golf course or a ballpark; only an RV keyword or style makes a park an
RV park.
"""

import pandas as pd

from openplaces.io.curator.indicators import score_decisions
from openplaces.recipe import get_recipe_by_id


def _rv_decision():
    recipe = get_recipe_by_id('US_parcel-openplaces-2026')
    for step in recipe['pipeline']:
        for decision in step.get('decisions') or []:
            if decision.get('class') == 'RV Park':
                return decision
    raise AssertionError('no RV Park decision')


def test_recreation_group_and_footprints_alone_are_not_an_rv_park():
    curated = pd.DataFrame(
        {
            'keyword_class': [None, 'RV Park'],
            'style_class': [None, None],
            'group_parcel': ['Entertainment/Recreation'] * 2,
            'n_buildings_per_parcel': [12, 12],
        }
    )
    winner = score_decisions(curated, [_rv_decision()])[0]
    assert pd.isna(winner.iloc[0])
    assert winner.iloc[1] == 'RV Park'
