"""The park rule decided for question 4, on the building side.

plans/occupancy-and-dwellings-one-decider.md, O5 (maintainer,
2026-11-22): a lot is a park where it holds 8 or more manufactured
homes, or 3 or more whose median aspect is 2.78 or more. Every lot,
class and aspect below is fabricated.
"""

from __future__ import annotations

import pandas as pd
import pytest

from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState
from openplaces.io.curator.estimators.manufactured_homes import (
    flag_manufactured_home_communities,
)
from openplaces.recipe import get_recipe_by_id

MH = 'Manufactured Home'
RULE = {'homes_at_least': 8, 'min_median_aspect': 2.78, 'aspect_min_homes': 3}


def _state(curated):
    return CurateState(
        recipe={},
        entity_recipe={},
        admin_id=AdminId('US', 'XX', 'YY'),
        verbose=False,
        timer=None,
        curated=curated,
    )


def _lot(lot, homes, aspect, extra=()):
    rows = [{'parcel_id': lot, 'occupancy_type': MH, 'aspect_ratio': aspect}] * homes
    rows += [
        {'parcel_id': lot, 'occupancy_type': c, 'aspect_ratio': a} for c, a in extra
    ]
    return rows


def _flag(rows, **params):
    frame = pd.DataFrame(rows)
    out = flag_manufactured_home_communities(_state(frame), **params).curated
    return out.groupby('parcel_id')['manufactured_home_community'].first()


@pytest.mark.parametrize(
    ('homes', 'aspect', 'park'),
    [
        (8, 1.5, True),
        (4, 3.0, True),
        (3, 3.0, True),
        (4, 2.0, False),
        (2, 3.0, False),
        (1, 4.0, False),
        (7, 2.7, False),
    ],
    ids=['8 low', '4 long', '3 long', '4 short', '2 long', '1 home', '7 short'],
)
def test_the_park_rule(homes, aspect, park):
    assert _flag(_lot('L', homes, aspect), **RULE)['L'] == park


def test_the_aspect_is_read_off_the_homes_only():
    # Two long sheds beside three short homes do not lift the median.
    rows = _lot('L', 3, 1.8, extra=[('Secondary', 5.0), ('Secondary', 5.0)])
    assert not _flag(rows, **RULE)['L']


def test_without_the_arms_the_old_count_rule_holds():
    assert _flag(_lot('L', 4, 1.0))['L']
    assert not _flag(_lot('L', 3, 9.0))['L']


def test_the_national_recipe_sets_the_decided_rule():
    step = next(
        e
        for e in get_recipe_by_id('US_building-openplaces-2026')['pipeline']
        if e.get('step') == 'flag_manufactured_home_communities'
    )
    assert {k: step[k] for k in RULE} == RULE
