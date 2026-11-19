"""A lot's structure style comes from its properties in the parcel curate.

The parcel geospine no longer keeps a roll's or a property layer's
`building_style` on the lot (withhold_property_values); the parcel
curate reduces it from the property spine, every distinct style once,
ahead of what the lot holds (plans/account-values-one-deciding-step.md,
V4, ruled 2026-11-18). All values are fabricated.
"""

import pandas as pd
import pytest

from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState, aggregation
from openplaces.recipe import get_recipe_by_id

PARCEL = 'US_parcel-openplaces-2026'


def _state(curated):
    return CurateState(
        recipe={},
        entity_recipe={},
        admin_id=AdminId('US', 'XX', 'YY'),
        verbose=False,
        timer=None,
        curated=curated,
    )


@pytest.fixture
def properties(monkeypatch):
    rows = pd.DataFrame(
        {
            'parcel_id_local': ['p1', 'p1', 'p1', 'p2', 'p3'],
            'building_style': ['Colonial', 'Row End', 'Colonial', 'Ranch', None],
        }
    )

    def fake_get_entities(recipe_id, admin_id, columns=None, missing='raise'):
        return rows[[c for c in columns if c in rows.columns]]

    monkeypatch.setattr(aggregation, 'get_entities', fake_get_entities)
    monkeypatch.setattr(
        aggregation,
        'describe_recipe',
        lambda recipe_id, admin_id: pd.DataFrame(index=rows.columns),
    )


def _rule():
    pipeline = get_recipe_by_id(PARCEL)['pipeline']
    step = next(s for s in pipeline if s['step'] == 'aggregate_from_entities')
    return {k: v for k, v in step.items() if k != 'step'}


def test_the_parcel_curate_reduces_style_from_properties():
    rule = _rule()

    assert rule['columns']['building_style'] == 'distinct'
    assert 'building_style' in rule['prefer']


def test_each_style_once_ahead_of_the_lots_copy(properties):
    # p1: an old geospine's joined copy; p2: none; p3: the parcel
    # layer's own style, which no property states.
    curated = pd.DataFrame(
        {
            'parcel_id_local': ['p1', 'p2', 'p3'],
            'building_style': ['Colonial + Row End', None, 'Cape'],
        }
    )
    state = aggregation.aggregate_from_entities(
        _state(curated),
        'r',
        key='parcel_id_local',
        columns={'building_style': 'distinct'},
        prefer=['building_style'],
    )
    assert state.curated['building_style'].tolist() == [
        'Colonial | Row End',
        'Ranch',
        'Cape',
    ]
