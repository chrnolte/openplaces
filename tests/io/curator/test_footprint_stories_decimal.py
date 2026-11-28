"""The delivered footprint keeps n_stories as a decimal.

Ruling of 2026-11-27: the CHEER footprint curate does not round the
story count it folds from the building curate. A roll's half story
stays 1.5, and an outline holding several buildings carries the mean
of their stories, which can also be a fraction. The building curate
has kept the decimal since 2026-11-15.
"""

import pandas as pd

from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState, aggregation
from openplaces.recipe import get_recipe_by_id

FOOTPRINT_RECIPE = 'US_footprint-cheer-2026'
# The whole-number cast, under its current name and the name a
# pending rename gives it.
ROUNDING_STEPS = {'cast_integers', 'round_to_integers'}


def _rounding_steps(recipe_id):
    pipeline = get_recipe_by_id(recipe_id)['pipeline']
    return [step for step in pipeline if step['step'] in ROUNDING_STEPS]


def test_the_footprint_rounding_step_does_not_list_n_stories():
    steps = _rounding_steps(FOOTPRINT_RECIPE)
    assert steps, 'the footprint curate has no whole-number cast'
    for step in steps:
        assert 'n_stories' not in step['columns'], step
        # The cast still serves the counts and years that are whole.
        assert 'year_built' in step['columns'], step


def _state(curated):
    return CurateState(
        recipe={},
        entity_recipe={},
        admin_id=AdminId('US', 'XX', 'YY'),
        verbose=False,
        timer=None,
        curated=curated,
    )


def test_a_fractional_story_count_survives_the_fold_and_the_cast(monkeypatch, tmp_path):
    """f1 holds one building with a roll's half story; f2 holds two
    buildings of one and two stories, whose mean is 1.5; f3 holds one
    two-story building. Fabricated ids and values throughout."""
    import openplaces.geo.link as link_module
    import openplaces.io.curator as curator
    import openplaces.io.harmonizer.entity_links as entity_links

    buildings = pd.DataFrame(
        {
            'footprint_id': ['f1', 'f2', 'f2', 'f3'],
            'n_stories': [1.5, 1.0, 2.0, 2.0],
            'year_built': [1950.0, 1990.0, 1970.0, 2001.0],
        },
        index=pd.Index(['b1', 'b2', 'b3', 'b4'], name='building_id'),
    )
    links = pd.DataFrame(
        {
            'building_id': ['b1', 'b2', 'b3', 'b4'],
            'footprint_id': ['f1', 'f2', 'f2', 'f3'],
            'link_method': 'footprint_id',
            'link_source': 'footprint',
            'share': pd.Series([pd.NA] * 4, dtype='Float64'),
        }
    )
    monkeypatch.setattr(
        aggregation,
        'get_entities',
        lambda recipe_id, admin_id, columns=None, missing='raise': buildings[
            [c for c in columns if c in buildings.columns]
        ],
    )
    monkeypatch.setattr(
        aggregation,
        'describe_recipe',
        lambda recipe_id, admin_id: pd.DataFrame(index=buildings.columns),
    )
    monkeypatch.setattr(
        aggregation, '_linked_recipe_id', lambda recipe_id: 'US_building-spine-2026'
    )
    monkeypatch.setattr(
        link_module,
        'get_link_owner_recipe_id',
        lambda recipe: (
            'US_building-geospine-2026'
            if recipe == 'US_building-spine-2026'
            else 'US_footprint-geospine-2026'
        ),
    )
    monkeypatch.setattr(
        link_module,
        'get_entity_link_path',
        lambda a, b, admin_id=None: tmp_path / f'{a}__{b}.parquet',
    )
    (tmp_path / 'US_building-geospine-2026__US_footprint-geospine-2026.parquet').touch()
    monkeypatch.setattr(
        entity_links,
        'read_entity_link',
        lambda path: links if path.name.startswith('US_building-geospine') else None,
    )

    curated = pd.DataFrame(
        index=pd.Index(['f1', 'f2', 'f3'], name='footprint_id'),
    )
    state = aggregation.aggregate_from_entities(
        _state(curated),
        'US_building-openplaces-2026',
        columns='all',
        key=None,
        aggregations={'year_built': 'min'},
        count_as='n_buildings_per_footprint',
    )
    # The footprint recipe's own cast, as the recipe declares it.
    curator._load_steps()
    for step in _rounding_steps(FOOTPRINT_RECIPE):
        fn = curator._STEP_REGISTRY[step['step']]
        state = fn(state, columns=step['columns'])

    out = state.curated
    assert out['n_stories'].tolist() == [1.5, 1.5, 2.0]
    assert out['year_built'].dtype == 'Int64'
    assert out['year_built'].tolist() == [1950, 1970, 2001]
