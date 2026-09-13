"""A supplement of an additional layer joins the spine built from it.

The property spine loads a statewide host's `property` layer through
`union_spine_sources`, which records the host's recipe id. A city
table declaring `supplements: <host>` and `supplements_layer:
property` is then selected by the supplements-only join, and its
columns land on the rows it describes. Ids and values are fabricated.
"""

import pandas as pd
import pytest

import openplaces.io.harmonizer.links as links
import openplaces.io.harmonizer.spine as spine_module
from openplaces.core.schema import AdminId, Entity
from openplaces.io.harmonizer import HarmonizeState

_HOST = 'US-XX_parcel-statewide-2025'
_CITY = 'US-XX-YY_property-citytable-2026'


def _state():
    return HarmonizeState(
        recipe={'admin_id': AdminId('US')},
        admin_id=AdminId('US-XX-YY'),
        verbose=False,
        timer=None,
        spine=None,
    )


def _host_recipe():
    return {
        'recipe_id': _HOST,
        'admin_id': AdminId('US-XX'),
        'stage': 'ingest',
        'entity': Entity('parcel', 'statewide', '2025'),
        'additional_layers': [
            {'layer': 'ASSESS', 'entity': Entity('property', 'statewide', '2025')}
        ],
    }


def _city_recipe(**extra):
    return {
        'recipe_id': _CITY,
        'admin_id': AdminId('US-XX-YY'),
        'stage': 'ingest',
        'entity': Entity('property', 'citytable', '2026'),
        'supplements': _HOST,
        'supplements_layer': 'property',
        'aggregation_function': {'n_bedrooms': 'sum'},
        **extra,
    }


def test_a_layer_source_is_recorded_under_its_hosts_id(monkeypatch):
    layer_rows = pd.DataFrame({'parcel_id_local': ['a', 'b']})
    monkeypatch.setattr(spine_module, 'get_entities', lambda *a, **k: layer_rows)
    monkeypatch.setattr(spine_module, 'restrict_to_admin_by_name', lambda df, *a: df)

    state = spine_module.union_spine_sources(
        _state(),
        sources=[{'recipe_id': _HOST, 'label': 'statewide', 'layer': 'property'}],
    )

    assert state.metadata['spine_source_recipe_ids'] == {_HOST}


def test_a_city_table_lands_on_the_layer_rows_it_details(monkeypatch):
    recipes = {_HOST: _host_recipe(), _CITY: _city_recipe()}
    monkeypatch.setattr(
        'openplaces.recipe.get_recipe_by_id', lambda rid, **k: recipes[rid]
    )
    monkeypatch.setattr(links, '_find_admin_scoped_recipe_ids', lambda *a: [_CITY])
    frames = {
        # Two building rows of 'a' sum; 'z' is not on the spine.
        _CITY: pd.DataFrame(
            {'parcel_id_local': ['a', 'a', 'z'], 'n_bedrooms': [2.0, 1.0, 9.0]}
        )
    }

    def _get_entities(recipe_id, *args, **kwargs):
        assert recipe_id in frames, f'{recipe_id} should not be loaded'
        return frames[recipe_id]

    monkeypatch.setattr(links, 'get_entities', _get_entities)
    monkeypatch.setattr(links, 'restrict_to_admin_by_name', lambda df, *a: df)
    monkeypatch.setattr(links, '_apply_remap_csvs', lambda state, recipe_id: state)

    state = _state()
    state.spine = pd.DataFrame(
        {'parcel_id_local': ['a', 'b'], 'source': ['statewide', 'statewide']}
    )
    state.metadata['spine_source_recipe_ids'] = {_HOST}

    state = links.link_by_id(
        state,
        auto_discover=True,
        entity_type='property',
        supplements_only=True,
        count_as=False,
    )

    spine = state.spine.set_index('parcel_id_local')
    assert spine.loc['a', 'n_bedrooms'] == 3.0
    assert pd.isna(spine.loc['b', 'n_bedrooms'])
    assert len(spine) == 2


def test_a_supplement_naming_an_unknown_layer_fails_at_discovery(monkeypatch):
    recipes = {
        _HOST: _host_recipe(),
        _CITY: _city_recipe(supplements_layer='transaction'),
    }
    monkeypatch.setattr(
        'openplaces.recipe.get_recipe_by_id', lambda rid, **k: recipes[rid]
    )
    monkeypatch.setattr(links, '_find_admin_scoped_recipe_ids', lambda *a: [_CITY])

    with pytest.raises(ValueError, match='no such additional layer'):
        links._discover_link_sources(_state(), 'property')
