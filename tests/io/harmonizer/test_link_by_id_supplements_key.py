"""A supplement declaring `supplements_key` joins its roll on that column.

Travis County TX's roll builds `parcel_id_local` from `geo_id` (its link
to parcels), while its room-count tables carry only `prop_id`. Such a
supplement names a second key both tables carry, and the property
spine's `supplements_only` pass joins it on that key, onto the roll's
rows only. Every other auto-discovered join skips it. Values are
fabricated.
"""

import pandas as pd
import pytest

import openplaces.io.harmonizer.links as links
from openplaces.core.schema import AdminId
from openplaces.io.harmonizer import HarmonizeState

_ROLL = 'US-XX-YY_property-roll-2026'
_DETAIL = f'{_ROLL}_bedrooms'
_KEY = 'parcel_id_assessor'


def _state(spine, spine_source_ids=(_ROLL,)):
    state = HarmonizeState(
        recipe={},
        admin_id=AdminId('US-XX-YY'),
        verbose=False,
        timer=None,
        spine=spine,
    )
    state.metadata['spine_source_recipe_ids'] = set(spine_source_ids)
    return state


def _matches():
    return [
        {
            'recipe_id': _ROLL,
            'layer': None,
            'key': 'parcel_id_local',
            'aggregation_function': None,
            'supplements': None,
            'supplements_key': None,
        },
        {
            'recipe_id': _DETAIL,
            'layer': None,
            'key': _KEY,
            'aggregation_function': {'n_bedrooms': 'sum'},
            'supplements': _ROLL,
            'supplements_key': _KEY,
        },
    ]


def _property_spine():
    # Three roll rows keyed on a geo-style parcel_id_local, plus a row
    # from another source whose own assessor id happens to equal a
    # roll row's.
    return pd.DataFrame(
        {
            'parcel_id_local': ['g1', 'g2', 'g3', None],
            _KEY: ['101', '102', '103', '101'],
            'source': ['roll', 'roll', 'roll', 'other'],
        },
        index=pd.Index(['p1', 'p2', 'p3', 'x1'], name='property_id'),
    )


def _patch(monkeypatch, frames):
    def _get_entities(recipe_id, *args, **kwargs):
        assert recipe_id in frames, f'{recipe_id} should not be loaded'
        return frames[recipe_id]

    monkeypatch.setattr(links, '_discover_link_sources', lambda *a, **k: _matches())
    monkeypatch.setattr(links, 'get_entities', _get_entities)
    monkeypatch.setattr(links, 'restrict_to_admin_by_name', lambda df, *a: df)
    monkeypatch.setattr(links, '_apply_remap_csvs', lambda state, recipe_id: state)


def test_supplement_joins_the_roll_on_its_supplements_key(monkeypatch):
    # The detail table has no parcel_id_local at all; it reaches the
    # roll's properties through the assessor id alone.
    detail = pd.DataFrame(
        {
            _KEY: ['101', '101', '102', '999'],
            'n_bedrooms': [2.0, 1.0, 3.0, 4.0],
        }
    )
    _patch(monkeypatch, {_DETAIL: detail})

    state = links.link_by_id(
        _state(_property_spine()),
        auto_discover=True,
        entity_type='property',
        supplements_only=True,
        count_as=False,
    )

    beds = state.spine['n_bedrooms']
    assert beds['p1'] == 3.0
    assert beds['p2'] == 3.0
    assert pd.isna(beds['p3'])
    # Same assessor id, different source: not the roll's property.
    assert pd.isna(beds['x1'])
    assert 'n_records_per_key' not in state.spine.columns


def test_a_caller_key_does_not_replace_the_supplements_key(monkeypatch):
    detail = pd.DataFrame({_KEY: ['102'], 'n_bedrooms': [4.0]})
    _patch(monkeypatch, {_DETAIL: detail})

    state = links.link_by_id(
        _state(_property_spine()),
        auto_discover=True,
        entity_type='property',
        supplements_only=True,
        count_as=False,
        spine_key='parcel_id_alnum',
        ref_key='parcel_id_alnum',
    )

    assert state.spine.loc['p2', 'n_bedrooms'] == 4.0


def test_other_discovery_skips_a_keyed_supplement(monkeypatch):
    # A parcel spine's property join: the roll joins on parcel_id_local,
    # and the keyed supplement is never loaded.
    roll = pd.DataFrame({'parcel_id_local': ['g1'], 'total_value': [100.0]})
    _patch(monkeypatch, {_ROLL: roll})
    parcels = pd.DataFrame(
        {'parcel_id_local': ['g1', 'g2']},
        index=pd.Index(['a', 'b'], name='parcel_id'),
    )

    state = links.link_by_id(
        _state(parcels, spine_source_ids=()),
        auto_discover=True,
        entity_type='property',
        columns=['total_value', 'n_bedrooms'],
        count_as=False,
    )

    assert state.spine.loc['a', 'total_value'] == 100.0
    assert 'n_bedrooms' not in state.spine.columns


def test_a_supplement_output_without_the_key_raises(monkeypatch):
    stale = pd.DataFrame({'parcel_id_local': ['101'], 'n_bedrooms': [2.0]})
    _patch(monkeypatch, {_DETAIL: stale})

    with pytest.raises(ValueError, match='supplements_key'):
        links.link_by_id(
            _state(_property_spine()),
            auto_discover=True,
            entity_type='property',
            supplements_only=True,
            count_as=False,
        )


def test_a_spine_without_the_key_raises(monkeypatch):
    detail = pd.DataFrame({_KEY: ['101'], 'n_bedrooms': [2.0]})
    _patch(monkeypatch, {_DETAIL: detail})
    spine = _property_spine().drop(columns=[_KEY])

    with pytest.raises(ValueError, match='spine has no such column'):
        links.link_by_id(
            _state(spine),
            auto_discover=True,
            entity_type='property',
            supplements_only=True,
            count_as=False,
        )


def test_discover_link_sources_reads_the_key_off_the_recipe(monkeypatch):
    recipes = {
        _ROLL: {
            'recipe_id': _ROLL,
            'columns': {'parcel_id_admin2_2': 'GEO'},
            'transformations': [{'input': 'ACCT', 'output': _KEY}],
            'parcel_id_local': {'source': 'parcel_id_admin2_2'},
        },
        _DETAIL: {
            'recipe_id': _DETAIL,
            'supplements': _ROLL,
            'supplements_key': _KEY,
            'columns': {'n_bedrooms': 'BEDS'},
            'transformations': [{'input': 'ACCT', 'output': _KEY}],
        },
    }
    monkeypatch.setattr(
        links, '_find_admin_scoped_recipe_ids', lambda *a, **k: list(recipes)
    )
    monkeypatch.setattr(
        'openplaces.recipe.get_recipe_by_id', lambda rid, **k: recipes[rid]
    )

    matches = {
        m['recipe_id']: m
        for m in links._discover_link_sources(_state(None), 'property')
    }

    assert matches[_ROLL]['key'] == 'parcel_id_local'
    assert matches[_ROLL]['supplements_key'] is None
    assert matches[_DETAIL]['key'] == _KEY
    assert matches[_DETAIL]['supplements_key'] == _KEY


def test_travis_supplements_are_discovered_on_the_assessor_id():
    state = HarmonizeState(
        recipe={}, admin_id=AdminId('US-TX-TRA'), verbose=False, timer=None
    )
    found = {
        m['recipe_id']: m['key']
        for m in links._discover_link_sources(state, 'property')
    }
    roll = 'US-TX-TRA_property-traviscad-2026'
    assert found[roll] == 'parcel_id_local'
    for table in ('bathrooms', 'bedrooms', 'half-bathrooms', 'stories'):
        assert found[f'{roll}_{table}'] == _KEY
