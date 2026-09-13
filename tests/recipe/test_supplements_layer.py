"""Supplements of an additional layer, at a scope inside their roll's.

A city's assessing table details properties a statewide layer already
holds (Massachusetts: the MassGIS assessing table is the `property`
layer of the parcel recipe). It names the host recipe and the layer
(`supplements_layer`), at a scope contained in the host's. Recipe
dicts, ids and values below are fabricated.
"""

import pytest

import openplaces.recipe as recipe_module
from openplaces.core.schema import AdminId, Entity
from openplaces.recipe import get_recipe_dependencies, get_supplemented_table

_HOST = 'US-XX_parcel-statewide-2025'
_CITY = 'US-XX-YY_property-citytable-2026'


def _host(**extra):
    return {
        'recipe_id': _HOST,
        'admin_id': AdminId('US-XX'),
        'stage': 'ingest',
        'entity': Entity('parcel', 'statewide', '2025'),
        'columns': {'parcel_id_assessor': 'MAP'},
        'additional_layers': [
            {
                'layer': 'ASSESS',
                'entity': Entity('property', 'statewide', '2025'),
                'columns': {'property_id_assessor': 'PROP'},
            }
        ],
        **extra,
    }


def _city(**extra):
    return {
        'recipe_id': _CITY,
        'admin_id': AdminId('US-XX-YY'),
        'stage': 'ingest',
        'entity': Entity('property', 'citytable', '2026'),
        'supplements': _HOST,
        'supplements_layer': 'property',
        'columns': {'n_bedrooms': 'BEDS'},
        **extra,
    }


@pytest.fixture
def tables(monkeypatch):
    found = {_HOST: _host()}
    monkeypatch.setattr(recipe_module, 'get_recipe_by_id', lambda rid, **k: found[rid])
    return found


def test_a_layer_supplement_resolves_to_the_layer_table(tables):
    table = get_supplemented_table(_city())
    assert str(table['entity'].entity_type) == 'property'
    assert table['columns'] == {'property_id_assessor': 'PROP'}
    # The layer inherits its host's scope, which contains the city's.
    assert str(table['admin_id']) == 'US-XX'


def test_a_recipe_that_supplements_nothing_returns_none(tables):
    recipe = _city(supplements=None, supplements_layer=None)
    assert get_supplemented_table(recipe) is None


def test_a_layer_without_supplements_raises(tables):
    with pytest.raises(ValueError, match='supplements_layer but no supplements'):
        get_supplemented_table(_city(supplements=None))


def test_a_layer_that_is_not_a_name_raises(tables):
    with pytest.raises(ValueError, match='must be an entity type'):
        get_supplemented_table(_city(supplements_layer=['property']))


def test_an_unknown_layer_raises(tables):
    with pytest.raises(ValueError, match="no such additional layer.*'property'"):
        get_supplemented_table(_city(supplements_layer='transaction'))


def test_a_layer_of_another_entity_type_raises(tables):
    tables[_HOST]['additional_layers'].append(
        {'layer': 'SALES', 'entity': Entity('transaction', 'statewide', '2025')}
    )
    with pytest.raises(ValueError, match='entities of its own type'):
        get_supplemented_table(_city(supplements_layer='transaction'))


def test_naming_the_host_without_its_layer_raises_with_a_hint(tables):
    # The host's own rows are parcels; a property table cannot add
    # columns to them, but the host's property layer is right there.
    with pytest.raises(ValueError, match='supplements_layer: property'):
        get_supplemented_table(_city(supplements_layer=None))


def test_a_supplement_outside_its_rolls_scope_raises(tables):
    with pytest.raises(ValueError, match='does not contain it'):
        get_supplemented_table(_city(admin_id=AdminId('US-ZZ-YY')))


def test_scope_is_contained_by_level_not_by_string_prefix(tables):
    # 'US-XX-WA' is a string prefix of 'US-XX-WAR' but not its parent.
    tables[_HOST]['admin_id'] = AdminId('US-XX-WA')
    with pytest.raises(ValueError, match='does not contain it'):
        get_supplemented_table(_city(admin_id=AdminId('US-XX-WAR')))


def test_a_supplement_of_a_supplement_raises(tables):
    tables[_HOST]['supplements'] = 'US-XX_parcel-other-2025'
    with pytest.raises(ValueError, match='itself a supplement'):
        get_supplemented_table(_city())


def test_a_same_scope_roll_still_resolves(tables):
    # The pre-existing case: a detail table beside its own roll.
    roll = 'US-XX-YY_property-roll-2026'
    tables[roll] = {
        'recipe_id': roll,
        'admin_id': AdminId('US-XX-YY'),
        'stage': 'ingest',
        'entity': Entity('property', 'roll', '2026'),
    }
    detail = _city(supplements=roll, supplements_layer=None)
    assert get_supplemented_table(detail) is tables[roll]


def _upstreams_by_step(recipe, admin_id):
    by_step: dict[str, set] = {}
    for edge in get_recipe_dependencies(recipe, admin_id=AdminId(admin_id)):
        by_step.setdefault(edge.step, set()).add(edge.upstream_recipe_id)
    return by_step


def test_a_city_layer_supplement_feeds_only_its_citys_supplements_join(monkeypatch):
    """Never a spine source; an input of the property spine's join.

    The statewide host reaches the union through its property layer;
    the city table reaches only the supplements_only join, and only
    for its own city.
    """
    scanned = (
        {
            'recipe_id': _CITY,
            'admin_id': 'US-XX-YY',
            'specificity': 3,
            'version': '2026',
            'supplements': _HOST,
        },
    )
    layer = {
        'recipe_id': _HOST,
        'layer': 'property',
        'label': 'statewide',
        'layer_key': 'parcel_id_admin2',
    }
    monkeypatch.setattr(
        recipe_module, '_scan_ingest_recipe_ids', lambda entity_type: scanned
    )
    monkeypatch.setattr(
        recipe_module, 'find_additional_layer_recipes', lambda *a, **k: [layer]
    )
    discover = {'auto_discover': True, 'entity_type': 'property'}
    spine = {
        'admin_id': AdminId('US'),
        'entity': Entity('property', 'spine', '2026'),
        'pipeline': [
            {'step': 'union_spine_sources', 'sources': [discover]},
            {
                'step': 'link_by_id',
                **discover,
                'supplements_only': True,
                'count_as': False,
            },
        ],
    }

    city = _upstreams_by_step(spine, 'US-XX-YY')
    assert city['union_spine_sources'] == {_HOST}
    assert _CITY in city['link_by_id']

    other = _upstreams_by_step(spine, 'US-XX-ZZ')
    assert all(_CITY not in upstreams for upstreams in other.values())
