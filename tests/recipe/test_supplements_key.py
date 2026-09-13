"""`supplements_key`: its validation, and the committed declarations.

A supplement joins its roll on `parcel_id_local` unless it names another
column both tables produce. `get_supplements_key` checks that from the
recipes alone. Recipe dicts below are fabricated.
"""

import re

import pytest

import openplaces.recipe as recipe_module
from openplaces.core.schema import ENTITY_TYPES
from openplaces.diagnostics import find_recipes
from openplaces.recipe import (
    get_recipe_by_id,
    get_recipe_dependencies,
    get_supplements_key,
)

_ROLL = 'US-XX-YY_property-roll-2026'
_KEY = 'parcel_id_assessor'
_STRIP = {'type': 'string', 'operation': 'lstrip', 'input': 'ACCT', 'output': _KEY}


def _roll(**extra):
    return {
        'recipe_id': _ROLL,
        'columns': {'parcel_id_admin2_2': 'GEO', 'total_value': 'VAL'},
        'transformations': [_STRIP],
        'parcel_id_local': {'source': 'parcel_id_admin2_2', 'kind': 'parcel'},
        **extra,
    }


def _detail(**extra):
    return {
        'recipe_id': f'{_ROLL}_bedrooms',
        'supplements': _ROLL,
        'supplements_key': _KEY,
        'columns': {'n_bedrooms': 'BEDS'},
        'transformations': [_STRIP],
        **extra,
    }


@pytest.fixture
def rolls(monkeypatch):
    tables = {_ROLL: _roll()}
    monkeypatch.setattr(recipe_module, 'get_recipe_by_id', lambda rid, **k: tables[rid])
    return tables


def test_a_key_both_tables_produce_is_returned(rolls):
    assert get_supplements_key(_detail()) == _KEY


def test_no_declared_key_returns_none(rolls):
    assert get_supplements_key(_detail(supplements_key=None)) is None


def test_a_key_without_supplements_raises(rolls):
    with pytest.raises(ValueError, match='no supplements'):
        get_supplements_key(_detail(supplements=None))


def test_a_key_that_is_not_a_name_raises(rolls):
    with pytest.raises(ValueError, match='must be a column name'):
        get_supplements_key(_detail(supplements_key=['a', 'b']))


def test_a_key_the_supplement_does_not_produce_raises(rolls):
    with pytest.raises(ValueError, match=re.escape(f'{_ROLL}_bedrooms does not')):
        get_supplements_key(_detail(transformations=[]))


def test_a_dropped_key_is_not_produced(rolls):
    with pytest.raises(ValueError, match='does not produce'):
        get_supplements_key(_detail(drop_columns=[_KEY]))


def test_a_key_the_roll_does_not_produce_raises(rolls):
    rolls[_ROLL] = _roll(transformations=[])
    with pytest.raises(ValueError, match=re.escape(f'but {_ROLL} does not')):
        get_supplements_key(_detail())


def test_a_key_mapped_in_columns_is_produced(rolls):
    rolls[_ROLL] = _roll(transformations=[], columns={_KEY: 'ACCT'})
    assert get_supplements_key(_detail()) == _KEY


def test_unnamed_columns_cannot_be_checked_statically(rolls):
    # The join raises instead if the column is missing from the data.
    rolls[_ROLL] = _roll(transformations=[], keep_unnamed_columns=True)
    assert get_supplements_key(_detail()) == _KEY


@pytest.mark.parametrize('entity_type', sorted(ENTITY_TYPES))
def test_every_committed_supplements_key_is_produced_by_both_tables(entity_type):
    df = find_recipes(entity_type, stage='ingest')
    if df.empty:
        return
    keyed = df[df['supplements_key'] != '']
    for recipe_id, key in zip(keyed['recipe_id'], keyed['supplements_key']):
        assert get_supplements_key(get_recipe_by_id(recipe_id)) == key, recipe_id


TRAVIS_ROLL = 'US-TX-TRA_property-traviscad-2026'
TRAVIS_TABLES = ['bathrooms', 'bedrooms', 'half-bathrooms', 'stories']


@pytest.mark.parametrize('table', TRAVIS_TABLES)
def test_travis_room_tables_join_the_roll_on_its_assessor_id(table):
    detail = get_recipe_by_id(f'{TRAVIS_ROLL}_{table}')
    assert detail['supplements'] == TRAVIS_ROLL
    assert get_supplements_key(detail) == _KEY
    # The key relates the table to its roll only; it needs no
    # parcel_id_local of its own.
    assert 'parcel_id_local' not in detail


def test_travis_roll_links_parcels_on_geo_id():
    roll = get_recipe_by_id(TRAVIS_ROLL)
    assert roll['columns']['parcel_id_admin2_2'] == 'geo_id'
    assert roll['parcel_id_local']['source'] == 'parcel_id_admin2_2'


def _upstreams(recipe_id):
    return {
        edge.upstream_recipe_id
        for edge in get_recipe_dependencies(recipe_id, admin_id='US-TX-TRA')
    }


def test_keyed_supplements_are_inputs_of_the_property_spine_only():
    # The parcel geospine's property join skips a keyed supplement, so
    # it must not be listed (and fingerprinted) as that job's input.
    tables = {f'{TRAVIS_ROLL}_{t}' for t in TRAVIS_TABLES}
    assert tables <= _upstreams('US_property-spine-2026')
    geospine = _upstreams('US_parcel-geospine-2026')
    assert not tables & geospine
    assert TRAVIS_ROLL in geospine
