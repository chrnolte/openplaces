"""A curated parcel gets its property attributes reduced from the property
spine, with the rule each column needs, filling only what it lacks.

The parcel geospine copies no property-level attribute (year built,
living area, room counts) from a roll any more; the curate step reads the
harmonized property spine for the unit and reduces it per parcel. The
year built is the earliest property's, the living area the sum, and a
parcel whose own layer already states a value keeps it.
"""

import pandas as pd
import pytest

from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState, aggregation


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
            'parcel_id_local': ['p1', 'p1', 'p2', None, 'p9', 'p4'],
            'year_built': [1990, 1965, 2001, 1950, 1880, None],
            'living_area_sqft': [1200, 800, 950, 300, 500, None],
            'n_bedrooms': [3, 2, 4, 1, 1, None],
        }
    )
    calls = []

    def fake_get_entities(recipe_id, admin_id, columns=None, missing='raise'):
        calls.append((recipe_id, str(admin_id), tuple(columns)))
        return rows[[c for c in columns if c in rows.columns]]

    monkeypatch.setattr(aggregation, 'get_entities', fake_get_entities)
    monkeypatch.setattr(
        aggregation,
        'describe_recipe',
        lambda recipe_id, admin_id: pd.DataFrame(index=rows.columns),
    )
    return calls


def test_each_column_is_reduced_with_its_own_rule(properties):
    curated = pd.DataFrame(
        {'parcel_id_local': ['p1', 'p2', 'p3'], 'year_built': [None, None, 1977]}
    )
    state = aggregation.aggregate_from_entities(
        _state(curated),
        'US_property-spine-2026',
        columns={'year_built': 'min', 'living_area_sqft': 'sum', 'n_bedrooms': 'sum'},
    )
    out = state.curated
    assert out['year_built'].tolist() == [1965, 2001, 1977]
    assert out['living_area_sqft'].tolist()[:2] == [2000, 950]
    assert pd.isna(out['living_area_sqft'][2])
    assert out['n_bedrooms'].tolist()[:2] == [5, 4]
    assert properties[0][0] == 'US_property-spine-2026'
    assert properties[0][1] == 'US-XX-YY'


def test_a_parcel_whose_properties_state_nothing_gets_no_zero(properties):
    curated = pd.DataFrame({'parcel_id_local': ['p4']})
    state = aggregation.aggregate_from_entities(
        _state(curated), 'r', columns={'living_area_sqft': 'sum', 'year_built': 'min'}
    )
    assert state.curated[['living_area_sqft', 'year_built']].isna().all().all()


def test_a_value_the_parcel_layer_states_is_kept(properties):
    curated = pd.DataFrame({'parcel_id_local': ['p1'], 'year_built': [1999]})
    state = aggregation.aggregate_from_entities(
        _state(curated), 'r', columns={'year_built': 'min'}
    )
    assert state.curated['year_built'].tolist() == [1999]


def test_fill_only_false_replaces_the_stated_value(properties):
    curated = pd.DataFrame({'parcel_id_local': ['p1'], 'year_built': [1999]})
    state = aggregation.aggregate_from_entities(
        _state(curated), 'r', columns={'year_built': 'min'}, fill_only=False
    )
    assert state.curated['year_built'].tolist() == [1965]


def test_a_list_uses_the_registry_rule(properties):
    curated = pd.DataFrame({'parcel_id_local': ['p1', 'p2']})
    state = aggregation.aggregate_from_entities(
        _state(curated), 'r', columns=['n_bedrooms']
    )
    # The registry averages a bedroom count across the rows sharing a key.
    assert state.curated['n_bedrooms'].tolist() == [2.5, 4]


def test_a_missing_reference_is_skipped_with_a_warning(monkeypatch):
    def missing(*a, **k):
        raise FileNotFoundError('no spine')

    monkeypatch.setattr(aggregation, 'describe_recipe', missing)
    curated = pd.DataFrame({'parcel_id_local': ['p1']})
    with pytest.warns(UserWarning, match='could not read'):
        state = aggregation.aggregate_from_entities(
            _state(curated), 'r', columns={'year_built': 'min'}
        )
    assert 'year_built' not in state.curated.columns


def test_the_first_key_present_on_both_sides_is_used(properties):
    curated = pd.DataFrame({'parcel_id_admin2': ['p1'], 'other_id': ['x']})
    state = aggregation.aggregate_from_entities(
        _state(curated),
        'r',
        columns={'year_built': 'min'},
        key=['other_id', 'parcel_id_admin2'],
        reference_key=['missing_here', 'parcel_id_local'],
    )
    assert state.curated['year_built'].tolist() == [1965]


def test_a_link_table_replaces_the_key_join(monkeypatch):
    # Lot A: two roll rows keyed on its units, plus the split's own
    # unit rows describing the same units (ids that did not converge).
    # Lots C and D: one account on both, three quarters on C by area.
    rows = pd.DataFrame(
        {
            'parcel_id_local': ['A1', 'A2', 'A1', 'A2', 'CD'],
            'living_area_sqft': [800.0, 900.0, 810.0, 910.0, 4000.0],
            'year_built': [1990, 1985, 1991, 1986, 1970],
        },
        index=pd.Index(['r1', 'r2', 'u1', 'u2', 'r4'], name='property_id'),
    )
    links = pd.DataFrame(
        {
            'property_id': ['r1', 'r2', 'u1', 'u2', 'r4', 'r4'],
            'parcel_id': ['pa', 'pa', 'pa', 'pa', 'pc', 'pd'],
            'link_method': ['stacked_units_crosswalk'] * 2
            + ['stacked_units'] * 2
            + ['stacked_units_crosswalk'] * 2,
            'link_source': [
                'roll',
                'roll',
                'layer:units',
                'layer:units',
                'roll',
                'roll',
            ],
            'share': pd.array([None, None, None, None, 0.75, 0.25], dtype='Float64'),
            'share_basis': [None, None, None, None, 'area', 'area'],
        }
    )
    from pathlib import Path

    from openplaces.geo import link as geo_link
    from openplaces.io.harmonizer import entity_links

    monkeypatch.setattr(
        aggregation,
        'get_entities',
        lambda recipe_id, admin_id, columns=None, missing='raise': rows[
            [c for c in columns if c in rows.columns]
        ],
    )
    monkeypatch.setattr(
        aggregation,
        'describe_recipe',
        lambda recipe_id, admin_id: pd.DataFrame(index=rows.columns),
    )
    monkeypatch.setattr(geo_link, 'get_link_owner_recipe_id', lambda recipe: 'owner')
    monkeypatch.setattr(
        geo_link, 'get_entity_link_path', lambda a, b, admin_id=None: Path('link')
    )
    monkeypatch.setattr(entity_links, 'read_entity_link', lambda path: links)

    curated = pd.DataFrame(
        {'parcel_id_local': ['A', 'C', 'D']},
        index=pd.Index(['pa', 'pc', 'pd'], name='parcel_id'),
    )
    state = aggregation.aggregate_from_entities(
        _state(curated),
        'US_property-spine-2026',
        columns={'living_area_sqft': 'sum', 'year_built': 'min'},
    )
    out = state.curated
    # The roll's two rows, not also the split's two: 1,700, not 3,420.
    assert out.loc['pa', 'living_area_sqft'] == 1700.0
    assert out.loc['pa', 'year_built'] == 1985
    # One account on two lots is divided, and its total is conserved.
    assert out.loc['pc', 'living_area_sqft'] == 3000.0
    assert out.loc['pd', 'living_area_sqft'] == 1000.0
    assert out.loc['pd', 'year_built'] == 1970


def test_no_shared_key_skips_with_a_warning(properties):
    curated = pd.DataFrame({'parcel_id_admin2': ['p1']})
    with pytest.warns(UserWarning, match='no key'):
        state = aggregation.aggregate_from_entities(
            _state(curated), 'r', columns={'year_built': 'min'}
        )
    assert 'year_built' not in state.curated.columns
