"""`link_by_id(layer_columns=...)`: a property layer bundled in a parcel
recipe brings only parcel-level columns onto lots.

All values are fabricated.
"""

import pandas as pd
import pytest

import openplaces.io.harmonizer.links as links
from openplaces.core.schema import AdminId
from openplaces.io.harmonizer import HarmonizeState
from openplaces.recipe import get_recipe_by_id
from tests.links_patching import patch_links

#: Lot L1 has no unit, so the layer matches half the spine and warns.
pytestmark = pytest.mark.filterwarnings('ignore:link_by_id. parcels matched only')

GEOSPINE = 'US_parcel-geospine-2026'

#: The columns a property layer brings to a lot (the geospine's list).
PARCEL_LEVEL = ['land_value', 'improvement_value', 'n_dwellings', 'use_group']

#: Attributes a property states about itself, which the parcel curate
#: reduces from the property spine with its own rule.
PROPERTY_LEVEL = ['year_built', 'n_stories', 'n_rooms', 'n_bedrooms']


def _run(monkeypatch, **kwargs):
    matches = [
        {
            'recipe_id': 'parcels',
            'layer': None,
            'key': 'parcel_id_local',
            'aggregation_function': None,
            'supplements': None,
        },
        {
            'recipe_id': 'parcels',
            'layer': 'property',
            'key': 'lot_id_local',
            'aggregation_function': None,
            'supplements': None,
            'stacked_units_layer': True,
        },
    ]
    tables = {
        # The parcel table's own rows: lot L1 states its own year and
        # stories, lot L2 states nothing of its own.
        None: pd.DataFrame(
            {
                'parcel_id_local': ['L1', 'L2'],
                'year_built': [1950.0, None],
                'n_stories': [2.0, None],
                'land_value': [1000.0, None],
            }
        ),
        # Three units stacked on lot L2.
        'property': pd.DataFrame(
            {
                'parcel_id_local': ['U1', 'U2', 'U3'],
                'lot_id_local': ['L2', 'L2', 'L2'],
                'year_built': [1900.0, 1960.0, 2020.0],
                'n_stories': [1.0, 3.0, 5.0],
                'n_rooms': [4.0, 5.0, 6.0],
                'n_bedrooms': [1.0, 2.0, 3.0],
                'land_value': [10.0, 20.0, 30.0],
                'improvement_value': [100.0, 200.0, 300.0],
                'n_dwellings': [1.0, 1.0, 1.0],
                'use_group': ['residential'] * 3,
            }
        ),
    }

    def _get_entities(recipe_id, admin_id, layer=None, **_):
        return tables[layer].copy()

    patch_links(monkeypatch, '_discover_link_sources', lambda *a, **k: matches)
    patch_links(monkeypatch, 'get_entities', _get_entities)
    patch_links(monkeypatch, '_apply_remap_csvs', lambda state, recipe_id: state)
    spine = pd.DataFrame(
        {'parcel_id_local': ['L1', 'L2']},
        index=pd.Index(['p1', 'p2'], name='parcel_id'),
    )
    state = HarmonizeState(
        recipe={},
        admin_id=AdminId('US-XX-YY'),
        verbose=False,
        timer=None,
        spine=spine,
    )
    return links.link_by_id(state, auto_discover=True, entity_type='parcel', **kwargs)


def test_property_layer_keeps_no_averaged_property_attribute(monkeypatch):
    spine = _run(monkeypatch, layer_columns={'property': PARCEL_LEVEL}).spine

    # Lot p2 carries three units: none of their own attributes lands
    # on it, neither as a mean nor otherwise.
    for column in PROPERTY_LEVEL:
        assert column not in spine.columns or pd.isna(spine.loc['p2', column])
    assert 'n_rooms' not in spine.columns
    assert 'n_bedrooms' not in spine.columns


def test_parcel_table_own_attributes_survive(monkeypatch):
    spine = _run(monkeypatch, layer_columns={'property': PARCEL_LEVEL}).spine

    # The parcel table's own row is still copied on the registry
    # default, so a county whose only building attributes sit there
    # keeps them.
    assert spine.loc['p1', 'year_built'] == 1950.0
    assert spine.loc['p1', 'n_stories'] == 2.0
    assert spine.loc['p1', 'land_value'] == 1000.0


def test_parcel_level_columns_still_arrive_from_the_layer(monkeypatch):
    spine = _run(monkeypatch, layer_columns={'property': PARCEL_LEVEL}).spine

    assert spine.loc['p2', 'land_value'] == 60.0
    assert spine.loc['p2', 'improvement_value'] == 600.0
    assert spine.loc['p2', 'n_dwellings'] == 3.0
    # join_nonnull: every unit's label, joined.
    assert set(spine.loc['p2', 'use_group'].split(' + ')) == {'residential'}


def test_without_layer_columns_the_layer_takes_the_registry_default(monkeypatch):
    # The behavior layer_columns exists to replace: the registry's
    # mean for a property's own counts and years.
    spine = _run(monkeypatch).spine

    assert spine.loc['p2', 'year_built'] == pytest.approx(1960.0)
    assert spine.loc['p2', 'n_stories'] == pytest.approx(3.0)
    assert spine.loc['p2', 'n_rooms'] == pytest.approx(5.0)


def test_layer_columns_needs_auto_discover():
    state = HarmonizeState(
        recipe={},
        admin_id=AdminId('US-XX-YY'),
        verbose=False,
        timer=None,
        spine=pd.DataFrame({'parcel_id_local': ['L1']}),
    )
    with pytest.raises(ValueError, match='auto_discover'):
        links.link_by_id(
            state, recipe_id='parcels', layer_columns={'property': PARCEL_LEVEL}
        )


def test_geospine_parcel_passes_copy_the_property_pass_list_from_layers():
    """Passes 1 to 3 name, for property layers, the list pass 4 copies
    from rolls, and that list holds no property-level attribute."""
    pipeline = get_recipe_by_id(GEOSPINE)['pipeline']
    passes = [
        s for s in pipeline if s.get('step') == 'link_by_id' and s.get('auto_discover')
    ]
    parcel_passes = [s for s in passes if s.get('entity_type') == 'parcel']
    property_pass = next(s for s in passes if s.get('entity_type') == 'property')

    assert len(parcel_passes) == 3
    for step in parcel_passes:
        assert 'columns' not in step
        assert step['layer_columns']['property'] == property_pass['columns']
    for column in PROPERTY_LEVEL + ['living_area_sqft', 'gross_floor_area_sqft']:
        assert column not in property_pass['columns']
