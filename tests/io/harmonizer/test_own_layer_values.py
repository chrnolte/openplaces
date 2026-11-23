"""A lot keeps the values of the parcel layer that draws it.

Fabricated layers: a statewide layer ('statelayer') and a county's own
layer ('countylayer') covering the same lots, as in the North Carolina
counties with both (plans/lot-keeps-own-layer-values.md). Ids, values
and labels are invented.
"""

from types import SimpleNamespace

import pandas as pd
import pytest

import openplaces.io.harmonizer.links as links
from openplaces.core.schema import AdminId
from openplaces.io.harmonizer import HarmonizeState
from openplaces.io.harmonizer.links.property_values import (
    PROPERTY_WRITES_KEY,
    note_write,
    snapshot,
    withhold_property_values,
)
from tests.links_patching import patch_links

# The county layer keys two of the three lots, below the match-rate
# warning's threshold, by design of the fixture.
pytestmark = pytest.mark.filterwarnings('ignore:link_by_id. countylayer matched')

STATE = 'statelayer'
COUNTY = 'countylayer'


def _match(recipe_id, layer=None, outdated=False):
    return {
        'recipe_id': recipe_id,
        'layer': layer,
        'key': 'parcel_id_local',
        'aggregation_function': None,
        'values_outdated': outdated,
    }


def _refs():
    return {
        STATE: pd.DataFrame(
            {
                'parcel_id_local': ['A', 'B', 'S'],
                'improvement_value': [900.0, 800.0, 500.0],
                'total_value': [1900.0, 1800.0, 1500.0],
                'use_group': ['', '', 'Residential'],
            }
        ),
        COUNTY: pd.DataFrame(
            {
                'parcel_id_local': ['A', 'B'],
                'improvement_value': [100.0, 200.0],
                'total_value': [1100.0, None],
                'use_group': ['Residential', 'Commercial'],
            }
        ),
    }


def _spine():
    # As resolve_spine leaves it: the keep columns from each lot's own
    # drawing layer. Lot c2's county row states no improvement.
    return pd.DataFrame(
        {
            'parcel_id_local': ['A', 'B', 'S'],
            'improvement_value': [100.0, None, 500.0],
            'use_group': ['Residential', 'Commercial', 'Residential'],
            'geometry_source': [COUNTY, COUNTY, STATE],
        },
        index=pd.Index(['c1', 'c2', 's1'], name='parcel_id'),
    )


def _run(monkeypatch, matches, refs=None, entity_type='parcel', spine=None):
    refs = refs or _refs()
    patch_links(monkeypatch, '_discover_link_sources', lambda *a, **k: matches)
    patch_links(monkeypatch, '_apply_remap_csvs', lambda state, recipe_id: state)
    patch_links(
        monkeypatch,
        'get_entities',
        lambda recipe_id, admin_id, layer=None, **kwargs: refs[recipe_id],
    )
    state = HarmonizeState(
        recipe={},
        admin_id=AdminId('US-NC-ZZ'),
        verbose=False,
        timer=None,
        spine=_spine() if spine is None else spine,
    )
    state.metadata['spine_source_recipe_ids'] = {STATE, COUNTY}
    state.metadata['spine_keep_columns'] = {'improvement_value', 'use_group'}
    state = links.link_by_id(
        state,
        auto_discover=True,
        entity_type=entity_type,
        columns=['improvement_value', 'total_value', 'use_group'],
        track_provenance=['improvement_value', 'total_value', 'use_group'],
    )
    return state.spine


def test_a_coarser_parcel_layer_only_fills_the_lots_another_layer_draws(
    monkeypatch,
):
    spine = _run(monkeypatch, [_match(STATE), _match(COUNTY)])
    # The county's own values stand on its lots, the statewide layer
    # covers every lot and still does not replace them.
    assert spine.loc['c1', 'improvement_value'] == 100.0
    # An empty cell is filled from the other layer.
    assert spine.loc['c2', 'improvement_value'] == 800.0
    assert spine.loc['c2', 'improvement_value_source'] == STATE
    assert spine.loc['c2', 'total_value'] == 1800.0
    # The statewide layer's lot keeps its own value.
    assert spine.loc['s1', 'improvement_value'] == 500.0
    assert spine.loc['s1', 'total_value'] == 1500.0


def test_one_lot_takes_every_column_from_its_own_layer(monkeypatch):
    # Before the rule, c1 read the statewide improvement (900) beside
    # the county total (1100).
    spine = _run(monkeypatch, [_match(STATE), _match(COUNTY)])
    assert spine.loc['c1', ['improvement_value', 'total_value']].tolist() == [
        100.0,
        1100.0,
    ]
    assert spine.loc['c1', 'total_value_source'] == COUNTY


def test_the_order_of_the_layers_does_not_decide(monkeypatch):
    spine = _run(monkeypatch, [_match(COUNTY), _match(STATE)])
    assert spine.loc['c1', 'improvement_value'] == 100.0
    assert spine.loc['c1', 'total_value'] == 1100.0
    assert spine.loc['s1', 'total_value'] == 1500.0


def test_an_empty_label_does_not_replace_the_own_layer_label(monkeypatch):
    spine = _run(monkeypatch, [_match(STATE), _match(COUNTY)])
    assert spine.loc['c1', 'use_group'] == 'Residential'
    assert spine.loc['c2', 'use_group'] == 'Commercial'


def test_values_outdated_lets_the_other_layer_replace_the_values(monkeypatch):
    spine = _run(monkeypatch, [_match(STATE), _match(COUNTY, outdated=True)])
    assert spine.loc['c1', 'improvement_value'] == 900.0
    assert spine.loc['c1', 'improvement_value_source'] == STATE
    assert spine.loc['c1', 'total_value'] == 1900.0
    assert spine.loc['c1', 'total_value_source'] == STATE
    # The statewide layer states no label (an empty string), so the
    # county's stays.
    assert spine.loc['c1', 'use_group'] == 'Residential'
    assert spine.loc['s1', 'total_value'] == 1500.0


def test_a_roll_still_overwrites_a_parcel_layer(monkeypatch):
    refs = {
        'countyroll': pd.DataFrame(
            {
                'parcel_id_local': ['A', 'B', 'S'],
                'improvement_value': [150.0, 250.0, 550.0],
            }
        )
    }
    spine = _spine()
    spine['total_value'] = [1100.0, None, 1500.0]
    out = _run(
        monkeypatch,
        [_match('countyroll')],
        refs=refs,
        entity_type='property',
        spine=spine,
    )
    assert out['improvement_value'].tolist() == [150.0, 250.0, 550.0]


def test_a_bundled_property_layer_keeps_the_coverage_rule(monkeypatch):
    refs = {
        STATE: pd.DataFrame(
            {
                'parcel_id_local': ['A', 'B', 'S'],
                'improvement_value': [150.0, 250.0, 550.0],
            }
        )
    }
    out = _run(monkeypatch, [_match(STATE, layer='property')], refs=refs)
    assert out['improvement_value'].tolist() == [150.0, 250.0, 550.0]


@pytest.mark.filterwarnings('ignore:link_by_id. stateunits matched')
def test_a_filling_layer_decides_a_cell_a_property_copy_occupied(monkeypatch):
    # Lot s1: the statewide layer draws it and states no total; its
    # bundled property layer copies one in; the county layer, which may
    # only fill there, states 1700. Once the property's copy is
    # withdrawn the county's value is the one the lot keeps.
    refs = {
        STATE: pd.DataFrame(
            {
                'parcel_id_local': ['A', 'B', 'S'],
                'total_value': [1900.0, 1800.0, None],
            }
        ),
        'stateunits': pd.DataFrame({'parcel_id_local': ['S'], 'total_value': [1234.0]}),
        COUNTY: pd.DataFrame(
            {
                'parcel_id_local': ['A', 'B', 'S'],
                'total_value': [1100.0, None, 1700.0],
            }
        ),
    }
    matches = [
        _match(STATE),
        _match('stateunits', layer='property'),
        _match(COUNTY),
    ]
    patch_links(monkeypatch, '_discover_link_sources', lambda *a, **k: matches)
    patch_links(monkeypatch, '_apply_remap_csvs', lambda state, recipe_id: state)
    patch_links(
        monkeypatch,
        'get_entities',
        lambda recipe_id, admin_id, layer=None, **kwargs: refs[recipe_id],
    )
    state = HarmonizeState(
        recipe={}, admin_id=AdminId('US-NC-ZZ'), verbose=False, timer=None
    )
    state.spine = _spine()
    state.metadata['spine_source_recipe_ids'] = {STATE, COUNTY}
    state = links.link_by_id(
        state,
        auto_discover=True,
        columns=['total_value'],
        track_provenance=['total_value'],
    )
    # In memory the property's copy stands, as the cut reads it.
    assert state.spine.loc['s1', 'total_value'] == 1234.0
    state = withhold_property_values(state, columns=['total_value'])
    assert state.spine.loc['s1', 'total_value'] == 1700.0
    assert state.spine.loc['s1', 'total_value_source'] == COUNTY
    assert state.spine.loc['c1', 'total_value'] == 1100.0


def test_an_empty_string_made_missing_is_no_property_write():
    # A property write states nothing for lot a, whose layer wrote an
    # empty string; the write still makes it missing, and that is no
    # property's value to take back.
    index = pd.Index(['a', 'b'], name='parcel_id')
    spine = pd.DataFrame(
        {'building_style': pd.Series(['', 'Ranch'], index=index, dtype=object)},
        index=index,
    )
    state = HarmonizeState(
        recipe={}, admin_id=AdminId('US-NC-ZZ'), verbose=False, timer=None
    )
    state.spine = spine
    before = snapshot(spine, 'building_style')
    links._write_prioritized(
        spine,
        'building_style',
        pd.Series([None, 'Colonial'], index=index, dtype=object),
    )
    note_write(state, spine, 'building_style', before, from_properties=True)
    held = state.metadata[PROPERTY_WRITES_KEY]['building_style']['held']
    # Lot b took the property's value; lot a took none.
    assert held.tolist() == [False, True]
    state = withhold_property_values(state, columns=['building_style'])
    assert pd.isna(state.spine.loc['a', 'building_style'])
    # Lot b returns to its layer's own statement.
    assert state.spine.loc['b', 'building_style'] == 'Ranch'


def test_write_prioritized_treats_an_empty_string_as_missing():
    spine = pd.DataFrame({'label': ['kept', '', None]}, index=['a', 'b', 'c'])
    links._write_prioritized(
        spine, 'label', pd.Series(['', 'new', ' '], index=spine.index)
    )
    assert spine['label'].tolist()[:2] == ['kept', 'new']
    assert pd.isna(spine.loc['c', 'label'])


def test_write_prioritized_first_write_drops_empty_strings():
    spine = pd.DataFrame(index=['a', 'b'])
    links._write_prioritized(
        spine, 'label', pd.Series(['', 'x'], index=['a', 'b'], dtype='string')
    )
    assert pd.isna(spine.loc['a', 'label'])
    assert spine.loc['b', 'label'] == 'x'


def test_write_prioritized_treats_a_blank_category_as_missing():
    spine = pd.DataFrame({'label': pd.Categorical(['kept', ''])}, index=['a', 'b'])
    new_vals = pd.Series(pd.Categorical(['', 'new']), index=['a', 'b'])
    links._write_prioritized(spine, 'label', new_vals)
    assert spine['label'].astype(object).tolist() == ['kept', 'new']


def test_discovery_carries_the_county_declaration():
    # Onslow's own layer declares its values outdated; NC OneMap does not.
    state = HarmonizeState(
        recipe={}, admin_id=AdminId('US-NC-ONS'), verbose=False, timer=None
    )
    matches = links._discover_link_sources(state, 'parcel')
    declared = {m['recipe_id']: m['values_outdated'] for m in matches if not m['layer']}
    assert declared == {
        'US-NC_parcel-nconemap-2025': False,
        'US-NC-ONS_parcel-onslowcounty-2026': True,
    }


def test_values_outdated_must_be_a_boolean_on_a_parcel_recipe():
    parcel = SimpleNamespace(entity_type='parcel')
    roll = SimpleNamespace(entity_type='property')
    assert links.values_outdated({'entity': parcel}) is False
    assert links.values_outdated({'entity': parcel, 'values_outdated': True})
    with pytest.raises(ValueError, match='true or false'):
        links.values_outdated({'entity': parcel, 'values_outdated': 'yes'})
    with pytest.raises(ValueError, match='parcel layer'):
        links.values_outdated({'entity': roll, 'values_outdated': True})
