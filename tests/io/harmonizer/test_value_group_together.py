"""A lot takes its value columns from one parcel layer or not at all.

Fabricated layers: a statewide layer ('statelayer') and a county's own
layer ('countylayer') over the same lots
(plans/lot-keeps-own-layer-values.md). Ids, values, years and labels
are invented.
"""

import pandas as pd
import pytest

import openplaces.io.harmonizer.links as links
from openplaces.core.schema import AdminId
from openplaces.io.harmonizer import HarmonizeState
from tests.links_patching import patch_links

pytestmark = pytest.mark.filterwarnings('ignore:link_by_id. countylayer matched')

STATE = 'statelayer'
COUNTY = 'countylayer'
GROUP = {
    'values': ['land_value', 'improvement_value', 'total_value'],
    'year': 'tax_year',
}
COLUMNS = ['land_value', 'improvement_value', 'total_value', 'tax_year', 'use_group']


def _match(recipe_id, outdated=False):
    return {
        'recipe_id': recipe_id,
        'layer': None,
        'key': 'parcel_id_local',
        'aggregation_function': None,
        'values_outdated': outdated,
    }


def _refs():
    # Lot A: the county states land and total, no improvement (the
    # case that mixed). Lot B: the county states no value at all. Lot
    # S: the statewide layer draws it and states only a total.
    return {
        STATE: pd.DataFrame(
            {
                'parcel_id_local': ['A', 'B', 'S'],
                'land_value': [400.0, 300.0, None],
                'improvement_value': [900.0, 800.0, None],
                'total_value': [1300.0, 1100.0, 1500.0],
                'tax_year': [2025.0, 2025.0, 2025.0],
                'use_group': ['Residential', 'Commercial', 'Residential'],
            }
        ),
        COUNTY: pd.DataFrame(
            {
                'parcel_id_local': ['A', 'B', 'S'],
                'land_value': [1000.0, None, 70.0],
                'improvement_value': [None, None, 80.0],
                'total_value': [1000.0, None, 150.0],
                'tax_year': [2022.0, 2022.0, 2022.0],
                'use_group': [None, None, 'Vacant'],
            }
        ),
    }


def _spine():
    # As resolve_spine leaves it: each lot's drawing layer's keep
    # columns, their `_source` naming that layer.
    spine = pd.DataFrame(
        {
            'parcel_id_local': ['A', 'B', 'S'],
            'land_value': [1000.0, None, None],
            'improvement_value': [None, None, None],
            'geometry_source': [COUNTY, COUNTY, STATE],
        },
        index=pd.Index(['c1', 'c2', 's1'], name='parcel_id'),
    )
    spine['land_value_source'] = pd.Series([COUNTY, None, None], index=spine.index)
    return spine


def _run(monkeypatch, matches, value_group=GROUP, refs=None, spine=None, **kwargs):
    refs = refs or _refs()
    patch_links(monkeypatch, '_discover_link_sources', lambda *a, **k: matches)
    patch_links(monkeypatch, '_apply_remap_csvs', lambda state, recipe_id: state)
    patch_links(
        monkeypatch,
        'get_entities',
        lambda recipe_id, admin_id, layer=None: refs[recipe_id],
    )
    state = HarmonizeState(
        recipe={}, admin_id=AdminId('US-NC-ZZ'), verbose=False, timer=None
    )
    state.spine = _spine() if spine is None else spine
    state.metadata['spine_source_recipe_ids'] = {STATE, COUNTY}
    state.metadata['spine_keep_columns'] = {'land_value', 'improvement_value'}
    state = links.link_by_id(
        state,
        auto_discover=True,
        columns=COLUMNS,
        track_provenance=['land_value', 'improvement_value', 'total_value'],
        value_group=value_group,
        **kwargs,
    )
    return state.spine


@pytest.mark.parametrize('order', [(STATE, COUNTY), (COUNTY, STATE)])
def test_a_lot_takes_no_value_from_the_other_layer_beside_its_own(monkeypatch, order):
    spine = _run(monkeypatch, [_match(r) for r in order])
    # The county states land and total for lot c1; the statewide
    # improvement does not join them, whichever layer comes first.
    row = spine.loc['c1']
    assert row['land_value'] == 1000.0
    assert row['total_value'] == 1000.0
    assert pd.isna(row['improvement_value'])
    assert pd.isna(row['improvement_value_source'])
    assert row['total_value_source'] == COUNTY
    # The year is the county's, the year of the values the lot keeps.
    assert row['tax_year'] == 2022.0


@pytest.mark.parametrize('order', [(STATE, COUNTY), (COUNTY, STATE)])
def test_the_other_layer_fills_the_whole_group_where_none_is_stated(monkeypatch, order):
    spine = _run(monkeypatch, [_match(r) for r in order])
    row = spine.loc['c2']
    assert row[['land_value', 'improvement_value', 'total_value']].tolist() == [
        300.0,
        800.0,
        1100.0,
    ]
    assert row['total_value_source'] == STATE
    assert row['tax_year'] == 2025.0


def test_other_columns_still_fill_cell_by_cell(monkeypatch):
    spine = _run(monkeypatch, [_match(STATE), _match(COUNTY)])
    # The county states no use_group for c1, so the statewide label
    # fills it, beside the county's values.
    assert spine.loc['c1', 'use_group'] == 'Residential'


def test_the_statewide_lot_keeps_its_own_group(monkeypatch):
    spine = _run(monkeypatch, [_match(COUNTY), _match(STATE)])
    # The statewide layer states only a total for s1: the county's
    # land and improvement do not join it.
    row = spine.loc['s1']
    assert row['total_value'] == 1500.0
    assert pd.isna(row['land_value'])
    assert pd.isna(row['improvement_value'])
    assert row['tax_year'] == 2025.0


def test_without_a_group_the_cells_fill_one_by_one(monkeypatch):
    spine = _run(monkeypatch, [_match(STATE), _match(COUNTY)], value_group=None)
    assert spine.loc['c1', 'improvement_value'] == 900.0


def test_an_outdated_layer_does_not_fill_beside_the_other_layers_group(
    monkeypatch,
):
    spine = _run(monkeypatch, [_match(STATE), _match(COUNTY, outdated=True)])
    # The statewide layer replaces the outdated county's group on c1.
    row = spine.loc['c1']
    assert row[['land_value', 'improvement_value', 'total_value']].tolist() == [
        400.0,
        900.0,
        1300.0,
    ]
    assert row['land_value_source'] == STATE
    assert row['tax_year'] == 2025.0


def test_a_lossy_pass_fills_the_group_together(monkeypatch):
    # Only the lossy pass meets the statewide rows here; it may not add
    # an improvement beside the county's land on c1.
    spine = _run(monkeypatch, [_match(STATE)], fill_only=True)
    assert spine.loc['c1', 'land_value'] == 1000.0
    assert pd.isna(spine.loc['c1', 'improvement_value'])
    assert pd.isna(spine.loc['c1', 'total_value'])
    # Lot c2 states nothing, so the whole group fills.
    assert spine.loc['c2', 'improvement_value'] == 800.0
    assert spine.loc['c2', 'land_value'] == 300.0


def test_a_lossy_pass_never_empties_what_a_precise_pass_wrote(monkeypatch):
    # The county's precise pass writes c1's total; its lossy pass meets
    # c1 again on a key whose rows state land but no total. The lot
    # holds only the county's own values, so the gap fill leaves the
    # total as it is.
    spine = _run(monkeypatch, [_match(COUNTY)])
    assert spine.loc['c1', 'total_value'] == 1000.0
    lossy = _refs()
    lossy[COUNTY] = lossy[COUNTY].assign(total_value=None)
    spine = _run(monkeypatch, [_match(COUNTY)], refs=lossy, spine=spine, fill_only=True)
    assert spine.loc['c1', 'total_value'] == 1000.0
    assert spine.loc['c1', 'land_value'] == 1000.0


def test_value_group_is_validated():
    assert links.parse_value_group(None) is None
    parsed = links.parse_value_group({'values': ['land_value'], 'year': 'tax_year'})
    assert parsed == {'values': ('land_value',), 'year': 'tax_year'}
    with pytest.raises(ValueError, match='unknown key'):
        links.parse_value_group({'values': ['land_value'], 'weights': [1]})
    with pytest.raises(ValueError, match='at least one column'):
        links.parse_value_group({'values': []})


def test_value_group_needs_auto_discover():
    state = HarmonizeState(
        recipe={}, admin_id=AdminId('US-NC-ZZ'), verbose=False, timer=None
    )
    with pytest.raises(ValueError, match='auto_discover only'):
        links.link_by_id(state, recipe_id=COUNTY, value_group=GROUP)
