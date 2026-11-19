"""`withhold_property_values`: a property's values leave the lots at the
end of the parcel geospine, and a parcel layer's own values stay.

All values are fabricated.
"""

import numpy as np
import pandas as pd
import pytest

import openplaces.io.harmonizer.links as links
from openplaces.core.schema import AdminId
from openplaces.io.harmonizer import HarmonizeState
from openplaces.io.harmonizer.attributes.property_counts import (
    estimate_property_counts,
)
from openplaces.io.harmonizer.links.additions import account_value_density
from openplaces.io.harmonizer.links.property_values import (
    PROPERTY_WRITES_KEY,
    note_write,
    snapshot,
    withhold_property_values,
)
from openplaces.recipe import get_recipe_by_id
from tests.links_patching import patch_links

pytestmark = pytest.mark.filterwarnings('ignore:link_by_id')

GEOSPINE = 'US_parcel-geospine-2026'
VALUES = ['land_value', 'improvement_value', 'total_value', 'n_dwellings']
TRACKED = ['land_value', 'improvement_value', 'n_dwellings']


def _state(spine):
    return HarmonizeState(
        recipe={}, admin_id=AdminId('US-XX-YY'), verbose=False, timer=None, spine=spine
    )


def _passes(monkeypatch, parcel_rows, unit_rows):
    """A parcel table and its bundled stacked-units layer, in one pass."""
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
    tables = {None: parcel_rows, 'property': unit_rows}

    def _get_entities(recipe_id, admin_id, layer=None, **_):
        return tables[layer].copy()

    patch_links(monkeypatch, '_discover_link_sources', lambda *a, **k: matches)
    patch_links(monkeypatch, 'get_entities', _get_entities)
    patch_links(monkeypatch, '_apply_remap_csvs', lambda state, recipe_id: state)
    spine = pd.DataFrame(
        {'parcel_id_local': ['L1', 'L2', 'L3']},
        index=pd.Index(['p1', 'p2', 'p3'], name='parcel_id'),
    )
    state = _state(spine)
    return links.link_by_id(
        state,
        auto_discover=True,
        entity_type='parcel',
        layer_columns={'property': VALUES},
        track_provenance=TRACKED,
    )


PARCEL_ROWS = pd.DataFrame(
    {
        'parcel_id_local': ['L1', 'L2', 'L3'],
        # The layer states its own land value on L1 and L2, none on L3.
        'land_value': [1000.0, 500.0, None],
        'n_dwellings': [1.0, 0.0, None],
    }
)
UNIT_ROWS = pd.DataFrame(
    {
        'parcel_id_local': ['U1', 'U2', 'U3', 'U4'],
        'lot_id_local': ['L2', 'L2', 'L3', 'L3'],
        'land_value': [10.0, 20.0, 30.0, 40.0],
        'improvement_value': [100.0, 200.0, 300.0, 400.0],
        'n_dwellings': [1.0, 1.0, 1.0, 1.0],
    }
)


def test_the_passes_still_write_property_values_in_memory(monkeypatch):
    # What the shared-land groups, the common-area links and the
    # building geospine's cut read: unchanged by this package.
    spine = _passes(monkeypatch, PARCEL_ROWS, UNIT_ROWS).spine

    assert spine.loc['p2', 'land_value'] == 30.0
    assert spine.loc['p3', 'improvement_value'] == 700.0


def test_a_layer_value_a_property_overwrote_returns(monkeypatch):
    state = _passes(monkeypatch, PARCEL_ROWS, UNIT_ROWS)
    before = state.spine['land_value_source'].astype(object).copy()
    state = withhold_property_values(state, columns=VALUES)
    spine = state.spine

    assert spine.loc['p2', 'land_value'] == 500.0
    assert spine.loc['p2', 'n_dwellings'] == 0.0
    # The lot the units alone described has no value left.
    assert pd.isna(spine.loc['p3', 'land_value'])
    assert pd.isna(spine.loc['p3', 'improvement_value'])
    assert pd.isna(spine.loc['p3', 'land_value_source'])
    # A lot no property wrote keeps its value and its label.
    assert spine.loc['p1', 'land_value'] == 1000.0
    assert spine.loc['p1', 'land_value_source'] == before['p1']


def test_the_label_returns_with_the_value(monkeypatch):
    state = _passes(monkeypatch, PARCEL_ROWS, UNIT_ROWS)
    record = state.metadata[PROPERTY_WRITES_KEY]['land_value']
    label_before = record.loc['p2', 'label']
    state = withhold_property_values(state, columns=['land_value'])

    assert state.spine.loc['p2', 'land_value_source'] == label_before


def test_preliminary_keeps_the_count_before_the_step(monkeypatch):
    state = _passes(monkeypatch, PARCEL_ROWS, UNIT_ROWS)
    state = withhold_property_values(
        state, columns=['n_dwellings'], preliminary=['n_dwellings']
    )
    spine = state.spine

    assert spine.loc['p2', 'n_dwellings_preliminary'] == 2.0
    assert spine.loc['p3', 'n_dwellings_preliminary'] == 2.0
    assert spine.loc['p2', 'n_dwellings'] == 0.0
    assert pd.isna(spine.loc['p3', 'n_dwellings'])


def test_a_later_parcel_write_is_not_taken_back():
    spine = pd.DataFrame(
        {'land_value': [np.nan, np.nan]}, index=pd.Index(['a', 'b'], name='parcel_id')
    )
    state = _state(spine)
    before = snapshot(spine, 'land_value')
    spine['land_value'] = [10.0, 20.0]
    note_write(state, spine, 'land_value', before, from_properties=True)
    # A parcel table overwrites lot a afterwards.
    before = snapshot(spine, 'land_value')
    spine.loc['a', 'land_value'] = 99.0
    note_write(state, spine, 'land_value', before, from_properties=False)
    state = withhold_property_values(state, columns=['land_value'])

    assert state.spine.loc['a', 'land_value'] == 99.0
    assert pd.isna(state.spine.loc['b', 'land_value'])


def test_a_value_the_roll_rule_emptied_stays_empty():
    spine = pd.DataFrame(
        {'land_value': [5.0], 'land_value_source': ['statewide']},
        index=pd.Index(['a'], name='parcel_id'),
    )
    state = _state(spine)
    before = snapshot(spine, 'land_value')
    spine['land_value'] = [7.0]
    spine['land_value_source'] = ['roll']
    note_write(state, spine, 'land_value', before, from_properties=True)
    # not_on_roll empties it afterwards (a write by no property).
    before = snapshot(spine, 'land_value')
    spine['land_value'] = [np.nan]
    spine['land_value_source'] = ['not_on_roll']
    note_write(state, spine, 'land_value', before, from_properties=False)
    state = withhold_property_values(state, columns=['land_value'])

    assert pd.isna(state.spine.loc['a', 'land_value'])
    assert state.spine.loc['a', 'land_value_source'] == 'not_on_roll'


def test_no_record_changes_nothing():
    spine = pd.DataFrame({'land_value': [5.0]}, index=pd.Index(['a'], name='parcel_id'))
    state = withhold_property_values(_state(spine.copy()), columns=['land_value'])

    pd.testing.assert_frame_equal(state.spine, spine)


def test_the_geospine_withholds_last_and_keeps_its_passes():
    pipeline = get_recipe_by_id(GEOSPINE)['pipeline']
    last = pipeline[-1]

    assert last['step'] == 'withhold_property_values'
    # The values and the dwelling count, and since the ruling of
    # 2026-11-18 the roll's tax year and the structure style; the use
    # labels and the address parts are still copied.
    assert set(last['columns']) == {*VALUES, 'tax_year', 'building_style'}
    assert last['preliminary'] == ['n_dwellings']
    # The passes still name the value columns: the building geospine's
    # cut replays them for its structure evidence, and its sidecar
    # fingerprint records them.
    passes = [
        s for s in pipeline if s.get('step') == 'link_by_id' and s.get('auto_discover')
    ]
    roll_pass = next(s for s in passes if s.get('entity_type') == 'property')
    for column in VALUES:
        assert column in roll_pass['columns']


def test_the_cut_and_the_stand_ins_still_read_the_rolls_value(monkeypatch):
    # lot_structure_evidence (the building geospine's cut, and the
    # stand-in gate's lot_values_from) replays the geospine's link
    # passes alone; the withholding step is not among them, so a lot
    # whose only value is a roll's still shows it there.
    from openplaces.io.harmonizer import entity_links
    from openplaces.io.harmonizer.links import parcel_pieces

    recipe = get_recipe_by_id(GEOSPINE)
    _, passes = parcel_pieces._lot_value_steps(recipe)
    assert 'withhold_property_values' not in {p['step'] for p in passes}

    roll = {
        'recipe_id': 'US-XX-YY_property-roll-2026',
        'layer': None,
        'key': 'parcel_id_local',
        'aggregation_function': None,
        'supplements': None,
    }

    def _discover(state, entity_type):
        return [roll] if entity_type == 'property' else []

    def _get_entities(recipe_id, admin_id, layer=None, **_):
        return pd.DataFrame(
            {'parcel_id_local': ['L1'], 'improvement_value': [80_000.0]}
        )

    patch_links(monkeypatch, '_discover_link_sources', _discover)
    patch_links(monkeypatch, 'get_entities', _get_entities)
    patch_links(monkeypatch, '_apply_remap_csvs', lambda state, recipe_id: state)
    monkeypatch.setattr(entity_links, 'load_unit_lot_pairs', lambda admin_id: None)
    lots = pd.DataFrame(
        {'geo_id': ['g1', 'g2'], 'parcel_id_local': ['L1', 'L2']},
    )
    state = _state(pd.DataFrame())
    evidence = parcel_pieces.lot_structure_evidence(state, lots, 'parcels', recipe)

    assert evidence.loc['g1', 'improvement_value'] == 80_000.0
    assert pd.isna(evidence.loc['g2', 'improvement_value'])


def test_estimate_property_counts_reads_the_preliminary_count():
    spine = pd.DataFrame(
        {
            'n_dwellings': [np.nan, np.nan],
            'n_dwellings_preliminary': [4.0, np.nan],
            'max_dwellings_per_building': [0.0, 0.0],
        },
        index=pd.Index(['a', 'b'], name='parcel_id'),
    )
    spine = estimate_property_counts(_state(spine)).spine

    assert spine.loc['a', 'n_properties_per_parcel'] == 4.0
    assert spine.loc['a', 'property_source'] == 'estimated'
    assert spine.loc['b', 'property_source'] == 'none'


def test_estimate_property_counts_on_an_old_geospine():
    # A geospine written before the step has no preliminary column;
    # its n_dwellings still holds the copied count.
    spine = pd.DataFrame(
        {'n_dwellings': [3.0], 'max_dwellings_per_building': [0.0]},
        index=pd.Index(['a'], name='parcel_id'),
    )
    spine = estimate_property_counts(_state(spine)).spine

    assert spine.loc['a', 'n_properties_per_parcel'] == 3.0


def _polygons():
    return pd.DataFrame(
        {
            # K1: one account the layer draws as three polygons and
            # repeats on each; K2: one polygon; K3: two polygons with
            # their own values.
            'parcel_id_local': ['K1', 'K1', 'K1', 'K2', 'K3', 'K3'],
            'improvement_value': [9000.0, 9000.0, 9000.0, 9000.0, 100.0, 5000.0],
            'area_ha': [1.0, 2.0, 6.0, 1.0, 1.0, 1.0],
        },
        index=pd.Index(['a', 'b', 'c', 'd', 'e', 'f'], name='parcel_id'),
    )


def test_a_repeated_account_value_counts_once_over_its_land():
    density = account_value_density(_polygons())

    assert density['a'] == pytest.approx(1000.0)
    assert density['c'] == pytest.approx(1000.0)


def test_a_single_lot_account_keeps_its_own_density():
    density = account_value_density(_polygons())

    assert density['d'] == pytest.approx(9000.0)


def test_differing_values_on_one_key_are_each_lots_own():
    density = account_value_density(_polygons())

    assert density['e'] == pytest.approx(100.0)
    assert density['f'] == pytest.approx(5000.0)


def test_a_polygon_stating_no_value_takes_none_from_its_key():
    polygons = _polygons()
    polygons.loc['c', 'improvement_value'] = np.nan
    density = account_value_density(polygons)

    assert pd.isna(density['c'])
    # The other two still repeat one value, over their own area only.
    assert density['a'] == pytest.approx(3000.0)


def test_a_placeholder_key_is_no_account():
    polygons = _polygons()
    polygons['parcel_id_local'] = ['0', '0', '0', 'K2', 'K3', 'K3']
    density = account_value_density(polygons)

    assert density['a'] == pytest.approx(9000.0)
