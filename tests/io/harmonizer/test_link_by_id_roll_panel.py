"""An auto-discovered roll that holds several tax years joins its latest.

Florida's DOR roll is ingested as a panel of yearly rolls, so aggregate
mode summed one account over every year it was on the roll. Values are
fabricated.
"""

import pandas as pd

import openplaces.io.harmonizer.entity_links as entity_links
import openplaces.io.harmonizer.links as links
from openplaces.core.schema import AdminId
from openplaces.io.harmonizer import HarmonizeState
from tests.links_patching import patch_links


def _state(spine):
    return HarmonizeState(
        recipe={},
        admin_id=AdminId('US-FL-LA'),
        verbose=False,
        timer=None,
        spine=spine,
    )


def _patch(monkeypatch, ref, entity_type):
    matches = [
        {
            'recipe_id': f'US-FL_{entity_type}-example-2026',
            'layer': None,
            'key': 'parcel_id_local',
            'aggregation_function': None,
        }
    ]
    patch_links(monkeypatch, '_discover_link_sources', lambda *a, **k: matches)
    patch_links(monkeypatch, '_apply_remap_csvs', lambda state, recipe_id: state)
    patch_links(monkeypatch, 'restrict_to_admin_by_name', lambda df, *a: df)
    patch_links(
        monkeypatch, 'get_entities', lambda recipe_id, admin_id, layer=None: ref
    )
    monkeypatch.setattr(entity_links, 'load_unit_lot_pairs', lambda admin_id: None)


def _spine():
    return pd.DataFrame(
        {'parcel_id_local': ['A1', 'B2']},
        index=pd.Index(['parcel-a', 'parcel-b'], name='parcel_id'),
    )


def test_property_roll_panel_joins_its_latest_year_only(monkeypatch):
    # Account A1 is on three rolls, B2 on the latest only. Summing the
    # panel would give A1 600; the latest roll says 300.
    ref = pd.DataFrame(
        {
            'parcel_id_local': ['A1', 'A1', 'A1', 'B2'],
            'tax_year': [2023, 2024, 2025, 2025],
            'land_value': [100.0, 200.0, 300.0, 50.0],
        },
        index=pd.Index(['a-23', 'a-24', 'a-25', 'b-25'], name='property_id'),
    )
    _patch(monkeypatch, ref, 'property')

    state = links.link_by_id(
        _state(_spine()),
        auto_discover=True,
        entity_type='property',
        columns=['land_value', 'tax_year'],
        count_as=False,
    )

    assert state.spine['land_value'].tolist() == [300.0, 50.0]


def test_property_roll_drops_accounts_retired_before_the_latest_year(monkeypatch):
    # The unit's latest roll stands, not each account's latest row: an
    # account last valued in 2023 is not revived with its old figure.
    ref = pd.DataFrame(
        {
            'parcel_id_local': ['A1', 'A1', 'A1', 'B2'],
            'tax_year': [2023, 2024, 2025, 2023],
            'land_value': [100.0, 200.0, 300.0, 80.0],
        },
        index=pd.Index(['a-23', 'a-24', 'a-25', 'b-23'], name='property_id'),
    )
    _patch(monkeypatch, ref, 'property')

    state = links.link_by_id(
        _state(_spine()),
        auto_discover=True,
        entity_type='property',
        columns=['land_value'],
        count_as=False,
    )

    assert state.spine.loc['parcel-a', 'land_value'] == 300.0
    assert pd.isna(state.spine.loc['parcel-b', 'land_value'])


def test_units_stamped_with_different_years_are_not_a_panel(monkeypatch):
    # Units split off a statewide layer whose records carry their own
    # roll year: no account repeats, so every unit's value stays.
    ref = pd.DataFrame(
        {
            'parcel_id_local': ['A1', 'B2'],
            'tax_year': [2025, 2024],
            'land_value': [300.0, 80.0],
        },
        index=pd.Index(['unit-a', 'unit-b'], name='property_id'),
    )
    _patch(monkeypatch, ref, 'property')

    state = links.link_by_id(
        _state(_spine()),
        auto_discover=True,
        entity_type='property',
        columns=['land_value'],
        count_as=False,
    )

    assert state.spine['land_value'].tolist() == [300.0, 80.0]


def test_transaction_years_are_all_kept(monkeypatch):
    # A sales table carries tax_year too, and each year holds other
    # sales: every one of them is counted.
    ref = pd.DataFrame(
        {
            'parcel_id_local': ['A1', 'A1', 'B2'],
            'tax_year': [2023, 2025, 2025],
            'price': [1000.0, 2000.0, 500.0],
        },
        index=pd.Index(['s1', 's2', 's3'], name='transaction_id'),
    )
    _patch(monkeypatch, ref, 'transaction')

    state = links.link_by_id(
        _state(_spine()),
        auto_discover=True,
        entity_type='transaction',
        columns={'price': 'last_sale_price'},
        count_as='n_transactions',
    )

    assert state.spine['n_transactions'].tolist() == [2, 1]
