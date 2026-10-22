"""A Massachusetts last sale reaches its parcel by id, not by address.

A MassGIS assessor row carries its parcel's statewide id and its own
account number but no `parcel_id_local`, so before `parcel_id_keys` the
national passes reached it by address alone. Fabricated rows.
"""

from __future__ import annotations

import pandas as pd

from openplaces.core.schema import AdminId, Entity
from openplaces.io.harmonizer import HarmonizeState
from openplaces.io.harmonizer import transactions as spine_steps
from openplaces.recipe import apply_recipe_patches, get_recipe_by_id


def _patch(monkeypatch, tmp_path):
    parcels = pd.DataFrame(
        {
            'parcel_id_local': [None, None, None],
            'parcel_id_admin2': ['F_1_1', 'F_2_2', 'F_3_3'],
            # The address points the second sale at the wrong parcel:
            # two lots on one road share the number on the assessor's
            # list, as a back lot behind its frontage lot can.
            'address_id_local_city': ['a1', 'a2', 'a2-wrong'],
            'area_ha': [1.0, 2.0, 3.0],
        },
        index=pd.Index(['p1', 'p2', 'p3'], name='parcel_id'),
    )
    properties = pd.DataFrame(
        {'property_id_assessor': ['1 1', '2 2'], 'parcel_id_local': [None, None]},
        index=pd.Index(['prop-1', 'prop-2'], name='property_id'),
    )
    frames = {
        'US_parcel-spine-2026': parcels,
        'US_property-spine-2026': properties,
    }
    monkeypatch.setattr(
        spine_steps, 'get_entities', lambda rid, admin_id, **k: frames.get(rid)
    )
    monkeypatch.setattr(spine_steps, 'get_recipe_id', lambda recipe: 'tx')
    monkeypatch.setattr(spine_steps, 'get_link_owner_recipe_id', lambda recipe: 'tx')
    monkeypatch.setattr(
        spine_steps,
        'get_entity_link_path',
        lambda own, other, admin_id: tmp_path / f'{own}_{other}.parquet',
    )
    monkeypatch.setattr(
        spine_steps,
        'write_entity_link',
        lambda path, links, **k: links.to_parquet(path),
    )


def _state():
    sales = pd.DataFrame(
        {
            'sale_record_kind': ['assessor_last_sale'] * 3,
            'parcel_id_admin2': ['F_1_1', 'F_3_3', None],
            'property_id_assessor': ['1 1', None, None],
            'address_id_local_city': ['a1', 'a2-wrong', 'a2'],
            'source': ['massgis'] * 3,
        },
        index=pd.Index(['s1', 's2', 's3'], name='transaction_id'),
    )
    return HarmonizeState(
        recipe={'entity': Entity('transaction', 'spine', '2026')},
        admin_id=AdminId('US-MA-XX'),
        verbose=False,
        timer=None,
        spine=sales,
    )


def _links(tmp_path, other):
    return pd.read_parquet(tmp_path / f'tx_US_{other}-spine-2026.parquet')


def test_the_parcel_id_pass_runs_before_the_addresses(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    state = spine_steps.link_transactions(
        _state(),
        parcel_id_keys=('parcel_id_admin2',),
        property_key='property_id_assessor',
    )
    links = _links(tmp_path, 'parcel').set_index('transaction_id')
    assert links.loc['s1', 'parcel_id'] == 'p1'
    assert links.loc['s1', 'link_method'] == 'parcel_id_admin2'
    assert links.loc['s2', 'parcel_id'] == 'p3'
    # A sale without the id still reaches its parcel by address.
    assert links.loc['s3', 'link_method'] == 'address_id_local_city'
    assert state.spine.loc['s1', 'parcel_link_method'] == 'parcel_id_admin2'
    properties = _links(tmp_path, 'property')
    assert properties[['transaction_id', 'property_id']].values.tolist() == [
        ['s1', 'prop-1']
    ]
    assert properties['link_method'].tolist() == ['property_id_assessor']


def test_without_the_option_nothing_changes(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    spine_steps.link_transactions(_state())
    links = _links(tmp_path, 'parcel')
    assert set(links['link_method']) == {'address_id_local_city'}
    assert _links(tmp_path, 'property').empty


def test_the_massachusetts_patch_switches_both_on():
    recipe = get_recipe_by_id('US_transaction-spine-2026')
    patched, applied = apply_recipe_patches(recipe, 'US-MA-NTC')
    assert 'US-MA_transaction-spine-2026' in applied
    (step,) = [s for s in patched['pipeline'] if s['step'] == 'link_transactions']
    assert step['parcel_id_keys'] == ['parcel_id_admin2']
    assert step['property_key'] == 'property_id_assessor'
    # Outside Massachusetts the national defaults hold.
    other, _ = apply_recipe_patches(recipe, 'US-FL-LA')
    (step,) = [s for s in other['pipeline'] if s['step'] == 'link_transactions']
    assert 'parcel_id_keys' not in step
