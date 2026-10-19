"""A Massachusetts deed and the roll row citing it name one document.

A registry and MassGIS pad one book and page differently, so the deed
and the assessor's last-sale row citing it named two documents; and a
registry deed carries no parcel id, so it linked by address alone. With
the zeros stripped the two share `sale_document_ref`, the deed takes the
citing rows' ids (`cited_document`), and the curate can flag the echo.
Fabricated rows.
"""

from __future__ import annotations

import pandas as pd

from openplaces.core.schema import AdminId, Entity
from openplaces.io import sale_records
from openplaces.io.harmonizer import HarmonizeState
from openplaces.io.harmonizer import transactions as spine_steps
from openplaces.recipe import apply_recipe_patches, get_recipe_by_id


def test_leading_zeros_are_stripped_only_when_asked():
    frame = pd.DataFrame(
        {'book': ['00313', '646', 'B12'], 'page': ['0278', '141', '07']}
    )
    plain = sale_records.derive_document_id(frame.copy(), [['book', 'page']])
    assert plain['sale_document_id'].tolist() == ['00313/0278', '646/141', 'B12/07']
    stripped = sale_records.derive_document_id(
        frame.copy(), [['book', 'page']], strip_leading_zeros=True
    )
    # A part with a letter is not a number and keeps its spelling.
    assert stripped['sale_document_id'].tolist() == ['313/278', '646/141', 'B12/7']


def _patch(monkeypatch, tmp_path):
    parcels = pd.DataFrame(
        {
            'parcel_id_local': [None, None, None],
            'parcel_id_admin2': ['F_1_1', 'F_2_2', 'F_3_3'],
            'address_id_local_city': ['a1', 'a2', 'a3'],
            'area_ha': [1.0, 2.0, 3.0],
        },
        index=pd.Index(['p1', 'p2', 'p3'], name='parcel_id'),
    )
    properties = pd.DataFrame(
        {'property_id_assessor': ['1 1', '2 2', '3 3'], 'parcel_id_local': [None] * 3},
        index=pd.Index(['prop-1', 'prop-2', 'prop-3'], name='property_id'),
    )
    frames = {'US_parcel-spine-2026': parcels, 'US_property-spine-2026': properties}
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
    # d1: a deed of two parcels, cited by two roll rows; d2: a deed that
    # links by its own address and is cited too; d3: cited by nothing.
    sales = pd.DataFrame(
        {
            'sale_record_kind': ['deed', 'deed', 'deed', *['assessor_last_sale'] * 3],
            'sale_document_ref': ['10/1', '10/2', '10/3', '10/1', '10/1', '10/2'],
            'parcel_id_admin2': [None, None, None, 'F_1_1', 'F_2_2', 'F_3_3'],
            'property_id_assessor': [None, None, None, '1 1', '2 2', '3 3'],
            'address_id_local_city': [None, 'a3', None, None, None, None],
            'source': ['registry'] * 3 + ['massgis'] * 3,
        },
        index=pd.Index(['d1', 'd2', 'd3', 'l1', 'l2', 'l3'], name='transaction_id'),
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


def test_a_deed_takes_the_ids_of_the_rows_citing_it(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    spine_steps.link_transactions(
        _state(),
        parcel_id_keys=('parcel_id_admin2',),
        property_key='property_id_assessor',
        document_key='sale_document_ref',
    )
    parcels = _links(tmp_path, 'parcel')
    d1 = parcels[parcels['transaction_id'] == 'd1']
    assert sorted(d1['parcel_id']) == ['p1', 'p2']
    assert set(d1['link_method']) == {'cited_document'}
    d2 = parcels[parcels['transaction_id'] == 'd2']
    assert d2['parcel_id'].tolist() == ['p3']
    assert 'd3' not in set(parcels['transaction_id'])
    properties = _links(tmp_path, 'property')
    assert sorted(
        properties.loc[properties['transaction_id'] == 'd1', 'property_id']
    ) == [
        'prop-1',
        'prop-2',
    ]
    assert set(properties.loc[properties['transaction_id'] == 'd1', 'link_method']) == {
        'cited_document'
    }


def test_without_a_document_key_deeds_link_as_before(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    spine_steps.link_transactions(
        _state(),
        parcel_id_keys=('parcel_id_admin2',),
        property_key='property_id_assessor',
    )
    parcels = _links(tmp_path, 'parcel')
    assert 'd1' not in set(parcels['transaction_id'])
    assert 'cited_document' not in set(parcels['link_method'])


def test_the_massachusetts_patches_switch_it_on():
    spine, _ = apply_recipe_patches(
        get_recipe_by_id('US_transaction-spine-2026'), 'US-MA-NTC'
    )
    derives = [s for s in spine['pipeline'] if s['step'] == 'derive_document_id']
    assert [s.get('output', 'sale_document_id') for s in derives] == [
        'sale_document_id',
        'sale_document_ref',
    ]
    assert all(s['strip_leading_zeros'] for s in derives)
    (link,) = [s for s in spine['pipeline'] if s['step'] == 'link_transactions']
    assert link['document_key'] == 'sale_document_ref'
    curate, _ = apply_recipe_patches(
        get_recipe_by_id('US_transaction-openplaces-2026'), 'US-MA-NTC'
    )
    flags = [
        s for s in curate['pipeline'] if s['step'] == 'flag_sales_matching_other_kind'
    ]
    assert flags[1]['key_column'] == 'sale_document_ref'
    other, _ = apply_recipe_patches(
        get_recipe_by_id('US_transaction-spine-2026'), 'US-FL-LA'
    )
    assert not any(
        s.get('strip_leading_zeros')
        for s in other['pipeline']
        if s['step'] == 'derive_document_id'
    )
