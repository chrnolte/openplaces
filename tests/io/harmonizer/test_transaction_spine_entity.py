"""The transaction spine establishes the entity: one row per recorded
sale, exact repeats dropped, the deed named, and the id minted once.

Since 2026-09-29 (plans/stage-contract-audit.md: transaction ids are
minted in the spine, curate never re-mints). Fabricated rows throughout.
"""

from __future__ import annotations

import pandas as pd
import pytest

from openplaces.core.schema import AdminId, Entity
from openplaces.io.harmonizer import HarmonizeState
from openplaces.io.harmonizer import transactions as spine_steps
from openplaces.io.sale_records import mint_transaction_ids
from openplaces.recipe import apply_recipe_patches, get_recipe_by_id


def _state(frame):
    return HarmonizeState(
        recipe={'entity': Entity('transaction', 'spine', '2026')},
        admin_id=AdminId('US-FL-LA'),
        verbose=False,
        timer=None,
        spine=frame,
    )


def _rows():
    # Deed BK1/PG1 conveys parcels A and B and is listed twice for A (an
    # overlapping source window); deed BK2/PG2 conveys C alone; the
    # fifth row is an assessor last-sale echo citing BK2/PG2; deed
    # BK3/PG3 conveys D, listed under two spellings of its number ('D'
    # and 'D-0', one parcel_id_local), as Florida's feed does across
    # vintages.
    return pd.DataFrame(
        {
            'sale_record_kind': [
                'deed',
                'deed',
                'deed',
                'deed',
                'assessor_last_sale',
                'deed',
                'deed',
            ],
            'parcel_id_assessor': ['A', 'A', 'B', 'C', 'C', 'D', 'D-0'],
            'parcel_id_local': ['A', 'A', 'B', 'C', 'C', 'D', 'D'],
            # What derive_parcel_link_key writes before the fold runs.
            'parcel_link_key': ['A', 'A', 'B', 'C', 'C', 'D', 'D'],
            'recorded_date': [
                '2020-03-05',
                '2020-03-05',
                '2020-03-05',
                '2021-07-01',
                None,
                '2022-01-09',
                '2022-01-09',
            ],
            'sale_year': [None, None, None, None, 2021.0, None, None],
            'sale_month': [None, None, None, None, 7.0, None, None],
            'price': [100.0, 100.0, 100.0, 50.0, 50.0, 70.0, 70.0],
            'sale_book': ['1', '1', '1', '2', '2', '3', '3'],
            'sale_page': ['1', '1', '1', '2', '2', '3', '3'],
            'area_ha': [1.0, 1.0, 2.0, 0.5, None, 3.0, 3.0],
        }
    )


def _run_pipeline(frame):
    state = _state(frame)
    state = spine_steps.derive_sale_period(state)
    state = spine_steps.dedup_transactions(
        state,
        key_columns=[
            'sale_record_kind',
            'parcel_id_local',
            # As in the recipe: a second spelling of the number is not
            # an exact repeat, so D's two rows reach the fold.
            'parcel_id_assessor',
            'sale_year',
            'sale_month',
            'price',
            'sale_book',
            'sale_page',
        ],
    )
    state = spine_steps.derive_document_id(
        state,
        candidates=[['sale_book', 'sale_page']],
        scope_column='sale_record_kind',
        unscoped_value='deed',
    )
    state = spine_steps.count_parcels_per_document(
        state, parcel_column=['parcel_id_local', 'parcel_id_assessor']
    )
    state = spine_steps.aggregate_multi_parcel_sales(state)
    return spine_steps.assign_transaction_ids(state)


def test_the_spine_ends_as_one_row_per_recorded_sale_with_its_document_id():
    state = _run_pipeline(_rows())
    spine = state.spine
    # Seven rows in: one exact repeat dropped, A and B folded into one
    # deed, D's two spellings folded into one.
    assert len(spine) == 4
    assert spine.index.name == 'transaction_id'
    assert spine.index.is_unique
    by_doc = spine.set_index('sale_document_id')
    assert by_doc.loc['1/1', 'n_parcels_per_sale'] == 2
    assert by_doc.loc['1/1', 'price'] == 100.0
    # What the members state for their own parcel is not folded onto the
    # deed: A's and B's areas and numbers differ, so the deed has none.
    assert pd.isna(by_doc.loc['1/1', 'area_ha'])
    assert pd.isna(by_doc.loc['1/1', 'parcel_id_local'])
    # The members themselves are kept for the link table, with the id.
    members = state.metadata[spine_steps.DEED_MEMBERS_KEY]
    assert members['parcel_id_local'].tolist() == ['A', 'B', 'D', 'D']
    assert members['transaction_id'].tolist()[:2] == ['US-FL-LA_1-1', 'US-FL-LA_1-1']
    assert by_doc.loc['2/2', 'n_parcels_per_sale'] == 1
    # One parcel under two spellings is one parcel: counted on the
    # normalized number, and the deed keeps what both rows state alike.
    assert by_doc.loc['3/3', 'n_parcels_per_sale'] == 1
    assert by_doc.loc['3/3', 'parcel_id_local'] == 'D'
    assert by_doc.loc['3/3', 'area_ha'] == 3.0
    assert pd.isna(by_doc.loc['3/3', 'parcel_id_assessor'])
    # The assessor echo keeps a scoped document of its own.
    assert 'assessor_last_sale:2/2' in by_doc.index
    # The period was derived from the date where the source stated none.
    assert (
        by_doc.loc['1/1', 'sale_year'] == 2020 and by_doc.loc['1/1', 'sale_month'] == 3
    )


def test_the_id_is_the_document_scoped_by_the_admin_unit_without_a_suffix():
    spine = _run_pipeline(_rows()).spine
    ids = set(spine.index)
    assert 'US-FL-LA_1-1' in ids
    assert 'US-FL-LA_2-2' in ids
    assert not any(i.endswith(('_2', '_3')) for i in ids)


def test_minting_twice_is_stable():
    """Curate loads the spine's index and never re-mints; minting again
    from the same rows gives the same ids anyway."""
    spine = _run_pipeline(_rows()).spine
    again, _ = mint_transaction_ids(spine, 'US-FL-LA')
    assert list(again.index) == list(spine.index)


def test_a_missing_document_column_warns_and_fingerprints():
    frame = _rows().drop(columns=['sale_book', 'sale_page'])
    state = spine_steps.derive_sale_period(_state(frame))
    with pytest.warns(UserWarning, match='named by its content fingerprint'):
        state = spine_steps.assign_transaction_ids(state)
    assert all(i.startswith('US-FL-LA_sale:') for i in state.spine.index)
    assert state.spine.index.is_unique


def test_the_recipes_divide_the_steps_between_spine_and_curate():
    spine = get_recipe_by_id('US_transaction-spine-2026')
    curate = get_recipe_by_id('US_transaction-openplaces-2026')
    spine_steps_listed = [s['step'] for s in spine['pipeline']]
    curate_steps_listed = [s['step'] for s in curate['pipeline']]
    moved = [
        'derive_sale_period',
        'dedup_transactions',
        'derive_document_id',
        'count_parcels_per_document',
        'aggregate_multi_parcel_sales',
        'assign_transaction_ids',
        'link_transactions',
    ]
    assert spine_steps_listed[-7:] == moved
    for step in moved:
        assert step not in curate_steps_listed
    # The fold precedes the ids, so a deed's id carries no suffix; the
    # links follow the ids, since they are keyed on them.
    assert spine_steps_listed.index('aggregate_multi_parcel_sales') < (
        spine_steps_listed.index('assign_transaction_ids')
    )
    # What a sale conveyed is in the link tables, not in spine columns:
    # no pass copies parcel values onto the sale any more.
    assert 'link_by_id' not in spine_steps_listed
    assert 'record_link_method' not in spine_steps_listed
    assert 'link_address_ranges' not in spine_steps_listed
    # And the curate gathers through the links.
    gathers = [s for s in curate['pipeline'] if s['step'] == 'aggregate_from_entities']
    assert {g['recipe_id'] for g in gathers} == {
        'US_parcel-spine-2026',
        'US_property-spine-2026',
    }
    assert all(g['key'] is None for g in gathers)


def test_massachusetts_names_its_documents_by_book_and_page():
    """A registry's instrument number restarts every year, so the state
    patch puts book and page first; elsewhere the national order holds."""
    spine = get_recipe_by_id('US_transaction-spine-2026')

    def _candidates(admin_id):
        patched, applied = apply_recipe_patches(spine, admin_id)
        (step,) = [s for s in patched['pipeline'] if s['step'] == 'derive_document_id']
        return step['candidates'], applied

    national, applied = _candidates('US-WI-VI')
    assert applied == []
    assert national[0] != ['book', 'page']
    massachusetts, applied = _candidates('US-MA-SOM')
    assert applied == ['US-MA_transaction-spine-2026']
    assert massachusetts[0] == ['book', 'page']
    assert sorted(map(tuple, massachusetts)) == sorted(map(tuple, national))


def _parcels():
    return pd.DataFrame(
        {
            'parcel_id_local': ['A', 'B', 'C', 'L', 'D'],
            'area_ha': [1.0, 3.0, 0.5, 2.0, 3.0],
            'address_id_local': ['k1', 'k2', 'k3', 'k4', 'k5'],
            'address_id_local_city': ['c1', 'c2', 'c3', 'c4', 'c5'],
            'address_street': ['Main St', 'Main St', 'Oak Ave', 'Oak Ave', 'Elm St'],
            'address_number': ['10', '12', '7', '9', '1'],
            'admin4_id': ['T'] * 5,
        },
        index=pd.Index(['pa', 'pb', 'pc', 'pl', 'pd'], name='parcel_id'),
    )


def _properties():
    return pd.DataFrame(
        {'parcel_id_local': ['U1', 'C', 'D']},
        index=pd.Index(['prop-u1', 'prop-c', 'prop-d'], name='property_id'),
    )


def test_links_pair_each_sale_with_what_it_conveyed(monkeypatch, tmp_path):
    """A deed of two parcels gets two link rows with area shares; a unit
    sale reaches its lot and its property; an address-only sale reaches
    its parcel by the address key; a last-sale row says so."""
    frames = {
        'US_parcel-spine-2026': _parcels(),
        'US_property-spine-2026': _properties(),
    }
    monkeypatch.setattr(
        spine_steps, 'get_entities', lambda rid, admin_id, **k: frames.get(rid)
    )
    monkeypatch.setattr(spine_steps, 'get_recipe_id', lambda recipe: 'tx-spine')
    monkeypatch.setattr(
        spine_steps, 'get_link_owner_recipe_id', lambda recipe: 'tx-spine'
    )
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
    state = _run_pipeline(_rows())
    spine = state.spine
    # Extra rows: a condo unit sale (its lot L through lot_id_local), a
    # Massachusetts-style sale with an address and no number, and a
    # range deed.
    extra = pd.DataFrame(
        {
            'sale_record_kind': ['deed', 'deed', 'deed'],
            'parcel_id_local': ['U1', None, None],
            'lot_id_local': ['L', None, None],
            'address_id_local_city': [None, 'c3', None],
            'address_street': [None, 'Oak Ave', 'Main St'],
            'address_number': [None, '7', '10-12'],
            'admin4_id': ['T', 'T', 'T'],
            'price': [1.0, 2.0, 3.0],
            'source': ['s', 's', 's'],
        },
        index=pd.Index(
            ['US-FL-LA_u', 'US-FL-LA_addr', 'US-FL-LA_range'], name='transaction_id'
        ),
    )
    spine = pd.concat([spine, extra])
    spine['parcel_link_key'] = spine['lot_id_local'].where(
        spine['lot_id_local'].notna(), spine['parcel_id_local']
    )
    state.spine = spine
    state = spine_steps.link_transactions(state)

    links = pd.read_parquet(tmp_path / 'tx-spine_US_parcel-spine-2026.parquet')
    by_sale = {
        sale: group.set_index('parcel_id')
        for sale, group in links.groupby('transaction_id')
    }
    deed = by_sale['US-FL-LA_1-1']
    assert set(deed.index) == {'pa', 'pb'}
    assert deed['link_method'].tolist() == ['parcel_id_local', 'parcel_id_local']
    assert deed['share'].astype(float).round(2).tolist() == [0.25, 0.75]
    assert deed['share_basis'].tolist() == ['area', 'area']
    single = by_sale['US-FL-LA_2-2']
    assert single.index.tolist() == ['pc']
    assert single['share'].isna().all()
    # D under two spellings: two member rows, one pair, no share.
    spelled = by_sale['US-FL-LA_3-3']
    assert spelled.index.tolist() == ['pd']
    assert spelled['share'].isna().all()
    assert by_sale['US-FL-LA_u']['link_method'].tolist() == ['stacked_units']
    assert by_sale['US-FL-LA_u'].index.tolist() == ['pl']
    assert by_sale['US-FL-LA_addr']['link_method'].tolist() == ['address_id_local_city']
    # '10-12' matches parcels A and B: ambiguous, so no link.
    assert 'US-FL-LA_range' not in by_sale
    # The echo's id is its scoped document, normalized to upper case.
    (echo_id,) = [k for k in by_sale if 'ASSESSOR' in k]
    assert by_sale[echo_id]['link_method'].tolist() == ['assessor_last_sale']

    properties = pd.read_parquet(tmp_path / 'tx-spine_US_property-spine-2026.parquet')
    assert set(properties['transaction_id']) == {
        'US-FL-LA_u',
        'US-FL-LA_2-2',
        'US-FL-LA_3-3',
        echo_id,
    }
    # The two spellings of D make one (sale, property) pair, not two.
    assert (properties['transaction_id'] == 'US-FL-LA_3-3').sum() == 1
    summary = state.spine['parcel_link_method']
    assert summary.loc['US-FL-LA_1-1'] == 'parcel_id_local'
    assert summary.loc['US-FL-LA_u'] == 'stacked_units'
    assert pd.isna(summary.loc['US-FL-LA_range'])
    # The link table holds two ids and the labels, nothing else.
    assert list(links.columns) == [
        'transaction_id',
        'parcel_id',
        'link_method',
        'link_source',
        'share',
        'share_basis',
    ]
