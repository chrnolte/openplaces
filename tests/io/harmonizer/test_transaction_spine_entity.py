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
from openplaces.recipe import get_recipe_by_id


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
    # overlapping source window); deed BK2/PG2 conveys C alone; the last
    # row is an assessor last-sale echo citing BK2/PG2.
    return pd.DataFrame(
        {
            'sale_record_kind': ['deed', 'deed', 'deed', 'deed', 'assessor_last_sale'],
            'parcel_id_assessor': ['A', 'A', 'B', 'C', 'C'],
            'parcel_id_local': ['A', 'A', 'B', 'C', 'C'],
            'recorded_date': [
                '2020-03-05',
                '2020-03-05',
                '2020-03-05',
                '2021-07-01',
                None,
            ],
            'sale_year': [None, None, None, None, 2021.0],
            'sale_month': [None, None, None, None, 7.0],
            'price': [100.0, 100.0, 100.0, 50.0, 50.0],
            'sale_book': ['1', '1', '1', '2', '2'],
            'sale_page': ['1', '1', '1', '2', '2'],
            'area_ha': [1.0, 1.0, 2.0, 0.5, None],
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
        state, parcel_column='parcel_id_assessor'
    )
    state = spine_steps.aggregate_multi_parcel_sales(state)
    return spine_steps.assign_transaction_ids(state)


def test_the_spine_ends_as_one_row_per_recorded_sale_with_its_document_id():
    spine = _run_pipeline(_rows()).spine
    # Five rows in: one exact repeat dropped, A and B folded into one deed.
    assert len(spine) == 3
    assert spine.index.name == 'transaction_id'
    assert spine.index.is_unique
    by_doc = spine.set_index('sale_document_id')
    assert by_doc.loc['1/1', 'n_parcels_per_sale'] == 2
    assert by_doc.loc['1/1', 'area_ha'] == 3.0
    assert by_doc.loc['1/1', 'price'] == 100.0
    assert by_doc.loc['2/2', 'n_parcels_per_sale'] == 1
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
    ]
    assert spine_steps_listed[-6:] == moved
    for step in moved:
        assert step not in curate_steps_listed
    # The fold precedes the ids, so a deed's id carries no suffix.
    assert spine_steps_listed.index('aggregate_multi_parcel_sales') < (
        spine_steps_listed.index('assign_transaction_ids')
    )
