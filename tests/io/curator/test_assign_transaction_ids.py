"""A curated sale carries an id, scoped to the admin unit that issued it.

Until 2026-09-23 it carried none: the curated output had an unnamed
RangeIndex, so every county numbered its rows 0, 1, 2. That is
invisible per county and destructive when counties are pooled, because
`export_delivery` de-duplicates on the index - pooling Wisconsin's 72
counties kept 291,024 of 2,751,753 sales.
"""

from __future__ import annotations

import pandas as pd
import pytest

from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState
from openplaces.io.curator.transactions import assign_transaction_ids


def _state(frame, admin_id='US-FL-MD'):
    return CurateState(
        recipe={},
        entity_recipe={},
        admin_id=AdminId(admin_id),
        verbose=False,
        timer=None,
        curated=frame,
    )


def _sales(**overrides):
    frame = pd.DataFrame(
        {
            'sale_document_id': ['BK1-PG1', 'BK1-PG2', 'BK2-PG9'],
            'sale_record_kind': ['deed', 'deed', 'deed'],
            'parcel_id_local': ['p1', 'p2', 'p3'],
            'sale_year': [2020, 2021, 2022],
            'sale_month': [1, 2, 3],
            'price': [100.0, 200.0, 300.0],
        }
    )
    for key, value in overrides.items():
        frame[key] = value
    return frame


def test_the_id_is_the_document_scoped_by_its_admin_unit():
    """Document numbers repeat between counties, so the unit is part of it."""
    state = assign_transaction_ids(_state(_sales()))

    index = state.curated.index
    assert index.name == 'transaction_id'
    assert index.nunique() == 3
    assert all(str(value).startswith('US-FL-MD_') for value in index)


def test_two_counties_do_not_share_an_id_for_the_same_document():
    """The failure that collapsed 89% of Wisconsin, in miniature."""
    one = assign_transaction_ids(_state(_sales(), 'US-FL-MD')).curated
    two = assign_transaction_ids(_state(_sales(), 'US-FL-AL')).curated

    assert set(one.index).isdisjoint(set(two.index))


def test_a_sale_naming_no_document_is_named_by_its_content():
    """Florida's roll writes a blank book and page on 5.2% of Miami-Dade.

    Those rows still need a name that survives a rebuild, which is the
    same fallback `assign_entity_ids` uses for a property whose
    assessor issued no account number.
    """
    frame = _sales()
    frame.loc[1, 'sale_document_id'] = None

    state = assign_transaction_ids(_state(frame))

    index = state.curated.index
    assert index.nunique() == 3
    assert index.isna().sum() == 0
    # The named-by-content row is distinguishable from the two that
    # carry a document reference.
    assert ':' in str(index[1])
    assert ':' not in str(index[0])


def test_two_sales_of_one_document_on_different_parcels_stay_apart():
    """A repeated document must not fold two sales into one id."""
    frame = _sales()
    frame['sale_document_id'] = ['BK1-PG1', 'BK1-PG1', 'BK2-PG9']

    state = assign_transaction_ids(_state(frame))

    assert state.curated.index.nunique() == 3


def test_a_duplicate_index_does_not_break_the_minting():
    """Polk County FL, 2026-09-24: 1 of 139 counties failed on this.

    `aggregate_multi_parcel_sales` indexes its aggregated rows by the
    representative row's label, so a label can appear twice.
    `mint_ids` assigns the content-named ids through a boolean mask,
    which is label-based, and a repeated label misaligned it with
    "Must have equal len keys and value when setting with an
    iterable" - a failure that only shows on a county with both a
    repeated label and a sale naming no document.
    """
    frame = _sales()
    frame.loc[1, 'sale_document_id'] = None
    frame.index = pd.Index(['a', 'a', 'b'])

    state = assign_transaction_ids(_state(frame))

    assert state.curated.index.name == 'transaction_id'
    assert state.curated.index.nunique() == 3
    # Row order is untouched: the ids line up with the rows they name.
    assert list(state.curated['parcel_id_local']) == ['p1', 'p2', 'p3']


def test_a_fingerprint_uses_price_and_date_before_anything_else():
    """The narrowest tier is what a sale is: a price on a date.

    Every column in the hash can change a published id, so a sale
    whose parcel key is corrected must keep its name where price and
    date already separate it.
    """
    frame = _sales()
    frame['sale_document_id'] = None

    before = assign_transaction_ids(_state(frame.copy())).curated.index
    moved = frame.copy()
    moved['parcel_id_local'] = ['x1', 'x2', 'x3']
    moved['sale_record_kind'] = ['assessor_last_sale'] * 3
    after = assign_transaction_ids(_state(moved)).curated.index

    assert list(before) == list(after)


def test_a_wider_tier_is_used_only_where_the_narrow_one_collides():
    """Two sales at one price in one month are told apart by parcel."""
    frame = _sales()
    frame['sale_document_id'] = None
    frame['sale_year'] = [2020, 2020, 2021]
    frame['sale_month'] = [1, 1, 3]
    frame['price'] = [100.0, 100.0, 300.0]

    state = assign_transaction_ids(_state(frame))

    # Separated by the parcel tier, not by a `_2` suffix.
    ids = [str(v) for v in state.curated.index]
    assert len(set(ids)) == 3
    assert not any(i.endswith('_2') for i in ids), ids


def test_two_indistinguishable_sales_still_get_an_id_each():
    """Osceola County FL, 2026-09-24: 789,903 ids for 793,283 sales.

    `mint_ids` gives exact duplicates one id, which is right for a
    property described by two sources and wrong for a sale: a row
    reaching this step already survived `dedup_transactions`, so it is
    one the recipe means to keep, and a shared index label would have
    the delivery's de-duplication drop it silently.
    """
    frame = _sales()
    # Two rows the source states identically and distinguishes by
    # nothing this step reads.
    frame.loc[1] = frame.loc[0]

    state = assign_transaction_ids(_state(frame))

    assert len(state.curated) == 3
    assert state.curated.index.nunique() == 3


def test_an_empty_table_is_left_alone():
    """A county with no sales must not raise on its way through."""
    state = _state(pd.DataFrame(columns=['sale_document_id']))

    assert assign_transaction_ids(state).curated.empty


def test_a_table_with_no_document_column_says_so():
    """Silently fingerprinting every sale would hide a recipe error."""
    frame = _sales().drop(columns=['sale_document_id'])

    with pytest.warns(UserWarning, match='no .*sale_document_id'):
        state = assign_transaction_ids(_state(frame))

    assert state.curated.index.nunique() == 3
