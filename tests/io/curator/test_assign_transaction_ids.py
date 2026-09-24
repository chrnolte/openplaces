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
