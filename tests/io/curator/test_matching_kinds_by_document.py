"""Flagging a last-sale row that repeats a deed, keyed on the document
where neither side carries a parcel id.

Massachusetts is that case: its registry deeds have no
`parcel_id_local`, so the parcel-keyed pass leaves every MA row
unflagged. Every value here is fabricated.
"""

import pandas as pd

from openplaces.io.curator import transactions as tx


def _state(curated):
    state = tx.CurateState.__new__(tx.CurateState)
    state.curated = curated
    state.admin_id = 'XX-YY-ZZ'
    state.verbose = False
    return state


def _rows():
    # A deed and the last-sale row repeating it, plus a last-sale row
    # with no matching deed.
    return pd.DataFrame(
        {
            'sale_record_kind': ['deed', 'assessor_last_sale', 'assessor_last_sale'],
            'book': ['100', '100', '200'],
            'page': ['5', '5', '9'],
            'sale_year': [2020, 2020, 2021],
            'sale_month': [3, 3, 7],
            'price': [500000.0, 500000.0, 250000.0],
        }
    )


def test_a_composite_document_key_finds_the_echo():
    out = tx.flag_sales_matching_other_kind(
        _state(_rows()), key_column=['book', 'page']
    ).curated
    assert pd.isna(out['sale_matches_deed'][0])  # the deed itself
    assert out['sale_matches_deed'][1] == 1.0  # repeats it
    assert out['sale_matches_deed'][2] == 0.0  # no deed to repeat


def test_a_missing_key_part_leaves_the_row_undecided():
    rows = _rows()
    rows.loc[1, 'page'] = None
    out = tx.flag_sales_matching_other_kind(
        _state(rows), key_column=['book', 'page']
    ).curated
    assert pd.isna(out['sale_matches_deed'][1])


def test_the_parcel_pass_is_not_overruled_by_the_document_pass():
    # The whole point of fill_only: a second key reaches what the first
    # could not, and never revises what it decided.
    rows = _rows()
    rows['parcel_id_local'] = ['a', 'a', 'b']
    state = _state(rows)
    state = tx.flag_sales_matching_other_kind(state, key_column='parcel_id_local')
    first = state.curated['sale_matches_deed'].copy()
    state = tx.flag_sales_matching_other_kind(
        state, key_column=['book', 'page'], fill_only=True
    )
    after = state.curated['sale_matches_deed']
    decided = first.notna()
    assert (after[decided] == first[decided]).all()


def test_a_state_without_the_key_columns_still_gets_the_column():
    rows = _rows().drop(columns=['book', 'page'])
    out = tx.flag_sales_matching_other_kind(
        _state(rows), key_column=['book', 'page']
    ).curated
    assert 'sale_matches_deed' in out.columns
    assert out['sale_matches_deed'].isna().all()
