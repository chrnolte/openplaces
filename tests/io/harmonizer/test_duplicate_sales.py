"""Pairs of sale rows that describe one sale (plans/transaction-sale-
deduplication.md, sections 4, 5 and 8). One test per rule, each with a
near miss that must not pair. Every value is fabricated.
"""

from __future__ import annotations

import pandas as pd
import pytest

from openplaces.core.attribute_registry import is_personal_attribute
from openplaces.io import sale_records as sr
from openplaces.io.harmonizer.last_sales import (
    last_sales_to_transactions,
    restated_last_sales,
)
from openplaces.recipe import apply_recipe_patches, get_recipe_by_id

DEED = 'deed'
LAST = 'assessor_last_sale'


def _sales(rows):
    """Sales indexed by transaction_id from (id, kind, doc, year, month,
    price[, date]) tuples."""
    frame = pd.DataFrame(
        [
            {
                'transaction_id': r[0],
                'sale_record_kind': r[1],
                'sale_document_id': r[2],
                'sale_year': r[3],
                'sale_month': r[4],
                'price': r[5],
                'recorded_date': r[6] if len(r) > 6 else None,
                'source': 'src',
            }
            for r in rows
        ]
    )
    return frame.set_index('transaction_id')


def _links(rows, id_column):
    return pd.DataFrame(
        rows, columns=['transaction_id', id_column, 'link_method']
    ).astype('string')


def _pairs(sales, parcel_rows=(), property_rows=(), floor=100.0, **kwargs):
    things = sr.sale_things(
        _links(list(parcel_rows), 'parcel_id'),
        _links(list(property_rows), 'property_id'),
    )
    return sr.duplicate_sale_pairs(sales, things, floor, **kwargs)


def _as_set(pairs):
    return {
        (a, b, m)
        for a, b, m in zip(
            pairs['transaction_id'], pairs['other_transaction_id'], pairs['link_method']
        )
    }


def test_echo_by_document_pairs_the_last_sale_with_its_deed():
    sales = _sales(
        [
            ('D1', DEED, '10/20', 2020, 5, 300_000),
            ('L1', LAST, 'assessor_last_sale:10/20', 2020, 5, 300_000),
            # Near miss: cites a document no deed here names.
            ('L2', LAST, 'assessor_last_sale:11/21', 2020, 5, 300_000),
        ]
    )
    pairs = _pairs(sales)
    assert _as_set(pairs) == {('L1', 'D1', sr.DEED_ECHO_DOCUMENT)}


def test_echo_of_a_multi_parcel_deed_is_reached_through_the_link_table():
    # The folded deed states no parcel; its link rows do.
    sales = _sales(
        [
            ('D1', DEED, '500', 2021, 3, 900_000),
            ('L1', LAST, None, 2021, 3, 900_000),
            # Near miss: same parcel and month, another price.
            ('L2', LAST, None, 2021, 3, 899_000),
        ]
    )
    parcels = [
        ('D1', 'pA', 'parcel_id_local'),
        ('D1', 'pB', 'parcel_id_local'),
        ('L1', 'pB', 'assessor_last_sale'),
        ('L2', 'pA', 'assessor_last_sale'),
    ]
    pairs = _pairs(sales, parcels)
    assert _as_set(pairs) == {('L1', 'D1', sr.DEED_ECHO_PERIOD_PRICE)}


def test_a_document_echo_is_not_labeled_twice():
    sales = _sales(
        [
            ('D1', DEED, '10/20', 2020, 5, 300_000),
            ('L1', LAST, 'assessor_last_sale:10/20', 2020, 5, 300_000),
        ]
    )
    parcels = [('D1', 'p', 'parcel_id_local'), ('L1', 'p', 'assessor_last_sale')]
    pairs = _pairs(sales, parcels)
    assert _as_set(pairs) == {('L1', 'D1', sr.DEED_ECHO_DOCUMENT)}


def test_one_last_sale_stated_twice_keeps_the_row_read_first():
    sales = _sales(
        [
            ('L1', LAST, 'assessor_last_sale:1/1', 2015, 2, 150_000),
            ('L2', LAST, 'assessor_last_sale:2/2', 2015, 2, 150_000),
            # Near miss: a different last sale of the same property.
            ('L3', LAST, 'assessor_last_sale:3/3', 2019, 8, 210_000),
        ]
    )
    props = [
        ('L1', 'acct', 'parcel_id_local'),
        ('L2', 'acct', 'parcel_id_local'),
        ('L3', 'acct', 'parcel_id_local'),
    ]
    pairs = _pairs(sales, property_rows=props)
    assert _as_set(pairs) == {('L2', 'L1', sr.ROLL_RESTATED)}


def test_a_market_double_closing_keeps_the_later_leg():
    sales = _sales(
        [
            ('D1', DEED, '100', 2022, 6, 400_000),
            ('D2', DEED, '101', 2022, 7, 400_000),
            # Near miss: same price, three months on.
            ('D3', DEED, '102', 2022, 10, 400_000),
        ]
    )
    parcels = [(t, 'p', 'parcel_id_local') for t in ('D1', 'D2', 'D3')]
    pairs = _pairs(sales, parcels)
    assert _as_set(pairs) == {('D1', 'D2', sr.DOUBLE_CLOSING)}


def test_a_nominal_pair_is_not_labeled():
    sales = _sales(
        [
            ('D1', DEED, '100', 2022, 6, 100),
            ('D2', DEED, '101', 2022, 6, 100),
            ('D3', DEED, '102', 2022, 6, 250_000),
        ]
    )
    parcels = [(t, 'p', 'parcel_id_local') for t in ('D1', 'D2', 'D3')]
    assert _pairs(sales, parcels, floor=100.0).empty


def test_a_resale_at_another_price_is_labeled_and_is_not_a_duplicate():
    sales = _sales(
        [
            ('D1', DEED, '100', 2022, 6, 200_000),
            ('D2', DEED, '101', 2022, 6, 260_000),
        ]
    )
    parcels = [(t, 'p', 'parcel_id_local') for t in ('D1', 'D2')]
    pairs = _pairs(sales, parcels)
    assert _as_set(pairs) == {('D1', 'D2', sr.RESALE_WITHIN_WINDOW)}
    summary = sr.duplicate_columns(sales.index, pairs)
    assert summary['sale_duplicate_of'].isna().all()
    assert summary['sale_resale_within_window'].tolist() == [1.0, 1.0]


def test_two_units_of_one_stacked_lot_are_never_paired():
    sales = _sales(
        [
            ('D1', DEED, '100', 2022, 6, 180_000),
            ('D2', DEED, '101', 2022, 6, 180_000),
        ]
    )
    # Both reach lot L through their units; no property link.
    parcels = [('D1', 'L', 'stacked_units'), ('D2', 'L', 'stacked_units')]
    assert _pairs(sales, parcels).empty
    # With property links, each is its own unit: still never paired.
    props = [('D1', 'U1', 'parcel_id_local'), ('D2', 'U2', 'parcel_id_local')]
    parcels = [('D1', 'L', 'parcel_id_local'), ('D2', 'L', 'parcel_id_local')]
    assert _pairs(sales, parcels, props).empty


def test_day_stated_pairs_need_the_same_day():
    sales = _sales(
        [
            ('D1', DEED, '100', 2022, 6, 300_000, '2022-06-30'),
            ('D2', DEED, '101', 2022, 7, 300_000, '2022-07-02'),
            ('D3', DEED, '102', 2022, 8, 300_000, '2022-08-15'),
            ('D4', DEED, '103', 2022, 8, 300_000, '2022-08-15'),
        ]
    )
    parcels = [(t, 'p', 'parcel_id_local') for t in ('D1', 'D2', 'D3', 'D4')]
    pairs = _pairs(sales, parcels, floor=0.0)
    assert _as_set(pairs) == {('D3', 'D4', sr.DOUBLE_CLOSING)}


def test_a_correction_names_the_return_it_restates():
    sales = _sales(
        [
            ('D1', DEED, '5001', 2021, 1, 120_000, '2021-01-10'),
            ('D2', DEED, '5090', 2021, 4, 0, '2021-04-02'),
            # Near miss: a land-contract deed naming its contract.
            ('D3', DEED, '5100', 2020, 1, 90_000, '2020-01-05'),
            ('D4', DEED, '6100', 2023, 9, 0, '2023-09-01'),
        ]
    )
    sales['previous'] = [None, '5001', None, '5100']
    sales['exemption'] = [None, '3 - Correct prior', None, '17 - Land contract']
    parcels = [
        ('D1', 'p', 'parcel_id_local'),
        ('D2', 'p', 'parcel_id_local'),
        ('D3', 'q', 'parcel_id_local'),
        ('D4', 'q', 'parcel_id_local'),
    ]
    pairs = _pairs(
        sales,
        parcels,
        floor=0.0,
        previous_document_column='previous',
        correction_column='exemption',
        correction_values=['3 - Correct prior'],
    )
    assert _as_set(pairs) == {('D1', 'D2', sr.RESTATES_DOCUMENT)}


def test_the_pair_table_holds_four_columns_and_known_labels():
    sales = _sales(
        [
            ('D1', DEED, '10/20', 2020, 5, 300_000),
            ('L1', LAST, 'assessor_last_sale:10/20', 2020, 5, 300_000),
        ]
    )
    pairs = _pairs(sales)
    assert list(pairs.columns) == list(sr.DUPLICATE_PAIR_COLUMNS)
    assert set(pairs['link_method']) <= set(sr.DUPLICATE_LABELS)


def _mixed_fixture():
    sales = _sales(
        [
            ('D1', DEED, '100', 2022, 6, 400_000),
            ('D2', DEED, '101', 2022, 6, 400_000),
            ('L1', LAST, None, 2022, 6, 400_000),
            ('D3', DEED, '102', 2022, 7, 410_000),
        ]
    )
    parcels = [(t, 'p', 'parcel_id_local') for t in ('D1', 'D2', 'L1', 'D3')]
    return sales, parcels


def test_no_personal_column_changes_the_pairs():
    sales, parcels = _mixed_fixture()
    plain = _pairs(sales, parcels)
    named = sales.copy()
    # Names that would split or join every group if any rule read them.
    named['grantor'] = ['A', 'B', 'C', 'D']
    named['grantee'] = ['E', 'E', 'E', 'E']
    assert is_personal_attribute('grantor') and is_personal_attribute('grantee')
    pd.testing.assert_frame_equal(_pairs(named, parcels), plain)


def test_no_value_ratio_or_index_changes_the_pairs():
    sales, parcels = _mixed_fixture()
    plain = _pairs(sales, parcels)
    valued = sales.copy()
    valued['land_value'] = [1.0, 9e6, 5.0, 3.0]
    valued['improvement_value'] = [2.0, 0.0, 8e5, 4.0]
    valued['sale_price_to_assessed_ratio'] = [0.1, 9.0, 1.0, 0.5]
    valued['price_index'] = [1.0, 2.0, 3.0, 4.0]
    pd.testing.assert_frame_equal(_pairs(valued, parcels), plain)
    summary = sr.duplicate_columns(valued.index, _pairs(valued, parcels))
    pd.testing.assert_frame_equal(summary, sr.duplicate_columns(sales.index, plain))


def test_summary_resolves_by_precedence_and_leaves_ids_alone():
    sales, parcels = _mixed_fixture()
    pairs = _pairs(sales, parcels)
    # D1-D2 double closing (D2 kept); L1 echoes both deeds; D3 a resale
    # of each.
    summary = sr.duplicate_columns(sales.index, pairs)
    assert summary.index.equals(sales.index)
    assert summary.loc['L1', 'sale_duplicate_of'] == 'D1'
    assert summary.loc['L1', 'sale_duplicate_reason'] == sr.DEED_ECHO_PERIOD_PRICE
    assert summary.loc['D1', 'sale_duplicate_of'] == 'D2'
    assert summary.loc['D1', 'sale_duplicate_reason'] == sr.DOUBLE_CLOSING
    assert pd.isna(summary.loc['D2', 'sale_duplicate_of'])
    assert summary['sale_resale_within_window'].tolist() == [1.0, 1.0, 0.0, 1.0]


def test_a_row_in_no_pair_has_nothing_and_no_table_leaves_all_missing():
    sales = _sales([('D1', DEED, '1', 2020, 1, 5.0)])
    summary = sr.duplicate_columns(sales.index, _pairs(sales))
    assert pd.isna(summary.loc['D1', 'sale_duplicate_of'])
    assert summary.loc['D1', 'sale_resale_within_window'] == 0.0
    missing = sr.duplicate_columns(sales.index, None)
    assert missing.isna().all().all()


def test_nominal_floor_is_one_mass_point():
    prices = pd.Series([0.0] * 30 + [100.0] * 40 + list(range(1000, 1030)))
    assert sr.nominal_price_floor(prices) == 100.0
    assert sr.nominal_price_floor(pd.Series([250_000.0, 300_000.0])) == 0.0


# The roll's restatement of one last sale (decision D2).


def _roll():
    # Account A's one sale under spelling 1/1 in roll years 2000-2002
    # and 9/9 from 2003; account B's sale stated under two spellings in
    # one roll year (overlap: both kept); account C sold twice.
    return pd.DataFrame(
        {
            'parcel_id_local': ['A'] * 5 + ['B'] * 2 + ['C'] * 2,
            'tax_year': [2000, 2001, 2002, 2003, 2004, 2010, 2010, 2005, 2006],
            'last_sale_price': [5e4] * 5 + [7e4] * 2 + [8e4, 9e4],
            'last_sale_year': [1999] * 5 + [2009] * 2 + [2004, 2006],
            'last_sale_month': [4] * 5 + [2] * 2 + [1, 5],
            'last_sale_book': ['1', '1', '1', '9', '9', '3', '4', '5', '6'],
            'last_sale_page': ['1', '1', '1', '9', '9', '3', '4', '5', '6'],
        }
    )


def test_a_restated_last_sale_keeps_the_latest_spelling():
    rows = last_sales_to_transactions(_roll(), 'roll')
    by_account = rows.groupby('parcel_id_local')['sale_book'].agg(sorted)
    assert by_account['A'] == ['9']
    assert by_account['B'] == ['3', '4']
    assert by_account['C'] == ['5', '6']
    assert 'tax_year' not in rows.columns


def test_restatement_needs_disjoint_roll_years():
    roll = _roll()
    rows = roll.rename(
        columns={
            'last_sale_year': 'sale_year',
            'last_sale_month': 'sale_month',
            'last_sale_price': 'price',
            'last_sale_book': 'sale_book',
            'last_sale_page': 'sale_page',
        }
    )
    # A's old spelling returns in 2005: the ranges overlap, both stay.
    rows.loc[len(rows)] = ['A', 2005, 5e4, 1999, 4, '1', '1']
    mask = restated_last_sales(rows, rows['tax_year'], ['parcel_id_local'])
    assert not mask.any()


def test_without_a_roll_year_nothing_is_restated():
    roll = _roll().drop(columns='tax_year')
    rows = last_sales_to_transactions(roll, 'roll')
    assert sorted(rows.loc[rows['parcel_id_local'] == 'A', 'sale_book']) == ['1', '9']


# Recipes.


def test_the_spine_links_duplicates_after_the_sales_are_linked():
    steps = [
        s['step'] for s in get_recipe_by_id('US_transaction-spine-2026')['pipeline']
    ]
    at = steps.index('link_duplicate_sales')
    assert steps.index('link_transactions') < at
    assert at < steps.index('check_parcel_links_by_address_points')


def test_wisconsin_reads_previous_documents_only_on_corrections():
    recipe, _ = apply_recipe_patches(
        get_recipe_by_id('US_transaction-spine-2026'), 'US-WI-VI'
    )
    (step,) = [s for s in recipe['pipeline'] if s['step'] == 'link_duplicate_sales']
    assert step['previous_document_column'] == 'Previous Document Number'
    assert step['correction_values'] == ['3 - Correct prior']


@pytest.mark.parametrize(
    'column',
    ['sale_duplicate_of', 'sale_duplicate_reason', 'sale_resale_within_window'],
)
def test_the_three_columns_are_shared_and_not_personal(column):
    recipe = get_recipe_by_id('US_transaction-openplaces-2026')
    assert column in recipe['share']['columns']
    assert not is_personal_attribute(column)
    steps = [s['step'] for s in recipe['pipeline']]
    assert 'derive_duplicate_columns' in steps
