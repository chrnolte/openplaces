"""One row per recorded sale, not one per parcel.

A deed covering several parcels is published once per parcel with the
same price on each row, so a consumer counting rows counts one transfer
several times and a per-area price uses one parcel's area against the
whole deed's price. The three steps here name the document, count the
parcels it covered, and collapse its rows into one observation whose
extensive columns are summed. The `value_map` indicator is what grades a
source's own sale vocabulary into a value a consumer can threshold.
"""

from __future__ import annotations

import pandas as pd
import pytest

from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState
from openplaces.io.curator.inferers import derive_indicators
from openplaces.io.curator.transactions import (
    aggregate_multi_parcel_sales,
    count_parcels_per_document,
    derive_document_id,
)


def _state(df: pd.DataFrame) -> CurateState:
    return CurateState(
        recipe={},
        entity_recipe={},
        admin_id=AdminId('US'),
        verbose=False,
        timer=None,
        curated=df,
    )


def _sales():
    # Deed 100/1 covers parcels A and B (a multi-parcel sale); deed 200/2
    # covers C alone; D's sale has a clerk instrument number and no book.
    return pd.DataFrame(
        {
            'parcel_id_assessor': ['A', 'B', 'C', 'D'],
            'sale_book': ['100', '100', '200', None],
            'sale_page': ['1', '1', '2', None],
            'sale_clerk_instrument': [None, None, None, '2024-55'],
            'price': [300000.0, 300000.0, 90000.0, 50000.0],
            'sale_year': [2020, 2020, 2021, 2024],
            'area_ha': [2.0, 6.0, 1.0, 0.5],
        },
        index=pd.Index(['t1', 't2', 't3', 't4'], name='transaction_id'),
    )


CANDIDATES = [['sale_clerk_instrument'], ['sale_book', 'sale_page']]


def test_document_id_takes_the_first_complete_candidate():
    state = derive_document_id(_state(_sales()), candidates=CANDIDATES)
    document = state.curated['sale_document_id']
    assert document.tolist() == ['100/1', '100/1', '200/2', '202455']


def test_document_id_is_missing_when_no_candidate_is_complete():
    df = _sales()
    df.loc['t3', 'sale_page'] = None
    state = derive_document_id(_state(df), candidates=CANDIDATES)
    assert pd.isna(state.curated.loc['t3', 'sale_document_id'])


def test_parcels_per_document_counts_distinct_parcels():
    state = derive_document_id(_state(_sales()), candidates=CANDIDATES)
    state = count_parcels_per_document(state, parcel_column='parcel_id_assessor')
    assert state.curated['n_parcels_per_sale'].tolist() == [2.0, 2.0, 1.0, 1.0]


def test_multi_parcel_deed_becomes_one_row_with_area_summed():
    state = derive_document_id(_state(_sales()), candidates=CANDIDATES)
    state = count_parcels_per_document(state, parcel_column='parcel_id_assessor')
    state = aggregate_multi_parcel_sales(state, weight_column='area_ha')
    curated = state.curated

    # Four rows became three; the deed's row keeps its heaviest
    # member's label (B, 6 ha) so the entity id is a real row's.
    assert list(curated.index) == ['t2', 't3', 't4']
    deed = curated.loc['t2']
    assert deed['area_ha'] == pytest.approx(8.0)
    assert deed['price'] == pytest.approx(300000.0)
    assert deed['n_parcels_per_sale'] == 2.0
    assert deed['sale_document_id'] == '100/1'
    assert deed['parcel_id_assessor'] == 'B'
    # Single-parcel rows are untouched.
    assert curated.loc['t3', 'area_ha'] == pytest.approx(1.0)
    assert curated.loc['t4', 'sale_document_id'] == '202455'


def test_aggregation_is_a_no_op_without_shared_documents():
    df = _sales().drop(index='t2')
    state = derive_document_id(_state(df), candidates=CANDIDATES)
    state = aggregate_multi_parcel_sales(state, weight_column='area_ha')
    assert len(state.curated) == 3


QUALIFIED = "Qualified arm's length - deed/instrument examination"
DISQUALIFIED = 'Disqualified - corrective/quit claim/tax deed or minimal stamps'


def test_value_map_grades_a_source_vocabulary():
    df = pd.DataFrame(
        {
            'sale_qualification_code': [
                QUALIFIED,
                DISQUALIFIED,
                'Pending - recorded/discovered in previous 90 days',
                None,
            ]
        }
    )
    state = derive_indicators(
        _state(df),
        indicators=[
            {
                'output': 'sale_arms_length_confidence',
                'type': 'value_map',
                'column': 'sale_qualification_code',
                'mapping': {QUALIFIED: 1.0, DISQUALIFIED: 0.0},
                'default': 0.5,
            }
        ],
    )
    graded = state.curated['sale_arms_length_confidence'].tolist()
    assert graded[:3] == [1.0, 0.0, 0.5]
    # A missing source value is not in the mapping and takes the default.
    assert graded[3] == 0.5


def test_value_map_without_default_leaves_unnamed_values_missing():
    df = pd.DataFrame({'kind': ['deed', 'other']})
    state = derive_indicators(
        _state(df),
        indicators=[
            {
                'output': 'sale_record_kind',
                'type': 'value_map',
                'column': 'kind',
                'mapping': {'deed': 'deed'},
            }
        ],
    )
    assert state.curated['sale_record_kind'].tolist()[0] == 'deed'
    assert pd.isna(state.curated['sale_record_kind'].tolist()[1])


def test_group_share_is_the_countys_disclosure_rate_on_every_row():
    df = pd.DataFrame(
        {
            'use_group': ['Single Family'] * 4 + ['Commercial'],
            'price': [100.0, None, 1000.0, 250000.0, 1.0],
        }
    )
    state = derive_indicators(
        _state(df),
        indicators=[
            {
                'output': 'single_family_sales_disclosure_share',
                'type': 'group_share',
                'column': 'price',
                'predicate': 'above',
                'threshold': 1000,
                'restrict': {'column': 'use_group', 'equals': 'Single Family'},
            }
        ],
    )
    share = state.curated['single_family_sales_disclosure_share']
    # One of four single-family sales is priced above the 1,000 dollar
    # floor (100 and 1,000 are nominal); the commercial row reads the
    # same county share.
    assert share.tolist() == pytest.approx([0.25] * 5)


def test_double_closing_can_be_flagged_instead_of_dropped():
    from openplaces.io.curator.transactions import collapse_double_closings

    df = pd.DataFrame(
        {
            'parcel_id_assessor': ['A', 'A', 'B'],
            'sale_year': [2020, 2020, 2020],
            'sale_month': [3, 4, 3],
            'price': [100000.0, 100000.0, 50000.0],
            'sale_book': ['1', '2', '3'],
            'sale_page': ['1', '1', '1'],
        },
        index=pd.Index(['t1', 't2', 't3'], name='transaction_id'),
    )
    state = collapse_double_closings(
        _state(df), key_column='parcel_id_assessor', output='earlier_leg'
    )
    assert len(state.curated) == 3
    assert state.curated['earlier_leg'].tolist() == [1, 0, 0]


def test_nominal_floor_is_the_largest_low_price_mass_point():
    # Florida: 100 dollars on a third of rows, 0 on a tenth, real prices
    # spread above. 1,000 appears a few times but not enough to count.
    real = list(range(20000, 320000, 1000))  # 300 spread prices
    prices = [100.0] * 150 + [0.0] * 50 + [1000.0] * 2 + [float(x) for x in real]
    df = pd.DataFrame({'price': prices})
    state = derive_indicators(
        _state(df),
        indicators=[{'output': 'floor', 'type': 'nominal_floor', 'column': 'price'}],
    )
    assert state.curated['floor'].unique().tolist() == [100.0]


def test_nominal_floor_is_zero_when_only_zero_repeats():
    prices = [0.0] * 200 + [float(x) for x in range(5000, 305000, 1000)]
    state = derive_indicators(
        _state(pd.DataFrame({'price': prices})),
        indicators=[{'output': 'floor', 'type': 'nominal_floor', 'column': 'price'}],
    )
    assert state.curated['floor'].unique().tolist() == [0.0]


def test_disclosure_share_reads_the_detected_floor():
    prices = (
        [100.0] * 150
        + [0.0] * 50
        + [500.0, 250000.0]
        + [float(x) for x in range(20000, 320000, 1000)]
    )
    df = pd.DataFrame({'price': prices, 'use_group': 'Single Family'})
    state = derive_indicators(
        _state(df),
        indicators=[
            {'output': 'floor', 'type': 'nominal_floor', 'column': 'price'},
            {
                'output': 'share',
                'type': 'group_share',
                'column': 'price',
                'predicate': 'above',
                'threshold_column': 'floor',
                'restrict': {'column': 'use_group', 'equals': 'Single Family'},
            },
        ],
    )
    # 302 of 502 sales sit above the 100 dollar floor; the 500 dollar
    # sale counts as disclosed, which a round 1,000 floor would lose.
    assert state.curated['share'].iloc[0] == pytest.approx(302 / 502)
