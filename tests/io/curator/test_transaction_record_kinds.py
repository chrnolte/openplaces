"""Curate steps that keep deeds and assessor last sales apart.

Every value here is fabricated.
"""

from __future__ import annotations

import pandas as pd

from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState
from openplaces.io.curator.transactions import (
    collapse_double_closings,
    dedup_transactions,
    derive_document_id,
    derive_sale_period,
    flag_sales_matching_other_kind,
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


def _pair() -> pd.DataFrame:
    """One sale reported as a deed and as the roll's last-sale field."""
    return pd.DataFrame(
        {
            'sale_record_kind': ['deed', 'assessor_last_sale'],
            'parcel_id_local': ['p1', 'p1'],
            'parcel_id_assessor': ['p1', 'p1'],
            'recorded_date': pd.to_datetime(['2021-06-17', None]),
            'sale_year': [None, 2021.0],
            'sale_month': [None, 6.0],
            'price': [210000.0, 210000.0],
            'book': ['40', None],
            'page': ['7', None],
            'sale_book': [None, '40'],
            'sale_page': [None, '7'],
        }
    )


def test_sale_period_is_filled_from_the_date_and_never_overwritten():
    df = pd.DataFrame(
        {
            'recorded_date': pd.to_datetime(['2021-06-17', '2019-03-02', None]),
            'sale_year': [None, 2018.0, 2020.0],
            'sale_month': [None, None, 4.0],
        }
    )
    out = derive_sale_period(_state(df)).curated
    assert out['sale_year'].tolist() == [2021.0, 2018.0, 2020.0]
    assert out['sale_month'].tolist() == [6.0, 3.0, 4.0]


def test_dedup_tolerates_a_key_column_the_county_lacks():
    df = _pair()
    key = ['sale_record_kind', 'parcel_id_local', 'price', 'sale_clerk_instrument']
    out = dedup_transactions(_state(df), key_columns=key).curated
    # Different kinds, so both stay; the absent column raised before.
    assert len(out) == 2


def test_deed_and_its_assessor_echo_do_not_share_a_document():
    out = derive_document_id(
        _state(_pair()),
        candidates=[['sale_book', 'sale_page'], ['book', 'page']],
        scope_column='sale_record_kind',
        unscoped_value='deed',
    ).curated
    assert out['sale_document_id'].tolist() == ['40/7', 'assessor_last_sale:40/7']


def test_placeholder_book_and_page_name_no_document():
    # A roll that writes a space (or zeros) where it has no book and
    # page would otherwise name one document for every row.
    df = pd.DataFrame(
        {
            'sale_book': [' ', '0', '000', '40', None],
            'sale_page': [' ', '0', '7', '7', '7'],
        }
    )
    out = derive_document_id(
        _state(df), candidates=[['sale_book', 'sale_page']]
    ).curated
    assert out['sale_document_id'].isna().tolist() == [True, True, True, False, True]
    assert out['sale_document_id'][3] == '40/7'


def test_assessor_rows_of_one_deed_still_share_a_document():
    df = pd.DataFrame(
        {
            'sale_record_kind': ['assessor_last_sale'] * 2,
            'sale_book': ['40', '40'],
            'sale_page': ['7', '7'],
        }
    )
    out = derive_document_id(
        _state(df),
        candidates=[['sale_book', 'sale_page']],
        scope_column='sale_record_kind',
        unscoped_value='deed',
    ).curated
    assert out['sale_document_id'].nunique() == 1


def test_deed_and_its_echo_are_not_a_double_closing():
    df = derive_sale_period(_state(_pair())).curated
    flagged = collapse_double_closings(
        _state(df.copy()), key_column='parcel_id_assessor', output='leg'
    ).curated
    assert flagged['leg'].sum() == 1  # the defect `within` removes
    scoped = collapse_double_closings(
        _state(df.copy()),
        key_column='parcel_id_assessor',
        output='leg',
        within=['sale_record_kind'],
    ).curated
    assert scoped['leg'].sum() == 0


def _set_keep_personal(monkeypatch, keep: bool) -> None:
    """Fix the installation's choice, whatever this machine's config says."""
    import openplaces.config as config

    monkeypatch.setattr(config, 'get_keep_personal_columns', lambda: keep)


def test_only_registered_non_personal_columns_leave_the_build(monkeypatch):
    from openplaces.io.curator.formatters import keep_registered_columns

    _set_keep_personal(monkeypatch, False)

    df = pd.DataFrame(
        {
            'price': [1.0],
            'land_value_parcel': [2.0],
            'address_source': ['fabricated'],
            'grantor': ['Fabricated Seller'],
            'grantee_address': ['1 Fabricated Way'],
            'owner_name_parcel': ['Fabricated Owner'],
            'Grantee Agent Name': ['Fabricated Agent'],
            'Preparer Name': ['Fabricated Preparer'],
            '_join_id': ['x'],
        }
    )
    out = keep_registered_columns(
        _state(df), exclude=['grantor', 'owner_name'], keep=['_join_id']
    ).curated
    assert list(out.columns) == [
        'price',
        'land_value_parcel',
        'address_source',
        '_join_id',
    ]


def test_an_installation_may_keep_personal_columns_but_never_raw_headers(
    monkeypatch,
):
    from openplaces.io.curator.formatters import keep_registered_columns

    _set_keep_personal(monkeypatch, True)
    df = pd.DataFrame(
        {
            'price': [1.0],
            'grantor': ['Fabricated Seller'],
            'owner_name_parcel': ['Fabricated Owner'],
            'Grantee Agent Name': ['Fabricated Agent'],
            'grantee_address': ['1 Fabricated Way'],
        }
    )
    out = keep_registered_columns(_state(df)).curated
    assert list(out.columns) == ['price', 'grantor', 'owner_name_parcel']
    # A recipe can still drop one for everybody.
    out = keep_registered_columns(_state(df), exclude=['grantor']).curated
    assert list(out.columns) == ['price', 'owner_name_parcel']


def test_no_recipe_setting_turns_personal_columns_on():
    import inspect

    from openplaces.io.curator.formatters import keep_registered_columns

    assert set(inspect.signature(keep_registered_columns).parameters) == {
        'state',
        'exclude',
        'keep',
    }


def test_a_delivery_never_carries_personal_columns():
    import pytest

    from openplaces.io import delivery

    for column in (
        'grantor',
        'grantee',
        'owner_name',
        'owner_name_parcel',
        'owner_address_source',
        'grantor_source',
    ):
        assert delivery._is_personal_column(column), column
    for column in ('price', 'land_value_parcel', 'address', 'address_source'):
        assert not delivery._is_personal_column(column), column
    delivery._refuse_personal_columns(['price', 'address'])
    with pytest.raises(delivery.PersonalColumnError):
        delivery._refuse_personal_columns(['price', 'owner_name_parcel'])


def test_curate_recipe_keeps_party_and_owner_columns_out_of_share(monkeypatch):
    # Runs the shipped recipe's own allow-list step, with its own
    # arguments, over every personal attribute the registry lists and
    # over raw source headers of the kind RETR brings along.
    from openplaces.core.attribute_registry import load_registry

    _set_keep_personal(monkeypatch, False)
    from openplaces.io.curator.formatters import keep_registered_columns
    from openplaces.recipe import get_recipe_by_id

    recipe = get_recipe_by_id('US_transaction-openplaces-2026')
    steps = [s for s in recipe['pipeline'] if s['step'] == 'keep_registered_columns']
    assert len(steps) == 1
    # After it, nothing may add a column again.
    later = recipe['pipeline'][recipe['pipeline'].index(steps[0]) + 1 :]
    assert [s['step'] for s in later] == ['cast_categoricals', 'order_columns']

    personal = [
        name
        for name in load_registry().index
        if name.startswith(('owner_', 'grantor', 'grantee'))
    ]
    assert {'grantor', 'grantee', 'owner_name', 'owner_address'} <= set(personal)
    raw = ['Grantor Name', 'Grantee Agent Address', 'Preparer Name', 'Tax Bill Name']
    unregistered = ['grantor_address', 'grantee_address']
    suffixed = [f'{name}_parcel' for name in personal]
    columns = ['price', *personal, *suffixed, *raw, *unregistered]
    df = pd.DataFrame({c: ['fabricated'] for c in columns})
    params = {k: v for k, v in steps[0].items() if k != 'step'}
    out = keep_registered_columns(_state(df), **params).curated
    assert list(out.columns) == ['price']


def _grading_specs(output):
    from openplaces.recipe import get_recipe_by_id

    recipe = get_recipe_by_id('US_transaction-openplaces-2026')
    return [
        spec
        for step in recipe['pipeline']
        for spec in step.get('indicators', [])
        if spec['output'] == output
    ]


def test_wisconsin_sale_is_graded_from_conveyance_and_relationship():
    from openplaces.io.curator.inferers import derive_indicators

    df = pd.DataFrame(
        {
            'conveyance_type': [
                'Sale',
                'Sale',
                'Sale',
                'Gift',
                'Sale',
                'Trust (conveyance to)',
                'Sale',
                None,
            ],
            'sale_party_relationship': [
                'No relationship',
                'Family',
                'Other',
                'No relationship',
                'Spouses',
                'No relationship',
                None,
                'Family',
            ],
        }
    )
    from openplaces.recipe import get_recipe_by_id

    recipe = get_recipe_by_id('US_transaction-openplaces-2026')
    wanted = {
        '_conveyance_grade',
        '_relationship_grade',
        'sale_arms_length_confidence',
        'sale_qualification_group',
    }
    specs = [
        spec
        for step in recipe['pipeline']
        for spec in step.get('indicators', [])
        if spec['output'] in wanted
    ]
    out = derive_indicators(_state(df), indicators=specs).curated
    conf = out['sale_arms_length_confidence']
    # A sale between strangers is the only 1.0; an unlisted conveyance
    # type is not a sale.
    assert conf.tolist()[:6] == [1.0, 0.0, 0.7, 0.0, 0.0, 0.0]
    # Half an answer is not graded.
    assert conf.isna().tolist()[6:] == [True, True]
    assert out['sale_qualification_group'].tolist()[:6] == [
        'qualified',
        'related_parties_family',
        'related_parties_unspecified',
        'not_a_sale',
        'related_parties_family',
        'not_a_sale',
    ]


def test_florida_codes_grade_like_their_labels():
    from pathlib import Path

    import openplaces

    specs = _grading_specs('sale_arms_length_confidence')
    florida = next(s for s in specs if s['column'] == 'sale_qualification_code')
    labels = pd.read_csv(
        Path(openplaces.__file__).parent
        / 'recipes/US/FL/_all/transaction/fldor/2026'
        / 'US-FL_transaction-fldor-2026_sale-qualification-code-labels.csv',
        dtype=str,
    )
    mapping = florida['mapping']
    for code, label in zip(labels['code'], labels['label']):
        key = f'{int(code):02d}'
        assert mapping.get(key) == mapping.get(label), (key, label)


def test_a_second_source_fills_only_what_the_first_left_missing():
    from openplaces.io.curator.inferers import derive_indicators

    df = pd.DataFrame({'a': ['x', None], 'b': ['y', 'y']})
    specs = [
        {'output': 'grade', 'type': 'value_map', 'column': 'a', 'mapping': {'x': 1.0}},
        {
            'output': 'grade',
            'type': 'value_map',
            'column': 'b',
            'mapping': {'y': 0.0},
            'fill_only': True,
        },
    ]
    out = derive_indicators(_state(df), indicators=specs).curated
    assert out['grade'].tolist() == [1.0, 0.0]


def test_echo_is_flagged_on_exact_month_and_price_only():
    df = pd.concat(
        [
            _pair(),
            pd.DataFrame(
                {
                    'sale_record_kind': ['assessor_last_sale'] * 3,
                    'parcel_id_local': ['p2', 'p1', 'p3'],
                    'sale_year': [2021.0, 2021.0, None],
                    'sale_month': [6.0, 6.0, 6.0],
                    # p2: no deed; p1: a dollar off; p3: no year.
                    'price': [210000.0, 210001.0, 210000.0],
                }
            ),
        ],
        ignore_index=True,
    )
    df = derive_sale_period(_state(df)).curated
    out = flag_sales_matching_other_kind(_state(df), key_column='parcel_id_local')
    flag = out.curated['sale_matches_deed']
    assert pd.isna(flag[0])  # the deed itself
    assert flag[1] == 1.0
    assert flag[2] == 0.0
    assert flag[3] == 0.0
    assert pd.isna(flag[4])
    assert len(out.curated) == 5  # nothing removed
