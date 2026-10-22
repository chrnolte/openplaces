"""New Jersey SR1A and MOD-IV parcel recipes: the shared lot key.

The SR1A sales file states a lot in fixed-width parts (zero-padded
block and lot, suffixes, qualifier); the NJOGIS parcel layer states it
as PAMS_PIN. The transaction recipe rebuilds the PAMS_PIN form, so the
two sides must agree string for string. All values are fabricated.
"""

import pandas as pd
import pytest

from openplaces.io.transform import apply_transformations
from openplaces.recipe import get_recipe_by_id

SALES = 'US-NJ_transaction-njtaxation-2026'
PARCELS = 'US-NJ-CU_parcel-njogis-2026'


def _sales_frame(rows):
    columns = [
        'COUNTY_CODE',
        'DISTRICT_CODE',
        'BLOCK',
        'BLOCK_SUFFIX',
        'LOT',
        'LOT_SUFFIX',
        'QUALIFICATION_CODES',
        'U_N_TYPE',
        'SR_NU_CODE',
    ]
    df = pd.DataFrame(rows, columns=columns, dtype=object)
    # The recipe's `columns` mapping runs before its transformations;
    # these are the mapped names the transformations read.
    for col in [
        'recorded_date',
        'price_raw',
        'land_value',
        'improvement_value',
        'total_value',
        'year_built',
        'living_area_sqft',
    ]:
        df[col] = '0000000'
    df['recorded_date'] = '250704'
    return df


@pytest.mark.parametrize(
    ('parts', 'expected'),
    [
        (['99', '98', '00123', None, '00045', None, None], '9998_123_45'),
        (['99', '98', '00123', '01', '00045', None, None], '9998_123.01_45'),
        (['99', '98', '00123', None, '00045', '02', None], '9998_123_45.02'),
        (['99', '98', '00123', None, '00045', None, 'C0007'], '9998_123_45_C0007'),
        (
            ['99', '98', '00123', '01', '00045', '02', 'QFARM'],
            '9998_123.01_45.02_QFARM',
        ),
    ],
)
def test_sales_key_takes_pams_pin_form(parts, expected):
    recipe = get_recipe_by_id(SALES)
    out = apply_transformations(_sales_frame([parts + ['U', None]]), recipe)
    assert out['parcel_id_assessor'].iloc[0] == expected


def test_sales_one_digit_lot_suffix_is_zero_filled_where_scoped():
    recipe = get_recipe_by_id(SALES)
    rows = [['99', '98', '00123', None, '00045', '1', None, 'U', None]]
    scoped = apply_transformations(
        _sales_frame(rows), recipe, process_admin_id='US-NJ-CU'
    )
    assert scoped['parcel_id_assessor'].iloc[0] == '9998_123_45.01'
    elsewhere = apply_transformations(
        _sales_frame(rows), recipe, process_admin_id='US-NJ-SA'
    )
    assert elsewhere['parcel_id_assessor'].iloc[0] == '9998_123_45.1'


def test_sales_qualification_code_and_recorded_date():
    recipe = get_recipe_by_id(SALES)
    rows = [
        ['99', '98', '00001', None, '00001', None, None, 'U', None],
        ['99', '98', '00001', None, '00002', None, None, 'N', '25'],
    ]
    out = apply_transformations(_sales_frame(rows), recipe)
    assert out['sale_qualification_code'].tolist() == ['U', 'N25']
    assert out['recorded_date'].iloc[0] == pd.Timestamp('2025-07-04')
    assert pd.isna(out['living_area_sqft'].iloc[0])


def test_curate_patch_grades_sr1a_codes_and_leaves_unknown_missing():
    from openplaces.core.schema import AdminId
    from openplaces.io.curator import CurateState
    from openplaces.io.curator.inferers import derive_indicators
    from openplaces.recipe.patches import apply_recipe_patches

    national = get_recipe_by_id('US_transaction-openplaces-2026')
    patched, applied = apply_recipe_patches(national, 'US-NJ-CU')
    assert 'US-NJ_transaction-openplaces-2026' in str(applied)
    # Every grading spec keyed on the code, national ones first, in
    # pipeline order, as the curator runs them.
    specs = [
        spec
        for step in patched['pipeline']
        if step['step'] == 'derive_indicators'
        for spec in step['indicators']
        if spec.get('column') == 'sale_qualification_code'
    ]
    codes = ['U', 'N07', 'N25', 'N01', 'N99', None]
    state = CurateState(
        recipe={},
        entity_recipe={},
        admin_id=AdminId('US-NJ-CU'),
        verbose=False,
        timer=None,
        curated=pd.DataFrame({'sale_qualification_code': codes}),
    )
    out = derive_indicators(state, indicators=specs).curated
    conf = out['sale_arms_length_confidence']
    assert conf.tolist()[:4] == [1.0, 0.7, 0.0, 0.0]
    assert conf.iloc[4:].isna().all()
    assert out['sale_qualification_group'].tolist()[:4] == [
        'qualified',
        'qualified_parcel_changed',
        'not_a_sale',
        'related_parties_family',
    ]
    assert out['sale_qualification_group'].iloc[4:].isna().all()


def test_parcel_deed_date_resolves_century():
    recipe = get_recipe_by_id(PARCELS)
    df = pd.DataFrame(
        {
            'last_sale_date': ['950412', '260101', '000000', None],
            'DEED_BOOK': ['1234', '1234', None, None],
            'DEED_PAGE': ['56', '57', None, None],
        }
    )
    out = apply_transformations(df, recipe)
    assert out['last_sale_date'].iloc[0] == pd.Timestamp('1995-04-12')
    assert out['last_sale_date'].iloc[1] == pd.Timestamp('2026-01-01')
    assert out['last_sale_date'].iloc[2:].isna().all()
    assert out['last_sale_book_page'].iloc[0] == '1234-56'
    assert out['last_sale_book_page'].iloc[2:].isna().all()
