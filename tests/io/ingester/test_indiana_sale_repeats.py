"""Indiana's sales disclosure headers keep one row per saved sale.

The SALEDISC export repeats a sale on rows that differ only in
workflow fields the recipe does not map. These tests run the real
recipe's preprocessing on fabricated rows: repeats collapse to one,
and rows sharing an SDF_ID with another price or date all stay.
"""

from pathlib import Path

import pandas as pd

from openplaces.io.ingester.table_ingester import TableIngester
from openplaces.recipe import get_recipe_by_id

RECIPE_ID = 'US-IN_transaction-statsindiana-2026'


class _Timer:
    def mark(self, *args, **kwargs):
        pass


def _preprocess(raw):
    recipe = get_recipe_by_id(RECIPE_ID)
    ingester = TableIngester(
        recipe, {}, {'admin_id_to_process': 'US-IN'}, Path('.'), _Timer()
    )
    return ingester._preprocess_recipe_data(raw)


def _raw(rows):
    columns = [
        'SDF_ID',
        'E1_Sales_Price',
        'C7_Conveyance_Date',
        'County_ID',
        'P2_18_Validated_By',
    ]
    return pd.DataFrame(rows, columns=columns, dtype=str)


def test_the_drop_list_is_every_mapped_column():
    recipe = get_recipe_by_id(RECIPE_ID)
    assert recipe['drop_duplicates'] == list(recipe['columns'])


def test_rows_repeating_a_sale_collapse_to_one():
    raw = _raw(
        [
            ['900001', '150000', '2025-03-01', '01', 'reviewer a'],
            ['900001', '150000', '2025-03-01', '01', 'reviewer b'],
            ['900001', '150000', '2025-03-01', '01', None],
            ['900002', '87000', '2025-04-15', '01', 'reviewer a'],
        ]
    )
    out = _preprocess(raw)
    assert len(out) == 2
    assert not out.duplicated().any()
    assert sorted(out['instrument_number']) == ['900001', '900002']


def test_same_id_with_another_price_or_date_stays_apart():
    raw = _raw(
        [
            ['900003', '210000', '2025-05-02', '02', 'reviewer a'],
            ['900003', '215000', '2025-05-02', '02', 'reviewer a'],
            ['900003', '210000', '2025-06-02', '02', 'reviewer b'],
        ]
    )
    out = _preprocess(raw)
    assert len(out) == 3
    assert out['instrument_number'].eq('900003').all()
    assert sorted(out['price'].astype(float)) == [210000.0, 210000.0, 215000.0]
