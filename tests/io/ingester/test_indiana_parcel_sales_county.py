"""Indiana's parcel-sale rows are chunked by the county in their parcel number.

SALEPARCEL.txt names no county; DLGF's state parcel number does, in its
first two digits. These tests run the real recipe's process_by on
fabricated parcel numbers: each row reaches its county, and a row whose
prefix names no county is reported with its count. The parcel numbers
are fabricated (only the county prefix is meaningful).
"""

import copy
from pathlib import Path

import pandas as pd
import pytest

from openplaces.io.ingester.table_ingester import TableIngester
from openplaces.recipe import get_recipe_by_id, get_save_admin_level

RECIPE_ID = 'US-IN_transaction-statsindianaparcel-2026'


class _Timer:
    def mark(self, *args, **kwargs):
        pass


def _assign(parcel_numbers):
    recipe = copy.deepcopy(get_recipe_by_id(RECIPE_ID))
    partition = {'admin_id_to_download': 'US-IN', 'table_fids': {}}
    ingester = TableIngester(
        recipe, partition, {'admin_id_to_process': 'US-IN-AD'}, Path('.'), _Timer()
    )
    raw = pd.DataFrame({'A1_Parcel_Number': parcel_numbers}, dtype=str)
    ingester._read_recipe_data = lambda **kwargs: raw.copy()
    ingester._prepare_admin_id_crosswalk()
    ingester._prepare_table_fid_filter()
    return partition['table_fids'][ingester.table_name]


def test_the_recipe_saves_one_file_per_county():
    assert get_save_admin_level(get_recipe_by_id(RECIPE_ID)) == 3


def test_each_row_reaches_the_county_its_prefix_names():
    assigned = _assign(['010000000000000001', '020000000000000002'])
    assert assigned.tolist() == ['US-IN-AD', 'US-IN-AL']


def test_a_row_naming_no_county_is_reported():
    with pytest.warns(UserWarning, match='1 of 3 rows'):
        assigned = _assign(
            ['010000000000000001', '990000000000000002', '020000000000000003']
        )
    assert assigned.isna().tolist() == [False, True, False]
