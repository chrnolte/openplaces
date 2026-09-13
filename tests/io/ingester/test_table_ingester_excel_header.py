"""An Excel recipe without a `header` key reads its first row as header.

The flat-table reader defaulted `header` to 'infer', read_csv's
vocabulary, and passed it to read_excel, which rejects it: every Excel
recipe that did not set `header` failed. Values are fabricated.
"""

from __future__ import annotations

import pandas as pd

from openplaces.io.ingester.table_ingester import TableIngester


def _reader(recipe):
    ti = TableIngester.__new__(TableIngester)
    ti.recipe = recipe
    return ti


def test_excel_without_header_key_uses_the_first_row(tmp_path):
    path = tmp_path / 'dwelling.xlsx'
    pd.DataFrame({'parcel': ['p1', 'p2'], 'bedrooms': [3, 2]}).to_excel(
        path, index=False
    )
    df = _reader({})._read_flat_table(path, None, None)
    assert list(df.columns) == ['parcel', 'bedrooms']
    assert df['bedrooms'].tolist() == [3, 2]


def test_header_none_still_reads_without_a_header_row(tmp_path):
    path = tmp_path / 'dwelling.xlsx'
    pd.DataFrame({'parcel': ['p1'], 'bedrooms': [3]}).to_excel(path, index=False)
    df = _reader({'header': 'none'})._read_flat_table(path, None, None)
    assert len(df) == 2
