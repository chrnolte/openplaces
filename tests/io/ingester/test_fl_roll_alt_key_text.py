"""The Florida roll reads its county account key (ALT_KEY) as text.

A county's ALT_KEY changes shape between years: 21 digits in some
(beyond int64, so pandas holds Python ints and the parquet write
overflows), 5 to 6 digits in others, and empty in years where the
county left it blank, which pandas reads as float. Read with the
recipe's `csv_dtype`, every year writes the same string column, so
the year files roll up into one county file. Values are fabricated.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openplaces.io import save_parquet
from openplaces.io.ingester.table_ingester import TableIngester
from openplaces.recipe import get_recipe_by_id

RECIPE_ID = 'US-FL_property-fldor-2026'

YEARS = {
    'long': 'PARCEL_ID,ALT_KEY\nA1,123456789012345678901\nA2,223456789012345678901\n',
    'short': 'PARCEL_ID,ALT_KEY\nA1,12345\nA2,123456\n',
    'empty': 'PARCEL_ID,ALT_KEY\nA1,\nA2,\n',
}


def _reader(recipe):
    ti = TableIngester.__new__(TableIngester)
    ti.recipe = recipe
    return ti


def test_recipe_reads_alt_key_as_text():
    assert get_recipe_by_id(RECIPE_ID)['csv_dtype']['ALT_KEY'] == 'str'


@pytest.mark.parametrize('year', sorted(YEARS))
def test_every_year_writes_alt_key_as_string(tmp_path, year):
    source = tmp_path / f'{year}.csv'
    source.write_text(YEARS[year])
    df = _reader(get_recipe_by_id(RECIPE_ID))._read_flat_table(source, None, None)
    out = tmp_path / f'{year}.parquet'
    save_parquet(df, out)
    field = pq.read_schema(out).field('ALT_KEY')
    assert pa.types.is_string(field.type) or pa.types.is_large_string(field.type)
    if year == 'long':
        assert df['ALT_KEY'].str.len().tolist() == [21, 21]


def test_a_21_digit_key_read_as_number_cannot_be_written(tmp_path):
    source = tmp_path / 'long.csv'
    source.write_text(YEARS['long'])
    df = _reader({})._read_flat_table(source, None, None)
    with pytest.raises(OverflowError):
        save_parquet(df, tmp_path / 'long.parquet')
