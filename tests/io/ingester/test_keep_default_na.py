"""A recipe may keep "NA" as a string: it is Namibia, not a missing value."""

import pandas as pd

from openplaces.io.ingester.table_ingester import TableIngester


def _read(tmp_path, recipe):
    path = tmp_path / 'units.csv'
    path.write_text('name,admin1_id\nErongo,NA\nKarasburg,\n', encoding='utf-8')
    ingester = TableIngester.__new__(TableIngester)
    ingester.recipe = recipe
    return TableIngester._read_flat_table(ingester, path, None, None)


def test_na_is_a_value_when_the_recipe_says_so(tmp_path):
    frame = _read(tmp_path, {'keep_default_na': False})
    assert frame['admin1_id'].tolist()[0] == 'NA'
    assert pd.isna(frame['admin1_id'].tolist()[1])


def test_the_default_still_reads_na_as_missing(tmp_path):
    frame = _read(tmp_path, {})
    assert pd.isna(frame['admin1_id'].tolist()[0])
