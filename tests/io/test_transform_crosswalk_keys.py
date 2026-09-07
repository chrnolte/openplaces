"""Key dtypes when a crosswalk table is read.

A county or class code is text with meaningful leading zeros. Read as a
number, '037' becomes 37, matches nothing in the column it maps, and the
all-NaN result raises nothing.
"""

import pandas as pd

from openplaces.io.transform import _apply_remap_file, _read_crosswalk_table


def _write_csv(path, rows):
    path.write_text('code,value\n' + '\n'.join(rows) + '\n', encoding='utf-8')
    return path


def test_zero_padded_keys_survive_as_text(tmp_path):
    path = _write_csv(tmp_path / 'remap.csv', ['037,Alleghany', '119,Mecklenburg'])

    result = _apply_remap_file(pd.Series(['037', '119']), str(path))

    assert result.tolist() == ['Alleghany', 'Mecklenburg']


def test_plain_numeric_keys_keep_their_numeric_values(tmp_path):
    # No leading zeros: the table is left exactly as pandas typed it, so a
    # crosswalk carrying numbers still returns numbers.
    path = _write_csv(tmp_path / 'remap.csv', ['37,1500', '119,2400'])

    table = _read_crosswalk_table(lambda dtype: pd.read_csv(path, dtype=dtype))

    assert table['code'].tolist() == [37, 119]
    assert table['value'].tolist() == [1500, 2400]
