"""The flat-table reader honors a recipe's opt-in `on_bad_lines`.

A delimited source with one malformed line (a stray delimiter inside a
text field) stops pandas' default read. A recipe that knows this sets
`on_bad_lines`, which the reader passes to `read_csv`; without the key
the read still fails, so no row is ever dropped unasked. Values are
fabricated.
"""

from __future__ import annotations

import pandas as pd
import pytest

from openplaces.io.ingester.table_ingester import TableIngester

ROWS = 'acct\tclass\n0001\tA1\n0002\tA1\textra\n0003\tB2\n'


def _reader(recipe):
    ti = TableIngester.__new__(TableIngester)
    ti.recipe = recipe
    return ti


def test_a_malformed_line_fails_without_the_key(tmp_path):
    path = tmp_path / 'roll.txt'
    path.write_text(ROWS)
    with pytest.raises(pd.errors.ParserError):
        _reader({'delimiter': '\t', 'csv_dtype': 'str'})._read_flat_table(
            path, None, None
        )


def test_on_bad_lines_skip_drops_only_the_malformed_line(tmp_path):
    path = tmp_path / 'roll.txt'
    path.write_text(ROWS)
    df = _reader(
        {'delimiter': '\t', 'csv_dtype': 'str', 'on_bad_lines': 'skip'}
    )._read_flat_table(path, None, None)
    assert df['acct'].tolist() == ['0001', '0003']
