"""Two flat-file traps of Indiana's sales disclosure export.

The export quotes every field but never doubles a quote inside one, and
a field can hold the delimiter itself. `all_fields_quoted: true` reads
such a file by splitting on quote-delimiter-quote. A file declared
'utf-16' without its byte-order mark is read by the byte order its zero
bytes show, and one that is not UTF-16 at all raises naming the file.
All values are fabricated.
"""

from __future__ import annotations

import pandas as pd
import pytest

from openplaces.io.ingester.table_ingester import TableIngester
from openplaces.io.ingester.table_read import resolve_utf16_byte_order

# Row 2 holds unescaped quotes and a tab inside its note; row 3 holds an
# unescaped quote that makes pandas' default read shift a value.
LINES = [
    '"id"\t"note"\t"value"',
    '"A-1"\t"plain"\t"100"',
    '"A-2"\t""Lot 9" - see\tplat 7"\t"200"',
    '"A-3"\t"a "b c\t"\t"300"',
    '"A-4"\t""\t"400"',
]
TEXT = '\r\n'.join(LINES) + '\r\n'


def _reader(recipe):
    ti = TableIngester.__new__(TableIngester)
    ti.recipe = recipe
    return ti


def _write(tmp_path, text, encoding):
    path = tmp_path / 'sales.txt'
    path.write_bytes(text.encode(encoding))
    return path


def test_default_read_fails_on_the_unescaped_quote(tmp_path):
    path = _write(tmp_path, TEXT, 'utf-16')
    with pytest.raises(pd.errors.ParserError):
        _reader({'delimiter': '\t', 'csv_dtype': 'str'})._read_flat_table(
            path, None, 'utf-16'
        )


def test_all_fields_quoted_reads_every_row_whole(tmp_path, capsys):
    path = _write(tmp_path, TEXT, 'utf-16')
    df = _reader(
        {'delimiter': '\t', 'csv_dtype': 'str', 'all_fields_quoted': True}
    )._read_flat_table(path, None, 'utf-16')
    assert df['id'].tolist() == ['A-1', 'A-2', 'A-3', 'A-4']
    assert df['note'].tolist()[:3] == [
        'plain',
        '"Lot 9" - see\tplat 7',
        'a "b c\t',
    ]
    assert pd.isna(df['note'].iloc[3])
    assert df['value'].tolist() == ['100', '200', '300', '400']
    assert '2 line(s) hold a quote or delimiter' in capsys.readouterr().out


def test_all_fields_quoted_honors_usecols(tmp_path):
    path = _write(tmp_path, TEXT, 'utf-16')
    df = _reader(
        {'delimiter': '\t', 'csv_dtype': 'str', 'all_fields_quoted': True}
    )._read_flat_table(path, ['id', 'value'], 'utf-16')
    assert list(df.columns) == ['id', 'value']
    assert len(df) == 4


def test_all_fields_quoted_refuses_an_unquoted_line(tmp_path):
    path = _write(tmp_path, '"id"\t"value"\nB-1\t"5"\n', 'utf-8')
    with pytest.raises(ValueError, match='line 2'):
        _reader(
            {'delimiter': '\t', 'csv_dtype': 'str', 'all_fields_quoted': True}
        )._read_flat_table(path, None, None)


@pytest.mark.parametrize('codec', ['utf-16-le', 'utf-16-be'])
def test_utf16_without_mark_is_read_by_its_byte_order(tmp_path, codec):
    path = _write(tmp_path, TEXT, codec)
    assert resolve_utf16_byte_order(path, 'utf-16') == codec
    df = _reader(
        {'delimiter': '\t', 'csv_dtype': 'str', 'all_fields_quoted': True}
    )._read_flat_table(path, None, 'utf-16')
    assert df['id'].tolist() == ['A-1', 'A-2', 'A-3', 'A-4']


def test_utf16_with_mark_is_unchanged(tmp_path):
    path = _write(tmp_path, TEXT, 'utf-16')
    assert resolve_utf16_byte_order(path, 'utf-16') == 'utf-16'
    assert resolve_utf16_byte_order(path, 'utf-8') == 'utf-8'
    assert resolve_utf16_byte_order(path, None) is None


def test_a_file_declared_utf16_that_is_not_raises_naming_it(tmp_path):
    path = _write(tmp_path, TEXT, 'utf-8')
    with pytest.raises(UnicodeError, match='sales.txt'):
        _reader({'delimiter': '\t', 'csv_dtype': 'str'})._read_flat_table(
            path, None, 'utf-16'
        )


@pytest.mark.parametrize(
    ('written', 'expected'), [('utf-8', 'utf-8'), ('cp1252', 'cp1252')]
)
def test_an_8bit_file_among_utf16_ones_reads_with_a_fallback(
    tmp_path, written, expected
):
    # One vintage of a UTF-16 source exported as 8-bit text. 'é' is two
    # bytes in UTF-8 and one invalid-as-UTF-8 byte in cp1252.
    text = TEXT.replace('plain', 'café')
    path = _write(tmp_path, text, written)
    assert resolve_utf16_byte_order(path, 'utf-16', 'cp1252') == expected
    df = _reader(
        {
            'delimiter': '\t',
            'csv_dtype': 'str',
            'all_fields_quoted': True,
            'encoding_fallback': 'cp1252',
        }
    )._read_flat_table(path, None, 'utf-16')
    assert df['note'].iloc[0] == 'café'
    assert df['id'].tolist() == ['A-1', 'A-2', 'A-3', 'A-4']
