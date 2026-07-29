"""Unit tests for the table-ingester flat-file reader dispatch.

Covers the fixed-width reader (`_read_fixed_width`) added for sources like the
New Hanover County, NC parcel tax roll, and the headerless Excel branch of
`_read_recipe_data` used by the county sales spreadsheets. Both are network-free.
"""

import pandas as pd

from openplaces.io.ingester.table_ingester import TableIngester


def _make_ingester(recipe):
    """Build a bare TableIngester carrying only a recipe (no I/O setup)."""
    ingester = TableIngester.__new__(TableIngester)
    ingester.recipe = recipe
    return ingester


def test_read_fixed_width_drops_fillers_strips_and_keeps_leading_zeros(tmp_path):
    # Two records: [id(6), filler(3), name(10)]. The id is zero-padded and the
    # name is right-padded with spaces, so stripping and dtype=str both matter.
    data = '000123XXXAlice     \n004560YYYBob       \n'
    path = tmp_path / 'roll.txt'
    path.write_text(data)

    ingester = _make_ingester(
        {
            'fixed_width': [
                ['parcel_id', 6],
                ['filler', 3],
                ['name', 10],
            ]
        }
    )
    df = ingester._read_fixed_width(path, str)

    # Filler column dropped, only mapped fields remain.
    assert list(df.columns) == ['parcel_id', 'name']
    # Leading zeros preserved (read as text).
    assert df['parcel_id'].tolist() == ['000123', '004560']
    # Fixed-width padding stripped from values.
    assert df['name'].tolist() == ['Alice', 'Bob']


def test_read_fixed_width_dedupes_repeated_filler_names(tmp_path):
    # Multiple `filler` fields must not collide on a single column name.
    data = 'AbBcC\n'
    path = tmp_path / 'f.txt'
    path.write_text(data)

    ingester = _make_ingester(
        {
            'fixed_width': [
                ['a', 1],
                ['filler', 1],
                ['b', 1],
                ['filler', 1],
                ['c', 1],
            ]
        }
    )
    df = ingester._read_fixed_width(path, str)

    assert list(df.columns) == ['a', 'b', 'c']
    assert df.iloc[0].tolist() == ['A', 'B', 'C']


def test_read_excel_headerless_uses_positional_names(tmp_path):
    # The county sales spreadsheets have no header row; columns are named
    # positionally via `names` with `header: none`.
    src = pd.DataFrame([['001', 'sale', 150000], ['002', 'sale', 225000]])
    path = tmp_path / 'sales.xlsx'
    src.to_excel(path, header=False, index=False)

    df = pd.read_excel(
        path,
        sheet_name=0,
        header=None,
        names=['parid', 'kind', 'price'],
        dtype=None,
    )

    assert list(df.columns) == ['parid', 'kind', 'price']
    assert df['price'].tolist() == [150000, 225000]
