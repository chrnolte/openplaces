"""The ingester reads Microsoft Access databases, one named table at a time.

Values are fabricated. The round-trip test builds its own database with
Jackcess and is skipped where no Java runtime is installed.
"""

from __future__ import annotations

import faulthandler

import pandas as pd
import pytest

from openplaces.core.constants import ACCESS_EXTENSIONS
from openplaces.io import access
from openplaces.io.ingester.table_ingester import TableIngester


def _reader(recipe):
    ti = TableIngester.__new__(TableIngester)
    ti.recipe = recipe
    return ti


def test_access_extensions_are_known():
    assert {'.mdb', '.accdb'} <= ACCESS_EXTENSIONS


def test_the_flat_table_reader_routes_access_files(tmp_path, monkeypatch):
    calls = {}

    def fake(path, table, columns=None):
        calls.update(path=path, table=table, columns=columns)
        return pd.DataFrame({'AssrNo': ['0001'], 'BedRms': [3]})

    monkeypatch.setattr('openplaces.io.ingester.table_ingester.read_access_table', fake)
    path = tmp_path / 'Fabricated.MDB'
    path.write_bytes(b'')

    df = _reader({'layer': 'RESIDENTIAL'})._read_flat_table(
        path, ['AssrNo', 'BedRms'], None
    )

    assert calls['table'] == 'RESIDENTIAL'
    assert calls['columns'] == ['AssrNo', 'BedRms']
    assert df['BedRms'].tolist() == [3]


def test_text_dtype_is_applied_when_asked(tmp_path, monkeypatch):
    monkeypatch.setattr(
        'openplaces.io.ingester.table_ingester.read_access_table',
        lambda path, table, columns=None: pd.DataFrame({'AssrNo': [1]}),
    )
    path = tmp_path / 'Fabricated.accdb'
    path.write_bytes(b'')

    df = _reader({'layer': 'GENERAL', 'csv_dtype': 'str'})._read_flat_table(
        path, None, None
    )

    assert df['AssrNo'].tolist() == ['1']


def test_a_table_name_is_required(tmp_path):
    with pytest.raises(ValueError, match='layer'):
        access.read_access_table(tmp_path / 'Fabricated.mdb', None)


def test_without_a_route_the_error_names_both(tmp_path, monkeypatch):
    monkeypatch.setattr(access, '_jvm_available', lambda: False)
    monkeypatch.setattr(access, '_odbc_available', lambda: False)

    with pytest.raises(access.AccessReaderUnavailableError) as raised:
        access.read_access_table(tmp_path / 'Fabricated.mdb', 'GENERAL')

    assert 'jpype1 openjdk' in str(raised.value)
    assert 'ODBC' in str(raised.value)


def test_a_jar_that_does_not_match_its_hash_is_refused(tmp_path, monkeypatch):
    jar = tmp_path / 'jackcess.jar'
    monkeypatch.setattr(access, '_jar_path', lambda: jar)

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            yield b'not the jar'

    monkeypatch.setattr(access.requests, 'get', lambda *args, **kwargs: Response())

    with pytest.raises(OSError, match='sha256'):
        access.jackcess_jar()
    assert not jar.exists()
    assert not list(tmp_path.glob('.jackcess.*'))


def test_a_round_trip_through_jackcess(tmp_path):
    """Build a small database with Jackcess, then read it back."""
    jpype = pytest.importorskip('jpype')
    if not access._jvm_available():
        pytest.skip('no Java runtime')
    # On Windows the JVM raises and handles access violations of its own
    # while starting; pytest's faulthandler reports them as a fatal
    # exception although nothing failed.
    was_enabled = faulthandler.is_enabled()
    faulthandler.disable()
    try:
        access._start_jvm()
    finally:
        if was_enabled:
            faulthandler.enable()
    jclass = jpype.JClass
    builder = jclass('com.healthmarketscience.jackcess.DatabaseBuilder')
    file_format = jclass('com.healthmarketscience.jackcess.Database$FileFormat')
    table_builder = jclass('com.healthmarketscience.jackcess.TableBuilder')
    column_builder = jclass('com.healthmarketscience.jackcess.ColumnBuilder')
    data_type = jclass('com.healthmarketscience.jackcess.DataType')
    big_decimal = jclass('java.math.BigDecimal')

    path = tmp_path / 'Fabricated.accdb'
    database = builder.create(file_format.V2010, jclass('java.io.File')(str(path)))
    table = (
        table_builder('RESIDENTIAL')
        .addColumn(column_builder('AssrNo', data_type.TEXT))
        .addColumn(column_builder('BedRms', data_type.LONG))
        .addColumn(
            column_builder('Stories', data_type.NUMERIC).setPrecision(5).setScale(2)
        )
        .toTable(database)
    )
    table.addRow('0001', 3, big_decimal('1.50'))
    table.addRow('0002', None, big_decimal('2.00'))
    database.close()

    df = access.read_access_table(path, 'RESIDENTIAL', ['AssrNo', 'Stories'])

    assert df['AssrNo'].tolist() == ['0001', '0002']
    assert df['Stories'].tolist() == [1.5, 2.0]
    with pytest.raises(KeyError, match='BATHS'):
        access.read_access_table(path, 'RESIDENTIAL', ['BATHS'])
