"""An empty source the download check cannot count raises at its read.

The check before any read counts rows only where that is cheap and the
partition is one file. A spreadsheet, an Access table, a large GeoJSON,
one unit's `process_by.file_pattern` file or `layer_pattern` layer, and
a `read_members` or `join_members` set are checked on the primary
table's first unfiltered read instead, from the frame read anyway.
Every file here is fabricated by the test.
"""

from __future__ import annotations

import json

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from openplaces.io.ingester import Ingester
from openplaces.io.ingester.download import EmptyDownloadError
from openplaces.io.ingester.table_ingester import TableIngester

RECIPE_ID = 'US-ZZ_parcel-example-2026'


class _Source:
    download_url = 'https://data.example.invalid/parcels.zip'
    download_url_source = None
    download_url_scraper = None
    portal_url = None
    verify_ssl = True


class _ManualSource(_Source):
    download_url = None


class _Entity:
    def __init__(self, source):
        self.source = source

    def __str__(self):
        return 'parcel-example-2026'


class _Timer:
    def mark(self, *args, **kwargs):
        pass


def _table_ingester(
    tmp_path, data_path, *, extra=None, source=None, primary=True, partition=None
):
    recipe = {
        'admin_id': 'US-ZZ',
        'entity': _Entity(source or _Source()),
        **(extra or {}),
    }
    download_partition = {
        'download_url': _Source.download_url,
        'downloaded_path': data_path,
        'data_path': data_path,
        'admin_id_to_download': 'US-ZZ',
        'partition_id_to_download': None,
        **(partition or {}),
    }
    table = TableIngester(
        recipe,
        download_partition,
        {'admin_id_to_process': 'US-ZZ-AAA'},
        tmp_path,
        _Timer(),
    )
    table.refuse_empty_read = primary
    return table


def _write_xlsx(path, rows):
    pd.DataFrame(rows, columns=['parcel_number', 'value']).to_excel(path, index=False)


def test_header_only_spreadsheet_raises_and_is_deleted(tmp_path):
    data = tmp_path / 'roll.xlsx'
    _write_xlsx(data, [])
    table = _table_ingester(tmp_path, data)

    with pytest.raises(EmptyDownloadError) as err:
        table._read_recipe_data()

    message = str(err.value)
    assert RECIPE_ID in message
    assert 'US-ZZ' in message
    assert 'roll.xlsx' in message
    assert 'data.example.invalid' in message
    assert 'empty_download_ok' in message
    assert not data.exists()


def test_spreadsheet_with_rows_passes(tmp_path):
    data = tmp_path / 'roll.xlsx'
    _write_xlsx(data, [['A-1', 100]])
    table = _table_ingester(tmp_path, data)

    assert len(table._read_recipe_data()) == 1
    assert data.exists()


def test_declared_empty_spreadsheet_proceeds_with_one_line(tmp_path, capsys):
    data = tmp_path / 'roll.xlsx'
    _write_xlsx(data, [])
    table = _table_ingester(tmp_path, data, extra={'empty_download_ok': ['US-ZZ']})

    assert len(table._read_recipe_data()) == 0

    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and 'empty_download_ok' in out[0]
    assert data.exists()


def test_empty_access_table_raises_naming_the_table(tmp_path, monkeypatch):
    data = tmp_path / 'roll.mdb'
    data.write_bytes(b'fabricated, never parsed')
    monkeypatch.setattr(
        'openplaces.io.ingester.table_read.read_access_table',
        lambda path, layer, columns: pd.DataFrame(columns=['parcel_number']),
    )
    table = _table_ingester(
        tmp_path, data, extra={'layer': 'Parcels'}, source=_ManualSource()
    )

    with pytest.raises(EmptyDownloadError, match='placed by hand') as err:
        table._read_recipe_data()

    assert 'layer Parcels' in str(err.value)
    assert data.exists()


def test_large_geojson_without_features_raises(tmp_path):
    data = tmp_path / 'layer.geojson'
    # Over the 64 KB the download check parses; padded by a member
    # GeoJSON readers ignore.
    payload = {
        'type': 'FeatureCollection',
        'name': 'layer',
        'description': 'x' * 70_000,
        'features': [],
    }
    data.write_text(json.dumps(payload), encoding='utf-8')
    table = _table_ingester(tmp_path, data)

    with pytest.raises(EmptyDownloadError, match='layer.geojson'):
        table._read_recipe_data()

    assert not data.exists()


def test_file_pattern_unit_file_raises_and_is_kept(tmp_path):
    shared = tmp_path / 'statewide.csv'
    shared.write_text('parcel_number\nA-1\n')
    unit_file = tmp_path / 'county_AAA.csv'
    unit_file.write_text('parcel_number\n')
    table = _table_ingester(
        tmp_path,
        shared,
        extra={'process_by': {'file_pattern': 'county_{code}.csv'}},
    )

    with pytest.raises(EmptyDownloadError) as err:
        table._read_recipe_data(data_path_override=unit_file)

    message = str(err.value)
    assert 'US-ZZ-AAA' in message
    assert 'county_AAA.csv' in message
    assert 'other units read' in message
    assert unit_file.exists() and shared.exists()


def test_file_pattern_unit_may_be_declared_by_itself(tmp_path, capsys):
    unit_file = tmp_path / 'county_AAA.csv'
    unit_file.write_text('parcel_number\n')
    table = _table_ingester(
        tmp_path, unit_file, extra={'empty_download_ok': ['US-ZZ-AAA']}
    )

    table._read_recipe_data(data_path_override=unit_file)

    assert 'accepted' in capsys.readouterr().out


def test_layer_pattern_empty_layer_raises(tmp_path):
    data = tmp_path / 'counties.gpkg'
    frame = gpd.GeoDataFrame({'a': ['x']}, geometry=[Point(0, 0)], crs=4326)
    frame.to_file(data, layer='ParcelsAAA', driver='GPKG')
    frame.iloc[:0].to_file(data, layer='ParcelsBBB', driver='GPKG')
    table = _table_ingester(tmp_path, data)

    assert len(table._read_recipe_data(layer_override='ParcelsAAA')) == 1
    with pytest.raises(EmptyDownloadError) as err:
        table._read_recipe_data(layer_override='ParcelsBBB')

    assert 'layer ParcelsBBB' in str(err.value)
    assert 'counties.gpkg' in str(err.value)
    assert data.exists()


def test_read_members_one_blank_member_is_legal(tmp_path):
    (tmp_path / 'm1.csv').write_text('')
    (tmp_path / 'm2.csv').write_text('parcel_number\nA-1\n')
    table = _table_ingester(
        tmp_path,
        tmp_path / 'm1.csv',
        partition={'read_members_glob': 'm*.csv'},
    )

    assert list(table._read_recipe_data()['parcel_number']) == ['A-1']


@pytest.mark.parametrize('content', ['', 'parcel_number\n'])
def test_read_members_all_empty_raise(tmp_path, content):
    for name in ('m1.csv', 'm2.csv'):
        (tmp_path / name).write_text(content)
    table = _table_ingester(
        tmp_path,
        tmp_path / 'm1.csv',
        partition={'read_members_glob': 'm*.csv'},
    )

    with pytest.raises(EmptyDownloadError, match='members matching m\\*.csv'):
        table._read_recipe_data()


def _join_partition(tmp_path, geo, counts):
    (tmp_path / 'geo.txt').write_text(geo)
    (tmp_path / 'counts.txt').write_text(counts)
    return {
        'join_members': {
            'key': 'record',
            'members': [
                (tmp_path / 'geo.txt', {0: 'record', 1: 'name'}),
                (tmp_path / 'counts.txt', {0: 'record', 1: 'total'}),
            ],
        }
    }


def test_join_members_all_blank_raise(tmp_path):
    table = _table_ingester(
        tmp_path, tmp_path / 'geo.txt', partition=_join_partition(tmp_path, '', '')
    )

    with pytest.raises(EmptyDownloadError, match='joined members geo.txt'):
        table._read_recipe_data()


def test_join_members_one_blank_member_is_named_by_the_join(tmp_path):
    partition = _join_partition(tmp_path, '0001,Area\n', '')
    table = _table_ingester(tmp_path, tmp_path / 'geo.txt', partition=partition)

    with pytest.raises(ValueError, match='not in every member'):
        table._read_recipe_data()


def test_join_members_with_rows_pass(tmp_path):
    partition = _join_partition(tmp_path, '0001,Area\n', '0001,5\n')
    table = _table_ingester(tmp_path, tmp_path / 'geo.txt', partition=partition)

    assert len(table._read_recipe_data()) == 1


def test_secondary_table_is_not_checked(tmp_path):
    data = tmp_path / 'roll.xlsx'
    _write_xlsx(data, [])
    table = _table_ingester(tmp_path, data, primary=False)

    assert len(table._read_recipe_data()) == 0
    assert data.exists()


def test_filtered_read_is_not_checked(tmp_path):
    data = tmp_path / 'layer.gpkg'
    gpd.GeoDataFrame({'a': ['x']}, geometry=[Point(0, 0)], crs=4326).to_file(
        data, driver='GPKG'
    )
    table = _table_ingester(tmp_path, data)

    out = table._read_recipe_data(bbox=(10, 10, 11, 11))

    assert len(out) == 0
    assert data.exists()


def test_file_counted_before_the_read_is_not_reported_again(tmp_path, capsys):
    data = tmp_path / 'roll.csv'
    data.write_text('parcel_number\n')
    table = _table_ingester(
        tmp_path,
        data,
        extra={'empty_download_ok': True},
        partition={'rows_counted': (str(data), None)},
    )

    table._read_recipe_data()

    assert capsys.readouterr().out == ''


def test_only_the_primary_table_is_marked(tmp_path, monkeypatch):
    ing = object.__new__(Ingester)
    ing.recipe = {'admin_id': 'US-ZZ', 'entity': _Entity(_Source())}
    ing.verbose = False
    ing.download_partition = {'admin_id_to_download': None}
    ing.recipe_heap_dir = tmp_path
    ing.admin_ids_to_save = []

    class _Recorder:
        def mark(self, *args, **kwargs):
            pass

    ing.timer = _Recorder()
    marked = []
    monkeypatch.setattr(
        TableIngester,
        'process',
        lambda self, **kwargs: marked.append(self.refuse_empty_read),
    )

    ing._process_recipe_data()

    assert marked == [True]
    assert TableIngester.refuse_empty_read is False
