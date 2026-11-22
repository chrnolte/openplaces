"""An empty download for a unit raises and is not kept as the cache.

A REST layer that answered a unit's filter with zero features wrote a
45-byte FeatureCollection; the file was cached and every later run
saved the unit's output without rows. An empty download is not an
observed "none", so the ingest refuses it unless the recipe declares
`empty_download_ok`.
"""

from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import Point

from openplaces.io.ingester import Ingester
from openplaces.io.ingester.download import EmptyDownloadError, count_source_rows

EMPTY_COLLECTION = '{"type": "FeatureCollection", "features": []}'


class _Source:
    download_url = None
    download_url_source = None
    download_url_scraper = 'arcgis_rest_scraper'
    portal_url = None
    verify_ssl = True


class _ManualSource(_Source):
    download_url_scraper = None


class _Entity:
    def __init__(self, source):
        self.source = source

    def __str__(self):
        return 'parcel-example-2026'


def _make_ingester(tmp_path, *, content, name='layer.geojson', extra=None, source=None):
    external = tmp_path / 'external'
    external.mkdir()
    data = external / name
    data.write_text(content, encoding='utf-8')
    ing = object.__new__(Ingester)
    ing.recipe = {
        'entity': _Entity(source or _Source()),
        'scraper_options': {
            'layer_url': 'https://gis.example.invalid/arcgis/rest/services/P/0'
        },
        **(extra or {}),
    }
    ing.verbose = False
    ing._recipe_id = lambda: 'US-ZZ_parcel-example-2026'
    ing.download_partition = {
        'download_url': None,
        'downloaded_path': data,
        'data_path': data,
        'admin_id_to_download': 'US-ZZ-AAA',
        'partition_id_to_download': None,
    }
    return ing, data


def test_empty_feature_collection_raises_and_is_deleted(tmp_path):
    ing, data = _make_ingester(tmp_path, content=EMPTY_COLLECTION)

    with pytest.raises(EmptyDownloadError) as err:
        ing._refuse_empty_download()

    message = str(err.value)
    assert 'US-ZZ_parcel-example-2026' in message
    assert 'US-ZZ-AAA' in message
    assert 'gis.example.invalid' in message
    assert 'layer.geojson' in message
    assert not data.exists()


def test_file_placed_by_hand_is_kept(tmp_path):
    ing, data = _make_ingester(
        tmp_path, content=EMPTY_COLLECTION, source=_ManualSource()
    )

    with pytest.raises(EmptyDownloadError, match='placed by hand'):
        ing._refuse_empty_download()

    assert data.exists()


def test_declared_empty_source_proceeds_with_one_line(tmp_path, capsys):
    ing, data = _make_ingester(
        tmp_path,
        content=EMPTY_COLLECTION,
        extra={'empty_download_ok': ['US-ZZ-AAA']},
    )

    ing._refuse_empty_download()

    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and 'empty_download_ok' in out[0]
    assert data.exists()


def test_declaration_for_another_unit_does_not_cover_this_one(tmp_path):
    ing, _ = _make_ingester(
        tmp_path,
        content=EMPTY_COLLECTION,
        extra={'empty_download_ok': ['US-ZZ-BBB']},
    )

    with pytest.raises(EmptyDownloadError):
        ing._refuse_empty_download()


def test_download_with_rows_passes(tmp_path):
    ing, data = _make_ingester(tmp_path, content='')
    gpd.GeoDataFrame(
        {'parcel_number': ['A-1']}, geometry=[Point(0, 0)], crs=4326
    ).to_file(data, driver='GeoJSON')

    ing._refuse_empty_download()

    assert data.exists()


def test_header_only_csv_is_empty(tmp_path):
    ing, data = _make_ingester(tmp_path, content='parcel_number,value\n', name='t.csv')

    with pytest.raises(EmptyDownloadError):
        ing._refuse_empty_download()
    assert not data.exists()


def test_row_counts_by_format(tmp_path):
    csv = tmp_path / 'rows.csv'
    csv.write_text('a,b\n1,2\n')
    fixed = tmp_path / 'rows.txt'
    fixed.write_text('0001ABC\n')
    blank = tmp_path / 'blank.txt'
    blank.write_text('\n\n')
    parquet = tmp_path / 'rows.parquet'
    gpd.GeoDataFrame({'a': []}, geometry=[], crs=4326).to_parquet(parquet)
    xlsx = tmp_path / 'sheet.xlsx'
    xlsx.write_bytes(b'not read')

    assert count_source_rows(csv) is None
    assert count_source_rows(fixed) is None
    assert count_source_rows(blank) == 0
    assert count_source_rows(parquet) == 0
    assert count_source_rows(xlsx) is None

    gpkg = tmp_path / 'layer.gpkg'
    frame = gpd.GeoDataFrame({'a': ['x']}, geometry=[Point(0, 0)], crs=4326)
    frame.to_file(gpkg, layer='rows', driver='GPKG')
    frame.iloc[:0].to_file(gpkg, layer='none', driver='GPKG')
    assert count_source_rows(gpkg, layer='rows') == 1
    assert count_source_rows(gpkg, layer='none') == 0


def test_partition_driver_refuses_before_processing(tmp_path, monkeypatch):
    """The per-partition driver checks the download before any read."""
    ing, _ = _make_ingester(tmp_path, content=EMPTY_COLLECTION)
    ing.download_partition = {}
    partition = {
        'download_url': None,
        'downloaded_path': tmp_path / 'external' / 'layer.geojson',
        'data_path': tmp_path / 'external' / 'layer.geojson',
    }

    monkeypatch.setattr(
        Ingester,
        '_resolve_download_url',
        lambda self: self.download_partition.update(partition),
    )
    monkeypatch.setattr(Ingester, '_resolve_downloaded_and_data_paths', lambda s: None)
    monkeypatch.setattr(
        Ingester, '_download_and_unzip_recipe_data', lambda s, redownload: None
    )
    processed = []
    monkeypatch.setattr(
        Ingester, '_process_recipe_data', lambda s, a=None: processed.append(a)
    )
    with pytest.raises(EmptyDownloadError):
        ing._ingest_download_partition(admin_id_to_download='US-ZZ-AAA')
    assert processed == []
