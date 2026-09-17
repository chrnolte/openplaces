"""`unzip` extracts archives nested inside a download.

Sources ship zips of zips: a county roll as one zip per table, or a file
host wrapping the roll's own zip in a second one. Extracting one level
left the inner archive in the heap, where no reader opens it. Every
archive here is fabricated.
"""

from __future__ import annotations

import io
import zipfile

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

import openplaces.io.ingester as ingester_module
from openplaces.io import unzip
from openplaces.io.ingester import Ingester

_TABLE = 'id,area\n"R001",1200\n"R002",950\n'


def _zip_bytes(members: dict[str, bytes | str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, data in members.items():
            z.writestr(name, data)
    return buffer.getvalue()


def _write_zip(path, members):
    path.write_bytes(_zip_bytes(members))
    return path


def test_nested_zip_is_extracted_and_its_table_read(tmp_path):
    inner = _zip_bytes({'Export123.txt': _TABLE})
    outer = _write_zip(
        tmp_path / 'roll.zip',
        {'Table - Property.zip': inner, 'readme.pdf': b'%PDF'},
    )

    out = unzip(outer, tmp_path / 'heap', verbose=False)

    assert (out / 'Export123.txt').exists()
    assert not (out / 'Table - Property.zip').exists()
    table = pd.read_csv(out / 'Export123.txt', dtype=str)
    assert table['id'].tolist() == ['R001', 'R002']


def test_doubly_nested_zip_is_extracted(tmp_path):
    innermost = _zip_bytes({'data.csv': _TABLE})
    middle = _zip_bytes({'wrapped/roll.zip': innermost})
    outer = _write_zip(tmp_path / 'download.zip', {'host.zip': middle})

    out = unzip(outer, tmp_path / 'heap', verbose=False)

    assert (out / 'data.csv').exists()
    assert list(out.rglob('*.zip')) == []


def test_members_select_inside_nested_archives(tmp_path):
    outer = _write_zip(
        tmp_path / 'roll.zip',
        {
            'Property.zip': _zip_bytes({'Export1.txt': _TABLE}),
            'Owner.zip': _zip_bytes({'Export2.txt': 'owner\n'}),
            'LAYOUT.xlsx': b'not read',
        },
    )

    out = unzip(outer, tmp_path / 'heap', members=['export1.TXT'], verbose=False)

    assert sorted(p.name for p in out.iterdir()) == ['Export1.txt']


def test_members_accept_glob_patterns(tmp_path):
    outer = _write_zip(
        tmp_path / 'roll.zip',
        {'X_APPRAISAL_INFO.TXT': 'a', 'X_ENTITY_INFO.TXT': 'b'},
    )

    out = unzip(outer, tmp_path / 'heap', members=['*_appraisal_*'], verbose=False)

    assert [p.name for p in out.iterdir()] == ['X_APPRAISAL_INFO.TXT']


def test_same_file_in_two_nested_archives_raises(tmp_path):
    outer = _write_zip(
        tmp_path / 'roll.zip',
        {
            'a.zip': _zip_bytes({'table.txt': 'first'}),
            'b.zip': _zip_bytes({'table.txt': 'second'}),
        },
    )

    with pytest.raises(ValueError, match='table.txt'):
        unzip(outer, tmp_path / 'heap', verbose=False)


def test_single_level_zip_is_unchanged(tmp_path):
    outer = _write_zip(
        tmp_path / 'parcels.zip',
        {'parcels/data.csv': _TABLE, 'parcels/notes/readme.txt': 'x'},
    )

    out = unzip(outer, tmp_path / 'heap', verbose=False)

    # The single top-level folder is still stripped.
    assert (out / 'data.csv').read_text() == _TABLE
    assert (out / 'notes' / 'readme.txt').exists()


def test_shapefile_in_zip_is_unchanged(tmp_path):
    shp_dir = tmp_path / 'shp'
    shp_dir.mkdir()
    gpd.GeoDataFrame(
        {'parcel_id': ['A1', 'A2']},
        geometry=[Point(0, 0), Point(1, 1)],
        crs='EPSG:4326',
    ).to_file(shp_dir / 'parcels.shp')
    members = {p.name: p.read_bytes() for p in shp_dir.iterdir()}
    # A KMZ is a zip container GDAL reads as it is: it must stay whole.
    members['overview.kmz'] = _zip_bytes({'doc.kml': '<kml/>'})
    outer = _write_zip(tmp_path / 'parcels.zip', members)

    out = unzip(outer, tmp_path / 'heap', verbose=False)

    assert sorted(p.name for p in out.iterdir()) == sorted(members)
    assert gpd.read_file(out / 'parcels.shp')['parcel_id'].tolist() == ['A1', 'A2']


class _Timer:
    def mark(self, *args, **kwargs):
        pass


class _Source:
    download_url = 'http://example.invalid/roll.zip'
    download_url_source = None
    download_url_scraper = None
    verify_ssl = True


class _Entity:
    source = _Source()


def test_ingester_finds_a_member_of_a_nested_archive(tmp_path, monkeypatch):
    external = tmp_path / 'external'
    heap = tmp_path / 'heap'
    external.mkdir()
    downloaded = _write_zip(
        external / 'roll.zip',
        {
            'wrapper/Table - Property.zip': _zip_bytes({'Export1.txt': _TABLE}),
            'wrapper/Table - Owner.zip': _zip_bytes({'Export2.txt': 'owner\n'}),
        },
    )

    ing = object.__new__(Ingester)
    ing.recipe = {'entity': _Entity(), 'extract_members': ['Export1.txt']}
    ing.verbose = False
    ing.timer = _Timer()
    ing.recipe_external_dir = external
    ing.recipe_heap_dir = heap
    ing.download_partition = {
        'download_url': _Source.download_url,
        'downloaded_path': downloaded,
        'data_path': heap / 'Export1.txt',
        'admin_id_to_download': None,
        'partition_id_to_download': None,
    }
    monkeypatch.setattr(
        ingester_module,
        'download',
        lambda *a, **k: pytest.fail('the download is already on disk'),
    )

    ing._download_and_unzip_recipe_data()

    assert ing.download_partition['data_path'] == heap / 'Export1.txt'
    assert sorted(p.name for p in heap.rglob('*') if p.is_file()) == ['Export1.txt']
