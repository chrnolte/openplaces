"""Merging a county's per-tile partials goes through the shared aggregation
core rather than a local concat that had diverged from it.

The local copy skipped `coerce_mixed_object_columns`, so a county straddling
two tiles whose partials typed one column differently (a source that is numeric
in one tile and text in another) produced a mixed object column that the
parquet write then refused. It also asked the tile link table one question per
(admin unit x tile), which is O(admins x tiles).
"""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from openplaces.io import ingester as ingester_module
from openplaces.io import read_parquet, save_parquet
from openplaces.io.ingester import Ingester
from openplaces.timing import Timer


def _partial(value, offset):
    return gpd.GeoDataFrame(
        {'building_style': [value], 'geo_id': [f'row{offset}']},
        geometry=[box(offset, 0, offset + 1, 1)],
        crs='EPSG:4326',
        index=pd.Index([f'row{offset}'], name='footprint_id'),
    )


@pytest.fixture
def tiled_ingester(tmp_path, monkeypatch):
    ingester = Ingester.__new__(Ingester)
    ingester.recipe = {}
    ingester.timer = Timer('test')
    ingester.verbose = False
    ingester._owns_timer = False
    ingester.admin_ids_to_save = ['XX-AA-AA']
    ingester.partition_ids_to_download = ['t1', 't2']
    ingester.tile_admin_link = pd.DataFrame(
        index=pd.MultiIndex.from_tuples(
            [('t1', 'XX-AA-AA'), ('t2', 'XX-AA-AA'), ('t3', 'XX-AA-BB')],
            names=['tile_id', 'admin_id'],
        )
    )

    paths = {
        None: tmp_path / 'XX-AA-AA_footprint.parquet',
        't1': tmp_path / 'XX-AA-AA_footprint_t1.parquet',
        't2': tmp_path / 'XX-AA-AA_footprint_t2.parquet',
    }
    monkeypatch.setattr(
        ingester_module,
        'get_output_path',
        lambda _recipe, _admin_id, partition_id=None, **kwargs: paths[partition_id],
    )
    return ingester, paths


def test_tiles_typed_differently_still_merge(tiled_ingester):
    ingester, paths = tiled_ingester
    save_parquet(_partial(2, 0), paths['t1'])
    save_parquet(_partial('two-story', 1), paths['t2'])

    ingester._merge_tile_partials()

    merged = read_parquet(paths[None], geom=True)
    assert sorted(merged['building_style'].astype(str)) == ['2', 'two-story']
    assert not paths['t1'].exists()
    assert not paths['t2'].exists()


def test_a_single_tile_county_is_renamed(tiled_ingester):
    ingester, paths = tiled_ingester
    save_parquet(_partial('two-story', 0), paths['t1'])

    ingester._merge_tile_partials()

    merged = read_parquet(paths[None], geom=True)
    assert merged['building_style'].tolist() == ['two-story']
    assert not paths['t1'].exists()


def test_the_link_table_is_read_once_per_run(tiled_ingester):
    """The admin-unit-to-tiles map is built once, not per (unit x tile)."""
    ingester, paths = tiled_ingester
    save_parquet(_partial('two-story', 0), paths['t1'])
    calls = []
    original_xs = ingester.tile_admin_link.xs

    def counted_xs(*args, **kwargs):
        calls.append(1)
        return original_xs(*args, **kwargs)

    ingester.tile_admin_link.xs = counted_xs
    ingester._merge_tile_partials()

    assert calls == []
