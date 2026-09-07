"""A spatially masked chunk drops the neighbors its bounding-box read picked up,
even when the recipe already carries the admin id column, and honors
`use_spatial_mask: false`.

`overlay_admin_ids` writes the admin id column in place. A recipe that maps or
derives that column itself therefore adds no new column, and the neighbor drop
used to be keyed on "columns the overlay added" - so it silently did nothing and
the neighbors were written into this unit's output.
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from openplaces.core.schema import AdminId
from openplaces.io.ingester import table_ingester as ti_module
from openplaces.io.ingester.table_ingester import TableIngester
from openplaces.timing import Timer


def _admin_geometries():
    """Two fabricated neighboring units, indexed like an admin layer."""
    return gpd.GeoSeries(
        [box(0, 0, 1, 1), box(1, 0, 2, 1)],
        index=pd.Index(['XX-AA-AA', 'XX-AA-BB'], name='admin3_id'),
        crs='EPSG:4326',
    )


def _bbox_read():
    """What a bbox read of unit AA returns: one of its own rows, one neighbor's."""
    return gpd.GeoDataFrame(
        {'admin3_id': ['XX-AA-AA', 'XX-AA-BB'], 'value': [1, 2]},
        geometry=[box(0.1, 0.1, 0.2, 0.2), box(1.1, 0.1, 1.2, 0.2)],
        crs='EPSG:4326',
        index=pd.Index(['a', 'b'], name='row_id'),
    )


def _ingester(recipe, admin_geometries, monkeypatch):
    def fake_overlay(gdf, admin_geometries, timer, **kwargs):
        # Stand in for the real sjoin: only rows inside the one unit the
        # mask passed in keep an id; the rest go null, in place.
        inside = gdf.geometry.centroid.within(admin_geometries.union_all())
        gdf[admin_geometries.index.name] = pd.Series(
            admin_geometries.index[0], index=gdf.index
        ).where(inside)
        return gdf

    monkeypatch.setattr(ti_module, 'overlay_admin_ids', fake_overlay)

    ingester = TableIngester.__new__(TableIngester)
    ingester.recipe = recipe
    ingester.download_partition = {'admin_geometries': admin_geometries.to_frame()}
    ingester.processing_chunk = {'admin_id_to_process': 'XX-AA-AA'}
    ingester.timer = Timer('test')
    ingester.verbose = False
    return ingester


def _recipe():
    return {
        'entity': None,
        'dataset': 'masked',
        'process_by': {'admin_level': 3, 'use_spatial_mask': True},
    }


def test_neighbors_are_dropped_when_the_recipe_already_maps_the_column(monkeypatch):
    admin = _admin_geometries()
    ingester = _ingester(_recipe(), admin, monkeypatch)
    ingester.download_partition['admin_geometries'] = admin.loc[['XX-AA-AA']]

    df = ingester._preprocess_recipe_data(_bbox_read())

    assert df['value'].tolist() == [1]


def test_use_spatial_mask_false_keeps_every_row(monkeypatch):
    admin = _admin_geometries()
    recipe = _recipe()
    recipe['process_by']['use_spatial_mask'] = False
    recipe['overlay_admin_ids'] = {'admin_level': 3}
    ingester = _ingester(recipe, admin, monkeypatch)
    ingester.download_partition['admin_geometries'] = admin.loc[['XX-AA-AA']]

    df = ingester._preprocess_recipe_data(_bbox_read())

    assert df['value'].tolist() == [1, 2]


def test_ingester_reads_no_bbox_when_the_mask_is_switched_off():
    """`use_spatial_mask: false` must not produce a bbox-clipped read.

    The Ingester tested key presence while the TableIngester tested
    truthiness, so an explicit `false` gave a bbox read whose out-of-unit
    rows nothing then dropped.
    """
    from openplaces.io.ingester import Ingester

    ingester = Ingester.__new__(Ingester)
    ingester.recipe = {
        'entity': None,
        'admin_id': AdminId('XX'),
        'process_by': {'admin_level': 3, 'use_spatial_mask': False},
    }
    ingester.download_partition = {
        'admin_id_to_download': 'XX-AA',
        'partition_id_to_download': None,
    }
    ingester.timer = Timer('test')
    ingester.verbose = False

    seen = {}

    class _Stub:
        def process(self, process_in_chunks=False, bbox=None):
            seen['bbox'] = bbox

    ingester._make_table_ingester = lambda _recipe: _Stub()
    ingester._process_recipe_data('XX-AA-AA')

    assert seen['bbox'] is None
