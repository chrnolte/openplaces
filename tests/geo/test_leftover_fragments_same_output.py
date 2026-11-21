"""The identity overlay's leftover pieces are the ones pandas' union per
group gave.

plans/slow-steps-same-output-faster.md, R3: `_leftover_fragments`
unions each polygon's overlapping others through `table.reduce_groups`
instead of ``groupby(...).agg(shapely.union_all)``. The previous
function is kept below; both run on the same fabricated polygons and
their ids and geometries (as WKB) are compared exactly, as is the
whole identity overlay.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely

from openplaces.geo import polygon


def _old_leftover_fragments(
    gdf_self, self_idx, gdf_other, other_idx, partial_ids, piece_intersection
):
    partial_self = gdf_self[gdf_self[self_idx].isin(partial_ids)].set_index(self_idx)
    other_geom = gdf_other.set_index(other_idx)[['geometry']]
    pairs = (
        piece_intersection[piece_intersection[self_idx].isin(partial_ids)][
            [self_idx, other_idx]
        ]
        .copy()
        .join(other_geom.rename(columns={'geometry': '_og'}), on=other_idx)
    )
    other_union = pairs.groupby(self_idx)['_og'].agg(shapely.union_all)
    leftover = shapely.difference(
        partial_self.geometry.reindex(other_union.index).values,
        other_union.values,
    )
    keep = ~shapely.is_empty(leftover)
    return other_union.index[keep], leftover[keep]


def _layers(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 400, n)
    y = rng.uniform(0, 400, n)
    w = rng.uniform(5, 30, n)
    left = gpd.GeoDataFrame(
        {'footprint_id': [f'f{i:04d}' for i in range(n)]},
        geometry=shapely.box(x, y, x + w, y + w * rng.uniform(0.5, 2, n)),
        crs='EPSG:32617',
    )
    m = n // 2
    x2 = rng.uniform(0, 400, m)
    y2 = rng.uniform(0, 400, m)
    right = gpd.GeoDataFrame(
        {'parcel_id': [f'p{i:04d}' for i in range(m)]},
        geometry=shapely.box(x2, y2, x2 + 40, y2 + 25),
        crs='EPSG:32617',
    )
    return left, right


@pytest.mark.parametrize('seed', [1, 2, 3])
def test_leftovers_match(seed):
    left, right = _layers(600, seed)
    pieces = gpd.overlay(left, right, how='intersection', keep_geom_type=False)
    partial = set(pieces['footprint_id'])
    args = (left, 'footprint_id', right, 'parcel_id', partial, pieces)
    new_ids, new_geoms = polygon._leftover_fragments(*args)
    old_ids, old_geoms = _old_leftover_fragments(*args)
    pd.testing.assert_index_equal(new_ids, old_ids, exact=True)
    assert list(shapely.to_wkb(new_geoms)) == list(shapely.to_wkb(old_geoms))


def test_identity_overlay_matches(monkeypatch):
    left, right = _layers(800, 9)
    left = left.set_index('footprint_id')
    right = right.set_index('parcel_id')
    new = polygon.overlay_polygons(left, right, how='identity', iou=True, geom=True)
    monkeypatch.setattr(polygon, '_leftover_fragments', _old_leftover_fragments)
    old = polygon.overlay_polygons(left, right, how='identity', iou=True, geom=True)
    assert list(shapely.to_wkb(new.geometry.values)) == list(
        shapely.to_wkb(old.geometry.values)
    )
    pd.testing.assert_frame_equal(
        pd.DataFrame(new.drop(columns='geometry')),
        pd.DataFrame(old.drop(columns='geometry')),
        check_exact=True,
        check_dtype=True,
    )


def test_no_partial_polygon():
    left, right = _layers(50, 4)
    empty = gpd.overlay(left, right, how='intersection').iloc[:0]
    args = (left, 'footprint_id', right, 'parcel_id', set(), empty)
    new_ids, new_geoms = polygon._leftover_fragments(*args)
    old_ids, old_geoms = _old_leftover_fragments(*args)
    pd.testing.assert_index_equal(new_ids, old_ids, exact=True)
    assert len(new_geoms) == len(old_geoms) == 0
