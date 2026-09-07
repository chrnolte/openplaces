"""`overlay_admin_ids` must survive a centroid on a shared admin boundary.

Admin units tile space, so a centroid landing exactly on the line between
two of them matches both. The left join then returns more rows than the
input frame, and the overlay fallback returns more than one piece per
entity.
"""

from __future__ import annotations

import geopandas as gpd
from shapely.geometry import box

from openplaces.geo.overlay import overlay_admin_ids


def _admin(geometries, ids):
    return gpd.GeoSeries(geometries, index=ids, crs='epsg:4326').rename_axis(
        'admin2_id'
    )


def _entities(geometries, ids):
    return gpd.GeoDataFrame(
        {'footprint_id': ids, 'geometry': geometries}, crs='epsg:4326'
    ).set_index('footprint_id')


def test_centroid_on_a_shared_boundary_gets_one_admin_id():
    admin = _admin([box(0, 0, 1, 1), box(1, 0, 2, 1)], ['US-XX-A', 'US-XX-B'])
    # Centroid is exactly (1.0, 0.5), on the line between the two units.
    gdf = _entities([box(0.5, 0, 1.5, 1)], ['f1'])

    result = overlay_admin_ids(gdf, admin_geometries=admin)

    assert len(result) == 1
    assert result['admin2_id'].iloc[0] in {'US-XX-A', 'US-XX-B'}


def test_overlay_fallback_keeps_the_largest_piece():
    # A gap between the two units, so the centroid matches neither and
    # the include_overlays branch has to resolve the polygon.
    admin = _admin([box(0, 0, 1, 1), box(2, 0, 3, 1)], ['US-XX-A', 'US-XX-B'])
    gdf = _entities([box(0.2, 0, 2.5, 1)], ['f1'])

    result = overlay_admin_ids(gdf, admin_geometries=admin, include_overlays=True)

    assert len(result) == 1
    # 0.8 of the polygon's width lies in A against 0.5 in B.
    assert result['admin2_id'].iloc[0] == 'US-XX-A'
