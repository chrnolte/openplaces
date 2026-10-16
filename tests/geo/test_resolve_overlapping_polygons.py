"""`resolve_overlapping_polygons` must reject an unusable `prefer_higher`.

A recipe sets `keep_overlapping_polygons: {prefer_higher: <column>}` and
the ingester passes it through untouched, so a misspelled column name
used to invert the recipe's intent (the area tiebreak ran instead) while
the run still reported success.
"""

from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import box

from openplaces.geo.polygon import resolve_overlapping_polygons


def _overlapping():
    """Two near-identical polygons differing in one attribute."""
    return gpd.GeoDataFrame(
        {
            'parcel_id': ['p1', 'p2'],
            'year_built': [1990, 2010],
            'geometry': [box(0, 0, 1, 1), box(0, 0, 1.01, 1)],
        },
        crs='epsg:4326',
    ).set_index('parcel_id')


def test_misspelled_prefer_higher_column_raises():
    gdf = _overlapping()

    with pytest.raises(ValueError, match='year_buillt'):
        resolve_overlapping_polygons(
            gdf, iou_threshold=0.5, keep={'prefer_higher': 'year_buillt'}
        )


@pytest.mark.parametrize('large_first', [True, False])
def test_identical_nested_pair_keeps_the_larger_polygon(large_first):
    # A small roof structure nested in a large building, both with the
    # same attributes: the pair counts as identical, and the large
    # building must survive whichever side of the pair it is on.
    large = box(0, 0, 0.001, 0.001)
    small = box(0.0004, 0.0004, 0.00045, 0.00045)
    geoms = [large, small] if large_first else [small, large]
    gdf = gpd.GeoDataFrame(
        {'fid': ['a', 'b'], 'geometry_source': ['massgis', 'massgis']},
        geometry=geoms,
        crs='epsg:4326',
    ).set_index('fid')

    result = resolve_overlapping_polygons(gdf, keep=False)

    assert len(result) == 1
    assert result.geometry.iloc[0].equals(large)


def test_prefer_higher_keeps_the_higher_value():
    gdf = _overlapping()

    result = resolve_overlapping_polygons(
        gdf, iou_threshold=0.5, keep={'prefer_higher': 'year_built'}
    )

    assert list(result['year_built']) == [2010]
