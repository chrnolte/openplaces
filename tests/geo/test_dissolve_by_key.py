"""`dissolve_by_key` folds the polygon rows of one source record.

A layer that draws one acquisition as several polygons repeats its price
on each; the record is one row with the union of its polygons, and a
value survives only where every polygon of the record states it.
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from openplaces.geo.polygon import dissolve_by_key


def _frame():
    return gpd.GeoDataFrame(
        {
            'deal': ['D1', 'D2', 'D1', None, None],
            'price': [100.0, 50.0, 100.0, 7.0, 8.0],
            'own_area': [1.0, 2.0, 3.0, 4.0, 5.0],
            'note': ['x', 'y', None, 'z', 'w'],
        },
        geometry=[
            box(0, 0, 1, 1),
            box(5, 5, 6, 6),
            box(1, 0, 2, 1),
            box(8, 8, 9, 9),
            box(10, 10, 11, 11),
        ],
        crs='EPSG:3857',
        index=pd.Index([10, 11, 12, 13, 14], name='fid'),
    )


def test_one_row_per_key_with_the_union_of_its_polygons():
    out, n_rows, n_records, _ = dissolve_by_key(_frame(), 'deal')

    assert (n_rows, n_records) == (2, 1)
    assert out.index.tolist() == [10, 11, 13, 14]
    d1 = out.loc[10]
    assert d1['deal'] == 'D1'
    assert d1.geometry.area == 2.0
    assert out.crs == 'EPSG:3857'


def test_a_repeated_price_is_kept_once_and_never_summed():
    out, *_ = dissolve_by_key(_frame(), 'deal')

    assert out.loc[10, 'price'] == 100.0
    assert out['price'].sum() == 100.0 + 50.0 + 7.0 + 8.0


def test_a_value_the_rows_disagree_on_is_left_missing():
    out, _, _, disagree = dissolve_by_key(_frame(), 'deal')

    assert pd.isna(out.loc[10, 'own_area'])
    assert disagree['own_area'] == 1
    assert disagree['price'] == 0


def test_a_missing_value_is_not_filled_from_another_row_of_the_key():
    # One polygon of D1 states a note and the other none: the note is
    # not taken for the record (design limit L25).
    out, _, _, disagree = dissolve_by_key(_frame(), 'deal')

    assert pd.isna(out.loc[10, 'note'])
    assert disagree['note'] == 1


def test_rows_without_a_key_or_alone_on_theirs_pass_through():
    frame = _frame()
    out, *_ = dissolve_by_key(frame, 'deal')

    for label in (11, 13, 14):
        assert out.loc[label].equals(frame.loc[label])


def test_nothing_to_fold_returns_the_frame_unchanged():
    frame = _frame().iloc[[1, 3]]
    out, n_rows, n_records, disagree = dissolve_by_key(frame, 'deal')

    assert out is frame
    assert (n_rows, n_records) == (0, 0)
    assert disagree.empty
