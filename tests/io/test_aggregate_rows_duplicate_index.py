"""aggregate_rows must not multiply rows whose index label repeats.

Its geometry-area ordering used to re-select rows by label, which turns
one repeated label into every combination of its occurrences, so the
groupby that follows aggregated an inflated input. The ordering is now
positional.
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from openplaces.table import aggregate_rows


def _first(series):
    return series.iloc[0]


def _duplicate_label_parcels():
    return gpd.GeoDataFrame(
        {
            'land_value': [100.0, 200.0, 400.0],
            'group': ['g', 'g', 'g'],
            'geometry': [box(0, 0, 1, 1), box(1, 0, 3, 1), box(3, 0, 4, 1)],
        },
        index=pd.Index(['a', 'a', 'b'], name='parcel_id'),
        crs='epsg:6933',
    )


def test_geometry_sorted_aggregation_does_not_inflate_the_sum():
    out = aggregate_rows(_duplicate_label_parcels(), by='group')
    assert out['land_value'].tolist() == [700.0]


def test_geometry_sorting_still_orders_by_descending_area():
    out = aggregate_rows(
        _duplicate_label_parcels(), by='group', aggregation_function=_first
    )
    # The two-unit polygon is the largest, so its value leads the group.
    assert out['land_value'].tolist() == [200.0]


def test_unique_index_result_is_unchanged_by_the_positional_ordering():
    parcels = _duplicate_label_parcels()
    parcels.index = pd.Index(['a', 'b', 'c'], name='parcel_id')
    out = aggregate_rows(parcels, by='group', aggregation_function=_first)
    assert out['land_value'].tolist() == [200.0]


def test_explicit_sort_column_path_is_unaffected():
    out = aggregate_rows(
        _duplicate_label_parcels(),
        by='group',
        sort_by='land_value',
        aggregation_function=_first,
    )
    assert out['land_value'].tolist() == [400.0]
