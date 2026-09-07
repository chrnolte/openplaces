"""The shared duplicate-index assertion names the labels it refuses on.

Every stage entry point and every repaired label-keyed site routes its
refusal through `require_unique_index`, so its message is the one a user
reads when a duplicated entity id reaches the pipeline.
"""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from openplaces.table import require_unique_index


def test_unique_index_passes():
    frame = pd.DataFrame({'a': [1, 2]}, index=pd.Index(['x', 'y'], name='parcel_id'))
    require_unique_index(frame, 'context')


def test_message_names_context_index_and_labels():
    frame = pd.DataFrame(
        {'a': [1, 2, 3]}, index=pd.Index(['x', 'x', 'y'], name='parcel_id')
    )
    with pytest.raises(ValueError) as excinfo:
        require_unique_index(frame, 'curate US-NC-WA')
    message = str(excinfo.value)
    assert 'curate US-NC-WA' in message
    assert 'parcel_id' in message
    assert "'x'" in message
    assert "'y'" not in message


def test_lists_at_most_max_labels():
    labels = [str(i) for i in range(8) for _ in range(2)]
    frame = pd.DataFrame({'a': range(16)}, index=pd.Index(labels, name='id'))
    with pytest.raises(ValueError) as excinfo:
        require_unique_index(frame, 'context', max_labels=3)
    message = str(excinfo.value)
    assert '8 repeated' in message
    assert '16 rows' in message
    assert message.count("'") == 6 + 2  # three labels plus the index name
    assert '...' in message


def test_accepts_an_index_a_series_and_a_geodataframe():
    index = pd.Index(['x', 'x'], name='footprint_id')
    with pytest.raises(ValueError):
        require_unique_index(index, 'context')
    with pytest.raises(ValueError):
        require_unique_index(pd.Series([1, 2], index=index), 'context')
    gdf = gpd.GeoDataFrame(
        {'geometry': [box(0, 0, 1, 1), box(1, 0, 2, 1)]}, index=index
    )
    with pytest.raises(ValueError):
        require_unique_index(gdf, 'context')


def test_multiindex_reports_its_level_names():
    index = pd.MultiIndex.from_tuples(
        [('a', 'p'), ('a', 'p'), ('b', 'q')], names=['footprint_id', 'parcel_id']
    )
    with pytest.raises(ValueError) as excinfo:
        require_unique_index(pd.DataFrame({'v': [1, 2, 3]}, index=index), 'context')
    assert "['footprint_id', 'parcel_id']" in str(excinfo.value)
