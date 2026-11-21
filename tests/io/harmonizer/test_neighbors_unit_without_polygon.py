"""A unit listed without a polygon fails the neighbor lookup by name.

The country's admin table can list a county before its state's admin
layer is built, with no geometry. The lookup then names the layer to
build instead of answering that the unit has no neighbors. Fabricated
units throughout.
"""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from openplaces.io.harmonizer import neighbors
from openplaces.io.harmonizer.neighbors import UnitGeometryMissingError


@pytest.fixture
def units(monkeypatch):
    table = gpd.GeoDataFrame(
        index=pd.Index(['US-XX-AAA', 'US-XX-BBB', 'US-XX-CCC'], name='admin_id'),
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), None],
        crs='EPSG:4326',
    )
    monkeypatch.setattr(neighbors, '_level_units', lambda country, level: table)
    neighbors.touching_units.cache_clear()
    yield table
    neighbors.touching_units.cache_clear()


def test_a_unit_without_a_polygon_names_the_layer_to_build(units):
    with pytest.raises(UnitGeometryMissingError, match='admin layer of US-XX first'):
        neighbors.touching_units('US-XX-CCC')


def test_a_unit_with_a_polygon_finds_its_neighbor(units):
    assert neighbors.touching_units('US-XX-AAA') == ('US-XX-BBB',)


def test_an_unknown_unit_has_no_neighbors(units):
    assert neighbors.touching_units('US-XX-ZZZ') == ()
