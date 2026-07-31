import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from openplaces.viz import get_admin_boundary_layer


@pytest.fixture
def sample_admin_gdf():
    poly1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    poly2 = Polygon([(1, 1), (2, 1), (2, 2), (1, 2)])
    return gpd.GeoDataFrame(
        {'admin_id': ['A1', 'A2'], 'name': ['Unit 1', 'Unit 2']},
        geometry=[poly1, poly2],
        crs='EPSG:4326',
    )


def test_get_admin_boundary_layer_from_gdf(sample_admin_gdf):
    layer = get_admin_boundary_layer(
        gdf=sample_admin_gdf, color='white', width=3, style='dotted'
    )
    assert layer is not None
    assert len(layer.table) == 2
    assert layer.get_width == 3.0
    assert layer.width_units == 'pixels'
    assert list(layer.get_color) == [255, 255, 255, 255]
    assert list(layer.get_dash_array) == [3, 3]


def test_get_admin_boundary_layer_floating_elevation(sample_admin_gdf):
    elevation_val = 150.0
    layer = get_admin_boundary_layer(
        gdf=sample_admin_gdf,
        elevation=elevation_val,
        color='yellow',
        width=4,
        style='dashed',
    )
    assert layer.get_width == 4.0
    assert list(layer.get_dash_array) == [6, 6]


def test_get_admin_boundary_layer_solid_style(sample_admin_gdf):
    layer = get_admin_boundary_layer(
        gdf=sample_admin_gdf, color='#00ff00', style='solid'
    )
    assert not hasattr(layer, 'get_dash_array') or layer.get_dash_array is None
    assert list(layer.get_color) == [0, 255, 0, 255]


def test_get_admin_boundary_layer_fence_mode(sample_admin_gdf):
    elevation_val = 150.0
    layer = get_admin_boundary_layer(
        gdf=sample_admin_gdf,
        elevation=elevation_val,
        color='#00ff00',
        mode='fence',
    )
    assert layer is not None
    assert layer.extruded is True
    assert layer.get_elevation == elevation_val
    assert layer.wireframe is True
    assert list(layer.get_line_color) == [0, 255, 0, 255]
    assert list(layer.get_fill_color) == [255, 255, 255, 30]
