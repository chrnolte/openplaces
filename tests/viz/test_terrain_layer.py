import warnings
from unittest.mock import patch

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Polygon

from openplaces.viz.terrain import show_value_terrain_layer


@pytest.fixture
def sample_imperfect_gdf():
    # A mix of polygon and non-polygon, zero area, and missing values
    poly1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    poly2 = Polygon([(0, 0), (0, 0), (0, 0), (0, 0)])  # zero area
    line1 = LineString([(0, 0), (1, 1)])  # non-polygon
    poly3 = Polygon([(1, 1), (2, 1), (2, 2), (1, 2)])

    return gpd.GeoDataFrame(
        {
            'land_value_imputed': [100.0, 50.0, 20.0, None],  # one missing value
            'area_m2': [100.0, 0.0, 10.0, 100.0],
        },
        geometry=[poly1, poly2, line1, poly3],
        crs='EPSG:4326',
    )


def test_show_value_terrain_layer_silent_false(sample_imperfect_gdf):
    # With silent=False, warnings should be raised for:
    # 1. Non-polygon geometry
    # 2. Zero/missing area
    # 3. Missing values
    with patch(
        'openplaces.viz.terrain.get_entities', return_value=sample_imperfect_gdf
    ):
        with pytest.warns(UserWarning) as record:
            show_value_terrain_layer(
                recipe='dummy_recipe',
                admin_id='US-MA-SU',
                value_column='land_value_imputed',
                area_m2_column='area_m2',
                silent=False,
            )

        # We expect 3 distinct UserWarnings (non-polygon, zero area, missing value)
        warning_messages = [str(r.message) for r in record]
        assert any('non-polygon geometry' in msg for msg in warning_messages)
        assert any('zero/missing area' in msg for msg in warning_messages)
        assert any("missing 'land_value_imputed'" in msg for msg in warning_messages)


def test_show_value_terrain_layer_silent_true(sample_imperfect_gdf):
    # With silent=True, no warnings should be raised
    with patch(
        'openplaces.viz.terrain.get_entities', return_value=sample_imperfect_gdf
    ):
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter('always')
            show_value_terrain_layer(
                recipe='dummy_recipe',
                admin_id='US-MA-SU',
                value_column='land_value_imputed',
                area_m2_column='area_m2',
                silent=True,
            )

            # Check if any UserWarning was raised
            user_warnings = [r for r in record if issubclass(r.category, UserWarning)]
            assert len(user_warnings) == 0


def test_show_value_terrain_layer_height_clip_value():
    poly1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    poly2 = Polygon([(1, 1), (2, 1), (2, 2), (1, 2)])
    gdf = gpd.GeoDataFrame(
        {
            'land_value_imputed': [100.0, 500.0],
            'area_m2': [10.0, 10.0],  # rates: 10.0, 50.0 $/m2
        },
        geometry=[poly1, poly2],
        crs='EPSG:4326',
    )

    with patch('openplaces.viz.terrain.get_entities', return_value=gdf):
        # We set height_clip_value to 25.0 $/m2
        terrain_layer = show_value_terrain_layer(
            recipe='dummy_recipe',
            admin_id='US-MA-SU',
            value_column='land_value_imputed',
            area_m2_column='area_m2',
            height_clip_value=25.0,  # 25.0 $/m2
            elevation_scale=1.0,
            silent=True,
        )

        # Verify rendered_elevation has a max value of 25.0
        import numpy as np

        np.testing.assert_allclose(terrain_layer.rendered_elevation, [10.0, 25.0])


def test_show_value_terrain_layer_zero_values_grey():
    poly1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    poly2 = Polygon([(1, 1), (2, 1), (2, 2), (1, 2)])
    poly3 = Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])
    gdf = gpd.GeoDataFrame(
        {
            'land_value_imputed': [
                100.0,
                0.0,
                None,
            ],  # one positive, one zero, one missing (which defaults to zero)
            'area_m2': [10.0, 10.0, 10.0],
        },
        geometry=[poly1, poly2, poly3],
        crs='EPSG:4326',
    )

    with patch('openplaces.viz.terrain.get_entities', return_value=gdf):
        terrain_layer = show_value_terrain_layer(
            recipe='dummy_recipe',
            admin_id='US-MA-SU',
            value_column='land_value_imputed',
            area_m2_column='area_m2',
            alpha=0.8,
            silent=True,
        )

        # Alpha 255: round(0.8 * 255) = 204
        # Positive value should be colormapped normally (not 50% grey)
        # Zero and missing values should be colored exactly [128, 128, 128, 204]
        fill_colors = terrain_layer.layer.get_fill_color.to_pylist()
        assert fill_colors[0] != [128, 128, 128, 204]
        assert fill_colors[1] == [128, 128, 128, 204]
        assert fill_colors[2] == [128, 128, 128, 204]
