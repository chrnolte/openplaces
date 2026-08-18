import warnings
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pytest
import shapely
from lonboard import PolygonLayer
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


def _base_z(geom):
    """The z-coordinate of a polygon's first vertex (its base offset)."""
    return np.asarray(shapely.get_coordinates(geom, include_z=True))[0, 2]


def test_show_value_terrain_layer_elevation_column_sets_base_z():
    poly1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    poly2 = Polygon([(1, 1), (2, 1), (2, 2), (1, 2)])
    gdf = gpd.GeoDataFrame(
        {
            'land_value_imputed': [100.0, 500.0],
            'area_m2': [10.0, 10.0],  # rates: 10.0, 50.0 $/m2
            'elevation': [30.0, None],  # one real, one missing -> 0
        },
        geometry=[poly1, poly2],
        crs='EPSG:4326',
    )

    with patch('openplaces.viz.terrain.get_entities', return_value=gdf):
        with pytest.warns(UserWarning, match="missing 'elevation'"):
            terrain_layer = show_value_terrain_layer(
                recipe='dummy_recipe',
                admin_id='US-MA-SU',
                value_column='land_value_imputed',
                area_m2_column='area_m2',
                elevation_column='elevation',
                elevation_scale=1.0,
                height_clip_percentile=None,
                silent=False,
            )

    # rendered_elevation stays purely value-based, unaffected by elevation_column
    np.testing.assert_allclose(terrain_layer.rendered_elevation, [10.0, 50.0])

    # Each polygon's own base z should equal its elevation (0 for the missing row)
    base_z = [_base_z(geom) for geom in terrain_layer.gdf.geometry]
    np.testing.assert_allclose(base_z, [30.0, 0.0])

    # total_elevation = base_z + rendered_elevation
    np.testing.assert_allclose(terrain_layer.total_elevation, [40.0, 50.0])


def test_show_value_terrain_layer_elevation_column_missing_silent():
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    gdf = gpd.GeoDataFrame(
        {
            'land_value_imputed': [100.0],
            'area_m2': [10.0],
            'elevation': [None],
        },
        geometry=[poly],
        crs='EPSG:4326',
    )

    with patch('openplaces.viz.terrain.get_entities', return_value=gdf):
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter('always')
            show_value_terrain_layer(
                recipe='dummy_recipe',
                admin_id='US-MA-SU',
                value_column='land_value_imputed',
                area_m2_column='area_m2',
                elevation_column='elevation',
                silent=True,
            )

            user_warnings = [r for r in record if issubclass(r.category, UserWarning)]
            assert len(user_warnings) == 0


def test_show_value_terrain_layer_stack_on_uses_total_elevation():
    # Same footprint for both layers so the fallback spatial-overlap match
    # in `_match_largest_overlap` unambiguously pairs them.
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    base_gdf = gpd.GeoDataFrame(
        {
            'land_value_imputed': [100.0],
            'area_m2': [10.0],  # rate: 10.0 $/m2
            'elevation': [50.0],
        },
        geometry=[poly],
        crs='EPSG:4326',
    )
    top_gdf = gpd.GeoDataFrame(
        {'land_value_imputed': [200.0], 'area_m2': [10.0]},  # rate: 20.0 $/m2
        geometry=[poly],
        crs='EPSG:4326',
    )

    with patch('openplaces.viz.terrain.get_entities', return_value=base_gdf):
        base_layer = show_value_terrain_layer(
            recipe='dummy_recipe',
            admin_id='US-MA-SU',
            value_column='land_value_imputed',
            area_m2_column='area_m2',
            elevation_column='elevation',
            elevation_scale=1.0,
            silent=True,
        )

    # base's total_elevation = 50 (ground) + 10 (value height)
    np.testing.assert_allclose(base_layer.total_elevation, [60.0])

    with patch('openplaces.viz.terrain.get_entities', return_value=top_gdf):
        top_layer = show_value_terrain_layer(
            recipe='dummy_recipe',
            admin_id='US-MA-SU',
            value_column='land_value_imputed',
            area_m2_column='area_m2',
            elevation_scale=1.0,
            stack_on=base_layer,
            silent=True,
        )

    # top's base should be base's total_elevation (60), not just its
    # rendered_elevation (10) -- i.e. it rides on top of parcel ground + height.
    top_base_z = _base_z(top_layer.gdf.geometry.iloc[0])
    assert top_base_z == pytest.approx(60.0)
    np.testing.assert_allclose(top_layer.total_elevation, [80.0])


def test_show_value_terrain_layer_elevation_recipe_drape_mode():
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    gdf = gpd.GeoDataFrame(
        {'land_value_imputed': [100.0], 'area_m2': [10.0]},  # rate: 10.0 $/m2
        geometry=[poly],
        crs='EPSG:4326',
    )

    # A stand-in draped geometry: same footprint, per-vertex Z varying
    # (as `viz.elevation.drape_parcel_elevation` would return).
    geom3d = shapely.force_3d(poly, z=0.0)
    xyz = np.asarray(shapely.get_coordinates(geom3d, include_z=True))
    xyz[:, 2] = [5.0, 6.0, 7.0, 8.0, 5.0]
    draped_geom = gpd.GeoSeries([shapely.set_coordinates(geom3d, xyz)], crs='EPSG:4326')
    mean_elevation = np.array([xyz[:, 2].mean()])

    with (
        patch('openplaces.viz.terrain.get_entities', return_value=gdf),
        patch(
            'openplaces.viz.terrain.elevation.drape_parcel_elevation',
            return_value=(draped_geom, mean_elevation),
        ) as mock_drape,
    ):
        terrain_layer = show_value_terrain_layer(
            recipe='dummy_recipe',
            admin_id='US-MA-SU',
            value_column='land_value_imputed',
            area_m2_column='area_m2',
            elevation_recipe='dummy_dem_recipe',
            elevation_mode='drape',
            elevation_scale=1.0,
            silent=True,
        )

    mock_drape.assert_called_once()
    out_coords = np.asarray(
        shapely.get_coordinates(terrain_layer.gdf.geometry.iloc[0], include_z=True)
    )
    # Per-vertex Z from the drape is preserved, not flattened to one scalar.
    np.testing.assert_allclose(out_coords[:, 2], [5.0, 6.0, 7.0, 8.0, 5.0])
    # total_elevation = mean_elevation (6.4) + rendered_elevation (10.0)
    np.testing.assert_allclose(terrain_layer.total_elevation, mean_elevation + 10.0)


def test_show_value_terrain_layer_terrain_exaggeration_scales_drape_and_mean():
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    gdf = gpd.GeoDataFrame(
        {'land_value_imputed': [100.0], 'area_m2': [10.0]},  # rate: 10.0 $/m2
        geometry=[poly],
        crs='EPSG:4326',
    )

    geom3d = shapely.force_3d(poly, z=0.0)
    xyz = np.asarray(shapely.get_coordinates(geom3d, include_z=True))
    xyz[:, 2] = [1.0, 2.0, 3.0, 4.0, 1.0]
    draped_geom = gpd.GeoSeries([shapely.set_coordinates(geom3d, xyz)], crs='EPSG:4326')
    mean_elevation = np.array([xyz[:, 2].mean()])  # 2.2

    with (
        patch('openplaces.viz.terrain.get_entities', return_value=gdf),
        patch(
            'openplaces.viz.terrain.elevation.drape_parcel_elevation',
            return_value=(draped_geom, mean_elevation),
        ),
    ):
        terrain_layer = show_value_terrain_layer(
            recipe='dummy_recipe',
            admin_id='US-MA-SU',
            value_column='land_value_imputed',
            area_m2_column='area_m2',
            elevation_recipe='dummy_dem_recipe',
            elevation_mode='drape',
            elevation_scale=1.0,
            terrain_exaggeration=3.0,
            silent=True,
        )

    out_coords = np.asarray(
        shapely.get_coordinates(terrain_layer.gdf.geometry.iloc[0], include_z=True)
    )
    # Rendered geometry's per-vertex Z is exaggerated 3x from the raw drape.
    np.testing.assert_allclose(out_coords[:, 2], [3.0, 6.0, 9.0, 12.0, 3.0])
    # total_elevation = 3x mean_elevation (6.6) + rendered_elevation (10.0),
    # not the raw (unexaggerated) mean.
    np.testing.assert_allclose(terrain_layer.total_elevation, [16.6])


def test_show_value_terrain_layer_stack_on_avoids_double_counting_ground():
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    base_gdf = gpd.GeoDataFrame(
        {
            'land_value_imputed': [100.0],
            'area_m2': [10.0],  # rate: 10.0 $/m2
            'elevation': [50.0],
        },
        geometry=[poly],
        crs='EPSG:4326',
    )
    top_gdf = gpd.GeoDataFrame(
        {'land_value_imputed': [200.0], 'area_m2': [10.0]},  # rate: 20.0 $/m2
        geometry=[poly],
        crs='EPSG:4326',
    )

    with patch('openplaces.viz.terrain.get_entities', return_value=base_gdf):
        base_layer = show_value_terrain_layer(
            recipe='dummy_recipe',
            admin_id='US-MA-SU',
            value_column='land_value_imputed',
            area_m2_column='area_m2',
            elevation_column='elevation',
            elevation_scale=1.0,
            silent=True,
        )
    # base.total_elevation = 50 (ground) + 10 (value height) = 60

    with (
        patch('openplaces.viz.terrain.get_entities', return_value=top_gdf),
        patch(
            'openplaces.viz.terrain.elevation.get_building_elevation',
            return_value=np.array([5.0]),
        ),
    ):
        top_layer = show_value_terrain_layer(
            recipe='dummy_recipe',
            admin_id='US-MA-SU',
            value_column='land_value_imputed',
            area_m2_column='area_m2',
            elevation_recipe='dummy_dem_recipe',
            elevation_mode='flat',
            elevation_scale=1.0,
            stack_on=base_layer,
            silent=True,
        )

    # top's base = its own ground (5.0) + base's *rendered* height only (10.0,
    # not base's total_elevation of 60) -- ground isn't double-counted.
    top_base_z = _base_z(top_layer.gdf.geometry.iloc[0])
    assert top_base_z == pytest.approx(15.0)
    np.testing.assert_allclose(
        top_layer.total_elevation, [35.0]
    )  # 15 + 20 (own height)


def _missing_value_gdf():
    """Two valued polygons, one zero-valued and one null-valued."""
    polys = [
        Polygon([(x, x), (x + 1, x), (x + 1, x + 1), (x, x + 1)]) for x in range(4)
    ]
    return gpd.GeoDataFrame(
        {
            'land_value_imputed': [100.0, 200.0, 0.0, None],
            'area_m2': [10.0, 10.0, 10.0, 10.0],
        },
        geometry=polys,
        crs='EPSG:4326',
    )


def test_missing_value_render_keeps_every_row():
    gdf = _missing_value_gdf()
    with patch('openplaces.viz.terrain.get_entities', return_value=gdf):
        layer = show_value_terrain_layer(
            recipe='dummy_recipe',
            admin_id='US-MA-SU',
            value_column='land_value_imputed',
            area_m2_column='area_m2',
            silent=True,
        )
    # Default mode is unchanged: valueless rows stay in the fill layer and
    # no ghost layer is built at all.
    assert len(layer.layer.table) == 4
    assert layer.ghost_layer is None


def test_missing_value_drop_removes_null_and_zero():
    gdf = _missing_value_gdf()
    with patch('openplaces.viz.terrain.get_entities', return_value=gdf):
        layer = show_value_terrain_layer(
            recipe='dummy_recipe',
            admin_id='US-MA-SU',
            value_column='land_value_imputed',
            area_m2_column='area_m2',
            missing_value='drop',
            silent=True,
        )
    assert len(layer.layer.table) == 2
    assert layer.ghost_layer is None
    assert layer.gdf['land_value_imputed'].tolist() == [100.0, 200.0]


def test_missing_value_ghost_splits_rows_between_layers():
    gdf = _missing_value_gdf()
    with patch('openplaces.viz.terrain.get_entities', return_value=gdf):
        layer = show_value_terrain_layer(
            recipe='dummy_recipe',
            admin_id='US-MA-SU',
            value_column='land_value_imputed',
            area_m2_column='area_m2',
            missing_value='ghost',
            ghost_rgba=(255, 255, 255, 140),
            silent=True,
        )
    # Each row is drawn exactly once: valued rows in the fill layer, the
    # zero/null pair in the ghost layer, never both.
    assert len(layer.layer.table) == 2
    assert len(layer.ghost_layer.table) == 2
    # `gdf` still carries every row, so a `stack_on` caller can find the
    # ghosted ones underneath whatever it stacks.
    assert len(layer.gdf) == 4
    assert list(layer.ghost_layer.get_line_color) == [255, 255, 255, 140]
    # A ghost is a ring, not a box: no fill, and no extrusion of its own.
    assert layer.ghost_layer.get_elevation == 0


def test_missing_value_ghost_floats_above_its_base():
    gdf = _missing_value_gdf()
    gdf['ground_m'] = [5.0, 5.0, 5.0, 5.0]

    # Capture the geometry actually handed to the ghost layer -- reading it
    # back off the built layer means an arrow round-trip, and the offset is
    # applied on the way in.
    captured = []
    real_from_geopandas = PolygonLayer.from_geopandas

    def spy(frame, **kwargs):
        captured.append(frame.copy())
        return real_from_geopandas(frame, **kwargs)

    with (
        patch('openplaces.viz.terrain.get_entities', return_value=gdf),
        patch.object(PolygonLayer, 'from_geopandas', staticmethod(spy)),
    ):
        layer = show_value_terrain_layer(
            recipe='dummy_recipe',
            admin_id='US-MA-SU',
            value_column='land_value_imputed',
            area_m2_column='area_m2',
            elevation_column='ground_m',
            missing_value='ghost',
            ghost_offset=2.0,
            silent=True,
        )

    # Every row's base sits at its 5 m ground elevation ...
    ghosted = layer.gdf[layer.gdf['land_value_imputed'].fillna(0) == 0]
    assert _base_z(ghosted.geometry.to_numpy()) == 5.0
    # ... and the ring uploaded for the ghosts sits ghost_offset above it.
    assert _base_z(captured[-1].geometry.to_numpy()) == 7.0


def test_missing_value_ghost_excludes_ghosts_from_value_range():
    gdf = _missing_value_gdf()
    with patch('openplaces.viz.terrain.get_entities', return_value=gdf):
        ghosted = show_value_terrain_layer(
            recipe='dummy_recipe',
            admin_id='US-MA-SU',
            value_column='land_value_imputed',
            area_m2_column='area_m2',
            missing_value='ghost',
            silent=True,
        )
        rendered = show_value_terrain_layer(
            recipe='dummy_recipe',
            admin_id='US-MA-SU',
            value_column='land_value_imputed',
            area_m2_column='area_m2',
            silent=True,
        )
    # Ghosts are never colored, so they must not drag the colorbar bounds
    # down to zero the way the rendered-as-gray mode does.
    assert rendered.value_range[0] == 0.0
    assert ghosted.value_range[0] > 0.0


def test_missing_value_invalid_mode_raises():
    gdf = _missing_value_gdf()
    with patch('openplaces.viz.terrain.get_entities', return_value=gdf):
        with pytest.raises(ValueError, match='missing_value'):
            show_value_terrain_layer(
                recipe='dummy_recipe',
                admin_id='US-MA-SU',
                value_column='land_value_imputed',
                area_m2_column='area_m2',
                missing_value='hide',
                silent=True,
            )


def test_drop_corridors_removes_road_shaped_polygons():
    lat = 42.0
    deg_y = 1 / 111_320
    deg_x = deg_y / np.cos(np.radians(lat))

    def strip(length_m, width_m, offset=0.0):
        dx, dy = length_m * deg_x, width_m * deg_y
        y = lat + offset
        return Polygon([(0, y), (dx, y), (dx, y + dy), (0, y + dy)])

    gdf = gpd.GeoDataFrame(
        {
            'land_value_imputed': [100.0, 200.0],
            'area_m2': [50_000.0, 40_000.0],
        },
        geometry=[strip(200, 200), strip(5000, 10, offset=0.05)],
        crs='EPSG:4326',
    )

    with patch('openplaces.viz.terrain.get_entities', return_value=gdf):
        kept = show_value_terrain_layer(
            recipe='dummy_recipe',
            admin_id='US-MA-SU',
            value_column='land_value_imputed',
            area_m2_column='area_m2',
            silent=True,
        )
        dropped = show_value_terrain_layer(
            recipe='dummy_recipe',
            admin_id='US-MA-SU',
            value_column='land_value_imputed',
            area_m2_column='area_m2',
            drop_corridors=True,
            silent=True,
        )
    # The corridor is removed on shape alone -- both rows carry a value, so
    # no missing_value mode would have caught it.
    assert len(kept.layer.table) == 2
    assert len(dropped.layer.table) == 1
    assert dropped.gdf['land_value_imputed'].tolist() == [100.0]


def test_drop_corridors_accepts_threshold_overrides():
    lat = 42.0
    deg_y = 1 / 111_320
    deg_x = deg_y / np.cos(np.radians(lat))
    dx, dy = 2000 * deg_x, 40 * deg_y  # elongation ~50: under the default
    gdf = gpd.GeoDataFrame(
        {'land_value_imputed': [100.0], 'area_m2': [80_000.0]},
        geometry=[Polygon([(0, lat), (dx, lat), (dx, lat + dy), (0, lat + dy)])],
        crs='EPSG:4326',
    )

    with patch('openplaces.viz.terrain.get_entities', return_value=gdf):
        default = show_value_terrain_layer(
            recipe='dummy_recipe',
            admin_id='US-MA-SU',
            value_column='land_value_imputed',
            area_m2_column='area_m2',
            drop_corridors=True,
            silent=True,
        )
        # Tightening the threshold removes the only row; deck.gl cannot
        # draw an empty layer, so that is an explicit error, not a
        # silently blank map.
        with pytest.raises(ValueError, match='No rows left to render'):
            show_value_terrain_layer(
                recipe='dummy_recipe',
                admin_id='US-MA-SU',
                value_column='land_value_imputed',
                area_m2_column='area_m2',
                drop_corridors={'max_elongation': 25},
                silent=True,
            )
    assert len(default.layer.table) == 1
