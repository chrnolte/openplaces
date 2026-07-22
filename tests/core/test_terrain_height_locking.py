"""Tests for height scaling locking in show_value_terrain_layer.

Verifies that height is consistently computed as value/m² * elevation_scale,
regardless of which area column or unit system is used. This is critical for
stacking layers — two values divided by the same area in m² must produce
proportional heights even if one uses area_m2 and the other uses area_ha.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
from shapely.geometry import box

from openplaces.geo.polygon import convert_area, resolve_area
from openplaces.viz.terrain import _height_value


def test_resolve_area_detects_area_m2_first():
    """When both area_m2 and area_ha exist, area_m2 should be used."""
    gdf = gpd.GeoDataFrame(
        {
            'area_m2': [100.0],
            'area_ha': [0.01],  # Same physical area, different units
        },
        geometry=[box(0, 0, 10, 10)],
        crs='epsg:4326',
    )
    result = resolve_area(gdf, unit='m2')
    np.testing.assert_allclose(result, [100.0], rtol=1e-10)


def test_resolve_area_converts_ha_to_requested_unit():
    """When only area_ha exists, convert it to the requested unit."""
    gdf = gpd.GeoDataFrame(
        {
            'area_ha': [0.01],  # 100 m2
        },
        geometry=[box(0, 0, 10, 10)],
        crs='epsg:4326',
    )
    # Ask for m2; should get 100
    result = resolve_area(gdf, unit='m2')
    np.testing.assert_allclose(result, [100.0], rtol=1e-10)

    # Ask for ha; should get 0.01
    result = resolve_area(gdf, unit='ha')
    np.testing.assert_allclose(result, [0.01], rtol=1e-10)


def test_convert_area_round_trip():
    """Converting ha -> m2 -> ha should round-trip exactly."""
    original_ha = 0.5
    converted = convert_area(original_ha, 'ha', 'm2')
    assert converted == 5_000.0

    back = convert_area(converted, 'm2', 'ha')
    assert abs(back - original_ha) < 1e-10


def test_height_computation_locked_across_area_units():
    """Critical test: same value/area ratio produces identical $/m² basis.

    Two GeoDataFrames with:
    - area_m2: value=2000, area_m2=100 → $/m² = 20
    - area_ha: value=2000, area_ha=0.01 (=100m²) → $/m² = 20

    Heights computed from each must be identical when both use
    elevation_scale=1.0 (meters per $1/m²).
    """
    # Footprint-style with area_m2
    gdf_footprint = gpd.GeoDataFrame(
        {'value': [2000.0], 'area_m2': [100.0]},
        geometry=[box(0, 0, 10, 10)],
        crs='epsg:4326',
    )

    # Parcel-style with area_ha
    gdf_parcel = gpd.GeoDataFrame(
        {'value': [2000.0], 'area_ha': [0.01]},
        geometry=[box(0, 0, 10, 10)],
        crs='epsg:4326',
    )

    # resolve_area with unit='m2' should handle both correctly
    area_m2_from_m2 = resolve_area(gdf_footprint, unit='m2')[0]
    area_m2_from_ha = resolve_area(gdf_parcel, unit='m2')[0]

    assert abs(area_m2_from_m2 - area_m2_from_ha) < 1e-10, (
        f'Areas should be identical: {area_m2_from_m2} vs {area_m2_from_ha}'
    )

    # Per-m² values should be identical
    value_per_m2_footprint = 2000.0 / area_m2_from_m2
    value_per_m2_parcel = 2000.0 / area_m2_from_ha

    assert abs(value_per_m2_footprint - value_per_m2_parcel) < 1e-10


def test_height_ratio_matches_value_per_m2_ratio():
    """Height should scale with value/m², the core requirement.

    Two rows with same area but different values should have
    heights proportional to their value/m² ratios.
    """
    gdf = gpd.GeoDataFrame(
        {
            'value': [1000.0, 2205.0],  # 2.205x ratio
            'area_m2': [100.0, 100.0],  # Same area in m2
        },
        geometry=[box(0, 0, 10, 10), box(10, 0, 20, 10)],
        crs='epsg:4326',
    )

    # Get m2 areas for both rows
    areas_m2 = resolve_area(gdf, unit='m2')
    assert np.allclose(areas_m2, [100.0, 100.0])

    # Compute value/m2 for each
    value_per_m2 = gdf['value'].values / areas_m2

    # Heights should be proportional to value/m2 (with elevation_scale=1.0)
    heights = value_per_m2 * 1.0  # elevation_scale = 1.0

    height_ratio = heights[1] / heights[0]
    expected_ratio = 2205.0 / 1000.0

    assert abs(height_ratio - expected_ratio) < 1e-6


def test_height_locking_with_mixed_area_columns():
    """Simulates rendering building and parcel with their native area columns.

    Building has area_m2, parcel has area_ha. Same $/m² value should produce
    same height regardless of which area column contains it.
    """
    elevation_scale = 0.0025  # meters per $1/m2

    # Scenario: building with area_m2, parcel with area_ha
    # Both have exactly 100 m2 (0.01 ha) area
    # Both have value that produces $20/m2

    value = 2000.0  # This will give $20/m2 when divided by 100 m2

    gdf_building = gpd.GeoDataFrame(
        {'value': [value], 'area_m2': [100.0]},
        geometry=[box(0, 0, 10, 10)],
        crs='epsg:4326',
    )

    gdf_parcel = gpd.GeoDataFrame(
        {'value': [value], 'area_ha': [0.01]},  # 0.01 ha = 100 m2
        geometry=[box(0, 0, 10, 10)],
        crs='epsg:4326',
    )

    # Both resolve to same m2 area
    area_m2_building = resolve_area(gdf_building, unit='m2')[0]
    area_m2_parcel = resolve_area(gdf_parcel, unit='m2')[0]

    assert abs(area_m2_building - area_m2_parcel) < 1e-10

    # Compute heights using the standard formula
    height_building = (value / area_m2_building) * elevation_scale
    height_parcel = (value / area_m2_parcel) * elevation_scale

    # Heights must be identical
    assert abs(height_building - height_parcel) < 1e-10, (
        f'Heights should match: {height_building:.8f} vs {height_parcel:.8f}'
    )


def test_height_value_function_with_mixed_area_units():
    """Integration test: _height_value produces consistent heights.

    This is the actual function used in show_value_terrain_layer. The
    height value should be proportional to value/area_m2, regardless of
    whether the area came from area_m2 or area_ha (converted).

    Note: With only 2 data points, height_clip_percentile=99.9 may clip
    the top value slightly. We disable clipping here to test the core
    proportionality.
    """
    area_m2_both = np.array([100.0, 100.0])
    value_both = np.array([1000.0, 2205.0])

    # Disable clipping (height_clip_percentile=None) to test pure proportionality
    per_area_m2, height_value, is_clipped = _height_value(
        value_both,
        area_m2_both,
        log_height=False,
        height_clip_percentile=None,
    )

    # Heights should be proportional to values
    height_ratio = height_value[1] / height_value[0]
    value_ratio = 2205.0 / 1000.0

    assert abs(height_ratio - value_ratio) < 1e-6, (
        f'Height ratio {height_ratio:.6f} should match value ratio {value_ratio:.6f}'
    )

    # Verify the actual per-area values are correct
    np.testing.assert_allclose(per_area_m2, [10.0, 22.05], rtol=1e-10)


def test_height_value_clipping_with_value_cutoff():
    """Verify that _height_value clips correctly.

    Checks behavior when height_clip_value_m2 is provided.
    """
    area_m2 = np.array([10.0, 10.0, 10.0])
    value = np.array([100.0, 250.0, 400.0])  # rates: 10.0, 25.0, 40.0

    per_area_m2, height_value, is_clipped = _height_value(
        value,
        area_m2,
        log_height=False,
        height_clip_percentile=99.0,
        height_clip_value_m2=25.0,  # takes precedence over percentile
    )

    np.testing.assert_allclose(per_area_m2, [10.0, 25.0, 40.0])
    # The height_value should be clipped to a max of 25.0
    np.testing.assert_allclose(height_value, [10.0, 25.0, 25.0])
    # Only the last item should be marked as clipped (rate 40.0 > 25.0)
    np.testing.assert_array_equal(is_clipped, [False, False, True])
