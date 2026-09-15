"""`get_admin_boundary_layer(snap_to=...)`: the fence follows the parcels.

A boundary draped on its own vertices and the parcel walls beside it
are two straight-segment approximations of the same terrain from
different vertex sets. Snapping inserts the parcel vertices into the
boundary and re-interpolates the boundary's own vertices along the
chord, so the two ground lines coincide.
"""

from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, Polygon

from openplaces.viz import interactive


@pytest.fixture
def town_and_parcels():
    """A square town; two parcels share its west edge (x=0)."""
    town = gpd.GeoDataFrame(
        {'name': ['town']},
        geometry=[Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])],
        crs='EPSG:32619',
    )
    # Parcel vertices on x=0 at y=0, 40, 60, 100, with scene z that the
    # flat 5 m drape below would never give.
    p1 = Polygon([(0, 0, 10), (30, 0, 10), (30, 40, 20), (0, 40, 20)])
    p2 = Polygon([(0, 60, 30), (30, 60, 30), (30, 100, 40), (0, 100, 40)])
    parcels = gpd.GeoDataFrame(geometry=[p1, p2], crs='EPSG:32619')
    return town, parcels


def _flat_drape(gdf, elevation_recipe, terrain_exaggeration, elevation_datum=0.0):
    out = gdf.copy()
    out.geometry = shapely.force_3d(gdf.geometry, z=5.0)
    return out


def test_snapped_boundary_carries_parcel_vertices_and_chords(town_and_parcels):
    town, parcels = town_and_parcels
    lines = town.copy()
    lines.geometry = town.geometry.boundary
    lines = _flat_drape(lines, None, 1.0)
    snapped = interactive._snap_boundary_to_vertices(
        lines, parcels, tolerance=1.0, max_gap=100.0
    )
    xyz = shapely.get_coordinates(snapped.geometry.to_numpy(), include_z=True)
    west = xyz[np.isclose(xyz[:, 0], 0.0)]
    by_y = {round(y, 3): z for _, y, z in west}
    # Parcel vertices inserted with their own z.
    assert by_y[40.0] == pytest.approx(20.0)
    assert by_y[60.0] == pytest.approx(30.0)
    # The town's own corners on that edge coincide with parcel corners
    # and take the parcel z, not the draped 5 m.
    assert by_y[0.0] == pytest.approx(10.0)
    assert by_y[100.0] == pytest.approx(40.0)
    # The east edge has no parcels: its vertices keep the draped z.
    east = xyz[np.isclose(xyz[:, 0], 100.0)]
    assert np.allclose(east[:, 2], 5.0)
    # Still one closed ring.
    assert snapped.geometry.iloc[0].is_closed


def test_snapped_boundary_reinterpolates_own_vertices_on_the_chord():
    # A boundary vertex at y=50 between parcel vertices at y=0 (z=0)
    # and y=100 (z=20): the draped 5 m gives way to the chord's 10.
    lines = gpd.GeoDataFrame(
        geometry=[LineString([(0, 0, 5), (0, 50, 5), (0, 100, 5)])],
        crs='EPSG:32619',
    )
    parcel = gpd.GeoDataFrame(
        geometry=[Polygon([(0, 0, 0), (10, 0, 0), (10, 100, 20), (0, 100, 20)])],
        crs='EPSG:32619',
    )
    snapped = interactive._snap_boundary_to_vertices(lines, parcel, 1.0, 100.0)
    xyz = shapely.get_coordinates(snapped.geometry.to_numpy(), include_z=True)
    assert xyz[np.isclose(xyz[:, 1], 50.0)][0, 2] == pytest.approx(10.0)
    # ... but not across a gap longer than max_gap.
    kept = interactive._snap_boundary_to_vertices(lines, parcel, 1.0, 50.0)
    xyz = shapely.get_coordinates(kept.geometry.to_numpy(), include_z=True)
    assert xyz[np.isclose(xyz[:, 1], 50.0)][0, 2] == pytest.approx(5.0)


def test_snap_survives_a_geographic_crs_round_trip():
    lines = gpd.GeoDataFrame(
        geometry=[LineString([(0, 0, 5), (0, 100, 5)])], crs='EPSG:32619'
    ).to_crs('EPSG:4326')
    lines.geometry = gpd.GeoSeries(
        shapely.force_3d(lines.geometry.to_numpy(), z=5.0), crs='EPSG:4326'
    )
    parcel = gpd.GeoDataFrame(
        geometry=[Polygon([(0, 50, 30), (10, 50, 30), (10, 60, 30), (0, 60, 30)])],
        crs='EPSG:32619',
    )
    snapped = interactive._snap_boundary_to_vertices(lines, parcel, 1.0, 100.0)
    z = shapely.get_coordinates(snapped.geometry.to_numpy(), include_z=True)[:, 2]
    assert 30.0 in np.round(z, 6)
    assert snapped.crs == lines.crs


def test_ribbon_vertices_take_line_z():
    lines = gpd.GeoDataFrame(
        geometry=[LineString([(0, 0, 0), (0, 100, 20)])], crs='EPSG:32619'
    )
    ribbon = interactive._ribbon_from_lines(lines, half_width=0.05)
    xyz = shapely.get_coordinates(ribbon.to_numpy(), include_z=True)
    assert np.allclose(xyz[:, 2], np.clip(xyz[:, 1], 0, 100) * 0.2, atol=0.05)
    assert ribbon.iloc[0].area == pytest.approx(0.1 * 100, rel=0.1)


def test_fence_layer_snaps_when_asked(town_and_parcels):
    town, parcels = town_and_parcels
    with patch.object(interactive, '_drape_boundary', side_effect=_flat_drape):
        layer = interactive.get_admin_boundary_layer(
            gdf=town,
            mode='fence',
            elevation=5,
            elevation_recipe='dummy-dem',
            snap_to=parcels,
        )
    assert layer.extruded
    assert len(layer.table) == 1


def test_split_rim_pieces_separates_edge_triangles():
    from openplaces.viz import terrain

    ring = shapely.Polygon([(0, 0, 1), (10, 0, 1), (10, 10, 1), (0, 10, 1)])
    # Four triangles around the center plus one interior triangle that
    # touches the ring at a single vertex only.
    center = (5, 5, 1)
    rim = [
        shapely.Polygon([(0, 0, 1), (10, 0, 1), center]),
        shapely.Polygon([(10, 0, 1), (10, 10, 1), center]),
        shapely.Polygon([(10, 10, 1), (0, 10, 1), center]),
    ]
    inner = shapely.Polygon([(0, 10, 1), (3, 6, 1), center])
    mesh = np.array([shapely.MultiPolygon(rim + [inner])], dtype=object)
    rim_out, interior_out = terrain._split_rim_pieces(mesh, np.array([ring]))
    assert shapely.get_num_geometries(rim_out[0]) == 3
    assert shapely.get_num_geometries(interior_out[0]) == 1
    single = np.array([shapely.MultiPolygon(rim)], dtype=object)
    _, interior_none = terrain._split_rim_pieces(single, np.array([ring]))
    assert interior_none[0] is None


def test_mesh_cells_for_budget_coarsens_with_row_count():
    from openplaces.viz import terrain

    assert terrain._mesh_cells_for_budget(3_000, 8, 2_000_000) == 8
    assert terrain._mesh_cells_for_budget(13_000, 8, 2_000_000) == 5
    # Six Boston-area towns: even a 2x2 grid per parcel would not fit.
    assert terrain._mesh_cells_for_budget(160_000, 8, 2_000_000) < 2
    assert terrain._mesh_cells_for_budget(160_000, 8, None) == 8
    assert terrain._mesh_cells_for_budget(0, 8, 2_000_000) == 8


def test_needs_reassert_only_above_the_default_ceiling():
    from lonboard.view_state import MapViewState

    from openplaces.viz import interactive

    echoed = MapViewState(pitch=30)  # lonboard's echo: ceiling back at 60
    assert not interactive._needs_reassert(echoed, 85, 0, 59.0)
    assert interactive._needs_reassert(echoed, 85, 0, None)
    high = MapViewState(pitch=70)
    assert interactive._needs_reassert(high, 85, 0, 59.0)
    held = MapViewState(pitch=70, max_pitch=85, min_pitch=0)
    assert not interactive._needs_reassert(held, 85, 0, 59.0)
    # A requested floor above the default can bind anywhere.
    assert interactive._needs_reassert(echoed, 85, 10, 59.0)
