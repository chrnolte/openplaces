import geopandas as gpd
from shapely.geometry import box

from openplaces.geo.overlay import overlay_polygons_with_duckdb


def test_duckdb_overlay_fallback(monkeypatch, capsys):
    # Create simple test GeoDataFrames with distinct index names
    gdf1 = gpd.GeoDataFrame(
        {'id_1': ['A'], 'geometry': [box(0, 0, 2, 2)]}, crs='epsg:4326'
    ).set_index('id_1')
    gdf2 = gpd.GeoDataFrame(
        {'id_2': ['B'], 'geometry': [box(1, 1, 3, 3)]}, crs='epsg:4326'
    ).set_index('id_2')

    # Verify that it normally works with duckdb
    res = overlay_polygons_with_duckdb(gdf1, gdf2, how='intersection')
    assert len(res) == 1

    # 1. Test Proactive Bypass (by mock complexity)
    def mock_is_too_complex(layer, **kwargs):
        return True

    monkeypatch.setattr('openplaces.geo.overlay._is_too_complex', mock_is_too_complex)

    # 1a. Bypass with default print
    capsys.readouterr()  # clear buffers
    res_bypass = overlay_polygons_with_duckdb(gdf1, gdf2, how='intersection')
    assert len(res_bypass) == 1
    assert 'Geometries are highly complex' in capsys.readouterr().out

    # 1b. Bypass silenced with silent=True
    capsys.readouterr()
    res_bypass_silent = overlay_polygons_with_duckdb(
        gdf1, gdf2, how='intersection', silent=True
    )
    assert len(res_bypass_silent) == 1
    assert capsys.readouterr().out == ''

    # 1c. Bypass silenced with verbose=False
    capsys.readouterr()
    res_bypass_verbose_false = overlay_polygons_with_duckdb(
        gdf1, gdf2, how='intersection', verbose=False
    )
    assert len(res_bypass_verbose_false) == 1
    assert capsys.readouterr().out == ''

    # Restore _is_too_complex
    monkeypatch.undo()

    # 2. Test Reactive Fallback (by raising exception during run)
    def mock_overlay_paths(*args, **kwargs):
        raise RuntimeError('Simulated DuckDB failure')

    monkeypatch.setattr(
        'openplaces.geo.overlay._overlay_polygons_paths', mock_overlay_paths
    )

    # 2a. Fallback with default print
    capsys.readouterr()
    res_fallback = overlay_polygons_with_duckdb(gdf1, gdf2, how='intersection')
    assert len(res_fallback) == 1
    captured = capsys.readouterr().out
    assert 'DuckDB spatial join failed' in captured
    assert 'Falling back to geopandas' in captured

    # 2b. Fallback silenced with silent=True
    capsys.readouterr()
    res_fallback_silent = overlay_polygons_with_duckdb(
        gdf1, gdf2, how='intersection', silent=True
    )
    assert len(res_fallback_silent) == 1
    assert capsys.readouterr().out == ''
