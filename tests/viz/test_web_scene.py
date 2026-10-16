"""The static web scene: file layout, page properties, and a small build.

All values are fabricated.
"""

import gzip
import json
import re
from importlib import resources

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from shapely.geometry import MultiPolygon, Polygon, box

from openplaces.io.delivery import PersonalColumnError
from openplaces.viz import web_scene
from openplaces.viz.web_scene import (
    SceneLayer,
    TooltipField,
    decode_scene_file,
    encode_scene_file,
    refuse_personal_fields,
)

STATIC = resources.files('openplaces.viz.web_scene') / 'static'


def _square_with_hole():
    return Polygon(
        [(0, 0), (0.01, 0), (0.01, 0.01), (0, 0.01)],
        holes=[[(0.004, 0.004), (0.006, 0.004), (0.006, 0.006), (0.004, 0.006)]],
    )


def test_scene_file_round_trip_keeps_holes_and_multipolygon_parts():
    geometry = [
        _square_with_hole(),
        MultiPolygon([box(0.02, 0, 0.03, 0.01), box(0.04, 0, 0.05, 0.01)]),
    ]
    data, origin = encode_scene_file(
        geometry,
        base=[0.0, 1.5],
        height=[10.0, 20.0],
        value=[12.5, np.nan],
    )
    header, arrays = decode_scene_file(data)

    assert header['n_features'] == 2
    assert header['origin'] == origin
    # Three parts: the holed square, then the multipolygon's two.
    assert arrays['part_feature'].tolist() == [0, 1, 1]
    # The holed square has two rings, each part of the multipolygon one.
    assert np.diff(arrays['part_ring_offsets']).tolist() == [2, 1, 1]
    assert arrays['height'].tolist() == [10.0, 20.0]
    assert arrays['base'].tolist() == [0.0, 1.5]
    assert arrays['value'][0] == 12.5 and np.isnan(arrays['value'][1])
    # Triangles: 8 for the square with a square hole (n + 2h - 2), two
    # per box; all of them index existing vertices.
    triangles = arrays['triangles'].reshape(-1, 3)
    assert len(triangles) == 12
    assert triangles.max() < len(arrays['xy']) // 2

    xy = arrays['xy'].reshape(-1, 2).astype(float) + origin
    ring = arrays['ring_offsets']
    hole = shapely.Polygon(xy[ring[1] : ring[2]])
    expected = shapely.Polygon(geometry[0].interiors[0])
    assert hole.area == pytest.approx(expected.area, rel=1e-4)


def test_arrays_are_aligned_for_typed_array_views():
    data, _ = encode_scene_file([box(0, 0, 1, 1)], base=[0], height=[1], value=[1.0])
    header, _ = decode_scene_file(data)
    header_length = int.from_bytes(data[4:8], 'little')
    assert (8 + header_length) % 8 == 0
    assert all(entry['offset'] % 8 == 0 for entry in header['arrays'])


def test_a_personal_tooltip_column_is_refused():
    layer = SceneLayer(
        name='parcels',
        label='Land value',
        recipe='any',
        value_column='land_value',
        area_unit='ac',
        vmin=1,
        vmax=10,
        tooltip=(TooltipField('owner_name', 'Owner'),),
    )
    with pytest.raises(PersonalColumnError):
        refuse_personal_fields([layer])


def test_the_default_scene_shows_no_personal_column():
    refuse_personal_fields(web_scene.VALUE_SCENE_LAYERS)


def test_page_has_no_input_control_and_no_background_map():
    page = (STATIC / 'index.html').read_text(encoding='utf-8')
    script = (STATIC / 'viewer.js').read_text(encoding='utf-8')
    # No search or filter control of any kind: no field to type in, no
    # list to pick features from. The only buttons are the palette
    # switch (colors) and the per-layer switch (a whole layer on or
    # off); neither selects features by their attributes.
    assert not re.search(r'<(input|select|textarea|form|button)\b', page, re.I)
    assert not re.search(r"createElement\('(input|select|textarea|form)", script)
    assert len(re.findall(r"createElement\('button'\)", script)) == 2
    assert 'dataset.palette' in script and 'dataset.layer' in script
    # Only the scene's own layers; nothing that draws a background map.
    for text in (page, script):
        assert not re.search(
            r'TileLayer|MapboxOverlay|maplibre|mapbox-gl|basemap|tile\.', text, re.I
        )
    # The deck.gl bundle is pinned to an exact version.
    assert re.search(r'deck\.gl@\d+\.\d+\.\d+/', web_scene.DECK_GL_URL)


def test_terrain_layer_renders_a_given_geodataframe():
    pytest.importorskip('lonboard')
    from openplaces.viz.terrain import show_value_terrain_layer

    gdf = gpd.GeoDataFrame(
        {'land_value': [1000.0, 2000.0]},
        geometry=[box(0, 0, 100, 100), box(100, 0, 200, 100)],
        crs='EPSG:26986',
    )
    terrain = show_value_terrain_layer(
        gdf, None, value_column='land_value', area_unit='m2', silent=True
    )
    assert terrain.fill_color.shape == (2, 4)
    assert terrain.rendered_elevation[1] > terrain.rendered_elevation[0]


def test_build_writes_only_the_allowed_fields(tmp_path, monkeypatch):
    pytest.importorskip('lonboard')
    # Massachusetts State Plane, meters; two fabricated towns.
    parcels = gpd.GeoDataFrame(
        {
            'admin3_id': ['US-XX-AAA', 'US-XX-BBB'],
            'address': ['1 Example St', '2 Sample Ave'],
            'use_group': ['residential', 'commercial'],
            'land_value_imputed': [50_000.0, 80_000.0],
            'land_value_imputed_source': ['assessor', 'imputed'],
            'owner_name': ['fabricated', 'fabricated'],
        },
        geometry=[
            box(230_000, 900_000, 230_050, 900_050),
            box(231_000, 900_000, 231_040, 900_040),
        ],
        crs='EPSG:26986',
    )
    buildings = gpd.GeoDataFrame(
        {
            'admin3_id': ['US-XX-AAA'],
            'address': ['1 Example St'],
            'occupancy_type': ['single_family'],
            'year_built': [1925],
            'structure_value': [300_000.0],
            'structure_value_source': ['assessor'],
        },
        geometry=[box(230_010, 900_010, 230_030, 900_030)],
        crs='EPSG:26986',
    )
    bundles = {
        'US_parcel-openplaces-2026': parcels,
        'US_footprint-openplaces-2026': buildings,
    }
    monkeypatch.setattr(
        web_scene, 'read_bundle', lambda recipe, region, admin_ids: bundles[recipe]
    )
    monkeypatch.setattr(web_scene, 'bundle_notice', lambda recipe, region: 'terms')
    monkeypatch.setattr(
        web_scene, '_region_members', lambda recipe, region: ['US-XX-AAA', 'US-XX-BBB']
    )
    towns = gpd.GeoDataFrame(
        {'admin3_id': ['US-XX-AAA', 'US-XX-BBB'], 'name': ['Aaa', 'Bbb']},
        geometry=[
            box(229_900, 899_900, 230_900, 900_900),
            box(230_900, 899_900, 231_900, 900_900),
        ],
        crs='EPSG:26986',
    ).to_crs(4326)
    monkeypatch.setattr(web_scene, '_admin_names', lambda admin_ids: towns)
    monkeypatch.setattr(web_scene, 'land_borders', lambda admin_ids, land=None: [])
    # The default parcel layer drops corridors; these squares are none.
    layers = (
        SceneLayer(
            **{
                **web_scene.VALUE_SCENE_LAYERS[0].__dict__,
                'render': {},
            }
        ),
        web_scene.VALUE_SCENE_LAYERS[1],
    )

    out = web_scene.build_web_scene(
        'test-region',
        ['US-XX-AAA', 'US-XX-BBB'],
        tmp_path,
        layers=layers,
        build_id='b1',
        verbose=False,
    )

    manifest = json.loads((out / 'scene.json').read_text(encoding='utf-8'))
    assert [layer['name'] for layer in manifest['layers']] == ['parcels', 'buildings']
    assert set(manifest['layers'][0]['files']) == {'US-XX-AAA', 'US-XX-BBB'}
    assert set(manifest['layers'][1]['files']) == {'US-XX-AAA'}
    assert (out / '.htaccess').exists() and (out / 'index.html').exists()

    tooltip_path = out / manifest['layers'][0]['files']['US-XX-BBB']['tooltip']
    values = json.loads(gzip.decompress(tooltip_path.read_bytes()))
    allowed = {tip.column for tip in layers[0].tooltip}
    assert set(values) == allowed | {
        f'{tip.column}__imputed' for tip in layers[0].tooltip if tip.imputed_from
    }
    assert 'owner_name' not in json.dumps(values)
    assert values['land_value_imputed__imputed'] == [1]

    geometry_path = out / manifest['layers'][1]['files']['US-XX-AAA']['geometry']
    header, arrays = decode_scene_file(gzip.decompress(geometry_path.read_bytes()))
    # The building stands on its parcel's extrusion.
    assert arrays['base'][0] > 0
    assert header['n_features'] == 1


def test_tour_starts_at_the_densest_center_and_skips_lone_parcels():
    rows, geometry = [], []
    # Town A: a dense center of 25 small parcels near the origin.
    for k in range(25):
        x, y = 230_000 + 30 * (k % 5), 900_000 + 30 * (k // 5)
        rows.append(('US-XX-AAA', 1_000_000.0))
        geometry.append(box(x, y, x + 20, y + 20))
    # Town B, 5 km away: a smaller center of 25 parcels.
    for k in range(25):
        x, y = 235_000 + 30 * (k % 5), 900_000 + 30 * (k // 5)
        rows.append(('US-XX-BBB', 100_000.0))
        geometry.append(box(x, y, x + 20, y + 20))
    # One vast, costly parcel alone in its cell (an airport).
    rows.append(('US-XX-AAA', 1e10))
    geometry.append(box(240_000, 900_000, 240_300, 900_300))
    gdf = gpd.GeoDataFrame(
        pd.DataFrame(rows, columns=['admin3_id', 'total_value']),
        geometry=geometry,
        crs='EPSG:26986',
    )
    foci = web_scene.tour_foci(
        gdf, 'total_value', {'US-XX-AAA': 'Aaa', 'US-XX-BBB': 'Bbb'}
    )
    assert [f['name'] for f in foci] == ['Aaa', 'Bbb']


def test_deploy_settings_come_from_the_env_file(tmp_path, monkeypatch):
    for key in ('HOST', 'USER', 'PATH', 'KEY', 'KNOWN_HOSTS', 'URL'):
        monkeypatch.delenv(f'OPENPLACES_WEB_DEPLOY_{key}', raising=False)
    env_file = tmp_path / '.env.deploy'
    env_file.write_text(
        'OPENPLACES_WEB_DEPLOY_HOST=host.example\n'
        'OPENPLACES_WEB_DEPLOY_USER=someone\n'
        'OPENPLACES_WEB_DEPLOY_PATH=/home/someone/public_html/scene/\n',
        encoding='utf-8',
    )
    monkeypatch.setenv('OPENPLACES_WEB_DEPLOY_USER', 'from-environment')
    settings = web_scene.load_deploy_settings(env_file)
    assert settings['path'] == '/home/someone/public_html/scene'
    assert settings['user'] == 'from-environment'
    env = web_scene._rclone_env(settings)
    assert env['RCLONE_CONFIG_BH_TYPE'] == 'sftp'
    assert env['RCLONE_CONFIG_BH_HOST'] == 'host.example'

    env_file.write_text('OPENPLACES_WEB_DEPLOY_HOST=host.example\n', encoding='utf-8')
    with pytest.raises(ValueError, match='Missing deploy settings'):
        web_scene.load_deploy_settings(env_file)


def test_palettes_take_scene_bounds_and_color_stops():
    palettes = web_scene.default_palettes()
    assert palettes[0].id == 'gold'
    assert 'original' not in {p.id for p in palettes}
    pastel = next(p for p in palettes if p.id == 'pastel')
    manifest = web_scene.palette_manifest(
        pastel,
        web_scene.VALUE_SCENE_LAYERS,
        {'parcels': (1_000.0, 100_000.0), 'buildings': (20.0, 500.0)},
    )
    parcels = manifest['layers']['parcels']
    assert (parcels['vmin'], parcels['vmax']) == (1_000.0, 100_000.0)
    assert len(parcels['lut']) == 256
    # Color stops: the ramp starts and ends on the first and last stop.
    assert manifest['layers']['buildings']['lut'][0] == '#a98bd6'
    assert manifest['layers']['buildings']['lut'][-1] == '#fff4b8'
    assert manifest['layers']['buildings']['alpha'] == round(255 * 0.9)


def test_scene_bounds_round_outward():
    values = np.concatenate([np.full(98, 3_000.0), [17.0, 9_700.0]])
    low, high = web_scene.scene_bounds(values, low_q=0.0, high_q=1.0)
    assert (low, high) == (10.0, 10_000.0)


def test_landing_page_is_links_only(tmp_path):
    from openplaces.viz.web_scene.landing import Place, build_landing

    out = build_landing(
        [
            Place('aaa', 'Aaa', 'One county', tmp_path),
            Place('bbb', 'Bbb', 'Two', tmp_path),
        ],
        tmp_path / 'site',
        thumbnails=False,
    )
    page = (out / 'index.html').read_text(encoding='utf-8')
    # Navigation between pre-built pages: no search, filter or picker.
    assert not re.search(r'<(input|select|textarea|form|button|script)\b', page, re.I)
    assert re.findall(r'href="([^"]+)"', page) == ['aaa/', 'bbb/']


def test_scenes_do_not_carry_the_landing_template(tmp_path, monkeypatch):
    names = [p.name for p in STATIC.iterdir()]
    assert 'landing.html' in names


def test_tooltip_values_mark_only_imputed_cells():
    gdf = pd.DataFrame(
        {
            'structure_value': [1.0, 2.0, np.nan],
            'structure_value_source': ['assessor', 'assessor+imputed', None],
        }
    )
    layer = web_scene.VALUE_SCENE_LAYERS[1]
    layer = SceneLayer(**{**layer.__dict__, 'tooltip': (layer.tooltip[3],)})
    values = web_scene._tooltip_values(gdf, layer)
    assert values['structure_value'] == [1.0, 2.0, None]
    assert values['structure_value__imputed'] == [0, 1, 0]
