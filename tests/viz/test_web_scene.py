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
    # The pinned tooltip's close button is the third; the fourth picks
    # the coloring (per area or per dwelling, with or without
    # openplaces' estimates), which recolors every feature and hides
    # none. The fifth and sixth fold the controls away and back; the
    # seventh shares a link to the view (camera and settings only); the
    # eighth turns the view north up.
    assert len(re.findall(r"createElement\('button'\)", script)) == 8
    for kind in (
        'dataset.palette',
        'dataset.layer',
        'dataset.close',
        'dataset.choice',
        'dataset.fold',
        'dataset.share',
        'dataset.compass',
    ):
        assert kind in script
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
            'land_use_class': ['Single-Family', 'Recreation'],
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
            'occupancy_type': ['Single-Family'],
            'year_built': [1925],
            'structure_value': [300_000.0],
            'structure_value_source': ['assessor'],
            'n_dwellings': [3.0],
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
    # Per dwelling: its own value over its own dwellings.
    assert arrays['value_dwelling'][0] == pytest.approx(100_000.0)
    assert manifest['layers'][1]['views']['dwelling']['unit'] == 'dwelling'
    # A home: per dwelling, not on the non-residential ramp.
    assert list(arrays['nonresidential']) == [0]
    assert manifest['layers'][1]['views']['dwelling']['other']['unit'] == 'sqft'
    # Which values openplaces filled: the second parcel's land value.
    parcel_path = out / manifest['layers'][0]['files']['US-XX-BBB']['geometry']
    _, parcel_arrays = decode_scene_file(gzip.decompress(parcel_path.read_bytes()))
    assert list(parcel_arrays['estimated']) == [1]
    # A lot counts the dwellings of the home on it: 3, so $50,000 of
    # land is $16,667 per dwelling; the lot without homes has none.
    aaa_path = out / manifest['layers'][0]['files']['US-XX-AAA']['geometry']
    _, aaa = decode_scene_file(gzip.decompress(aaa_path.read_bytes()))
    assert aaa['value_dwelling'][0] == pytest.approx(50_000.0 / 3)
    assert np.isnan(parcel_arrays['value_dwelling'][0])
    assert manifest['layers'][0]['n_estimated'] == 1


def test_large_unclassed_lot_with_one_counted_home_is_not_residential(tmp_path):
    # Land use says Multi-Family, one outline counted as a home, but
    # neither the roll's own count nor its use group confirms it.
    lots = gpd.GeoDataFrame(
        {
            'land_use_class': ['Multi-Family', 'Multi-Family'],
            'use_group': [None, 'residential'],
            'n_dwellings': [np.nan, np.nan],
        },
        geometry=[box(0, 0, 1000, 1000), box(2000, 0, 2020, 20)],
        crs='EPSG:26986',
    )
    homes = gpd.GeoDataFrame(
        {'occupancy_type': ['Multi-Family', 'Multi-Family'], 'n_dwellings': [1.0, 3.0]},
        geometry=[box(500, 500, 510, 510), box(2005, 5, 2015, 15)],
        crs='EPSG:26986',
    )
    frames = {'parcels': lots, 'buildings': homes}
    web_scene._count_dwellings_from_above(frames, web_scene.VALUE_SCENE_LAYERS)
    assert list(frames['parcels']['n_dwellings']) == [1.0, 3.0]
    assert frames['parcels'][web_scene.ROLL_DWELLINGS].isna().all()


def test_public_land_with_a_stray_home_is_not_residential():
    # A home standing on a park lot: one source (the outline) says
    # dwelling, the roll's land use does not; the lot stays
    # non-residential (two sources must agree).
    assert 'Recreation' not in web_scene.RESIDENTIAL_LAND_USE
    assert 'Single-Family' in web_scene.RESIDENTIAL_LAND_USE


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
    # The chips' order; the first visit shows DEFAULT_PALETTE (Pastel).
    assert web_scene.DEFAULT_PALETTE == 'pastel'
    assert [p.id for p in palettes] == ['landscape', 'sunset', 'neon', 'pastel']
    manifest = web_scene.palette_manifest(
        palettes[1],
        web_scene.VALUE_SCENE_LAYERS,
        {'parcels': (1_000.0, 100_000.0), 'buildings': (20.0, 500.0)},
    )
    parcels = manifest['layers']['parcels']
    assert (parcels['vmin'], parcels['vmax']) == (1_000.0, 100_000.0)
    assert len(parcels['lut']) == 256
    # Opaque: translucent walls shade by draw order.
    assert manifest['layers']['buildings']['alpha'] == 255
    # Color stops: the ramp starts and ends on the first and last stop.
    landscape = web_scene.palette_manifest(palettes[0], web_scene.VALUE_SCENE_LAYERS)
    assert landscape['layers']['parcels']['lut'][0] == '#3f6e8c'
    assert landscape['layers']['parcels']['lut'][-1] == '#a8443f'
    assert landscape['layers']['parcels']['line'] is None
    assert landscape['blend'] == 'normal'
    # Neon: value-colored edges on dark, opaque, unshaded faces, blended
    # normally, so brightness does not follow how many faces overlap.
    neon = web_scene.palette_manifest(palettes[2], web_scene.VALUE_SCENE_LAYERS)
    buildings = neon['layers']['buildings']
    assert buildings['alpha'] == 255
    assert buildings['line_lut'][0] == '#0c4a1c'
    assert buildings['line_lut'][-1] == '#b54dff'
    assert len(buildings['line_lut']) == len(buildings['lut'])
    assert neon['material']['diffuse'] == 0.0
    assert neon['blend'] == 'normal'
    # The light page carries its own no-value and border colors.
    pastel = web_scene.palette_manifest(palettes[3], web_scene.VALUE_SCENE_LAYERS)
    assert '#ffffff' in pastel['background']
    assert pastel['no_value'][:3] == [212, 208, 201]
    # The land ramp starts at a faint tint, not white (no value is a
    # separate, darker gray).
    assert pastel['layers']['parcels']['lut'][0] == '#eef1f7'
    assert pastel['theme'] == 'light'


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
    places = json.loads((out / 'places.json').read_text(encoding='utf-8'))
    assert [p['slug'] for p in places] == ['aaa', 'bbb']


def test_place_menu_is_links_only():
    script = (STATIC / 'viewer.js').read_text(encoding='utf-8')
    start = script.index('function buildPlaceNav')
    body = script[start : script.index('\n}\n', start)]
    # The menu of other places and the back link are anchors to pages
    # built ahead of time; no form control, nothing fetched but the
    # site's list of places.
    assert "createElement('a')" in body
    assert not re.search(r"createElement\('(input|select|textarea|form|button)", body)
    assert 'fetch(' not in body


def test_border_paths_sit_on_the_nearest_parcel_ground():
    line = shapely.LineString([(0.0, 0.0), (0.001, 0.0)])
    draped = gpd.GeoDataFrame(
        geometry=[
            shapely.Polygon([(0, 0, 30), (0.0002, 0, 30), (0.0002, 0.0002, 30)]),
            shapely.Polygon([(0.001, 0, 50), (0.0012, 0, 50), (0.0012, 0.0002, 50)]),
        ],
        crs='EPSG:4326',
    )
    (path,) = web_scene.border_paths([line], draped)
    assert [z for _, _, z in path] == [30.0, 50.0]
    flat = draped.set_geometry(shapely.force_2d(draped.geometry))
    (path,) = web_scene.border_paths([line], flat)
    assert [z for _, _, z in path] == [0.0, 0.0]


def test_crop_keeps_whole_features_by_their_center():
    # Two lots in longitude and latitude; the area covers the first
    # one's center and only a corner of the second.
    lots = gpd.GeoDataFrame(
        geometry=[box(-71.10, 42.30, -71.09, 42.31), box(-71.08, 42.30, -71.07, 42.31)],
        crs='EPSG:4326',
    )
    area = box(-71.105, 42.295, -71.078, 42.315)
    kept = web_scene.crop_to_area(lots, area)
    assert list(kept.index) == [0]
    # Kept whole, not clipped to the area.
    assert kept.geometry.iloc[0].equals(lots.geometry.iloc[0])


def test_data_files_are_named_by_content_and_pruned(tmp_path):
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    first = web_scene._write_content_file(data_dir, 'parcels_US-XX-AAA', b'same bytes')
    again = web_scene._write_content_file(data_dir, 'parcels_US-XX-AAA', b'same bytes')
    other = web_scene._write_content_file(data_dir, 'parcels_US-XX-AAA', b'new bytes')
    # Same data, same name and the same file bytes (no gzip timestamp).
    assert first == again != other
    assert first.startswith('data/parcels_US-XX-AAA.') and first.endswith('.bin')
    assert gzip.decompress((tmp_path / first).read_bytes()) == b'same bytes'
    # An old build folder and an unused file go; kept files stay.
    old = data_dir / '20260101T000000Z'
    old.mkdir()
    (old / 'parcels_US-XX-AAA.bin').write_bytes(b'x')
    web_scene._prune_local_data(tmp_path, {other})
    assert (tmp_path / other).exists()
    assert not (tmp_path / first).exists()
    assert not old.exists()
    manifest = {
        'layers': [{'files': {'US-XX-AAA': {'geometry': other, 'tooltip': first}}}]
    }
    assert web_scene.manifest_files(manifest) == {other, first}


def test_file_list_uploads_drop_filters():
    # rclone refuses --files-from beside --filter rules.
    flags = web_scene._without_filters(web_scene.RCLONE_SFTP_FLAGS)
    assert '--filter' not in flags
    assert '--transfers' in flags


def test_local_only_scene_is_never_deployed(tmp_path):
    (tmp_path / 'index.html').write_text('<!doctype html>', encoding='utf-8')
    (tmp_path / 'scene.json').write_text(
        json.dumps({'build_id': 'x', 'preview': False, 'local_only': True}),
        encoding='utf-8',
    )
    with pytest.raises(ValueError, match='local use only'):
        web_scene.deploy_web_scene(tmp_path, go=False)


def test_identical_outlines_merge_into_one_feature():
    layer = web_scene.VALUE_SCENE_LAYERS[0]
    lot = box(0, 0, 10, 10)
    gdf = gpd.GeoDataFrame(
        {
            'land_value_imputed': [100.0, 50.0, 7.0],
            'land_value_imputed_source': ['assessor', 'assessor+imputed', 'assessor'],
        },
        geometry=[
            lot,
            shapely.Polygon(list(lot.exterior.coords)[::-1]),
            box(20, 0, 30, 10),
        ],
        crs='EPSG:26986',
    )
    out = web_scene.merge_identical_outlines(gdf, layer)
    assert len(out) == 2
    merged = out.iloc[0]
    assert merged['land_value_imputed'] == 150.0
    assert merged[web_scene.RECORDS_COLUMN] == 2
    assert 'imputed' in merged['land_value_imputed_source']
    assert pd.isna(out.iloc[1][web_scene.RECORDS_COLUMN])


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
