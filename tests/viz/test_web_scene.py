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
    # eighth turns the view north up; the ninth, in the menu under
    # "Modeled" on a team page, picks which model colors the lots; the
    # tenth, beside it, the models' scale (to the assessments, or in
    # today's dollars); the eleventh replays the guided tour; the
    # twelfth sets the color bounds (local, or nationwide).
    assert len(re.findall(r"createElement\('button'\)", script)) == 12
    for kind in (
        'dataset.palette',
        'dataset.layer',
        'dataset.close',
        'dataset.choice',
        'dataset.fold',
        'dataset.share',
        'dataset.compass',
        'dataset.replay',
        'dataset.range',
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
    monkeypatch.setattr(web_scene, 'land_borders', lambda admin_ids, **kwargs: [])
    monkeypatch.setattr(web_scene, 'lake_polygons', lambda admin_ids: None)
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


def test_sources_document_drops_the_recipe_folders():
    notice = '\n'.join(
        [
            'Data licence and attribution for US_footprint-openplaces-2026 (US-XX)',
            '=' * 70,
            '',
            'Sources',
            '-' * 70,
            ' 97.0%  obm                  ODbL-1.0',
            '                             https://opendatacommons.org/licenses/odbl/1-0/',
            '  3.0%  examplegis (US-XX_footprint-examplegis-2026) Public domain. '
            'Credit requested: "Example GIS".',
            '',
            'Recipes',
            '-' * 70,
            'openplaces 0.1.0',
            '        C:\\Users\\someone\\Code\\openplaces\\src\\openplaces\\recipes',
        ]
    )
    doc = web_scene.notice_sources(notice + '\n\n' + notice)
    # Folders on the builder's machine never reach the page.
    assert 'Users' not in json.dumps(doc)
    names = [s['name'] for s in doc['sources']]
    assert names == ['OpenBuildingMap', 'examplegis']
    assert doc['sources'][0]['share'] == pytest.approx(0.97)
    assert doc['sources'][0]['url'].startswith('https://opendatacommons.org')
    assert 'OpenStreetMap contributors' in doc['sources'][0]['credit']
    assert doc['sources'][1]['terms'] == 'Public domain.'
    assert 'ODbL' in doc['share_alike'][0]


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
    # The chips' order: the first visit's palette, DEFAULT_PALETTE
    # (Pastel), comes first.
    assert web_scene.DEFAULT_PALETTE == 'pastel'
    assert [p.id for p in palettes] == ['pastel', 'landscape', 'sunset', 'neon']
    by_id = {p.id: p for p in palettes}
    manifest = web_scene.palette_manifest(
        by_id['sunset'],
        web_scene.VALUE_SCENE_LAYERS,
        {'parcels': (1_000.0, 100_000.0), 'buildings': (20.0, 500.0)},
    )
    parcels = manifest['layers']['parcels']
    assert (parcels['vmin'], parcels['vmax']) == (1_000.0, 100_000.0)
    assert len(parcels['lut']) == 256
    # Opaque: translucent walls shade by draw order.
    assert manifest['layers']['buildings']['alpha'] == 255
    # Color stops: the ramp starts and ends on the first and last stop.
    landscape = web_scene.palette_manifest(
        by_id['landscape'], web_scene.VALUE_SCENE_LAYERS
    )
    assert landscape['layers']['parcels']['lut'][0] == '#3f6e8c'
    assert landscape['layers']['parcels']['lut'][-1] == '#a8443f'
    assert landscape['layers']['parcels']['line'] is None
    assert landscape['blend'] == 'normal'
    # Neon: value-colored edges on dark, opaque, unshaded faces, blended
    # normally, so brightness does not follow how many faces overlap.
    neon = web_scene.palette_manifest(by_id['neon'], web_scene.VALUE_SCENE_LAYERS)
    buildings = neon['layers']['buildings']
    assert buildings['alpha'] == 255
    assert buildings['line_lut'][0] == '#0c4a1c'
    assert buildings['line_lut'][-1] == '#b54dff'
    assert len(buildings['line_lut']) == len(buildings['lut'])
    assert neon['material']['diffuse'] == 0.0
    assert neon['blend'] == 'normal'
    # The light page carries its own no-value and border colors.
    pastel = web_scene.palette_manifest(by_id['pastel'], web_scene.VALUE_SCENE_LAYERS)
    assert '#ffffff' in pastel['background']
    assert pastel['no_value'][:3] == [212, 208, 201]
    # The land ramp starts at a faint tint, not white (no value is a
    # separate, darker gray).
    assert pastel['layers']['parcels']['lut'][0] == '#eef1f7'
    assert pastel['theme'] == 'light'


def test_imputed_twin_supplies_only_the_models_cells(tmp_path, monkeypatch):
    # Fabricated: three buildings. The regular bundle has a recorded
    # value for the first two; its imputed twin keeps the first,
    # replaces the second and fills the third, marking only those two.
    regular_dir = tmp_path / 'region-x'
    twin_dir = tmp_path / 'region-x.imputed'
    regular_dir.mkdir()
    twin_dir.mkdir()
    name = 'US-XX_footprint-openplaces-2026.parquet'
    notice = 'US-XX_footprint-openplaces-2026_LICENSE.txt'
    monkeypatch.setattr(
        web_scene,
        'delivery_paths',
        lambda recipe, region: {
            'canonical': regular_dir / name,
            'terms': regular_dir / notice,
        },
    )
    # A fabricated recipe without the height check.
    monkeypatch.setattr(web_scene, 'ghost_rule', lambda recipe: None)
    # The twin's notice names the model's code; the page reads it from
    # there (fabricated commit).
    (twin_dir / notice).write_text(
        'Imputed values\n... Model code: openplaces branch\nbuilding-imputation, '
        'commit 0123abcd.\n',
        encoding='utf-8',
    )
    code = web_scene.imputed_model_code('r', 'region-x')
    assert code == [
        'Model code: openplaces branch building-imputation, commit 0123abcd.'
    ]
    assert web_scene.imputation_note(code + code)['credit'].endswith('commit 0123abcd.')
    index = pd.Index(['a', 'b', 'c'], name='footprint_id')
    part = pd.DataFrame(
        {
            'structure_value': [100.0, 0.0, np.nan],
            'structure_value_source': ['roll', 'roll', None],
        },
        index=index,
    )
    pd.DataFrame(
        {
            'structure_value': [100.0, 250.0, 300.0],
            'structure_value_source': [
                'roll',
                'building-imputation-v0.3+imputed',
                'building-imputation-v0.3+imputed',
            ],
        },
        index=index,
    ).to_parquet(twin_dir / name)
    out, taken = web_scene.overlay_imputed_values(
        part, 'r', 'region-x', ('structure_value',)
    )
    assert taken == 2
    assert list(out['structure_value']) == [100.0, 250.0, 300.0]
    assert list(out['structure_value_source'].astype(str)) == [
        'roll',
        'building-imputation-v0.3+imputed',
        'building-imputation-v0.3+imputed',
    ]
    # The page marks them estimated through the usual marker.
    from openplaces.core.provenance import is_imputed

    assert list(is_imputed(out['structure_value_source'])) == [False, True, True]
    # Without a twin, nothing changes.
    (twin_dir / name).unlink()
    same, none = web_scene.overlay_imputed_values(
        part, 'r', 'region-x', ('structure_value',)
    )
    assert none == 0 and same is part


def test_lot_total_value_falls_back_to_land_and_improvement():
    # Fabricated lots: a roll total; none but land and improvement;
    # land alone; nothing.
    lots = pd.DataFrame(
        {
            'total_value': [500.0, np.nan, np.nan, np.nan],
            'land_value': [100.0, 200.0, 300.0, np.nan],
            'improvement_value': [400.0, 50.0, np.nan, np.nan],
            'land_value_imputed': [100.0, 200.0, 300.0, 7.0],
        }
    )
    total = web_scene.lot_total_value(lots, 'land_value_imputed')
    assert list(total) == [500.0, 250.0, 300.0, 7.0]


def test_type_codes_give_lots_and_buildings_one_list_of_types():
    # A lot class and a building class of the same kind share a type;
    # an unlisted class and a missing one are unknown (code 0).
    classes = pd.Series(
        ['Single-Family', 'Low-Rise Multi-Family', 'Multi-Family', 'Mystery', None]
    )
    codes = web_scene.type_codes(classes)
    keys = [web_scene.TYPE_CATEGORIES[c]['key'] for c in codes]
    assert codes.dtype == np.uint8
    assert keys == [
        'single_family',
        'multi_family',
        'multi_family',
        'unknown',
        'unknown',
    ]


def test_type_categories_name_each_class_once():
    values = [v for c in web_scene.TYPE_CATEGORIES for v in c['values']]
    assert len(values) == len(set(values))
    assert web_scene.TYPE_CATEGORIES[0]['key'] == 'unknown'


def test_scene_bounds_follow_the_mean_and_spread_and_round_outward():
    # Fabricated: 98 lots at $1,000 and two outliers, $1 and $1M. On a
    # log scale the mean is 3 and the standard deviation 0.42, so the
    # ramp spans about $141 to $7,079, rounded outward; the outliers
    # saturate the ends instead of stretching it.
    values = np.concatenate([np.full(98, 1_000.0), [1.0, 1_000_000.0]])
    assert web_scene.scene_bounds(values) == (100.0, 10_000.0)
    # Never wider than the values themselves.
    assert web_scene.scene_bounds([100.0, 1_000.0, 10_000.0]) == (100.0, 10_000.0)
    # Every display unit has nationwide bounds.
    assert set(web_scene.NATIONAL_BOUNDS) >= {'ac', 'sqft', 'dwelling'}


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
    assert re.findall(r'<a\b[^>]*href="([^"]+)"', page) == ['aaa/', 'bbb/']
    # The logo as its icon, copied beside it.
    assert all((out / icon).exists() for icon in web_scene.ICON_FILES)
    places = json.loads((out / 'places.json').read_text(encoding='utf-8'))
    assert [p['slug'] for p in places] == ['aaa', 'bbb']
    # The methods page beside it, plain text and links, with the
    # section the unexplained buildings' switch links to.
    methods = (out / 'methods.html').read_text(encoding='utf-8')
    assert not re.search(
        r'<(input|select|textarea|form|button|script)\b', methods, re.I
    )
    assert 'id="unexplained"' in methods
    assert '../methods.html#unexplained' in (STATIC / 'viewer.js').read_text(
        encoding='utf-8'
    )


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


def test_contrast_focus_lies_between_the_dearest_and_cheapest_nearby_homes():
    # Three blocks of fabricated homes, 1 km apart in a row: $200K,
    # $250K and $2M per dwelling. The sharpest contrast within 1.2 km is
    # between the second and third blocks.
    lots, values = [], []
    for k, value in enumerate([200_000, 250_000, 2_000_000]):
        for n in range(50):
            x = 230_000 + 1000 * k + (n % 10) * 20
            y = 900_000 + (n // 10) * 20
            lots.append(box(x, y, x + 10, y + 10))
            values.append(value)
    gdf = gpd.GeoDataFrame(
        {'land_value_per_dwelling': values}, geometry=lots, crs='EPSG:26986'
    )
    focus = web_scene.contrast_focus(
        gdf, 'land_value_per_dwelling', cell_m=500, radius_m=1200, min_homes=40
    )
    assert focus['ratio'] == pytest.approx(8.0)
    x, y = (
        gpd.GeoSeries(
            gpd.points_from_xy([focus['longitude']], [focus['latitude']]), crs=4326
        )
        .to_crs(26986)
        .iloc[0]
        .coords[0]
    )
    assert 231_500 < x < 232_500
    assert focus['km'] == pytest.approx(1.0) and focus['claim']
    # From a downtown beside the first two blocks, their milder contrast
    # (1.25 times) is preferred over the sharper one 2 km away.
    downtown = (
        gpd.GeoSeries(gpd.points_from_xy([230_600], [900_050]), crs=26986)
        .to_crs(4326)
        .iloc[0]
    )
    focus = web_scene.contrast_focus(
        gdf,
        'land_value_per_dwelling',
        cell_m=500,
        radius_m=1200,
        min_homes=40,
        near=(downtown.x, downtown.y),
        near_m=800,
    )
    assert focus['near'] and focus['ratio'] == pytest.approx(1.2, abs=0.05)
    # The claim holds only if the colors on screen agree: land per home
    # higher on the costly side.
    agree = gdf.assign(land=np.where(np.array(values) > 1e6, 90_000.0, 30_000.0))
    focus = web_scene.contrast_focus(
        agree,
        'land_value_per_dwelling',
        cell_m=500,
        radius_m=1200,
        min_homes=40,
        shown='land',
    )
    assert (focus['high_shown'], focus['low_shown'], focus['claim']) == (
        90_000,
        30_000,
        True,
    )
    reverse = agree.assign(land=120_000 - agree['land'])
    focus = web_scene.contrast_focus(
        reverse,
        'land_value_per_dwelling',
        cell_m=500,
        radius_m=1200,
        min_homes=40,
        shown='land',
    )
    assert not focus['claim']
    # Too small a gap to say aloud.
    close = gdf.assign(
        land_value_per_dwelling=np.where(np.array(values) > 1e6, 500_000, 200_000)
    )
    focus = web_scene.contrast_focus(
        close, 'land_value_per_dwelling', cell_m=500, radius_m=1200, min_homes=40
    )
    assert focus['ratio'] < web_scene.CONTRAST_CLAIM_RATIO and not focus['claim']
    # Homes only: no per-dwelling values, no stop.
    assert (
        web_scene.contrast_focus(
            gdf.assign(land_value_per_dwelling=np.nan), 'land_value_per_dwelling'
        )
        is None
    )


def test_buildings_on_exempt_lots_are_not_homes_unless_the_roll_counts_them():
    # Fabricated: an exempt lot (an armory), an exempt lot whose roll
    # counts 40 dwellings (public housing), and a residential lot, each
    # with one building classed "Multi-Family".
    lots = gpd.GeoDataFrame(
        {
            'use_group': ['exempt', 'exempt', 'residential'],
            'n_dwellings': [np.nan, 40.0, np.nan],
        },
        geometry=[box(0, 0, 100, 100), box(200, 0, 300, 100), box(400, 0, 500, 100)],
        crs='EPSG:26986',
    )
    buildings = gpd.GeoDataFrame(
        {
            web_scene.BUILDING_CLASS: ['Multi-Family'] * 3,
            'n_dwellings': [12.0, 40.0, 3.0],
        },
        geometry=[box(40, 40, 60, 60), box(240, 40, 260, 60), box(440, 40, 460, 60)],
        crs='EPSG:26986',
    )
    gdfs = {'parcels': lots, 'buildings': buildings}
    scale = {
        'recipe': 'r',
        'value_column': 'v',
        'area_unit': 'sqft',
        'vmin': 1,
        'vmax': 2,
    }
    layers = [
        SceneLayer(name='parcels', label='Land', **scale),
        SceneLayer(name='buildings', label='B', stack_on='parcels', **scale),
    ]
    web_scene._mark_exempt_ground(gdfs, layers)
    assert list(gdfs['buildings'][web_scene.BUILDING_CLASS]) == [
        web_scene.EXEMPT_CLASS,
        'Multi-Family',
        'Multi-Family',
    ]


def test_local_command_never_uploads():
    from openplaces.viz.web_scene import local

    source = (STATIC.parent / 'local.py').read_text(encoding='utf-8')
    # No deploy code at all: no upload path to take by mistake.
    assert not re.search(r'deploy_web_scene\(|deploy_landing|rclone|ssh', source, re.I)
    with pytest.raises(SystemExit, match='never uploads'):
        local.main(['boston', '--deploy'])
    # Every place has a state and a center, or whole units and the
    # line its landing card shows; the order is editorial.
    places = local.load_places()
    assert places['radius_km'] > places['phone_radius_km'] > 0
    for p in places['places']:
        assert p['state']
        if local.has_circle(p):
            assert len(p['center']) == 2 and p['around']
        else:
            assert p['admin_ids'] and p['extent_note'] and 'center' not in p
    # A team page is a separate place, never the public one.
    team = [p for p in places['places'] if p.get('team')]
    assert all(p['slug'].endswith('-team') for p in team)


def test_places_are_listed_by_parcel_count(tmp_path):
    from openplaces.viz.web_scene import site

    # Fabricated pages: one records its count; one, built earlier, has
    # only its tooltip file (three lots).
    new = tmp_path / 'new'
    new.mkdir()
    (new / 'scene.json').write_text(
        json.dumps({'layers': [{'n_features': 5}]}), encoding='utf-8'
    )
    old = tmp_path / 'old'
    (old / 'data').mkdir(parents=True)
    (old / 'data' / 'tip.bin').write_bytes(
        gzip.compress(json.dumps({'address': ['a', 'b', 'c']}).encode())
    )
    (old / 'scene.json').write_text(
        json.dumps({'layers': [{'files': {'US-XX-A': {'tooltip': 'data/tip.bin'}}}]}),
        encoding='utf-8',
    )
    assert site.parcel_count(new) == 5
    assert site.parcel_count(old) == 3


def test_site_publishes_each_phone_page_first_and_the_landing_last(monkeypatch):
    from openplaces.viz.web_scene import site

    calls = []
    monkeypatch.setattr(
        site,
        'deploy_web_scene',
        lambda folder, go, subpath: calls.append((subpath, go)),
    )
    monkeypatch.setattr(
        site, 'deploy_landing', lambda root, go: calls.append(('/', go))
    )
    monkeypatch.setattr(site.time, 'sleep', lambda seconds: None)
    site.main(['publish', 'durham', 'nantucket', 'boston'])
    # A dry run unless --go; the phone page before the page that sends
    # phones to it; a place of whole units has none; the landing page
    # last.
    assert calls == [
        ('durham-phone', False),
        ('durham', False),
        ('nantucket', False),
        ('boston-phone', False),
        ('boston', False),
        ('/', False),
    ]
    # `--all` leaves team pages out; they go up only when named.
    calls.clear()
    site.main(['publish', '--all'])
    assert calls and not any(slug.endswith('-team') for slug, _ in calls)
    with pytest.raises(SystemExit, match='Not in places.yaml'):
        site.main(['publish', 'atlantis'])
    with pytest.raises(SystemExit, match='--all'):
        site.main(['build'])


def test_value_sources_by_unit_set_withheld_values_apart():
    # Fabricated: one county's values from the roll, another's withheld.
    provenance = pd.Series(['parcel.massgis'] * 3 + ['withheld'] * 2)
    units = pd.Series(['US-XX-A'] * 3 + ['US-XX-B'] * 2)
    by_unit = web_scene.value_sources_by_unit(
        provenance, units, {'US-XX-A': 'County A', 'US-XX-B': 'County B'}
    )
    assert [u['unit'] for u in by_unit] == ['County A', 'County B']
    assert by_unit[0]['values'][0]['name'] == 'MassGIS'
    withheld = by_unit[1]['values'][0]
    assert withheld['name'] == web_scene.WITHHELD_BY_TERMS and withheld['withheld']
    assert 'withheld' not in by_unit[0]['values'][0]
    # A unit with features but no value at all says so.
    provenance = pd.Series(['parcel.massgis', None])
    units = pd.Series(['US-XX-A', 'US-XX-C'])
    by_unit = web_scene.value_sources_by_unit(
        provenance, units, {'US-XX-A': 'County A', 'US-XX-C': 'County C'}
    )
    assert by_unit[1]['values'][0]['name'] == web_scene.NO_VALUES_IN_BUNDLE


def test_value_sources_are_counted_apart_from_outline_sources():
    provenance = pd.Series(
        ['parcel.massgis'] * 6
        + ['parcel.share_class_city+imputed'] * 3
        + ['parcel']
        + [None] * 5
    )
    shares = {s['name']: s['share'] for s in web_scene.value_sources(provenance)}
    assert shares == pytest.approx(
        {
            'MassGIS': 0.6,
            web_scene.FILLED_BY_OPENPLACES: 0.3,
            'Assessor roll, via the lot (source not recorded)': 0.1,
        }
    )
    assert 'section.values' in (STATIC / 'viewer.js').read_text(encoding='utf-8')


def test_known_issues_follow_the_scene_units(tmp_path):
    issues = tmp_path / 'issues.yaml'
    issues.write_text(
        'issues:\n'
        '  - admin_ids: [US-MA-AAA]\n'
        '    found: 2026-01-02\n'
        '    text: >-\n'
        '      Town A: a fabricated\n'
        '      problem.\n'
        '  - admin_ids: all\n'
        '    found: 2026-01-01\n'
        '    text: Everywhere.\n'
        '  - admin_ids: [US-NC-BBB]\n'
        '    found: 2026-01-03\n'
        '    text: Elsewhere.\n',
        encoding='utf-8',
    )
    assert web_scene.known_issues(['US-MA-AAA', 'US-MA-CCC'], path=issues) == [
        {'found': '2026-01-02', 'text': 'Town A: a fabricated problem.'},
        {'found': '2026-01-01', 'text': 'Everywhere.'},
    ]
    # The bundled list parses, and the Sources panel renders it.
    assert isinstance(web_scene.known_issues(['US-MA-SOM']), list)
    assert 'scene.known_issues' in (STATIC / 'viewer.js').read_text(encoding='utf-8')


def test_town_border_breaks_at_a_lake_and_not_at_a_road(monkeypatch):
    # Two towns meeting along x = 1000 (meters from an origin in
    # Massachusetts). Lots line the border except for a 60 m road
    # crossing at y = 400-460; a pond covers y = 700-850.
    x0, y0 = 230_000, 899_000

    def shifted(xmin, ymin, xmax, ymax):
        return box(x0 + xmin, y0 + ymin, x0 + xmax, y0 + ymax)

    towns = gpd.GeoDataFrame(
        {'admin3_id': ['US-MA-AAA', 'US-MA-BBB']},
        geometry=[shifted(0, 0, 1000, 1000), shifted(1000, 0, 2000, 1000)],
        crs='EPSG:26986',
    )
    monkeypatch.setattr(web_scene, 'get_admin', lambda *args, **kwargs: towns)
    lots = gpd.GeoDataFrame(
        geometry=[shifted(900, 0, 990, 400), shifted(900, 460, 990, 1000)],
        crs='EPSG:26986',
    )
    pond = gpd.GeoSeries([shifted(950, 700, 1050, 850)], crs='EPSG:26986')
    borders = web_scene.land_borders(
        ['US-MA-AAA', 'US-MA-BBB'], land=lots, lakes=pond.to_crs(4326)
    )
    lengths = sorted(gpd.GeoSeries(borders, crs=4326).to_crs(26986).length)
    # One stretch across the road up to the pond, one above it (the
    # 5 m outline tolerance adds a little at the ends).
    assert lengths == pytest.approx([150, 700], abs=15)


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


def test_a_scene_reading_a_team_bundle_is_a_team_scene():
    assert web_scene.scene_audience('nantucket-ma.team') == 'team'
    assert web_scene.scene_audience(['western-nc', 'nantucket-ma.team']) == 'team'
    assert web_scene.scene_audience(['western-nc']) == 'public'
    assert web_scene.scene_audience(None) == 'public'


def test_dollars_factors_extend_a_short_county_series_with_its_state():
    # Fabricated indexes: CPI 200 in October 2010 and 250 in January
    # 2022 (to 2010 dollars: 0.8). County 11111 runs 2010-2025 and
    # doubles; county 22222 stops in 2021 at 1.5 times its 2010 value,
    # and its state grows 1.2 times from 2021 to 2025.
    consumer = pd.DataFrame(
        {
            'series_id_bls': [web_scene.CONSUMER_PRICE_SERIES] * 2,
            'year': [2010, 2022],
            'period_bls': ['M10', 'M01'],
            'consumer_price_index': [200.0, 250.0],
        }
    )
    house = pd.DataFrame(
        {
            'county_fips': ['11111', '11111', '22222', '22222', '22222'],
            'year': [2010, 2025, 2010, 2021, 2025],
            'house_price_index': [100.0, 200.0, 100.0, 150.0, np.nan],
        }
    )
    state = pd.DataFrame(
        {
            'admin2_id': ['US-YY', 'US-YY'],
            'year': [2021, 2025],
            'house_price_index': [100.0, 120.0],
        }
    )
    counties = {'11111': 'US-XX', '22222': 'US-YY', '33333': 'US-YY'}
    factors = web_scene.dollars_factors(counties, house, consumer, state)
    assert factors['11111'] == {'factor': 1.6, 'to_year': 2025, 'estimated': False}
    assert factors['22222'] == {'factor': 1.44, 'to_year': 2025, 'estimated': True}
    # No 2010 value: no factor.
    assert '33333' not in factors
    # Without the state index, the short series stops at its last year.
    alone = web_scene.dollars_factors(counties, house, consumer)
    assert alone['22222'] == {'factor': 1.2, 'to_year': 2021, 'estimated': False}


def test_residential_lots_take_a_single_home_class_as_its_own_evidence():
    # Fabricated lots, each with a counted home on it:
    lots = pd.DataFrame(
        {
            'n_dwellings': [1.0, 1.0, 1.0, 1.0, 40.0, 0.0],
            'land_use_class': [
                # a house whose roll keeps its own use code (no count);
                'Single-Family',
                # an exempt lot classed single-family;
                'Single-Family',
                # an arboretum classed "Multi-Family", exempt;
                'Multi-Family',
                # an office;
                'Office',
                # public housing, exempt, with the roll's own count;
                'Multi-Family',
                # a house lot without any home counted.
                'Single-Family',
            ],
            'use_group': [
                'SFR GOLF',
                'exempt',
                'exempt',
                'commercial',
                'exempt',
                'SFR',
            ],
            web_scene.ROLL_DWELLINGS: [np.nan, np.nan, np.nan, np.nan, 40.0, np.nan],
        }
    )
    assert list(web_scene.residential_lots(lots)) == [
        True,
        False,
        False,
        False,
        True,
        False,
    ]


def test_residential_lots_follow_the_roll_where_it_names_no_home():
    # Fabricated lots, each classed as homes with a home counted on it:
    rows = [
        # forest land at use value, no building on the roll;
        ('Single-Family', 'Use Value | Forestry', 0.0, 20.0, 1.0),
        # industrial land with two outlines seen from above;
        ('Multiple Single-Family', 'INDUSTRIAL', 5e5, 40.0, 2.0),
        # open land with no use group and a token improvement;
        ('Single-Family', None, 200.0, 40.0, 1.0),
        # a house on commercially coded land;
        ('Single-Family', 'COMMERCIAL | SINGLE FAMILY RESIDENTIAL', 0.0, 0.3, 1.0),
        # a mobile home whose land record carries no building;
        ('Manufactured Home', 'MOBILE HOME SUBDIVISION', 0.0, 0.2, 1.0),
        # a park of mobile homes that the roll files as commercial;
        ('Manufactured Home', 'COMMERCIAL', 0.0, 8.0, 30.0),
        # a new house on a lot the roll still calls vacant;
        ('Single-Family', 'open_space | vacant', 0.0, 0.1, 1.0),
        # a house the roll states neither use group nor value for;
        ('Single-Family', None, np.nan, 0.4, 1.0),
        # a farmhouse on farmland at use value: its land is drawn per
        # acre, the farmhouse a home on the building layer;
        ('Single-Family', 'Use Value | Agricultural', 1.5e5, 40.0, 1.0),
        # the homesite cut from a farm at use value.
        ('Single-Family', 'USE VALUE HOMESITE', 1.5e5, 0.8, 1.0),
    ]
    lots = pd.DataFrame(
        rows,
        columns=[
            'land_use_class',
            'use_group',
            'improvement_value',
            'area_ha',
            'n_dwellings',
        ],
    ).assign(**{web_scene.ROLL_DWELLINGS: np.nan})
    assert list(web_scene.residential_lots(lots)) == [
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        False,
        True,
    ]


def test_the_page_reads_the_height_check_from_the_footprint_recipe():
    rule = web_scene.ghost_rule('US_footprint-openplaces-2026')
    assert rule['min_value_per_sqft'] > 0
    assert rule['max_value_per_floor_sqft'] > 0
    assert web_scene.ghost_rule('US_parcel-openplaces-2026') is None


def test_the_models_values_get_the_height_check(monkeypatch):
    # Fabricated outlines of 100 m2 on one floor. The first two carry
    # recorded values and the curate's own flags, which stay as they
    # are; the last two took the model's value, a needle and a house.
    rule = {'min_value_per_sqft': 3_500, 'max_value_per_floor_sqft': 3_000}
    monkeypatch.setattr(web_scene, 'ghost_rule', lambda recipe: rule)
    part = pd.DataFrame(
        {
            'structure_value': [9e8, 3e5, 4e8, 3e5],
            'area_m2': [100.0] * 4,
            web_scene.GHOST_COLUMN: pd.array([False, False, pd.NA, pd.NA]),
        }
    )
    filled = pd.Series([False, False, True, True])
    out = web_scene.flag_model_heights(part, 'r', {'structure_value': filled})
    assert list(out[web_scene.GHOST_COLUMN]) == [False, False, True, False]


def test_tiny_lots_carrying_an_accounts_value_are_land_ghosts(monkeypatch):
    # Fabricated: 2,000 lots of 1,000 m2 worth $1,000 to $2,000 each,
    # then a 20 m2 sliver carrying $5M, a 20 m2 lot at a high but real
    # price, and a large lot at the sliver's value per area.
    monkeypatch.setattr(web_scene, 'ghost_rule', lambda recipe: None)
    side = np.sqrt(1_000.0)
    geoms = [box(i * 100, 0, i * 100 + side, side) for i in range(2_000)]
    values = list(np.linspace(1_000, 2_000, 2_000))
    sliver = np.sqrt(20.0)
    geoms += [
        box(0, 1_000, sliver, 1_000 + sliver),
        box(100, 1_000, 100 + sliver, 1_000 + sliver),
        box(0, 5_000, 1_000, 6_000),
    ]
    values += [5e6, 200.0, 5e6 / 20 * 1e6]
    lots = gpd.GeoDataFrame({'land_value': values}, geometry=geoms, crs=5070)
    layer = web_scene.SceneLayer(
        name='parcels',
        label='Land',
        recipe='land',
        value_column='land_value',
        area_unit='ac',
        vmin=1.0,
        vmax=10.0,
    )
    gdfs = {'parcels': lots}
    web_scene.mark_land_ghosts(gdfs, [layer])
    ghosts = gdfs['parcels'][web_scene.GHOST_COLUMN].to_numpy()
    assert ghosts[-3] and not ghosts[-2] and not ghosts[-1]
    assert not ghosts[:-3].any()


def test_downtown_stop_names_the_unit_holding_it():
    admin = gpd.GeoDataFrame(
        {'admin3_id': ['US-XX-AAA', 'US-XX-BBB']},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs=4326,
    )
    stop = web_scene.downtown_stop((1.5, 0.5), admin)
    assert stop == {'longitude': 1.5, 'latitude': 0.5, 'admin3_id': 'US-XX-BBB'}
    assert web_scene.downtown_stop(None, admin) is None
    # Every place has a downtown for its tour: its own, or its center.
    from openplaces.viz.web_scene import local

    for place in local.load_places()['places']:
        assert len(local.downtown_of(place)) == 2


def test_a_layer_without_a_team_twin_reads_the_public_bundle(monkeypatch):
    from openplaces.io import delivery

    declared = {
        'parcels': [{'region_id': 'nantucket-ma'}, {'region_id': 'nantucket-ma.team'}],
        'footprints': [{'region_id': 'nantucket-ma'}],
    }
    monkeypatch.setattr(delivery, 'delivery_regions', lambda recipe: declared[recipe])
    assert web_scene.layer_region('parcels', 'nantucket-ma.team') == 'nantucket-ma.team'
    assert web_scene.layer_region('footprints', 'nantucket-ma.team') == 'nantucket-ma'
    assert web_scene.layer_region('footprints', 'nantucket-ma') == 'nantucket-ma'


def test_model_files_pool_counties_and_keep_their_views(tmp_path):
    # Fabricated: two counties' files with the same sidecar; one view's
    # column is missing from the files and is left out; a third file
    # does not exist.
    side = {
        'year': 2025,
        'basis': 'nominal 2025 USD per ha',
        'views': [
            {
                'key': 'm_a',
                'label': 'A',
                'log_column': 'log_a',
                'aoa_column': 'aoa_a',
                'credit': 'Model A, fabricated.',
            },
            {'key': 'm_b', 'label': 'B', 'log_column': 'log_b'},
        ],
    }
    for county, ids in (('X-1', ['p1', 'p2']), ('X-2', ['p3'])):
        pd.DataFrame(
            {'parcel_id': ids, 'log_a': [9.0] * len(ids), 'aoa_a': [0.5] * len(ids)}
        ).to_parquet(tmp_path / f'{county}.parquet')
        (tmp_path / f'{county}.json').write_text(json.dumps(side), encoding='utf-8')
    models = web_scene.read_model_files(
        [tmp_path / 'X-1.parquet', tmp_path / 'X-2.parquet', tmp_path / 'X-3.parquet']
    )
    assert list(models.table.index) == ['p1', 'p2', 'p3']
    assert [v.key for v in models.views] == ['m_a']
    assert models.views[0].aoa_column == 'aoa_a'
    assert models.year == 2025
    assert models.credits == ({'name': 'A', 'credit': 'Model A, fabricated.'},)
    assert models.data_credits == ()
    assert web_scene.read_model_files([tmp_path / 'none.parquet']) is None
    # Training sales per unit, named in one line of the model's credit.
    models = web_scene.ModelFiles(
        table=models.table,
        views=models.views,
        year=2025,
        credits=({'name': 'A', 'credit': 'Model A'},),
        training_sales={'A': [('X-1', 1200), ('X-2', 30)]},
    )
    credit = web_scene.model_credits(models, {'X-1': 'County One'})[0]['credit']
    assert credit == 'Model A. Trained on sales: County One 1,200; X-2 30.'


def test_lots_from_a_model_file_refuse_personal_columns(tmp_path):
    # Fabricated lots; with no dwelling count the layer loses its
    # per-home view and the tooltip rows that need it.
    lots = gpd.GeoDataFrame(
        {
            'parcel_id': ['p1', 'p2'],
            'admin3_id': ['US-XX-A', 'US-XX-A'],
            'land_value': [1000.0, 2000.0],
        },
        geometry=[box(0, 0, 0.001, 0.001), box(0.002, 0, 0.003, 0.001)],
        crs='EPSG:4326',
    )
    lots.to_parquet(tmp_path / 'lots.parquet')
    layer = web_scene.VALUE_SCENE_LAYERS[0]
    gdf, shown = web_scene.lots_from_model_file(tmp_path / 'lots.parquet', layer)
    assert list(gdf.index) == ['p1', 'p2']
    assert list(gdf[layer.value_column]) == [1000.0, 2000.0]
    assert shown.per_dwelling_label is None
    assert 'n_dwellings' not in [t.column for t in shown.tooltip]
    # Two units' files pool into one page.
    other = lots.assign(parcel_id=['q1', 'q2'], admin3_id='US-XX-B')
    other.to_parquet(tmp_path / 'other.parquet')
    both, _ = web_scene.lots_from_model_file(
        [tmp_path / 'lots.parquet', tmp_path / 'other.parquet'], layer
    )
    assert sorted(both['admin3_id'].unique()) == ['US-XX-A', 'US-XX-B']
    lots.assign(owner_name=['A', 'B']).to_parquet(tmp_path / 'owners.parquet')
    with pytest.raises(PersonalColumnError):
        web_scene.lots_from_model_file(tmp_path / 'owners.parquet', layer)


def test_palettes_of_a_land_only_scene_keep_the_land_colors():
    lots_only = web_scene.VALUE_SCENE_LAYERS[:1]
    palettes = web_scene.default_palettes(lots_only)
    assert len(palettes) == len(web_scene.default_palettes())
    assert all(set(p.layers) == {'parcels'} for p in palettes)


def test_viewer_names_no_2010_for_models_in_todays_dollars():
    script = (
        resources.files('openplaces.viz.web_scene') / 'static' / 'viewer.js'
    ).read_text(encoding='utf-8')
    # The year comes from the manifest; 2010 is only its default.
    assert 'scene.model_year' in script
    assert 'Model, 2010' not in script


def test_model_views_scale_to_the_assessment_and_carry_the_aoa():
    # Fabricated: four 1-ha lots whose recorded land is twice what the
    # model predicts, so the scale is 2; one lot lacks a prediction.
    # The shown value (with fills) differs and must not enter it.
    lots = gpd.GeoDataFrame(
        {
            'log_v': np.log([10_000.0, 20_000.0, 40_000.0, np.nan]),
            'aoa_v': [0.5, 1.0, 3.0, np.nan],
            'land_value': [20_000.0, 40_000.0, 80_000.0, 5_000.0],
            'v': [90_000.0, 90_000.0, 90_000.0, 90_000.0],
        },
        geometry=[box(1000 * k, 0, 1000 * k + 100, 100) for k in range(4)],
        crs='EPSG:26986',
    )
    layer = SceneLayer(
        name='parcels',
        label='Land',
        recipe='r',
        value_column='v',
        area_unit='ha',
        vmin=1,
        vmax=2,
        model_views=(web_scene.ModelView('model_v', 'V', 'log_v', 'aoa_v'),),
    )
    arrays, views = web_scene.model_view_arrays(lots, layer)
    assert views['model_v'] == {'label': 'V', 'unit': 'ha', 'model': True, 'scale': 2.0}
    assert arrays['value_model_v'][:3] == pytest.approx([20_000, 40_000, 80_000])
    assert np.isnan(arrays['value_model_v'][3])
    # Inside where at most 1, outside above, unknown without a value.
    assert list(arrays['aoa_model_v']) == [2, 2, 1, 0]
    # Per home: the lot's scaled model value over its homes, on the
    # per-home view's own scale; lots that are not homes have none.
    home_view = {'vmin': 1_000, 'vmax': 1_000_000, 'other': {'unit': 'ha'}}
    homes = np.array([2.0, np.nan, 4.0, 1.0])
    arrays, views = web_scene.model_view_arrays(
        lots, layer, homes=homes, home_view=home_view
    )
    home = views['model_v__home']
    assert (home['unit'], home['base'], home['per_home'], home['vmax']) == (
        'dwelling',
        'model_v',
        True,
        1_000_000,
    )
    # 1 ha lots: $20,000 over 2 homes, none, $80,000 over 4 homes.
    per_home = arrays['value_model_v__home']
    assert per_home[0] == pytest.approx(10_000) and np.isnan(per_home[1])
    assert per_home[2] == pytest.approx(20_000)
    # No predictions: no view.
    empty = lots.assign(log_v=np.nan)
    assert web_scene.model_view_arrays(empty, layer) == ({}, {})
    # No recorded value to scale on: no view either.
    assert web_scene.model_view_arrays(lots.drop(columns='land_value'), layer) == (
        {},
        {},
    )


def _team_scene(tmp_path):
    (tmp_path / 'index.html').write_text('<!doctype html>', encoding='utf-8')
    (tmp_path / '.htaccess').write_text('Options -Indexes\n', encoding='utf-8')
    (tmp_path / 'scene.json').write_text(
        json.dumps({'build_id': 'x', 'audience': 'team', 'layers': []}),
        encoding='utf-8',
    )
    return tmp_path


FAKE_TOKEN = 'AbCdEfGhIjKlMnOpQrSt_-'


def _fake_host(monkeypatch, team_passwd, on_host):
    settings = {
        'host': 'host.example',
        'user': 'someone',
        'path': '/home/someone/public_html/values',
        'key': '',
        'known_hosts': '',
        'url': 'https://host.example/values',
        'team_passwd': team_passwd,
        'access_tokens': 'tokens.tsv',
        'access_contact': 'someone@host.example',
    }
    calls = []
    monkeypatch.setattr(web_scene, 'load_deploy_settings', lambda env_file: settings)
    # A fabricated token.
    monkeypatch.setattr(
        web_scene, 'read_access_tokens', lambda path: [(FAKE_TOKEN, 'a reader')]
    )
    monkeypatch.setattr(web_scene, '_rclone_env', lambda settings: {})
    monkeypatch.setattr(
        web_scene,
        '_rclone_output',
        lambda env, *args: (
            'passwd\n' if args[0] == 'lsf' and args[1] in on_host else None
        ),
    )
    monkeypatch.setattr(web_scene, '_rclone', lambda env, *args: calls.append(args))
    return calls


def test_team_page_goes_up_only_behind_a_password(tmp_path, monkeypatch):
    scene = _team_scene(tmp_path)
    passwd = '/home/someone/.htpasswds/values/nantucket-team/passwd'
    _fake_host(monkeypatch, '', on_host=set())
    with pytest.raises(ValueError, match='only behind a password'):
        web_scene.deploy_web_scene(scene, go=True, subpath='nantucket-team')

    _fake_host(monkeypatch, passwd, on_host=set())
    with pytest.raises(FileNotFoundError, match='names no file on the host'):
        web_scene.deploy_web_scene(scene, go=True, subpath='nantucket-team')

    calls = _fake_host(monkeypatch, passwd, on_host={f'bh:{passwd}'})
    web_scene.deploy_web_scene(scene, go=True, subpath='nantucket-team')
    # The token gate and the password go up before any file of the page.
    assert calls[0][0] == 'copyto' and calls[0][2].endswith('/nantucket-team/.htaccess')
    htaccess = (scene / '.htaccess').read_text(encoding='utf-8')
    assert f'AuthUserFile "{passwd}"' in htaccess and 'Require valid-user' in htaccess
    assert FAKE_TOKEN in htaccess and 'access.html [R=302,L,QSD]' in htaccess
    # Apache's own 401 page, or the host's rewrite rules turn the
    # password request into a 404 and browsers never ask.
    assert 'ErrorDocument 401 default' in htaccess


def test_team_password_path_is_checked_and_public_pages_carry_none(
    tmp_path, monkeypatch
):
    with pytest.raises(ValueError, match='absolute path'):
        web_scene.team_htaccess('passwd')
    with pytest.raises(ValueError, match='absolute path'):
        web_scene.team_htaccess('/home/a b/passwd')
    scene = _team_scene(tmp_path)
    (scene / 'scene.json').write_text(
        json.dumps({'build_id': 'x', 'audience': 'public', 'layers': []}),
        encoding='utf-8',
    )
    # A public page left with a team page's .htaccess is rewritten:
    # the token gate, no password.
    (scene / '.htaccess').write_text(web_scene.team_htaccess('/p'), encoding='utf-8')
    calls = _fake_host(monkeypatch, '/p', on_host={'bh:/p'})
    web_scene.deploy_web_scene(scene, go=False)
    htaccess = (scene / '.htaccess').read_text(encoding='utf-8')
    assert 'AuthUserFile' not in htaccess and FAKE_TOKEN in htaccess
    assert calls[0][0] == 'copyto' and calls[0][2].endswith('/.htaccess')


def test_access_gate_lets_tokens_in_and_sends_others_to_the_access_page():
    gate = web_scene.access_htaccess([FAKE_TOKEN, 'b' * 22], '/values/', 'host.example')
    # A token in the address sets the cookie, scoped to the site.
    assert 'CO=op_access:%1:host.example:43200:/values:secure:httponly' in gate
    # Without a token in the address or the cookie: the access page,
    # which (with the icons) passes always.
    # QSD: a refused token does not travel on to the access page.
    assert 'RewriteRule ^ /values/access.html [R=302,L,QSD]' in gate
    assert '!(?:^|/)(?:access\\.html|icon\\-32\\.png|icon\\-180\\.png)$' in gate
    assert f'({FAKE_TOKEN}|{"b" * 22})' in gate
    with pytest.raises(ValueError):
        web_scene.access_htaccess(['short'], '/values', 'host.example')
    with pytest.raises(ValueError):
        web_scene.access_htaccess([FAKE_TOKEN], '/values', 'bad host"')


def test_access_gate_goes_up_in_three_connections(tmp_path, monkeypatch):
    from pathlib import Path

    # Fabricated site: a public page built here, a team page built
    # elsewhere (its .htaccess online asks for a password), an old
    # address that only redirects.
    site = tmp_path / 'site'
    (site / 'town').mkdir(parents=True)
    (site / 'town' / 'scene.json').write_text(
        '{"audience": "public"}', encoding='utf-8'
    )
    calls = _fake_host(monkeypatch, '/p', on_host=set())
    monkeypatch.setattr(
        web_scene,
        '_rclone_output',
        lambda env, *args: 'town/\nelsewhere-team/\nold/\nthumbs/\n',
    )
    uploaded = {}

    def fake_rclone(env, *args):
        calls.append(args)
        source, destination = args[1], args[2]
        if destination.startswith('bh:'):
            for path in Path(source).rglob('.htaccess'):
                uploaded[path.parent.name] = path.read_text(encoding='utf-8')
            return
        current = Path(destination)
        (current / 'elsewhere-team').mkdir(parents=True)
        (current / 'elsewhere-team' / '.htaccess').write_text(
            'AuthUserFile "/p"\n', encoding='utf-8'
        )
        (current / 'old').mkdir()
        (current / 'old' / '.htaccess').write_text('RedirectMatch 301 ^.*$ x\n')

    monkeypatch.setattr(web_scene, '_rclone', fake_rclone)
    written = web_scene.deploy_access(site, go=True)
    assert written == ['', 'elsewhere-team', 'town']
    # One listing (patched above), one copy down, one copy up.
    assert [c[0] for c in calls] == ['copy', 'copy']
    assert 'AuthUserFile' in uploaded['elsewhere-team']
    assert 'AuthUserFile' not in uploaded['town'] and FAKE_TOKEN in uploaded['town']
    assert 'old' not in uploaded


def test_access_tokens_file_needs_a_token(tmp_path):
    path = tmp_path / 'tokens.tsv'
    with pytest.raises(ValueError, match='No access tokens'):
        web_scene.read_access_tokens('')
    with pytest.raises(ValueError, match='holds no access token'):
        web_scene.read_access_tokens(path)
    path.write_text(f'# comment\n{FAKE_TOKEN}\ta reader\n', encoding='utf-8')
    assert web_scene.read_access_tokens(path) == [(FAKE_TOKEN, 'a reader')]
    path.write_text('not a token\tsomeone\n', encoding='utf-8')
    with pytest.raises(ValueError, match='not an access token'):
        web_scene.read_access_tokens(path)


def test_tokens_add_and_revoke_one_person(tmp_path):
    from openplaces.viz.web_scene import tokens

    path = tmp_path / 'tokens.tsv'
    link = tokens.add(path, 'reader one', 'https://host.example/values')
    tokens.add(path, 'reader two', 'https://host.example/values')
    token = link.rsplit('=', 1)[1]
    assert link.startswith('https://host.example/values/?t=')
    assert web_scene.read_access_tokens(path)[0] == (token, 'reader one')
    tokens.revoke(path, 'reader one')
    assert [n for _, n in web_scene.read_access_tokens(path)] == ['reader two']


def test_access_page_takes_the_contact_from_the_settings():
    page = web_scene.access_page({'access_contact': 'someone@host.example'})
    assert 'mailto:someone@host.example' in page and '{{CONTACT}}' not in page
    with pytest.raises(ValueError, match='ACCESS_CONTACT'):
        web_scene.access_page({'access_contact': ''})


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
    assert 'methods.html' in names


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
