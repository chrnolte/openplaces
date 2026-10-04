"""openplaces never draws its data over a basemap or tile service.

Every map, static or interactive, and every generated QGIS project
carries only openplaces' own layers on a plain background. A user who
wants a background map adds one in their own tool.
"""

import inspect
import re
import xml.etree.ElementTree as ET
import zipfile
from importlib import resources
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402
from shapely.geometry import box  # noqa: E402

import openplaces  # noqa: E402
import openplaces.viz as viz  # noqa: E402
from openplaces.core.schema import AdminId  # noqa: E402
from openplaces.viz import maps  # noqa: E402

SRC = Path(openplaces.__file__).parent

# A map tile client, or a call that fetches tiles, anywhere in src/.
_TILE_CLIENT = re.compile(
    r'^\s*(import|from)\s+(contextily|xyzservices)\b|add_basemap|bounds2img'
    r'|BitmapTileLayer|MaplibreBasemap',
    re.MULTILINE,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


def test_viz_exposes_no_basemap_helpers():
    public = set(viz.__all__) | set(viz._LAZY_SOURCES)
    assert not {name for name in public if 'basemap' in name.lower()}
    assert not {name for name in public if 'tile' in name.lower()}


def test_no_source_module_uses_a_tile_client():
    offenders = [
        str(path.relative_to(SRC))
        for path in SRC.rglob('*.py')
        if _TILE_CLIENT.search(path.read_text(encoding='utf-8'))
    ]
    assert not offenders


@pytest.mark.parametrize(
    'function',
    [maps.show_geometry_context, maps.show_building, maps.show_ingested_geometries],
)
def test_static_maps_take_no_basemap_argument(function):
    parameters = inspect.signature(function).parameters
    assert not [name for name in parameters if 'basemap' in name]


def test_geometry_context_draws_no_background_image():
    gdf = gpd.GeoDataFrame(
        {'name': ['a', 'b']},
        geometry=[
            box(-78.0, 35.0, -77.999, 35.001),
            box(-77.999, 35.0, -77.998, 35.001),
        ],
        crs='EPSG:4326',
    )
    fig, (ax_map, _) = maps.show_geometry_context(gdf, 0)
    assert ax_map.collections
    assert not ax_map.images


def test_interactive_map_has_no_basemap(monkeypatch):
    pytest.importorskip('lonboard')
    from openplaces.viz import interactive

    parcels = gpd.GeoDataFrame(
        {'value': [1.0, 2.0]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs='EPSG:4326',
    )
    monkeypatch.setattr(interactive, 'get_entities', lambda *a, **k: parcels)
    map_widget = interactive.show_entities_interactive('any-recipe', legend=False)
    assert map_widget.basemap is None


def _tile_layers(qgz_path: Path) -> list[str]:
    from openplaces.viz.qgis_map.generator import _is_tile_layer

    with zipfile.ZipFile(qgz_path) as zf:
        [qgs] = [n for n in zf.namelist() if n.endswith('.qgs')]
        root = ET.fromstring(zf.read(qgs))
    return [
        m.findtext('layername')
        for m in root.find('projectlayers').findall('maplayer')
        if _is_tile_layer(m)
    ]


def _packaged_template() -> Path:
    return Path(
        resources.files('openplaces.qgis') / 'templates' / 'openplaces_template.qgz'
    )


def test_packaged_qgis_template_has_no_tile_layer():
    assert not _tile_layers(_packaged_template())


def test_generated_qgis_project_has_no_tile_layer(tmp_path, monkeypatch):
    from openplaces.viz.qgis_map import generator

    monkeypatch.setattr(
        generator,
        'get_admin',
        lambda *a, **k: gpd.GeoDataFrame(
            geometry=[box(-80.0, 34.0, -79.0, 35.0)], crs='EPSG:4326'
        ),
    )
    out = generator.generate_qgz(
        {'recipe_id': 'test-recipe', 'stage': 'curate'},
        AdminId('US', 'NC', 'CAR'),
        layer_specs=[],
        template_path=_packaged_template(),
        output_path=tmp_path / 'out.qgz',
    )
    assert not _tile_layers(out)
    with zipfile.ZipFile(out) as zf:
        [qgs] = [n for n in zf.namelist() if n.endswith('.qgs')]
        assert b'type=xyz' not in zf.read(qgs)
