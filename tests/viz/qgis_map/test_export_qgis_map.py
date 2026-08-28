"""Thin end-to-end test for the public `export_qgis_map` entrypoint."""

import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pandas as pd
import pytest

import openplaces.viz.qgis_map.generator as generator_module
from openplaces.core.schema import AdminId
from openplaces.io import to_parquet
from openplaces.recipe import get_output_path, get_recipe_by_id
from openplaces.viz.qgis_map import export_qgis_map, style_registry

FIXTURES = Path(__file__).parent / 'fixtures'
TEMPLATE_PATH = FIXTURES / 'tiny_template.qgz'

PARCEL_RECIPE = 'US_parcel-openplaces-2026'
FOOTPRINT_RECIPE = (
    'US_footprint-openplaces-2026'  # combined save_to; matches fixture 'test-output'
)
COUNTY = 'US-NC-CAR'


@pytest.fixture
def fixture_registry() -> pd.DataFrame:
    df = pd.read_csv(FIXTURES / 'style_registry.csv', dtype=str, keep_default_na=False)
    df['default_visible'] = df['default_visible'].str.lower().isin({'true', '1', 'yes'})
    df['priority'] = df['priority'].replace('', '0').astype(int)
    return df.set_index('style_key')


@pytest.fixture(autouse=True)
def _patch_style_registry(monkeypatch, fixture_registry):
    monkeypatch.setattr(
        generator_module,
        'get_style',
        lambda et, s, role: style_registry.get_style(
            et, s, role, registry=fixture_registry
        ),
    )
    monkeypatch.setattr(
        generator_module,
        'get_style_variants',
        lambda style_key: style_registry.get_style_variants(
            style_key, registry=fixture_registry
        ),
    )
    monkeypatch.setattr(
        generator_module,
        'get_fallback_style',
        lambda: style_registry.get_fallback_style(registry=fixture_registry),
    )
    monkeypatch.setattr(
        generator_module,
        'get_static_styles',
        lambda: style_registry.get_static_styles(registry=fixture_registry),
    )


class TestExportQgisMap:
    def test_returns_valid_qgz_when_nothing_exists_on_disk(self, mock_data_root):
        # filter_existing defaults to True; an empty data root means every
        # resolved layer is dropped, leaving only always-kept static/basemap
        # template layers -- still a valid, openable project.
        out = export_qgis_map(
            PARCEL_RECIPE,
            COUNTY,
            template_path=TEMPLATE_PATH,
            output_path=mock_data_root / 'out.qgz',
        )
        assert out.exists()
        with zipfile.ZipFile(out) as zf:
            qgs_members = [n for n in zf.namelist() if n.endswith('.qgs')]
            assert len(qgs_members) == 1

    def test_returns_populated_qgz_when_output_data_exists(self, mock_data_root):
        # footprint-openplaces-2026 saves attributes+geometry combined in one
        # file, so only that single path needs to exist on disk.
        recipe = get_recipe_by_id(FOOTPRINT_RECIPE)
        output_path = get_output_path(recipe, admin_id=AdminId(COUNTY))
        to_parquet(pd.DataFrame({'a': [1]}), output_path)

        out = export_qgis_map(
            FOOTPRINT_RECIPE,
            COUNTY,
            template_path=TEMPLATE_PATH,
            output_path=mock_data_root / 'out.qgz',
        )
        assert out.exists()
        with zipfile.ZipFile(out) as zf:
            [qgs_name] = [n for n in zf.namelist() if n.endswith('.qgs')]
            root = ET.fromstring(zf.read(qgs_name))
        layernames = {
            m.find('layername').text
            for m in root.find('projectlayers').findall('maplayer')
        }
        assert output_path.stem in layernames
