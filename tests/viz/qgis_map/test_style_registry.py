"""Tests for openplaces.viz.qgis_map.style_registry."""

from pathlib import Path

import pandas as pd
import pytest

from openplaces.viz.qgis_map import style_registry

FIXTURE_CSV = Path(__file__).parent / 'fixtures' / 'style_registry.csv'


@pytest.fixture
def fixture_registry() -> pd.DataFrame:
    df = pd.read_csv(FIXTURE_CSV, dtype=str, keep_default_na=False)
    df['default_visible'] = df['default_visible'].str.lower().isin({'true', '1', 'yes'})
    df['priority'] = df['priority'].replace('', '0').astype(int)
    return df.set_index('style_key')


class TestGetStyle:
    def test_exact_match(self, fixture_registry):
        style = style_registry.get_style(
            'footprint', 'obm', 'input', registry=fixture_registry
        )
        assert style is not None
        assert style.style_key == 'test-input'
        assert style.template_layer_name == 'proto_input'
        assert style.default_visible is False

    def test_wildcard_source_fallback(self, fixture_registry):
        style = style_registry.get_style(
            'footprint',
            'wholly-unregistered-source',
            'output',
            registry=fixture_registry,
        )
        assert style is not None
        assert style.style_key == 'test-output'

    def test_role_disambiguates_same_entity_type(self, fixture_registry):
        # 'footprint' has both an 'output' wildcard row (test-output) and an
        # 'input' exact row (test-input, source='obm'); role must select the
        # right one even when source alone would be ambiguous.
        output_style = style_registry.get_style(
            'footprint', 'obm', 'output', registry=fixture_registry
        )
        input_style = style_registry.get_style(
            'footprint', 'obm', 'input', registry=fixture_registry
        )
        assert output_style.style_key == 'test-output'
        assert input_style.style_key == 'test-input'

    def test_no_match_returns_none(self, fixture_registry):
        style = style_registry.get_style(
            'transaction', 'anything', 'input', registry=fixture_registry
        )
        assert style is None

    def test_basemap_and_static_rows_are_not_matchable(self, fixture_registry):
        # basemap-open/_fallback rows have blank entity_type and must never
        # be returned by ordinary entity_type/source/role lookups.
        assert (
            style_registry.get_style('', '', 'output', registry=fixture_registry)
            is None
        )


class TestGetStyleVariants:
    def test_returns_registered_variants(self, fixture_registry):
        variants = style_registry.get_style_variants(
            'test-output', registry=fixture_registry
        )
        assert [v.style_key for v in variants] == ['test-output-by-roof-shape']
        assert variants[0].variant_of == 'test-output'
        assert variants[0].variant_label == 'Roof shape'
        assert variants[0].template_layer_name == 'proto_output_by_roof_shape'
        assert variants[0].default_visible is False

    def test_no_variants_returns_empty_list(self, fixture_registry):
        variants = style_registry.get_style_variants(
            'test-input', registry=fixture_registry
        )
        assert variants == []

    def test_unknown_style_key_returns_empty_list(self, fixture_registry):
        variants = style_registry.get_style_variants('nope', registry=fixture_registry)
        assert variants == []

    def test_variant_rows_are_excluded_from_get_style(self, fixture_registry):
        # 'test-output-by-roof-shape' shares (entity_type='footprint',
        # role='output') with the base 'test-output' row; get_style must
        # never surface the variant itself.
        style = style_registry.get_style(
            'footprint', None, 'output', registry=fixture_registry
        )
        assert style.style_key == 'test-output'


class TestGetFallbackStyle:
    def test_returns_reserved_row(self, fixture_registry):
        style = style_registry.get_fallback_style(registry=fixture_registry)
        assert style.style_key == style_registry.FALLBACK_STYLE_KEY
        assert style.template_layer_name == 'proto_fallback'

    def test_missing_fallback_row_raises(self):
        reg = pd.DataFrame(
            {
                'entity_type': ['footprint'],
                'source': [''],
                'role': ['output'],
                'template_layer_name': ['x'],
                'group_path': ['Curated'],
                'default_visible': [True],
                'priority': [0],
                'notes': [''],
            },
            index=pd.Index(['test-output'], name='style_key'),
        )
        with pytest.raises(ValueError, match='_fallback'):
            style_registry.get_fallback_style(registry=reg)


class TestGetStaticStyles:
    def test_returns_basemap_and_static_rows_only(self, fixture_registry):
        static = style_registry.get_static_styles(registry=fixture_registry)
        keys = {s.style_key for s in static}
        # '_fallback' is role='static' too but must be excluded: it should
        # only survive pruning when a spec actually falls back to it, not
        # unconditionally like a real basemap/static layer.
        assert keys == {'basemap-open'}


class TestLoadRegistry:
    def test_production_registry_loads_and_caches(self):
        first = style_registry.load_registry()
        second = style_registry.load_registry()
        assert first is second
        assert style_registry.FALLBACK_STYLE_KEY in first.index
