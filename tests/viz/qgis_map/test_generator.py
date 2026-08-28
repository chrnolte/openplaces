"""Tests for openplaces.viz.qgis_map.generator.

Exercises `generate_qgz` against the synthetic `fixtures/tiny_template.qgz`
with directly-constructed `LayerSpec`s (bypassing the resolver), matched
against `fixtures/style_registry.csv` (not the production registry).
"""

import datetime as dt_module
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

import openplaces.viz.qgis_map.generator as generator_module
from openplaces.core.schema import AdminId, Entity
from openplaces.viz.qgis_map import style_registry
from openplaces.viz.qgis_map.generator import generate_qgz
from openplaces.viz.qgis_map.resolver import LayerSpec

FIXTURES = Path(__file__).parent / 'fixtures'
TEMPLATE_PATH = FIXTURES / 'tiny_template.qgz'


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


@pytest.fixture(autouse=True)
def _patch_get_admin(monkeypatch):
    def _fake_get_admin(admin_id, geom=True):
        return gpd.GeoDataFrame(
            {'geometry': [box(-80.0, 34.0, -79.0, 35.0)]}, crs='EPSG:4326'
        )

    monkeypatch.setattr(generator_module, 'get_admin', _fake_get_admin)


def _spec(
    tmp_path,
    *,
    role,
    entity_type,
    source,
    display_name,
    combined=False,
    recipe_id=None,
    depth=0,
) -> LayerSpec:
    attr_path = tmp_path / f'{display_name}.parquet'
    geo_path = attr_path if combined else tmp_path / f'{display_name}_geo.parquet'
    return LayerSpec(
        role=role,
        recipe_id=recipe_id or f'{entity_type}-{source}-recipe',
        entity_type=entity_type,
        source=source,
        version='2025',
        admin_id=AdminId('US', 'NC', 'CAR'),
        display_name=display_name,
        attr_path=attr_path,
        geo_path=geo_path,
        exists=True,
        depth=depth,
        combined=combined,
    )


@pytest.fixture
def basic_specs(tmp_path):
    return [
        _spec(
            tmp_path,
            role='output',
            entity_type='footprint',
            source='cheer',
            display_name='US-NC-CAR_footprint-openplaces-2026',
            depth=0,
        ),
        _spec(
            tmp_path,
            role='output',
            entity_type='parcel',
            source='openplaces',
            display_name='US-NC-CAR_parcel-openplaces-2026',
            combined=True,
            depth=0,
        ),
        _spec(
            tmp_path,
            role='input',
            entity_type='footprint',
            source='obm',
            display_name='US-NC-CAR_footprint-obm-2025',
            depth=2,
        ),
        _spec(
            tmp_path,
            role='admin',
            entity_type='admin',
            source='census',
            display_name='US_admin-census-2021',
            depth=0,
        ),
    ]


def _footprint_cheer_spec(specs):
    return next(
        s for s in specs if s.entity_type == 'footprint' and s.source == 'cheer'
    )


MINIMAL_RECIPE = {'recipe_id': 'test-recipe', 'stage': 'curate'}


def _run(tmp_path, specs, **kwargs):
    return generate_qgz(
        MINIMAL_RECIPE,
        AdminId('US', 'NC', 'CAR'),
        layer_specs=specs,
        template_path=TEMPLATE_PATH,
        output_path=tmp_path / 'out.qgz',
        **kwargs,
    )


def _read_qgs_root(qgz_path: Path) -> ET.Element:
    with zipfile.ZipFile(qgz_path) as zf:
        qgs_names = [n for n in zf.namelist() if n.endswith('.qgs')]
        assert len(qgs_names) == 1
        return ET.fromstring(zf.read(qgs_names[0]))


def _find_tree_layer(root: ET.Element, name: str) -> ET.Element:
    [layer] = [
        el
        for el in root.find('layer-tree-group').iter('layer-tree-layer')
        if el.get('name') == name
    ]
    return layer


def _find_legendlayer(root: ET.Element, name: str) -> ET.Element:
    [layer] = [
        el for el in root.find('legend').iter('legendlayer') if el.get('name') == name
    ]
    return layer


class TestDefaultOutputPath:
    def test_admin_entity_prefix_appears_once(self, mock_data_root):
        recipe = {
            'recipe_id': 'US_footprint-openplaces-2026',
            'entity': Entity('footprint', 'openplaces', '2026'),
        }
        output_path = generator_module._default_output_path(
            recipe, AdminId('US', 'NC', 'CAR')
        )
        assert output_path.name == 'US-NC-CAR_footprint-openplaces-2026_map.qgz'
        assert output_path == (
            mock_data_root
            / 'data'
            / 'share'
            / 'US'
            / 'NC'
            / 'CAR'
            / '_all'
            / 'footprint'
            / 'openplaces'
            / '2026'
            / 'US-NC-CAR_footprint-openplaces-2026_map.qgz'
        )


class TestZipStructure:
    def test_exactly_one_qgs_and_styles_member(self, tmp_path, basic_specs):
        out = _run(tmp_path, basic_specs)
        with zipfile.ZipFile(out) as zf:
            names = zf.namelist()
        qgs = [n for n in names if n.endswith('.qgs')]
        others = [n for n in names if not n.endswith('.qgs')]
        assert len(qgs) == 1
        assert len(others) == 1

    def test_styles_db_passed_through_unmodified(self, tmp_path, basic_specs):
        out = _run(tmp_path, basic_specs)
        with zipfile.ZipFile(TEMPLATE_PATH) as zf:
            original_styles = zf.read('tiny_template_styles.db')
        with zipfile.ZipFile(out) as zf:
            [styles_name] = [n for n in zf.namelist() if not n.endswith('.qgs')]
            output_styles = zf.read(styles_name)
        assert output_styles == original_styles

    def test_output_is_well_formed_xml(self, tmp_path, basic_specs):
        out = _run(tmp_path, basic_specs)
        root = _read_qgs_root(out)
        assert root.tag == 'qgis'


class TestLayerCloning:
    def test_maplayer_count(self, tmp_path, basic_specs):
        # Per matched spec: only its clone (2 layers, 1 if combined) — the
        # original template prototype it was cloned from is pruned, not
        # kept. Plus the always-kept basemap.
        out = _run(tmp_path, basic_specs)
        root = _read_qgs_root(out)
        maplayers = root.find('projectlayers').findall('maplayer')
        expected = sum(1 if spec.combined else 2 for spec in basic_specs)
        expected += 1  # proto_basemap, always kept regardless of resolved specs
        # The footprint/cheer output spec resolves to 'test-output', which has
        # one registered variant (test-output-by-roof-shape): +1 new variant
        # clone. Likewise the combined parcel/openplaces spec resolves to
        # 'test-output-combined', which has one registered variant
        # (test-output-combined-by-zone): another +1.
        expected += 1 + 1
        assert len(maplayers) == expected

    def test_datasources_point_at_resolved_or_static_paths(
        self, tmp_path, basic_specs, fixture_registry
    ):
        # basic_specs' data files and the output .qgz all live directly under
        # tmp_path, so every relative datasource collapses to a bare filename
        # (the "same folder as the curated data file" case).
        out = _run(tmp_path, basic_specs)
        root = _read_qgs_root(out)
        resolved_names = set()
        for spec in basic_specs:
            resolved_names.add(spec.attr_path.name)
            resolved_names.add(spec.geo_path.name)

        static_layernames = {
            s.template_layer_name
            for s in style_registry.get_static_styles(registry=fixture_registry)
        }
        template_root = _read_qgs_root(TEMPLATE_PATH)
        static_datasources = {
            m.find('datasource').text
            for m in template_root.find('projectlayers').findall('maplayer')
            if m.find('layername').text in static_layernames
        }

        for maplayer in root.find('projectlayers').findall('maplayer'):
            datasource = maplayer.find('datasource').text
            assert datasource in resolved_names or datasource in static_datasources

    def test_datasource_is_relative_not_absolute(self, tmp_path, basic_specs):
        out = _run(tmp_path, basic_specs)
        root = _read_qgs_root(out)
        footprint_spec = _footprint_cheer_spec(basic_specs)
        maplayers = root.find('projectlayers').findall('maplayer')
        [clone] = [
            m
            for m in maplayers
            if m.find('layername').text == footprint_spec.display_name
        ]
        datasource = clone.find('datasource').text
        assert datasource == footprint_spec.geo_path.name
        assert not Path(datasource).is_absolute()

    def test_datasource_falls_back_to_absolute_across_drives(
        self, tmp_path, basic_specs, monkeypatch
    ):
        def _raise_different_drive(*args, **kwargs):
            raise ValueError("path is on mount 'C:', start on mount 'D:'")

        monkeypatch.setattr(generator_module.os.path, 'relpath', _raise_different_drive)
        out = _run(tmp_path, basic_specs)
        root = _read_qgs_root(out)
        footprint_spec = _footprint_cheer_spec(basic_specs)
        maplayers = root.find('projectlayers').findall('maplayer')
        [clone] = [
            m
            for m in maplayers
            if m.find('layername').text == footprint_spec.display_name
        ]
        datasource = clone.find('datasource').text
        assert datasource == str(footprint_spec.geo_path.resolve())
        assert Path(datasource).is_absolute()

    def test_no_dangling_join_references(self, tmp_path, basic_specs):
        out = _run(tmp_path, basic_specs)
        root = _read_qgs_root(out)
        all_ids = {
            maplayer.find('id').text
            for maplayer in root.find('projectlayers').findall('maplayer')
        }
        for maplayer in root.find('projectlayers').findall('maplayer'):
            vectorjoins = maplayer.find('vectorjoins')
            if vectorjoins is None:
                continue
            for join in vectorjoins.findall('join'):
                assert join.get('joinLayerId') in all_ids

    def test_combined_spec_produces_single_layer_with_no_join(
        self, tmp_path, basic_specs
    ):
        out = _run(tmp_path, basic_specs)
        root = _read_qgs_root(out)
        parcel_spec = next(s for s in basic_specs if s.entity_type == 'parcel')
        maplayers = root.find('projectlayers').findall('maplayer')
        matches = [
            m for m in maplayers if m.find('layername').text == parcel_spec.display_name
        ]
        assert len(matches) == 1
        vectorjoins = matches[0].find('vectorjoins')
        assert vectorjoins is None or len(vectorjoins.findall('join')) == 0
        # And no separate _attr clone was created for it.
        attr_matches = [
            m
            for m in maplayers
            if m.find('layername').text == f'{parcel_spec.display_name}_attr'
        ]
        assert attr_matches == []


class TestStyleVariants:
    def test_variant_clone_appears_with_expected_display_name(
        self, tmp_path, basic_specs
    ):
        out = _run(tmp_path, basic_specs)
        root = _read_qgs_root(out)
        footprint_spec = _footprint_cheer_spec(basic_specs)
        layernames = {
            m.find('layername').text
            for m in root.find('projectlayers').findall('maplayer')
        }
        assert f'Roof shape - {footprint_spec.display_name}' in layernames

    def test_variant_joins_the_base_clones_shared_attr_id(self, tmp_path, basic_specs):
        out = _run(tmp_path, basic_specs)
        root = _read_qgs_root(out)
        footprint_spec = _footprint_cheer_spec(basic_specs)
        maplayers = root.find('projectlayers').findall('maplayer')
        [base_attr] = [
            m
            for m in maplayers
            if m.find('layername').text == f'{footprint_spec.display_name}_attr'
        ]
        base_attr_id = base_attr.find('id').text
        [variant] = [
            m
            for m in maplayers
            if m.find('layername').text == f'Roof shape - {footprint_spec.display_name}'
        ]
        [join] = variant.find('vectorjoins').findall('join')
        assert join.get('joinLayerId') == base_attr_id

    def test_variant_defaults_unchecked(self, tmp_path, basic_specs):
        out = _run(tmp_path, basic_specs)
        root = _read_qgs_root(out)
        footprint_spec = _footprint_cheer_spec(basic_specs)
        variant_name = f'Roof shape - {footprint_spec.display_name}'
        tree_layers = root.find('layer-tree-group').iter('layer-tree-layer')
        [entry] = [layer for layer in tree_layers if layer.get('name') == variant_name]
        assert entry.get('checked') == 'Qt::Unchecked'

    def test_combined_spec_variant_has_no_join(self, tmp_path, basic_specs):
        # Combined specs have no attr sibling, but variants are still cloned
        # (e.g. footprint-openplaces-2026's own curate output is combined) — the
        # variant clone just shares the base's datasource with no join.
        out = _run(tmp_path, basic_specs)
        root = _read_qgs_root(out)
        parcel_spec = next(s for s in basic_specs if s.entity_type == 'parcel')
        variant_name = f'Zone - {parcel_spec.display_name}'
        maplayers = root.find('projectlayers').findall('maplayer')
        [variant] = [m for m in maplayers if m.find('layername').text == variant_name]
        vectorjoins = variant.find('vectorjoins')
        assert vectorjoins is None or len(vectorjoins.findall('join')) == 0
        assert variant.find('datasource').text == parcel_spec.geo_path.name

    def test_missing_variant_template_layer_raises(self, tmp_path, monkeypatch):
        bad_variant = style_registry.LayerStyle(
            style_key='test-output-broken-variant',
            entity_type='footprint',
            source=None,
            role='output',
            template_layer_name='does_not_exist_in_template',
            group_path='Curated',
            default_visible=False,
            priority=0,
            variant_of='test-output',
            variant_label='Broken',
            notes=None,
            dynamic_categorize_attr=None,
            attr_override=None,
            category_colors=None,
            attr_breaks=None,
            attr_scale=None,
            attr_unit=None,
        )
        monkeypatch.setattr(
            generator_module, 'get_style_variants', lambda style_key: [bad_variant]
        )
        spec = _spec(
            tmp_path,
            role='output',
            entity_type='footprint',
            source='cheer',
            display_name='US-NC-CAR_footprint-openplaces-2026',
        )
        with pytest.raises(ValueError, match='does_not_exist_in_template'):
            _run(tmp_path, [spec])


class TestPruning:
    def test_unused_non_static_prototypes_are_removed(self, tmp_path, basic_specs):
        # basic_specs never resolves to the 'test-admin' style's own prototype
        # under a *different* source than 'census' is fine, but the fallback
        # prototype (proto_fallback/_attr) is never cloned from here and must
        # be pruned along with its tree/legend entries.
        out = _run(tmp_path, basic_specs)
        root = _read_qgs_root(out)
        layernames = {
            m.find('layername').text
            for m in root.find('projectlayers').findall('maplayer')
        }
        assert 'proto_fallback' not in layernames
        assert 'proto_fallback_attr' not in layernames

    def test_matched_prototypes_are_pruned_alongside_their_clones(
        self, tmp_path, basic_specs
    ):
        # A matched spec's original template prototype (and its variants)
        # must not survive alongside the clone made from it -- keeping both
        # would leave an unresolvable, un-substituted placeholder datasource
        # (e.g. './proto_output.parquet') in the shipped project.
        out = _run(tmp_path, basic_specs)
        root = _read_qgs_root(out)
        layernames = {
            m.find('layername').text
            for m in root.find('projectlayers').findall('maplayer')
        }
        matched_prototypes = {
            'proto_output',
            'proto_output_attr',
            'proto_output_by_roof_shape',
            'proto_output_combined',
            'proto_output_combined_by_zone',
            'proto_input',
            'proto_input_attr',
            'proto_admin',
            'proto_admin_attr',
        }
        assert not (matched_prototypes & layernames)

    def test_static_basemap_prototype_survives_pruning(self, tmp_path, basic_specs):
        out = _run(tmp_path, basic_specs)
        root = _read_qgs_root(out)
        layernames = {
            m.find('layername').text
            for m in root.find('projectlayers').findall('maplayer')
        }
        assert 'proto_basemap' in layernames

    def test_pruned_layers_removed_from_tree_and_legend(self, tmp_path, basic_specs):
        out = _run(tmp_path, basic_specs)
        root = _read_qgs_root(out)
        tree_xml = ET.tostring(root.find('layer-tree-group'), encoding='unicode')
        legend_xml = ET.tostring(root.find('legend'), encoding='unicode')
        assert 'proto_fallback' not in tree_xml
        assert 'proto_fallback' not in legend_xml


class TestUnstyledFallback:
    def test_unmatched_spec_uses_fallback_and_warns(self, tmp_path, basic_specs):
        unmatched = _spec(
            tmp_path,
            role='input',
            entity_type='transaction',
            source='wholly-unregistered',
            display_name='US-NC-CAR_transaction-unregistered-2025',
            depth=3,
        )
        with pytest.warns(UserWarning, match='No style registered'):
            out = _run(tmp_path, [*basic_specs, unmatched])
        root = _read_qgs_root(out)
        matches = [
            m
            for m in root.find('projectlayers').findall('maplayer')
            if m.find('layername').text == unmatched.display_name
        ]
        assert len(matches) == 1
        tree_xml = ET.tostring(root.find('layer-tree-group'), encoding='unicode')
        assert f'name="{unmatched.display_name}"' in tree_xml, (
            'expected the fallback clone to be inserted into the tree'
        )
        # Placed under an "Unstyled" group.
        unstyled_group = next(
            g
            for g in root.find('layer-tree-group').findall('layer-tree-group')
            if g.get('name') == 'Unstyled'
        )
        assert any(
            layer.get('name') == unmatched.display_name
            for layer in unstyled_group.findall('layer-tree-layer')
        )


class TestProjectVariables:
    def test_sets_recipe_admin_and_timestamp_variables(self, tmp_path, basic_specs):
        out = _run(tmp_path, basic_specs)
        root = _read_qgs_root(out)
        names = [v.text for v in root.find('properties/Variables/variableNames')]
        values = [v.text for v in root.find('properties/Variables/variableValues')]
        pairs = dict(zip(names, values, strict=True))
        assert pairs['recipe_id'] == 'test-recipe'
        assert pairs['admin_id'] == 'US-NC-CAR'
        assert 'generated_at' in pairs

    def test_title_set_when_given(self, tmp_path, basic_specs):
        out = _run(tmp_path, basic_specs, title='My Test Map')
        root = _read_qgs_root(out)
        assert root.find('title').text == 'My Test Map'


class TestExtent:
    def test_extent_is_updated_from_template_default(self, tmp_path, basic_specs):
        out = _run(tmp_path, basic_specs)
        root = _read_qgs_root(out)
        extent = root.find('mapcanvas/extent')
        values = [
            float(extent.find(tag).text) for tag in ('xmin', 'ymin', 'xmax', 'ymax')
        ]
        assert values != [0.0, 0.0, 1.0, 1.0]
        assert all(v == v for v in values)  # not NaN


class TestMissingTemplateLayer:
    def test_raises_when_style_points_at_a_missing_template_layer(
        self, tmp_path, monkeypatch
    ):
        bad_style = style_registry.LayerStyle(
            style_key='broken',
            entity_type='footprint',
            source=None,
            role='output',
            template_layer_name='does_not_exist_in_template',
            group_path='Curated',
            default_visible=True,
            priority=0,
            variant_of=None,
            variant_label=None,
            notes=None,
            dynamic_categorize_attr=None,
            attr_override=None,
            category_colors=None,
            attr_breaks=None,
            attr_scale=None,
            attr_unit=None,
        )
        monkeypatch.setattr(
            generator_module, 'get_style', lambda et, s, role: bad_style
        )
        spec = _spec(
            tmp_path,
            role='output',
            entity_type='footprint',
            source='x',
            display_name='x',
        )
        with pytest.raises(ValueError, match='does_not_exist_in_template'):
            _run(tmp_path, [spec])


class TestDeterminism:
    def test_two_identical_calls_produce_identical_qgs_bytes(
        self, tmp_path, basic_specs, monkeypatch
    ):
        counter_box = {'n': 0}

        class _FakeUUID:
            def __init__(self, hex_value):
                self.hex = hex_value

        def _fake_uuid4():
            counter_box['n'] += 1
            return _FakeUUID(f'{counter_box["n"]:032x}')

        class _FixedDatetime(dt_module.datetime):
            @classmethod
            def now(cls, tz=None):
                return cls(2026, 1, 1, tzinfo=tz)

        monkeypatch.setattr(generator_module.uuid, 'uuid4', _fake_uuid4)
        monkeypatch.setattr(generator_module, 'datetime', _FixedDatetime)

        def _generate(output_name):
            counter_box['n'] = 0
            return generate_qgz(
                MINIMAL_RECIPE,
                AdminId('US', 'NC', 'CAR'),
                layer_specs=basic_specs,
                template_path=TEMPLATE_PATH,
                output_path=tmp_path / output_name,
            )

        out1 = _generate('run1.qgz')
        out2 = _generate('run2.qgz')
        with zipfile.ZipFile(out1) as zf1:
            [qgs1] = [n for n in zf1.namelist() if n.endswith('.qgs')]
            bytes1 = zf1.read(qgs1)
        with zipfile.ZipFile(out2) as zf2:
            [qgs2] = [n for n in zf2.namelist() if n.endswith('.qgs')]
            bytes2 = zf2.read(qgs2)
        assert bytes1 == bytes2


class TestMissingTemplate:
    def test_missing_template_path_raises_friendly_error(self, tmp_path, basic_specs):
        with pytest.raises(FileNotFoundError, match='template'):
            generate_qgz(
                MINIMAL_RECIPE,
                AdminId('US', 'NC', 'CAR'),
                layer_specs=basic_specs,
                template_path=tmp_path / 'does_not_exist.qgz',
                output_path=tmp_path / 'out.qgz',
            )


class TestStaticLayerCheckedState:
    def test_default_visible_false_overrides_template_baked_checked(
        self, tmp_path, basic_specs, fixture_registry, monkeypatch
    ):
        modified = fixture_registry.copy()
        modified.loc['basemap-open', 'default_visible'] = False
        monkeypatch.setattr(
            generator_module,
            'get_static_styles',
            lambda: style_registry.get_static_styles(registry=modified),
        )
        out = _run(tmp_path, basic_specs)
        root = _read_qgs_root(out)
        assert _find_tree_layer(root, 'proto_basemap').get('checked') == 'Qt::Unchecked'
        assert (
            _find_legendlayer(root, 'proto_basemap').get('checked') == 'Qt::Unchecked'
        )

    def test_default_visible_true_keeps_checked(self, tmp_path, basic_specs):
        # Fixture registry's basemap-open row is default_visible=true, matching
        # the template's own baked Qt::Checked -- unaffected regression guard.
        out = _run(tmp_path, basic_specs)
        root = _read_qgs_root(out)
        assert _find_tree_layer(root, 'proto_basemap').get('checked') == 'Qt::Checked'
        assert _find_legendlayer(root, 'proto_basemap').get('checked') == 'Qt::Checked'


class TestProjectCrs:
    def test_project_crs_mirrors_mapcanvas_destinationsrs(self, tmp_path, basic_specs):
        out = _run(tmp_path, basic_specs)
        root = _read_qgs_root(out)
        project_crs_wkt = root.find('projectCrs/spatialrefsys/wkt')
        canvas_wkt = root.find('.//mapcanvas/destinationsrs/spatialrefsys/wkt')
        assert project_crs_wkt is not None
        assert canvas_wkt is not None
        assert project_crs_wkt.text == canvas_wkt.text

    def test_exactly_one_project_crs(self, tmp_path, basic_specs):
        out = _run(tmp_path, basic_specs)
        root = _read_qgs_root(out)
        assert len(root.findall('projectCrs')) == 1

    def test_projections_enabled_set(self, tmp_path, basic_specs):
        out = _run(tmp_path, basic_specs)
        root = _read_qgs_root(out)
        projections_enabled = root.find('properties/SpatialRefSys/ProjectionsEnabled')
        assert projections_enabled is not None
        assert projections_enabled.text == '1'


class TestCollapsedLayerTree:
    def test_all_groups_and_layers_collapsed(self, tmp_path, basic_specs):
        out = _run(tmp_path, basic_specs)
        root = _read_qgs_root(out)
        tree_root = root.find('layer-tree-group')
        groups = tree_root.findall('.//layer-tree-group')
        layers = tree_root.findall('.//layer-tree-layer')
        assert groups, 'expected at least one group in the output tree'
        assert layers, 'expected at least one layer in the output tree'
        assert all(g.get('expanded') == '0' for g in groups)
        assert all(el.get('expanded') == '0' for el in layers)

    def test_all_legend_groups_and_layers_closed(self, tmp_path, basic_specs):
        out = _run(tmp_path, basic_specs)
        root = _read_qgs_root(out)
        legend_root = root.find('legend')
        legend_groups = legend_root.findall('.//legendgroup')
        legend_layers = legend_root.findall('.//legendlayer')
        assert legend_groups, 'expected at least one legendgroup in the output'
        assert legend_layers, 'expected at least one legendlayer in the output'
        assert all(g.get('open') == 'false' for g in legend_groups)
        assert all(el.get('open') == 'false' for el in legend_layers)


def _dynamic_cat_spec(dir_path: Path, *, values: list) -> LayerSpec:
    spec = _spec(
        dir_path,
        role='output',
        entity_type='footprint',
        source='dynamiccat',
        display_name='US-NC-CAR_footprint-dynamiccat-test',
        combined=True,
    )
    pd.DataFrame({'conflict_field': values}).to_parquet(spec.attr_path)
    return spec


def _symbol_color_for_value(
    renderer: ET.Element, value: str
) -> tuple[int, int, int, int]:
    [category] = [
        c
        for c in renderer.find('categories').findall('category')
        if c.get('value') == value
    ]
    symbol_name = category.get('symbol')
    [symbol] = [
        s
        for s in renderer.find('symbols').findall('symbol')
        if s.get('name') == symbol_name
    ]
    [color_option] = [
        opt for opt in symbol.iter('Option') if opt.get('name') == 'color'
    ]
    r, g, b, a = (int(x) for x in color_option.get('value').split(',')[:4])
    return r, g, b, a


class TestDynamicCategories:
    def test_categories_regenerated_from_real_data(self, tmp_path):
        spec = _dynamic_cat_spec(
            tmp_path, values=['new_y', 'new_x', 'new_x', None, 'new_z']
        )
        out = _run(tmp_path, [spec])
        root = _read_qgs_root(out)
        renderer = root.find('.//renderer-v2')
        assert renderer.get('attr') == 'conflict_field'
        values_found = {
            c.get('value') for c in renderer.find('categories').findall('category')
        }
        assert values_found == {'new_x', 'new_y', 'new_z'}
        for value in values_found:
            *_, alpha = _symbol_color_for_value(renderer, value)
            assert alpha == 127

    def test_color_stable_for_same_label_across_runs(self, tmp_path):
        dir1, dir2 = tmp_path / 'run1', tmp_path / 'run2'
        dir1.mkdir()
        dir2.mkdir()
        spec1 = _dynamic_cat_spec(dir1, values=['shared_label'])
        spec2 = _dynamic_cat_spec(dir2, values=['shared_label'])
        out1 = generate_qgz(
            MINIMAL_RECIPE,
            AdminId('US', 'NC', 'CAR'),
            layer_specs=[spec1],
            template_path=TEMPLATE_PATH,
            output_path=dir1 / 'out.qgz',
        )
        out2 = generate_qgz(
            MINIMAL_RECIPE,
            AdminId('US', 'NC', 'CAR'),
            layer_specs=[spec2],
            template_path=TEMPLATE_PATH,
            output_path=dir2 / 'out.qgz',
        )
        renderer1 = _read_qgs_root(out1).find('.//renderer-v2')
        renderer2 = _read_qgs_root(out2).find('.//renderer-v2')
        assert _symbol_color_for_value(
            renderer1, 'shared_label'
        ) == _symbol_color_for_value(renderer2, 'shared_label')

    def test_missing_column_leaves_baked_categories_and_warns(self, tmp_path):
        spec = _spec(
            tmp_path,
            role='output',
            entity_type='footprint',
            source='dynamiccat',
            display_name='US-NC-CAR_footprint-dynamiccat-nodata',
            combined=True,
        )
        pd.DataFrame({'other_column': ['x', 'y']}).to_parquet(spec.attr_path)
        with pytest.warns(UserWarning, match='conflict_field'):
            out = _run(tmp_path, [spec])
        root = _read_qgs_root(out)
        renderer = root.find('.//renderer-v2')
        values_found = {
            c.get('value') for c in renderer.find('categories').findall('category')
        }
        assert values_found == {'old_a', 'old_b'}


class TestRenderModes:
    """Re-rendering a cloned template layer for differently-shaped data.

    The delivery bundle splits the schema: centroids carry the attributes,
    polygons carry only shape. The template was authored for polygons that
    carry both, so its symbology has to be rewritten for each.
    """

    FILL_LAYER = (
        '<maplayer geometry="Polygon" wkbType="MultiPolygon">'
        '<renderer-v2 type="categorizedSymbol" attr="occupancy_type">'
        '<symbols>'
        '<symbol type="fill" name="0" alpha="1">'
        '<layer class="SimpleFill">'
        '<Option type="Map">'
        '<Option type="QString" name="color" value="48,191,226,126"/>'
        '<Option type="QString" name="outline_color" value="0,0,0,255"/>'
        '</Option></layer></symbol>'
        '<symbol type="fill" name="1" alpha="1">'
        '<layer class="SimpleFill">'
        '<Option type="Map">'
        '<Option type="QString" name="color" value="200,10,10,255"/>'
        '</Option></layer></symbol>'
        '</symbols></renderer-v2></maplayer>'
    )

    def _clone(self):
        return ET.fromstring(self.FILL_LAYER)

    def test_points_convert_fills_to_markers(self):
        clone = self._clone()

        generator_module._fills_to_markers(clone)

        symbols = clone.findall('renderer-v2/symbols/symbol')
        assert [sym.get('type') for sym in symbols] == ['marker', 'marker']
        assert all(sym.find('layer').get('class') == 'SimpleMarker' for sym in symbols)

    def test_points_keep_each_category_color(self):
        """A points view must stay readable against the polygon views."""
        clone = self._clone()

        generator_module._fills_to_markers(clone)

        colors = [
            option.get('value')
            for option in clone.iter('Option')
            if option.get('name') == 'color'
        ]
        assert colors == ['48,191,226,126', '200,10,10,255']

    def test_points_restate_the_geometry_type(self):
        clone = self._clone()

        generator_module._fills_to_markers(clone)

        assert clone.get('geometry') == 'Point'
        assert clone.get('wkbType') == 'Point'

    def test_points_keep_the_classifying_attribute(self):
        clone = self._clone()

        generator_module._fills_to_markers(clone)

        assert clone.find('renderer-v2').get('attr') == 'occupancy_type'

    def test_outline_drops_the_classification(self):
        """Geometry-only data cannot be classified, only shown."""
        clone = self._clone()

        generator_module._to_outline(clone)

        renderer = clone.find('renderer-v2')
        assert renderer.get('type') == 'singleSymbol'
        assert renderer.get('attr') is None

    def test_outline_is_unfilled(self):
        clone = self._clone()

        generator_module._to_outline(clone)

        options = {
            option.get('name'): option.get('value')
            for option in clone.iter('Option')
            if option.get('name')
        }
        assert options['style'] == 'no'
        assert options['outline_style'] == 'solid'

    def test_default_render_changes_nothing(self):
        clone = self._clone()

        generator_module._apply_render_mode(clone, 'default')

        assert clone.find('renderer-v2/symbols/symbol').get('type') == 'fill'
        assert clone.get('geometry') == 'Polygon'

    def test_classifying_attr_reads_the_renderer(self):
        assert generator_module._classifying_attr(self._clone()) == 'occupancy_type'
        assert generator_module._classifying_attr(ET.fromstring('<maplayer/>')) is None
