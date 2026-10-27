"""Observed/imputed split of classified QGIS layers."""

import xml.etree.ElementTree as ET

import pandas as pd

from openplaces.viz.qgis_map.generator import (
    _add_join,
    _apply_imputed_split,
    _category_literal,
    _order_numbered_groups,
    _split_group_path,
    _split_observed_imputed,
)


def test_untyped_category_literals_quote_codes_that_only_parse_as_numbers():
    assert _category_literal('2.5', None) == '2.5'
    assert _category_literal('-3', None) == '-3'
    assert _category_literal('01', None) == "'01'"
    assert _category_literal('nan', None) == "'nan'"
    assert _category_literal('inf', None) == "'inf'"


def _symbol(name: str, color: str) -> str:
    return (
        f'<symbol type="fill" name="{name}" alpha="1">'
        '<layer class="SimpleFill"><Option type="Map">'
        f'<Option type="QString" name="color" value="{color}"/>'
        '</Option></layer></symbol>'
    )


def _categorized(attr: str = 'roof_shape') -> ET.Element:
    return ET.fromstring(
        '<maplayer><id>a</id><layername>Roof shape</layername>'
        '<datasource>./points.parquet</datasource>'
        f'<renderer-v2 type="categorizedSymbol" attr="{attr}" symbollevels="0">'
        '<categories>'
        '<category value="Gable" label="Gable" symbol="0" render="true"/>'
        '<category value="It\'s flat" label="Flat" symbol="1" render="false"/>'
        '<category type="double" value="2" label="Two" symbol="0" render="true"/>'
        '<category type="NULL" value="NULL" label="None" symbol="2"/>'
        '<category value="" label="Other" symbol="2" render="true"/>'
        '</categories><symbols>'
        + _symbol('0', '1,1,1,255')
        + _symbol('1', '2,2,2,255')
        + _symbol('2', '3,3,3,255')
        + '</symbols><effect enabled="0"/></renderer-v2>'
        '<blendMode>0</blendMode></maplayer>'
    )


def _graduated() -> ET.Element:
    return ET.fromstring(
        '<maplayer><id>g</id>'
        '<renderer-v2 type="graduatedSymbol"'
        ' attr="coalesce(&quot;year_built&quot;, 1500.0)">'
        '<ranges>'
        '<range lower="1950" upper="2000" label="1950 - 2000" symbol="1"'
        ' render="true"/>'
        '<range lower="1499.5" upper="1500.5" label="No value" symbol="null"'
        ' render="true"/>'
        '<range lower="1900" upper="1950" label="1900 - 1950" symbol="0"'
        ' render="true"/>'
        '</ranges><symbols>'
        + _symbol('0', '1,1,1,255')
        + _symbol('1', '2,2,2,255')
        + _symbol('null', '9,9,9,255')
        + '</symbols></renderer-v2></maplayer>'
    )


def _parents(clone: ET.Element) -> list[ET.Element]:
    renderer = clone.find('renderer-v2')
    assert renderer.get('type') == 'RuleRenderer'
    return renderer.find('rules').findall('rule')


PROBE = "strpos(coalesce(\"roof_shape_source\", ''), 'imputed')"
IMPUTED = f'{PROBE} > 0'
OBSERVED = f'{PROBE} = 0'


class TestCategorized:
    def test_two_checked_parents_with_provenance_filters(self):
        clone = _categorized()
        assert _split_observed_imputed(clone, source_field='roof_shape_source')
        observed, imputed = _parents(clone)
        assert (observed.get('label'), observed.get('filter')) == (
            'Observed',
            OBSERVED,
        )
        assert (imputed.get('label'), imputed.get('filter')) == ('Imputed', IMPUTED)
        assert observed.get('checkstate') is None
        assert imputed.get('checkstate') is None
        assert observed.get('symbol') is None

    def test_children_carry_category_filters_and_labels(self):
        clone = _categorized()
        _split_observed_imputed(clone, source_field='roof_shape_source')
        for parent in _parents(clone):
            children = parent.findall('rule')
            assert [c.get('filter') for c in children] == [
                '"roof_shape" = \'Gable\'',
                "\"roof_shape\" = 'It''s flat'",
                '"roof_shape" = 2',
                '"roof_shape" IS NULL',
                'ELSE',
            ]
            assert [c.get('label') for c in children] == [
                'Gable',
                'Flat',
                'Two',
                'None',
                'Other',
            ]
            assert children[1].get('checkstate') == '0'
            assert children[0].get('checkstate') is None

    def test_every_rule_has_its_own_copy_of_its_symbol(self):
        clone = _categorized()
        _split_observed_imputed(clone, source_field='roof_shape_source')
        renderer = clone.find('renderer-v2')
        keys = [r.get('symbol') for r in renderer.iter('rule') if r.get('symbol')]
        names = [s.get('name') for s in renderer.find('symbols')]
        assert len(keys) == 10
        assert len(set(keys)) == 10
        assert sorted(keys) == sorted(names)
        colors = {
            s.get('name'): s.find('.//Option/Option').get('value')
            for s in renderer.find('symbols')
        }
        first = _parents(clone)[0].findall('rule')
        assert colors[first[0].get('symbol')] == '1,1,1,255'
        assert colors[first[2].get('symbol')] == '1,1,1,255'
        assert colors[first[1].get('symbol')] == '2,2,2,255'

    def test_keep_writes_only_the_named_parent(self):
        clone = _categorized()
        assert _split_observed_imputed(
            clone, source_field='roof_shape_source', keep=('Imputed',)
        )
        [imputed] = _parents(clone)
        assert (imputed.get('label'), imputed.get('filter')) == ('Imputed', IMPUTED)

    def test_renderer_keeps_its_position_and_effect(self):
        clone = _categorized()
        _split_observed_imputed(clone, source_field='roof_shape_source')
        assert [c.tag for c in clone] == [
            'id',
            'layername',
            'datasource',
            'renderer-v2',
            'blendMode',
        ]
        assert clone.find('renderer-v2/effect') is not None
        assert len({r.get('key') for r in clone.iter('rule')}) == 12


class TestGraduated:
    def test_ranges_sorted_and_open_below_only_on_a_shared_bound(self):
        clone = _graduated()
        assert _split_observed_imputed(clone, source_field='year_built_source')
        ref = '(coalesce("year_built", 1500.0))'
        for parent in _parents(clone):
            children = parent.findall('rule')
            assert [c.get('filter') for c in children] == [
                f'{ref} >= 1499.5 AND {ref} <= 1500.5',
                f'{ref} >= 1900 AND {ref} <= 1950',
                f'{ref} > 1950 AND {ref} <= 2000',
            ]
            assert [c.get('label') for c in children] == [
                'No value',
                '1900 - 1950',
                '1950 - 2000',
            ]
        assert 'year_built_source' in _parents(clone)[1].get('filter')


def _points(tmp_path, sources) -> object:
    path = tmp_path / 'points.parquet'
    pd.DataFrame(
        {'roof_shape': ['Gable'] * len(sources), 'roof_shape_source': sources}
    ).to_parquet(path)
    return path


class TestApplyImputedSplit:
    COLUMNS = {'roof_shape', 'roof_shape_source'}

    def test_point_layer_gets_token_subsets_and_keeps_its_renderer(self, tmp_path):
        data = _points(tmp_path, ['parcel', 'm-v1+imputed', None, "it's+imputed"])
        clone = _categorized()
        twin = _apply_imputed_split(clone, 'roof_shape', self.COLUMNS, data_path=data)
        listed = "'it''s+imputed', 'm-v1+imputed'"
        assert clone.findtext('datasource') == (
            './points.parquet|subset="roof_shape_source" IS NULL OR '
            f'"roof_shape_source" NOT IN ({listed})'
        )
        assert twin.findtext('datasource') == (
            f'./points.parquet|subset="roof_shape_source" IN ({listed})'
        )
        assert clone.find('renderer-v2').get('type') == 'categorizedSymbol'
        assert twin.find('renderer-v2').get('type') == 'categorizedSymbol'
        assert (twin.findtext('id'), twin.findtext('layername')) == (
            'a_imputed',
            'Roof shape (imputed)',
        )

    def test_nothing_imputed_leaves_the_view_whole_and_the_copy_empty(self, tmp_path):
        data = _points(tmp_path, ['parcel', None])
        clone = _categorized()
        twin = _apply_imputed_split(clone, 'roof_shape', self.COLUMNS, data_path=data)
        assert clone.findtext('datasource') == './points.parquet'
        assert twin.findtext('datasource') == './points.parquet|subset=1 = 0'

    def test_joined_polygon_splits_into_one_rule_each(self):
        clone = _categorized()
        _add_join(clone, attr_id='attr', field='footprint_id', subset=['roof_shape'])
        twin = _apply_imputed_split(
            clone,
            'roof_shape',
            self.COLUMNS,
            join_layer_name='Attributes - points',
        )
        assert [p.get('filter') for p in _parents(clone)] == [OBSERVED]
        assert [p.get('filter') for p in _parents(twin)] == [IMPUTED]
        for layer in (clone, twin):
            fields = [f.get('name') for f in layer.iter('field')]
            assert fields == ['roof_shape', 'roof_shape_source']
        assert clone.findtext('datasource') == './points.parquet'

    def test_join_without_custom_prefix_uses_the_layer_name(self):
        clone = _categorized(attr='pts_roof_shape')
        _add_join(clone, attr_id='attr', field='footprint_id', subset=['roof_shape'])
        clone.find('vectorjoins/join').set('hasCustomPrefix', '0')
        twin = _apply_imputed_split(
            clone,
            'pts_roof_shape',
            self.COLUMNS,
            join_layer_name='pts',
        )
        [imputed] = _parents(twin)
        assert imputed.get('filter') == (
            "strpos(coalesce(\"pts_roof_shape_source\", ''), 'imputed') > 0"
        )
        assert imputed.find('rule').get('filter') == '"pts_roof_shape" = \'Gable\''

    def test_layer_without_a_source_column_is_unchanged(self):
        clone = _categorized()
        before = ET.tostring(clone)
        assert _apply_imputed_split(clone, 'roof_shape', {'roof_shape'}) is None
        assert ET.tostring(clone) == before

    def test_single_symbol_renderer_is_unchanged(self):
        clone = ET.fromstring(
            '<maplayer><id>s</id><renderer-v2 type="singleSymbol"><symbols>'
            + _symbol('0', '1,1,1,255')
            + '</symbols></renderer-v2></maplayer>'
        )
        before = ET.tostring(clone)
        assert _apply_imputed_split(clone, 'roof_shape', self.COLUMNS) is None
        assert ET.tostring(clone) == before


def test_split_groups_are_renamed_and_numbered_first():
    assert _split_group_path('Buildings/Point geometries') == 'Buildings/2. Original'
    assert _split_group_path('Buildings/Polygon geometries') == 'Buildings/3. Polygons'
    assert _split_group_path('Administrative units') == 'Administrative units'
    parent = ET.fromstring(
        '<g><g name="2. Original"/><layer/><g name="3. Polygons"/>'
        '<g name="1. Imputed"/><g name="Other"/></g>'
    )
    _order_numbered_groups(parent, 'g')
    assert [c.get('name') for c in parent] == [
        '1. Imputed',
        None,
        '2. Original',
        '3. Polygons',
        'Other',
    ]
