"""Tests for openplaces.viz.qgis_map.resolver.

Exercises `resolve_layers` against real, committed curate recipes (rather
than synthetic ones) so dependency-graph walking runs against the actual
recipes/ tree, per the project's stated preference for real recipe fixtures
over mocks (see tests/io/curator/test_merge_enrichments.py). No real data
files are required: `filter_existing=False` resolves paths without
touching the filesystem, and `mock_data_root` points every bucket at an
empty tmp_path so `filter_existing=True` behavior is deterministic.
"""

import pandas as pd
import pytest

from openplaces.core.schema import AdminId, Entity
from openplaces.io import to_parquet
from openplaces.recipe import get_output_path, get_recipe_by_id
from openplaces.viz.qgis_map.resolver import (
    RENDER_OUTLINE,
    RENDER_POINTS,
    _build_spec,
    resolve_layers,
)

PARCEL_RECIPE = 'US_parcel-openplaces-2026'
FOOTPRINT_RECIPE = 'US_footprint-cheer-2026'
HARMONIZE_RECIPE = 'US_footprint-spine-2026'
COUNTY = 'US-NC-CE'


class TestResolveLayersStageValidation:
    def test_rejects_non_curate_recipe(self):
        with pytest.raises(ValueError, match='curate'):
            resolve_layers(HARMONIZE_RECIPE, COUNTY)


class TestResolveLayersRealRecipeGraph:
    def test_output_spec_present(self, mock_data_root):
        specs = resolve_layers(PARCEL_RECIPE, COUNTY, filter_existing=False)
        outputs = [s for s in specs if s.role == 'output']
        assert len(outputs) == 1
        assert outputs[0].depth == 0
        assert outputs[0].entity_type == 'parcel'
        assert outputs[0].source == 'openplaces'

    def test_input_specs_are_ingest_stage_only(self, mock_data_root):
        specs = resolve_layers(PARCEL_RECIPE, COUNTY, filter_existing=False)
        inputs = [s for s in specs if s.role == 'input']
        assert len(inputs) > 0
        for spec in inputs:
            recipe = get_recipe_by_id(spec.recipe_id)
            assert recipe.get('stage', 'ingest') == 'ingest'

    def test_only_known_roles_present(self, mock_data_root):
        # Harmonize/enrich intermediates are walked through, not materialized.
        specs = resolve_layers(PARCEL_RECIPE, COUNTY, filter_existing=False)
        assert {s.role for s in specs} <= {'output', 'input', 'admin'}

    def test_admin_context_layer_present_by_default(self, mock_data_root):
        specs = resolve_layers(PARCEL_RECIPE, COUNTY, filter_existing=False)
        admin_specs = [s for s in specs if s.role == 'admin']
        assert len(admin_specs) >= 1
        assert all(s.entity_type == 'admin' for s in admin_specs)

    def test_restricting_admin_context_levels_never_adds_admin_specs(
        self, mock_data_root
    ):
        default_specs = resolve_layers(PARCEL_RECIPE, COUNTY, filter_existing=False)
        restricted_specs = resolve_layers(
            PARCEL_RECIPE, COUNTY, admin_context_levels=(), filter_existing=False
        )
        default_admin_ids = {s.recipe_id for s in default_specs if s.role == 'admin'}
        restricted_admin_ids = {
            s.recipe_id for s in restricted_specs if s.role == 'admin'
        }
        assert restricted_admin_ids <= default_admin_ids

    def test_combined_output_recipe_has_matching_attr_and_geo_paths(
        self, mock_data_root
    ):
        specs = resolve_layers(FOOTPRINT_RECIPE, COUNTY, filter_existing=False)
        output = next(s for s in specs if s.role == 'output')
        assert output.combined is True
        assert output.attr_path == output.geo_path

    def test_non_combined_specs_have_distinct_attr_and_geo_paths(self, mock_data_root):
        specs = resolve_layers(PARCEL_RECIPE, COUNTY, filter_existing=False)
        output = next(s for s in specs if s.role == 'output')
        assert output.combined is False
        assert output.attr_path != output.geo_path

    def test_deterministic_ordering(self, mock_data_root):
        first = resolve_layers(PARCEL_RECIPE, COUNTY, filter_existing=False)
        second = resolve_layers(PARCEL_RECIPE, COUNTY, filter_existing=False)
        assert [s.recipe_id for s in first] == [s.recipe_id for s in second]


class TestResolveLayersFilterExisting:
    def test_empty_data_root_drops_everything(self, mock_data_root):
        specs = resolve_layers(PARCEL_RECIPE, COUNTY, filter_existing=True)
        assert specs == []

    def test_keeps_only_specs_with_files_on_disk(self, mock_data_root):
        unfiltered = resolve_layers(PARCEL_RECIPE, COUNTY, filter_existing=False)
        output = next(s for s in unfiltered if s.role == 'output')
        to_parquet(pd.DataFrame({'a': [1]}), output.attr_path)
        to_parquet(pd.DataFrame({'a': [1]}), output.geo_path)

        filtered = resolve_layers(PARCEL_RECIPE, COUNTY, filter_existing=True)
        assert [s.recipe_id for s in filtered] == [output.recipe_id]


class TestBuildSpecAdminTruncation:
    def test_resolves_at_coarser_save_level_than_target(self, mock_data_root):
        # A synthetic ingest recipe saved at admin level 2 (state), resolved
        # against a level-3 (county) target: _build_spec's caller is
        # responsible for truncating first, mirroring resolve_layers' own
        # `admin_id.truncate_to_level(get_save_admin_level(upstream))` call.
        recipe = {
            'stage': 'ingest',
            'admin_id': AdminId('US', 'NC'),
            'entity': Entity('footprint', 'teststate', '2025'),
            'save_to': {'data_dir': 'cache'},
            'process_by': {'admin_level': 2},
        }
        target = AdminId('US', 'NC', 'CE')
        resolved_admin = target.truncate_to_level(2)
        assert resolved_admin == AdminId('US', 'NC')

        spec = _build_spec(recipe, resolved_admin, role='input', depth=1, verbose=False)
        assert spec is not None
        assert spec.admin_id == AdminId('US', 'NC')
        expected_path = get_output_path(recipe, admin_id=AdminId('US', 'NC'))
        assert spec.attr_path == expected_path


class TestIncludeInputs:
    def test_inputs_dropped_leaves_output_and_admin(self, mock_data_root):
        specs = resolve_layers(
            FOOTPRINT_RECIPE, COUNTY, filter_existing=False, include_inputs=False
        )

        assert {s.role for s in specs} == {'output', 'admin'}

    def test_inputs_kept_by_default(self, mock_data_root):
        specs = resolve_layers(FOOTPRINT_RECIPE, COUNTY, filter_existing=False)

        assert any(s.role == 'input' for s in specs)


class TestDeliveryBundleLayers:
    """At the region the recipe delivers, the map is built from the bundle.

    The curated output is written per county, so its own save level cannot
    resolve a path for the region -- without this the output layer silently
    vanished from the map.
    """

    REGION = 'US-NC'

    def test_bundle_resolves_points_and_outline(self, mock_data_root):
        specs = resolve_layers(
            FOOTPRINT_RECIPE, self.REGION, filter_existing=False, include_inputs=False
        )
        outputs = [s for s in specs if s.role == 'output']

        assert [s.render for s in outputs] == [RENDER_POINTS, RENDER_OUTLINE]
        assert outputs[0].attr_path.name.endswith('_point.parquet')
        assert outputs[1].attr_path.name.endswith('_geo.parquet')

    def test_bundle_layers_stand_alone(self, mock_data_root):
        """Neither layer joins: points carry attributes, polygons show shape."""
        outputs = [
            s
            for s in resolve_layers(
                FOOTPRINT_RECIPE,
                self.REGION,
                filter_existing=False,
                include_inputs=False,
            )
            if s.role == 'output'
        ]

        assert all(s.combined for s in outputs)
        assert all(s.attr_path == s.geo_path for s in outputs)

    def test_county_scope_is_unchanged(self, mock_data_root):
        outputs = [
            s
            for s in resolve_layers(FOOTPRINT_RECIPE, COUNTY, filter_existing=False)
            if s.role == 'output'
        ]

        assert len(outputs) == 1
        assert outputs[0].render == 'default'
