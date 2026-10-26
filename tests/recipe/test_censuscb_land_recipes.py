"""The land-only boundary recipes are for drawing and never join the spine.

The Census cartographic boundary files are generalized and clipped to the
shoreline. Spine auto-discovery ranks every `*_admin{N}` ingest recipe by
scope and version and takes polygons from the winner, so each of these
recipes carries `exclude_from_auto_discover: true`; this file pins that,
and the way `get_admin(geom='land')` chooses among them.
"""

import pandas as pd
import pytest
import yaml

from openplaces.core.schema import AdminId
from openplaces.diagnostics import find_recipes
from openplaces.io.readers import (
    LAND_GEOMETRY_FINEST_SOURCE,
    LAND_GEOMETRY_SOURCES,
    _land_geometry_recipe_ids,
)
from openplaces.path import recipe_roots
from openplaces.recipe import get_recipe_by_id

NEW_ENGLAND = ['CT', 'ME', 'MA', 'NH', 'RI', 'VT']


def _land_rows():
    admin = find_recipes('admin', stage='ingest')
    return admin[admin['source_id'].isin(LAND_GEOMETRY_SOURCES.values())]


def _recipe_file(stem, suffix):
    """Path of the recipe file or sidecar named `stem` + `suffix`."""
    for root in recipe_roots():
        for path in root.rglob(stem + suffix):
            return path
    raise FileNotFoundError(stem + suffix)


def _recipe_yaml(recipe_id):
    path = _recipe_file(recipe_id, '.yaml')
    return yaml.safe_load(path.read_text(encoding='utf-8'))


class TestRecipesStayOutOfTheSpine:
    def test_every_land_recipe_is_excluded_from_auto_discovery(self):
        rows = _land_rows()
        assert len(rows) == 3 * 2 + 1 + len(NEW_ENGLAND)
        assert rows['exclude_from_auto_discover'].all()

    def test_every_land_recipe_records_unrestricted_terms(self):
        for recipe_id in _land_rows()['recipe_id']:
            source = _recipe_yaml(recipe_id)['entity']['source']
            assert source['redistribution_restricted'] is False, recipe_id
            assert source['license'].startswith('public-domain'), recipe_id

    def test_no_land_recipe_is_the_spine_source_of_a_level(self):
        # The spine's pick for a unit is the first non-excluded recipe;
        # none of these may be among the candidates.
        admin = find_recipes('admin', stage='ingest')
        candidates = admin[~admin['exclude_from_auto_discover']]
        assert not candidates['source_id'].isin(LAND_GEOMETRY_SOURCES.values()).any()

    def test_county_id_overrides_match_the_tiger_recipe(self):
        # The unit ids the spine lacks are one list, copied beside each
        # source because the index function reads it by source.
        tiger = pd.read_csv(
            _recipe_file('US_admin-census-2025_admin3-ids', '.csv'), dtype=str
        )
        for source_id in LAND_GEOMETRY_SOURCES.values():
            ours = pd.read_csv(
                _recipe_file(f'US_admin-{source_id}-2025_admin3-ids', '.csv'),
                dtype=str,
            )
            pd.testing.assert_frame_equal(ours, tiger)

    @pytest.mark.parametrize('abbr', NEW_ENGLAND)
    def test_new_england_towns_are_level_three_recipes_of_their_state(self, abbr):
        recipe = get_recipe_by_id(f'US-{abbr}_admin-censuscb-2025_admin3')
        assert str(recipe['admin_id']) == f'US-{abbr}'


class TestLandRecipeChoice:
    def test_a_state_outside_new_england_reads_the_national_county_file(self):
        ids = _land_geometry_recipe_ids('censuscb', 3, [AdminId('US-MI')])
        assert ids == ['US_admin-censuscb-2025_admin3']

    def test_new_england_reads_its_towns(self):
        # The national county file covers the scope too; it holds no
        # New England row, so reading it adds nothing.
        ids = _land_geometry_recipe_ids('censuscb', 3, [AdminId('US-MA')])
        assert ids == [
            'US-MA_admin-censuscb-2025_admin3',
            'US_admin-censuscb-2025_admin3',
        ]

    def test_a_whole_country_reads_counties_and_every_new_england_state(self):
        ids = _land_geometry_recipe_ids('censuscb', 3, None)
        assert len(ids) == 1 + len(NEW_ENGLAND)

    def test_a_coarse_resolution_uses_it_where_the_source_has_it(self):
        ids = _land_geometry_recipe_ids('censuscb5m', 3, [AdminId('US-MI')])
        assert ids == ['US_admin-censuscb5m-2025_admin3']
        ids = _land_geometry_recipe_ids('censuscb20m', 2, [AdminId('US')])
        assert ids == ['US_admin-censuscb20m-2025_admin2']

    def test_a_scope_without_the_coarse_file_falls_back_to_the_finest(self):
        # Towns and county subdivisions exist at 500k only.
        ids = _land_geometry_recipe_ids('censuscb5m', 3, [AdminId('US-MA')])
        assert ids == [
            f'US-MA_admin-{LAND_GEOMETRY_FINEST_SOURCE}-2025_admin3',
            'US_admin-censuscb5m-2025_admin3',
        ]
        ids = _land_geometry_recipe_ids('censuscb20m', 4, [AdminId('US-MI')])
        assert ids == ['US_admin-censuscb-2025_admin4']
