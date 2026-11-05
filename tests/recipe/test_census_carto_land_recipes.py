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
    LAND_GEOMETRY_FINEST_PRODUCT,
    LAND_GEOMETRY_PRODUCTS,
    LAND_GEOMETRY_SOURCE,
    _land_geometry_recipe_ids,
    _land_product,
)
from openplaces.path import recipe_roots
from openplaces.recipe import find_admin_recipe_id, get_recipe_by_id

NEW_ENGLAND = ['CT', 'ME', 'MA', 'NH', 'RI', 'VT']


def _land_rows():
    admin = find_recipes('admin', stage='ingest')
    return admin[
        admin['source_id'].eq(LAND_GEOMETRY_SOURCE)
        & admin['version'].map(_land_product).isin(LAND_GEOMETRY_PRODUCTS.values())
    ]


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
        assert candidates['version'].map(_land_product).isna().all()

    def test_county_id_overrides_match_the_tiger_recipe(self):
        # The unit ids the spine lacks are one list, copied beside each
        # recipe because the index function reads it by recipe.
        tiger = pd.read_csv(
            _recipe_file('US_admin-census-2025_admin3-ids', '.csv'), dtype=str
        )
        for product in LAND_GEOMETRY_PRODUCTS.values():
            ours = pd.read_csv(
                _recipe_file(f'US_admin-census-{product}2025_admin3-ids', '.csv'),
                dtype=str,
            )
            pd.testing.assert_frame_equal(ours, tiger)

    @pytest.mark.parametrize('abbr', NEW_ENGLAND)
    def test_new_england_towns_are_level_three_recipes_of_their_state(self, abbr):
        recipe = get_recipe_by_id(f'US-{abbr}_admin-census-carto500k2025_admin3')
        assert str(recipe['admin_id']) == f'US-{abbr}'

    @pytest.mark.parametrize('level', [2, 3, 4])
    def test_tiger_stays_the_guessed_census_recipe(self, level):
        # A version that leads with letters sorts below a bare year, so
        # an un-sourced lookup never lands on a land-only layer.
        recipe_id = find_admin_recipe_id('US', level, silent=True)
        assert _land_product(recipe_id.split('-')[-1].split('_')[0]) is None


class TestLandRecipeChoice:
    def test_a_state_outside_new_england_reads_the_national_county_file(self):
        ids = _land_geometry_recipe_ids('carto500k', 3, [AdminId('US-MI')])
        assert ids == ['US_admin-census-carto500k2025_admin3']

    def test_new_england_reads_its_towns(self):
        # The national county file covers the scope too; it holds no
        # New England row, so reading it adds nothing.
        ids = _land_geometry_recipe_ids('carto500k', 3, [AdminId('US-MA')])
        assert ids == [
            'US-MA_admin-census-carto500k2025_admin3',
            'US_admin-census-carto500k2025_admin3',
        ]

    def test_a_whole_country_reads_counties_and_every_new_england_state(self):
        ids = _land_geometry_recipe_ids('carto500k', 3, None)
        assert len(ids) == 1 + len(NEW_ENGLAND)

    def test_a_coarse_resolution_uses_it_where_the_source_has_it(self):
        ids = _land_geometry_recipe_ids('carto5m', 3, [AdminId('US-MI')])
        assert ids == ['US_admin-census-carto5m2025_admin3']
        ids = _land_geometry_recipe_ids('carto20m', 2, [AdminId('US')])
        assert ids == ['US_admin-census-carto20m2025_admin2']

    def test_a_scope_without_the_coarse_file_falls_back_to_the_finest(self):
        # Towns and county subdivisions exist at 500k only.
        ids = _land_geometry_recipe_ids('carto5m', 3, [AdminId('US-MA')])
        assert ids == [
            f'US-MA_admin-census-{LAND_GEOMETRY_FINEST_PRODUCT}2025_admin3',
            'US_admin-census-carto5m2025_admin3',
        ]
        ids = _land_geometry_recipe_ids('carto20m', 4, [AdminId('US-MI')])
        assert ids == ['US_admin-census-carto500k2025_admin4']

    @pytest.mark.parametrize(
        ('version', 'expected'),
        [
            ('carto500k2025', 'carto500k'),
            ('carto5m2025', 'carto5m'),
            ('carto20m2031', 'carto20m'),
            ('2025', None),
            (2025, None),
        ],
    )
    def test_the_product_is_the_version_without_its_year(self, version, expected):
        assert _land_product(version) == expected
