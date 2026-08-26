"""Tests that the GADM admin recipes pin identifiers to the committed spine.

The GADM parquets are the world's default admin geometry, so an identifier
minted here rather than looked up in the spine puts two vintages in one
file: `get_admin(..., geom=True)` then outer-joins them and returns rows
that carry a name but no geometry beside rows that carry geometry but no
name. That is what the 2026-06 vintage did, at 27,060 of 48,695 level-3
units.

These are recipe-shape checks, so they read no data.
"""

import pytest

from openplaces.recipe import get_recipe_by_id

MINTER = 'openplaces.io.admin_codes.assign_admin_ids'

# level -> the level its parent id is crosswalked from, and the source
# column that crosswalk joins on
PARENT_OF = {
    2: (1, 'admin1_id_a3'),
    3: (2, 'admin2_id_gadm'),
    4: (3, 'admin3_id_gadm'),
}


@pytest.fixture(scope='module', params=sorted(PARENT_OF))
def level_and_recipe(request):
    level = request.param
    return level, get_recipe_by_id(f'admin-gadm-4~1_admin{level}')


class TestIdentifiersComeFromTheSpine:
    def test_uses_the_shared_minter(self, level_and_recipe):
        level, recipe = level_and_recipe
        create_index = recipe.get('create_index') or {}
        assert create_index.get('function') == MINTER, (
            f'admin-gadm-4~1_admin{level} must build its index with '
            f'{MINTER}, which reproduces the code the committed spine '
            'already records for a unit.'
        )

    def test_names_the_level_it_is_indexing(self, level_and_recipe):
        level, recipe = level_and_recipe
        args = (recipe.get('create_index') or {}).get('args') or {}
        assert args.get('new_admin_id_col') == f'admin{level}_id'
        assert args.get('parent_admin_id_col') == f'admin{PARENT_OF[level][0]}_id'

    def test_does_not_also_mint_its_own(self, level_and_recipe):
        level, recipe = level_and_recipe
        # The legacy waterfall in `io.admin` is kept for reference only.
        # Wiring it back in would mint under pre-2026 rules and re-create
        # the mixed-vintage parquet.
        assert 'index_function' not in recipe, (
            f'admin-gadm-4~1_admin{level} declares both an index_function '
            'and a create_index; only the latter pins to the spine.'
        )


class TestParentsResolveBeforeChildren:
    def test_crosswalks_the_level_above(self, level_and_recipe):
        level, recipe = level_and_recipe
        parent_level, join_column = PARENT_OF[level]
        crosswalk = recipe.get('admin_id_crosswalk') or {}
        assert crosswalk.get('admin_level') == parent_level
        assert crosswalk.get('admin_id_column') == join_column
        assert crosswalk.get('admin_recipe_id') == f'admin-gadm-4~1_admin{parent_level}'

    def test_the_join_column_is_mapped_from_the_source(self, level_and_recipe):
        level, recipe = level_and_recipe
        _, join_column = PARENT_OF[level]
        # `assign_admin_ids` raises on a blank parent, so the column the
        # crosswalk joins on has to survive the column mapping.
        assert join_column in recipe['columns'], (
            f'admin-gadm-4~1_admin{level} crosswalks on {join_column}, '
            'which its `columns` block does not map from the source.'
        )
