"""Tests for the per-parent admin layer set.

The point of `context_layers` is that drawing one place should not build a
global layer per level. Each layer is scoped to the ancestor it sits
inside, so the tests that matter are the ones pinning that scoping rather
than the shape of the returned dicts.
"""

import pytest

from openplaces.io.admin import context_layers
from openplaces.io.readers import get_admin_ids


class TestLayerSet:
    def test_a_country_needs_only_its_own_outline(self):
        assert context_layers('US') == [
            {'admin_level': 1, 'scope': 'US', 'role': 'outline'}
        ]

    def test_each_layer_is_scoped_to_the_ancestor_it_sits_inside(self):
        # A level-4 identifier: the country outline, the country's level-2
        # units, the level-2 parent's level-3 units, and the level-3
        # parent's level-4 units.
        layers = context_layers('DE-BY-MK-ML')
        assert [(d['admin_level'], d['scope']) for d in layers] == [
            (1, 'DE'),
            (2, 'DE'),
            (3, 'DE-BY'),
            (4, 'DE-BY-MK'),
        ]

    def test_the_innermost_layer_holds_the_requested_unit(self):
        layers = context_layers('US-MA-SOM')
        focus = [d for d in layers if d['role'] == 'focus']
        assert len(focus) == 1
        assert focus[0] == {'admin_level': 3, 'scope': 'US-MA', 'role': 'focus'}
        assert 'US-MA-SOM' in get_admin_ids(3, 'US-MA')

    def test_outermost_first(self):
        levels = [d['admin_level'] for d in context_layers('DE-BY-MK-ML')]
        assert levels == sorted(levels)

    @pytest.mark.parametrize('bad', ['', '   '])
    def test_a_non_identifier_is_refused(self, bad):
        with pytest.raises(ValueError):
            context_layers(bad)


class TestScopingIsWorthIt:
    def test_one_town_reads_far_less_than_the_world(self):
        # The whole justification for per-parent products. Asserted as a
        # ratio rather than a fixed count so it tracks the claim, not a
        # snapshot of how many units the spine happens to hold.
        layers = [d for d in context_layers('US-MA-SOM') if d['role'] != 'outline']
        scoped = sum(len(get_admin_ids(d['admin_level'], d['scope'])) for d in layers)
        unscoped = sum(len(get_admin_ids(d['admin_level'])) for d in layers)
        assert scoped < unscoped / 50


class TestRecipesArePartitionedToMatch:
    @pytest.mark.parametrize(('level', 'parent_level'), [(2, 1), (3, 2), (4, 3)])
    def test_each_product_is_stored_under_its_parent(self, level, parent_level):
        # `context_layers` scopes a layer to its parent, so the recipe has
        # to partition its output the same way or the notebook would write
        # one unit's children over another's.
        from openplaces.recipe import get_recipe_by_id

        recipe = get_recipe_by_id(f'admin-openplaces-2026_admin{level}')
        assert recipe['save_to']['admin_level'] == parent_level

    def test_the_country_layer_stays_global(self):
        from openplaces.recipe import get_recipe_by_id

        recipe = get_recipe_by_id('admin-openplaces-2026_admin1')
        assert 'admin_level' not in recipe['save_to']
