"""get_admin() selects the spine by admin level, not by string prefix."""

import pytest

from openplaces.core.schema import AdminId
from openplaces.io.readers import get_admin, get_admin_ids


class TestGetAdminSelectionIsLevelBounded:
    """A mixed-width sibling must not be selected by a shorter id.

    'US-NC-WA' was Wake County's id before North Carolina widened to
    three-character codes, and 'US-NC-WAR' is Warren County's current
    one. The first is a plain string prefix of the second and names no
    unit today.
    """

    def test_a_shorter_sibling_id_selects_nothing(self):
        with pytest.raises(ValueError, match='No admin IDs'):
            get_admin(level=3, admin_id='US-NC-WA', silent=True)

    def test_a_unit_selects_only_itself(self):
        admin = get_admin(level=3, admin_id='US-NC-WAK', silent=True)
        assert list(admin.index) == ['US-NC-WAK']

    def test_a_parent_still_selects_its_children(self):
        admin = get_admin(level=3, admin_id='US-NC', silent=True)
        assert 'US-NC-WAR' in admin.index
        assert 'US-NC-WAK' in admin.index

    def test_the_empty_scope_still_selects_the_world(self):
        """Level 0 is the planet, so it covers every unit.

        A global recipe passes its own empty admin_id down to the
        selection below, where the level-boundary test above matches
        nothing: the separator it appends cannot start any id. Before
        this case was restored, every global recipe resolved to no rows.
        """
        admin = get_admin(level=1, admin_id=AdminId(), silent=True)
        assert 'US' in admin.index


class TestAWholeLevelWithGeometry:
    """A whole level with geometry reads the recipe's whole scope.

    The spine rebuild asks exactly this, to weight every unit. The
    default geometry recipe is chosen after the one place that bound
    `admin_ids` from a recipe, so the call raised UnboundLocalError
    instead of resolving every file the recipe wrote.
    """

    def test_the_whole_recipe_scope_is_requested(self, monkeypatch):
        from openplaces.io import readers

        seen = []

        class Resolved(Exception):
            pass

        def record(recipe, admin_id):
            seen.append(admin_id)
            raise Resolved

        monkeypatch.setattr(readers, '_get_output_admin_ids', record)
        with pytest.raises(Resolved):
            get_admin(level=2, geom=True, silent=True)
        assert seen == [None]


class TestAnEmptyScopeIsAnAnswer:
    """A unit with nothing below it at the requested level.

    Antarctica has no second-level units and an Andorran parish no
    third-level ones. A per-unit harmonize run of the world admin spine
    asks for exactly those, and both raised, which stopped the whole
    level-2 and level-3 rebuild under the orchestrator.
    """

    def test_no_unit_at_the_level_raises_by_default(self):
        with pytest.raises(ValueError, match='No admin IDs'):
            get_admin(level=2, admin_id='AQ', silent=True)

    def test_allow_empty_returns_an_empty_table(self):
        admin = get_admin(level=2, admin_id='AQ', silent=True, allow_empty=True)
        assert len(admin) == 0

    def test_allow_empty_returns_an_empty_id_list(self):
        assert get_admin_ids(3, 'AD-AL', allow_empty=True) == []

    def test_allow_empty_changes_nothing_for_a_populated_scope(self):
        assert 'US-NC-WAK' in get_admin_ids(3, 'US-NC', allow_empty=True)


def test_a_placeholder_output_is_skipped_when_files_are_concatenated(tmp_path):
    import pandas as pd

    from openplaces.io.readers import _concat_recipe_files

    real = tmp_path / 'US_x.parquet'
    pd.DataFrame({'admin2_id': ['US-MA'], 'name': ['Massachusetts']}).to_parquet(real)
    placeholder = tmp_path / 'AW_x.parquet'
    pd.DataFrame({'_join_id': pd.Series([], dtype='int64')}).to_parquet(placeholder)
    got = _concat_recipe_files(
        [placeholder, real], filters=[('admin2_id', 'in', ['US-MA'])]
    )
    assert got['name'].tolist() == ['Massachusetts']
    only = _concat_recipe_files([placeholder])
    assert only.empty
