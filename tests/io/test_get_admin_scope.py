"""get_admin() selects the spine by admin level, not by string prefix."""

import pytest

from openplaces.core.schema import AdminId
from openplaces.io.readers import get_admin


class TestGetAdminSelectionIsLevelBounded:
    """A mixed-width sibling must not be selected by a shorter id.

    'US-NC-WA' is Wake County's pre-2026 id, superseded when North
    Carolina widened to three-character codes, and 'US-NC-WAR' is Warren
    County's current one. Both are real entries in the admin spine, and
    the first is a plain string prefix of the second.
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
