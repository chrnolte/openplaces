"""admin_ancestor is the accessor a consumer uses instead of splitting an
id on its separator: the separator, level widths and level numbering are
the hub's to change."""

import pytest

from openplaces.core.schema import AdminId, admin_ancestor


@pytest.mark.parametrize(
    ('admin_id', 'level', 'expected'),
    [
        ('US-NC-WAR', 1, 'US'),
        ('US-NC-WAR', 2, 'US-NC'),
        ('US-NC-WAR', 3, 'US-NC-WAR'),
        ('US-NC-WAR', 0, None),
        ('US-NC', 3, None),
    ],
)
def test_ancestor_by_level(admin_id, level, expected):
    assert admin_ancestor(admin_id, level) == expected


def test_accepts_an_admin_id_object_and_returns_a_string():
    assert admin_ancestor(AdminId('US', 'NC', 'WAR'), 2) == 'US-NC'
