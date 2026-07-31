import pytest

from openplaces.core.schema import AdminId


class TestAdminIdTruncateToLevel:
    def test_truncates_to_fewer_levels(self):
        admin_id = AdminId('US', 'MA', 'MI')
        assert admin_id.truncate_to_level(2) == AdminId('US', 'MA')

    def test_full_level_returns_equal_admin_id(self):
        admin_id = AdminId('US', 'MA', 'MI')
        assert admin_id.truncate_to_level(3) == admin_id

    @pytest.mark.parametrize('level', [0, -1])
    def test_non_positive_level_returns_none(self, level):
        admin_id = AdminId('US', 'MA', 'MI')
        assert admin_id.truncate_to_level(level) is None

    def test_level_beyond_depth_is_a_no_op(self):
        admin_id = AdminId('US', 'MA')
        assert admin_id.truncate_to_level(5) == admin_id
