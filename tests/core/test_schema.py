import pytest

from openplaces.core.schema import AdminId, Source


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


class TestSourceTerms:
    """The license fields a recipe records after checking a source's terms."""

    def test_unchecked_terms_default_to_none(self):
        source = Source('massgis')
        assert source.license is None
        assert source.terms_url is None
        # None means nobody checked, which is not the same as False
        assert source.redistribution_restricted is None

    def test_terms_are_kept_as_recorded(self):
        source = Source(
            source_id='gadm',
            license='non-commercial, no redistribution without permission',
            terms_url='https://example.org/license.html',
            redistribution_restricted=True,
        )
        assert source.license.startswith('non-commercial')
        assert source.terms_url == 'https://example.org/license.html'
        assert source.redistribution_restricted is True

    def test_terms_load_from_a_recipe_source_mapping(self):
        # Recipes reach Source as **source, so a YAML `license:` key only
        # works if it is an accepted parameter
        source = Source(**{'source_id': 'example', 'license': 'public-domain'})
        assert source.license == 'public-domain'
