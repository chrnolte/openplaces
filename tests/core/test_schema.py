import pytest

from openplaces.core.schema import (
    AdminId,
    Source,
    UsageRequirement,
    admin_scope_covers,
)


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


class TestAdminScopeCovers:
    """Containment is a level test, not a string-prefix test.

    The pair is real: 'US-NC-WA' is Wake County's pre-2026 id and
    'US-NC-WAR' is Warren County's current one, both in the admin spine.
    """

    def test_ancestor_covers_descendant(self):
        assert admin_scope_covers('US-NC', 'US-NC-WAR')

    def test_unit_covers_itself(self):
        assert admin_scope_covers('US-NC-WAR', 'US-NC-WAR')

    def test_mixed_width_sibling_is_not_covered(self):
        assert not admin_scope_covers('US-NC-WA', 'US-NC-WAR')

    def test_empty_scope_is_global(self):
        assert admin_scope_covers('', 'US-NC-WAR')
        assert admin_scope_covers(None, 'US-NC-WAR')

    def test_empty_admin_id_is_covered_only_by_global(self):
        assert not admin_scope_covers('US-NC', '')

    def test_admin_id_objects_are_accepted(self):
        assert admin_scope_covers(AdminId('US', 'NC'), AdminId('US', 'NC', 'WAR'))


class TestAdminIdLevelValidation:
    def test_a_level_holding_a_separator_is_rejected(self):
        with pytest.raises(ValueError):
            AdminId('US', 'NC-WAR')

    def test_a_level_holding_trailing_space_is_rejected(self):
        with pytest.raises(ValueError):
            AdminId('US ', 'NC')

    def test_superseded_long_level_codes_are_still_accepted(self):
        assert str(AdminId('US-MN-LO-X157')) == 'US-MN-LO-X157'


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


class TestUsageRequirement:
    """Access-eligibility conditions recorded on a source's terms."""

    def test_defaults_to_none_on_source(self):
        assert Source('massgis').usage_requirement is None

    def test_loads_from_a_recipe_source_mapping(self):
        # Recipes reach Source as **source, so a YAML `usage_requirement:`
        # mapping only works if it is an accepted parameter that converts
        source = Source(
            **{
                'source_id': 'scag',
                'usage_requirement': {'non_commercial': True},
            }
        )
        assert isinstance(source.usage_requirement, UsageRequirement)
        assert source.usage_requirement.non_commercial is True
        assert not source.usage_requirement.is_empty()

    def test_no_conditions_is_empty(self):
        assert UsageRequirement().is_empty()

    def test_unknown_environment_flag_raises(self):
        with pytest.raises(ValueError, match='Unknown environment flag'):
            UsageRequirement(environment=['licenced'])

    def test_undeclared_commercial_does_not_satisfy_non_commercial(self):
        requirement = UsageRequirement(non_commercial=True)
        assert requirement.unmet({'commercial': None})
        assert requirement.unmet({'commercial': True})
        assert requirement.unmet({'commercial': False}) == []

    @pytest.mark.parametrize(
        'declared, expected_met',
        [
            ({'restricted': True, 'encrypted_at_rest': True}, True),
            ({'restricted': True, 'offline_only': True}, True),
            ({'restricted': True}, False),
            ({'encrypted_at_rest': True, 'offline_only': True}, False),
            ({}, False),
        ],
    )
    def test_environment_is_and_of_or_groups(self, declared, expected_met):
        # ZTRAX's shape: restricted AND (encrypted-at-rest OR offline-only)
        requirement = UsageRequirement(
            environment=['restricted', ['encrypted_at_rest', 'offline_only']]
        )
        unmet = requirement.unmet({'environment': declared})
        assert (unmet == []) is expected_met

    @pytest.mark.parametrize(
        'interest, admin_id, expected_met',
        [
            ('US-MA', 'US-MA', True),
            ('US-MA', 'US-MA-MI', True),  # recipe inside the interest
            ('US-MA-MI', 'US-MA', True),  # statewide file serves the county
            ('US-MA', 'US-NC', False),
            ('US-MA', None, False),
        ],
    )
    def test_admin_interest_matches_ancestors_and_descendants(
        self, interest, admin_id, expected_met
    ):
        requirement = UsageRequirement(admin_interest=True)
        profile = {'admin_interests': [interest]}
        unmet = requirement.unmet(profile, admin_id=admin_id)
        assert (unmet == []) is expected_met
