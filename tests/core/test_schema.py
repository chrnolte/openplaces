import pytest

from openplaces.core.constants import ESCAPE_DIR
from openplaces.core.schema import (
    AdminId,
    DataSet,
    Entity,
    Source,
    UsageRequirement,
    admin_scope_covers,
    cast_dataset_or_entity,
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

    def test_a_no_resale_clause_is_its_own_fact(self):
        # "Not to be resold" forbids selling, not sharing: recorded apart
        # from redistribution_restricted, and unchecked by default.
        assert Source('example').resale_restricted is None
        source = Source(
            **{
                'source_id': 'examplepa',
                'license': 'free download; not to be resold without consent',
                'redistribution_restricted': False,
                'resale_restricted': True,
            }
        )
        assert source.resale_restricted is True
        assert source.redistribution_restricted is False


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


class TestAdminIdConstruction:
    """The single-argument branch, which used to assume a string."""

    def test_global_id_round_trips(self):
        assert AdminId(str(AdminId())) == AdminId()
        assert AdminId('').levels == ()

    def test_an_admin_id_can_be_rewrapped(self):
        assert AdminId(AdminId('US', 'MA')) == AdminId('US', 'MA')
        assert AdminId(AdminId()) == AdminId()

    def test_a_sequence_is_accepted_as_the_docstring_promises(self):
        assert AdminId(['US', 'MA']) == AdminId('US', 'MA')
        assert AdminId(('US', 'MA', 'MI')) == AdminId('US', 'MA', 'MI')

    def test_a_non_string_level_still_raises_value_error(self):
        # Not TypeError, and not a message about the type of `levels`:
        # two parsers use AdminId raising ValueError as their
        # discriminator between an admin prefix and another token.
        with pytest.raises(ValueError, match='invalid at level'):
            AdminId(7)


class TestEntityRoundTrip:
    """An entity id must parse back into the entity that produced it."""

    def test_a_separator_in_the_version_does_not_leak_into_the_id(self):
        entity = Entity('transaction', 'retr', '2024-01')
        assert Entity(str(entity)) == entity
        assert str(entity).count('-') == 2

    def test_a_separator_in_the_source_does_not_leak_into_the_id(self):
        entity = Entity('parcel', 'two-words', '2025')
        assert Entity(str(entity)) == entity

    def test_a_compact_string_cannot_be_combined_with_explicit_parts(self):
        # Silently discarding them produced an entity, and a path, for a
        # source nobody asked for.
        with pytest.raises(ValueError, match='already carries'):
            Entity('parcel-massgis-2025', source='other')
        with pytest.raises(ValueError, match='already carries'):
            Entity('parcel-massgis-2025', version='2030')

    def test_too_many_parts_are_refused_by_name(self):
        with pytest.raises(ValueError, match='at most three'):
            Entity('parcel-massgis-2025-extra')


class TestCastDatasetOrEntity:
    """Dispatch on the first token's vocabulary, not on exceptions."""

    def test_a_two_token_theme_is_a_dataset(self):
        # This is path.py's own documented example, and it used to raise
        # IndexError out of Theme, which no caller handles.
        dataset = cast_dataset_or_entity('bio-species')
        assert isinstance(dataset, DataSet)
        assert str(dataset.theme) == 'bio-species'

    def test_a_two_token_entity_is_an_entity(self):
        entity = cast_dataset_or_entity('parcel-massgis')
        assert isinstance(entity, Entity)
        assert str(entity.entity_type) == 'parcel'

    def test_a_full_dataset_id_is_a_dataset(self):
        dataset = cast_dataset_or_entity('land-elevation-usgs-3dep')
        assert isinstance(dataset, DataSet)
        assert dataset.version == '3dep'

    def test_an_unknown_first_token_raises_value_error(self):
        with pytest.raises(ValueError, match='neither a registered'):
            cast_dataset_or_entity('admin1-something')


class TestDataSetVersion:
    """A dataset's version is recorded, never invented."""

    def test_a_missing_version_stays_missing(self):
        # It used to default to today's date, so a bare theme token
        # parsed as a live dataset whose output path moved every day.
        dataset = DataSet('bio-species')
        assert dataset.version is None
        assert str(dataset) == 'bio-species'
        assert ESCAPE_DIR in dataset.to_path().parts

    def test_a_float_version_is_sanitized_like_a_recipe_id(self):
        # A YAML `version: 4.1` gave a '4.1' directory beside the '4~1'
        # the recipe id spells.
        assert DataSet('land', 'example', 4.1).version == '4~1'
        assert '4~1' in DataSet('land', 'example', 4.1).to_path().parts

    def test_a_full_compact_string_still_splits(self):
        dataset = DataSet('land-elevation-usgs-3dep')
        assert str(dataset.theme) == 'land-elevation'
        assert str(dataset.source) == 'usgs'
        assert dataset.version == '3dep'


class TestUsageRequirementProfileShapes:
    """`unmet` is documented as never raising, on any config shape."""

    def test_a_list_environment_declaration_is_read_not_crashed_on(self):
        # The natural mirror of the recipe-side list syntax. It used to
        # raise AttributeError, aborting the ingest gate instead of
        # prompting.
        requirement = UsageRequirement(environment=['restricted'])
        assert requirement.unmet({'environment': ['restricted']}) == []
        assert requirement.unmet({'environment': ['licensed']})

    def test_a_string_environment_declaration_is_read(self):
        requirement = UsageRequirement(environment=['restricted'])
        assert requirement.unmet({'environment': 'restricted'}) == []

    def test_an_unusable_profile_leaves_every_condition_unmet(self):
        requirement = UsageRequirement(non_commercial=True, environment=['licensed'])
        assert len(requirement.unmet('not a profile')) == 2
        assert len(requirement.unmet({'environment': 3})) == 2
