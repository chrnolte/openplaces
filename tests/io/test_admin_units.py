"""Tests for the flat unit registry.

The registry is a re-shaping of data that is already committed, so the tests
that matter are the ones proving it loses nothing: every level regenerates
identically from it, and every declared parent resolves. Only once that
holds is it safe for anything to prefer the flat table over the per-level
CSVs.
"""

import pandas as pd
import pytest

from openplaces.io.admin_units import (
    UNIT_COLUMNS,
    build_units,
    load_level,
    orphans,
    parent_of,
    to_level_view,
)

LEVELS = (1, 2, 3, 4)


@pytest.fixture(scope='module')
def units():
    return build_units()


class TestParentOf:
    @pytest.mark.parametrize(
        ('admin_id', 'expected'),
        [
            ('US', ''),
            ('US-MA', 'US'),
            ('US-MA-MI', 'US-MA'),
            # Parentage is pure string work, so a four-deep id still
            # exercises it even though New England has no counties now.
            ('DE-BY-MU-AA', 'DE-BY-MU'),
        ],
    )
    def test_parent_is_the_id_minus_its_last_segment(self, admin_id, expected):
        assert parent_of(admin_id) == expected

    def test_works_at_any_depth(self):
        # The property the migration depends on: parentage does not need to
        # know how deep the branch is, so a shallower branch still resolves.
        assert parent_of('US-CT-HA') == 'US-CT'
        assert parent_of('US-NY-WE-YO') == 'US-NY-WE'


class TestBuildUnits:
    def test_has_the_declared_columns(self, units):
        assert list(units.columns) == UNIT_COLUMNS

    def test_covers_every_level(self, units):
        assert set(units['depth']) == set(LEVELS)

    def test_row_count_matches_the_per_level_csvs(self, units):
        expected = 0
        for level in LEVELS:
            rows = load_level(level)
            expected += int((rows[f'admin{level}_id'].str.strip() != '').sum())
        assert len(units) == expected

    def test_identifiers_are_unique(self, units):
        assert units['admin_id'].is_unique

    def test_every_unit_is_on_the_path_for_now(self, units):
        # Level omission has not happened yet. Writing False anywhere here
        # would assert something the committed data does not support.
        assert units['in_path'].all()


class TestHierarchyIsClosed:
    def test_no_orphaned_units(self, units):
        # A unit hanging off an unregistered parent is what silently drops
        # rows in a join, so this is a defect rather than a warning.
        missing = orphans(units)
        assert missing.empty, (
            f'{len(missing)} unit(s) name a parent that is not registered: '
            f'{sorted(missing["admin_id"].head(5))}'
        )

    def test_depth_agrees_with_the_parent_chain(self, units):
        by_id = dict(zip(units['admin_id'], units['depth']))
        parented = units[units['parent_admin_id'].str.strip() != '']
        mismatched = [
            row.admin_id
            for row in parented.itertuples(index=False)
            if by_id.get(row.parent_admin_id) != row.depth - 1
        ]
        assert not mismatched, f'depth disagrees with parent for {mismatched[:5]}'


class TestRoundTrip:
    @pytest.mark.parametrize('level', LEVELS)
    def test_level_view_regenerates_identically(self, units, level):
        source = load_level(level)
        id_column = f'admin{level}_id'
        source = source[source[id_column].str.strip() != '']
        expected = (
            pd.DataFrame(
                {
                    id_column: source[id_column],
                    'name': source['name'] if 'name' in source else '',
                    'type': source['type'] if 'type' in source else '',
                }
            )
            .sort_values(id_column)
            .reset_index(drop=True)
        )
        assert to_level_view(units, level).equals(expected)
