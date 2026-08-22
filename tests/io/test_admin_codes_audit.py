"""Tests for the identifier audit and the safe reference resolver."""

import pandas as pd

from openplaces.io.admin_codes.audit import _read, audit_spine, resolve_identifier


class TestAuditSpine:
    def test_the_committed_spine_satisfies_every_invariant(self):
        report = audit_spine(reproduce=False)
        assert (report['bad_format'] == 0).all()
        assert (report['orphan_parents'] == 0).all()
        assert (report['duplicate_ids'] == 0).all()
        assert (report['mixed_width_parents'] == 0).all()

    def test_reports_one_row_per_level(self):
        report = audit_spine(levels=(2, 3), reproduce=False)
        assert list(report.index) == [2, 3]
        assert (report['units'] > 0).all()

    def test_level_four_reproduces_exactly(self):
        # Pinning is what makes identifiers durable, so this is the check
        # that must not regress: minting again returns the same spine.
        report = audit_spine(levels=(4,), reproduce=True)
        assert report.loc[4, 'reproduced'] == report.loc[4, 'units']


class TestResolveIdentifier:
    def test_resolution_follows_the_unit_the_id_used_to_name(self):
        # There is deliberately no "this id looks live, keep it" shortcut.
        # US-NY-AL named Allegany before the rebuild and names Albany
        # after it, so resolving it must return *Allegany's* new id --
        # returning the unchanged string would silently hand a caller the
        # wrong county.
        assert resolve_identifier('US-NY-AL') == 'US-NY-LG'

    def test_a_retired_identifier_resolves_by_name(self):
        # US-NC-AA was Alamance before the rebuild.
        assert resolve_identifier('US-NC-AA') == 'US-NC-AL'

    def test_a_recycled_identifier_follows_the_unit_not_the_string(self):
        # US-NC-HE named Hyde before the rebuild and names a different
        # county now. Resolving it must land on Hyde, never on the unit
        # that happens to hold the string today.
        #
        # Asserted against the *unit* rather than a fixed code: a
        # re-mint moves codes, and an expected value frozen here would
        # go stale silently. It already did once -- Hyde was US-NC-HY
        # until the weighted re-mint, and US-NC-HY is now Haywood, so
        # the old literal would have asserted the wrong county.
        live = _read(3)
        name_by_id = dict(zip(live['admin3_id'], live['name']))

        resolved = resolve_identifier('US-NC-HE')
        assert name_by_id[resolved] == 'Hyde'
        assert name_by_id['US-NC-HE'] != 'Hyde', (
            'US-NC-HE still names Hyde, so this no longer tests recycling'
        )

    def test_an_unknown_identifier_returns_none(self):
        assert resolve_identifier('US-ZZ-QQ') is None

    def test_extra_name_knowledge_can_be_supplied(self):
        assert resolve_identifier('XX-YY-ZZ') is None
        assert (
            resolve_identifier('US-NY-QQ', past_names={'US-NY-QQ': 'Albany'})
            == 'US-NY-AL'
        )

    def test_a_unit_whose_level_disappeared_still_resolves(self):
        # Connecticut's towns moved from level 4 to level 3 when the
        # county level was removed. The old id names a county that no
        # longer exists, so the parent cannot be resolved and the scope
        # has to widen to the state -- the case that used to return None.
        resolved = resolve_identifier('US-CT-FA-BE')
        assert resolved is not None
        assert resolved.startswith('US-CT-')
        assert resolved.count('-') == 2


class TestAuditReportShape:
    def test_report_is_a_frame_of_counts(self):
        report = audit_spine(levels=(2,), reproduce=False)
        assert isinstance(report, pd.DataFrame)
        for column in ('units', 'bad_format', 'duplicate_ids'):
            assert pd.api.types.is_integer_dtype(report[column])
