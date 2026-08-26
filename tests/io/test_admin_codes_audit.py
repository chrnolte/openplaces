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


def _name_by_id(level=3):
    """Map every live admin id at *level* to the unit it names."""
    live = _read(level)
    column = f'admin{level}_id'
    return dict(zip(live[column], live['name']))


def _resolves_to(identifier, name, **kwargs):
    """Assert that *identifier* resolves to the unit called *name*.

    Every assertion in this class is written this way on purpose. What
    the resolver promises is about *units*, not about strings, and an
    expected code frozen into a test goes stale on the next re-mint
    without saying so: these tests pinned `US-NY-LG` and `US-NC-LG` until
    a re-mint moved both, at which point they asserted the wrong county.
    """
    resolved = resolve_identifier(identifier, **kwargs)
    assert resolved is not None, f'{identifier} resolved to nothing'
    assert _name_by_id().get(resolved) == name, (
        f'{identifier} resolved to {resolved}, which names '
        f'{_name_by_id().get(resolved)!r}, not {name!r}'
    )


class TestResolveIdentifier:
    def test_resolution_follows_the_unit_the_id_used_to_name(self):
        # There is deliberately no "this id looks live, keep it"
        # shortcut. US-NY-LG named Allegany before the rebuild and names
        # a different county after it, so resolving it must land on
        # *Allegany* -- returning the unchanged string would silently
        # hand a caller the wrong county.
        _resolves_to('US-NY-LG', 'Allegany')

    def test_a_retired_identifier_resolves_by_name(self):
        # US-NC-AL was Alamance before the rebuild.
        _resolves_to('US-NC-AL', 'Alamance')

    def test_a_recycled_identifier_follows_the_unit_not_the_string(self):
        # US-NC-HD named Hyde before the rebuild. Resolving it must land
        # on Hyde, never on whatever unit happens to hold the string
        # today.
        name_by_id = _name_by_id()
        _resolves_to('US-NC-HD', 'Hyde')
        assert name_by_id.get('US-NC-HD') != 'Hyde', (
            'US-NC-HD still names Hyde, so this no longer tests recycling'
        )

    def test_an_unknown_identifier_returns_none(self):
        assert resolve_identifier('US-ZZ-QQ') is None

    def test_extra_name_knowledge_can_be_supplied(self):
        assert resolve_identifier('XX-YY-ZZ') is None
        _resolves_to('US-NY-QQ', 'Albany', past_names={'US-NY-QQ': 'Albany'})

    def test_a_unit_whose_level_disappeared_still_resolves(self):
        # Connecticut's towns moved from level 4 to level 3 when the
        # county level was removed. The old id names a county that no
        # longer exists, so the parent cannot be resolved and the scope
        # has to widen to the state -- the case that used to return None.
        resolved = resolve_identifier('US-CT-BET')
        assert resolved is not None
        assert resolved.startswith('US-CT-')
        assert resolved.count('-') == 2


class TestAuditReportShape:
    def test_report_is_a_frame_of_counts(self):
        report = audit_spine(levels=(2,), reproduce=False)
        assert isinstance(report, pd.DataFrame)
        for column in ('units', 'bad_format', 'duplicate_ids'):
            assert pd.api.types.is_integer_dtype(report[column])
