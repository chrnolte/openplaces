"""Tests for the identifier audit and the safe reference resolver."""

from collections import Counter

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


class TestResolveIdentifier:
    """Every expectation here is read from the spine, never written down.

    A re-mint moves codes, so a literal frozen in a test goes stale
    silently and then asserts the wrong unit. That has now happened
    twice: Hyde was `US-NC-HY` until the weighted re-mint, and Albany
    was `US-NY-LG` until `af3b5bf` widened North Carolina and taught the
    mint to prefer a name's first letter. Both times the fix was to
    assert on the *unit* a resolution lands on.
    """

    @staticmethod
    def _names(level=3):
        live = _read(level)
        superseded = _read(level, superseded=True)
        column = f'admin{level}_id'
        return (
            dict(zip(live[column], live['name'])),
            dict(zip(superseded[column], superseded['name'])),
        )

    def test_resolution_follows_the_unit_the_id_used_to_name(self):
        # There is deliberately no "this id looks live, keep it"
        # shortcut. Every id the superseded snapshot records must resolve
        # to whatever names that same unit today -- returning the
        # unchanged string would silently hand a caller a different
        # county.
        live_name, past_name = self._names()
        moved = [
            admin_id
            for admin_id, name in past_name.items()
            if name and live_name.get(admin_id) not in (None, name)
        ]
        assert moved, 'no id changed meaning; this no longer tests anything'

        for admin_id in moved[:200]:
            resolved = resolve_identifier(admin_id)
            if resolved is None:
                continue  # ambiguous by name; refusing is correct
            assert live_name[resolved] == past_name[admin_id], (
                f'{admin_id} used to name {past_name[admin_id]!r} but '
                f'resolved to {resolved} ({live_name[resolved]!r})'
            )

    def test_a_retired_identifier_resolves_by_name(self):
        # An id the spine no longer issues still points at its unit,
        # because the unit itself did not go anywhere.
        live_name, past_name = self._names()
        retired = [
            admin_id
            for admin_id, name in past_name.items()
            if name and admin_id not in live_name
        ]
        assert retired, 'the spine retired no identifiers'

        resolved = {a: resolve_identifier(a) for a in retired[:200]}
        landed = {a: r for a, r in resolved.items() if r is not None}
        assert landed, 'no retired identifier resolved at all'
        for admin_id, target in landed.items():
            assert live_name[target] == past_name[admin_id]

    def test_a_recycled_identifier_follows_the_unit_not_the_string(self):
        # An id that survived but changed meaning must land on the unit
        # it used to name, never on the one holding the string today.
        live_name, past_name = self._names()
        recycled = [
            admin_id
            for admin_id, name in past_name.items()
            if name and live_name.get(admin_id) not in (None, name)
        ]
        assert recycled, 'no identifier was recycled; nothing to test'

        admin_id = recycled[0]
        resolved = resolve_identifier(admin_id)
        assert resolved is not None
        assert live_name[resolved] == past_name[admin_id]
        assert resolved != admin_id, (
            f'{admin_id} resolved to itself, so this no longer tests recycling'
        )

    def test_an_unknown_identifier_returns_none(self):
        assert resolve_identifier('US-ZZ-QQ') is None

    def test_extra_name_knowledge_can_be_supplied(self):
        live_name, _ = self._names()
        assert resolve_identifier('XX-YY-ZZ') is None
        resolved = resolve_identifier('US-NY-QQ', past_names={'US-NY-QQ': 'Albany'})
        assert resolved is not None
        assert live_name[resolved] == 'Albany'

    def test_a_parent_that_cannot_resolve_widens_to_the_state(self):
        # When the parent is gone the scope widens to the state, which is
        # readable straight off the identifier. This is what carried
        # Connecticut's towns through their move from level 4 to level 3;
        # that vintage has since been regenerated out of the superseded
        # snapshot, so the case is built here instead of borrowed from it.
        live_name, _ = self._names()
        resolved = resolve_identifier(
            'US-CT-XX-QQ', past_names={'US-CT-XX-QQ': 'Bethel'}
        )
        assert resolved is not None, 'a dead parent should widen, not give up'
        assert resolved.startswith('US-CT-')
        assert live_name[resolved] == 'Bethel'

    def test_widening_still_requires_a_unique_match(self):
        # Widening loosens the parent, not the uniqueness rule: a name
        # borne by two units in one state must stay unresolved rather
        # than resolve to whichever one comes first.
        live = _read(3)
        state = live['admin3_id'].str.rsplit('-', n=1).str[0]
        named = live['name'].str.strip() != ''
        pairs = list(zip(state[named], live['name'][named].str.strip()))
        counts = Counter(pairs)
        ambiguous = next((pair for pair, n in counts.items() if n > 1), None)
        assert ambiguous is not None, (
            'no state repeats a level-3 name, so ambiguity cannot be tested'
        )

        scope, name = ambiguous
        probe = f'{scope}-XX-QQ'
        assert resolve_identifier(probe, past_names={probe: name}) is None


class TestAuditReportShape:
    def test_report_is_a_frame_of_counts(self):
        report = audit_spine(levels=(2,), reproduce=False)
        assert isinstance(report, pd.DataFrame)
        for column in ('units', 'bad_format', 'duplicate_ids'):
            assert pd.api.types.is_integer_dtype(report[column])
