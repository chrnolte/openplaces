"""Tests for the recipe-facing admin id assignment."""

import warnings

import pandas as pd
import pytest

from openplaces.io.admin_codes import assign_admin_ids
from openplaces.io.admin_codes import frame as frame_module
from openplaces.io.admin_codes.anchors import normalize_name
from openplaces.io.admin_codes.frame import _placeholder_codes
from openplaces.io.admin_codes.registry import load_registry


def frame(parents, names, **columns):
    return pd.DataFrame({'admin2_id': parents, 'name': names, **columns})


def spine_id(parent, name, level=3):
    """Return the id the committed spine records for one unit.

    Read rather than written down: a re-mint moves codes, and a literal
    frozen here goes stale silently. Albany was 'US-NY-LG' until the
    2026-08 re-mint and is 'US-NY-AL' now.
    """
    pins, _, _ = load_registry(level)
    code = pins.get((parent, normalize_name(name)))
    assert code is not None, f'the spine no longer names {name} under {parent}'
    return f'{parent}-{code}'


def assign(df, **kwargs):
    kwargs.setdefault('new_admin_id_col', 'admin3_id')
    kwargs.setdefault('parent_admin_id_col', 'admin2_id')
    # These cases exercise the derivation rules, so they opt out of the
    # spine registry: with pinning on, a real unit's code comes from the
    # committed spine and no rule runs at all. TestPinning covers that.
    kwargs.setdefault('pin_to_spine', False)
    return assign_admin_ids(df, **kwargs)


class TestIdentifiers:
    def test_builds_full_id_from_parent_and_code(self):
        out = assign(frame(['US-MA', 'US-MA'], ['Middlesex', 'Suffolk']))
        assert set(out.index) == {'US-MA-MI', 'US-MA-SU'}
        assert out.index.name == 'admin3_id'

    def test_records_the_rule_behind_each_code(self):
        # Pinning is off here, so this is the derivation rule firing, not
        # the spine's record. Assert the rule, not the code it produced.
        out = assign(frame(['US-NY'], ['Albany']))
        assert len(out) == 1
        assert out['admin3_id_source'].iloc[0] == 'name'

    def test_keeps_the_input_columns(self):
        out = assign(frame(['US-MA'], ['Middlesex'], fips=['25017']))
        assert out.loc['US-MA-MI', 'fips'] == '25017'

    def test_groups_are_independent_so_a_code_may_repeat(self):
        out = assign(frame(['US-MA', 'US-ME'], ['Middlesex', 'Middlesex']))
        assert set(out.index) == {'US-MA-MI', 'US-ME-MI'}

    def test_row_order_does_not_change_the_assignment(self):
        names = ['Berkshire', 'Bristol', 'Barnstable', 'Norfolk']
        first = assign(frame(['US-MA'] * 4, names))
        second = assign(frame(['US-MA'] * 4, names[::-1]))
        assert dict(zip(first['name'], first.index)) == dict(
            zip(second['name'], second.index)
        )


class TestAwkwardRows:
    def test_siblings_sharing_a_name_get_different_codes(self):
        out = assign(frame(['CO-AN'] * 2, ['Santo Domingo', 'Santo Domingo']))
        assert len(set(out.index)) == 2

    def test_nameless_rows_get_placeholder_codes(self):
        out = assign(frame(['US-MA', 'US-MA'], ['Middlesex', '']))
        assert out.loc['US-MA-MI', 'admin3_id_source'] == 'name'
        placeholders = out[out['admin3_id_source'] == 'placeholder']
        assert len(placeholders) == 1
        assert placeholders.index[0].startswith('US-MA-X')

    def test_missing_name_is_treated_as_nameless(self):
        out = assign(frame(['US-MA', 'US-MA'], ['Middlesex', None]))
        assert (out['admin3_id_source'] == 'placeholder').sum() == 1

    def test_placeholder_codes_avoid_taken_codes(self):
        codes = _placeholder_codes(2, 2, {'X0'})
        assert 'X0' not in codes
        assert len(set(codes)) == 2
        assert all(len(code) == 2 for code in codes)


class TestUniformWidth:
    def test_one_width_per_parent(self):
        # Enough same-initial names that two characters cannot
        # carry them.
        names = [f'Springfield {n}' for n in range(60)] + ['Boston']
        out = assign(frame(['US-MA'] * len(names), names))
        widths = {len(i.rsplit('-', 1)[1]) for i in out.index}
        assert len(widths) == 1

    def test_explicit_lengths_are_honored(self):
        out = assign(frame(['US-NY'], ['Albany']), lengths=(3,))
        assert out.index[0] == 'US-NY-ALB'


class TestWeighting:
    # Sanford and Sanborn both propose SA first, so only one can
    # have it.
    NAMES = ['Sanford', 'Sanborn']

    def contested(self, populations):
        df = frame(['US-XX'] * 2, self.NAMES, population=populations)
        out = assign(df, weight_col='population')
        return out.loc['US-XX-SA', 'name']

    def test_the_heavier_unit_keeps_the_code_from_its_name(self):
        assert self.contested([500000, 900]) == 'Sanford'

    def test_swapping_the_weights_swaps_the_winner(self):
        assert self.contested([900, 500000]) == 'Sanborn'

    def test_without_weights_the_outcome_is_still_deterministic(self):
        df = frame(['US-XX'] * 2, self.NAMES)
        assert list(assign(df).index) == list(assign(df).index)


class TestValidation:
    def test_missing_name_column_is_reported(self):
        df = pd.DataFrame({'admin2_id': ['US-MA']})
        with pytest.raises(ValueError, match="'name' not found"):
            assign(df)

    def test_missing_parent_column_is_reported(self):
        df = pd.DataFrame({'name': ['Middlesex']})
        with pytest.raises(ValueError, match="'admin2_id' not found"):
            assign(df)

    def test_blank_parent_is_rejected(self):
        with pytest.raises(ValueError, match='no admin2_id'):
            assign(frame(['US-MA', ''], ['Middlesex', 'Suffolk']))

    def test_ids_use_only_the_allowed_charset(self):
        out = assign(frame(['US-MA'] * 3, ['Ávila', "L'Aquila", 'Zürich']))
        assert out.index.str.fullmatch(r'[A-Z0-9-]+').all()


class TestPinning:
    """The spine is the registry of record; see admin_codes.registry."""

    def pinned(self, parents, names, **kwargs):
        return assign_admin_ids(
            frame(parents, names),
            new_admin_id_col='admin3_id',
            parent_admin_id_col='admin2_id',
            **kwargs,
        )

    # New York counties, not Massachusetts ones: New England's counties
    # left the hierarchy when its towns became level 3, and these tests
    # need units the spine actually names.
    def test_a_unit_the_spine_names_keeps_its_code(self):
        out = self.pinned(['US-NY'], ['Albany'])
        assert out.loc[spine_id('US-NY', 'Albany'), 'admin3_id_source'] == 'pinned'

    def test_pinning_overrides_an_explicit_length(self):
        # The point of a pin is that the recorded id wins, so a caller
        # asking for three characters still gets the two-character code
        # the spine already issued.
        recorded = spine_id('US-NY', 'Albany')
        assert len(recorded.rsplit('-', 1)[1]) == 2, 'need a 2-char pin to test'
        out = self.pinned(['US-NY'], ['Albany'], lengths=(3,))
        assert out.index[0] == recorded

    def test_a_new_sibling_does_not_move_existing_codes(self):
        names = ['Albany', 'Erie', 'Monroe']
        before = self.pinned(['US-NY'] * 3, names)
        after = self.pinned(['US-NY'] * 4, [*names, 'Albion Vale'])
        was = dict(zip(before['name'], before.index))
        now = dict(zip(after['name'], after.index))
        assert all(was[name] == now[name] for name in names)

    def test_an_unknown_unit_is_still_assigned(self):
        out = self.pinned(['US-MA'], ['Nonexistent Placeville'])
        assert out['admin3_id_source'].iloc[0] != 'pinned'
        assert out.index[0].startswith('US-MA-')


class TestSharedSourceCode:
    """A source code two siblings share cannot pin either of them."""

    @pytest.fixture
    def two_pinned_siblings(self, monkeypatch):
        # A fabricated parent whose two units the spine records under
        # distinct names and distinct source codes.
        pins = {
            ('XX-AA', normalize_name('Alpha')): 'AL',
            ('XX-AA', normalize_name('Bravo')): 'BR',
        }
        by_external = {('XX-AA', '101'): 'AL', ('XX-AA', '202'): 'BR'}
        monkeypatch.setattr(
            frame_module,
            'load_registry',
            lambda level, sep: (pins, by_external, {'XX-AA': ('AL', 'BR')}),
        )
        monkeypatch.setattr(frame_module, 'load_group_code_lengths', lambda: {})

    def assign_pinned(self, names, codes):
        return assign_admin_ids(
            frame(['XX-AA', 'XX-AA'], names, admin3_id_admin1=codes),
            new_admin_id_col='admin3_id',
            parent_admin_id_col='admin2_id',
        )

    def test_a_copied_code_falls_back_to_the_names(self, two_pinned_siblings):
        # The source gives Bravo the code it records for Alpha, as GADM
        # 4.1 does for two Philippine provinces. Pinning on the code
        # would put both rows on AL.
        out = self.assign_pinned(['Alpha', 'Bravo'], ['101', '101'])
        assert dict(zip(out['name'], out.index)) == {
            'Alpha': 'XX-AA-AL',
            'Bravo': 'XX-AA-BR',
        }
        assert (out['admin3_id_source'] == 'pinned').all()

    def test_a_unique_code_still_pins_a_renamed_unit(self, two_pinned_siblings):
        # Bravo arrives under a name the spine does not record, so only
        # its unambiguous source code can keep it on its recorded code.
        out = self.assign_pinned(['Alpha', 'Bravo Renamed'], ['101', '202'])
        assert dict(zip(out['name'], out.index)) == {
            'Alpha': 'XX-AA-AL',
            'Bravo Renamed': 'XX-AA-BR',
        }


class TestReviewedGroupLength:
    """A reviewed length may not split a group's width in two."""

    @pytest.fixture
    def pinned_two_reviewed_three(self, monkeypatch):
        # A fabricated parent whose one existing sibling was minted two
        # characters wide, with a reviewed row asking for three.
        pins = {('XX-AA', normalize_name('Alpha')): 'AL'}
        monkeypatch.setattr(
            frame_module,
            'load_registry',
            lambda level, sep: (pins, {}, {'XX-AA': ('AL',)}),
        )
        monkeypatch.setattr(
            frame_module, 'load_group_code_lengths', lambda: {'XX-AA': 3}
        )

    def test_the_pinned_width_wins(self, pinned_two_reviewed_three):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = assign_admin_ids(
                frame(['XX-AA', 'XX-AA'], ['Alpha', 'Bravo']),
                new_admin_id_col='admin3_id',
                parent_admin_id_col='admin2_id',
            )
        widths = {len(i.rsplit('-', 1)[1]) for i in out.index}
        assert widths == {2}, f'mixed-width parent: {sorted(out.index)}'

    def test_the_ignored_review_is_reported(self, pinned_two_reviewed_three):
        with pytest.warns(UserWarning, match='already 2 characters'):
            assign_admin_ids(
                frame(['XX-AA', 'XX-AA'], ['Alpha', 'Bravo']),
                new_admin_id_col='admin3_id',
                parent_admin_id_col='admin2_id',
            )

    def test_a_remint_adopts_the_reviewed_length(self, pinned_two_reviewed_three):
        # Nothing is pinned when the group is minted from scratch, so
        # the reviewed length applies to the whole group at once, which
        # is the way it is meant to take effect.
        out = assign_admin_ids(
            frame(['XX-AA', 'XX-AA'], ['Alpha', 'Bravo']),
            new_admin_id_col='admin3_id',
            parent_admin_id_col='admin2_id',
            pin_to_spine=False,
        )
        widths = {len(i.rsplit('-', 1)[1]) for i in out.index}
        assert widths == {3}
