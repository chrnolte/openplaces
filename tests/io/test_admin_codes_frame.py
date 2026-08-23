"""Tests for the recipe-facing admin id assignment."""

import pandas as pd
import pytest

from openplaces.io.admin_codes import assign_admin_ids
from openplaces.io.admin_codes.frame import _placeholder_codes


def frame(parents, names, **columns):
    return pd.DataFrame({'admin2_id': parents, 'name': names, **columns})


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
        out = assign(frame(['US-NY'], ['Albany']))
        assert out.loc['US-NY-LG', 'admin3_id_source'] == 'name'

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
        assert out.loc['US-NY-LG', 'admin3_id_source'] == 'pinned'

    def test_pinning_overrides_an_explicit_length(self):
        # The point of a pin is that the recorded id wins, so a caller
        # asking for three characters still gets the two-character code
        # the spine already issued.
        out = self.pinned(['US-NY'], ['Albany'], lengths=(3,))
        assert out.index[0] == 'US-NY-LG'

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
