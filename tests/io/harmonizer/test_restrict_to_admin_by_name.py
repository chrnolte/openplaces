"""Restricting an over-broad source to one admin unit by its name.

Every name here is fabricated.
"""

import pandas as pd
import pytest

from openplaces.io import harmonizer as hz


class _Admin(str):
    """An admin id whose level is its part count, like AdminId."""

    def get_level(self):
        return len(self.split('-'))


@pytest.fixture
def _spine(monkeypatch):
    def fake_get_admin(admin_id, level):
        return pd.DataFrame({'name': ['St. Quillby']})

    monkeypatch.setattr(hz, 'get_admin', fake_get_admin)
    monkeypatch.setattr(
        hz, 'get_recipe_by_id', lambda rid: {'admin_id': _Admin('XX-YY')}
    )
    monkeypatch.setattr(hz, 'AdminId', _Admin)


def _frame(names):
    return pd.DataFrame({'admin3_name': names, 'v': range(len(names))})


def test_a_source_punctuating_the_name_differently_still_matches(_spine):
    # The source drops the period and spells Saint out; the spine does
    # neither. Every row belongs to the unit and must be kept.
    df = _frame(
        ['ST CROIX'.replace('CROIX', 'QUILLBY'), 'Saint Quillby', 'st. quillby']
    )
    kept = hz.restrict_to_admin_by_name(df, 'r', _Admin('XX-YY-ZZ'))
    assert len(kept) == 3


def test_another_units_rows_are_still_dropped(_spine):
    df = _frame(['St. Quillby', 'Marlowe', 'St. Quillby'])
    kept = hz.restrict_to_admin_by_name(df, 'r', _Admin('XX-YY-ZZ'))
    assert kept['v'].tolist() == [0, 2]


def test_a_type_word_is_not_folded_away(_spine):
    # 'Quillby city' is a different unit from 'St. Quillby'; keeping the
    # type word is what tells them apart (Maryland's Baltimore pair).
    df = _frame(['Quillby city', 'St. Quillby'])
    kept = hz.restrict_to_admin_by_name(df, 'r', _Admin('XX-YY-ZZ'))
    assert kept['v'].tolist() == [1]


def test_dropping_every_row_warns_instead_of_looking_like_no_coverage(_spine):
    df = _frame(['Marlowe', 'Marlowe'])
    with pytest.warns(UserWarning, match='no row of'):
        kept = hz.restrict_to_admin_by_name(df, 'r', _Admin('XX-YY-ZZ'))
    assert len(kept) == 0


def test_a_source_already_scoped_to_the_unit_is_left_alone(monkeypatch, _spine):
    monkeypatch.setattr(
        hz, 'get_recipe_by_id', lambda rid: {'admin_id': _Admin('XX-YY-ZZ')}
    )
    df = _frame(['Marlowe', 'anything'])
    assert len(hz.restrict_to_admin_by_name(df, 'r', _Admin('XX-YY-ZZ'))) == 2
