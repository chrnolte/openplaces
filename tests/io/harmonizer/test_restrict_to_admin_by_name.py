"""Tests for `restrict_to_admin_by_name`, the plain-text admin-name fallback.

`get_entities` cannot restrict a source to a finer admin unit than it was
saved at when the source has no matching admin-id column of its own (e.g. a
statewide transaction table with only a free-text county name column) -- it
returns the whole unfiltered table instead. This would otherwise duplicate
the whole table into every child admin unit's output, or pool every admin
unit's counts/aggregates together (the WI widor transaction table: no
`admin3_id`, only a `admin3_name` "County" column, real bug found by
re-ingesting Dane county and seeing the full 2.3M-row statewide table come
back instead of ~156k rows).
"""

import pandas as pd

import openplaces.io.harmonizer as harmonizer_module
from openplaces.core.schema import AdminId


def test_filters_by_name_when_source_is_coarser(monkeypatch):
    df = pd.DataFrame({'admin3_name': ['Dane', 'Oconto', 'Dane'], 'price': [1, 2, 3]})
    monkeypatch.setattr(
        harmonizer_module,
        'get_recipe_by_id',
        lambda recipe_id: {'admin_id': AdminId('US-WI')},  # state-level source
    )
    monkeypatch.setattr(
        harmonizer_module,
        'get_admin',
        lambda admin_id, level: pd.DataFrame({'name': ['Dane']}),
    )

    result = harmonizer_module.restrict_to_admin_by_name(
        df, 'US-WI_transaction-widor-2026', AdminId('US-WI-DA')
    )

    assert result['admin3_name'].tolist() == ['Dane', 'Dane']


def test_no_op_when_source_already_scoped_at_or_finer(monkeypatch):
    df = pd.DataFrame({'admin3_name': ['New Hanover'], 'price': [1]})
    monkeypatch.setattr(
        harmonizer_module,
        'get_recipe_by_id',
        lambda recipe_id: {'admin_id': AdminId('US-NC-NE')},  # already county-scoped
    )

    result = harmonizer_module.restrict_to_admin_by_name(
        df, 'US-NC-NE_transaction-nhcgov-2026', AdminId('US-NC-NE')
    )

    assert len(result) == 1


def test_no_op_when_name_column_absent():
    df = pd.DataFrame({'price': [1, 2]})

    result = harmonizer_module.restrict_to_admin_by_name(
        df, 'US-WI_transaction-widor-2026', AdminId('US-WI-DA')
    )

    assert len(result) == 2


def test_case_and_whitespace_insensitive_match(monkeypatch):
    df = pd.DataFrame({'admin3_name': [' DANE ', 'Oconto']})
    monkeypatch.setattr(
        harmonizer_module,
        'get_recipe_by_id',
        lambda recipe_id: {'admin_id': AdminId('US-WI')},
    )
    monkeypatch.setattr(
        harmonizer_module,
        'get_admin',
        lambda admin_id, level: pd.DataFrame({'name': ['Dane']}),
    )

    result = harmonizer_module.restrict_to_admin_by_name(
        df, 'US-WI_transaction-widor-2026', AdminId('US-WI-DA')
    )

    assert len(result) == 1
