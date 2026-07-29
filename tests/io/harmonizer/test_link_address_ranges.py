"""Tests for the harmonizer `link_address_ranges` step.

A multi-unit deed sale's address number (e.g. '704-706', a duplex) never
matches a parcel's own single-number `address_id_local` key via the ordinary
`link_by_id` steps. This step splits the range and tries each individual
number instead, linking only when exactly one distinct parcel matches; an
ambiguous range (its two numbers resolving to two different parcels) is left
unmatched with a warning rather than guessed at.
"""

import warnings

import pandas as pd

import openplaces.io.harmonizer.links as links
from openplaces.io.harmonizer import HarmonizeState


def _state(spine, admin_id='US-MA-MI-SO'):
    return HarmonizeState(
        recipe={}, admin_id=admin_id, verbose=False, timer=None, spine=spine
    )


def _spine(numbers, streets=None):
    n = len(numbers)
    return pd.DataFrame(
        {
            'address_street': streets or ['Broadway'] * n,
            'address_number': numbers,
            'admin4_id': ['US-MA-MI-SO'] * n,
        }
    )


def test_single_match_links_the_range(monkeypatch):
    spine = _spine(['704-706'])
    ref = pd.DataFrame(
        {
            'address_street': ['Broadway'],
            'address_number': ['704'],
            'admin4_id': ['US-MA-MI-SO'],
            'use_group': ['Residential'],
        }
    )
    monkeypatch.setattr(links, 'get_entities', lambda *a, **k: ref)

    state = links.link_address_ranges(
        _state(spine), recipe_id='ref_recipe', columns=['use_group']
    )

    assert state.spine['use_group'].iloc[0] == 'Residential'


def test_ambiguous_match_left_unmatched_and_warns(monkeypatch):
    spine = _spine(['20-22'], streets=['Main St'])
    ref = pd.DataFrame(
        {
            'address_street': ['Main St', 'Main St'],
            'address_number': ['20', '22'],
            'admin4_id': ['US-MA-MI-SO', 'US-MA-MI-SO'],
            'use_group': ['Residential', 'Commercial'],
        }
    )
    monkeypatch.setattr(links, 'get_entities', lambda *a, **k: ref)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        state = links.link_address_ranges(
            _state(spine), recipe_id='ref_recipe', columns=['use_group']
        )

    assert pd.isna(state.spine['use_group'].iloc[0])
    assert len(caught) == 1
    assert 'more than one distinct parcel' in str(caught[0].message)


def test_zero_match_is_silent(monkeypatch):
    spine = _spine(['10-12'], streets=['Elm St'])
    ref = pd.DataFrame(
        {
            'address_street': ['Broadway'],
            'address_number': ['704'],
            'admin4_id': ['US-MA-MI-SO'],
            'use_group': ['Residential'],
        }
    )
    monkeypatch.setattr(links, 'get_entities', lambda *a, **k: ref)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        state = links.link_address_ranges(
            _state(spine), recipe_id='ref_recipe', columns=['use_group']
        )

    assert pd.isna(state.spine['use_group'].iloc[0])
    assert len(caught) == 0


def test_does_not_overwrite_an_already_linked_row(monkeypatch):
    # Mirrors _write_prioritized's real gap-fill/majority-coverage rule (the
    # same one link_by_id's own passes rely on): with this step's own match
    # coverage kept a minority of the spine (realistic -- most range rows
    # won't resolve), an already-filled row (from an earlier link_by_id
    # pass) is preserved rather than overwritten, and only genuinely empty
    # rows get filled in.
    spine = _spine(['704-706', '20-22', '30-32'])
    spine['use_group'] = ['Commercial', None, None]  # row 0 already filled
    ref = pd.DataFrame(
        {
            'address_street': ['Broadway'],
            'address_number': ['704'],
            'admin4_id': ['US-MA-MI-SO'],
            'use_group': ['Residential'],
        }
    )
    monkeypatch.setattr(links, 'get_entities', lambda *a, **k: ref)

    state = links.link_address_ranges(
        _state(spine), recipe_id='ref_recipe', columns=['use_group']
    )

    assert state.spine['use_group'].iloc[0] == 'Commercial'


def test_plain_unmatched_number_still_gets_a_direct_shot(monkeypatch):
    # Not range-shaped, and link_by_id's own passes are what normally
    # resolve a plain number -- but if this step is reached with the row
    # still unmatched (e.g. a canonicalization gap upstream), it isn't
    # skipped just because it's not a range: the same key lookup covers a
    # plain 1:1 match too.
    spine = _spine(['704'])
    ref = pd.DataFrame(
        {
            'address_street': ['Broadway'],
            'address_number': ['704'],
            'admin4_id': ['US-MA-MI-SO'],
            'use_group': ['Residential'],
        }
    )
    monkeypatch.setattr(links, 'get_entities', lambda *a, **k: ref)

    state = links.link_address_ranges(
        _state(spine), recipe_id='ref_recipe', columns=['use_group']
    )

    assert state.spine['use_group'].iloc[0] == 'Residential'


def test_already_matched_plain_row_is_not_reconsidered(monkeypatch):
    spine = _spine(['704'])
    spine['use_group'] = ['Commercial']  # already filled by an earlier pass
    ref = pd.DataFrame(
        {
            'address_street': ['Broadway'],
            'address_number': ['704'],
            'admin4_id': ['US-MA-MI-SO'],
            'use_group': ['Residential'],
        }
    )
    monkeypatch.setattr(links, 'get_entities', lambda *a, **k: ref)

    state = links.link_address_ranges(
        _state(spine), recipe_id='ref_recipe', columns=['use_group']
    )

    assert state.spine['use_group'].iloc[0] == 'Commercial'


def test_plain_number_matches_a_reference_range_half(monkeypatch):
    # The mirror case: the *reference* (parcel) side is recorded as a range
    # while the spine (transaction) lists a single plain number that's one
    # of the two halves.
    spine = _spine(['20'], streets=['Main St'])
    ref = pd.DataFrame(
        {
            'address_street': ['Main St'],
            'address_number': ['20-22'],
            'admin4_id': ['US-MA-MI-SO'],
            'use_group': ['Residential'],
        }
    )
    monkeypatch.setattr(links, 'get_entities', lambda *a, **k: ref)

    state = links.link_address_ranges(
        _state(spine), recipe_id='ref_recipe', columns=['use_group']
    )

    assert state.spine['use_group'].iloc[0] == 'Residential'


def test_plain_number_ambiguous_between_range_half_and_distinct_parcel(monkeypatch):
    # '22 Main St' is claimed by both a real standalone parcel at '22' and
    # a range parcel '20-22' -- genuinely ambiguous, must not guess.
    spine = _spine(['22'], streets=['Main St'])
    ref = pd.DataFrame(
        {
            'address_street': ['Main St', 'Main St'],
            'address_number': ['20-22', '22'],
            'admin4_id': ['US-MA-MI-SO', 'US-MA-MI-SO'],
            'use_group': ['Residential', 'Commercial'],
        }
    )
    monkeypatch.setattr(links, 'get_entities', lambda *a, **k: ref)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        state = links.link_address_ranges(
            _state(spine), recipe_id='ref_recipe', columns=['use_group']
        )

    assert pd.isna(state.spine['use_group'].iloc[0])
    assert len(caught) == 1
    assert 'more than one distinct parcel' in str(caught[0].message)


def test_missing_number_column_is_a_no_op():
    spine = pd.DataFrame({'other': [1, 2]})
    state = links.link_address_ranges(
        _state(spine), recipe_id='ref_recipe', columns=['use_group']
    )
    assert 'use_group' not in state.spine.columns
