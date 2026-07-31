"""Tests for `join_nonnull_addresses` (io/aggregate.py).

Guards the fix for a condo/apartment building's per-unit property records
getting `join_nonnull`-concatenated into one un-parseable multi-address
string (e.g. `'11 PLACEHOLDER AVE #2 + 11 PLACEHOLDER AVE #1'`), which used to leave
downstream address parsing unable to recover a clean `address_street` --
starving `impute_land_value`'s local street/city donor tier.
"""

from __future__ import annotations

from openplaces.io.aggregate import _agg_func_for, join_nonnull_addresses


def test_unit_suffixed_variants_of_the_same_address_collapse_to_one():
    values = ['11 PLACEHOLDER AVE #2', '11 PLACEHOLDER AVE #1']
    assert join_nonnull_addresses(values) == '11 PLACEHOLDER AVE #2'


def test_unit_suffixed_variants_collapse_regardless_of_order_or_spacing():
    values = ['11  PLACEHOLDER AVE  #1', '11 Placeholder Ave #2', '11 placeholder ave #3']
    # First-seen base wins; later unit-suffixed duplicates of the same base
    # are dropped rather than concatenated.
    assert join_nonnull_addresses(values) == '11  PLACEHOLDER AVE  #1'


def test_genuinely_different_addresses_still_join():
    values = ['12 MAIN ST', '45 OAK AVE']
    assert join_nonnull_addresses(values) == '12 MAIN ST + 45 OAK AVE'


def test_mixed_same_building_and_different_address():
    # Two units at the same building collapse to one; the genuinely
    # separate address still gets its own segment.
    values = ['11 PLACEHOLDER AVE #2', '11 PLACEHOLDER AVE #1', '45 OAK AVE']
    assert join_nonnull_addresses(values) == '11 PLACEHOLDER AVE #2 + 45 OAK AVE'


def test_all_null_returns_none():
    assert join_nonnull_addresses([None, None]) is None


def test_agg_func_for_routes_address_to_the_address_aware_joiner():
    assert _agg_func_for('address', 'join_nonnull') is join_nonnull_addresses


def test_agg_func_for_leaves_other_join_nonnull_columns_alone():
    # e.g. use_group: concatenation is safe there, no re-parsing downstream.
    func = _agg_func_for('use_group', 'join_nonnull')
    assert func is not join_nonnull_addresses
    assert func(['A', 'B']) == 'A + B'
