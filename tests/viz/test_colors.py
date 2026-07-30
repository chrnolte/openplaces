"""Tests for `resolve_category_colors` and the CSV-backed category palettes."""

import pandas as pd

from openplaces.viz.colors import (
    CATEGORY_COLORS,
    MISSING_LABEL,
    RESERVED_NEUTRAL_COLOR,
    resolve_category_colors,
)


def test_occupancy_type_has_no_stale_mobile_home_alias():
    assert 'Mobile Home' not in CATEGORY_COLORS['occupancy_type']


def test_multi_family_and_low_rise_are_distinguishable():
    palette = CATEGORY_COLORS['occupancy_type']
    assert palette['Multi-Family'] != palette['Low-Rise Multi-Family']


def test_resolve_category_colors_uses_curated_palette():
    values = pd.Series(['Single-Family', 'Manufactured Home', 'Single-Family'])
    resolved = resolve_category_colors(values, col_name='occupancy_type')
    curated = CATEGORY_COLORS['occupancy_type']
    assert resolved['Single-Family'] == curated['Single-Family']
    assert resolved['Manufactured Home'] == curated['Manufactured Home']


def test_resolve_category_colors_gives_uncurated_labels_distinct_colors():
    # 'Retail', 'Government Services', 'Professional Technical Services' are
    # not in _OCCUPANCY_TYPE -- real occupancy_type leak-through from the
    # curator's group_footprint_fema/group_parcel evidence vote. They must
    # not all collapse onto one shared fallback color.
    values = pd.Series(
        [
            'Single-Family',
            'Retail',
            'Government Services',
            'Professional Technical Services',
        ]
    )
    resolved = resolve_category_colors(values, col_name='occupancy_type')
    assert len(set(resolved.values())) == len(resolved)


def test_resolve_category_colors_is_deterministic_across_scopes():
    # Same uncurated label must resolve to the same color whether it's
    # queried alongside a small or large set of other labels (e.g. a single
    # county vs. a full state) -- the color is purely a function of the
    # label, not of what else happens to be present in a given call.
    color_alone = resolve_category_colors(
        pd.Series(['Retail']), col_name='occupancy_type'
    )
    color_with_others = resolve_category_colors(
        pd.Series(['Retail', 'Bank', 'Hospital', 'Single-Family']),
        col_name='occupancy_type',
    )
    assert color_alone['Retail'] == color_with_others['Retail']


def test_resolve_category_colors_missing_values():
    values = pd.Series(['Single-Family', None, None])
    resolved = resolve_category_colors(values, col_name='occupancy_type')
    assert resolved[MISSING_LABEL] == RESERVED_NEUTRAL_COLOR


def test_resolve_category_colors_no_missing_values_omits_the_label():
    values = pd.Series(['Single-Family', 'Manufactured Home'])
    resolved = resolve_category_colors(values, col_name='occupancy_type')
    assert MISSING_LABEL not in resolved


def test_resolve_category_colors_never_returns_none_for_unregistered_column():
    values = pd.Series(['some_value', 'another_value'])
    resolved = resolve_category_colors(values, col_name='not_a_registered_column')
    assert resolved
    assert len(set(resolved.values())) == len(resolved)
