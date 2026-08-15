"""Tests for suffix-independent attribute-registry lookups.

Covers openplaces.recipe.resolve_attribute_name directly and its consumer
aggregate_rows, which must apply registry aggregation rules to
provenance-suffixed columns.
"""

import pandas as pd

from openplaces.io.aggregate import aggregate_rows
from openplaces.recipe import resolve_attribute_name


def test_exact_registry_row_wins():
    # Each name ends in a source-like token but has its own registry row,
    # so it must resolve to itself rather than being suffix-stripped.
    for name in [
        'n_footprints_per_parcel',
        'priority_on_parcel',
        'parcel_id_local',
        'occupancy_type_review',
        'max_parcels_per_footprint',
        'max_dwellings_per_footprint',
        'year_first_buildup_hisdac',
    ]:
        assert resolve_attribute_name(name) == name


def test_suffixed_evidence_falls_back_to_base():
    assert resolve_attribute_name('improvement_value_parcel') == 'improvement_value'
    assert resolve_attribute_name('n_dwellings_overture') == 'n_dwellings'
    # These two have no registry rows of their own; they resolve to the base
    # occupancy_type row.
    assert resolve_attribute_name('occupancy_type_parcel') == 'occupancy_type'
    assert resolve_attribute_name('occupancy_type_footprint_fema') == 'occupancy_type'


def test_unresolvable_name_returned_unchanged():
    assert (
        resolve_attribute_name('definitely_not_an_attribute')
        == 'definitely_not_an_attribute'
    )


def test_aggregate_rows_includes_suffixed_numeric_column():
    # improvement_value_parcel resolves to improvement_value (registry sum),
    # so the suffixed column is aggregated instead of silently dropped.
    df = pd.DataFrame(
        {
            'group_id': ['a', 'a', 'b'],
            'improvement_value_parcel': [1.0, 2.0, 4.0],
        }
    )
    out = aggregate_rows(df, by='group_id')
    assert out is not None
    assert out.loc['a', 'improvement_value_parcel'] == 3.0
    assert out.loc['b', 'improvement_value_parcel'] == 4.0


def test_aggregate_rows_keeps_unregistered_source_sidecar():
    # year_built_source has no registry row of its own (unlike
    # improvement_value_parcel above, it isn't a recognized recipe-derived
    # provenance suffix either) -- get_agg_func's generic _source fallback
    # is what keeps it from being silently dropped here.
    df = pd.DataFrame(
        {
            'group_id': ['a', 'a', 'b'],
            'year_built': [1964, 1980, 1990],
            'year_built_source': ['countyx', 'countyx', 'countyy'],
        }
    )
    out = aggregate_rows(df, by='group_id')
    assert out is not None
    assert 'year_built_source' in out.columns
    assert out.loc['a', 'year_built_source'] == 'countyx'
    assert out.loc['b', 'year_built_source'] == 'countyy'
