"""Tests for the attribute registry: unique names and entity/stage filtering.

The registry is the authoritative list of canonical attribute names across every
entity type and pipeline stage. Names must stay unique (the loader indexes by
name), and `get_attributes` must scope by entity type while always including the
shared (blank entity_type) rows.
"""

from openplaces.core.attribute_registry import (
    get_agg_func,
    get_attributes,
    get_data_type,
    get_null_placeholder,
    load_registry,
)


def test_registry_names_unique():
    assert load_registry().index.is_unique


def test_source_suffix_falls_back_to_base_without_registry_row():
    # year_built_source has no CSV row of its own -- it's a provenance
    # sidecar of the registered year_built attribute, and both accessors
    # must recognize that generically.
    assert get_agg_func('year_built_source') == 'first'
    assert get_data_type('year_built_source') == 'categorical'


def test_source_suffix_fallback_requires_registered_base():
    assert get_agg_func('totally_made_up_source') is None
    assert get_data_type('totally_made_up_source') is None


def test_get_null_placeholder():
    assert get_null_placeholder('year_built') == 0
    assert get_null_placeholder('address') is None


def test_get_attributes_parcel_includes_shared_excludes_other_entities():
    parcel = get_attributes(entity_type='parcel')
    # parcel-specific
    assert 'owner_name' in parcel.index
    assert 'last_sale_price' in parcel.index
    # shared (blank entity_type) are always included
    assert 'land_value' in parcel.index
    assert 'address' in parcel.index
    # transaction-only must not leak in
    assert 'grantor' not in parcel.index


def test_get_attributes_stage_filter():
    curate = get_attributes(stage='curate')
    assert 'occupancy_type' in curate.index
    # an ingest/general (blank stage) attribute is not returned for a stage query
    assert 'land_value' not in curate.index
