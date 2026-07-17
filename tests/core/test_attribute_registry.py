"""Tests for the attribute registry: unique names and entity/stage filtering.

The registry is the authoritative list of canonical attribute names across every
entity type and pipeline stage. Names must stay unique (the loader indexes by
name), and `get_attributes` must scope by entity type while always including the
shared (blank entity_type) rows.
"""

from openplaces.core.attribute_registry import get_attributes, load_registry


def test_registry_names_unique():
    assert load_registry().index.is_unique


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
