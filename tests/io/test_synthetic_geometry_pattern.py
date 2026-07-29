"""Tests for `synthetic_geometry_pattern` (core.schema).

The harmonizer's infer_spine_additions labels a synthetic fallback geometry
'{entity_type}.{source}' (e.g. 'parcel.spine'); real geometry sources are
bare labels ('obm', 'microsoft'). The pattern must match exactly the typed
prefixes, optionally excluding the spine's own entity type.
"""

import re

import pandas as pd

from openplaces.core.schema import is_synthetic_geometry, synthetic_geometry_pattern


def test_matches_reference_derived_labels():
    pattern = re.compile(synthetic_geometry_pattern())
    assert pattern.match('parcel.spine')
    assert pattern.match('building.nsi')


def test_does_not_match_real_geometry_sources():
    pattern = re.compile(synthetic_geometry_pattern())
    assert not pattern.match('obm')
    assert not pattern.match('microsoft')
    # Prefix must be a whole entity type followed by the dot separator.
    assert not pattern.match('parcelx.spine')


def test_exclude_omits_own_entity_type():
    pattern = re.compile(synthetic_geometry_pattern(exclude='parcel'))
    assert not pattern.match('parcel.spine')
    assert pattern.match('building.nsi')


def test_is_synthetic_geometry_matches_reference_derived_rows():
    df = pd.DataFrame({'geometry_source': ['parcel.spine', 'obm']})
    mask = is_synthetic_geometry(df)
    assert list(mask) == [True, False]


def test_is_synthetic_geometry_excludes_own_entity_type():
    df = pd.DataFrame({'geometry_source': ['parcel.spine', 'building.nsi']})
    mask = is_synthetic_geometry(df, entity={'entity_type': 'parcel'})
    assert list(mask) == [False, True]


def test_is_synthetic_geometry_false_when_column_absent():
    df = pd.DataFrame({'value': [1, 2]})
    mask = is_synthetic_geometry(df)
    assert list(mask) == [False, False]
