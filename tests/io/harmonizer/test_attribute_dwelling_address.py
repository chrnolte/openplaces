"""Tests for `attribute_dwelling_address`.

Mirrors `tests/io/curator/test_summarize_footprint_morphology.py`'s
monkeypatch pattern for mocking `get_entities`. Covers: relaying only a
parcel's primary footprint's evidence (not secondary), breaking ties among
multiple primary footprints by largest overlap area, a parcel with no
footprints at all, and a footprint entity missing the evidence columns.
"""

from __future__ import annotations

import pandas as pd

import openplaces.io.harmonizer.attributes as attrs
import openplaces.io.readers as readers
from openplaces.io.harmonizer import HarmonizeState


def _parcel_spine(parcel_ids):
    spine = pd.DataFrame(index=pd.Index(parcel_ids, name='parcel_id'))
    return HarmonizeState(
        recipe={}, admin_id='US-NC-AR', verbose=False, timer=None, spine=spine
    )


def _footprints(rows):
    defaults = {
        'priority_on_parcel': 'primary',
        'area_intersection_m2_parcel': 1000.0,
        'address_street_dwelling_overture': None,
        'address_number_dwelling_overture': None,
        'city_dwelling_overture': None,
        'postal_code_dwelling_overture': None,
    }
    keys = set(defaults) | {k for r in rows for k in r}
    data = {k: [r.get(k, defaults.get(k)) for r in rows] for k in keys}
    return pd.DataFrame(data)


def test_relays_only_primary_footprint_evidence(monkeypatch):
    state = _parcel_spine(['A'])
    footprints = _footprints(
        [
            {
                'parcel_id': 'A',
                'priority_on_parcel': 'primary',
                'address_street_dwelling_overture': 'SAMPLE AVE',
                'city_dwelling_overture': 'North Billerica',
            },
            {
                'parcel_id': 'A',
                'priority_on_parcel': 'secondary',
                'address_street_dwelling_overture': 'WRONG ST',
                'city_dwelling_overture': 'Wrong City',
            },
        ]
    )
    monkeypatch.setattr(readers, 'get_entities', lambda *a, **k: footprints)

    state = attrs.attribute_dwelling_address(state, footprint_recipe_id='fp')

    assert state.spine.loc['A', 'address_street_dwelling_footprint'] == 'SAMPLE AVE'
    assert state.spine.loc['A', 'city_dwelling_footprint'] == 'North Billerica'


def test_multiple_primary_footprints_largest_overlap_wins(monkeypatch):
    state = _parcel_spine(['A'])
    footprints = _footprints(
        [
            {
                'parcel_id': 'A',
                'area_intersection_m2_parcel': 50.0,
                'address_street_dwelling_overture': 'SMALL BUILDING ST',
            },
            {
                'parcel_id': 'A',
                'area_intersection_m2_parcel': 500.0,
                'address_street_dwelling_overture': 'LARGE BUILDING ST',
            },
        ]
    )
    monkeypatch.setattr(readers, 'get_entities', lambda *a, **k: footprints)

    state = attrs.attribute_dwelling_address(state, footprint_recipe_id='fp')

    assert (
        state.spine.loc['A', 'address_street_dwelling_footprint'] == 'LARGE BUILDING ST'
    )


def test_parcel_with_no_footprints_stays_missing(monkeypatch):
    state = _parcel_spine(['A', 'B'])
    footprints = _footprints(
        [{'parcel_id': 'A', 'address_street_dwelling_overture': 'SAMPLE AVE'}]
    )
    monkeypatch.setattr(readers, 'get_entities', lambda *a, **k: footprints)

    state = attrs.attribute_dwelling_address(state, footprint_recipe_id='fp')

    assert pd.isna(state.spine.loc['B', 'address_street_dwelling_footprint'])


def test_missing_evidence_columns_is_noop(monkeypatch):
    state = _parcel_spine(['A'])
    footprints = pd.DataFrame({'parcel_id': ['A'], 'priority_on_parcel': ['primary']})
    monkeypatch.setattr(readers, 'get_entities', lambda *a, **k: footprints)

    state = attrs.attribute_dwelling_address(state, footprint_recipe_id='fp')

    assert 'address_street_dwelling_footprint' not in state.spine.columns


def test_missing_priority_column_is_noop(monkeypatch):
    state = _parcel_spine(['A'])
    footprints = pd.DataFrame(
        {'parcel_id': ['A'], 'address_street_dwelling_overture': ['SAMPLE AVE']}
    )
    monkeypatch.setattr(readers, 'get_entities', lambda *a, **k: footprints)

    state = attrs.attribute_dwelling_address(state, footprint_recipe_id='fp')

    assert 'address_street_dwelling_footprint' not in state.spine.columns


def test_spine_none_is_noop():
    state = HarmonizeState(
        recipe={}, admin_id='US-NC-AR', verbose=False, timer=None, spine=None
    )
    result = attrs.attribute_dwelling_address(state, footprint_recipe_id='fp')
    assert result.spine is None


def test_no_shared_on_column_is_noop(monkeypatch):
    # Footprint entity has no 'parcel_id' column at all.
    state = _parcel_spine(['A'])
    footprints = pd.DataFrame(
        {
            'priority_on_parcel': ['primary'],
            'address_street_dwelling_overture': ['SAMPLE AVE'],
        }
    )
    monkeypatch.setattr(readers, 'get_entities', lambda *a, **k: footprints)

    state = attrs.attribute_dwelling_address(state, footprint_recipe_id='fp')

    assert 'address_street_dwelling_footprint' not in state.spine.columns
