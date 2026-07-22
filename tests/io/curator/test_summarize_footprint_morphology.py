"""Tests for `summarize_footprint_morphology`.

Covers: joining by the globally-unique `parcel_id` (the parcel spine's own true
id lives on its index at this point in the pipeline, not a plain column -- see
`resolve_spine`), the `min_overlap_m2` sliver floor, the `priority_on_parcel`
exclusion feeding `n_primary_footprints_per_parcel`, and the synthetic-fallback
bypass of both.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

import openplaces.io.harmonizer.attributes as attrs
import openplaces.io.readers as readers
from openplaces.io.harmonizer import HarmonizeState


def _parcel_spine(parcel_ids):
    spine = gpd.GeoDataFrame(
        index=pd.Index(parcel_ids, name='parcel_id'),
        geometry=[box(i, 0, i + 1, 1) for i in range(len(parcel_ids))],
        crs='EPSG:4326',
    )
    return HarmonizeState(
        recipe={}, admin_id='US-NC-CE', verbose=False, timer=None, spine=spine
    )


def _footprints(rows):
    """Build a footprint GeoDataFrame from a list of dicts (defaults filled in)."""
    defaults = {
        'geometry_source': 'obm',
        'area_intersection_m2_parcel': 1000.0,
        'priority_on_parcel': 'primary',
        'n_dwellings_overture': 0,
        'n_parcels_per_footprint': 1,
    }
    keys = set(defaults) | {k for r in rows for k in r if k != 'geometry'}
    data = {k: [r.get(k, defaults.get(k)) for r in rows] for k in keys}
    return gpd.GeoDataFrame(
        data,
        geometry=[r['geometry'] for r in rows],
        crs='EPSG:32617',  # metric CRS so areas are physically meaningful
    )


def test_joins_by_parcel_id_not_parcel_id_local(monkeypatch):
    # Two genuinely distinct parcels ('A', 'B') share a non-unique
    # parcel_id_local ('DUP') -- exactly the condo-PIN-collision bug. Each has
    # its own single footprint. Joining by the default 'parcel_id' must keep
    # their counts separate (1 each), not merged (2 each).
    state = _parcel_spine(['A', 'B'])
    footprints = _footprints(
        [
            {'parcel_id': 'A', 'parcel_id_local': 'DUP', 'geometry': box(0, 0, 10, 10)},
            {'parcel_id': 'B', 'parcel_id_local': 'DUP', 'geometry': box(0, 0, 10, 10)},
        ]
    )
    monkeypatch.setattr(readers, 'get_entities', lambda *a, **k: footprints)

    state = attrs.summarize_footprint_morphology(state, footprint_recipe_id='fp')

    assert state.spine.loc['A', 'n_footprints_per_parcel'] == 1
    assert state.spine.loc['B', 'n_footprints_per_parcel'] == 1


def test_sliver_below_floor_excluded_from_all_counts(monkeypatch):
    state = _parcel_spine(['A'])
    footprints = _footprints(
        [
            {
                'parcel_id': 'A',
                'area_intersection_m2_parcel': 5.0,
                'geometry': box(0, 0, 10, 10),
            },
        ]
    )
    monkeypatch.setattr(readers, 'get_entities', lambda *a, **k: footprints)

    state = attrs.summarize_footprint_morphology(
        state, footprint_recipe_id='fp', min_overlap_m2=10.0
    )

    assert state.spine.loc['A', 'n_footprints_per_parcel'] == 0
    assert state.spine.loc['A', 'n_small_elongated_footprints_per_parcel'] == 0
    assert state.spine.loc['A', 'n_primary_footprints_per_parcel'] == 0


def test_overlap_at_or_above_floor_counted(monkeypatch):
    state = _parcel_spine(['A'])
    footprints = _footprints(
        [
            {
                'parcel_id': 'A',
                'area_intersection_m2_parcel': 10.0,
                'geometry': box(0, 0, 10, 10),
            },
        ]
    )
    monkeypatch.setattr(readers, 'get_entities', lambda *a, **k: footprints)

    state = attrs.summarize_footprint_morphology(
        state, footprint_recipe_id='fp', min_overlap_m2=10.0
    )

    assert state.spine.loc['A', 'n_footprints_per_parcel'] == 1
    assert state.spine.loc['A', 'n_primary_footprints_per_parcel'] == 1


def test_synthetic_fallback_bypasses_floor(monkeypatch):
    # A synthetic parcel-shaped fallback footprint (geometry_source contains
    # '.') has no meaningful overlap area, but must still count.
    state = _parcel_spine(['A'])
    footprints = _footprints(
        [
            {
                'parcel_id': 'A',
                'geometry_source': 'parcel.nconemap',
                'area_intersection_m2_parcel': 0.0,
                'priority_on_parcel': 'secondary',
                'geometry': box(0, 0, 10, 10),
            },
        ]
    )
    monkeypatch.setattr(readers, 'get_entities', lambda *a, **k: footprints)

    state = attrs.summarize_footprint_morphology(
        state, footprint_recipe_id='fp', min_overlap_m2=10.0
    )

    assert state.spine.loc['A', 'n_footprints_per_parcel'] == 1
    # Synthetic rows bypass the priority exclusion too.
    assert state.spine.loc['A', 'n_primary_footprints_per_parcel'] == 1
    # But size/shape aggregates still exclude synthetic rows.
    assert state.spine.loc['A', 'n_small_elongated_footprints_per_parcel'] == 0


def test_secondary_priority_excluded_from_primary_count_only(monkeypatch):
    # A parcel with one primary home and one legitimate, substantial (not a
    # sliver) accessory structure: n_footprints_per_parcel should count both,
    # n_primary_footprints_per_parcel only the primary one.
    state = _parcel_spine(['A'])
    footprints = _footprints(
        [
            {
                'parcel_id': 'A',
                'priority_on_parcel': 'primary',
                'geometry': box(0, 0, 10, 10),
            },
            {
                'parcel_id': 'A',
                'priority_on_parcel': 'secondary',
                'geometry': box(20, 0, 30, 10),
            },
        ]
    )
    monkeypatch.setattr(readers, 'get_entities', lambda *a, **k: footprints)

    state = attrs.summarize_footprint_morphology(state, footprint_recipe_id='fp')

    assert state.spine.loc['A', 'n_footprints_per_parcel'] == 2
    assert state.spine.loc['A', 'n_primary_footprints_per_parcel'] == 1


def test_dwelling_confirmed_footprint_sets_span_and_dwelling_maxima(monkeypatch):
    # Footprint 1 is dwelling-confirmed (n_dwellings_overture > 0) and spans 2
    # parcels and holds 3 dwellings -> both maxima should reflect it.
    # Footprint 2 has no dwelling confirmation at all (n_dwellings_overture=0)
    # but claims a much larger span/dwelling count -- it must be ignored,
    # since is_primary_candidate alone (sole footprint on its own parcel, NSI-
    # only, or synthetic) does not confirm a real dwelling.
    state = _parcel_spine(['A'])
    footprints = _footprints(
        [
            {
                'parcel_id': 'A',
                'n_dwellings_overture': 3,
                'n_parcels_per_footprint': 2,
                'geometry': box(0, 0, 10, 10),
            },
            {
                'parcel_id': 'A',
                'n_dwellings_overture': 0,
                'n_parcels_per_footprint': 9,
                'geometry': box(20, 0, 30, 10),
            },
        ]
    )
    monkeypatch.setattr(readers, 'get_entities', lambda *a, **k: footprints)

    state = attrs.summarize_footprint_morphology(state, footprint_recipe_id='fp')

    assert state.spine.loc['A', 'max_dwellings_per_footprint'] == 3
    assert state.spine.loc['A', 'max_parcels_per_footprint'] == 2


def test_no_dwelling_confirmation_leaves_maxima_zero(monkeypatch):
    # A sole footprint on its own parcel is is_primary_candidate=True but has
    # no dwelling evidence -- it must not count toward either maximum, even
    # though it reports a span of 2.
    state = _parcel_spine(['A'])
    footprints = _footprints(
        [
            {
                'parcel_id': 'A',
                'n_dwellings_overture': 0,
                'n_parcels_per_footprint': 2,
                'geometry': box(0, 0, 10, 10),
            },
        ]
    )
    monkeypatch.setattr(readers, 'get_entities', lambda *a, **k: footprints)

    state = attrs.summarize_footprint_morphology(state, footprint_recipe_id='fp')

    assert state.spine.loc['A', 'max_dwellings_per_footprint'] == 0
    assert state.spine.loc['A', 'max_parcels_per_footprint'] == 0


def test_missing_dwelling_and_span_columns_does_not_crash(monkeypatch):
    state = _parcel_spine(['A'])
    footprints = gpd.GeoDataFrame(
        {'parcel_id': ['A'], 'geometry_source': ['obm']},
        geometry=[box(0, 0, 10, 10)],
        crs='EPSG:32617',
    )
    monkeypatch.setattr(readers, 'get_entities', lambda *a, **k: footprints)

    state = attrs.summarize_footprint_morphology(state, footprint_recipe_id='fp')

    assert state.spine.loc['A', 'max_dwellings_per_footprint'] == 0
    assert state.spine.loc['A', 'max_parcels_per_footprint'] == 0


def test_sum_footprint_area_reflects_total_not_just_the_largest(monkeypatch):
    # Two differently-sized real footprints on one parcel: max_footprint_area_m2
    # should be the larger one alone (100), sum_footprint_area_m2 the total of
    # both (150) -- distinct aggregates, not aliases of each other.
    state = _parcel_spine(['A'])
    footprints = _footprints(
        [
            {'parcel_id': 'A', 'geometry': box(0, 0, 10, 10)},  # 100 m2
            {'parcel_id': 'A', 'geometry': box(20, 0, 25, 10)},  # 50 m2
        ]
    )
    monkeypatch.setattr(readers, 'get_entities', lambda *a, **k: footprints)

    state = attrs.summarize_footprint_morphology(state, footprint_recipe_id='fp')

    max_area = state.spine.loc['A', 'max_footprint_area_m2']
    sum_area = state.spine.loc['A', 'sum_footprint_area_m2']
    assert max_area == pytest.approx(100.0, rel=1e-2)
    assert sum_area == pytest.approx(150.0, rel=1e-2)
    assert sum_area > max_area


def test_synthetic_fallback_excluded_from_sum_area(monkeypatch):
    # A parcel whose only footprint is a synthetic parcel-shaped fallback: like
    # max_footprint_area_m2, sum_footprint_area_m2 must stay NaN (no real
    # footprint area evidence), not silently become 0.
    state = _parcel_spine(['A'])
    footprints = _footprints(
        [
            {
                'parcel_id': 'A',
                'geometry_source': 'parcel.nconemap',
                'area_intersection_m2_parcel': 0.0,
                'geometry': box(0, 0, 10, 10),
            },
        ]
    )
    monkeypatch.setattr(readers, 'get_entities', lambda *a, **k: footprints)

    state = attrs.summarize_footprint_morphology(
        state, footprint_recipe_id='fp', min_overlap_m2=10.0
    )

    assert pd.isna(state.spine.loc['A', 'max_footprint_area_m2'])
    assert pd.isna(state.spine.loc['A', 'sum_footprint_area_m2'])


def test_missing_overlap_and_priority_columns_does_not_crash(monkeypatch):
    state = _parcel_spine(['A'])
    footprints = gpd.GeoDataFrame(
        {'parcel_id': ['A'], 'geometry_source': ['obm']},
        geometry=[box(0, 0, 10, 10)],
        crs='EPSG:32617',
    )
    monkeypatch.setattr(readers, 'get_entities', lambda *a, **k: footprints)

    state = attrs.summarize_footprint_morphology(state, footprint_recipe_id='fp')

    # No floor/priority columns available -> no filtering applied, matching
    # prior (pre-fix) behavior.
    assert state.spine.loc['A', 'n_footprints_per_parcel'] == 1
    assert state.spine.loc['A', 'n_primary_footprints_per_parcel'] == 1
