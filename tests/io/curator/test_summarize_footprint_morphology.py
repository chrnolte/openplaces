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
        recipe={}, admin_id='US-NC-AR', verbose=False, timer=None, spine=spine
    )


def _parcel_spine_m(parcel_ids, geometries):
    """Like `_parcel_spine`, but caller-supplied geometries in the same
    metric CRS as `_footprints`, so a real spatial overlay against footprint
    geometry can be exercised (unlike `_parcel_spine`'s placeholder boxes,
    which live in a different CRS and location and never spatially overlap
    `_footprints`' boxes).
    """
    spine = gpd.GeoDataFrame(
        index=pd.Index(parcel_ids, name='parcel_id'),
        geometry=geometries,
        crs='EPSG:32617',
    )
    return HarmonizeState(
        recipe={}, admin_id='US-NC-AR', verbose=False, timer=None, spine=spine
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


def test_primary_footprint_area_excludes_secondary(monkeypatch):
    # Same setup as above, but checking the area sum rather than the count:
    # footprint_area_m2_dominant totals both footprints (200), while
    # footprint_area_m2_primary reflects only the primary one (100).
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

    dominant_area = state.spine.loc['A', 'footprint_area_m2_dominant']
    primary_area = state.spine.loc['A', 'footprint_area_m2_primary']
    assert dominant_area == pytest.approx(200.0, rel=1e-2)
    assert primary_area == pytest.approx(100.0, rel=1e-2)


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


def test_dominant_footprint_area_reflects_total_not_just_the_largest(monkeypatch):
    # Two differently-sized real footprints on one parcel: max_footprint_area_m2
    # should be the larger one alone (100), footprint_area_m2_dominant the
    # total of both (150) -- distinct aggregates, not aliases of each other.
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
    dominant_area = state.spine.loc['A', 'footprint_area_m2_dominant']
    assert max_area == pytest.approx(100.0, rel=1e-2)
    assert dominant_area == pytest.approx(150.0, rel=1e-2)
    assert dominant_area > max_area


def test_synthetic_fallback_excluded_from_sum_area(monkeypatch):
    # A parcel whose only footprint is a synthetic parcel-shaped fallback: like
    # max_footprint_area_m2, footprint_area_m2_dominant must stay NaN (no real
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
    assert pd.isna(state.spine.loc['A', 'footprint_area_m2_dominant'])
    # The synthetic fallback's geometry is the parcel boundary itself, not a
    # real structure -- it must not count as coverage either.
    assert pd.isna(state.spine.loc['A', 'footprint_area_m2_in_parcel'])


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
    # This fixture's spine (EPSG:4326, near-origin unit boxes) and footprint
    # (EPSG:32617, a UTM-meter box) never spatially coincide -- the new
    # overlay-based column must degrade to NaN, not crash or silently 0-fill.
    assert pd.isna(state.spine.loc['A', 'footprint_area_m2_in_parcel'])


def test_dominant_sum_vs_clipped_coverage_sum_diverge_on_a_straddling_footprint(
    monkeypatch,
):
    # Two adjoining parcels, A (x in [0, 10]) and B (x in [10, 20]), and one
    # footprint straddling the boundary (x in [5, 15]), split 50/50 between
    # them by area. The footprint's `parcel_id`/`area_intersection_m2_parcel`
    # fixture fields independently declare A as its dominant parcel (these
    # signals are set explicitly by the fixture, not derived from geometry).
    state = _parcel_spine_m(['A', 'B'], [box(0, 0, 10, 10), box(10, 0, 20, 10)])
    footprints = _footprints(
        [
            {
                'parcel_id': 'A',
                'area_intersection_m2_parcel': 50.0,
                'geometry': box(5, 0, 15, 10),
            },
        ]
    )
    monkeypatch.setattr(readers, 'get_entities', lambda *a, **k: footprints)

    state = attrs.summarize_footprint_morphology(
        state, footprint_recipe_id='fp', min_overlap_m2=10.0
    )

    # Full, unclipped area (100) credited only to the dominant parcel A.
    assert state.spine.loc['A', 'footprint_area_m2_dominant'] == pytest.approx(
        100.0, rel=1e-2
    )
    assert pd.isna(state.spine.loc['B', 'footprint_area_m2_dominant'])

    # Clipped area correctly split between both touching parcels.
    assert state.spine.loc['A', 'footprint_area_m2_in_parcel'] == pytest.approx(
        50.0, rel=1e-2
    )
    assert state.spine.loc['B', 'footprint_area_m2_in_parcel'] == pytest.approx(
        50.0, rel=1e-2
    )

    # The exact discrepancy this whole change fixes: the dominant sum alone
    # overstates A's coverage share; the in-parcel sum cannot.
    assert (
        state.spine.loc['A', 'footprint_area_m2_dominant']
        > state.spine.loc['A', 'footprint_area_m2_in_parcel']
    )


def test_in_parcel_sum_excludes_a_sub_floor_sliver_on_a_non_dominant_parcel(
    monkeypatch,
):
    # A third parcel, C (x in [20, 21]), touches the same straddling footprint
    # (x in [5, 15]) not at all -- use a footprint that clips a <10 m2 sliver
    # onto C instead, to confirm the per-pair min_overlap_m2 floor excludes it
    # even though the footprint clears the floor on its dominant parcel A.
    state = _parcel_spine_m(['A', 'C'], [box(0, 0, 10, 10), box(9.5, 0, 19.5, 10)])
    footprints = _footprints(
        [
            {
                'parcel_id': 'A',
                'area_intersection_m2_parcel': 95.0,
                'geometry': box(0, 0, 10, 10),
            },
        ]
    )
    monkeypatch.setattr(readers, 'get_entities', lambda *a, **k: footprints)

    state = attrs.summarize_footprint_morphology(
        state, footprint_recipe_id='fp', min_overlap_m2=10.0
    )

    # A's full footprint area, correctly clipped (the footprint is entirely
    # within A here, so dominant == in-parcel).
    assert state.spine.loc['A', 'footprint_area_m2_in_parcel'] == pytest.approx(
        100.0, rel=1e-2
    )
    # C only overlaps a 0.5 x 10 = 5 m2 sliver, below the 10 m2 floor.
    assert pd.isna(state.spine.loc['C', 'footprint_area_m2_in_parcel'])
