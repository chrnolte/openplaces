"""Tests for the old-to-new parcel fractional crosswalk builder and its QA check."""

import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely.affinity
from shapely.geometry import box

from openplaces.geo.crosswalk import (
    build_id_or_overlay_crosswalk,
    warn_on_geo_id_area_mismatch,
)


def _gdf(boxes, crs='EPSG:4326'):
    return gpd.GeoDataFrame(
        {'parcel_id': list(boxes)},
        geometry=[boxes[k] for k in boxes],
        crs=crs,
    ).set_index('parcel_id')


def _grid_boxes(n, prefix, side_deg=0.0005, spacing_deg=0.002, x0=0.0, y0=0.0):
    """A dict of n x n small, well-separated square boxes (no fixture needed)."""
    boxes = {}
    for i in range(n):
        for j in range(n):
            x = x0 + i * spacing_deg
            y = y0 + j * spacing_deg
            boxes[f'{prefix}_{i}_{j}'] = box(x, y, x + side_deg, y + side_deg)
    return boxes


def test_geo_id_fast_path_matches_identical_geometry():
    old = _gdf({'old_1': box(0, 0, 0.001, 0.001)})
    new = _gdf({'new_1': box(0, 0, 0.001, 0.001)})

    crosswalk = build_id_or_overlay_crosswalk(new, old, min_overlap_m2=0)

    assert len(crosswalk) == 1
    row = crosswalk.iloc[0]
    assert row['match_type'] == 'geo_id'
    assert row['parcel_id_old'] == 'old_1'
    assert row['parcel_id_new'] == 'new_1'
    assert row['fraction_of_old'] == 1.0
    assert row['area_ha'] > 0


def test_overlay_fallback_splits_area_fractionally():
    old = _gdf({'old_2': box(0.01, 0, 0.03, 0.001)})
    new = _gdf(
        {
            'new_2': box(0.01, 0, 0.02, 0.001),
            'new_3': box(0.02, 0, 0.03, 0.001),
        }
    )

    crosswalk = build_id_or_overlay_crosswalk(new, old, min_overlap_m2=0)

    assert len(crosswalk) == 2
    assert set(crosswalk['match_type']) == {'overlay'}
    fractions = crosswalk.set_index('parcel_id_new')['fraction_of_old']
    assert fractions['new_2'] == pytest.approx(0.5, rel=1e-3)
    assert fractions['new_3'] == pytest.approx(0.5, rel=1e-3)
    assert crosswalk['fraction_of_old'].sum() == pytest.approx(1.0, rel=1e-6)


def test_min_overlap_m2_excludes_sliver():
    # new_4 overlaps old_3 by a ~5.5cm-wide sliver at the shared edge — well
    # under the default 10 m^2 threshold, and old_3 has no other overlap, so
    # it should simply be absent from the crosswalk.
    old = _gdf({'old_3': box(0.04, 0, 0.05, 0.001)})
    new = _gdf({'new_4': box(0.0499995, 0, 0.06, 0.001)})

    crosswalk = build_id_or_overlay_crosswalk(new, old, min_overlap_m2=10.0)

    assert crosswalk.empty


def test_geo_id_computed_automatically_when_missing():
    # Neither frame carries a 'geo_id' column (e.g. a harmonized spine, whose
    # resolve_spine step drops it) — the crosswalk should compute it from
    # geometry on both sides and still find the exact match.
    old = _gdf({'old_1': box(0, 0, 0.001, 0.001)})
    new = _gdf({'new_1': box(0, 0, 0.001, 0.001)})
    assert 'geo_id' not in old.columns
    assert 'geo_id' not in new.columns

    crosswalk = build_id_or_overlay_crosswalk(new, old, min_overlap_m2=0)

    assert len(crosswalk) == 1
    assert crosswalk.iloc[0]['match_type'] == 'geo_id'


def test_verbose_reports_geo_id_match_rate_before_overlay(capsys):
    # One geo_id-matched pair (old_1/new_1) plus one overlay-fallback pair
    # (old_2 split between new_2 and new_3) -- the geo_id report should count
    # only the former and be printed before the overlay call runs.
    old = _gdf(
        {
            'old_1': box(0, 0, 0.001, 0.001),
            'old_2': box(0.01, 0, 0.03, 0.001),
        }
    )
    new = _gdf(
        {
            'new_1': box(0, 0, 0.001, 0.001),
            'new_2': box(0.01, 0, 0.02, 0.001),
            'new_3': box(0.02, 0, 0.03, 0.001),
        }
    )

    build_id_or_overlay_crosswalk(new, old, min_overlap_m2=0, verbose=True)

    out = capsys.readouterr().out
    assert 'geo_id match: 1/3 new parcels (33.3%), 1/2 old parcels (50.0%)' in out
    assert 'remaining 2/1' in out


def test_duplicate_geo_id_raises_by_default():
    # Two old parcels with identical geometry (and thus identical geo_id) --
    # e.g. two condo-unit records sharing one physical footprint -- is a
    # structural fact about the input, not matching noise, so it raises
    # rather than silently deferring to overlay.
    shared = box(0, 0, 0.001, 0.001)
    old = _gdf({'old_a': shared, 'old_b': shared})
    new = _gdf({'new_1': shared})

    with pytest.raises(ValueError, match="'groupby'.*'first'|'first'.*'groupby'"):
        build_id_or_overlay_crosswalk(new, old, min_overlap_m2=0)


def test_duplicate_geo_id_first_keeps_one_arbitrary_row():
    shared = box(0, 0, 0.001, 0.001)
    old = _gdf({'old_a': shared, 'old_b': shared})
    new = _gdf({'new_1': shared})

    crosswalk = build_id_or_overlay_crosswalk(
        new, old, min_overlap_m2=0, on_duplicate_geo_id='first'
    )

    assert len(crosswalk) == 1
    assert crosswalk.iloc[0]['match_type'] == 'geo_id'
    assert crosswalk.iloc[0]['parcel_id_old'] in {'old_a', 'old_b'}


def test_duplicate_geo_id_groupby_aggregates_attributes():
    # old_a/old_b share a footprint (e.g. two condo-unit assessment records)
    # with different land_value ('sum' in the attribute registry) -- groupby
    # should aggregate that column and preserve both original ids.
    shared = box(0, 0, 0.001, 0.001)
    old = gpd.GeoDataFrame(
        {
            'parcel_id': ['old_a', 'old_b'],
            'land_value': [100.0, 50.0],
            'geometry': [shared, shared],
        },
        crs='EPSG:4326',
    ).set_index('parcel_id')
    new = _gdf({'new_1': shared})

    crosswalk = build_id_or_overlay_crosswalk(
        new, old, min_overlap_m2=0, on_duplicate_geo_id='groupby'
    )

    assert len(crosswalk) == 1
    assert crosswalk.iloc[0]['match_type'] == 'geo_id'
    # The resolved parcel_id_old is a representative of the merged group, and
    # the original ids/aggregated value should be recoverable from `old`'s
    # own resolution (checked directly against the helper below).
    from openplaces.geo.crosswalk import _resolve_duplicate_geo_ids
    from openplaces.geo.ids import get_geo_ids

    old_with_id = old.reset_index().rename(columns={'parcel_id': 'parcel_id_old'})
    old_with_id['geo_id'] = get_geo_ids(old_with_id, handle_duplicates=False)
    resolved = _resolve_duplicate_geo_ids(
        old_with_id, 'geo_id', 'parcel_id_old', 'groupby', 'old'
    )
    assert len(resolved) == 1
    assert resolved.iloc[0]['land_value'] == pytest.approx(150.0)
    assert set(resolved.iloc[0]['parcel_id_old_list']) == {'old_a', 'old_b'}


def test_no_duplicate_geo_id_unaffected_by_on_duplicate_geo_id():
    # With no duplicates present, on_duplicate_geo_id should never trigger --
    # behavior is identical to the default across all three settings.
    old = _gdf({'old_1': box(0, 0, 0.001, 0.001)})
    new = _gdf({'new_1': box(0, 0, 0.001, 0.001)})

    for mode in ('raise', 'first', 'groupby'):
        crosswalk = build_id_or_overlay_crosswalk(
            new, old, min_overlap_m2=0, on_duplicate_geo_id=mode
        )
        assert len(crosswalk) == 1
        assert crosswalk.iloc[0]['match_type'] == 'geo_id'


def _mismatched_geo_id_crosswalk():
    # Construct a crosswalk claiming a geo_id match between two geometries
    # whose true areas differ substantially (simulating a hash collision or
    # reprojection bug), bypassing the builder so the mismatch is guaranteed.
    old = _gdf({'old_1': box(0, 0, 0.001, 0.001)})
    new = _gdf({'new_1': box(0, 0, 0.002, 0.002)})  # ~4x the area
    crosswalk = pd.DataFrame(
        {
            'parcel_id_old': ['old_1'],
            'parcel_id_new': ['new_1'],
            'area_ha': [0.0],
            'match_type': ['geo_id'],
            'fraction_of_old': [1.0],
        }
    )
    return crosswalk, new, old


def test_warn_on_geo_id_area_mismatch_flags_and_plots(monkeypatch):
    crosswalk, new, old = _mismatched_geo_id_crosswalk()

    shown = []
    monkeypatch.setattr('matplotlib.pyplot.show', lambda: shown.append(True))

    with pytest.warns(UserWarning, match='geo_id-matched'):
        flagged = warn_on_geo_id_area_mismatch(crosswalk, new, old, tolerance=0.01)

    assert len(flagged) == 1
    assert flagged.iloc[0]['ratio'] == pytest.approx(4.0, rel=1e-2)
    assert shown


def test_warn_on_geo_id_area_mismatch_silent_suppresses_warning():
    crosswalk, new, old = _mismatched_geo_id_crosswalk()

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        flagged = warn_on_geo_id_area_mismatch(
            crosswalk, new, old, tolerance=0.01, silent=True
        )

    assert len(flagged) == 1


def test_calibration_recovers_systematic_shift():
    # A 5x5 grid (25 parcels per side) where `old` is offset from `new` by a
    # small, constant shift -- large enough (several grid_degrees cells) to
    # miss the exact hash, far too small (well under half the ~222m grid
    # spacing) to risk matching the wrong neighbor.
    new_boxes = _grid_boxes(5, 'new')
    dx, dy = 0.000004, 0.000003  # ~0.44m, ~0.33m at the equator
    old_boxes = {
        k.replace('new', 'old'): shapely.affinity.translate(v, xoff=dx, yoff=dy)
        for k, v in new_boxes.items()
    }
    new = _gdf(new_boxes)
    old = _gdf(old_boxes)

    crosswalk = build_id_or_overlay_crosswalk(new, old, min_overlap_m2=0)

    assert len(crosswalk) == 25
    assert set(crosswalk['match_type']) == {'geo_id_calibrated'}
    assert crosswalk['fraction_of_old'].eq(1.0).all()


def test_calibrate_group_col_recovers_per_group_shifts_pooling_fails():
    # Two "towns" with different constant shifts, far enough apart
    # spatially that neither group's sampling can see the other. Pooling
    # both into one calibration attempt can't find a single dominant
    # cluster (bimodal, evenly split 25/25) -- exactly the real-MA-data
    # motivation for calibrate_group_col: a shared county-wide estimate
    # blurs two individually-clean per-town shifts into noise. Grouping by
    # town lets each resolve on its own, trivially (zero noise within a
    # group, same setup as test_calibration_recovers_systematic_shift).
    new_boxes_a = _grid_boxes(5, 'new_a', x0=0.0, y0=0.0)
    new_boxes_b = _grid_boxes(5, 'new_b', x0=0.05, y0=0.0)
    dx_a, dy_a = 0.000004, 0.000003  # ~0.44m, ~0.33m
    dx_b, dy_b = -0.000004, -0.000003  # opposite direction
    old_boxes_a = {
        k.replace('new_a', 'old_a'): shapely.affinity.translate(v, xoff=dx_a, yoff=dy_a)
        for k, v in new_boxes_a.items()
    }
    old_boxes_b = {
        k.replace('new_b', 'old_b'): shapely.affinity.translate(v, xoff=dx_b, yoff=dy_b)
        for k, v in new_boxes_b.items()
    }

    new = _gdf({**new_boxes_a, **new_boxes_b})
    old = _gdf({**old_boxes_a, **old_boxes_b})
    new['town'] = ['A'] * 25 + ['B'] * 25
    old['town'] = ['A'] * 25 + ['B'] * 25

    grouped = build_id_or_overlay_crosswalk(
        new, old, min_overlap_m2=0, calibrate_group_col='town'
    )
    assert len(grouped) == 50
    assert set(grouped['match_type']) == {'geo_id_calibrated'}

    pooled = build_id_or_overlay_crosswalk(new, old, min_overlap_m2=0)
    assert not pooled['match_type'].eq('geo_id_calibrated').all()


def test_calibrate_pool_retries_after_a_failed_attempt(monkeypatch):
    # A single None from _estimate_offset_from_sample must not end
    # calibration for the whole pool: the old `break`-on-None behavior
    # meant one unlucky random sample silently gave up, matching the
    # near-zero recovery rate seen against real, noisy production data.
    import openplaces.geo.crosswalk as cw

    real_estimate = cw._estimate_offset_from_sample
    calls = []

    def flaky_estimate(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            return None
        return real_estimate(*args, **kwargs)

    monkeypatch.setattr(cw, '_estimate_offset_from_sample', flaky_estimate)

    new_boxes = _grid_boxes(5, 'new')
    dx, dy = 0.000004, 0.000003
    old_boxes = {
        k.replace('new', 'old'): shapely.affinity.translate(v, xoff=dx, yoff=dy)
        for k, v in new_boxes.items()
    }
    new = _gdf(new_boxes)
    old = _gdf(old_boxes)

    crosswalk = build_id_or_overlay_crosswalk(new, old, min_overlap_m2=0)

    assert len(calls) >= 2
    assert set(crosswalk['match_type']) == {'geo_id_calibrated'}


def test_kdtree_recovers_uncorrelated_fine_grained_offsets():
    # Each old parcel is nudged by an independent small random offset (not a
    # shared systematic shift) -- too scattered for the calibration stage's
    # dominant-cluster check to accept, but each pair is still well within
    # the direct KD-tree match's search radius and shape gate.
    rng = np.random.default_rng(0)
    new_boxes = _grid_boxes(5, 'new')
    old_boxes = {}
    for k, v in new_boxes.items():
        dx, dy = rng.uniform(
            -0.000003, 0.000003, size=2
        )  # up to ~0.33m, random direction
        old_boxes[k.replace('new', 'old')] = shapely.affinity.translate(
            v, xoff=dx, yoff=dy
        )
    new = _gdf(new_boxes)
    old = _gdf(old_boxes)

    crosswalk = build_id_or_overlay_crosswalk(new, old, min_overlap_m2=0)

    assert len(crosswalk) == 25
    assert set(crosswalk['match_type']) == {'geo_id_kdtree'}


def test_scattered_geometry_still_falls_back_to_overlay_at_scale():
    # The "old parcel split into two new parcels" scenario (a genuine area
    # mismatch, not a small offset), repeated at a scale large enough
    # (>= 20 pairs) to actually exercise the calibration/direct-match stages
    # rather than being skipped by the minimum-size guard -- both should
    # correctly decline to match (shape gate fails: half the area each),
    # leaving overlay as today's only resolver.
    old_boxes, new_boxes = {}, {}
    for i in range(25):
        x0 = i * 0.01
        old_boxes[f'old_{i}'] = box(x0, 0, x0 + 0.002, 0.001)
        new_boxes[f'new_{i}a'] = box(x0, 0, x0 + 0.001, 0.001)
        new_boxes[f'new_{i}b'] = box(x0 + 0.001, 0, x0 + 0.002, 0.001)
    old = _gdf(old_boxes)
    new = _gdf(new_boxes)

    crosswalk = build_id_or_overlay_crosswalk(new, old, min_overlap_m2=0)

    assert len(crosswalk) == 50
    assert set(crosswalk['match_type']) == {'overlay'}


def test_calibration_skipped_below_minimum_size():
    # Same constant-shift setup as test_calibration_recovers_systematic_shift
    # but with only a handful of parcels (below the calibration/direct-match
    # minimum-size guard) -- both stages must be skipped, falling straight to
    # overlay exactly as before they existed.
    new_boxes = _grid_boxes(2, 'new')  # 4 parcels, well under the size guard
    dx, dy = 0.000004, 0.000003
    old_boxes = {
        k.replace('new', 'old'): shapely.affinity.translate(v, xoff=dx, yoff=dy)
        for k, v in new_boxes.items()
    }
    new = _gdf(new_boxes)
    old = _gdf(old_boxes)

    crosswalk = build_id_or_overlay_crosswalk(new, old, min_overlap_m2=0)

    assert (
        not crosswalk['match_type'].isin(['geo_id_calibrated', 'geo_id_kdtree']).any()
    )


def test_warn_on_geo_id_area_mismatch_within_tolerance_does_not_flag():
    old = _gdf({'old_1': box(0, 0, 0.001, 0.001)})
    new = _gdf({'new_1': box(0, 0, 0.001, 0.001)})
    crosswalk = pd.DataFrame(
        {
            'parcel_id_old': ['old_1'],
            'parcel_id_new': ['new_1'],
            'area_ha': [0.0],
            'match_type': ['geo_id'],
            'fraction_of_old': [1.0],
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        flagged = warn_on_geo_id_area_mismatch(crosswalk, new, old, tolerance=0.01)

    assert flagged.empty
