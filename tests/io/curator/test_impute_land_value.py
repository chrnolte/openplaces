"""Tests for `impute_land_value` (land_value.py)."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon

from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState
from openplaces.io.curator.land_value import impute_land_value
from openplaces.io.curator.provenance import is_imputed


def _state(df: pd.DataFrame) -> CurateState:
    return CurateState(
        recipe={},
        entity_recipe={},
        admin_id=AdminId('US'),
        verbose=False,
        timer=None,
        curated=df,
    )


def _row(**overrides) -> dict:
    row = {
        'land_value': None,
        'improvement_value': None,
        'footprint_area_m2_in_parcel': None,
        'area_ha': None,
        'land_use_class': None,
        'use_group_combined': None,
        'address_street': None,
        'city': None,
    }
    row.update(overrides)
    return row


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_imputes_land_value_for_residential_candidate():
    # Three donors with a known, constant $/m2-of-lot rate (200), residential
    # like the candidate (no street/city on either side, so this exercises
    # the fallback tier -- grouped by residential-ness, not use_group_combined).
    donors = [
        _row(
            land_use_class='Single-Family',
            land_value=v,
            area_ha=100,
        )
        for v in (10_000, 20_000, 30_000)
    ]
    candidate = _row(
        land_use_class='Single-Family',
        land_value=0,
        improvement_value=50_000,
        area_ha=100,
    )
    df = _frame([*donors, candidate])
    out = impute_land_value(_state(df), min_group_size=3).curated

    assert out['land_value_imputed'].iloc[-1] == 200 * 100
    assert out['improvement_value_imputed'].iloc[-1] == 50_000 - 200 * 100


def test_rate_and_estimate_use_parcel_area_not_footprint_area():
    # Two donors with IDENTICAL land_value/m2 (same true rate, 200/m2 of lot)
    # but WILDLY DIFFERENT footprint_area_m2_in_parcel -- if the rate were ever
    # computed against footprint area instead of lot area, these two donors
    # would disagree and/or the final estimate would be wrong. A candidate
    # with its own small footprint on a much larger lot must get an estimate
    # based on its *lot* area (m2), not its footprint.
    donors = [
        _row(
            land_use_class='Single-Family',
            land_value=20_000,
            area_ha=100,
            footprint_area_m2_in_parcel=10,  # tiny building on this lot
        ),
        _row(
            land_use_class='Single-Family',
            land_value=20_000,
            area_ha=100,
            footprint_area_m2_in_parcel=90,  # huge building on the same-size lot
        ),
        _row(
            land_use_class='Single-Family',
            land_value=20_000,
            area_ha=100,
            footprint_area_m2_in_parcel=50,
        ),
    ]
    candidate = _row(
        land_use_class='Single-Family',
        land_value=0,
        improvement_value=50_000,
        area_ha=500,  # a large lot...
        footprint_area_m2_in_parcel=5,  # ...with a tiny footprint on it
    )
    df = _frame([*donors, candidate])
    out = impute_land_value(_state(df), min_group_size=3).curated

    # Donor rate is 200/m2 of LOT regardless of their footprint sizes; the
    # candidate's estimate must be rate * its own LOT area (500), not its
    # footprint area (5).
    assert out['land_value_imputed'].iloc[-1] == 200 * 500


def test_donor_rate_never_reflects_improvement_value():
    # A donor with a distinctive land_value but a wildly different
    # improvement_value: the learned rate must track land_value only.
    donors = [
        _row(
            land_use_class='Single-Family',
            land_value=v,
            improvement_value=999_999_999,  # deliberately unrelated/huge
            area_ha=100,
        )
        for v in (10_000, 20_000, 30_000)
    ]
    candidate = _row(
        land_use_class='Single-Family',
        land_value=0,
        improvement_value=50_000,
        area_ha=100,
    )
    df = _frame([*donors, candidate])
    out = impute_land_value(_state(df), min_group_size=3).curated
    assert out['land_value_imputed'].iloc[-1] == 200 * 100


def test_real_land_value_passes_through_unchanged():
    # A row with a real, already-present land_value: land_value_imputed and
    # improvement_value_imputed must passthrough the real values unchanged
    # (not stay missing, and not have anything subtracted) -- these are the
    # parcel's final, best-available values, not sparse imputation-only
    # columns.
    df = _frame(
        [
            _row(
                land_use_class='Single-Family',
                land_value=5_000,
                improvement_value=50_000,
                area_ha=100,
            )
        ]
    )
    out = impute_land_value(_state(df), min_group_size=1).curated
    assert out['land_value_imputed'].iloc[0] == 5_000
    assert out['improvement_value_imputed'].iloc[0] == 50_000


def test_improvement_value_passes_through_when_not_a_candidate():
    # land_value is missing but the row is not eligible for imputation
    # (non-residential, not footprint-heavy relative to its peers): no land
    # estimate is made, so land_value_imputed stays missing, but
    # improvement_value_imputed must still passthrough the real
    # improvement_value unchanged -- there's nothing to subtract when no land
    # value was ever derived from it.
    peers = [
        _row(
            use_group_combined='COMMERCIAL',
            land_use_class='Commercial',
            land_value=v,
            improvement_value=100_000 - v,
            area_ha=100,
            footprint_area_m2_in_parcel=100,
        )
        for v in (10_000, 20_000, 30_000)
    ]
    not_a_candidate = _row(
        use_group_combined='COMMERCIAL',
        land_use_class='Commercial',
        land_value=0,
        improvement_value=500_000,
        area_ha=25,
        # Exactly 25% of the group median (100, computed including this row)
        # -- the ">" comparison isn't satisfied, so not footprint-heavy.
        footprint_area_m2_in_parcel=25,
    )
    df = _frame([*peers, not_a_candidate])
    out = impute_land_value(_state(df), min_group_size=3).curated
    assert pd.isna(out['land_value_imputed'].iloc[-1])
    assert out['improvement_value_imputed'].iloc[-1] == 500_000


def test_footprint_ratio_detection_fires_above_threshold():
    # Three non-residential peers (footprint area 100, also donors) plus a
    # candidate whose FOOTPRINT area (1000) is far more than 25% of the
    # group's median (100) -> eligible on footprint-share grounds despite
    # not being residential. The candidate's LOT area (m2) is set to a
    # different value (2000) than its footprint area (1000), to confirm the
    # eligibility test and the value estimate correctly use their own,
    # separate area columns. Fallback tier groups by residential-ness: all
    # rows here are non-residential (Commercial), so they share one group.
    peers = [
        _row(
            use_group_combined='COMMERCIAL',
            land_use_class='Commercial',
            land_value=v,
            improvement_value=100_000 - v,
            area_ha=100,
            footprint_area_m2_in_parcel=100,
        )
        for v in (10_000, 20_000, 30_000)
    ]
    candidate = _row(
        use_group_combined='COMMERCIAL',
        land_use_class='Commercial',
        land_value=0,
        improvement_value=500_000,
        area_ha=2000,
        footprint_area_m2_in_parcel=1000,
    )
    df = _frame([*peers, candidate])
    out = impute_land_value(_state(df), min_group_size=3).curated
    # rate (200/m2 of lot) * the candidate's own LOT area (2000), not its
    # footprint area (1000).
    assert out['land_value_imputed'].iloc[-1] == 200 * 2000


def test_footprint_ratio_detection_does_not_fire_at_threshold():
    # Candidate's own FOOTPRINT area (25) is exactly 25% of the group median
    # (100, computed including the candidate itself) -> the ">" comparison
    # is not satisfied, and land_use_class is non-residential, so it's not a
    # candidate at all.
    peers = [
        _row(
            use_group_combined='COMMERCIAL',
            land_use_class='Commercial',
            land_value=v,
            improvement_value=100_000 - v,
            area_ha=100,
            footprint_area_m2_in_parcel=100,
        )
        for v in (10_000, 20_000, 30_000)
    ]
    candidate = _row(
        use_group_combined='COMMERCIAL',
        land_use_class='Commercial',
        land_value=0,
        improvement_value=500_000,
        area_ha=25,
        footprint_area_m2_in_parcel=25,
    )
    df = _frame([*peers, candidate])
    out = impute_land_value(_state(df), min_group_size=3).curated
    assert pd.isna(out['land_value_imputed'].iloc[-1])


def test_tier1_street_city_wins_over_broader_fallback():
    # Fallback-tier donors (different street, non-residential so they'd never
    # match the residential fallback bucket either) carry a $/m2-of-lot rate
    # of 1000; same-street/city donors carry a distinct rate of 500. The
    # candidate shares the street/city cohort, so it must pick up the local
    # rate (500), not the broader fallback rate (1000) -- proving tier order,
    # not just that *some* estimate was produced. address_street here is
    # already the parsed, canonical street name (as
    # reconcile_addresses_df would write it) -- no house number, no unit
    # suffix to worry about at this layer.
    fallback_donors = [
        _row(
            land_use_class='Single-Family',
            land_value=100_000,
            area_ha=100,
            address_street='OTHER AVE',
            city='TOWNA',
        )
        for _ in range(5)
    ]
    street_donors = [
        _row(
            land_use_class='Single-Family',
            land_value=50_000,
            area_ha=100,
            address_street='MAIN ST',
            city='TOWNA',
        )
        for _ in range(2)
    ]
    candidate = _row(
        land_use_class='Single-Family',
        land_value=0,
        improvement_value=200_000,
        area_ha=100,
        address_street='MAIN ST',
        city='TOWNA',
    )
    df = _frame([*fallback_donors, *street_donors, candidate])
    out = impute_land_value(_state(df), min_group_size=2).curated

    assert out['land_value_imputed'].iloc[-1] == 500 * 100
    assert out['land_value_imputed_source'].iloc[-1] == 'address_street_city+imputed'


def test_city_tier_wins_over_county_wide_fallback():
    # Regression test for a real Middlesex County, MA case: the candidate's
    # own street has no donors at all (e.g. an address-parsing gap left it
    # unmatched to any real same-street neighbor), so tier 1 must fail. The
    # candidate's city (TOWNA) has a distinct, lower $/area-of-lot rate (500)
    # than a different city (TOWNB, rate 5000) -- if the final,
    # unconditional _is_residential-only tier fired instead of the
    # intermediate city tier, the estimate would reflect a blend of both
    # cities (a ~5.5x-too-high rate here), not the pure TOWNA rate.
    city_donors = [
        _row(
            land_use_class='Single-Family',
            land_value=50_000,
            area_ha=100,
            address_street=f'OTHER ST {i}',
            city='TOWNA',
        )
        for i in range(5)
    ]
    other_city_donors = [
        _row(
            land_use_class='Single-Family',
            land_value=500_000,
            area_ha=100,
            address_street=f'FAR ST {i}',
            city='TOWNB',
        )
        for i in range(5)
    ]
    candidate = _row(
        land_use_class='Condominium',
        land_value=0,
        improvement_value=200_000,
        area_ha=100,
        address_street='MAIN ST',
        city='TOWNA',
    )
    df = _frame([*city_donors, *other_city_donors, candidate])
    out = impute_land_value(_state(df), min_group_size=5).curated

    assert out['land_value_imputed'].iloc[-1] == 50_000
    assert out['land_value_imputed_source'].iloc[-1] == 'city__is_residential+imputed'


def test_tier1_gracefully_skipped_when_city_missing():
    # The candidate's own city is NaN, so its street/city key can never match
    # any donor group (by construction, regardless of what donor groups
    # exist) -- it must fall straight through to the residential fallback
    # without raising.
    fallback_donors = [
        _row(
            land_use_class='Single-Family',
            land_value=100_000,
            area_ha=100,
        )
        for _ in range(3)
    ]
    candidate = _row(
        land_use_class='Single-Family',
        land_value=0,
        improvement_value=200_000,
        area_ha=100,
        address_street='MAIN ST',
        city=None,
    )
    df = _frame([*fallback_donors, candidate])
    out = impute_land_value(_state(df), min_group_size=3).curated

    assert out['land_value_imputed'].iloc[-1] == 1_000 * 100
    assert out['land_value_imputed_source'].iloc[-1] == '_is_residential+imputed'


def test_improvement_value_imputed_clipped_at_zero():
    # Donor rate (2000/m2 of lot) applied to the candidate's area would
    # exceed its own improvement_value -> improvement_value_imputed must
    # clip at 0, not go negative.
    donors = [
        _row(
            land_use_class='Single-Family',
            land_value=v,
            area_ha=100,
        )
        for v in (190_000, 200_000, 210_000)
    ]
    candidate = _row(
        land_use_class='Single-Family',
        land_value=0,
        improvement_value=50_000,
        area_ha=100,
    )
    df = _frame([*donors, candidate])
    out = impute_land_value(_state(df), min_group_size=3).curated

    assert out['land_value_imputed'].iloc[-1] == 200_000
    assert out['improvement_value_imputed'].iloc[-1] == 0


def test_group_below_min_group_size_left_unimputed():
    donor = _row(land_use_class='Single-Family', land_value=100_000, area_ha=100)
    candidate = _row(
        land_use_class='Single-Family',
        land_value=0,
        improvement_value=200_000,
        area_ha=100,
    )
    df = _frame([donor, candidate])
    out = impute_land_value(_state(df), min_group_size=2).curated
    assert pd.isna(out['land_value_imputed'].iloc[-1])


def test_condominium_imputed_like_other_residential_classes():
    # Donors are Single-Family, not Condominium -- the whole point of the
    # residential-wide fallback bucket is that a condo candidate can learn
    # from its (far more numerous, non-condo) residential neighbors instead
    # of being isolated with a thin, systematically-low condo-only sample.
    donors = [
        _row(
            land_use_class='Single-Family',
            land_value=v,
            area_ha=100,
        )
        for v in (10_000, 20_000, 30_000)
    ]
    candidate = _row(
        land_use_class='Condominium',
        land_value=0,
        improvement_value=50_000,
        area_ha=100,
    )
    df = _frame([*donors, candidate])
    # Default residential_classes (not overridden) must already include
    # 'Condominium'.
    out = impute_land_value(_state(df), min_group_size=3).curated
    assert out['land_value_imputed'].iloc[-1] == 200 * 100


def test_provenance_distinguishes_imputed_rows_from_passthrough_rows():
    donors = [
        _row(
            land_use_class='Single-Family',
            land_value=v,
            area_ha=100,
        )
        for v in (10_000, 20_000, 30_000)
    ]
    untouched = _row(
        land_use_class='Single-Family',
        land_value=15_000,
        improvement_value=50_000,
        area_ha=100,
    )
    candidate = _row(
        land_use_class='Single-Family',
        land_value=0,
        improvement_value=50_000,
        area_ha=100,
    )
    df = _frame([*donors, untouched, candidate])
    out = impute_land_value(_state(df), min_group_size=3).curated

    sources = out['land_value_imputed_source']
    assert sources.iloc[-1] == '_is_residential+imputed'

    # The 'untouched' row's real land_value passes through into the final
    # column, and now says so: a passthrough carries the incoming column's
    # own provenance rather than nothing, so every row of the column can be
    # told apart as observed or estimated. Leaving it null was the gap --
    # a reader could not distinguish "real value" from "not decided".
    assert out['land_value_imputed'].iloc[-2] == 15_000
    assert sources.iloc[-2] == 'land_value'
    assert not is_imputed(sources).iloc[-2]
    assert is_imputed(sources).iloc[-1]


def test_derived_improvement_value_is_marked_imputed():
    donors = [
        _row(land_use_class='Single-Family', land_value=v, area_ha=100)
        for v in (10_000, 20_000, 30_000)
    ]
    untouched = _row(
        land_use_class='Single-Family',
        land_value=15_000,
        improvement_value=50_000,
        area_ha=100,
    )
    candidate = _row(
        land_use_class='Single-Family',
        land_value=0,
        improvement_value=50_000,
        area_ha=100,
    )
    df = _frame([*donors, untouched, candidate])
    out = impute_land_value(_state(df), min_group_size=3).curated

    # improvement_value_imputed is improvement_value minus the imputed land
    # value on a candidate row -- derived from a derived number -- and is
    # what reaches a footprint as structure_value.
    sources = out['improvement_value_imputed_source']
    assert sources.iloc[-1] == '_is_residential+imputed'
    assert is_imputed(sources).iloc[-1]
    assert sources.iloc[-2] == 'improvement_value'
    assert not is_imputed(sources).iloc[-2]


def test_missing_required_column_is_a_no_op():
    df = _frame([_row(land_value=0, improvement_value=50_000, area_ha=100)]).drop(
        columns=['land_use_class']
    )
    out = impute_land_value(_state(df), min_group_size=1).curated
    assert 'land_value_imputed' not in out.columns


def test_missing_parcel_area_column_is_a_no_op():
    df = _frame([_row(land_value=0, improvement_value=50_000)]).drop(
        columns=['area_ha']
    )
    out = impute_land_value(_state(df), min_group_size=1).curated
    assert 'land_value_imputed' not in out.columns


def test_rate_is_a_median_not_a_mean():
    # Four honest donors at 200/ha and one extreme one. The mean rate is
    # 40,160/ha and the median 200/ha, so the two statistics disagree by
    # 200x on the same cohort -- which is the whole reason the default is
    # the median. Measured on real data, the mean donor rate across the
    # ten surveyed NC counties is 1,017x the median.
    donors = [
        _row(land_use_class='Single-Family', land_value=20_000, area_ha=100)
        for _ in range(4)
    ]
    outlier = _row(land_use_class='Single-Family', land_value=200_000, area_ha=1)
    candidate = _row(
        land_use_class='Single-Family',
        land_value=0,
        improvement_value=5_000_000,
        area_ha=100,
    )
    df = _frame([*donors, outlier, candidate])
    out = impute_land_value(_state(df), min_group_size=3).curated

    assert out['land_value_imputed'].iloc[-1] == 200 * 100  # median
    assert out['land_value_imputed'].iloc[-1] != 40_160 * 100  # mean


def test_mean_is_still_available_when_a_recipe_asks_for_it():
    donors = [
        _row(land_use_class='Single-Family', land_value=20_000, area_ha=100)
        for _ in range(4)
    ]
    outlier = _row(land_use_class='Single-Family', land_value=200_000, area_ha=1)
    candidate = _row(
        land_use_class='Single-Family',
        land_value=0,
        improvement_value=5_000_000,
        area_ha=100,
    )
    df = _frame([*donors, outlier, candidate])
    out = impute_land_value(_state(df), min_group_size=3, statistic='mean').curated
    assert out['land_value_imputed'].iloc[-1] == 40_160 * 100


def test_a_donor_with_a_rounding_artifact_area_cannot_teach_a_rate():
    # A parcel recorded at 0.000001 ha with a real land value implies a
    # rate of 20 billion per hectare. It is not a lot, so it is not a
    # donor -- the guard drops it before it can teach anything.
    donors = [
        _row(land_use_class='Single-Family', land_value=20_000, area_ha=100)
        for _ in range(3)
    ]
    degenerate = _row(
        land_use_class='Single-Family', land_value=20_000, area_ha=0.000001
    )
    candidate = _row(
        land_use_class='Single-Family',
        land_value=0,
        improvement_value=10_000_000,
        area_ha=100,
    )
    df = _frame([*donors, degenerate, candidate])
    out = impute_land_value(_state(df), min_group_size=3, statistic='mean').curated
    # With the degenerate donor excluded the mean is the honest 200/ha.
    assert out['land_value_imputed'].iloc[-1] == 200 * 100


def test_a_condominium_is_still_imputed_and_still_subtracted():
    # Shared-lot classes stay covered: the point of the rate fix was to
    # stop over-imputing them, never to drop them. A condo's land value is
    # still learned from non-condo neighbours, still written, and still
    # subtracted from its improvement value -- that subtraction is how a
    # condo's combined assessed figure gets split into land and structure.
    donors = [
        _row(land_use_class='Single-Family', land_value=20_000, area_ha=100)
        for _ in range(3)
    ]
    condo = _row(
        land_use_class='Condominium',
        land_value=0,
        improvement_value=300_000,
        area_ha=1,
    )
    df = _frame([*donors, condo])
    out = impute_land_value(_state(df), min_group_size=3).curated

    assert out['land_value_imputed'].iloc[-1] == 200 * 1
    assert out['improvement_value_imputed'].iloc[-1] == 300_000 - 200
    assert is_imputed(out['land_value_imputed_source']).iloc[-1]


def test_over_imputation_can_still_erase_a_structure_value():
    # Documents the residual the rate fix does NOT remove: where the
    # imputed land value exceeds the whole improvement value, the clip
    # takes the structure value to zero. 402 parcels across the ten
    # surveyed counties still land here (down from 755 under the mean).
    donors = [
        _row(land_use_class='Single-Family', land_value=1_000_000, area_ha=1)
        for _ in range(3)
    ]
    condo = _row(
        land_use_class='Condominium', land_value=0, improvement_value=5_000, area_ha=1
    )
    df = _frame([*donors, condo])
    out = impute_land_value(_state(df), min_group_size=3).curated
    assert out['land_value_imputed'].iloc[-1] == 1_000_000
    assert out['improvement_value_imputed'].iloc[-1] == 0


def _geo_frame(rows: list[dict], geoms) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(rows, geometry=list(geoms), crs='EPSG:4326')


def _box(x: float) -> Polygon:
    return Polygon([(x, 0), (x + 1, 0), (x + 1, 1), (x, 1)])


def test_records_sharing_a_lot_split_its_value_instead_of_repeating_it():
    # Three donors establish 200/ha. Four condo records share ONE lot
    # geometry, each repeating the lot's whole 1 ha area -- the shape that
    # charged every unit for the entire development.
    donors = [
        _row(land_use_class='Single-Family', land_value=20_000, area_ha=100)
        for _ in range(3)
    ]
    units = [
        _row(
            land_use_class='Condominium',
            land_value=0,
            improvement_value=100_000,
            area_ha=1,
        )
        for _ in range(4)
    ]
    geoms = [_box(0), _box(2), _box(4)] + [_box(10)] * 4
    out = impute_land_value(
        _state(_geo_frame([*donors, *units], geoms)), min_group_size=3
    ).curated

    lot = 200 * 1
    shares = out['land_value_imputed'].iloc[-4:]
    assert (shares == lot / 4).all()
    # Conservation: the lot never gives out more land value than it holds.
    assert shares.sum() == pytest.approx(lot)


def test_a_lot_is_split_by_improvement_value_not_evenly():
    # A penthouse holds more of the common land than a studio, which is how
    # a condominium declaration assigns percentage interest.
    donors = [
        _row(land_use_class='Single-Family', land_value=20_000, area_ha=100)
        for _ in range(3)
    ]
    big = _row(
        land_use_class='Condominium', land_value=0, improvement_value=300_000, area_ha=1
    )
    small = _row(
        land_use_class='Condominium', land_value=0, improvement_value=100_000, area_ha=1
    )
    geoms = [_box(0), _box(2), _box(4), _box(10), _box(10)]
    out = impute_land_value(
        _state(_geo_frame([*donors, big, small], geoms)), min_group_size=3
    ).curated

    lot = 200 * 1
    assert out['land_value_imputed'].iloc[-2] == pytest.approx(lot * 0.75)
    assert out['land_value_imputed'].iloc[-1] == pytest.approx(lot * 0.25)


def test_identical_area_with_different_geometry_is_never_split():
    # The trap this keys on geometry to avoid: a platted subdivision sells
    # many lots cut to the same size. Splitting their land value between
    # unrelated neighbours would be a worse bug than the one being fixed --
    # measured in Beaufort NC, 635 Single-Family parcels share an area with
    # another while only 18 share a geometry.
    donors = [
        _row(land_use_class='Single-Family', land_value=20_000, area_ha=100)
        for _ in range(3)
    ]
    neighbours = [
        _row(
            land_use_class='Single-Family',
            land_value=0,
            improvement_value=150_000,
            area_ha=1,
        )
        for _ in range(3)
    ]
    # Same area_ha, deliberately DIFFERENT geometries.
    geoms = [_box(0), _box(2), _box(4), _box(20), _box(22), _box(24)]
    out = impute_land_value(
        _state(_geo_frame([*donors, *neighbours], geoms)), min_group_size=3
    ).curated
    assert (out['land_value_imputed'].iloc[-3:] == 200 * 1).all()


def test_a_shared_lot_with_no_improvement_values_splits_evenly():
    donors = [
        _row(land_use_class='Single-Family', land_value=20_000, area_ha=100)
        for _ in range(3)
    ]
    # improvement_value present (candidacy needs it) but equal, so the
    # weighting degenerates to an even split by construction.
    units = [
        _row(
            land_use_class='Condominium',
            land_value=0,
            improvement_value=50_000,
            area_ha=1,
        )
        for _ in range(2)
    ]
    geoms = [_box(0), _box(2), _box(4), _box(10), _box(10)]
    out = impute_land_value(
        _state(_geo_frame([*donors, *units], geoms)), min_group_size=3
    ).curated
    assert (out['land_value_imputed'].iloc[-2:] == 100).all()


def test_a_frame_without_geometry_behaves_exactly_as_before():
    donors = [
        _row(land_use_class='Single-Family', land_value=20_000, area_ha=100)
        for _ in range(3)
    ]
    candidate = _row(
        land_use_class='Condominium', land_value=0, improvement_value=100_000, area_ha=1
    )
    out = impute_land_value(
        _state(_frame([*donors, candidate])), min_group_size=3
    ).curated
    assert out['land_value_imputed'].iloc[-1] == 200 * 1


def test_a_recorded_total_recovers_land_directly_without_a_rate():
    # total > improvement means the source recorded a land component all
    # along: land is exactly total - improvement, and no neighbour's rate
    # should get a vote. Donors here imply 200/ha, which on this 1 ha lot
    # would give 200 -- the test fails if the estimate is used instead.
    donors = [
        _row(land_use_class='Single-Family', land_value=20_000, area_ha=100)
        for _ in range(3)
    ]
    recorded = _row(
        land_use_class='Single-Family',
        land_value=0,
        improvement_value=180_000,
        total_value=250_000,
        area_ha=1,
    )
    out = impute_land_value(
        _state(_frame([*donors, recorded])), min_group_size=3
    ).curated

    assert out['land_value_imputed'].iloc[-1] == 70_000
    assert (
        out['land_value_imputed_source'].iloc[-1] == 'total_minus_improvement+imputed'
    )
    assert is_imputed(out['land_value_imputed_source']).iloc[-1]


def test_a_recorded_total_does_not_also_reduce_the_improvement_value():
    # The subtraction exists only for a source that folded land INTO the
    # improvement figure. Where total > improvement it plainly did not, so
    # subtracting would charge the parcel for its land twice.
    donors = [
        _row(land_use_class='Single-Family', land_value=20_000, area_ha=100)
        for _ in range(3)
    ]
    recorded = _row(
        land_use_class='Single-Family',
        land_value=0,
        improvement_value=180_000,
        total_value=250_000,
        area_ha=1,
    )
    out = impute_land_value(
        _state(_frame([*donors, recorded])), min_group_size=3
    ).curated
    assert out['improvement_value_imputed'].iloc[-1] == 180_000


def test_an_estimate_is_capped_at_the_parcels_own_total_value():
    # total == improvement: the source folded land in, so the rate is the
    # right tool -- but a parcel cannot hold more land value than its whole
    # assessed value. Uncapped the rate would give 1,000,000 here.
    donors = [
        _row(land_use_class='Single-Family', land_value=1_000_000, area_ha=1)
        for _ in range(3)
    ]
    candidate = _row(
        land_use_class='Single-Family',
        land_value=0,
        improvement_value=90_000,
        total_value=90_000,
        area_ha=1,
    )
    out = impute_land_value(
        _state(_frame([*donors, candidate])), min_group_size=3
    ).curated
    assert out['land_value_imputed'].iloc[-1] == 90_000
    assert out['land_value_imputed_source'].iloc[-1].endswith('+imputed')


def test_no_recorded_total_leaves_the_rate_uncapped():
    donors = [
        _row(land_use_class='Single-Family', land_value=1_000_000, area_ha=1)
        for _ in range(3)
    ]
    candidate = _row(
        land_use_class='Single-Family',
        land_value=0,
        improvement_value=90_000,
        area_ha=1,
    )
    out = impute_land_value(
        _state(_frame([*donors, candidate])), min_group_size=3
    ).curated
    assert out['land_value_imputed'].iloc[-1] == 1_000_000


def test_a_total_below_the_improvement_value_is_left_to_the_rate():
    # An internally inconsistent record (total < improvement) is not
    # evidence of a land component; it falls through to the normal path
    # rather than producing a negative land value.
    donors = [
        _row(land_use_class='Single-Family', land_value=20_000, area_ha=100)
        for _ in range(3)
    ]
    odd = _row(
        land_use_class='Single-Family',
        land_value=0,
        improvement_value=100_000,
        total_value=80_000,
        area_ha=1,
    )
    out = impute_land_value(_state(_frame([*donors, odd])), min_group_size=3).curated
    assert out['land_value_imputed'].iloc[-1] >= 0
    assert (
        out['land_value_imputed_source'].iloc[-1] != 'total_minus_improvement+imputed'
    )


def test_nullable_dtypes_do_not_break_the_lot_split():
    # A source whose values arrive as pandas nullable Float64/Int64 rather
    # than numpy floats propagates pd.NA into the groupby, and numpy then
    # refuses to evaluate the mask ("boolean value of NA is ambiguous").
    # New Hanover NC failed a whole county rebuild on exactly this.
    donors = [
        _row(land_use_class='Single-Family', land_value=20_000, area_ha=100)
        for _ in range(3)
    ]
    units = [
        _row(
            land_use_class='Condominium',
            land_value=0,
            improvement_value=100_000,
            area_ha=1,
        )
        for _ in range(2)
    ]
    orphan = _row(
        land_use_class='Condominium', land_value=0, improvement_value=None, area_ha=1
    )
    geoms = [_box(0), _box(2), _box(4), _box(10), _box(10), _box(30)]
    df = _geo_frame([*donors, *units, orphan], geoms)
    for col in ('land_value', 'improvement_value', 'area_ha'):
        df[col] = df[col].astype('Float64')

    out = impute_land_value(_state(df), min_group_size=3).curated
    assert out['land_value_imputed'].iloc[-3:-1].tolist() == [100.0, 100.0]
