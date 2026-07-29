"""Tests for `impute_land_value` (land_value.py)."""

from __future__ import annotations

import pandas as pd

from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState
from openplaces.io.curator.land_value import impute_land_value


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
    assert out['land_value_imputed_source'].iloc[-1] == 'address_street_city'


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
    assert out['land_value_imputed_source'].iloc[-1] == '_is_residential'


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


def test_provenance_recorded_only_for_imputed_rows():
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

    assert out['land_value_imputed_source'].iloc[-1] == '_is_residential'
    assert pd.isna(out['land_value_imputed_source'].iloc[-2])
    # The 'untouched' row's real land_value still passes through into the
    # final column, just with no provenance token (it's the real value).
    assert out['land_value_imputed'].iloc[-2] == 15_000


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
