"""Tests for the parcel land-use vote and the shared indicator vocabulary.

Covers `resolve_by_vote`'s score_columns and review_column (the two features
the retired `classify_parcel_land_use` wrapper contributed), the real recipe's
rule shapes, and `evaluate_indicator`'s predicate types.
"""

from __future__ import annotations

import pandas as pd
import pytest

from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState
from openplaces.io.curator.indicators import evaluate_indicator
from openplaces.io.curator.reconcilers import resolve_by_vote


def _classify(state, rules, output='land_use_class', **kwargs):
    """Score parcel land-use rules through the shared vote.

    The parcel lane used to call a `classify_parcel_land_use` wrapper; these
    are the two settings that distinguished it. `preserve_base=False` means
    the vote *is* the classification rather than a correction to one, so
    parcels no decision claims stay null for `reconcile_land_use` to fill.
    """
    return resolve_by_vote(
        state,
        target=output,
        decisions=rules,
        preserve_base=False,
        default_source='rule',
        **kwargs,
    )


RULES = [
    {
        'class': 'Manufactured Home Park',
        'min_score': 2,
        'indicators': [
            {'type': 'keyword', 'column': 'use_group_combined', 'pattern': 'MOBILE'},
            {
                'type': 'in_set',
                'column': 'group_parcel',
                'values': ['Manufactured Home'],
            },
        ],
    },
    {
        'class': 'RV Park',
        'min_score': 2,
        'indicators': [
            {'type': 'keyword', 'column': 'use_group_combined', 'pattern': 'RV'},
            {
                'type': 'in_set',
                'column': 'group_parcel',
                'values': ['Entertainment/Recreation'],
            },
        ],
    },
    {
        'class': 'Vacant',
        'min_score': 1,
        'indicators': [
            {
                'type': 'numeric_at_most',
                'column': 'footprint_area_log_zscore',
                'max': -1.5,
            },
        ],
    },
]


def _state(df: pd.DataFrame) -> CurateState:
    return CurateState(
        recipe={},
        entity_recipe={},
        admin_id=AdminId('US'),
        verbose=False,
        timer=None,
        curated=df,
    )


def _frame(**overrides) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            'use_group_combined': ['RESIDENTIAL'],
            'group_parcel': ['Single Family'],
            'footprint_area_log_zscore': [0.0],
        }
    )
    for key, value in overrides.items():
        df[key] = value
    return df


def test_winner_selection_unchanged_with_shared_indicators():
    df = _frame(
        use_group_combined=['MOBILE HOME PARK'], group_parcel=['Manufactured Home']
    )
    out = _classify(_state(df), rules=RULES).curated
    assert out['land_use_class'].iloc[0] == 'Manufactured Home Park'


def test_score_columns_exposes_raw_score_even_when_not_winning():
    df = _frame(
        use_group_combined=['MOBILE HOME PARK'],
        group_parcel=['Manufactured Home'],
        footprint_area_log_zscore=[-2.0],
    )
    out = _classify(
        _state(df), rules=RULES, score_columns={'Vacant': 'vacancy_score'}
    ).curated
    # Manufactured Home Park wins (score 2), but Vacant's own score (1) is
    # still recorded, since score_columns is not gated by min_score / winning.
    assert out['land_use_class'].iloc[0] == 'Manufactured Home Park'
    assert out['vacancy_score'].iloc[0] == 1.0


def test_score_columns_zero_when_no_indicator_matches():
    df = _frame(footprint_area_log_zscore=[0.0])
    out = _classify(
        _state(df), rules=RULES, score_columns={'Vacant': 'vacancy_score'}
    ).curated
    assert out['vacancy_score'].iloc[0] == 0.0


def test_review_column_false_when_no_rule_wins():
    # No rule's indicators match this generic residential parcel, so no
    # decision reaches its min_score; review stays False (nothing to flag).
    df = _frame()
    out = _classify(_state(df), rules=RULES, review_column='land_use_review').curated
    assert not out['land_use_review'].iloc[0]


def test_review_column_flags_narrow_margin_between_winner_and_runner_up():
    rules = [
        {
            'class': 'A',
            'min_score': 1,
            'indicators': [{'type': 'equals', 'column': 'x', 'value': 1}],
        },
        {
            'class': 'B',
            'min_score': 1,
            'indicators': [
                {'type': 'equals', 'column': 'x', 'value': 1},
                {'type': 'equals', 'column': 'y', 'value': 1},
            ],
        },
    ]
    df = pd.DataFrame({'x': [1], 'y': [1]})
    out = _classify(
        _state(df), rules=rules, review_column='review', review_margin=2.0
    ).curated
    # A scores 1, B scores 2: winner is B, margin is 1 < review_margin 2.
    assert out['land_use_class'].iloc[0] == 'B'
    assert bool(out['review'].iloc[0]) is True


def test_review_column_false_for_clear_winner():
    rules = [
        {
            'class': 'A',
            'min_score': 1,
            'indicators': [{'type': 'equals', 'column': 'x', 'value': 1}],
        },
        {
            'class': 'B',
            'min_score': 5,
            'indicators': [{'type': 'equals', 'column': 'y', 'value': 1}],
        },
    ]
    df = pd.DataFrame({'x': [1], 'y': [0]})
    out = _classify(_state(df), rules=rules, review_column='review').curated
    assert out['land_use_class'].iloc[0] == 'A'
    assert bool(out['review'].iloc[0]) is False


def test_flag_column_still_works():
    df = _frame(
        use_group_combined=['MOBILE HOME PARK'], group_parcel=['Manufactured Home']
    )
    out = _classify(
        _state(df),
        rules=RULES,
        flag_column='manufactured_home_community',
        flag_class='Manufactured Home Park',
    ).curated
    assert bool(out['manufactured_home_community'].iloc[0]) is True


# Mirrors the real US_parcel-openplaces-2026.yaml Multiple Single-Family rule:
# min_score raised from 2 to 3 so the two indicators true for almost any
# ordinary single-family parcel (no small-elongated footprints, group is
# Single Family) can't satisfy the rule without a genuine multiplicity signal
# (the keyword or n_primary_footprints_per_parcel) also being true.
MULTIPLE_SF_RULE = {
    'class': 'Multiple Single-Family',
    'min_score': 3,
    'indicators': [
        {
            'type': 'keyword',
            'column': 'use_group_combined',
            'pattern': 'MULTIPLE HOUSES',
        },
        {
            'type': 'count_at_least',
            'column': 'n_primary_footprints_per_parcel',
            'min': 2,
        },
        {
            'type': 'numeric_at_most',
            'column': 'n_small_elongated_footprints_per_parcel',
            'max': 0,
        },
        {'type': 'in_set', 'column': 'group_parcel', 'values': ['Single Family']},
    ],
}


def _ordinary_single_family_frame(**overrides) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            'use_group_combined': ['RESIDENTIAL PRIMARY'],
            'group_parcel': ['Single Family'],
            'n_primary_footprints_per_parcel': [1],
            'n_small_elongated_footprints_per_parcel': [0],
        }
    )
    for key, value in overrides.items():
        df[key] = value
    return df


def test_multiple_single_family_does_not_fire_on_two_weak_indicators_alone():
    # An ordinary single-family parcel: only the two "always true for a
    # normal home" indicators (no small-elongated footprints, group is
    # Single Family) match. Score 2 < min_score 3 -> rule must not fire.
    df = _ordinary_single_family_frame()
    out = _classify(_state(df), rules=[MULTIPLE_SF_RULE]).curated
    assert pd.isna(out['land_use_class'].iloc[0])


def test_multiple_single_family_ignores_stale_n_footprints_per_parcel():
    # n_footprints_per_parcel=2 (the old, noisier signal -- e.g. inflated by a
    # sliver or an accessory structure) must not affect the outcome; only
    # n_primary_footprints_per_parcel (here 1) is read.
    df = _ordinary_single_family_frame(n_footprints_per_parcel=[2])
    out = _classify(_state(df), rules=[MULTIPLE_SF_RULE]).curated
    assert pd.isna(out['land_use_class'].iloc[0])


def test_multiple_single_family_fires_with_genuine_multiplicity_evidence():
    # A genuine multi-home parcel: n_primary_footprints_per_parcel=2 supplies
    # the third indicator alongside the two weak ones -> score 3 = min_score.
    df = _ordinary_single_family_frame(n_primary_footprints_per_parcel=[2])
    out = _classify(_state(df), rules=[MULTIPLE_SF_RULE]).curated
    assert out['land_use_class'].iloc[0] == 'Multiple Single-Family'


def test_multiple_single_family_fires_on_keyword_alone_plus_weak_indicators():
    df = _ordinary_single_family_frame(use_group_combined=['MULTIPLE HOUSES'])
    out = _classify(_state(df), rules=[MULTIPLE_SF_RULE]).curated
    assert out['land_use_class'].iloc[0] == 'Multiple Single-Family'


# --- any_of indicator -------------------------------------------------------


def test_any_of_true_when_any_sub_indicator_matches():
    df = pd.DataFrame({'a': ['X'], 'b': ['Y']})
    matched = evaluate_indicator(
        df,
        {
            'type': 'any_of',
            'indicators': [
                {'type': 'equals', 'column': 'a', 'value': 'NOPE'},
                {'type': 'equals', 'column': 'b', 'value': 'Y'},
            ],
        },
    )
    assert matched.iloc[0]


def test_any_of_false_when_no_sub_indicator_matches():
    df = pd.DataFrame({'a': ['X'], 'b': ['Y']})
    matched = evaluate_indicator(
        df,
        {
            'type': 'any_of',
            'indicators': [
                {'type': 'equals', 'column': 'a', 'value': 'NOPE'},
                {'type': 'equals', 'column': 'b', 'value': 'NOPE'},
            ],
        },
    )
    assert not matched.iloc[0]


def test_any_of_missing_column_sub_indicator_contributes_false():
    df = pd.DataFrame({'a': ['X']})
    matched = evaluate_indicator(
        df,
        {
            'type': 'any_of',
            'indicators': [
                {'type': 'equals', 'column': 'missing', 'value': 'X'},
                {'type': 'equals', 'column': 'a', 'value': 'X'},
            ],
        },
    )
    assert matched.iloc[0]


def test_unknown_indicator_type_still_raises():
    df = pd.DataFrame({'a': ['X']})
    with pytest.raises(ValueError, match='Unknown voting indicator type'):
        evaluate_indicator(df, {'type': 'bogus', 'column': 'a'})


# --- value_share_below / value_share_at_least -------------------------------


def _value_share_indicator(kind: str, **extra) -> dict:
    return {
        'type': kind,
        'value': 'land_value',
        'total': ['land_value', 'improvement_value'],
        **extra,
    }


def test_value_share_below_true_under_ratio():
    df = pd.DataFrame({'land_value': [1_000], 'improvement_value': [199_000]})
    matched = evaluate_indicator(
        df, _value_share_indicator('value_share_below', max_ratio=0.01)
    )
    assert matched.iloc[0]


def test_value_share_below_false_at_or_above_ratio():
    df = pd.DataFrame({'land_value': [50_000], 'improvement_value': [150_000]})
    matched = evaluate_indicator(
        df, _value_share_indicator('value_share_below', max_ratio=0.01)
    )
    assert not matched.iloc[0]


def test_value_share_below_include_zero_matches_zero_total():
    df = pd.DataFrame({'land_value': [0], 'improvement_value': [0]})
    matched = evaluate_indicator(
        df,
        _value_share_indicator('value_share_below', max_ratio=0.01, include_zero=True),
    )
    assert matched.iloc[0]


def test_value_share_at_least_true_when_ratio_meets_minimum():
    df = pd.DataFrame({'land_value': [50_000], 'improvement_value': [150_000]})
    matched = evaluate_indicator(
        df, _value_share_indicator('value_share_at_least', min_ratio=0.01)
    )
    assert matched.iloc[0]


def test_value_share_at_least_false_when_ratio_below_minimum():
    df = pd.DataFrame({'land_value': [0], 'improvement_value': [200_000]})
    matched = evaluate_indicator(
        df, _value_share_indicator('value_share_at_least', min_ratio=0.01)
    )
    assert not matched.iloc[0]


def test_value_share_at_least_false_when_total_is_zero():
    df = pd.DataFrame({'land_value': [0], 'improvement_value': [0]})
    matched = evaluate_indicator(
        df, _value_share_indicator('value_share_at_least', min_ratio=0.01)
    )
    assert not matched.iloc[0]


# --- Condominium vs. Townhome (structural exclusion via land-value share) --

# Mirrors the real US_parcel-openplaces-2026.yaml rules.
CONDOMINIUM_RULE = {
    'class': 'Condominium',
    'min_score': 2,
    'indicators': [
        {
            'type': 'value_share_below',
            'value': 'land_value',
            'total': ['land_value', 'improvement_value'],
            'max_ratio': 0.01,
            'include_zero': True,
        },
        {
            'type': 'any_of',
            'indicators': [
                {
                    'type': 'in_set',
                    'column': 'group_parcel',
                    'values': ['Multi Family', 'Single Family'],
                },
                {
                    'type': 'in_set',
                    'column': 'group_footprint_fema',
                    'values': ['Multi Family', 'Single Family'],
                },
            ],
        },
        {
            'type': 'keyword',
            'column': 'use_group_combined',
            'pattern': 'CONDO|CONDOMINIUM',
        },
    ],
}

TOWNHOME_RULE = {
    'class': 'Townhome',
    'min_score': 4,
    'indicators': [
        {'type': 'numeric_at_most', 'column': 'n_footprints_per_parcel', 'max': 1},
        {
            'type': 'any_of',
            'indicators': [
                {
                    'type': 'numeric_at_least',
                    'column': 'max_parcels_per_footprint',
                    'min': 2,
                },
                {
                    'type': 'numeric_at_least',
                    'column': 'max_dwellings_per_footprint',
                    'min': 2,
                },
                {
                    'type': 'keyword',
                    'column': 'use_group_combined',
                    'pattern': 'TOWNHOUSE|TOWNHOME|ROW HOUSE|ROW-HOUSE',
                },
            ],
        },
        {
            'type': 'any_of',
            'indicators': [
                {
                    'type': 'in_set',
                    'column': 'group_parcel',
                    'values': ['Multi Family', 'Single Family'],
                },
                {
                    'type': 'in_set',
                    'column': 'group_footprint_fema',
                    'values': ['Single Family', 'Multi Family'],
                },
            ],
        },
        {
            'type': 'value_share_at_least',
            'value': 'land_value',
            'total': ['land_value', 'improvement_value'],
            'min_ratio': 0.01,
        },
    ],
}


def _townhome_shaped_frame(**overrides) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            'use_group_combined': ['RESIDENTIAL PRIMARY'],
            'group_parcel': ['Single Family'],
            'n_footprints_per_parcel': [1],
            'max_parcels_per_footprint': [2],
        }
    )
    for key, value in overrides.items():
        df[key] = value
    return df


def test_condominium_wins_over_townhome_on_zero_land_value_morphology():
    # Regression test for US-MA-MI-SO parcel 48e681aa651e741b736033de:
    # Townhome-shaped morphology (footprint shared across parcels) but
    # land_value is 0 with a real improvement_value -- the Massachusetts
    # condominium signature. Townhome's value_share_at_least indicator fails
    # (0/(0+200000)=0 < 0.01), capping its score at 3 < min_score 4, so it
    # never becomes eligible; Condominium (score 2 = min_score 2) wins.
    df = _townhome_shaped_frame(land_value=[0], improvement_value=[200_000])
    out = _classify(_state(df), rules=[CONDOMINIUM_RULE, TOWNHOME_RULE]).curated
    assert out['land_use_class'].iloc[0] == 'Condominium'


def test_townhome_still_wins_with_nonzero_land_value():
    # Same morphology, but a real per-unit land_value (share 0.2, well above
    # the 0.01 cutoff): Condominium's value_share_below fails, capping its
    # score at 1 < min_score 2 (not eligible); Townhome reaches all 4
    # indicators (score 4 = min_score 4) and wins.
    df = _townhome_shaped_frame(land_value=[50_000], improvement_value=[200_000])
    out = _classify(_state(df), rules=[CONDOMINIUM_RULE, TOWNHOME_RULE]).curated
    assert out['land_use_class'].iloc[0] == 'Townhome'


def test_condominium_fires_on_keyword_alone_plus_residential_context():
    # A non-trivial land-value share (0.15) makes the value-share indicator
    # fail, but the assessor "condominium" keyword plus residential context
    # alone still reach min_score 2 -- an intentional consequence worth
    # pinning down explicitly.
    df = pd.DataFrame(
        {
            'use_group_combined': ['RESIDENTIAL CONDOMINIUM'],
            'group_parcel': ['Single Family'],
            'land_value': [15_000],
            'improvement_value': [85_000],
        }
    )
    out = _classify(_state(df), rules=[CONDOMINIUM_RULE]).curated
    assert out['land_use_class'].iloc[0] == 'Condominium'


MULTIPLE_SF_RULE_WITH_VALUE_SHARE = {
    **MULTIPLE_SF_RULE,
    'min_score': 4,
    'indicators': [
        *MULTIPLE_SF_RULE['indicators'],
        {
            'type': 'value_share_at_least',
            'value': 'land_value',
            'total': ['land_value', 'improvement_value'],
            'min_ratio': 0.01,
        },
    ],
}


def test_multiple_single_family_does_not_fire_on_zero_land_value_condo_shape():
    # Same genuine-multiplicity morphology as
    # test_multiple_single_family_fires_with_genuine_multiplicity_evidence,
    # but land_value is 0 (condo shape): the original 4 indicators still
    # score 3, but the new value_share_at_least indicator fails, capping the
    # total at 3 < min_score 4 -- must not fire.
    df = _ordinary_single_family_frame(
        n_primary_footprints_per_parcel=[2], land_value=[0], improvement_value=[200_000]
    )
    out = _classify(_state(df), rules=[MULTIPLE_SF_RULE_WITH_VALUE_SHARE]).curated
    assert pd.isna(out['land_use_class'].iloc[0])


def test_multiple_single_family_fires_with_genuine_multiplicity_and_normal_land_value():
    df = _ordinary_single_family_frame(
        n_primary_footprints_per_parcel=[2],
        land_value=[50_000],
        improvement_value=[150_000],
    )
    out = _classify(_state(df), rules=[MULTIPLE_SF_RULE_WITH_VALUE_SHARE]).curated
    assert out['land_use_class'].iloc[0] == 'Multiple Single-Family'


# --- FEMA occupancy corroborating the residential land-use rules -----------

# Mirrors the real US_parcel-openplaces-2026.yaml Multiple Single-Family rule
# with FEMA wired in via any_of (Fix 5): FEMA backs up group_parcel rather
# than adding an independent 4th vote, so it must not weaken the min_score:3
# guard against the two generic indicators that are true for almost any
# ordinary single-family parcel.
MULTIPLE_SF_RULE_WITH_FEMA = {
    **MULTIPLE_SF_RULE,
    'indicators': [
        *MULTIPLE_SF_RULE['indicators'][:3],
        {
            'type': 'any_of',
            'indicators': [
                {
                    'type': 'in_set',
                    'column': 'group_parcel',
                    'values': ['Single Family'],
                },
                {
                    'type': 'in_set',
                    'column': 'group_footprint_fema',
                    'values': ['Single Family'],
                },
            ],
        },
    ],
}


def test_multiple_single_family_any_of_does_not_fire_from_group_and_fema_alone():
    # An ordinary single-family parcel where FEMA also happens to agree
    # (Single Family): the any_of group+FEMA indicator still counts once, so
    # score is 2 (no-small-elongated + any_of), not 3 -- the rule must not
    # fire. Without any_of (FEMA as an independent vote), this would
    # incorrectly reach score 3 and fire.
    df = _ordinary_single_family_frame(group_footprint_fema=['Single Family'])
    out = _classify(_state(df), rules=[MULTIPLE_SF_RULE_WITH_FEMA]).curated
    assert pd.isna(out['land_use_class'].iloc[0])


MH_PARK_RULE = {
    'class': 'Manufactured Home Park',
    'min_score': 2,
    'indicators': [
        {
            'type': 'keyword',
            'column': 'use_group_combined',
            'pattern': 'MOBILE|MANUFACTURED|MH PARK',
        },
        {'type': 'in_set', 'column': 'group_parcel', 'values': ['Manufactured Home']},
        {
            'type': 'count_at_least',
            'column': 'n_small_elongated_footprints_per_parcel',
            'min': 4,
        },
        {
            'type': 'in_set',
            'column': 'group_footprint_fema',
            'values': ['Manufactured Home'],
        },
    ],
}


def test_manufactured_home_park_fires_on_group_and_fema_vote():
    # Assessor vocabulary doesn't say MOBILE/MANUFACTURED (a local code the
    # keyword list doesn't know), and the small-elongated footprint count is
    # too low -- but the linked-NSI group and the independent FEMA vote
    # together still reach min_score 2.
    df = pd.DataFrame(
        {
            'use_group_combined': ['WVC'],
            'group_parcel': ['Manufactured Home'],
            'n_small_elongated_footprints_per_parcel': [1],
            'group_footprint_fema': ['Manufactured Home'],
        }
    )
    out = _classify(_state(df), rules=[MH_PARK_RULE]).curated
    assert out['land_use_class'].iloc[0] == 'Manufactured Home Park'


def test_rule_wins_are_recorded_in_source_sidecar():
    # Rows a rule classifies carry 'rule' provenance, distinguishing them
    # from the evidence-vote defaults reconcile_land_use fills afterwards.
    df = pd.concat(
        [
            _frame(
                use_group_combined=['MOBILE HOME PARK'],
                group_parcel=['Manufactured Home'],
            ),
            _frame(),
        ],
        ignore_index=True,
    )
    out = _classify(_state(df), rules=RULES).curated
    assert out['land_use_class_source'].iloc[0] == 'rule'
    assert pd.isna(out['land_use_class_source'].iloc[1])
