"""Tests for the generic weighted-voting reconciler (resolve_by_vote).

Each decision tallies the weights of its matched indicators; among decisions that
reach their min_score, the highest score wins (ties broken by recipe order). The
manufactured-home scenario checks that low value plus a parcel/NSI
manufactured-home signal outvote a lone n_dwellings>=2 multi-family vote, while
a genuine multi-unit parcel still resolves to Multi-Family.
"""

from __future__ import annotations

import pandas as pd

from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState
from openplaces.io.curator.reconcilers import resolve_by_vote

MANUFACTURED_HOME_DECISION = {
    'class': 'Manufactured Home',
    'min_score': 2,
    'indicators': [
        {
            'type': 'value_share_below',
            'value': 'improvement_value_parcel',
            'total': ['improvement_value_parcel', 'land_value_parcel'],
            'max_ratio': 0.025,
            'include_zero': True,
        },
        {
            'type': 'keyword',
            'column': 'use_group_combined_parcel',
            'pattern': 'MOBILE|MANUFACTURED|SINGLE WIDE|DOUBLE WIDE',
            'regex': True,
        },
        {'type': 'equals', 'column': 'group_parcel', 'value': 'Manufactured Home'},
    ],
}

MULTI_FAMILY_DECISION = {
    'class': 'Multi-Family',
    'source': 'multi_family_dwellings',
    'min_score': 1,
    'indicators': [{'type': 'numeric_at_least', 'column': 'n_dwellings', 'min': 2}],
}

DECISIONS = [MANUFACTURED_HOME_DECISION, MULTI_FAMILY_DECISION]


def _state(df: pd.DataFrame) -> CurateState:
    return CurateState(
        recipe={'entity': 'footprint-cheer-2026', 'admin_id': 'US'},
        entity_recipe={},
        admin_id=AdminId('US'),
        verbose=False,
        timer=None,
        curated=df,
    )


def _frame(**overrides) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            'occupancy_type': ['Single-Family'],
            'use_group_combined_parcel': ['RESIDENTIAL'],
            'group_parcel': ['Single Family'],
            'improvement_value_parcel': [100000.0],
            'land_value_parcel': [50000.0],
            'n_dwellings': [1.0],
        }
    )
    for key, value in overrides.items():
        df[key] = value
    return df


def _vote(df: pd.DataFrame) -> pd.DataFrame:
    return resolve_by_vote(
        _state(df), target='occupancy_type', decisions=DECISIONS
    ).curated


def test_low_value_plus_keyword_beats_dwellings():
    # Low value (0) + manufactured-home keyword = 2 votes for Manufactured Home;
    # n_dwellings=3 is 1 vote for Multi-Family. Manufactured Home wins.
    out = _vote(
        _frame(
            improvement_value_parcel=[0.0],
            use_group_combined_parcel=['MOBILE HOME PARK'],
            n_dwellings=[3.0],
        )
    )
    assert out['occupancy_type'].iloc[0] == 'Manufactured Home'
    assert out['occupancy_type_source'].iloc[0] == 'vote'


def test_low_value_plus_nsi_group_tendency_beats_dwellings():
    # Low value + NSI-mode group_parcel == Manufactured Home = 2 votes; the
    # manufactured-home keyword is absent. Still beats the lone n_dwellings vote.
    out = _vote(
        _frame(
            improvement_value_parcel=[500.0],  # 500/50500 = 0.0099 < 0.025
            group_parcel=['Manufactured Home'],
            use_group_combined_parcel=['RESIDENTIAL'],
            n_dwellings=[4.0],
        )
    )
    assert out['occupancy_type'].iloc[0] == 'Manufactured Home'


def test_genuine_multifamily_resolves_to_multifamily():
    # No manufactured-home evidence: Manufactured Home scores 0, Multi-Family
    # scores 1.
    out = _vote(
        _frame(
            improvement_value_parcel=[300000.0],
            use_group_combined_parcel=['APARTMENTS'],
            group_parcel=['Multi Family'],
            n_dwellings=[4.0],
        )
    )
    assert out['occupancy_type'].iloc[0] == 'Multi-Family'
    # The token is the decision's own name, not the bare 'dwellings' this used
    # to share with impute_occupancy_type's single_family_dwellings rule.
    assert out['occupancy_type_source'].iloc[0] == 'multi_family_dwellings'


def test_single_manufactured_home_indicator_does_not_reach_min_score():
    # Only the keyword fires (1 vote < min_score 2) and n_dwellings is 1, so no
    # decision is eligible: the base class is kept and no source is stamped.
    out = _vote(
        _frame(
            improvement_value_parcel=[100000.0],
            use_group_combined_parcel=['MOBILE HOME'],
            group_parcel=['Single Family'],
            n_dwellings=[1.0],
        )
    )
    assert out['occupancy_type'].iloc[0] == 'Single-Family'
    assert pd.isna(out.get('occupancy_type_source', pd.Series([pd.NA])).iloc[0])


def test_no_eligible_decision_keeps_base():
    out = _vote(_frame())  # nothing matches
    assert out['occupancy_type'].iloc[0] == 'Single-Family'


def test_tie_broken_by_recipe_order():
    # Two decisions tie at score 1; the earlier one in the list wins.
    decisions = [
        {
            'class': 'First',
            'min_score': 1,
            'indicators': [
                {'type': 'numeric_at_least', 'column': 'n_dwellings', 'min': 2}
            ],
        },
        {
            'class': 'Second',
            'min_score': 1,
            'indicators': [{'type': 'equals', 'column': 'group_parcel', 'value': 'X'}],
        },
    ]
    df = pd.DataFrame(
        {
            'occupancy_type': ['Base'],
            'n_dwellings': [3.0],
            'group_parcel': ['X'],
        }
    )
    out = resolve_by_vote(
        _state(df), target='occupancy_type', decisions=decisions
    ).curated
    assert out['occupancy_type'].iloc[0] == 'First'


def test_missing_columns_contribute_no_votes():
    # When indicator columns are absent the indicator is simply skipped; here only
    # n_dwellings exists, so Multi-Family wins on its single vote.
    df = pd.DataFrame({'occupancy_type': ['Single-Family'], 'n_dwellings': [2.0]})
    out = resolve_by_vote(
        _state(df), target='occupancy_type', decisions=DECISIONS
    ).curated
    assert out['occupancy_type'].iloc[0] == 'Multi-Family'


def test_require_blocks_decision_even_when_score_is_reached():
    # All three Manufactured Home indicators fire (score 3 >= min_score 2), but
    # `require` (m2 >= 20) fails for a tiny footprint: the decision must not
    # win, and the existing 'Secondary' value is preserved untouched.
    decisions = [
        {
            **MANUFACTURED_HOME_DECISION,
            'require': [{'type': 'numeric_at_least', 'column': 'area_m2', 'min': 20}],
        }
    ]
    df = _frame(
        occupancy_type=['Secondary'],
        improvement_value_parcel=[0.0],
        use_group_combined_parcel=['MOBILE HOME PARK'],
        area_m2=[5.0],
    )
    out = resolve_by_vote(
        _state(df), target='occupancy_type', decisions=decisions
    ).curated
    assert out['occupancy_type'].iloc[0] == 'Secondary'
    assert pd.isna(out.get('occupancy_type_source', pd.Series([pd.NA])).iloc[0])


def test_require_allows_decision_once_size_threshold_is_met():
    decisions = [
        {
            **MANUFACTURED_HOME_DECISION,
            'require': [{'type': 'numeric_at_least', 'column': 'area_m2', 'min': 20}],
        }
    ]
    df = _frame(
        occupancy_type=['Secondary'],
        improvement_value_parcel=[0.0],
        use_group_combined_parcel=['MOBILE HOME PARK'],
        area_m2=[25.0],
    )
    out = resolve_by_vote(
        _state(df), target='occupancy_type', decisions=decisions
    ).curated
    assert out['occupancy_type'].iloc[0] == 'Manufactured Home'


def test_require_with_no_indicators_behaves_like_before():
    # A decision with no `require` key is unaffected (fully backward compatible).
    out = _vote(_frame(improvement_value_parcel=[300000.0], n_dwellings=[4.0]))
    assert out['occupancy_type'].iloc[0] == 'Multi-Family'


def test_is_null_indicator_matches_only_missing_values():
    # is_null is the absence half of not_null: inside an any_of with an
    # in_set allow-list it expresses "one of these values, or nothing
    # known" -- the vocabulary's stand-in for negation.
    decisions = [
        {
            'class': 'Allowed',
            'min_score': 1,
            'require': [
                {
                    'type': 'any_of',
                    'indicators': [
                        {'type': 'in_set', 'column': 'base', 'values': ['Res']},
                        {'type': 'is_null', 'column': 'base'},
                    ],
                }
            ],
            'indicators': [{'type': 'equals', 'column': 'x', 'value': 1}],
        }
    ]
    df = pd.DataFrame(
        {
            'occupancy_type': [None, None, 'Retail'],
            'base': ['Res', None, 'Retail'],
            'x': [1, 1, 1],
        }
    )
    out = resolve_by_vote(
        _state(df), target='occupancy_type', decisions=decisions
    ).curated
    # Residential and missing bases resolve; a positively different base
    # vetoes the decision and the incoming value stands.
    assert out['occupancy_type'].tolist() == ['Allowed', 'Allowed', 'Retail']


def test_not_null_indicator_scores_presence_only():
    # not_null votes wherever the column holds any value, and never where it
    # is missing -- the presence test the vocabulary previously faked with a
    # numeric_at_least min: 0.
    decisions = [
        {
            'class': 'Counted',
            'min_score': 1,
            'indicators': [{'type': 'not_null', 'column': 'n_stories'}],
        }
    ]
    df = pd.DataFrame({'occupancy_type': [None, None], 'n_stories': [3.0, None]})
    out = resolve_by_vote(
        _state(df), target='occupancy_type', decisions=decisions
    ).curated
    assert out['occupancy_type'].iloc[0] == 'Counted'
    assert pd.isna(out['occupancy_type'].iloc[1])


def test_uncontested_winner_has_no_margin_and_is_never_review_flagged():
    # Rows: (0) contested 2-vs-1 -> margin 1, flagged at review_margin 2;
    # (1) uncontested lone winner -> no runner-up, no margin, never flagged,
    # however large the review_margin. Before this contract an uncontested
    # score-1 win reported margin 2.0 (score minus a -1 sentinel) and read
    # as *more* corroborated than a genuine 3-vs-2 contest.
    decisions = [
        {
            'class': 'A',
            'min_score': 1,
            'indicators': [
                {'type': 'equals', 'column': 'x', 'value': 1},
                {'type': 'equals', 'column': 'y', 'value': 1},
            ],
        },
        {
            'class': 'B',
            'min_score': 1,
            'indicators': [{'type': 'equals', 'column': 'x', 'value': 1}],
        },
    ]
    df = pd.DataFrame({'occupancy_type': [None, None], 'x': [1, 0], 'y': [1, 1]})
    out = resolve_by_vote(
        _state(df),
        target='occupancy_type',
        decisions=decisions,
        review_column='review',
        review_margin=100.0,
    ).curated
    assert out['occupancy_type'].tolist() == ['A', 'A']
    assert bool(out['review'].iloc[0]) is True  # contested: margin 1 < 100
    assert bool(out['review'].iloc[1]) is False  # uncontested: no margin


def test_score_decisions_is_pure_and_returns_requested_scores():
    # The scoring core never writes to the frame it reads -- score columns
    # are returned and assigned by the resolve_by_vote wrapper.
    from openplaces.io.curator.indicators import score_decisions

    decisions = [
        {
            'class': 'A',
            'min_score': 1,
            'indicators': [{'type': 'equals', 'column': 'x', 'value': 1}],
        }
    ]
    df = pd.DataFrame({'x': [1, 0]})
    columns_before = list(df.columns)
    winner, token, best, second, scores = score_decisions(df, decisions, {'A'})
    assert list(df.columns) == columns_before
    assert scores['A'].tolist() == [1.0, 0.0]
    assert winner.tolist()[0] == 'A' and pd.isna(winner.iloc[1])
    # A lone winner has no runner-up: second_score is missing, not -1.
    assert second.isna().all()
