"""Tests for the generic `derive_indicators` curate step.

One parity case per derivation ``type``, plus the cross-cutting contracts
(unknown type raises, absent inputs skip rather than fail) that let a recipe
declare indicators it may not have the data for.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon

from openplaces.io.curator import CurateState
from openplaces.io.curator.inferers import derive_indicators


def _state(df, recipe=None) -> CurateState:
    return CurateState(
        recipe=recipe or {},
        entity_recipe={},
        admin_id=None,
        verbose=False,
        timer=None,
        curated=df,
    )


def _ruleset_recipe(tmp_path, rows):
    """Write a ruleset CSV where load_ruleset will look for it."""
    csv_path = tmp_path / 'land-use-keywords.csv'
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def ruleset_state(tmp_path, monkeypatch):
    """State whose recipe-relative ruleset lookups resolve into tmp_path."""

    def _fake_recipe_path(admin_id, entity, filename):
        return tmp_path / filename

    monkeypatch.setattr('openplaces.path.recipe_path', _fake_recipe_path)
    return tmp_path


def test_ruleset_class_first_match_wins_and_unmatched_is_missing(ruleset_state):
    _ruleset_recipe(
        ruleset_state,
        [
            # Park pattern precedes the bare one, so the two can never both
            # claim a row -- the ambiguity the old inline regexes allowed.
            {
                'pattern': 'MH PARK|MANUFACTURED HOME PARK',
                'match_type': 'regex',
                'occupancy_type': 'Manufactured Home Park',
                'reviewed': 'true',
            },
            {
                'pattern': 'MOBILE|MANUFACTURED',
                'match_type': 'regex',
                'occupancy_type': 'Manufactured Home',
                'reviewed': 'true',
            },
            {
                'pattern': 'MODULAR',
                'match_type': 'contains',
                'occupancy_type': 'Single-Family',
                'reviewed': 'false',
            },
        ],
    )
    df = pd.DataFrame(
        {'use_group': ['MH PARK', 'MANUFACTURED HOME', 'MODULAR HOME', 'WAREHOUSE']}
    )
    out = derive_indicators(
        _state(df, recipe={'admin_id': 'US', 'entity': {}}),
        [
            {
                'output': 'keyword_class',
                'type': 'ruleset_class',
                'column': 'use_group',
                'ruleset': 'land-use-keywords.csv',
            }
        ],
    ).curated

    assert out['keyword_class'].tolist()[:3] == [
        'Manufactured Home Park',
        'Manufactured Home',
        'Single-Family',
    ]
    # Unmatched classifies as missing rather than passing the raw label
    # through -- this answers "which class?", not "normalize this label".
    assert pd.isna(out['keyword_class'].iloc[3])


def test_ruleset_class_reviewed_only_drops_unreviewed_matches(ruleset_state):
    _ruleset_recipe(
        ruleset_state,
        [
            {
                'pattern': 'MANUFACTURED',
                'match_type': 'contains',
                'occupancy_type': 'Manufactured Home',
                'reviewed': 'true',
            },
            {
                'pattern': 'MODULAR',
                'match_type': 'contains',
                'occupancy_type': 'Single-Family',
                'reviewed': 'false',
            },
        ],
    )
    df = pd.DataFrame({'use_group': ['MANUFACTURED HOME', 'MODULAR HOME']})
    out = derive_indicators(
        _state(df, recipe={'admin_id': 'US', 'entity': {}}),
        [
            {
                'output': 'keyword_class',
                'type': 'ruleset_class',
                'column': 'use_group',
                'ruleset': 'land-use-keywords.csv',
                'reviewed_only': True,
            }
        ],
    ).curated

    assert out['keyword_class'].iloc[0] == 'Manufactured Home'
    assert pd.isna(out['keyword_class'].iloc[1])


def test_pooled_vote_majority_and_tiebreaker():
    df = pd.DataFrame(
        {
            # Row 0: fema+nsi agree, outvoting parcel.
            # Row 1: two-way tie, broken by the tiebreaker column.
            'group_parcel': ['Retail', 'Single Family'],
            'group_fema': ['Manufactured Home', 'Multi Family'],
            'group_nsi': ['Manufactured Home', None],
        }
    )
    out = derive_indicators(
        _state(df),
        [
            {
                'output': 'group_bucket',
                'type': 'pooled_vote',
                'tiebreaker': 'group_parcel',
                'columns': [
                    {'column': 'group_parcel', 'label': 'parcel'},
                    {'column': 'group_fema', 'label': 'fema'},
                    {'column': 'group_nsi', 'label': 'nsi'},
                ],
            }
        ],
    ).curated

    assert out['group_bucket'].iloc[0] == 'Manufactured Home'
    assert out['group_bucket'].iloc[1] == 'Single Family'


def test_pooled_vote_weight_lets_one_source_outrank_another():
    df = pd.DataFrame({'strong': ['Multi-Family'], 'weak': ['Single-Family']})
    out = derive_indicators(
        _state(df),
        [
            {
                'output': 'voted',
                'type': 'pooled_vote',
                'columns': [
                    {'column': 'weak', 'label': 'weak', 'weight': 0.5},
                    {'column': 'strong', 'label': 'strong', 'weight': 1.0},
                ],
            }
        ],
    ).curated
    assert out['voted'].iloc[0] == 'Multi-Family'


def test_ratio_over_column_sum_and_nonpositive_total_is_missing():
    df = pd.DataFrame({'improvement_value': [25.0, 0.0], 'land_value': [75.0, 0.0]})
    out = derive_indicators(
        _state(df),
        [
            {
                'output': 'value_share_improvement',
                'type': 'ratio',
                'numerator': 'improvement_value',
                'denominator': ['improvement_value', 'land_value'],
            }
        ],
    ).curated

    assert out['value_share_improvement'].iloc[0] == 0.25
    # A zero total yields a missing share, never an infinite one.
    assert pd.isna(out['value_share_improvement'].iloc[1])


def test_ratio_over_own_area_uses_the_entitys_own_area():
    # 1 ha = 10,000 m2, so a 500 m2 footprint is a 5% share. Resolves the
    # area from an existing area_ha column rather than recomputing geometry.
    df = pd.DataFrame({'area_ha': [1.0], 'footprint_area_m2': [500.0]})
    out = derive_indicators(
        _state(df),
        [
            {
                'output': 'share',
                'type': 'ratio',
                'numerator': 'footprint_area_m2',
                'denominator': 'own_area',
            }
        ],
    ).curated

    assert out['share'].iloc[0] == 0.05


def test_shape_metric_aspect_ratio_is_measured_in_meters():
    # A 30m x 6m box near 35N: aspect 5.0 only if the geometry is projected
    # to meters first. Measured on raw lon/lat degrees the x axis is
    # compressed by cos(35) ~ 0.82, skewing the ratio badly.
    lon, lat = -78.0, 35.0
    dx = 30.0 / (111_320.0 * 0.8192)
    dy = 6.0 / 110_574.0
    box = Polygon(
        [
            (lon, lat),
            (lon + dx, lat),
            (lon + dx, lat + dy),
            (lon, lat + dy),
        ]
    )
    gdf = gpd.GeoDataFrame({'id': [1]}, geometry=[box], crs='EPSG:4326')
    out = derive_indicators(
        _state(gdf),
        [{'output': 'aspect_ratio', 'type': 'shape_metric', 'metric': 'aspect_ratio'}],
    ).curated

    assert out['aspect_ratio'].iloc[0] == pytest.approx(5.0, rel=0.02)


def test_group_statistic_zscore_scores_within_each_cohort():
    import numpy as np

    df = pd.DataFrame(
        {
            'use_group': ['A', 'A', 'A', 'B', 'B'],
            'area': [10.0, 20.0, 300.0, 5.0, 5.0],
        }
    )
    out = derive_indicators(
        _state(df),
        [
            {
                'output': 'z',
                'type': 'group_statistic',
                'group_column': 'use_group',
                'value_column': 'area',
                'statistic': 'zscore',
                'transform': 'log1p',
            }
        ],
    ).curated

    # Cohort A is scored against its own log-space mean and population std.
    values = np.log1p(df.loc[:2, 'area'])
    expected = (values - values.mean()) / values.std(ddof=0)
    assert out['z'].iloc[:3].to_numpy() == pytest.approx(expected.to_numpy())
    # A zero-variance cohort scores missing rather than dividing by zero.
    assert out['z'].iloc[3:].isna().all()


def test_cohort_threshold_emits_a_ratio_not_a_boolean():
    # Cohort = the three NSI-classed rows, mean area 100 -> threshold
    # max(floor 25, 0.5 * 100) = 50. Every row is scored against it,
    # cohort member or not, as a value/threshold ratio -- the vote applies
    # the >= 1.0 cutoff, not this derivation.
    df = pd.DataFrame(
        {
            'nsi_class': ['MH', 'MH', 'MH', None, None],
            'area_m2': [80.0, 100.0, 120.0, 40.0, None],
        }
    )
    out = derive_indicators(
        _state(df),
        [
            {
                'output': 'habitable_size_ratio',
                'type': 'cohort_threshold',
                'value_column': 'area_m2',
                'cohort_column': 'nsi_class',
                'cohort_value': 'MH',
                'fraction': 0.5,
                'floor': 25,
                'fallback': 90,
                'min_samples': 3,
            }
        ],
    ).curated

    assert out['habitable_size_ratio'].iloc[0] == pytest.approx(80.0 / 50.0)
    assert out['habitable_size_ratio'].iloc[3] == pytest.approx(40.0 / 50.0)
    # A missing value has no ratio; the vote's numeric_at_least casts no vote.
    assert pd.isna(out['habitable_size_ratio'].iloc[4])


def test_cohort_threshold_small_cohort_uses_the_fallback_mean():
    # Only one cohort sample (< min_samples 3): the mean comes from the
    # fallback (90) -> threshold max(25, 0.5 * 90) = 45.
    df = pd.DataFrame({'nsi_class': ['MH', None], 'area_m2': [100.0, 45.0]})
    out = derive_indicators(
        _state(df),
        [
            {
                'output': 'ratio',
                'type': 'cohort_threshold',
                'value_column': 'area_m2',
                'cohort_column': 'nsi_class',
                'cohort_value': 'MH',
                'fraction': 0.5,
                'floor': 25,
                'fallback': 90,
                'min_samples': 3,
            }
        ],
    ).curated
    assert out['ratio'].iloc[0] == pytest.approx(100.0 / 45.0)
    assert out['ratio'].iloc[1] == pytest.approx(1.0)


def test_absent_input_columns_skip_the_spec():
    df = pd.DataFrame({'present': [1.0]})
    out = derive_indicators(
        _state(df),
        [
            {
                'output': 'never_written',
                'type': 'ratio',
                'numerator': 'missing',
                'denominator': ['present'],
            }
        ],
    ).curated
    assert 'never_written' not in out.columns


def test_unknown_derivation_type_raises():
    df = pd.DataFrame({'a': [1.0]})
    with pytest.raises(ValueError, match='Unknown indicator derivation type'):
        derive_indicators(_state(df), [{'output': 'x', 'type': 'nonsense'}])
