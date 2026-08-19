"""Tests for the generic ground-truth validation helpers."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon

from openplaces.io.curator.validation import (
    classify_validation_result,
    compare_classifications_paired,
    link_points_to_entities,
    normalize_house_number,
    score_classification,
    summarize_sources,
)


def _box(lon, lat, size=0.0002):
    return Polygon(
        [(lon, lat), (lon + size, lat), (lon + size, lat + size), (lon, lat + size)]
    )


def test_normalize_house_number_bridges_float_and_string():
    # The trap this exists for: a number that round-tripped through a float
    # column never matches the string form, and fails silently.
    assert normalize_house_number(5114.0) == '5114'
    assert normalize_house_number('5114') == '5114'
    assert normalize_house_number('  5114 ') == '5114'
    assert normalize_house_number(None) is None
    assert normalize_house_number(float('nan')) is None


def test_address_match_beats_a_nearer_wrong_building():
    # The pin sits almost on top of entity 0 (a shed), but the address
    # identifies entity 1. Identity must win over proximity.
    entities = gpd.GeoDataFrame(
        {
            'address_number': ['99', '5114'],
            'address_street': ['OTHER ST', 'S VIRGINIA DARE TRL'],
            'occupancy_type': ['Secondary', 'Single-Family'],
        },
        geometry=[_box(-78.0, 35.0), _box(-78.01, 35.0)],
        crs='EPSG:4326',
    )
    points = pd.DataFrame(
        {
            'address_number': [5114.0],
            'address_street': ['SOUTH VIRGINIA DARE TRAIL'],
            'lon': [-78.0],
            'lat': [35.0],
        }
    )
    linked = link_points_to_entities(points, entities, admin1_id='US-NC')

    assert len(linked) == 1
    assert linked['matched_by'].iloc[0] == 'address'
    assert linked['occupancy_type_inv'].iloc[0] == 'Single-Family'


def _shared_address_entities():
    """A house and its garage at one address, garage listed first.

    Listing the garage first is the point: without a tie-break the linker
    returns whichever entity the row order happens to put first.
    """
    return gpd.GeoDataFrame(
        {
            'address_number': ['5114', '5114'],
            'address_street': ['S VIRGINIA DARE TRL', 'S VIRGINIA DARE TRL'],
            'priority_on_parcel': ['secondary', 'primary'],
            'occupancy_type': ['Secondary', 'Single-Family'],
        },
        geometry=[_box(-78.0, 35.0), _box(-78.01, 35.0)],
        crs='EPSG:4326',
    )


def _shared_address_point():
    return pd.DataFrame(
        {
            'address_number': [5114.0],
            'address_street': ['SOUTH VIRGINIA DARE TRAIL'],
            'lon': [-78.0],
            'lat': [35.0],
        }
    )


def test_prefer_column_breaks_an_address_tie_toward_the_primary_structure():
    linked = link_points_to_entities(
        _shared_address_point(),
        _shared_address_entities(),
        admin1_id='US-NC',
        prefer_column='priority_on_parcel',
        prefer_values=('primary', 'unknown'),
    )

    assert linked['matched_by'].iloc[0] == 'address'
    assert linked['occupancy_type_inv'].iloc[0] == 'Single-Family'


def test_without_prefer_column_the_first_matching_row_still_wins():
    # Pins the historical behavior: the tie-break is opt-in, so a caller
    # that does not ask for it sees exactly what it saw before.
    linked = link_points_to_entities(
        _shared_address_point(),
        _shared_address_entities(),
        admin1_id='US-NC',
    )

    assert linked['occupancy_type_inv'].iloc[0] == 'Secondary'


def test_prefer_column_is_ignored_when_the_entities_lack_it():
    # A caller may share one config across entity sets that do not all
    # carry the ranking column; that must not raise.
    entities = _shared_address_entities().drop(columns='priority_on_parcel')
    linked = link_points_to_entities(
        _shared_address_point(),
        entities,
        admin1_id='US-NC',
        prefer_column='priority_on_parcel',
        prefer_values=('primary',),
    )

    assert linked['occupancy_type_inv'].iloc[0] == 'Secondary'


def test_distance_fallback_applies_and_respects_the_radius():
    entities = gpd.GeoDataFrame(
        {'occupancy_type': ['Single-Family']},
        geometry=[_box(-78.0, 35.0)],
        crs='EPSG:4326',
    )
    near = pd.DataFrame({'lon': [-78.0], 'lat': [35.0]})
    far = pd.DataFrame({'lon': [-78.5], 'lat': [35.5]})

    assert link_points_to_entities(near, entities)['matched_by'].iloc[0] == 'distance'
    assert link_points_to_entities(far, entities).empty


def test_entity_columns_are_suffixed_so_they_cannot_shadow_the_truth():
    # Both sides carry occupancy_type. If the entity value overwrote the
    # ground-truth one, every comparison would agree with itself.
    entities = gpd.GeoDataFrame(
        {'occupancy_type': ['Multi-Family']},
        geometry=[_box(-78.0, 35.0)],
        crs='EPSG:4326',
    )
    points = pd.DataFrame(
        {'occupancy_type': ['Single-Family'], 'lon': [-78.0], 'lat': [35.0]}
    )
    linked = link_points_to_entities(points, entities)

    assert linked['occupancy_type'].iloc[0] == 'Single-Family'
    assert linked['occupancy_type_inv'].iloc[0] == 'Multi-Family'


def test_validation_result_separates_omission_from_commission():
    truth = pd.Series(['Single-Family', 'Single-Family', 'Multi-Family'])
    predicted = pd.Series(['Single-Family', None, 'Single-Family'])
    result = classify_validation_result(truth, predicted)

    # Declining to answer and answering wrongly have different fixes, so they
    # get different labels.
    assert result.tolist() == ['correct', 'omission', 'commission']


def test_summarize_sources_renders_missing_values_rather_than_dropping_them():
    summary = summarize_sources(
        {
            'truth': pd.Series(['Multi-Family']),
            'nsi': pd.Series([None]),
            'fema': pd.Series(['Single-Family']),
        }
    )
    # A source that said nothing is itself evidence about the disagreement.
    assert summary.iloc[0] == 'truth: Multi-Family | nsi: - | fema: Single-Family'


def test_score_classification_separates_precision_from_recall():
    # Predicts Single-Family for everything: perfect recall for that class,
    # poor precision. Recall alone would call this a success.
    truth = pd.Series(['Single-Family', 'Single-Family', 'Multi-Family'])
    predicted = pd.Series(['Single-Family'] * 3)
    table = score_classification(
        truth, predicted, ['Single-Family', 'Multi-Family']
    ).set_index('class')

    # Scores are rounded to 4 places, so compare at that resolution.
    assert table.loc['Single-Family', 'recall'] == 1.0
    assert table.loc['Single-Family', 'precision'] == pytest.approx(2 / 3, abs=1e-4)
    assert table.loc['Multi-Family', 'recall'] == 0.0
    assert table.loc['ALL', 'recall'] == pytest.approx(2 / 3, abs=1e-4)


def test_score_classification_ignores_unpredicted_rows_in_recall():
    truth = pd.Series(['Single-Family', 'Single-Family'])
    predicted = pd.Series(['Single-Family', None])
    table = score_classification(truth, predicted, ['Single-Family']).set_index('class')

    # One of two was scored, and it was right: recall over scored rows is 1.0
    # while n_scored records that half the population went unanswered.
    assert table.loc['Single-Family', 'n_truth'] == 2
    assert table.loc['Single-Family', 'n_scored'] == 1
    assert table.loc['Single-Family', 'recall'] == 1.0


def test_paired_comparison_of_an_unchanged_run_is_exactly_zero():
    # The property the whole gate rests on. Comparing two independent point
    # estimates on a sample this size leaves several F1 points of spread, so
    # a neutral change looks like a regression; drawing the same indices for
    # both predictions cancels it exactly.
    truth = pd.Series(['A'] * 60 + ['B'] * 40)
    result = compare_classifications_paired(
        truth, truth.copy(), truth.copy(), ['A', 'B'], n_draws=50
    )
    assert (result['d_f1'] == 0).all()
    assert (result['d_low'] == 0).all()
    assert (result['d_high'] == 0).all()


def test_paired_comparison_resolves_a_regression_below_the_old_tolerance():
    # Ten wrong calls out of 400 move ALL F1 by well under the 0.01 the
    # superseded gate needed to see, and the paired interval still separates
    # it from zero.
    truth = pd.Series(['A'] * 200 + ['B'] * 200)
    baseline = truth.copy()
    proposed = truth.copy()
    proposed.iloc[:10] = 'B'
    result = compare_classifications_paired(
        truth, baseline, proposed, ['A', 'B'], n_draws=200
    ).set_index('class')
    assert abs(result.loc['ALL', 'd_f1']) < 0.03
    assert result.loc['A', 'd_high'] < 0
    assert result.loc['A', 'p_worse'] == 1.0


def test_paired_comparison_rejects_misaligned_inputs():
    truth = pd.Series(['A', 'B', 'A'])
    with pytest.raises(ValueError, match='aligned'):
        compare_classifications_paired(
            truth, truth, pd.Series(['A', 'B']), ['A', 'B'], n_draws=5
        )


def test_paired_comparison_is_reproducible_for_a_seed():
    truth = pd.Series(['A'] * 30 + ['B'] * 30)
    proposed = truth.copy()
    proposed.iloc[:5] = 'B'
    kwargs = dict(classes=['A', 'B'], n_draws=40, seed=7)
    first = compare_classifications_paired(truth, truth.copy(), proposed, **kwargs)
    second = compare_classifications_paired(truth, truth.copy(), proposed, **kwargs)
    pd.testing.assert_frame_equal(first, second)
